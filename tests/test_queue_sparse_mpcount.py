from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/queue_sparse_mpcount.py"
spec = importlib.util.spec_from_file_location("queue_sparse_mpcount", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def request(**changes):
    return {"min_free_mib": 7168, "stable_checks": 3, "poll_seconds": 30, "gpus": [], **changes}


def test_gpu_inventory_uses_free_memory_not_utilization():
    assert module.parse_gpus("0, GPU-a, 4000, 12288\n1, GPU-b, 10000, 12288\n") == [
        {"index": 0, "uuid": "GPU-a", "free_mib": 8288},
        {"index": 1, "uuid": "GPU-b", "free_mib": 2288},
    ]


def test_wait_requires_consecutive_stable_memory_checks():
    inventories = iter([
        [{"index": 0, "uuid": "GPU-a", "free_mib": value}]
        for value in (6000, 8000, 7000, 7900, 7600, 7400)
    ])
    updates, sleeps = [], []
    selected = module.wait_for_gpu(
        request(), lambda **values: updates.append(values), lambda: False,
        query=lambda: next(inventories), sleep=sleeps.append,
    )
    assert selected["uuid"] == "GPU-a"
    assert [entry["stable_checks"] for entry in updates] == [0, 1, 0, 1, 2, 3]
    assert sleeps == [30] * 5


def test_wait_obeys_gpu_restriction():
    inventory = [{"index": 0, "uuid": "GPU-a", "free_mib": 10000},
                 {"index": 1, "uuid": "GPU-b", "free_mib": 8000}]
    selected = module.wait_for_gpu(
        request(gpus=[1], stable_checks=1), lambda **_: None, lambda: False,
        query=lambda: inventory, sleep=lambda _: pytest.fail("Should launch immediately"),
    )
    assert selected["index"] == 1


def test_cancelled_wait_does_not_query_or_launch():
    with pytest.raises(InterruptedError):
        module.wait_for_gpu(
            request(), lambda **_: None, lambda: True,
            query=lambda: pytest.fail("Cancelled queue must not query GPUs"),
        )


def test_transient_gpu_query_failure_resets_stability():
    results = iter([8000, subprocess.TimeoutExpired("nvidia-smi", 15), 8100, 8200, 8300])
    updates = []

    def query():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return [{"index": 0, "uuid": "GPU-a", "free_mib": result}]

    module.wait_for_gpu(request(), lambda **values: updates.append(values), lambda: False,
                        query=query, sleep=lambda _: None)
    assert [entry["stable_checks"] for entry in updates] == [1, 0, 1, 2, 3]
    assert updates[1]["gpu_query_error"] == "TimeoutExpired"
    assert updates[-1]["gpu_query_error"] is None


@pytest.mark.parametrize("version", ["../escape", "stb_10_seed2_effbs16_retry1", "stb_10_seed3_effbs16"])
def test_retry_requires_safe_new_seed_specific_name(version):
    with pytest.raises(ValueError):
        module.validate_version(3, version)


def test_failed_child_exit_code_and_output_are_preserved(tmp_path):
    updates = []
    log = tmp_path / "training.log"
    code = module.run_child(
        [sys.executable, "-c", "print('failure captured'); raise SystemExit(7)"],
        {"uuid": "GPU-test", "index": 0}, log,
        lambda **values: updates.append(values), lambda: False,
    )
    assert code == 7
    assert "failure captured" in log.read_text(encoding="utf-8")
    assert updates[-1] == {"child_pid": None, "child_exit_code": 7}


def test_cancelled_child_is_not_started(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_, **__: pytest.fail("Must not launch"))
    with pytest.raises(InterruptedError):
        module.run_child(["not-run"], {"uuid": "GPU-test"}, tmp_path / "training.log",
                         lambda **_: None, lambda: True)


def test_child_is_stopped_when_queue_is_cancelled(tmp_path):
    updates = []
    with pytest.raises(InterruptedError):
        module.run_child(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            {"uuid": "GPU-test"}, tmp_path / "training.log",
            lambda **values: updates.append(values), lambda: bool(updates),
        )
    assert updates[-1]["child_pid"] is None
    assert updates[-1]["child_exit_code"] != 0


def fake_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    upstream = project / "external/MPCount"
    monkeypatch.setattr(module, "PROJECT_ROOT", project)
    monkeypatch.setattr(module, "MPCOUNT_ROOT", upstream)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_, **__: "commit\n")
    config_rel = "configs/mpcount/stb_10_train_accum4_seed3.yml"
    config = yaml.safe_load((SCRIPT.parents[1] / config_rel).read_text(encoding="utf-8"))
    (project / config_rel).parent.mkdir(parents=True)
    (project / config_rel).write_text(yaml.safe_dump(config), encoding="utf-8")
    train = upstream / config["train_dataset"]["params"]["root"] / "train"
    val = train.parent / "val"
    train.mkdir(parents=True)
    val.mkdir()
    for index in range(1, 33):
        for suffix in (".jpg", ".npy", "_dmap.npy"):
            (train / f"IMG_{index}{suffix}").write_bytes(b"sample")
    for index in range(33, 113):
        (val / f"IMG_{index}.jpg").write_bytes(b"sample")
    split_path = project / "splits/stb_10_seed3.json"
    split_path.parent.mkdir()
    split_path.write_text(json.dumps({"labeled_ids": [f"IMG_{index}" for index in range(1, 33)]}), encoding="utf-8")
    for relative in ("uv.lock", "scripts/queue_sparse_mpcount.py", "scripts/train_mpcount_accum.py",
                     "scripts/evaluate_sparse_mpcount.py", "scripts/select_mpcount_checkpoint.py",
                     "src/sparse2unseen/monitoring/mpcount_log.py",
                     "configs/mpcount/stb_100_effbs16_test_stb.yml",
                     "configs/mpcount/stb_100_effbs16_test_sta.yml",
                     "configs/mpcount/stb_100_effbs16_test_qnrf.yml"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    args = argparse.Namespace(seed=3, run_version="stb_10_seed3_effbs16_retry1", min_free_mib=7168,
                              poll_seconds=30, stable_checks=3, gpu=None, wandb=True, evaluate=True)
    return project, args


def test_request_freezes_source_inputs_and_rejects_changes(tmp_path, monkeypatch):
    project, args = fake_project(tmp_path, monkeypatch)
    prepared = module.prepare_request(args)
    assert prepared["seed"] == 3
    assert prepared["evaluate"] is True
    assert prepared["min_free_mib"] == 7168
    module.check_inputs(prepared)
    (project / "scripts/train_mpcount_accum.py").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Queued input changed"):
        module.check_inputs(prepared)


def test_queue_refuses_existing_training_directory(tmp_path, monkeypatch):
    _, args = fake_project(tmp_path, monkeypatch)
    (module.MPCOUNT_ROOT / "logs" / args.run_version).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.prepare_request(args)
