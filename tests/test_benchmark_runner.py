from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import threading

import pytest


def runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts/run_benchmark_suite.py"
    definition = importlib.util.spec_from_file_location("test_benchmark_runner", script)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fail,workers", [(False, 2), (True, 1)])
def test_detached_runner_coordinates_jobs_and_bounds_failed_attempts(tmp_path, monkeypatch, fail, workers):
    module = runner_module()
    suite = tmp_path / "runs/benchmark_v1"
    suite.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SUITE", suite)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    gpu_list = [{"index": index, "uuid": f"GPU-{index}", "free_mib": 12000} for index in range(workers)]
    monkeypatch.setattr(module, "query_gpus", lambda: gpu_list)
    monkeypatch.setattr(module, "wait_for_gpu", lambda request, *_: gpu_list[request["gpus"][0]])
    jobs = []
    for index in range(1 if fail else 3):
        name = f"job{index}"
        spec = suite / "specs" / f"{name}.json"
        module.write_json(spec, {"name": name})
        jobs.append({"name": name, "state": "pending", "attempts": 0, "spec": str(spec.relative_to(tmp_path))})
    module.write_json(suite / "matrix.json", {"jobs": jobs, "inputs": {}})
    commands, lock = [], threading.Lock()

    class Child:
        def __init__(self, command, **kwargs):
            spec = json.loads(Path(command[command.index("--spec")+1]).read_text())
            name = spec["name"]
            phase = Path(command[1]).stem
            with lock:
                commands.append((name, phase))
                self.pid = 1000 + len(commands)
            self.returncode = 1 if fail else 0
            directory = suite / "jobs" / name
            if phase == "train_benchmark":
                assert not directory.exists(), "Console capture must not pre-create the training output directory"
                if not fail:
                    directory.mkdir(parents=True)
            else:
                module.write_json(directory / "results.json", {"domains": {domain: {} for domain in ("stb", "sta", "qnrf")}})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    args = argparse.Namespace(workers=workers, min_free_mib=8192, stable_checks=3, poll_seconds=30)
    assert module.worker(args) == (1 if fail else 0)
    state = module.load_json(suite / "queue_state.json")
    if fail:
        assert state["state"] == "failed"
        assert commands == [("job0", "train_benchmark")] * 3
        assert state["jobs"][0]["attempts"] == 3
    else:
        assert state["state"] == "complete"
        assert Counter(commands) == Counter((job["name"], phase) for job in jobs for phase in ("train_benchmark", "evaluate_benchmark"))
        assert all(job["state"] == "complete" for job in state["jobs"])
