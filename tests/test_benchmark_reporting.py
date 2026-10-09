from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sparse2unseen.experiments.common import digest, write_json


@pytest.fixture
def completed(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/report_benchmark_suite.py"
    definition = importlib.util.spec_from_file_location("benchmark_reporting_test", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    domains = ("stb", "sta", "qnrf")
    monkeypatch.setattr(module, "SOURCE_COUNTS", {domain: (4, 1, 2) for domain in domains})
    output = tmp_path / "output"
    output.mkdir()
    spec = {"name": "fixture", "source": "stb", "fraction": 0.1, "method": "label_only", "seed": 1,
            "samples_per_epoch": 320, "updates_per_epoch": 20,
            "settings": {"epochs": 2, "warmup_epochs": 1,
                         "test_patch_sizes": {"stb": 10000, "sta": 10000, "qnrf": 1024}}}
    write_json(output / "spec.json", spec)
    checkpoint = output / "best.pt"
    checkpoint.write_bytes(b"selected model fixture")
    sha = digest(checkpoint)
    marker = {"experiment": "fixture", "selection_domain": "stb_val", "model_kind": "student",
              "smoke": False, "epochs": 2, "selected_epoch": 1, "source_val_mae": 3,
              "checkpoint_sha256": sha, "spec_sha256": digest(output / "spec.json")}
    write_json(output / "training_complete.json", marker)
    history = [{"epoch": epoch, "source_val_mae": mae, "source_val_rmse": mae * 2,
                "optimizer_updates": 20, "labeled_samples": 320, "unlabeled_samples": 0}
               for epoch, mae in enumerate([5, 3])]
    (output / "epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    result = {**{key: spec[key] for key in ("source", "fraction", "method", "seed")},
              "experiment": "fixture", "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
              "selection_domain": "stb_val", "model_kind": "student", "selected_epoch": 1, "domains": {}}
    for domain in domains:
        records = [{"id": f"{domain}_1", "count": 1, "predicted": 2.0},
                   {"id": f"{domain}_2", "count": 2, "predicted": 4.0}]
        manifest = tmp_path / f"data/manifests/{domain}_test.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps({"id": row["id"], "count": row["count"]}) + "\n" for row in records))
        cache = {"n": 2, "mae": 1.5, "mse": 2.5, "rmse": math.sqrt(2.5),
                 "predictions": records, "checkpoint_sha256": sha,
                 "patch_size": spec["settings"]["test_patch_sizes"][domain]}
        write_json(output / f"test_{domain}.json", cache)
        result["domains"][domain] = {key: value for key, value in cache.items() if key != "predictions"}
    return module, spec, result, output, marker, history


def test_complete_source_selected_predictions_pass(completed):
    module, spec, result, output, _, _ = completed
    assert module.audit(spec, result, output) == result


def test_a_reported_score_must_match_saved_predictions(completed):
    module, spec, result, output, _, _ = completed
    result["domains"]["qnrf"]["mae"] = 1.0
    with pytest.raises(ValueError, match="saved per-image"):
        module.audit(spec, result, output)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_count", "wrong_tile", "wrong_checkpoint"])
def test_prediction_coverage_and_recipe_are_audited(completed, change):
    module, spec, result, output, _, _ = completed
    cache_path = output / "test_sta.json"
    cache = json.loads(cache_path.read_text())
    if change == "missing":
        cache["predictions"].pop()
    elif change == "duplicate":
        cache["predictions"][1]["id"] = cache["predictions"][0]["id"]
    elif change == "wrong_count":
        cache["predictions"][0]["count"] = 99
    elif change == "wrong_tile":
        cache["patch_size"] = 1024
    elif change == "wrong_checkpoint":
        cache["checkpoint_sha256"] = "other weights"
    write_json(cache_path, cache)
    with pytest.raises(ValueError):
        module.audit(spec, result, output)


def test_best_checkpoint_must_be_minimum_source_validation_mae(completed):
    module, spec, result, output, marker, _ = completed
    marker.update(selected_epoch=0, source_val_mae=5)
    result["selected_epoch"] = 0
    write_json(output / "training_complete.json", marker)
    with pytest.raises(ValueError, match="minimum source validation"):
        module.audit(spec, result, output)


@pytest.mark.parametrize("change", ["partial", "budget", "wrong_variant", "smoke"])
def test_full_training_budget_and_model_variant_are_required(completed, change):
    module, spec, result, output, marker, history = completed
    if change == "partial":
        history.pop()
    elif change == "budget":
        history[1]["optimizer_updates"] = 19
    elif change == "wrong_variant":
        result["model_kind"] = "ema"
    elif change == "smoke":
        marker["smoke"] = True
    write_json(output / "training_complete.json", marker)
    (output / "epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    with pytest.raises(ValueError):
        module.audit(spec, result, output)


def test_identical_epoch_replay_is_allowed_but_inconsistent_replay_is_not(completed):
    module, spec, result, output, _, history = completed
    replay = copy.deepcopy(history[0])
    (output / "epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [*history, replay]))
    module.audit(spec, result, output)
    replay["source_val_mae"] += 1
    (output / "epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [*history, replay]))
    with pytest.raises(ValueError, match="Replayed epoch"):
        module.audit(spec, result, output)


def test_legacy_results_are_rechecked_not_exempted_by_a_flag(completed, monkeypatch):
    module, spec, result, output, _, _ = completed
    result["legacy_version"] = "legacy_fixture"
    called = []
    def verify(actual_spec, version):
        called.append((actual_spec, version))
        return {**result, "selected_epoch": 0}
    monkeypatch.setitem(sys.modules, "build_benchmark_suite", SimpleNamespace(audit_legacy=verify))
    with pytest.raises(ValueError, match="upstream logs"):
        module.audit(spec, result, output)
    assert called == [(spec, "legacy_fixture")]
