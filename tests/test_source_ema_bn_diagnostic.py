from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_source_ema_bn.py"
    definition = importlib.util.spec_from_file_location("source_ema_bn_test", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def test_bn_recalibration_changes_only_statistics_and_keeps_dropout_disabled(diagnostic):
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(.9), nn.Conv2d(3, 1, 1)).eval()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    modes = []
    handle = model[1].register_forward_pre_hook(lambda module, _: modes.append(module.training))
    result = diagnostic.recalibrate_batch_norm(model, [torch.full((4, 3, 4, 4), 2.),
                                                     torch.full((4, 3, 4, 4), 4.)])
    handle.remove()
    assert result == {"images": 8, "batches": 2, "bn_layers": 1}
    assert modes == [False, False]
    assert torch.equal(model[0].running_mean, torch.full((3,), 3.))
    assert model[0].num_batches_tracked.item() == 2
    assert model[0].momentum == .1
    assert not any(module.training for module in model.modules())
    assert all(torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    assert all(value.grad is None for value in model.parameters())


def test_no_images_cannot_be_reported_as_success(diagnostic):
    model = nn.Sequential(nn.BatchNorm2d(3)).eval()
    with pytest.raises(ValueError, match="no source images"):
        diagnostic.recalibrate_batch_norm(model, [])
    assert not model.training and not model[0].training
    assert model[0].momentum == .1


def test_calibration_only_reads_images_in_declared_source_train(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    folder = tmp_path / "data/processed/mpcount/stb/train"
    folder.mkdir(parents=True)
    rows = []
    for index in range(4):
        path = folder / f"image{index}.png"
        Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)
        rows.append({"id": str(index), "image": str(path), "points": "DO_NOT_OPEN.npy",
                     "density": "DO_NOT_OPEN_dmap.npy", "count": object()})
    batches = list(diagnostic.source_image_batches(list(reversed(rows)), "stb", 4, 2, 4))
    assert len(batches) == 2 and batches[0].shape == (2, 3, 4, 4)
    assert torch.allclose(batches[0], torch.full_like(batches[0], 128 / 255 * 2 - 1), atol=1e-7)
    rows[0]["image"] = str(tmp_path / "data/processed/mpcount/sta/test/image0.png")
    with pytest.raises(ValueError, match="source training images"):
        list(diagnostic.source_image_batches(rows, "stb", 4, 2, 4))


@pytest.mark.parametrize("count", [0, 3, 6])
def test_invalid_calibration_budgets_are_rejected(diagnostic, count):
    with pytest.raises(ValueError, match="whole batch"):
        list(diagnostic.source_image_batches([{}] * 4, "stb", count, 2, 4))


def test_manifests_must_be_source_only(diagnostic):
    spec = {"name": "fixture", "source": "stb", "method": "mean_teacher",
            "train_manifest": "data/manifests/stb_train.jsonl",
            "val_manifest": "data/manifests/stb_val.jsonl"}
    diagnostic.validate_spec(spec)
    spec["val_manifest"] = "data/manifests/sta_test.jsonl"
    with pytest.raises(ValueError, match="declared source"):
        diagnostic.validate_spec(spec)


@pytest.mark.parametrize("change", ["smoke", "spec", "teacher", "unstarted", "beyond_budget"])
def test_only_matching_active_production_ema_snapshots_are_accepted(diagnostic, change):
    snapshot = {"spec_hash": "frozen", "smoke": False, "teacher": {}, "epoch": 2}
    if change == "smoke":
        snapshot["smoke"] = True
    elif change == "spec":
        snapshot["spec_hash"] = "other"
    elif change == "teacher":
        snapshot["teacher"] = None
    elif change == "unstarted":
        snapshot["epoch"] = -1
    else:
        snapshot["epoch"] = 180
    with pytest.raises(ValueError):
        diagnostic.validate_snapshot(snapshot, "frozen", 180)
