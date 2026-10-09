from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import sparse2unseen.experiments.common as common
from sparse2unseen.data.density_dataset import DensityDataset
from sparse2unseen.data.manifest import Sample


def fixture_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "SOURCE_COUNTS", {"stb": (4, 1, 1)})
    root = tmp_path / "data/processed/mpcount/stb"
    rows = []
    for phase, ids in (("train", range(4)), ("val", [4])):
        directory = root / phase
        directory.mkdir(parents=True)
        part = []
        for index in ids:
            row = {"id": f"IMG_{index}", "image": str(directory / f"IMG_{index}.jpg"),
                   "points": str(directory / f"IMG_{index}.npy"), "density": str(directory / f"IMG_{index}_dmap.npy"), "count": 10}
            for field in ("image", "points", "density"):
                Path(row[field]).write_bytes(b"fixture")
            part.append(row)
        common.write_json(tmp_path / f"{phase}.json", part)
        (tmp_path / f"{phase}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in part))
        if phase == "train":
            rows = part
    split = {"labeled_ids": ["IMG_0", "IMG_1"], "unlabeled_ids": ["IMG_2", "IMG_3"]}
    common.write_json(tmp_path / "split.json", split)
    sparse = tmp_path / "sparse/train"
    sparse.mkdir(parents=True)
    for row in rows[:2]:
        (sparse / Path(row["image"]).name).symlink_to(row["image"])
    return {"source": "stb", "method": "mean_teacher", "train_manifest": "train.jsonl",
            "val_manifest": "val.jsonl", "split": "split.json", "labeled_root": "sparse",
            "labeled_images": 2, "unlabeled_images": 2}


def test_source_partitions_are_checked_before_training(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path, monkeypatch)
    result = common.check_source_inputs(spec)
    assert len(result["split"]["labeled_ids"]) == 2
    split = common.load_json(tmp_path / "split.json")
    split["unlabeled_ids"][0] = "IMG_0"
    common.write_json(tmp_path / "split.json", split)
    with pytest.raises(ValueError, match="partition"):
        common.check_source_inputs(spec)


def test_target_image_path_in_training_manifest_is_rejected(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path, monkeypatch)
    path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["image"] = str(tmp_path / "data/processed/mpcount/sta/train/target.jpg")
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="declared source"):
        common.check_source_inputs(spec)


def test_unlabeled_dataset_never_loads_annotation_files(tmp_path, monkeypatch):
    image = tmp_path / "unlabeled.jpg"
    Image.new("RGB", (320, 320), "white").save(image)
    monkeypatch.setattr(np, "load", lambda *_, **__: pytest.fail("Unlabeled annotations were opened"))
    sample = Sample("unlabeled", str(image), str(tmp_path / "forbidden.npy"), str(tmp_path / "forbidden_dmap.npy"), 0)
    dataset = DensityDataset([sample], None, 320, labeled=False, train=True)
    weak, strong, identifier = dataset[0]
    assert weak.shape == strong.shape == (3, 320, 320)
    assert identifier == "unlabeled"


def test_partial_training_cannot_trigger_target_evaluation(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_benchmark.py"
    definition = importlib.util.spec_from_file_location("test_evaluate_benchmark", script)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "check_source_inputs", lambda _: None)
    monkeypatch.setattr(module, "make_model", lambda *_, **__: pytest.fail("Must not construct model/open targets"))
    spec_path = tmp_path / "spec.json"
    common.write_json(spec_path, {"name": "partial", "settings": {"epochs": 180}})
    common.write_json(tmp_path / "runs/benchmark_v1/jobs/partial/training_complete.json", {"epochs": 149, "smoke": False})
    with pytest.raises(RuntimeError, match="completed"):
        module.evaluate(spec_path)
