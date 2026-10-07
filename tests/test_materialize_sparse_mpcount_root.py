from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/materialize_sparse_mpcount_root.py"
spec = importlib.util.spec_from_file_location("materialize_sparse_mpcount_root", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def example_source(tmp_path: Path):
    source = tmp_path / "source"
    manifest = tmp_path / "manifest.jsonl"
    split = tmp_path / "split.json"
    rows = []
    for sample_id in ("IMG_1", "IMG_2", "IMG_3"):
        train = source / "train"
        train.mkdir(parents=True, exist_ok=True)
        files = {
            "image": train / f"{sample_id}.jpg",
            "points": train / f"{sample_id}.npy",
            "density": train / f"{sample_id}_dmap.npy",
        }
        for path in files.values():
            path.write_bytes(b"test")
        rows.append({"id": sample_id, **{key: str(path) for key, path in files.items()}})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split.write_text(json.dumps({
        "labeled_ids": ["IMG_1", "IMG_3"], "unlabeled_ids": ["IMG_2"],
        "n_total": 3, "n_labeled": 2, "fraction": 2 / 3, "seed": 1,
    }), encoding="utf-8")
    for phase in ("val", "test"):
        phase_dir = source / phase
        phase_dir.mkdir()
        (phase_dir / "IMG_4.jpg").write_bytes(b"test")
        (phase_dir / "IMG_4.npy").write_bytes(b"test")
        (phase_dir / "IMG_4_dmap.npy").write_bytes(b"test")
    return source, manifest, split


def test_materializes_only_labeled_train_and_preserves_source_eval(tmp_path):
    source, manifest, split = example_source(tmp_path)
    output = tmp_path / "sparse"
    counts = module.materialize(source, manifest, split, output)
    assert counts == {"labeled_train_images": 2, "val_images": 1, "test_images": 1}
    assert len(list((output / "train").iterdir())) == 6
    assert not (output / "train/IMG_2.jpg").exists()
    assert (output / "train/IMG_1_dmap.npy").is_symlink()
    assert (output / "train/IMG_1_dmap.npy").resolve() == (source / "train/IMG_1_dmap.npy")
    for phase in ("val", "test"):
        assert (output / phase / "IMG_4.jpg").is_symlink()
        assert (output / phase / "IMG_4.npy").is_symlink()


def test_rejects_existing_output_without_changing_it(tmp_path):
    source, manifest, split = example_source(tmp_path)
    output = tmp_path / "sparse"
    output.mkdir()
    marker = output / "user_file"
    marker.write_text("leave me alone", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.materialize(source, manifest, split, output)
    assert marker.read_text(encoding="utf-8") == "leave me alone"


def test_rejects_invalid_split_before_creating_output(tmp_path):
    source, manifest, split = example_source(tmp_path)
    selection = json.loads(split.read_text(encoding="utf-8"))
    selection["unlabeled_ids"] = ["IMG_1"]
    split.write_text(json.dumps(selection), encoding="utf-8")
    output = tmp_path / "sparse"
    with pytest.raises(ValueError, match="partition"):
        module.materialize(source, manifest, split, output)
    assert not output.exists()


def test_rejects_missing_density_before_creating_output(tmp_path):
    source, manifest, split = example_source(tmp_path)
    (source / "train/IMG_3_dmap.npy").unlink()
    output = tmp_path / "sparse"
    with pytest.raises(FileNotFoundError):
        module.materialize(source, manifest, split, output)
    assert not output.exists()
