from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_COUNTS = {"stb": (320, 80, 316), "sta": (240, 60, 182), "qnrf": (1081, 120, 334)}
METHODS = ("label_only", "mean_teacher", "mpcount", "ssl_dg", "domain_stable")
SSL_METHODS = ("mean_teacher", "ssl_dg", "domain_stable")
DG_METHODS = ("mpcount", "ssl_dg", "domain_stable")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_source_inputs(spec: dict) -> dict:
    """Validate the source partition without opening any target files."""
    source = spec["source"]
    if source not in SOURCE_COUNTS or spec["method"] not in METHODS:
        raise ValueError("Unknown source or method")
    pool = [json.loads(line) for line in (ROOT / spec["train_manifest"]).read_text().splitlines() if line]
    validation = [json.loads(line) for line in (ROOT / spec["val_manifest"]).read_text().splitlines() if line]
    split = load_json(ROOT / spec["split"])
    train_ids = [row["id"] for row in pool]
    selected = split["labeled_ids"]
    unselected = split["unlabeled_ids"]
    if len(train_ids) != SOURCE_COUNTS[source][0] or len(set(train_ids)) != len(train_ids):
        raise ValueError("Source training manifest count/IDs do not match the protocol")
    if len(validation) != SOURCE_COUNTS[source][1] or set(train_ids) & {row["id"] for row in validation}:
        raise ValueError("Source validation partition is incorrect or overlaps training")
    if len(set(selected + unselected)) != len(selected + unselected) or set(selected + unselected) != set(train_ids):
        raise ValueError("Labeled and unlabeled IDs must partition the entire source pool")
    if len(selected) != spec["labeled_images"] or len(unselected) != spec["unlabeled_images"]:
        raise ValueError("Annotation budget does not match the frozen experiment")
    source_dir = (ROOT / f"data/processed/mpcount/{source}").resolve()
    for phase, rows in (("train", pool), ("val", validation)):
        for row in rows:
            if Path(row["image"]).resolve().parent != source_dir / phase:
                raise ValueError("Manifest image is outside the declared source partition")
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            if phase == "val" or row["id"] in set(selected):
                for field in ("points", "density"):
                    path = Path(row[field]).resolve()
                    if path.parent != source_dir / phase or not path.is_file():
                        raise ValueError("Labeled annotations do not belong to the source partition")
    images = list((ROOT / spec["labeled_root"] / "train").glob("*.jpg"))
    images += list((ROOT / spec["labeled_root"] / "train").glob("*.png"))
    if {path.stem for path in images} != set(selected) or len(images) != len(selected):
        raise ValueError("Materialized source root does not match the frozen labeled subset")
    for path, expected in spec.get("input_sha256", {}).items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen experiment input changed: {path}")
    return {"pool": pool, "validation": validation, "split": split}
