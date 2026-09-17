from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .manifest import Sample


def make_random_split(samples: list[Sample], fraction: float, seed: int) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ids = [s.id for s in samples]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_labeled = len(ids) if fraction == 1 else max(1, math.floor(len(ids) * fraction))
    labeled = sorted(ids[:n_labeled])
    unlabeled = sorted(ids[n_labeled:])
    return {
        "strategy": "random",
        "seed": seed,
        "fraction": fraction,
        "n_total": len(ids),
        "n_labeled": len(labeled),
        "labeled_ids": labeled,
        "unlabeled_ids": unlabeled,
    }


def save_split(split: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)
        f.write("\n")


def load_split(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    required = {"labeled_ids", "unlabeled_ids", "seed", "fraction"}
    missing = required.difference(split)
    if missing:
        raise ValueError(f"Split missing fields: {sorted(missing)}")
    overlap = set(split["labeled_ids"]) & set(split["unlabeled_ids"])
    if overlap:
        raise ValueError(f"Split contains labeled/unlabeled overlap: {sorted(overlap)[:5]}")
    return split
