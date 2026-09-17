from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    id: str
    image: str
    points: str | None
    density: str | None
    count: float


def read_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    rows: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                rows.append(
                    Sample(
                        id=str(raw["id"]),
                        image=str(raw["image"]),
                        points=raw.get("points"),
                        density=raw.get("density"),
                        count=float(raw["count"]),
                    )
                )
            except KeyError as e:
                raise ValueError(f"Missing key {e} in {path}:{line_no}") from e
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    ids = [r.id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in manifest: {path}")
    return rows


def write_manifest(path: str | Path, samples: Iterable[Sample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__, sort_keys=True) + "\n")
