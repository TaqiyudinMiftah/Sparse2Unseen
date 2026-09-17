#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_density(image: Path) -> Path | None:
    stem = image.stem
    candidates = [
        image.with_name(stem + "_dmap.npy"),
        image.with_name(stem + "_dmap2.npy"),
        image.with_name(stem + "_density.npy"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Processed dataset root containing train/val/test")
    parser.add_argument("--phase", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    phase_dir = root / args.phase
    images = sorted(list(phase_dir.glob("*.jpg")) + list(phase_dir.glob("*.png")))
    if not images:
        raise SystemExit(f"No images found in {phase_dir}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for image in images:
            points = image.with_suffix(".npy")
            if not points.exists():
                raise FileNotFoundError(f"Missing point annotation: {points}")
            pts = np.load(points)
            density = find_density(image)
            row = {
                "id": image.stem,
                "image": str(image),
                "points": str(points),
                "density": str(density) if density else None,
                "count": int(len(pts)),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(images)} samples to {out}")


if __name__ == "__main__":
    main()
