#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    with open(args.split, "r", encoding="utf-8") as f:
        split = json.load(f)
    labeled = set(split["labeled_ids"])

    # Sparse train phase: include every file sharing a labeled image stem prefix.
    train_src = source / "train"
    for sample_id in labeled:
        matches = list(train_src.glob(sample_id + ".*")) + list(train_src.glob(sample_id + "_*.npy"))
        for src in set(matches):
            if src.is_file():
                link_or_copy(src, output / "train" / src.name, args.copy)

    # Preserve val/test phases for source-domain evaluation and checkpoint selection.
    for phase in ["val", "test"]:
        phase_src = source / phase
        if not phase_src.exists():
            continue
        for src in phase_src.iterdir():
            if src.is_file():
                link_or_copy(src, output / phase / src.name, args.copy)

    print(f"Materialized sparse MPCount root at {output}")
    print(f"Labeled train IDs: {len(labeled)}")


if __name__ == "__main__":
    main()
