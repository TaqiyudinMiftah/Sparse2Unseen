#!/usr/bin/env python3
"""Expose a validated sparse MPCount source root without duplicating images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {dst}")
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(strict=True), dst)


def _read_manifest(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("Manifest is empty or contains duplicate IDs")
    return by_id


def _validated_sources(source: Path, manifest: Path, split: Path) -> tuple[list[Path], dict[str, list[Path]], int]:
    rows = _read_manifest(manifest)
    selection = json.loads(split.read_text(encoding="utf-8"))
    labeled = selection["labeled_ids"]
    unlabeled = selection["unlabeled_ids"]
    all_ids = labeled + unlabeled
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(rows):
        raise ValueError("Split IDs must partition the manifest exactly")
    if selection["n_total"] != len(rows) or selection["n_labeled"] != len(labeled):
        raise ValueError("Split counts do not match the manifest")
    if not labeled:
        raise ValueError("Sparse training set is empty")

    train_dir = source / "train"
    train_files: list[Path] = []
    for sample_id in labeled:
        row = rows[sample_id]
        for key in ("image", "points", "density"):
            manifest_file = Path(row[key]).resolve(strict=True)
            source_file = train_dir / manifest_file.name
            if not source_file.is_file() or source_file.resolve() != manifest_file:
                raise ValueError(f"Manifest {key} does not belong to source train: {sample_id}")
            train_files.append(source_file)

    phases: dict[str, list[Path]] = {}
    for phase in ("val", "test"):
        phase_dir = source / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(f"Missing source {phase} directory: {phase_dir}")
        entries = [entry for entry in phase_dir.iterdir() if entry.is_file()]
        images = [entry for entry in entries if entry.suffix.lower() in {".jpg", ".png"}]
        if not images or any(not image.with_suffix(".npy").is_file() for image in images):
            raise ValueError(f"Source {phase} images or annotations are incomplete")
        phases[phase] = entries
    return train_files, phases, len(labeled)


def materialize(
    source_root: Path, manifest: Path, split: Path, output_root: Path, copy: bool = False
) -> dict[str, int]:
    source = source_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to reuse existing output root: {output}")
    train_files, phases, n_labeled = _validated_sources(source, manifest, split)

    output.mkdir(parents=True)
    for src in train_files:
        link_or_copy(src, output / "train" / src.name, copy)
    for phase, files in phases.items():
        for src in files:
            link_or_copy(src, output / phase / src.name, copy)
    return {
        "labeled_train_images": n_labeled,
        "val_images": sum(path.suffix.lower() in {".jpg", ".png"} for path in phases["val"]),
        "test_images": sum(path.suffix.lower() in {".jpg", ".png"} for path in phases["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    counts = materialize(
        args.source_root, args.manifest, args.split, args.output_root, args.copy
    )
    print(f"Materialized sparse MPCount root at {args.output_root.resolve()}")
    print(
        f"Labeled train IDs: {counts['labeled_train_images']}; "
        f"source val/test images: {counts['val_images']}/{counts['test_images']}"
    )


if __name__ == "__main__":
    main()
