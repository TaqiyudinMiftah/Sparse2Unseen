#!/usr/bin/env python3
"""Expose MPCount's source-validation-selected checkpoint as best.pth.

MPCount saves the selected checkpoint as ``best_<epoch>.pth``, while its test
configurations conventionally refer to ``best.pth``. This helper reads the
completed training log and creates that compatibility link without copying a
large checkpoint or selecting a model using target-domain results.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "external" / "MPCount" / "logs" / "stb_100_seed2023"
BEST_EPOCH = re.compile(r"^Best epoch: (\d+), best criterion: .+$")


def select_checkpoint(run_dir: Path) -> Path:
    log_path = run_dir / "log.txt"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    starts = [
        index for index, line in enumerate(lines) if line.startswith("Start training at ")
    ]
    if not starts:
        raise RuntimeError(f"No training run found in {log_path}")
    start = starts[-1]
    completed = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if index > start and (match := BEST_EPOCH.match(line))
    ]
    if not completed:
        raise RuntimeError(f"No source-validation best epoch found in {log_path}")

    index, epoch = completed[-1]
    if not any(line.startswith("End training at ") for line in lines[index + 1 :]):
        raise RuntimeError(f"Training has not completed in {log_path}")

    checkpoint = run_dir / f"best_{epoch}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = select_checkpoint(run_dir)
    link = run_dir / "best.pth"
    if link.is_symlink():
        if link.resolve() != checkpoint.resolve():
            raise RuntimeError(f"Refusing to replace an existing checkpoint link: {link}")
    elif link.exists():
        raise RuntimeError(f"Refusing to replace an existing checkpoint file: {link}")
    else:
        link.symlink_to(checkpoint.name)
    print(f"Source-validation checkpoint: {link} -> {checkpoint.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
