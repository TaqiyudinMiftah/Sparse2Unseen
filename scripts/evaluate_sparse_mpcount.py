#!/usr/bin/env python3
"""Evaluate a source-selected sparse MPCount checkpoint on fixed test domains."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import yaml

from select_mpcount_checkpoint import select_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPCOUNT_ROOT = PROJECT_ROOT / "external/MPCount"
sys.path.insert(0, str(MPCOUNT_ROOT))

from main import load_config  # noqa: E402
from trainers.dgtrainer import DGTrainer  # noqa: E402


def validate_run_version(seed: int, version: str | None) -> str:
    canonical = f"stb_10_seed{seed}_effbs16"
    version = version or canonical
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version):
        raise ValueError("Run version must be a single safe path component")
    if version != canonical and not version.startswith(canonical + "_"):
        raise ValueError("Run version does not match the sparse seed")
    return version


def build_eval_config(seed: int, domain: str, checkpoint: Path, run_version: str | None = None) -> dict:
    if seed not in (1, 2, 3):
        raise ValueError("Expected a committed sparse split seed: 1, 2, or 3")
    if domain not in ("stb", "sta", "qnrf"):
        raise ValueError("Unknown test domain")
    run_version = validate_run_version(seed, run_version)
    template = PROJECT_ROOT / f"configs/mpcount/stb_100_effbs16_test_{domain}.yml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["seed"] = seed
    config["version"] = f"{run_version}_test_{domain}"
    if domain == "qnrf":
        config["version"] += "_ps1024"
    config["checkpoint"] = str(checkpoint.resolve(strict=True))
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--run-version", help="Explicit clean-retry run name; defaults to the canonical run")
    parser.add_argument(
        "--domain", choices=("stb", "sta", "qnrf", "all"), default="all"
    )
    args = parser.parse_args()

    run_version = validate_run_version(args.seed, args.run_version)
    train_dir = MPCOUNT_ROOT / "logs" / run_version
    checkpoint = select_checkpoint(train_dir)
    domains = ("stb", "sta", "qnrf") if args.domain == "all" else (args.domain,)
    configs = [build_eval_config(args.seed, domain, checkpoint, run_version) for domain in domains]
    for config in configs:
        destination = MPCOUNT_ROOT / "logs" / config["version"]
        if destination.exists():
            raise FileExistsError(f"Refusing to reuse test log directory: {destination}")

    os.chdir(MPCOUNT_ROOT)
    for config in configs:
        destination = MPCOUNT_ROOT / "logs" / config["version"]
        destination.mkdir(parents=True)
        config_path = destination / "config.yml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        init_params, task_params = load_config(str(config_path), "test")
        trainer = DGTrainer(**init_params)
        trainer.test(**task_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
