#!/usr/bin/env python3
"""Freeze the documented benchmark matrix and validate existing completed runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import yaml

from sparse2unseen.data.manifest import read_manifest
from sparse2unseen.data.splits import make_random_split, save_split
from sparse2unseen.experiments.common import ROOT, SOURCE_COUNTS, check_source_inputs, digest, load_json, now, write_json

sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
from materialize_sparse_mpcount_root import materialize
from select_mpcount_checkpoint import select_checkpoint

SUITE = ROOT / "runs/benchmark_v1"
CODE_INPUTS = [
    "configs/suite/matrix.yaml", "uv.lock", "scripts/train_benchmark.py",
    "scripts/evaluate_benchmark.py", "scripts/run_benchmark_suite.py",
    "scripts/build_benchmark_suite.py", "src/sparse2unseen/experiments/common.py",
    "src/sparse2unseen/experiments/sampling.py", "src/sparse2unseen/experiments/training.py",
    "src/sparse2unseen/experiments/inference.py", "src/sparse2unseen/data/density_dataset.py",
    "src/sparse2unseen/confidence/region_stability.py", "src/sparse2unseen/dg/augment.py",
    "src/sparse2unseen/ssl/ema.py", "src/sparse2unseen/losses.py",
]


def audit_legacy(spec: dict, version: str) -> dict:
    root = ROOT / "external/MPCount/logs" / version
    checkpoint = select_checkpoint(root)
    text = (root / "log.txt").read_text()
    epoch = int(checkpoint.stem.split("_")[-1])
    if set(map(int, re.findall(r"^Epoch (\d+): accumulation=4, microbatches=80, optimizer steps=20$", text, re.M))) != set(range(180)):
        raise ValueError("Legacy run did not complete the fixed update budget")
    config_file = next(root.glob("*.yml"))
    train_config = yaml.safe_load(config_file.read_text())
    for phase in ("train", "val"):
        source_root = train_config[f"{phase}_dataset"]["params"]["root"]
        if not source_root.startswith("data/stb"):
            raise ValueError("Legacy model was not trained/selected on STB only")
    test_results = {}
    for domain, counts in SOURCE_COUNTS.items():
        suffix = f"_test_{domain}" + ("_ps1024" if domain == "qnrf" else "")
        test_root = ROOT / "external/MPCount/logs" / (version + suffix)
        config = yaml.safe_load((test_root / "config.yml").read_text()) if (test_root / "config.yml").is_file() else None
        if config is None:
            # Earlier full-label evaluations copy the project-owned config name.
            configs = list(test_root.glob("*.yml"))
            config = yaml.safe_load(configs[0].read_text()) if configs else yaml.safe_load((ROOT / f"configs/mpcount/stb_100_effbs16_test_{domain}.yml").read_text())
        if config["test_dataset"]["params"]["root"] != f"data/{domain}":
            raise ValueError("Legacy test used the wrong domain")
        if config["patch_size"] != spec["settings"]["test_patch_sizes"][domain]:
            raise ValueError("Legacy inference protocol differs")
        log = (test_root / "log.txt").read_text()
        if "End testing at " not in log:
            raise ValueError("Legacy test is incomplete")
        loaded = re.search(r"^Loading checkpoint from (.+)$", log, re.M)
        if loaded is None:
            raise ValueError("Legacy test lacks a loaded-checkpoint audit")
        loaded_path = Path(loaded.group(1))
        if not loaded_path.is_absolute():
            loaded_path = ROOT / "external/MPCount" / loaded_path
        if loaded_path.resolve() != checkpoint.resolve():
            raise ValueError("Legacy test did not use the source-selected weights")
        match = re.search(r"Testing results: mae: ([\d.]+) mse: ([\d.]+)", log)
        if not match:
            raise ValueError("Missing legacy test metrics")
        mae, mse = map(float, match.groups())
        test_results[domain] = {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "n": counts[2]}
    return {"experiment": spec["name"], "source": spec["source"], "method": spec["method"],
            "fraction": spec["fraction"], "seed": spec["seed"], "selected_epoch": epoch,
            "selection_domain": "stb_val", "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint), "domains": test_results,
            "legacy_version": version, "imported_at": now(), "legacy_train_workers": 16}


def build(*, preflight=False) -> dict:
    settings = yaml.safe_load((ROOT / "configs/suite/matrix.yaml").read_text())
    jobs = []
    frozen = {path: digest(ROOT / path) for path in CODE_INPUTS}
    for folder in ("models", "datasets", "utils"):
        for path in sorted((ROOT / "external/MPCount" / folder).glob("*.py")):
            frozen[str(path.relative_to(ROOT))] = digest(path)
    source_samples = {source: read_manifest(ROOT / f"data/manifests/{source}_train.jsonl") for source in settings["sources"]}
    for source, samples in source_samples.items():
        if len(samples) != SOURCE_COUNTS[source][0]:
            raise ValueError("Source preprocessing counts do not match the protocol")
    # First complete the STB 10% ladder, then other fractions and source domains.
    combinations = [(source, fraction, seed, method)
                    for source in settings["sources"] for fraction in settings["fractions"]
                    for method in settings["methods"] for seed in settings["seeds"]]
    combinations += [(source, 1.0, settings["full_label_seed"], method)
                     for source in settings["sources"] for method in settings["full_label_methods"]]
    if preflight:
        combinations = [("stb", 0.10, 1, method) for method in settings["methods"] if method != "mpcount"]
    suite_dir = ROOT / "runs/benchmark_preflight" if preflight else SUITE
    for source, fraction, seed, method in combinations:
        percent = round(fraction * 100)
        name = f"{source}_{percent}_{method}_seed{seed}_v1"
        split_rel = f"splits/{source}_{percent}_seed{seed}.json"
        split_path = ROOT / split_rel
        expected = make_random_split(source_samples[source], fraction, seed)
        if split_path.exists():
            if load_json(split_path) != expected:
                raise ValueError(f"Existing split differs: {split_path}")
        else:
            save_split(expected, split_path)
        root_rel = f"external/MPCount/data/{source}"
        if fraction < 1:
            root_rel += f"_sparse{percent}_seed{seed}"
            if not (ROOT / root_rel).exists():
                materialize(ROOT / f"data/processed/mpcount/{source}",
                            ROOT / f"data/manifests/{source}_train.jsonl", split_path, ROOT / root_rel)
        n_images = len(source_samples[source])
        samples_per_epoch = math.ceil(n_images / 16) * 16
        spec = {"name": name, "source": source, "fraction": fraction, "seed": seed,
                "method": method, "split": split_rel, "labeled_root": root_rel,
                "train_manifest": f"data/manifests/{source}_train.jsonl",
                "val_manifest": f"data/manifests/{source}_val.jsonl",
                "labeled_images": expected["n_labeled"], "unlabeled_images": len(expected["unlabeled_ids"]),
                "samples_per_epoch": samples_per_epoch, "updates_per_epoch": samples_per_epoch // 16,
                "settings": settings, "input_sha256": {**frozen, split_rel: digest(split_path)},
                "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "external/MPCount", text=True).strip()}
        check_source_inputs(spec)
        destination = suite_dir / "specs" / f"{name}.json"
        if destination.exists() and not preflight and load_json(destination) != spec:
            raise ValueError(f"Refusing to change a frozen experiment: {name}")
        write_json(destination, spec)
        job = {"name": name, "spec": str(destination.relative_to(ROOT)), "state": "pending", "attempts": 0}
        legacy = None
        if source == "stb" and method == "mpcount":
            if fraction == 0.10:
                legacy = f"stb_10_seed{seed}_effbs16" + ("_retry1" if seed == 3 else "")
            elif fraction == 1:
                legacy = "stb_100_seed2023_effbs16"
        if legacy and not preflight and (ROOT / "external/MPCount/logs" / legacy).exists():
            result = audit_legacy(spec, legacy)
            write_json(SUITE / "jobs" / name / "results.json", result)
            job["state"] = "imported_complete"
        jobs.append(job)
    matrix = {"created_at": now(), "protocol_version": 1, "jobs": jobs,
              "settings": settings, "inputs": frozen, "primary_run_count": len(jobs),
              "external_extension": "JHU-CROWD++ is deferred until the A/B/Q protocol is stable, as specified in docs/PROTOCOL.md."}
    if (suite_dir / "matrix.json").exists() and not preflight:
        raise FileExistsError("Benchmark matrix already exists; use its frozen inputs")
    write_json(suite_dir / "matrix.json", matrix)
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Prepare only isolated STB smoke configs")
    args = parser.parse_args()
    matrix = build(preflight=args.preflight)
    print(f"Frozen {len(matrix['jobs'])} training experiments; imported "
          f"{sum(job['state'] == 'imported_complete' for job in matrix['jobs'])} completed references")
