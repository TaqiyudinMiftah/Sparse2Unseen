#!/usr/bin/env python3
"""Evaluate completed, source-selected weights unchanged on the fixed test sets."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sparse2unseen.experiments.common import ROOT, SOURCE_COUNTS, check_source_inputs, digest, load_json, now, write_json
from sparse2unseen.experiments.inference import upstream, count_metrics
from sparse2unseen.experiments.training import make_model


def evaluate(spec_path: Path):
    spec = load_json(spec_path)
    check_source_inputs(spec)
    output = ROOT / "runs/benchmark_v1/jobs" / spec["name"]
    marker = load_json(output / "training_complete.json")
    if marker["smoke"] or marker["epochs"] != spec["settings"]["epochs"]:
        raise RuntimeError("Only completed predeclared training may be test-evaluated")
    checkpoint = output / "best.pt"
    if digest(checkpoint) != marker["checkpoint_sha256"] or digest(spec_path) != marker["spec_sha256"]:
        raise RuntimeError("Completed checkpoint/config audit differs")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    model = make_model(spec["method"], pretrained=False).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    _, _, dataset_type, _, _, _ = upstream()
    results = {}
    for domain, sizes in SOURCE_COUNTS.items():
        cache = output / f"test_{domain}.json"
        if cache.exists():
            result = load_json(cache)
            if result["checkpoint_sha256"] != marker["checkpoint_sha256"] or result["n"] != sizes[2]:
                raise RuntimeError("Cached test result belongs to a different checkpoint/partition")
        else:
            dataset = dataset_type(root=str(ROOT / f"data/processed/mpcount/{domain}"), crop_size=320,
                                   downsample=1, method="test", is_grey=False, unit_size=16, pre_resize=1)
            if len(dataset) != sizes[2]:
                raise ValueError("Test preprocessing count mismatch")
            loader = DataLoader(dataset, batch_size=1, num_workers=spec["settings"]["validation_workers"])
            result = count_metrics(model, loader, device, spec["settings"]["test_patch_sizes"][domain], spec["settings"]["log_para"])
            result["checkpoint_sha256"] = marker["checkpoint_sha256"]
            result["patch_size"] = spec["settings"]["test_patch_sizes"][domain]
            write_json(cache, result)
        results[domain] = {key: value for key, value in result.items() if key != "predictions"}
        print(f"{spec['name']} {domain}: MAE {result['mae']:.4f}, RMSE {result['rmse']:.4f}", flush=True)
    result = {"experiment": spec["name"], "source": spec["source"], "fraction": spec["fraction"],
              "method": spec["method"], "seed": spec["seed"], "selected_epoch": marker["selected_epoch"],
              "selection_domain": marker["selection_domain"], "model_kind": marker["model_kind"],
              "checkpoint": str(checkpoint), "checkpoint_sha256": marker["checkpoint_sha256"],
              "completed_at": now(), "domains": results}
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.spec)
