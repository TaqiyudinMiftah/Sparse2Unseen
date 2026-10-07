#!/usr/bin/env python3
"""Report audited completed benchmark results and all remaining matrix entries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from sparse2unseen.experiments.common import ROOT, SOURCE_COUNTS, SSL_METHODS, digest, load_json, now, write_json


def audit_predictions(spec, result, output):
    """Recompute finalized test metrics; never called for an unfinished run."""
    for domain, values in result["domains"].items():
        cache = load_json(output / f"test_{domain}.json")
        if cache["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("Test predictions belong to a different checkpoint")
        if cache["patch_size"] != spec["settings"]["test_patch_sizes"][domain]:
            raise ValueError("Test tiling differs from the fixed evaluation recipe")
        records = cache["predictions"]
        if len(records) != values["n"] or cache["n"] != values["n"]:
            raise ValueError("Per-image test prediction count differs")
        ids = [row["id"] for row in records]
        manifest = [json.loads(line) for line in
                    (ROOT / f"data/manifests/{domain}_test.jsonl").read_text().splitlines() if line]
        truth = {row["id"]: row["count"] for row in manifest}
        if len(set(ids)) != len(ids) or set(ids) != set(truth) or len(manifest) != len(truth):
            raise ValueError("Test predictions do not cover the exact manifest IDs")
        errors = []
        for row in records:
            if row["count"] != truth[row["id"]] or not math.isfinite(row["predicted"]):
                raise ValueError("Invalid per-image prediction or ground-truth count")
            errors.append(row["predicted"] - row["count"])
        mse = sum(error**2 for error in errors) / len(errors)
        metrics = {"mae": sum(abs(error) for error in errors) / len(errors),
                   "mse": mse, "rmse": math.sqrt(mse)}
        for key, expected in metrics.items():
            if not all(math.isclose(row[key], expected, rel_tol=1e-10, abs_tol=1e-9)
                       for row in (cache, values)):
                raise ValueError("Test metric does not match the saved per-image predictions")


def audit(spec, result, output):
    if result["experiment"] != spec["name"]:
        raise ValueError("Result experiment identity differs")
    for key in ("source", "method", "fraction", "seed"):
        if result[key] != spec[key]:
            raise ValueError("Result metadata differs from its frozen experiment")
    if result["selection_domain"] != spec["source"] + "_val":
        raise ValueError("Checkpoint selection did not use the declared source validation")
    if set(result["domains"]) != set(SOURCE_COUNTS):
        raise ValueError("Evaluation did not cover the fixed test matrix")
    for domain, values in result["domains"].items():
        if values["n"] != SOURCE_COUNTS[domain][2]:
            raise ValueError("Test count differs from the declared dataset partition")
        if any(not math.isfinite(values[metric]) or values[metric] < 0
               for metric in ("mae", "mse", "rmse")):
            raise ValueError("Invalid/non-finite test metric")
        if not math.isclose(values["rmse"]**2, values["mse"], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("RMSE does not agree with mean squared error")
    if digest(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
        raise ValueError("Result checkpoint hash differs")
    if "legacy_version" in result:
        from build_benchmark_suite import audit_legacy
        verified = audit_legacy(spec, result["legacy_version"])
        for key in ("checkpoint_sha256", "selected_epoch", "domains"):
            if verified[key] != result[key]:
                raise ValueError("Legacy result differs from the audited upstream logs")
    else:
        if Path(result["checkpoint"]).resolve() != (output / "best.pt").resolve():
            raise ValueError("Test result did not use this run's selected checkpoint")
        if load_json(output / "spec.json") != spec:
            raise ValueError("Saved training recipe differs from its frozen spec")
        marker = load_json(output / "training_complete.json")
        if marker["smoke"] or marker["epochs"] != spec["settings"]["epochs"]:
            raise ValueError("Incomplete/smoke training cannot be reported")
        if marker["checkpoint_sha256"] != result["checkpoint_sha256"] or marker["spec_sha256"] != digest(output / "spec.json"):
            raise ValueError("Training-completion checkpoint/config audit differs")
        if marker["selected_epoch"] != result["selected_epoch"]:
            raise ValueError("Tested epoch differs from the source-selected epoch")
        kind = "ema" if spec["method"] in SSL_METHODS else "student"
        if marker["model_kind"] != kind or result["model_kind"] != kind:
            raise ValueError("Tested model is not the predeclared student/EMA variant")
        if marker["experiment"] != spec["name"] or marker["selection_domain"] != result["selection_domain"]:
            raise ValueError("Completion marker belongs to a different selection protocol")
        history = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines() if line]
        epochs = {}
        for row in history:
            # An interrupted epoch may be logged before its resume snapshot saves.
            # Replay must agree on the scientific measurements, not wall-clock time.
            if row["epoch"] in epochs:
                keys = ("source_val_mae", "source_val_rmse", "optimizer_updates", "labeled_samples", "unlabeled_samples")
                if any(row[key] != epochs[row["epoch"]][key] for key in keys):
                    raise ValueError("Replayed epoch has inconsistent measurements")
            epochs[row["epoch"]] = row
        if set(epochs) != set(range(spec["settings"]["epochs"])):
            raise ValueError("Training history does not cover every declared epoch")
        for row in epochs.values():
            expected_unlabeled = spec["samples_per_epoch"] if kind == "ema" and row["epoch"] >= spec["settings"]["warmup_epochs"] else 0
            if (row["optimizer_updates"] != spec["updates_per_epoch"] or
                row["labeled_samples"] != spec["samples_per_epoch"] or
                row["unlabeled_samples"] != expected_unlabeled):
                raise ValueError("Completed training exposure/update budget differs")
            if not math.isfinite(row["source_val_mae"]):
                raise ValueError("Non-finite source validation measurement")
        selected = min(epochs.values(), key=lambda row: (row["source_val_mae"], row["epoch"]))
        if marker["selected_epoch"] != selected["epoch"] or marker["source_val_mae"] != selected["source_val_mae"]:
            raise ValueError("Checkpoint was not selected by minimum source validation MAE")
        audit_predictions(spec, result, output)
    return result


def report(require_complete=False):
    suite = ROOT / "runs/benchmark_v1"
    matrix = load_json(suite / "matrix.json")
    state_path = suite / "queue_state.json"
    state = load_json(state_path) if state_path.exists() else {"jobs": matrix["jobs"]}
    statuses = {job["name"]: job for job in state["jobs"]}
    completed, remaining, groups = [], [], defaultdict(list)
    for entry in matrix["jobs"]:
        spec = load_json(ROOT / entry["spec"])
        output = suite / "jobs" / entry["name"]
        path = output / "results.json"
        if path.exists():
            result = audit(spec, load_json(path), output)
            completed.append(result)
            groups[(spec["source"], spec["fraction"], spec["method"])].append(result)
        else:
            status = statuses[entry["name"]]
            remaining.append({"name": entry["name"], "state": status["state"], "error": status.get("error")})
    if require_complete and remaining:
        raise RuntimeError(f"Conclusion gate: {len(remaining)} primary training/evaluation entries remain incomplete")
    aggregate = []
    for (source, fraction, method), entries in sorted(groups.items()):
        expected = 1 if fraction == 1 else 3
        summary = {"source": source, "fraction": fraction, "method": method,
                   "completed_seeds": [row["seed"] for row in entries], "expected_seeds": expected,
                   "all_seeds_complete": len(entries) == expected, "domains": {}}
        if len(entries) == expected:
            for domain in SOURCE_COUNTS:
                summary["domains"][domain] = {
                    metric: {"mean": statistics.mean(row["domains"][domain][metric] for row in entries),
                             "sample_sd": statistics.stdev(row["domains"][domain][metric] for row in entries) if len(entries)>1 else None}
                    for metric in ("mae", "rmse")}
        aggregate.append(summary)
    payload = {"updated_at": now(), "primary_total": len(matrix["jobs"]), "completed_count": len(completed),
               "remaining_count": len(remaining), "completed": completed, "remaining": remaining, "aggregate": aggregate}
    write_json(ROOT / "reports/benchmark_results.json", payload)
    lines = ["# Complete benchmark progress", "", f"Updated {payload['updated_at']}.", "",
             f"Audited training + all-test completion: **{len(completed)} / {len(matrix['jobs'])}**.", "",
             "The final conclusion is pending the complete primary matrix. The protocol is in",
             "[BENCHMARK_SUITE.md](../docs/BENCHMARK_SUITE.md). JHU remains the later external extension.", "",
             "## Completed training results", "", "Scores are MAE / RMSE. QNRF uses fixed 1024 tiles.", "",
             "| Source | Labels | Method | Seed | STB | STA | QNRF |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in completed:
        values = [f"{row['domains'][domain]['mae']:.4f} / {row['domains'][domain]['rmse']:.4f}" for domain in SOURCE_COUNTS]
        lines.append(f"| {row['source']} | {row['fraction']:.0%} | {row['method']} | {row['seed']} | " + " | ".join(values) + " |")
    lines += ["", "## Complete seed aggregates", "", "Only groups with all their required seeds appear here.", "",
              "| Source | Labels | Method | STB MAE ± sample SD | STA MAE ± sample SD | QNRF MAE ± sample SD |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for group in aggregate:
        if not group["all_seeds_complete"]:
            continue
        cells = []
        for domain in SOURCE_COUNTS:
            values = group["domains"][domain]["mae"]
            cells.append(f"{values['mean']:.4f} ± {values['sample_sd']:.4f}" if values["sample_sd"] is not None else f"{values['mean']:.4f} (single seed)")
        lines.append(f"| {group['source']} | {group['fraction']:.0%} | {group['method']} | " + " | ".join(cells) + " |")
    lines += ["", "## Remaining experiments", "", "| Experiment | Current state |", "| --- | --- |"]
    for row in remaining:
        lines.append(f"| {row['name']} | {row['state']} |")
    temporary = ROOT / "reports/benchmark_progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(ROOT / "reports/benchmark_progress.md")
    print(f"Audited {len(completed)}/{len(matrix['jobs'])} completed experiments; {len(remaining)} remaining")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report(args.require_complete)
