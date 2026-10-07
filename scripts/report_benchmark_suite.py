#!/usr/bin/env python3
"""Report audited completed benchmark results and all remaining matrix entries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import statistics

from sparse2unseen.experiments.common import ROOT, SOURCE_COUNTS, digest, load_json, now, write_json


def audit(spec, result, output):
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
    from pathlib import Path
    if digest(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
        raise ValueError("Result checkpoint hash differs")
    if "legacy_version" not in result:
        marker = load_json(output / "training_complete.json")
        if marker["smoke"] or marker["epochs"] != spec["settings"]["epochs"]:
            raise ValueError("Incomplete/smoke training cannot be reported")
        if marker["checkpoint_sha256"] != result["checkpoint_sha256"] or marker["spec_sha256"] != digest(output / "spec.json"):
            raise ValueError("Training-completion checkpoint/config audit differs")
        if marker["selected_epoch"] != result["selected_epoch"]:
            raise ValueError("Tested epoch differs from the source-selected epoch")
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
