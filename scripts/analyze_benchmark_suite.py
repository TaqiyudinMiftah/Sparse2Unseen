#!/usr/bin/env python3
"""Generate post-evaluation statistics without changing the frozen recipes."""
from __future__ import annotations

import argparse

from sparse2unseen.benchmark_analysis import METRICS, analyze, expected_keys
from sparse2unseen.experiments.common import ROOT, load_json, write_json
from report_benchmark_suite import report


def cell(values):
    if values is None:
        return "undefined (zero reference)"
    text = f"{values['mean']:.4f}"
    return text + (f" ± {values['sample_sd']:.4f}" if values["sample_sd"] is not None else " (single seed)")


def markdown(data):
    lines = ["# Benchmark analysis", "", f"Updated {data['updated_at']}.", "",
             f"Status: {'complete' if data['complete'] else 'partial — final conclusion pending'}; "
             f"{data['completed_runs']}/{data['expected_runs']} audited runs.", "",
             "All statistics use source-selected, completed training and fixed test evaluations.",
             "Only complete seed groups and matched contrasts are displayed. Cells give",
             "**MAE / RMSE**, with sample SD across seeds, not standard error or pooled-image RMSE.", "",
             "## Complete groups", "", "| Source | Labels | Method | STB | STA | QNRF |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    domains = data["source_order"]
    for row in data["groups"]:
        if row["complete"]:
            cells = [" / ".join(cell(row["domains"][domain][metric]) for metric in METRICS) for domain in domains]
            lines.append(f"| {row['source']} | {row['fraction']:.0%} | {row['method']} | " + " | ".join(cells) + " |")
    lines += ["", "## Label-scarcity degradation", "",
              "Positive percentage = worse than the 100% reference. Full-label references have",
              "one seed (2023), so these are descriptive changes, not matched-seed significance tests.",
              "B1 uses the B0 full-label reference; B3/the prototype use B2. These supervised-family",
              "references are not separately trained 100% EMA models.", "",
              "| Source | Labels | Method | Full reference | STB Δ% | STA Δ% | QNRF Δ% |",
              "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for row in data["degradation"]:
        cells = [" / ".join(cell(row["domains"][domain][metric]["relative_increase_pct"]) for metric in METRICS)
                 for domain in domains]
        lines.append(f"| {row['source']} | {row['fraction']:.0%} | {row['method']} | {row['reference_method']} | " + " | ".join(cells) + " |")
    lines += ["", "## Matched sparse-seed method contrasts", "",
              "Positive reduction = candidate improves on baseline; negative reduction = worse.",
              "Percentage changes are computed per matched seed before averaging. Historical B2",
              "and new jobs do not share identical sampler trajectories; no p-values are claimed.", "",
              "| Source | Labels | Candidate vs baseline | STB reduction % | STA reduction % | QNRF reduction % |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in data["contrasts"]:
        cells = [" / ".join(cell(row["domains"][domain][metric]["relative_reduction_pct"]) for metric in METRICS)
                 for domain in domains]
        lines.append(f"| {row['source']} | {row['fraction']:.0%} | {row['candidate']} vs {row['baseline']} | " + " | ".join(cells) + " |")
    if not data["contrasts"]:
        lines += ["", "No complete matched method pair is available yet."]
    lines += ["", "## Does scarcity disproportionately hurt unseen domains?", "",
              "Positive percentage-point difference = unseen relative degradation exceeds source",
              "relative degradation. Negative values contradict that pattern for this comparison.", "",
              "| Source | Target | Labels | Method | Target Δ% − source Δ% (MAE / RMSE) |",
              "| --- | --- | ---: | --- | ---: |"]
    for row in data["disproportion"]:
        values = " / ".join(cell(row["target_minus_source_degradation_pp"][metric]) for metric in METRICS)
        lines.append(f"| {row['source']} | {row['target']} | {row['fraction']:.0%} | {row['method']} | {values} |")
    lines += ["", "This analysis is not evidence of a benefit from any pending method. The",
              "[protocol](../docs/BENCHMARK_SUITE.md), [novelty limitations](../docs/NOVELTY_AUDIT_2026_10.md),",
              "and [remaining experiments](benchmark_progress.md) bound any eventual conclusion.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    matrix = load_json(ROOT / "runs/benchmark_v1/matrix.json")
    declared = expected_keys(matrix["settings"])
    specs = [load_json(ROOT / entry["spec"]) for entry in matrix["jobs"]]
    actual = [(spec["source"], spec["fraction"], spec["method"], spec["seed"]) for spec in specs]
    if len(actual) != len(set(actual)) or set(actual) != declared:
        raise ValueError("Runtime matrix does not cover the complete declared protocol")
    payload = report(args.require_complete)
    data = analyze(payload["completed"], matrix["settings"], require_complete=args.require_complete)
    data.update(updated_at=payload["updated_at"], production_commit=matrix["production_commit"],
                source_order=matrix["settings"]["sources"])
    write_json(ROOT / "reports/benchmark_analysis.json", data)
    destination = ROOT / "reports/benchmark_analysis.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(markdown(data), encoding="utf-8")
    temporary.replace(destination)
    print(f"Analysis: {data['completed_runs']}/{data['expected_runs']} completed; "
          f"{len(data['contrasts'])} complete matched contrasts")


if __name__ == "__main__":
    main()
