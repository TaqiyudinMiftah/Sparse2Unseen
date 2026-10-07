# Benchmark analysis

Updated 2026-10-07T05:08:15+00:00.

Status: partial — final conclusion pending; 4/141 audited runs.

All statistics use source-selected, completed training and fixed test evaluations.
Only complete seed groups and matched contrasts are displayed. Cells give
**MAE / RMSE**, with sample SD across seeds, not standard error or pooled-image RMSE.

## Complete groups

| Source | Labels | Method | STB | STA | QNRF |
| --- | ---: | --- | ---: | ---: | ---: |
| stb | 10% | mpcount | 9.5516 ± 0.5365 / 16.2051 ± 0.5346 | 118.0166 ± 16.6290 / 199.8379 ± 24.7980 | 257.0374 ± 11.2870 / 422.0574 ± 14.1263 |
| stb | 100% | mpcount | 7.7991 (single seed) / 13.3822 (single seed) | 110.6855 (single seed) / 181.4593 (single seed) | 242.7133 (single seed) / 418.9947 (single seed) |

## Label-scarcity degradation

Positive percentage = worse than the 100% reference. Full-label references have
one seed (2023), so these are descriptive changes, not matched-seed significance tests.
B1 uses the B0 full-label reference; B3/the prototype use B2. These supervised-family
references are not separately trained 100% EMA models.

| Source | Labels | Method | Full reference | STB Δ% | STA Δ% | QNRF Δ% |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| stb | 10% | mpcount | mpcount | 22.4710 ± 6.8789 / 21.0940 ± 3.9947 | 6.6234 ± 15.0237 / 10.1282 ± 13.6659 | 5.9017 ± 4.6503 / 0.7310 ± 3.3715 |

## Matched sparse-seed method contrasts

Positive reduction = candidate improves on baseline; negative reduction = worse.
Percentage changes are computed per matched seed before averaging. Historical B2
and new jobs do not share identical sampler trajectories; no p-values are claimed.

| Source | Labels | Candidate vs baseline | STB reduction % | STA reduction % | QNRF reduction % |
| --- | ---: | --- | ---: | ---: | ---: |

No complete matched method pair is available yet.

## Does scarcity disproportionately hurt unseen domains?

Positive percentage-point difference = unseen relative degradation exceeds source
relative degradation. Negative values contradict that pattern for this comparison.

| Source | Target | Labels | Method | Target Δ% − source Δ% (MAE / RMSE) |
| --- | --- | ---: | --- | ---: |
| stb | sta | 10% | mpcount | -15.8476 ± 11.0719 / -10.9658 ± 12.9935 |
| stb | qnrf | 10% | mpcount | -16.5693 ± 5.3007 / -20.3630 ± 0.7506 |

This analysis is not evidence of a benefit from any pending method. The
[protocol](../docs/BENCHMARK_SUITE.md), [novelty limitations](../docs/NOVELTY_AUDIT_2026_10.md),
and [remaining experiments](benchmark_progress.md) bound any eventual conclusion.
