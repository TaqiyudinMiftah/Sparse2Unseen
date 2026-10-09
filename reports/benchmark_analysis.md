# Benchmark analysis

Updated 2026-10-09T07:38:12+00:00.

Status: partial — final conclusion pending; 20/141 audited runs.

All statistics use source-selected, completed training and fixed test evaluations.
Only complete seed groups and matched contrasts are displayed. Cells give
**MAE / RMSE**, with sample SD across seeds, not standard error or pooled-image RMSE.

## Complete groups

| Source | Labels | Method | STB | STA | QNRF |
| --- | ---: | --- | ---: | ---: | ---: |
| stb | 10% | domain_stable | 10.8947 ± 1.2033 / 21.1660 ± 4.1058 | 105.1514 ± 11.1450 / 180.0757 ± 13.9624 | 197.4813 ± 15.2570 / 343.0076 ± 15.3696 |
| stb | 10% | label_only | 11.3052 ± 1.0206 / 19.1705 ± 1.8644 | 120.6158 ± 9.5117 / 198.4541 ± 17.0236 | 225.4980 ± 36.5880 / 388.8891 ± 61.6990 |
| stb | 10% | mean_teacher | 13.8956 ± 1.9685 / 24.8913 ± 4.1022 | 111.9733 ± 19.4752 / 186.5893 ± 34.9879 | 214.9966 ± 32.9339 / 361.8875 ± 38.9802 |
| stb | 10% | mpcount | 9.5516 ± 0.5365 / 16.2051 ± 0.5346 | 118.0166 ± 16.6290 / 199.8379 ± 24.7980 | 257.0374 ± 11.2870 / 422.0574 ± 14.1263 |
| stb | 10% | ssl_dg | 13.4552 ± 1.0461 / 25.4374 ± 2.7235 | 116.0782 ± 20.5864 / 203.0121 ± 40.4259 | 229.7437 ± 35.3215 / 401.8718 ± 51.9954 |
| stb | 100% | mpcount | 7.7991 (single seed) / 13.3822 (single seed) | 110.6855 (single seed) / 181.4593 (single seed) | 242.7133 (single seed) / 418.9947 (single seed) |

## Label-scarcity degradation

Positive percentage = worse than the 100% reference. Full-label references have
one seed (2023), so these are descriptive changes, not matched-seed significance tests.
B1 uses the B0 full-label reference; B3/the prototype use B2. These supervised-family
references are not separately trained 100% EMA models.

| Source | Labels | Method | Full reference | STB Δ% | STA Δ% | QNRF Δ% |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| stb | 10% | domain_stable | mpcount | 39.6920 ± 15.4285 / 58.1648 ± 30.6807 | -4.9998 ± 10.0691 / -0.7625 ± 7.6945 | -18.6360 ± 6.2860 / -18.1356 ± 3.6682 |
| stb | 10% | mpcount | mpcount | 22.4710 ± 6.8789 / 21.0940 ± 3.9947 | 6.6234 ± 15.0237 / 10.1282 ± 13.6659 | 5.9017 ± 4.6503 / 0.7310 ± 3.3715 |
| stb | 10% | ssl_dg | mpcount | 72.5228 ± 13.4134 / 90.0834 ± 20.3517 | 4.8721 ± 18.5990 / 11.8774 ± 22.2782 | -5.3436 ± 14.5528 / -4.0867 ± 12.4096 |

## Matched sparse-seed method contrasts

Positive reduction = candidate improves on baseline; negative reduction = worse.
Percentage changes are computed per matched seed before averaging. Historical B2
and new jobs do not share identical sampler trajectories; no p-values are claimed.

| Source | Labels | Candidate vs baseline | STB reduction % | STA reduction % | QNRF reduction % |
| --- | ---: | --- | ---: | ---: | ---: |
| stb | 10% | domain_stable vs ssl_dg | 18.2353 ± 15.4316 / 17.2090 ± 9.7050 | 7.9399 ± 15.6505 / 9.8580 ± 11.4942 | 13.1543 ± 10.5146 / 13.8546 ± 10.0349 |
| stb | 10% | mean_teacher vs label_only | -24.5489 ± 28.8722 / -32.0260 ± 34.9412 | 7.4462 ± 10.9838 / 6.4151 ± 10.3020 | 3.9303 ± 13.3048 / 5.8570 ± 13.1755 |
| stb | 10% | mpcount vs label_only | 14.7863 ± 12.1170 / 14.7494 ± 11.2027 | 1.0889 ± 21.6832 / -1.3734 ± 17.5466 | -15.5559 ± 14.9648 / -10.1746 ± 15.7917 |
| stb | 10% | ssl_dg vs mpcount | -40.7920 ± 5.1627 / -56.8639 ± 14.3738 | 0.6467 ± 19.8959 / -2.1034 ± 19.3976 | 10.8970 ± 9.6222 / 4.9689 ± 9.1766 |

## Does scarcity disproportionately hurt unseen domains?

Positive percentage-point difference = unseen relative degradation exceeds source
relative degradation. Negative values contradict that pattern for this comparison.

| Source | Target | Labels | Method | Target Δ% − source Δ% (MAE / RMSE) |
| --- | --- | ---: | --- | ---: |
| stb | sta | 10% | domain_stable | -44.6918 ± 25.4644 / -58.9273 ± 34.3155 |
| stb | qnrf | 10% | domain_stable | -58.3280 ± 20.3116 / -76.3004 ± 33.5327 |
| stb | sta | 10% | mpcount | -15.8476 ± 11.0719 / -10.9658 ± 12.9935 |
| stb | qnrf | 10% | mpcount | -16.5693 ± 5.3007 / -20.3630 ± 0.7506 |
| stb | sta | 10% | ssl_dg | -67.6507 ± 18.1650 / -78.2059 ± 18.1822 |
| stb | qnrf | 10% | ssl_dg | -77.8664 ± 18.4774 / -94.1701 ± 13.5926 |

This analysis is not evidence of a benefit from any pending method. The
[protocol](../docs/BENCHMARK_SUITE.md), [novelty limitations](../docs/NOVELTY_AUDIT_2026_10.md),
and [remaining experiments](benchmark_progress.md) bound any eventual conclusion.
