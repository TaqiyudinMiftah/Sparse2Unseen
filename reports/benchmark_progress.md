# Complete benchmark progress

Updated 2026-10-07T05:08:15+00:00.

Audited training + all-test completion: **4 / 141**.

The final conclusion is pending the complete primary matrix. The protocol is in
[BENCHMARK_SUITE.md](../docs/BENCHMARK_SUITE.md). JHU remains the later external extension.

## Completed training results

Scores are MAE / RMSE. QNRF uses fixed 1024 tiles.

| Source | Labels | Method | Seed | STB | STA | QNRF |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| stb | 10% | mpcount | 1 | 8.9392 / 15.8303 | 102.2181 / 172.8133 | 250.3209 / 410.7431 |
| stb | 10% | mpcount | 2 | 9.9386 / 16.8172 | 116.4645 / 205.1531 | 270.0684 / 437.8901 |
| stb | 10% | mpcount | 3 | 9.7771 / 15.9677 | 135.3673 / 221.5474 | 250.7229 / 417.5391 |
| stb | 100% | mpcount | 2023 | 7.7991 / 13.3822 | 110.6855 / 181.4593 | 242.7133 / 418.9947 |

## Complete seed aggregates

Only groups with all their required seeds appear here.

| Source | Labels | Method | STB MAE ± sample SD | STA MAE ± sample SD | QNRF MAE ± sample SD |
| --- | ---: | --- | ---: | ---: | ---: |
| stb | 10% | mpcount | 9.5516 ± 0.5365 | 118.0166 ± 16.6290 | 257.0374 ± 11.2870 |
| stb | 100% | mpcount | 7.7991 (single seed) | 110.6855 (single seed) | 242.7133 (single seed) |

## Remaining experiments

| Experiment | Current state |
| --- | --- |
| stb_10_label_only_seed1_v1 | training |
| stb_10_mean_teacher_seed1_v1 | training |
| stb_10_ssl_dg_seed1_v1 | pending |
| stb_10_domain_stable_seed1_v1 | pending |
| stb_10_label_only_seed2_v1 | pending |
| stb_10_mean_teacher_seed2_v1 | pending |
| stb_10_ssl_dg_seed2_v1 | pending |
| stb_10_domain_stable_seed2_v1 | pending |
| stb_10_label_only_seed3_v1 | pending |
| stb_10_mean_teacher_seed3_v1 | pending |
| stb_10_ssl_dg_seed3_v1 | pending |
| stb_10_domain_stable_seed3_v1 | pending |
| stb_5_label_only_seed1_v1 | pending |
| stb_5_mean_teacher_seed1_v1 | pending |
| stb_5_mpcount_seed1_v1 | pending |
| stb_5_ssl_dg_seed1_v1 | pending |
| stb_5_domain_stable_seed1_v1 | pending |
| stb_5_label_only_seed2_v1 | pending |
| stb_5_mean_teacher_seed2_v1 | pending |
| stb_5_mpcount_seed2_v1 | pending |
| stb_5_ssl_dg_seed2_v1 | pending |
| stb_5_domain_stable_seed2_v1 | pending |
| stb_5_label_only_seed3_v1 | pending |
| stb_5_mean_teacher_seed3_v1 | pending |
| stb_5_mpcount_seed3_v1 | pending |
| stb_5_ssl_dg_seed3_v1 | pending |
| stb_5_domain_stable_seed3_v1 | pending |
| stb_40_label_only_seed1_v1 | pending |
| stb_40_mean_teacher_seed1_v1 | pending |
| stb_40_mpcount_seed1_v1 | pending |
| stb_40_ssl_dg_seed1_v1 | pending |
| stb_40_domain_stable_seed1_v1 | pending |
| stb_40_label_only_seed2_v1 | pending |
| stb_40_mean_teacher_seed2_v1 | pending |
| stb_40_mpcount_seed2_v1 | pending |
| stb_40_ssl_dg_seed2_v1 | pending |
| stb_40_domain_stable_seed2_v1 | pending |
| stb_40_label_only_seed3_v1 | pending |
| stb_40_mean_teacher_seed3_v1 | pending |
| stb_40_mpcount_seed3_v1 | pending |
| stb_40_ssl_dg_seed3_v1 | pending |
| stb_40_domain_stable_seed3_v1 | pending |
| stb_100_label_only_seed2023_v1 | pending |
| sta_10_label_only_seed1_v1 | pending |
| sta_10_mean_teacher_seed1_v1 | pending |
| sta_10_mpcount_seed1_v1 | pending |
| sta_10_ssl_dg_seed1_v1 | pending |
| sta_10_domain_stable_seed1_v1 | pending |
| sta_10_label_only_seed2_v1 | pending |
| sta_10_mean_teacher_seed2_v1 | pending |
| sta_10_mpcount_seed2_v1 | pending |
| sta_10_ssl_dg_seed2_v1 | pending |
| sta_10_domain_stable_seed2_v1 | pending |
| sta_10_label_only_seed3_v1 | pending |
| sta_10_mean_teacher_seed3_v1 | pending |
| sta_10_mpcount_seed3_v1 | pending |
| sta_10_ssl_dg_seed3_v1 | pending |
| sta_10_domain_stable_seed3_v1 | pending |
| sta_5_label_only_seed1_v1 | pending |
| sta_5_mean_teacher_seed1_v1 | pending |
| sta_5_mpcount_seed1_v1 | pending |
| sta_5_ssl_dg_seed1_v1 | pending |
| sta_5_domain_stable_seed1_v1 | pending |
| sta_5_label_only_seed2_v1 | pending |
| sta_5_mean_teacher_seed2_v1 | pending |
| sta_5_mpcount_seed2_v1 | pending |
| sta_5_ssl_dg_seed2_v1 | pending |
| sta_5_domain_stable_seed2_v1 | pending |
| sta_5_label_only_seed3_v1 | pending |
| sta_5_mean_teacher_seed3_v1 | pending |
| sta_5_mpcount_seed3_v1 | pending |
| sta_5_ssl_dg_seed3_v1 | pending |
| sta_5_domain_stable_seed3_v1 | pending |
| sta_40_label_only_seed1_v1 | pending |
| sta_40_mean_teacher_seed1_v1 | pending |
| sta_40_mpcount_seed1_v1 | pending |
| sta_40_ssl_dg_seed1_v1 | pending |
| sta_40_domain_stable_seed1_v1 | pending |
| sta_40_label_only_seed2_v1 | pending |
| sta_40_mean_teacher_seed2_v1 | pending |
| sta_40_mpcount_seed2_v1 | pending |
| sta_40_ssl_dg_seed2_v1 | pending |
| sta_40_domain_stable_seed2_v1 | pending |
| sta_40_label_only_seed3_v1 | pending |
| sta_40_mean_teacher_seed3_v1 | pending |
| sta_40_mpcount_seed3_v1 | pending |
| sta_40_ssl_dg_seed3_v1 | pending |
| sta_40_domain_stable_seed3_v1 | pending |
| sta_100_label_only_seed2023_v1 | pending |
| sta_100_mpcount_seed2023_v1 | pending |
| qnrf_10_label_only_seed1_v1 | pending |
| qnrf_10_mean_teacher_seed1_v1 | pending |
| qnrf_10_mpcount_seed1_v1 | pending |
| qnrf_10_ssl_dg_seed1_v1 | pending |
| qnrf_10_domain_stable_seed1_v1 | pending |
| qnrf_10_label_only_seed2_v1 | pending |
| qnrf_10_mean_teacher_seed2_v1 | pending |
| qnrf_10_mpcount_seed2_v1 | pending |
| qnrf_10_ssl_dg_seed2_v1 | pending |
| qnrf_10_domain_stable_seed2_v1 | pending |
| qnrf_10_label_only_seed3_v1 | pending |
| qnrf_10_mean_teacher_seed3_v1 | pending |
| qnrf_10_mpcount_seed3_v1 | pending |
| qnrf_10_ssl_dg_seed3_v1 | pending |
| qnrf_10_domain_stable_seed3_v1 | pending |
| qnrf_5_label_only_seed1_v1 | pending |
| qnrf_5_mean_teacher_seed1_v1 | pending |
| qnrf_5_mpcount_seed1_v1 | pending |
| qnrf_5_ssl_dg_seed1_v1 | pending |
| qnrf_5_domain_stable_seed1_v1 | pending |
| qnrf_5_label_only_seed2_v1 | pending |
| qnrf_5_mean_teacher_seed2_v1 | pending |
| qnrf_5_mpcount_seed2_v1 | pending |
| qnrf_5_ssl_dg_seed2_v1 | pending |
| qnrf_5_domain_stable_seed2_v1 | pending |
| qnrf_5_label_only_seed3_v1 | pending |
| qnrf_5_mean_teacher_seed3_v1 | pending |
| qnrf_5_mpcount_seed3_v1 | pending |
| qnrf_5_ssl_dg_seed3_v1 | pending |
| qnrf_5_domain_stable_seed3_v1 | pending |
| qnrf_40_label_only_seed1_v1 | pending |
| qnrf_40_mean_teacher_seed1_v1 | pending |
| qnrf_40_mpcount_seed1_v1 | pending |
| qnrf_40_ssl_dg_seed1_v1 | pending |
| qnrf_40_domain_stable_seed1_v1 | pending |
| qnrf_40_label_only_seed2_v1 | pending |
| qnrf_40_mean_teacher_seed2_v1 | pending |
| qnrf_40_mpcount_seed2_v1 | pending |
| qnrf_40_ssl_dg_seed2_v1 | pending |
| qnrf_40_domain_stable_seed2_v1 | pending |
| qnrf_40_label_only_seed3_v1 | pending |
| qnrf_40_mean_teacher_seed3_v1 | pending |
| qnrf_40_mpcount_seed3_v1 | pending |
| qnrf_40_ssl_dg_seed3_v1 | pending |
| qnrf_40_domain_stable_seed3_v1 | pending |
| qnrf_100_label_only_seed2023_v1 | pending |
| qnrf_100_mpcount_seed2023_v1 | pending |
