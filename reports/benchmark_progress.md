# Complete benchmark progress

Updated 2026-10-09T07:38:12+00:00.

Audited training + all-test completion: **20 / 141**.

The final conclusion is pending the complete primary matrix. The protocol is in
[BENCHMARK_SUITE.md](../docs/BENCHMARK_SUITE.md). JHU remains the later external extension.

## Completed training results

Scores are MAE / RMSE. QNRF uses fixed 1024 tiles.

| Source | Labels | Method | Seed | STB | STA | QNRF |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| stb | 10% | label_only | 1 | 12.3003 / 20.9783 | 124.6992 / 196.1592 | 189.6028 / 320.0466 |
| stb | 10% | mean_teacher | 1 | 12.8001 / 22.0650 | 105.4457 / 165.9934 | 207.9065 / 345.8290 |
| stb | 10% | mpcount | 1 | 8.9392 / 15.8303 | 102.2181 / 172.8133 | 250.3209 / 410.7431 |
| stb | 10% | ssl_dg | 1 | 12.3041 / 26.2497 | 100.5579 / 176.0550 | 212.4765 / 375.8871 |
| stb | 10% | domain_stable | 1 | 12.0589 / 24.4588 | 93.6947 / 165.1239 | 180.3025 / 325.3261 |
| stb | 10% | label_only | 2 | 10.2609 / 17.2541 | 127.4041 / 216.5088 | 262.7416 / 439.1943 |
| stb | 10% | mean_teacher | 2 | 16.1681 / 29.5965 | 133.8739 / 226.9873 | 250.8980 / 406.3317 |
| stb | 10% | mpcount | 2 | 9.9386 / 16.8172 | 116.4645 / 205.1531 | 270.0684 / 437.8901 |
| stb | 10% | ssl_dg | 2 | 13.7137 / 27.6623 | 139.4313 / 249.4943 | 270.3771 / 461.7376 |
| stb | 10% | domain_stable | 2 | 10.9694 / 22.4735 | 105.8033 / 192.7747 | 209.4532 / 350.5271 |
| stb | 10% | label_only | 3 | 11.3545 / 19.2792 | 109.7442 / 182.6944 | 224.1496 / 407.4265 |
| stb | 10% | mean_teacher | 3 | 12.7186 / 23.0126 | 96.6004 / 166.7873 | 186.1852 / 333.5017 |
| stb | 10% | mpcount | 3 | 9.7771 / 15.9677 | 135.3673 / 221.5474 | 250.7229 / 417.5391 |
| stb | 10% | ssl_dg | 3 | 14.3479 / 22.4001 | 108.2454 / 183.4868 | 206.3773 / 367.9907 |
| stb | 10% | domain_stable | 3 | 9.6558 / 16.5657 | 115.9561 / 182.3283 | 202.6882 / 353.1697 |
| stb | 5% | label_only | 1 | 15.2076 / 26.2179 | 99.7527 / 172.2054 | 191.7432 / 330.7981 |
| stb | 5% | mean_teacher | 1 | 14.7594 / 27.9319 | 134.9504 / 235.7737 | 247.6525 / 428.1902 |
| stb | 5% | mpcount | 1 | 11.9381 / 21.6808 | 102.8572 / 160.9640 | 217.9951 / 349.2645 |
| stb | 5% | ssl_dg | 1 | 13.0600 / 29.3999 | 105.7965 / 191.6251 | 201.8030 / 356.5608 |
| stb | 100% | mpcount | 2023 | 7.7991 / 13.3822 | 110.6855 / 181.4593 | 242.7133 / 418.9947 |

## Complete seed aggregates

Only groups with all their required seeds appear here.

| Source | Labels | Method | STB MAE ± sample SD | STA MAE ± sample SD | QNRF MAE ± sample SD |
| --- | ---: | --- | ---: | ---: | ---: |
| stb | 10% | domain_stable | 10.8947 ± 1.2033 | 105.1514 ± 11.1450 | 197.4813 ± 15.2570 |
| stb | 10% | label_only | 11.3052 ± 1.0206 | 120.6158 ± 9.5117 | 225.4980 ± 36.5880 |
| stb | 10% | mean_teacher | 13.8956 ± 1.9685 | 111.9733 ± 19.4752 | 214.9966 ± 32.9339 |
| stb | 10% | mpcount | 9.5516 ± 0.5365 | 118.0166 ± 16.6290 | 257.0374 ± 11.2870 |
| stb | 10% | ssl_dg | 13.4552 ± 1.0461 | 116.0782 ± 20.5864 | 229.7437 ± 35.3215 |
| stb | 100% | mpcount | 7.7991 (single seed) | 110.6855 (single seed) | 242.7133 (single seed) |

## Remaining experiments

| Experiment | Current state |
| --- | --- |
| stb_5_domain_stable_seed1_v1 | training |
| stb_5_label_only_seed2_v1 | training |
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
