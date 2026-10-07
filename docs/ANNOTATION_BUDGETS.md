# Annotation budgets in the primary benchmark

The nominal 5/10/40/100% label fraction applies to **MPCount's training pool**,
after source validation has been reserved. Source validation remains fully
labeled for checkpoint selection. It must be counted when describing total
annotation consumption; target labels are evaluation-only, not supervision.

The 141 frozen experiment specs were checked on 7 October 2026. All methods
and required seeds agree on these counts for each source/fraction:

| Source | Nominal training fraction | Labeled train | Unlabeled train pool | Labeled validation | Total labeled source images | Canonical source train images | Total annotation fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STA | 5% | 12 | 228 | 60 | 72 | 300 | 24.00% |
| STA | 10% | 24 | 216 | 60 | 84 | 300 | 28.00% |
| STA | 40% | 96 | 144 | 60 | 156 | 300 | 52.00% |
| STA | 100% | 240 | 0 | 60 | 300 | 300 | 100.00% |
| STB | 5% | 16 | 304 | 80 | 96 | 400 | 24.00% |
| STB | 10% | 32 | 288 | 80 | 112 | 400 | 28.00% |
| STB | 40% | 128 | 192 | 80 | 208 | 400 | 52.00% |
| STB | 100% | 320 | 0 | 80 | 400 | 400 | 100.00% |
| QNRF | 5% | 54 | 1027 | 120 | 174 | 1201 | 14.49% |
| QNRF | 10% | 108 | 973 | 120 | 228 | 1201 | 18.98% |
| QNRF | 40% | 432 | 649 | 120 | 552 | 1201 | 45.96% |
| QNRF | 100% | 1081 | 0 | 120 | 1201 | 1201 | 100.00% |

Total annotation fraction is `(labeled_train + validation) / (train_pool +
validation)`. Sparse labeled sizes use `max(1, floor(fraction * train_pool))`;
the QNRF training fraction is therefore slightly below its nominal percentage.
The full-label subset uses the complete training pool.

Counts refer to images with point annotations, not the number of annotated
heads, human labeling hours, or equal annotation effort across datasets.
The cost is per experiment: labels are reused across competing methods, and
the same validation set is reused across seeds. The unlabeled pool exists for
every sparse split, but label-only and sparse MPCount do not train on it.

## Reporting language

Use: “10% of source training annotations, with a fixed fully labeled source
validation set for model selection.” Include the table's actual annotation
cost when comparing label efficiency.

Do not use “10% of all source labels” for the current STB/STA 10% regime.
Do not present the supervised 100% B0/B2 reference as an independently trained
100% EMA method: [the protocol](BENCHMARK_SUITE.md) explicitly uses those
supervised-family references rather than separate full-label SSL jobs.

These disclosures do not change the frozen datasets, splits, training budgets,
or checkpoint-selection rule. A validation-within-budget experiment would be
a different protocol and would need separately declared splits and runs.
