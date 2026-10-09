# STB 10%: completed five-method, three-seed comparison

Audited on **2026-10-09**. This cohort is complete; the broader 141-run
primary benchmark is not. These are descriptive results, not the final
project conclusion or a state-of-the-art claim.

## Protocol and audit

Each method uses the same 32 labeled and 288 unlabeled STB training IDs for
each of seeds 1, 2, and 3. Label-only and sparse MPCount do not use the
unlabeled pool. The 80 fully labeled STB validation images are outside the
10% training-label fraction: total annotated source images are **112/400
(28%)**, not 10% of all source annotations.

Every run completed 180 epochs with physical batch 4, accumulation 4, and
20 optimizer updates per epoch. Checkpoints were selected solely by minimum
STB validation MAE. B0/B2 infer with the selected student; B1/B3/the prototype
infer with the selected EMA teacher. STB and STA use whole-image inference;
QNRF uses the fixed 1024-tile protocol. No target data or target-derived
selection affected training. Recipes remain frozen at production commit
`63aa4d2de830aca2c81f1f7df157486a1e4a7115`.

The result audit checks completed training/exposure histories, source-only
selection, selected-checkpoint hashes, canonical test IDs and ground-truth
counts, and MAE/RMSE recomputed from saved per-image predictions. The three
historical sparse MPCount runs retain their audited original sampler
trajectories; they are not reruns with identical trajectories to the new
methods. The [protocol](../docs/BENCHMARK_SUITE.md) documents these adapters.

## Three-seed test results

Cells give **MAE ± sample SD / RMSE ± sample SD** across seeds 1/2/3.
Lower is better. SD is across seed-level scores, not standard error or
pooled-image RMSE.

| Method | STB source | STA unseen | QNRF unseen |
| --- | ---: | ---: | ---: |
| B0: label-only | 11.31 ± 1.02 / 19.17 ± 1.86 | 120.62 ± 9.51 / 198.45 ± 17.02 | 225.50 ± 36.59 / 388.89 ± 61.70 |
| B1: Mean Teacher | 13.90 ± 1.97 / 24.89 ± 4.10 | 111.97 ± 19.48 / 186.59 ± 34.99 | 215.00 ± 32.93 / 361.89 ± 38.98 |
| B2: sparse MPCount | 9.55 ± 0.54 / 16.21 ± 0.53 | 118.02 ± 16.63 / 199.84 ± 24.80 | 257.04 ± 11.29 / 422.06 ± 14.13 |
| B3: ordinary SSL + MPCount | 13.46 ± 1.05 / 25.44 ± 2.72 | 116.08 ± 20.59 / 203.01 ± 40.43 | 229.74 ± 35.32 / 401.87 ± 52.00 |
| Domain-stable prototype | 10.89 ± 1.20 / 21.17 ± 4.11 | 105.15 ± 11.15 / 180.08 ± 13.96 | 197.48 ± 15.26 / 343.01 ± 15.37 |

Unrounded per-seed results, source-selected checkpoint provenance, and complete
aggregates are in [benchmark_results.json](benchmark_results.json) and
[benchmark_analysis.md](benchmark_analysis.md).

## What this cohort supports

The prototype has the lowest mean MAE and RMSE on both unseen datasets among
these five methods. Sparse MPCount has the lowest source MAE and RMSE.
Better source scores therefore do not identify the best transfer model in
this cohort.

Against B3, the prototype lowers MAE in all three seeds on STB and QNRF,
but only two of three on STA. STA seed 3 worsens from 108.25 to 115.96 MAE;
the effect is not uniformly positive. Matched-seed relative MAE reductions
are **18.24 ± 15.43% on STB, 7.94 ± 15.65% on STA, and 13.15 ± 10.51% on
QNRF**. These percentages are computed per seed and then averaged, not by
taking a ratio of aggregate means. No statistical-significance claim is made.

Mean Teacher does **not** recover source performance in this frozen recipe:
its STB MAE is worse than label-only in all three seeds. It improves mean
unseen scores, but not every seed. This contradicts the anticipated
"large source recovery, little unseen recovery" pattern here; that hypothesis
must not be presented as an observed result. The separate
[source-only EMA/BN diagnostic](ema_bn_diagnostic.md) establishes early
normalization sensitivity, not a generic failure of SSL or a cause proven
for these final scores. Scratch diagnostic models are excluded from this table.

The existing 100% MPCount reference has STB/STA/QNRF MAE
7.80 / 110.69 / 242.71 (single seed 2023). Sparse B2's relative MAE degradation
is +22.47% / +6.62% / +5.90%, respectively. This comparison does **not** support
the claim that label scarcity hurts unseen MAE disproportionately. The 100%
label-only reference is still pending, so corresponding B0/B1 degradation
claims are not yet available.

## Limits and next step

Only one source and one sparse fraction have a complete five-method,
three-seed comparison. Three seeds do not establish robustness across other
sources, fractions, or datasets. B0/B1 and B2/B3 use the documented base/DG
model families; historical B2 and new jobs have different sampling
trajectories. The prototype also uses four extra diversified teacher forward
passes to estimate regional weights, so B3 versus prototype is not a
compute-matched or identical-random-trajectory ablation. It cannot by itself
prove that stability weighting, rather than all associated changes, causes
the improvement.

The [current literature audit](../docs/NOVELTY_AUDIT_2026_10.md) documents
direct prior-work overlap. TMTB/UGSDA reproduction and the required mechanism
ablations are absent from this primary matrix; no novelty or SOTA claim
follows from this result.

Continue the unchanged 5%/40% and STA/QNRF-source queue, complete the remaining
full-label references, and apply the strict full-matrix completion gate
before writing the final conclusion. Test observations from this cohort
must not be used to retune the ongoing frozen suite.
