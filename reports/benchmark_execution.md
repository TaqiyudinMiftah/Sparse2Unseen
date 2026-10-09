# Production benchmark launch

The production queue launched on **2026-10-07 at 04:44:29 UTC**. The recipe,
code, and splits were frozen at 04:44:19 UTC from commit
`63aa4d2de830aca2c81f1f7df157486a1e4a7115`. Subsequent reporting and isolated
diagnostic additions do not change the frozen production recipe.

Both first training jobs began at 04:45:29 UTC:

| Experiment | GPU index at launch | W&B monitoring |
| --- | ---: | --- |
| STB 10%, label-only, seed 1 | 1 | [Live run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/lygqd1ht) |
| STB 10%, Mean Teacher, seed 1 | 0 | [Live run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/4uvxhx3s) |

At the 04:55 UTC verification, both jobs had finished seven of their 180
epochs (zero-based epoch 6). Process checks and GPU allocations confirmed
that the queue manager and both trainers were live; both GPUs showed 100%
utilization at the earlier 04:54 UTC sample. Other users' GPU jobs were left
untouched. Rolling resume checkpoints and source-selected best checkpoints
were present for both runs.

At that snapshot, Mean Teacher was still in its declared ten-epoch supervised
warmup. Its
unlabeled path starts after that warmup; zero unlabeled visits at this stage
are expected. W&B logs source training and source validation only. These early
validation measurements are not final benchmark results.

The warmup transition was subsequently verified at **05:00:46 UTC**. Mean
Teacher completed zero-based epoch 10 with 320 labeled visits, 320 unlabeled
visits, and 20 optimizer updates. Its supervised and unsupervised losses were
finite (1.20213 and 0.13853); no trainer exception or out-of-memory failure was
observed. The label-only run had completed epoch 11 at the next process check.
This verifies that the declared unlabeled path is active in production, not
only in a smoke run. It does not establish final counting performance.

## Source-only instability check, 05:35 UTC

The label-only run had completed 39/180 epochs (zero-based epoch 38), with
best source-validation MAE 10.59355 at epoch 10. Mean Teacher had completed
28/180 epochs (epoch 27), with best source-validation MAE 18.72449 at epoch
11. Both trainers and the detached queue manager remained live. No new run
had completed training or reached test evaluation.

Mean Teacher's validation performance became unstable after SSL began:
MAE reached 407.55699 at epoch 23 and was 213.46828 at epoch 27. Its losses
remained finite. A read-only CPU check loaded the epoch-26 rolling checkpoint
and compared student and EMA inference on the first three **STB validation**
images in the manifest, using identical whole-image inputs, native
normalization, and density sum divided by 1000:

| Source validation image | Ground-truth count | Student count | EMA teacher count |
| --- | ---: | ---: | ---: |
| IMG_106 | 132 | 151.79850 | 374.09891 |
| IMG_108 | 209 | 142.48777 | 361.54066 |
| IMG_11 | 139 | 165.53491 | 341.88459 |

These are diagnostic examples, **not full-set MAE/RMSE or final results**.
They show a large student/teacher inference gap at that checkpoint, but do
not establish its cause or predict the final selected checkpoint's quality.
The rolling checkpoint is subsequently overwritten by normal training; this
table is an observation record, not a retained-checkpoint reproduction audit.

Inspection found matching labeled/unlabeled input normalization and confirmed
that all teacher buffers equaled the student buffers, as the declared helper
requires. Parameter averaging combined with copied current batch-normalization
statistics is a possible mechanism to investigate, not a demonstrated coding
bug. No recalibration, target access, model replacement, checkpoint change,
or recipe adjustment was performed. Production remains on the frozen recipe,
with EMA checkpoints selected by source validation only. This instability
must be considered when interpreting recipe-specific SSL results; it cannot
support a general claim that source-only SSL fails.

The subsequent [full source-only BN diagnostic](ema_bn_diagnostic.md) completed
at 06:09:55 UTC. On an archived epoch-36 EMA checkpoint, scratch recalibration
using 32 source training images reduced 80-image source-validation MAE from
359.51998 to 15.55872 while preserving every model parameter. These diagnostic
scores are excluded from the primary results. The unchanged production run
later selected epoch 131, rather than its early unstable checkpoint.

## Completed production audits, 16:40 UTC

Three new runs completed 180 epochs and fixed STB/STA/QNRF test evaluation:

| STB 10% run, seed 1 | Source-selected epoch (zero-based) | Source-val MAE | Test evaluation completed (UTC) |
| --- | ---: | ---: | --- |
| Label-only | 10 | 10.59355 | 08:40:08 |
| Mean Teacher | 131 | 13.38286 | 10:38:50 |
| Ordinary SSL + MPCount | 154 | 15.28983 | 14:59:53 |

All seven completed entries, including the four historical MPCount references,
passed the checkpoint, complete-epoch/exposure, source-selection, fixed test
ID/count, and saved prediction/metric audits. Updated MAE/RMSE are in
[benchmark_progress.md](benchmark_progress.md). No complete three-seed matched
method contrast is available yet; seed-1 differences are descriptive, not
final SSL/DG conclusions. The 127-test suite passes, and the strict conclusion
gate still correctly refuses completion with 134 primary entries remaining.

The queue automatically advanced to the STB 10% stability prototype, seed 1
([W&B](https://wandb.ai/Tim-1/Sparse2Unseen/runs/2n48d40j)), and label-only,
seed 2 ([W&B](https://wandb.ai/Tim-1/Sparse2Unseen/runs/h0il2ddv)). Both child
processes and the detached manager were confirmed live; both GPUs showed
100% utilization. No training recipe, target-dependent selection, or external
user process was changed.

## Queue and reporting

At the **2026-10-09 07:38 UTC** verification, the matrix contains 141 primary
runs: four audited historical MPCount runs and sixteen new runs are complete,
two new runs are active, and 119 runs are pending. The queue advances automatically, using up
to two GPUs, and performs
the fixed three-domain test evaluation only after successful full training.
The running manager is detached from the interactive shell; this does not
provide automatic restart after a host reboot.

Runtime state is stored in `runs/benchmark_v1/queue_state.json`, with per-run
progress, checkpoints, and logs under `runs/benchmark_v1/`. These artifacts
remain local and are excluded from Git. Aggregate results and remaining work
are recorded in [benchmark_progress.md](benchmark_progress.md).

Code, splits, protocol, and report snapshots are being pushed through
[draft PR #13](https://github.com/TaqiyudinMiftah/Sparse2Unseen/pull/13).
The final scientific conclusion remains pending until the full primary matrix
passes the completion audit. No conclusion about SSL or the stability
prototype is inferred from the initial training epochs.

## Completed STB 10% cohort, 2026-10-09

All five methods have completed all three STB 10% seeds and their fixed
three-domain tests. Four STB 5% seed-1 methods have also completed; their
single-seed scores do not form a complete aggregate. All 20 completed entries
passed the checkpoint, source-selection, epoch/exposure, test ID/count, and
per-image metric audits. The 127-test suite passes. The strict completion gate
still refuses a final conclusion with 121 primary entries incomplete.

The [STB 10% cohort report](stb_10_benchmark_seed123.md) records the descriptive
findings and limitations. The prototype's unseen mean MAE is 105.15 on STA
and 197.48 on QNRF, compared with B3's 116.08 and 229.74. It improves MAE in
all three QNRF seeds but only two STA seeds. Sparse MPCount remains best on
STB. Mean Teacher worsens source MAE versus label-only in all three seeds,
so the proposed source-recovery hypothesis is not supported by this recipe's
results. None of these observations changes the frozen production settings.

The detached manager (PID 91577) and both active child trainers were verified
live at 07:38 UTC; both RTX 3060 GPUs showed 100% utilization. The current jobs
are:

| STB 5% experiment | Completed epochs at snapshot | W&B |
| --- | ---: | --- |
| Domain-stable prototype, seed 1 | 85 / 180 | [Run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/xeqk9t0y) |
| Label-only, seed 2 | 29 / 180 | [Run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/dywpedao) |

The host had 98 GiB available disk space and approximately 46 GiB available
RAM. No duplicate training process was launched, no other user's process was
changed, and no training recipe or target-dependent selection was introduced.
