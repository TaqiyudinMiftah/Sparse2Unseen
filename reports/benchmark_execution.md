# Production benchmark launch

The production queue launched on **2026-10-07 at 04:44:29 UTC**. The recipe,
code, and splits were frozen at 04:44:19 UTC from commit
`63aa4d2de830aca2c81f1f7df157486a1e4a7115`. Report-only commits after this
freeze do not change the running experiments.

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

Mean Teacher is still in its declared ten-epoch supervised warmup. Its
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

## Queue and reporting

The matrix contains 141 primary runs: four audited historical MPCount runs
are complete, two new runs are active, and 135 runs are pending at this
snapshot. The queue advances automatically, using up to two GPUs, and performs
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
