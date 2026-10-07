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
