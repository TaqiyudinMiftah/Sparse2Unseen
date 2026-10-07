# Sparse MPCount B2 — STB 10%, three-seed protocol and results

Status: all three sparse seeds completed 180-epoch training and STB/STA/QNRF
test evaluations. The original seed-3 run was interrupted after epoch 149;
the unchanged clean retry completed on 2026-09-28. Only the completed retry
contributes to the three-seed result.
Do not use any STA or QNRF result to choose
checkpoints, training schedules, augmentation, or other hyperparameters.

## Frozen protocol

- The 320-image STB source-training pool is split into 32 labeled and 288
  unused images by each of the committed split files below. MPCount B2 uses
  the 32 labeled images only; the 288 images are reserved for later SSL
  baselines. The same labeled IDs must be used by B0 and B1.
- A separate, fixed set of 80 fully labeled STB validation images selects the
  checkpoint. These labels are **not included** in the nominal 10% of source
  training annotations. The true annotation budget is therefore 32 labeled
  training images plus 80 labeled validation images.
- The model is current-upstream MPCount `final` with deterministic upsampling.
  Each run uses 180 epochs, physical batch 4, gradient accumulation 4, and 20
  optimizer updates per epoch, matching the
  [full-label anchor](mpcount_stb_100_seed2023_effbs16.md). The 32 labeled
  images are shuffled with exactly ten visits each per epoch, giving 320
  training samples and 80 minibatches. MPCount applies stochastic crops and
  photometric transforms at each visit. This repeated sampling is an explicit
  adapter for a fixed update budget, not additional labels.
- Split/model seeds are 1, 2, and 3. Validation and the test inference
  settings remain fixed across seeds. STB and STA use the 10000-pixel
  whole-image setting; QNRF uses 1024-pixel tiles because of GPU memory.
  QNRF tile results are not exact whole-image reproductions.
- The official MPCount checkout remains
  `6eb06772bcf7dfb771c43a14d67146fce767f103`. All dataset roots,
  configs, logs, and checkpoints for a run are recorded locally; only configs,
  split IDs, metrics, and the report are committed to Git.

| Seed | Committed split SHA-256 | Labeled / unlabeled |
| ---: | --- | ---: |
| 1 | `edf0c765c81df681e9a0bbe549b1dc957ccec6176dbabd6f4355a56a49b81fcb` | 32 / 288 |
| 2 | `db77763b14a4b615ddab78b319f0e1e01464e6764666f9de077c2fbe8072bbd4` | 32 / 288 |
| 3 | `0ec751fcb861132ba4832628af3e7c1288387c497accb303cabaa2380b960067` | 32 / 288 |

Each sparse root was validated to expose 32 train, 80 val, and 316 source
test images. The seed-1 one-epoch smoke run completed 80 minibatches and 20
optimizer updates, with STB source-validation MAE 42.2647. This smoke score
does not enter the results table or choose a hyperparameter.
The full seed-1 run started at 2026-09-27 13:40:40 UTC and is tracked in
[W&B](https://wandb.ai/Tim-1/Sparse2Unseen/runs/oowwjjf4); this training run
logs source-domain metrics only.
The full seed-2 run started at 2026-09-27 13:43:20 UTC and is tracked in
[W&B](https://wandb.ai/Tim-1/Sparse2Unseen/runs/0cubr7tj); it follows the same
source-only protocol on the second shared GPU.
Both runs completed 180 epochs and W&B reports them as finished. Seed 1 ended
at 2026-09-27 17:56:17 UTC; seed 2 ended at 2026-09-27 17:50:53 UTC.
Seed 3 started at 2026-09-28 02:11:18 UTC on GPU 0 and is tracked in
[W&B](https://wandb.ai/Tim-1/Sparse2Unseen/runs/wzkb55yg). Its predeclared
training config is unchanged after viewing the seed-1/2 test results.

### Seed-3 interruption audit

At the 2026-09-28 08:30 UTC check, the seed-3 process was absent and W&B
reported `crashed`. The local training log last changed at 05:44:20 UTC and
ends after completed epoch 149 (150 of the planned 180 epochs), without a
`Best epoch` / `End training` completion record. W&B's last synced epoch is
148. No traceback or CUDA out-of-memory error was found in the saved console
or W&B logs, and the system journal is not accessible to this user; the cause
is therefore unconfirmed.

The best source-validation MAE observed before interruption was 10.1391 at
epoch 135. This incomplete-run checkpoint is **not** treated as a final
seed-3 result and has not been evaluated on targets. The completed-run
selection guard remains in force. MPCount saves model weights only, so
optimizer, scheduler, and RNG state needed for an exact training resume are
unavailable. The original interrupted artifacts remain preserved; the
separately audited clean retry below uses the unchanged recipe.

### Seed-3 clean retry queue

At 2026-09-28 09:00:10 UTC, a detached queue worker was activated for
`stb_10_seed3_effbs16_retry1`, using project commit
`5e0df2736b4fe0ff7888d9ae0d3cdd53b4b96345`. The original run, weights,
logs, and crashed W&B record are preserved. The retry starts from the same
pretrained initialization with seed 3 and the unchanged 180-epoch config;
it is not an exact resume and does not replace the original audit.

The queue requires at least 7168 MiB free on one GPU for three consecutive
30-second polls. Its initial inventory showed 4491 MiB free on GPU 0 and
1350 MiB on GPU 1, so training had not started at activation. This is a
best-effort memory gate, not an exclusive GPU reservation; other users and
their processes are not modified. The worker is detached from the invoking
shell and records child output and exit codes, without automatically retrying
a failure.

Local audit artifacts are under
`runs/queues/stb_10_seed3_effbs16_retry1/`: `request.json` records commits and
SHA-256 hashes of the split, training config, evaluation templates, lockfile,
and adapter code; `status.json` records the current state and PIDs;
`training.log` and `evaluation.log` are created when those phases start.
The queue refuses to launch if frozen input hashes or the upstream checkout
change. These runtime artifacts and credentials are not committed to Git.

The worker started training at 2026-09-28 10:27:20 UTC. The MPCount log
records all 180 epochs with 80 microbatches and 20 optimizer steps per epoch,
then `End training` at 14:40:13 UTC. The worker finished its fixed tests at
14:50:48 UTC with exit code 0 and state `complete`. Source-only monitoring
is in the completed [W&B retry run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/9rlen9r2).
All three tests loaded the retry's same STB-validation-selected
`best_135.pth`; STB/STA used whole-image inference and QNRF used the fixed
1024-pixel tiles. Retry test logs use the retry prefix and did not overwrite
earlier runs. No target evaluation was launched from the interrupted run.

The original run's epoch-135 weight file has the **same SHA-256** as the
independently completed retry's selected epoch-135 file. This confirms the
selected weights matched, but does not turn the interrupted run into a
completed training run or change the source-only selection rule. Queue,
cancellation, and retry-specific evaluation tests passed as part of the
74-test suite.

## Source-only checkpoint selection

| Seed | Selected epoch | STB validation MAE / RMSE | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 1 | 166 | 11.2401 / 20.6144 | `8932b730aeb62e9555042f1093956e5e991f386ef93c22e89da071d7e614ddb8` |
| 2 | 119 | 11.7258 / 22.2547 | `80ed8abaff54f51d02134ca32470b20dbe61cf70b14298bc4c087d1b608142a3` |
| 3, clean retry | 135 | 10.1391 / 17.8949 | `f150cf69b0efd6492cde1990f11c56eb975c1a0fa8396d6f489bb2f2b873846d` |

The selected weights are local files under
`external/MPCount/logs/stb_10_seed{seed}_effbs16/best_{epoch}.pth` for seeds
1 and 2, and under the distinct `stb_10_seed3_effbs16_retry1` directory for
seed 3.
These validation metrics are not test results. RMSE is the square root of the
logged mean squared error. Only the selected checkpoint is evaluated; target
performance never selects among epochs.

## Results

The full-label reference is a single seed-2023 run, not a paired three-seed
estimate. Test scores are MAE / RMSE. The final row reports the mean and
**sample** standard deviation across the three sparse split/model seeds.

| Method | Source-training labels | Seed | STB test | STA test | QNRF test, 1024-pixel tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPCount full-label anchor | 320 / 320 | 2023 | 7.7991 / 13.3822 | 110.6855 / 181.4593 | 242.7133 / 418.9947 |
| MPCount sparse | 32 / 320 | 1 | 8.9392 / 15.8303 | 102.2181 / 172.8133 | 250.3209 / 410.7431 |
| MPCount sparse | 32 / 320 | 2 | 9.9386 / 16.8172 | 116.4645 / 205.1531 | 270.0684 / 437.8901 |
| MPCount sparse, clean retry | 32 / 320 | 3 | 9.7771 / 15.9677 | 135.3673 / 221.5474 | 250.7229 / 417.5391 |
| MPCount sparse mean ± sample SD | 32 / 320 | 1–3 | 9.5516 ± 0.5365 / 16.2051 ± 0.5346 | 118.0166 ± 16.6290 / 199.8379 ± 24.7980 | 257.0374 ± 11.2870 / 422.0574 ± 14.1263 |

Raw MPCount mean squared errors are retained for the RMSE calculation:

| Seed | STB test logged `mse` | STA test logged `mse` | QNRF test logged `mse` |
| ---: | ---: | ---: | ---: |
| 1 | 250.5985 | 29864.4378 | 168709.8849 |
| 2 | 282.8193 | 42087.7813 | 191747.7812 |
| 3, clean retry | 254.9669 | 49083.2533 | 174338.8891 |

Seed 1 completed all final tests at 2026-09-28 02:10:43 UTC; seed 2 at
2026-09-28 02:10:19 UTC. Test logs and their effective configs are in ignored
local directories named `external/MPCount/logs/stb_10_seed{seed}_effbs16_test_{domain}`;
the QNRF directory additionally ends in `_ps1024`. All domains used the same
source-selected checkpoint within each seed.
The clean seed-3 retry completed the STB, STA, and QNRF tests at
2026-09-28 14:42:42, 14:43:19, and 14:50:47 UTC, respectively. Its test
configs independently confirm `data/stb`, `data/sta`, and `data/qnrf` roots,
the deterministic model, and patch sizes 10000, 10000, and 1024.

## Descriptive MAE degradation against the full-label anchor

Each cell is absolute MAE change / relative percentage change. A negative
change means that the sparse run scored better than the reference run on that
test set; it does not justify choosing that seed or tuning on the target.

| Seed | STB | STA | QNRF, 1024-pixel tiles |
| ---: | ---: | ---: | ---: |
| 1 | +1.1401 / +14.62% | -8.4674 / -7.65% | +7.6076 / +3.13% |
| 2 | +2.1395 / +27.43% | +5.7790 / +5.22% | +27.3551 / +11.27% |
| 3, clean retry | +1.9780 / +25.36% | +24.6818 / +22.30% | +8.0096 / +3.30% |
| Three-seed mean vs. full-label anchor | +1.7525 / +22.47% | +7.3311 / +6.62% | +14.3241 / +5.90% |

All three seeds worsen source STB MAE relative to the full-label anchor. STA
varies substantially across seeds (one sparse seed improves on the reference),
while mean relative MAE degradation on both unseen sets is smaller than on
STB. These results do **not** support a general claim that 10% labels damage
unseen-domain counting disproportionately. Do not interpret the favorable
first seed as a general sparse-label advantage. The planned B0/B1 comparisons
are needed to distinguish sparse supervision from the effects of MPCount's DG
mechanism and source-only SSL.

For each domain, report absolute MAE degradation as sparse MAE minus the
full-label anchor MAE, and relative degradation as that difference divided by
the full-label anchor MAE. Compare source and unseen degradation together.
Because the full-label reference has one seed and the sparse runs use both
split and model seeds 1–3, these are descriptive comparisons, not a paired
seed-controlled significance test. Sample SD describes only the three sparse
runs; it is not an uncertainty estimate for the full-label anchor. All
checkpoints above were selected exclusively on STB validation.
