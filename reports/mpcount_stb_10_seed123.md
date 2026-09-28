# Sparse MPCount B2 — STB 10%, three-seed protocol and results

Status: seeds 1 and 2 completed training; test evaluation is in progress,
2026-09-28. Seed 3 is prepared but not yet started. Do not use any STA or QNRF result to choose
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

## Source-only checkpoint selection

| Seed | Selected epoch | STB validation MAE / RMSE | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 1 | 166 | 11.2401 / 20.6144 | `8932b730aeb62e9555042f1093956e5e991f386ef93c22e89da071d7e614ddb8` |
| 2 | 119 | 11.7258 / 22.2547 | `80ed8abaff54f51d02134ca32470b20dbe61cf70b14298bc4c087d1b608142a3` |
| 3 | pending | pending | pending |

The selected weights are local files under
`external/MPCount/logs/stb_10_seed{seed}_effbs16/best_{epoch}.pth`.
These validation metrics are not test results. RMSE is the square root of the
logged mean squared error. Only the selected checkpoint is evaluated; target
performance never selects among epochs.

## Results

The full-label reference is a single seed-2023 run, not a paired three-seed
estimate. Test scores are MAE / RMSE; target columns remain blank until a
source-validation-selected checkpoint completes training and is evaluated.

| Method | Source-training labels | Seed | STB test | STA test | QNRF test, 1024-pixel tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPCount full-label anchor | 320 / 320 | 2023 | 7.7991 / 13.3822 | 110.6855 / 181.4593 | 242.7133 / 418.9947 |
| MPCount sparse | 32 / 320 | 1 | 8.9392 / 15.8303 | 102.2181 / 172.8133 | pending |
| MPCount sparse | 32 / 320 | 2 | 9.9386 / 16.8172 | 116.4645 / 205.1531 | pending |
| MPCount sparse | 32 / 320 | 3 | pending | pending | pending |
| MPCount sparse mean ± sample SD | 32 / 320 | 1–3 | pending | pending | pending |

Raw MPCount mean squared errors are retained for the RMSE calculation:

| Seed | STB test logged `mse` | STA test logged `mse` | QNRF test logged `mse` |
| ---: | ---: | ---: | ---: |
| 1 | 250.5985 | 29864.4378 | pending |
| 2 | 282.8193 | 42087.7813 | pending |
| 3 | pending | pending | pending |

For each domain, report absolute MAE degradation as sparse MAE minus the
full-label anchor MAE, and relative degradation as that difference divided by
the full-label anchor MAE. Compare source and unseen degradation together.
Because the full-label reference has one seed and the sparse runs use both
split and model seeds 1–3, these are descriptive comparisons, not a paired
seed-controlled significance test. Source-validation selection metrics and
checkpoint hashes are recorded above as each run completes.
