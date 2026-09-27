# Sparse MPCount B2 — STB 10%, three-seed protocol and results

Status: in progress, 2026-09-27. Do not use any STA or QNRF result to choose
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

## Results

The full-label reference is a single seed-2023 run, not a paired three-seed
estimate. Test scores are MAE / RMSE; target columns remain blank until a
source-validation-selected checkpoint completes training and is evaluated.

| Method | Source-training labels | Seed | STB test | STA test | QNRF test, 1024-pixel tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPCount full-label anchor | 320 / 320 | 2023 | 7.7991 / 13.3822 | 110.6855 / 181.4593 | 242.7133 / 418.9947 |
| MPCount sparse | 32 / 320 | 1 | pending | pending | pending |
| MPCount sparse | 32 / 320 | 2 | pending | pending | pending |
| MPCount sparse | 32 / 320 | 3 | pending | pending | pending |
| MPCount sparse mean ± sample SD | 32 / 320 | 1–3 | pending | pending | pending |

For each domain, report absolute MAE degradation as sparse MAE minus the
full-label anchor MAE, and relative degradation as that difference divided by
the full-label anchor MAE. Compare source and unseen degradation together.
Because the full-label reference has one seed and the sparse runs use both
split and model seeds 1–3, these are descriptive comparisons, not a paired
seed-controlled significance test. The raw source-validation selection metrics
and checkpoint hashes will be added when each run completes.
