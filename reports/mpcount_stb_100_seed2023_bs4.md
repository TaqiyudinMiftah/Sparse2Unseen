# MPCount full-label STB baseline — seed 2023, batch size 4

Run date: 2026-09-26. This is the first end-to-end, source-only MPCount baseline
for Sparse2Unseen. It is a hardware-constrained run, not an exact reproduction
of the upstream batch-16 configuration.

## Protocol and provenance

- Source: ShanghaiTech Part B (STB), with 320 labeled training images and 80
  source-validation images. Checkpoint selection used STB validation MAE only.
- Final evaluation: STB test (316 images), ShanghaiTech Part A (STA) test (182),
  and UCF-QNRF test (334). No target images or labels were read by the training
  or checkpoint-selection configuration.
- Model and optimizer: upstream MPCount `final` model and STB settings, seed
  2023, 180 epochs. The only training-config changes from
  [`stb_100_train.yml`](../configs/mpcount/stb_100_train.yml) were batch size
  16 → 4 and a separate run name. A batch-4 training step fit the shared 12 GB
  GPU; batches 8 and 16 exceeded an approximately 8 GB per-process memory cap
  while other jobs were active. This changes the number of optimizer updates
  per epoch, so the scores should not be presented as exact upstream results.
- Training config: [`stb_100_train_bs4.yml`](../configs/mpcount/stb_100_train_bs4.yml).
  Test configs: [STB](../configs/mpcount/stb_100_bs4_test_stb.yml),
  [STA](../configs/mpcount/stb_100_bs4_test_sta.yml), and
  [QNRF](../configs/mpcount/stb_100_bs4_test_qnrf.yml).
- Software: Python 3.10.12, PyTorch 2.0.1+cu117, torchvision 0.15.2+cu117,
  NumPy 1.26.4; NVIDIA GeForce RTX 3060, 12 GB. Official MPCount checkout:
  `6eb06772bcf7dfb771c43a14d67146fce767f103`. Sparse2Unseen base commit:
  `2262bf891bc787b24a7fa9a6276e09a34383653b`.
- Selected checkpoint: epoch 171 (`external/MPCount/logs/stb_100_seed2023_bs4/best_171.pth`),
  SHA-256 `61df05cd64eb156545bc1a7753a1ac957f09476b1eb4b8a0330c2d8857ba5170`.
  This 127 MB checkpoint and the datasets are local experiment artifacts, not
  stored in Git.

## Results

| Evaluation set | Images | MAE | MPCount logged `mse` | RMSE |
| --- | ---: | ---: | ---: | ---: |
| STB validation, selected epoch 171 | 80 | 7.4018 | 168.0633 | 12.9639 |
| STB test (source) | 316 | 7.2288 | 142.9401 | 11.9558 |
| STA test (unseen) | 182 | 114.2961 | 42744.6712 | 206.7478 |
| QNRF test (unseen) | 334 | 227.0574 | 147692.0879 | 384.3073 |

RMSE here is `sqrt(mean((predicted_count - ground_truth_count)^2))`, calculated
from MPCount's logged mean squared error. The validation row is included only
to document checkpoint selection; test rows were measured after training.

STB and STA used the original 10000-pixel inference patch setting, effectively
whole-image inference. Whole-image QNRF inference ran out of GPU memory after
18% of its test pass. The final QNRF test used MPCount's built-in 1024-pixel
tile inference, selected for memory before a complete QNRF score existed.
This is an evaluation-protocol deviation and should accompany every QNRF
comparison. The model weights were unchanged across all three test sets.

The [upstream MPCount README](https://github.com/Shimmer93/MPCount) reports
approximately B→A 99.6/182.9 and B→QNRF 165.6/290.4 (MAE and the paper's
root-error convention). This run's cross-domain errors are higher. The batch
size and QNRF inference differences limit attribution; no target-derived
hyperparameter change was made to improve these numbers.

## Local audit trail

MPCount writes its full logs and checkpoints under the ignored
`external/MPCount/logs/` tree:

| Run | Local log |
| --- | --- |
| Training and STB validation | `stb_100_seed2023_bs4/log.txt` |
| STB test | `stb_100_seed2023_bs4_test_stb/log.txt` |
| STA test | `stb_100_seed2023_bs4_test_sta/log.txt` |
| QNRF test, 1024-pixel tiles | `stb_100_seed2023_bs4_test_qnrf_ps1024/log.txt` |

The failed whole-image QNRF attempt has a separate incomplete log at
`stb_100_seed2023_bs4_test_qnrf/log.txt`; it contains no aggregate metric.

## Next experiment

Freeze this hardware-constrained protocol before any sparse run. Generate
deterministic STB 10% source-training splits (32 labeled of 320) for seeds
1, 2, and 3, and use identical labeled IDs across methods. First validate the
sparse MPCount data root and run one seed with STB-only model selection. Fix
the custom trainer's training-loss checkpoint selection before treating its
label-only or SSL results as research baselines. STA and QNRF remain final
evaluation sets only.
