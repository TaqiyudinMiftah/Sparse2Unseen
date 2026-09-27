# MPCount full-label STB anchor — effective batch 16, seed 2023

Run date: 2026-09-27. This is the predeclared full-label anchor using physical
batch 4 and four gradient-accumulation minibatches. The training run completed
all 180 epochs; the checkpoint was selected on STB validation only.

## Protocol and provenance

- Source: ShanghaiTech Part B (STB), 320 labeled training images and 80 labeled
  source-validation images. No STA or QNRF data were used in training,
  checkpoint selection, or hyperparameter selection.
- Test sets: STB (316 images), ShanghaiTech Part A (STA; 182), and UCF-QNRF
  (334). The selected weights are unchanged across all three evaluations.
- Model: current-upstream MPCount `final` with its deterministic upsampler.
  The project-owned adapter preserves the model and losses, but performs one
  optimizer update per four physical batches of four (20 updates per epoch).
  Batch normalization still observes batches of four, so this is not
  bitwise-equivalent to physical batch 16. The scheduler remains once per
  epoch, as in MPCount.
- Training config: [`stb_100_train_accum4.yml`](../configs/mpcount/stb_100_train_accum4.yml).
  Test configs: [STB](../configs/mpcount/stb_100_effbs16_test_stb.yml),
  [STA](../configs/mpcount/stb_100_effbs16_test_sta.yml), and
  [QNRF](../configs/mpcount/stb_100_effbs16_test_qnrf.yml).
- Official MPCount checkout: `6eb06772bcf7dfb771c43a14d67146fce767f103`.
  The run used the UV-locked Python 3.10.12 / PyTorch 2.0.1 environment on a
  shared NVIDIA GeForce RTX 3060 12 GB GPU.
- Source-validation-selected checkpoint: epoch 179,
  `external/MPCount/logs/stb_100_seed2023_effbs16/best_179.pth`, SHA-256
  `ce5dc85a7c38195aa91b211e405fc5ef84480ef847f2194b16eb31dec856e9a2`.
  The 127 MB checkpoint and datasets remain local, ignored artifacts.
- Source-only training monitor:
  [W&B run](https://wandb.ai/Tim-1/Sparse2Unseen/runs/gjsnu0wi).

## Results

| Evaluation set | MAE | MPCount logged `mse` | RMSE |
| --- | ---: | ---: | ---: |
| STB validation, selected epoch 179 | 7.0770 | 161.4720 | 12.7072 |
| STB test (source) | 7.7991 | 179.0840 | 13.3822 |
| STA test (unseen) | 110.6855 | 32927.4856 | 181.4593 |
| QNRF test (unseen), 1024-pixel tiles | 242.7133 | 175556.5914 | 418.9947 |

RMSE is `sqrt(mean((predicted_count - ground_truth_count)^2))`, calculated
from MPCount's logged mean squared error. The STB validation row documents
checkpoint selection; all test metrics were measured after training.

STB and STA use a 10000-pixel inference patch setting, effectively whole-image
inference. QNRF uses a fixed 1024-pixel tile setting to fit the shared 12 GB
GPU. Tiled QNRF inference is a protocol deviation from the upstream
whole-image setting; its score must not be presented as an exact reproduction
of the published QNRF number.

## Comparison and interpretation

The [earlier physical-batch-4 run](mpcount_stb_100_seed2023_bs4.md) selected
epoch 171 on source validation. Relative to that run, effective-batch-16
accumulation changes the test results as follows:

| Test set | Physical batch 4 MAE/RMSE | Effective batch 16 MAE/RMSE |
| --- | ---: | ---: |
| STB | 7.2288 / 11.9558 | 7.7991 / 13.3822 |
| STA | 114.2961 / 206.7478 | 110.6855 / 181.4593 |
| QNRF, 1024-pixel tiles | 227.0574 / 384.3073 | 242.7133 / 418.9947 |

The accumulation run improves STA but worsens STB and QNRF. It is not a
uniform improvement, and a single seed does not establish a robust
generalization advantage. The target results were not used to change the
checkpoint or this predeclared protocol. Both runs use the current
deterministic upsampler, whereas the
[official original-checkpoint diagnostic](mpcount_stb_official_checkpoint_audit.md)
uses the earlier bilinear-upsample variant and scored 102.6109/183.0300 on
STA and 170.1513/300.5704 on tiled QNRF. These are different model variants,
not interchangeable checkpoints. The gap to the original QNRF result warrants
further source-only investigation before interpreting sparse-label outcomes;
it does not justify selecting a model or setting from QNRF test performance.

## Local audit trail

MPCount writes logs under the ignored `external/MPCount/logs/` tree:

| Run | Local log |
| --- | --- |
| Training and STB validation | `stb_100_seed2023_effbs16/log.txt` |
| STB test | `stb_100_seed2023_effbs16_test_stb/log.txt` |
| STA test | `stb_100_seed2023_effbs16_test_sta/log.txt` |
| QNRF test, 1024-pixel tiles | `stb_100_seed2023_effbs16_test_qnrf_ps1024/log.txt` |

The deterministic STB 10% splits for [seed 1](../splits/stb_10_seed1.json),
[seed 2](../splits/stb_10_seed2.json), and [seed 3](../splits/stb_10_seed3.json)
each contain 32 labeled and 288 unlabeled IDs from the same 320-image STB
training partition. The fixed 80 labeled STB validation images are outside
that sparse-training fraction and must be disclosed with every sparse result.
Identical split IDs must be shared across competing methods. Before sparse
training, investigate the current-upstream deterministic model's gap to the
official original B checkpoint using source-only diagnostics; do not tune on
STA or QNRF test metrics.
