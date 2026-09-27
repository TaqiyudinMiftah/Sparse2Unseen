# Official MPCount B-source checkpoint: pipeline diagnostic

Run date: 2026-09-27. This is a one-time evaluation-path check before the
predeclared effective-batch-16 training run. It did not select a training
checkpoint or tune its settings.

## Provenance and compatibility

- Official MPCount repository: [`Shimmer93/MPCount`](https://github.com/Shimmer93/MPCount),
  checkout `6eb06772bcf7dfb771c43a14d67146fce767f103`.
- Official original B-source checkpoint from the repository's Google Drive
  link, file ID `1sYGMGNOqj0OUEz-5zE9S1G7hjOzmtJsZ`.
- Downloaded weight SHA-256:
  `9f2e73c62cd289d4a63786663e8fd1ae0833c365b6f84714c5541f8c6fc92b4e`.
  The 133 MB weight is ignored locally and is not committed to Git.
- Checked all 142 state-dict entries against MPCount `final` with
  `deterministic=False`, `pretrained=False`: no missing, unexpected, or
  shape/dtype-mismatched entries. A strict load succeeded. The original
  bilinear-upsample architecture is required for this checkpoint; the current
  MPCount checkout defaults to a different deterministic upsampler.
- Processed test sizes: STB 316, STA 182, QNRF 334. The official MPCount
  dataset and test trainer were used throughout. QNRF pointed to `data/qnrf`,
  avoiding the incorrect QNRF root in the upstream test config.

## Results

| Test set | Inference patch size | MAE | Logged mean squared error | RMSE |
| --- | ---: | ---: | ---: | ---: |
| STB (source) | 10000 | 10.2131 | 335.0823 | 18.3053 |
| STA (unseen) | 10000 | 102.6109 | 33499.9653 | 183.0300 |
| QNRF (unseen) | 1024 | 170.1513 | 90342.5612 | 300.5704 |

RMSE is the square root of MPCount's logged `mse` value. The [upstream
README](https://github.com/Shimmer93/MPCount) lists original B-source scores
of 99.6/182.9 on STA and 165.6/290.4 on QNRF (MAE/root-error convention).
Our STA MAE is 3.0% above that reference and RMSE is 0.1% above it, within
the predeclared 10% diagnostic tolerance. QNRF MAE/RMSE are 2.7%/3.5% above
the reference, but our 1024-pixel tiled inference is not the paper's
whole-image setting, so this is not an exact reproduction claim.

This check supports the STA test pipeline and provides a useful QNRF sanity
check. It does **not** validate training density-map generation, prove that
tiling is numerically equivalent to whole-image QNRF inference, or justify
target-derived training changes. The effective-batch-16 protocol was fixed
before these metrics were used.

## Local audit trail

The repository-owned test configs are
[STB](../configs/mpcount/stb_official_original_test_stb.yml),
[STA](../configs/mpcount/stb_official_original_test_sta.yml), and
[QNRF](../configs/mpcount/stb_official_original_test_qnrf.yml). The ignored
upstream logs are under `external/MPCount/logs/` in directories named
`stb_official_original_test_stb`, `stb_official_original_test_sta`, and
`stb_official_original_test_qnrf_ps1024`.
