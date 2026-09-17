# Experimental Protocol

## Main question

Can unlabeled images from a single source dataset compensate for scarce point annotations while improving direct generalization to unseen crowd datasets?

## Initial datasets

- ShanghaiTech Part A (STA)
- ShanghaiTech Part B (STB)
- UCF-QNRF (QNRF)

Main cross-domain matrix:

- STA -> STB, QNRF
- STB -> STA, QNRF
- QNRF -> STA, STB

JHU-CROWD++ is reserved as an additional external generalization test after the A/B/Q protocol is stable.

## Label fractions

5%, 10%, 40%, 100% of source training images are labeled. Remaining source images are treated as unlabeled for SSL regimes.

For 5%, 10%, and 40%, run at least three deterministic seeds and report mean ± standard deviation.

## Forbidden leakage

No target-domain image may be used for:

- training
- augmentation statistics
- hyperparameter selection
- checkpoint selection
- pseudo-label generation
- normalization/statistics adaptation

Target datasets are evaluation-only.

## Baseline ladder

- B0: label-only sparse source
- B1: Mean Teacher using labeled + unlabeled source
- B2: sparse-label MPCount
- B3: naive SSL + DG
- Ours: domain-stability-weighted pseudo labels

## Metrics

Primary: image-level count MAE and RMSE.

Always report in-domain source-test performance next to unseen-domain performance. This exposes whether an SSL method only helps the source distribution.
