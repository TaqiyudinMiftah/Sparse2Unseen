# Sparse2Unseen

**Label-Efficient Single-Domain Generalization for Crowd Counting**

Sparse2Unseen is a research scaffold for studying a specific question:

> Can unlabeled images from one source domain compensate for scarce crowd annotations while improving generalization to completely unseen crowd domains, without using target-domain data during training?

The initial protocol follows the ShanghaiTech A/B and UCF-QNRF cross-domain setting used by MPCount, then adds sparse source-label regimes (5%, 10%, 40%, 100%).

## Research protocol

For source dataset `S`, split the source training set into:

- `S_L`: labeled source subset (5%, 10%, 40%, or 100%)
- `S_U`: remaining source images treated as unlabeled
- `T`: unseen target dataset; **never accessed during training**

Initial development experiment:

```text
source: ShanghaiTech Part B
labels: 10% (3 deterministic seeds)
unlabeled source: remaining 90%
targets: ShanghaiTech Part A, UCF-QNRF
```

## Baselines

1. `label_only`: train only on the sparse labeled source subset.
2. `mean_teacher`: add unlabeled source images with an EMA teacher.
3. `sparse_mpcount`: run official MPCount using only the sparse labeled source subset.
4. `ssl_dg`: prototype domain-stable pseudo-labeling using unlabeled source images.

The repository intentionally keeps the first three baselines separate from the proposed method so that gains can be attributed cleanly.

## Repository layout

```text
configs/experiments/       experiment YAMLs
data/manifests/            JSONL manifests (not raw datasets)
docs/                      protocol and MPCount integration notes
external/                  external repositories (gitignored)
splits/                    deterministic labeled/unlabeled split JSONs
src/sparse2unseen/         research code
tools/                     manifest/split utilities
train.py                   training entry point
evaluate.py                cross-domain evaluation entry point
```

## Environment

For the standalone Sparse2Unseen scaffold:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For exact MPCount reproduction, use the upstream versions documented in `docs/MPCOUNT_INTEGRATION.md` (Python 3.10.12, PyTorch 2.0.1, torchvision 0.15.2).

## 1. Prepare datasets with MPCount

The easiest reproducible path is to use the official MPCount preprocessing for ShanghaiTech A/B and UCF-QNRF. Clone it into `external/MPCount`:

```bash
bash scripts/bootstrap_mpcount.sh
```

Then follow `docs/MPCOUNT_INTEGRATION.md` to preprocess the datasets and generate density maps.

Raw datasets and processed images are intentionally excluded from git.

## 2. Build manifests

After preprocessing, create a JSONL manifest. Example:

```bash
python tools/build_manifest.py \
  --root /path/to/mpcount/data/stb \
  --phase train \
  --output data/manifests/stb_train.jsonl
```

Repeat for source/target test sets.

Each row contains an ID, image path, point annotation path, optional density-map path, and ground-truth count.

## 3. Generate deterministic sparse-label splits

```bash
python tools/generate_splits.py \
  --manifest data/manifests/stb_train.jsonl \
  --fraction 0.10 \
  --seed 1 \
  --output splits/stb_10_seed1.json
```

For the paper, generate seeds 1, 2, and 3 at 5%, 10%, 40%, and 100%.

## 4. Sanity-check the first experiment

The repository can validate configuration/splits before any GPU run:

```bash
python train.py --config configs/experiments/stb_10_label_only.yaml --dry-run
```

Once manifest paths are populated, train with:

```bash
python train.py --config configs/experiments/stb_10_label_only.yaml
python train.py --config configs/experiments/stb_10_mean_teacher.yaml
python train.py --config configs/experiments/stb_10_ssl_dg.yaml
```

Evaluate on all configured domains:

```bash
python evaluate.py \
  --config configs/experiments/stb_10_label_only.yaml \
  --checkpoint runs/stb_10_label_only_seed1/best.pt
```

## Domain-stable pseudo-label prototype

For an unlabeled source image, an EMA teacher predicts on multiple photometric/domain-diversified views. The image is divided into a region grid. The variance of regional predicted counts becomes a stability score:

```text
low variance  -> high pseudo-label weight
high variance -> low pseudo-label weight
```

This is implemented in `src/sparse2unseen/confidence/region_stability.py` and is deliberately isolated so it can be ablated against ordinary Mean Teacher training.

## Reproducibility rules

- Target-domain images are forbidden during training and model selection.
- Sparse split IDs are stored in JSON and committed.
- Report mean ± std over at least 3 sparse-label seeds.
- Tune hyperparameters only on source-domain validation data.
- Report source in-domain MAE/RMSE alongside every unseen-domain result.
- Keep a 100%-label result in each source-domain table as an annotation-cost reference.

## Upstream reference

The project protocol and reproduction setup are based in part on:

- Peng & Chan, **Single Domain Generalization for Crowd Counting**, CVPR 2024 (MPCount).
- Official code: `https://github.com/Shimmer93/MPCount`

Sparse2Unseen does not vendor MPCount source code. The bootstrap script checks out the upstream repository into `external/MPCount` so its license and history remain intact.

## Status

This initial scaffold is meant to establish B0/B1 and the first domain-stability prototype. The next milestone is to reproduce `SHB -> SHA/QNRF` with 100% source labels, then rerun at 10% labels before modifying the method.
