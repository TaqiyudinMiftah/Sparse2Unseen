# MPCount integration

Sparse2Unseen uses MPCount as an external baseline rather than copying its source code.

Official repository:

`https://github.com/Shimmer93/MPCount`

The upstream README specifies:

- Python 3.10.12
- PyTorch 2.0.1
- torchvision 0.15.2
- preprocessed roots named `sta`, `stb`, `qnrf`
- 320 x 320 training crops in the published configs

## Bootstrap

```bash
bash scripts/bootstrap_mpcount.sh
```

## Preprocess

From `external/MPCount`, follow the upstream commands. Conceptually:

```bash
python utils/preprocess_data.py --dataset sta --origin-dir /path/to/ShanghaiTech/part_A --data-dir data/sta
python utils/preprocess_data.py --dataset stb --origin-dir /path/to/ShanghaiTech/part_B --data-dir data/stb
python utils/preprocess_data.py --dataset qnrf --origin-dir /path/to/UCF-QNRF --data-dir data/qnrf
python utils/dmap_gen.py --path data/sta
python utils/dmap_gen.py --path data/stb
python utils/dmap_gen.py --path data/qnrf
```

Check the current upstream README/scripts before running; preprocessing details may change.

## Sparse-label MPCount baseline

MPCount's dataset implementation enumerates image files in each `train` directory. To create B2 without modifying upstream code, materialize a sparse processed dataset root containing only the labeled training items while keeping source validation/test data unchanged.

Use:

```bash
python tools/materialize_sparse_mpcount_root.py \
  --source-root /path/to/external/MPCount/data/stb \
  --manifest data/manifests/stb_train.jsonl \
  --split splits/stb_10_seed1.json \
  --output-root /path/to/external/MPCount/data/stb_sparse10_seed1
```

The tool uses symlinks by default, so it does not duplicate the dataset.
