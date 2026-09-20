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

## Automated preparation

After downloading the raw datasets, run:

```bash
uv run python scripts/prepare_datasets.py all
```

The wrapper calls MPCount's own preprocessing scripts rather than reimplementing them. It also handles two integration details automatically:

1. The common ShanghaiTech archive uses `part_A_final` / `part_B_final`, while MPCount's preprocessing logic distinguishes STA from STB using the literal origin-directory basename `part_A` / `part_B`. The wrapper creates safe symlink aliases under `data/raw/.mpcount_alias/`.
2. Kaggle or other mirrors may add an extra parent directory around UCF-QNRF. The wrapper searches for the directory that actually contains both `Train/` and `Test/`.

Outputs are written to:

```text
data/processed/mpcount/
  sta/
  stb/
  qnrf/
```

Those directories are then symlinked into `external/MPCount/data/`, so the upstream configs continue to work unchanged.

The wrapper validates phase sizes using MPCount's checked-in split files, runs `utils/dmap_gen.py`, and generates the nine Sparse2Unseen manifests automatically.

Useful options:

```bash
# Show paths/commands without processing
uv run python scripts/prepare_datasets.py all --dry-run

# Rebuild one dataset from scratch
uv run python scripts/prepare_datasets.py stb --force

# Skip expensive density-map generation temporarily
uv run python scripts/prepare_datasets.py all --skip-density
```

For reference, the equivalent upstream commands are still:

```bash
python utils/preprocess_data.py --dataset sta --origin-dir /path/to/ShanghaiTech/part_A --data-dir data/sta
python utils/preprocess_data.py --dataset stb --origin-dir /path/to/ShanghaiTech/part_B --data-dir data/stb
python utils/preprocess_data.py --dataset qnrf --origin-dir /path/to/UCF-QNRF --data-dir data/qnrf
python utils/dmap_gen.py --path data/sta
python utils/dmap_gen.py --path data/stb
python utils/dmap_gen.py --path data/qnrf
```

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
