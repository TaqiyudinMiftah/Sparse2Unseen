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

The wrapper calls MPCount's own preprocessing scripts rather than reimplementing them. It also handles three integration details automatically:

1. The common ShanghaiTech archive uses `part_A_final` / `part_B_final`, while MPCount's preprocessing logic distinguishes STA from STB using the literal origin-directory basename `part_A` / `part_B`. The wrapper creates safe symlink aliases under `data/raw/.mpcount_alias/`.
2. Some ShanghaiTech mirrors spell the annotation directory `ground_truth`, while MPCount expects `ground-truth`. The wrapper validates image/annotation pairs and creates compatibility symlinks when needed.
3. Kaggle or other mirrors may add an extra parent directory around UCF-QNRF. The wrapper searches for the directory that actually contains both `Train/` and `Test/`.

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

## Full-label STB baseline

Train the official MPCount model with all 320 STB training images. The
project-owned config uses only STB training and validation data during training
and model selection:

```bash
cd external/MPCount
../../.venv/bin/python main.py --task train --config ../../configs/mpcount/stb_100_train.yml
cd ../..
```

The default training config preserves the upstream STB model and optimizer
settings, including batch size 16. This batch size may exceed the memory of a
12 GB GPU; record any smaller batch size as a deviation from upstream. Do not
use STA or QNRF performance to choose a checkpoint or hyperparameters.

For the tested batch-4 variant on a shared 12 GB GPU, use the separately named
run and evaluation configs. The training config differs from the upstream-style
one only in batch size and run name:

```bash
cd external/MPCount
../../.venv/bin/python main.py --task train --config ../../configs/mpcount/stb_100_train_bs4.yml
cd ../..
uv run python scripts/select_mpcount_checkpoint.py \
  --run-dir external/MPCount/logs/stb_100_seed2023_bs4
cd external/MPCount
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_bs4_test_stb.yml
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_bs4_test_sta.yml
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_bs4_test_qnrf.yml
cd ../..
```

Report the batch-4 result separately from the upstream batch-16 reference. A
smaller batch changes the number of optimizer updates per epoch, even though
the model and other config values are unchanged.
For this run, whole-image QNRF inference exceeded the memory of the shared
12 GB GPU. The batch-4 QNRF test config therefore uses MPCount's existing
1024-pixel patch inference and writes to a separately named log directory.
Report this evaluation setting with the QNRF result.

MPCount saves the validation-selected model as `best_<epoch>.pth`, whereas its
test configurations refer to `best.pth`. After training completes, create that
link using the completed STB training log:

```bash
uv run python scripts/select_mpcount_checkpoint.py
```

Then evaluate the same checkpoint on the source test set and both unseen test
sets:

```bash
cd external/MPCount
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_test_stb.yml
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_test_sta.yml
../../.venv/bin/python main.py --task test --config ../../configs/mpcount/stb_100_test_qnrf.yml
cd ../..
```

Each evaluation writes to a separate directory under `external/MPCount/logs/`.
MPCount logs `mae` and `mse`, where its `mse` is the **mean squared error**;
report RMSE as the square root of that logged value. The upstream
`stb_test_qnrf.yml` currently points to `data/sta`; the project-owned QNRF
config above uses `data/qnrf`.

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
