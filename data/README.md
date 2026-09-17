# Data

Do not commit raw crowd-counting datasets to this repository.

## Download raw datasets

Sparse2Unseen includes a resumable downloader for the two datasets used in the first experiments:

```bash
# Download + extract ShanghaiTech A/B and UCF-QNRF
python scripts/download_datasets.py all

# Or download them individually
python scripts/download_datasets.py shanghaitech
python scripts/download_datasets.py ucf_qnrf

# Preview actions without downloading
python scripts/download_datasets.py all --dry-run
```

The default destination is `data/raw/`. Use `--dest /path/to/raw` to store the datasets elsewhere.

The downloader uses:

- ShanghaiTech A/B: the long-standing Dropbox raw archive referenced by crowd-counting implementations (`ShanghaiTech.zip`).
- UCF-QNRF: the University of Central Florida CRCV archive (`UCF-QNRF_ECCV18.zip`).

It supports partial-download resume when the server accepts HTTP Range requests, validates ZIP signatures, extracts the archives, and checks for the expected dataset folders. By default the ZIP is removed after successful extraction; pass `--keep-archive` to retain it.

Please follow the datasets' respective research/use terms and cite their original papers.

## Processed layout

Recommended processed layout (compatible with MPCount-style preprocessing):

```text
data_root/
  train/
    IMG_1.jpg
    IMG_1.npy          # point coordinates (N x 2)
    IMG_1_dmap.npy     # density map (optional but required by this trainer)
  val/
  test/
```

After downloading the raw data, follow `docs/MPCOUNT_INTEGRATION.md` for preprocessing, then use `tools/build_manifest.py` to index a processed phase into JSONL.
