# Data

Do not commit raw crowd-counting datasets to this repository.

## Download raw datasets

Sparse2Unseen includes a downloader for the two datasets used in the first experiments:

```bash
# Download + extract ShanghaiTech A/B and UCF-QNRF
uv run python scripts/download_datasets.py all

# Or download them individually
uv run python scripts/download_datasets.py shanghaitech
uv run python scripts/download_datasets.py ucf_qnrf

# Preview actions without downloading
uv run python scripts/download_datasets.py all --dry-run
```

The default destination is `data/raw/`. Use `--dest /path/to/raw` to store the datasets elsewhere.

The downloader uses:

- ShanghaiTech A/B: a public `ShanghaiTech.zip` Google Drive mirror linked by the official TencentYoutuResearch SASNet repository. The script uses `gdown` so Google Drive confirmation pages are handled correctly.
- UCF-QNRF: the University of Central Florida CRCV archive (`UCF-QNRF_ECCV18.zip`). Direct HTTPS downloads use `requests` with the `certifi` CA bundle rather than relying on the host/container CA store.

Direct HTTP(S) downloads support resume when the server accepts HTTP Range requests. Google Drive downloads use `gdown` resume support. Every downloaded archive is checked for a ZIP signature before extraction, and extracted folders are checked for the expected dataset markers. By default the ZIP is removed after successful extraction; pass `--keep-archive` to retain it.

If the official UCF server still fails TLS verification even with `certifi`, you can explicitly opt into an insecure transport fallback:

```bash
uv run python scripts/download_datasets.py ucf_qnrf --insecure-ssl
```

Only use `--insecure-ssl` for the known official UCF URL after verifying the URL printed by the script. It disables certificate verification for that direct HTTPS download.

If an older checkout left an HTML response such as `data/raw/ShanghaiTech.zip.part`, the downloader automatically removes that stale non-ZIP partial file before retrying from Google Drive. You can also force a clean download with:

```bash
uv run python scripts/download_datasets.py shanghaitech --force
```

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
