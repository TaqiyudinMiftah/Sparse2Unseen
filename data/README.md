# Data

Do not commit raw crowd-counting datasets to this repository.

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

Use `tools/build_manifest.py` to index a processed phase into JSONL.
