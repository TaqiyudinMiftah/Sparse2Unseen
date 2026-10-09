# Source-only EMA batch-normalization diagnostic

Completed **2026-10-07 at 06:09:55 UTC**, before any new production run reached
test evaluation. This is a diagnostic of an intermediate checkpoint, **not a
primary benchmark result or a replacement model**.

## Controlled comparison

The active STB 10% Mean Teacher seed-1 run supplied its completed zero-based
epoch-36 EMA weights. The diagnostic retained those input weights, source spec,
and provenance before evaluating all 80 STB validation images on CPU.

A scratch copy then recomputed only its 20 BN layers' running statistics on
32 source training images: the first 32 lexicographically sorted training IDs,
irrespective of labeled membership. Calibration used deterministic 320-pixel
center crops, batches of four, native 0.5 normalization, cumulative BN
statistics, and disabled dropout. No training annotations were read during
calibration. Model parameters were checked to remain identical. The same
80-image source validation partition was evaluated again.

| Intermediate EMA variant | Source-val MAE | Source-val RMSE | Validation images |
| --- | ---: | ---: | ---: |
| Original copied student BN buffers | 359.519981 | 364.362123 | 80 |
| Scratch source-training BN recalibration | 15.558716 | 23.752668 | 80 |

Changing BN statistics alone had a large effect at this checkpoint. This
supports BN-statistics sensitivity as a contributor to the early teacher
counting problem. It does not isolate averaging lag from all other possible
BN distribution effects, demonstrate an implementation error, or establish
any unseen-domain benefit. No recalibrated test result was computed.

The unchanged production Mean Teacher subsequently completed all 180 epochs
and selected **epoch 131**, with source-validation MAE **13.382863**, before
its fixed test evaluation. Thus the early spike did not remain its final
selected validation score. The scratch diagnostic was not used to replace
the live model, change the recipe, or select a benchmark checkpoint.

## Audit and retained artifacts

The two sets of per-image predictions were independently checked against
all 80 canonical STB validation IDs and counts; MAE/MSE/RMSE were recomputed.
The 32 calibration IDs are unique source training IDs and do not intersect
validation. Input checkpoint/spec/code hashes match the retained artifacts.
The tool's parameter-preservation, source-image isolation, and snapshot guards
are covered by the **127 passing repository tests**.

Artifacts remain local, outside the primary `jobs/` result tree:

```text
runs/benchmark_v1/diagnostics/stb_10_mean_teacher_seed1_v1/
  ema_bn_epoch36_20261007T055319735998Z/
    input_teacher.pt
    source_spec.json
    provenance.json
    original_source_val.json
    diagnostic.json
```

Input checkpoint SHA256:
`cdb1cfd9dd278fc431086cc3de6e4f196a5212119cf2608e73425896e0c74afd`.
The diagnostic implementation is retained at commit `eb10859`; the production
training recipe remains frozen at `63aa4d2`. The CPU original-EMA score differs
slightly from the GPU training log at the same epoch (359.690443 / 364.531990);
this report does not claim bitwise CPU/GPU parity.

Historical command used while the rolling EMA snapshot was active:

```bash
uv run --no-sync python scripts/diagnose_source_ema_bn.py \
  --spec runs/benchmark_v1/specs/stb_10_mean_teacher_seed1_v1.json \
  --threads 2
```

The command reads an active rolling snapshot, not the archived diagnostic
weights. Completed production snapshots are compacted; the retained
`input_teacher.pt` and `source_spec.json` preserve the exact diagnostic input
for reconstruction after completion. Diagnostic scores are explicitly excluded
from [primary results](benchmark_progress.md) and the final conclusion gate.
