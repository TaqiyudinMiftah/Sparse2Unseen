# Complete benchmark protocol, version 1

The primary matrix follows `docs/PROTOCOL.md`: source domains STB, STA, and
QNRF; each source is evaluated unchanged on its own test partition and the
other two datasets. Sparse training fractions are 5%, 10%, and 40%, with split
and model seeds 1, 2, and 3. Methods are B0 label-only, B1 Mean Teacher, B2
MPCount, B3 ordinary SSL + MPCount, and the domain-stability prototype. Full
label references use seed 2023 for B0 and B2. With no unlabeled images, B1
reduces to B0 and B3/the prototype reduce to B2; these references are reused
rather than training duplicate 100% runs. This gives **141 training runs**.
The existing STB full-label B2 and three STB 10% B2 runs are audited and reused.
JHU-CROWD++ remains the subsequent external extension specified in the protocol.

## Models and supervision

B0/B1 use upstream `DGModel_base`; B2/B3/the prototype use upstream
`DGModel_final`. All share MPCount's pretrained VGG16-BN encoder, density
decoder, deterministic upsampling, normalization, and full-resolution density
units. MPCount retains its memory/classification heads and supervised loss.
B0/B1 use density MSE on the same two labeled source views, without MPCount's
memory/classification/consistency mechanisms. These models differ in DG heads;
the standalone compact VGG density scaffold is not the primary comparator.
No upstream source files are edited.

B1 adds ordinary source Mean Teacher consistency. B3 adds that same consistency
to the full MPCount supervised objective. The prototype adds regional weighting
to B3's pseudo-density loss, isolating the proposed change. All SSL methods use
EMA decay ceiling 0.999, a ten-epoch supervised warmup, a ten-epoch linear ramp, and
unsupervised weight 1. The EMA update occurs after each optimizer step, including
warmup. The startup decay is `min(0.999, 1 - 1/(completed_updates+1))`, following
the [original Mean Teacher implementation](https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py).
This avoids leaving a newly initialized density head almost unchanged during
the early low-update-budget epochs. A source-only preflight exposed that
issue with the constant-decay scaffold before the production recipe was frozen.
SSL validation and final inference use the EMA model; B0/B2 use the
student. The model choice is fixed before target evaluation.

Unlabeled datasets receive only source image paths and IDs; annotation paths
and counts are stripped, and annotation files are never opened by the
unlabeled loader. Weak and strong views share crop/flip geometry. Labeled
views use upstream `DenClsDataset`; unlabeled views use the existing
image-only `DensityDataset` crop/photometric transforms. All methods use the
same labeled IDs for each source/fraction/seed.

The prototype's fixed settings are four photometrically diversified teacher
views, a 4x4 regional partition, and `exp(-4 * variance)` weights. Regional
variance uses densities divided by MPCount's log factor 1000, so regional sums
are in people. Canonical weak-view EMA predictions provide the pseudo labels;
the four extra views determine weights. B3 has no stability weights. This is
an experimental hypothesis, with no claim of novelty or benefit until results
and a current literature review support it.

## Training and validation budgets

Every run has 180 epochs, physical batch 4, and four-batch gradient
accumulation. The supervised exposure budget is `ceil(source_train_size/16)*16`
per epoch: 320/20 updates on STB, 240/15 on STA, and 1088/68 on QNRF. A
seeded sampler covers every image with visit counts differing by at most one;
this handles fractions whose labeled sizes do not divide the exposure budget.
After warmup, SSL uses the same number of unlabeled visits per epoch and covers
every unlabeled image. There is no cached cycling of already augmented batches.

AdamW uses learning rate 0.001 and weight decay 0.0001. To keep the original
MPCount anchor convention, OneCycleLR is configured with 300 epochs and the
source's updates per epoch, but is **stepped once per epoch**, as upstream does.
This is explicitly not the usual per-optimizer-update OneCycle schedule.
The historical STB runs retain their original stochastic trajectories; new
runs use explicitly separate, saved sampler/worker generators for labeled,
unlabeled, and validation data. Batch normalization sees physical batches of
four. AMP is disabled to match the MPCount anchor's arithmetic and environment.

Each source uses its fixed fully labeled validation partition: STB 80, STA 60,
QNRF 120. These labels are outside the nominal training-label fraction and must
be disclosed. Only source validation MAE selects checkpoints. QNRF source
validation uses 1024 tiles; STB/STA validation uses whole images. Target files,
target labels, target-derived statistics, and target adaptation are excluded
from training and model selection. All settings are frozen before new tests.

## Completion, evaluation, and storage

The preflight runs are isolated one-epoch checks; SSL warmup is disabled only
there to exercise its computation. They cannot enter the final results or
trigger target evaluation. Production specs retain the full warmup/epoch count.

The benchmark builder freezes code/config/split hashes, materializes source
roots as symlinks, and audits completed legacy checkpoints and test logs. The
runner starts up to two jobs when a GPU has at least 8192 MiB free for three
30-second checks. GPU locks coordinate only this project's workers; they do
not reserve the GPU against other users. No other users' processes are changed.

Rolling checkpoints save model, EMA, optimizer, scheduler, best model, global
RNG, sampler RNG, and worker RNG states after each complete epoch. An interrupted
epoch is replayed from that boundary with the same recipe. Failed jobs have at
most three attempts before being flagged for investigation. Completed jobs
retain the selected and final model weights; large optimizer/RNG snapshots are
compacted after verified completion. Existing historical artifacts are preserved.

Test evaluation is gated on successful 180-epoch completion and a matching
selected-checkpoint hash. The chosen model is evaluated unchanged on all three
test datasets: whole-image STB/STA and fixed 1024-pixel QNRF tiles, as in the
anchor. There is no target-dependent checkpoint choice or parameter update.
Each result retains per-image predictions and MAE/MSE/RMSE locally. The report
aggregates only audited completed results, with sample SD across sparse seeds.

```bash
uv run --no-sync python scripts/build_benchmark_suite.py --preflight
# Run the generated isolated smoke specs with train_benchmark.py --smoke-epochs 1.
uv run --no-sync python scripts/build_benchmark_suite.py
# Commit tested code and generated splits on the experiment branch, then:
uv run --no-sync python scripts/finalize_benchmark_suite.py
uv run --no-sync python scripts/run_benchmark_suite.py --launch
uv run --no-sync python scripts/report_benchmark_suite.py
```

Runtime specs, checkpoints, predictions, and queue state are under
`runs/benchmark_v1/`, ignored by Git. W&B logs source training and source
validation only, using existing machine authentication. No credentials are
stored in specs or reports. Code, splits, protocol, and aggregate reports are
pushed via the experiment PR. The final conclusion requires every primary
matrix entry and test domain to pass the completion audit.
