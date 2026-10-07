# Benchmark preflight validation

These are source-only implementation checks from 2026-10-07, not benchmark
test scores. The one-epoch smoke runs disabled SSL warmup to exercise the
unlabeled path. Production keeps the declared ten-epoch warmup and 180 epochs.
No target dataset was evaluated in any preflight run.

| Path | Optimizer steps | Labeled visits | Unlabeled visits | STB validation MAE | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 label-only | 20 | 320 | 0 | 46.6791 | 3610.07 |
| B1 Mean Teacher, corrected startup | 20 | 320 | 320 | 37.8417 | 3738.28 |
| B3 ordinary SSL + MPCount | 20 | 320 | 320 | 112.6685 | 4648.60 |
| Domain-stability prototype | 20 | 320 | 320 | 112.7406 | 4654.48 |

The stability path's mean regional weight was 0.99239 in this early check.
These scores do not rank trained methods or justify target-domain claims.

A preliminary constant-decay EMA smoke had source-validation MAE 922.9470.
The production setup uses the original Mean Teacher startup correction with
decay ceiling 0.999, shared across all SSL methods. This decision was made
using source diagnostics and the primary implementation before production
specs were finalized. The preliminary run is preserved locally and excluded
from the production matrix.

The `resume_audit` smoke was deliberately interrupted after a source epoch
checkpoint had saved optimizer, scheduler, random generators, and best-model
state. The resumed invocation restored that state and finalized its two-epoch
completion record, selecting epoch 1 at STB validation MAE 25.5785. Its
checkpoint hash is `44b5fa4ca4aa0770228ed7a71613fcbe2a189cc3df0d593853f4fcd788a0f524`.
Unit tests separately prove that interrupted and continuous optimizer/EMA/RNG
updates produce identical weights, including sampler replay.

Validation: 93 tests passed, including source-path guards, rejection of partial
training before test evaluation, unlabeled annotation-file isolation, equality
with upstream MPCount's supervised training step, accumulation/exposure budgets,
and coordination of two queue workers. Lockfile and whitespace checks passed.
