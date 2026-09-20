# Research TL;DR

## One-sentence version

**Sparse2Unseen studies whether a crowd-counting model can learn from a very small labeled subset plus the remaining unlabeled images of a single source domain, then generalize directly to completely unseen crowd datasets without using any target-domain data during training or model selection.**

## Why this problem matters

Crowd counting still depends heavily on point annotations, which are expensive in dense scenes. Semi-supervised methods reduce annotation cost, but they are usually evaluated in-domain. Domain-generalization methods improve transfer to unseen scenes, but they usually assume the source domain is fully labeled.

Sparse2Unseen focuses on the intersection:

> **few labeled source images + many unlabeled source images -> unseen target domains, with zero target access**

The research question is therefore:

> **Can unlabeled source data compensate for scarce source annotations while also improving generalization to previously unseen crowd domains?**

## Formal setting

For a single source dataset:

- `S_L`: labeled source subset, using 5%, 10%, 40%, or 100% of the source training images.
- `S_U`: remaining source images, treated as unlabeled.
- `T`: unseen target dataset, used only for final evaluation.

Target-domain images are not used for training, pseudo-labeling, augmentation statistics, hyperparameter selection, checkpoint selection, normalization adaptation, or validation.

## Initial benchmark protocol

Datasets:

- ShanghaiTech Part A (STA)
- ShanghaiTech Part B (STB)
- UCF-QNRF (QNRF)
- JHU-CROWD++ later as an additional external generalization test

Main source-to-target matrix:

- STA -> STB, QNRF
- STB -> STA, QNRF
- QNRF -> STA, STB

The first development path is:

```text
source: STB
source labels: 10%
unlabeled source: remaining 90%
targets: STA and QNRF
seeds: 1, 2, 3
```

## Baseline ladder

The experiments are deliberately incremental:

| ID | Training setup | Purpose |
| --- | --- | --- |
| B0 | Sparse labeled source only | Measures the cost of source-label scarcity |
| B1 | Mean Teacher with labeled + unlabeled source | Measures ordinary semi-supervised gains |
| B2 | Sparse-label MPCount | Measures source-only domain generalization under label scarcity |
| B3 | Naive SSL + DG | Tests whether simply combining SSL and DG is already sufficient |
| Ours | Domain-stability-weighted pseudo supervision | Tests the Sparse2Unseen hypothesis |

## Core method hypothesis

Ordinary semi-supervised learning often trusts a pseudo label because the teacher is confident.

Sparse2Unseen asks for a stronger criterion:

> **A useful pseudo label for domain generalization should be both confident and stable under plausible domain changes.**

For an unlabeled source image, the teacher is evaluated across multiple domain-diversified views. The image is divided into regions, regional predicted counts are compared, and unstable regions receive lower pseudo-label weight.

Conceptually:

```text
low prediction variance across domain perturbations
    -> domain-stable region
    -> high pseudo-label weight

high prediction variance across domain perturbations
    -> source-style-sensitive region
    -> low pseudo-label weight
```

The initial objective is organized around:

- supervised counting loss,
- source pseudo-label loss,
- consistency / stability loss,
- existing source-only domain-generalization loss.

The current prototype lives in `src/sparse2unseen/confidence/region_stability.py`.

## What the paper needs to demonstrate

A successful result should show more than lower in-domain MAE.

The central comparison is:

| Method | Source labels | Unlabeled source | Target data during training | In-domain | Unseen-domain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Label-only | 5/10/40% | No | No | ... | ... |
| SSL | 5/10/40% | Yes | No | ... | ... |
| Sparse DG | 5/10/40% | No | No | ... | ... |
| Naive SSL + DG | 5/10/40% | Yes | No | ... | ... |
| Sparse2Unseen | 5/10/40% | Yes | No | ... | ... |
| Full-label reference | 100% | -- | No | ... | ... |

Primary metrics:

- MAE
- RMSE
- mean +/- standard deviation over sparse-label seeds

If the method later becomes point-based, localization precision / recall / F1 should also be reported.

## The empirical question to answer first

Before adding more model complexity:

> **Does reducing source annotation disproportionately damage unseen-domain performance, and does ordinary source-domain SSL mainly recover in-domain performance rather than cross-domain performance?**

If the answer is yes, that directly motivates Sparse2Unseen.

## Novelty boundaries

The project should **not** claim the following as novel by themselves:

- single-domain generalization for crowd counting -- MPCount and later work already study it;
- uncertainty-guided domain generalization -- UGSDA already combines uncertainty and source-only DG;
- semi-supervised crowd counting -- established by multiple recent methods including TMTB, P2R, and S4Crowd;
- target-unlabeled adaptation -- P2R also reports UDA experiments, which are different from the zero-target-data setting here.

The intended contribution is the **label-efficient source-only DG setting and a training mechanism designed for the interaction between source pseudo-label reliability and unseen-domain generalization**.

A defensible claim should therefore be phrased conservatively:

> In the reviewed 2023-2026 literature, we did not find a method whose central evaluation protocol combines a sparsely labeled single source domain, the remaining unlabeled source data, and direct deployment to completely unseen target crowd datasets with no target-domain access.

This is a working literature-based hypothesis, not a substitute for a final novelty search before submission.

## Current milestone

1. Reproduce the 100%-label MPCount STB -> STA/QNRF baseline.
2. Freeze deterministic 5%, 10%, and 40% source splits.
3. Run B0 at STB 10%.
4. Run B1 Mean Teacher at STB 10%.
5. Measure in-domain vs unseen-domain degradation.
6. Run sparse MPCount and naive SSL + DG.
7. Only then evaluate domain-stability-weighted pseudo supervision.

See also:

- [Experimental protocol](PROTOCOL.md)
- [Literature review 2023-2026](LITERATURE_REVIEW_2023_2026.md)
- [MPCount integration](MPCOUNT_INTEGRATION.md)
