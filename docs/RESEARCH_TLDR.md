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
- sparse-source unseen-domain transfer -- TMTB already evaluates this combination;
- semi-supervised crowd counting -- established by multiple recent methods including TMTB, P2R, and S4Crowd;
- target-unlabeled adaptation -- P2R also reports UDA experiments, which are different from the zero-target-data setting here.

The intended contribution is an **audited label-efficiency study and a tested
training mechanism for the interaction between source pseudo-label reliability
and unseen-domain generalization**, not the invention of the setting.

A defensible claim should therefore be phrased conservatively:

> Sparse2Unseen tests whether regional prediction stability improves a matched
> source-only SSL + DG baseline across label budgets and unseen domains.

The earlier absence-of-prior-work statement is withdrawn. The
[October audit](NOVELTY_AUDIT_2026_10.md) records the overlap and publication
limitations. A different statistic alone does not establish a novel principle.

## Current milestone

The full-label STB MPCount anchor and three-seed STB 10% MPCount comparison
are complete. The remaining [primary benchmark](BENCHMARK_SUITE.md) is frozen
and running, beginning with STB 10% B0/B1. The
[progress report](../reports/benchmark_progress.md) tracks all required runs.
The final conclusion must include negative or mixed findings and distinguish
completed baseline evidence from pending SSL/prototype results.

See also:

- [Experimental protocol](PROTOCOL.md)
- [Literature review 2023-2026](LITERATURE_REVIEW_2023_2026.md)
- [MPCount integration](MPCOUNT_INTEGRATION.md)
