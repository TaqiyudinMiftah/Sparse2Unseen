# Novelty and evidence audit — 7 October 2026

This update supports the eventual conclusion; it does not change the frozen
training recipe. Searches covered sparse-source SSL, single-source DG,
pseudo-label consistency/stability, and crowd-counting work from 2025–2026.
Only author, conference, journal, or institutional sources support the notes
below. This is a targeted check, not an exhaustive proof of novelty.

## Correction that changes the framing

[TMTB, CVPR 2025](https://arxiv.org/html/2503.17984v1), sections 3.1 and 4.1,
already combines source SSL with unseen-domain evaluation. Tables 3–4 use
40% source labels for STA↔QNRF and explicitly mark target data unavailable.
It also weights inpainted regions using weak/strong EMA classification
disagreement. This is both a protocol precedent and a reliability-method
collision. Its [official implementation](https://github.com/syhien/taste_more_taste_better)
is public.

Our inference: neither sparse-source DG nor augmentation-conditioned regional
reliability can be claimed as a new general principle. The current prototype
differs by using variance of regional density integrals over multiple
photometric views, but a different formula alone is not evidence of novelty
or usefulness. The earlier literature gap statement has been withdrawn.

## Other boundaries checked

| Work | Verified relevance | Boundary for this project |
| --- | --- | --- |
| [UGSDA, ACM MM 2024, author project](https://chenlab.comp.polyu.edu.hk/Papers/paper1_guanchen/paper1_guanchen.html) | Source-only counting with uncertainty-guided style diversification and density-distribution consistency. | Uncertainty + DG is established. |
| [P2R, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_Point-to-Region_Loss_for_Semi-Supervised_Point-Based_Crowd_Counting_CVPR_2025_paper.pdf) | Table 2 separates target-unlabeled adaptation from DG. | UDA results are not a no-target-access comparison. |
| [SinCount, Scientific Reports, April 2026](https://www.nature.com/articles/s41598-026-46286-3) | Frequency-aware single-source crowd-counting DG. | Generic single-domain transfer is not new. |
| [GBDGC preprint, March 2026, v2](https://arxiv.org/html/2603.24106v2) | Stable latent pseudo-domain discovery in source-only counting. | Its stability concerns domain assignments, not unlabeled crowd-density pseudo labels; distinguish the two. |
| [WQCount, CVIU, October 2026](https://www.sciencedirect.com/science/article/pii/S1077314226002638) | The publisher abstract describes scale-decoupled single-source DG. | Abstract-level check only; do not infer its sparse-label protocol or training details. |

CVF fetches intermittently returned HTTP 403. TMTB's author-linked arXiv full
text supplied the relevant sections; UGSDA's author project supplied its
method description. GBDGC is a preprint, not a verified conference acceptance.
WQCount's issue date is not evidence of its exact first online publication day.

## What the running benchmark can establish

The fixed B0/B1/B2/B3/prototype comparisons can test source-versus-unseen
degradation and whether stability weighting improves the same-network B3
baseline. Report all required seeds, MAE/RMSE, fully labeled validation costs,
training budgets, and the added teacher-view computation. Mixed or negative
results remain valid outcomes. Historical B2 stochastic trajectories differ
from the new sampler setup, so do not claim a bitwise paired B2/B3 comparison.

The primary matrix does not reproduce TMTB or UGSDA and does not contain a
complete view-count/weighting ablation study. Its completion therefore cannot
establish superiority over those methods, state of the art, or publication-
grade algorithmic novelty. Such comparisons would need a separately declared
protocol, not silent changes to the live matrix based on target results.

## Dataset-count citation check

The original [ShanghaiTech paper, section 3.2](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
reports STA 300 train / 182 test and STB 400 train / 316 test.
The original [UCF-QNRF paper, section 4](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.pdf)
reports 1201 train / 334 test.

Our local MPCount manifests were counted again on 7 October: STA
240/60/182, STB 320/80/316, QNRF 1081/120/334 (train/validation/test).
The reserved source-validation images are part of each canonical training
partition, not extra data. These checks agree with the current upstream split
files and preserve the canonical test sizes. Annotation budgets refer only
to the smaller MPCount training pools, with validation labels disclosed
separately in [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md).
