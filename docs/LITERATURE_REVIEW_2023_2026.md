# Crowd Counting Literature Review, 2023-2026

**Scope:** recent work most relevant to label efficiency, point supervision, domain generalization, test-time adaptation, video, multimodal counting, robustness, and uncertainty.

**Last consolidated:** 20 September 2026.

This document records the literature review used to define the Sparse2Unseen research problem. It is not intended to be an exhaustive survey of all crowd-counting papers.

## Main takeaway

Recent crowd-counting research has several mature lines:

1. **Semi-supervised / weakly supervised counting** reduces annotation cost.
2. **Point-based counting and localization** improves interpretable individual-level prediction.
3. **Single-domain / cross-scene generalization** targets deployment to unseen environments.
4. **Test-time adaptation, video, multimodal sensing, and robustness** address increasingly realistic deployment settings.

The most relevant gap for Sparse2Unseen is at the intersection of the first two major deployment constraints:

> **Only a small subset of one source domain is labeled, the rest of that source is unlabeled, and the trained model must generalize to completely unseen target datasets without seeing target images during training.**

Within the reviewed set, this exact protocol is not the central setting of the closest semi-supervised or domain-generalization methods.

## Literature matrix

| Year | Paper | Venue | Theme | Supervision | Relevance / main idea | Limitation relative to Sparse2Unseen | Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | [CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model](https://openaccess.thecvf.com/content/CVPR2023/html/Liang_CrowdCLIP_Unsupervised_Crowd_Counting_via_Vision-Language_Model_CVPR_2023_paper.html) | CVPR | Unsupervised / VLM | No crowd labels | Uses CLIP-derived vision-language priors and ranking prompts to count without standard crowd annotations. | Zero-label learning is the focus; explicit source-only cross-domain generalization is not. | [GitHub](https://github.com/dk-liang/CrowdCLIP) |
| 2023 | [Optimal Transport Minimization: Crowd Localization on Density Maps for Semi-Supervised Counting](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.html) | CVPR | Semi-supervised / localization | Point labels + unlabeled images | Converts density estimates to pseudo points with optimal transport and confidence weighting. | Pseudo-label reliability is studied mainly for semi-supervised learning rather than source-only DG. | [GitHub](https://github.com/Elin24/OT-M) |
| 2023 | [Calibrating Uncertainty for Semi-Supervised Crowd Counting](https://openaccess.thecvf.com/content/ICCV2023/html/LI_Calibrating_Uncertainty_for_Semi-Supervised_Crowd_Counting_ICCV_2023_paper.html) | ICCV | Semi-supervised / uncertainty | Point labels + unlabeled images | Uses patch-wise uncertainty to select more reliable pseudo supervision. | Important uncertainty baseline, but unseen-domain generalization is not explicitly optimized. | -- |
| 2023 | [Point-Query Quadtree for Crowd Counting, Localization, and More (PET)](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html) | ICCV | Point counting / localization | Full point supervision | Point-query transformer with quadtree structure for counting and localization. | Strong point baseline, but requires dense point labels and does not target domain shift. | [GitHub](https://github.com/cxliu0/PET) |
| 2023 | [Counting Crowds in Bad Weather](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Counting_Crowds_in_Bad_Weather_ICCV_2023_paper.html) | ICCV | Robustness | Fully supervised | Improves robustness to haze, rain, and snow using weather-aware modeling. | Targets weather degradation rather than sparse-source annotation plus generic unseen-domain shift. | -- |
| 2023 | [Dynamic Mixture of Counter Network for Location-Agnostic Crowd Counting](https://openaccess.thecvf.com/content/WACV2023/html/Wang_Dynamic_Mixture_of_Counter_Network_for_Location-Agnostic_Crowd_Counting_WACV_2023_paper.html) | WACV | Weak supervision | Image-level counts | Learns from total-count labels without point annotations using a dynamic mixture of counters. | Reduces annotation cost but does not provide the source-only DG setting. | -- |
| 2023 | [Domain-General Crowd Counting in Unseen Scenarios](https://ojs.aaai.org/index.php/AAAI/article/view/25131) | AAAI | Domain generalization | Fully labeled source | Learns domain-invariant / domain-specific memories after splitting the source into subdomains. | Core DG prior work, but assumes a labeled source rather than a sparse labeled + unlabeled source. | [GitHub](https://github.com/ZPDu/Domain-general-Crowd-Counting-in-Unseen-Scenarios) |
| 2023 | [Semi-Supervised Crowd Counting via Multiple Representation Learning](https://doi.org/10.1109/TIP.2023.3313490) | IEEE TIP | Semi-supervised | Labeled + unlabeled source | Combines multiple density representations, consistency, and distribution matching. | Focuses on in-domain semi-supervision rather than source-only DG. | -- |
| 2023 | [Multi-task Semi-supervised Crowd Counting via Global to Local Self-correction](https://www.sciencedirect.com/science/article/pii/S0031320323002066) | Pattern Recognition | Semi-supervised / multi-task | Labeled + unlabeled images | Uses density regression plus auxiliary tasks and global/local pseudo-label correction. | Additional tasks improve SSL but do not directly solve unseen-domain generalization. | -- |
| 2023 | [Motional Foreground Attention-based Video Crowd Counting](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005897) | Pattern Recognition | Video | Supervised video | Uses motion foreground attention and frame differences for temporal counting. | Temporal modeling is central; sparse frame annotation and cross-domain transfer remain separate issues. | -- |
| 2024 | [Glance To Count: Learning To Rank With Anchors for Weakly-Supervised Crowd Counting](https://openaccess.thecvf.com/content/WACV2024/html/Xiong_Glance_To_Count_Learning_To_Rank_With_Anchors_for_Weakly-Supervised_WACV_2024_paper.html) | WACV | Weak supervision | Ranking + count anchors | Learns ordinal crowd-density ranking and calibrates it using a small number of count anchors. | Metric calibration still needs anchor counts; localization and source-only DG are not central. | -- |
| 2024 | [CrowdDiff: Multi-hypothesis Crowd Density Estimation using Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/html/Ranasinghe_CrowdDiff_Multi-hypothesis_Crowd_Density_Estimation_using_Diffusion_Models_CVPR_2024_paper.html) | CVPR | Generative density estimation | Fully supervised | Generates multiple plausible narrow-kernel density maps with conditional diffusion. | Strong density estimation but adds sampling cost and does not target label efficiency / DG. | [Project](https://dylran.github.io/crowddiff.github.io/) |
| 2024 | [Single Domain Generalization for Crowd Counting (MPCount)](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.html) | CVPR | Single-domain generalization | Fully labeled source | Uses memory prototypes, a content-error mask, attention consistency, and patch classification for unseen domains. | Closest DG baseline; its central setting assumes labeled source data rather than source label scarcity. | [GitHub](https://github.com/Shimmer93/MPCount) |
| 2024 | [Regressor-Segmenter Mutual Prompt Learning for Crowd Counting (mPrompt)](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_Regressor-Segmenter_Mutual_Prompt_Learning_for_Crowd_Counting_CVPR_2024_paper.html) | CVPR | Density regression / segmentation | Fully supervised | Mutually prompts density regression and segmentation to reduce annotation variance. | Full point supervision; unlabeled-source exploitation and DG are not the objective. | [GitHub](https://github.com/csguomy/mPrompt) |
| 2024 | [Improving Point-Based Crowd Counting and Localization Based on Auxiliary Point Guidance (APGCC)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3537_ECCV_2024_paper.php) | ECCV | Point counting / localization | Fully supervised | Improves proposal-target matching using auxiliary point guidance and feature interpolation. | Strong localization baseline, but dense point labels remain expensive and DG is not central. | [GitHub](https://github.com/AaronCIH/APGCC) |
| 2024 | [Multi-modal Crowd Counting via a Broker Modality](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9445_ECCV_2024_paper.php) | ECCV | Multimodal | Fully supervised paired modalities | Introduces a broker modality to improve cross-modal interaction and reduce alignment/ghosting issues. | Requires paired multimodal data and full labels. | [GitHub](https://github.com/HenryCilence/Broker-Modality-Crowd-Counting) |
| 2024 | [Domain-Agnostic Crowd Counting via Uncertainty-Guided Style Diversity Augmentation (UGSDA)](https://doi.org/10.1145/3664647.3681310) | ACM Multimedia | Single-domain generalization / uncertainty | Fully labeled source | Uses uncertainty-guided style perturbation and density-distribution consistency to diversify one source domain. | Critical novelty boundary: uncertainty + DG already exists; Sparse2Unseen must distinguish itself through sparse-source SSL + DG. | [GitHub](https://github.com/gcding/UGSDA-pytorch) |
| 2024 | [Crowd Counting Using Meta-Test-Time Adaptation (CrowdTTA)](https://pubmed.ncbi.nlm.nih.gov/39252679/) | International Journal of Neural Systems | Test-time adaptation | Supervised source + unlabeled test sample | Meta-learns an initialization and adapts at test time using pseudo supervision / uncertainty. | Target test samples are used during adaptation, unlike Sparse2Unseen's no-target-access training protocol. | -- |
| 2024 | [MRC-Crowd: Semi-Supervised Crowd Counting via Multi-Representation Consistency](https://arxiv.org/abs/2310.10352) | Preprint | Semi-supervised | Labeled + unlabeled source | Uses multiple crowd representations and consistency constraints. | Useful SSL baseline, but primarily in-domain rather than source-only DG. | [GitHub](https://github.com/cha15yq/MRC-Crowd) |
| 2025 | [Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting (TMTB)](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Taste_More_Taste_Better_Diverse_Data_and_Strong_Model_Boost_CVPR_2025_paper.html) | CVPR | Semi-supervised | Point labels + unlabeled source | Combines crowd-aware inpainting augmentation, a visual state-space model, and anti-noise classification. | Strong recent label-efficient baseline, but source-only unseen-domain generalization is not the formal target. | [GitHub](https://github.com/syhien/taste_more_taste_better) |
| 2025 | [Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting (P2R)](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Point-to-Region_Loss_for_Semi-Supervised_Point-Based_Crowd_Counting_CVPR_2025_paper.html) | CVPR | Semi-supervised / point-based / UDA | Partial point labels + unlabeled data | Relaxes point-to-point matching to point-to-region supervision and studies semi-supervised counting and UDA. | Very close methodologically, but its UDA setting can use target-domain unlabeled data; Sparse2Unseen forbids target access. | [GitHub](https://github.com/Elin24/P2RLoss) |
| 2025 | [Free Lunch Enhancements for Multi-modal Crowd Counting](https://openaccess.thecvf.com/content/CVPR2025/html/Meng_Free_Lunch_Enhancements_for_Multi-modal_Crowd_Counting_CVPR_2025_paper.html) | CVPR | Multimodal RGB-T | Fully supervised paired modalities | Adds cross-modal alignment and regional density supervision without extra inference parameters. | Fully supervised paired sensors; source label scarcity is not addressed. | [GitHub](https://github.com/HenryCilence/Free-Lunch-Multimodal-Counting) |
| 2025 | [Perspective-assisted Prototype-based Learning for Semi-supervised Crowd Counting](https://www.sciencedirect.com/science/article/pii/S0031320324008240) | Pattern Recognition | Semi-supervised / perspective | Labeled + unlabeled images | Uses perspective information, prototypes, and consistency learning. | Primarily an in-domain SSL method rather than source-only DG. | -- |
| 2025 | [Video Individual Counting for Moving Drones](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Video_Individual_Counting_for_Moving_Drones_ICCV_2025_paper.html) | ICCV | Video / drone | Supervised video | Introduces MovingDroneCrowd and spatiotemporal cross-frame modeling for moving viewpoints. | Strong video direction, but annotation efficiency and source-to-unseen-scene generalization are separate open axes. | -- |
| 2025 | [CSCC: Cross-Scene Crowd Counting via Learning to Diversify for Domain Generalization](https://doi.org/10.1109/TMM.2025.3535302) | IEEE TMM | Domain generalization | Supervised source | Learns diversified source variations for cross-scene generalization. | Strengthens the DG literature but does not center on a sparsely labeled source plus unlabeled-source training. | -- |
| 2026 | [Semi-supervised Crowd Counting from Unlabeled Data (S4Crowd)](https://www.sciencedirect.com/science/article/pii/S0031320325014505) | Pattern Recognition | Semi-supervised | Labeled + unlabeled source | Dynamically weights pseudo labels and introduces self-supervised crowd regularizers. | A strong newest SSL reference; unseen cross-dataset transfer is not its primary protocol. | -- |
| 2026 | [Adapting Lightweight Image-based Counting Models for Video Crowd Counting](https://openaccess.thecvf.com/content/CVPR2026/html/Shu_Adapting_Lightweight_Image-based_Counting_Models_for_Video_Crowd_Counting_CVPR_2026_paper.html) | CVPR | Video / efficiency | Supervised video training | Injects spatiotemporal information during training while keeping image-model inference lightweight. | Focuses on efficient temporal use, not sparse source labels or source-only DG. | -- |
| 2026 | [Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](https://openaccess.thecvf.com/content/CVPR2026/html/Anisha_Generative_Adversarial_Perturbations_with_Cross-paradigm_Transferability_on_Localized_Crowd_Counting_CVPR_2026_paper.html) | CVPR | Adversarial robustness | Attack evaluation | Studies transferable adversarial perturbations across density-map and point-regression counting paradigms. | Primarily an attack/robustness paper, not annotation efficiency or source-only DG. | -- |
| 2026 | [Fourier Transform-based Single Domain Generalization for Crowd Counting (SinCount)](https://doi.org/10.1038/s41598-026-46286-3) | Scientific Reports | Single-domain generalization | Fully labeled source | Uses frequency-domain transformations to improve single-source generalization. | Further evidence that generic SDG alone is crowded; sparse-source DG needs a distinct contribution. | [GitHub](https://github.com/Twiwq/SinCount) |
| 2026 | [A Benchmark for Semi-supervised Multi-modal Crowd Counting](https://arxiv.org/abs/2606.03646) | Preprint | Semi-supervised multimodal | 5% / 10% / 40% paired labels | Formalizes semi-supervised RGB-T crowd-counting protocols on RGBT-CC and DroneRGBT. | The semi-supervised multimodal task/benchmark is already established; novelty must come from an algorithm. | [GitHub](https://github.com/HenryCilence/Semi-supervised-Multimodal-Crowd-Counting) |

## Closest prior work to Sparse2Unseen

### MPCount -- source-only single-domain generalization

MPCount is the main baseline for the **unseen-domain** half of the problem. It trains from one source domain and evaluates on unseen crowd datasets.

What Sparse2Unseen changes:

- MPCount's core protocol is source-only DG.
- Sparse2Unseen additionally imposes severe source-label scarcity.
- The remaining source images are explicitly exploited as unlabeled data.

This is why the first reproducibility milestone is the MPCount STB -> STA/QNRF setting.

### TMTB / P2R / S4Crowd -- label-efficient source learning

These methods establish that unlabeled crowd images can substantially reduce annotation requirements.

What Sparse2Unseen changes:

- performance is not judged only on the source distribution;
- the target domains are completely unseen;
- target data is not available for training or adaptation.

P2R is particularly important because it also reports UDA. That setting must not be confused with Sparse2Unseen: UDA may use unlabeled target images, while Sparse2Unseen does not.

### Calibrating Uncertainty -- pseudo-label reliability

This work shows that pseudo supervision in crowd counting benefits from uncertainty estimation.

Sparse2Unseen builds on the broader reliability idea, but the research hypothesis is different:

> pseudo labels should be reliable **under plausible domain variation**, not only confident on the current source appearance.

### UGSDA -- uncertainty and source-only DG

UGSDA is the most important novelty warning. It already uses uncertainty-guided style diversity for crowd-counting domain generalization.

Therefore the paper should **not** claim that combining uncertainty and DG is itself new.

The distinction must come from:

- sparse source annotation,
- unlabeled source exploitation,
- the behavior of pseudo-label reliability under source-domain diversification,
- and direct evaluation on unseen targets.

### PET / APGCC -- possible point-based extension

PET and APGCC are strong modern point-localization baselines. If Sparse2Unseen later moves from density regression to point prediction, they provide a route to evaluate localization as well as total counting error.

## Research gaps extracted from the review

### 1. Label-efficient single-domain generalization

**Primary Sparse2Unseen direction.**

Reviewed literature has mature SSL and mature source-only DG, but the central protocol

```text
5-10% labeled source
+ remaining unlabeled source
-> completely unseen targets
with zero target data
```

is not the focus of the closest reviewed methods.

Working method hypothesis:

- EMA teacher / student,
- regional pseudo supervision,
- domain-diversifying source perturbations,
- stability-aware pseudo-label weighting,
- existing source-only DG objective.

### 2. Point-based test-time adaptation

CrowdTTA establishes test-time adaptation for crowd counting, while PET/APGCC provide strong point localization. A point-localization-specific TTA method was not obvious in this review and could be a follow-up direction, but it requires a dedicated novelty search.

### 3. Label-efficient video counting under scene shift

Recent drone/video work advances temporal modeling, but combining sparse frame labels with cross-scene generalization remains attractive. This is scientifically interesting but has higher data and engineering cost.

### 4. Semi-supervised multimodal counting

A benchmark now exists, so the task itself cannot be claimed as new. A potential algorithmic direction is cross-modal disagreement as a pseudo-label confidence signal.

### 5. Robustness / adversarial defense

Bad-weather counting and transferable attacks are established. Robust defense for localized counting could be a follow-up, but it is less aligned with the core Sparse2Unseen question.

## Recommended reading order

1. **P2R, CVPR 2025** -- closest semi-supervised point-based method and key UDA boundary.
2. **TMTB, CVPR 2025** -- strong recent semi-supervised recipe.
3. **MPCount, CVPR 2024** -- primary single-domain generalization baseline.
4. **UGSDA, ACM MM 2024** -- critical novelty check for uncertainty + DG.
5. **Calibrating Uncertainty, ICCV 2023** -- pseudo-label uncertainty foundation.
6. **Domain-General Crowd Counting in Unseen Scenarios, AAAI 2023** -- earlier source-only DG formulation.
7. **APGCC, ECCV 2024** -- strong point-localization architecture.
8. **PET, ICCV 2023** -- point-query counting/localization baseline.
9. **S4Crowd, Pattern Recognition 2026** -- newest semi-supervised reference in this review.
10. **CrowdTTA, 2024** -- relevant if test-time adaptation becomes a branch.
11. **Video Individual Counting, ICCV 2025** -- relevant for drone/video extensions.
12. **Semi-supervised multimodal benchmark, 2026** -- relevant for RGB-T extensions.

## Evaluation implications

The paper should not rely only on one fully supervised MAE comparison.

Recommended reporting:

- 5%, 10%, 40%, and 100% source-label fractions;
- at least three deterministic sparse-label seeds;
- mean +/- standard deviation;
- source in-domain MAE / RMSE;
- unseen-target MAE / RMSE;
- full-label reference result;
- B0/B1/B2/B3 ladder;
- parameters / FLOPs / latency if the new method adds meaningful compute;
- localization metrics if the final model becomes point-based.

A particularly important diagnostic is whether ordinary semi-supervised learning improves the source domain much more than unseen domains.

## Working novelty statement

A safe working statement for internal use is:

> **Sparse2Unseen investigates label-efficient single-domain generalization for crowd counting: learning from a sparsely annotated source domain together with its unlabeled source images, then evaluating directly on previously unseen crowd domains without target-domain access.**

Before paper submission, this claim must be re-checked against literature published after this review date and against broader search terms beyond the papers listed here.
