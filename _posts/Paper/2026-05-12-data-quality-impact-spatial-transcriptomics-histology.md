---
title: "Impact of Data Quality on Deep Learning Prediction of Spatial Transcriptomics from Histology Images"
excerpt: "A controlled cross-swap study shows training-label quality, not architecture, drives ST-from-H&E performance: Xenium-trained PCC 0.715 vs. Visium-trained 0.519 (~38% relative), and KNN/MAGIC/SCVI imputation boosts held-out PCC but degrades generalization on a serial replicate."
categories:
  - Paper
tags:
  - Spatial-Transcriptomics
  - Computational-Pathology
  - Data-Centric-AI
  - Xenium
  - Visium
  - Imputation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- **Holds architecture fixed, varies data.** A pretrained ResNet50 + 4-layer MLP head is trained on serial-section breast cancer tissue assayed by both Visium and Xenium, with images and molecular labels rasterized to a shared 250x250 px patch grid. The only thing that changes between runs is which technology supplied the training data.
- **Cross-swap factorial isolates the source of the gap.** Pairing Xenium molecular labels with the *Visium* image still gives **PCC 0.605**, well above the Visium-mol + Visium-image baseline (0.519). A sparsity titration (zero Xenium counts wherever paired Visium count <= t for t = 0,1,5,10,15,20) and a Poisson noise titration (lambda = 5, 15, 45) reproduce Visium-like performance on Xenium monotonically — and the dose-response replicates on an independent Xenium serial section (Rep2).
- **Headline result + a negative result.** Xenium training yields mean per-gene **PCC 0.715 vs. 0.519 for Visium (~38% relative)** on breast cancer; the ordering persists with UNI features (**0.734 vs. 0.529**) and with the RedeHist single-cell model (**0.633 vs. 0.385**), and on COAD (Xenium 5K 0.568, CosMx 6K 0.532, VisiumHD 0.434). KNN-smoothing, MAGIC, and SCVI imputation of Visium counts **increase** held-out PCC but **decrease** PCC on Rep2 — i.e., imputation gains do not generalize.

## Motivation

ST-from-H&E methods (ST-Net, HisToGene, BLEEP, EGN, RedeHist) routinely report ~0.3 mean PCC on Visium and substantially higher numbers on Xenium, and the community has attributed those gaps to architecture. This conflates model design with the sparsity, noise, and resolution of the underlying training labels and images. For medical AI the question is load-bearing: if clinical pipelines plan to infer transcriptomes from cheap H&E slides, we need to know whether reported gains are real architectural progress or simply a property of the assay used to label the training data.

## Core Innovation

- **A controlled cross-swap design.** Pair Visium and Xenium serial sections on the same FFPE block, co-register with STalign, rasterize both to a common 250x250 px patch grid, and train identical ResNet50+MLP models. Then swap (image fixed / molecular swapped, and vice versa) to factor out architecture and isolate which input axis carries the signal.
- **In-silico data-quality ablations.** *Sparsity:* zero Xenium counts where the paired Visium count is below thresholds 0/1/5/10/15/20. *Noise:* add Poisson(lambda) samples to Xenium counts for lambda in {5, 15, 45}. *Image:* Gaussian blur with kernels 5/25/125, plus a Visium re-scan WSI (20,511x27,587). Each ablation is evaluated both on a 15% held-out test split and on an independent serial replicate (Rep2) of the same block.
- **Imputation rescue audit.** Train on KNN-smoothed (k=50, 20 PCs), MAGIC (knn=50, t=auto), and SCVI-imputed (<=300 epochs, anchored on 5' Chromium scRNA-seq from a serial section) Visium counts. Evaluate on test *and* on Rep2 — the failure mode that the rest of the field's leaderboards do not surface.
- **Cross-technology + cross-tissue generalization.** Repeat the same protocol on COAD serial sections assayed by Xenium 5K, VisiumHD, and CosMx 6K.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Training on Xenium molecular data outperforms Visium by ~38% mean PCC under matched architecture. | Fig. 2A: 0.715 vs. 0.519; Fig. 2B: almost all genes above the diagonal; Supp. Fig. 2 rMSE. | Breast, 5 seeds, test split. | ⭐⭐⭐ |
| C2 | The Xenium advantage is **not** an artifact of architecture choice. | UNI: 0.734 vs. 0.529 (Supp. Fig. 3A-B); RedeHist: 0.633 vs. 0.385 (Supp. Fig. 3C-D). | Breast. | ⭐⭐⭐ |
| C3 | Molecular data quality alone (sparsity + noise) drives most of the gap. | Cross-swap (Fig. 3A): Xenium mol. + Visium image 0.605 > Visium mol. + Visium image 0.519; sparsity titration (Fig. 3E) recapitulates Visium-like PCC at t<=1; Poisson noise titration (Fig. 3F). Trends replicate on Rep2. | Breast, test + Rep2. | ⭐⭐⭐ |
| C4 | Image resolution alone has modest impact when molecular labels are poor, large impact when they are clean. | Fig. 4A vs. 4C: Visium mol. swap 0.519 vs. 0.492 (within noise); Xenium mol. swap 0.605 vs. 0.715. | Breast. | ⭐⭐ |
| C5 | **Imputation (KNN / MAGIC / SCVI) on Visium yields apparent gains that do not generalize.** | **Fig. 3G: held-out PCC up, Rep2 PCC down across all three methods.** | **Breast, test + Rep2 (same FFPE block).** | **⭐⭐ — convincing on Rep2, but Rep2 is still a serial section of one block.** |
| C6 | Findings reproduce across technologies (Xenium 5K, VisiumHD, CosMx 6K) and a second tissue (COAD). | Supp. Fig. 8: Xenium 5K 0.568 > CosMx 6K 0.532 > VisiumHD 0.434. CosMx 6K rMSE 0.200 (highest). | COAD, test only. | ⭐⭐ |
| C7 | Reduced image resolution degrades Grad-CAM interpretability, not just accuracy. | Fig. 4F: heatmaps diffuse with kernel >= 25; cellular landmarks lost. | Breast, two genes. | ⭐⭐ |
| C8 | Off-target probe binding does not explain the Xenium advantage. | OPT pad-10 flags 14/306 genes; removing them preserves the trend (Supp. Fig. 10). | Breast. | ⭐⭐ |
| C9 | Higher-quality training data improves cell-type marker prediction specifically. | Supp. Fig. 11: nearly all marker-gene PCCs improve under Xenium training. | Breast. | ⭐⭐ |
| C10 | Data-quality improvement is an *orthogonal* strategy to architectural tuning. | Synthesis of C2 + C3 + C5. | All. | ⭐⭐ — well-argued but framed as a recommendation; no direct "better data vs. bigger model" trade-off run. |

**Honest read.** The cross-swap design is exactly the right factorial: changing one axis at a time, with image and molecular signal cleanly separated by rasterization. C1-C3 are unusually well-supported for an ST benchmarking paper — the sparsity titration produces a monotonic dose-response that *bridges* the technologies on the same data, and the Rep2 evaluation correctly catches the imputation overfitting mode that the rest of the field's leaderboards miss. C5 is the paper's most actionable contribution and directly challenges papers that report imputation-boosted PCC on Visium without an external validation. The weak spots are mostly about scope:

1. **One FFPE block per tissue.** Rep2 is a serial section of the same patient block, so "generalization" here is closer to internal cross-validation than to out-of-distribution. Error bars are over 5 seed reruns, capturing training stochasticity rather than biological variability. The 38% headline does not include any patient-level variance.
2. **C4 is asymmetric.** "Visium mol. is insensitive to image resolution" could equally mean "Visium labels are too noisy for any visual signal to help" — a floor effect rather than evidence that image quality only matters with clean labels. The paper does not disentangle these.
3. **Grad-CAM interpretability story has a self-undermining caveat.** Supp. Fig. 12 shows attention is *not* cell-type-specific: switching the target gene barely changes the heatmap. The authors flag this honestly, but it weakens C7.
4. **Missing concurrent comparisons.** HEST-Bench (Jaume et al., 1,229 paired samples), OmiCLIP, and STPath all argue for curated high-quality training pools — consistent with this paper's thesis — but are neither cited as concurrent evidence nor reused as splits. The "data quality matters" framing lands as more novel than the literature warrants.
5. **No statistical significance testing.** Differences such as 0.519 vs. 0.526 (re-scanned Visium WSI) are described as "slight increases" without t-tests, paired permutation tests, or bootstrap confidence intervals.

Net: C1-C3 and C5 deserve their ratings; C4, C6-C9 are good single-cohort evidence; C10 is a framing recommendation the experiments are consistent with but cannot prove against architectural innovation directly.

## Method & Architecture

![Data-quality benchmarking pipeline: paired Visium/Xenium serial sections, STalign co-registration, common 250x250 px patch grid, ResNet50+MLP head, and the ablation menu of sparsity / Poisson noise / blur / imputation](/assets/images/paper/data-quality-impact-st/fig_p004_01.png)
*Figure 1: Data-quality benchmarking pipeline. Paired Visium and Xenium serial sections are co-registered with STalign, rasterized to a common 250x250 px patch grid, and used to train identical ResNet50+MLP models under controlled molecular (sparsity, Poisson noise, imputation) and image (Gaussian blur, resolution swap) ablations.*

The base model is a pretrained ResNet50 with the final FC removed, feeding a 4-layer MLP (each layer = Linear + BN + ReLU + 20% Dropout) and a linear head that predicts log1p expression for all 306 shared breast cancer genes per patch. Patches are resized 250x250 -> 224x224, channel-normalized with ResNet50 stats, augmented with random H/V flips and 90-degree rotations. Loss = MSE; Adam lr=1e-3, weight_decay=1e-5; batch 64; 150 epochs; 75/10/15 split; 5 independent seeds on an RTX 6000 Ada. Robustness variants swap ResNet50 for UNI embeddings (same head) or use RedeHist with cell-level predictions summed back into 250x250 patches. STalign performs affine alignment with 8 landmarks followed by diffeomorphic metric mapping (a=2500, epV=1, niter=2000, sigmaA=0.11, sigmaB=0.10, sigmaM=0.15, sigmaP=50). Xenium single-cell coordinates are mapped to the WSI via 30-um rasterization plus a 4-landmark affine + DMM.

![Molecular cross-swap on the Visium image: Xenium-trained PCC 0.605 vs. Visium-trained 0.519](/assets/images/paper/data-quality-impact-st/fig_p009_01.png)
*Figure 2 (Fig. 3A in paper): Even when the image is held fixed as the Visium WSI, Xenium molecular labels yield PCC 0.605 vs. 0.519 for Visium labels — isolating molecular quality from imaging.*

![Image-only sweep and Grad-CAM across blur levels: Xenium-mol benefits from higher-resolution images while Visium-mol is insensitive; attention diffuses as blur kernel grows](/assets/images/paper/data-quality-impact-st/fig_p011_01.png)
*Figure 3 (Fig. 4 in paper): Image resolution matters only when molecular labels are clean. Xenium-mol benefits from the higher-resolution Xenium WSI (0.605 -> 0.715), while Visium-mol is insensitive (0.519 vs. 0.492). Grad-CAM heatmaps for CD4 and PDGFRA lose nuclear/cellular localization as the Gaussian blur kernel grows; note however that Supp. Fig. 12 shows the heatmaps are not cell-type-specific.*

## Experimental Results

Mean per-gene PCC across 5 independent seeds. Bold rows are the paper's own headline configurations.

| Configuration | Image | Molecular | Mean PCC | Dataset / split |
|---|---|---|---|---|
| **Baseline Xenium** | Xenium WSI | Xenium | **0.715** | Breast, test |
| **Baseline Visium** | Visium hi-res | Visium | **0.519** | Breast, test |
| Cross: Visium image + Xenium mol. | Visium hi-res | Xenium | 0.605 | Breast, test |
| Cross: Xenium image + Visium mol. | Xenium WSI | Visium | 0.492 | Breast, test |
| UNI features | Xenium WSI | Xenium | **0.734** | Breast, test |
| UNI features | Visium hi-res | Visium | 0.529 | Breast, test |
| RedeHist | Xenium WSI | Xenium | 0.633 | Breast, test |
| RedeHist | Visium hi-res | Visium | 0.385 | Breast, test |
| Visium re-scanned WSI | Visium WSI | Xenium | 0.678 | Breast, test |
| Visium re-scanned WSI | Visium WSI | Visium | 0.526 | Breast, test (vs. 0.519 hi-res) |
| **Xenium 5K baseline** | Xenium 5K WSI | Xenium 5K | **0.568** (rMSE 0.148) | COAD, test |
| VisiumHD baseline | VisiumHD WSI | VisiumHD | 0.434 (rMSE 0.162) | COAD, test |
| CosMx 6K baseline | CosMx 6K WSI | CosMx 6K | 0.532 (rMSE 0.200) | COAD, test |

Ablation highlights:

- **Sparsity titration (Fig. 3E).** Imposing Visium-style zeros on Xenium at threshold <=1 drops Xenium PCC to roughly Visium's level; degradation is monotonic with threshold and reproduces on Rep2. Strong evidence that molecular sparsity, not architecture, accounts for most of the Visium-Xenium gap.
- **Poisson noise (Fig. 3F).** lambda = 5 -> 15 -> 45 monotonically lowers PCC on both test and Rep2; curves stay parallel — noise affects in-distribution and serial-replicate evaluation equally.
- **Imputation (Fig. 3G).** KNN, MAGIC, and SCVI all increase PCC on held-out test but decrease PCC on Rep2 — a clean demonstration of test-set overfitting via imputation and a direct challenge to a common ST community practice.
- **Image blur.** Slow PCC decline through kernel 25, sharp drop at 125.
- **Off-target check.** Removing 14 OPT-flagged genes does not change the trend (Supp. Fig. 10).
- **Marker genes (Supp. Fig. 11).** Nearly all cell-type marker genes improve under Xenium training, strengthening biological relevance.

![COAD cross-technology: Xenium 5K 0.568 > CosMx 6K 0.532 > VisiumHD 0.434; CosMx exhibits a unimodal distribution and the highest rMSE](/assets/images/paper/data-quality-impact-st/fig_p030_01.png)
*Figure 4 (Supp. Fig. 8): COAD generalization. Across three technologies on serial sections of one COAD block, Xenium 5K beats CosMx 6K and VisiumHD; CosMx 6K shows the highest rMSE (0.200).*

![CosMx 6K stain quality and per-gene spatial maps for PDYN and MAP4 illustrate technology-specific trade-offs](/assets/images/paper/data-quality-impact-st/fig_p032_01.png)
*Figure 5 (Supp. Fig. 9): CosMx 6K H&E shows pale eosin and diffuse hematoxylin in epithelial regions. PDYN appears better predicted in CosMx while MAP4 is sparser — technology-specific gene-level trade-offs that probe sequences alone (unpublished for CosMx) cannot adjudicate.*

## Limitations

**Authors admit.** Off-target probe binding in imaging-based assays may inflate Xenium PCC; CosMx probe sequences are unavailable so an OPT-style audit is impossible; H&E protocol timing differs between Visium (pre-assay) and Xenium (post-assay); slide artifacts (folds, bubbles, scanner variability) are uncontrolled; throughput-quality trade-off means Visium's untargeted full-transcriptome coverage may still be preferable for some use cases; Grad-CAM attention is not cell-type-specific (Supp. Fig. 12).

**Not adequately addressed.**

- **No external patient cohort.** Rep2 is a serial section of the same FFPE block; this is internal validation, not HEST-Bench-style generalization. Only one tissue block per tissue type.
- **No quantification of training-data quantity vs. quality trade-off.** Could a larger Visium training pool compensate for lower per-sample quality? The paper flags this as future work but does not run it.
- **No comparison to curated multi-cohort efforts** (HEST-1k / HEST-Bench, STimage-1K4M, OmiCLIP, STPath) as either competitors or concurrent evidence.
- **Patch-level pseudobulking of Xenium** discards single-cell resolution before training; Visium spots are already pseudobulked by the assay. A fair single-cell-vs-spot comparison is not run.
- **No stain normalization, color augmentation, or domain-adaptive foundation models** are tried — image-side rescue is left open in contrast to the negative result for molecular imputation.
- **No statistical significance tests** on PCC differences and no per-patient variance estimates.
- **Conservative imputation menu.** ST-tailored methods such as gimVI, Tangram, SpatialScope, and Sprout are not evaluated; they may rescue more cleanly.
- **Headline 38% is on a single test split.** Whether the gain shrinks under multi-cohort training where the model sees more variability is an open question.

## Why It Matters for Medical AI

If you are building an H&E-to-transcriptome pipeline for clinical deployment, the practical message is: stop tuning your encoder and start auditing your labels. Two operational consequences.

1. **Treat training-label quality as the primary lever.** The cross-swap result says that under matched architecture, swapping in higher-quality molecular labels buys you ~0.20 PCC — more than any architecture swap the field has reported on the same baseline. Curated, low-sparsity assays (Xenium-class panels) are worth the cost even if you intend to ship a cheap H&E inference model at test time.
2. **Do not train on imputed Visium counts unless you validate on an unseen replicate.** KNN, MAGIC, and SCVI all increase held-out PCC and *decrease* PCC on Rep2 — the held-out gain is overfitting to imputed structure rather than true biology. This invalidates a non-trivial fraction of the Visium-imputation literature and is the most clinically consequential takeaway: a model that looks better on your local test set may regress on the next patient's serial section.

The scope caveats matter for medical adoption. One FFPE block per tissue is not a generalization story; no patient-level variance is reported; no external cohort is used; and Grad-CAM attention is not cell-type-specific so don't sell the heatmaps as interpretability. Treat this as a methodologically rigorous proof-of-principle for a data-centric mindset, not as a deployment-ready benchmark.

## References

- Hallinan, C., Lucas, C.-H. G., & Fan, J. *Impact of Data Quality on Deep Learning Prediction of Spatial Transcriptomics from Histology Images.* bioRxiv preprint, Feb 19, 2026. DOI: [10.1101/2025.09.04.674228](https://doi.org/10.1101/2025.09.04.674228).
- Code: [https://github.com/calebhallinan/dataquality_geneprediction](https://github.com/calebhallinan/dataquality_geneprediction)
- Co-registration: STalign (https://github.com/JEFworks-Lab/STalign).
- Imputation baselines: KNN-smoothing (Wagner et al.), MAGIC (van Dijk et al.), SCVI (Lopez et al.).
- Related architectures: ST-Net, HisToGene, BLEEP, EGN, RedeHist.
- Concurrent data-curation efforts (not directly compared by the paper): HEST-1k / HEST-Bench, STimage-1K4M, OmiCLIP, STPath.
- Foundation model: UNI (Mahmood Lab).
- Off-target audit: Off-Target Probe Tracker (OPT, pad-length 10).
