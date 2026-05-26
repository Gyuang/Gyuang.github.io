---
title: "Impact of Data Quality on Deep Learning Prediction of Spatial Transcriptomics from Histology Images"
excerpt: "Holding architecture fixed at a single ResNet50 + 4-layer MLP, swapping Visium for Xenium training labels on the same FFPE breast cancer block lifts mean per-gene PCC from 0.519 to 0.715 — but the headline is one tissue, one panel, one backbone, and KNN/MAGIC/SCVI imputation that boosts test PCC silently degrades it on the Rep2 serial section."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/data-quality-st/
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
date: 2026-05-26
last_modified_at: 2026-05-26
---

> **Version note.** This post covers the **bioRxiv v1 (Aug 20, 2025; posted Sept 9, 2025)** preprint, which trains a **single** ResNet50 + 4-layer MLP on a **single** FFPE breast cancer block with Visium and Xenium serial sections. A separate post on this site covers a later expanded version that adds UNI features, RedeHist, COAD, VisiumHD, and CosMx — none of which are in v1. The v1 headline numbers should be read with that scope in mind.

## TL;DR

- **Fixed architecture, varied data.** Hallinan et al. fix the model to **one pretrained ResNet50 (frozen-style use) + 4-layer MLP head** and vary only the training-data assay. Visium CytAssist and Xenium In Situ serial sections from the same FFPE breast cancer block (Janesick et al. 2023) are co-registered with **STalign** (affine + LDDMM), rasterized into ~250x250 px patches anchored to Visium spots, and Xenium counts are pseudobulked per patch. 306 genes shared between the Visium full transcriptome and the Xenium panel, 3,958 patches, 5 seeds per configuration.
- **Cross-swap factorial + ablation menu.** With image and molecular inputs disentangled, the paper runs (i) all 2x2 combinations of {Visium, Xenium} image x {Visium, Xenium} counts, (ii) **sparsity** titration on Xenium counts (zero where the paired Visium count <= t for t in {0, 1, 5, 10, 15, 20}), (iii) **additive Poisson noise** on Xenium counts at lambda in {5, 15, 45}, (iv) **Gaussian blur** of Xenium images at kernel 5/25/125, and (v) **imputation rescue** of Visium counts with KNN-smoothing, MAGIC, and SCVI. Every condition is scored on both the held-out 15% test split **and** an independent Xenium serial replicate **Rep2** of the same block.
- **Headline + the negative result that matters most.** Matched-condition mean per-gene **PCC = 0.715 (Xenium) vs. 0.519 (Visium)**, ~38% relative. The cross-swap shows the gap is mostly molecular (Xenium counts + Visium image still gives **0.605**, vs. **0.492** when only the image is swapped). The sharpest finding is the **imputation overfitting**: KNN/MAGIC/SCVI **raise** test PCC and **lower** Rep2 PCC — gains the rest of the field's leaderboards would happily report and that would not transfer.

## Motivation

Spatial transcriptomics costs thousands of dollars per slide; H&E is essentially free and already routine. A growing body of work (ST-Net, HisToGene, BLEEP, EGN, HEST-Bench) predicts spot-level gene expression from H&E, with reported mean PCC ranging from ~0.3 on Visium to substantially higher on Xenium. The community has attributed those gaps to architectural progress. This paper argues the dominant lever is the **molecular and imaging quality of the training data itself**, which differs systematically between **sequencing-based** (Visium: spot-level, transcriptome-wide, sparse, noisy, lateral diffusion, PCR bias) and **imaging-based** (Xenium: subcellular, panel-limited, off-target probe binding, segmentation errors) assays. For clinical pathology pipelines that plan to infer transcriptomes from cheap H&E, the choice of training assay — and the QC applied to it — may matter more than the choice of backbone.

## Core Innovation

- **A controlled cross-swap factorial.** Pair Visium and Xenium serial sections on the same FFPE block, co-register with STalign, rasterize both to a shared 250x250 px patch grid, and train identical ResNet50+MLP models. Then swap image and molecular axes independently to isolate which input carries the signal — a design the architecture-tuning literature has largely skipped.
- **In-silico data-quality ablations.** *Sparsity:* zero Xenium counts where the paired Visium count is below threshold t. *Noise:* add Poisson(lambda) samples to Xenium counts. *Image:* Gaussian blur with kernels 5/25/125. Each ablation is dose-response, with 5 seeds per point, and every point is evaluated on both the held-out test split **and** the independent Xenium serial replicate Rep2.
- **Imputation rescue audit.** Train on KNN-smoothed (k=50, 20 PCs), MAGIC (knn=50, t=auto), and SCVI-imputed (<=300 epochs, anchored on a 5' Chromium scRNA-seq sample from a serial section) Visium counts. Evaluate on test **and** Rep2 — the second axis is what surfaces the overfitting that this paper's most important finding hinges on.
- **Off-target sensitivity check.** Run Off-Target Probe Tracker (OPT, pad-length 10), flag 16 of 306 Xenium genes with protein-coding off-target hits, and retrain without them to confirm the Xenium > Visium gap is not an artifact of probe-design bias.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Training on Xenium gives ~38% higher mean PCC than Visium (0.715 vs. 0.519). | Fig. 2A PCC histogram, Fig. 2B gene-wise scatter (nearly all genes above y=x), 5 seeds. | **One breast cancer block**, 306 shared genes, held-out test patches. | ⭐⭐ — convincing on this slide but n=1 tissue, 1 tumor type, 1 panel, 1 architecture. |
| C2 | The advantage is largely driven by molecular (not image) quality. | Cross-swap: Xenium counts + Visium image = 0.605 (still above 0.519); Visium counts + Xenium image = 0.492 (no gain). | Same. | ⭐⭐ — clean factorial design; still one tissue. |
| C3 | Sparsity in Xenium counts monotonically degrades PCC and at t<=1 collapses to Visium-level performance. | Fig. 3E, 6 sparsity thresholds x {test, Rep2}, 5 seeds, SEM bars. | Test + Rep2 agree. | ⭐⭐⭐ — well-controlled ablation replicated on an out-of-distribution serial section. |
| C4 | Additive Poisson noise (lambda = 5, 15, 45) on Xenium counts monotonically reduces PCC. | Fig. 3F, 3 lambda values, test + Rep2, 5 seeds, SEM bars. | Test + Rep2. | ⭐⭐ — direction is consistent, but only 3 lambda points and additive Poisson is a poor proxy for real PCR + dropout + lateral diffusion. |
| C5 | **Imputation (KNN, MAGIC, SCVI) on Visium boosts test PCC but fails to generalize to Rep2.** | **Fig. 3G — held-out PCC up, Rep2 PCC down across all three methods.** | **Test + Rep2 (same FFPE block).** | **⭐⭐⭐ — the cleanest result; the test/Rep2 divergence would not have surfaced without Rep2.** |
| C6 | Image resolution matters, but less than molecular quality. | Fig. 4E (3 blur levels) + cross-swap delta of ~0.027 for image swap vs. ~0.1-0.2 for molecular swap. | Same. | ⭐⭐ — directional support; only 3 blur levels, isotropic Gaussian only. |
| C7 | Lower image resolution also degrades Grad-CAM interpretability. | Fig. 4F heatmaps for CD4 (PCC 0.79) and PDGFRA (PCC 0.76) under each blur level. | Two genes, qualitative. | ⭐ — visually suggestive on two cherry-picked well-predicted genes; **Supp. Fig. 7 shows saliency does not track gene identity**, which undercuts the interpretability story. |
| C8 | Off-target probe binding does not drive the Xenium advantage. | Supp. Fig. 6 — drop 16/306 OPT-flagged genes, gap persists. | Same. | ⭐⭐ — sensible robustness check, but OPT is a probe-design heuristic from the same lab. |
| C9 | "Improving data quality is an orthogonal strategy to architecture tuning." | Implied from C2-C6. | Same. | ⭐ — **the paper never varies architecture**, so "orthogonal" is asserted, not measured. |

**Honest read.** The strongest results in this paper are **C3, C5, and to a lesser extent C4**, because the authors had the discipline to evaluate every condition on an independent serial replicate (Rep2). The imputation-overfitting finding (C5) is the kind of negative result that the rest of the field's single-split leaderboards would silently miss, and it is the most actionable contribution here.

The weakest claims are the ones that get quoted out of context. Six things readers should hold in mind before generalizing the 0.715 vs. 0.519 headline:

1. **Only one architecture.** ResNet50 + 4-layer MLP, no BLEEP, ST-Net, HisToGene, EGN, ViT, UNI, or Virchow baselines. The C9 framing — "data quality is orthogonal to architecture" — is **asserted, not measured**. A 3-star version of that claim would compare across 2-3 backbones on the same data and show that the data-quality delta is invariant to the choice.
2. **Dataset-specific.** One FFPE block, one tumor type (breast cancer), one panel pre-selected by Xenium-Visium intersection. The 306-gene set is genes Xenium was **designed** to do well on. The 0.715 vs. 0.519 headline must **not** be read as "imaging-based ST > sequencing-based ST" in general. No VisiumHD, CosMx, Slide-seqV2, MERFISH, STARmap, or GeoMx is included in v1.
3. **Pseudobulking discards Xenium's native advantage.** Aggregating Xenium subcellular transcripts up to Visium-spot-sized patches throws away exactly the resolution that motivates buying Xenium. The comparison measures Xenium-when-degraded-to-Visium-grid and therefore **underestimates** Xenium's potential in its native deployment.
4. **Imputation rescue is a clean overfitting result, not a fix.** All three imputers **raise test PCC and lower Rep2 PCC**. Practitioners should treat any paper that reports imputation-boosted PCC without an external serial replicate as not having shown generalization.
5. **Image resolution matters less, but the interpretability story partially self-destructs.** Blurring the image degrades PCC more slowly than degrading counts (good for the headline), and Grad-CAM saliency loses cellular landmarks under blur. But **Supp. Fig. 7 shows that swapping the predicted gene at inference does not move attention to cell-type-specific structures** — the heatmap stays on generic nuclei. The authors flag this honestly, and it weakens C7.
6. **Strongest finding, said plainly:** Molecular quality dominates image quality on this dataset. Swapping counts moves mean PCC by ~0.1-0.2; swapping images moves it by ~0.03. Sparsity and Poisson noise on Xenium counts monotonically degrade PCC; at t<=1 the Xenium model collapses to Visium-level. That is the part of the paper that survives all the scope caveats.

## Method & Architecture

![Data-quality benchmarking pipeline](/assets/images/paper/data-quality-st/fig_p004_01.png)
*Figure 1: Data-quality benchmarking pipeline. Paired serial-section Visium + Xenium breast cancer slides are co-registered with STalign (affine + LDDMM), rasterized to a common 250x250 px patch grid anchored on Visium spots, and used to train an identical ResNet50 + MLP that predicts 306 genes per patch under matched and ablated conditions. Xenium transcripts inside each patch are pseudobulked; Visium counts are used as is. Five seeds per configuration, single RTX 6000 Ada GPU.*

### 1. Inputs and co-registration

- **Source.** Two Xenium In Situ replicates (Rep1, Rep2; WSI 20511x27587 and 18728x27788 px) and a Visium CytAssist section (high-resolution 2000x1809 px), all serial sections from the same FFPE breast cancer block (10x public dataset; Janesick et al. *Nat Commun* 2023).
- **STalign v1.0.1.** Affine Visium -> Xenium with 8 manual landmarks; Xenium transcripts rasterized at 30 microns, then 4-landmark affine, then LDDMM (a = 2500, epV = 1, niter = 2000, sigmaA = 0.11, sigmaB = 0.10, sigmaM = 0.15, sigmaP = 50). After registration, Visium spots and Xenium cells share coordinates on either WSI.

### 2. Common patch grid

- Each Visium spot defines a ~250x250 px patch on the aligned image.
- Xenium per-cell counts inside each patch are **pseudobulked** (SEraster-style) so both technologies emit per-patch count vectors over the same locations.
- Counts log-transformed: $\log(x+1)$. All-zero patches dropped.
- **Final corpus: 3,958 patches, 306 genes** shared between Visium full-transcriptome and the Xenium panel.

### 3. Model and training

- Pretrained **ResNet50** (classification head removed) -> 2048-D feature -> **4-layer MLP** (Linear -> BatchNorm -> ReLU -> Dropout 0.2) -> linear output predicting all 306 genes per patch.
- Patches resized 250 -> 224 px, channel-normalized; augmentations are random H/V flips and 90-degree rotations.
- **MSE loss**, Adam (lr 1e-3, wd 1e-5), batch 64, 150 epochs. Splits 75/10/15 train/val/test.
- **Five independent seeds** per configuration. Single RTX 6000 Ada GPU.
- **Crucially: only one architecture.** No BLEEP/ST-Net/HisToGene/EGN/ViT comparison.

### 4. Ablation menu

- **Sparsity (Fig. 3E).** Zero Xenium counts inside each patch whenever the paired Visium count <= t for t in {0, 1, 5, 10, 15, 20}.
- **Noise (Fig. 3F).** Add Poisson(lambda) to Xenium counts for lambda in {5, 15, 45}.
- **Image (Fig. 4E).** Gaussian blur kernels 5, 25, 125 on Xenium images at train+test time.
- **Imputation (Fig. 3G).** Visium counts smoothed with **KNN** (k=50, 20 PCs, raw UMI), **MAGIC** (knn=50, t=auto, log input), and **SCVI** (300 epochs, trained on a serial-section Chromium 5' scRNA-seq sample + 80% of spatial data; TKT/SLC39A4/GABARAPL2 dropped).
- **Interpretability (Fig. 4F).** Grad-CAM on the last ResNet50 conv layer for CD4 and PDGFRA at each blur level. Supp. Fig. 7 additionally swaps the predicted gene at inference.

### 5. Metrics

- **Per-gene Pearson correlation coefficient (PCC)** between predicted and observed log-counts on test patches and on Rep2.
- **Range-normalized RMSE** as secondary (Supp. Fig. 2-5).

## Experimental Results

![Matched-condition Visium vs. Xenium](/assets/images/paper/data-quality-st/fig_p006_01.png)
*Figure 2: Matched-condition comparison. Mean per-gene PCC is 0.715 for Xenium vs. 0.519 for Visium on the held-out test set (5 seeds, 306 genes); almost all genes lie above the y=x line. Per-gene panels: HDC (Xenium >> Visium), ANKRD30A (both good), AHSP (both poor — coverage floor), GZMK (Visium slightly above Xenium — single counter-example flagged honestly).*

### Main quantitative comparison

Mean PCC across 306 genes, 5 seeds, held-out test split:

| Molecular input | Image input | Mean PCC | Notes |
|---|---|---|---|
| **Xenium counts** | **Xenium WSI (matched)** | **0.715** | **best; ~38% relative gain over Visium-matched** |
| Visium counts | Visium hi-res image (matched) | 0.519 | baseline |
| Xenium counts | Visium image | 0.605 | molecular swap — molecular quality dominates |
| Visium counts | Xenium WSI | 0.492 | image swap — within noise of Visium-matched |

(Supp. Fig. 2-5 report rMSE trends consistent with PCC; no paired statistical tests across genes are reported.)

![Molecular swap and ablations](/assets/images/paper/data-quality-st/fig_p009_01.png)
*Figure 3: Molecular swap and ablations. (A-D) Holding image fixed (Visium) and swapping counts, Xenium molecular data still yields PCC 0.605 vs. 0.519, implicating molecular quality as the dominant factor. (E) Sparsity titration: zeroing Xenium counts wherever the paired Visium count <= t monotonically degrades test + Rep2 PCC; at t<=1 the Xenium model collapses to Visium-level. (F) Additive Poisson noise (lambda = 5, 15, 45) monotonically degrades PCC on both splits. (G) Imputation rescue: KNN/MAGIC/SCVI raise test PCC but lower Rep2 PCC across all three methods — the cleanest negative result in the paper.*

### Ablation summary

| Condition | Mean PCC trend (test) | Mean PCC trend (Rep2) | Read |
|---|---|---|---|
| Xenium + sparsify t<=1 | drops to ~Visium baseline | drops to ~Visium baseline | sparsity is a primary driver |
| Xenium + Poisson lambda increase | monotone down | monotone down | noise is a driver; only 3 lambda points |
| Xenium + Gaussian blur (5/25/125) | slow down | slow down | image resolution matters less than counts |
| **Visium + KNN-smoothing** | **up** | **down** | **imputation overfits the same-block split** |
| **Visium + MAGIC** | **up** | **down** | **imputation overfits** |
| **Visium + SCVI** | **up** | **down** | **imputation overfits** |
| Drop 16 OPT-flagged genes | Xenium gap persists | — | off-target binding does not explain the gap |

![Image ablation and Grad-CAM interpretability](/assets/images/paper/data-quality-st/fig_p011_01.png)
*Figure 4: Image-quality ablation and interpretability. (A-D) Image swap: replacing Xenium WSI with Visium image moves PCC from 0.715 to 0.605 (small) — but swapping the molecular axis moves it from 0.715 to 0.492. (E) Gaussian blur (5/25/125) on Xenium images: PCC drops slowly. (F) Grad-CAM saliency for CD4 (PCC 0.79) and PDGFRA (PCC 0.76) at each blur level — cellular landmarks diffuse with kernel >= 25. Caveat (Supp. Fig. 7): saliency does not move with the predicted gene at inference, so this is a Grad-CAM limitation, not evidence of cell-type-specific localization.*

### Per-gene qualitative panel (Fig. 2C)

- **HDC** — Xenium >> Visium.
- **ANKRD30A** — both good.
- **AHSP** — both poor (likely coverage floor, not technology).
- **GZMK** — Visium slightly > Xenium. A single counter-example the authors flag honestly. The "Xenium wins" headline is therefore a population-mean statement, not gene-wise universal.

### Off-target audit

![Off-target probe sensitivity](/assets/images/paper/data-quality-st/fig_p027_01.png)
*Supp. Fig. 6: Off-target probe sensitivity. Removing 16 OPT-flagged off-target-prone Xenium genes leaves the Xenium > Visium gap intact, ruling out off-target inflation as the sole explanation for the headline.*

### Interpretability caveat

![Grad-CAM does not track gene identity](/assets/images/paper/data-quality-st/fig_p028_01.png)
*Supp. Fig. 7: Saliency does not move with predicted gene identity. Swapping the target gene at inference barely changes Grad-CAM heatmaps — saliency stays on generic nuclei rather than cell-type-specific structures. This undercuts gene-level interpretability claims and is acknowledged by the authors.*

## Limitations

The authors flag most of these directly; the list below is the union of their caveats and additional ones a careful reader should add.

- **One architecture only (v1).** ResNet50 + 4-layer MLP. No BLEEP, ST-Net, HisToGene, EGN, ViT, UNI, or Virchow comparison. The "orthogonal to architecture" framing is **not actually measured** in v1.
- **One tissue block, one tumor type.** A single FFPE breast cancer block; Rep2 is a serial section of the same block, so "generalization" is closer to internal cross-validation than to true OOD. No cross-patient variance, no other cancer types, no normal tissue.
- **One panel, pre-selected by Xenium intersection.** 306 genes are exactly those the Xenium panel was designed to detect well. The comparison is therefore on Xenium's home turf and does not generalize to "imaging > sequencing" in the broader ST landscape (VisiumHD, CosMx, Slide-seqV2, MERFISH, STARmap, GeoMx are all absent in v1).
- **Pseudobulking discards Xenium's resolution.** Aggregating Xenium subcellular transcripts to Visium-sized patches throws away the property that motivates buying Xenium in the first place. v1 measures Xenium-degraded-to-Visium-grid, not Xenium at native scale, and therefore **underestimates** Xenium's true ceiling.
- **Same-block serial sections** mean Visium and Xenium are not measuring identical biology even ignoring technology — slight cell-content differences persist (acknowledged but not bounded).
- **Noise model is simplistic.** Additive Poisson at three lambda is a poor proxy for real Visium dropout + PCR bias + lateral diffusion.
- **Statistical testing is mostly visual.** Mean +/- SEM across 5 seeds and overlapping histograms; no paired tests across the 306 genes.
- **Imputation set is small and dated.** KNN/MAGIC/SCVI are 2017-2018-era; no scGen, SpaGE, or Tangram-style spatial-specific imputers tested.
- **Grad-CAM interpretability story has a self-undermining caveat.** Supp. Fig. 7 shows saliency does not move with predicted gene identity — undercutting cell-type-specific interpretability claims.
- **H&E protocol differences not modeled.** Xenium H&E is post-assay; Visium H&E is pre-assay. Stain and scanner artifacts could contribute to the "image quality" axis independently of the assay.
- **No cost-per-PCC-point trade-off.** The headline recommendation ("use higher-quality data") is technology- and budget-specific but never quantified against throughput or per-slide cost.

## Why It Matters for Medical AI

For any clinical pathology pipeline that plans to deploy an H&E -> expression model — biomarker pre-screening, tumor microenvironment profiling, theranostic stratification — the practically important takeaway is **not** "Xenium beats Visium" (the v1 scope does not justify that as a general claim) but the methodological lessons:

1. **Pick training assays with QC in mind.** Sparsity and per-spot noise of the labeling assay can dominate downstream model accuracy at fixed architecture.
2. **Always test on an independent serial replicate.** Imputation gains on a single test split can be silent overfitting. A second slice from the same block is the cheapest external check possible and would catch a large class of false positives in this literature.
3. **Beware "architecture wins" claims.** If two methods report different PCC on different assays, the assay difference is a confound at least as large as the architecture difference. Reviewers should ask for the cross-swap factorial, not just the bake-off table.
4. **Saliency for "explainability" is risky on this task.** Grad-CAM on H&E -> expression may be visually compelling but Supp. Fig. 7 shows it does not track gene identity in this setting — do not promise pathologists cell-type-specific attention without a sanity check.

## References

- Hallinan C., Lucas C.-H. G., Fan J. **Impact of Data Quality on Deep Learning Prediction of Spatial Transcriptomics from Histology Images.** bioRxiv 2025.09.04.674228 (v1, manuscript dated Aug 20, 2025; posted Sept 9, 2025). DOI: [10.1101/2025.09.04.674228](https://doi.org/10.1101/2025.09.04.674228). CC-BY 4.0. **Not peer-reviewed.**
- Code: [github.com/calebhallinan/dataquality_geneprediction](https://github.com/calebhallinan/dataquality_geneprediction)
- Janesick A. et al. **High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis.** *Nat Commun* 14, 8353 (2023). The paired Visium + Xenium breast cancer dataset.
- STalign — Clifton K. et al. **STalign: alignment of spatial transcriptomics data using diffeomorphic metric mapping.** *Nat Commun* 14, 8123 (2023).
- Off-Target Probe Tracker (OPT) — Janesick A. et al., 10x Genomics technical note.
- Imputation baselines: Wagner F. et al. **KNN-smoothing for single-cell RNA-seq** (bioRxiv 2018); van Dijk D. et al. **MAGIC** (*Cell* 2018); Lopez R. et al. **scVI / SCVI** (*Nat Methods* 2018).
- Related ST-from-H&E work covered on this site: [BLEEP]({{ site.baseurl }}/paper/bleep-bi-modal-contrastive-learning-spatial-transcriptomics/), [HisToGene]({{ site.baseurl }}/paper/histogene-vision-transformer-spatial-transcriptomics/), [EGN]({{ site.baseurl }}/paper/egn-exemplar-guided-network-spatial-transcriptomics/), [HEST-1k]({{ site.baseurl }}/paper/hest-1k-spatial-transcriptomics-histology-benchmark/).
- Related companion review on this site (later expanded version with UNI / RedeHist / COAD): [Impact of Data Quality on Deep Learning Prediction of Spatial Transcriptomics (v2)]({{ site.baseurl }}/paper/data-quality-impact-spatial-transcriptomics-histology/).
