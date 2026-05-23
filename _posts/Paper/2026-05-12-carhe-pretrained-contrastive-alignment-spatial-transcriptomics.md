---
title: "CarHE: Predicting Spatial Transcriptomics from H&E Image by Pretrained Contrastive Alignment Learning"
excerpt: "Contrastive alignment of HIPT image features to scGPT cell-cluster embeddings plus Grad-CAM refinement, yielding lung-cancer DFS AUC = 0.734."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/carhe-pretrained-contrastive-alignment-spatial-transcriptomics/
tags:
  - CarHE
  - Spatial-Transcriptomics
  - Contrastive-Learning
  - HIPT
  - scGPT
  - Grad-CAM
  - Computational-Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- CarHE predicts spot- and cell-level spatial transcriptomics (ST) from H&E alone by contrastively aligning HIPT image patches to 100 scGPT-derived "cell cluster" embeddings, then iteratively refining the patch-to-cluster assignment with a Grad-CAM masking loop.
- Pretraining spans ~2M cells/spots across Visium, Visium HD, ST (HER2ST), Xenium, and MERFISH; the design swaps noisy per-spot count regression for a low-dimensional, cell-type-anchored target that is supposed to transfer across tissues and species.
- Headline metrics: **top-500 variable-gene PCC = 0.70** on BRCA HER2ST slide H1, melanoma Xenium cell-type **ARI = 0.77**, mouse intestine Visium HD **AUROC > 0.9**, and — the strongest clinical evidence — in-house lung-cancer **DFS AUC = 0.734** on an 880-patient / 1,600+ slide cohort. DLPFC drops to **PCC = 0.32**, which the authors attribute to HIPT's TCGA pretraining bias.

## Motivation
Spatial transcriptomics is biologically rich but cost-prohibitive and rarely scaled to large patient cohorts. Prior image-to-ST models (HisToGene, Hist2ST, BLEEP, iSTAR, STEM, ENLIGHT-DeepPT) share three structural weaknesses: they typically regress only a few hundred to <10,000 genes, per-spot regression of high-dimensional sparse counts is noisy and overfits, and a model trained on one tissue rarely generalizes to another.

CarHE's pitch is that *cell types* — what pathologists already read from H&E — are the natural cross-tissue invariant. So rather than aligning patches to raw count vectors, the model aligns them to scGPT-clustered cell-type embeddings, and converts the resulting cluster weights into spot-resolution expression with a Grad-CAM-based refinement loop. The medical-AI angle is concrete: DFS prediction across 1,600+ lung-cancer slides from four Chinese centers, using TLS regions as the interpretable intermediate biomarker.

## Core Innovation
1. **Cell-cluster targets, not per-spot counts.** scGPT embeddings of >2M pretraining cells/spots are KNN-clustered into m = 100 cell clusters, each represented by its mean embedding. This turns sparse, noisy per-cell expression into a stable, low-dimensional contrastive target.
2. **Softened CLIP target matrix.** The contrastive loss uses a target T with diagonal = 1 and off-diagonal = a constant α < 1 instead of 0 — recognizing that two patches may genuinely belong to similar clusters. This is what prevents the cell-type-clustered objective from collapsing under hard negatives.
3. **Grad-CAM refinement loop.** Per query patch, the model computes Grad-CAM activations w.r.t. each candidate cluster, masks the patch with cluster-specific visual prompts, re-runs CarHE, and drops candidates with probability < 0.4 until both the weight vector and the candidate set converge. This is closer to weakly-supervised segmentation than to standard CLIP inference, and the paper claims it is where spatial accuracy beyond cluster-centroid averaging comes from — though no ablation isolates this step.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|-------|----------|------------|----------|
| C1 | Predicts ST data with >10,000 genes from H&E alone | scGPT input is up to 10k HVGs; ccRCC supp reports PCC=0.65 on >17,000 genes | ccRCC (1 cohort) | ⭐⭐ |
| C2 | Accuracy exceeds 0.7 across diverse tissue types and species | Top-500 PCC=0.70 BRCA H1; TLS PCC=0.75/0.82 BRCA F/H; ARI=0.77 melanoma — but DLPFC PCC=0.32, mouse intestine reported only as AUROC | Selectively true | ⭐ |
| C3 | Up to 1.7× higher PCC than second-best method | Fig 2A/B; holds on cherry-picked genes; baselines limited to HisToGene/Hist2ST/THItoGene/STEM (pre-2024); no HEST-Bench / OmiCLIP / STPath / BLEEP / iSTAR | **BRCA H1 only** | ⭐ |
| C4 | Identifies subtle pathological features (TLS) across cancers | TLS PCC=0.75 & 0.82 on BRCA F/H; qualitative agreement on 62 annotated lung slides; one putative TLS originally labeled cancer | Multi-cancer mixed qual/quant | ⭐⭐ |
| C5 | Reconstructs 3D ST from images alone (DLPFC) | k=7 k-means on predicted expression recovers L1–L6+WM laminae; PAGA L1→WM trajectory matches biology | DLPFC 4 slides; expression PCC only 0.32 | ⭐ |
| C6 | Cross-species: mouse intestine cell-type AUROC > 0.9 | Fig 5C ROC for 21 clusters; Lyz1 gland localization qualitative | 1 mouse slide; train/test = right/left halves of same slide | ⭐ |
| C7 | **DFS AUC = 0.734 vs unnamed SOTA at 0.586 / 0.643** | Fig 6E ROC; MLP on CarHE features; baselines unnamed; single validation cohort | 880-patient lung cohort | ⭐⭐ |
| C8 | Cell-type concordance >93% / 95% / 98% for cancer / stromal / TLS | Fig 6D heatmap on 62 annotated slides | In-house only; "correct" undefined | ⭐⭐ |
| C9 | Cross-tissue generalizability | Held-out splits are *slide-level inside the same studies used for pretraining*, not external cohorts | All evaluations | ⭐ |
| C10 | Gene coverage >17× prior SOTA | Architectural argument: competitors are sub-1,000-gene models | Methods | ⭐⭐ |

The strongest claims are C7, C8, and C4 — downstream-clinically-relevant metrics with at least multi-slide evidence. The weakest are C2, C3, and C9: the "across diverse tissues" and "1.7× over second-best" framings rely almost entirely on BRCA H1 against a baseline set that excludes the current strongest published models. DLPFC PCC = 0.32 is itself evidence that "PCC > 0.7" is a *cancer-specific* statement rather than a universal one.

## Method & Architecture

![CarHE overview: contrastive alignment of HIPT patches to scGPT cell-cluster embeddings, followed by Grad-CAM iterative refinement](/assets/images/paper/carhe/page_004.png)
*Figure 1: CarHE overview — scGPT cell-cluster embeddings are contrastively aligned to HIPT image features, then iteratively refined by Grad-CAM masking.*

**Inputs.** Paired H&E WSI (stain-normalized via `staintools.LuminosityStandardizer`) and ST counts. Each spot/cell sits at the center of a 256×256 px patch at ~0.5 µm/px.

**Pretrained encoders.**
- *Image encoder = HIPT* (TCGA-pretrained). A small ViT (n=8, h=6, d=384) extracts 16-px features inside each 256-px patch; a second ViT (n=4, h=6, d=192) aggregates them. Only the last two transformer blocks are fine-tuned; the rest is frozen. Patch feature `Q_k` is 384-d.
- *Gene-expression encoder = scGPT* pretrained on 33M human cells, ingesting up to 10,000 highly variable genes per cell/spot. Output `sC_k` is 512-d.
- *Projection heads.* Two small MLPs (linear → GELU → linear → dropout → residual → LayerNorm) project image features (384 → 256) and gene features (512 → 256) into a shared 256-d space.

**Cell-cluster construction.** scGPT embeddings of all pretraining cells/spots are KNN-clustered into m = 100 cell clusters. Each cluster `S_m` is represented by the *mean* embedding `C_m` of its members.

**Step 1 — contrastive alignment.** For each batch of paired (patch, cell) examples, train a modified CLIP loss with a softened target T (diagonal = 1, off-diagonal = α < 1):

$$
\mathcal{L}_{CLIP} = \frac{1}{2}\big(\text{CE}(\text{softmax}(P C^\top/\tau), T) + \text{CE}(\text{softmax}(C P^\top/\tau), T)\big)
$$

Output is a weight matrix `W_0` mapping each patch to a probability distribution over the 100 clusters.

**Step 2 — Grad-CAM spatial refinement.** Per query patch q:
1. Initialize `W^0_q` from the Step-1 model.
2. For every candidate cluster m, compute Grad-CAM w.r.t. that "class" on the patch and mask the patch with cluster-specific visual prompts.
3. Re-run CarHE on the masked image to obtain refined `W^{t+1}_q`.
4. Drop any cluster whose probability falls below τ_drop = 0.4.
5. Iterate 2–4 until both `W` and the candidate set converge.

Predicted gene expression is then `g_{i,q} = Σ_m W_{q,m} · ḡ_{i,m}` — a weighted average of cluster-mean gene expressions.

The softened off-diagonal α is what prevents the contrastive objective from collapsing under genuinely similar clusters; the Grad-CAM loop is what the paper credits with sub-patch spatial accuracy. Neither contribution is isolated by ablation.

## Experimental Results

### Main quantitative comparison

| Task | Dataset | Metric | **CarHE** | HisToGene | Hist2ST | THItoGene | STEM | Notes |
|------|---------|--------|-----------|-----------|---------|-----------|------|-------|
| CD3D | BRCA H1 | PCC | **0.64** | 0.13 | 0.17 | low | low | Fig 2A |
| CD74 | BRCA H1 | PCC | **0.68** | 0.45 | 0.35 | low | low | Fig 2A |
| COL3A1 | BRCA H1 | PCC | **0.57** | 0.13 | 0.06 | 0.49 | low | Fig 2A |
| Top-300 var. genes | BRCA H1 | PCC | **0.68** | lower | lower | lower | lower | Fig 2B, p<0.0001 |
| Top-500 var. genes | BRCA H1 | PCC | **0.70** | lower | lower | lower | lower | Fig 2B |
| Top-1000 var. genes | BRCA H1 | PCC | **0.68** | lower | lower | lower | lower | Fig 2B |
| Top-2000 var. genes | BRCA H1 | PCC | **0.60** | lower | lower | lower | lower | Fig 2B |
| Cell-type pred. | Melanoma Xenium | ARI | **0.77** | — | — | — | — | Fig 2D, no baselines |
| >17,000 genes | ccRCC c2–c4 | PCC | **0.65** | — | — | — | — | Supp Fig 1 |
| TLS score | BRCA patient 1 | PCC | **0.75** | — | — | — | — | Fig 3B vs 3C |
| TLS score | BRCA patient 2 | PCC | **0.82** | — | — | — | — | Fig 3F vs 3G |
| Gene exp. | DLPFC 151673–6 | PCC | **0.32** | — | — | — | — | Supp Fig 2 |
| Cluster ID | Mouse intestine HD | AUROC | **>0.9** (all 21) | — | — | — | — | Fig 5C/D |
| **DFS prediction** | **In-house lung 1,600+ slides** | **AUC** | **0.734** | — | — | — | — | vs. unnamed SOTA 0.586 / 0.643 |

![BRCA H1 marker genes and melanoma Xenium cell-type recovery](/assets/images/paper/carhe/page_006.png)
*Figure 2: BRCA H1 and melanoma Xenium benchmarks — CarHE recovers marker-gene patterns (CD3D / CD74 / COL3A1) and cell-type composition where HisToGene / Hist2ST / THItoGene / STEM flatline.*

### Reading between the rows

- **The "1.7× over second-best" claim is BRCA HER2ST slide H1 only, against pre-2024 baselines.** The 1.7× factor is computed against HisToGene/Hist2ST/THItoGene/STEM on top-variable genes from BRCA H1 (and per-gene ratios on three cherry-picked markers reach 1.5×–9.5×). On the cross-tissue/cross-species evaluations (melanoma Xenium, ccRCC, DLPFC, mouse intestine), **no baseline numbers are reported at all** — so the "1.7× across diverse tissues" framing is unsupported by the data presented.
- **Significance.** Fig 2B reports p < 0.0001, but the boxplot variance reflects variance across genes on a single test slide. No cross-slide CV, no bootstrap CI, no per-cohort variance.
- **DLPFC PCC = 0.32 is the anomaly.** It is also the only completely non-cancer test set. The k-means cortical-layer recovery (Fig 4) is consistent with "the encoder learned cortical-layer-relevant features" rather than "the encoder predicts gene expression accurately."
- **Mouse intestine.** AUROC > 0.9 is for *cluster identity* — a classification problem. No scalar PCC is given for individual genes; Lyz1 villi-vs-gland is qualitative only.
- **DFS AUC = 0.734 is the strongest clinical evidence.** But the comparison baselines (AUC 0.586, 0.643) are unnamed in the main text — presumably image-only deep models — so we cannot tell whether they are a fairly trained Prov-GigaPath or an off-the-shelf weak baseline. No K-M survival curves or hazard ratios are shown, only an ROC, and there is a single validation cohort.
- **Effectively no ablations.** Step-1-only vs full pipeline, raw-counts vs scGPT-cluster targets, k ≠ 100 — none reported.

![TLS detection in HER2ST BRCA patients 1 and 2](/assets/images/paper/carhe/page_023.png)
*Figure 3: TLS recovery vs. pathologist annotation in BRCA patients 1 and 2 — CarHE flags a putative TLS initially annotated as cancer that shows immune infiltration on zoom.*

![DLPFC 3D reconstruction across 4 consecutive sections](/assets/images/paper/carhe/page_024.png)
*Figure 4: DLPFC 3D reconstruction — even with raw-expression PCC = 0.32, CarHE recovers the L1–WM laminar layout and an L1→WM PAGA trajectory.*

![Mouse intestine Visium HD cross-species transfer](/assets/images/paper/carhe/page_025.png)
*Figure 5: Cross-species transfer — on mouse intestine Visium HD, CarHE achieves per-cluster AUROC > 0.9 and correctly localizes Lyz1 to gland regions.*

![In-house lung cohort: TLS detection and DFS prediction](/assets/images/paper/carhe/page_026.png)
*Figure 6: In-house 880-patient lung cohort — CarHE-identified TLS regions match pathologist annotation, and a downstream MLP on CarHE features reaches DFS AUC = 0.734 (vs. unnamed baselines at 0.586 and 0.643).*

## Limitations

**Authors admit.** Performance depends on input H&E and ST quality; performance on rare or underrepresented tissues "requires further validation"; DLPFC accuracy is lower than cancer datasets, attributed to HIPT's TCGA pretraining bias.

**Not addressed.**
- No ablation isolating the Grad-CAM refinement (Step 2) or the softened-α CLIP target — the two design contributions the paper most leans on.
- No sensitivity analysis on k = 100 cluster count. The 100-cluster quantization physically bounds the granularity of downstream cell typing.
- No external benchmark against HEST-Bench, BLEEP, iSTAR, OmiCLIP, STPath, or ENLIGHT-DeepPT — only pre-2024 ST-prediction models.
- The DFS baseline models are unnamed; the "15% improvement" claim is not independently reproducible from the text.
- Held-out splits are *within-cohort* (slide held out from a dataset whose other slides were in pretraining), not leave-one-cohort-out.
- No statistical correction across the many reported PCC values; p < 0.0001 on a single slide is not the right framing.
- The iterative Grad-CAM loop requires repeated forward passes per cluster per patch — WSI-scale compute is not characterized.
- DFS validation is single-cohort, all-Chinese, all-lung-cancer.

## Why It Matters for Medical AI

The clinically actionable claim here is C7: predicting disease-free survival in lung cancer from H&E alone, via an interpretable intermediate (TLS regions inferred from ST) on an 880-patient cohort, reaching AUC = 0.734. If that number holds under external validation — non-Chinese cohorts, named baselines including modern pathology foundation models like Prov-GigaPath or UNI, and reported confidence intervals — then "ST-as-an-intermediate-feature" becomes a credible route for routine prognostic biomarkers from a stain that every pathology lab already produces.

That is a large "if." The strongest framing in the paper today is: CarHE shows that contrastive alignment to cell-type prototypes (rather than to raw counts) is a viable design for image-to-ST models, and that the resulting features carry enough signal for a downstream prognostic head to beat the unnamed image-only baselines used here. The weakest framings — cross-tissue generalization and "1.7× over second-best" — should not be quoted without qualification: that lead is BRCA HER2ST slide H1 against pre-2024 baselines, while DLPFC collapses to PCC = 0.32 on the same model. The DFS AUC = 0.734 result, not the PCC headline, is what would justify clinical follow-up.

## References

- Paper (bioRxiv): [10.1101/2025.06.15.659438](https://www.biorxiv.org/content/10.1101/2025.06.15.659438)
- Code: [github.com/Jwzouchenlab/CarHE](https://github.com/Jwzouchenlab/CarHE)
- HIPT — Chen et al., *Hierarchical Image Pyramid Transformer*, CVPR 2022
- scGPT — Cui et al., *scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics*, Nature Methods 2024
- HEST-Bench — Jaume et al., *HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis*, NeurIPS 2024
- BLEEP — Xie et al., *Spatially Resolved Gene Expression Prediction from Histology via Bi-modal Contrastive Learning*, NeurIPS 2023
- HER2ST — Andersson et al., *Spatial deconvolution of HER2-positive breast cancer*, Nat. Commun. 2021
