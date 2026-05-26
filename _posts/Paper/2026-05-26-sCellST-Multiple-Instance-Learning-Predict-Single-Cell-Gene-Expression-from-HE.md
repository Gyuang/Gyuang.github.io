---
title: "sCellST: A Multiple Instance Learning Approach to Predict Single-Cell Gene Expression from H&E Images Using Spatial Transcriptomics"
excerpt: "Instance-MIL on HoVer-Net-detected nuclei, supervised by Visium FFPE spot means, beats Istar in 8/12 Wilcoxon cells on 3 PDAC slides — but the single-cell claim is never validated against any real single-cell ground truth."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/scellst/
tags:
  - sCellST
  - Multiple-Instance-Learning
  - Spatial-Transcriptomics
  - Visium
  - HoVer-Net
  - MoCo-v3
  - Self-Supervised-Learning
  - Computational-Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- sCellST reformulates Visium spot-to-expression prediction as **instance-based MIL on segmented nuclei**: HoVer-Net detects every cell, MoCo v3 (ResNet-50) embeds each 48x48 px cell crop, an MLP predicts a per-cell expression vector, and only the **mean over a spot's cells** is supervised against the Visium readout. The trained per-cell head then doubles as a "virtual single-cell" expression estimator at inference time.
- On three PDAC Visium FFPE slides (321 shared HVGs, leave-one-slide-out), sCellST beats **Istar** (deliberately reduced from a 5-model ensemble to 1 model "for fairness") in **8/12** Wilcoxon signed-rank cells, and dominates **HisToGene** (~300 M params, overfit on ~2,000 training spots). sCellST has ~900 K trainable parameters.
- **The headline "single-cell" claim is never validated against a real single-cell ground truth.** No Xenium overlay, no IHC, no multiplex IF. Every per-cell evaluation is either an author-designed simulation or qualitative agreement with HoVer-Net + canonical markers — i.e. circular, since HoVer-Net itself is the cell-typer being "improved upon".

## Motivation

Existing H&E-to-spatial-transcriptomics models (HisToGene, ST-Net, BLEEP, EGN, Hist2ST, Istar) regress expression at the **spot** level — an average over 10-20 cells inside one 55 um Visium spot. Even methods that go finer (Istar, iSTAR) operate on rectangular image patches rather than segmented nuclei, so they cannot answer "what does *this* cell express?" The authors argue that the natural unit of histology is the cell, and since a Visium spot is literally a bag of cells with one expression label, **MIL with cell instances is the right supervision pattern**. The medical-AI payoff would be substantial: retrospective H&E archives (TCGA, hospital cohorts) could be re-mined for cell-type-specific spatial biology without paying for ST acquisition.

## Core Innovation

- **Bag = Visium spot, instance = segmented nucleus.** Each spot's bag is constructed from HoVer-Net detections that fall inside the spot radius. No neighbour expansion across adjacent spots — pure instance-MIL, not attention-MIL with context.
- **MoCo v3 cell embeddings tuned for histology.** ResNet-50 backbone, with non-default augmentations: tweaked colour jitter, random erasing, random rotation. SSL is trained **only on cells from the training slide** (<=300 K cells) to avoid cross-fold leakage, accepting lower SSL quality as the price.
- **Softplus per-cell head + mean aggregation.** Per-cell prediction f_theta(h_i) >= 0; spot prediction is the simple mean over the spot's cells. MSE loss on library-normalised, log1p, per-gene min-max-scaled expression. NB-NLL was tried in simulation and offered no benefit, so it was dropped from main results.
- **~900 K trainable parameters vs HisToGene's ~300 M**, the same order as Istar — the authors argue this parameter-to-data ratio is the reason HisToGene collapses on the ~2,000-spot regime.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | sCellST predicts **single-cell** gene expression | Simulation cell-scenario (Fig. 2d): MIL reaches Pearson r = 0.54 on marker genes vs fully-supervised oracle 0.68 | 1 ovarian H&E + matched scRNA, **simulated only** | ⭐⭐ — proven *in simulation*; **no real single-cell ground truth** (no Xenium co-registration, no IHC, no multiplex IF) anywhere in the paper |
| C2 | Outperforms state-of-the-art spot-level GE predictors | Fig. 3c: sCellST > Istar in 8/12 Wilcoxon cells (3 train slides x 2 test slides x 2 metrics); dominates HisToGene | 3 PDAC Visium FFPE slides, 321 shared HVGs | ⭐⭐ — only 2 baselines, and Istar is **deliberately weakened from its 5-model ensemble to a single model "for fairness"**; no BLEEP, EGN, Hist2ST, iSTAR (full), TRIPLEX, GHIST, Henium |
| C3 | Versatile across tissues and cancer types | Qualitative DE-gene agreement on breast + ovarian + PDAC | 3 cancers but only 1 slide each outside PDAC; **no PDAC->breast transfer** | ⭐ — no cross-cancer quantitative generalisation; "versatility" = "applied to three slides" |
| C4 | Identifies finer cell types than HoVer-Net (fibroblast vs endothelial; lymphocyte vs plasma cell) | Fig. 6 score distributions + top-100 image galleries; SupFig. 5 shows plasma-cell-scoring cells HoVer-Net mislabels | 1 ovarian Visium slide | ⭐⭐ — visually compelling but no quantitative gold standard (no IHC, no multiplex IF, no Xenium); "discovery" is judged by the same pretrained pathology gaze |
| C5 | SSL embedding is necessary / superior to ImageNet transfer | "On par with" SSL variant (SupFig. 3) on spot-level; more homogeneous galleries (SupFig. 4) for plasma + endothelial cells | PDAC (quant) + ovarian (qual) | ⭐ — by the authors' own numbers SSL ~= ImageNet on the quantitative task; the win is qualitative, on one slide |
| C6 | Mean aggregation models the measurement process correctly | Loss derivation; no ablation against gated-attention MIL, MaxMIL, or learned-weight aggregation | — | ⭐ — design choice not benchmarked against any other MIL aggregator |
| C7 | NB loss offers no benefit, MSE is sufficient | SupFig. 2 (simulation) | Simulation only | ⭐⭐ — fair given simulation scope, but real ST count distributions are zero-inflated; NB ablation on real PDAC is missing from the main text |
| C8 | Generalises across slides | Leave-one-slide-out among 3 PDAC slides (Sec. 2.3) | 3 PDAC slides from one cohort (GSE211895) | ⭐ — within-cohort, within-organ, within-protocol; **no real cross-cancer or cross-cohort generalisation** |

**Honest read.** The *formulation* is the strongest part — instance-based MIL on cells is the right abstraction, and the simulation suite (Fig. 2) bounds what is and is not recoverable in a principled way. The benchmark, however, is thin: 2 baselines on 3 slides of one cancer type, with the stronger baseline (Istar) explicitly hobbled "for fair comparison" by cutting its 5-model ensemble to 1. An 8/12 Wilcoxon margin against a weakened opponent is not a Nature-Communications-grade headline on its own. The "single-cell" framing — which is the entire selling point — is never validated against an actual single-cell measurement; every cell-level evaluation is either (a) inside an author-designed simulation, or (b) qualitative agreement with HoVer-Net + marker plausibility, which is **circular** because HoVer-Net is the very tool sCellST claims to surpass in C4. No variance / CI reporting, no per-gene win/loss table, no multiple-testing correction. The medical-AI promise (re-mining TCGA at cell-level resolution) is real, but the evidence does not yet support it.

## Method & Architecture

![sCellST pipeline: HoVer-Net cell detection, MoCo v3 SSL embedding, and instance-MIL regression](/assets/images/paper/scellst/page_005.png)
*Figure 1: The three stages of sCellST — nucleus detection with HoVer-Net (PanNuke weights, 3 retained classes), MoCo v3 cell embeddings on 48x48 px crops, and instance-MIL regression of spot expression as the mean over per-cell softplus predictions.*

### 1. Cell detection and tissue masking

Pretrained **HoVer-Net** (PanNuke weights, 6 classes) is run on the whole Visium WSI. Only the three dominant PanNuke classes are retained — **neoplastic epithelial, connective/soft-tissue, inflammatory**. The PanNuke tissue mask is replaced with an Otsu + morphological-opening pipeline from the `Democratizing_WSI` repo because it produced cleaner masks on these slides.

### 2. Cell crop extraction

For each detected nucleus, a **12 um x 12 um** box centered on the segmentation centroid (typical cell diameter) is extracted from the WSI (native resolution 0.2-0.5 um/px, FFPE) and resized to **48 x 48 px** as the input to the embedding network.

### 3. Cell embedding via MoCo v3

Backbone is **ResNet-50**. Augmentations diverge from natural-image defaults: colour-jitter parameters retuned, **random erasing and random rotation** added (rotation requires extracting a larger crop first to avoid corner artifacts). SSL training uses **all cells from the training slide only** (<=300 K cells) — the authors explicitly accept lower SSL quality to avoid data leakage across cross-validation folds. ImageNet-transfer and one-hot cell-type vectors are both kept as embedding ablations.

### 4. MIL bag construction

For spot s with detected cells {x_1, ..., x_{k_s}}, the bag is exactly the cells whose centroids fall inside the Visium spot radius. **There is no neighbour expansion** — a cell does not participate in adjacent spots' bags. This is pure instance-MIL, not attention-MIL with neighbour context.

### 5. Per-cell predictor and aggregation

An MLP maps each cell embedding $h_i \in \mathbb{R}^d$ to a per-cell expression vector $\hat{y}_i \in \mathbb{R}_+^G$:

$$
\hat{y}_i = \text{softplus}(f_\theta(h_i))
$$

The softplus head guarantees non-negative per-cell scores. Spot prediction is the **mean** of per-cell predictions:

$$
\hat{y}_s = \frac{1}{k_s} \sum_{i=1}^{k_s} \hat{y}_i
$$

(The PDF prints "1/S" in the equation but the surrounding text makes clear the divisor is the cell count $k_s$ — a typesetting bug.) Total trainable parameters are ~900 K, vs ~300 M for HisToGene — the cited reason HisToGene fails on ~2,000-spot regimes.

### 6. Loss

Two losses were tried:

1. **MSE** on library-normalised, log1p, per-gene min-max-scaled spot expression.
2. **NB negative log-likelihood** with a learned per-gene dispersion $\alpha_j$ and observed library size $l_i$ absorbing the per-cell scale.

Simulation (SupFig. 2) showed no significant NB advantage, so **all main results use MSE**.

### 7. Downstream single-cell scoring

After training, per-cell predictions feed Scanpy's `score_genes` to build cell-type signature scores (mean over marker list $G_m$ minus mean over control list $G_c$), used in the ovarian fine-cell-type experiment.

### Simulation framework

![Simulation framework bounding what MIL can recover](/assets/images/paper/scellst/page_006.png)
*Figure 2: Simulation framework. Three scenarios — random (no morphology-expression link), centroid (perfect 1:1 morphology-mean expression), and cell (morphology cluster -> scRNA cluster, but per-cell expression drawn at random within cluster). On the cell scenario, MIL recovers median Pearson r = 0.54 on marker genes vs r = 0.68 for a fully-supervised cell-level oracle — i.e. ~55-80% of the achievable signal with only spot-level supervision.*

## Experimental Results

### Spot-level benchmark on PDAC (Fig. 3)

Distribution of per-gene Pearson / Spearman correlations on 321 shared HVGs, leave-one-slide-out across 3 PDAC slides -> **3 train x 2 test x 2 metrics = 12 cells** in the Wilcoxon signed-rank table.

| Method | Params | Result vs sCellST | Notes |
|---|---|---|---|
| HisToGene (ViT) | ~300 M | sCellST wins decisively in all settings (boxplots clearly higher) | Authors attribute failure to parameter-to-data ratio |
| Istar (weakly-supervised patch ensemble, **reduced from 5 -> 1 model "for fairness"**) | ~900 K | sCellST significantly better in **8 / 12** experiments | Remaining 4 cells: direction not stated |
| **sCellST (SSL / MoCo v3)** | **~900 K** | **Best overall** | Headline configuration |
| sCellST (ImageNet-transfer) | ~900 K | "On par" with SSL variant (SupFig. 3) | SSL win is not quantitative on spot-level |
| sCellST (one-hot cell-type) | tiny | Outperformed by SSL/ImageNet variants in most cases | Embedding does matter, but morphology > class label |

The paper does **not** publish a numeric mean-correlation table in the main text — only boxplots and a Wilcoxon count. No confidence intervals, no per-gene breakdown beyond visual.

![Leave-one-slide-out benchmark on three PDAC Visium FFPE slides](/assets/images/paper/scellst/page_008.png)
*Figure 3: Leave-one-slide-out benchmark across three PDAC Visium FFPE slides (GSE211895). sCellST beats Istar in 8/12 Wilcoxon signed-rank comparisons on per-gene Pearson and Spearman correlations across 321 shared HVGs, and dominates HisToGene throughout.*

### Simulation (Fig. 2)

- **Random scenario** (no morphology-expression link): per-gene Pearson ~ 0, max ~ 0.1 — sanity check passes.
- **Centroid scenario** (perfect 1:1 morphology-mean expression): median r > 0.8 for most genes — MIL solves it when signal exists.
- **Cell scenario** (cluster-level link, intra-cluster expression random): median Pearson **r = 0.10 on HVGs**, **r = 0.54 on marker genes** at spot level. Fully-supervised cell-level oracle reaches **0.18 / 0.68** — MIL recovers ~55-80% of the achievable signal without per-cell labels.
- SSL embedding vs one-hot ideal cluster vector: only "mild" improvement for one-hot, arguing intra-cluster noise is not the bottleneck.

### Qualitative cell-level results (Figs. 4-5, breast + ovarian)

![HoVer-Net per-label differential expression from sCellST predictions on breast and ovarian](/assets/images/paper/scellst/page_009.png)
*Figure 4: Per-HoVer-Net-label differential expression from sCellST predictions recovers canonical markers — COL1A2/3A1/10A1/11A1, MMP2, MYLK for connective; PTPRC, IGHM, LCP1, LSP1, SRGN, POU2AF1 for inflammatory; patient-specific markers (PTPN3 in breast; WNT6, FGF19 in ovarian) for neoplastic.*

![Single-cell virtual expression maps — the immune-excluded ovarian pattern](/assets/images/paper/scellst/page_010.png)
*Figure 5: Single-cell virtual expression. sCellST resolves CD68+ macrophages to interstitial stromal locations in an immune-excluded ovarian tumour — a spatial pattern Visium spot-level resolution cannot recover. This is the paper's strongest qualitative argument, but it remains qualitative.*

### Fine cell-type discovery (Fig. 6, ovarian)

Trained on 93 ovarian cell-type marker genes (from a CellxGene scRNA reference); Scanpy signature scores then separate fibroblasts vs endothelial cells, and lymphocytes vs plasma cells — distinctions HoVer-Net cannot make.

![Marker-gene training schema for fine cell-type discovery](/assets/images/paper/scellst/fig_p012_03.png)
*Figure 6a: Marker-gene training pipeline. Predict 93 ovarian cell-type marker genes from H&E cell crops, then compute Scanpy signature scores per cell.*

Top-100 image galleries per signature show coherent morphology — spindle-shaped fibroblasts vs vessel-lining endothelium, small dark lymphocyte nuclei vs larger ovoid plasma cells:

![Top-scoring cell galleries: fibroblasts](/assets/images/paper/scellst/fig_p012_04.png)
*Figure 6c (fibroblasts): Top-scoring cells form a coherent spindle-cell morphology.*

![Top-scoring cell galleries: endothelial](/assets/images/paper/scellst/fig_p012_05.png)
*Figure 6c (endothelial): Vessel-lining morphology emerges among top-scoring cells.*

![Top-scoring cell galleries: lymphocytes](/assets/images/paper/scellst/fig_p012_06.png)
*Figure 6c (lymphocytes): Small dark round nuclei dominate the top-scoring cohort.*

![Top-scoring cell galleries: plasma cells](/assets/images/paper/scellst/fig_p012_07.png)
*Figure 6c (plasma cells): Larger, ovoid nuclei with eccentric morphology — SupFig. 5 shows these are systematically mislabeled by HoVer-Net.*

SupFig. 4 reports that ImageNet embeddings give *less homogeneous* galleries for plasma + endothelial cells specifically — the one place SSL clearly beats ImageNet beyond ties.

## Limitations

**Authors acknowledge (p.13-14):**

- Hard dependency on HoVer-Net — segmentation errors propagate uncharacterised.
- Domain shift in cell embeddings across staining / scanning protocols; no out-of-the-box generalisation without retraining.
- Small ST training datasets; FFPE Visium with paired high-resolution H&E is still scarce.
- Visium HD (2 um bins) mentioned as future direction but needs custom training because 8 um super-bins still don't align with cells.

**Not addressed (this review's read):**

- **No real single-cell ground truth anywhere.** Xenium / CosMx / MERFISH co-registration would have settled C1 and C4 — and would have been the natural follow-up given the "single-cell" framing in the title. The omission is conspicuous.
- **Baselines limited to HisToGene + Istar.** No comparison to BLEEP, EGN, Hist2ST, iSTAR (full 5-model ensemble), TRIPLEX, ST-Net, or any 2024-era foundation-model baseline (Henium, GHIST, HEST-1k baselines, scMMGPT). The Istar comparison itself is asymmetric — its 5-model ensemble was cut to 1 "for fairness".
- **Gene set is 321-2000 HVGs**, not a Xenium-style targeted panel. The "single-cell expression" framing is misleading on this axis too — Visium HVGs are very different from imaging-ST panels.
- **No cross-cancer generalisation.** The leave-one-slide-out is *within* 3 PDAC slides from one GEO cohort. No PDAC -> breast or PDAC -> ovarian transfer test, despite the paper showing qualitative breast and ovarian results.
- **HoVer-Net cell-detection errors propagate uncharacterised.** No reporting of how many cells per spot are typical post-detection, how spots with k_s = 1-2 cells behave, or what happens in dense lymphocyte clusters where HoVer-Net is known to undercount.
- **No ablation on aggregation function** (mean vs gated-attention MIL vs max), which is the standard MIL design choice and where most of the recent MIL literature lives.
- **No multiple-testing correction** on the gene-wise correlations.
- **Code "will be" available** — not verifiable from the preprint PDF.

## Why It Matters for Medical AI

If a virtual single-cell expression head trained from cheap Visium spots actually worked, the path to scaled spatial biology runs through retrospective H&E cohorts (TCGA, hospital archives) — no new wet-lab acquisition needed. That is the right ambition, and instance-MIL on segmented nuclei is the right abstraction. The gap between this paper's framing and its evidence is a roadmap for the next study: register cell-level predictions to Xenium or multiplex IF on the same tissue, benchmark against the current generation of H&E -> ST methods (not just HisToGene + a 1-model Istar), and report per-cell accuracy on real single-cell labels rather than only on simulation. Until then, sCellST is best understood as a **well-formulated MIL framework with promising qualitative results**, not as a validated H&E-to-single-cell predictor.

## References

- Paper (bioRxiv): [Chadoutaud et al., "sCellST: A Multiple Instance Learning Approach to Predict Single-Cell Gene Expression from H&E Images Using Spatial Transcriptomics", bioRxiv 2024.11.07.622225 (8 Nov 2024)](https://www.biorxiv.org/content/10.1101/2024.11.07.622225v1) — Nature Communications version 2026.
- Code: [github.com/loicchadoutaud/sCellST](https://github.com/loicchadoutaud/sCellST) (announced as forthcoming in the preprint).
- PDAC Visium dataset: [GEO GSE211895](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211895) (Wu et al.).
- Related work — H&E -> ST regression baselines: ST-Net (Nat. Biomed. Eng. 2020), HisToGene (Briefings in Bioinformatics 2022), BLEEP (NeurIPS 2023), EGN, Hist2ST, Istar / iSTAR.
- Component models: HoVer-Net (Med. Image Anal. 2019), MoCo v3 (ICCV 2021), Scanpy `score_genes` (Wolf et al., Genome Biol. 2018).
