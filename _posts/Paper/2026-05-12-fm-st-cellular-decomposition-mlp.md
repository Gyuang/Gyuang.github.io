---
title: "Integrating Pathology Foundation Models and Spatial Transcriptomics for Cellular Decomposition from Histology Images"
excerpt: "A 7-layer MLP on CONCH+UNI embeddings matches Hist2Cell on per-spot cellular decomposition while collapsing GPU memory from 44,951 MB to 570 MB and STNet leave-one-out training from 383m 24s to 6m 5s."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/fm-st-cellular-decomposition-mlp/
tags:
  - Pathology-Foundation-Models
  - CONCH
  - UNI
  - Spatial-Transcriptomics
  - Cellular-Decomposition
  - cell2location
  - Hist2Cell
  - MLP
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- The paper plugs frozen pathology foundation model (FM) embeddings — **CONCH (512-D) concatenated with UNI (1024-D)** — into a small 7-layer MLP and regresses per-spot cell-type abundances directly from H&E patches, replacing Hist2Cell's heavy Vision Graph-Transformer.
- Supervision comes from **cell2location** abundances derived with a Human Breast Cell Atlas (HBCA) scRNA-seq reference; the loss is a composite of **MSE + MAE + Pearson** that forces both magnitude and rank alignment.
- Headline result on STNet leave-one-out: **CC and L1 are on par with or better than Hist2Cell**, while GPU memory drops from **44,951 MB to 570 MB** and per-run training time from **383m 24s to 6m 5s** — a roughly 79x memory reduction and a 63x speedup with no accuracy loss.

## Motivation

10x Visium gives per-spot gene expression on the same H&E slide pathologists already read, and cell2location can deconvolve those spots into cell-type abundances given a paired scRNA-seq reference. The bottleneck is acquisition: ST is expensive, slow, and tied to a matched single-cell experiment. Image-only predictors like Hist2Cell remove the ST requirement at inference time, but their Vision Graph-Transformer backbone needs roughly 45 GB of GPU memory and several hours per leave-one-out fold — which puts the method out of reach for most labs.

The authors observe that two pathology FMs trained on hundreds of millions of H&E tiles — **CONCH** (vision-language, MGH/MIT) and **UNI** (vision-only, Mahmood Lab) — already encode the morphological priors a cell-type regressor would otherwise have to learn from scratch. If the FM embeddings are rich enough, the downstream head should be a small MLP, not a graph transformer. The medical-AI angle is practical accessibility: bring cellular-level readouts to histology slides without ST data and without a multi-GPU training budget.

## Core Innovation

The paper's contribution is architectural minimalism, not a new representation. Three choices do the work:

1. **Concatenate two complementary FMs.** CONCH is vision-language, trained with pathology image-text pairs; UNI is vision-only, trained purely on histopathology. Stacking their embeddings (1536-D) consistently beats either alone on cross-dataset transfer (Table 2: 0.57 L1 vs. 0.60 / 0.59 on her2st→STNet), and the ablation (Figures 4 / A2) shows that *which* FMs are combined matters more than total embedding dimensionality — a 1536-D UNI2 alone does not beat the 1536-D CONCH+UNI concatenation.
2. **A 7-layer MLP with SiLU activations** as the regression head. No graph structure, no attention over neighboring spots — each spot is decoded independently from its FM embedding.
3. **Composite loss with a Pearson term.** $L_{total} = \text{MSE}(\hat{y}, y) + \lambda_1 \cdot \text{MAE}(\hat{y}, y) + \lambda_2 \cdot L_{Pearson}(\hat{y}, y)$. MSE penalizes magnitude error, MAE adds robustness to abundance outliers, and the Pearson term forces relative-abundance rank to align with the cell2location target — which is what colocalization metrics ultimately reward.

Crucially, the supervision target is *not* an orthogonal cellular readout — it is **cell2location output**. The MLP is being trained to mimic a probabilistic deconvolution model, so the entire benchmark measures fidelity to that deconvolution rather than to ground-truth cellular composition. The paper does not flag this as a structural limitation; we do below.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | FM-embedding + MLP matches or surpasses Hist2Cell on per-spot CC and L1. | Figures 2 (her2st) and 3 (STNet) show CC/L1 per held-out patient; CONCH+UNI wins in most folds with 2-3 exceptions (her2st D; STNet 24223, 23508). | her2st, STNet | ⭐⭐ — multi-fold but single-run, no variance reported, exceptions exist. |
| C2 | Concatenating CONCH+UNI outperforms either FM alone. | Figures 2, 3, A2; cross-dataset Table 2 (CONCH+UNI strictly best on her2st→STNet, tied with CONCH on STNet→her2st). | her2st, STNet | ⭐⭐ — consistent in cross-dataset; LOO has counterexamples on STNet 23288, 23377. |
| C3 | Generalization to external datasets is improved over Hist2Cell. | Table 2: lower L1 in both directions (0.57 vs 0.77; 0.55 vs 0.61); CC tied (0.40) on STNet→her2st, better (0.32 vs 0.27) on her2st→STNet. | her2st↔STNet | ⭐⭐ — clear L1 wins; CC wins modest and within breast tissue only. |
| C4 | Colocalization patterns are recovered comparably to Hist2Cell. | Table 1: cos sim 0.82 vs 0.81 (her2st) and 0.78 vs 0.77 (STNet); Figure A1(b). | her2st, STNet | ⭐⭐ — differences within 0.01; "matches" is fair, "surpasses" would not be. |
| C5 | The method drastically reduces compute (~43 GB GPU, >4.5 h training saved). | Table 3: 570 MB vs 44,951 MB; 6m 5s vs 383m 24s on STNet LOO; similar on her2st LOO. | her2st, STNet | ⭐⭐⭐ — large, reproducible-looking numbers across four experimental settings. |
| C6 | The approach "generalizes across tissue types by leveraging foundation model embeddings". | Only breast cancer datasets are evaluated. | her2st, STNet | ⭐ — not evidenced; overreach. |
| C7 | Selection of embedding sources matters more than embedding dimensionality. | Figures 4 and A2 — UNI2 (1536-D) alone often beats CONCH+UNI (1536-D); adding UNI2 to CONCH+UNI does not always help. | her2st, STNet | ⭐⭐ — qualitatively supported but not statistically tested. |

**Honest read.** The strongest, well-supported claim is **C5** — the compute reduction is an order-of-magnitude effect reproduced across four experimental settings. C1-C4 (accuracy parity or modest gains) are credible but reported as point estimates without error bars or significance tests, and the gains over Hist2Cell are within 0.01 on colocalization and mixed on per-fold CC. **C6 is the load-bearing overreach**: the abstract frames the method as tissue-agnostic, but both datasets are breast cancer and share the same HBCA scRNA-seq reference, so true cross-tissue generalization remains untested. Two methodological gaps the paper does not address: (a) **the ground truth is itself a model output** (cell2location), so what is being measured is fidelity to a deconvolution model, not to orthogonal cellular composition; (b) **no shallow baseline** — linear regression or k-NN on the same FM embeddings — is reported, so the "7-layer MLP with SiLU" choice is unjustified.

## Method & Architecture

![FM + MLP pipeline overview: H&E patches at Visium spots are encoded by frozen CONCH and UNI, concatenated, and regressed by a 7-layer MLP to cell-type abundances; cell2location with an HBCA scRNA-seq reference provides the supervision target.](/assets/images/paper/fm-st-cellular-decomp/page_003.png)
*Figure 1: Overview — H&E patches at ST spots are encoded by pre-trained pathology FMs (CONCH, UNI), concatenated, and regressed to cell-type abundances; cell2location provides the supervision target.*

The pipeline is intentionally short:

1. **Patch extraction.** From each ST slide, crop 224×224 H&E patches centered on Visium spots (Hist2Cell preprocessing). RGB-thresholded background-dominated patches are dropped.
2. **FM embedding.** Pass each patch through frozen CONCH (512-D) and UNI (1024-D); concatenate to a 1536-D vector. The paper also evaluates CONCH-only, UNI-only, UNI2-only (ViT-h/14, 1536-D), and CONCH+UNI+UNI2 ablations.
3. **Standardization.** Zero-mean, unit-variance normalization across the training set.
4. **Ground-truth labels.** Run **cell2location** on the paired spatial gene-count matrix with the Human Breast Cell Atlas (HBCA) scRNA-seq reference to produce per-spot cell-type abundances $y$. The same HBCA reference is used for both her2st and STNet, which is what makes cross-dataset evaluation feasible — they share a label space.
5. **MLP regressor.** Seven fully-connected layers with SiLU activations; a final linear layer outputs the cell-type abundance vector $\hat{y}$.
6. **Composite loss.** $L_{total} = \text{MSE}(\hat{y}, y) + \lambda_1 \cdot \text{MAE}(\hat{y}, y) + \lambda_2 \cdot L_{Pearson}(\hat{y}, y)$, with $L_{Pearson} = -\frac{\sum (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum(\hat{y}_i - \bar{\hat{y}})^2} \cdot \sqrt{\sum(y_i - \bar{y})^2} + \epsilon}$. Specific $\lambda_1, \lambda_2$ values are not reported.
7. **Evaluation.** (i) Leave-one-patient-out CV within each dataset; (ii) cross-dataset (her2st↔STNet) out-of-distribution test. Metrics: Pearson correlation coefficient (CC), L1, and bivariate Moran's R colocalization (row-wise cosine similarity and correlation between predicted and GT colocalization clustermaps).

## Experimental Results

### Main quantitative comparison

| Setting | Method | Metric | Value |
|---|---|---|---|
| Cross-dataset her2st→STNet | **CONCH+UNI MLP** | **L1 ↓ / CC ↑** | **0.57 / 0.32** |
| Cross-dataset her2st→STNet | CONCH MLP | L1 / CC | 0.60 / 0.23 |
| Cross-dataset her2st→STNet | UNI MLP | L1 / CC | 0.59 / 0.28 |
| Cross-dataset her2st→STNet | Hist2Cell | L1 / CC | 0.77 / 0.27 |
| Cross-dataset STNet→her2st | **CONCH+UNI MLP** | **L1 / CC** | **0.55 / 0.40** |
| Cross-dataset STNet→her2st | CONCH MLP | L1 / CC | 0.55 / 0.40 |
| Cross-dataset STNet→her2st | UNI MLP | L1 / CC | 0.56 / 0.36 |
| Cross-dataset STNet→her2st | Hist2Cell | L1 / CC | 0.61 / 0.40 |
| Colocalization (her2st) | **CONCH+UNI MLP** | **cos sim / corr** | **0.82 / 0.87** |
| Colocalization (her2st) | Hist2Cell | cos sim / corr | 0.81 / 0.87 |
| Colocalization (STNet) | **CONCH+UNI MLP** | **cos sim / corr** | **0.78 / 0.78** |
| Colocalization (STNet) | Hist2Cell | cos sim / corr | 0.77 / 0.77 |
| Compute (STNet LOO) | **CONCH+UNI MLP** | **GPU / time** | **570 MB / 6m 5s** |
| Compute (STNet LOO) | Hist2Cell | GPU / time | 44,951 MB / 383m 24s |
| Compute (her2st LOO) | **CONCH+UNI MLP** | **GPU / time** | **570 MB / 9m 13s** |
| Compute (her2st LOO) | Hist2Cell | GPU / time | 44,951 MB / 171m 39s |

![Per-patient CC and L1 across 23 STNet patients (leave-one-out); CONCH+UNI MLP consistently leads except on samples 23288 and 23377.](/assets/images/paper/fm-st-cellular-decomp/page_004.png)
*Figure 3: STNet leave-one-out — CC (top) and L1 (bottom) across 23 patients; CONCH+UNI MLP leads with two exceptions (23288, 23377).*

### Ablation: which FMs to combine

![Ablation on her2st combining CONCH, UNI, and UNI2 in single, two-, and three-FM configurations; UNI2 alone or CONCH+UNI is consistently best.](/assets/images/paper/fm-st-cellular-decomp/page_005.png)
*Figure 4: her2st ablation — adding UNI2 to CONCH+UNI does not consistently help; FM selection matters more than embedding dimensionality.*

The ablation evaluates three single-FM configurations (CONCH, UNI, UNI2), three two-FM concatenations, and the three-FM stack. The reported pattern is that **UNI2 alone or CONCH+UNI** wins on most folds, while CONCH+UNI+UNI2 does *not* consistently improve. The takeaway is that **complementarity of FMs** (vision-language CONCH plus vision-only UNI) carries more signal than raw dimensionality — UNI2 (1536-D, ViT-h/14 on 200M H&E tiles) alone does not strictly dominate CONCH+UNI (also 1536-D). UNI alone beats CONCH alone, plausibly because UNI is pathology-specific while CONCH is trained more broadly via vision-language pairs.

### Qualitative colocalization

![Predicted vs. ground-truth cell-type spatial distributions on her2st slide B1 (a) and bivariate Moran's R clustermaps (b); CONCH+UNI MLP reduces false-positive spots vs. Hist2Cell.](/assets/images/paper/fm-st-cellular-decomp/page_007.png)
*Figure A1: (a) cell-type spatial maps for her2st slide B1; (b) bivariate Moran's R colocalization clustermaps. The authors report an average CC improvement of 0.1 across four representative cell types of slide B1, with the largest gains on cell types with less distinct morphology.*

## Limitations

The authors' Impact Statement is generic boilerplate and the Discussion only frames future work implicitly, so most of the following gaps are unacknowledged in the paper.

- **Circular ground truth.** Supervision is cell2location output, not an orthogonal cellular readout (e.g., paired single-cell, IHC, Xenium). The benchmark therefore measures fidelity to a deconvolution model, not to ground-truth cellular composition. Any systematic bias in cell2location — reference-mismatch, prior mis-specification — is baked into both the labels and the evaluation.
- **"Generalizes across tissue types" is untested.** Both datasets (her2st, STNet) are breast cancer, and both use the same HBCA scRNA-seq reference for cell2location. The cross-dataset evaluation is within tissue and within label space. The abstract's claim of tissue-type generalization is unsupported.
- **No shallow baseline.** The paper does not compare against linear regression or k-NN on the same CONCH+UNI embeddings, leaving the choice of a 7-layer MLP unjustified. Given the strength of the FM features, a much shallower head may suffice.
- **No variance / CI.** Per-fold metrics are point estimates; no error bars across LOO folds or seeds.
- **No loss ablation.** The composite loss is introduced without ablating $\lambda_1, \lambda_2$ or removing the Pearson term — the choice of three loss components is unjustified.
- **License / access constraints** of CONCH and UNI for downstream clinical use are not discussed.
- **Equation formatting.** The Pearson loss is split awkwardly across lines in the PDF, and per-cell-type vs. per-spot reduction is not fully specified.

## Why It Matters for Medical AI

The result that matters is C5. If FM-embedding + MLP can recover cell-type composition at Hist2Cell-level accuracy while fitting on a single consumer GPU and training in minutes, then cellular decomposition from H&E becomes a commodity downstream task — like patch classification or grading — rather than a research-infrastructure problem. The architectural lesson generalizes beyond cellular decomposition: when a pathology FM has already absorbed the morphological priors, the supervised head can often be small. The honest caveat is that this paper does not yet show that lesson outside breast tissue, and the supervision is a model output rather than ground truth, so practical deployment should expect to **revalidate against an orthogonal cellular readout** before trusting the abundance estimates clinically.

## References

- Paper: [Integrating Pathology Foundation Models and Spatial Transcriptomics for Cellular Decomposition from Histology Images (arXiv 2507.07013)](https://arxiv.org/abs/2507.07013) — ICML 2025 Workshop on Multi-modal Foundation Models and LLMs for Life Sciences.
- Foundation models: [CONCH (Lu et al., Nat. Med. 2024)](https://www.nature.com/articles/s41591-024-02856-4); [UNI (Chen et al., Nat. Med. 2024)](https://www.nature.com/articles/s41591-024-02857-3).
- Deconvolution target: [cell2location (Kleshchevnikov et al., Nat. Biotechnol. 2022)](https://www.nature.com/articles/s41587-021-01139-4).
- Baseline: [Hist2Cell (Zhao et al., 2024)](https://www.biorxiv.org/content/10.1101/2023.12.13.571603) — Vision Graph-Transformer for histology-to-cell decomposition.
- Datasets: [her2st (Andersson et al., Nat. Commun. 2021)](https://www.nature.com/articles/s41467-021-26271-2); [STNet (He et al., Nat. Biomed. Eng. 2020)](https://www.nature.com/articles/s41551-020-0578-x); [Human Breast Cell Atlas (Kumar et al., 2023)](https://www.nature.com/articles/s41586-023-06252-9).
