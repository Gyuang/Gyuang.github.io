---
title: "Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images"
excerpt: "Dual-pathway contrastive objective (multi-scale instance + cross-level instance-group) on top of a frozen UNI encoder wins 9/9 metrics across HER2ST, STNet, and skinST — e.g., HER2ST PCC(H) 0.541 vs TRIPLEX 0.497."
categories: [Paper, Spatial-Transcriptomics, Pathology]
permalink: /paper/gene-dml/
tags:
  - Gene-DML
  - Spatial-Transcriptomics
  - Contrastive-Learning
  - Histopathology
  - UNI
  - Multi-Scale
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-24
last_modified_at: 2026-05-24
---

## TL;DR
- Gene-DML adds two discrimination pathways on top of a CLIP-style bimodal contrastive backbone for histology-to-gene-expression prediction: (1) **multi-scale instance-level** alignment over local / neighbor / global image tiles, and (2) a **cross-level instance-group** path that pulls each instance toward the nearest centroid in the *other* modality.
- Built on a **frozen UNI** pathology foundation model, the method sweeps SOTA on **9/9 cells** (MSE, PCC(A), PCC(H)) across **HER2ST, STNet, skinST** under patient-disjoint cross-validation with stds reported. Headline: **HER2ST PCC(H) 0.541 vs TRIPLEX 0.497**; **skinST PCC(A) 0.433 vs 0.374**.
- Ablations are unusually clean: **the cross-level instance-group pathway is the larger single contributor**, but the paper omits a neighbor-only ablation, never benchmarks CARHE / THItoGene, and quietly loses on **MSE/MAE for external Visium** generalization even while winning PCC.

## Motivation
Spatial transcriptomics (ST) maps gene expression onto WSI coordinates but requires expensive specialized assays (Visium, MERFISH, seqFISH+, STARmap), motivating regressors that recover expression from routine H&E tiles. Two prior failure modes set up the gap Gene-DML wants to close:

1. **Unimodal regressors** — ST-Net, HistoGene, Hist2ST, EGN — overfit high-dimensional counts because they receive no semantic supervision from gene structure itself.
2. **Bimodal contrastive models** — BLEEP, MclSTExp, RankByGene, ST-Align — align *only* single-tile to single-profile pairs, missing both **multi-scale morphology** (neighborhood context, slide-level structure) and **shared semantic groups** (tiles with similar morphology should map near gene-expression clusters).

Gene-DML targets both in one dual-pathway objective. The medical-AI promise is a scalable, non-invasive route to molecular profiling from routine H&E for precision oncology.

## Core Innovation
The structurally novel ingredient is the **cross-level instance-group contrastive objective**: each image embedding is pulled toward the nearest *gene-expression* cluster centroid, and each gene embedding toward the nearest *image* cluster centroid — a bidirectional instance-to-group pull that complements the standard instance-to-instance alignment. Layered on top of three-scale image features (local 224x224, neighbor 1120x1120, concatenated global) fused by self-attention, this turns a BLEEP-style backbone into a multi-level discriminator with one extra loss term (weight `lambda`).

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | Dual-pathway multi-level discrimination beats prior SOTA. | Tab. 1: best on **9/9** metrics (MSE, PCC(A), PCC(H)) with std reported. | HER2ST, STNet, skinST | star-star-star |
| C2 | Each pathway contributes additively. | Tab. 3 (skinST) + Tabs. 9, 13 (HER2ST, STNet): cleanly factorial ablation, both individually positive, combined best. | 3 datasets | star-star-star |
| C3 | Generalizes better to unseen Visium slides. | Tab. 2: best PCC(A)/PCC(H) on 3 breast Visium; Tab. 8: best on 2 skin Visium. **But MSE/MAE often lose.** | 5 unseen slides | star-star |
| C4 | The instance-group pathway is the structurally novel ingredient. | Tab. 3 "I-G only" beats "I-L only" on PCC; removing feature grouping drops more than removing global-scale. | skinST (HER2ST/STNet in suppl.) | star-star |
| C5 | Multi-scale (local + neighbor + global) is necessary, not just TRIPLEX-style multi-resolution. | Global-rep ablation row in Tab. 3 (0.247 / 0.421 / 0.478 vs full 0.237 / 0.433 / 0.520) shows global helps modestly; **neighbor scale never ablated independently.** | skinST | star |
| C6 | Approach matches biological biomarker semantics. | Fig. 3 qualitative on GNAS, FASN with cherry-picked slides + oncology citations. | 2-4 slides/gene | star |
| C7 | Robust to hyperparameters. | Tabs. 4-6 (and 10-12, 14-16): MSE varies 0.01-0.02 around optimum. | 3 datasets | star-star |
| C8 | UNI freezing avoids overfitting. | Asserted as design rationale; **no UNI-finetuned or alternate-FM ablation**. | none | star |

## Method & Architecture

![Gene-DML framework architecture](/assets/images/paper/gene-dml/fig_p003_01.png)
*Figure 1: Gene-DML framework — a multi-scale image encoder (local / neighbor / global, UNI frozen) and a gene encoder feed two discrimination pathways: multi-scale instance-level (local/neighbor/global to gene) and cross-level instance-group via per-modality K-means centroids.*

Step by step:

1. **Inputs.** Per ST spot, a WSI tile triplet at three scales and the matched 250-gene vector: $I \in \mathbb{R}^{N \times H \times W \times 3}$, $G \in \mathbb{R}^{N \times M}$ with $M = 250$ morphology-correlated genes.
2. **Multi-scale image encoder — UNI frozen.**
   - **Local:** 224x224 tile, UNI features projected to 512-d.
   - **Neighbor:** 1120x1120 region (center + 5 neighbor tiles) tokenized into 25 patches; learnable self-attention blocks refine the frozen UNI features.
   - **Global:** all local-tile features for the slide are concatenated and passed through a transformer block, producing one TRIPLEX-style global token.
3. **Scale-wise fusion.** Concatenate per-spot $\{I_{L}, I_{N}, I_{G}\}$ along channels and apply self-attention to re-weight scales into instance embeddings $I^{E_{ins}}$.
4. **Gene encoder.** Two `Linear + GELU + Dropout` blocks then an FFN project raw counts into the shared latent space, yielding $G^{E_{ins}}_i \in \mathbb{R}^d$.
5. **Feature grouping.** A separate FC head produces cluster-friendly embeddings $I^{E_{clu}}, G^{E_{clu}}$. K-means (k = 18 / 25 / 90 for HER2ST / skinST / STNet) on unit-normalized vectors yields per-modality centroids $\{C^I_j\}, \{C^G_j\}$. Clustering on a *separate* projection isolates clustering noise from alignment.
6. **Cross-level assignment.** Each instance gets $A(i) = j$ as its nearest centroid in the *other* modality. Two alignments are enforced bidirectionally: $G^{E_{ins}}_i \leftrightarrow C^I_j$ and $I^{E_{ins}}_i \leftrightarrow C^G_j$.

![Cross-level instance-group alignment schematic](/assets/images/paper/gene-dml/fig_p005_01.png)
*Figure 2: Cross-level instance-group pathway — each image instance is pulled toward its nearest gene-expression centroid (and vice versa), enforcing consistency across representational hierarchies.*

7. **Losses.**
   - **Multi-scale instance-level (Eq. 4):** for each scale $s \in \{L, N, G\}$, form soft target $T^s = \sigma((\mathrm{sim}(I^s, I^s) + \mathrm{sim}(G, G)) / 2\tau)$, then symmetric cross-entropy between scale-$s$ image embeddings and gene embeddings; averaged over scales. BLEEP-style internal-similarity soft targets, replicated per scale, so the model never assumes hard one-hot positive pairs.
   - **Cross-level instance-group (Eq. 5):** soft-target cross-entropy between $(I^{E_{ins}}_i, C^G_j)$ and $(G^{E_{ins}}_j, C^I_i)$ with $\tau_{I\text{-}G} = 0.07$.
   - **Prediction (Eq. 6):** MSE between a head on $I'_i$ and ground truth $G_i$.
   - **Total (Eq. 7):** $L = L_{multi\_ins} + \lambda \cdot L_{cross} + L_{pred}$, default $\lambda = 0.8$ on all three datasets.
8. **Training.** Adam, lr 1e-4 with StepLR decay 0.95 every 20 epochs, batch 256, single NVIDIA RTX A6000 (48 GB).

## Experimental Results

### Cross-validation (patient-disjoint)

All numbers as `mean +/- std` over 8/8/4-fold patient-disjoint CV. PCC(A) = all 250 genes; PCC(H) = top-50 highly predictive genes.

| Method | HER2ST MSE / PCC(A) / PCC(H) | STNet MSE / PCC(A) / PCC(H) | skinST MSE / PCC(A) / PCC(H) |
|---|---|---|---|
| ST-Net (20) | 0.260 / 0.194 / 0.345 | 0.209 / 0.116 / 0.223 | 0.294 / 0.274 / 0.382 |
| HistoGene (21) | 0.314 / 0.168 / 0.302 | 0.194 / 0.100 / 0.219 | 0.270 / 0.133 / 0.261 |
| Hist2ST (22) | 0.285 / 0.118 / 0.248 | 0.181 / 0.044 / 0.099 | 1.291 / 0.004 / 0.053 |
| EGN (23) | 0.241 / 0.197 / 0.328 | 0.192 / 0.111 / 0.203 | 0.281 / 0.281 / 0.388 |
| BLEEP (23) | 0.277 / 0.151 / 0.277 | 0.235 / 0.095 / 0.193 | 0.297 / 0.269 / 0.396 |
| TRIPLEX (24) | 0.228 / 0.314 / 0.497 | 0.202 / 0.206 / 0.352 | 0.268 / 0.374 / 0.490 |
| M2OST (25) | 0.302 / 0.231 / 0.410 | 0.278 / 0.201 / 0.353 | 0.271 / 0.300 / 0.490 |
| **Gene-DML (ours)** | **0.210 / 0.331 / 0.541** | **0.179 / 0.237 / 0.384** | **0.237 / 0.433 / 0.520** |

Clean sweep across **9/9** metric cells. **CARHE and THItoGene are not benchmarked** in the paper (neither in main nor supplementary), so no direct comparison can be made to those concurrent baselines.

### External Visium generalization (Tab. 2)
On three unseen breast Visium slides Gene-DML wins **PCC(A) and PCC(H) everywhere** (e.g., Visium-1 PCC(H) **0.415 vs TRIPLEX 0.241**), but **MSE/MAE often lose** to TRIPLEX/BLEEP (Visium-1 MSE 0.410 vs TRIPLEX 0.351; MAE 0.512 vs 0.464). Authors frame this as a semantic-alignment vs per-spot regression trade-off — defensible for biomarker discovery, but the abstract does not carry this caveat.

### Ablations (skinST, Tab. 3)

| Variant | MSE | PCC(A) | PCC(H) |
|---|---|---|---|
| Plain bimodal baseline (no MS, no I-G) | 0.297 | 0.269 | 0.396 |
| + Multi-scale I-L only | — | +0.039 | +0.053 |
| + Cross-level I-G only | — | +0.051 | +0.082 |
| **Both (full Gene-DML)** | **0.237** | **0.433** | **0.520** |
| Full minus global-scale rep | 0.247 | 0.421 | 0.478 |
| Full minus feature grouping (k-means) | 0.253 | 0.398 | 0.456 |

Pathways are clearly **complementary**, and the **cross-level instance-group path is the larger single contributor**. Removing feature grouping hurts more than removing the global-scale representation.

### Qualitative (Fig. 3, supplementary Figs. 4-5)

![Spatial GNAS and FASN prediction on breast cancer slides](/assets/images/paper/gene-dml/fig_p007_01.png)
*Figure 3: Spatial GNAS and FASN prediction on breast cancer — Gene-DML produces visibly sharper maps and higher per-slide PCC than TRIPLEX (e.g., FASN 0.772 vs 0.621; GNAS 0.608 vs 0.440). Both genes are well-known oncology biomarkers; the selection is rhetorically convenient but biologically appropriate.*

![Supplementary HER2ST GNAS/FASN predictions](/assets/images/paper/gene-dml/fig_p014_01.png)
*Figure 4: Supplementary HER2ST GNAS/FASN predictions — Gene-DML outperforms TRIPLEX across diverse patient samples, supporting the qualitative claim.*

### Hyperparameters
- **k sweep (Tab. 4):** skinST optimum at 25; HER2ST at 18 (Tab. 10); STNet at 90 (Tab. 14) — moderately k-sensitive, requires per-dataset tuning.
- **lambda sweep (Tab. 5):** 0.8 optimal on all three datasets.
- **tau_{I-G} (Tab. 6):** 0.07 optimal across datasets.

## Limitations

**Acknowledged by the authors.**
- MSE/MAE not always best in external generalization (framed as semantic vs regression trade-off).
- K is dataset-specific (18 / 25 / 90), requires per-dataset tuning.

**Not addressed — flag.**
1. **External-Visium absolute-error loss.** Abstract emphasises PCC dominance but elides that Gene-DML often loses on **MSE/MAE** versus TRIPLEX/BLEEP on the same external slides — readers comparing on absolute error will be surprised.
2. **No CARHE or THItoGene benchmark.** Two natural concurrent baselines requested in this review are absent from both main paper and supplementary tables — any "vs CARHE/THItoGene" claim cannot be made from this work.
3. **No neighbor-only ablation.** Only the global scale is toggled. Without a local-only and a local+neighbor row, the multi-scale claim does not isolate whether the neighbor scale specifically (vs simply using TRIPLEX-style multi-resolution) does the work.
4. **UNI frozen as a fiat.** No UNI-unfrozen control, and no alternate pathology FM (Virchow, Phikon, CONCH) ablation. We cannot tell whether the encoder choice drives the gain or whether the new losses do.
5. **No statistical significance tests** despite stds being reported — overlapping error bars at small fold counts (especially skinST with 4 folds) are not assessed.
6. **250-gene panel pre-filtered for morphology correlation.** Genes were ranked per-dataset for predictability, so the achievable correlation is inflated. No result on the full ~20k transcriptome or on a biologically-orthogonal gene panel.
7. **K-means run on-the-fly** with no schedule/EMA discussion; centroid drift across epochs and batch-size dependence are not analyzed.
8. **Inference cost vs TRIPLEX is not reported.** Multi-scale + grouping adds non-trivial compute.
9. **All H&E cancer ST.** No IHC, no normal-tissue benchmark, no cross-stain transfer.

## Why It Matters for Medical AI
Recovering gene expression from H&E alone — if reliable — would route around the cost and tissue-destruction of spatial transcriptomics assays for downstream tasks like biomarker discovery, spatial heterogeneity quantification, and cohort-scale molecular phenotyping. Gene-DML pushes the contrastive-alignment frontier on the public ST benchmarks under the most defensible protocol (patient-disjoint CV, std reported) and confirms — for the third time in this subfield — that **pathology foundation models work best when kept frozen and supervised by gene-cluster structure** rather than naive per-spot regression. The honest read for clinicians is: PCC dominance does not translate to absolute-error dominance under domain shift to Visium, so this is biomarker-discovery quality, not yet quantitative-assay replacement.

## References
- **Paper:** Song, Fan, Chang, Cai. *Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images.* WACV 2026. arXiv: [2507.14670](https://arxiv.org/abs/2507.14670)
- **Code:** [github.com/YXSong000/Gene-DML](https://github.com/YXSong000/Gene-DML)
- **Backbone:** Chen et al. *Towards a general-purpose foundation model for computational pathology (UNI).* Nature Medicine, 2024.
- **Datasets:** HER2ST (Andersson et al., Nat Comm 2021); STNet (He et al., Nat Biomed Eng 2020); skinST (Ji et al., Cell 2020); curated through HEST-1k (Jaume et al., NeurIPS 2024).
- **Closest prior art:** TRIPLEX (CVPR 2024); BLEEP (NeurIPS 2023); EGN (WACV 2023); Hist2ST (Briefings in Bioinformatics 2022); ST-Net (Nat Biomed Eng 2020).
