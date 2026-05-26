---
title: "Spatial Gene Expression at Single-Cell Resolution from Histology Using Deep Learning with GHIST"
excerpt: "UNet 3+ multitask framework predicts per-nucleus gene expression from H&E by fusing a cell-type reference prior with neighborhood-composition cross-attention — beats spot-level baselines on HER2ST (PCC 0.16 vs ST-Net 0.14) under patient-grouped CV."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/ghist/
tags:
  - GHIST
  - Spatial Transcriptomics
  - Xenium
  - UNet 3+
  - Cross-Attention
  - Single-Cell
  - HER2ST
  - TCGA
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR
- GHIST is a multitask UNet 3+ framework that predicts gene expression for **individual nuclei** in H&E by training on subcellular spatial transcriptomics (10x Xenium), jointly optimizing morphology, cell-type classification, neighborhood composition, and per-cell expression.
- The mechanism that does the load-bearing work is the **cell-type reference prior**: a small MLP predicts soft weights `W` over reference rows, computes `S = WR` (averaged scRNA-seq profiles), then a multi-head cross-attention with neighborhood composition `p_est` as query and `S` as key/value emits per-cell expression. Removing the reference collapses SVG PCC in ablation.
- **Headline result:** on HER2ST spot-level benchmark under **patient-grouped 4-fold CV**, GHIST reaches **PCC=0.16 / SSIM=0.10** (all genes) and **PCC=0.27 / SSIM=0.26** on top-20 SVGs, beating ST-Net (0.14/0.08) and DeepPT (0.11/0.07). Single-cell Xenium SVG PCC ≈ 0.7 looks strong but is measured on horizontal strips of the **same slide** as training — no cross-patient single-cell PCC is reported.

## Motivation
Subcellular spatial transcriptomics platforms (Xenium, CosMx, MERSCOPE) measure RNA at single-cell resolution but remain expensive. H&E is universal but every existing histology→ST method — ST-Net, HisToGene, Hist2ST, DeepPT, BLEEP, DeepSpaCE, GeneCodeR — operates at the **Visium spot** level (~50–100 µm), where each spot is a mixture of many cells of multiple types. The authors' own 2023 benchmark (Chan et al., bioRxiv 2023.12.12.571251) showed those spot-level predictors have limited translational potential. None natively exploit the subcellular SST signal that Xenium now provides, and none formulate the prediction at single-cell resolution from H&E alone at inference. GHIST positions itself as the first such method, enabling in-silico enrichment of large H&E archives (TCGA, clinical) with a spatial-omics modality without wet-lab SRT.

## Core Innovation
Three ideas, all small in code and load-bearing in effect:

1. **Per-cell output from a single per-patch forward pass.** A UNet 3+ backbone (no per-cell crops, no ViT, no pathology FM pretraining) processes a 256×256 H&E patch with thousands of nuclei at once. For each nucleus mask, features from the first and last conv layers are masked, summed, area-normalized, concatenated with the patch-mean of the same volumes, and projected through 2× FC+ReLU to a 256-d per-cell embedding `x_nucleus` that always carries patch context.
2. **Cell-type reference as a prior over rows, not as a prompt.** A small MLP on `x_nucleus` predicts a softmax `W ∈ R^{n_ref}` over reference cell types. The baseline per-cell expression is `S_j = Σ_k W_k R_{kj}` (Eq. 10) — a weighted average of averaged scRNA-seq profiles (HCA / CELLxGENE). Then multi-head cross-attention (8 heads) with query = `p_est` (estimated neighborhood composition) and key/value = `S` emits an additive bias `b`; final `y' = ReLU(S + b)`. Reference profiles enter as a prior, neighborhood composition as a modulator — not as raw H&E tokens.
3. **Cell-type / expression consistency losses + inference-time recovery rule.** A second cell-type module takes ground-truth or predicted expressions and predicts cell type; cross-entropy, cosine, and MSE losses tie the two modules together. At inference an additive logit bump `v_t = α · n_cells · (p_est,t − p_CT,t) · |Φ_t|` rescues B/myeloid/T-cells from T-cell dominance. `α=2` for Xenium, `α=10,000` (with malignant prior zeroed) for TCGA — a heavy-handed swap with no sensitivity analysis.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | First method to predict at single-cell resolution from H&E. | Method description; Fig. 3 comparison set is all spot-based. | ⭐⭐ — true at preprint time (Jul 2024); concurrent single-cell methods (Henium, sCellST, Hist2Cell, BLEEP) are not benchmarked. |
| C2 | Top-20 SVGs reach median PCC ≈ 0.7 at single-cell level (BreastCancer1/2). | Fig. 2f boxplots; marker genes SCD 0.74, FASN 0.77, FOXA1 0.80, EPCAM 0.84. | ⭐⭐ — CV folds are **horizontal strips of the same slide**; no patient-grouped single-cell PCC. |
| C3 | Outperforms SOTA spot methods on HER2ST. | PCC 0.16/SSIM 0.10 all genes; PCC 0.27/SSIM 0.26 top-20 SVGs; beats ST-Net (0.14/0.08), DeepPT (0.11/0.07), HisToGene, Hist2ST, DeepSpaCE, GeneCodeR under **4-fold patient-grouped CV**. | ⭐⭐⭐ — patient-grouped CV, multiple metrics, multiple gene subsets, multiple baselines. The cleanest claim. |
| C4 | Generalizes to other cancer tissues (lung adeno, melanoma). | Supp. Fig. 4–5: cell-type-proportion PCC = 0.97 (lung), 0.92 (melanoma). | ⭐ — only **cell-type-proportion** correlation reported, not per-gene PCC; one slide each; same Xenium scanner. |
| C5 | Translational potential on TCGA-BRCA: significant survival stratification. | Fig. 3g KM p=0.017; C-index 0.57 (3-fold × 100 repeats). | ⭐⭐ — significant but C-index 0.57 only marginally above the RNA-Seq-STgene baseline 0.55; in-distribution HER2+ subtype. |
| C6 | Spatial L statistic (χ²=6.34, p=0.012) beats bulk RNA-seq (χ²=1.9) for survival. | Fig. 5a. | ⭐⭐ — n=92 single cohort, no external replication, no variance bars on χ². |
| C7 | Single-cell reference input is required for good SVG PCC. | Supp. Fig. 1d ablation: removing reference collapses top-20/50 SVG PCC. | ⭐⭐⭐ — clean, large effect, biologically motivated. |
| C8 | Neighborhood-composition recovery rescues B and myeloid cells from T-cell dominance. | Supp. Fig. 1b ablation: minority classes vanish without it. | ⭐⭐ — convincing, but the recovery is an inference-time, hand-tuned rule (`α`, `prob>0.6` mask) selected on validation. |
| C9 | Predicts roughly double the cell types of BCSS-style approaches. | Discussion paragraph; ~10–12 vs ~4 types. | ⭐ — qualitative; no head-to-head with a BCSS-trained classifier. |
| C10 | "Single-cell SGE on TCGA." | TCGA inference uses Hover-Net nuclei + GHIST pseudobulk; no per-cell ground truth on TCGA. | ⭐ — single-cell in **spatial granularity** only; not validated at per-cell level on TCGA. |

**Honest analysis.** Two results carry the paper: the HER2ST spot-level head-to-head (C3) and the reference-input ablation (C7). Both are properly cross-validated, multi-metric, biologically motivated. Everything else has weaker scaffolding:

1. The "single-cell SVG PCC ≈ 0.7" headline (C2) is measured on horizontal strips of the same Xenium slide as training. Spatial autocorrelation means train/test strips share tissue context. No leave-one-slide-out, no cross-patient single-cell PCC.
2. External-tissue generalization (C4) reports only cell-type proportion correlation, not per-gene PCC — a much weaker bar.
3. TCGA survival (C5, p=0.017, C-index 0.57) trains on a single Xenium HER2+ slide and tests on the HER2+ TCGA subset — in-distribution by subtype. C-index 0.57 is barely above the gene-subset baseline 0.55.
4. No error bars on any single-number metric (PCC 0.16, SSIM 0.10, C-index 0.57); variance is shown over genes via boxplots, not over training seeds. PCC 0.16 vs 0.14 is presented as superiority without a statistical test.
5. Concurrent single-cell methods (BLEEP, Henium, sCellST, Hist2Cell) are not benchmarked. The "first / only single-cell" framing depends on this gap.
6. No foundation-model baseline. UNet 3+ from scratch with He init; Discussion recommends FMs but provides no UNI / Phikon / Virchow comparison.
7. The TCGA inference uses `α = 10,000` (vs `α = 2` for Xenium) with no sensitivity analysis — a hand-tuned prior swap that essentially overrides the model's cell-type head.
8. "Single-cell SGE on TCGA" is unvalidated at the per-cell level — no GT exists; all downstream validation collapses to bulk pseudobulk or cell-type proportion.

## Method & Architecture

![GHIST overall framework](/assets/images/paper/ghist/fig_p030_01.png)
*Figure 1: GHIST overview — H&E patch → UNet 3+ backbone (5-scale full-skip CNN, He init, no FM pretraining) → four multitask heads: pixel morphology / cell type (CE), per-nucleus cell type (auxiliary CE), patch-level neighborhood composition (KL), and per-cell gene expression. The expression head computes a soft weighted average `S = WR` over reference profiles and modulates it with multi-head cross-attention conditioned on neighborhood composition.*

The per-nucleus embedding `x_nucleus ∈ R^{256}` pools first/last-conv-layer features within each nucleus mask, area-normalizes to counter nucleus-size bias, and concatenates with the patch-mean of the same volumes. The expression head with reference is:

$$S_j = \sum_k W_k\, R_{kj}, \qquad W = \mathrm{softmax}(\mathrm{MLP}(x_{nucleus}))$$

$$b = \mathrm{Linear}\big(\mathrm{MultiHeadCrossAttn}(Q=p_{est},\; K=V=S)\big), \qquad y' = \mathrm{ReLU}(S + b)$$

So the reference enters as a **prior over rows** (not as prompts, not as raw expression injection), and the neighborhood composition is the **query** of cross-attention against the reference-weighted profile — not against raw H&E tokens. The ablation in Supp. Fig. 1d shows that removing `R` (regressing `S` directly from `x_nucleus`) markedly degrades SVG PCC.

Auxiliary losses tie cell-type and expression predictions together: `L_{CT,expr}` (CE on cell type predicted from predicted expressions), `L_{CT,embed}` (cosine distance between embeddings from GT vs predicted expressions, weighted ×100), `L_{CT,logits}` (MSE between logits). Neighborhood composition has two-way KL consistency (`L_{NC,est}` against patch GT, `L_{NC,pr}` against per-cell-head aggregation). For HER2ST spot-mode, per-cell predictions in a spot are summed (Eq. 14) and nuclei labels come from Hover-Net retrained on NuCLS (6 types).

Training is AdamW, lr 1e-3, wd 1e-4, batch 8, 50 epochs, single RTX 4090. Inference uses 30-px patch overlap with largest-area voting per nucleus. TCGA inference is an ensemble of 3 BreastCancer2 checkpoints.

## Experimental Results

### Datasets

| Dataset | Platform | Tissue | Genes | Use |
|---|---|---|---|---|
| BreastCancer1 (Xenium) | 10x Xenium | breast ER+/HER2+/PR− | 313 | train+test (5-fold horizontal-strip CV, **same slide**) |
| BreastCancer2 (Xenium) | 10x Xenium | breast ER−/HER2+/PR− | 280 | train+test, ablation, source for TCGA model |
| LungAdenocarcinoma | 10x Xenium FFPE | lung | 377 | external tissue eval (cell-type proportion only) |
| Melanoma | 10x Xenium | skin | 382 | external tissue eval (cell-type proportion only) |
| HER2ST | spot-level (Andersson 2021) | HER2+ breast, 8 patients, 36 sections | 785 (filtered) | spot-mode benchmark, **4-fold patient-grouped CV** |
| NuCLS | H&E + nuclei | breast (TCGA-sourced) | — | trains Hover-Net for HER2ST cell-type labels |
| TCGA-BRCA HER2+ | diagnostic WSI + bulk RNA + WGS + clinical | breast, 92 patients | — | H&E inference, downstream survival |

### Single-cell expression on Xenium (Fig. 2)

![Per-cell predictions on Xenium](/assets/images/paper/ghist/fig_p031_01.png)
*Figure 2: Per-cell predictions on BreastCancer1/2. SVG PCC ≈ 0.7 (top-20) / 0.6 (top-50) on horizontal-strip CV; marker-gene scatter SCD 0.74, FASN 0.77, FOXA1 0.80, EPCAM 0.84. Non-SVGs sit near zero as a built-in negative control.*

| Metric | BreastCancer1 | BreastCancer2 | Lung adeno | Melanoma |
|---|---|---|---|---|
| Top-20 SVG median PCC | ≈ 0.7 | ≈ 0.7 | n/a | n/a |
| Top-50 SVG median PCC | ≈ 0.6 | ≈ 0.6 | n/a | n/a |
| Non-SVG median PCC | ≈ 0–0.1 | ≈ 0–0.1 | n/a | n/a |
| Cell-type-proportion PCC | high | high | r = 0.97 | r = 0.92 |

### Spot-level vs SOTA on HER2ST (Fig. 3a–e) — 4-fold patient-grouped CV

![HER2ST benchmark and TCGA survival](/assets/images/paper/ghist/fig_p032_01.png)
*Figure 3: HER2ST 4-fold patient-grouped CV: **GHIST tops PCC/SSIM** (all genes, SVGs, HVGs) over ST-Net, DeepPT, Hist2ST, HisToGene, DeepSpaCE, GeneCodeR. Applied to 92 TCGA-BRCA HER2+ patients, GHIST pseudobulk gives a significant KM separation (p=0.017).*

| Method | PCC (all) | SSIM (all) | PCC (top-20 SVG) | SSIM (SVG) |
|---|---|---|---|---|
| **GHIST** | **0.16** | **0.10** | **0.27** | **0.26** |
| ST-Net | 0.14 | 0.08 | — | — |
| DeepPT | 0.11 | 0.07 | — | — |
| HisToGene / Hist2ST / DeepSpaCE / GeneCodeR | lower (Fig. 3a/b, Supp. Fig. 6) | — | — | — |

Top-5 genes per GHIST on HER2ST: GNAS 0.42, FASN 0.42, SCD 0.34, MYL12B 0.32, CLDN4 0.32. PCC on top-10% HVGs = 0.20.

### TCGA-BRCA HER2+ application (Fig. 4 & Fig. 5a)

![TCGA in-silico modality and survival](/assets/images/paper/ghist/fig_p033_01.png)
*Figure 4: GHIST applied to 92 TCGA-BRCA HER2+ patients — predicted single-cell expression of malignant marker DSP and TME marker SFRP4, plus per-patient cell-type composition. No per-cell ground truth exists on TCGA; all downstream validation is bulk/pseudobulk or cell-type proportion.*

![Downstream: spatial L statistic and CNA hotspots](/assets/images/paper/ghist/fig_p034_01.png)
*Figure 5: Spatial L statistic enables survival stratification (χ²=6.34, p=0.012) above bulk RNA-seq (χ²=1.9); CNA → differential patterning recovers known 8q24 and 17q11–21 (HER2 amplicon) hotspots.*

| Source | C-index (3-fold × 100 repeats) | KM log-rank p |
|---|---|---|
| **GHIST pseudobulk (from H&E)** | **0.57** | **0.017** |
| RNA-Seq-STgene (HER2ST panel subset) | 0.55 | — |
| Cell-type-specific GE (GHIST) | χ² = 2.16 | n.s. |
| Bulk RNA-seq vs Spatial L statistic | χ² 1.9 vs **6.34** | L: **p = 0.012** |

### Ablation (Supp. Fig. 1, BreastCancer2)

![Ablation: reference and neighborhood heads](/assets/images/paper/ghist/fig_p036_01.png)
*Supp. Fig. 1: Removing the single-cell reference input collapses top-20/50 SVG PCC; removing the cell-type or neighborhood-composition heads erases B and myeloid cells from predictions (the model collapses minorities into the dominant T-cell class). The Eq. 7–8 inference-time recovery is essential for the immune-cell story.*

### External tissue generalization

![Lung adenocarcinoma external eval](/assets/images/paper/ghist/fig_p039_01.png)
*Supp. Fig. 4: Lung adenocarcinoma Xenium — cell-type proportion PCC = 0.97. Per-gene PCC is not reported.*

![Melanoma external eval](/assets/images/paper/ghist/fig_p040_01.png)
*Supp. Fig. 5: Melanoma Xenium — cell-type proportion PCC = 0.92. Same scanner (10x Xenium); per-gene PCC not reported.*

![TCGA stain QC failure modes](/assets/images/paper/ghist/fig_p043_01.png)
*Supp. Fig. 8: Stain-quality failure modes in TCGA-BRCA — TCGA-AO-A12G understained vs TCGA-PE-A5DD overstained, both flagged by pathologist QC and excluded.*

## Limitations
- **No cross-slide / cross-patient single-cell PCC.** The 5-fold Xenium CV splits by horizontal strips within the *same* slide; only HER2ST CV is patient-grouped (and only at spot level). The "single-cell PCC ≈ 0.7" headline is intra-slide.
- **External tissues report only cell-type proportion correlation** (PCC 0.97 lung, 0.92 melanoma), not per-gene PCC — a much weaker bar than the in-distribution Xenium SVG numbers.
- **No comparison vs concurrent single-cell methods** (BLEEP, Henium, sCellST, Hist2Cell). BLEEP is named in related work but not in Fig. 3. The "first / only single-cell" framing depends on this gap.
- **No foundation-model baseline.** UNet 3+ trained from scratch with He init; no UNI / Phikon / Virchow comparison despite the authors recommending FMs in the Discussion.
- **TCGA inference uses hand-tuned `α = 10,000`** (vs `α = 2` for Xenium) with the malignant prior zeroed — an aggressive prior swap with no sensitivity analysis. The "single-cell SGE on TCGA" framing is unvalidated at the per-cell level because no GT exists; all downstream validation is bulk pseudobulk, cell-type proportion, or L-statistic.
- **Survival C-index 0.57 (p=0.017)** is only marginally above the RNA-Seq-STgene panel baseline 0.55 on an in-distribution HER2+ TCGA subset; no replication on another cohort or cancer type.
- **No error bars on single-number metrics** (PCC 0.16, SSIM 0.10, C-index 0.57); variance is over genes via boxplots, not over training seeds. PCC 0.16 vs 0.14 is reported as superiority without a statistical test.
- **Gene panel size (280–382 genes)** limits identifiable cell types — myoepithelial cells cannot be discriminated without P63 in panel (acknowledged by authors).
- **Only two Xenium breast slides** train the main per-cell model — small slide count, even if cell count is large.
- **Inference depends on Hover-Net + BIDCell**; segmentation errors and stain shift on TCGA propagate into predictions. Authors exclude 2/92 slides by pathologist QC but do not quantify segmentation-driven prediction noise.

## Why It Matters for Medical AI
The interesting move for medical AI is the *substrate switch*: train on Xenium (where transcriptomics, H&E, and nucleus segmentation are co-registered at single-cell resolution), then deploy the image branch on archival H&E (TCGA) where no transcriptomics exists. If the per-cell predictions truly transfer, this enables retrospective spatial-biomarker discovery on existing pathology archives without wet-lab SRT. The mechanistic contribution that is likely to be portable is the **reference-as-prior-over-rows + neighborhood-as-cross-attn-query** design: it gives the model a biologically motivated initialization (`S = WR`) and uses the patch-level composition as a soft modulator rather than a hard label. The clearest empirical wins — HER2ST under patient-grouped CV (C3) and the reference ablation (C7) — are real and reproducible. The single-cell-on-TCGA narrative (C10) is the part that needs the most scepticism: per-cell PCC on out-of-distribution H&E remains unvalidated because the ground truth does not exist. Survival C-index 0.57 on an in-distribution HER2+ cohort is interesting but not yet clinically actionable.

## References
- Paper: *Nature Methods* (Sep 2025; DOI [10.1038/s41592-025-02795-z](https://doi.org/10.1038/s41592-025-02795-z))
- Preprint: bioRxiv [2024.07.02.601790](https://doi.org/10.1101/2024.07.02.601790) (analysis based on this version)
- Code: [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST) (MIT-style, public)
- Backbone: UNet 3+ ([Huang et al. 2020](https://arxiv.org/abs/2004.08790))
- Cell-type references: [Human Cell Atlas](https://www.humancellatlas.org/), [CELLxGENE](https://cellxgene.cziscience.com/)
- Segmentation: [Hover-Net](https://github.com/vqdang/hover_net), [BIDCell (Fu et al.)](https://github.com/SydneyBioX/BIDCell)
- Cell-type labels: [scClassify](https://github.com/SydneyBioX/scClassify), [NuCLS](https://sites.google.com/view/nucls/)
- Stain normalization: [torchstain](https://github.com/EIDOSLAB/torchstain)
- Spot-level baselines: [ST-Net](https://github.com/bryanhe/ST-Net), [HisToGene](https://github.com/maxpmx/HisToGene), [Hist2ST](https://github.com/biomed-AI/Hist2ST), DeepPT, DeepSpaCE, GeneCodeR
- Related (not benchmarked): [BLEEP](https://arxiv.org/abs/2306.01859), Henium, sCellST, Hist2Cell
- External cohorts: [HER2ST (Andersson 2021)](https://www.nature.com/articles/s41467-021-26271-2), [TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
