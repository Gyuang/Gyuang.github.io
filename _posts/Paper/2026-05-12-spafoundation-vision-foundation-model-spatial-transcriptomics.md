---
title: "SpaFoundation: a visual foundation model for spatial transcriptomics"
excerpt: "ViT-B/16 iBOT-pretrained on 1.84M HEST-1K patches hits PCC 0.84 for ERBB2 on a single Visium HD slide — strong number, thin evidence."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/spafoundation-vision-foundation-model-spatial-transcriptomics/
tags:
  - SpaFoundation
  - Spatial-Transcriptomics
  - iBOT
  - Vision-Transformer
  - HEST-1K
  - Foundation-Model
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- SpaFoundation is a vision-only ST foundation model: a **ViT-B/16 (~80M params)** pretrained with **iBOT** (self-distillation + masked image modeling) on **1.84M HEST-1K-3 patches** from 1,113 samples across 26 tissue types — no gene-expression supervision during pretraining.
- Headline result on Visium HD breast cancer high-resolution inference (16μm → 8μm): **ERBB2 PCC 0.84 vs ST-Net's 0.65 (+29.23%)**; median PCC on the top-1,000 HVGs is **0.19** (+91.28% relative over ST-Net).
- The headline numbers come from a **single Visium HD slide** with no variance bars, and the paper invokes UNI/CONCH/CHIEF/STPath/OmiCLIP as motivation but never benchmarks against any of them — comparators are 2020-era ST-Net, His2ST, BLEEP, and ImageNet ResNet/VGG.

## Motivation

Spatial transcriptomics (ST) faces a structural trade-off: sequencing-based platforms (Visium, Stereo-seq) recover genome-wide expression at low spatial resolution, while imaging-based platforms (MERSCOPE, seqFISH+, CosMx) give subcellular resolution for only a small gene panel. The cheap bridge is in-silico gene-expression prediction from H&E. Existing image-to-ST models (ST-Net, HistoGene, His2ST, THItoGene, BLEEP, TRIPLEX) are trained per-dataset, per-task, on small cohorts, and generalize poorly across organs and platforms. Meanwhile, general computational-pathology foundation models (UNI, CONCH, CHIEF) target whole-slide H&E, not the small-patch-around-a-spot regime ST analysis demands. The paper's pitch is that a domain-specific visual backbone, pretrained on the new HEST-1K corpus, fills this niche.

## Core Innovation

SpaFoundation's design choice is to keep pretraining **vision-only** — the paired gene expression in HEST-1K is intentionally not used during pretraining. That places it in direct contrast with paired image–RNA contrastive models (BLEEP, STPath, OmiCLIP/Loki). The pretraining loop is standard iBOT: a teacher network sees global crops, a student sees local crops plus masked global crops, and the two are aligned through (i) a self-distillation cross-entropy on `[CLS]` tokens (semantic alignment across views) and (ii) a BEiT-style masked-image-modeling loss on patch tokens (local feature reconstruction). The teacher is updated as an EMA of the student. Pretraining runs 10 epochs on 1.84M 224×224 patches from a curated HEST-1K-3 subset that deliberately excludes the three downstream datasets to prevent leakage.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset(s) | Strength |
|-------|----------|------------|----------|
| **C1.** Accurately predicts spatial gene expression from H&E alone. | Fig. 1d qualitative maps (COL6A2, SPINK5); Fig. 1e per-gene PCC bars; Fig. 1f,g boxplots over 250 HEG / 171 HVG; HER2+ supplementary replication. | cSCC (4 patients), HER2+ (supp). | ⭐⭐ Two datasets, LOPO CV, but small cohorts, no significance tests. |
| **C2.** Achieves PCC 0.84 on ERBB2 for 8μm high-resolution inference, +29.23% over ST-Net. | Fig. 2b–d qualitative + bar + boxplot. | **Single** Visium HD breast cancer slide. | ⭐ Single sample, no replicates, no variance on 0.84; the 91.28% relative gain sits on a low absolute baseline (median PCC < 0.2). |
| **C3.** Outperforms ImageNet ResNet-50 / VGG-16 on tumor detection (AUC 0.94). | Fig. 2e–h, ROC across all test samples. | HBCIS, 5 test patients. | ⭐⭐ Multi-sample test, but baselines are deliberately weak — 2014–2016 ImageNet CNNs, not pathology FMs. |
| **C4.** Zero-shot spatial clustering with state-of-the-art ARI/NMI. | Fig. 2i–l; median ARI 0.3176, NMI 0.3761 across 8 sections (+29.94% / +30.77% vs second-best). | HER2+ (8 annotated sections). | ⭐⭐ Multi-section, multi-metric, single dataset; Scanpy/SpaGCN/stLearn baselines use gene expression, so it isn't strictly image-only vs image-only. |
| **C5.** Domain-specific pretraining beats general supervised pretraining for ST tasks. | Tumor detection vs ResNet/VGG only. | HBCIS. | ⭐ No direct comparison against UNI/CONCH/CHIEF — the actual peer group mentioned in the intro. ImageNet CNNs are not a fair stand-in. |
| **C6.** Generalizes "across tasks, organs, platforms." | 4 downstream tasks: cSCC (skin), HER2+ (breast), HBCIS (breast), Visium HD (breast). | 3 of 4 are breast cancer. | ⭐ Organ diversity is overstated; despite 26 organs in pretraining, downstream is breast-heavy + skin. No CNS / lung / liver evaluation. |
| **C7.** SD + MIM captures multi-level semantic and local features. | Ablation referenced (page 2) but values deferred to supplementary, not in main text. | — | Unverifiable from the manuscript. |

## Method & Architecture

![SpaFoundation overview: iBOT pretraining on HEST-1K-3 and HEG/HVG prediction on cSCC](/assets/images/paper/spafoundation/page_014.png)
*Figure 1: SpaFoundation overview — iBOT teacher–student pretraining on HEST-1K-3 (1,113 WSIs, 26 organs), the four downstream tasks, and qualitative + boxplot results for HEG/HVG gene-expression prediction on cSCC.*

**Backbone.** ViT-B/16 (~80M params, patch size 16, input 224×224) is used as both student and teacher inside the iBOT framework.

**View construction.** For each histology patch the augmentation pipeline produces 2 global crops (32%–100% crop ratio), 10 local crops (5%–32% crop ratio), and masked global crops (random subset of non-overlapping 16×16 patch tokens masked). The teacher encodes the global crops only; the student encodes the local + masked-global crops.

**Self-distillation loss (semantic).** Cross-entropy between teacher and student `[CLS]` distributions across views of the same image:

$$\mathcal{L}_{SD} = -P_{\theta_s}(X_s)^T \log P_{\theta_t}(X_t)$$

**Masked image modeling loss (local).** Student must recover the teacher's patch-token distribution on masked positions:

$$\mathcal{L}_{MIM} = -\sum_{j=1}^{N} m_j \cdot P_t(h_j^t)^T \log P_s(h_j^s)$$

where $m_j \in \{0, 1\}$ is the per-patch mask indicator.

**Joint objective.** $\mathcal{L} = \lambda_1 \mathcal{L}_{SD} + \lambda_2 \mathcal{L}_{MIM}$. The values of $\lambda_1, \lambda_2$ are not reported in the main text.

**Teacher update.** EMA of student weights into teacher — standard self-distillation stabilization.

**Pretraining setup.** 10 epochs, AdamW, LR 2e-4 with cosine annealing, mixed precision, gradient clipping, final layer frozen at start; batch size 64 per GPU on 8× V100 32GB NVlink. 10 epochs on a 1.84M-patch corpus is short by foundation-model standards (UNI is trained on ~125K WSIs with substantially more compute).

**Downstream heads.** Lightweight: LayerNorm + Linear regressor with MSE for gene expression (LOPO CV, 20–25 fine-tune epochs); a two-layer head with BCE for tumor detection (1 fine-tune epoch); a zero-shot path for clustering — PCA on patch embeddings, KNN k=6, Leiden resolution swept 0.1–3.0 step 0.01.

**Pretraining data — HEST-1K-3.** A filtered subset of HEST-1K (Jaume et al., 2024) with the three downstream datasets (HER2+, cSCC, HBCIS) removed: **1,113 samples, 26 tissue types, 1.84M 224×224 patches**. The organ distribution is dominated by spinal cord, brain, bowel, breast, and skin — a meaningful bias the paper does not foreground.

## Experimental Results

![Three downstream tasks: high-resolution Visium HD inference, HBCIS tumor detection, HER2+ zero-shot clustering](/assets/images/paper/spafoundation/page_016.png)
*Figure 2: Three downstream tasks — (a–d) high-resolution ERBB2 inference on Visium HD breast cancer with PCC 0.84 vs ST-Net 0.65; (e–h) tumor detection on HBCIS with AUC 0.94; (i–l) zero-shot spatial clustering on HER2+ samples H1 and C1 with best ARI/NMI across 8 annotated sections.*

| Task | Dataset | Metric | **SpaFoundation** | Best baseline | Δ |
|------|---------|--------|--------------------|---------------|---|
| HEG spatial gene expression | cSCC (top 250 HEG) | PCC (Fig. 1f, boxplot median) | **highest of 6 methods** | BLEEP strong on HEG | qualitative |
| HVG spatial gene expression | cSCC (top 1,000 HVG) | PCC (Fig. 1g) | **highest in "most cases"** | His2ST competitive | qualitative |
| High-res inference (ERBB2) | Visium HD breast cancer | PCC (Fig. 2c) | **0.84** | ST-Net 0.65 | **+29.23%** |
| High-res inference (top-1,000 HVG) | Visium HD breast cancer | median PCC (Fig. 2d) | **0.19** | ST-Net (not given) | **+91.28%** relative |
| High-res inference (top-50 genes) | Visium HD breast cancer | median PCC | **0.60** | ST-Net (not given) | best |
| Tumor detection | HBCIS, all test samples | AUC (Fig. 2h) | **0.94** | VGG-16 −1.55% / ResNet-50 −4.16% | +1.55% / +4.16% |
| Spatial clustering (zero-shot) | HER2+ H1 | ARI / NMI (Fig. 2j) | **0.4219 / 0.5309** | second-best lower | best |
| Spatial clustering (zero-shot) | HER2+ C1 | ARI / NMI (Fig. 2k) | **0.6058 / 0.4687** | second-best lower | best |
| Spatial clustering (8 sections) | HER2+ | median ARI / NMI (Fig. 2l) | **0.3176 / 0.3761** | second-best | **+29.94% / +30.77%** relative |

**Critical comparator audit.** The paper explicitly invokes UNI, CONCH, and CHIEF as the motivating peer group for a domain-specific ST FM, but never benchmarks against them. It also makes no mention of **STPath** (vision-only ST FM, April 2025) or **OmiCLIP/Loki** (paired image–RNA contrastive FM, Nature Methods 2025) — the two most direct competitors that were public when this preprint was posted. Several stronger baselines (HistoGene, His2ST, BLEEP, TRIPLEX, THItoGene) are dropped from the high-resolution Visium HD task "due to high computational complexity," leaving only ST-Net (a 2020 DenseNet-121) as the comparator behind the headline 0.84 PCC.

**Baseline-stability observation.** His2ST is stronger on HVG than HEG; BLEEP is stronger on HEG than HVG; SpaFoundation is stable on both. This is a defensible secondary claim, but it is shown only on cSCC.

## Limitations

1. **No comparison against contemporaneous ST foundation models** (STPath, OmiCLIP/Loki, HEST-1K's own pretrained encoder, or fine-tuned CONCH/UNI). The introduction's positioning against pathology FMs is never tested.
2. **No variance / error bars** on headline metrics — the 0.84 PCC and 0.94 AUC are point estimates from single runs.
3. **No multi-seed pretraining or fine-tuning.** 10 pretraining epochs on 1.84M patches is short for the foundation-model regime.
4. **Spot-count inconsistency** in the Visium HD section: main text says ~530K spots at 8μm (page 3) but Methods says ~580K (page 9).
5. **Loss-weight hyperparameters $\lambda_1, \lambda_2$ are not disclosed.**
6. **Downstream evaluation is breast-dominated.** 3 of 4 downstream datasets are breast; cSCC is skin. The 26-organ pretraining is never evaluated on the 24+ unseen organs (CNS, bowel, lung, etc.).
7. **No species-stratified analysis.** HEST-1K is ~52.8% mouse, ~47.2% human; downstream is all human. Domain shift is not quantified.
8. **HVG performance is modest.** Median PCC 0.19 across 1,000 HVGs on Visium HD means the long tail is largely unpredicted — but the abstract reads as if high-resolution inference is solved.
9. **No deconvolution or cell–cell communication task** — both are natural next benchmarks for an ST FM and would have stress-tested the embedding more honestly than zero-shot clustering on 8 sections.

## Why It Matters for Medical AI

Spatial transcriptomics is one of the few biomedical modalities where an image-only foundation model can plausibly substitute for an expensive wet-lab assay, since H&E is essentially free and Visium/Visium HD experiments are not. A working vision-only ST FM would be valuable: cheap retrospective expression maps over archival WSIs, candidate-biomarker localization without re-sequencing, and a clean starting point for downstream tasks. SpaFoundation is a reasonable engineering execution of that idea — iBOT on HEST-1K is the right recipe to try, and the high-resolution Visium HD experiment is a genuinely interesting setting. The problem is calibration. The paper sells a "+91.28%" improvement, but the absolute median PCC is 0.19 on a single slide, with no variance and no comparison against the obvious 2025-era peers (STPath, OmiCLIP, UNI). Readers in medical AI should treat SpaFoundation as a credible **proof-of-concept for vision-only ST pretraining**, not as evidence that the problem is solved or that this design beats the paired-modality alternatives.

## References

- **Paper (bioRxiv preprint, Aug 11, 2025):** [https://doi.org/10.1101/2025.08.07.669202](https://doi.org/10.1101/2025.08.07.669202)
- **Code:** [https://github.com/NingZhangCSUBio/SpaFoundation](https://github.com/NingZhangCSUBio/SpaFoundation)
- **Pretraining corpus — HEST-1K (Jaume et al., 2024):** [https://huggingface.co/datasets/MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest)
- **iBOT (Zhou et al., 2022):** Image BERT Pre-Training with Online Tokenizer.
- **BEiT (Bao et al., 2022):** masked image modeling reference for the MIM loss term.
- **Image-to-ST baselines:** ST-Net (He et al., 2020), HistoGene, His2ST, THItoGene, BLEEP, TRIPLEX.
- **Pathology FMs invoked but not benchmarked:** UNI, CONCH, CHIEF.
- **Direct ST FM competitors not cited:** STPath (vision-only, Apr 2025), OmiCLIP / Loki (paired image–RNA contrastive, Nature Methods 2025).
- **Downstream datasets:** cSCC (Andrew et al., GSE144240), HER2+ (Andersson et al., almaan/her2st), HBCIS (He et al., 2020, Mendeley 29ntw7sh4r), Visium HD Human Breast Cancer (10x Genomics).
