---
title: "Virchow: A Million-Slide Digital Pathology Foundation Model"
excerpt: "ViT-H/14 (632M) trained with DINOv2 on 1.49M MSKCC H&E WSIs reaches 0.949 pan-cancer specimen-level AUC across 17 tissues and drops only -0.005 weighted F1 on unnormalized CRC vs CTransPath's -0.118."
categories:
  - Paper
tags:
  - Virchow
  - DINOv2
  - ViT-H
  - Pathology-Foundation-Model
  - Self-Supervised-Learning
  - Whole-Slide-Imaging
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

> **Scope note.** This post reviews **Virchow1** (arXiv:2309.07778 v5, *Nature Medicine* 2024) trained on **1,488,550 MSKCC H&E WSIs**. The follow-on **Virchow2** (arXiv:2408.00738, ~3.1M slides, DINOv2 with register tokens) is a different paper and is **not** analyzed here. Where the user prompt or external commentary references "3M slides," that is Virchow2.

## TL;DR

- **Pretraining at pathology-unprecedented scale.** A vanilla **ViT-H/14 (632M params)** is trained with **DINOv2** SSL on **1,488,550 H&E WSIs from 119,629 MSKCC patients across 17 tissue groups** — roughly **50x more slides** than TCGA-based predecessors. No bespoke pathology priors: 224x224 tiles, 2 global + 8 local crops, iBOT masked-patch loss, 131,072-prototype projection head.
- **Headline pan-cancer detector.** Frozen Virchow tile embeddings + a small attention-MIL aggregator (Agata) reach **0.949 specimen-level AUC across 17 tissues on 23,408 slides** and **0.937 AUC on 7 rare cancer types**, vs Phikon 0.930 and CTransPath 0.904 (all pairwise differences p << 0.001, DeLong + Holm). Internal-vs-external AUC drop is only **-0.006**.
- **Strongest evidence: stain-shift robustness.** On the unnormalized variant of NCT-CRC-HE-100K (a deliberate OOD stain stress test), Virchow's weighted F1 drops **only -0.005** vs the normalized variant, while **CTransPath drops -0.118** and Phikon **-0.071** — by far the cleanest single result in the paper.

## Motivation

Computational pathology has lagged general computer vision on the scaling-law axis. Most academic pathology foundation models are trained on **TCGA (~29k WSIs, 32 cancer types)** with backbones under 100M parameters — and their downstream gains plateau quickly. Yet the clinically valuable tasks live in the long tail: detecting cancer in *rare* tissues, predicting molecular biomarkers (MSI, FGFR3, EGFR) directly from H&E to bypass costly IHC/sequencing, and surviving stain/scanner shift between institutions. Those need a model that has seen the long tail of cellular morphologies, stains, and tissue architectures that 30k slides cannot cover. Virchow is the pathology version of the JFT-300M experiment: take a faithful natural-image DINOv2 recipe, swap in **1.5M proprietary MSKCC WSIs** and a ViT-H, and ask whether scaling alone produces a generalist H&E encoder.

## Core Innovation

The contribution is **not algorithmic** — it is the empirical demonstration that:

- A **faithful natural-image DINOv2 recipe** (cross-view DINO CLS loss + iBOT masked-patch loss + EMA teacher + 131,072 prototypes) transfers to histopathology with **no pathology-specific design changes**, when you scale both data and capacity.
- **Tile embedding = `concat(CLS, mean(patch_tokens))`** -> 2,560-D (1,280x2), which diverges from Phikon's CLS-only and CTransPath's mean-only conventions and is shown empirically to help.
- Two DINOv2 hyperparameter tweaks for scale: LR warmup over **495,000 iterations** (vs 100k default) and teacher temperature schedule **0.04 -> 0.07 over 186,000 iterations**.
- A **frozen-encoder + Agata attention-MIL aggregator** workflow for slide/specimen-level tasks: learned query Q with K = GELU(W1^T x + b1) (256-D), V = GELU(W2^T K + b2) (512-D), backprop restricted to the slide with the highest cancer probability per specimen for memory reasons.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Virchow is the first million-scale pathology foundation model | Corpus stats: 1.49M WSIs / 119,629 patients vs ~29k TCGA in prior FMs | MSKCC corpus | ⭐⭐⭐ |
| C2 | Pan-cancer detection across 17 tissues at 0.949 specimen-level AUC | Fig. 2b ROC + AUC table; DeLong p << 0.001 vs baselines | MSKCC + external (23,408 slides) | ⭐⭐⭐ |
| C3 | Robust on 7 rare cancer types (0.937 AUC) | Fig. 2a per-type AUCs (brain 0.954, cervix 0.875, bone 0.841); Fig. 2c sensitivity-at-95%-specificity | 8,798 rare-cancer slides | ⭐⭐ |
| C4 | Robust to external/OOD deployment | Internal-vs-External AUC drop only 0.006-0.016 | 7,467 external slides | ⭐⭐ |
| C5 | SOTA on biomarker prediction (ColonMSI, BladderFGFR, LungEGFR) | Table 1 AUC + 1,000-bootstrap 95% CIs | MSK-IMPACT cohorts | ⭐⭐ |
| C6 | SOTA on tile-level linear-probing benchmarks | Top-1 on 5 of 6 datasets (PanMSK, CRC, CRC-nonorm, PCam, MHIST); WILDS tied | 6 datasets, 6 baselines | ⭐⭐⭐ |
| C7 | Robust to stain variation without normalization | CRC-nonorm: Virchow -0.005 F1 vs CTransPath -0.118, Phikon -0.071 | NCT-CRC-HE-100K-NONORM | ⭐⭐⭐ |
| C8 | Semantically meaningful embeddings (cell-type discovery) | Fig. 3d PCA on CoNSeP qualitatively isolates malignant epithelium / inflammatory cells | CoNSeP | ⭐ |
| C9 | "Scaling laws apply to pathology FMs" | Monotonic gains vs Phikon (~300M, ~6k WSIs) and CTransPath (~28M, ~32k WSIs) | All benchmarks | ⭐⭐ |

**Honest read.** **C2, C6, and C7** are the most defensible: confidence intervals, statistical testing, multi-dataset OOD evaluation, and a stain-normalization stress test that makes the encoder's invariance properties legible. **C5 is partially oversold** — Bladder (+0.016 vs Phikon, +0.020 vs CTransPath) and Lung (+0.032 vs Phikon, +0.046 vs CTransPath) gains are clean, but **ColonMSI Δ = +0.002 vs CTransPath with heavily overlapping CIs** is effectively noise. **C9 is the structural weakness.** Virchow simultaneously changes three variables — data (1.5M MSKCC vs 6k-32k), architecture (ViT-H vs ViT-S/B), and SSL recipe (DINOv2 vs iBOT/MoCov2) — so the paper cannot decompose how much of the lift is from data vs capacity vs algorithm. **Most-missing experiments:** no data-scale ablation (100k / 500k / 1.5M), no model-size ablation at fixed data, **no UNI or Prov-GigaPath comparison** despite both being contemporaneous and obvious head-to-heads (UNI is conspicuously absent), no inter-pathologist baseline on pan-cancer, no compute-budget reporting.

## Method & Architecture

![Virchow training and inference pipeline: H&E WSI to 224x224 tiles to ViT-H DINOv2 to tile embeddings to Agata MIL aggregator to slide-level task head](/assets/images/paper/virchow/fig_p002_41.png)
*Figure 1: Virchow pipeline — H&E WSI -> 224x224 foreground tiles -> ViT-H/14 trained with DINOv2 -> 2,560-D tile embedding (concat of CLS and mean patch tokens) -> Agata attention-MIL aggregator -> specimen-level prediction.*

**Pretraining at a glance.**

1. **Tile extraction.** Each WSI is downsampled 16x; foreground is detected with HSV thresholds (H in [90,180], S in [8,255], V in [103,255]); non-overlapping 224x224 tiles with >=25% tissue area are kept.
2. **DINOv2 objective.** Per tile, 2 global crops (224) + 8 local crops (98). Student-teacher EMA setup runs two losses simultaneously: (i) **DINO CLS loss** — student CLS on one global crop must match the teacher's CLS on the *opposite* global view (cross-view); (ii) **iBOT masked-patch loss** — random subset of student patches are masked and must predict the teacher's unmasked patch tokens at those positions. Local crops are fed only to the student.
3. **Hyperparameters.** AdamW (beta1=0.9, beta2=0.999), float16, **131,072 prototypes** (much larger than DINOv2 default to match capacity), LR warmup over **495,000 iterations**, teacher temperature **0.04 -> 0.07 over 186,000 iterations**. Each minibatch samples one WSI per GPU x 256 tiles/WSI.
4. **Tile embedding.** `concat(CLS_token, mean(patch_tokens))` -> 2,560-D vector (1,280 + 1,280). Distinct from Phikon (CLS-only) and CTransPath (mean-only).
5. **Slide adapter.** Agata attention-MIL aggregator over frozen tile embeddings, trained with cross-entropy for 25 epochs, AdamW lr=3e-4. Backprop is restricted to the slide with the highest cancer probability per specimen due to memory.

![Dataset cohort overview: 119,629 patients to 1.49M H&E WSIs across 17 tissue types](/assets/images/paper/virchow/fig_p002_36.png)
*Figure 2: MSKCC pretraining cohort — 1,488,550 H&E WSIs from 119,629 patients / 208,815 cases / 392,268 specimens across 17 tissue groups (breast 24.9%, skin 18.4%, lymph node 16.6%, lung 6.1%, ...). 63% biopsy / 37% resection, scanned at 20x (0.5 mpp) on Leica scanners. Single-center, single-vendor.*

## Experimental Results

### Pan-cancer detection — specimen-level AUC

![Pan-cancer detection: AUC bars, ROC curves, and ID-vs-OOD breakdown across 17 tissues](/assets/images/paper/virchow/page_004.png)
*Figure 3: Fig. 2 of the paper — pan-cancer detection across 17 tissues. Virchow achieves 0.949 overall AUC, 0.937 on rare cancer types, with only a -0.006 internal-to-external AUC drop.*

| Model | Overall | Internal (MSKCC) | External | Rare cancers | Brain | Cervix | Bone |
|---|---|---|---|---|---|---|---|
| CTransPath | 0.904 | 0.896 | 0.880 | — | 0.795 | 0.753 | 0.728 |
| Phikon | 0.930 | 0.920 | 0.912 | — | 0.898 | 0.810 | 0.822 |
| **Virchow** | **0.949** | **0.938** | **0.932** | **0.937** | **0.954** | **0.875** | **0.841** |

All overall pairwise differences are significant at p << 0.001 (DeLong's test with Holm correction). Caveat: the "external" set is composed of **MSKCC consultation submissions** — slides sent to MSKCC for a second read — not a true foreign-hospital held-out set.

### Biomarker prediction — case-level AUC (95% CI)

![Biomarker prediction Table 1: ColonMSI, BladderFGFR, LungEGFR AUCs with 95% CIs](/assets/images/paper/virchow/page_007.png)
*Figure 4: Page containing Table 1 — biomarker prediction AUCs with bootstrap 95% CIs on MSK-IMPACT confirmed cohorts.*

| Backbone | ColonMSI | BladderFGFR | LungEGFR |
|---|---|---|---|
| CTransPath | 0.970 (0.946, 0.989) | 0.882 (0.824, 0.930) | 0.807 (0.758, 0.852) |
| Phikon | 0.957 (0.905, 0.992) | 0.886 (0.838, 0.930) | 0.821 (0.771, 0.864) |
| **Virchow** | **0.972 (0.950, 0.989)** | **0.902 (0.862, 0.941)** | **0.853 (0.804, 0.891)** |

The LungEGFR gain (+0.046 vs CTransPath, +0.032 vs Phikon) is the most substantial. **ColonMSI's +0.002 over CTransPath is within noise** — the 95% CIs overlap almost entirely — so the abstract's "SOTA on three biomarkers" framing should be read as "SOTA on two and tied on one."

### Tile-level linear probing — weighted F1

![Tile-level linear-probing results: per-task weighted F1 across PanMSK, CRC, CRC-nonorm, WILDS, PCam, MHIST](/assets/images/paper/virchow/page_006.png)
*Figure 5: Fig. 3 of the paper — tile-level linear probing on 6 benchmarks (frozen embeddings, SGD, cosine LR 0.01->0, batch 4,096, 12,500 iterations), plus the CoNSeP PCA cell-type discovery panel.*

| Dataset | NatImg (1.1B) | PLIP | CTransPath | DINO p=8 | Phikon | **Virchow** |
|---|---|---|---|---|---|---|
| PanMSK (ID, 17-tissue) | 0.883 | 0.862 | 0.897 | 0.903 | 0.923 | **0.950** |
| CRC (stain-normalized) | 0.952 | 0.944 | 0.962 | 0.959 | 0.959 | **0.973** |
| **CRC (no norm, OOD)** | 0.927 | 0.806 | 0.844 | 0.949 | 0.888 | **0.968** |
| WILDS (OOD hospital) | 0.934 | 0.867 | 0.947 | 0.957 | **0.971** | 0.970 |
| PCam | 0.886 | 0.873 | 0.872 | 0.918 | 0.905 | **0.933** |
| MHIST | 0.827 | 0.801 | 0.816 | 0.769 | 0.796 | **0.835** |

**The CRC-nonorm row is the paper's most compelling single number.** Under the stain distribution shift, Virchow's weighted F1 falls **only -0.005** vs the normalized variant; CTransPath collapses by **-0.118** and Phikon by **-0.071**. This is the clearest evidence that scale + DINOv2 buys genuine invariance to a routine clinical confounder. WILDS is essentially a statistical tie with Phikon (Δ = -0.001).

### Qualitative — semantic part discovery

![CoNSeP PCA visualization: Virchow tile features approximately segment malignant epithelium, miscellaneous, and inflammatory cells](/assets/images/paper/virchow/fig_p006_01.png)
*Figure 6: Fig. 3d — PCA over Virchow tile features on CoNSeP. First PC isolates malignant epithelium; second PC isolates miscellaneous vs inflammatory cells. Qualitative only — no segmentation IoU/F1 is reported.*

## Limitations

**Acknowledged by the authors:**

- Single-center pretraining (MSKCC only) and single-vendor scanners (Leica only).
- Tile-level (not slide-level) embeddings — an aggregator is always required for slide tasks.
- No aggregator architecture exploration.
- No claim of clinical readiness; stratified prospective validation still needed.
- SSL algorithm choice (DINOv2 vs MAE vs MoCo) left as open question.
- Domain-specific augmentation (color, stain, cropping) under-explored.

**Not addressed (independent observations):**

- **No comparison against contemporaneous pathology FMs of similar scale.** UNI (ViT-L, ~100k WSIs, *Nat Med* 2024) is the obvious head-to-head and is conspicuously absent. Prov-GigaPath (ViT-G + tile + slide encoder, 1.3B params, ~1.4B tiles, *Nature* 2024) likewise.
- **No data-scaling curve.** The central scientific claim is "scale wins," yet there is no 30k -> 100k -> 500k -> 1.5M ablation. The data-vs-capacity-vs-algorithm contributions are not decomposed.
- **No model-size scan** at fixed data (ViT-S/B/L/H).
- **"External" validation is referral-biased.** The external set is MSKCC consultation submissions — cases the originating institution couldn't resolve and sent to MSKCC. True external validation would use TCGA, CPTAC, or a foreign hospital with no MSKCC referral relationship.
- **ColonMSI Δ = +0.002** is not statistically meaningful, but the paper frames all three biomarker tasks as wins without explicit pairwise testing on biomarker AUCs.
- **No failure-mode analysis** on cervix/bone cancers (<0.9 AUC for all three models — either evaluation-set artifact or genuinely hard morphology).
- **Cell-type discovery (Fig. 3d) is qualitative only** — no segmentation IoU/F1 to back the "DINOv2-style part discovery" claim.
- **No release** of pretraining data, training code, or weights in v5 (Virchow1 weights were later gated-released on HuggingFace).

## Why It Matters for Medical AI

Virchow1 is the closest thing computational pathology has to a JFT-scale encoder experiment, and three of its results have direct clinical translation value:

- **Pan-cancer detection at 0.949 AUC over 17 tissues** is a credible foundation for a screening / triage assistant — a single encoder feeding a small attention head, rather than a separate model per cancer type.
- **Stain-shift robustness without explicit normalization** (-0.005 F1 vs CTransPath's -0.118 on CRC-nonorm) is the closest thing the paper offers to deployment-grade evidence: scanners, stains, and lab protocols vary across institutions, and a model that does not require Macenko/Vahadane normalization at inference is materially easier to ship.
- **Rare-cancer coverage (0.937 AUC across 7 rare types)** is the long-tail story that TCGA-scale FMs cannot tell. Bone (0.841) and cervix (0.875) remain weak even for Virchow, so this is a step, not a solved problem.

The caveats matter, though. The "external" validation is referral-biased toward MSKCC, no UNI/Prov-GigaPath baselines are reported, the scaling-laws claim is not decomposed, and ColonMSI's headline win is statistical noise. For a clinical reader: take pan-cancer detection and stain robustness as well-evidenced; treat biomarker SOTA and rare-cancer generalization as promising but in need of multi-institutional prospective validation before deployment.

## References

- **Paper.** Vorontsov, Bozkurt, Casson, et al. *Virchow: A Million-Slide Digital Pathology Foundation Model.* arXiv:2309.07778 (v5, 18 Jan 2024). Journal version: *Nature Medicine* (2024).
- **Follow-on (not this paper).** *Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology.* arXiv:2408.00738 (~3.1M slides, DINOv2 with register tokens).
- **Related encoders cited above.**
  - UNI: Chen et al., *Towards a General-Purpose Foundation Model for Computational Pathology*, *Nat Med* (2024).
  - Prov-GigaPath: Xu et al., *A Whole-Slide Foundation Model for Digital Pathology from Real-World Data*, *Nature* (2024).
  - CTransPath: Wang et al., *Transformer-based unsupervised contrastive learning for histopathological image classification*, *Med Image Anal* (2022).
  - Phikon: Filiot et al., *Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling* (2023).
  - DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023).
- **Datasets.** NCT-CRC-HE-100K (Kather et al.), WILDS-Camelyon17, PCam, MHIST, CoNSeP; MSK-IMPACT cohorts (ColonMSI, BladderFGFR, LungEGFR) are MSKCC-internal.
- **Model weights.** Virchow1 gated release on HuggingFace (`paige-ai/Virchow`); not released in v5 of the paper.
