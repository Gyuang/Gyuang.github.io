---
title: "H&Enium: Applying Foundation Models to Computational Pathology and Spatial Transcriptomics to Learn an Aligned Latent Space"
excerpt: "Single-cell H&E↔Xenium contrastive alignment with frozen UNI2 + CellPLM and two MLPs improves gene-expression PCC by +10–11% across three slides."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/h-and-enium-single-cell-contrastive-alignment-xenium/
tags:
  - H&Enium
  - Contrastive Alignment
  - Spatial Transcriptomics
  - Xenium
  - UNI2
  - CellPLM
  - BLEEPinput
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- H&Enium is the first **single-cell-resolution** image↔transcriptomics contrastive aligner — using Xenium nuclei as anchor points instead of the ~55 µm Visium spots that OmiCLIP, CarHE, and BLEEP operate on.
- The architecture is deliberately minimal: two **frozen** foundation models (UNI2 for H&E, CellPLM for gene panels) and just **two single-layer MLP projection heads** to a 128-d aligned space — trained with a soft-target contrastive loss.
- **Headline result:** gene-expression prediction PCC improves by **+10–11% relative** over the non-aligned UNI2 baseline on all three Xenium slides; out-of-sample PanNuke F1 improves by **+16–34% relative**. The proposed `BLEEPinput` soft target is statistically tied with plain CLIP in-distribution and only clearly separates from CLIP on one PanNuke subset.

## Motivation
The tumor microenvironment (TME) is fundamentally a single-cell, neighborhood-scale phenomenon, but the existing image↔gene contrastive frameworks — BLEEP, ST-Align, PathOmCLIP, OmiCLIP/Loki, CarHE — all align at the **Visium spot** level (≈55 µm, multiple cells per spot), so they can only emit a pseudo-bulk signal per region. Xenium changes the substrate: H&E, transcriptomics, and nucleus segmentation are co-registered at single-cell resolution. That makes a true per-cell contrastive alignment possible for the first time. The pitch is to train projection heads on Xenium and then deploy the image branch on archival H&E (PanNuke), recovering cell types from images alone — with alignment regularizing the H&E embedding by transcriptomic context.

## Core Innovation
Two ideas, both small in code and load-bearing in effect:

1. **Single-cell anchoring instead of spot anchoring.** Per Xenium cell, take a 224-px H&E patch centered on the nucleus (after a 1.33× Lanczos upsample so single cells fit), paired with the Xenium gene-count vector (C = 474 for Pancreas, 380 for Breast) and a 4-class PanNuke cell type derived from expression.
2. **`BLEEPinput` soft target.** BLEEP builds its soft target matrix `T` from the **post-projection aligned embeddings `A`**, which are essentially random at the start of training. `BLEEPinput` computes the same formula on the **pre-projection FM embeddings `Z`** — which are already meaningful out of UNI2 / CellPLM — to avoid target collapse. Everything else (loss form, temperature, normalization) is identical.

Only the two single-layer MLPs (linear → GELU → LayerNorm → dropout 0.3, output dim 128) are trained. Both FMs stay frozen.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | First single-cell-resolution image↔transcriptomics contrastive alignment. | Positional vs BLEEP/ST-Align/PathOmCLIP (all Visium-spot). No empirical head-to-head at single-cell scale. | ⭐⭐ |
| C2 | `BLEEPinput` outperforms CLIP and BLEEP targets. | In-distribution: tied with CLIP to 3 decimals on F1/PCC/RVD across all three slides. Out-of-sample: BLEEPinput wins on all three PanNuke subsets, CLIP regresses on BreastIDC (F1 0.498 → 0.416, ±0.22 std). Original BLEEP underperforms non-aligned baseline by 7–13% F1. | ⭐⭐ |
| C3 | Alignment improves PanNuke cell-type F1 by >16% relative. | Pancreas 0.224 → 0.264 (+17.9%); BreastIDC 0.498 → 0.580 (+16.6%); BreastILC 0.356 → 0.476 (+33.7%). | ⭐⭐ |
| C4 | Alignment improves gene-expression PCC by >10% relative on all three Xenium slides. | Pancreas +11.1%, BreastIDC +10.5%, BreastILC +10.8%. RVD improves in lockstep. CLIP gives the same gain. | ⭐⭐⭐ |
| C5 | UNI2 outperforms CONCH at single-cell resolution. | F1 0.655 vs 0.604 (Pancreas), 0.743 vs 0.700 (BreastIDC), 0.522 vs 0.512 (BreastILC). | ⭐⭐ |
| C6 | FMs beat morphological + majority baselines for single-cell typing. | UNI2 F1 0.522–0.743 vs morphological 0.39–0.48 vs majority 0.13–0.22. Hand-crafted geometric baseline only — no PanNuke-trained ResNet. | ⭐⭐⭐ |
| C7 | CellPLM preferred over scGPT. | Driven by Pancreas (+5% F1); Breast is a tie. Raw expression beats both FMs on Pancreas (0.908 vs 0.860). | ⭐⭐ |
| C8 | Method generalizes "consistently across samples." | 3 slides (1 pancreas + 2 breast), all from 10x demo data. No leave-one-cohort-out. | ⭐ |
| C10 | Foundation for single-cell H&E-only clinical analysis. | PanNuke F1 0.26–0.58 — direction supported, not yet clinically actionable; only 4 generic PanNuke classes. | ⭐⭐ |

The strongest experimental contribution is the **clean negative result** on BLEEP: its original post-projection soft target *underperforms* the non-aligned baseline by 7–13% F1 at single-cell scale on all three slides. The proposed fix (compute the target from pre-projection `Z` instead) is sensible and reproducible. But the framing of `BLEEPinput` as a *novel target that outperforms alternatives* is overclaimed — on in-distribution Xenium folds it is statistically indistinguishable from the plain CLIP identity target, and the cleanest CLIP-vs-BLEEPinput separation (PanNuke BreastIDC) looks more like a stability win than a fundamental loss-design win given the ±0.22 F1 std on the CLIP run.

## Method & Architecture

![H&Enium architecture overview](/assets/images/paper/henium/page_002.png)
*Figure 1: H&Enium architecture — frozen pathology FM (UNI2) and frozen transcriptomic FM (CellPLM) feed two trainable single-layer MLP projection heads (P_I, P_G) aligned via a soft-target contrastive loss at single-cell resolution.*

For each Xenium cell, image embedding `z_I = FM_I(I) ∈ R^{d_I}` from UNI2 and gene embedding `z_G = FM_G(G) ∈ R^{d_G}` from CellPLM are projected to a shared 128-d space via `P_I` and `P_G`. After L2-normalizing rows of `A_I, A_G ∈ R^{B×d_a}`, the cosine-similarity logits `S = cossim(A_G, A_I)` are matched against a target matrix `T` with a bidirectional soft cross-entropy:

$$\mathcal{L}_\text{contrast} = \lambda\,\mathrm{SoftCE}(S^\top, T^\top) + (1-\lambda)\,\mathrm{SoftCE}(S, T)$$

Three target variants are evaluated: the **CLIP** identity `T = I_B`; the **BLEEP** target `T = \text{softmax}_{τ'}(α·\text{cossim}(A_G, A_G) + (1−α)·\text{cossim}(A_I, A_I))` (on post-projection `A`); and the proposed **`BLEEPinput`** target computed identically but on the **pre-projection FM embeddings `Z`**. Training is AdamW, lr 1e-3, weight decay 1e-4, batch B = 64, up to 20 epochs, early-stop patience 5. Hold-out is 5-fold spatial cross-validation within each slide.

Downstream evaluation is intentionally non-end-to-end: balanced-class L2-penalized logistic regression for cell typing, Ridge regression (top-50 most-variable genes, PCA-64) for gene-expression prediction.

![UNI2 image UMAP on Pancreas](/assets/images/paper/henium/page_004.png)
*Figure 2A: UNI2 frozen image embeddings on Pancreas form overlapping clusters with cell types mixed — the image side has the most to gain from alignment.*

## Experimental Results

### Non-aligned baselines (Table 1)
UNI2 beats CONCH at single-cell resolution on every slide — non-obvious because CONCH usually wins at the patch level. At 224-px single-cell crops, UNI2's pure SSL pretraining wins.

| Slide | Image FM | Accuracy | BAC | F1 |
|---|---|---|---|---|
| Pancreas | **UNI2** | 0.686 ± 0.011 | 0.655 ± 0.025 | **0.655 ± 0.027** |
| Pancreas | CONCH | 0.631 ± 0.007 | 0.604 ± 0.026 | 0.604 ± 0.026 |
| BreastIDC | **UNI2** | 0.810 ± 0.060 | 0.785 ± 0.039 | **0.743 ± 0.052** |
| BreastILC | **UNI2** | 0.764 ± 0.089 | 0.586 ± 0.035 | **0.522 ± 0.022** |
| Pancreas | Morphological (16 feat.) | 0.458 | 0.436 | 0.427 |
| Pancreas | Majority | 0.354 | 0.250 | 0.130 |

### In-distribution Xenium folds (Table 2)
**`BLEEPinput` ≈ CLIP** to 3 decimals; original **BLEEP target hurts** by 7–13% F1 on all slides.

| Slide | Target | F1 | Δ vs non-aligned |
|---|---|---|---|
| Pancreas | Non-aligned (UNI2) | 0.655 ± 0.027 | — |
| Pancreas | CLIP | 0.667 ± 0.028 | +1.8% |
| Pancreas | BLEEP | 0.567 ± 0.035 | **−13.4%** |
| Pancreas | **BLEEPinput** | **0.668 ± 0.030** | +1.9% |
| BreastIDC | CLIP | 0.747 ± 0.051 | +0.6% |
| BreastIDC | BLEEP | 0.688 ± 0.045 | −7.4% |
| BreastIDC | **BLEEPinput** | **0.749 ± 0.051** | +0.8% |
| BreastILC | CLIP | 0.559 ± 0.015 | +7.0% |
| BreastILC | BLEEP | 0.475 ± 0.038 | −9.1% |
| BreastILC | **BLEEPinput** | **0.559 ± 0.020** | +7.0% |

### Out-of-sample PanNuke (Table 3) — the most informative result
`BLEEPinput` is the only target that improves on all three subsets; CLIP regresses sharply on BreastIDC.

| Slide | Target | F1 | Δ vs non-aligned |
|---|---|---|---|
| Pancreas | Non-aligned | 0.224 ± 0.023 | — |
| Pancreas | CLIP | 0.247 ± 0.050 | +10.3% |
| Pancreas | **BLEEPinput** | **0.264 ± 0.047** | **+17.9%** |
| BreastIDC | Non-aligned | 0.498 ± 0.033 | — |
| BreastIDC | CLIP | 0.416 ± 0.220 | **−16.5%** |
| BreastIDC | **BLEEPinput** | **0.580 ± 0.032** | **+16.6%** |
| BreastILC | Non-aligned | 0.356 ± 0.082 | — |
| BreastILC | CLIP | 0.490 ± 0.068 | +37.7% |
| BreastILC | **BLEEPinput** | 0.476 ± 0.074 | +33.7% |

### Gene-expression prediction (Table 4)
CLIP and `BLEEPinput` are statistically tied at +10–11% relative PCC; original BLEEP underperforms or no-ops. RVD improves in lockstep, so the PCC gain is not a mean-shift artifact.

| Slide | Target | PCC ↑ | RVD ↓ | Δ PCC vs non-aligned |
|---|---|---|---|---|
| Pancreas | Non-aligned | 0.361 ± 0.034 | 0.728 ± 0.017 | — |
| Pancreas | CLIP | 0.401 ± 0.032 | 0.660 ± 0.030 | +11.1% |
| Pancreas | **BLEEPinput** | **0.401 ± 0.032** | 0.662 ± 0.027 | +11.1% |
| BreastIDC | CLIP | 0.475 ± 0.009 | 0.543 ± 0.016 | +10.2% |
| BreastIDC | **BLEEPinput** | **0.476 ± 0.009** | 0.548 ± 0.013 | **+10.5%** |
| BreastILC | CLIP | 0.369 ± 0.012 | 0.673 ± 0.030 | +10.5% |
| BreastILC | **BLEEPinput** | **0.370 ± 0.011** | 0.674 ± 0.030 | **+10.8%** |

## Limitations
- **Only 3 slides total** (1 pancreas + 2 breast), all from public 10x demos. The "across samples" framing is across 3 biopsies, not across cohorts; no leave-one-slide-out is reported.
- **Severe class imbalance on Breast** (Neoplastic 60–78%, Epithelial <1% on BreastILC) makes accuracy misleading and drags F1 down through minority classes.
- **Training labels are derived from Xenium expression itself**, so the gene branch has direct access to label-generating signal (`a_G` F1 ≥ 0.86). The fair test is the image-only PanNuke path — where absolute F1 falls to 0.26–0.58.
- **PanNuke labels are pathologist-by-eye** with no molecular ground truth and class distributions that differ from Xenium training by 2–10×. "Alignment helps generalization" cannot be fully separated from "alignment regularizes against label-distribution shift."
- **No head-to-head against BLEEP / ST-Align / PathOmCLIP / OmiCLIP / CarHE / STPath** at the single-cell-classification task — only the non-aligned UNI2/CONCH baselines.
- **No ablation on `α`, `λ`, `τ'`, projection-head depth, `d_a = 128`, or batch size B = 64.** The single most interesting knob in `BLEEPinput` (`α`) is neither reported nor swept.
- **No spatial-context analysis.** All inputs are 224-px nucleus-centered patches — no neighborhood signal, even though the TME is fundamentally a neighborhood phenomenon.
- **Gene FMs run frozen on Xenium 474/380-gene panels**, well outside their whole-transcriptome scRNA-seq pretraining regime. Raw expression beats both CellPLM and scGPT on Pancreas (F1 0.908 vs 0.860), suggesting the gene-side FM is not load-bearing.
- **PanNuke 4-class schema** collapses CAF subtypes, M1/M2 macrophages, exhausted vs naive T cells into "Connective" / "Inflammatory" — not yet a clinically actionable readout.
- **No public code link** in the preprint.

## Why It Matters for Medical AI
The interesting move for medical AI is the *scale* at which the alignment happens. OmiCLIP and CarHE operate on Visium spots (pseudo-bulk), so they cannot directly answer "what cell type is this nucleus?" — they can only answer "what does this 55-µm region look like in expression space?" H&Enium shows that with Xenium as a training substrate, you can push that question down to per-cell resolution and then deploy the image branch on archival H&E (PanNuke) where no transcriptomics exists. The +16–34% relative F1 gain on PanNuke is real, but absolute F1 0.26–0.58 on a 4-class generic schema is far from a clinical readout. The mechanistic insight that *the soft target must be computed from a stable embedding space* (the `BLEEPinput` fix) is the most portable contribution — it likely transfers to any contrastive-with-soft-targets setup that gets stuck at the start of training.

## References
- Paper (bioRxiv preprint, 2025-07-26): [`10.1101/2025.07.22.665986`](https://doi.org/10.1101/2025.07.22.665986)
- Venue: ICML 2025 Workshop on Multi-modal Foundation Models and LLMs for Life Sciences
- License: CC-BY-NC 4.0
- Code: no public GitHub link in the preprint
- Foundation models: [UNI2](https://github.com/mahmoodlab/UNI), [CONCH](https://github.com/mahmoodlab/CONCH), [CellPLM](https://github.com/OmicsML/CellPLM), [scGPT](https://github.com/bowang-lab/scGPT)
- Related spot-level work: [BLEEP (Xie et al. 2023)](https://arxiv.org/abs/2306.01859), [OmiCLIP/Loki](https://www.nature.com/articles/s41592-025-02707-1), CarHE, PathOmCLIP, ST-Align, STPath
- External evaluation dataset: [PanNuke](https://jgamper.github.io/PanNukeDataset/)
