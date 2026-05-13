---
title: "Stem: Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images"
excerpt: "First conditional-diffusion formulation of H&E→ST as p(X|V), with a DiT denoiser modulated by pooled CONCH+UNI embeddings; on HER2ST B1 it drives RVD from 0.6025 (BLEEP) down to 0.0693 while matching or beating PCC on the authors' four datasets."
categories:
  - Paper
tags:
  - Stem
  - Diffusion-Model
  - Spatial-Transcriptomics
  - DiT
  - CONCH
  - UNI
  - Computational-Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- Stem is the **first generative-modeling approach** to H&E→spatial-transcriptomics prediction, recasting the task as learning the conditional `p_gene(X | V)` rather than a deterministic regression `X ≈ f(V)`. Earlier methods (ST-Net, HisToGene, Hist2ST, BLEEP, EGN, TRIPLEX, M2ORT) all collapse one-to-many H&E→expression maps to a single averaged profile.
- A 12-block **DiT** denoises a length-C sequence of gene-count tokens (count MLP + learnable gene-type embedding) under adaLN-Zero modulation from a single pooled image vector formed by concatenating **CONCH (512-d) + UNI (1024-d)** patch embeddings — no spot-wise retrieval, no patch-token cross-attention.
- Headline gain is on the authors' newly introduced **Relative Variation Distance (RVD)** metric: **HER2ST B1 HMHVG RVD 0.6025 (BLEEP) → 0.0693 (Stem)**, with PCC-200 0.4257 vs. 0.3165 (TRIPLEX) on Kidney AKI slide 20-0038. The strongest evidence that the diffusion model itself (not just the foundation-model encoders) drives the gain is Table 9: a CONCH+UNI kNN baseline reaches only PCC-200 0.2547 / RVD 0.3269, vs. Stem's 0.4257 / 0.0751.

## Motivation

Spatial transcriptomics couples gene expression to tissue location and is central to tumor-microenvironment studies, but Visium-class assays are expensive and slow. H&E slides, by contrast, are ubiquitous, so a reliable in-silico H&E→ST model is the obvious dream. Every prior patch-level predictor (ST-Net, HisToGene, Hist2ST, BLEEP, EGN, TRIPLEX, M2ORT) treats the mapping as a deterministic regression. The authors argue this is doubly wrong:

1. The map is intrinsically **one-to-many** — cells of the same type can be in different transcriptomic states even when their H&E patches look identical.
2. The dominant evaluation metric (PCC) **rewards prediction of the mean** and penalizes the very biological variation that regression-based models destroy by averaging.

Stem's pitch is therefore both methodological (use a conditional generative model that natively expresses one-to-many) and evaluative (introduce RVD to detect mode-collapsed predictors).

## Core Innovation

- **Generative reframing.** Learn `p_gene(X | V)` over the C-dim gene-count vector `X` given an H&E patch `V`. If the conditional collapses to a delta, the regression case is recovered.
- **DDPM in gene-expression space, not pixel space.** Forward noise schedule `q(X_t | X_0, V) = N(sqrt(ᾱ_t) X_0, (1 − ᾱ_t) I)` is applied to the **gene vector**; the image is only ever read by the condition encoder.
- **Pooled bi-FM conditioning.** A 224×224 patch is encoded by CONCH (CoCa, 0.1B) and UNI (DINOv2, 0.3B); each is attention-pooled, their 1536-d concatenation is projected by an MLP to D=384, and the resulting `c_hist` modulates every DiT block via adaLN-Zero. No image-patch cross-attention; the entire image conditioning is one vector.
- **Gene tokenization without an image patcher.** For gene i: `h_i = MLP_{1→D}(x_i) + E_type[i]` (count MLP + learnable gene-type embedding). DiT's image-only patch/unpatch modules are stripped.
- **20-sample inference, sample mean as point estimate.** Because the model is generative, inference produces a distribution; the predicted expression is the mean over 20 samples (median and mode are slightly better for some metrics — see Table 7).

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|-------|----------|----------|----------|
| C1 | First generative-modeling approach to H&E→ST prediction. | Related-work survey covers all major patch-level baselines; no prior generative counterexample found. | — | ⭐⭐⭐ |
| C2 | Stem achieves SOTA on standard metrics (MSE, MAE, PCC) across multiple datasets. | Tables 1, 2, 10, 11. | Kidney, HER2ST, PRAD, Mouse brain | ⭐⭐ — true in aggregate, but **Kidney HVG MSE 1.7529 is worse than TRIPLEX 1.4500**; **HER2ST DEG PCC-1000 0.5423 is worse than TRIPLEX 0.5575**; **PRAD PCC-10 0.6103 is worse than TRIPLEX 0.6173**; TRIPLEX is missing on mouse brain (OOM). The abstract's blanket "outperforms all existing approaches" overstates this. |
| C3 | Stem captures biological heterogeneity better, as measured by RVD. | Tables 1, 2, 6, 10, 11 — RVD margins are largest (often 5–10×). | All 4 datasets | ⭐⭐⭐ on its own metric, but **RVD is introduced in this same paper**; no external validation that lower RVD implies better downstream biological utility. |
| C4 | Stem improves computational efficiency over BLEEP/EGN by eliminating reference-set retrieval. | Stated in the intro; no wall-clock or memory comparison. Inference still runs 20× through a 12-block DiT plus CONCH+UNI. | — | ⭐ — qualitative argument only. |
| C5 | Predictions are biologically meaningful — Leiden clustering of Stem's predicted 296-DEG vectors recovers invasive-cancer and breast-gland regions on HER2ST B1. | Figure 3; Appendix E marker-gene heatmaps (CCL19, TRAC, IGHA1, GPX3, RAB11FIP1, COL4A1). | HER2ST (B1, G2) | ⭐⭐ — visually compelling, but only 1–2 slides shown, no ARI/AMI vs. pathologist, no comparison to TRIPLEX's clustering. |
| C6 | The diffusion model — not just the foundation-model encoder — drives the gain. | **Table 9.** CONCH+UNI kNN retrieval gives PCC-200 0.2547 / RVD 0.3269; full Stem gives 0.4257 / 0.0751. ResNet18-CL (TRIPLEX's encoder) only 0.2297 / 0.4064. | Kidney HMHVG | ⭐⭐⭐ — the strongest ablation in the paper. |
| C7 | CONCH + UNI is better than either alone. | Table 3 (PCC), Table 8 (PCC, MSE). | Kidney HMHVG | ⭐⭐ — supported on PCC, but **CONCH alone has the best RVD (0.0625) in Table 3** — variance-preservation is partially inherited from CONCH itself. |
| C8 | Image augmentation significantly improves performance. | Table 4 — PCC-200 climbs monotonically from 0.3298 (2:1) to 0.3859 (1:4). | Kidney HMHVG | ⭐⭐ — PCC scaling is clean; **RVD is non-monotone (0.1058 → 0.1391 → 0.0813 → 0.1316)**; no significance test. |
| C9 | Larger pathology FMs do not necessarily help. | Table 8 — Virchow-2 (0.6B) and H-Optimus-0 (1.1B) underperform UNI (0.3B); all underperform CONCH+UNI. | Kidney HMHVG | ⭐⭐ |

**Honest read.** C1 and C6 are the load-bearing claims and are well supported — the conditional-diffusion reframing is genuinely novel within the patch-level family, and Table 9 isolates the diffusion model as the source of the gain over a CONCH+UNI kNN baseline that uses the same encoders. C3 is the headline marketing claim, and the margins are massive — but **RVD is the authors' own metric**, so this partially reduces to "we win on the metric we invented." Critical gaps: (1) **no variance / confidence intervals on any number** despite 20-sample inference enabling free CIs; (2) **single holdout slide per dataset** for headline tables, with no leave-one-patient-out — HER2ST B1 is not patient-disjoint from training data; (3) **no wall-clock inference comparison** vs. BLEEP/EGN to substantiate the efficiency claim. STPath and PathGen are operating on different axes (slide-level foundation model and CLIP-style pathology FM respectively) and should not be positioned as direct comparators to Stem.

## Method & Architecture

![Stem overview](/assets/images/paper/stem/page_003.png)
*Figure 1: Stem pipeline. H&E patches are encoded by CONCH + UNI and pooled into a single conditioning vector c_hist; gene counts are diffused in expression space and a DiT denoiser predicts the noise conditioned on c_hist. At inference, 20 samples are drawn and the sample mean is reported.*

The forward process is standard DDPM in **gene-expression space**: `q(X_t | X_0, V) = N(sqrt(ᾱ_t) X_0, (1 − ᾱ_t) I)` with linear schedule `β_t = (t/T) β_max + (1 − t/T) β_min` and `ᾱ_t = Π α_s`. The reverse process is `p_θ(X_{t−1} | X_t, V) = N(μ_θ(X_t, V, t), σ_t² I)` parameterized via ε-prediction, giving the training objective

$$\mathcal{L}_\epsilon(\theta) = \mathbb{E}_{t, X_t}\, \lVert \epsilon_\theta(X_t, V, t) - \epsilon_t \rVert_2^2.$$

![Stem DiT block](/assets/images/paper/stem/page_006.png)
*Figure 2: DiT block detail. Gene count is embedded by a 2-layer SiLU MLP and added to a learnable gene-type embedding; the sinusoid time embedding plus the pooled image embedding c_hist drive adaLN-Zero modulation in every block. Output head: adaLN + linear → R¹ per gene token (predicted ε for that gene's noisy count).*

Key knobs: D=384, 12 DiT blocks, 6 attention heads; AdamW lr 1e-4, batch 256, 250k iterations, EMA 0.9999; 7 distortion-free augmentations (flips, 90/180/270° rotations, transpose, transverse) with augmentation ratio up to 1:4.

## Experimental Results

### Kidney Visium (holdout slide 20-0038, AKI)

| Gene set | Model | PCC-10↑ | PCC-50↑ | PCC-200↑ | MAE↓ | MSE↓ | RVD↓ |
|---|---|---|---|---|---|---|---|
| HMHVG | HisToGene | 0.4294 | 0.3503 | 0.0905 | 0.9298 | 1.4105 | 0.9962 |
| HMHVG | BLEEP | 0.4998 | 0.4221 | 0.3143 | 0.9451 | 1.5261 | 0.2170 |
| HMHVG | TRIPLEX | 0.4654 | 0.4105 | 0.3165 | **0.8969** | **1.3015** | 0.5871 |
| HMHVG | **Stem** | **0.5893** | **0.5332** | **0.4257** | 0.8792 | 1.3513 | **0.0751** |
| HVG | HisToGene | 0.4237 | 0.3296 | 0.0774 | 0.9776 | 1.5609 | 0.9965 |
| HVG | BLEEP | 0.4902 | 0.3953 | 0.2474 | 0.9931 | 1.7658 | 0.3293 |
| HVG | TRIPLEX | 0.4621 | 0.3997 | 0.2726 | 0.9962 | **1.4500** | 0.6984 |
| HVG | **Stem** | **0.5366** | **0.4699** | **0.3047** | **0.9763** | 1.7529 | **0.1325** |

Stem wins PCC at every k and RVD by a wide margin, but **on HVG MSE is 1.7529 vs. TRIPLEX 1.4500** — the model trades MSE for variance preservation.

### HER2ST (holdout B1)

| Gene set | Model | PCC-10↑ | PCC-50↑ | PCC-300↑ | MAE↓ | MSE↓ | RVD↓ |
|---|---|---|---|---|---|---|---|
| HMHVG | HisToGene | 0.6812 | 0.6345 | 0.5250 | 0.9367 | 1.3468 | 10.3407 |
| HMHVG | BLEEP | 0.7727 | 0.7141 | 0.5652 | 0.8328 | 1.2428 | 0.6025 |
| HMHVG | TRIPLEX | 0.7907 | 0.7394 | 0.5766 | 0.9311 | 1.3456 | 0.6428 |
| HMHVG | **Stem** | **0.8298** | **0.7726** | **0.5984** | **0.7547** | **1.0742** | **0.0693** |
| DEG | HisToGene | 0.6816 | 0.6369 | 0.5112 | 0.8791 | 1.2627 | 9.7057 |
| DEG | BLEEP | 0.7711 | 0.7188 | 0.5518 | 0.7590 | 1.1297 | 0.6383 |
| DEG | TRIPLEX | 0.7919 | 0.7432 | 0.5709 | 0.8768 | 1.2887 | 0.6533 |
| DEG | **Stem** | **0.8365** | **0.7651** | **0.5748** | **0.6881** | **0.9631** | **0.0862** |

This is the cleanest sweep: Stem wins every column on both gene panels. The RVD drop from 0.6025 (BLEEP) to 0.0693 is the headline figure of the paper.

![HER2ST HMHVG variance curves](/assets/images/paper/stem/page_016.png)
*Figure 3: Sorted per-gene variance on HER2ST HMHVG. The blue curve is ground-truth variance; orange dots are each model's predicted variance. HisToGene/BLEEP/TRIPLEX flatten to near-zero variance (mode collapse); Stem tracks the ground-truth curve.*

### Prostate PRAD and mouse brain

- **PRAD HMHVG-200**: Stem PCC-200 0.3832 vs. TRIPLEX 0.3601, BLEEP 0.3158, HisToGene 0.2235; RVD 0.1975 vs. TRIPLEX 0.7954. **TRIPLEX wins PCC-10 (0.6173 vs. Stem 0.6103) and MSE (1.4819 vs. 1.4873)** by a hair — the only headline metrics where Stem is not best on a held-out dataset.
- **Mouse brain HMHVG-200**: Stem PCC-200 0.2791 vs. BLEEP 0.1555 vs. HisToGene −0.0008. TRIPLEX missing (OOM, not Stem's fault, but the comparison is incomplete).

### Ablations worth highlighting

**Table 9 — encoders alone vs. Stem (the most informative ablation):**

| Encoder / Model | PCC-200↑ | RVD↓ |
|---|---|---|
| ResNet18-CL (TRIPLEX encoder), kNN | 0.2297 | 0.4064 |
| CONCH+UNI, kNN | 0.2547 | 0.3269 |
| **Stem (CONCH+UNI + DiT)** | **0.4257** | **0.0751** |

The diffusion model contributes **+0.17 PCC-200 and a 4× RVD improvement** over a kNN baseline that uses the same encoders — strong evidence that the modeling component, not the encoder, drives the gain.

**Table 4 — augmentation ratio (Kidney HMHVG-200):**

| Ratio | PCC-200 | RVD |
|---|---|---|
| 2:1 | 0.3298 | 0.1058 |
| 1:1 | 0.3576 | 0.1391 |
| 1:2 | 0.3712 | 0.0813 |
| 1:4 | **0.3859** | 0.1316 |

PCC scales cleanly with augmentation; RVD is non-monotone, so "more augmentation always helps" is not the right summary.

**Table 6 — 1000-DEG scalability on HER2ST:** Stem PCC-1000 0.5423 < TRIPLEX 0.5575 (Stem **loses** at PCC-1000) but retains the RVD lead (0.1208 vs. TRIPLEX 0.6632 / BLEEP 0.5220) and wins MAE/MSE.

### Qualitative finding

Leiden clustering of Stem's predicted 296-DEG vectors on HER2ST B1 recovers two clusters matching the pathologist's invasive-cancer and breast-gland regions; the same pipeline on BLEEP's predictions merges them. The result is shown only against BLEEP — no clustering comparison to TRIPLEX, which is otherwise the strongest baseline.

## Limitations

**Acknowledged by the authors:**
- PCC alone is misleading on this task — motivation for RVD.
- Larger pathology FMs do not necessarily help (Virchow-2, H-Optimus-0 underperform UNI).
- Sample mean is not optimal for every gene; some are better summarized by median or mode.
- TRIPLEX could not be run on mouse-brain dataset due to GPU memory.

**Not addressed:**
- **No variance / confidence intervals on any metric**, despite 20-sample inference enabling free CIs.
- **Per-dataset training, not a foundation model.** Unlike STPath, Stem must be re-trained for each dataset / gene set; no cross-organ transfer test.
- **No cross-validation / patient-level holdout.** Every headline number is one holdout slide. HER2ST B1 is from patient B, whose other slides are in training.
- **RVD is introduced in this same paper.** No independent biological validation that low RVD implies better downstream utility (cell-type deconvolution, prognostic signal, etc.).
- **Compute efficiency claim is qualitative.** No wall-clock vs. BLEEP/EGN despite the abstract advertising the scaling benefit.
- **Gene-vector size scaling.** The C-dim gene vector becomes a length-C token sequence with O(C²) attention; the 1000-DEG ablation already loses PCC-1000 to TRIPLEX. Whether this scales to full transcriptomes (~20k genes) is unclear.
- **No comparison to STPath, OmiCLIP/Loki, or any post-2024 generative-ST baseline** despite the paper's Jan 2025 submission. Note: STPath (slide-level foundation model with masked gene-expression regression) and PathGen (CLIP-style pathology FM) operate on different axes and are not direct comparators — STPath sacrifices per-dataset accuracy for cross-organ generalization, while PathGen is an encoder, not a generative ST model.

## Why It Matters for Medical AI

The promise is "in-silico ST" — running gene-expression inference on archival H&E slides without wet-lab Visium runs. Stem's contribution to that program is two-fold. First, it shows that the H&E→ST map should be modeled generatively, not as deterministic regression — and it backs this up with Table 9, where the diffusion model adds +0.17 PCC-200 over a kNN baseline using the same encoders. Second, it surfaces a calibration problem in the field: PCC rewards mean-prediction, so any deterministic regressor with strong encoders will look good on PCC while flattening genuine biological variation. RVD is the right kind of metric to expose this, even if its own validation as a biological-utility proxy is still future work. For downstream use cases that depend on within-tissue heterogeneity — TME deconvolution, regional biomarker discovery, spatial prognostic signal — that distinction is load-bearing.

## References

- Paper: Zhu, Zhu, Tao, Qiu. *Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images.* ICLR 2025. arXiv:2501.15598
- Code: <https://github.com/SichenZhu/Stem>
- DiT: Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
- CONCH: Lu et al. *A visual-language foundation model for computational pathology.* Nature Medicine 2024.
- UNI: Chen et al. *Towards a general-purpose foundation model for computational pathology.* Nature Medicine 2024.
- HEST-1K (data convention): Jaume et al., NeurIPS 2024.
- Baselines: ST-Net (He et al. 2020), HisToGene (Pang et al. 2021), Hist2ST (Zeng et al. 2022), BLEEP (Xie et al. 2024), EGN (Yang et al. 2023), TRIPLEX (Chung et al. 2024), M2ORT (Wang et al. 2024).
