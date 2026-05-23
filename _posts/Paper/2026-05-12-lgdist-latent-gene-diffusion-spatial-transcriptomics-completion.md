---
title: "LGDiST: Latent Gene Diffusion for Spatial Transcriptomics Completion"
excerpt: "A reference-free latent DiT over Visium neighborhoods reports an 18% average MSE drop over SpaCKLE on 26 SpaRED datasets, plus up to 10% MSE / 188% PCC gains on six downstream histology-to-expression models — but the headline number lives entirely inside one boxplot and the largest PCC claim has no absolute baseline."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/lgdist-latent-gene-diffusion-spatial-transcriptomics-completion/
tags:
  - LGDiST
  - Latent-Diffusion
  - DiT
  - Spatial-Transcriptomics
  - Visium
  - Dropout-Imputation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- LGDiST is the **first reference-free latent diffusion model** for Visium dropout completion: a 4-layer transformer autoencoder maps a (6+1)-spot hexagonal neighborhood into a 128-dim latent, and a 12-layer DiT (16 heads) denoises the central spot conditioned on its neighbors. No scRNA-seq reference is needed.
- The key data trick is **inflating the working gene set to the top 1,024 Moran's-I genes ("context genes")** during training, then filtering back to the original HSAGs at inference. Ablations show this gives ~23% MSE over an HSAG-only latent.
- Headline: **18% lower average MSE vs. SpaCKLE across 26 SpaRED datasets**, with downstream gains of **up to 10% MSE (BLEEP)** and **up to 188% PCC (SEPAL)** on six histology-to-expression models — though both numbers come from boxplots / bar charts with no per-dataset table.

## Motivation

Sequencing-based ST (10x Visium) measures spot-level transcriptomes at near-cellular resolution, but suffers heavy stochastic dropout: transcripts that exist biologically go undetected per spot. This degrades direct ST analyses (spatial domains, cell typing) and, more consequentially for medical AI, poisons the training signal for histology-to-expression models that learn to read transcriptomes off H&E.

Existing imputation falls into two camps:

1. **Reference-based** (SpaGE, stPlus, stDiff, Tangram) — lean on scRNA-seq as a prior. They inherit batch effects, alignment failures, and an external-data dependency that may not exist for every tissue.
2. **Reference-free** (median completion, SpaCKLE) — practical but limited. Median is over-smoothed; SpaCKLE is a masked autoencoder restricted to the Highly Spatially Associated Genes (HSAGs) and ignores the rest of the transcriptome.

LGDiST attacks both gaps: reference-free *and* expressive enough to exploit the "context" genes prior work discarded.

## Core Innovation

- **Generative, reference-free.** Unlike SpaCKLE (a masked AE) or stDiff (which needs scRNA-seq), LGDiST is a true diffusion model trained on ST alone.
- **Context Genes (CGs) as scaffolding.** Top-1,024 Moran's-I genes are used as training context — including low-autocorrelation genes that SpaRED's pipeline normally drops — and are filtered out at inference. The CGs only exist to give the latent a richer signal.
- **Hexagonal neighborhood conditioning.** The DiT is conditioned on the 6 immediate Visium neighbors via a binary mask that only noises the central spot, exploiting Visium's lattice topology directly.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | First reference-free **generative** model for ST dropout | By construction (stDiff is reference-based; SpaCKLE is a masked AE) | n/a | ⭐⭐⭐ |
| C2 | 18% lower average MSE vs SpaCKLE | Fig. 4 boxplot only; no per-dataset table, no CI, no significance test | 26 SpaRED | ⭐⭐ |
| C3 | Beats reference-free-adapted stDiff | Same Fig. 4 boxplot; stDiff was re-implemented by LGDiST authors with scRNA-seq dependency stripped — a self-built crippled baseline | 26 SpaRED | ⭐ |
| C4 | Context genes (low-Moran's-I) carry useful info | Table 1: removing CGs adds ~23% MSE | 6 SpaRED subsets | ⭐⭐ |
| C5 | The AE latent space is necessary | Table 1: removing AE adds ~62% MSE on top of the no-CG variant | 6 datasets | ⭐⭐ |
| C6 | 6-neighbor conditioning is necessary; 1-hop ≈ 2-hop | Table 2: no-conditioning blows MSE to 1.891; 1-hop 0.573; 2-hop 0.588 | 6 datasets | ⭐⭐⭐ |
| C7 | Robust to 10-80% missingness | Fig. 7 on **a single dataset (10XGMBSP)** | 1 SpaRED | ⭐ |
| C8 | Improves 6 downstream histology-to-expression models | Fig. 8 bar charts; raw per-model numbers absent from the text | 26 SpaRED × 6 models | ⭐⭐ |
| C9 | SEPAL +188% PCC | Fig. 8b; absolute baseline PCC undisclosed — 0.05 → 0.14 cannot be distinguished from 0.30 → 0.86 | SpaRED | ⭐ |
| C10 | LGDiST "cleans" without smoothing or sparsifying | Fig. 5 (cherry-picked optimal) but Fig. 6 shows median-bias failure where LGDiST is *beaten* by SpaCKLE per-gene | qualitative | ⭐ |

**Overall rating: ⭐⭐.** The ablation block (Tables 1-2) is the strongest part: clean isolation of (a) latent space, (b) context genes, (c) neighborhood conditioning, with deltas large enough to convince on six datasets. The headline 18% number is plausible but presented only as a 26-point boxplot — no per-dataset breakdown, no variance, no significance. The most aggressive number in the abstract (188% PCC on SEPAL) is the weakest, because percentage gains are misleading when baselines are near zero. The stDiff* baseline is a self-stripped reduction of a reference-based method and mostly tells us "stDiff without scRNA-seq is bad." Robustness (Fig. 7) is claimed generally but evaluated on **1 of 26** datasets. Finally, the DiT is **trained to recover fully-empty central spots** but **deployed against partial dropout** — this train/inference distribution mismatch is not discussed anywhere in the paper.

## Method & Architecture

![LGDiST pipeline overview: neighborhood-conditioned latent diffusion for Visium spot completion](/assets/images/paper/lgdist/page_002.png)
*Figure 1: LGDiST pipeline — the central Visium spot's missing HSAG values are recovered from a (6+1)-spot hexagonal neighborhood through a latent autoencoder and a conditional DiT.*

### Stage 0 — Preprocessing

1. Take SpaRED's preselected **HSAGs**: 32 or 128 highest-Moran's-I genes per dataset.
2. Add **CGs**: re-run SpaRED's pipeline to keep the top **1,024** genes by Moran's I. The HSAGs are preserved as a subset; the rest are the low-autocorrelation context.
3. **Pre-complete** the training matrix with **median completion** (from SEPAL, following SpaCKLE) — accelerates convergence and ensures non-zero supervision. Still reference-free.
4. For each spot, build a neighborhood $X \in \mathbb{R}^{(n+1) \times g}$ with $n=6$ (1-hop hexagonal), $g=1024$.

### Stage 1 — Genomics-based Latent Autoencoder

![Stage 1 autoencoder: maps (n+1) x g neighborhoods into a 128-dim latent](/assets/images/paper/lgdist/page_003.png)
*Figure 2: Stage 1 — a 4-layer transformer encoder (1 head, tanh, dropout 0.1) maps the neighborhood with 2D positional encoding into a 128-dim latent; a 2-layer MLP decoder reconstructs it.*

5. Add 2D positional encoding: $X_p = X + \text{PE}_{2D}$.
6. Encoder $E_\theta$: **4-layer transformer, 1 attention head**, dropout 0.1, tanh. Maps $X_p \to X_E \in \mathbb{R}^{(n+1) \times 128}$.
7. Decoder $D_\phi$: 2-layer MLP, tanh + 0.1 dropout. $\hat{X} = D_\phi(X_E)$.
8. **Regularization:** Gaussian noise injected into $X_p$ with probability 0.5 during training.
9. **Weighted reconstruction loss** prioritizes HSAGs over CGs:

$$\mathcal{L}_{\text{rec} } = \alpha\,\text{MSE}(X_{\text{HSAGs} }, \hat{X}_{\text{HSAGs} }) + (1-\alpha)\,\text{MSE}(X_{\text{CGs} }, \hat{X}_{\text{CGs} })$$

The exact $\alpha$ is **not stated in the paper**. AE training: 5,000 epochs, lr 1e-6, AdamW, batch 128.

### Stage 2 — Conditional Latent DiT

![Stage 2 DiT training: central-spot-only masking, neighbor latents as condition](/assets/images/paper/lgdist/page_004.png)
*Figure 3(a): Stage 2 — only the central row of the latent neighborhood is noised; the un-noised neighbor rows serve as the conditioning signal alongside the timestep.*

10. Architecture: **12-layer DiT, 16 attention heads** (Peebles & Xie 2023); ~3.22 GFLOPs.
11. Encode the neighborhood: $X_E = E_\theta(X_p)$.
12. **Central-spot-only mask** $M \in \{0,1\}^{(n+1)\times d}$; forward process at step $t$:

$$\tilde X_E = (1-M) \odot \varepsilon + M \odot X_E,\quad \varepsilon \sim \mathcal{N}(0, I)$$

13. **Condition** the DiT on $C = M \odot X_E$ plus timestep $t$.
14. Noise-prediction loss:

$$\mathcal{L}_{\text{DDPM} } = \mathbb{E}_{X_E, t, \varepsilon}\bigl[\,\|\varepsilon - \hat\varepsilon_\theta(\tilde X_{E,t}, C, t)\|_2^2\,\bigr]$$

15. **Schedule:** cosine, 1500 training steps, 50 sampling steps. Train 1500 epochs, lr 1e-4, AdamW, batch 128, 1× RTX 8000.

### Stage 3 — Inference

*filter CGs**, and write only the originally missing HSAG positions.*

16. Encode the (median-pre-completed) neighborhood, run the DiT to denoise the central latent row, decode with $D_\phi$.
17. **Filter CGs** from the decoded output — CGs are scaffolding only.
18. Write **only the missing positions**; do not overwrite measured values.

A subtle issue: the DiT is trained to recover *fully-empty* central spots and deployed against *partial* dropout. The decoder filtering is the inference-time workaround, but the training objective never saw the partial-mask setting.

## Experimental Results

### Main completion comparison — 26 SpaRED datasets (Fig. 4)

The paper reports the average relative improvement only; no per-dataset table is provided in the text.

| Method | Avg. MSE (relative) | Datasets | Reference-free |
|---|---|---|---|
| stDiff* (LGDiST-stripped re-impl. of stDiff) | worst of three (Fig. 4) | 26 | Forced into RF mode |
| SpaCKLE (prior reference-free SOTA) | baseline | 26 | Yes |
| **LGDiST (ours)** | **−18% vs. SpaCKLE** | 26 | Yes |

### Ablations on 6 random SpaRED datasets (Table 1)

| Variant | Latent | CGs | Avg. MSE |
|---|---|---|---|
| No AE, no CGs (diffusion in raw gene space) | — | — | 1.007 ± 0.200 |
| No CGs (HSAGs only in latent) | ✓ | — | 0.710 ± 0.214 |
| **Full LGDiST** | **✓** | **✓** | **0.573 ± 0.136** |

### Neighborhood ablation (Table 2)

| n neighbors | Avg. MSE | FLOPs (G) |
|---|---|---|
| 0 (no conditioning) | 1.891 ± 0.988 | 0.64 |
| **6 (1-hop, default)** | **0.573 ± 0.136** | **3.22** |
| 18 (2-hop) | 0.588 ± 0.155 | 8.74 |

Removing neighborhood conditioning inflates MSE ~230% — the model is heavily reliant on neighbor context, which makes sense given that's the only signal at inference once the central row is replaced with noise.

### Robustness to missing fraction (Fig. 7, 1 dataset only)

On the 10XGMBSP dataset, LGDiST holds MSE roughly flat at ~0.53-0.55 from 10% to 80% masked; SpaCKLE degrades sharply above 70% masking, exceeding 0.62. This is **a single-dataset experiment**, so the robustness claim does not generalize across the other 25.

### Downstream gene-prediction enhancement (Fig. 8, 6 models)

![Downstream MSE bar chart: 6 histology-to-expression models trained on LGDiST-completed data](/assets/images/paper/lgdist/page_008.png)
*Figure 8(a): All six histology-to-expression models (ST-Net, SEPAL, HisToGene-class, BLEEP, EGGN-style, EGN) improve in MSE when trained on LGDiST-completed data versus SpaCKLE; BLEEP shows the largest reduction (up to 10%).*

- **BLEEP:** up to **10%** MSE reduction.
- **SEPAL:** up to **188%** PCC improvement (absolute baseline undisclosed).
- All six models improve on more than half of the 26 datasets — but raw per-model numbers are not tabulated.

### Qualitative (Fig. 5, 6)

![Optimal qualitative case: LGDiST recovers spatial patterns SpaCKLE flattens](/assets/images/paper/lgdist/page_006.png)
*Figure 5: Optimal case (gene ENSG00000075624) — LGDiST achieves MSE 0.439 vs. SpaCKLE 1.049 and preserves the spatial pattern.*

![Failure case: LGDiST regresses predictions toward tissue median, sometimes worse than SpaCKLE](/assets/images/paper/lgdist/page_007.png)
*Figure 6: Suboptimal case — LGDiST predictions compress toward the tissue median; on this gene LGDiST MSE 1.254 is **worse** than SpaCKLE's 0.969.*

## Limitations

**Acknowledged:**

- Suboptimal cases compress predictions toward the tissue median, underestimating expression range (Fig. 6).
- LGDiST is computationally expensive (3.22 GFLOPs per neighborhood for the DiT alone).

**Unacknowledged but material:**

- **No variance / significance** on the 18% headline. Is it driven by uniform improvement or by SpaCKLE outliers on a few large-MSE datasets?
- **Train/inference distribution mismatch.** Trained on fully-empty central spots, deployed against partial dropout. Filtering CGs at inference is a patch, not a fix.
- **Absolute baselines for PCC in Fig. 8b are missing.** A 188% PCC improvement on a near-zero baseline (e.g. 0.05 → 0.14) is far less impressive than the abstract implies.
- **stDiff* is a self-built crippled baseline** — the LGDiST authors stripped its scRNA-seq dependency. The original stDiff is not evaluated.
- **Reference-based methods are excluded "for fairness."** Users who have scRNA-seq available have no head-to-head data on whether to pick LGDiST or Tangram / stPlus / original stDiff.
- **Robustness (Fig. 7) is shown on 1 of 26 datasets** but framed as a general property.
- **No imaging-based ST** (MERFISH, Xenium, Slide-seq) — Visium-only.
- **No sweep on CG count** (1,024 is fixed); no $\alpha$ value reported for the AE loss.
- **No external dataset** outside SpaRED — generalization beyond this group's own benchmark is unverified, and this is the same group's third paper on SpaRED (SEPAL → SpaCKLE → LGDiST).

## Why It Matters for Medical AI

Sequencing-based ST is the upstream signal for an entire generation of digital-pathology models that learn to predict transcriptomes from H&E (ST-Net, HisToGene, BLEEP, SEPAL, EGN, EGGN-style). Whatever cleans the ST training target propagates into every downstream model. LGDiST's design point — reference-free, neighborhood-conditioned, latent — is a sensible direction because it removes the scRNA-seq dependency that makes reference-based imputation impractical for many clinical tissues. If the 18% MSE / 10% downstream-MSE story replicates on imaging-based ST and on cohorts outside SpaRED, this becomes a default preprocessing step. Until then, treat it as a strong reference-free baseline whose headline magnitudes need an independent per-dataset audit.

## References

- **Paper:** Cárdenas, Manrique, Vega, Ruiz, Arbeláez. *Latent Gene Diffusion for Spatial Transcriptomics Completion.* arXiv:2509.01864, 2025. <https://arxiv.org/abs/2509.01864>
- **Code:** <https://github.com/BCV-Uniandes/LGDiST>
- **SpaRED benchmark:** Mejia et al., MICCAI 2024 — the 26-dataset collection used here.
- **SpaCKLE (prior RF SOTA):** Mejia et al., MICCAI 2024.
- **SEPAL (prior median-completion pipeline + downstream model):** Mejia et al., 2023.
- **DiT (architecture):** Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023.
- **stDiff (reference-based baseline):** Li et al., 2024.
- **Related downstream models:** ST-Net (He et al.), HisToGene, BLEEP, EGN, EGGN-style.
