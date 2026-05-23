---
title: "SpaDiT: Diffusion Transformer for Spatial Gene Expression Prediction using scRNA-seq"
excerpt: "Per-gene conditional DiT with Flash-Attention scRNA-seq prior: best mean PCC on 9 of 10 paired ST benchmarks, e.g. MC 0.812 vs SpatialScope 0.776."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/spadit-diffusion-transformer-spatial-transcriptomics-imputation/
tags:
  - SpaDiT
  - Diffusion Transformer
  - Spatial Transcriptomics
  - scRNA-seq
  - Flash-Attention
  - adaLN
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- SpaDiT reframes ST gene imputation as a **per-gene** generative problem: each gene `i` becomes one sample `(x_st^i, x_sc^i)`, and a DiT denoises its masked ST expression conditioned on the full paired scRNA-seq matrix.
- A **Flash-Attention condition projector** compresses the entire scRNA-seq matrix into a low-dim embedding `x_ψ` that is injected at every DiT block via **adaLN-style scale/shift modulation** — the scRNA-seq atlas is used as a *prior*, not a retrieval anchor like Tangram/SpaGE.
- Best mean PCC on **9 of 10** paired scRNA-seq + ST datasets vs. 8 baselines — e.g. **MC 0.812±0.039 vs SpatialScope 0.776±0.006**, **MG 0.657±0.035 vs 0.612±0.143**, **MPMC 0.808±0.043 vs 0.769±0.022**. The headline "SOTA on all five metrics across all ten datasets" is slightly overstated (ML is a soft spot, std bars overlap, within-gene only).

## Motivation

Spatial transcriptomics platforms trade coverage for resolution: seq-based assays (Visium / Slide-seqV2 / Stereo-seq) suffer from high per-spot dropout (the paper reports 6.3%–83.9% across its ten datasets), and image-based assays (seqFISH / MERFISH / STARmap) measure only a few hundred pre-selected genes. The standard fix is to use a paired scRNA-seq atlas to *impute* unmeasured ST genes.

Existing methods — Tangram, SpaGE, stPlus, novoSpaRc, SpaOTsc, SpatialScope — align ST spots to scRNA-seq cells through similarity on the small set of *shared* genes, then transfer expression. SpaDiT's claim is that this anchoring is the bottleneck: it ignores most of the scRNA-seq matrix, inherits SC batch bias, and is fundamentally retrieval-based. A *generative* model that consumes the entire scRNA-seq atlas as a conditioning prior should give a cleaner formulation.

Note that SpaDiT is **transcriptomic-only** (scRNA-seq -> ST). It is **not** an H&E -> ST method and is therefore complementary to histology-conditioned approaches like Stem, ST-Net, BLEEP, or EGN rather than a direct competitor to them.

## Core Innovation

Three design choices, all coupled:

1. **Per-gene sample reframe.** Most ST imputation methods treat a *spot* as a sample. SpaDiT treats a *gene* as a sample: for gene `i`, the input pair is `(x_st^i ∈ R^p, x_sc^i ∈ R^q)` where `p` = #spots and `q` = #cells. Concatenation along the shared-gene axis becomes geometrically meaningful.

2. **Latent Embedding `φ`.** A feed-forward projector maps `x_st^i` and `x_sc^i` to a common dimension and concatenates: `x_φ = x̂_st^i ⊕ x̂_sc^i`. Two masks — zero-mask `m_1` and non-zero mask `m_2` — force the network to reconstruct both expressed and dropped-out entries.

3. **Condition Embedding `ψ` via Flash-Attention.** The full scRNA-seq matrix `X_sc` is reduced via Flash-Attention with extra dim-reduction maps `Φ_K, Φ_V`:

   $$x_\psi = \text{softmax}\!\left(\frac{Q\,\Phi_K(K)^\top}{\sqrt{d_k} }\right)\Phi_V(V)$$

   `x_ψ` becomes the conditioning token `x_0^c` injected into every DiT block as adaLN-style scale/shift modulation — the Peebles & Xie 2023 trick, ported to gene expression.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | SOTA across multiple metrics | Table 2: PCC/SSIM/RMSE/JS over 10 datasets, 8 baselines | ⭐⭐⭐ on PCC, ⭐⭐ overall (loses to SpatialScope on ML across PCC/RMSE) |
| C2 | "Best on all five indicators" | Table 2 + Fig 2 AS | ⭐⭐ — overstated; ML is a soft spot, MH SSIM also loses |
| C3 | Strongest gene-level correlation | Table 2 PCC column | ⭐⭐⭐ on 9/10 datasets, but variance bars often overlap competitors (HBC, MVC, MPMC); no significance test |
| C4 | Spatial similarity preserved | Figure 5 spatial overlays | ⭐⭐ — qualitative; displayed gene per dataset is the *highest-PCC* gene = cherry-picked |
| C5 | Clean spatial boundaries | Figure 5 + supp | ⭐⭐ — no quantitative spatial-coherence metric (e.g. Moran's I) |
| C6 | Robust under downsampling | Figure 6 RS at 0.1/0.3/0.5/0.7 | ⭐ — single dataset (MH) in main text; MH has the **lowest dropout (6.3%)** = easiest case |
| C7 | Transformer backbone is critical | Table 3 (U-Net / Mamba / Transformer) | ⭐⭐⭐ — Transformer wins on every dataset by 4–15 AS points |
| C8 | Condition `ψ` is critical | Table 4 (`w/o ψ`) | ⭐⭐⭐ — removing it is the largest drop in every dataset |
| C9 | Flash-Attention beats MLP for `ψ` | Table 4 (`w/o Flash-Attention`) | ⭐⭐ — consistent gap, but no other attention variant tested |
| C10 | Full SC matrix beats common-gene-only `ψ` | Table 4 (`w/ Common Gene in ψ`) | ⭐⭐ — small-to-medium positive effect |
| C11 | Concat in `φ` is essential | Table 4 (`w/o Concat`) | ⭐⭐ — dataset-dependent (MPMC drops sharply, ME drops too) |
| C12 | Preserves gene-similarity structure | Figures 3, 4 (UMAP, hierarchical clustering) | ⭐ — entirely qualitative; no Mantel / kBET / LISI |

**Honest read.** The PCC and ablation evidence is genuinely strong — SpaDiT beats 8 baselines across 10 datasets and the ablations are well-controlled. Three things weaken the headline. First, **all reported standard deviations are within-dataset across the test genes**, not across training runs or random seeds, so std bars overlap competitors in several cells (HBC, MVC, MPMC). Second, "best on all five indicators" is not what Table 2 shows — ML is a consistent soft spot. Third, the robustness story (Figure 6) is reported in the main text on only the **least-sparse dataset (MH, 6.3% dropout)**, so the strongest robustness claim leans on the easiest sample.

## Method & Architecture

![SpaDiT architecture overview](/assets/images/paper/spadit/page_003.png)
*Figure 1: SpaDiT architecture — (A) training masks ST genes and uses the full scRNA-seq matrix as conditioning; (B) inference denoises Gaussian noise into predicted gene expression. The Latent Embedding `φ` concatenates shared-gene representations and the Condition Embedding `ψ` compresses the full scRNA-seq matrix via Flash-Attention into adaLN modulation in every DiT block.*

**Diffusion in latent space.** Standard DDPM with conditional reverse:

$$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\alpha_t}\,x_0,\,(1-\alpha_t) I), \quad x_t = \sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\varepsilon$$

$$p_\theta(x^*_{t-1} \mid x^*_t, x_0^c) = \mathcal{N}\!\left(\mu_\theta(x^*_t, t \mid x_0^c),\,\sigma_\theta(x^*_t, t \mid x_0^c) I\right)$$

**Training loss** is the usual ε-prediction with the conditioning argument:

$$\mathcal{L} = \mathbb{E}_{x_0,\,\varepsilon \sim \mathcal{N}(0,I),\,t}\;\bigl\|\,\varepsilon - \varepsilon_\theta\bigl(\sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\varepsilon,\,t \mid x_0^c\bigr)\bigr\|_2^2$$

**Backbone.** Stack of `N` DiT blocks (exact `N` not stated in the main text), each: LayerNorm -> adaLN scale/shift by `x_ψ` -> Multi-Head Self-Attention -> residual -> LayerNorm -> adaLN scale/shift -> pointwise FFN -> residual. Residual blocks initialize to identity. A linear decoder produces the noise + diagonal-covariance prediction in original gene-expression space — there is **no separate reconstruction network**.

**Inference (Algorithm 2).** Initialize `x_T ~ N(0, I)`. For `t = T..1`: recompute `x_t^c = ψ(X_sc)`, build `x_t = Φ(x_t, x_sc^i)`, denoise with the trained `ε_θ(x_t, t | x_t^c)`. Output `x_0` as the predicted held-out gene expression.

The paper does not report `N`, hidden size, `T` (timesteps), learning rate, optimizer, batch size, or total iterations in the main text — presumably in the supplementary.

## Experimental Results

**Datasets.** Ten paired scRNA-seq + ST datasets, gene-wise split 7:2:1 (train/val/test). Image-based ST: MH, MHPR, ML, MG, MVC. Seq-based ST: MHM, HBC, ME, MPMC, MC. Dropout ranges from 6.3% (MH) to 83.9% (MC). Two cautions on the setup: (i) ME has only 198 spots; (ii) the split is over genes, not spots/slides/patients, so spatial neighborhoods leak between train and test.

**Main comparison — PCC↑ (mean ± std over test genes):**

| Method | MG | MH | MHPR | MVC | MHM | HBC | ME | MPMC | MC | ML |
|---|---|---|---|---|---|---|---|---|---|---|
| Tangram | 0.458±0.203 | 0.523±0.116 | 0.683±0.012 | 0.623±0.117 | 0.536±0.053 | 0.703±0.142 | 0.503±0.025 | 0.727±0.026 | 0.745±0.003 | 0.714±0.056 |
| scVI | 0.476±0.157 | 0.446±0.157 | 0.691±0.143 | 0.594±0.023 | 0.511±0.117 | 0.656±0.005 | 0.496±0.007 | 0.716±0.014 | 0.736±0.015 | 0.637±0.001 |
| SpaGE | 0.526±0.114 | 0.438±0.163 | 0.653±0.063 | 0.603±0.107 | 0.545±0.226 | 0.639±0.025 | 0.512±0.013 | 0.753±0.066 | 0.769±0.011 | 0.653±0.007 |
| stPlus | 0.503±0.233 | 0.401±0.037 | 0.483±0.231 | 0.574±0.059 | 0.476±0.007 | 0.597±0.111 | 0.526±0.026 | 0.689±0.007 | 0.701±0.099 | 0.699±0.014 |
| SpaOTsc | 0.522±0.014 | 0.485±0.107 | 0.657±0.002 | 0.629±0.147 | 0.496±0.018 | 0.587±0.107 | 0.547±0.006 | 0.734±0.201 | 0.738±0.064 | 0.723±0.005 |
| novoSpaRc | 0.563±0.158 | 0.567±0.252 | 0.613±0.146 | 0.656±0.037 | 0.515±0.003 | 0.647±0.122 | 0.569±0.013 | 0.756±0.015 | 0.756±0.015 | 0.766±0.056 |
| SpatialScope | 0.612±0.143 | 0.582±0.183 | 0.637±0.031 | 0.683±0.114 | 0.547±0.103 | 0.733±0.183 | 0.563±0.056 | 0.769±0.022 | 0.776±0.006 | **0.803±0.014** |
| stDiff | 0.482±0.021 | 0.527±0.013 | 0.621±0.007 | 0.601±0.043 | 0.471±0.009 | 0.544±0.021 | 0.553±0.014 | 0.629±0.011 | 0.604±0.019 | 0.736±0.099 |
| **SpaDiT** | **0.657±0.035** | **0.621±0.099** | **0.770±0.043** | **0.725±0.106** | **0.573±0.083** | **0.772±0.057** | **0.590±0.146** | **0.808±0.043** | **0.812±0.039** | 0.784±0.096 |

SpaDiT wins PCC on 9/10 datasets; **ML is the exception** — SpatialScope wins both PCC and RMSE there. Std bars are within-gene only (no run-to-run variance).

![Accuracy Score across 10 paired datasets](/assets/images/paper/spadit/page_005.png)
*Figure 2: Accuracy Score (AS, composite) across 10 paired scRNA-seq + ST datasets — SpaDiT's box sits above all eight baselines on every dataset (no numeric table provided, which is a real omission).*

**Backbone ablation (Table 3, AS metric):**

| Dataset | U-Net | Mamba | Transformer (SpaDiT) |
|---|---|---|---|
| MG | 0.454±0.011 | 0.477±0.008 | **0.514±0.032** |
| MH | 0.453±0.011 | 0.471±0.026 | **0.553±0.057** |
| MHPR | 0.477±0.013 | 0.475±0.014 | **0.506±0.038** |
| MVC | 0.482±0.011 | 0.474±0.011 | **0.572±0.033** |
| MHM | 0.466±0.011 | 0.461±0.102 | **0.553±0.037** |
| HBC | 0.478±0.012 | 0.462±0.086 | **0.613±0.024** |
| ME | 0.470±0.010 | 0.489±0.051 | **0.589±0.060** |
| MPMC | 0.458±0.013 | 0.421±0.022 | **0.488±0.033** |
| MC | 0.470±0.010 | 0.478±0.015 | **0.564±0.026** |
| ML | 0.487±0.013 | 0.488±0.021 | **0.619±0.024** |

Transformer wins on all 10 datasets; Mamba is essentially tied with U-Net. Clean ablation.

**Module ablation (Table 4, AS):**

| Variant | MG | MH | MHPR | MVC | MHM | HBC | ME | MPMC | MC | ML |
|---|---|---|---|---|---|---|---|---|---|---|
| **SpaDiT (full)** | **0.514** | **0.553** | **0.506** | **0.572** | **0.553** | **0.613** | **0.589** | **0.488** | **0.564** | **0.619** |
| w/o Flash-Attention | 0.439 | 0.485 | 0.431 | 0.429 | 0.415 | 0.431 | 0.422 | 0.425 | 0.438 | 0.459 |
| w/o Condition ψ (zero) | 0.383 | 0.336 | 0.404 | 0.394 | 0.318 | 0.407 | 0.376 | 0.401 | 0.417 | 0.423 |
| w/ Common Gene in ψ only | 0.483 | 0.503 | 0.437 | 0.533 | 0.489 | 0.537 | 0.426 | 0.411 | 0.503 | 0.526 |
| w/o Concat in φ | 0.462 | 0.501 | 0.432 | 0.489 | 0.485 | 0.407 | 0.512 | 0.311 | 0.489 | 0.503 |

Dropping the condition entirely (`w/o ψ`) is the worst — the scRNA-seq prior is doing real work. Flash-Attention vs. MLP costs ~7–15 AS points. Concatenating shared-gene reps in `φ` is consistently positive (~3–15 AS).

![UMAP overlays of predicted vs. true expression](/assets/images/paper/spadit/page_007.png)
*Figure 3: UMAP of predicted vs. true gene expressions across 10 datasets and 9 methods — SpaDiT's predictions (orange) nearly overlap the ground truth (blue), while baselines show visibly larger displacement. Qualitative.*

![Hierarchical clustering of pairwise gene distances](/assets/images/paper/spadit/page_008.png)
*Figure 4: Hierarchical clustering on pairwise gene-distance matrices per method — SpaDiT's heatmap structure most closely matches ground truth, indicating gene-gene similarity is preserved. No Mantel/LISI/kBET quantification.*

![Spatial expression patterns per dataset](/assets/images/paper/spadit/page_009.png)
*Figure 5: Spatial expression patterns for a marker gene per dataset (MG / MHPR / HBC / MPMC / MC) — SpaDiT reproduces spatial contours more faithfully than the 8 baselines. Caveat: the displayed gene per dataset is explicitly the highest-PCC gene, so this is best-case visualization.*

![Robustness under downsampling on MH](/assets/images/paper/spadit/page_010.png)
*Figure 6: Robustness Score (fraction of genes with PCC > 0.5) on the MH dataset at downsampling rates 0.1/0.3/0.5/0.7 — SpaDiT retains a higher RS than baselines. Important caveat: shown only for MH in the main text, and MH has the **lowest dropout (6.3%)** of all 10 datasets — i.e. the easiest case for downsampling.*

## Limitations

**Authors acknowledge.**
- When ST lacks sufficient marker information, SpaDiT's reliance on existing expression signals degrades — shared with competing methods.
- Diffusion is relatively new for ST imputation; future work could combine diffusion with similarity-based anchoring.

**Not addressed.**
- **Run-to-run variance.** All ± values are over test genes in a single split. No multi-seed training, no cross-validation, no statistical significance tests (paired t-test / Wilcoxon / bootstrap CI). Std bars overlap competitors in many cells.
- **Computational cost.** No FLOPs, wallclock, or GPU memory comparison vs. baselines. Algorithm 2 recomputes `ψ(X_sc)` at every denoising step, which is expected to be expensive — but this is never quantified.
- **Hyperparameter sensitivity.** Number of DiT blocks, hidden size, `T`, learning rate, batch size all absent from the main text.
- **Batch effect.** The motivation argues scRNA-seq introduces batch bias and that conditioning beats anchoring on this axis, but it is never directly tested (e.g. pair SC and ST from different cohorts).
- **Cell-type preservation.** No downstream cell-typing evaluation (ARI/NMI on predicted vs. true expression). Only gene-level PCC/SSIM/RMSE/JS.
- **External validation.** All 10 datasets are pre-curated benchmarks; no external clinical ST cohort.
- **vs. stDiff.** stDiff (Li et al. 2024) is the only other diffusion baseline and is among the *worst* performers. The paper does not analyze *why* SpaDiT so dramatically outperforms the only other ST-imputation diffusion model — backbone? conditioning? per-gene reframe?
- **vs. histology-conditioned ST.** SpaDiT is purely transcriptomic and is **not** directly comparable to H&E-conditioned ST methods (Stem, ST-Net, BLEEP, EGN). The two lines should be read as complementary modalities for the same imputation problem, not as direct competitors.

## Why It Matters for Medical AI

Two reasons. First, on clinical-grade ST slides like HBC (human breast cancer), rescuing low-expression marker genes matters for downstream cell-type and pathway analysis, and SpaDiT's per-gene + adaLN conditioning recipe transfers cleanly to any paired SC+ST cohort without a separate retrieval index. Second, the recipe — Flash-Attention condition projector + adaLN scale/shift modulation per DiT block — is reusable: any tabular-omics imputation problem with a large auxiliary matrix (multi-omic, cross-platform, cross-tissue) can plug into this framework. The complementary modality (H&E -> ST, e.g. Stem) is orthogonal to SpaDiT, so in a real clinical pipeline both can in principle be ensembled.

## References

- Paper: SpaDiT: Diffusion Transformer for Spatial Gene Expression Prediction using scRNA-seq. Xiaoyu Li, Fangfang Zhu, Wenwen Min. *Briefings in Bioinformatics* 2024. arXiv:2407.13182 — https://arxiv.org/abs/2407.13182
- DiT backbone: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
- Flash-Attention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022.
- Diffusion baseline: stDiff — Li et al., "stDiff: a diffusion model for imputing spatial transcriptomics", *Briefings in Bioinformatics* 2024.
- Anchor-based baselines: Tangram (Biancalani et al. 2021), SpaGE (Abdelaal et al. 2020), stPlus (Shengquan et al. 2021), SpaOTsc (Cang & Nie 2020), novoSpaRc (Nitzan et al. 2019), SpatialScope (Wan et al. 2024), scVI (Lopez et al. 2018).
- Complementary modality (H&E -> ST): Stem (Zhu et al., ICLR 2025), ST-Net, BLEEP, EGN — covered in separate posts on this blog.
