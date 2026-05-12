---
title: "GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction"
excerpt: "Next-scale AR over k-means gene groups sweeps four HEST-1k slides, peaking at PRAD PCC-200 0.512 vs STEM 0.403 (+27.0%) — on single-slide holdouts with no CV."
categories:
  - Paper
tags:
  - GenAR
  - Spatial-Transcriptomics
  - Autoregressive-Generation
  - Next-Scale-Prediction
  - UNI
  - AdaLN
  - FiLM
  - HEST-1k
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- GenAR reframes H&E-to-spatial-transcriptomics as a **coarse-to-fine next-scale autoregressive token generation** problem over raw integer counts, hierarchically clustered into k-means gene groups along a 1 → 4 → 8 → 40 → 100 → 200 scale ladder.
- A codebook-free VAR-style decoder conditions on UNI image features + sinusoidal spatial PE via **AdaLN**, modulates per-gene outputs via **FiLM**, and trains with **soft-KL on coarser scales + heteroscedastic Gaussian NLL** ($\sigma^2 = \alpha\mu + \beta$) at the finest scale.
- On four HEST-1k slides GenAR beats BLEEP/M2OST/TRIPLEX/STEM on 24 of 25 metric cells; the headline is **PRAD PCC-200 0.512 vs. STEM 0.403 (+27.0% relative)** — but the entire evaluation is single-slide holdouts with no CV, no per-patient bootstrap, and no variance reporting.

## Motivation

Spatial transcriptomics (Visium and friends) anchors gene expression to tissue location and has become a workhorse for tumor-microenvironment biology, but assays cost hundreds to thousands of dollars per sample. H&E images are essentially free, so "predict ST from H&E" has grown into an active sub-field — ST-Net, Hist2ST, HisToGene, BLEEP, EGN, TRIPLEX, M2OST, and most recently the ICLR 2025 diffusion baseline STEM.

The authors argue every prior method makes the same two modeling mistakes: (i) **per-gene independent regression**, ignoring the co-expression/regulatory structure biology depends on; and (ii) **log-transform + continuous regression**, which is incompatible with count-aware downstream pipelines like DESeq2 differential expression and pathway enrichment. The medical-AI pitch is that if you can predict counts that round-trip into standard biological tools, "in silico ST on archival H&E" becomes practically usable rather than a benchmark curiosity.

## Core Innovation

GenAR's bet is that **gene expression has a natural multi-scale structure** — pathway-level → module-level → individual-gene — and that **next-scale autoregressive generation** (VAR-style, but without a VQ-VAE codebook) is a better inductive bias than either independent regression or joint diffusion. K-means clustering on Z-score-normalized training expression induces the hierarchy; a causal Transformer decoder generates count tokens scale-by-scale, factorizing

$$ p(\mathbf{y} \mid \mathbf{H}) = \prod_{k=1}^{K} p(\mathbf{y}^{(k)} \mid \mathbf{H}, \mathbf{y}^{(<k)}) $$

with $K = 6$ and scale dimensions $(1, 4, 8, 40, 100, 200)$. The model never sees log-transformed counts during training — it operates on raw integers and uses a heteroscedastic Gaussian likelihood that gracefully handles the count-magnitude variance.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|
| C1: Discrete AR token generation beats continuous regression and diffusion-in-log-space | Tables 1, 2, 5 (PCC-10/50/200, MSE, MAE) | HER2ST, PRAD, Kidney, Mouse Brain, ccRCC | ⭐⭐ (multi-dataset but single slide each; no CI) |
| C2: Multi-scale next-scale generation is the most important component | Table 3: PCC-10 0.702 → 0.651, MSE 1.191 → 1.406 without it | PRAD only | ⭐⭐ (single dataset, no variance) |
| C3: Codebook-free raw-count prediction preserves biological semantics | Theoretical argument (Eq. 4 derivation); no count-space vs log-space head-to-head | n/a | ⭐ (eval log-transforms predictions, undercutting the claim) |
| C4: Gene-identity FiLM is necessary for per-gene precision | Table 3: PCC-200 0.512 → 0.481 without FiLM | PRAD only | ⭐⭐ (single-slide ablation) |
| C5: Soft-KL + Gaussian-NLL beats cross-entropy | Table 3: PCC-10 0.702 → 0.662 with CE | PRAD only | ⭐⭐ (single CE config, no label-smoothing/focal comparison) |
| C6: Generalizes across tissues (cancer + healthy, breast/prostate/kidney/brain/ccRCC) | Tables 1, 2, 5 | 5 datasets | ⭐⭐ (good coverage but N=1 slide each) |
| C7: (1, 4, 8, 40, 100, 200) is the optimal scale design | Table 6 | PRAD only | ⭐ (3-scale config beats 6-scale on PCC-200) |
| C8: Robust across tissue types, larger margins on cancer | Cross-dataset reading of Tables 1, 2, 5 | 5 datasets | ⭐⭐ (descriptive only; no statistical test possible at N=1) |

**Honest read.** GenAR wins on 24 of 25 main-table metric cells plus 4 of 5 ccRCC appendix cells — that's hard to fake. But every "+X%" in the abstract is **one slide vs one slide**: the PRAD +27.0% headline is computed against two patients of held-out data. The ablations are PRAD-only, so "multi-scale matters most" may or may not transfer to HER2ST/Kidney. Two specific tells worth surfacing: the evaluation **log-transforms GenAR's raw-count outputs** before computing PCC (so the "raw-count semantics" pitch is conceptual, not metric-driven), and Table 6 shows the chosen 6-scale config **loses to a 3-scale config on PCC-200** — the "best overall" claim is true only if you average across metrics.

## Method & Architecture

![GenAR architecture overview](/assets/images/paper/genar/page_004.png)
*Figure 1: GenAR architecture. (a) Genes are hierarchically clustered from 1 → 4 → 8 → 40 → 100 → 200; (b) UNI image features and sinusoidal spatial encoding fuse into a 768-d conditioning embedding H; (c) a causal Transformer generates count tokens scale-by-scale, AdaLN-conditioned on H and FiLM-modulated by gene identity.*

**1. Histology + spatial fusion.** A 224×224 H&E patch per spot $I_u$ is encoded by UNI (Mahmood lab pathology foundation model, 1024-d output). 2D coordinates $S_u$ pass through a sinusoidal positional encoding and a linear projector + LayerNorm. The two streams concatenate and feed a 2-layer MLP (GELU + dropout) producing the conditioning vector $\mathbf{H} \in \mathbb{R}^{768}$.

**2. Hierarchical gene clustering (load-bearing).** K-means on Z-score-normalized training expression first partitions the 200 genes into 4 major clusters, then subdivides each into groups of ~12 genes. This induces the gene ordering at the finest scale; intermediate scales are produced by `AdaptiveAvgPool1d(y, d_k)` over that ordering.

![Hierarchical gene clustering](/assets/images/paper/genar/fig_p004_01.png)
*Figure 2: Circular dendrogram with heatmap visualization of the k-means gene clustering on Z-score-normalized expression profiles — this induces the autoregressive ordering at the finest scale.*

**3. Causal Transformer decoder.** 12 stacked Transformer blocks. AdaLN-Zero (DiT-style) injects $\mathbf{H}$ into every block; gene-identity embeddings $\mathbf{E}_{\text{identity}} \in \mathbb{R}^{200 \times 768}$ produce per-gene scale/shift parameters via FiLM at the output head. At scale $k$, the input sequence is `[start_token, GeneEmbed(y^(<k)), GeneUpsampling(E_outputs, k)] + PosEmbed(k) + ScaleEmbed(k)`; a causal mask is applied; the last $d_k$ positions are sliced, FiLM-modulated, and decoded.

**4. Multi-scale loss.** Coarser scales ($k < K$) use **soft-label KL divergence** against temperature-smoothed pooled targets $q^{(k)} = \mathrm{softmax}(\mathrm{AdaptiveAvgPool1d}(y, d_k) / \tau)$. The **finest scale** uses a **heteroscedastic Gaussian NLL** with variance $\sigma^2 = \alpha \cdot \hat{\mu} + \beta$. Total loss is the unweighted mean across scales.

![Training vs inference sequence construction](/assets/images/paper/genar/page_006.png)
*Figure 3: Training uses teacher forcing with ground-truth coarser-scale tokens; inference autoregressively chains the model's own predictions across the 6 scales, starting from a START token.*

**5. Training and inference asymmetry.** Adam, lr 1e-4, batch size 64, NVIDIA H100 80GB. Inference is greedy-argmax by default (temperature sampling optional). Importantly, **outputs are post-hoc log2-transformed during evaluation** to match the baseline convention — so the headline PCC numbers are computed in log-space, not count-space.

## Experimental Results

### Main quantitative comparison (HER2ST + PRAD)

| Method | HER2ST PCC-10 | HER2ST PCC-50 | HER2ST PCC-200 | HER2ST MSE | HER2ST MAE | PRAD PCC-10 | PRAD PCC-50 | PRAD PCC-200 | PRAD MSE | PRAD MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| BLEEP (NeurIPS'23) | 0.773 | 0.714 | 0.565 | 1.243 | 0.833 | 0.580 | 0.510 | 0.316 | 2.475 | 1.091 |
| M2OST (AAAI'25) | 0.810 | 0.759 | 0.660 | 1.151 | 0.820 | 0.602 | 0.551 | 0.442 | 1.290 | 0.862 |
| TRIPLEX (CVPR'24) | 0.783 | 0.714 | 0.586 | 1.212 | 0.857 | 0.620 | 0.544 | 0.423 | 1.319 | 0.836 |
| STEM (ICLR'25) | 0.831 | 0.770 | 0.625 | 1.199 | 0.787 | 0.636 | 0.555 | 0.403 | 1.457 | 0.857 |
| **GenAR** | **0.842** | **0.784** | **0.663** | **1.082** | **0.745** | **0.702** | **0.650** | **0.512** | **1.191** | **0.771** |

### Main quantitative comparison (Kidney + Healthy Mouse Brain)

| Method | Kidney PCC-10 | Kidney PCC-50 | Kidney PCC-200 | Kidney MSE | Kidney MAE | Brain PCC-10 | Brain PCC-50 | Brain PCC-200 | Brain MSE | Brain MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| BLEEP | 0.500 | 0.422 | 0.314 | 1.926 | 0.945 | 0.342 | 0.280 | 0.156 | 1.591 | 0.987 |
| M2OST | 0.494 | 0.447 | 0.318 | 1.785 | 0.925 | 0.456 | 0.387 | 0.231 | 1.148 | 0.861 |
| TRIPLEX | 0.542 | 0.469 | 0.336 | 1.732 | 0.887 | 0.501 | 0.445 | 0.312 | 1.157 | 0.822 |
| STEM | 0.567 | 0.483 | 0.322 | 1.832 | 0.997 | 0.526 | 0.452 | 0.331 | 1.235 | 0.864 |
| **GenAR** | **0.589** | **0.514** | **0.354** | **1.636** | **0.871** | **0.568** | **0.503** | **0.367** | **1.138** | **0.805** |

Margins over STEM (the most recent generative baseline) on PCC-200: **PRAD +27.0%**, Mouse Brain +10.9%, Kidney +9.9%, HER2ST +6.1%. The ccRCC appendix (INT2 holdout) gives GenAR PCC-200 0.276 vs STEM 0.256 (+7.8%), with MSE 1.465 vs 1.422 being the **only metric in the entire paper** where GenAR loses.

### Ablations (PRAD only)

- **Multi-scale removed** (single scale = 200): PCC-10 0.702 → 0.651, MSE 1.191 → 1.406 — the largest single hit.
- **Gene-identity FiLM removed**: PCC-200 0.512 → 0.481.
- **Soft-KL + Gaussian-NLL → vanilla cross-entropy**: PCC-10 0.702 → 0.662.
- **Scale design** (Table 6): single (200) → 0.622; (1, 20, 200) → 0.681; (1, 40, 100, 200) → 0.683; **(1, 4, 8, 40, 100, 200) → 0.702** on PCC-10. But on **PCC-200 the 3-scale config (0.534) actually beats the chosen 6-scale (0.512)** — an internal contradiction with the "optimal balance" framing.
- **Backbone** (Table 4): ResNet-18 → 0.273, CONCH → 0.286, UNI → 0.383 with a simple head; full GenAR (UNI + multi-scale AR head) → 0.512. So the +33.7% the paper attributes to GenAR is really the AR head, not UNI.

### Qualitative

![SSR4 spatial prediction on HER2ST SPA148](/assets/images/paper/genar/fig_p009_01.png)
*Figure 4: Predicted spatial expression of SSR4 on HER2ST SPA148. GenAR most closely tracks the high-expression yellow-green hotspots in the ground truth; BLEEP and M2OST oversmooth, STEM blurs high-expression boundaries.*

![Additional gene qualitative comparison](/assets/images/paper/genar/fig_p025_01.png)
*Figure 5: Appendix qualitative comparison on an additional gene; the coarse-to-fine factorization preserves sharper expression boundaries than the baselines.*

## Limitations

**Author-admitted.**
- Weaker performance in extremely sparse regions (token rarity / gradient sparsity for low-count genes); pathway/ontology-informed grouping suggested as future work.
- Future work proposed on integrating gene regulatory networks and pathway-level priors.

**Not addressed in the paper.**
- **No CV, no CI, no leave-one-patient-out, no statistical test.** Every reported margin is one slide vs one slide; the same flaw STEM had, and GenAR doesn't fix it.
- **Evaluation log-transforms the raw-count outputs**, undercutting the "discrete count semantics preserved" pitch — the headline metric is computed in log-space.
- **Scale-design ablation contradicts the chosen config**: PCC-200 prefers fewer, coarser scales (3-scale 0.534 > 6-scale 0.512). The paper averages over metrics to call 6-scale "best overall."
- **PRAD-only ablations** — the architectural claims may not transfer.
- **No comparison to STPath** (npj Digital Medicine 2025, the foundation-model ST-from-WSI baseline trained on 928 WSIs over 38,984 gene symbols), no comparison to UMPIRE or zero-shot STImage-1k4m. The peer set is restricted to per-organ-trained models.
- **No per-gene PCC distribution** (only top-10/50/200 averages), so it's unclear whether GenAR helps housekeeping genes, marker genes, or just the high-correlation tail.
- **No compute/latency reporting** — autoregressive across 6 scales should be much faster than STEM's 20-sample DDPM but it's not quantified.
- **Vocabulary size for count tokens is never specified.** For a method whose central pitch is "discrete count tokens," this is a load-bearing omission.
- **No mention of how $\alpha, \beta$ in $\sigma^2 = \alpha\mu + \beta$ are set** — learned, fixed, or per-dataset tuned?
- **No downstream-bio experiment** (differential expression on predicted counts, pathway enrichment); the "raw counts for biology" argument would land harder with one.

## Why It Matters for Medical AI

If H&E-to-ST prediction matures into a tool that produces count-space outputs compatible with standard biological pipelines, it changes the cost structure of spatial-omics research — archival H&E slides become a substrate for in-silico Visium-style assays. GenAR is the cleanest published argument so far that **discrete autoregressive generation over biologically-structured gene hierarchies is the right inductive bias** for this task, and its sweep across four HEST-1k datasets is the strongest such result to date. The caveats — single-slide evaluation, PRAD-only ablations, post-hoc log-transformed metrics, no comparison to STPath — mean the "state-of-the-art" claim deserves the qualifier "on this evaluation protocol." A natural sequel is to slot a next-scale AR head into a pretrained STPath-style backbone and run leave-one-patient-out.

## References

- Paper (arXiv): [2510.04315](https://arxiv.org/abs/2510.04315)
- Code: [github.com/oyjr/genar](https://github.com/oyjr/genar)
- Related: STEM (ICLR 2025, conditional diffusion in log-space), TRIPLEX (CVPR 2024), M2OST (AAAI 2025), BLEEP (NeurIPS 2023), HEST-1k (Jaume et al., NeurIPS 2024), UNI (Mahmood lab pathology FM)
- Methodological lineage: VAR (next-scale prediction), DiT / AdaLN-Zero (Peebles & Xie; Dhariwal & Nichol), FiLM (Perez et al.)
