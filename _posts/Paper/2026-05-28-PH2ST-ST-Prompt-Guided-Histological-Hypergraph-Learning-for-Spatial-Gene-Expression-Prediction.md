---
title: "PH2ST: Prompt-Guided Hypergraph Learning for Spatial Transcriptomics Prediction in Whole Slide Images"
excerpt: "PH2ST reframes WSI-to-ST prediction as inference-time prompting — at test time the model consumes 10% measured spots from the same WSI as a prompt — and reports PCC 0.338 on HER2+ (vs TRIPLEX 0.318) and 0.479 on cSCC (vs 0.431), with std devs that swallow the deltas."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/ph2st/
tags:
  - PH2ST
  - Spatial-Transcriptomics
  - Hypergraph
  - Prompt-Learning
  - Computational-Pathology
  - Cross-Attention
  - UNI
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- PH2ST reframes WSI-to-ST gene-expression prediction as **inference-time prompting**: at test time, a small subset of *measured* spots from the *same test WSI* (default **10%**) is supplied as a prompt and the model predicts the remaining unmeasured spots. This is the central conceptual move and also the central caveat — the test slide is not blind to the model.
- The architecture stacks (i) a **dual-scale histological hypergraph** (slide-level spot hypergraph + per-spot neighborhood sub-hypergraph over 25 sub-patches in a 1120×1120 region with α=5) on top of frozen **UNI** features, (ii) ViT global fusion, and (iii) a **cross-attention prompt-refinement** module where ST-prompt embeddings act as queries against neighborhood histology tokens.
- Headline numbers (5-fold patient-stratified CV, 10% prompts at test time): PCC **0.338** on HER2+ (vs TRIPLEX 0.318) and **0.479** on cSCC (vs 0.431). The deltas (+0.020 / +0.048) are real but smaller than the std devs (±0.10–0.15). Performance scales monotonically with prompt ratio to PCC ≈ **0.606 at 90% prompts** on HER2+.

## Motivation

The paper argues three weaknesses in existing image→ST predictors (ST-Net, HisToGene, Hist2ST, HGGEP, BLEEP, TRIPLEX):

1. **The pure image-only setting does not match clinical reality.** In practice, partial ST is often already measured — small chip designs, dropout, low resolution, manually-annotated regions — and a model that ignores this signal is leaving free information on the table.
2. **Prior work under-models multi-scale spatial structure.** Most methods operate at one of "single spot patch" or "slide-wide graph", not both at once.
3. **Cross-patient / cross-section domain shift remains severe.** ST datasets are tiny (typically a dozen patients), so any pure regression model has very little to generalize from.

PH2ST's pitch: *use* the partial ST that clinicians already have at hand to anchor inference. Reframe ST prediction as imputation / super-resolution / local-to-global extrapolation rather than blind regression from H&E.

## Core Innovation

**Inference-time prompting with measured spots from the same WSI.** During training, 30% of spots per WSI are randomly masked and the unmasked rows act as a "prompt"; at inference, 10% of spots from the *same test WSI being scored* are fed in as the prompt. A learnable projection $\Phi_{\text{prompt}} = \text{LN}(\text{Dropout}(\text{FC}(\text{GELU}(Y_{\text{prompt}}W))))$ turns these gene-expression rows into tokens that **query** per-spot neighborhood histology tokens via cross-attention.

This is conceptually different from previous WSI→ST work in two ways:

- The prompt comes from the *same slide* the model is predicting, so the model is conditioning on a per-test-WSI oracle subset.
- The four sampling strategies (fully random, Poisson disc, sparse square, dense square) are rhetorically mapped onto imputation, super-resolution, and local-to-global scenarios — though all are evaluated with the same held-out PCC, so the task-distinction is more in framing than in metric.

The fairness compromise: **every non-BLEEP baseline is additionally fine-tuned on the same 10% prompt spots** of each test WSI for 5 epochs on the final linear layer, so all methods see the same intra-slide oracle. The reader should still mentally tag PH2ST's headline PCC as "with 10% intra-slide ground truth" — these numbers are not directly comparable to pure image-only image→ST PCCs in other papers.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | PH2ST is the *first* framework to cast ST prediction as inference-time prompting. | Novelty claim; Sec. 2.2 survey of prior work shows no method takes per-test-WSI ST as input. | — | ⭐⭐ |
| C2 | PH2ST achieves SOTA on HER2+ and cSCC. | Table 1 — outperforms TRIPLEX on every metric × dataset cell; non-parametric significance Fig. 3. | HER2+, cSCC | ⭐⭐ |
| C3 | Dual-scale hypergraph is necessary. | Fig. 5a dashed lines on cSCC — removing either branch hurts. | cSCC only | ⭐⭐ |
| C4 | Prompt-guided refinement is the key gain over TRIPLEX. | No ablation removing only the prompt module while keeping dual-scale hypergraph. Indirect control via 10%-prompt-finetuned baselines. | HER2+, cSCC | ⭐ |
| C5 | UNI is the best encoder for ST prediction. | Table 2 on HER2+ only; UNI vs Virchow2 differ by ≈0 on PCC. | HER2+ only | ⭐ |
| C6 | Method is robust across realistic sampling regimes. | Table 3 / Fig. 6 on HER2+ only; Poisson is clearly weaker than random/dense at high ratios. | HER2+ only | ⭐⭐ |
| C7 | Performance scales smoothly with prompt ratio (10→90%). | Table 3, PCC 0.338 → 0.606 monotonic. | HER2+ | ⭐⭐⭐ |
| C8 | PH2ST supports imputation, super-resolution, and local-to-global prediction. | Sampling strategies *named* after these scenarios but evaluated with identical held-out PCC; no comparison to gimVI / SpaGCN / Tangram. | HER2+ | ⭐ |
| C9 | Robust to domain shift. | 5-fold patient-stratified CV on two datasets; no external-dataset transfer; no train-on-HER2 → test-on-cSCC. | HER2+, cSCC | ⭐ |

**Honest read.** The headline win is real but smaller than the abstract implies. On HER2+ the delta over TRIPLEX is **+0.020 PCC** with std dev **±0.139**; on cSCC it is **+0.048 PCC** with std dev **±0.103**. The "significance" reported in Fig. 3 rests on per-gene paired tests, not per-fold means — so the apparent statistical separation comes from gene-count power, not from a robust fold-level effect. Critically, there is **no "PH2ST without the cross-attention prompt branch" ablation**, so the central architectural-vs-prompt-signal contribution is not isolated. Given the small Δ over TRIPLEX (which itself sees the same 10% prompt via fine-tuning), this is the experiment that would settle the paper's central claim. The "imputation / super-resolution / local-to-global" framing is rhetorical re-labeling of sampling patterns; without dedicated comparisons to gimVI / SpaGCN / Tangram on a task-specific benchmark, that framing does not earn more than ⭐.

## Method & Architecture

![PH2ST architecture: UNI spot + neighborhood features, dual-scale hypergraph, ViT global fusion, ST-prompt cross-attention refinement](/assets/images/paper/ph2st/fig_p006_01.png)
*Figure 1 — PH2ST overview: (a) UNI spot + neighborhood features, (b) dual-scale hypergraph + ViT global fusion, (c) ST-prompt projection with four sampling strategies, (d) cross-attention prompt-guided refinement.*

**Problem formulation.** Given $X \in \mathbb{R}^{n \times H \times W \times 3}$ spot patches from one WSI and a small set $Y_{\text{prompt}} \in \mathbb{R}^{n_{\text{prompt}} \times m}$ of *known* gene-expression vectors (default 10% of spots in that same WSI), predict $Y_{\text{pred}} \in \mathbb{R}^{(n-n_{\text{prompt}}) \times m}$ for the remaining spots. The per-WSI prompt is supplied *at inference*.

1. **Spot / neighborhood feature extraction.** Each 224×224 spot patch is encoded by frozen **UNI** → $\Phi_s \in \mathbb{R}^{n \times d}$. For each spot, a 1120×1120 region (α=5) is tiled into 25 non-overlapping 224×224 sub-patches and encoded → $\Phi_n \in \mathbb{R}^{n \times 25 \times d}$. CTransPath, CONCH 1.5, Virchow2, ResNet50 are tested as drop-in replacements.

2. **ST-prompt embedding.** During training, expression rows for non-prompt spots are masked to zero, leaving the prompt rows. A learnable projection produces

   $$\Phi_{\text{prompt}} = \text{LN}(\text{Dropout}(\text{FC}(\text{GELU}(Y_{\text{prompt}}\,W))))$$

3. **Dual-scale hypergraph construction.**
   - *Spot-level (slide-wide).* Incidence matrix $I(v_i, v_j) = \text{Norm}(\text{Sim}(v_i, v_j)) + \text{Norm}(\text{Pos}(v_i, v_j))$ where Sim is L2 between spot features and Pos is L2 between normalized spatial coordinates. Each spot's $P=4$ nearest combined-distance neighbors form a hyperedge.
   - *Neighbor-level (per spot).* For each spot, L2 distance among the 25 sub-patch features connects each sub-patch to its $P=4$ nearest → $n$ sub-hypergraphs.
   - Aggregation uses HypergraphConv (Bai et al. 2021): $\text{HConv}(\Phi, I) = D^{-1} I W_H D^{-1} I^T \Phi$, with ReLU + LN + Dropout. Outputs: $H_s$, $H_n$.

4. **Global ViT fusion.** Standard multi-head self-attention blocks consume $H_s$ and $H_n$ to produce slide-wide tokens $T_s, T_n \in \mathbb{R}^{n \times d}$.

5. **Prompt-guided cross-attention refinement.** ST-prompt embeddings act as **queries**, neighborhood tokens $T_n$ act as **keys / values**: $T_n^G = \text{CrossAttn}(\Phi_{\text{prompt}}, T_n)$. Fused representation $Z^G = T_n^G + T_s$ feeds a linear head producing per-gene predictions.

6. **Loss.** Triple MSE with consistency regularization: each branch (spot, neighbor) is trained both against the ground truth and against the fused prediction $p_Z$, weighted by $\lambda = 0.3$; $\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{spot}} + \mathcal{L}_{\text{neighbor}} + \mathcal{L}_Z$.

7. **Training / inference protocol.** Adam, lr=1e-4, step decay (step=50, γ=0.9), 200 epochs with early stopping (patience=20 on PCC), single RTX 4090. **Training:** randomly mask 30% of spots per WSI as prompt each iteration. **Inference:** 10% of spots per WSI used as prompt (default), drawn from one of four strategies — fully random, Poisson disc, sparse square, dense square.

## Experimental Results

### Datasets

| Dataset | Platform | Patients | Sections | Spots | Genes after filtering | Resolution |
|---|---|---|---|---|---|---|
| HER2+ breast cancer (Andersson 2021) | Legacy ST (10x) | 8 | 36 | 13,620 | 785 | 100 μm spot, 224×224 px patch |
| cSCC (Ji 2020, GSE144240) | 10x Visium | 4 | 12 | 23,205 | 171 | 55 μm spot, 224×224 px patch |

Preprocessing: per-spot counts normalized to CPM × 1e6 then log-transformed; top-1000 HVG retained then further reduced by gene-presence filter. **Splits:** 5-fold CV at *patient level* (following TRIPLEX's benchmark protocol to avoid same-patient train/test leakage). Within each held-out WSI, the 10% prompt spots are randomly drawn from that same slide.

### Main comparison (5-fold CV, 10% test-time prompt)

| Method | HER2+ MAE↓ | HER2+ CCC↑ | HER2+ PCC↑ | cSCC MAE↓ | cSCC CCC↑ | cSCC PCC↑ |
|---|---|---|---|---|---|---|
| ST-Net | 0.654±0.065 | 0.129±0.092 | 0.187±0.116 | 0.699±0.125 | 0.248±0.048 | 0.307±0.052 |
| HisToGene | 0.656±0.054 | 0.104±0.058 | 0.176±0.092 | 1.081±0.105 | 0.079±0.029 | 0.238±0.047 |
| Hist2ST | 0.679±0.075 | 0.106±0.070 | 0.201±0.117 | 1.076±0.108 | 0.078±0.044 | 0.224±0.067 |
| HGGEP | 0.643±0.067 | 0.123±0.081 | 0.190±0.094 | 0.736±0.161 | 0.125±0.096 | 0.173±0.084 |
| BLEEP | 0.742±0.049 | 0.177±0.010 | 0.202±0.046 | 0.913±0.101 | 0.170±0.009 | 0.174±0.008 |
| TRIPLEX | 0.543±0.110 | 0.225±0.131 | 0.318±0.147 | 0.702±0.306 | 0.263±0.130 | 0.431±0.060 |
| **PH2ST** | **0.516±0.096** | **0.256±0.148** | **0.338±0.139** | **0.698±0.320** | **0.312±0.147** | **0.479±0.103** |

All baselines except BLEEP are additionally fine-tuned on the same 10% prompt spots of each test WSI (final linear layer, 5 epochs) for fair comparison. BLEEP is excluded from fine-tuning because of its contrastive design.

![Per-gene PCC comparison with non-parametric significance on HER2+ and cSCC](/assets/images/paper/ph2st/fig_p008_01.png)
*Figure 2 — Per-gene PCC comparison with non-parametric significance on HER2+ and cSCC; PH2ST significantly outperforms all baselines on per-gene paired tests.*

![Qualitative spatial heatmaps of predicted vs ground-truth marker genes](/assets/images/paper/ph2st/fig_p009_01.png)
*Figure 3 — Predicted vs ground-truth spatial expression heatmaps for selected marker genes (TMSB10 / CISD3 / CD74 / COL6A2 on HER2+; MT-CO2 / PKP1 / RPS5 / SFN on cSCC) on representative slides.*

### Ablations

![Ablation curves: branch contributions, test-time prompt ratio, training-time prompt ratio on cSCC](/assets/images/paper/ph2st/fig_p010_01.png)
*Figure 4 — Ablations on cSCC: (a) fine-tuning baselines and branch ablation, (b) test-time prompt ratio, (c) training-time prompt ratio.*

- **Branches** (Fig. 4a dashed): removing either the spot branch or the neighbor branch degrades both PCC and CCC — both scales contribute.
- **Fine-tuning baselines** (Fig. 4a solid): fine-tuning on 10% prompts helps spot-based methods (ST-Net) but actively *hurts* some graph / ViT baselines — evidence that the prompt signal alone is hard to leverage without an architectural slot for it.
- **Training prompt ratio** (Fig. 4c): excessive training-time prompt causes diminishing returns / over-reliance on prompts versus histology. 30% is the picked sweet spot.
- **Test prompt ratio** (Fig. 4b): PCC monotonically increases from 0.338 @ 10% → **0.606 @ 90%** (random sampling, HER2+).
- **Encoder ablation** (Table 2, HER2+): ResNet50 PCC 0.180 ≪ CTransPath 0.320 ≈ CONCH 1.5 0.325 ≈ Virchow2 0.338 ≈ UNI 0.338. UNI wins on MAE / CCC and ties on PCC.

### Sampling-strategy robustness (HER2+)

Random ≈ Dense Square > Sparse Square > **Poisson Disc**, with the gap widening at higher prompt ratios. At 90%: PCC random=0.606, dense=0.607, sparse=0.601, **Poisson=0.492**. The authors interpret Poisson's weakness as an "island effect" — isolated prompts lacking neighborhood context, consistent with the cross-attention design that queries *neighboring* histology features.

![Prompt-region selection on pathologist tissue annotations](/assets/images/paper/ph2st/fig_p011_01.png)
*Figure 5 — Prompt-region selection: tumor-stroma boundary prompts yield the strongest downstream PCC; pure-tumor or pure-stroma prompts underperform — moderate tissue heterogeneity is more informative.*

## Limitations

**Authors admit:**
- Hypergraph construction is CPU-bound and not GPU-accelerated → training cost grows with WSI size.
- All experimental settings are simulation-based on existing low-resolution ST; "true" super-resolution / cellular-resolution evaluation requires datasets the authors do not have.

**Reviewer-noticed gaps that the post-reviewer should weigh heavily:**

1. **Within-slide spot leakage is the entire premise.** Test prompts and test labels are drawn from the *same WSI* in 5-fold patient-stratified CV. Patient-level CV prevents patient leakage; *spot-level* leakage within a slide is by design. A practitioner should **not** compare PH2ST's headline PCC to image-only image→ST PCC numbers from other papers without the "10% intra-slide oracle" caveat.
2. **No "PH2ST without the cross-attention prompt branch" ablation.** The contribution of the prompting mechanism is not isolated from that of UNI features + dual-scale hypergraph + ViT global fusion. Given the small Δ over TRIPLEX (+0.020 / +0.048 PCC), this is the experiment that would settle the central claim.
3. **Headline deltas are dwarfed by std devs.** HER2+: +0.020 PCC over TRIPLEX with std ±0.139; cSCC: +0.048 with std ±0.103. Reported significance is per-gene paired tests, not per-fold means.
4. **Poisson-disc sampling is markedly weaker** (PCC 0.492 vs 0.606 at 90% prompts on HER2+) — undermines the "super-resolution" framing, since super-resolution naturally implies spatially-spread prompt grids.
5. **Imputation / super-resolution / local-to-global are named but not evaluated as separate tasks** — all four sampling strategies are scored with the same held-out PCC; no comparison to dedicated ST-imputation baselines (gimVI, SpaGCN, Tangram).
6. **Only 2 small datasets, 2 diseases, n=12 patients total.** "Generalization" conclusions rest on a very thin base. No external-dataset transfer; no train-on-HER2 → test-on-cSCC; no Visium-HD generalization.
7. **Variance reporting is ambiguous.** Std devs appear to be across genes within a fold, not across folds, so per-fold variability that would let readers compute proper paired tests is not tabulated.
8. **Hyperparameter sensitivity unreported.** $P=4$ hyperedge size, $\alpha=5$ neighborhood, $\lambda=0.3$ loss weight are stated without sweep evidence.

## Why It Matters for Medical AI

The honest contribution is procedural rather than architectural: PH2ST shows that **if** you allow yourself a small fraction of measured spots on the test slide — which clinicians often have already — there is a clean architectural slot (cross-attention with prompt tokens as queries against neighborhood histology keys / values) for using them, and the resulting predictor monotonically improves with prompt budget all the way to PCC ≈ 0.6 on HER2+. For real deployment scenarios where partial ST is cheap (low-density Visium chip, manual annotation of a few ROIs), this is a more honest framing of the WSI→ST task than blind regression. The architectural specifics (dual-scale hypergraph, UNI features) appear to do less work than the prompt mechanism — but the paper does not run the ablation that would prove it, so practitioners should treat the framing as the contribution and the headline PCC delta as suggestive rather than conclusive.

## References

- Paper: [arXiv:2503.16816v2](https://arxiv.org/abs/2503.16816) (v2 posted 20 Apr 2025)
- Code: [github.com/NIUYI0511/PH2ST](https://github.com/NIUYI0511/PH2ST/)
- HER2+ dataset: Andersson et al. 2021 (Spatial transcriptomics of HER2-positive breast cancer)
- cSCC dataset: Ji et al. 2020, GSE144240 (Multimodal analysis of cutaneous squamous cell carcinoma)
- Compared baselines: ST-Net, HisToGene, Hist2ST, HGGEP, BLEEP, TRIPLEX
- Backbone: UNI histopathology foundation model (Chen et al. 2024); HypergraphConv (Bai et al. 2021)
