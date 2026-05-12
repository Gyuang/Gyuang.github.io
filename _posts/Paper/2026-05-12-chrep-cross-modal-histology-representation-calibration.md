---
title: "CHRep: Cross-modal Histology Representation and Post-hoc Calibration for Spatial Gene Expression Prediction"
excerpt: "Two-phase histology-to-ST pipeline that decouples representation learning from deployment-time bias correction, lifting PCC(ACG) on Alex+10x from 0.1949 to 0.2718 (+39.5%) over mclSTExp under strict LOSO."
categories:
  - Paper
tags:
  - CHRep
  - Spatial-Transcriptomics
  - Histology
  - Contrastive-Learning
  - Post-hoc-Calibration
  - LOSO
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- CHRep splits histology-to-spatial-gene-expression into two stages: a representation phase trained with **correlation-aware regression + symmetric image–gene contrastive alignment + coordinate-induced multi-hop topology**, and a frozen-backbone post-hoc calibration phase combining a similarity-weighted neighbor Estimate Module with a magnitude-regularized residual Correction Module.
- The calibration gallery and corrector are trained only on training slides, so the held-out slide is excluded from both — making this a strict LOSO protocol rather than the leakier "same-slide neighbors" common in retrieval-based ST predictors.
- Headline numbers: on Alex+10x, **PCC(ACG) rises from 0.1949 (mclSTExp) to 0.2718 (+39.5%)** with 9.7% MSE and 9.0% MAE reductions; CHRep also tops HAGE by **+9.8% PCC(ACG) on HER2+** and **+4.0% on cSCC**. The paper is transparent that **HAGE still wins HER2+ MSE (0.4830 vs 0.5616) and MAE (0.3606 vs 0.5002)**.

## Motivation
H&E whole-slide images are cheap and routinely scanned, while spatial transcriptomics (ST) is expensive and low-throughput — so predicting spot-level expression from histology is a high-leverage problem. But under leave-one-slide-out (LOSO), prior predictors (ST-Net, HisToGene, Hist2ST, THItoGene, BLEEP, mclSTExp, HAGE) degrade because of three coupled failures: (i) slide-level staining and scanner shifts perturb visual features, (ii) point-wise MSE/MAE objectives encourage mean-seeking predictions that flatten gene-wise variance and depress correlation even when absolute error looks fine, and (iii) immediate-neighbor message passing only captures local context and misses the higher-order mesoscopic organization of tissue.

CHRep's argument is structural: stable representation learning and systematic bias correction are different problems and should not be entangled inside a single end-to-end regressor. Phase 1 learns features that respect gene-wise correlation and tissue topology; Phase 2, with the backbone frozen, fixes the residual deployment-time bias using a retrieval-then-correct mechanism whose training data never includes the test slide.

## Core Innovation
1. **Correlation-aware regression loss.** Beyond MSE/MAE, the regression head adds a `(1 − PCC)` term so that the optimizer is explicitly penalized for the mean-seeking collapse that hurts gene-wise correlation. This is intended to address the "low PCC despite low MSE" failure mode of plain regressors.
2. **Symmetric image–gene contrastive alignment.** A coordinate-guided morphology feature `F_M` and a gene feature `F_G` are pulled together by a symmetric InfoNCE objective with learnable temperature `τ`, so that visual similarity reflects molecular similarity at the spot level.
3. **Coordinate-induced multi-hop topology regularization.** A kNN graph in coordinate space yields per-hop adjacency `A^{(h)}`; the multi-hop prior `A_topo = Σ_h α_h A^{(h)}` is matched against the gene-feature Gram matrix `S(F_G) = F_G F_G^⊤` via a Frobenius loss — pushing the representation to preserve mesoscopic neighborhood structure, not just immediate neighbors.
4. **Two-stage decoupling with strict LOSO.** After Phase 1, the regression head is discarded; only the frozen image embedding is reused. Phase 2 retrieves top-k training-spot neighbors in embedding space, softmax-weights their expression into a non-parametric estimate, and a small MLP `r_η` produces a residual `Δ_i` with explicit magnitude regularization `λ_Δ ||Δ_i||_2^2`. The held-out slide enters neither the gallery nor the corrector training set.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|-------|----------|------------|----------|
| C1 | CHRep beats prior methods in PCC under strict LOSO | Table I: best PCC(ACG) and PCC(HEG@50) on all three cohorts | HER2+, cSCC, Alex+10x | ⭐⭐⭐ |
| C2 | +39.5% PCC(ACG) over mclSTExp on Alex+10x | Table I: 0.1949 → 0.2718 (0.2718/0.1949 − 1 ≈ 39.5%) | Alex+10x | ⭐⭐⭐ |
| C3 | +4.0% / +9.8% PCC(ACG) over HAGE on cSCC / HER2+ | Table I: 0.3397 → 0.3532; 0.2489 → 0.2733 | cSCC, HER2+ | ⭐⭐⭐ |
| C4 | 9.7% MSE and 9.0% MAE reduction over mclSTExp on Alex+10x | Table I: 0.2329 → 0.2102; 0.3897 → 0.3545 | Alex+10x | ⭐⭐⭐ |
| C5 | Decoupling representation learning and post-hoc calibration is beneficial | Table II: full > estimate-only / correction-only / no-constraint variants | Alex+10x only | ⭐⭐ |
| C6 | All three representation objectives are complementary | Table III: full objective beats every two-of-three variant | cSCC only | ⭐⭐ |
| C7 | Multi-hop topology beats immediate-neighbor smoothness | Inferred from Table III; `H_hop` and `α_h` not isolated | cSCC | ⭐ |
| C8 | Gains reflect stable cross-slide generalization, not lucky slides | Figs. 2, 4 per-slide boxplots show higher medians and tighter IQRs | HER2+, cSCC | ⭐⭐ |
| C9 | CHRep produces spatial patterns closer to ground truth | Fig. 5: SBSN, PAICS, CELF1 on one cSCC slide | cSCC qualitative | ⭐ |
| C10 | Correlation-aware loss avoids mean-seeking attenuation | Discussion only; no ablation removing the PCC term | — | ⭐ |
| C11 | CHRep does **not** always minimize MSE/MAE — HAGE wins HER2+ MSE 0.4830 vs 0.5616 and MAE 0.3606 vs 0.5002 | Table I, explicitly acknowledged in §III-E | HER2+ | ⭐⭐⭐ (transparent negative result) |

**Honest read.** Headline gains (C1–C4) arithmetically check out and come with reported standard deviations (e.g., ±0.0018 on HER2+ PCC(ACG)). But the ablations are localized: calibration ablation is Alex+10x-only, representation ablation is cSCC-only, so no single dataset shows that *every* ingredient helps. Critically, **the image backbone `f_θ` is never named** (no architecture, parameter count, or pretraining source given), which is a major reproducibility gap when the entire Phase 2 retrieval mechanism depends on embedding quality — and which makes it impossible to know whether the gains would survive plugging a modern pathology foundation encoder (UNI, CONCH, GigaPath, Phikon) into the baselines. HAGE is also missing on Alex+10x for lack of public code, which is exactly the cohort where CHRep claims its largest gain. No statistical test against the runner-up is reported, and the 39.5% relative figure is computed off a low absolute base (0.1949), where relative improvements amplify small absolute differences. The paper's candid acknowledgement of losing HER2+ MSE/MAE to HAGE (C11) is the most trust-building element here.

## Method & Architecture

![CHRep two-phase pipeline: training-time representation learning with three branches and three losses, followed by inference-time post-hoc calibration with a frozen encoder, Estimate Module, and Correction Module](/assets/images/paper/chrep/fig_p003_01.png)
*Figure 1: CHRep two-phase pipeline — training combines an H&E branch, a coordinate branch, and a gene branch under correlation-aware regression + symmetric contrastive alignment + coordinate-induced multi-hop topology; inference freezes the encoder, retrieves top-k training spots for a similarity-weighted Estimate, then adds a magnitude-regularized residual Correction.*

**Phase 1 — representation learning.** For each spot triplet `(x, p, g)` with an H&E patch `x ∈ R^{3×224×224}`, coordinate `p ∈ R^2`, and HVG expression `g ∈ R^G`, counts are library-size normalized, log-transformed, and per-gene standardized using train-fold μ, σ to give `g̃`.

- Histology branch: an (unnamed) image encoder `f_θ` produces `F_H(i) = f_θ(x_i)`; a regression head `h_φ` outputs `ĝ_i = h_φ(F_H(i))`.
- Coordinate branch: position embedding `e_ψ` lifts `p_i` to `F_C(i)`, fused with `F_H` into the coordinate-guided morphology feature `F_M(i)`. Coordinates are **not** fed as image-channel inputs.
- Gene branch: `g_ω(g̃_i)` → `F_G(i)`.

Three losses combine in the total objective `L = λ_con L_con + λ_reg L_reg + λ_spa L_spa`:

$$\mathcal{L}_{reg} = \frac{1}{BG}\sum_i \|\hat g_i - \tilde g_i\|_2^2 + \lambda_{mae}\frac{1}{BG}\sum_i\|\hat g_i - \tilde g_i\|_1 + \lambda_{PCC}(1 - \mathrm{PCC})$$

`L_con = ½(L_{M→G} + L_{G→M})` is a symmetric InfoNCE on projection heads of `F_M` and `F_G` with learnable temperature `τ`, and `L_spa = ||S̃(F_G) − Ã_topo||_F^2` matches the (normalized) gene-feature Gram matrix to the multi-hop coordinate prior `A_topo = Σ_h α_h A^{(h)}`.

**Phase 2 — post-hoc calibration (frozen backbone).** The regression head `h_φ` is discarded; only `z_i := F_H(i)` is reused.

1. **Estimate Module.** For query `i`, cosine similarities `s_ij = z_i^⊤ z_j` against training spots, top-k neighbors `N_k(i) ⊆ D_tr \ {i}`, softmax weights `w_ij = exp(s_ij/τ_t) / Σ exp(s_ij'/τ_t)`, and a non-parametric estimate `ĝ^{(E)}_i = Σ w_ij g̃_j`. Self-copy is explicitly blocked.
2. **Correction Module.** A small MLP `r_η(z_i) = Δ_i` provides the final prediction `ĝ_i = ĝ^{(E)}_i + Δ_i`.
3. **Magnitude regularization.** `L_Δ = (1/B) Σ ||Δ_i||_2^2`. Calibration loss `L_cal = L^{cal}_reg + λ_Δ L_Δ` reuses the MSE+MAE+PCC form on calibrated predictions; only `η` is updated.
4. **Strict LOSO.** The held-out slide is excluded from both the gallery bank and corrector training data.

## Experimental Results

### Main comparison (Table I, LOSO, standardized space)

| Method | HER2+ PCC(ACG) | HER2+ PCC(HEG@50) | HER2+ MSE | HER2+ MAE | cSCC PCC(ACG) | cSCC PCC(HEG@50) | cSCC MSE | cSCC MAE | Alex+10x PCC(ACG) | Alex+10x PCC(HEG@50) | Alex+10x MSE | Alex+10x MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ST-Net | 0.0561 ± 0.017 | 0.0134 ± 0.013 | 0.5312 | 0.6306 | 0.0012 ± 0.022 | 0.0018 | 0.6806 | 0.6404 | 0.0009 | 0.0452 | 0.4721 | 0.5042 |
| HisToGene | 0.0842 | 0.0711 | 0.5202 | 0.6422 | 0.0771 | 0.0919 | 0.6805 | 0.6234 | 0.0618 | 0.0984 | 0.4565 | 0.4973 |
| Hist2ST | 0.1443 | 0.1849 | 0.5135 | 0.6087 | 0.1838 | 0.2175 | 0.6748 | 0.6107 | 0.1299 | 0.1784 | 0.3788 | 0.4492 |
| THItoGene | 0.1726 | 0.2809 | 0.5012 | 0.5956 | 0.2373 | 0.2719 | 0.6546 | 0.6012 | 0.1384 | 0.2156 | 0.3672 | 0.4315 |
| BLEEP | 0.1873 | 0.2909 | 0.6015 | 0.5824 | 0.2449 | 0.3122 | 0.5163 | 0.5399 | 0.1552 | 0.2825 | 0.2593 | 0.4050 |
| mclSTExp | 0.2304 | 0.3866 | 0.5897 | 0.5813 | 0.3235 | 0.4261 | 0.4302 | 0.5208 | 0.1949 | 0.3611 | 0.2329 | 0.3897 |
| HAGE | 0.2489 | 0.4458 | **0.4830** | **0.3606** | 0.3397 | 0.4607 | 0.4248 | **0.3296** | — | — | — | — |
| **CHRep (Ours)** | **0.2733 ± 0.0018** | **0.4634 ± 0.0020** | 0.5616 | 0.5002 | **0.3532 ± 0.004** | **0.4682 ± 0.005** | **0.4183** | 0.4048 | **0.2718 ± 0.016** | **0.4659 ± 0.020** | **0.2102 ± 0.033** | **0.3545 ± 0.010** |

CHRep wins every PCC column across all three cohorts. The MSE/MAE picture is not uniform: HAGE retains the HER2+ error wins (0.4830 vs 0.5616 MSE; 0.3606 vs 0.5002 MAE) and the cSCC MAE win (0.3296 vs 0.4048). The paper's positioning is therefore "best correlation, not always best absolute error" — and explicitly attributes the gap to a magnitude-vs-correlation tension on HER2+.

![Per-slide gene-wise PCC distributions on HER2+ across 32 sections](/assets/images/paper/chrep/fig_p006_01.png)
*Figure 2: Per-slide gene-wise PCC distribution on HER2+ (785 genes); CHRep shows higher medians and tighter IQRs across most slides versus representative baselines.*

![PCC at varying held-out gene-set sizes K on HER2+](/assets/images/paper/chrep/fig_p007_01.png)
*Figure 3: HER2+ PCC across gene sets (ACG, HEG@50/100/200) with error bars across folds — CHRep stays on top throughout the gene-set sweep.*

![Per-slide gene-wise PCC distributions on cSCC across 12 sections](/assets/images/paper/chrep/fig_p008_01.png)
*Figure 4: Per-slide gene-wise PCC distribution on cSCC under LOSO; CHRep's gain is broadly distributed across held-out slides rather than concentrated on a single favorable section.*

![Qualitative spatial expression maps for SBSN, PAICS, CELF1](/assets/images/paper/chrep/fig_p009_01.png)
*Figure 5: Held-out cSCC slide — ground-truth vs CHRep vs mclSTExp spatial expression for SBSN, PAICS, CELF1; red boxes highlight regions where CHRep recovers localized high-expression structure more faithfully.*

### Ablations

**Post-hoc calibration (Table II, Alex+10x):**

| Variant | PCC(ACG) | PCC(HEG@50) | MSE | MAE |
|---|---|---|---|---|
| Estimate only | 0.2079 | — | — | — |
| Correction only | 0.2434 | — | 0.2300 | — |
| Estimate + Correction (no constraint) | 0.2604 | — | 0.2652 | — |
| **Full (Estimate + Correction + magnitude reg.)** | **0.2676** | **0.4573** | **0.2150** | **0.3551** |

The pattern is instructive: Estimate alone is conservative and over-smooths; Correction alone over-adjusts to slide noise (PCC up, MSE up); only the magnitude-regularized combination gets both PCC and MSE moving in the right direction together.

**Representation objective (Table III, cSCC):**

| Variant | PCC(ACG) | PCC(HEG@50) | MSE | MAE |
|---|---|---|---|---|
| Regression + Contrastive (no topology) | 0.3412 | 0.4482 | — | — |
| Topology + Regression (no contrastive) | 0.3321 | 0.4272 | — | 0.4103 |
| Contrastive + Topology (no regression) | 0.3482 | 0.4563 | — | — |
| **Full (all three)** | **0.3522** | **0.4663** | **0.4191** | **0.4051** |

Each two-of-three variant trails the full objective, but margins are tight (≤ 0.02 PCC) and the topology-only-removal variant is the closest, suggesting the three losses are complementary rather than redundant — though no single dataset shows this across cohorts.

## Limitations

**Authors acknowledge:**
- CHRep does not always minimize MSE/MAE; HAGE wins HER2+ error metrics and cSCC MAE.
- HAGE could not be reproduced on Alex+10x because no official code is available.
- Table II/III full-model numbers differ slightly from Table I because they come from independently repeated experiment batches.

**Gaps I noticed:**
- **The image backbone `f_θ` is never named** — no architecture, parameter count, or pretraining source. Because Phase 2 retrieval is entirely embedding-quality-bound, this is the single most important reproducibility omission. It also leaves the obvious comparison — does CHRep still win when every baseline uses a modern pathology foundation encoder (UNI, CONCH, GigaPath, Phikon)? — unanswered.
- Hyperparameters `λ_con`, `λ_reg`, `λ_spa`, `λ_Δ`, `λ_PCC`, `λ_mae`, top-k, `τ_t`, `H_hop`, `α_h` are referenced in equations but not tabulated.
- Localized ablations: calibration only on Alex+10x, representation only on cSCC. No single cohort shows every ingredient helps; no ablation isolates the PCC term in `L_reg`, the multi-hop depth `H_hop`, the `α_h` schedule, or top-k / `τ_t`.
- HAGE missing on Alex+10x weakens the strongest-baseline narrative exactly where the largest gain is reported.
- Cohort sizes are small (9–32 sections). LOSO on 9 Alex+10x slides has high per-fold variance, and the +39.5% relative gain is computed off a low base PCC of 0.1949.
- No statistical significance test against the runner-up.
- No external validation cohort, no cross-cohort transfer (e.g., train cSCC → deploy HER2+), no held-out tissue type.
- No code or model release mentioned in the manuscript.
- No analysis of which gene categories (cell-type markers, signaling, housekeeping) benefit most — biological interpretability is left implicit.

## Why It Matters for Medical AI
ST is one of the most directly clinically actionable molecular modalities — it preserves spatial context that bulk and single-cell sequencing destroy — but the throughput cost makes large-cohort molecular pathology impractical. A reliable H&E → ST predictor would convert millions of archived slides into molecular substrates for survival modeling, response prediction, and biomarker discovery. CHRep's contribution is methodological honesty about *what is hard* in this problem: it is not modeling capacity, it is the slide-level distribution shift that strict LOSO exposes. The post-hoc calibration design — train-only gallery, train-only corrector, frozen embedding — is a transferable recipe that other histology-to-omics models can adopt without retraining, and the candid acknowledgement of the HAGE MSE/MAE loss makes the paper's positioning easier to trust than the more common "best on every metric" framing. The unnamed backbone is the audit gap that prevents wholesale adoption; once that is closed (and ideally swapped for a UNI/CONCH-class encoder), the design is poised to become a default baseline for cross-slide ST prediction.

## References
- **Paper**: CHRep: Cross-modal Histology Representation and Post-hoc Calibration for Spatial Gene Expression Prediction (arXiv:2604.21573v1, 2026)
- **Authors**: Changfan Wang, Xinran Wang, Donghai Liu, Fei Su, Lulu Sun, Zhicheng Zhao, Zhu Meng (BUPT + Peking University Third Hospital)
- **Code**: not released in the manuscript
- **Related work**: ST-Net (Genome Medicine 2020), HisToGene (Briefings in Bioinformatics 2022), Hist2ST (Briefings in Bioinformatics 2022), THItoGene (Briefings in Bioinformatics 2024), BLEEP (NeurIPS 2023), mclSTExp (Briefings in Bioinformatics 2024), HAGE
- **Datasets**: HER2+ breast cancer (Andersson et al. 2020), cSCC (Ji et al. Cell 2020), Alex+10x Visium (Wu et al. Nature Genetics 2021)
