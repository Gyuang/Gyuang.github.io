---
title: "BioMARL: Biological Pathway Guided Gene Selection Through Collaborative Reinforcement Learning"
excerpt: "A two-stage KEGG-prior + multi-agent DQN gene selector wins 5/6 TCGA tasks at k=100 (BRCA-ER AUC 0.9706, LUAD AUC 0.8327) — but ablates only Phase 2 and ships an unflagged regression on OV."
categories:
  - Paper
tags:
  - BioMARL
  - Multi-Agent Reinforcement Learning
  - Gene Selection
  - KEGG Pathways
  - Graph Neural Network
  - TCGA
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- BioMARL frames HDLSS gene selection as a two-stage pipeline: a KEGG-pathway-weighted statistical pre-filter produces a candidate pool, then a per-gene DQN with a GNN state, centralized critic, shared synergy memory, and a perturbation-based ensemble reward estimator picks the final top-k.
- On 6 TCGA classification tasks at k = 100 it beats 8 baselines (K-Best, mRMR, LASSO, RFE, LASSONet, GFS, RRA, MCDM) on **5/6 datasets** — headline numbers: **BRCA-ER AUC 0.9706 ± 0.0198**, **BRCA-HER2 0.8357 ± 0.0296**, **LUAD 0.8327 ± 0.0601**, **OV 0.6439 ± 0.065**.
- The honest read: Phase 1 is never ablated, the full model *regresses* on OV vs. the −Memory variant (0.6439 < 0.6494) without acknowledgement, and the Vanderbilt gastric-cancer "real-world" validation is narrative-only with no AUC, no test split, and no baseline comparison.

## Motivation
The paper opens on the MammaPrint vs. 76-gene-Wang panel anecdote: two breast-cancer signatures aimed at the same outcome that share only **three** genes. The authors argue this instability is a direct symptom of biology-agnostic feature selection on HDLSS (high-dimension low-sample-size) data — small sample perturbations re-rank Lasso/χ² scores, so the selected panel changes between runs even when the downstream AUC does not. For clinical translation that instability is the actual blocker: druggable targets and reproducible assays demand pathway-coherent biomarkers, not just predictive ones.

KEGG provides decades of curated pathway annotations, but plugging that prior into selection without losing statistical discipline is non-trivial — most pathway-enrichment-then-filter approaches are heuristic. BioMARL frames selection as a cooperative decision process so that each gene can co-decide its inclusion based on both individual signal and pathway-level synergy with the other agents.

## Core Innovation
Three ideas the paper actually combines for the first time, in this order:

1. **KEGG-weighted statistical pre-filter (Phase 1).** Three importance scorers (χ², Random Forest importance, SVM ranking) are blended with validation-performance-derived weights, then each gene's meta-score is multiplied by a pathway bonus $(1 + \beta \log(1 + \bar{S}_p(g)))$ derived from how well classifiers trained *only* on each KEGG pathway's gene set perform. Unmapped genes pass through with no penalty. This collapses ~20,530 genes to a pathway-aware candidate pool $G_{pre}$.
2. **Per-gene DQN with a GNN state on a correlation⊕Jaccard graph (Phase 2).** Every gene is its own agent. The shared environment state is computed by a two-layer GNN on edges $E_{ij} = \rho C_{ij} + (1-\rho) J_{ij}$ combining Pearson correlation and KEGG-pathway Jaccard similarity ($\rho = 0.7$). A centralized critic $V(s)$ blends into the target Q-value $Q_{target} = \lambda_a (r_t + \gamma \max_a Q) + \lambda_b v_t$ with $(\lambda_a, \lambda_b) = (0.7, 0.3)$ — a baseline that cuts variance in the implicit policy gradients.
3. **Shared synergy memory + perturbation-based reward.** A pairwise synergy matrix $M$ accumulates $\Delta p$ from every successful subset and biases action probabilities via $P(a_i = 1) = \sigma(Q(s_i, 1) + \eta \sum_{j \in P_i} M[i,j])$. The per-agent reward is an ensemble meta-learner (RF + XGBoost + LightGBM + a small NN) that predicts $\Delta R_t$ from a bit-flip perturbation matrix $D_t$ in parallel, avoiding sequential re-evaluation. The composite reward $r_i = \omega r_{base} + \xi \Delta \phi_i + \zeta \Delta \psi_i$ with $(\omega, \xi, \zeta) = (0.5, 0.25, 0.25)$ adds pathway-centrality and coverage differentials.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|-------|----------|----------|----------|
| C1 | BioMARL outperforms 8 baselines on the majority of TCGA tasks | Figure 2 box-plots; Table 2 exact numbers | 5/6 TCGA tasks; not OV | ⭐⭐ |
| C2 | Each architectural component (reward, critic, memory) contributes positively | Table 2 ablation across BRCA(ER), BRCA(HER2), LUAD, OV | 4 datasets | ⭐⭐ — **OV regression for full vs −Mem (0.6439 < 0.6494) is not addressed** |
| C3 | Selected genes are biologically meaningful (pathway-coherent) | GO enrichment (Fig. 5); PR-split heatmap (Fig. 4); Table 3 literature review of 8 genes | BRCA, LUAD, OV | ⭐⭐ — counts depend on chosen p-threshold; Table 3 is curated, not systematic |
| C4 | BioMARL is robust to downstream classifier choice | Radar plot Fig. 6 on BRCA(TN) only | 1 dataset, 5 classifiers | ⭐ — single-dataset evidence |
| C5 | BioMARL is robust to train/test split | Hold-out study Fig. 3 on BRCA(ER) and LUAD | 2 datasets | ⭐⭐ |
| C6 | BioMARL-selected genes have prognostic (survival) value | KM plots Fig. 7 for one gene per dataset | 2 datasets, 2 genes | ⭐ — n = 1 gene per dataset is anecdotal |
| C7 | First framework to unify statistical FS with pathway knowledge via MARL | §1 contributions | — | ⭐⭐ — strictly true if "via MARL"; pathway-aware filtering broadly exists |
| **C8** | **Pathway-guided pre-filter is necessary** | **No ablation toggles Phase 1 on/off** | — | **unsupported — only Phase 2 components are ablated** |
| C9 | Real-world validation on Vanderbilt gastric-cancer cohort | §4.2 narrative — no AUC, no test split, no baseline | 1 private dataset | ⭐ — not reproducible from the paper |

**Honest read.** The Phase 2 ablation on three of four datasets is real and statistically meaningful (10 runs with reported variance). What is weaker than the abstract implies: (i) Phase 1 is *never* ablated, so we cannot say whether the pathway prior or the MARL block is doing the work; (ii) the full model regresses on OV (0.6439 vs. 0.6494 for −Mem), contradicting the "every component helps" narrative; (iii) the Vanderbilt gastric-cancer "real-world" validation is asserted rather than measured; (iv) all baselines share a Random Forest downstream — pairing each with its preferred classifier might narrow the gap; (v) only 6 tasks all from TCGA — no METABRIC, no microarray, no external transcriptomic cohort.

## Method & Architecture

![BioMARL two-stage framework](/assets/images/paper/pathway-guided-marl/fig_p003_01.png)
*Figure 1: BioMARL framework. (a) Pathway-guided statistical pre-filter using KEGG, three statistical scorers, and a validation-performance-weighted pathway bonus. (b) MARL refined selection — per-gene DQN with a GNN state on the correlation⊕Jaccard graph, centralized critic, shared synergy memory, and an ensemble-based Personal Performance Impact Estimator.*

**Phase 1 — Pathway-guided pre-filtering.** Three importance vectors $S_i$ from $f_i \in \{\chi^2, \text{RF importance}, \text{SVM ranking}\}$. For each KEGG pathway $p_i$ with gene set $G_i$, train a classifier on $D[G_i]$ and record $S_p(p_i)$. Weights $w_i = S_p(f_i) / \sum_j S_p(f_j)$ define the meta-score $m_g = \sum_i w_i S_{i,g}$. Pathway bonus: $\hat{s}_g = m_g \cdot (1 + \beta \log(1 + \bar{S}_p(g)))$ with $\beta = 0.2$. Threshold $G_{pre} = \{g : \hat{s}_g > \mu + 2\sigma\}$.

**Phase 2 — MARL refined selection.** Each gene is a binary agent.

- **State (GNN):** $E_{ij} = \rho C_{ij} + (1-\rho) J_{ij}$ with $\rho = 0.7$; two graph-conv layers, $s_t = \mathrm{pool}(\sigma(\tilde A h W^{(1)}))$, hidden dim 64.
- **Per-agent DQN:** 4-layer MLP (256-128-64-2), ReLU + layer norm, prioritized replay (buffer 1700, batch 64, 3000 exploration steps), target network every 50 steps, Huber loss, $\epsilon$-greedy ($\epsilon: 0.95 \to 0.1$, decay 0.99), lr 3e-4, $\gamma = 0.85$.
- **Centralized critic** $V(s)$ trained with MSE against observed global improvement $I_t$. Blended target $Q_{target} = \lambda_a (r_t + \gamma \max_a Q(s_{t+1}, a)) + \lambda_b v_t$, $(\lambda_a, \lambda_b) = (0.7, 0.3)$.
- **Shared memory** = (collaboration record $H$, synergy matrix $M \in \mathbb{R}^{d \times d}$). On a successful subset $S$ with improvement $\Delta p$: $H(S) \leftarrow \max(H(S), \Delta p)$, $M[i,j]\,{+}{=}\,\Delta p$. Action prior: $P(a_i = 1) = \sigma(Q(s_i, 1) + \eta \sum_{j \in P_i} M[i,j])$ with $\eta$ ramping 0.08 → 0.3; memory decays at 0.99.
- **Personal Performance Impact Estimator.** Flip each bit of $a_t$ to build perturbation matrix $D_t$; ensemble meta-learner $f_{meta}$ (RF + XGBoost + LightGBM + 256-128-64-32-1 NN with a linear meta) predicts $\Delta R_t = f_{meta}(D_t) - R_t$. Uncertainty $u_t$ gives $c_t = 1/(1+u_t)$ and $r_{base} = \Delta R_t \odot c_t - \log(1+u_t) + I_t \mathbf{1}$. Online update every 50 steps minimizes $\sum (f_{meta}(a_t) - R_t)^2 + \lambda \Omega(f_{meta})$.
- **Pathway-aware reward.** Centrality $\phi_i = n_i \sum_{p \in P_i} |G_{i,p}|$; aggregate $\Phi(S) = \sum_{i \in S} n_i \sum_{p \in P_i} |G_{i,p} \cap S|$; differential $\Delta \phi_i$ from toggling $i$. Coverage $\psi_p(S) = |S \cap G_p|/|G_p|$ with $\Delta \psi_i$.
- **Composite reward** per agent: $r_i = \omega r_{base} + \xi \Delta \phi_i + \zeta \Delta \psi_i$, $(\omega, \xi, \zeta) = (0.5, 0.25, 0.25)$.
- **Final ranking:** $w_i = \sum_t \gamma^{T-t}(Q^t_i(s_t,1) - Q^t_i(s_t,0))$; sort, take top $k = 100$.

## Experimental Results

### Main comparison (AUC at k = 100, Random Forest downstream, 10-run mean)

Exact numbers for the four datasets that appear in Table 2; the two remaining BRCA tasks (PR, TN) come from Figure 2 box-plots (means approximated). Baselines: K-Best, mRMR, LASSO, RFE, LASSONet, GFS, RRA, MCDM.

| Method | BRCA(ER) | BRCA(HER2) | BRCA(PR) | BRCA(TN) | LUAD | OV |
|--------|----------|------------|----------|----------|------|----|
| **BioMARL** | **0.9706 ± 0.0198** | **0.8357 ± 0.0296** | **≈ 0.95+** | **≈ 0.96+** | **0.8327 ± 0.0601** | 0.6439 ± 0.065 |
| Best baseline (Fig. 2) | < BioMARL | < BioMARL | ≈ BioMARL | < BioMARL | < BioMARL | **wins** (paper does not name it) |

![Main benchmark — AUC across 6 TCGA tasks](/assets/images/paper/pathway-guided-marl/page_006.png)
*Figure 2: AUC comparison of BioMARL vs. 8 baselines across BRCA-ER/HER2/PR/TN, LUAD, and OV (10 runs each). BioMARL wins 5/6; OV is the exception and the paper does not name the winning baseline in the prose.*

### Ablation (Table 2, exact numbers)

| Variant | Reward | Critic | Shared mem | BRCA(ER) | BRCA(HER2) | LUAD | OV |
|---|---|---|---|---|---|---|---|
| BioMARL−Rwd | ✗ | ✓ | ✓ | 0.9598 ± 0.023 | 0.8184 ± 0.061 | 0.7064 ± 0.107 | 0.6276 ± 0.106 |
| BioMARL−Crt | ✓ | ✗ | ✓ | 0.9605 ± 0.029 | 0.8073 ± 0.030 | 0.7623 ± 0.058 | 0.6298 ± 0.095 |
| BioMARL−Mem | ✓ | ✓ | ✗ | 0.9618 ± 0.0166 | 0.7914 ± 0.093 | 0.7712 ± 0.089 | **0.6494 ± 0.100** |
| **BioMARL (full)** | ✓ | ✓ | ✓ | **0.9706 ± 0.0198** | **0.8357 ± 0.0296** | **0.8327 ± 0.0601** | 0.6439 ± 0.065 |

The pathway-aware reward is the single highest-impact component on LUAD (+0.126 AUC over BioMARL−Rwd). Note that on OV the full model is *worse* than BioMARL−Mem (0.6439 vs. 0.6494) — this regression contradicts the "every component helps" framing and is not acknowledged in the text.

![Ablation table + qualitative panels](/assets/images/paper/pathway-guided-marl/page_007.png)
*Figure 3: Top — Table 2 Phase-2 ablation of personalized reward, centralized critic, and shared memory. Bottom — Fig. 4 heatmap of 100 BioMARL-selected genes separating PR+/PR− patients on BRCA; Fig. 5 GO-term enrichment counts at p ≤ 0.01 (BRCA-PR 90, LUAD 24, OV 23 — all highest among methods).*

### Robustness, downstream-agnostic behaviour, and survival

![Classifier-agnostic robustness + KM survival curves](/assets/images/paper/pathway-guided-marl/page_008.png)
*Figure 4: Fig. 6 (left) — classifier-agnostic robustness on BRCA(TN) across RF, XGBoost, Decision Tree, Ridge, SVM (radar). Fig. 7 (right) — KM survival curves for C1GALT1C1 stratifying BRCA-ER (p = 8.6e-4) and PIK3CD stratifying LUAD (p = 4.2e-5).*

- **Hold-out study (Fig. 3 in paper):** train/test splits 15/30/45/60% on BRCA(ER) and 25/40/50/60% on LUAD — BioMARL-selected features consistently beat the unfiltered baseline; LUAD shows much wider swings due to its tiny minority class (n = 21).
- **GO enrichment (Fig. 5 in paper):** 90 GO terms at p ≤ 0.01 on BRCA(PR), 24 on LUAD, 23 on OV — highest among compared methods on every task.
- **Literature case study (Table 3 in paper):** 8 BioMARL-unique genes (CYP3A7, NME1, FGA, GRB7, ERBB2, AGTR1, PSMC1, TUBB3) each have published breast-cancer associations and KEGG pathway hits — useful face-validity, but curated rather than systematic.

## Limitations

Author-acknowledged:
- HDLSS data scenarios stress MARL convergence — addressed by Phase 1 reduction.
- KEGG is incomplete — unmapped genes retain $m_g$ with no penalty.

Not addressed in the paper:
- **No Phase 1 ablation.** Whether the pathway-weighted statistical filter is load-bearing or whether MARL on the raw input could match the headline numbers is unknown.
- **OV regression unflagged.** Full BioMARL underperforms BioMARL−Mem on OV (0.6439 vs. 0.6494); the text claims uniform component contribution.
- **No comparison against newer MARL/RL feature selectors** (MARLFS, CAESAR variants, GAINS-style autoregressive selectors) at matched $k$.
- **Fixed $(\omega, \xi, \zeta) = (0.5, 0.25, 0.25)$** — no sensitivity study; the "balance" claim is unverified.
- **KEGG bias.** KEGG over-represents well-studied pathways (cancer, metabolism), so genes in poorly annotated pathways get smaller bonuses — baking in established biology and arguably *reducing* novel-biomarker discovery, the opposite of one stated motivation.
- **No computational cost reported.** A per-gene DQN + GNN + ensemble meta-learner + perturbation matrix on thousands of agents must be substantial; runtime/memory belong in any reproducibility audit.
- **Single-gene KM survival** is not the standard test — multivariate Cox regression on the full 100-gene panel would be.
- **No significance testing** between BioMARL and the second-best baseline per dataset in Figure 2 (no t-test, no Wilcoxon).
- **Vanderbilt gastric-cancer validation** has no AUC, no held-out test set, no baseline comparison — the "real-world" claim is asserted, not measured.
- **TCGA-only, Western-ancestry-skewed cohort.** LUAD and OV use very few class-2 samples (21 and 49), and the chosen survival cutoffs (<25 mo, >50 mo) discard intermediate-survival patients — a non-trivial label-selection effect.

## Why It Matters for Medical AI
Clinical biomarker discovery is gated less by accuracy than by *stability and biological coherence* — two signatures for the same outcome should overlap by more than three genes. BioMARL is a clean engineering example of pushing pathway priors into the *reward function* of a feature selector rather than only into a pre- or post-hoc filter, and the synergy-memory term is a concrete way to surface gene combinations that work together rather than gene rankings that happen to coincide. The cautionary points carry through too: an unablated Phase 1, an unflagged ablation regression, and a narrative "real-world" validation are all common patterns in genomic-ML papers, and the audit framework here (per-component ablation across four tasks, KM stratification, GO enrichment, literature review) is exactly the level of evidence a medical-AI reader should *expect* before trusting any HDLSS gene panel.

## References
- Paper: Azim et al., "Biological Pathway Guided Gene Selection Through Collaborative Reinforcement Learning," KDD '25, Toronto, August 3–7 2025.
- arXiv: https://arxiv.org/abs/2505.24155 (v1, 30 May 2025, cs.LG)
- DOI: https://doi.org/10.1145/3711896.3737198
- Code: https://github.com/ehtesam3154/bioMARL
- Background — KEGG: Kanehisa and Goto, "KEGG: Kyoto Encyclopedia of Genes and Genomes," Nucleic Acids Res., 2000.
- Related — MARLFS: Liu et al., "Automating Feature Subspace Exploration via Multi-Agent Reinforcement Learning," KDD 2019.
- Background — TCGA cohort and PAM50 subtyping: Parker et al., 2009.
