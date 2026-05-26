---
title: "DeepSpot2Cell: Predicting Virtual Single-Cell Spatial Transcriptomics from H&E images using Spot-Level Supervision"
excerpt: "A DeepSet over (cell, spot, neighbor) tiles encoded by Phikon-v2 lifts top-100 single-cell Pearson r over scstGCN by +22% (lung ISin), +46% (lung OOS), +65% (breast OOS), +38% (pancreas OOS) — but every 'Visium' label is pseudo-Visium synthesized from Xenium, the per-cell branch is never directly supervised, and the abstract cherry-picks the OOS column of each organ."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/deepspot2cell/
tags:
  - DeepSpot2Cell
  - Spatial-Transcriptomics
  - DeepSets
  - Pathology-Foundation-Model
  - Phikon-v2
  - Virtual-Single-Cell
  - Xenium
  - Visium
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- DeepSpot2Cell models each Visium-style spot as a **bag of cells** and applies a **DeepSet sum aggregator** over per-cell pathology-foundation-model (PFM) embeddings of three tiles — the cell, its host spot, and neighboring spots — supervised only by the spot-level expression total. At inference, the per-cell branch alone is read out to produce virtual single-cell expression.
- The backbone choice — **Phikon-v2** over UNI and H-Optimus-0 — comes from a **single-dataset (lung ISin) ablation**, and the "sum is principled because transcripts add" argument is rhetorical: a nonlinear per-gene head $\rho_\text{gene}$ means $\sum_j \rho_\text{gene}(h_j) \neq \rho_\text{gene}(\sum_j h_j)$, so the per-cell inference path is **never directly supervised**.
- Abstract headlines — **+46% lung, +65% breast, +38% pancreas** — are the **maxima of the result matrix** (OOS columns), not across-the-board gains. Pancreas OOS rests on **n=2 samples with absolute r=0.11**. There is also no real Visium evaluation: all "spot-level" supervision is **pseudo-Visium synthesized from Xenium** by summing transcripts within 55 µm circular regions.

## Motivation

Spot-based Visium data are abundant (e.g., the 7,000-patient MOSAIC cohort) but each spot pools 1–10 cells, while Xenium-style single-cell ST is costly, low-sensitivity, and gene-panel-restricted. Prior H&E→ST work either predicts spot-level expression (DeepSpot, BLEEP, HisToGene) or generates super-pixel grids (iStar, scstGCN) that must be averaged post-hoc onto cell bounding boxes — a lossy approximation. The clinical pull: a cheap, retrospective way to attach virtual single-cell profiles to any H&E slide would let researchers mine pathology archives for cell-type-specific biomarker signal without re-sequencing tissue. DeepSpot2Cell positions itself as a *deconvolution by construction* alternative to super-resolution — train on what we have (spot totals), evaluate what we want (per-cell expression).

## Core Innovation

- **Cells as a permutation-invariant set per spot.** Aggregation uses a DeepSet sum, deliberately chosen so the spot prediction is a sum over cells — mirroring the additivity of transcript counts. (Caveat: this clean correspondence only holds *before* the nonlinear $\rho_\text{gene}$ head — see audit.)
- **Three-tile multi-scale context.** Each cell is encoded together with its host spot tile and a neighbor spot tile, all by a *frozen* PFM. The cell head $\varphi_\text{cell}$ and a shared spot/neighbor head $\varphi_\text{spot}$ project these into a concatenated per-cell representation.
- **Train-on-spot, infer-on-cell.** The model is trained with spot-level MSE only, but at inference the per-cell branch $\rho_\text{gene}(h_j)$ is read out alone — never supervised directly, never aggregated. This is the methodological gamble that the paper rides on.
- **PFM ablation picks Phikon-v2.** Over UNI and H-Optimus-0 on lung ISin (one dataset, one split).

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | DeepSpot2Cell improves single-cell expression correlation over prior super-resolution baselines | Table 1 (top-100); Tables 3, 4 (top-50/200); Figure 5 (correlation curves) | Lung, breast, pancreas Xenium (HEST-1k) | ⭐⭐⭐ — consistent direction across 3 organs × 4 splits × 3 top-K cutoffs; bootstrapped SE 0.00–0.01; non-overlapping with baselines on most cells. |
| C2 | "+46% improvement on lung top-100" (abstract) | Lung OOS row, Table 1: 0.24 → 0.35 = +46% relative | Lung OOS only | ⭐⭐ — the 46% is the OOS lung delta, not across-the-board. Lung ISin is +22%, lung ISout is +17%. The abstract sells the strongest sub-cell of the matrix. |
| C3 | "+65% on breast" (abstract) | Best matching delta is breast OOS 0.17 → 0.28 (+65%) | Breast OOS only | ⭐⭐ — again the cherry of the matrix; breast ISin is only +16% (0.37 → 0.43) and breast ISout +24% (0.33 → 0.41). |
| C4 | "+38% on pancreas" | Pancreas ISin 0.29 → 0.32 (+10%), ISout 0.14 → 0.16 (+14%), OOS 0.08 → 0.11 (+38%) — so +38% is the OOS column | Pancreas (n=2 samples) | ⭐ — only 2 pancreatic samples and the smallest absolute correlations in the table (r=0.11 is itself weak); statistically these effects are tiny in absolute terms. |
| C5 | Model does not overfit to spot-level signal (ISin ≈ ISout) | Table 1: lung 0.39 vs 0.41; breast 0.43 vs 0.41; pancreas 0.32 vs 0.16 | All three | ⭐⭐ — holds for lung and breast; **breaks for pancreas** (0.32 → 0.16, a 50% drop ISin→ISout), undermining universality. |
| C6 | DeepSet sum aggregation is principled because it matches transcript additivity | Section 3.1 conceptual argument + ablation in Figure 3 showing sum > mean > GRU | Lung only | ⭐⭐ — empirical support is one ablation on one dataset; the theoretical "sum of cells = spot" argument is loose because $\rho_\text{gene}$ is nonlinear, so $\sum_j \rho_\text{gene}(h_j) \neq \rho_\text{gene}(\sum_j h_j)$. |
| C7 | Generalizes to OOD samples | Breast OOD: 0.37 vs scstGCN 0.24 (Table 1, OOD column) | Breast OOD only | ⭐⭐ — "OOD" here means same tissue, same vendor (10x Genomics FFPE breast), different panel. No cross-organ, cross-scanner, or cross-institution OOD test. |
| C8 | Phikon-v2 outperforms UNI / H-Optimus-0 as PFM backbone | Figure 3 bar chart (PFM panel) | Lung ISin only | ⭐ — single dataset, single split, no SE on the bars in the readable camera-ready text. |
| C9 | Inferring cell-level expression from spot-level supervision is well-posed via DeepSets | Architecture description + empirical correlations | All splits | ⭐⭐ — single-cell correlations top out at r ≈ 0.39 on the easiest (ISin lung) setting and drop sharply in OOD/OOS. The spot→cell decomposition is mathematically under-determined; the model is implicitly relying on H&E morphology to break the tie, restricting it to genes whose expression is visible in histology. The paper acknowledges this only obliquely in "future work." |

## Method & Architecture

![DeepSpot2Cell architecture: per-cell, per-spot, and per-neighbor tiles encoded by a frozen pathology FM (Phikon-v2), projected through two MLP heads, concatenated, summed across cells in a spot via DeepSet aggregation, and decoded to gene expression](/assets/images/paper/deepspot2cell/fig_p003_01.png)

*Figure 1: DeepSpot2Cell predicts virtual single-cell ST by encoding three tiles (cell, host spot, neighbor spots) through a pathology foundation model and aggregating per-cell embeddings with a DeepSet sum before predicting spot-level expression. At inference, the per-cell branch alone yields virtual single-cell predictions — a path never directly supervised during training.*

Step-by-step:

1. **Cell segmentation (preprocessing, not learned).** Nuclei come from Xenium ground truth in the benchmark; in deployment, CellViT (as used in HEST-1k) is the assumed substitute. Each cell tile is the smallest square fully containing the segmented cell.
2. **Pseudo-spot construction.** Tile the H&E (20×) into 224×224-px non-overlapping tiles centered on circular 160-px pseudospots spaced 224 px apart; a cell $j$ is assigned to spot $i$ if its nucleus is ≥10 µm inside the spot boundary. Spot expression = sum of contained cells' Xenium counts.
3. **Triple-input encoding via frozen PFM** for each cell $j$ in spot $i$: cell tile $x_\text{cell}^j$, host spot tile $x_\text{spot}^i$, neighbor spot tile $x_\text{neighbor}^i$ — all encoded by Phikon-v2 (UNI and H-Optimus-0 benchmarked).
4. **Two MLP projection heads.** $\varphi_\text{cell}$ on the cell-tile embedding; a shared $\varphi_\text{spot}$ on the spot and neighbor tiles. Concatenate $h_j = \text{Concat}(\varphi_\text{cell}(x_\text{cell}^j), \varphi_\text{spot}(x_\text{spot}^i), \varphi_\text{spot}(x_\text{neighbor}^i))$.
5. **DeepSet sum aggregation, then per-gene head:**

$$\hat{s}_i = \rho_\text{gene}\Big(\sum_{j \in C_i} h_j\Big)$$

with $\rho_\text{gene}$ a two-layer MLP producing a per-spot gene-expression vector $\hat{s}_i \in \mathbb{R}^G$.

6. **Training loss:** MSE between $\hat{s}_i$ and the observed (pseudo-)spot expression $s_i$:

$$\mathcal{L} = \frac{1}{N}\sum_i \|\hat{s}_i - s_i\|^2$$

7. **Inference for single-cell prediction.** Evaluate only the cell branch $\rho_\text{gene}(h_j)$ per cell — **not** the sum. This is the critical inconsistency: because $\rho_\text{gene}$ is nonlinear, the per-cell predictions are not constrained at training time to sum to the spot total, and the network never sees a per-cell target.
8. **Hyperparameters.** Adam, lr $= 10^{-4}$, batch size 256 spots, dropout 0.3 on $\varphi$ and $\rho_\text{gene}$ MLPs, early stopping on validation loss, single NVIDIA RTX 4090.

## Experimental Results

All datasets are HEST-1k Xenium subsets (Vannan 2023 lung; Janesick 2023 + FFPE breast; 10x Xenium human multi-tissue + FFPE pancreas). Preprocessing: drop genes expressed in <20 cells; remove blank/negative controls; per-spot scale to 10,000 counts then `log1p`. **All "spot" supervision is pseudo-Visium synthesized from Xenium by summing transcripts within 55 µm regions — no real Visium spot is ever used.**

Main quantitative table (top-100 genes, mean Pearson r with bootstrapped SE; reproduced from Table 1):

| Model | Lung ISin | Lung ISout | Lung OOS | Breast ISin | Breast ISout | Breast OOS | Breast OOD | Pancreas ISin | Pancreas ISout | Pancreas OOS |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.20 (0.01) | 0.24 (0.00) | 0.19 (0.00) | 0.30 (0.01) | 0.34 (0.01) | 0.14 (0.00) | 0.25 (0.01) | 0.18 (0.00) | 0.09 (0.00) | 0.10 (0.00) |
| iStar | 0.28 (0.01) | 0.28 (0.00) | 0.15 (0.00) | 0.34 (0.01) | 0.25 (0.01) | 0.10 (0.00) | 0.17 (0.00) | 0.28 (0.01) | 0.13 (0.00) | 0.10 (0.00) |
| scstGCN | 0.32 (0.01) | 0.35 (0.01) | 0.24 (0.01) | 0.37 (0.01) | 0.33 (0.01) | 0.17 (0.00) | 0.24 (0.01) | 0.29 (0.01) | 0.14 (0.00) | 0.08 (0.00) |
| **DeepSpot2Cell** | **0.39 (0.01)** | **0.41 (0.01)** | **0.35 (0.01)** | **0.43 (0.01)** | **0.41 (0.01)** | **0.28 (0.01)** | **0.37 (0.01)** | **0.32 (0.01)** | **0.16 (0.00)** | **0.11 (0.00)** |

Tables 3 and 4 (top-50 and top-200 genes) preserve this ordering across all 10 cells of the matrix.

![Table 1 — top-100 gene Pearson correlation across lung/breast/pancreas, ISin/ISout/OOS/OOD splits](/assets/images/paper/deepspot2cell/page_004.png)

*Figure 2: Page 4 — Table 1 in context. DeepSpot2Cell dominates every cell at top-100, but headline percentages (+46/+65/+38) in the abstract are read from disparate OOS columns, and absolute correlations remain weak in the harder splits (pancreas OOS r = 0.11).*

**Qualitative (Figure 2 in the paper, slide NCBI867, MSLN — NSCLC marker).** Spot-level r jumps from 0.21 (iStar) / 0.30 (scstGCN) to 0.45 (DeepSpot2Cell); cell-level r 0.43 / 0.51 / 0.67. DeepSpot2Cell predictions look spatially coherent vs the diffuse blobs from the super-resolution baselines.

![MSLN qualitative comparison and ablation panels on page 5](/assets/images/paper/deepspot2cell/page_005.png)

*Figure 3: Page 5 — MSLN qualitative (Figure 2) and the ablation panels of Figure 3 (input context, PFM choice, aggregation operator). Sum > mean > GRU; cell+spot+neighbors > cell+spot > cell; Phikon-v2 > H-Optimus-0 > UNI on lung ISin.*

![High-resolution MSLN qualitative panel (Figure 2 in the paper)](/assets/images/paper/deepspot2cell/fig_p005_01.png)

*Figure 4: MSLN expression on lung slide NCBI867. iStar (r = 0.21) and scstGCN (r = 0.30) produce diffuse predictions; DeepSpot2Cell (r = 0.45) recovers spatially coherent expression matching the Xenium ground truth.*

**Generalization framing.** iStar and scstGCN drop below the MLP baseline on breast OOD (iStar 0.17, scstGCN 0.24 vs MLP 0.25 vs DeepSpot2Cell 0.37 on top-100), which the authors use to argue those models overfit to training images while DeepSpot2Cell learns transferable mappings. Note "OOD" here is still 10x Genomics FFPE breast cancer — a modest panel shift, not a tissue, scanner, or institution shift.

## Limitations

**Acknowledged by the authors:**

- Pseudo-Visium spots from Xenium ≠ real Visium measurements.
- Old ~300-gene Xenium panels; not the newer 5k panel.
- Deployment assumes accurate cell segmentation; CellViT errors will propagate.
- Downstream biological utility of "virtual cells" is unverified.

**Not addressed / additional concerns:**

- **Aggregation–prediction non-commutativity (the core methodological gamble).** Training optimizes $\hat{s}_i = \rho_\text{gene}(\sum_j h_j)$ but inference reads $\rho_\text{gene}(h_j)$ per cell. Because $\rho_\text{gene}$ is nonlinear, $\sum_j \rho_\text{gene}(h_j) \neq \rho_\text{gene}(\sum_j h_j)$ — the per-cell prediction path is never directly supervised. The rhetorical link "sum aggregation matches transcript additivity" only holds at the linear/pre-head stage. The paper should ablate an alternative architecture in which $\rho_\text{gene}$ is applied per cell and then summed — making training and inference consistent.
- **Inverse-problem ill-posedness.** Spot-level loss admits infinitely many per-cell decompositions. No analysis of which genes are decomposable in principle vs which are forever indistinguishable from H&E. The model will confidently output per-cell expression for genes whose signal is invisible in morphology — a real **hallucination risk** for genes uncorrelated with cell-type/tissue context.
- **Missing baselines.** No comparison to spot-deconvolution methods (CARD, cell2location, RCTD, Tangram) or to GHIST and sCellST (the obvious per-cell-from-H&E peers). Only iStar and scstGCN are compared, both *super-resolution* models, not deconvolution. The MLP baseline is too weak as a control.
- **No real Visium evaluation.** The entire benchmark is synthetic pseudo-spots; the actual production use-case (apply to true Visium and predict cells) is untested.
- **Scanner / institution generalization not assessed.** "OOD" is still 10x Genomics FFPE breast, not a different lab, scanner, or institution.
- **Cell-type stratification missing.** No systematic breakdown of accuracy per cell type (tumor vs rare immune). Figure 2 hints (it nails neoplastic MSLN) but no quantification.
- **PFM ablation is single-dataset, single-split** (lung ISin only). The Phikon-v2 > UNI > H-Optimus-0 claim is fragile.
- **No cell-count sensitivity analysis.** Spots with 1 cell vs 10 cells are aggregated identically; performance vs cell density is not reported.
- **C5 contradicted on pancreas.** ISin 0.32 → ISout 0.16 is a 50% drop, undermining the "no overfitting" framing.
- **No statistical-significance test** between DeepSpot2Cell and scstGCN; Figure 3 ablation lacks reported SE bars.

## Why It Matters for Medical AI

If the per-cell branch is even partially decodable, it is a cheap retrospective lens onto historical pathology archives: H&E slides exist by the tens of millions; Xenium-grade ST does not. The honest reading is narrower than the abstract: DeepSpot2Cell predicts the **cell-type/morphology-correlated component** of expression at single-cell resolution, not the full transcriptome. For markers whose expression is visibly tied to cell type and tissue context (e.g., MSLN in neoplastic NSCLC cells), the qualitative gain over super-resolution baselines is genuine and clinically useful as a hypothesis-generation tool. For genes whose signal is invisible in H&E morphology, the model will still produce numbers — and they should not be trusted at the single-cell level without orthogonal validation.

## References

- Paper: Nonchev, Manaiev, Koelzer, Rätsch. *DeepSpot2Cell: Predicting Virtual Single-Cell Spatial Transcriptomics from H&E images using Spot-Level Supervision.* NeurIPS 2025 Workshop on Imageomics (3rd edition). bioRxiv 2025.09.23.678121 (v3, October 2025).
- Code: <https://github.com/ratschlab/DeepSpot2Cell>
- Related: DeepSpot, BLEEP, HisToGene (spot-level prediction); iStar, scstGCN (H&E super-resolution); GHIST, sCellST (per-cell from H&E); CARD, cell2location, RCTD, Tangram (spot deconvolution); HEST-1k benchmark; Phikon-v2, UNI, H-Optimus-0 (pathology foundation models).
