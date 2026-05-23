---
title: "Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis"
excerpt: "Per-concept adapters plus a small learnable unknown-concept branch push CBM to 93.96 AUC / 88.15 ACC on Derm7pt and lift Skincon concept AUC from 62.14 to 82.14."
categories:
  - Paper
  - Pathology
  - LLM
permalink: /paper/concept-complement-bottleneck-medical-diagnosis/
tags:
  - CCBM
  - Concept Bottleneck Models
  - Cross-Attention
  - ClinicalBERT
  - Interpretability
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- CCBM upgrades the standard CBM with (i) one **per-concept FCL adapter** that produces a concept-specific visual query and (ii) a small **unknown-concept complement branch** of `n_u` learnable embeddings to recover the residual gap to a black-box backbone.
- On Derm7pt the model reaches **93.96% AUC / 88.15% ACC** for diagnosis and **83.86% AUC** for concept detection; on Skincon concept AUC jumps from CBM's 62.14% to **82.14%** — a 20-point gain that is the paper's most striking number.
- Ablations are less flattering than the abstract: with `n_u=0` the architecture **already matches CBM**, so the adapter+MHCA design carries most of the gain, and on LIDC-IDRI CCBM's diagnosis lead over the black-box is **within one standard deviation**.

## Motivation

Concept bottleneck models earn clinician trust by routing predictions through human-readable concepts (BI-RADS descriptors, the 7-point dermoscopy checklist, LIDC malignancy attributes), but in medical imaging they have two persistent problems. They need fine-grained per-concept annotations from radiologists or dermatologists, and they still trail black-box backbones by a few points of accuracy. LLM/VLM-discovered concepts (LaBo, LM4CV) sidestep the annotation cost but drift away from clinically meaningful evidence.

There is also a subtler issue. Every existing CBM shares **one** image encoding across every concept. That is unfair: a coarse "Brown Hyperpigmentation" patch and a subtle "Atypical Pigment Network" need very different visual evidence, but both are read off the same feature vector. CCBM is positioned as a hybrid that keeps clinically annotated concepts at the center, gives each one a dedicated feature pathway, and uses a small bank of latent unknown concepts only to close the remaining gap to a black-box.

## Core Innovation

Two structural changes to CBM:

1. **Concept adapters.** For each known concept $i$, a separate fully-connected layer $C_i : \mathbb{R}^d \to \mathbb{R}^{d_k}$ projects the shared image feature into a concept-specific query $Q_i$. Multi-head cross-attention then attends $Q_i$ over frozen ClinicalBERT embeddings of the concept names. Crucially, a separate aggregator FCL $f_i$ converts each row of the attended output into a scalar concept score $S_i$ — scores are **not** averaged, so each concept lives in its own channel.

2. **Unknown-concept complement.** A bank of `n_u` extra adapters produces unknown-concept queries $Q^u_j$, paired with `n_u` learnable key/value vectors that stand in for missing text. A cosine-similarity penalty pushes these embeddings apart from each other **and** from the known-concept text embeddings, so the unknown branch adds information rather than re-encoding what is already there. The concatenated $[S, L]$ vector feeds a single FCL decision head, preserving CBM-style bottleneck interpretability.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Concept adapters produce fairer per-concept learning (more uniform per-concept AUC). | Fig. 2 fine-grained per-concept AUC on Derm7pt / BrEaST / Skincon. | ⭐⭐ Visual evidence is convincing for outlier concepts (BrH, ErY, PF, CAL) but no quantitative fairness metric (variance across concepts, worst-concept AUC) is reported. |
| C2 | CCBM beats SOTA explainable models on diagnosis and concept detection. | Table 2 with 5-fold mean ± std across 4 datasets, vs CBM, PCBM(-H), ECBM, AEC, CBIVLM. | ⭐⭐⭐ Strongest claim — multi-dataset, multi-metric, with variance. One caveat: Skincon ACC narrowly **loses** to AEC (80.00 vs 80.24) yet the text says "outperforming other competitors." |
| C3 | CCBM closes / exceeds the black-box gap. | Same-backbone Inception-v3 / ResNet50 rows in Table 2. | ⭐⭐ CCBM > black-box on Derm7pt, Skincon, BrEaST by 1–5%; on LIDC-IDRI the AUC lead is **0.30%, inside one std band** — effectively tied. "Closes the gap" is fair; "exceeds black-box" generalizes too far. |
| C4 | The unknown-concept branch provides the additional gain over an adapter-only CBM. | Table 3: `n_u=0` vs best `n_u`; Fig. 5 qualitative cases. | ⭐⭐ `n_u=0` → best `n_u` adds only 1–3% AUC. On BrEaST `n_u=0` (87.76) is statistically tied with `n_u=2` (88.49) once stds are considered. The adapter+MHCA carries the gain; the complement branch is incremental. |
| C5 | The dissimilarity loss $L_{sim}$ is necessary. | Table 4 with vs without $L_{sim}$. | ⭐ Improvements are 0.2–1.4% on diagnosis and **negative** on concept detection. **No standard deviations** are reported in Table 4 — the effect could plausibly be noise. |
| C6 | Explanations are faithful — decisions depend on the bottleneck. | Fig. 3 inference-time intervention. | ⭐⭐ AUC collapses toward 50% under aggressive zeroing, the expected behavior; BrEaST and LIDC-IDRI curves are noisy and the authors acknowledge it. |
| C7 | Label-efficient: graceful degradation as labels shrink. | Fig. 4. | ⭐⭐ Drops only ~5–10% AUC down to 30% of labels — but no baseline comparison under the same regime, so "efficient relative to what" is unclear. |
| C8 | Generates diverse visual + textual explanations. | Fig. 5 (two cherry-picked cases). | ⭐ Qualitative only; no user study, no plausibility metric. |

**Honest summary.** The load-bearing claim — C2 — is well supported. Five-fold CV, four datasets, multiple metrics, sensible baselines including a same-backbone black-box, and the Skincon concept-detection delta (+20 AUC) is mechanistically explained by giving each rare concept its own adapter and aggregator. Where the paper over-reaches: (i) the abstract sells CCBM as exceeding the black-box, but on LIDC-IDRI the AUC margin sits inside one std; (ii) Table 4's $L_{sim}$ ablation drops std and shows mixed signs, so the case for the dissimilarity loss is weaker than presented; (iii) the unknown-concept branch is heavily marketed, but Table 3 shows `n_u=0` already matches CBM — the headline win is mostly the adapter+MHCA design, with `n_u` adding incremental gains. Missing rigor: no significance tests, no external-domain validation, no comparison to recent LLM-discovered-concept CBMs (LaBo, LM4CV, IRCBM) on these medical datasets.

## Method & Architecture

![CCBM framework](/assets/images/paper/concept-complement-bottleneck/page_008.png)
*Figure 1: CCBM framework. Per-concept adapters $C_i$ produce concept-specific queries that attend over frozen ClinicalBERT text embeddings (known) and learnable embeddings (unknown); separate aggregator FCLs $f_i$ yield concept scores, which feed a single-layer decision head.*

The full pipeline:

1. **Backbones.** Image encoder $E_I$ — Inception-v3 for Derm7pt / Skincon, ResNet50 for BrEaST / LIDC-IDRI — outputs a $d$-dim feature. A **frozen ClinicalBERT** $E_T$ encodes each of the $n_k$ known textual concept names to a $d_k$-dim embedding used as Keys and Values.
2. **Per-concept queries.** Each concept $i$ has its own FCL $C_i$, producing $Q_i = C_i(E_I(X)) \in \mathbb{R}^{d_k}$. Stacking gives $Q \in \mathbb{R}^{n_k \times d_k}$.
3. **Multi-head cross-attention.** $A(Q, K) = \mathrm{softmax}(QK^\top / \sqrt{d_k})$, attended output $A_w = A(Q, K) V$. Each concept's visual query attends across all textual concept embeddings.
4. **Per-concept aggregators.** A separate FCL $f_i$ maps the $i$-th row of $A_w$ to a scalar $S_i$. This is the "fairness" move — no averaging, no shared head.
5. **Complement branch.** `n_u` extra adapters produce unknown-concept queries $Q^u_j$; `n_u` learnable $K^u, V^u$ stand in for the missing text. Symmetric MHCA + aggregators yield unknown-concept scores $l_j$.
6. **Decision head.** $\hat Y = f'_d([S, L])$ over the concatenated known+unknown bottleneck.
7. **Loss.**

   $$\mathcal{L} = \lambda_1 \mathcal{L}_{ce}(\hat Y, Y) + \mathcal{L}_{cep}(S, C) + \lambda_2 \sum_i \Big[ \sum_{j \neq i} \mathcal{L}_{sim}(K^u_i, K^u_j) + \sum_j \mathcal{L}_{sim}(K^u_i, K_j) \Big]$$

   $\mathcal{L}_{cep}$ is multi-label BCE for classification-style concepts (Derm7pt, Skincon, BrEaST) or MSE for the LIDC-IDRI regression concepts. $\mathcal{L}_{sim}$ is cosine similarity that pushes unknown-concept embeddings apart from each other **and** from known-concept embeddings.

**Hyperparameters.** Adam, ≤300 epochs, early stop on training loss plateau. $(\lambda_1, \lambda_2)$ = (0.2, 10) for Derm7pt and BrEaST, (0.1, 5) for Skincon, (0.5, 10) for LIDC-IDRI. Default `n_u = #classes` — at least as many latent concepts as classes are required to discriminate.

## Experimental Results

5-fold CV mean ± std. CCBM rows in bold.

| Dataset | Model | Diag AUC | Diag ACC | Diag F1 | Concept AUC | Concept ACC |
|---|---|---|---|---|---|---|
| Derm7pt | CBM | 92.88 ± 1.90 | 85.89 ± 1.92 | 82.18 ± 2.67 | 82.15 ± 2.68 | 80.00 ± 1.87 |
| Derm7pt | AEC | 91.27 ± 2.02 | 84.88 ± 2.05 | 80.99 ± 3.32 | 76.61 ± 1.61 | 75.30 ± 0.72 |
| Derm7pt | Inception-v3 (black-box) | 92.02 ± 2.53 | 86.46 ± 2.54 | 83.13 ± 3.31 | — | — |
| Derm7pt | **CCBM** | **93.96 ± 0.95** | **88.15 ± 1.64** | **85.61 ± 2.79** | **83.86 ± 1.00** | **80.90 ± 1.56** |
| Skincon | AEC | 83.86 ± 0.61 | **80.24 ± 0.52** | 63.89 ± 1.63 | 58.64 ± 0.90 | 90.64 ± 0.10 |
| Skincon | CBM | 80.01 ± 1.25 | 78.42 ± 1.31 | 60.57 ± 3.04 | 62.14 ± 1.37 | 89.32 ± 0.14 |
| Skincon | Inception-v3 | 79.92 ± 1.48 | 77.52 ± 1.47 | 59.86 ± 2.78 | — | — |
| Skincon | **CCBM** | **84.55 ± 1.87** | 80.00 ± 1.75 | **67.14 ± 1.87** | **82.14 ± 0.33** | **91.35 ± 0.25** |
| BrEaST | CBM | 87.42 ± 4.27 | 77.21 ± 8.62 | 76.29 ± 8.31 | 70.76 ± 1.38 | 77.49 ± 1.43 |
| BrEaST | ResNet50 | 86.97 ± 6.14 | 77.61 ± 6.23 | 76.39 ± 6.09 | — | — |
| BrEaST | **CCBM** | **88.49 ± 5.85** | **79.21 ± 6.00** | **77.95 ± 5.53** | **77.78 ± 4.00** | **80.06 ± 3.46** |
| LIDC-IDRI | CBM | 89.76 ± 1.30 | 83.63 ± 1.67 | 80.22 ± 1.85 | RMSE 0.2981 ± 0.0096 | MAE 0.2461 ± 0.0095 |
| LIDC-IDRI | ResNet50 | 90.18 ± 1.17 | 84.08 ± 1.71 | 81.70 ± 1.52 | — | — |
| LIDC-IDRI | **CCBM** | **90.48 ± 0.94** | **84.82 ± 0.43** | **82.56 ± 0.67** | **RMSE 0.1890 ± 0.0046** | **MAE 0.1380 ± 0.0032** |

The most striking deltas are on **Skincon concept detection** (82.14 vs 62.14 AUC — a 20-point jump that mechanistically traces to per-concept adapters preventing the shared encoder from collapsing rare concepts) and **LIDC-IDRI concept regression** (RMSE 0.189 vs 0.298). On diagnosis CCBM beats every explainable baseline on all four datasets and beats the same-backbone black-box on three of four — the LIDC margin over ResNet50 (0.30% AUC) is within one std and should be read as "tied," not "exceeds."

![Per-concept AUC on Derm7pt](/assets/images/paper/concept-complement-bottleneck/page_017.png)
*Figure 2(a): Per-concept AUC on Derm7pt. CCBM matches or beats CBM/AEC on all 7 clinical concepts with smaller variance.*

**Ablations and additional findings.**

- **`n_u` sweep (Table 3).** Optimum sits at `n_u = #classes` on Derm7pt (2) and BrEaST (2); Skincon's best is `n_u=3` (matching classes); LIDC-IDRI ties between `n_u=2` and `n_u=3`. Critically, **with `n_u=0` (no unknown branch) CCBM already matches CBM**, so the adapter+MHCA design does the heavy lifting and the complement branch closes a smaller residual gap.
- **$L_{sim}$ ablation (Table 4).** Removing the dissimilarity loss slightly **improves** concept detection but consistently hurts diagnosis. The direction is what the authors claim, but Table 4 reports means only — without std, the case for $L_{sim}$ is weaker than the narrative suggests.
- **Intervention faithfulness (Fig. 3).** Zeroing high-scoring concepts crashes diagnosis AUC toward 50%, evidence the decision actually depends on the bottleneck.

![Intervention on Derm7pt](/assets/images/paper/concept-complement-bottleneck/page_018.png)
*Figure 3(a): Inference-time intervention on Derm7pt — as more high-scoring concepts are zeroed, diagnosis AUC falls toward chance.*

- **Label efficiency (Fig. 4).** Down to 30% of training labels, CCBM degrades by only 5–10% AUC; the sharp drop happens at 10%. BrEaST is the exception, dominated by its tiny size.

![Label efficiency on Derm7pt](/assets/images/paper/concept-complement-bottleneck/page_019.png)
*Figure 4(a): Label-efficiency on Derm7pt — CCBM holds within ~5% AUC down to 30% of training data.*

- **Qualitative explanations (Fig. 5).** Two case studies show the unknown-concept branch flipping a misdiagnosis: without it (`n_u=0`) the model errs; with `n_u=c` the learned C1/C2 concepts intervene and the concept scores stay closer to ground truth. Honest read: two examples, no user study, no plausibility metric.

![Qualitative explanations](/assets/images/paper/concept-complement-bottleneck/page_020.png)
*Figure 5: Visual + textual explanations on Derm7pt and LIDC-IDRI. The learned unknown concepts (blue text) intervene on cases that `n_u=0` misdiagnoses; concept scores in brackets stay closer to ground truth.*

## Limitations

- **Unknown concepts are not textualized.** They are learned embeddings with no semantic grounding — calling them "concepts" is generous, and the qualitative figure shows them only as heatmaps labeled C1/C2. Generalization across diseases is untested.
- **The "exceeds black-box" framing over-generalizes.** On LIDC-IDRI the diagnosis AUC lead over ResNet50 is 0.30%, inside one std. The honest framing is "closes the black-box gap."
- **The complement branch carries less weight than advertised.** Table 3's `n_u=0` row already matches CBM. The headline architectural contribution is per-concept adapters + MHCA + per-concept aggregators; the unknown branch is incremental.
- **$L_{sim}$ ablation has no variance estimates.** Table 4 reports means only; with stds the effect could be noise on several rows.
- **No statistical significance tests, no external-domain validation, no LLM-discovered-concept baselines** (LaBo, LM4CV, IRCBM) on the medical datasets — only CBIVLM appears.
- **Compute/parameter cost is not reported.** $n_k$ separate adapters and $n_k$ separate aggregator FCLs add real cost (e.g., 22 adapters on Skincon), but the paper makes no parameter-count or wall-clock comparison.
- **Dataset-level issues.** BrEaST has only ~254 images, producing diagnosis stds near 6 and many wins inside noise. Binarizing LIDC malignancy at score > 3 discards ordinal information. No Fitzpatrick skin-tone subgroup analysis on Skincon, despite the parent dataset (Fitzpatrick17k) being built for that purpose.
- **Hyperparameters $\lambda_1, \lambda_2$ differ per dataset** and were tuned via grid search; no sensitivity curves.

## Why It Matters for Medical AI

For clinically deployed CBMs the central question is whether interpretability has to cost accuracy. CCBM answers "not much" on four medical imaging datasets — the diagnosis-vs-black-box gap closes or inverts, and the per-concept AUC profile becomes dramatically flatter on Skincon's long-tailed 22-concept set, which is exactly the regime where shared-feature CBMs collapse on rare findings. The architectural lesson generalizes: in any concept-based clinical setup with heterogeneous concepts (BI-RADS descriptors, LIDC attributes, 7-point checklists), a per-concept feature pathway is likely to help more than chasing better text encoders or richer prompts. The unknown-concept branch is the weaker half of the contribution, but the honest framing — adapters do most of the work, unknown concepts close the residual gap — is still a useful recipe for teams who do not want to give up bottleneck interpretability when going beyond a fixed clinical concept set.

## References

- Hongmei Wang, Junlin Hou, Hao Chen. *Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis.* arXiv:2410.15446 (v2, Dec 25 2024). Preprint submitted to *Medical Image Analysis*. <https://arxiv.org/abs/2410.15446>
- Koh et al., *Concept Bottleneck Models.* ICML 2020. — the CBM baseline beaten throughout.
- Yuksekgonul et al., *Post-hoc Concept Bottleneck Models (PCBM, PCBM-H).* ICLR 2023.
- Espinosa Zarlenga et al., *Concept Embedding Models (CEM/ECBM).* NeurIPS 2022.
- Daneshjou et al., *Skincon dataset on Fitzpatrick17k.* NeurIPS 2022.
- Kawahara et al., *Derm7pt — 7-point checklist dataset.* IEEE JBHI 2018.
- Pawłowska et al., *BrEaST breast ultrasound dataset.* 2024.
- Armato et al., *LIDC-IDRI.* Medical Physics 2011.
