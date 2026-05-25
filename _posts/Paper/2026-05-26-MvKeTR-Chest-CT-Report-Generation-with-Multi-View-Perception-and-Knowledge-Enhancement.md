---
title: "MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancement"
excerpt: "Three per-view CT-ViTs (axial/coronal/sagittal) plus a CT-CLIP retrieval branch are late-concat fused through a single KAN layer, reaching BLEU-4 37.86 / ROUGE-L 54.25 on CTRG-Chest-548K — but the entire result rests on one templated 1,804-pair benchmark and a single KAN-vs-MLP ablation row with no parameter accounting."
categories: [Paper, CT-Report-Generation]
permalink: /paper/mvketr/
tags:
  - MvKeTR
  - 3D CT
  - Radiology Report Generation
  - Multi-View Perception
  - KAN
  - CT-CLIP
  - Cross-Modal Retrieval
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- MvKeTR mirrors the radiologist's reading workflow with two branches — a **Multi-View Perception Aggregator (MVPA)** that runs three independent CT-ViTs over axial, coronal, and sagittal permutations of the same 224³ volume, and a **Cross-Modal Knowledge Enhancer (CMKE)** that retrieves the top-16 most similar reports from CT-RATE via a frozen CT-CLIP and cross-attends them into the axial tokens.
- Every MLP block inside MVPA and CMKE is swapped for a **Kolmogorov-Arnold Network (KAN)** layer, justified by Wang et al. (ICLR 2025) bounds on parameter scaling and spectral bias — the bounds are invoked theoretically but never measured in this paper.
- On CTRG-Chest-548K the model reports **BLEU-1/2/3/4 = 58.36 / 48.79 / 42.43 / 37.86, METEOR = 28.36, ROUGE-L = 54.25**, beating CAMANet (the strongest prior on this benchmark) by an average of 2.51 points; Reg2RG keeps the best METEOR (49.71) on the same table.

## Motivation

The paper targets two failure modes in prior chest-CT report generators (CT2Rep, Dia-LLaMA, SL-DG, Reg2RG):

1. **Single-view collapse.** Existing 3D extractors typically process the volume as a stack of axial slices, losing the cross-plane cues that radiologists rely on (the motivating example is small pulmonary nodules whose morphology is easier to characterise on coronal/sagittal projections).
2. **No external knowledge.** Real radiologists routinely consult analogous prior cases when drafting reports; image-only generators ignore that loop entirely.

MvKeTR explicitly mirrors that two-step diagnostic workflow with a multi-view branch and a retrieval branch.

## Core Innovation

Three components, in order of how much weight they bear in the ablations:

1. **MVPA — three independent CT-ViTs + view-aware attention.** Each view gets its own CT-ViT (weights not shared). A learnable view-embedding bias $E_v$ is added to the attention logits — $\text{softmax}((QK^\top + QE_v^\top)/\sqrt{d_k})V$ — and the three per-view outputs are **late-concatenated** before a single fusion KAN layer. Despite "multi-view", there is no cross-view cross-attention here.
2. **CMKE — CT-CLIP retrieval + cross-attention.** Top-$k{=}16$ reports are retrieved from CT-RATE (25,692 volumes / 21,304 patients) using cosine similarity on frozen CT-CLIP embeddings. The retrieved report embeddings serve as K/V in cross-attention with **axial-only** image tokens as Q — an internal inconsistency given the paper's central thesis that single-view is insufficient.
3. **KAN everywhere.** All MLP blocks in MVPA and CMKE are replaced by KAN layers $\Phi_{k-1}\circ\cdots\circ\Phi_0\,x$, with each $\Phi_k$ a matrix of learnable univariate spline activations. The motivation is the theoretical $O(GW^2L)$ KAN scaling vs $O(G^2W^4L)$ MLP scaling plus a Hessian-eigenvalue bound on spectral bias — both invoked, neither measured for the actual MvKeTR configuration.

## Claims & Evidence Analysis

| # | Claim | Evidence | Rating |
|---|---|---|---|
| C1 | MvKeTR sets a new SOTA across "almost all" metrics on chest CTRG | Table II — best on 5/6 metrics, only loses METEOR (28.36 vs 49.71) to Reg2RG | ⭐⭐ Single 1,804-pair benchmark with a 931-token templated vocabulary; CT2Rep / Reg2RG / Dia-LLaMA are evaluated *outside* their native CT-RATE setting, so the like-for-like comparison those methods were tuned for is missing. |
| C2 | Multi-view perception contributes more than knowledge enhancement | Table III: BASE+MVPA = +17.2% vs BASE+CMKE = +14.5% over BASE | ⭐⭐ Cleanly separated by ablation, but no breakdown of MVPA gain into "view-aware attention" vs "3× the tokens" — early concat, cross-view cross-attention, and gated fusion are never ablated. |
| C3 | KAN > MLP for this task | Table III: Ours = +23.6% vs Ours-MLP = +14.1% (single row) | ⭐ One ablation row, one dataset, single run, no error bars; KAN reproductions in the broader literature frequently fail to beat well-tuned MLPs. |
| C4 | KAN delivers superior parameter efficiency in MvKeTR | None — only the theoretical $O(GW^2L)$ vs $O(G^2W^4L)$ result from Wang et al. (ref [50]) is cited; no parameter count for Ours vs Ours-MLP | ⭐ Theoretical only. |
| C5 | View-aware attention is robust to misalignment | Table VI: ±5°–±20° rotations cause only −2.3 to −3.8% degradation | ⭐⭐ Non-monotonic in rotation magnitude (±5° hurts more than ±20°) which the authors hand-wave as "adaptive attention redistribution" rather than within-noise variance. |
| C6 | Robust to acquisition artifacts | Table VI: motion +0.8%, ring +0.7%, MPR −0.3% | ⭐ Performance *improving* under noise is physically implausible and most likely run-to-run variance — the paper reports no seeds, no variance, no confidence intervals anywhere. |
| C7 | Radiologists prefer MvKeTR reports | Table IV: 4.20 / 3.90 / 4.15 / 4.10 vs M2KT 3.70 / 2.80 / 3.05 / 2.95 | ⭐ n=10 cases, n=2 raters, no inter-rater κ, no statistical test, and the three strongest automatic baselines (CT2Rep, Reg2RG, Dia-LLaMA) are *omitted* from the human eval. |
| C8 | CT-ViT is the most suitable 3D extractor | Table V — CT-ViT best on 5/6 metrics; 3D ViT ties on BLEU-1 (58.39 vs 58.36) | ⭐⭐ Defensible within this dataset but the CT-ViT initialisation comes from GenerateCT pretraining, which is itself the biggest source of the gap. |
| C9 | $F_{mv} + F_{ke}$ fusion implements a Bayesian posterior | Eq. 22 is a narrative reframing of concatenation; no inference rule is implemented | ⭐ Motivational, not load-bearing. |

**Honest read — seven things to flag before quoting the numbers.**

1. **"Multi-view" is late concat, not cross-view cross-attention.** $F_{mv} = \text{Norm}(\text{KANLayer}([AN_a; AN_s; AN_c]))$ is just channel-concat through a single fusion layer. No early-concat, cross-view-cross-attention, or gated-fusion baseline is ablated, so the +17.2% MVPA gain cannot be cleanly attributed to view-aware attention as opposed to having 3× the visual tokens.
2. **CMKE retrieval query is axial-only**, despite the entire motivation arguing single-view is insufficient. The paper does not address this internal inconsistency.
3. **The KAN vs MLP ablation is a single row.** No parameter count, no FLOPs, no latency, no variance, no seeds. The headlined +9.5 pp swing is uncorroborated and the parameter-efficiency claim is a purely theoretical bound (Wang ICLR'25) — never measured for the actual MvKeTR configuration.
4. **Three independent CT-ViTs roughly triple the visual-encoder cost** vs CT2Rep, which uses a single CT-ViT. There is no parameter / FLOP accounting and no weight-shared-CT-ViT ablation, so the parameter-fairness comparison to baselines is invisible.
5. **One benchmark, one language, ~1,800 pairs.** CTRG-Chest-548K is 1,804 image-report pairs (the "548K" refers to Chinese-character report tokens, not cases) with a 931-token vocabulary and heavily templated sentences ("thorax is symmetrical", "no pleural effusion"). Templated reports systematically inflate BLEU/ROUGE for any method that learns the template. CT2Rep, Reg2RG, and Dia-LLaMA are all evaluated outside their native CT-RATE setting they were tuned for.
6. **No retrieval-leakage analysis.** Whether CTRG-Chest-548K-similar cases appear in the CT-RATE retrieval bank is never checked. Even if the datasets are nominally disjoint, semantic overlap from templated Chinese chest-CT reporting style could effectively turn retrieval into near-neighbour memorisation.
7. **Human eval is n=10 cases, 2 raters, no κ, and only beats the three weakest baselines** (MRMA, M2KT, M2TR). The strongest automatic baselines (CT2Rep, Reg2RG, Dia-LLaMA) are notably absent from the human eval.

The headline number on CTRG-Chest-548K is real and internally consistent with the ablations, but the framing of clinical reliability that the abstract reaches for is not supported by the evidence.

## Method & Architecture

![MvKeTR overall architecture](/assets/images/paper/mvketr/fig_p004_01.png)
*Figure 1: MvKeTR overview — three CT-ViTs (axial/coronal/sagittal, weights not shared) feed the Multi-View Perception Aggregator; the Cross-Modal Knowledge Enhancer retrieves top-16 similar reports via a frozen CT-CLIP and cross-attends them with axial-only image tokens as queries; concatenated $[F_{mv}; F_{ke}]$ features go to an R2GenCMN-style decoder.*

**Step-by-step.**

1. **Input formatting.** Resize the volume to $224\times224\times224$ and produce three view-permuted tensors — axial $(d{\times}h{\times}w)$, coronal $(h{\times}w{\times}d)$, sagittal $(h{\times}d{\times}w)$. Patch size 28 in all dimensions, yielding $8{\times}8{\times}8 = 512$ tokens per view embedded to $D$-dim.
2. **3D visual extractor.** Three *independent* CT-ViTs (one per view, **weights not shared**) embed each view to $Z_v \in \mathbb{R}^{8\times8\times8\times512}$. Each CT-ViT is a stack of spatial transformer layers (intra-slice) followed by causal transformer layers (depth-wise), initialised from GenerateCT (Hamamci et al., ECCV 2024).

![CT-ViT extractor — spatial then causal transformers](/assets/images/paper/mvketr/fig_p004_04.png)
*Figure 2: CT-ViT extracts per-view tokens by stacking spatial transformer layers (intra-slice attention) followed by causal transformer layers (depth-wise attention). One CT-ViT is instantiated per view; weights are not shared.*

3. **KAN layer.** Every MLP block inside MVPA and CMKE is replaced by $\text{KAN}(x) = \Phi_{k-1}\circ\cdots\circ\Phi_0\,x$, where each $\Phi_k$ is a matrix of learnable univariate spline activations $\phi_{i,j}$. KAN hyperparameters (grid size $G$, spline degree $k$, exact KAN variant) are not specified in the paper.

![KAN layer stack](/assets/images/paper/mvketr/fig_p006_11.png)
*Figure 3: KAN layer — each layer is a matrix of learnable univariate spline activations $\phi_{i,j}$, stacked $k$ deep. MvKeTR replaces every MLP block inside MVPA and CMKE with a KAN layer.*

4. **View-aware attention (MVPA).** Standard scaled dot-product attention is modified with an additive learnable view-embedding bias on the logits:

$$\text{VAA}_v(Q,K,V,E_v) = \text{softmax}\!\left(\frac{QK^\top + QE_v^\top}{\sqrt{d_k}}\right)V,$$

with $E_v \in \mathbb{R}^{N\times d_k}$ and $v \in \{a, c, s\}$. Each per-view output is residual-normed: $AN_v = \text{Norm}(\text{VAA}_v + Z_v)$.

![View-aware attention vs vanilla attention](/assets/images/paper/mvketr/fig_p006_10.png)
*Figure 4: View-aware attention (right) adds a learnable view-embedding bias $QE_v^\top$ to the standard attention logits (left). $E_v$ is one of three view-specific embedding matrices for axial/coronal/sagittal.*

5. **Multi-view aggregation.** $\text{Concat}_{mv} = [AN_a; AN_s; AN_c];\ F_{mv} = \text{Norm}(\text{KANLayer}(\text{Concat}_{mv}))$. **This is late concat + KAN fusion** — there is no cross-view cross-attention.
6. **Cross-modal knowledge enhancer (CMKE).**
   - *Retrieval.* A frozen CT-CLIP (Hamamci et al., 2024; pretrained on CT-RATE) encodes the **axial** volume $v = E_\text{image}(X_a)$ and each candidate report $r_i = E_\text{text}(R_i)$; cosine similarity on L2-normalised embeddings picks the top-$k$ report embeddings $Z_{\text{top-}k}$. The hyperparameter is swept in Fig. 10 over $\{8, 16, 32, 64\}$ and $k=16$ is selected.
   - *Knowledge enhancement.* Cross-attention with $Q$ from axial tokens $Z_a$, $K$ and $V$ from $Z_{\text{top-}k}$; output is residual-normed and passed through one KAN layer: $F_{ke} = \text{Norm}(\text{KANLayer}(AN_a) + AN_a)$.
7. **Report generator.** R2GenCMN-style cross-modal memory: $F_S = [F_{mv}; F_{ke}]$ is run through a learnable memory matrix $M$ to produce memory responses $r_{f_i}$, and a 3-encoder / 3-decoder Transformer with cross-modal memory generates tokens autoregressively. The paper frames the $F_{mv} + F_{ke}$ fusion as a Bayesian posterior $p(Y\,|\,X, K) \propto p(X\,|\,Y)\cdot p(K\,|\,Y)$ (Eq. 22); this is narrative, not algorithmic — no Bayesian inference is performed.
8. **Training.** Negative log-likelihood; LR = 5e-5 (each CT-ViT) and 1e-4 (rest); decay 0.8/epoch; batch size 2; 30 epochs; beam size 3; Adam + AMP on a single RTX 4090 D; max generation length 150; vocab 931 tokens (freq > 3).

## Experimental Results

### Main comparison on CTRG-Chest-548K (Table II)

| Method | Year | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Vanilla Transformer* | 2017 | 32.37 | 28.40 | 25.82 | 23.93 | 22.22 | 51.28 |
| MRMA* | 2018 | 40.85 | 30.16 | 23.77 | 19.54 | 19.52 | 33.45 |
| M2TR* | 2021 | 39.46 | 32.76 | 28.56 | 25.72 | 22.35 | 47.63 |
| R2GenCMN* | 2022 | 48.29 | 38.34 | 32.53 | 28.42 | 23.89 | 48.91 |
| M2KT* | 2023 | 39.71 | 33.17 | 29.12 | 26.16 | 20.11 | 47.19 |
| TSGET* | 2024 | 46.12 | 38.15 | 32.92 | 28.84 | 23.02 | 50.19 |
| UDT* | 2024 | 46.54 | 39.17 | 34.48 | 31.08 | 23.42 | 53.36 |
| CAMANet* | 2024 | 54.28 | 45.26 | 39.42 | 35.38 | 26.48 | 54.20 |
| CT2Rep* | 2024 | 49.81 | 40.70 | 35.19 | 31.28 | 24.82 | 51.18 |
| SL-DG‡ | 2024 | — | — | — | 23.70 | 21.90 | 43.80 |
| Dia-LLaMA‡ | 2024 | 51.16 | — | — | 29.64 | 26.28 | 42.15 |
| LHR-RFL* | 2024 | 55.56 | 44.76 | 37.62 | 32.64 | 25.40 | 47.70 |
| Reg2RG‡ | 2024 | 49.63 | 41.43 | 35.91 | 32.04 | **49.71** | 47.76 |
| **MvKeTR (ours)** | — | **58.36** | **48.79** | **42.43** | **37.86** | 28.36 | **54.25** |

`*` = replicated by the authors on CTRG-Chest-548K; `‡` = numbers cited from the original paper. Note that Reg2RG's METEOR (49.71) is from its native CT-RATE evaluation, not directly comparable to the other rows.

### Ablation (Table III)

| Variant | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | Avg Δ vs BASE |
|---|---|---|---|---|---|---|---|
| BASE (R2GenCMN + CT-ViT) | 48.29 | 38.34 | 32.53 | 28.42 | 23.89 | 48.91 | — |
| BASE + MVPA | 54.81 | 45.82 | 39.85 | 35.49 | 26.54 | 54.68 | +17.2% |
| BASE + CMKE | 53.76 | 44.61 | 38.63 | 34.43 | 26.77 | 52.59 | +14.5% |
| Ours-MLP (KAN→MLP) | 53.98 | 45.30 | 39.55 | 35.39 | 27.16 | 54.05 | +14.1% |
| **Ours (full)** | **58.36** | **48.79** | **42.43** | **37.86** | **28.36** | **54.25** | **+23.6%** |

The Ours-MLP row is the only direct **KAN vs MLP ablation** in the paper. The +9.5 pp gap is attributed to KAN, but no parameter / FLOPs / variance numbers are provided to fairness-check the gain — the parameter-efficiency claim is invoked theoretically (Wang ICLR'25 bound), never measured for this configuration.

### Visual extractor ablation (Table V)

CT-ViT vs 3D ViT vs CT-Net vs U-Net: CT-ViT wins on 5/6 metrics; 3D ViT ties on BLEU-1 (58.39 vs 58.36) and beats on nothing else. The CT-ViT advantage is plausibly dominated by its GenerateCT pretraining initialisation, which other backbones do not get.

### Robustness (Table VI)

| Perturbation | Avg Δ |
|---|---|
| Rotation ±5° | −3.8% |
| Rotation ±10° | −2.6% |
| Rotation ±15° | −2.3% |
| Rotation ±20° | −3.4% |
| Motion artifact | **+0.8%** |
| Ring artifact | **+0.7%** |
| MPR blurring | −0.3% |

Rotations are non-monotonic (±5° hurts more than ±20°) which the authors frame as "adaptive attention redistribution"; with no seeds / variance reported, an "improvement under perturbation" reading is more parsimonious as within-noise variance.

### Human eval (Table IV)

n=10 cases, 2 board-certified radiologists, 5-point Likert. MvKeTR scores 4.20 / 3.90 / 4.15 / 4.10 (completeness / precision / readability / confidence) vs runner-up M2KT 3.70 / 2.80 / 3.05 / 2.95. No inter-rater κ, no statistical test, and CT2Rep, Reg2RG, Dia-LLaMA are absent — only the three weakest automatic baselines are in the comparison.

### Top-k retrieval sweep (Fig. 10)

BLEU-1 peaks at top-$k = 16$ over $\{8, 16, 32, 64\}$; modest unimodal curve.

### Qualitative examples

![Qualitative case 1](/assets/images/paper/mvketr/fig_p010_01.png)
*Figure 5: Qualitative case 1 — MvKeTR correctly identifies increased lung transparency and patchy shadows in the left lower lobe; R2GenCMN misses both (Fig. 9, top of the paper).*

![Qualitative case 2](/assets/images/paper/mvketr/fig_p010_02.png)
*Figure 6: Qualitative case 2 — MvKeTR detects multiple bilateral nodules that R2GenCMN omits (Fig. 9, bottom of the paper).*

## Limitations

**Acknowledged by the authors.**

- Modest degradation under synthetic rotations and acquisition artifacts.
- CMKE quality depends on the CT-RATE retrieval bank; future work flagged on uncertainty-aware retrieval, human-in-the-loop validation, adversarial debiasing.
- Generalisation to other modalities (MRI) and body parts is unvalidated.

**Not addressed.**

- **No external benchmark.** A single 1,804-pair, single-language, single-source dataset. No CT-RATE evaluation, no cross-hospital test, no multi-language test. The Reg2RG and CT2Rep numbers on their native CT-RATE setting are not reproduced here.
- **No parameter / FLOPs / latency comparison.** Three independent CT-ViTs roughly triple the visual-encoder cost vs single-view baselines; weight-shared CT-ViTs (the cheaper alternative) are not ablated. The "KAN saves parameters" claim is purely theoretical.
- **No variance / seed runs / significance tests.** All reported numbers are single runs — including a robustness table where the model "improves under noise".
- **Retrieval-leakage check is missing.** Whether CTRG-Chest-548K-similar cases appear in the CT-RATE retrieval bank is never analysed.
- **No clinical-efficacy metrics.** No CheXpert F1, RadGraph F1, or clinical-entity recall — only n-gram metrics, which are known to be uninformative for templated medical reports.
- **MVPA fusion ablation is incomplete.** Late concat vs early concat vs cross-view cross-attention vs gated fusion is not compared, so the MVPA gain cannot be cleanly attributed to view-aware attention vs simply having 3× more tokens.
- **CMKE uses only axial as the retrieval query**, contradicting the paper's central thesis that single-view processing is insufficient.
- **KAN hyperparameters** (grid size $G$, spline degree $k$, exact KAN variant) are not specified.
- **Bayesian-posterior framing of $F_{mv} + F_{ke}$ is post-hoc narrative** — Eq. 22 motivates concatenation but no Bayesian inference is implemented.

## Why It Matters for Medical AI

The architectural story — orthogonal-view processing plus retrieval-augmented decoding — is the right direction for 3D radiology report generation, and the per-view CT-ViT design at least makes the multi-view hypothesis testable. But the evidence is currently confined to a single small, templated benchmark with no parameter accounting, no clinical-efficacy metric, and no significance testing. For practitioners weighing a multi-view + retrieval recipe, the right takeaways are: (1) view-aware attention with late concat is a cheap drop-in that helps on templated benchmarks; (2) retrieval-augmented decoding with a frozen CT-CLIP is plausibly worth the engineering cost — but should be validated with a leakage check and a clinical-efficacy metric; (3) the KAN-over-MLP claim is interesting but the evidence here is insufficient to prefer KAN in production, and the parameter-efficiency framing is theoretical only.

## References

- Paper: Deng et al., *MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancement*, arXiv:2411.18309, Nov 2024 (rev. Jun 2025). [arXiv link](https://arxiv.org/abs/2411.18309)
- Code: [github.com/xiweideng/MvKeTR](https://github.com/xiweideng/MvKeTR)
- CT-ViT / GenerateCT: Hamamci et al., *GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes*, ECCV 2024.
- CT-CLIP / CT-RATE: Hamamci et al., *A Foundation Model Utilizing Chest CT Volumes and Radiology Reports for Supervised-Level Zero-Shot Detection of Abnormalities*, 2024.
- CTRG-Chest-548K dataset: Tang et al., *CTRG: A Large-Scale Chinese Chest CT Report Generation Benchmark*, Expert Systems with Applications, 2024. [Repo](https://github.com/tangyuhao2016/CTRG)
- R2GenCMN: Chen et al., *Cross-Modal Memory Networks for Radiology Report Generation*, ACL 2022.
- KAN: Liu et al., *KAN: Kolmogorov-Arnold Networks*, 2024; Wang et al., theoretical analysis of KAN parameter scaling, ICLR 2025.
- Related CT report generators: CT2Rep, Reg2RG, Dia-LLaMA, CAMANet, SL-DG, LHR-RFL.
