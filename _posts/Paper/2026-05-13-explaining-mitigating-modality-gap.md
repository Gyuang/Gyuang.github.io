---
title: "Explaining and Mitigating the Modality Gap in Contrastive Multimodal Learning"
excerpt: "Best mitigation only shifts the modality gap from 0.294 to 0.088 — and Theorem 3.2 proves it cannot reach zero."
categories:
  - Paper
  - Multimodal-Alignment
tags:
  - CLIP
  - Modality-Gap
  - Contrastive-Learning
  - Temperature-Scheduling
  - Gradient-Flow
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-14
permalink: /paper/explaining-mitigating-modality-gap/
---

## TL;DR
- The persistent CLIP modality gap is explained by a gradient-flow analysis: the learnable inverse temperature $\beta = \exp(\nu)$ grows fast enough to throttle the decay of the gap parameter $\gamma$, so the gap can only shrink at rate **$\Omega(1/\log(t)^2)$** — astronomically slow.
- Two mitigation families fall out of the analysis: **temperature control** (scheduling, softplus reparameterization, smaller LR for $\nu$, fixed-large $\tau$) and **modality swapping** (hard / soft cross-modality entry exchange).
- On MSCOCO / CLIP-RN50 from scratch, the best variant (Temperature Scheduling S3) cuts the gap **0.294 → 0.088** and lifts Image / Text Retrieval Top-1 to **75.72 / 85.31**, but Table 3 shows the gap-vs-retrieval relationship is **inverted-U** — pushing the gap below 0.02 collapses retrieval to single digits.

## Motivation

Liang et al. (NeurIPS'22) and follow-ups documented empirically that CLIP's image and text embeddings live on distinct "cones" — a phenomenon directly contradicting the contrastive loss's global optimum (perfect alignment / neural collapse). Prior accounts blamed initialization geometry (Shi 2023) or modality information imbalance (Schrodi 2024), but only empirically.

This paper aims at the first **rigorous gradient-flow characterization** of why the gap appears, why it persists, and crucially how it interacts with CLIP's learnable temperature $\tau$ — a component used everywhere in practice (OpenCLIP, SigLIP precursors) yet never theoretically scrutinized in the modality-gap context. The downstream relevance is direct for any medical contrastive VLM (BiomedCLIP, MedCLIP, PubMedCLIP) where gap-induced retrieval degradation is a well-documented pain point.

![Hero](/assets/images/paper/explaining-mitigating-modality-gap/fig_p004_01.png)
*Figure 1: Modality gap on a real CLIP run — at initialization the modalities sit on opposite cones, and after training they remain in distinct, parallel clusters. The gap stabilizes rather than closes. (Fig. 2a)*

![Init enlargement](/assets/images/paper/explaining-mitigating-modality-gap/fig_p004_02.png)
*Figure 2: Even when initialized with a small modality gap, training enlarges it — the empirical motivation for Theorem 3.3. (Fig. 2b)*

## Core Innovation

Two analytical levers do all the work:

1. **Parallel-modality reparameterization** (after Zhang et al. 2022) absorbs all inter-modality offset into a single scalar $\gamma \in [-1, 1]$, so that the modality gap satisfies $\Delta = \lVert c_X - c_Y \rVert \geq 2\gamma$. The high-dimensional dynamics of the two encoders reduce to a tractable joint flow on $(\theta, \phi, \nu, \gamma)$.
2. **Coupling between $\nu$ and $\gamma$.** The ratio $R = (d\gamma/dt)/(d\beta/dt) = -2\beta\gamma / [\beta'^2 (1-\gamma^2)] = \Theta(1/\beta)$ shows that as $\beta = \exp(\nu)$ explodes (which it does naturally during training), $\gamma$'s decay is throttled. This is the mechanism behind gap stabilization at non-zero $\Delta$.

From these come two formal results: **Theorem 3.2** proves $\Delta(t) \geq \Omega(1/\log(t)^2)$ closure under standard learnable $\tau$, and **Theorem 3.3** proves $d\Delta/dt|_{t=0} > 0$ with high probability under uniform-on-sphere init — i.e., the gap **enlarges** in early training.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Modality gap closes at rate $\Omega(1/\log(t)^2)$ under standard CLIP $\beta=\exp(\nu)$ | Theorem 3.2 (proof in App. B.3); Fig. 4a shows linear $\Delta$ vs $-1/\log(t)^2$ | Synthetic + MSCOCO/CLIP-RN50 | ⭐⭐⭐ — proof + matching empirical curve |
| C2 | Random init causes the gap to *enlarge* in early training ($d\Delta/dt\|_0 > 0$) | Theorem 3.3 (App. B.4); Fig. 4b histogram across 100 inits | MSCOCO/CLIP-RN50, 100 trials | ⭐⭐⭐ — both theory and direct empirical density |
| C3 | Learnable temperature is the **key** stabilizer of the gap | Lemma 3.1 derives $R=\Theta(1/\beta)$; FLT/TS/TR all reduce gap (Table 2) | MSCOCO | ⭐⭐ — theory shows $\beta$ throttles $\gamma$ but doesn't isolate it from architecture / init / projection-head; experimental separation is informal |
| C4 | Softplus reparameterization gives $\gamma(t) = \Theta(1/(t \log t))$, much faster | App. C derivation; TR row in Table 2 (gap 0.284 vs 0.294 baseline) | MSCOCO | ⭐ — theory $\neq$ practice. Empirical TR barely moves the gap; the predicted dramatic acceleration does not appear in 100 epochs |
| C5 | Reducing modality gap improves image-text retrieval | Fig. 6a (downward trend), Table 2 | MSCOCO | ⭐⭐ — clear positive correlation in $[0.08, 0.29]$, but Table 3 shows the trend reverses at gap $\leq 0.02$ (retrieval collapses to ~7%). The abstract does not flag this |
| C6 | Reducing modality gap has limited impact on zero-shot / linear probe | Fig. 7, Table 1 | MSCOCO → CIFAR-10 | ⭐ — only one downstream classification benchmark, and CIFAR-10 zero-shot at ~14% is below useful range. Cannot generalize |
| C7 | Uniformity, not gap, drives classification accuracy | Fig. 7b, Table 3 (linear probe $69.44 \to 37.97$ as $\tau$ grows) | MSCOCO → CIFAR-10 | ⭐⭐ — internally consistent but single benchmark, single architecture |
| C8 | Modality swapping mitigates the gap by breaking the parallel constraint | HS, SS rows in Table 2; Table 5 ablation | MSCOCO | ⭐⭐ — gap clearly drops; theoretical link is hand-wavy (no theorem; the parallel ansatz is just dropped) |
| C9 | The proposed methods are "principled" / "theory-guided" | Sec. 1, Sec. 4 | — | ⭐⭐ — TS, FLT, SLRT, TR fall out of the temperature analysis; HS/SS are post-hoc inspirations, not derivations |

**Honest read.** Theorems 3.2 and 3.3 are the strongest contributions and are well-supported by both proofs and matching empirical curves. The mitigation story is **partially supported** but **the abstract overstates it**: the methods do not "close" the gap, they **shift** it (best $\Delta = 0.088$ is still ~30% of baseline, and Theorem 3.2 itself proves $\Delta = 0$ is unreachable in practice). Calling that "closing" is a stretch by the authors' own theory.

## Method & Architecture

![Method overview](/assets/images/paper/explaining-mitigating-modality-gap/fig_p010_01.png)
*Figure 3: The two mitigation families — Control Temperature (keep $\tau$ from collapsing) and Swap Modality (break the parallel-cone constraint). (Fig. 5a)*

![Parallel-cone abstraction](/assets/images/paper/explaining-mitigating-modality-gap/fig_p005_01.png)
*Figure 4: The "perfectly matched" vs "mismatched" parallel-cone abstraction used throughout the analysis. $\gamma$ measures the cone offset; $\Delta \geq 2\gamma$. (Fig. 3)*

**Family A — Temperature Control** (all derived from the $\tau$-scaling argument):
- **TS (Temperature Scheduling).** Don't learn $\tau$; linearly increase from $10^{-2}$ to $\{2,3,4,5\} \cdot 10^{-2}$ over training (S0–S3).
- **TR (Temperature Reparameterization).** Replace $1/\tau = \exp(\nu)$ with $1/\tau = \log(1 + e^\nu)$ (softplus) or $1/\tau = \exp(\nu/s)$ for $s > 1$. Appendix C predicts $\gamma(t) = \Theta(1/(t \log t))$ — much faster than the baseline $\Omega(1/\log(t)^2)$.
- **SLRT (Smaller LR for Temperature).** Multiply $\nu$'s learning rate by 0.1.
- **FLT (Fixed Large Temperature).** Freeze $\tau \in [10^{-2}, 4 \cdot 10^{-1}]$.

**Family B — Modality Swapping** (breaks the parallel-cone assumption underlying the analysis):
- **HS (Hard Swap).** For a random $p$-fraction of training, swap $H_X[i,j] \leftrightarrow H_Y[i,j]$ independently w.p. 0.5. Best $p = 10^{-3}$.
- **SS (Soft Swap).** For a random $p$-fraction, mix $H_X[i,j] \leftarrow \lambda_{ij} H_X[i,j] + (1 - \lambda_{ij}) H_Y[i,j]$ with $\lambda_{ij} \sim U[0,1]$. Best $p = 5 \cdot 10^{-2}$.

## Experimental Results

**Setup.** From-scratch CLIP / RN50 + CLIP-Transformer text encoder, MSCOCO train (100 epochs, batch 1024, cosine LR, 20-epoch warmup, single A100, 3 seeds averaged). Modality gap measured as the mean over 512 random pairs in MSCOCO test.

Main results (Table 2, MSCOCO, 3-seed mean):

| Metric | Baseline | FLT ($\tau=4 \cdot 10^{-2}$) | **TS S3 ($10^{-2} \to 5 \cdot 10^{-2}$)** | TR (softplus) | SLRT (×0.1) | HS ($p=10^{-3}$) | SS ($p=5 \cdot 10^{-2}$) |
|---|---|---|---|---|---|---|---|
| Modality Gap ↓ | 0.294 | 0.122 | **0.088** | 0.284 | 0.268 | 0.225 | 0.139 |
| Uniformity ↑ | −1.165 | −1.183 | **−1.108** | −1.153 | −1.152 | −1.156 | −1.152 |
| Zero-shot ↑ | 13.65 | 14.40 | **17.17** | 14.86 | 15.13 | 13.91 | 14.33 |
| Linear Probe ↑ | 65.55 | 61.98 | **67.50** | 66.46 | 66.63 | 65.95 | 64.94 |
| Image Retrieval ↑ | 68.23 | 73.44 | **75.72** | 70.06 | 70.32 | 69.72 | 68.91 |
| Text Retrieval ↑ | 78.37 | 82.66 | **85.31** | 80.10 | 80.40 | 79.74 | 79.16 |

**Critical ablation — FLT temperature sweep (Table 3) reveals an inverted-U.** Pushing the gap below ~0.1 *destroys* retrieval, and linear probe degrades monotonically as $\tau$ grows:

| $\tau$ (FLT) | Mod. Gap | Img Retr. | Text Retr. | Linear Probe |
|---|---|---|---|---|
| $10^{-2}$ | 0.669 | 69.89 | 80.68 | 69.44 |
| $2 \cdot 10^{-2}$ | 0.334 | 70.54 | 81.30 | 68.41 |
| $4 \cdot 10^{-2}$ | 0.122 | 73.44 | 82.66 | 61.98 |
| $7 \cdot 10^{-2}$ | 0.033 | 75.53 | 82.88 | 49.61 |
| $10^{-1}$ | 0.019 | 66.55 | 73.58 | 48.29 |
| $2 \cdot 10^{-1}$ | 0.008 | **23.97** | **26.94** | 50.14 |
| $4 \cdot 10^{-1}$ | 0.009 | 6.90 | 8.70 | 37.97 |

So gap ↓ correlates with retrieval ↑ **only inside $[0.03, 0.30]$**; below that, models collapse. **Hard Swap (Table 5)** behaves the same way — $p = 1.0$ (full swap) gives gap 0.028 but tanks Linear Probe to 39.72 and Img Retr to 47.17. Aggressive gap closing destroys downstream utility.

**MMVP-VLM (Table 1)** is essentially flat across all variants (range 11.85–14.82) and shows no correlation with gap or uniformity — so the "fine-grained vision" ability that the gap was hypothesized to gate is **largely unaffected** by closing it.

![Theorem 3.2 verification](/assets/images/paper/explaining-mitigating-modality-gap/fig_p008_01.png)
*Figure 5: $\Delta(t)$ plotted against $-1/\log(t)^2$ is approximately linear, confirming the $\Omega(1/\log(t)^2)$ closure rate of Theorem 3.2 on MSCOCO/CLIP. (Fig. 4a)*

![Theorem 3.3 verification](/assets/images/paper/explaining-mitigating-modality-gap/fig_p008_02.png)
*Figure 6: Histogram of $\Delta\Delta$ over the first 10 steps across 100 random inits — modality gap grows at the start of training in every trial, mean $\approx 0.15$. (Fig. 4b)*

![Retrieval vs gap](/assets/images/paper/explaining-mitigating-modality-gap/fig_p011_01.png)
*Figure 7: Image- and text-retrieval Top-1 vs modality gap across all proposed methods. Lower gap → higher retrieval, but the displayed range $[0.03, 0.30]$ crops out the collapse regime documented in Table 3. (Fig. 6a)*

![Retrieval vs uniformity](/assets/images/paper/explaining-mitigating-modality-gap/fig_p011_02.png)
*Figure 8: Retrieval vs uniformity — no clean correlation, supporting the paper's claim that retrieval gains are gap-driven, not uniformity-driven. (Fig. 6b)*

![Zero-shot vs gap](/assets/images/paper/explaining-mitigating-modality-gap/fig_p012_01.png)
*Figure 9: Zero-shot and linear-probe accuracy do **not** correlate with modality gap — a key qualifier on the "close the gap" narrative. (Fig. 7a)*

![Linear probe vs uniformity](/assets/images/paper/explaining-mitigating-modality-gap/fig_p012_02.png)
*Figure 10: Conversely, uniformity tracks classification accuracy — gap and uniformity are independent axes of representation quality. (Fig. 7b)*

## Limitations

What the authors acknowledge:
- Theorem 3.2 assumes positive margin $\alpha > 0$ throughout training (relaxed only empirically).
- Theorem 3.3 assumes parallel modalities and uniform-on-sphere init.
- "Closing modality gap is not all you need" — uniformity matters too.

What I'd add as significant unaddressed weaknesses:
1. **The mitigation does not close the gap — it shifts it.** Best $\Delta = 0.088$ (TS-S3) is still ~30% of baseline, not zero. Theorem 3.2 by the authors themselves says you cannot close it; the methods just navigate the slow regime more efficiently. Calling $\Delta = 0.088$ "closed" is overstated.
2. **Downstream gains correlate with gap reduction only inside a sweet spot $[0.03, 0.30]$.** Table 3's FLT sweep is the single most important ablation — the gap-vs-retrieval relationship is **inverted-U**, not monotonic. The headline figure (Fig. 6a) crops the x-axis to hide the collapse regime $[0, 0.03]$. This is the biggest piece of selective reporting.
3. **The authors' own data refutes the clean "smaller gap = better downstream" story.** Zero-shot and linear probe track **uniformity**, not gap (Fig. 7); MMVP-VLM is flat across all variants (Table 1).
4. **All claims rest on one pretrain corpus + one architecture** — MSCOCO / RN50 / CLIP. No LAION-400M, no ViT, no SigLIP. Whether the dynamics generalize to large-scale CLIPs (where the $\beta$ trajectory is qualitatively different — see OpenCLIP scaling laws) is open.
5. **No std-dev reported.** Numbers are 3-seed means; with deltas of ~0.5 retrieval points between TS-S3 and FLT, variance would matter.
6. **CIFAR-10 zero-shot at ~14% is essentially noise**, so the headline claim "zero-shot doesn't depend on gap" rests on a benchmark where the model can't actually do zero-shot.
7. **No finetuning experiments.** The motivating use case (CLIP finetuning for medical / domain-specific data) is left to "future work."
8. **Concurrent work uncited or unbenchmarked** — Fahim et al. 2024 ("It's not a modality gap") and Eslami & de Melo 2024 are mentioned but never compared. SigLIP, which uses sigmoid loss without the same learnable-$\tau$ structure, is conspicuously absent.
9. **TR (softplus) is a near-no-op empirically** ($\Delta: 0.294 \to 0.284$) despite theory predicting $\Theta(1/(t \log t))$ vs $\Omega(1/\log^2 t)$ — the theory's practical relevance is questionable.

## Why It Matters for Medical AI

Medical contrastive VLMs (BiomedCLIP, MedCLIP, PubMedCLIP) all inherit CLIP's modality-gap pathology, and report it as a documented obstacle for image-report retrieval. This paper's actionable takeaways for that setting:

- **Use temperature scheduling, not learnable $\tau$, when training a medical CLIP from scratch.** The TS-S3 schedule ($10^{-2} \to 5 \cdot 10^{-2}$) gave the best gap and the best retrieval simultaneously.
- **Don't aim for $\Delta \to 0$.** The Table 3 sweep shows that pushing the gap below $\sim 0.03$ collapses retrieval. Aim for the $[0.05, 0.15]$ band.
- **Track uniformity separately from gap.** Linear-probe / zero-shot performance on downstream classification (e.g., disease prediction from a frozen image encoder) is driven by uniformity, not gap. Optimizing only the gap can degrade the very classifier you want.
- **Caveat for the medical setting.** The paper trains only on MSCOCO / RN50; medical pair corpora (MIMIC-CXR, ROCO, OpenI) have stronger asymmetric information and much smaller scale, where the dynamics may differ. Treat the recipe as a hypothesis to validate, not a drop-in fix.

## References

- Paper: [arXiv:2412.07909v2](https://arxiv.org/abs/2412.07909)
- Background: Liang et al., "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning," NeurIPS 2022.
- Concurrent: Fahim et al., "It's not a modality gap: Characterizing and addressing the contrastive gap," 2024.
- Concurrent: Schrodi et al., "Modality information imbalance in contrastive learning," 2024.
- Reparameterization basis: Zhang et al., "Why does CLIP's geometry look like that?" 2022.
- Related downstream benchmark: Tong et al., "MMVP-VLM," CVPR 2024.
- Medical CLIPs that inherit the issue: BiomedCLIP, MedCLIP, PubMedCLIP.
