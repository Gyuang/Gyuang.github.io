---
title: "On the Duality Between Contrastive and Non-Contrastive Self-Supervised Learning"
excerpt: "A one-line Frobenius identity unifies SimCLR-style and VICReg-style SSL, and tuned SimCLR closes the gap to 68.6% top-1 on ImageNet."
categories:
  - Paper
  - Multimodal-Alignment
tags:
  - SSL
  - Contrastive-Learning
  - VICReg
  - SimCLR
  - Barlow-Twins
  - Representation-Learning
permalink: /paper/duality-contrastive-noncontrastive-ssl/
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-14
---

## TL;DR
- The "contrastive vs non-contrastive" split in self-supervised learning is largely cosmetic: the two prototypical regularizers satisfy an exact algebraic identity `L_nc + Σ_j ‖K_{j,·}‖^4 = L_c + Σ_i ‖K_{·,i}‖^4`, which collapses to `L_nc = L_c + N − M` under double L2 normalization.
- The paper renames the families **sample-contrastive** (SimCLR, DCL, SCL) and **dimension-contrastive** (Barlow Twins, VICReg, TCR), and builds two interpolating losses (VICReg-exp, VICReg-ctr) to isolate which engineering choice — LogSumExp, transpose, negative sampling — actually moves the needle. None of them do.
- **Tuned SimCLR reaches 68.45–68.68% top-1** on ImageNet (100 ep, ResNet-50, online linear) versus VICReg's 68.13%, matching almost every projector/embedding-dim cell of a 5×3 grid; the 1000-ep run still leaves a 0.7-pt residual (72.6 vs 73.3) that the authors attribute to tuning cost.

## Motivation

The SSL literature treats InfoNCE-style contrastive methods (SimCLR, MoCo, DCL) and covariance-regularization methods (Barlow Twins, VICReg, TCR) as conceptually different beasts, with marketing copy attached: "contrastive needs huge batches", "non-contrastive needs huge embedding dim", "BYOL avoids negatives". Garrido, Chen, Bardes, Najman, and LeCun argue this taxonomy is an artifact of implementation details — projector width, temperature, learning rate, normalization choices, and a notorious gradient-gather bug in DDP SimCLR — rather than a property of the loss family. The constructive payoff is a single identity that exposes the two criteria as the same object under row- vs column-normalization of the embedding matrix, together with the empirical demonstration that retuning SimCLR removes its historical disadvantage.

## Core Innovation

The contribution sits at the loss-function level, not at the architecture level. The authors define two prototypical regularizers on the projector output matrix `K ∈ R^{M×N}` (M = embedding dim, N = batch size):

- **Sample-contrastive:** `L_c = ‖K^T K − diag(K^T K)‖_F^2` — off-diagonal of the Gram matrix.
- **Dimension-contrastive:** `L_nc = ‖K K^T − diag(K K^T)‖_F^2` — off-diagonal of the covariance matrix.

Theorem 3.3 then says the two quantities differ only by row vs column fourth-norms of `K`, so under double L2 normalization they are optimization-equivalent up to a constant `N − M`. Two interpolating losses, **VICReg-exp** (replace the sum-of-squares cov term with LogSumExp) and **VICReg-ctr** (transpose `K` before applying VICReg), let the paper walk from VICReg to SimCLR one design knob at a time.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | `L_c` and `L_nc` are equivalent up to row/column normalization. | Theorem 3.3, exact algebraic identity, App. C proof, Cor. 3.4.1 single-side bounds. | ⭐⭐⭐ |
| C2 | Popular SSL methods map cleanly onto sample- vs dimension-contrastive. | Prop. 3.2 + App. A reformulation. Exact for SCL, Barlow Twins, VICReg, TCR, DCL-sq/abs, SimCLR-sq/abs; **asymptotic only** for SimCLR/DCL via Prop. 3.1 (assumes infinite negatives + Wang-Isola uniformity); **informal only** for BYOL/SimSiam/DINO/MoCo (App. B). | ⭐⭐⭐ for the prototypical criteria; ⭐⭐ for SimCLR/DCL; ⭐ for distillation-style methods |
| C3 | The contrastive vs non-contrastive distinction does not by itself drive empirical performance. | Fig. 1 / Table S5: VICReg 68.13 ≈ VICReg-exp 67.93 ≈ VICReg-ctr 67.92 ≈ SimCLR-Tuned 68.68 across a 5 embedding-dim × 3 projector grid. | ⭐⭐ — convincing within ImageNet/ResNet-50/100 ep, but a single seed and no transfer. |
| C4 | SimCLR can be tuned to match VICReg. | Table S5 SimCLR-Tuned row at 100 ep; one 1000-ep run reporting 72.6 vs 73.3. | ⭐⭐ — solid at 100 ep; the residual 0.7-pt gap at 1000 ep is hand-waved as "tuning is expensive". |
| C5 | "Non-contrastive needs huge embedding dim" is wrong. | VICReg 256-d: 55.9 (original) → 65.01 (8192-8192-d projector) → 67.72 (VICReg-ctr). | ⭐⭐ — improvement is real and reproducible across variants, but the cost moved into a ~150M-param projector rather than disappearing. |
| C6 | Normalization strategy is not decisive. | Fig. 2 — SimCLR ≈ SimCLR+centering ≈ SimCLR+dim-standardization across the projector grid. | ⭐⭐ — clean ablation but only on SimCLR; reverse direction (VICReg without dim-normalization) is not shown. |
| C7 | A DDP gradient-gather bug explains SimCLR's historical underperformance. | Footnote 1 on p. 8. | ⭐⭐ — believable, important, but not isolated as its own before/after row. |

## Method & Architecture

![Empirical equivalence across embedding dimensions and projector shapes](/assets/images/paper/duality-contrastive-noncontrastive/page_007.png)
*Figure 1: VICReg, VICReg-exp, and VICReg-ctr behave identically across embedding dimensions; tuned SimCLR catches up to VICReg's ~68% top-1 once the projector and hyperparameters match.*

The pipeline is the standard two-view SSL setup — encoder `f_θ`, projector `p_θ`, two augmentations, MSE/cosine invariance term plus a regularizer — with the regularizer being the only knob the paper varies. The four-step walk from VICReg to SimCLR is the empirical workhorse:

1. **VICReg** (dimension-contrastive, sum-of-squares cov term).
2. **VICReg-exp** swaps the sum-of-squares cov term for LogSumExp, `c_exp(K) = (1/d) Σ_i log Σ_{j≠i} exp(C(K)_{i,j}/τ)`. Tests LSE alone.
3. **VICReg-ctr** then transposes `K` before applying the variance + LSE-cov regularizers. Tests sample- vs dimension-contrastive holding LSE fixed.
4. **SimCLR** changes negative-pair sampling. Tests engineering details only.

![Theorem 3.3 and supporting lemmas](/assets/images/paper/duality-contrastive-noncontrastive/page_005.png)
*Figure 2: Theorem 3.3 — the algebraic identity at the heart of the duality result, with Cor. 3.4.1 bounding the single-side-normalization gap by O(N^2/M) or O(M^2/N).*

![Normalization conventions of each method](/assets/images/paper/duality-contrastive-noncontrastive/page_008.png)
*Figure 3: Normalization conventions used by each method — VICReg and SimCLR sit at opposite ends of a continuum that the paper bridges with VICReg-ctr.*

To make the comparison fair the paper retunes SimCLR: projector `8192-8192-d` (the VICReg shape), τ = 0.15, base LR = 0.5, and a fix for the DDP gradient-gather bug (gradients otherwise get divided by world size). All evaluation uses an online linear classifier with stop-gradient, trained alongside the backbone — App. E argues correlation with the canonical offline linear probe is near-perfect, but this remains a proxy.

## Experimental Results

Headline numbers from Table S5 (ImageNet top-1, 100 ep, ResNet-50, online linear):

| Method | Projector | d=256 | d=512 | d=1024 | d=2048 | d=8192 |
|---|---|---|---|---|---|---|
| VICReg | d-d-d | 61.36 | 63.50 | 65.35 | 66.74 | 68.13 |
| VICReg | 8192-8192-d | 65.01 | 66.72 | 68.06 | 68.07 | 68.13 |
| VICReg-exp | 8192-8192-d | 65.24 | 66.71 | 67.86 | 68.00 | 67.93 |
| VICReg-ctr | 8192-8192-d | 67.72 | 67.86 | 67.84 | 67.92 | 67.73 |
| SimCLR-Orig | 8192-8192-d | 66.11 | 66.33 | 66.00 | 66.02 | 66.08 |
| **SimCLR-Tuned** | **8192-8192-d** | **68.45** | **68.61** | **68.49** | **68.48** | **68.68** |
| SimCLR-Tuned | d-d-d | 63.42 | 65.35 | 66.36 | 67.62 | 68.68 |

At 1000 ep, **SimCLR-Tuned + dim-standardization 72.6%** vs **VICReg 73.3%** — a 0.7-pt residual the authors leave unexplained. Within the 100-ep grid, the four methods sit within ~0.6 pt of each other across every cell where projector capacity is matched.

![Normalization ablation on SimCLR](/assets/images/paper/duality-contrastive-noncontrastive/page_009.png)
*Figure 4: Adding centering or full dimension-standardization to SimCLR leaves accuracy essentially unchanged — normalization strategy is not the source of the gap.*

Three secondary findings worth flagging:

- **Robustness to embedding dim is a tuning artifact.** VICReg at d=256 jumps 55.9 → 65.01 → 67.72 as the projector grows and as VICReg-ctr's tuning is applied. The "non-contrastive needs huge output dim" folklore is mostly false — but the win required a projector with ~150M parameters, so the cost simply moved.
- **Projector capacity dominates.** Fig. S7 shows top-1 trending almost monotonically with projector parameter count up to ~150M params, across all four methods. The projector is the real lever.
- **Optimization-quality probe.** Fig. S8 shows that all four methods yield diagonally dominant Gram and covariance matrices in representation space; VICReg/VICReg-exp are marginally cleaner in embedding space (LSE only penalizes the largest off-diagonals), but downstream the matrices look essentially the same.

## Limitations

- **Strong duality is narrower than the title suggests.** Theorem 3.3 is exact only between the prototypical sample-/dimension-contrastive criteria. SimCLR and DCL fit *asymptotically* via Prop. 3.1, which assumes infinite negatives and Wang-Isola uniformity. BYOL, SimSiam, DINO, and MoCo are placed in the framework only *informally* in App. B. The blanket framing "all SSL methods are two sides of the same coin" outruns the proofs.
- **One dataset, one backbone, one regime.** All numbers are ImageNet linear probe with ResNet-50; no transfer to detection, segmentation, few-shot, OOD, or any non-natural-image domain; no ViT or ConvNeXt; one seed per cell; no variance bars in Fig. 1 or Table S5. Many of the claimed equivalences sit inside what could plausibly be 1σ noise.
- **Residual 1000-ep gap.** SimCLR-Tuned 72.6 vs VICReg 73.3 is hand-waved as "tuning is expensive". For a paper whose thesis is that the gap is engineering-only, leaving 0.7 pt unexplained at the headline regime is awkward.
- **Confounds in the matched-grid comparison.** SimCLR-Tuned uses batch 2048 while VICReg variants use 1024; the DDP gather-bug fix is mentioned in a footnote rather than isolated as its own ablation row; the projector that closes the dim-256 gap is itself ~150M parameters.
- **Online linear-probe proxy.** All headline numbers are online linear classifier accuracy, not the canonical offline linear probe. App. E argues correlation is near-perfect; this is still a proxy and matters when comparing against headline numbers from other papers.

## Why It Matters for Medical AI

The result is foundations-level rather than medical-specific, but the practical implication transfers directly: medical pretraining pipelines (CT, MRI, histology, retinal) frequently pick one SSL family off the shelf based on perceived strengths — "use VICReg because batches are small", "use SimCLR because output dim is small". The duality result implies that any of these methods should behave similarly *if* the projector is sized appropriately, the temperature/LR are tuned, and known implementation bugs are fixed. For modality-aligned pretraining (CLIP-style image-text or image-genomics), the framing also suggests the dimension-contrastive view is a viable reformulation of InfoNCE alignment — useful when one modality has many fewer samples than embedding dimensions and the Gram matrix is ill-conditioned. The caveat for medical deployment: none of the empirical claims have been validated outside ImageNet linear probe, so transferring the "they're all equivalent" headline to a domain shift comes with no evidence and should be treated as a hypothesis, not a finding.

## References

- Paper: Garrido, Chen, Bardes, Najman, LeCun. *On the Duality Between Contrastive and Non-Contrastive Self-Supervised Learning.* ICLR 2023. [arXiv:2206.02574](https://arxiv.org/abs/2206.02574)
- VICReg: Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022.
- Barlow Twins: Zbontar et al. *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* ICML 2021.
- SimCLR: Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020.
- DCL: Yeh et al. *Decoupled Contrastive Learning.* ECCV 2022.
- Spectral Contrastive Loss: HaoChen et al. *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss.* NeurIPS 2021.
- TCR: Liu et al. *Self-supervised Learning via Maximum Entropy Coding.* NeurIPS 2022.
- Alignment & Uniformity (asymptotic argument behind Prop. 3.1): Wang & Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020.
