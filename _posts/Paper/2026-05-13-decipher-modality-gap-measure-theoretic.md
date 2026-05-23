---
title: "Decipher the Modality Gap in Multimodal Contrastive Learning: From Convergent Representations to Pairwise Alignment"
excerpt: "A measure-theoretic analysis of symmetric InfoNCE proves the modality gap is caused by dimension collapse — not the cone effect — and a post-hoc Shared Space Projection drops CLIP ViT-B/32's CIFAR-10 gap from 74.69° to 5.37° while keeping zero-shot top-1 at 86.43%."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/decipher-modality-gap-measure-theoretic/
tags:
  - Modality-Gap
  - CLIP
  - Contrastive-Learning
  - InfoNCE
  - vMF
  - Dimension-Collapse
  - Shared-Space-Projection
  - Representation-Geometry
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-14
---

## TL;DR

- **First measure-theoretic framework for the convergent optimal representation (COR) of symmetric InfoNCE.** Modeling each modality as a vMF on $S^{h-1}$, the authors take $\lim_{N \to \infty} \mathcal{L}_{\text{MCL}} - 2\log N$ in closed form and prove three convergence theorems (no constraint, cone constraint, subspace constraint) plus a non-center-pair theorem for the actual gap behavior at the InfoNCE optimum.
- **The cone hypothesis is refuted at equilibrium, not dynamically.** Theorem 2 shows the argmin of $\mathcal{L}_{\text{MCL}}^c$ is independent of cone concentration $\kappa$ — but the proof is purely analytical and the paper never sweeps $\kappa$ empirically. The structural cause of the gap is **dimension collapse**: under the subspace constraint (Theorem 3), $\Delta\theta \to \varphi_{\min}$, the smallest principal angle between the two collapsed hyperplanes.
- **Headline result: post-hoc Shared Space Projection (SSP) on CLIP ViT-B/32 reduces $\Delta\theta$ on CIFAR-10 from 74.69° to 5.37° while keeping zero-shot top-1 at 86.43%** (vs. 89.00 baseline; vs. 80.97 for translation, 14.91 for dimension removal). On ImageNet-1K, $\Delta\theta$ drops 71.02° → 50.40° with R@1 only sliding 63.34 → 62.45.

## Motivation

Liang et al. (NeurIPS 2022) catalogued the modality gap in CLIP-style models and floated the **cone effect** — random initialization places image and text embeddings in disjoint hypercones, and the contrastive loss preserves them. Subsequent work blamed temperature (Yaras 2024), insufficient training (Shi 2023), the contrastive objective itself (Fahim 2024), or information bias between image and caption (Schrodi 2025). All of these explanations were argued via **small numerical examples**. None proved what the InfoNCE optimum actually looks like for two coupled distributions on $S^{h-1}$, and there is no theoretical account of why narrowing the gap post-hoc — say by translation — sometimes hurts downstream retrieval.

The medical-AI relevance is indirect but real: BioMedCLIP, PubMedCLIP and other domain CLIPs inherit the same gap, and any gap-reduction recipe that simultaneously degrades retrieval is a problem for cross-modal radiology workflows. A principled answer to "where does the gap come from at the optimum, and how do we close it without losing alignment quality?" matters for those pipelines too.

## Core Innovation

- **A closed-form limit for symmetric InfoNCE on the unit hypersphere.** Define $\Delta\theta = \cos^{-1}(c_x \cdot c_y)$ as the angle between vMF centers (a stronger, more interpretable choice than Liang's L2 mean-difference $\Delta\mu$), and the convergence function $J$ built from modified Bessel ratios. The infinite-$N$ behavior of $\mathcal{L}_{\text{MCL}} - 2\log N$ then admits clean argmin analysis.
- **Three convergence theorems.** Without constraints, the optimum is $\Delta\theta \to 0$ and $\kappa \to 0$ (Theorem 1). Under the cone constraint (fixed $\kappa > 0$), $\Delta\theta \to 0$ still, and crucially the optimum is **independent of $\kappa$** (Theorem 2) — this is the equilibrium-level refutation of the cone hypothesis. Under the subspace constraint, $\Delta\theta \to \varphi_{\min}$ and **cannot go below the principal angle** between the two collapsed hyperplanes (Theorem 3).
- **A pair-alignment theorem with a strong assumption.** Theorem 4 proves $P_C x_i = P_C y_i$ at the optimum — but only **under the Intra-Modal Isometry (IMS) assumption** $x_i \cdot c_x = y_i \cdot c_y$. The authors themselves note (Sec. 5.2) that **CLIP violates IMS**, which limits the practical bite of Corollary 2.
- **Shared Space Projection (SSP).** A post-hoc, training-free algorithm: SVD per modality, intersect bases via SVD of $G = B_X^T B_Y$, project onto the shared subspace $C$ and renormalize. Run on a frozen CLIP, no retraining.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|-------|----------|------------|----------|
| C1 | Without constraints, MCL drives $\Delta\theta \to 0$ and $\kappa \to 0$ | Theorem 1 (App. E.1), Corollary 1 | — (theory) | ⭐⭐⭐ if vMF + symmetric InfoNCE assumptions hold |
| C2 | Under the cone constraint, $\Delta\theta \to 0$ still; cone size $\kappa$ has **no** effect on the gap's convergence | Theorem 2 (App. E.2): argmin of $J(\cdot;\kappa,\nu) + J(\cdot;\kappa,\nu)$ is at $\Delta\theta=0$ for all $\kappa>0$ | — (theory) | ⭐⭐ — proof solid in the vMF + symmetric InfoNCE setting; **no empirical $\kappa$-sweep** to corroborate |
| C3 | Dimension collapse is the true cause; under subspace constraint $\Delta\theta \to \varphi_{\min}$ | Theorem 3 (App. E.3); MSCOCO Fig. 2c shows zero-valued singular values; Fig. 2d shows non-trivial principal angles | MSCOCO, ViT-B/32 | ⭐⭐ — theorem proved; empirical support is one model, one dataset |
| C4 | Paired samples cannot be perfectly aligned under subspace constraint | Theorem 4 + Corollary 2 (App. E.4); requires IMS | — (theory; IMS not verified) | ⭐⭐ — clean proof, but IMS is acknowledged to fail for CLIP, so the corollary's bite is weakened |
| **C5** | **SSP reduces modality gap without harming downstream performance** | **Tab. 1, Tab. 2, Tab. 3** | **CIFAR-10/100, ImageNet-1K, MSCOCO; ViT-B/32 + ViT-L/14** | **⭐⭐⭐ — two backbones, four benchmarks; consistent pattern; effect size large but no error bars** |
| C6 | Translation alters $X$'s distribution unpredictably and can hurt downstream | Sec. B.1 toy example; Tab. 1 (translation drops CIFAR-10 R@1 by 8 points) | CIFAR-10/100, ImageNet | ⭐⭐ — toy is stylized 4D; empirical drop is real on CIFAR but small on ImageNet |
| C7 | Hyperplane rotation can also achieve perfect alignment | Corollary 3 | — (theory) | ⭐ — never instantiated as an algorithm or tested |
| C8 | Information bias and temperature operate **via** dimension collapse, not directly | Sec. A.3 — discussion only | — | ⭐ — speculation; framed as "we suspect" |

**Honest read.** The theoretical core (C1–C4) is internally consistent, with $J$ and $\tilde{J}$ derived from Bessel-function identities (Baricz 2010, Olver 2010). But the abstract's "refutation of the cone hypothesis" should be read narrowly — what is refuted is the **equilibrium** version ("at the optimum, the cone produces the gap"), **not** the dynamical one ("the cone biases SGD trajectories toward suboptima with a gap"), and Liang's original framing was partly dynamical. C5 is the strongest empirical claim and survives the missing error bars by sheer effect size. What is missing across the board: (i) no error bars or repeated runs, (ii) no sensitivity analysis on the 99% variance threshold or the SSP rank $k=10$, (iii) **no test on non-CLIP encoders** (SigLIP, BLIP, ALIGN, medical CLIPs), (iv) no validation that vMF is a defensible distributional model for actual CLIP features beyond Fig. 2a, (v) no empirical $\kappa$-sweep — surprising for a paper attacking the cone hypothesis.

## Method & Architecture

![COR of MCL under three constraints](/assets/images/paper/decipher-modality-gap/page_002.png)
*Figure 1: COR of MCL under three constraints. (a) initialization: two cones; (b) no constraint: paired uniform, $\Delta\theta \to 0$; (c) cone constraint: $\Delta\theta \to 0$ — refutes the equilibrium version of the cone hypothesis; (d–e) subspace constraint: $\Delta\theta \to \varphi_{\min}$, the structural origin of the gap.*

### Setup

- $N$ image–text pairs; encoders map to the unit hypersphere $S^{h-1}$.
- $\text{vMF}(c, \kappa)$ is the distributional proxy: $c$ is the modal center, $1/\kappa$ the concentration ($1/\kappa \to \infty$ ⇒ uniform; $1/\kappa \to 0$ ⇒ point mass).
- Modality gap $\Delta\theta = \cos^{-1}(c_x \cdot c_y)$ — **angle between centers**, more interpretable than Liang's L2 $\Delta\mu$.
- Symmetric InfoNCE: $\mathcal{L}_{\text{MCL}} = \frac{1}{N} \sum_i [\mathcal{L}_{X \to Y}(x_i; Y) + \mathcal{L}_{Y \to X}(y_i; X)]$.

### Three convergence theorems

**Theorem 1 (no constraint).** As $N \to \infty$,

$$\lim \mathcal{L}_{\text{MCL} } - 2\log N \ge -\tfrac{2}{\tau} + 2\log\!\big(\Gamma(\nu+1)(2\tau)^\nu I_\nu(1/\tau)\big)$$

with $\nu = h/2 - 1$, with equality iff (A1) $x_i = y_i\ \forall i$ and (A2) $\mu_x = \mu_y = \sigma_{h-1}$ (uniform on the sphere). Conclusion: $\Delta\theta \to 0$, $\kappa \to 0$. The MCL objective by itself wants to dissolve the gap.

**Theorem 2 (cone constraint, $\kappa > 0$).** Restrict $X, Y$ to $\text{vMF}(c_x, \kappa_x), \text{vMF}(c_y, \kappa_y)$ with fixed positive $\kappa$. Then

$$\lim \mathcal{L}_{\text{MCL} }^c - 2\log N = J(\cos\Delta\theta;\kappa_y,\nu) + J(\cos\Delta\theta;\kappa_x,\nu) \ge J(1;\kappa_y,\nu) + J(1;\kappa_x,\nu),$$

with equality iff $\Delta\theta = 0$. **Critically, the bound depends on $\kappa$ but the optimum $\Delta\theta$ does not.** Cone size and temperature shift the loss value but not its argmin.

**Theorem 3 (subspace constraint).** If $X$ collapses into hyperplane $A$ and $Y$ into $B$ with normals $n_A, n_B$, intersection $C = A \cap B$, principal angle $\varphi \in (0, \pi/2)$, then

$$\lim \mathcal{L}_{\text{MCL} }^c - 2\log N \ge \tilde{J}(\cos\varphi_{\min}, \cos\varphi_{\min}, \cos\varphi_{\min}; \kappa_y, \tilde{\nu}) + \text{(symmetric)},$$

with $\tilde\nu = (h-1)/2 - 1$ and $\tilde J$ generalizing $J$ via the projection norm $\tilde M_\kappa(w,t) = \sqrt{\kappa^2 + 2\kappa w/\tau + t^2/\tau^2}$. Equality iff (A6) $c_x \perp C, c_y \perp C$ and (A7) $\Delta\theta = \varphi_{\min}$. **Once representations live in two distinct hyperplanes, the gap converges to the smallest angle between them and cannot go below.**

**Theorem 4 (non-center pairs).** Under (A6) and the **Intra-Modal Isometry (IMS)** assumption $x_i \cdot c_x = y_i \cdot c_y$, optimizing $\mathcal{L}_{\text{MCL}}^{i\ne c}$ forces (A8) $P_C x_i = P_C y_i$. Corollary 2 then proves that under (A6) and (A8) with $\varphi > 0$, $P_B x_i$ is **not** parallel to $y_i$, so non-center pairs are not perfectly aligned. The bite of Theorem 4 is bounded by the fact that **CLIP itself violates IMS** (Sec. 5.2; cited from Udandarao 2022 and Schrodi 2025).

### Why the subspace constraint is plausible: empirical motivation

![Empirical evidence of dimension collapse on MSCOCO](/assets/images/paper/decipher-modality-gap/page_005.png)
*Figure 2: Evidence on MSCOCO that CLIP suffers dimension collapse — zero-valued singular values in (c) and non-trivial principal angles in (d) license the subspace assumption of Theorem 3. The cosine-similarity density (a) and UMAP (b) show the modality gap qualitatively.*

### Two repair routes (Corollaries 3, 4)

![Why projection onto the shared space yields aligned pairs](/assets/images/paper/decipher-modality-gap/page_007.png)
*Figure 3: Non-center pairs under subspace constraint, why $P_B x_i \ne y_i$, hyperplane rotation (Cor. 3), and Shared Space Projection (Cor. 4 → SSP).*

- **Hyperplane rotation (Cor. 3):** rotate $A$ onto $B$ until $A = B = C$ and $\Delta\theta = \varphi = 0$; perfect alignment follows. Practically expensive in high dimensions and never instantiated as code.
- **Shared Space Projection (Cor. 4 → SSP):** project $x_i, y_i$ onto $C$ and renormalize: $x_i^* = P_C x_i / \|P_C x_i\|$. The pair $(x_i^*, y_i^*)$ is perfectly aligned.

### SSP algorithm (Sec. C.3)

1. SVD on $X$ and $Y$: $X = U_X \Sigma_X V_X^T$.
2. Pick first $d_X, d_Y$ right singular vectors with cumulative variance $> 99\%$ → bases $B_X, B_Y$.
3. SVD $G = B_X^T B_Y$; count singular values $> 1 - \epsilon$ (with $\epsilon = 10^{-3}$) to get $d_{\text{overlap}}$.
4. Shared basis $B_S = B_X U_G[:, :d_{\text{overlap}}]$.
5. (Optional) Pick the top $k < d_{\text{overlap}}$ dimensions of $B_S$ by projected variance (paper uses $k = 10$).
6. Project $X, Y$ onto $B_S^k$ and renormalize.

For MSCOCO, $d_{\text{overlap}} \approx 212$, but the visualization uses $k = 10$.

## Experimental Results

![Table 1 + SSP visualization on MSCOCO](/assets/images/paper/decipher-modality-gap/page_009.png)
*Figure 4: Quantitative comparison (Tab. 1) and the SSP visualization on MSCOCO. After SSP, UMAP clusters merge, paired cosine similarity (P-I2T) shifts right, intra-modal similarities shift left.*

### Zero-shot image classification, CLIP ViT-B/32 (Tab. 1)

| Method | CIFAR-10 $\Delta\theta$ | CIFAR-10 R@1 | CIFAR-10 R@5 | CIFAR-100 $\Delta\theta$ | CIFAR-100 R@1 | CIFAR-100 R@5 | ImageNet-1K $\Delta\theta$ | ImageNet-1K R@1 | ImageNet-1K R@5 |
|---|---|---|---|---|---|---|---|---|---|
| CLIP | 74.69° | 89.00 | 99.36 | 74.19° | 65.23 | 88.88 | 71.02° | 63.34 | 88.82 |
| CLIP + Translation | 7.02° | 80.97 | 96.09 | 30.50° | 54.46 | 77.25 | 51.68° | 60.37 | 86.93 |
| CLIP + Removal | 72.50° | 14.91 | 56.22 | 73.16° | 6.44 | 16.82 | 69.71° | 49.50 | 78.55 |
| **CLIP + SSP** | **5.37°** | **86.43** | **99.27** | **30.39°** | **64.51** | **88.79** | **50.40°** | **62.45** | **88.41** |

### ViT-L/14 (Tab. 2)

SSP on CIFAR-10 reduces $\Delta\theta$ 77.63° → 13.27° while **exactly preserving** R@1 (95.12 → 95.12) and R@5 (99.46 → 99.46). On CIFAR-100, 77.13° → 37.73° with R@1 78.44 → 77.72.

### MSCOCO retrieval, ViT-L/14 (Tab. 3)

SSP gets the largest $\Delta\theta$ reduction (78.16° → 68.06°) while losing only 0.5 R@1 on I→T (56.06 → 55.54) and 0.1 R@1 on T→I (35.33 → 35.22). Translation loses 1.9 R@1 I→T for a smaller gap reduction. Removal collapses to 49.56 R@1 I→T.

### Empirical evidence used to refute the cone hypothesis

The paper does **not** run a controlled experiment that varies $\kappa$ and reads off $\Delta\theta$. The "refutation" of the cone hypothesis is theoretical (Theorem 2) — analytical statement that the argmin of $\mathcal{L}_{\text{MCL}}^c$ is independent of $\kappa$. Fig. 2 (UMAP, cosine-similarity histogram, SVD spectrum, principal-angle plot on MSCOCO) is used to motivate the *subspace* constraint, not to disprove the cone constraint directly. Sec. A.3 explicitly re-states this.

### Qualitative findings

After SSP, the MSCOCO UMAP no longer shows two clusters and P-I2T cosine density shifts to the right (better paired alignment) while T2T and I2I shift to the left (better intra-modal uniformity). Sec. A.3 also notes that information bias (Schrodi 2025) is not contradicted but reframed: the authors hypothesize information bias *induces* dimension collapse, which then causes the gap.

## Limitations

**Authors admit (Sec. A.2):** they do **not** explain *why* dimension collapse happens — only that it suffices to produce the gap. Candidate causes cited but not analyzed: Jing et al. 2022 (negative eigenvalues in weight matrices), Schrodi et al. 2025 (information bias), Chun 2025. SSP also does **not improve** downstream accuracy because IMS does not strictly hold for CLIP (Sec. 5.2 — vision and text neighborhoods differ).

**Reviewer-style concerns the paper does not address:**

- **vMF is unverified.** Real CLIP embeddings have known anisotropies that vMF cannot model. If the distribution is mis-specified, Theorem 2's $\kappa$-invariance result inherits that mis-specification.
- **Shared-space estimation has free knobs.** $\epsilon = 10^{-3}$ and $k = 10$ are picked without ablation.
- **Frozen CLIP only.** The pretraining-time application teased in the conclusion ("optimize alignment in the shared space to achieve intra-modal isometry") is left for future work and would be the more impactful experiment.
- **No comparison to Yaras et al. 2024 or Eslami & de Melo 2025** — both cited but never benchmarked.
- **Symmetric InfoNCE only.** SigLIP's pairwise sigmoid loss is not analyzed; given its large practical footprint this is a notable omission.
- **Single encoder family.** No SigLIP, ALIGN, BLIP, or medical CLIPs.
- **No error bars, no $\kappa$-sweep, no rank/threshold sensitivity.**

## Why It Matters for Medical AI

Medical CLIPs (BioMedCLIP, PubMedCLIP, MedCLIP, PathCLIP) all inherit the symmetric-InfoNCE training recipe and therefore the same modality-gap geometry. Two implications:

1. **Post-hoc gap-reduction recipes need to preserve retrieval.** Translation-style fixes that move the text mean to the image mean look attractive in toy plots but consistently lose R@1 on real benchmarks (8 points on CIFAR-10 in Tab. 1). For radiology image–report retrieval, where the downstream metric *is* R@k, that is a serious cost. SSP's conservative R@1 drop (≤ 0.5 on MSCOCO) is the more honest baseline to compare against.
2. **The "shared subspace" view suggests a clean diagnostic.** SVD spectra on a frozen medical CLIP can quickly reveal whether dimension collapse is happening — and if it is, the principal-angle plot bounds how much of the gap is structurally unavoidable without retraining. This is a cheap, model-agnostic check that medical-CLIP teams can run before committing to a retraining cycle.

The caveat: the theory's bite depends on Intra-Modal Isometry, which CLIP — and almost certainly its medical descendants — violate. The corollary about *perfect* pair alignment is unlikely to land in clinical pipelines as stated; the SSP empirical recipe is what travels.

## References

- **Paper (arXiv 2510.03268, v2 7 Oct 2025):** [https://arxiv.org/abs/2510.03268](https://arxiv.org/abs/2510.03268)
- **Liang et al. — *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning* (NeurIPS 2022):** the paper this work directly responds to.
- **Schrodi et al. 2025** — information-bias account of the modality gap; reframed (not refuted) here.
- **Yaras et al. 2024** — temperature-based explanation; cited but not benchmarked.
- **Jing et al. 2022** — dimension collapse via negative eigenvalues; the candidate mechanistic cause the authors leave open.
- **Baricz 2010 / Olver 2010** — modified Bessel-function identities used to derive $J$ and $\tilde J$.
