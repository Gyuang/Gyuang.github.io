---
title: "The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence"
excerpt: "ICML 2026 theory paper recasts InfoNCE in the large-batch limit as gradient descent on a population free energy, exposes a convex unimodal vs cross-coupled multimodal bifurcation, and identifies a negative symmetric KL term as the structural origin of the CLIP modality gap."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/geometric-mechanics-contrastive-representation-learning/
tags:
  - InfoNCE
  - Contrastive-Learning
  - Modality-Gap
  - CLIP
  - Representation-Theory
  - Free-Energy
  - Alignment-Uniformity
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- InfoNCE is rederived in the **large-batch limit as gradient descent on a deterministic population free energy** over densities on a fixed embedding manifold; this energy bifurcates into a **convex Gibbs landscape (unimodal)** and a **cross-coupled, repulsive landscape (symmetric multimodal)**.
- The new analytical object is a **negative symmetric KL coupling** $-D^{Sym}_{KL}(\rho_1, \rho_2)$ inside the multimodal intrinsic energy. Each modality acts as a **barrier** in the other's effective field, so the population-level **modality gap is structural**, not an optimization or finite-sample artifact.
- Empirical hook on MS-COCO val2017: **PE-Core-L-14 has the best Recall@1 (I→T 0.741 / T→I 0.559) but the largest centroid gap (0.854)**, while ViT-bigG-14 has worse R@1 (0.670 / 0.506) yet the **smallest gap (0.598)** across 9 OpenCLIP checkpoints — better alignment does not close the gap.

## Motivation

Wang & Isola (2020) gave the field its first geometric vocabulary for InfoNCE on the hypersphere via the alignment–uniformity decomposition, and Liang et al. (2022) "Mind the Gap" empirically named the modality gap. Neither offered a first-principles account of *why* symmetric multimodal InfoNCE should sustain a marginal gap even when pairwise alignment is excellent. Density-ratio analyses (Gutmann & Hyvärinen; Oord et al.) characterize the optimal critic pointwise, not the descent directions induced by the softmax denominator. Identifiability work (Zimmermann et al.; Daunhawer et al.; Cai et al. 2024/2025) studies what is recoverable in principle, not the objective-induced geometry of learned marginals.

This paper fills that gap with a measure-theoretic, large-batch analysis. There is no medical-AI angle — it is pure theory plus toy and MS-COCO validation.

## Core Innovation

- **Population free energy.** InfoNCE is shown to be (up to a normalization) a finite-sample stochastic estimator of a deterministic functional $J_\tau(\theta)$ over the encoder pushforward laws; both **value and gradient** converge to that limit as the negatives count $N \to \infty$.
- **Bifurcation.** The intrinsic free energy is **strictly convex** in the unimodal case (unique Gibbs equilibrium $\rho^* \propto \exp(-U/\tau)$) but **concave coordinate-wise** in the symmetric multimodal case, with a persistent $-D^{Sym}_{KL}(\rho_1, \rho_2)$ term.
- **Reframings.** "Uniformity" becomes an **intra-basin entropy tie-breaker** selected by the Gibbs equation rather than a global counter-force; the modality gap becomes a **best-response barrier effect** rather than an initialization or temperature artifact.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Stochastic InfoNCE → deterministic population energy in value and gradient as $N \to \infty$ (unimodal). | Theorem 3.1 (proof App. C) + Fig. 3 (gradient cosine and relative error vs $N$, two critics, 20 seeds). | Synthetic GMM | ⭐⭐⭐ |
| C2 | Same large-batch consistency holds for symmetric multimodal InfoNCE. | Theorem 4.1 (proof App. D). | None directly | ⭐⭐ — main-paper gradient check is unimodal-only |
| C3 | Unimodal intrinsic energy has a unique Gibbs equilibrium; "uniformity" is **entropy-driven dispersion within the aligned basin**, not global repulsion. | Prop. 3.1 (strict convexity) + Fig. 4 (S² particles vs Gibbs across $\tau$) + Fig. 2a. | Synthetic two-well on $S^2$ | ⭐⭐⭐ |
| C4 | Symmetric multimodal intrinsic energy contains a persistent **negative symmetric KL** term; each modality is a barrier in the other's effective field. | Def. 4.3 + Prop. 4.1 (concave coordinate-wise) + Theorem 4.2 (negative-KL term survives sharp-kernel limit). | Pure analysis | ⭐⭐⭐ |
| C5 | Modality gap is therefore a **structural** consequence of symmetric InfoNCE rather than an optimization or finite-sample artifact. | Latent-angle synthetic (Fig. 5, Fig. 2b) + MS-COCO observational (Tab. 2) + caption-corruption intervention (Fig. 6b). | $S^1$ synthetic + MS-COCO val2017/train2017 (5k subset) | ⭐⭐ — only one real dataset; "structural" outruns the empirical surface |
| C6 | Stronger retrieval can imply *larger* modality gap. | Tab. 2 — PE-Core-L-14 best AvgR@1 yet largest centroid gap; ViT-bigG-14 inverts. | MS-COCO val2017, 5k | ⭐⭐ — 9 checkpoints differ in scale, data, architecture, and objective variant; confounders large |
| C7 | Weakening cross-modal compatibility systematically enlarges the modality gap. | Fig. 6b: same-category caption corruption, $p \in [0,1]$, monotone gap inflation for RN50 and ViT-B-16. | MS-COCO train2017 | ⭐⭐ — clean intervention but only two backbones, no variance bands, no quantitative theory→measured-gap match |

**Honest read.** The mathematical core (C1, C3, C4) is genuinely strong — the large-batch consistency theorems and the negative-symmetric-KL identification are mathematically novel structural results. The "unification" framing is **partial**: Wang & Isola is **subsumed and reinterpreted** (uniformity becomes an intra-basin entropy tie-breaker via Gibbs equilibrium, not just renamed); Liang et al. is **explained mechanistically** by the negative symmetric KL coupling rather than re-derived; identifiability work (Yi 2025, Yin 2026) is explicitly ceded as scope-orthogonal. The empirical case (C5–C7) is directionally consistent but underpowered: 1 real dataset, 9 confounded checkpoints, 2 fine-tuned backbones, no Tab. 2 variance bands, no Flickr30k/CC3M transfer, no quantitative magnitude check.

## Method & Architecture

![Unified analytical pipeline](/assets/images/paper/geometric-mechanics-cl/page_002.png)
*Figure 1: Unified analytical pipeline — stochastic InfoNCE on finite batches → large-batch parametric energy $J_\tau(\theta)$ on encoder pushforward laws → intrinsic free energy on densities, bifurcating into a convex unimodal Gibbs landscape vs a cross-coupled, repulsive multimodal landscape.*

**Setup.** The embedding space $Z$ (e.g., $S^{n-1}$) is a compact Riemannian manifold with volume measure $\mu$. Encoders $f_\theta, g_\phi$ push data laws onto $Z$, inducing marginals $q_\theta, q_\phi$ and positive-pair laws $\pi_{\theta\theta}, \pi_{\theta\phi}$. The exponential kernel $\kappa_\tau(z,w) = \exp(s(z,w)/\tau)$ uses a critic $s$. **Asm. 2.1 (constant volume):** $\int_Z \kappa_\tau(z,w) d\mu(w) = V_\kappa(\tau)$ independent of $z$, satisfied exactly on compact homogeneous manifolds with isotropic kernels (the standard L2-normalized hypersphere with cosine similarity works).

**Population partition field and smoothed density.**

$$\Gamma_{\theta,\tau}(z) := \int_Z \kappa_\tau(z,w) dq_\theta(w), \qquad \tilde\rho_{\theta,\tau} := \Gamma_{\theta,\tau}/V_\kappa(\tau).$$

**Alignment potential field (Def. 3.1).** Disintegrate the positive-pair law $\pi_{\theta\theta}(dz, dw) = q_\theta(dz)\nu_{\theta,z}(dw)$, then

$$U_\theta(z) := -\int_Z s(z,w) d\nu_{\theta,z}(w)$$

is the average negative similarity that anchor $z$ feels from its positive partners.

**Unimodal parametric energy (Def. 3.2).**

$$J_\tau(\theta) = \frac{1}{\tau}\int_Z U_\theta(z) dq_\theta(z) - H^\times(q_\theta, \tilde\rho_{\theta,\tau})$$

with $H^\times$ the cross-entropy of $q_\theta$ against the smoothed density. **Theorem 3.1** then proves $L_{NCE}(\theta) - J_\tau(\theta) - \log(NV_\kappa(\tau)) \to 0$ and $\nabla_\theta L_{NCE} - \nabla_\theta J_\tau \to 0$ as $N \to \infty$ — large-batch value and gradient consistency.

**Intrinsic unimodal free energy and Gibbs equilibrium.** Lift to the variational level, fix $U$, and minimize

$$F_{\tau,U}(\rho) = \frac{1}{\tau}\int U \rho d\mu - H(\rho)$$

over densities. **Strictly convex**, with unique Gibbs minimizer $\rho^*(z) = \exp(-U(z)/\tau)/Z_\tau$ (Prop. 3.1). As $\tau \to 0^+$, mass concentrates on near-minimizers of $U$ (Prop. 3.2). **Theorem 3.2** ties $J_\tau$ to $F_{\tau, U_\theta}$ in the sharp-kernel limit under a quadratic diagonal-peak condition (Asm. 3.2).

> **Punchline 1.** Within the aligned basin, the entropy term $-H(\rho)$ acts as a **tie-breaker** that selects the maximally dispersed near-minimizer — not as a global counter-force pushing toward the uniform distribution.

**Symmetric multimodal directional potentials (Def. 4.1).** Disintegrating the cross-modal pair law two ways gives two distinct potential fields $U_{\theta\phi}, U_{\phi\theta}$. Each direction's parametric energy uses **the other modality's smoothed density** in the cross-entropy:

$$J^{xy}_\tau = \frac{1}{\tau}\int U_{\theta\phi} dq_\theta - H^\times(q_\theta, \tilde\rho_{\phi,\tau}),$$

and symmetrically (Def. 4.2). **Theorem 4.1** is the multimodal analogue of large-batch consistency.

**Intrinsic multimodal energy (Def. 4.3) — the load-bearing equation.**

$$F^{Sym}_{\tau, U_{1,2}}(\rho_1, \rho_2) = \frac{1}{2}\big(F_{\tau,U_{12}}(\rho_1) + F_{\tau,U_{21}}(\rho_2)\big) - D^{Sym}_{KL}(\rho_1, \rho_2).$$

The **negative** symmetric KL is what kills the convex-Gibbs structure of the unimodal case.

**Barrier best-response geometry (Prop. 4.1).** Holding $\rho_2$ fixed, $\rho_1 \mapsto F^{Sym}$ is **concave** with effective field $V_{1\mid 2}(z) = U_{12}(z)/\tau + \log\rho_2(z)$. The best response places excess mass on **minimizers of $V_{1\mid 2}$**, i.e., where $\rho_2$ is *low*. Modality 2's high-density regions are a **barrier** in modality 1's effective landscape — winner-take-all co-adaptation, not Gibbs-like equilibrium. Exact marginal matching is a "knife-edge compatibility condition" (Remark 4.5). **Theorem 4.2** confirms the negative-KL term **does not vanish** in the sharp-kernel limit.

> **Punchline 2.** Symmetric multimodal InfoNCE is "pairwise attractive yet population-level repulsive." The modality gap is structural in the analyzed regime.

![Bifurcation diagnostic](/assets/images/paper/geometric-mechanics-cl/page_005.png)
*Figure 2: Numerical bifurcation summary. (a) Unimodal cap-mass concentration as $\tau \downarrow$ on $S^2$. (b) Symmetric KL between learned multimodal marginals grows with cross-modal misalignment $\sigma_{mis}$.*

![Gradient consistency](/assets/images/paper/geometric-mechanics-cl/page_007.png)
*Figure 3: Finite-batch InfoNCE gradients converge to the large-batch reference as $N$ grows, for both cosine-on-sphere and RBF-on-bounded-Euclidean critics (mean ± std, 20 seeds). Empirical fingerprint of Theorem 3.1.*

![Unimodal Gibbs concentration](/assets/images/paper/geometric-mechanics-cl/page_008.png)
*Figure 4: Two-well potential on $S^2$ — trained particles (orange) track analytical Gibbs samples (blue) across temperatures $\tau \in \{10, 2.5, 0.5, 0.1\}$. Visual validation of Prop. 3.2.*

## Experimental Results

### MS-COCO val2017, 9 frozen OpenCLIP checkpoints (Table 2)

5,000 images of val2017 with up to 5 captions each. Metrics: Recall@1 in both directions, energy distance $\mathcal{E}(p_I, p_T)$, RBF-MMD$^2$ at median bandwidth, centroid gap $\|\mu_I - \mu_T\|$, and centroid cosine similarity $\cos(\mu_I, \mu_T)$.

| Model | I→T R@1 | T→I R@1 | $\mathcal{E}$ | MMD$^2$ | $\|\mu_I-\mu_T\|$ | $\cos(\mu_I, \mu_T)$ |
|-------|---------|---------|------|------|------|------|
| **PE-Core-L-14** | **0.741** | **0.559** | 0.599 | 0.255 | **0.854** | 0.097 |
| PE-Core-B-16 | 0.703 | 0.499 | 0.545 | 0.237 | 0.808 | 0.189 |
| ViT-bigG-14 | 0.670 | 0.506 | **0.288** | **0.129** | **0.598** | **0.384** |
| ViT-H-14 | 0.662 | 0.485 | 0.375 | 0.159 | 0.697 | 0.048 |
| ViT-L-14 | 0.569 | 0.357 | 0.697 | 0.303 | 0.898 | 0.184 |
| ResNet-50x16 | 0.559 | 0.354 | 0.502 | 0.230 | 0.761 | 0.349 |
| RN101 | 0.498 | 0.299 | 0.443 | 0.228 | 0.670 | 0.604 |
| ViT-B-16 | 0.492 | 0.302 | 0.688 | 0.313 | 0.871 | 0.309 |
| ResNet-50 | 0.480 | 0.289 | 0.627 | 0.276 | 0.849 | 0.237 |

PE-Core-L-14 is bolded as the headline retrieval result; ViT-bigG-14 marks the cleanest counterexample to "better alignment closes the gap." Best-in-column entries are bold. The non-monotone retrieval-vs-gap pattern is the empirical hook for C5–C6.

### Latent-angle multimodal misalignment (Fig. 5, $S^1$ synthetic)

Joint-angle histograms broaden and skew off-diagonal as the cross-modal angular noise $\sigma_{mis}$ rises from 0 to 0.4. Symmetric KL between learned marginals grows roughly monotonically with $\sigma_{mis}$ (Fig. 2b shows mean ± SEM; exact values are not tabulated, only plotted).

![sigma_mis = 0.0](/assets/images/paper/geometric-mechanics-cl/fig_p008_01.png)
*Figure 5a: Joint-angle histogram at $\sigma_{mis} = 0.0$ — tight diagonal coupling, minimal modality gap.*

![sigma_mis = 0.1](/assets/images/paper/geometric-mechanics-cl/fig_p008_03.png)
*Figure 5b: $\sigma_{mis} = 0.1$ — diagonal band begins broadening as cross-modal compatibility weakens.*

![sigma_mis = 0.4](/assets/images/paper/geometric-mechanics-cl/fig_p008_10.png)
*Figure 5c: $\sigma_{mis} = 0.4$ — diagonal substantially deformed, marginal gap inflated.*

### MS-COCO interventional caption corruption (Fig. 6b)

For both fine-tuned RN50 and ViT-B-16, increasing the caption-corruption probability $p$ from 0 to 1 with same-category swaps **simultaneously degrades AvgR@1 and inflates the centroid gap**. The direction matches theory; the figure has no variance bands and no test of magnitude alignment.

![MS-COCO modality-gap diagnostics](/assets/images/paper/geometric-mechanics-cl/page_009.png)
*Figure 6: (a) Across 9 OpenCLIP checkpoints, stronger retrieval does not imply smaller modality gap. (b) Controlled caption corruption monotonically enlarges the centroid gap and degrades retrieval for two fine-tuned backbones.*

### What's missing

- No ablation on temperature $\tau$ in real-data experiments (CLIP learns $\tau$; theory takes it fixed).
- No statistical significance test on Tab. 2 differences and no variance bands in Tab. 2 or Fig. 6b.
- No held-out external dataset (Flickr30k, CC3M) for the OpenCLIP study.
- Multimodal gradient-consistency check (Theorem 4.1's empirical analogue) is not done in the main paper.

## Limitations

**Authors acknowledge.**
- Constant-volume kernel (Asm. 2.1) is exact only on compact homogeneous manifolds with isotropic kernels (the standard L2-normalized hypersphere works).
- Sharp-kernel results need the quadratic diagonal-peak condition Asm. 3.2 and a strictly positive density floor.
- Parametric inheritance (Cor. B.1, B.2) needs sufficient encoder expressivity.

**Audit flags.**
- The theory takes $\tau$ fixed; real CLIP learns $\tau$, which directly modulates the kernel sharpness and hence the relative weight of the negative-KL term.
- No treatment of in-batch dependent negatives (paper analyzes iid negatives and cites convergence to the same limit but does not measure finite-sample bias).
- No analysis of asymmetric losses such as SigLIP, even though the cited mechanism is sensitive to objective design.
- The MS-COCO interventional study uses a **mild** corruption protocol (same-category caption swap). No test on stronger semantic perturbations where the predicted gap inflation should be larger.
- The "knife-edge compatibility" claim for exact marginal matching is not given an explicit measure-zero/genericity argument in the main text — it lives in App. D.
- **No quantitative theory→experiment magnitude check**: experiments confirm directions (gap exists, gap grows with misalignment) but never test whether the *size* of the gap matches what the negative-KL term would predict.
- "Structural" is a strong word. The paper does not rule out competing mechanisms (initialization geometry, temperature schedule, normalization choice) on real CLIP training, and only one real dataset is used.

## References

- Paper: [arXiv:2601.19597](https://arxiv.org/abs/2601.19597) (v4, 10 May 2026; ICML 2026, PMLR 306)
- Project page: [yichaocai.com/nce_geo.github.io](https://yichaocai.com/nce_geo.github.io)
- Wang, T. & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. ICML.
- Liang, V. W. et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*. NeurIPS.
- Oord, A. v. d., Li, Y. & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
- Gutmann, M. & Hyvärinen, A. (2010). *Noise-Contrastive Estimation*. AISTATS.
- Zimmermann, R. et al. (2021). *Contrastive Learning Inverts the Data Generating Process*. ICML.
- Daunhawer, I. et al. (2023). *Identifiability Results for Multimodal Contrastive Learning*. ICLR.
- Cai, Y. et al. (2024/2025); Yi (2025); Yin (2026). Identifiability and modality-gap follow-ups (cited in Related Work).
- Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. ICCV.
