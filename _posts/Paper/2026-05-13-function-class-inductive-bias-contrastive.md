---
title: "A Theoretical Study of Inductive Biases in Contrastive Learning"
excerpt: "The function class — not the choice of augmentation — filters the positive-pair graph; Thm 3.6 and 4.5 replace $r_0$ clusters with $m$ minimal-implementable clusters as the dimensionality budget, backed by a single 4-row CIFAR-10 surrogate table."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/function-class-inductive-bias-contrastive/
tags:
  - Contrastive Learning
  - Inductive Bias
  - Spectral Contrastive Loss
  - Augmentation Graph
  - Self-Supervised Learning Theory
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- The paper is **not** about which augmentations to choose, and it is **not** a bias-variance decomposition. It holds the augmentation graph fixed and asks how the **function class** $\mathcal{F}$ filters which clusters of the positive-pair graph the representation can actually recover.
- Two headline theorems (Thm 3.6, Thm 4.5) replace the prior dependence on $r_0$ (the number of clusters in the augmentation graph) with **$m$ = the number of clusters the architecture can implement**. Concretely, Thm 3.6 gives $\|W\hat f(x) - e_{y(x)}\|^2 \lesssim (\alpha/\beta) \cdot P_{\max}/(P_{\min}-\alpha)$ at $k = m$, and Thm 4.5 sharpens this through an eigenfunction reformulation.
- The empirical evidence is **one CIFAR-10 table** (Table 1, 4 architectures, $\varphi_m$ at $m\in\{10,30,100,300\}$) with no seeds, no error bars, no downstream linear-probe accuracy, and a surrogate quantity measured after whitening from a *pre-trained* spectral-CL checkpoint. The empirical side rates 1 star.

## Motivation

HaoChen et al. (2021) reduced contrastive pre-training to spectral clustering on a *positive-pair graph* — vertices are augmented images, edge weights are positive-pair probabilities — and proved that a $k$-dimensional representation recovers the spectral structure when $k$ is at least $r_0$, the number of (near-)disconnected clusters. That bound treats the neural network as a universal approximator, so the dimensionality budget tracks a property of the *data* alone.

Saunshi et al. (2022) gave a single linear toy where contrastive learning succeeds with $k \ll r_0$ — possible only because the linear function class *cannot* implement the fine partition the augmentation graph technically allows. This paper generalizes that observation: the relevant dimensionality is not how many clusters the graph has, but how many clusters the architecture can actually implement on top of it. There is **no medical-AI angle** here — the paper is pure SSL theory on synthetic distributions plus one CIFAR-10 sanity check.

## Core Innovation

Replace $r_0$ (graph-only quantity) with $m$ (graph $\cap$ function-class quantity), called the number of **minimal implementable clusters**. A partition $\{S_1, \dots, S_m\}$ is *minimal implementable* w.r.t. $\mathcal{F}$ when (i) inter-cluster positive-pair leakage is at most $\alpha$, (ii) every $f \in \mathcal{F}$ has a lower-bounded per-cluster expansion $Q_{S_i}(g) \geq \beta$ (so $\mathcal{F}$ cannot split clusters any finer), and (iii) there exists an $f \in \mathcal{F}$ implementing the partition exactly. Under those assumptions, Theorem 3.6 bounds the linear-probe MSE at $k = m$ by

$$\mathbb{E}_{p_{\text{data} }}\|W\hat f(x) - e_{y(x)}\|^2 \;\leq\; \frac{\alpha}{\beta}\cdot\frac{P_{\max} }{P_{\min}-\alpha}.$$

Theorem 4.5 strengthens this through eigenfunctions of the positive-pair Laplacian: $m$ approximate eigenfunctions inside $\mathcal{F}$ suffice, which can be strictly smaller than the count of minimal implementable clusters when the implementable eigenfunctions have geometric structure.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Constrained $\mathcal{F}$ recovers only the cluster structures $\mathcal{F}$ can implement; $k = m$ suffices instead of $k = r_0$. | Thm 3.6 / Eq. 6 (proof in App. B). | ⭐⭐⭐ Theoretically rigorous; no empirical test of the bound itself. |
| C2 | Eigenfunction reformulation is strictly stronger and can drive $k$ below the count of minimal implementable clusters when those clusters have geometric structure. | Thm 4.5 (proof App. C); Example 5.1 exhibits a $k = s < 2^s$ gap. | ⭐⭐⭐ on theory; ⭐ on empirical confirmation. |
| C3 | Linear $\mathcal{F}$ reaches zero MSE on the hypercube with $k = s$; universal approximators with $k \leq 2^{d-1}$ provably fail. | Thm 5.3 (both directions). | ⭐⭐⭐ theorem; synthetic only. |
| C4 | ReLU MLPs solve a strictly richer family with $k = 2^s$ than linear models on the same data. | Thm 5.6. | ⭐⭐⭐ theorem; synthetic only. |
| C5 | Lipschitz constraints reduce required $k$ from $r_0$ to $r$. | Thm 5.9 / Example 5.7. | ⭐⭐⭐ theorem; synthetic only. |
| C6 | CNNs solve translation-invariant patch tasks with $k = 2^s$; ReLU MLPs require $k \geq d \cdot 2^{s-1}$. | Thm 5.12. | ⭐⭐⭐ theorem; synthetic only. |
| C7 | Real architectures on CIFAR-10 cannot implement arbitrarily fine partitions: $\varphi_m$ rises monotonically with $m$. | Table 1 ($\varphi_{10}\to\varphi_{300}$ rises ~4× for ResNet-18, ~9× for ResNet-101). | ⭐ Single dataset, no variance, surrogate quantity. |
| C8 | The framework distinguishes architectures (ResNet-101 / WRN / ViT $<$ ResNet-18 at small $m$). | Table 1 row comparison. | ⭐ Four numbers per column; gaps within plausible noise; no error bars. |
| C9 | Improves sample complexity over the prior $r_0$-dependent bound. | Thm 4.7 (Eq. 24) stated. | ⭐⭐ Form of the bound supports the claim; no quantitative comparison shown. |
| C10 (prompt-implied) | "Bias-variance decomposition" of linear-probe error. | **Not in the paper.** Eq. 24 is approximation + Rademacher + concentration, not bias-variance. | not supported. |
| C11 (prompt-implied) | "Implications for which augmentations to choose for which task." | **Not in the paper.** The augmentation distribution is fixed input; the variable is $\mathcal{F}$. | not supported. |

**Honest summary.** The theory (C1-C6, C9) is clean and the four worked examples genuinely exercise different function classes. The empirical case (C7-C8) is thin: one CIFAR-10 table, no seeds, a surrogate quantity that depends on a whitening step and on initialization from another pre-trained spectral-CL ResNet. The two prompt-style framings — bias-variance, augmentation-choice — do not appear in the paper.

## Method & Architecture

![Inductive bias filters the positive-pair graph](/assets/images/paper/augmentation-graph-ib/page_003.png)
*Figure 1 (HaoChen and Ma, 2023): on the hypercube example, a linear function class can implement the partition along the augmentation-invariant axis (left) but not along the spurious axis (right) — the inductive bias of $\mathcal{F}$ filters which clusters of the positive-pair graph the representation can recover.*

The setup is the spectral / Barlow-Twins-style contrastive loss

$$\mathcal{L}_\lambda(f) \;=\; \mathbb{E}_{(x,x^+)\sim p_{\text{pos} }}\|f(x)-f(x^+)\|^2 \;+\; \lambda \cdot \big\|\mathbb{E}_{x\sim p_{\text{data} }}[f(x)f(x)^\top] - I\big\|_F^2,$$

with $f \in \mathcal{F}: \mathcal{X} \to \mathbb{R}^k$.

![Positive-pair graph as a clustered graph](/assets/images/paper/augmentation-graph-ib/page_005.png)
*Figure 2: the positive-pair graph — vertices are augmented images, edges weighted by positive-pair probability; semantically related augmentations form a cluster while unrelated images sit in separate clusters.*

**Minimal implementable clusters (Section 3).** Assumption 3.1 ($\alpha$-separability) requires $\Pr_{p_{\text{pos}}}(\text{idx}\neq\text{idx}^+) \leq \alpha$. Assumption 3.3 lower-bounds the per-cluster expansion $Q_S(g)\geq\beta$ for every $g$ that $\mathcal{F}$ and a linear head can implement — this is the statement that $\mathcal{F}$ **cannot** carve clusters any finer. Assumption 3.4 demands that the partition itself be implementable by some $f \in \mathcal{F}$. Assumption 3.5 is a closure-under-elementwise-scaling technicality satisfied by any net with a linear last layer.

Under these assumptions, **Theorem 3.6** gives, at $k = m$ with $\lambda > \alpha/P_{\min}$,

$$\mathbb{E}_{p_{\text{data} }}\|W\hat f(x) - e_{y(x)}\|^2 \;\leq\; \frac{\alpha}{\beta}\cdot\frac{P_{\max} }{P_{\min}-\alpha}.$$

**Eigenfunction reformulation (Section 4).** Define the graph Laplacian $L(g)(x) = g(x) - \int p_{\text{pos}}(x,x')/p_{\text{data}}(x)\cdot g(x')\,dx'$. Assumption 4.2 posits $m$ orthogonal unit-norm "approximate eigenfunctions" inside $\mathcal{F}$ with total quadratic form $\leq \varphi$. Assumption 4.3 requires any small-eigenvalue function in $\mathcal{F}$ to be (up to $\varepsilon$) a linear combination of those eigenfunctions. Assumption 4.4 makes the downstream label linear in the eigenfunctions with norm $B$ and residual $\zeta$. **Theorem 4.5** then gives

$$\mathbb{E}_{p_{\text{data} }}\|W\hat f(x) - \vec y(x)\|^2 \;\lesssim\; \zeta + B^2 k \cdot (\varepsilon + \varphi/\lambda).$$

This is strictly stronger than Thm 3.6: the same downstream task can sometimes be solved with $k = m$ smaller than the count of minimal implementable clusters.

![Eigenfunction grouping on a 2x2 cluster grid](/assets/images/paper/augmentation-graph-ib/page_008.png)
*Figure 3: with four disconnected clusters on a 2×2 grid, only two of the four eigenfunctions are linear; restricting $\mathcal{F}$ to linear models picks out the axis-aligned 2-vs-2 grouping (left) and rejects the diagonal grouping (right), so $k = 2$ suffices instead of $k = 4$ — the geometric intuition behind Theorem 4.5.*

**Theorem 4.7 (finite-sample).** Adds Rademacher complexity $\hat R_{n_{\text{pre}}}(\mathcal{F})$ and labeled sample size $n_{\text{ds}}$; the bound decomposes into the Thm 4.5 population error plus an estimation term scaling like $O(C_f^2 k^2\lambda + B^2 k/(\tilde\varphi - 2\varphi))\cdot(\hat R_{n_{\text{pre}}}(\mathcal{F}) + \sqrt{\log(k^2/\delta)/n_{\text{pre}}}) + BC_f\sqrt{\log(1/\delta)/n_{\text{ds}}}$. This is **not a bias-variance decomposition** in the classical sense — it is an approximation-error ($\zeta, \varepsilon, \varphi$) plus estimation-error split.

**Instantiations (Section 5).** Apply Thm 4.5 to four classes on synthetic distributions: linear models on the hypercube (Thm 5.3), ReLU MLPs on the same data with a richer label (Thm 5.6), Lipschitz functions on separated manifolds with internal sub-clusters (Thm 5.9), and CNNs on a translation-invariant patch task (Thm 5.12). In each case the constrained class reaches zero error with a small $k$ while a universal approximator with much larger $k$ provably fails.

## Experimental Results

The paper has exactly one quantitative table. The proxy $\varphi_m$ is computed by training a model with the spectral contrastive loss at output dim $k = m$, whitening so empirical $ff^\top = I/m$, then reporting $\min_\lambda \mathbb{E}_{p_{\text{pos}}}\|\bar f(x)-\bar f(x^+)\|^2$ over $\lambda\in\{0.1,0.3,1,3,10,30,100,300,1000\}$. The networks are initialized from an 800-epoch spectral-CL ResNet checkpoint, then fine-tuned 200 epochs on CIFAR-10.

| Architecture | $\varphi_{10}$ | $\varphi_{30}$ | $\varphi_{100}$ | $\varphi_{300}$ |
|---|---|---|---|---|
| ResNet-18  | 0.131 | 0.166 | 0.224 | 0.521 |
| **ResNet-101** | **0.053** | **0.090** | **0.163** | **0.459** |
| WRN        | 0.080 | 0.093 | 0.138 | 0.340 |
| ViT        | 0.072 | 0.108 | 0.168 | 0.389 |

Smaller $\varphi_m$ means the architecture can implement an $m$-way partition with less inter-cluster positive-pair leakage. The authors read the monotonic rise as evidence for Assumption 3.3 / 4.3, and the across-row gaps as evidence the framework can distinguish architectures.

**Ablations / robustness.** None reported. No batch-size or learning-rate sweep, no per-seed variance, no cross-dataset replication, no head-to-head comparison with the baseline $r_0$-dependent bound, and **no downstream linear-probe accuracy** — the very quantity Theorem 4.5 directly bounds. Figures 1, 2, 3 are conceptual schematics; there are no result figures.

## Limitations

**Acknowledged by the authors (Section 7).**

- Theory requires $k = m$ exactly; the overparameterized $k > m$ regime is not handled.
- All proofs target the spectral / Barlow-Twins loss $\mathcal{L}_\lambda$; InfoNCE is left open.
- The framework concerns architectural inductive bias only — the implicit bias of SGD / weight decay on the *learned* representation is not modeled.

**Under-addressed.**

- Every Section 5 instantiation assumes $\mathcal{L}_\lambda$ is *globally* minimized over $\mathcal{F}$ — not what optimization actually delivers for ReLU nets, CNNs, or Lipschitz classes.
- Assumption 3.3 (the $\beta$-lower bound on per-cluster expansion) is never empirically estimated. The empirical proxy $\varphi_m$ measures the *opposite* side (Assumption 4.2), so half the framework remains untested.
- The CIFAR-10 experiment initializes from a fully pre-trained 800-epoch spectral-CL ResNet checkpoint before measuring $\varphi_m$. Whether the same checkpoint is used to initialize ViT / WideResNet / ResNet-101 — or whether each is trained from scratch — is not described per-architecture, materially affecting the cross-architecture comparison.
- No statistical-significance reporting and no confidence intervals on $\varphi_m$.
- No attempt to connect $\varphi_m$ to downstream linear-probe accuracy on CIFAR-10, which would be the natural validation of Theorem 4.5.

## References

- Paper: [arXiv:2211.14699](https://arxiv.org/abs/2211.14699) — *A Theoretical Study of Inductive Biases in Contrastive Learning*, Jeff Z. HaoChen and Tengyu Ma.
- HaoChen, Wei, Gaidon, Ma (2021), *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss*, NeurIPS.
- Saunshi, Ash, Goel, Misra, Zhang, Arora, Krishnamurthy, Kakade (2022), *Understanding Contrastive Learning Requires Incorporating Inductive Biases*, ICML.
- Zbontar, Jing, Misra, LeCun, Deny (2021), *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*, ICML.
