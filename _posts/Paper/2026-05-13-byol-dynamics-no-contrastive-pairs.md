---
title: "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs"
excerpt: "A linear-toy gradient-flow analysis of BYOL/SimSiam yields an analytic anti-collapse mechanism (predictor + stop-gradient + EMA + weight decay) and a closed-form predictor — DirectPred — that matches BYOL's 2-layer MLP on ImageNet (72.4% vs 72.5% top-1, 300 ep) and beats a gradient-trained linear predictor by +2.5 pt at 300 ep / +5.0 pt at 60 ep."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/byol-dynamics-no-contrastive-pairs/
tags:
  - BYOL
  - SimSiam
  - DirectPred
  - Self-Supervised-Learning
  - Non-Contrastive-Learning
  - Stop-Gradient
  - EMA
  - Weight-Decay
  - Representation-Learning
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- **Why does BYOL/SimSiam not collapse without negatives?** The authors derive a closed-form gradient-flow theory for a 2-layer linear BYOL and show that **predictor + stop-gradient + EMA + weight decay jointly stabilize a non-collapsed fixed point** — none of the four ingredients is decorative, and the explanation does **not** rely on batch-norm tricks.
- **Headline theoretical object.** Per eigen-mode the dynamics live in a 3-D state $(p_j, s_j, \tau)$ and admit an exact integral of motion $s_j(t) = \alpha_p^{-1} p_j^2(t) + e^{-2\eta t} c_j$. With $\eta > 0$ trajectories are forced onto an **invariant parabola** $s_j = \alpha_p^{-1} p_j^2$ — that parabola is the structured attractor that prevents collapse.
- **Headline empirical payoff — DirectPred.** A theory-derived, optimizer-free linear predictor $W_p = \hat U \, \mathrm{diag}(\sqrt{s_j} + \varepsilon \max_j s_j) \, \hat U^\top$ read straight off the eigen-decomposition of the running input correlation $\hat F$ reaches **ImageNet 72.4% / 91.0% top-1/top-5 at 300 epochs** — matching BYOL's default 2-layer MLP (72.5 / 90.8) and beating an SGD-trained linear predictor by **+2.5 pt (300 ep)** and **+5.0 pt (60 ep, 64.4 vs 59.4)**. No variance bars are reported on ImageNet.

## Motivation

Contrastive SSL (SimCLR, MoCo) leans on negative pairs and the standard intuition that "without negatives, everything collapses to a constant." BYOL and SimSiam broke that intuition: with only positive pairs, a predictor head, stop-gradient, and (optionally) an EMA target, they reach SOTA. The community had only empirical recipes — large batch, slow EMA, predictor LR > backbone LR, weight decay — and a popular "BatchNorm is secretly doing contrastive learning" folk-explanation, but no falsifiable model linking the hyperparameters to the absence of collapse.

This paper exists to (a) **prove** that the mechanism survives in a bias-free, BN-free *linear* model — isolating it from batch-norm side effects — and (b) **act on the proof** by replacing the trained predictor with a closed-form one (DirectPred). For multimodal-alignment and medical-AI readers, the practical relevance is that BYOL/SimSiam-style non-contrastive pretraining is widely used (CT, MRI, pathology) on small datasets where the SimCLR-style large batch is infeasible — and the analysis here pinpoints which knobs actually drive non-collapse rather than which knobs happen to be popular.

## Core Innovation

- **A 2-layer linear gradient-flow model of BYOL/SimSiam** (Lemma 1) that exposes three ODEs over $(W_p, W, W_a)$ with augmented-mean covariance $X$, augmentation-noise covariance $X'$, predictor LR multiplier $\alpha_p$, EMA rate $\beta$, and weight decay $\eta$.
- **Two structural theorems with no further assumptions.** Theorem 1 (analytic balancing $W W^\top = \alpha_p^{-1} W_p^\top W_p + e^{-2\eta t} C$) shows weight decay forces predictor learning into the backbone; Theorem 2 proves that *removing* stop-gradient gives $\dot{\mathrm{vec}}(W) = -H(t) \mathrm{vec}(W)$ with $H(t) \succeq 0$, so $W \to 0$ — collapse is unavoidable without stop-gradient.
- **Decoupled per-mode dynamics with an invariant parabola.** Under proportional-EMA, isotropic-data, symmetric-$W_p$ assumptions, Theorem 3 + Eqs 11-13 reduce the joint dynamics to scalar ODEs $(\dot p_j, \dot s_j, \dot\tau)$ with the integral of motion $s_j = \alpha_p^{-1} p_j^2$ — the dynamical core of the entire paper.
- **DirectPred.** A closed-form predictor read off the running input correlation $\hat F$, no SGD on $W_p$ at all. Two practical defaults that matter: $\rho = 0.3$ (EMA on $\hat F$), $\varepsilon = 0.1$ (eigenvalue-floor), and $\ell_2$-normalized features so absolute $s_j$ magnitude is irrelevant.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Predictor + stop-gradient is essential; removing either causes collapse. | Theorem 2 — analytic proof for SimSiam-form linear model; cited BYOL/SimSiam ablations. | Linear theory + Chen & He 2020 | ⭐⭐⭐ |
| C2 | A "balancing" identity ties online and predictor weights; weight decay accelerates it. | Theorem 1 closed-form $W W^\top = \alpha_p^{-1} W_p^\top W_p + e^{-2\eta t} C$; Tbl. 2 shows $\eta = 0$ underperforms $\eta = 4\cdot 10^{-4}$ (70.6-71.4 vs 74.5). | STL-10, ResNet-18 | ⭐⭐ |
| C3 | Eigenspaces of $W_p$ and $F$ align during training. | Theorem 3 + Fig. 2 cosine-alignment curves on a real ResNet-18/STL-10 run. | STL-10 (and synthetic) | ⭐⭐⭐ |
| C4 | EMA acts as an automatic curriculum on $\tau$; large $\alpha_p$ substitutes for it. | Synthetic decoupled-ODE plots (Fig. 5) sweeping $\beta, \eta$; Fig. 6 real STL-10/CIFAR-10 sweep over $\alpha_p$ without EMA; BYOL Tbl. 22 cited. | STL-10, CIFAR-10, synthetic | ⭐⭐ |
| C5 | Weight decay on the predictor alone makes symmetric $W_p$ work without EMA. | Tbl. 5: $\bar\eta_p = 4\cdot 10^{-4}, \bar\eta_s = 0$, sym $W_p$, no EMA $\to$ 73.01-73.67%, vs 36-43% when both are 0. | STL-10 only | ⭐⭐ |
| C6 | Plugging in the closed-form "always-optimal" predictor every minibatch fails — the popular folklore is wrong. | Tbl. 1: 32-41% top-1 vs ~74% for SGD predictor. | STL-10 only | ⭐⭐ |
| C7 | DirectPred matches the 2-layer MLP predictor on ImageNet without gradient training on $W_p$. | Tbl. 9: 72.4 / 91.0 vs 72.5 / 90.8 (300 ep, single run, no error bars). | ImageNet (300 ep) | ⭐⭐ |
| C8 | DirectPred outperforms the gradient-trained linear predictor by +2.5 pt (300 ep) and +5 pt (60 ep) on ImageNet. | Tbl. 9. | ImageNet | ⭐⭐ |
| C9 | Linear-theory predictions transport to nonlinear ResNet-18/ResNet-50. | Same direction of effect in Fig. 2 (real), Tbl. 3, Tbl. 5, Tbl. 6 across STL-10/CIFAR/ImageNet. | Three datasets | ⭐⭐ |
| C10 | "Strong weight decay $\to$ collapse unavoidable" once $\eta > \tau^2/(4(1+\sigma^2))$ (Obs#5). | Synthetic Fig. 3 (right) and Fig. 5 only; no real-image experiment driving $\bar\eta$ past the threshold. | Synthetic | ⭐ |

**Honest read.** The strongest contributions are C1 and C3 — Theorem 2 is a clean analytic collapse proof and Fig. 2 visualizes eigenspace alignment in a real (non-linear) ResNet-18 run, which is a non-trivial bridge from the linear toy to practice. C6 is the most underrated finding: it kills the obvious "the predictor just tracks its input correlation optimally" heuristic with one short table. The ImageNet headline numbers (C7, C8) are reported as **single point estimates with no variance bars** — the 0.1 pt gap between DirectPred and 2-layer MLP should not be read as "matches" without seeds, although the +2.5 pt and +5.0 pt gaps over the linear-SGD baseline are large enough to be believable. Where the paper over-reaches: the abstract sells the theory as "explaining" BYOL/SimSiam, but it explains only a *linear, bias-free, isotropic-data, symmetric-predictor* slice; real BYOL has BN, ReLU, a wide 4096-hidden predictor, and anisotropic image statistics (Section 5 acknowledges this with intuition only).

## Method & Architecture

![Per-mode state-space flow with the invariant parabola](/assets/images/paper/byol-dynamics/fig_p006_01.png)
*Figure 1 (paper Fig. 3): Flow of the per-mode dynamics $(p_j, s_j)$ under the decoupled ODEs of Theorem 3. **Left:** no weight decay — two perpendicular fixed-point branches. **Middle:** weak decay — trajectories converge onto the invariant parabola $s_j = \alpha_p^{-1} p_j^2$, which is the non-collapsed attractor. **Right:** strong decay — only the collapsed fixed point at the origin survives. This single figure is the dynamical core of the paper: BYOL has a structured attractor, not a magical anti-collapse trick.*

### Setup — a 2-layer linear BYOL

Online branch $f_1 = W x_1$, target branch $f_{2a} = W_a x_2$, predictor output $W_p f_1$, loss

$$J = \tfrac{1}{2} \, \mathbb{E} \, \| W_p f_1 - \mathrm{sg}(f_{2a}) \|^2,$$

with stop-gradient on the target branch and EMA on $W_a$. Setting $W_a = W$ recovers SimSiam.

### Lemma 1 — continuous-time gradient flow

$$\dot W_p = \alpha_p \big(-W_p W (X + X') + W_a X\big) W^\top - \eta W_p,$$

$$\dot W = W_p^\top \big(-W_p W (X + X') + W_a X\big) - \eta W,$$

$$\dot W_a = \beta \, (-W_a + W),$$

with $X = \mathbb{E}[\bar x \bar x^\top]$ the augmented-mean covariance, $X' = \mathbb{E}[V_{x'\mid x}[x']]$ the augmentation noise covariance, $\alpha_p$ the predictor LR multiplier, $\beta$ the EMA rate, $\eta$ the weight decay.

### Theorems 1-2 — balancing and stop-gradient-required

**Theorem 1 (balancing).** $W W^\top = \alpha_p^{-1} W_p^\top W_p + e^{-2\eta t} C$ — weight decay erases the initial imbalance term, so anything the predictor learns is forced into the backbone weights, which are the only weights used downstream.

**Theorem 2 (stop-gradient is essential).** Without stop-gradient, the SimSiam-form update reduces to $\dot{\mathrm{vec}}(W) = -H(t)\, \mathrm{vec}(W)$ with $H(t) \succeq 0$, so $W \to 0$ — **analytic collapse**. The proof is constructive and does not depend on data covariance shape.

### Theorem 3 + Eqs 11-13 — per-mode $(p_j, s_j, \tau)$ dynamics

Under proportional EMA $W_a = \tau W$, isotropic data $X = I$, $X' = \sigma^2 I$, and symmetric $W_p$, **Theorem 3 (eigenspace alignment)** reads

$$\frac{d}{dt} [F, W_p] = -[F, W_p] K - K [F, W_p], \qquad K = (1+\sigma^2) \big(\tfrac{\alpha_p}{2} F + W_p^2 - \tfrac{\tau}{1+\sigma^2} W_p\big) + \tfrac{3}{2} \eta I.$$

Whenever $\lambda_{\min}(K) \ge \lambda_0 > 0$, the commutator decays exponentially and $W_p$ and $F = W X W^\top$ share an eigenbasis. With that alignment in hand, the joint dynamics decouple into per-mode scalar ODEs (Eqs 11-13):

$$\dot p_j = \alpha_p s_j \big(\tau - (1+\sigma^2) p_j\big) - \eta p_j,$$

$$\dot s_j = 2 p_j s_j \big(\tau - (1+\sigma^2) p_j\big) - 2 \eta s_j,$$

$$s_j \dot \tau = \beta (1 - \tau) s_j - \tau \dot s_j / 2.$$

The exact integral of motion

$$s_j(t) = \alpha_p^{-1} p_j^2(t) + e^{-2\eta t} c_j$$

shows that, with $\eta > 0$, every trajectory is asymptotically pinned to the **invariant parabola** $s_j = \alpha_p^{-1} p_j^2$ visualized in Figure 1. From this scalar system the authors read off ten observations (Obs#1-#10, Tbl. 4) linking each hyperparameter to a direction of effect: large $\alpha_p$ flattens the parabola so the basin of collapse shrinks; too large $\alpha_p$ stops $s_j$ from growing with $p_j$; weight decay forgets initialization but expands the collapse basin via the unstable fixed point $p^*_{j-}$; EMA acts as an automatic curriculum by keeping $\tau$ small early so a small $p_j$ already satisfies the alignment-stability bound.

### Ablation vector fields — why each component is needed

![Vector fields without target learning, with tied predictor, and without predictor](/assets/images/paper/byol-dynamics/fig_p022_01.png)
*Figure 2 (paper appendix): Vector fields under three ablations. With a fixed target ($W_a$ not learned) the only stable fixed point is the origin (collapse). With a tied predictor $W_p = I$ a non-trivial fixed point on the parabola survives. Removing the predictor entirely sends $W \to 0$. Together this is the geometric companion to Theorem 2.*

### From theory to practice — DirectPred

Maintain a running estimate $\hat F \leftarrow \rho \hat F + (1-\rho) \, \mathbb{E}_B[f f^\top]$ — **uncentered correlation, not covariance** (centring hurts in their experiments). Eigen-decompose $\hat F = \hat U \Lambda_F \hat U^\top$ with $\Lambda_F = \mathrm{diag}(s_1, \ldots, s_d)$, set $p_j = \sqrt{s_j} + \varepsilon \cdot \max_j s_j$, and write the predictor in closed form:

$$W_p = \hat U \, \mathrm{diag}(p_j) \, \hat U^\top.$$

No SGD on $W_p$ at all. Optional hybrid: refresh $W_p$ from $\hat F$ every `freq` minibatches and let gradient updates on $W_p$ run in between — `freq = 5` works best on STL-10. Defaults that matter: $\rho = 0.3$, $\varepsilon = 0.1$, EMA on the backbone preserved, $\ell_2$-normalized features.

## Experimental Results

ImageNet-1k uses ResNet-50 with two recipes — 60 epochs (asymmetric loss, batch 256, SGD) and 300 epochs (symmetric loss, batch 4096, LARS, BYOL-faithful recipe). Smaller-scale ablations are on STL-10 (96×96, 100 ep, ResNet-18) and CIFAR-10. **Single seed, no variance bars on ImageNet.** Augmentations follow BYOL/SimCLR (random crop, color jitter, blur, solarisation).

### Table 1 — ImageNet predictor comparison (Tbl. 9; ResNet-50, top-1 / top-5 %)

| Predictor variant | 60 ep top-1 | 60 ep top-5 | 300 ep top-1 | 300 ep top-5 |
|---|---|---|---|---|
| 2-layer MLP predictor (BYOL default) | 64.7 | 85.8 | 72.5 | 90.8 |
| Linear predictor (SGD) | 59.4 | 82.3 | 69.9 | 89.6 |
| **DirectPred (ours, linear, no SGD on $W_p$)** | **64.4** | **85.8** | **72.4** | **91.0** |

DirectPred matches the 2-layer MLP within 0.1 pt at 300 ep (no error bars — interpret with caution) and beats the SGD linear predictor by **+5.0 pt at 60 ep** and **+2.5 pt at 300 ep**. The 60-ep gap is the more informative number because it is large relative to typical SSL seed variance.

### Table 2 — STL-10 100-ep BYOL with linear predictor (Tbl. 3; sym = symmetric $W_p$, top-1 %)

| Setting | sym, no bias | regular, no bias | sym, bias | regular, bias |
|---|---|---|---|---|
| EMA | 75.09 ± 0.48 | 74.51 ± 0.47 | 74.52 ± 0.29 | 74.16 ± 0.33 |
| no EMA | 36.62 ± 1.85 | **72.85 ± 0.16** | 36.04 ± 2.74 | **72.13 ± 0.53** |

This is one half of the load-bearing falsifier: symmetric $W_p$ + no EMA collapses to ~36% (consistent with theory predicting $K \succ 0$ requires either EMA or a non-symmetric escape route), while regular (non-symmetric) $W_p$ survives without EMA at ~72-73%.

### Table 3 — STL-10 DirectPred sweep over $\rho$, $\varepsilon$ (Tbl. 6, EMA on, 100 ep, top-1 %)

| ε → | 0 | 0.01 | 0.1 | 0.5 |
|---|---|---|---|---|
| ρ = 0.3 | 76.77 ± 0.24 | 77.11 ± 0.35 | **77.86 ± 0.16** | 75.06 ± 1.10 |
| ρ = 0.5 | 76.65 ± 0.20 | 76.76 ± 0.33 | 77.56 ± 0.25 | 75.22 ± 0.81 |

Best at $(\rho, \varepsilon) = (0.3, 0.1)$. At longer epochs (Tbl. 8), DirectPred reaches **80.28 ± 0.62** at 500 ep on STL-10 (vs SGD baseline 75.25 ± 0.74) and 89.56 ± 0.13 on CIFAR-10 (vs 89.33 ± 0.27 baseline; the gap closes at long epochs, as expected).

### The two load-bearing ablations

The most discriminating evidence against the "BatchNorm secretly does it" story is the pair **Tbl. 1 + Tbl. 5**, both run in the BN-free linear-predictor setup:

- **Tbl. 1 — naive "always-optimal predictor" collapses.** Plugging in the closed-form least-squares optimum every $N$ minibatches makes STL-10 top-1 fall to ~35%, far below the 74% a gradient-trained linear predictor reaches. The predictor must **track** $\hat F$ while the backbone is also being shaped by it — mere optimality is insufficient. This kills the obvious "the predictor just memorizes the input correlation" explanation.
- **Tbl. 5 — symmetric $W_p$ works without EMA iff $\eta_p > 0$, $\eta_s = 0$.** Setting weight decay on the predictor only ($\bar\eta_p = 4\cdot 10^{-4}$) and zero weight decay on the trunk ($\bar\eta_s = 0$) recovers 73.01-73.67% top-1 with sym $W_p$ + bias and **no EMA**, vs ~36-43% when both decays are zero. Fig. 6 is the second escape route: symmetric $W_p$ no-EMA also works once $\alpha_p > 1$.

Together these two tables sharply discriminate the proposed dynamics from the BN-trick story, **because there is no BN anywhere in this linear-predictor setup**. Fig. 2 then confirms eigenspace alignment empirically on a real ResNet-18/STL-10 run: eigenvalues of $F$ and $W_p$ evolve in step-function fashion, their eigenspaces align (normalized correlation $\to 1$), and $W_p$ drifts toward symmetry — none of which is enforced.

## Limitations

**Authors acknowledge:**
- Theory covers only **2-layer linear bias-free** networks. No analysis of multi-layer or non-linear predictors; the "fat 2-layer MLP wins by lucky initialisation" claim (Section 5) is intuition only.
- Non-symmetric $W_p$ is left to future work — only an $A + B$ decomposition is sketched.
- The two-layer predictor used by real BYOL is not analyzed; DirectPred only sets a **linear** $W_p$.

**Not addressed (my read).**
- **No variance on ImageNet.** Tbl. 9 reports single numbers — the 0.1 pt gap between DirectPred and 2-layer is well within typical run-to-run noise.
- **Small-batch / small-dataset regime.** DirectPred relies on a good moving-average estimate of $\hat F$. The paper uses batch 256 and 4096; no test of how performance degrades when $\hat F$ is estimated from, say, 32-64 samples — exactly the regime where medical SSL pretraining operates.
- **No domain-shift or transfer evaluation.** All eval is linear-probe on the same domain. Whether DirectPred's representation transfers as well as BYOL's to detection, segmentation, or medical fine-tuning is untested.
- **Cost.** A per-minibatch 256×256 eigen-decomposition is cheap; for ViT-style 768/1024-dim features `freq > 1` becomes necessary. No wall-clock benchmark.
- **BatchNorm interaction.** The whole pitch is that BN is *not* the explanation, but real BYOL uses BN heavily. The paper does not isolate which fraction of BYOL's gain comes from BN vs from the dynamics they describe.
- **Empirical falsifier for Obs#5 missing.** No real-image experiment drives $\bar\eta$ past the predicted "strong weight decay $\to$ collapse" threshold.

## Why It Matters for Multimodal Alignment

The paper is the cleanest dynamical-systems account of why a *non-contrastive*, no-negatives objective avoids representation collapse — exactly the question that downstream alignment work (BYOL-style two-tower medical encoders, SimSiam variants for histopathology, predictor-augmented multimodal alignment recipes) inherits whenever it drops InfoNCE negatives. The invariant parabola $s_j = \alpha_p^{-1} p_j^2$ is a structured-attractor description that complements the alignment-uniformity and dimensional-collapse pictures studied for contrastive SSL: where InfoNCE methods fail by under-filling the embedding ball, BYOL/SimSiam methods succeed by riding a parabolic curve in $(p, s)$ space, and **weight decay + stop-gradient + EMA + predictor** are the four ingredients that keep the system on that curve. The two load-bearing falsifiers (Tbl. 1, Tbl. 5) are also a useful template for any future "the projector / predictor secretly does X" claim — they show how to disprove a folk explanation by stripping the architecture to a BN-free linear model and re-running. The honest gap is that the theory is linear and bias-free, so DirectPred should be treated as a controlled validation of the dynamics rather than a drop-in replacement for the standard BYOL MLP head, especially on small medical-imaging batches where $\hat F$ is poorly estimated.

## References

- Paper (arXiv 2102.06810v4, 8 Oct 2021): https://arxiv.org/abs/2102.06810
- Code (DirectPred): https://github.com/facebookresearch/luckmatters/tree/master/ssl
- Venue: ICML 2021 (PMLR 139)
- Related work: Grill et al. 2020 (BYOL); Chen & He 2020 (SimSiam); Chen et al. 2020 (SimCLR); He et al. 2019 (MoCo); Richemond et al. 2020 (BYOL without BatchNorm); Hua et al. 2021 (dimensional collapse in non-contrastive SSL).
