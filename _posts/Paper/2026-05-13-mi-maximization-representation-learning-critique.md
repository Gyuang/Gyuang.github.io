---
title: "On Mutual Information Maximization for Representation Learning"
excerpt: "A bijective-encoder negative result that decouples MI from downstream accuracy — same true MI ranges from ~0.40 (adversarial) to ~0.90 (I_NCE/I_NWJ) linear-probe acc on MNIST."
categories:
  - Paper
  - Multimodal-Alignment
tags:
  - InfoNCE
  - InfoMax
  - Mutual-Information
  - Contrastive-Learning
  - Self-Supervised
  - Metric-Learning
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-14
last_modified_at: 2026-05-14
permalink: /paper/mi-maximization-representation-learning-critique/
---

## TL;DR
- **MI is not what's being optimized.** A bijective RealNVP encoder fixes the true `I(g_1(X^{(1)}); g_2(X^{(2)})) = I(X^{(1)}; X^{(2)})` by construction, yet linear-probe accuracy on MNIST sweeps from **~0.40 (adversarial training) to ~0.90 (I_NCE / I_NWJ)** at *identical* true MI — Figure 1 is essentially a proof.
- **Tighter MI bounds hurt.** An MLP critic gives a strictly higher I_NWJ value than a bilinear critic, but **~5 pp lower linear-probe accuracy** (Fig 3). The matched-bound protocol (Fig 4a,b) forces I_EST = t and still finds ConvNet encoders beating MLPs by **~8–10 pp** — encoder inductive bias drives downstream, not the MI value.
- **CPC / DIM are metric learning in MI clothing.** I_NCE with separable inner-product critic and shared encoder is *algebraically identical* (App C) to Sohn's (2016) multi-class K-pair triplet loss. Under the non-i.i.d. negatives CPC actually uses on images, I_NCE is provably **neither** a lower nor an upper bound on true MI (Fig 4c, App D) — so the "MI maximization" framing of CPC / DeepInfoMax / CMC is largely post-hoc rationalization of a triplet-loss method.

## Motivation

By 2019 a family of self-supervised methods — CPC, DeepInfoMax, CMC, AMDIM, TCN — had pushed ImageNet linear-probe SOTA under a single banner: maximize `I_EST(g_1(X^{(1)}); g_2(X^{(2)}))`, a sample-based lower bound on the mutual information between two views. The story was clean: bigger MI ⇒ more "shared content" preserved ⇒ better features.

Two facts make that story uncomfortable. First, MI is invariant under any smooth bijection — `I(g(X); Y) = I(X; Y)` whenever `g` is invertible — so by itself MI says **nothing** about representation geometry. Second, MI lower bounds in high dimension suffer exponential sample complexity (McAllester & Stratos 2018), so the gradient signal driving CPC at large `K` is unlikely to be tracking true MI in any honest sense. Tschannen et al. push on both points and ask: is the recipe really an MI recipe, or are the gains coming from encoder architecture, critic capacity, and contrastive loss's resemblance to deep metric learning?

The medical-AI relevance is indirect but real: practitioners porting CPC/SimCLR-style pretraining to chest X-rays or histology often treat "more MI = better features" as a design heuristic. This paper says that heuristic is wrong, and the right knobs to turn are encoder bias and negative-sample design.

## Core Innovation

The paper does not propose a method; it proposes a controlled-experiment protocol that **isolates MI from everything else**. Three moves carry the argument:

1. **Bijective encoders fix true MI exogenously.** Use a 30-coupling-layer RealNVP for `g_1, g_2`. By the change-of-variables, `I(g_1(X^{(1)}); g_2(X^{(2)})) = I(X^{(1)}; X^{(2)})` for **any** parameter setting. Now sweep the training loss and watch downstream accuracy move while true MI stays put.
2. **Matched-bound protocol.** Replace `max I_EST` with `min |I_EST - t|` for `t ∈ {2, 4}`. Different encoder families converge to *the same* I_EST value; any remaining downstream gap is entirely encoder inductive bias.
3. **Algebraic bridge to metric learning.** Re-derive I_NCE as a K-pair triplet loss to show that the practical loss landscape of CPC is metric-learning, not information-theoretic.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Larger MI is not predictive of downstream accuracy. | Bijective RealNVP holds true MI constant by construction; I_EST-trained encoders reach ~0.89–0.90 linear probe (Fig 1a,b), an adversarially trained bijective encoder collapses to ~0.40 at the **same** true MI (Fig 1c). | ⭐⭐⭐ — a *mathematical* decoupling, not a statistical correlation. The strongest claim in the paper. |
| C2 | I_NCE / I_NWJ gradients prefer ill-conditioned (hard-to-invert) encoders even when initialized at the identity. | Skip-MLP initialized at identity; Jacobian condition number rises monotonically to ~10^8 while linear-probe accuracy rises (Fig 2c, Fig 5). | ⭐⭐ — clear within the setting; single architecture family. |
| C3 | Higher-capacity critics give tighter MI bounds but **worse** representations. | Fig 3: MLP critic achieves higher I_NWJ but barely beats the pixel baseline (~0.84); bilinear/separable critics give higher accuracy (~0.89) at lower I_NWJ. Replicated on CIFAR-10 (App G). | ⭐⭐⭐ — large effect size, two estimators, two datasets. The rank inversion the paper's title is built on. |
| C4 | Encoder architecture matters more than estimator value. | Matched-bound protocol Fig 4a,b: ConvNet beats MLP by ~8–10 pp at every matched `(estimator, t)` pair on MNIST; replicated on CIFAR-10 (Fig 9). | ⭐⭐⭐ — cleanest controlled comparison in the paper. |
| C5 | I_NCE with non-i.i.d. negatives is neither a lower nor upper bound on MI. | Theorem in App D + synthetic 2-D Gaussian (shared `Z` per batch) where the closed-form true MI is *exceeded* by both I_NCE and I_NWJ (Fig 4c). | ⭐⭐⭐ for the theorem; ⭐⭐ for empirical relevance — synthetic regime; not demonstrated that CPC-on-ImageNet operates where the bound is violated by a meaningful margin. |
| C6 | I_NCE with separable inner-product critic = Sohn (2016) K-pair metric-learning loss. | Algebraic rewriting in Sec 4 / App C. | ⭐⭐⭐ — an identity, not an experiment. |
| C7 | Metric learning is a better explanation than InfoMax for the practical success of CPC/DIM/CMC. | C6 identity + observation that recent design choices (bilinear critic, dependent negatives, hard-negative mining) conflict with the MI view but align with metric learning. | ⭐⭐ — well argued, but no head-to-head experiment. The over-stated half of the paper. |

**Honest reading.** The bijection-based experiments (C1) and matched-bound experiments (C4) are essentially proofs; they survive at full strength. The over-statement is that **all empirical evidence is MNIST + CIFAR-10 at modest scale.** The CPC/AMDIM/CMC ImageNet numbers are never re-run; the paper argues by mechanism. That leaves the obvious counter — "maybe at ImageNet scale the MI signal does dominate" — formally open. No CIs / no significance tests on the accuracy curves either.

## Method & Architecture

![Figure 1](/assets/images/paper/mi-maximization/page_004.png)
*Figure 1: Same true MI, three very different linear-probe outcomes. (a) Bijective RealNVP encoders push I_EST up; (b) linear-probe accuracy follows in lock-step despite the true MI being held constant by construction; (c) an adversarially trained bijective encoder collapses accuracy to ~0.40 at the same true MI. This single figure carries the paper's argument.*

The experimental setup that supports Figure 1 is small and reproducible:

- **Two-view task.** Split an MNIST image into top half `X^{(1)} ∈ [0,1]^{392}` and bottom half `X^{(2)} ∈ [0,1]^{392}`. Train encoders `g_1, g_2` and a critic `f` to maximize a sample-based lower-bound estimator `I_EST`. Evaluate by training a linear (SAGA) classifier on `g_1(x_{top})` over all training labels; report 20-run mean test accuracy. Baselines: linear-on-pixels ≈ 85%; supervised MLP/ConvNet ≈ 94%.
- **Estimators.**

  $$\hat{I}_{\text{NCE}}(X; Y) = \mathbb{E}\!\left[\frac{1}{K}\sum_{i=1}^{K} \log \frac{e^{f(x_i, y_i)}}{\frac{1}{K}\sum_{j=1}^{K} e^{f(x_i, y_j)}}\right]$$

  $$\hat{I}_{\text{NWJ}}(X; Y) = \mathbb{E}_{p(x,y)}[f(x,y)] - e^{-1}\,\mathbb{E}_{p(x)}\mathbb{E}_{p(y)}[e^{f(x,y)}]$$

  Note `I_NCE` is upper-bounded by `log K ≈ 4.85` at `K = 128` — so it cannot track high MI even in principle.
- **Critic families.** Bilinear `f(x,y) = x^T W y`; separable `f = φ_1(x)^T φ_2(y)` (~40k params); concatenated MLP `f = φ([x, y])` (~40k params). Higher capacity → tighter lower bound, in theory.
- **Encoder families.** RealNVP (30 coupling layers, bijective by construction); 4-layer skip-MLP initialized at identity; ConvNet (two 5×5 conv layers, stride 2, LayerNorm before pool).

![Figure 2](/assets/images/paper/mi-maximization/page_005.png)
*Figure 2: The skip-MLP encoder is initialized at the identity — a true-MI optimum — and the I_EST gradient drives it monotonically toward an ill-conditioned (Jacobian condition number ~10^8), hard-to-invert solution. The estimator is actively destroying MI to make features linearly separable.*

The algebraic identity that grounds the metric-learning view (Section 4 / Appendix C):

$$\hat{I}_{\text{NCE}} = \log K - \mathbb{E}\!\left[\frac{1}{K}\sum_{i=1}^{K} \log\!\left(1 + \sum_{j \ne i} e^{f(x_i, y_j) - f(x_i, y_i)}\right)\right]$$

With symmetric separable critic `f(x, y) = φ(x)^T φ(y)` and shared encoder `g = g_1 = g_2`, the right-hand expectation **is** Sohn (2016)'s multi-class K-pair loss up to constants. Maximizing I_NCE ≡ minimizing a K-pair triplet loss with an inner-product metric.

## Experimental Results

This paper has no leaderboard — every claim is a behavioural curve. The qualitative scoreboard reconstructed from Figures 1–4:

| Setting | Encoder | Critic | Estimator | Linear-probe acc (MNIST, top half) | Notes |
|---|---|---|---|---|---|
| Pixel baseline | — | — | — | ≈ 0.85 | linear on raw `x_top` |
| Supervised ceiling | MLP / ConvNet end-to-end | — | — | ≈ 0.94 | reference |
| Bijective RealNVP, MI-max | RealNVP | bilinear | I_NCE | ≈ 0.89 | true MI constant by construction (Fig 1a,b) |
| Bijective RealNVP, MI-max | RealNVP | bilinear | I_NWJ | ≈ 0.90 | same true MI (Fig 1a,b) |
| **Bijective RealNVP, adversarial** | **RealNVP** | **—** | **adversarial CE** | **≈ 0.40** | **same true MI (Fig 1c) — the smoking gun** |
| Skip-MLP, init=identity | skip-MLP | bilinear | I_NCE | 0.84 → 0.90 as Jacobian cond. ~10^8 (Fig 2) | |
| MLP encoder, bilinear critic | MLP | bilinear | I_NWJ | ≈ 0.89 | looser bound (Fig 3) |
| MLP encoder, separable critic | MLP | separable | I_NWJ | ≈ 0.87 | tighter bound (Fig 3) |
| **MLP encoder, MLP critic** | **MLP** | **MLP** | **I_NWJ** | **≈ 0.84** | **tightest bound, barely above pixels (Fig 3)** |
| Matched-MI MLP encoder | MLP (~238k) | bilinear | I_NCE / I_NWJ at `t ∈ {2, 4}` | ≈ 0.80–0.84 | Fig 4a,b |
| **Matched-MI ConvNet encoder** | **ConvNet (~220k)** | **bilinear** | **I_NCE / I_NWJ at `t ∈ {2, 4}`** | **≈ 0.90–0.92** | **+8–10 pp over MLP at identical I_EST (Fig 4a,b)** |
| CIFAR-10, MLP encoder, MLP critic | MLP | MLP | I_NCE / I_NWJ | ≈ 0.27 | pixel baseline ≈ 0.24 (App G) |
| CIFAR-10, ConvNet, matched `t` | ConvNet | bilinear | I_NCE / I_NWJ | ≈ 0.36–0.40 | replicates MNIST pattern |

![Figure 3](/assets/images/paper/mi-maximization/page_006.png)
*Figure 3: Critic capacity ablation. The MLP critic produces a strictly tighter I_NWJ estimate than the bilinear / separable critics — and lands ~5 pp **lower** on linear-probe accuracy. The exact analogue of "tighter ELBOs are not necessarily better" (Rainforth et al. 2018), and the rank inversion the paper is named for.*

![Figure 4](/assets/images/paper/mi-maximization/page_007.png)
*Figure 4: (a,b) Matched-MI protocol — minimize `|I_EST - t|` for `t ∈ {2, 4}` so both encoders reach the same lower-bound value; ConvNet beats MLP by ~8–10 pp at every matched point, both estimators. (c) Synthetic 2-D Gaussian with non-i.i.d. negatives (shared `Z` per batch): both I_NCE and I_NWJ **exceed** the closed-form true MI — they are no longer lower bounds.*

The Figure 4c synthetic experiment is the formal cornerstone of the metric-learning argument. In a 2-D Gaussian with `Σ_Z = [[1, -0.5], [-0.5, 1]]` and shared `Z` per batch, true MI is closed-form computable, and both I_NCE and I_NWJ estimates blow past it. The non-i.i.d. regime is exactly what CPC-on-images and CMC use in practice — negatives are other patches from the same image, other utterances from the same speaker. The "MI lower bound" guarantee evaporates the moment you adopt those design choices.

## Limitations

**Acknowledged:**
- No replacement objective is proposed. The paper argues the field needs one (F-information, V-information, Wasserstein-dependency) but doesn't supply it.
- The empirical relevance of the non-i.i.d. negative-sample violation isn't quantified on real benchmarks — true MI is uncomputable there.
- Linear-probe evaluation itself might be a bad protocol; non-linear evaluation (Bachman 2019, Tian 2019) "defeats the purpose of learning efficiently transferable representations."

**Not addressed:**
- No experiments at ImageNet scale. All evidence is MNIST + CIFAR-10; whether the same biases dominate with ResNet-50 encoders on 1M images is *assumed*, not shown.
- No standard deviations or significance tests on accuracy curves — only 20-run means.
- The matched-bound experiment uses a bilinear critic for both encoder architectures, but C3 says critic choice matters a lot. The full cross-product `{critic} × {encoder} × {estimator} × {t}` isn't run.
- The metric-learning re-interpretation predicts that explicit metric-learning improvements (hard-negative mining, margin-based losses, semi-hard mining) should transfer cleanly to self-supervised pretraining. The paper doesn't run such a comparison.
- No medical-imaging or out-of-domain transfer experiments — relevant given how heavily CPC/SimCLR-style pretraining is now used on histology and CXR.

## Why It Matters for Medical AI

Self-supervised pretraining on unlabelled medical images is now standard practice — SimCLR / MoCo / CPC variants on chest X-rays, OCT, histology tiles. If "more MI" is wrong as a design heuristic, what should replace it for medical practitioners?

This paper's answer, by elimination: tune the **encoder architecture**, the **critic family**, and the **negative-sample distribution**, not the MI estimator. Concretely:

- Use a bilinear (or low-capacity separable) critic; do not assume tighter MI bounds help.
- Match the encoder inductive bias to the modality (ConvNets / hierarchical Transformers for 2-D / 3-D images; not generic MLPs).
- Design negatives explicitly as a metric-learning problem — hard-negative mining, patient-level or scanner-level negatives — rather than reaching for "more MI" via larger queues.

The combination of this paper, Wang & Isola 2020 (alignment + uniformity), and Saunshi et al. 2019 (latent-class analysis) means the InfoMax framing is no longer the right mental model for *any* of CPC, SimCLR, MoCo, or their medical-domain ports. The right mental model is metric learning on the hypersphere.

## References

- **Paper:** Tschannen, Djolonga, Rubenstein, Gelly, Lucic. *On Mutual Information Maximization for Representation Learning*. ICLR 2020. arXiv:1907.13625.
- **Closely related:** Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020 — the constructive geometric replacement for the MI framing.
- **Closely related:** McAllester & Stratos, *Formal Limitations on the Measurement of Mutual Information*, AISTATS 2020 — the exponential-sample-complexity result this paper leans on.
- **Methods critiqued:** van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, arXiv:1807.03748; Hjelm et al., *Learning Deep Representations by Mutual Information Estimation and Maximization* (DeepInfoMax), ICLR 2019; Tian et al., *Contrastive Multiview Coding* (CMC), ECCV 2020; Bachman et al., *Learning Representations by Maximizing Mutual Information Across Views* (AMDIM), NeurIPS 2019.
- **Metric-learning identity:** Sohn, *Improved Deep Metric Learning with Multi-class N-pair Loss Objective*, NeurIPS 2016.
- **MI lower bounds:** Nguyen, Wainwright, Jordan (2010) for I_NWJ; Belghazi et al. (2018) MINE; Poole et al. (2019) variational-bound survey.
