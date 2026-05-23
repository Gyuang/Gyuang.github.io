---
title: "Formal Limitations on the Measurement of Mutual Information"
excerpt: "Any distribution-free, high-confidence MI lower bound from N samples is capped at ~2 ln N + 5 — meaning InfoNCE-based SSL (CLIP, SimCLR) is provably not measuring MI in any regime that matters."
categories:
  - Paper
  - Multimodal-Alignment
tags:
  - Mutual-Information
  - InfoNCE
  - CPC
  - MINE
  - DV-Bound
  - Self-Supervised
  - Contrastive-Learning
  - Theory
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-14
last_modified_at: 2026-05-14
permalink: /paper/mi-formal-limits-mutual-information-bounds/
---

## TL;DR
- Any distribution-free, high-confidence lower bound on mutual information computed from `N` iid samples is provably ≤ **2 ln N + 5** with probability ≥ 0.96 under an adversarial distribution (Theorem 1.1). No estimator — MINE, NWJ, DV, CPC/InfoNCE — escapes this ceiling without distributional assumptions.
- The proof is one adversarial construction reused twice: mix the target with a small ln-N-entropy "spoiler" at rate `1/N`, invoke the outlier-risk / birthday-paradox argument so the spoiler is statistically invisible at sample size `N`, then turn the high-confidence premise on its head to cap the bound at `ln N`. Tight, parsimonious, decisive.
- For InfoNCE-based self-supervised learning (CLIP, SimCLR, contrastive medical pretraining): the "we are maximizing mutual information" framing is **statistically vacuous in the large-MI regime** — CPC's `ln K` ceiling literally *is* the `ln N` wall this theorem proves unavoidable. The synthetic Gaussian experiment makes this concrete: with true `I(X,Y) = 106.29` nats and `ln N ≈ 4.85`, CPC saturates at exactly **4.85 nats**.

## Motivation

By 2018–2019 the orthodoxy in self-supervised representation learning was "maximize the mutual information between views/features." MINE (Belghazi et al. 2018), CPC / InfoNCE (Oord et al. 2018), Deep InfoMax (Hjelm et al. 2019), and the entire downstream SimCLR / CLIP family pitched their losses as variational lower bounds on `I(X, Y)` and reported the optimized bound value as a *measurement* of MI — with claimed numbers in the hundreds of bits on images, text, and audio.

McAllester and Stratos's paper is the formal autopsy of that framing. The argument is two-pronged. First, the Donsker–Varadhan bound that anchors MINE has a moment-generating-function structure dominated by rare events that no realistic sample sees — so the "tight bound" at the empirical optimum is an artifact of overfitting to the in-sample distribution. Second, this is not a quirk of DV: *any* distribution-free, high-confidence lower bound on MI from `N` samples runs into the same `ln N` ceiling, because the adversary can always hide an arbitrary amount of true MI in a low-probability spoiler that `N` samples cannot detect.

This matters directly for medical AI self-supervision. Every contrastive recipe applied to paired clinical modalities — CT–report (BiomedCLIP-style), H&E–spatial-transcriptomics (BLEEP, ST-Net), retinal-OCT–EHR — is justified as "we are maximizing `I(image, text)`." If `N` is the batch size and `K` the number of negatives, the bound caps at `ln K ≤ ln N` — typically ≤ ln 1024 ≈ 6.9 nats — while the true MI between a chest CT and its radiology report is plausibly hundreds of bits. The numerical value of the bound is uninformative; whatever is making these systems work, it is not "measuring mutual information."

## Core Innovation

The paper's single technical move is to convert a **statistical** premise (high-confidence lower bound on KL or entropy) into a **structural** ceiling via one adversarial distribution. The proof template runs three times, at increasing generality:

1. **DV-specific (Sec 2.1)**: directly bound the empirical DV estimator by `ln N` via the outlier risk lemma.
2. **General KL (Theorem 3.1)**: replace `q` with `q̃ = (1 − 1/N) q + (1/N) p`, so `D_KL(p ‖ q̃) ≤ ln N`; samples from `q^N` and `q̃^N` are indistinguishable on the "no-spoiler-drawn" event whose probability is `≥ (1 − 1/N)^N ≥ 1/4`.
3. **Entropy and hence MI (Theorems 4.1 and 1.1)**: a `2kN²`-atom uniform spoiler caps entropy at `ln(2kN²)`; the birthday-paradox bound makes spoiler atoms collision-free with high probability, so `N` samples still cannot tell the truncated distribution from the original. Since `I(X, Y) ≤ H(X)` and continuous MI is the sup over discrete binnings (Appendix B), the entropy ceiling transfers verbatim.

The trick is the same each time — *one* counter-distribution, *one* outlier-risk / birthday argument — but the conclusion is universal: distribution-free + high-confidence is incompatible with measuring large MI from a sample.

The second contribution is the constructive alternative: **Difference of Entropies (DoE)**,

$$
\hat I = \inf_{q_X} \hat H(p_X, q_X) - \inf_{q_{X|Y} } \hat H(p_{X|Y}, q_{X|Y}),
$$

a difference of two cross-entropy *upper* bounds. DoE is **neither** an upper nor a lower bound on `I(X, Y)` — that is the price of escaping the `ln N` wall — but each cross-entropy admits an honest `O(1/√N)` Chernoff interval once log-loss is bounded by some `F_max` (e.g. via character-level backoff). The paper is explicit that this is *engineering*: a sufficiently more competent unseen "superintelligent" model could always drive the conditional entropy lower, so DoE has no formal guarantee against being fooled in this direction.

## Claims & Evidence Analysis

| # | Claim | Evidence | Setting | Strength |
|---|---|---|---|---|
| C1 | Any distribution-free high-confidence MI lower bound from `N` samples is ≤ `2 ln N + 5` w.p. ≥ 0.96 under an adversarial distribution. | Theorem 1.1, via Theorem 4.1 (entropy) and the `I ≤ H` + binning argument in Appendix B. | Worst-case; assumes only the two premises of the bound. | ⭐⭐⭐ |
| C2 | The DV bound (and MINE's objective) is dominated by unseen large-deviation events; empirical DV ≤ `ln N` under the high-confidence requirement. | Sec 2.1 outlier risk lemma; Table 1 / Figure 2 confirm CPC saturates exactly at `ln N`. | Theoretical with synthetic confirmation. | ⭐⭐⭐ |
| C3 | MINE's polynomial-sample-complexity Theorem 3 is incorrect; Hoeffding is applied to `e^X` instead of `X` in their appendix eqs. (46)–(49). | Direct proof-reading; the error is local and unambiguous. | No counter-example needed. | ⭐⭐⭐ |
| C4 | CPC's bound (InfoNCE) structurally cannot exceed `ln K`. | Oord et al. (2018) bound; Table 1 row 3 shows CPC = 4.85 ≈ ln 128 in the high-MI regime. | Structural property of the bound, not an estimation artifact. | ⭐⭐⭐ |
| C5 | DoE accurately measures MI even when MI ≫ ln N. | Table 1: 104.18 vs true 106.29 nats; Table 2: 120 / 54 bits with near-zero shuffled controls. | One synthetic family + two NLP datasets; no seeds, no error bars, no vision or medical data. | ⭐⭐ |
| C6 | DoE outperforms existing lower-bound estimators. | Table 1, with hyperparameters **oracle-tuned per estimator** to minimize `|I − Î|`. | Best-case-per-estimator protocol favors DoE since instability is what hurts the baselines. | ⭐⭐ |
| C7 | DoE produces "realistic estimates of large MI in real-world datasets." | Table 2 numbers (120 / 54 bits) and shuffled controls. | True MI is unknown for these pairs; "realistic" reduces to internal consistency (shuffled ≈ 0), not ground truth. | ⭐ |

**Honest take.** C1–C4 are the kind of impossibility result that closes a debate. The proof is parsimonious — *one* adversarial construction, reused — and the tightness is not incidental: the adversarial distribution achieves exactly `D_KL ≤ ln N` and `H ≤ ln(2kN²)`, so the ceiling is the right one, not a loose worst case. C5–C7 are the soft underbelly. The empirical section uses a single Gaussian family and two NLP datasets without seed variance, error bars, or any external ground truth on Table 2 — the "DoE measures MI well" claim there is a plausibility argument (shuffled ≈ 0), not a verification, and the paper says so. C6's hyperparameter-oracle protocol is honest within its stated rules but means the table is "best-case per estimator," which is more punishing of unstable baselines than of DoE.

## Method & Architecture

![Adversarial distribution construction](/assets/images/paper/mi-formal-limits/page_005.png)
*Figure 1 (cropped from page 5): the single trick behind the ln N ceiling. Replace the long tail of `p` with a uniform spoiler distribution on `2kN²` atoms. The resulting `p̃` has `H(p̃) ≤ ln(2kN²)`, and the birthday-paradox bound makes the spoiler statistically invisible at sample size `N` — so any bound that holds for `p` is forced to hold for `p̃` too, capping it at `ln(2kN²)`.*

### The DV warm-up

The DV variational form

$$
\mathrm{DV}_f(p \,\|\, q) = \mathbb{E}_p[f(x)] - \ln \mathbb{E}_q[e^{f(x)}]
$$

has a log-moment-generating term that is dominated by the heaviest tail event in the support. Suppose an empirical estimate uses `f(x_i) = F_max` on `p`-samples and `f(x'_i) = 0` on `q`-samples — a structurally favorable best case. The **outlier risk lemma** (Lemma 2.2) says that if `Pr[Φ] ≤ 1/N` then with probability at least `1/4` *no sample hits `Φ`*, because `(1 − 1/N)^N ≥ 1/4` for `N ≥ 2`. So an unseen event of mass `1/N` and height `e^{F_max}` contributes `(1/N) · e^{F_max}` to the true expectation that the sample never observed. Honoring the high-confidence guarantee forces

$$
\hat{\mathrm{DV} } \le F_\text{max} - \ln\!\big(e^{F_\text{max} } / N\big) = \ln N.
$$

This is the entire mechanism. Every subsequent theorem is just the same outlier risk applied to a different functional.

### The general KL theorem

Define the spoiler-mixed counter-distribution

$$
\tilde q(x) = (1 - 1/N)\, q(x) + (1/N)\, p(x).
$$

Then `D_KL(p ‖ q̃) ≤ ln N` by direct calculation. Samples drawn from `q^N` and `q̃^N` are *jointly indistinguishable* on the "Pure" event — no draw of the `(1/N)`-coin came up `1` — and Pr["Pure"] ≥ `1/4`. The bound `B` is required to hold for *every* distribution, so apply it with the population being `(p, q̃)` and condition on "Pure":

$$
B(p, S, \delta) \le D_\text{KL}(p \,\|\, \tilde q) \le \ln N \quad \text{w.p.} \ge 1 - 4\delta.
$$

The trick is **one** counter-distribution doing all the work.

### Generalization to entropy and MI

For entropy, sort `p(x_1) ≥ p(x_2) ≥ …` and build `p̃` by keeping the top `kN²` atoms and replacing the remaining mass with a uniform spoiler on a fresh set of `2kN² − kN²` atoms. Then `H(p̃) ≤ ln(2kN²)`. A birthday-paradox bound (using `1 − z ≥ e^{−1.01z}` for small `z`) shows the probability of no collision among the spoiler atoms is `≥ 1 − 0.505 / k`, making `p` and `p̃` statistically interchangeable on the "Pure" event. Hence

$$
B(T(S), \delta) \le \ln(2kN^2) \quad \text{w.p.} \ge 1 - \delta - 1.01/k.
$$

Plugging into `I(X, Y) ≤ H(X)` and taking the sup over discrete binnings (continuous case, Appendix B) yields Theorem 1.1: `B ≤ 2 ln N + 5` w.p. ≥ 0.96.

### DoE — the constructive escape hatch

Use `I(X, Y) = H(X) − H(X | Y)` and estimate each entropy by a cross-entropy **upper** bound:

$$
\hat I = \inf_{q_X} \hat H(p_X, q_X) - \inf_{q_{X|Y} } \hat H(p_{X|Y}, q_{X|Y}).
$$

A difference of upper bounds is neither an upper nor a lower bound on the truth — but each cross-entropy individually has `O(1/√N)` Chernoff intervals once log-loss is capped by `F_max` (Theorem 5.1). This is engineering: you trade the formal lower-bound guarantee for an honest pointwise estimate that can actually exceed `ln N`.

## Experimental Results

### Synthetic Gaussian — true MI vs estimator value (N = 128, ln N ≈ 4.85, nats)

| True `I(X,Y)` | DV | MINE | NWJ | NWJ (JS) | CPC | CPC+NWJ | **DoE (Gaussian)** | **DoE (Logistic)** |
|---|---|---|---|---|---|---|---|---|
| 4.13 | 2.72 | 2.57 | 1.99 | 1.50 | 2.73 | 2.77 | **4.19** | **4.13** |
| 18.41 | 10.27 | 9.38 | 9.25 | 5.55 | 4.82 | 8.18 | **18.38** | **18.42** |
| 106.29 | 61.96 | 34.56 | 50.46 | 13.41 | 4.85 | 10.45 | **104.18** | **104.16** |

Note CPC saturating at exactly `ln N ≈ 4.85` in the high-MI row — the structural prediction of the theorem made visible in a single number. DV, MINE, and NWJ produce values that *exceed* `ln N` in this table, but the authors flag this as misleading: those estimators are highly unstable with frequent numerical overflow, and the reported values are best-case hyperparameter-oracle final estimates rather than confidence-honoring outputs.

### Real-data DoE estimates (bits)

| Pairing | `Î(X, Y)` |
|---|---|
| **Related article pairs (Who-Did-What)** | **120.34** |
| Shuffled article pairs | −2.38 |
| **Translation pairs (IWSLT EN-DE)** | **54.72** |
| Shuffled translation pairs | −2.64 |

Both shuffled controls are near zero, consistent with no MI. The positive values are far above the `ln N` ceiling that limits any lower-bound estimator on these sample sizes — but they are not verifiable against ground truth, so "120 bits between related articles" is internal consistency, not measurement.

## Limitations

Acknowledged by the authors:
- DoE provides no formal upper or lower bound on MI; its accuracy is contingent on the cross-entropy models being close to optimal in *both* the marginal and the conditional. Any model gap on either side biases the estimate, and the direction of the bias is unconstrained.
- Lower bounds on entropy would amount to proving non-existence of a better predictive model ("superintelligence"); the authors argue such proofs are infeasible from data alone, which is also why DoE's escape from the `ln N` ceiling cannot itself be turned into a guarantee.
- The continuous-to-discrete reduction is proved for `ℝ` with Riemann-integrable densities; the fully general measure-theoretic case is conjectured (Appendix B).
- The result is distribution-free; distribution-specific estimators (small-support assumptions à la Valiant–Valiant, parametric families) can in principle do better and are not ruled out.

Missing from the paper:
- **No empirical comparison on images or any high-dimensional non-text data**, even though the InfoMax / SimCLR / CPC line of work is overwhelmingly applied to vision. The implications for vision SSL are inferred from the theorem, not demonstrated.
- **No variance reporting** in Tables 1 and 2 — no seeds, no error bars, no significance tests. The "DoE outperforms" claim is on hyperparameter-oracle single runs.
- The paper does **not engage with the alignment + uniformity reframing** (Wang & Isola 2020, contemporaneous), which is the constructive answer to "if InfoNCE is not measuring MI, then what is it doing?" The deeper answer the field converged on is that InfoNCE is a **geometric objective on the hypersphere**, decomposing into alignment of positive pairs and uniformity of the marginal — not an information-theoretic one. McAllester and Stratos close the door on the MI interpretation; Wang and Isola open the geometric one.
- The ln N ceiling is for the **numerical value of the bound**, not for the gradient direction. The paper does not address whether maximizing a vacuous-as-measurement lower bound still produces useful representations — which is the question the field actually cared about post-2020 and the paper leaves open.

## Why It Matters for Medical AI

Every contrastive-pretraining recipe applied to paired clinical modalities — BiomedCLIP and ChatCAD on CT/X-ray–report pairs, BLEEP and ST-Net on histology–spatial-transcriptomics, retinal SSL on OCT–fundus — is built on an InfoNCE loss and justified, in the literature, as "maximizing mutual information between modalities." This paper says: that justification is statistically vacuous. With a typical batch size `K = 1024`, the InfoNCE bound saturates at `ln K ≈ 6.9` nats regardless of dataset; the true MI between a chest CT and its radiology report is plausibly hundreds of bits. The bound is not measuring MI, it is hitting the ceiling.

What the loss is then doing is *not* what the marketing says. The paper's own constructive answer (DoE = difference of cross-entropy upper bounds) abandons the bound guarantee in exchange for an estimator that can numerically exceed `ln N` — useful for diagnostics, but not a justification of the contrastive loss. The deeper answer the field has converged on is Wang & Isola's **alignment + uniformity**: InfoNCE optimizes a geometric objective on the unit hypersphere — pull positive pairs together, push the marginal toward the uniform distribution on `S^{d-1}`. This reframing explains effects MI cannot (dimensional collapse, the CLIP modality gap, why temperature `τ` controls uniformity), and it transfers cleanly to clinical pretraining: the relevant question is no longer "how many bits of information does my model share between CT and report" but "how aligned are paired CT–report embeddings, and how uniformly does my CT marginal cover the sphere."

The pragmatic implication for medical-AI practitioners: stop reporting InfoNCE-derived MI estimates as if they were measurements. Report alignment and uniformity. Report retrieval metrics. Report downstream linear-probe accuracy. The MI number on its own is not load-bearing — the theorem in this paper says it cannot be.

## References

- Paper: McAllester, D. and Stratos, K. *Formal Limitations on the Measurement of Mutual Information.* AISTATS 2020. arXiv: [1811.04251](https://arxiv.org/abs/1811.04251).
- MINE (the estimator critiqued in Sec 2.2): Belghazi et al., *Mutual Information Neural Estimation*, ICML 2018.
- CPC / InfoNCE (the `ln K`-capped baseline in Table 1): van den Oord, Li, Vinyals, *Representation Learning with Contrastive Predictive Coding*, 2018.
- DV bound: Donsker and Varadhan, *Asymptotic Evaluation of Certain Markov Process Expectations for Large Time*, 1983.
- Empirical falsification of InfoMax: Tschannen et al., *On Mutual Information Maximization for Representation Learning*, ICLR 2020.
- The constructive geometric sequel: Wang and Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020.
- Synthetic Gaussian benchmark protocol: Poole et al., *On Variational Bounds of Mutual Information*, ICML 2019.
- Datasets: Onishi et al., *Who Did What* (LDC Gigaword), EMNLP 2016; IWSLT 2014 EN–DE.
