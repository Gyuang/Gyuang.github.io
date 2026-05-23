---
title: "V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization"
excerpt: "Adds a CFG-style vision-conditional log-ratio to DPO's implicit reward; AMBER 88.4 / MMHal 2.36 with 5K synthetic pairs vs DPO 87.9 / 2.12 — but Δ vs DPO is 0.1–0.5 with no variance and no mDPO baseline."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/v-dpo/
tags:
  - V-DPO
  - DPO
  - LVLM Hallucination
  - Classifier-Free Guidance
  - Preference Optimization
  - Vision-Language Models
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- V-DPO reframes LVLM hallucination as **over-reliance on the LLM's language prior** and injects a **Classifier-Free-Guidance (CFG)** style term `(γ−1)·log[π(y|v,x)/π(y|x)]` into DPO's implicit reward so the policy is graded on *image-specificity*, not just response preference.
- Pairs the CFG-style objective with a **diffusion-inpainting preference pipeline** (5K pairs from COCO/VG/VCR, one object replaced via DDPM, CLIPScore-ratio filtered ≥1.5) that produces both response-contrast and image-contrast `(v_w, v_l, x, y)` pairs.
- Headline numbers on LLaVA-v1.5-7B: **AMBER 88.4 / MMHal 2.36 / POPE F1 86.92 / HallusionBench qAcc 22.20** vs **DPO 87.9 / 2.12 / 86.42 / 21.97**. The Δ-vs-DPO is 0.1–0.5 across benchmarks with **no variance reported, no mDPO comparison, and no medical-domain evaluation**.

## Motivation

LVLMs hallucinate because the LLM backbone supplies a fluent text prior that overwhelms the visual signal. The paper's Fig. 1b shows this directly: on hallucination pairs, the **textual-only distribution** `p(y|x)` separates accurate from hallucinatory samples nearly as well as the **vision-conditioned** distribution `p(y|v,x)` — meaning vanilla DPO can move log-likelihoods toward preferred responses *without ever learning to use the image*.

Decoding-time fixes (OPERA, HALC, Volcano) are expensive and architecture-specific. The paper aims for a **training-time, model-agnostic** alignment objective that explicitly penalizes vision-independent likelihood. There is no medical-domain evaluation here — all benchmarks are general-VQA (POPE, AMBER, HallusionBench, MMHal-Bench) — but the idea is in principle transferable to medical LVLMs.

![Hallucination examples on unconventional images](/assets/images/paper/v-dpo/fig_p001_02.png)
*Figure 1a: Hallucination examples on unconventional images (kids/worms, lobsters, region descriptions) — the failure modes V-DPO targets.*

![Distribution gap between textual-only and vision-conditioned likelihoods](/assets/images/paper/v-dpo/fig_p001_03.png)
*Figure 1b: Log-likelihood gap between accurate and hallucinatory samples. The textual-only branch `p(y|x)` already separates the two classes — evidence that vanilla DPO can optimize the response branch without engaging the vision branch.*

## Core Innovation

The technical move is a **CFG-form vision-conditional regularizer** inside the DPO objective. The reward optimization becomes

$$\max_\pi \; \mathbb{E}\left[\, r(v,x,y) - \beta\,\mathrm{KL}\!\big(\pi \,\|\, \pi_{\text{ref} } \mid v,x\big) + \alpha\,\mathrm{KL}\!\big(\pi(y|v,x)\,\|\,\pi(y|x)\big)\,\right],$$

whose optimal-policy solution introduces a multiplicative correction `φ_θ(v,x,y) = [π_θ(y|v,x)/π_θ(y|x)]^(γ−1)` with `γ = 1 − α/β`. Setting `α > 0` gives `γ < 1` — the **opposite sign** of inference-time CFG, which uses `γ > 1` to amplify the visual ratio at decode. The V-DPO insight is that during gradient updates you *strengthen* the visual-specificity ratio rather than amplify it post-hoc.

The implicit reward used inside the Bradley–Terry sigmoid becomes

$$f_\theta(v,x,y) = \log\!\left[\frac{\pi_\theta(y|v,x)\cdot \phi_\theta(v,x,y)}{\pi_{\text{ref} }(y|v,x)}\right],$$

with the gradient stopped on `φ_θ` so the textual-only branch acts as a stable reference target.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | V-DPO outperforms vanilla DPO on hallucination benchmarks. | POPE F1 86.92/87.22 vs 86.42/87.12; AMBER 88.4/88.0 vs 87.9/87.6; MMHal 2.36/2.16 vs 2.12/2.08; HBench qAcc 22.20 vs 21.97. | ⭐⭐ Direction is consistent across two preference data sources and four benchmarks, but Δ is **0.10–0.50** in F1/Score with **no seeds / no variance reported**. "Significant" is editorial. |
| C2 | V-DPO mitigates over-reliance on language priors. | Fig. 5: textual-only distribution shift 6.35 (V-DPO) vs 7.58 (DPO); image-conditional shift 19.17 vs 18.54. | ⭐⭐ Direction supports the claim but only one model, one slice; effect size ~1.2 nats on a metric the paper defines itself. |
| C3 | V-DPO is especially effective with image-contrast preference pairs. | Image-contrast shift 11.01 vs 9.37; HBench qAcc +9.01 over SFT on synthetic vs +4.17 on RLHF-V. | ⭐⭐⭐ Largest qualitative gap; rationale (image-contrast pairs hold `x` fixed and only vary `v`, exactly the signal the CFG term reads) is internally consistent. |
| C4 | 5K V-DPO pairs beat 16K HA-DPO pairs. | AMBER 88.4 vs 85.7 (+2.7). | ⭐⭐ HA-DPO targets *style consistency*, not visual grounding — the comparison is somewhat unfair to HA-DPO's design intent. |
| C5 | V-DPO does not sacrifice general capability. | MMBench 65.12 vs SFT 65.21 (−0.09); LLaVA-1.6 66.15 vs 66.41 (−0.26). | ⭐⭐ Small drop, comparable to vanilla DPO's drop; no significance test. |
| C6 | Method generalizes across backbones. | App. D Table 7: LLaVA-1.6 AMBER 88.0 → 88.9. | ⭐ One additional backbone, one benchmark — not really "general." |
| C7 | Visual-guidance strength γ is a critical hyperparameter. | Fig. 4 γ-sweep is non-monotonic. | ⭐⭐ Clean ablation, but optimum **differs across data** (γ=0.75 synthetic vs γ=0.0 RLHF-V); user must tune. |

**Honest read.** The CFG-style derivation in Appendix A is correct, and the sign flip (`γ < 1` at training vs `γ > 1` at decode) is a real and clean insight. But there are four issues worth naming out loud:

1. **The added term is not fully identifiable from standard DPO.** When `π_ref` already conditions on `v`, vanilla DPO's gradient already moves `log π(y|v,x)`. V-DPO differs from DPO by `(γ−1)·[log π(y|v,x) − log π(y|x)]` — a regularizer that **penalizes the textual-only branch**, but the paper does not isolate this from a generic "down-weight language-prior tokens" baseline.
2. **No mDPO comparison or citation.** [mDPO (Wang et al., NeurIPS 2024, arXiv 2406.11839)](https://arxiv.org/abs/2406.11839) independently proposed adding image-conditional preference pairs plus a chosen-response anchor to DPO for LVLM hallucination, and predates V-DPO's arXiv post by ~5 months with essentially the same image-contrast idea. V-DPO does not cite or compare to mDPO in the main results. The image-contrast preference pair idea is therefore **not novel in late 2024** — V-DPO's distinct contribution narrows to the CFG-form reward term plus the diffusion-inpainting data pipeline.
3. **Δ-vs-DPO is small and reported without variance.** HallusionBench qAcc "improves by 9.01" is the gap vs SFT, not vs DPO; the honest Δ over DPO is **+0.23**, a single-evaluation noise-level number. AMBER Δ is **+0.5**, POPE F1 Δ is **0.10–0.50**. With 5K pairs and stochastic preference learning, one seed is not enough to call any of these wins significant.
4. **No medical-domain evaluation**, despite the obvious applicability of "stop hallucinating about what's in the image" to radiology and pathology LVLMs.

## Method & Architecture

![Synthetic data pipeline — left half](/assets/images/paper/v-dpo/fig_p004_03.png)
*Figure 2 (left): Synthetic preference-data pipeline. The LVLM grounds objects with bbox coordinates, an LLM (ChatGPT) proposes an unconventional replacement (e.g. cake → pile of rocks), and Stable-Diffusion / DDPM inpaints the bbox to produce `v_l`. CLIPScore filters retain only pairs with `CLIPScore_w / CLIPScore_l ≥ 1.5`.*

![Preference data formulation — right half](/assets/images/paper/v-dpo/fig_p004_04.png)
*Figure 2 (right): The pipeline yields both response-contrast `(v, x, y_w, y_l)` and image-contrast `(v_w, v_l, x, y_w)` preference pairs that feed V-DPO.*

The pipeline runs as follows.

1. **Start from an SFT LVLM** (LLaVA-v1.5-7B) and the standard DPO loss on response-contrast pairs (Eq. 3):

   $$\mathcal{L}_{\text{DPO} } = -\mathbb{E}\,\log\sigma\!\left(\beta\!\left[\log\frac{\pi_\theta(y_w|v,x)}{\pi_{\text{ref} }(y_w|v,x)} - \log\frac{\pi_\theta(y_l|v,x)}{\pi_{\text{ref} }(y_l|v,x)}\right]\right).$$

2. **Image-contrast extension** (Eq. 4): construct `D_v = {(v_w, v_l, x, y)}` where the same response is preferred under `v_w` and dispreferred under `v_l`. Same Bradley–Terry form, but the log-ratio is over images.

3. **Vision-guided reward** (Eq. 7, Eq. 8): add the `α·KL(π(y|v,x) ‖ π(y|x))` term, solve, and obtain the implicit reward

   $$f_\theta(v,x,y) = \log\!\left[\frac{\pi_\theta(y|v,x)\cdot [\pi_\theta(y|v,x)/\pi_\theta(y|x)]^{\gamma-1} }{\pi_{\text{ref} }(y|v,x)}\right],$$

   where `γ = 1 − α/β`. Gradient is stopped on the `φ_θ = [π_θ(y|v,x)/π_θ(y|x)]^(γ−1)` factor.

4. **Textual-only distribution** (Eq. 10): implemented by zeroing the visual tokens — `π_θ(·|x) := π_θ(·|0, x)`. The ablation shows that using a **frozen "static-lm" reference** (initial SFT model for `π(·|x)`) is empirically better — it keeps the language-prior reference from drifting during training.

5. **Normalization variant** (Eq. 11): the unnormalized form is renormalized in logit space via `softmax(h_θ(v,x) + (γ−1)(h_θ(v,x) − h_θ(0,x)))`. This **inflates guidance as training progresses** — helpful for generative CHAIR, harmful for discriminative F1.

6. **Training config (App. B).** LR 1e-6, batch 64, 4 epochs, β=0.1, γ=0.75 on synthetic data, γ=0.0 on RLHF-V data, 4×A100 40GB.

## Experimental Results

### Main hallucination benchmarks

| Benchmark | Metric | SFT | DPO (synth) | **V-DPO (synth)** | DPO (RLHF-V) | **V-DPO (RLHF-V)** | HA-DPO (16K) |
|---|---|---|---|---|---|---|---|
| POPE | F1 ↑ | 85.98 | 86.42 | **86.92** | 87.12 | **87.22** | 86.87 |
| POPE | Yes-Ratio % | 54.20 | 44.22 | 47.43 | 47.88 | 48.66 | 51.03 |
| AMBER | Score ↑ | 83.5 | 87.9 | **88.4** | 87.6 | **88.0** | 85.7 |
| AMBER | CHAIR ↓ | 7.8 | 7.3 | **6.6** | 5.7 | **5.6** | 6.7 |
| AMBER | Hal ↓ | 36.4 | 33.6 | **30.8** | 27.3 | **27.3** | 30.9 |
| AMBER | F1 ↑ | 74.7 | 83.1 | **83.5** | 80.9 | **81.6** | 78.1 |
| HallusionBench | qAcc ↑ | 13.19 | 21.97 | **22.20** | 16.70 | **17.36** | — |
| HallusionBench | aAcc ↑ | 48.16 | 55.52 | 55.31 | 51.31 | **51.63** | — |
| MMHal-Bench | Score ↑ | 1.97 | 2.12 | **2.36** | 2.08 | **2.16** | — |
| MMHal-Bench | Hal ↓ | 0.62 | 0.59 | **0.53** | 0.60 | **0.56** | — |
| MMBench overall | Acc ↑ | 65.21 | 65.03 | 65.12 | 64.78 | 64.95 | — |

V-DPO wins on essentially every hallucination cell — but the Δ over vanilla DPO is in the 0.1–0.5 range on F1/Score and 0.04 on MMHal Hal-rate, with **no seeds or confidence intervals reported anywhere**. MMBench is essentially unchanged (Δ ≈ 0.09 vs SFT, comparable to DPO's drop) — the paper's "no capability loss" claim.

![MMHal-Bench per-question-type breakdown](/assets/images/paper/v-dpo/fig_p008_01.png)
*Figure 3: MMHal-Bench meso-analysis. V-DPO gains are concentrated on the comparison / environment / adversarial question types — exactly the categories where the answer depends most on the image rather than world knowledge.*

### γ ablation

![Gamma sweep on AMBER](/assets/images/paper/v-dpo/fig_p008_03.png)
*Figure 4: AMBER CHAIR and F1 versus the visual-guidance weight γ. The optimum is non-monotonic and dataset-dependent — γ ≈ 0.75 for the synthetic preference pairs, γ = 0.0 for RLHF-V. Pushing γ to ±1 degrades both metrics.*

The γ-sweep is the cleanest piece of the experimental section but also the most awkward for the headline claim: when the optimal γ on RLHF-V data is **exactly 0**, the framework collapses to a version where the CFG term contributes nothing operationally, and V-DPO reduces to a peculiar limit case. The authors do not engage with whether the CFG framing adds anything in that regime versus a simple `KL(π‖π_text-only)` regularizer.

### Distribution-shift analysis

![Distribution-shift histograms](/assets/images/paper/v-dpo/fig_p008_04.png)
*Figure 5 (top): Shift in `log[p(y|v,x)/p(y|x)]` after training. V-DPO produces a larger shift (11.01 vs 9.37 on image-contrast pairs) — direct evidence the visual branch is moving more than the textual branch.*

![Distribution-shift histograms — second panel](/assets/images/paper/v-dpo/fig_p008_05.png)
*Figure 5 (bottom): Shift on the textual-only `log p(y|x)` is **smaller** under V-DPO (6.35 vs 7.58), supporting the "less language-prior movement" claim.*

This figure is the qualitative core of the paper. It is also the only place where the language-prior claim is supported with anything beyond accuracy deltas — and the effect size is around 1.2 nats on a metric of the paper's own construction.

### LLaVA-1.6 replication (App. D)

The pattern replicates on LLaVA-1.6 at smaller magnitude: AMBER 88.0 → 88.9 with V-DPO. One additional backbone, one additional benchmark — not enough to call the method "general."

### Ablations

- **Static-lm** (frozen SFT reference for `π(·|x)`): CHAIR −0.3 / −0.4, F1 +0.2 / +0.8. Suggests the reference choice for the textual-only branch matters more than the paper acknowledges.
- **Normalization variant** (Eq. 11): lowers CHAIR (better generation) but lowers discriminative F1 (−0.4 / −1.2). A real trade-off, not a free win.

## Limitations

**Acknowledged by authors.**

- No exploration of domains where the language prior *should* dominate (e.g., fluency-driven preferences).
- Noise and bias inherited from the automated LLM + Stable-Diffusion pipeline.
- Slight MMBench degradation.

**Under-addressed.**

- **No mDPO baseline.** The most damaging omission. mDPO (Wang et al., NeurIPS 2024) predates V-DPO's arXiv post by ~5 months and uses the same image-contrast preference idea. Without a head-to-head, the image-contrast contribution is unverifiable.
- **No variance / seeds.** Every reported gain is single-run. With 5K pairs and stochastic preference learning, Δ = 0.10 F1 on POPE is well inside noise.
- **GPT-4 evaluator drift.** MMHal-Bench scores depend on GPT-4-0613 from June 2024 — reproducibility risk over time.
- **γ tuning is dataset-specific** (0.75 vs 0.0), undermining the framing as a clean general method.
- **Single backbone family.** Only LLaVA-1.5 / LLaVA-1.6 evaluated. No LLaVA-Next, InternVL, Qwen-VL, Idefics, or any modern open LVLM.
- **No qualitative failure-mode analysis.** Does V-DPO ever create new hallucinations by overweighting irrelevant visual tokens? Not investigated.
- **Identifiability.** The `(γ−1)·log[π(y|v,x)/π(y|x)]` term is not isolated from generic "down-weight language-prior tokens" regularizers — no such baseline is tested.
- **No medical evaluation.** Despite the obvious applicability to medical LVLM hallucination (where false confabulated findings have real clinical cost), all evaluation is general-domain VQA.

## Why It Matters for Medical AI

V-DPO does not evaluate on any medical benchmark, but the failure mode it targets — an LVLM confabulating content that is supported by the language prior but not by the image — is the dominant hallucination mode in radiology and pathology vision-language models. Any medical LVLM team that already has a preference dataset (e.g. radiologist-corrected report pairs) and is using DPO can plug in the CFG-form term with minimal infrastructure change; the diffusion-inpainting data pipeline could also be reapplied with medical-image-aware editors (organ swaps, lesion insertions) to build image-contrast pairs.

The honest caveat is that without medical-domain evaluation, none of the headline numbers transfer directly. The Δ-vs-DPO of 0.1–0.5 on general benchmarks may shrink further on noisier medical data, especially when the underlying SFT model has weaker visual grounding. A medical-AI reader should treat V-DPO as a **plausible mechanism** rather than a validated method until somebody runs it on POPE-style probes against radiology reports or pathology VQA.

## References

- [Paper (ACL Findings 2025 / arXiv 2411.02712)](https://arxiv.org/abs/2411.02712)
- [Code: github.com/YuxiXie/V-DPO](https://github.com/YuxiXie/V-DPO)
- Related: [mDPO (Wang et al., NeurIPS 2024)](https://arxiv.org/abs/2406.11839) — concurrent image-contrast DPO, not cited by V-DPO
- Related: [HA-DPO (Zhao et al., 2023)](https://arxiv.org/abs/2311.16839) — style-consistent hallucination pairs baseline
- Related: [POVID](https://arxiv.org/abs/2402.11411), [Silkie](https://arxiv.org/abs/2312.10665), [RLHF-V (Yu et al., 2023)](https://arxiv.org/abs/2312.00849)
- Benchmarks: [POPE](https://arxiv.org/abs/2305.10355), [AMBER](https://arxiv.org/abs/2311.07397), [HallusionBench](https://arxiv.org/abs/2310.14566), [MMHal-Bench](https://arxiv.org/abs/2309.14525)
