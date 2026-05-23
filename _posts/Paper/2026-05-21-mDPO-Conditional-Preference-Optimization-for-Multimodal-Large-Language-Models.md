---
title: "mDPO: Conditional Preference Optimization for Multimodal Large Language Models"
excerpt: "Standard DPO on MLLMs collapses to a language-only signal (DPO No-Image ≈ DPO on MMHalBench); mDPO adds an image-axis Bradley-Terry term plus a chosen-reward anchor and lifts Bunny-3B from 2.28 → 2.96 on MMHalBench."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/mdpo/
tags:
  - mDPO
  - DPO
  - Preference-Optimization
  - Multimodal-LLM
  - Hallucination
  - CoPO
  - AncPO
  - Bunny
  - LLaVA
  - MMHalBench
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- Standard multimodal DPO **silently collapses to a language-only objective**: training DPO with images *removed* performs about the same as DPO trained with images on MMHalBench. The image side of the preference pair is being ignored.
- mDPO is a drop-in fix with two pieces. **CoPO** swaps the contrast axis from response to image -- hold (q, y_w) fixed, contrast a chosen image m_w against a hard-negative crop m_l -- so the only way to widen the Bradley-Terry margin is to actually use image features. **AncPO** adds an absolute floor r(m_w, q, y_w) >= 0 to stop the chosen-response log-prob from drifting down.
- Headline on Bunny-v1.0-3B: MMHalBench score **2.28 -> 2.96**, HalRate **0.56 -> 0.42**, Object HalBench CHAIRs **44.3 -> 27.0**, CHAIRi **7.6 -> 4.6** vs. standard DPO; human win-or-tie 89%. The cleanest evidence is the CoPO ablation -- removing it drops MMHal 2.96 -> 2.36 (vs 2.50 for removing AncPO), so **CoPO carries the gains**.

## Motivation

Multimodal DPO has so far been a "swap in image-text preference pairs and hope" affair (Silkie, POVID, HA-DPO), with mixed wins and occasionally *worsened* hallucination. The authors' diagnostic in Figure 1 is unusually sharp: in a controlled run on Bunny-v1.0-3B with Silkie 10K preferences, **DPO trained with images removed scores essentially the same as DPO trained with images** on MMHalBench. The image was contributing nothing to the optimization signal. This is the *unconditional preference* failure mode -- DPO's objective is r(m, q, y_w) - r(m, q, y_l), and because preference pairs almost always share (m, q), the model can satisfy the margin by sorting responses on language priors alone.

For high-stakes vision-language applications -- radiology VQA, pathology captioning, anything with a stereotyped report template -- this failure is exactly the worst-case: the model learns to sound right rather than to look at the pixels.

![mDPO motivating diagnostic: DPO (No Image) is on par with DPO on MMHalBench](/assets/images/paper/mdpo/fig_p001_01.png)
*Figure 1: The diagnostic that motivates mDPO. On Bunny-v1.0-3B + Silkie-10K, standard DPO (with images) performs essentially the same as DPO (No Image) on MMHalBench -- the image condition is being ignored. mDPO restores image-conditional learning and lifts the score from 2.28 to 2.96.*

## Core Innovation

mDPO replaces the single DPO term with an unweighted **sum of three losses**: standard multimodal DPO (kept as backbone), CoPO (image-axis preference), and AncPO (absolute reward floor on the chosen sample).

- **CoPO (Conditional Preference Optimization)** is the load-bearing piece. It is structurally identical to DPO -- same Bradley-Terry form, same beta -- but contrasts **images** instead of responses with (q, y_w) held fixed. Because q and y_w are identical on both sides of the margin, the only lever the model has to widen it is image-grounded reward. This mechanically eliminates the language-only shortcut.
- **Hard-negative image via random crop (0-20%)** of m_w. Visually similar enough to be a hard negative, but missing enough information that y_w is no longer optimal. Crucially this is **training-free, deterministic, no generative model in the loop** -- ablations show this beats both random images (too easy) and MoCo-v2 augmentation (too similar).
- **AncPO (Anchored Preference Optimization)** is an *orthogonal* fix to a known DPO pathology -- DPO only enforces a relative margin, so the chosen log-prob can drift down so long as the rejected drifts down faster. AncPO adds an absolute floor r(m_w, q, y_w) > delta with delta = 0.
- **No coefficient sweep.** The three terms are summed 1:1:1 with no per-term weight tuning, which is a small audit gap (more in the Limitations section).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Standard multimodal DPO suffers from "unconditional preference" -- the image is largely ignored. | DPO (No Image) ~= DPO on MMHalBench (Fig. 1); qualitative image-blind answers (Fig. 3). | MMHalBench, Silkie 10K, Bunny-3B | ⭐⭐ -- diagnostic is sharp but single base model + single benchmark; no log-prob-on-image-features probe to mechanistically confirm. |
| C2 | CoPO + AncPO fix this, with CoPO being the dominant term. | Ablation Tab. 2: -CoPO drops MMHal 2.96 -> 2.36 (Δ -0.60); -AncPO drops to 2.50 (Δ -0.46); both removed = DPO baseline 2.28. | Bunny-3B; MMHal + ObjHal | ⭐⭐⭐ -- clean ablation, both metrics move consistently, CoPO is unambiguously the workhorse. |
| C3 | mDPO reduces hallucination across model scales. | Bunny-3B and LLaVA-v1.5-7B both improve on all three benchmarks (Tab. 1). | MMHal, ObjHal, AMBER | ⭐⭐⭐ -- 2 models x 3 benchmarks, consistent direction; LLaVA-7B gains are smaller (ΔMMHal +0.25 vs Bunny's +0.68). |
| C4 | mDPO is data-scale-effective; DPO is not. | Fig. 5 -- mDPO scales monotonically with preference data, DPO flatlines / regresses. | MMHal, Bunny-3B | ⭐⭐ -- only one model + one benchmark; no error bars. |
| C5 | Hard-negative cropping is the right image-perturbation strategy. | Tab. 3 -- Crop 0-20% (2.96) > Crop 20-50% (2.92) > MoCo-v2 (2.82) > Random image (2.81) on MMHal score. | MMHal + ObjHal, Bunny-3B | ⭐⭐ -- clear ordering, but no semantic-mask / object-erase baseline; 0-20% vs 20-50% delta (~0.04) is below the noise floor of a single-run study. |
| C6 | Anchoring only on y_w is sufficient. | Tab. 4 -- adding anchors on y_l or m_l yields no gain (and can hurt CHAIR). | MMHal + ObjHal, Bunny-3B | ⭐⭐ -- well-isolated, single model. |
| C7 | 3B + mDPO is competitive with 7B + 8x data. | Tab. 1 -- Bunny-3B + mDPO (2.96 / 0.42) ~= Qwen-VL-Chat + Silkie-80K (3.01 / 0.41) on MMHal. | MMHal | ⭐ -- mixes base model, data scale, and objective; mDPO's contribution is not isolated. |

**Honest read.** The diagnostic in C1 is the paper's intellectual contribution -- the unconditional-preference failure mode is real and the fix is principled -- but it is shown on a single model + single benchmark, so it is more "suggestive" than "established." The strongest piece of evidence in the paper is **C2's ablation**: removing CoPO halves the gains while removing AncPO leaves most of them intact, which cleanly attributes the improvement to the image-axis preference term rather than to anchor magic or training-config artifacts. **CoPO is the dominant mechanism; AncPO is a useful but secondary anchor.** Weaknesses worth naming: (i) no variance reporting -- every cell is a single run, so the 0.04-point differences in Table 3 are not statistically meaningful; (ii) only two backbones, both CLIP/SigLIP + decoder-only LM with linear projector, so generalization to LLaVA-NeXT, Qwen2-VL, or Q-Former-style encoders is untested; (iii) GPT-4 judge dependence on MMHalBench; (iv) no medical / document / chart eval, which is the natural follow-up because the language-prior-dominates failure should be *worse* in templated-report domains; (v) the 1:1:1 loss weighting is fixed with no sweep, so the CoPO-vs-AncPO attribution may shift under tuning.

**vs. V-DPO (Xie et al., EMNLP 2024 -- concurrent).** Same diagnosis, different perturbation. Both papers identify the image-collapse failure of DPO on MLLMs and both inject an image-conditioned preference signal. The differences are concrete: (a) mDPO's rejected image is a **training-free 0-20% random crop**, V-DPO uses **diffusion-based or learned visual perturbations**, which are more semantic but introduce a generative dependency; (b) mDPO keeps the image-preference term in the *same* DPO/Bradley-Terry form (just with the image axis swapped instead of the response axis), making it a **drop-in addition** to any DPO pipeline -- V-DPO frames the image as an explicit "visual preference variable" in a generalized DPO formulation; (c) mDPO additionally adds the **orthogonal AncPO anchor**, V-DPO does not. **Neither paper cites the other** in its main body (concurrent submission, same venue cycle). mDPO is the more minimalist and reproducible setup; V-DPO's semantic negatives may scale better when crops are too easy (scene-level reasoning).

## Method & Architecture

![mDPO method overview: image-axis preference loss plus chosen-reward anchor](/assets/images/paper/mdpo/fig_p002_01.png)
*Figure 2: mDPO overview. Top: the gap between DPO's intended joint (image, question)-conditional reward and what it actually learns in practice (language-only). Bottom: mDPO adds an image-preference term (chosen image vs. cropped rejected image) and a reward anchor that keeps r(m_w, q, y_w) positive.*

**(1) Standard multimodal DPO** (backbone):

$$\mathcal{L}_{\text{DPO}_m} = -\log \sigma\!\left(\beta \log \tfrac{\pi_\theta(y_w \mid m, q)}{\pi_{\text{ref} }(y_w \mid m, q)} - \beta \log \tfrac{\pi_\theta(y_l \mid m, q)}{\pi_{\text{ref} }(y_l \mid m, q)}\right)$$

This maximizes $\sigma(r(m,q,y_w) - r(m,q,y_l))$ -- the standard chosen-vs-rejected response margin.

**(2) Conditional Preference Optimization (CoPO)** -- swap the contrast axis from response to image. Hold $(q, y_w)$ fixed and contrast a chosen image $m_w$ against a hard-negative cropped image $m_l$:

$$\mathcal{L}_{\text{CoPO} } = -\log \sigma\!\left(\beta \log \tfrac{\pi_\theta(y_w \mid m_w, q)}{\pi_{\text{ref} }(y_w \mid m_w, q)} - \beta \log \tfrac{\pi_\theta(y_w \mid m_l, q)}{\pi_{\text{ref} }(y_w \mid m_l, q)}\right)$$

Maximizes $\sigma(r(m_w,q,y_w) - r(m_l,q,y_w))$. Because $q$ and $y_w$ are identical on both sides, the only way the model can drive the margin up is by **using image features**, which mechanically removes the language-only shortcut.

The hard-negative $m_l$ is constructed by **random-cropping 0-20%** of the original image -- visually similar enough to be a hard negative, but missing enough information that $y_w$ is no longer optimal. Ablations rule out random images (too easy) and MoCo-v2 augmentation (too similar) as alternatives.

**(3) Anchored Preference Optimization (AncPO)** -- absolute floor on the chosen reward, fixing the well-known DPO pathology where chosen-response log-prob drifts down:

$$\mathcal{L}_{\text{AncPO} } = -\log \sigma\!\left(\beta \log \tfrac{\pi_\theta(y_w \mid m_w, q)}{\pi_{\text{ref} }(y_w \mid m_w, q)} - \delta\right)$$

with $\delta = 0$ (default), forcing $r(m_w, q, y_w) > 0$.

**Combined objective**:

$$\mathcal{L}_{\text{mDPO} } = \mathcal{L}_{\text{DPO}_m} + \mathcal{L}_{\text{CoPO} } + \mathcal{L}_{\text{AncPO} }$$

Unweighted sum -- no coefficient sweep is reported, a minor audit gap (see Limitations).

**Training setup.** LoRA (alpha=128, r=64); 3 epochs; batch 32; lr 1e-5 cosine, warmup 0.1; beta=0.1; delta=0. Identical config for DPO baseline and mDPO -- so head-to-head numbers in Table 1 isolate the objective change.

**Data.** 10K preference instances from **Silkie** (LLaVA-Instruct-150K subset of the full 80K). Bias caveat: Silkie itself is distilled from GPT-4V-family judges, so the "human preference" being optimized for is in fact a model preference.

![Qualitative comparison on MMHalBench: DPO ignores the image, mDPO grounds in it](/assets/images/paper/mdpo/fig_p003_01.png)
*Figure 3: Qualitative examples on MMHalBench. Standard DPO accepts an adversarial false-premise (top) and answers from a language prior while ignoring the image (bottom). mDPO grounds in the image in both cases -- the same behavioural signature as the diagnostic in Figure 1.*

## Experimental Results

**Main numbers** (Table 1; identical LoRA training config across rows -- so the deltas isolate the objective):

| Model | Objective | MMHal Score ↑ | MMHal HalRate ↓ | ObjHal CHAIRs ↓ | ObjHal CHAIRi ↓ | AMBER CHAIRs ↓ | AMBER Cover. ↑ | AMBER HalRate ↓ | AMBER Cog. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Bunny-v1.0-3B | base | 2.11 | 0.58 | 43.0 | 8.9 | 9.8 | 75.6 | 64.9 | 6.0 |
| Bunny-v1.0-3B | + DPO | 2.28 | 0.56 | 44.3 | 7.6 | 7.9 | 74.1 | 58.9 | 4.8 |
| **Bunny-v1.0-3B** | **+ mDPO** | **2.96** | **0.42** | **27.0** | **4.6** | **4.9** | 67.4 | **37.7** | **2.4** |
| LLaVA-v1.5-7B | base | 2.19 | 0.57 | 54.7 | 15.9 | 7.4 | 51.8 | 34.7 | 4.1 |
| LLaVA-v1.5-7B | + DPO | 2.14 | 0.65 | 49.0 | 13.0 | 6.5 | 55.1 | 34.5 | 2.3 |
| **LLaVA-v1.5-7B** | **+ mDPO** | **2.39** | **0.54** | **35.7** | **9.8** | **4.4** | 52.4 | **24.5** | 2.4 |

For Bunny-3B, mDPO closes most of the gap to Qwen-VL-Chat + Silkie-80K (a 7B model trained on 8x the preference data: 3.01 / 0.41 on MMHalBench).

**Ablations (Bunny-3B, MMHalBench score):**

| Variant | MMHal Score | Δ vs mDPO |
|---|---|---|
| **mDPO (full)** | **2.96** | -- |
| -- CoPO (DPO + AncPO only) | 2.36 | -0.60 |
| -- AncPO (DPO + CoPO only) | 2.50 | -0.46 |
| -- both (= DPO baseline) | 2.28 | -0.68 |

CoPO is the dominant term; AncPO alone barely beats vanilla DPO (2.50 vs 2.28). This is the **cleanest piece of evidence in the paper** -- it cleanly attributes the gain to the image-axis preference signal.

**Rejected-image strategy** (Table 3, Bunny-3B, MMHal score): Crop 0-20% (2.96) > Crop 20-50% (2.92) > MoCo-v2 (2.82) > Random image (2.81). Hard-negative design matters; the 0-20%-vs-20-50% gap is small enough that it should be read as a noise-floor result rather than a tuned hyperparameter.

**Anchor placement** (Table 4): anchoring only on $y_w$ is sufficient; adding anchors on $y_l$ or $m_l$ yields no gain and can hurt CHAIR.

**Fine-grained MMHalBench** (Table 5): mDPO wins on 7/8 categories; the **adversarial** category (false-premise questions about the image) jumps from 1.50 -> 4.17 -- the largest single-category gain, exactly where image-grounding should matter most. The only clear loss is **attribute** (3.25 -> 3.08).

![Human evaluation on MMHalBench: 89% win-or-tie for mDPO over DPO](/assets/images/paper/mdpo/fig_p006_01.png)
*Figure 4: Human evaluation on MMHalBench. On 96 items, mDPO is preferred or tied on 89% of instances vs. standard DPO, with the preference-vs-tie split roughly even.*

![Data scaling on MMHalBench: mDPO improves with preference data, DPO does not](/assets/images/paper/mdpo/fig_p006_02.png)
*Figure 5: Data scaling on MMHalBench (Bunny-3B). mDPO's score and hallucination rate improve monotonically with preference-data size; standard DPO flatlines or regresses -- supporting the paper's claim that the failure is in the objective, not the data.*

## Limitations

**Authors admit.** (a) only two MLLMs evaluated, both linear-projector decoder-only; (b) orthogonal DPO improvements (length-debiasing, KTO, IPO) are not combined; (c) only three benchmarks, all on general images.

**Flagged but unaddressed.**

- **Why does cropping work as a hard negative?** The 0-20% crop is empirical. A semantic-region-erase ablation (object segmentation, attention-based masking) would clarify whether CoPO is teaching "see the whole image" or "see *any* image content."
- **Reward-margin diagnostic absent.** The paper never plots the implicit reward distributions for $(m_w, q, y_w)$ vs $(m_l, q, y_w)$ after mDPO training -- which would directly verify that the image-conditional reward gap actually opens up.
- **AncPO claim under-instrumented.** AncPO is presented as solving a DPO pathology (chosen-log-prob drift), but the chosen-response log-prob trajectory is not plotted. A before/after curve would have made the claim concrete.
- **Coverage trade-off.** AMBER Coverage drops from 75.6 (base) -> 67.4 with mDPO on Bunny-3B. The paper dismisses this as "minor", but an 8.2-point coverage drop in exchange for hallucination reduction is a real precision-recall trade -- not a free lunch.
- **No loss-weight sweep.** The 1:1:1 sum across DPO + CoPO + AncPO is unjustified empirically. Under tuning, the CoPO-vs-AncPO attribution could shift.
- **Medical / safety transfer untested.** Given that "language prior dominates" is exactly how MLLMs fail in radiology VQA and pathology captioning, the medical follow-up is the most natural extension and the most under-discussed in the paper.

## Why It Matters for Medical AI

The mDPO diagnostic ("DPO collapses to language-only when image-text pairs are mostly redundant") is most concerning precisely in **templated-report domains** -- radiology, pathology, ophthalmology -- where the language prior is highly structured (CXR reads, biopsy descriptions, OCT impressions) and image-grounded variance is the entire clinical signal. If a medical MLLM trained with off-the-shelf DPO learns to sound like a radiologist without looking at the pixels, it will pass linguistic metrics while missing rare findings -- mirroring the macro-F1 vs micro-F1 split we see in CXR report-generation work (MedRAX, M4CXR). CoPO is a drop-in, training-free addition to existing DPO pipelines (no extra annotation, no diffusion negatives), which makes it unusually low-friction to try on **Med-PaLM-M-, MedRAX-, or LLaVA-Med-style** preference-fine-tuning loops. The untested but plausible extension: replace the random-crop hard negative with a clinically meaningful occlusion (mask the lesion, blank the affected lobe) so the conditional reward gap is forced to depend on the *diagnostically relevant* region rather than the whole-image gist.

## References

- **Paper.** Wang, Zhou, Huang, Xu, Zhang, Poon, Chen. "mDPO: Conditional Preference Optimization for Multimodal Large Language Models." NeurIPS 2024 / EMNLP 2024 main. arXiv:2406.11839.
- **Project / code.** [https://feiwang96.github.io/mDPO/](https://feiwang96.github.io/mDPO/)
- **Concurrent: V-DPO.** Xie et al. "V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization." EMNLP 2024.
- **Preference data.** Li et al. "Silkie: Preference Distillation for Large Visual Language Models." 2023.
- **Benchmarks.** Sun et al. (MMHalBench, 2023); Rohrbach et al. (Object HalBench / CHAIR, 2018); Wang et al. (AMBER, 2023).
- **Base models.** Bunny-v1.0-3B (SigLIP + Phi-2); LLaVA-v1.5-7B (CLIP + Vicuna).
- **Related preference work.** RLHF-V; HA-DPO; POVID; KTO; IPO.
