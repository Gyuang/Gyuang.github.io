---
title: "POVID: Aligning Modalities in Vision Large Language Models via Preference Fine-tuning"
excerpt: "DPO with gold-truth y_w and fully AI-synthesized y_l (GPT-4V rewrite + noisy-image online decode) drops LLaVA-1.5-7B CHAIRs from 66.8 to 31.8 and CHAIRi from 12.7 to 5.4 while raising POPE and MMHal."
categories: [Paper, VLM-Alignment, LLM]
permalink: /paper/povid/
tags:
  - POVID
  - DPO
  - VLLM Hallucination
  - Preference Tuning
  - LLaVA-1.5
  - GPT-4V
  - Modality Alignment
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- POVID reframes VLLM hallucination as a **modality-alignment** problem and tunes LLaVA-1.5-7B with DPO where the *preferred* response is the gold LLaVA-Instruct answer and *only the dispreferred side* is synthesized — (i) GPT-4V rewrites the gold answer with co-occurrence, attribute, and logical-relation hallucinations to produce `y_l^t` offline, and (ii) the target model decodes `y_l^n` online from a Gaussian-noised image at diffusion step `k=500`. A two-coefficient DPO loss (`β₁`, `β₂`) merges both branches over a 3+1-epoch schedule.
- **Headline numbers (LLaVA-1.5-7B):** CHAIRs **66.8 → 31.8**, CHAIRi **12.7 → 5.4**, POPE **85.9 → 86.9**, MMHal **2.42 → 2.69**, with +5.3 average on four general benchmarks. POVID is the average-rank-1 model across **all eight benchmarks** vs. Silkie/Vlfeedback, LLaVA-RLHF/Human-Preference, and RLHF-V at matched backbone.
- The *design* (truth-anchored y_w + synthetic y_l) is conceptually clean, but the paper never runs the head-to-head that would isolate it — every reported delta also bundles a different dispreference-data source. The noise-branch may additionally be equivalent to a training-time VCD regularizer, which the paper does not test against.

![POVID radar comparison](/assets/images/paper/povid/page_001.png)
*Figure 1: Radar comparison of POVID against prior VLLM preference-tuning methods (Vlfeedback, Human-Preference / LLaVA-RLHF, RLHF-V) across 8 benchmarks — POVID dominates all axes on the LLaVA-1.5-7B backbone.*

## Motivation

By late 2023 the dominant VLLM hallucination story had two strands: decoding-time / post-hoc fixes (VCD, Woodpecker, OPERA) and RLHF-style preference tuning. Strand (ii) inherited a structural flaw from LLM-RLHF — both `y_w` and `y_l` are sampled from the model, so neither is guaranteed correct. In VLLMs, where the answer has to be grounded in a specific image, comparing two wrong captions is a noisy signal: nothing in the contrast pins the loss to the image. RLHF-V (Yu et al. 2023b) bypasses this with expensive human-written segment-level corrections.

POVID's wager is that you can keep the cheap fully-synthetic recipe of Silkie/Vlfeedback *and* recover the ground-truth anchor of RLHF-V by inverting the data pipeline: keep the gold caption as `y_w` and synthesize *only* the dispreferred side. The introduction frames the stakes around medical and autonomous-driving deployment — domains where hallucinated objects are a safety failure rather than a UX complaint.

## Core Innovation

1. **Truth-anchored DPO pairs.** `y_w` is always the unchanged LLaVA-Instruct-150K ground-truth answer; only `y_l` is synthesized. The contrast is "truth vs. AI-generated hallucination," not "sample vs. sample."
2. **Two complementary dispreference branches.**
   - `y_l^t` (text-side, offline): GPT-4V is prompted to rewrite the gold answer into a lexically-similar but plausibly-wrong version, steered toward three hallucination archetypes — object co-occurrence, logical-relation errors, and incorrect attributes.
   - `y_l^n` (image-side, online): The target VLLM decodes greedily under a Gaussian-noised image at diffusion step `k=500`, teacher-forced on the preferred prefix so the contrast localizes to tokens that actually diverge under noise.
3. **Two-coefficient DPO loss.** Separate `β₁`, `β₂` weight the textual and noisy-image dispreferences inside a single sigmoid log-ratio objective.
4. **Two-phase schedule.** 3 epochs with `β₂=0` (text-only), then 1 epoch with both branches active and online noisy-image decoding.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | Truth-anchored y_w + AI-synthesized y_l beats all prior DPO-for-VLLM recipes at matched backbone | Table 1: POVID best on all 8 benchmarks vs. Vlfeedback / Human-Preference / RLHF-V on LLaVA-1.5-7B | ⭐⭐⭐ |
| C2 | Using gold y_w avoids the "both responses wrong" failure mode of prior DPO-for-VLLM | Conceptual argument in Sec. 1/3; no controlled experiment swapping in two-sample y_w on the same 17K seed prompts | ⭐ |
| C3 | Noising the image triggers inherent text-prior hallucinations that the model learns to suppress | Figure 4 logit-shift curve (one prompt, one image); Table 3 image-only ablation (66.8 → 50.4 CHAIRs) | ⭐⭐ |
| C4 | Both dispreference branches are individually necessary | Table 3 ablation: text-only 39.6 / image-only 50.4 / both 31.8 CHAIRs — monotone | ⭐⭐⭐ |
| C5 | POVID improves general VLLM capability, not just hallucination | SciQA-IMG 66.8 → 68.8, MM-Vet 30.5 → 31.8, MMBench 63.0 → 64.9, LLaVA-Bench 63.4 → 68.7 | ⭐⭐ |
| C6 | POVID realigns attention toward image tokens (mechanistic claim) | Figure 5: 2 qualitative attention maps; no aggregate image-attention metric | ⭐ |
| C7 | Beats Qwen-VL-Chat / InstructBLIP / mPLUG-Owl2 despite a smaller vision backbone | Table 2 average ranking 1.50 / 1.75 across 8 benchmarks | ⭐⭐ |
| C8 | 31.78% average reduction across hallucination benchmarks | Headline number in §4.2; not normalization-specified, recomputable to ~30.6–31.8% depending on weighting | ⭐ |
| C9 | "Easily scalable" — 6 hours on one A100 | Sec. 4.1 says 6h, Appendix A.1 says 20h — internal contradiction | ⭐ |

**Honest read.** C1 and C4 are well-supported: at matched backbone, POVID beats every other DPO-for-VLLM recipe across eight benchmarks, and the ablation cleanly shows both branches contribute. The weaker links are the mechanism stories (C3 and C6) and the central conceptual claim (C2). The most important missing experiment is the **direct head-to-head between "gold y_w + synthetic y_l" and "two synthetic samples"** on the same 17K seed prompts and the same loss — without it, every C1 delta confounds dispreference design with dispreference *source* data quality. The noise-branch story has a related issue: a noised image is approximately uninformative, so `y_l^n` is essentially the model's text-only prior continuation of `y_w` — i.e., the loss is partly a training-time analogue of visual-contrastive decoding (Leng et al. 2023), but the paper never benchmarks against VCD-style decoding-time fixes on the same checkpoint. Variance reporting is absent (single run, no seeds, no CIs), gains concentrate on captioning (**+6.7**) over reasoning (**+1.9**) in the LLaVA-Bench breakdown, the attention-realignment claim rests on **only two qualitative cases**, and Sec. 4.1 ("6 hours on one A100") vs. Appendix A.1 ("20 hours") contradict each other on training cost.

## Method & Architecture

![POVID framework](/assets/images/paper/povid/page_003.png)
*Figure 2: POVID framework. Stage 1 has GPT-4V inject plausible hallucinations into the ground-truth caption to form `y_l^t`; Stage 2 corrupts the image with diffusion noise to provoke the VLLM's own text-prior continuation `y_l^n`. Both feed a two-term DPO loss.*

### Stage 1 — GPT-4V-injected textual dispreference

Seeds: a 17K random subset of LLaVA-Instruct-150K, spanning captioning, simple VQA, and reasoning. `y_w` is the unchanged gold answer. GPT-4V is shown the image plus `y_w` and asked to produce a "highly confusing" lexically-similar rewrite, with explicit steering toward three archetypes:

1. **Object co-occurrence** — insert plausible-but-absent objects (e.g., "twenty oranges", "fruit knife") that often co-occur in the training distribution with what is actually present.
2. **Logical-relation errors** — swap relations between entities ("The moon is shining in from outside" instead of "The sun is shining").
3. **Incorrect attributes** — change colors, counts, or shapes.

For reasoning items, a separate prompt asks GPT-4V to corrupt the `Reason: ... Result: ...` chain while keeping the rewrite image-consistent and subtle.

![POVID textual-dispreference examples](/assets/images/paper/povid/page_004.png)
*Figure 3: Two GPT-4V-generated dispreferred examples — (a) captioning task with injected hallucinatory entities, attribute errors, and logical-relation swaps; (b) reasoning task with corrupted reasoning chain.*

### Stage 2 — noisy-image-triggered dispreference

The second branch corrupts the image with diffusion-style Gaussian noise:

$$x^{(k)} = \sqrt{\bar{\xi}_k}\, x + \sqrt{1 - \bar{\xi}_k}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \bar{\xi}_k = \prod_{i=0}^{k} \xi_i.$$

To pick the noise level, the authors track next-token logits for the prompt "In the image there are knife and ___" as `k` increases: `plate` (ground truth) dominates at `k=0`, `fork` (text-prior co-occurrence) takes over around `k=500`, and `pixel` (degenerate noise-text artifact) wins by `k=999`. POVID operates at **`k=500`** — noisy enough to expose the text prior, not yet meaningless.

![Logit curve vs. noise step](/assets/images/paper/povid/page_005.png)
*Figure 4: Next-token logits for "In the image, there are knife and ___" as a function of diffusion noise step k. Ground-truth `plate` peaks at k=0; co-occurrence prior `fork` dominates around k=500 (POVID's operating point); `pixel` takes over by k=999.*

The dispreferred tokens `y_l^n` are **decoded online** during DPO, each chosen as the argmax of `π_θ(·|x^n, y_w,<i)` — teacher-forced on the preferred prefix so the contrast localizes to tokens that actually diverge under noise.

### Combined DPO objective

Both branches are folded into one loss with separate coefficients:

$$\mathcal{L}_{\text{POVID} } = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D} } \left[\log \sigma\left( \alpha \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref} }(y_w|x)} - \left(\beta_1 \log \frac{\pi_\theta(y_l^t|x)}{\pi_{\text{ref} }(y_l^t|x)} + \beta_2 \log \frac{\pi_\theta(y_l^n|x^n)}{\pi_{\text{ref} }(y_l^n|x^n)} \right) \right)\right]$$

`y_l^n` is conditioned on the noisy image `x^n`; `y_l^t` shares the clean image `x` with `y_w`. The reference policy `π_ref` is the SFT'd LLaVA-1.5-7B before POVID.

### Training schedule and hyperparameters

- **Phase 1 (3 epochs):** `β₂=0`, text-only DPO on `(x, y_w, y_l^t)`.
- **Phase 2 (1 epoch):** full objective with online noisy-image decoding at `k=500`.
- Backbone: **LLaVA-1.5 7B** (Vicuna-7B + ViT-L/14 CLIP), LoRA `r=128, α=256`, mm-projector LR `2e-5`, batch size 1, LR `1e-7`, max len 1024.
- Compute: ~**6 hours on one A100 80GB** per Sec. 4.1, but Appendix A.1 reports **20 hours** — an internal inconsistency the paper does not resolve.

## Experimental Results

### Main comparison at matched LLaVA-1.5-7B backbone (Table 1)

| Method | CHAIRs ↓ | CHAIRi ↓ | POPE ↑ | MMHal ↑ | SciQA-IMG ↑ | MM-Vet ↑ | MMBench ↑ | LLaVA-Bench ↑ |
|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5 (base) | 66.8 | 12.7 | 85.90 | 2.42 | 66.8 | 30.5 | 63.0 | 63.4 |
| + Vlfeedback (Silkie data) | 56.3 | 11.4 | 83.72 | 2.62 | 66.2 | 31.2 | 63.9 | 62.1 |
| + Human-Preference (LLaVA-RLHF data) | 54.0 | 9.3 | 81.50 | 2.53 | 65.8 | 31.1 | 60.4 | 63.7 |
| + RLHF-V | 44.6 | 7.9 | 86.20 | 2.59 | 67.1 | 30.9 | 63.6 | 65.4 |
| **POVID (ours)** | **31.8** | **5.4** | **86.90** | **2.69** | **68.8** | **31.8** | **64.9** | **68.7** |

Relative to base, POVID cuts CHAIRs by 52.4% and CHAIRi by 57.5%; relative to the strongest baseline (RLHF-V) it cuts CHAIRs by another 12.8 absolute points. POPE and MMHal both move up rather than degrading — the "wins everywhere, no trade-off" framing.

### Ablation (Table 3)

| Text dispref. | Image distortion | CHAIRs ↓ | CHAIRi ↓ | POPE ↑ | MMHal ↑ | LLaVA-Bench ↑ |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 66.8 | 12.7 | 85.90 | 2.42 | 62.1 |
| ✓ | ✗ | 39.6 | 6.3 | 86.04 | 2.65 | 67.5 |
| ✗ | ✓ | 50.4 | 9.6 | 85.19 | 2.54 | 66.9 |
| **✓** | **✓** | **31.8** | **5.4** | **86.90** | **2.69** | **68.7** |

Text-dispreference alone does most of the heavy lifting (66.8 → 39.6 CHAIRs); image-distortion alone is weaker (66.8 → 50.4) but additive. The image-distortion branch is what is novel relative to prior DPO work, but it is **not** the dominant signal.

### Fine-grained LLaVA-Bench (Table 4)

POVID's biggest gain is on **Captioning / Detail-description** (53.4 → 60.1, **+6.7**), consistent with long-form ground-truth captions dominating the 17K seed. Reasoning and Conversation gains are smaller (**+1.9**, +2.6). POVID is mostly a captioning-hallucination fix.

### Modality-alignment qualitative analysis

![POVID attention maps](/assets/images/paper/povid/page_008.png)
*Figure 5: Attention maps for LLaVA-1.5 vs. POVID on a captioning case (left) and a yes/no VQA case (right). Red boxes mark image-token regions where POVID's attention mass increases relative to the base model.*

Two cases. There is no aggregate "image-attention share" metric on a held-out set — this is a vignette, not a measurement.

## Limitations

**Acknowledged by the authors.** Noise step `k` must be in a "reasonable range" with only anecdotal calibration (Figure 4); the recipe relies on GPT-4V (cost/TOS); only the LLaVA-1.5-7B backbone is evaluated.

**Not addressed.**

- No 13B / 34B scaling — does the recipe still help when the base model is already less hallucination-prone?
- No medical-domain evaluation despite the medical-AI framing — no test on LLaVA-Med, RadFM, or any radiology VQA benchmark, where hallucination cost is highest and the co-occurrence priors GPT-4V exploits (kitchen ↔ oven, knife ↔ fork) are not the dominant failure mode.
- **No isolation of C2**: the central design claim (gold y_w + synthetic y_l beats two-sample y_w) is never tested with both arms on the same 17K seed prompts and the same loss; every Table 1 delta also changes the dispreference-data source.
- **No VCD baseline**: the noise-branch is mechanistically similar to a training-time visual-contrastive-decoding regularizer (a noised image makes y_l^n ≈ text-only prior continuation of y_w), but the paper never measures the marginal contribution over a cheap decoding-time fix on the same checkpoint.
- **Attention-realignment claim is n=2** — there is no aggregate metric for "image attention share" on a held-out set.
- **Single run** — every number in Tables 1-4 is one seed with no CIs / significance test, despite POPE / MMBench / MMHal having known run-to-run swings of 1-2 points.
- **Gains skew to captioning** — Captioning +6.7 vs. Reasoning +1.9 on LLaVA-Bench suggests the recipe is mainly a long-form-caption fix.
- **No safety / over-refusal audit** — DPO often makes models more conservative; a refusal-rate or AMBER-style judgment-bias check is missing.
- **No audit of dispreference quality** — how often GPT-4V's "hallucinated" rewrites are still factually correct (which would inject label noise into DPO) is not reported.
- **Internal inconsistency on cost** — Sec. 4.1 reports 6 hours on one A100; Appendix A.1 reports 20 hours, unexplained.

## Why It Matters for Medical AI

The introduction explicitly motivates POVID with medical deployment, but the empirical scope is general-domain (LLaVA-Instruct, COCO-based CHAIR, MMBench, LLaVA-Bench). The transferability claim is therefore aspirational — the hallucination archetypes GPT-4V is steered toward (kitchen co-occurrences, generic attribute errors) are not the dominant failure modes in radiology (where the costly hallucinations are missed nodules, mislabeled laterality, and incorrect severity). Anyone who wants to apply POVID to a medical VLLM (LLaVA-Med, RadFM, MAIRA) should expect to (1) replace the GPT-4V dispreference generator with a clinically-grounded counterpart, (2) re-calibrate the noise step `k` for medical image statistics, and (3) re-validate on hallucination metrics that are sensitive to clinically meaningful errors rather than COCO-object presence. As a *framework* — gold y_w plus targeted synthetic y_l, with a separate β coefficient for an image-corruption regularizer — the recipe is portable; as published numbers, it is a general-domain captioning-hallucination fix.

## References

- Paper: [Aligning Modalities in Vision Large Language Models via Preference Fine-tuning (arXiv 2402.11411)](https://arxiv.org/abs/2402.11411)
- Code: [github.com/YiyangZhou/POVID](https://github.com/YiyangZhou/POVID)
- LLaVA-1.5 backbone: Liu et al. 2023, "Improved Baselines with Visual Instruction Tuning."
- Companion DPO-for-VLLM work: Silkie / Vlfeedback (Li et al. 2023d), LLaVA-RLHF (Sun et al. 2023), RLHF-V (Yu et al. 2023b), HA-DPO (Zhao et al. 2023).
- Decoding-time hallucination fixes referenced for context: VCD (Leng et al. 2023), Woodpecker, OPERA.
- Benchmarks: CHAIR (Rohrbach et al. 2018), POPE (Li et al. 2023), MMHal (Sun et al. 2023), MM-Vet, MMBench, ScienceQA, LLaVA-Bench.
