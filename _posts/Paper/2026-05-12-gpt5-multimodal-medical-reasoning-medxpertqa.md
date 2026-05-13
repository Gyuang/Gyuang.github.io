---
title: "Capabilities of GPT-5 on Multimodal Medical Reasoning"
excerpt: "A zero-shot CoT API evaluation puts GPT-5 at 69.99% / 74.37% on MedXpertQA MM Reasoning/Understanding — +29.26 / +26.18 over GPT-4o — but the multimodal headline rides on a single benchmark and a re-quoted pre-licensed-trainee baseline."
categories:
  - Paper
  - LLM
permalink: /paper/gpt5-multimodal-medical-reasoning-medxpertqa/
tags:
  - GPT-5
  - MedXpertQA
  - VQA-RAD
  - Chain-of-Thought
  - LLM-Evaluation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- A black-box, zero-shot chain-of-thought (CoT) evaluation of GPT-5, GPT-5-mini, GPT-5-nano, and GPT-4o-2024-11-20 across MedQA, MMLU-Medical, USMLE Self Assessment, MedXpertQA (Text + MM), and VQA-RAD — same prompt template, single run per item, no fine-tuning or retrieval.
- Headline: on **MedXpertQA MM** GPT-5 scores **69.99% Reasoning / 74.37% Understanding**, beating GPT-4o by **+29.26 / +26.18** points and the cited pre-licensed-trainee baseline by **+24.23 / +29.40** points.
- Caveats the abstract drops: the "human expert" baseline is **re-quoted from MedXpertQA's own paper** and refers to **pre-licensed trainees, not attending physicians**; the multimodal lead rests on a single benchmark; GPT-5 actually **loses VQA-RAD-yes/no to GPT-5-mini** (70.92 vs 74.90).

## Motivation

Prior medical-LLM evaluations have been mostly text-only, used heterogeneous prompts and scoring, and compared models across non-comparable splits — making it hard to attribute gains to the model versus the harness. The authors run a controlled A/B between GPT-4o and the GPT-5 family on the same benchmarks under one fixed zero-shot CoT prompt, to ask whether a single instruction-following generalist can act as a "hub" for multimodal medical decision support without specialist fine-tuning. The framing is timely because MedXpertQA (released January 2025) is the first widely available expert-level medical benchmark with both text and image subsets explicitly designed to resist leakage.

## Core Innovation

- There is **no new model and no new training pipeline**. The contribution is methodological discipline: identical zero-shot CoT template applied to every model and every item.
- **Two-turn convergence protocol.** Turn 1 asks the model to "think step by step"; Turn 2 cues a single-letter final answer with `"Therefore, among A through {END_LETTER}, the answer is"`. This separates free-form rationale from exact-match scoring.
- **Multimodal handling is deliberately minimal** — images are passed as `image_url` blocks inside the first user turn, no captioning, no cropping, no tool use. The harness is the same for QA and VQA except for the image block.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | GPT-5 "consistently outperforms all baselines, achieving state-of-the-art accuracy across all QA benchmarks." | Tables 1–3, monotone improvement over GPT-4o on every text benchmark. | MedQA, MMLU-Med (6 subj), USMLE Self Assessment, MedXpertQA Text | ⭐⭐ — point estimates only; no variance; no non-OpenAI baseline (Claude, Gemini, Med-PaLM 2, MEDITRON). "SOTA" as written means "beats GPT-4o-2024-11-20." |
| C2 | +29.26% Reasoning / +26.18% Understanding on MedXpertQA MM vs GPT-4o. | Table 3, MedXpertQA MM row. | MedXpertQA MM | ⭐⭐ — deltas are large enough to survive noise, but single-run, single-prompt, no per-specialty breakdown. |
| C3 | "Surpasses pre-licensed human experts by +24.23 / +29.40" on MedXpertQA MM. | Table 4. | MedXpertQA MM | ⭐ — human numbers are **re-quoted from the MedXpertQA paper**, not collected by the authors. "Pre-licensed" = trainees/students pre-board-certification, not attending specialists; humans presumably without internet vs LLM with full CoT. Not apples-to-apples. |
| C4 | GPT-4o remains below human expert performance on most dimensions. | Table 4, GPT-4o vs expert deltas (-15.90 to +3.22). | MedXpertQA Text + MM | ⭐⭐ — directionally supported by the same secondary-source human numbers as C3. |
| C5 | A representative case shows GPT-5 integrating visual and textual cues into a coherent diagnostic chain. | Figure 3, MM-1993 (Boerhaave). | n=1, MedXpertQA MM | ⭐ — single cherry-picked example; no failure cases shown. |
| C6 | Gains "suggest enhancements in cross-modal attention and alignment within the model's architecture or training." | Discussion §4. | — | ⭐ — pure speculation; the authors have no visibility into GPT-5 internals. |
| C7 | VQA-RAD inversion attributed to "scaling-related calibration." | Table 3 (70.92 vs 74.90). | VQA-RAD yes/no (n=251) | ⭐ — at n=251 a 4-point gap is within the Wilson 95% CI (~±6 pts); the proposed explanation is post-hoc. |
| C8 | "Zero-shot CoT synergizes with GPT-5's enhanced internal reasoning capacity." | Larger gains on MedXpertQA Text Reasoning than on MMLU factual splits. | MedXpertQA Text + MMLU | ⭐⭐ — pattern is consistent but there is no CoT-off ablation isolating the prompt's contribution. |

## Method & Architecture

![GPT-5 zero-shot CoT prompt template on a MedXpertQA multimodal case](/assets/images/paper/gpt5-medical-reasoning/page_005.png)

*Figure 1: the zero-shot CoT prompt template used for every multimodal item — system message, user turn with text + image_url block, free-form rationale, then a convergence cue that forces a single-letter final answer.*

The harness has five turns per item:

1. **System.** `"You are a helpful medical assistant."`
2. **User.** `Q: {QUESTION_TEXT}\nA: Let's think step by step.` — for VQA, all images for the item are appended as `image_url` entries in this same user turn.
3. **Assistant.** Free-form rationale, stored as `prediction_rationale`.
4. **User (convergence).** `"Therefore, among A through {END_LETTER}, the answer is"`
5. **Assistant.** Single letter, stored as `prediction`. Scoring is exact-match on this letter.

Decoding settings are not reported — no temperature, top-p, seed, or number of samples per question is given, and a single run per item is implied. All four models (GPT-5, GPT-5-mini, GPT-5-nano, GPT-4o-2024-11-20) are accessed through the OpenAI API, so reproducibility is bounded by whatever OpenAI's API defaults were on the date of querying.

![QA vs VQA prompt skeletons](/assets/images/paper/gpt5-medical-reasoning/page_004.png)

*Figure 2: QA vs VQA prompt skeletons — the only structural difference is the `image_url` block in the first user turn.*

## Experimental Results

### Main text benchmarks

![Text-benchmark tables — MedQA, MMLU-Medical, USMLE, MedXpertQA Text](/assets/images/paper/gpt5-medical-reasoning/page_003.png)

*Figure 3: Tables 1 and 2 from the paper. The lead over GPT-4o widens on the hardest text benchmark (MedXpertQA Text, +25–26 points) while MMLU-Medical gains are small because GPT-4o is already near ceiling.*

| Benchmark | Metric | **GPT-5** | GPT-5-mini | GPT-5-nano | GPT-4o-2024-11-20 |
|---|---|---|---|---|---|
| MedQA US (4-opt) | Acc | **95.84** (↑4.80) | 93.48 | 91.44 | 91.04 |
| MMLU Anatomy | Acc | **92.59** (↑1.48) | 92.59 | 88.15 | 91.11 |
| MMLU Clinical Knowledge | Acc | **95.09** (↑2.64) | 91.32 | 89.81 | 92.45 |
| MMLU College Biology | Acc | **99.31** (↑2.09) | 99.31 | 97.92 | 97.22 |
| MMLU College Medicine | Acc | **91.91** (↑1.74) | 88.44 | 85.55 | 90.17 |
| MMLU Medical Genetics | Acc | **100.00** (↑4.00) | 99.00 | 98.00 | 96.00 |
| MMLU Professional Med | Acc | **97.79** (↑1.10) | 97.43 | 96.69 | 96.69 |
| USMLE Step 1 | Acc | **93.28** (↑0.84) | 93.28 | 93.28 | 92.44 |
| USMLE Step 2 | Acc | **97.50** (↑4.17) | 95.83 | 90.00 | 93.33 |
| USMLE Step 3 | Acc | **94.89** (↑3.65) | 94.89 | 92.70 | 91.24 |
| USMLE Avg | Acc | **95.22** (↑2.88) | 94.67 | 91.99 | 92.34 |
| MedXpertQA Text | Reasoning | **56.96** (↑26.33) | 45.94 | 36.38 | 30.63 |
| MedXpertQA Text | Understanding | **54.84** (↑25.30) | 43.80 | 33.96 | 29.54 |

### Multimodal benchmarks and the headline claim

![Multimodal tables — MedXpertQA MM, VQA-RAD, and the vs-expert comparison](/assets/images/paper/gpt5-medical-reasoning/page_006.png)

*Figure 4: Tables 3 and 4. The +26–29 point lift on MedXpertQA MM and the +24.23 / +29.40 gap over the pre-licensed-trainee baseline drive the abstract's super-human framing. Same page also shows the VQA-RAD inversion: GPT-5 (70.92) under-performs GPT-5-mini (74.90).*

| Benchmark | Metric | **GPT-5** | GPT-5-mini | GPT-5-nano | GPT-4o-2024-11-20 | Pre-licensed trainee |
|---|---|---|---|---|---|---|
| MedXpertQA MM | Reasoning | **69.99** (↑29.26) | 60.51 | 45.44 | 40.73 | 45.76 |
| MedXpertQA MM | Understanding | **74.37** (↑26.18) | 61.37 | 45.85 | 48.19 | 44.97 |
| VQA-RAD (yes/no, n=251) | Acc | 70.92 | **74.90** (↑4.99) | 65.34 | 69.91 | — |

A qualitative case (Figure 3 in the paper, MedXpertQA MM-1993) shows GPT-5 correctly distinguishing Boerhaave syndrome from a Mallory-Weiss tear and selecting Gastrografin swallow as the next study. It is one cherry-picked case; the paper shows no failure examples.

![GPT-5 chain-of-thought on the Boerhaave case (MM-1993)](/assets/images/paper/gpt5-medical-reasoning/page_007.png)

*Figure 5: GPT-5's rationale on MedXpertQA MM-1993. Useful as color; not evidence of anything beyond a single case.*

### What is missing

- **No ablations.** No prompt-sensitivity test, no temperature sweep, no self-consistency, no CoT-on vs CoT-off, no per-specialty breakdown on MedXpertQA.
- **No variance reporting.** Every number is a single point estimate. At n=251 on VQA-RAD-binary the Wilson 95% CI is roughly ±6 points, so the GPT-5-mini > GPT-5 gap is not statistically distinguishable from zero.
- **No non-OpenAI frontier baseline.** No Claude, no Gemini, no MedGemma, no Med-PaLM 2. The "state of the art" claim is, in practice, a comparison to GPT-4o-2024-11-20.

## Limitations

**Author-acknowledged.** Benchmarks reflect idealized testing conditions and do not capture real-world variability, uncertainty, or ethics. The VQA-RAD inversion is flagged as unexplained. Prospective clinical trials, domain-adapted fine-tuning, and calibration are listed as future work.

**Things a careful reviewer should add:**

- **The "human expert" baseline is a re-quote, not a re-collection.** The MedXpertQA paper's pre-licensed trainee numbers are pasted into Table 4 with no description of sample size, demographics, time limits, or instructions. "Pre-licensed" likely means medical students or residents pre-board-certification — meaningfully weaker than the abstract's "trained professionals" framing.
- **The multimodal headline rides on one benchmark.** MedXpertQA MM is the sole multimodal benchmark where GPT-5 dominates. On VQA-RAD-binary it loses to GPT-5-mini and is statistically tied with GPT-4o. Concentration of risk on one benchmark — and on that benchmark's own paper-supplied human numbers — is the largest structural weakness.
- **No contamination probe for MedXpertQA.** MedXpertQA was posted on arXiv in January 2025; GPT-5 shipped in August 2025. The window is non-trivial and the paper offers no n-gram overlap, membership-inference, or perturbed-question control.
- **Ceiling effects on text benchmarks.** Four of six MMLU-Medical subjects are above 95% and Medical Genetics is at 100%. The authors implicitly concede that GPT-5's "real" gains are on reasoning, not factual recall.
- **VQA-RAD's open-ended split is silently excluded.** Only the 251 yes/no items are scored. Calling the result "VQA-RAD performance" without that qualifier is selective.
- **API determinism.** Results depend on OpenAI's decoding settings on the date of querying. Without a frozen model snapshot guarantee, the numbers are not reproducible by a third party.

## Why It Matters for Medical AI

The defensible reading of this paper is narrow but real: under one fixed zero-shot CoT harness, GPT-5 delivers a large lift over GPT-4o on the **text** medical-reasoning benchmark that was explicitly designed to resist memorization (MedXpertQA Text, +25–26 points), and lifts saturated factual benchmarks (MMLU-Medical, USMLE) close to ceiling. That is consistent with the "super-human medical reasoning on text" framing.

The **multimodal** story is weaker than the abstract suggests. The +26–29 point lift on MedXpertQA MM is striking but unconfirmed by any second multimodal benchmark — VQA-RAD-binary shows GPT-5 losing to its own mini variant — and the super-human comparison is to pre-licensed trainees whose performance was not collected under matched conditions. For anyone reading this paper to inform a deployment decision, the practical question is not "is GPT-5 super-human?" but "how does GPT-5 compare to Claude, Gemini, and domain-tuned open models on more than one multimodal benchmark, with variance estimates and a contamination check?" — and this paper does not answer that.

## References

- arXiv: <https://arxiv.org/abs/2508.08224> ("Capabilities of GPT-5 on Multimodal Medical Reasoning", Wang et al., Aug 2025)
- Code (claimed in abstract): "GPT-5-Evaluation" — no URL embedded in the paper body
- Related benchmark: MedXpertQA (Zuo et al., 2025) — source of the text/MM splits and the cited pre-licensed-trainee baseline
- Related benchmarks: MedQA (Jin et al.), MMLU (Hendrycks et al.), USMLE Self Assessment via Nori et al. 2023, VQA-RAD (Lau et al. 2018)
