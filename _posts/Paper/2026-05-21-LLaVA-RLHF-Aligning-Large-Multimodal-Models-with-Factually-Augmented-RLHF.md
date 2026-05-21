---
title: "LLaVA-RLHF: Aligning Large Multimodal Models with Factually Augmented RLHF"
excerpt: "First end-to-end RLHF for an LMM. Fact-RLHF feeds COCO captions / A-OKVQA rationales / GT answers into the reward model to block reward hacking — but the headline 'Fact-RLHF' bundles three changes, MMHal-Bench is adversarially built around the baseline, and POPE F1 actually drops vs SFT+."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/llava-rlhf/
tags:
  - LLaVA-RLHF
  - Fact-RLHF
  - RLHF
  - Vision-Language Model
  - Hallucination
  - MMHal-Bench
  - Reward Modeling
  - PPO
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- LLaVA-RLHF is the **first paper to actually run RLHF on a large multimodal model** end-to-end (LLaVA-7B/13B + Vicuna-v1.5 + CLIP ViT-L/14), with 10K human preferences (~$3000) labelled for hallucination-first / helpfulness-second.
- The core innovation is **Factually Augmented RLHF (Fact-RLHF)**: at training and inference time the reward model sees `(image, factual info, prompt, response)` — where "factual info" is 5 COCO captions or 3 A-OKVQA rationales — so it can detect ungrounded text that an SFT-initialized RM would otherwise reward-hack into existence.
- Headline numbers: **LLaVA-Bench 95.6% of text-GPT-4**, **MMHal-Bench 2.53 / hallucination rate 0.57** at 13B, **MMBench 60.1%**. But the same model **drops on POPE F1 (82.8 → 81.9)** vs LLaVA-SFT+, and LLaVA-RLHF-7B *under*performs LLaVA-SFT+-7B on MMBench (51.4 vs 52.1) — facts the abstract glosses over.

![LLaVA-RLHF method overview (Fig. 1)](/assets/images/paper/llava-rlhf/page_003.png)
*Figure 1 (page 3 render). (a) SFT-stage misalignment: synthetic dialogue is built from captions, not pixels. (b) Crowd preference collection on LLaVA-SFT+ outputs. (c) The Fact-RLHF reward model receives extra ground-truth captions so it cannot be fooled by confident-but-ungrounded responses.*

## Motivation

LLaVA-style LMMs are tuned on ~150K GPT-4-synthesized image dialogues — orders of magnitude less than text-only Flan (100M+). Worse, the synthetic prompts only see captions/bounding boxes rather than the actual pixels, so they bake **vision-language misalignment** (read: hallucination) into the SFT stage. RLHF is the standard text-domain fix, but naively transferring it to LMMs has two failure modes:

1. **Reward hacking.** An SFT-initialized RM inherits the same blind spots as the policy. The policy then learns to be *long and confident* rather than *grounded* — the classic verbosity hack flagged in AlpacaFarm (Dubois et al., 2023).
2. **Annotation drift.** Touvron et al. 2023b solved (1) for text by iterating fresh annotation rounds; that is prohibitively expensive in the vision setting.

The authors' bet: instead of refreshing annotation, **inject oracle textual facts into the reward model** so it stops being fooled.

![Parked motorcycle qualitative example (Table 1)](/assets/images/paper/llava-rlhf/fig_p002_02.png)
*Motorcycle example from Table 1. LLaVA and LLaVA-SFT+ invent a rider's gender; LLaVA-RLHF refuses. This is the one qualitative example that most cleanly separates capability from honesty.*

## Method

### Stage 1 — Augmented SFT (LLaVA-SFT+)

- Hold out 60K of LLaVA's 158K synthetic conversations for RM + RL (98K kept for SFT).
- Add three human-annotated datasets to the SFT mixture:
  - **VQA-v2** yes/no questions, 83K (Goyal et al., 2017).
  - **A-OKVQA** multiple-choice, 16K (Schwenk et al., 2022).
  - **Flickr30k** turned into a "Spotting Captioning" task, 23K (Chen et al., 2023a).
- Architecture is unchanged from LLaVA: CLIP ViT-L/14 + linear projection + Vicuna-v1.5 (7B at 256×256, 13B at 336×336). The projection matrix is reused from LLaVA.

### Stage 2 — Hallucination-aware preference collection

- Resample the last turn of 10K held-out LLaVA conversations from LLaVA-SFT+ at $T = 0.7$ to get response pairs $(y_0, y_1)$.
- AMT workers answer two questions: (Q1) which response has **fewer hallucinations**? (Q2) on a tie, which is **more helpful**? Honesty before helpfulness.
- Reward model is initialized from SFT+; the last-token embedding is linearly projected to a scalar. Trained with the standard pairwise cross-entropy:

$$
\mathcal{L}(r_\theta) \;=\; -\mathbb{E}\!\left[\log \sigma\big(r_\theta(I, x, y_i) - r_\theta(I, x, y_{1-i})\big)\right].
$$

### Stage 3 — Factually Augmented RLHF (Fact-RLHF)

The reward-model input is rewritten from `(image, prompt, response)` to:

```
(image, factual info, prompt, response)
```

where `factual info` is one of:

- **5 ground-truth COCO captions** for general LLaVA conversations, or
- **3 A-OKVQA rationales** for multi-choice items.

Two additional changes are folded into the same condition:

- **Symbolic rewards** — when the question has a known ground-truth label (VQA-v2 yes/no, A-OKVQA ABCD), the reward gets an explicit correctness penalty.
- **Length penalty** — token count is added as a penalizing factor to suppress the verbose-output reward-hacking pattern.

PPO objective with per-token KL to the SFT initial policy:

$$
\mathcal{L}(\pi_\phi^{\text{RL}}) \;=\; -\mathbb{E}\!\left[\, r_\theta(I, x, y) - \beta \cdot D_{\mathrm{KL}}\!\left(\pi_\phi^{\text{RL}} \,\|\, \pi_{\text{INIT}}\right) \right].
$$

- **RL data pool**: 50K LLaVA first-turn prompts (to avoid pre-existing context hallucinations) + 12K A-OKVQA + 10K VQA-v2 = 72K prompts.
- **Engineering**: all four models (policy, reward, value, initial policy) fit on one GPU via LoRA on every component. The value head is initialized from the reward model. For LLaVA-RLHF-7B, the reward and value are **13B-scale** — an explicit "small policy, larger reward" lever.

## Results

### LLaVA-Bench (relative to text-only GPT-4 %)

| Model | Conv | Detail | Complex | Full-Set |
|-------|------|--------|---------|----------|
| LLaVA-7B | 75.1 | 75.4 | 92.3 | 81.0 |
| LLaVA-SFT+ 7B | 88.8 | 74.6 | 95.0 | 86.3 |
| **LLaVA-RLHF-7B** | **93.0** | **79.0** | **109.5** | **94.1** |
| LLaVA-13B×336 | 87.2 | 74.3 | 92.9 | 84.9 |
| LLaVA-SFT+ 13B×336 | 85.8 | 75.5 | 93.9 | 85.2 |
| **LLaVA-RLHF-13B×336** | **93.9** | **82.5** | **110.1** | **95.6** |

### MMHal-Bench (overall ↑, hallucination rate ↓)

![MMHal-Bench radar plot (Fig. 2)](/assets/images/paper/llava-rlhf/fig_p007_01.png)
*Figure 2. 8-axis MMHal-Bench radar across IDEFICS-80B, Kosmos-2, InstructBLIP-13B, LLaVA-13B×336, and LLaVA-RLHF-13B. Orange envelopes the others on most axes — but see the audit below for caveats.*

| Model | Score | Hall. rate |
|-------|-------|------------|
| Kosmos-2 | 1.69 | 0.68 |
| IDEFICS-80B | 2.05 | 0.61 |
| InstructBLIP-13B | 2.14 | 0.58 |
| LLaVA-13B×336 | 1.11 | 0.84 |
| LLaVA-SFT+ 13B×336 | 2.43 | 0.55 |
| **LLaVA-RLHF-13B** | **2.53** | **0.57** |

### MMBench (CircularEval overall %)

| Model | SFT data | Overall |
|-------|----------|---------|
| LLaVA-SFT+ 7B | 220K | **52.1** |
| LLaVA-RLHF-7B | 280K | 51.4 |
| LLaVA-SFT+ 13B×336 | 220K | 57.5 |
| **LLaVA-RLHF-13B×336** | 280K | **60.1** |

Note the 7B regression: alignment tax is real at small scale.

### POPE (F1 by split)

| Model | Random | Popular | Adversarial | Overall | Yes % |
|-------|--------|---------|-------------|---------|-------|
| LLaVA-SFT+ 7B | 85.5 | 82.4 | 80.1 | **82.7** | 47.1 |
| LLaVA-RLHF-7B | 83.3 | 81.8 | 79.5 | 81.5 | 41.8 |
| LLaVA-SFT+ 13B | 84.8 | 82.6 | 81.1 | **82.8** | 41.9 |
| LLaVA-RLHF-13B | 83.5 | 81.8 | 80.5 | 81.9 | 39.0 |

On the cleaner third-party hallucination benchmark, **RLHF does not help and slightly hurts F1** at both scales. The paper attributes this to the more conservative "Yes %" — a plausible but unsupported story.

### Ablations (Table 5, 7B policy)

- SFT data is super-additive: A-OKVQA alone moves MMBench 38.7 → 48.5; VQA-v2 alone moves POPE 76.0 → 82.0; combining all three reaches 52.1 / 82.7 / 86.3 / 1.8.
- **Fact-RLHF vs vanilla RLHF** (same SFT, 7B policy / 13B reward): LLaVA-Bench 93.4 → 94.1, MMHal 1.8 → 2.1. Vanilla RLHF improves LLaVA-Bench but **underperforms on MMHal** — the reward-hacking-by-length story.
- **Reward-model size matters**: 7B policy + 13B reward beats 7B policy + 7B reward on LLaVA-Bench (93.4 vs 87.8). Some of the "Fact-RLHF" gain is plausibly a bigger-judge effect.
- **Data filtering ≠ RLHF**: using the Fact-RLHF reward as a filter (top 30/50/70% of LLaVA data) gets LLaVA-Bench to ~81.2-81.8 but MMHal/POPE/MMBench essentially do not move — the gradient signal matters, not just data quality.

## Claims-vs-Evidence Audit

| Claim | Strength | Notes |
|-------|----------|-------|
| C1: First successful RLHF for LMMs | ⭐⭐⭐ | No comparable prior LMM-RLHF system at submission time. Well-supported as a "first". |
| C2: 95.6% of text-only GPT-4 on LLaVA-Bench (13B) | ⭐⭐ | One benchmark, single seed, GPT-4-judged — known optimism + verbosity bias. |
| C3: "+60% improvement on MMHal-Bench" | ⭐ | Loose phrasing. Relative to original LLaVA-13B (1.11 → 2.53) it is +128%; relative to InstructBLIP-13B / IDEFICS-80B it is only +20-25%. Single 96-item in-house benchmark. |
| C4: Fact-RLHF reduces reward hacking | ⭐⭐ | The A/B exists, but **"Fact-RLHF" bundles three changes** (factual context, symbolic reward, length penalty) and they are **never independently ablated**. The attribution to "factual augmentation" specifically is weaker than the abstract suggests. |
| C5: SFT+ closes most of the capability gap; RLHF adds alignment | ⭐⭐⭐ | Visible across MMBench/POPE; the paper is honest about alignment tax. |
| C6: MMHal-Bench agrees with humans (94%) | ⭐ | **Single pairwise check** (LLaVA-13B vs IDEFICS-80B). Generalisation to all other model comparisons — including LLaVA-RLHF's own outputs — is asserted, not measured. |
| C7: New SOTA at 7B | ⭐⭐ | Abstract's 52.4% MMBench does not match the 51.4 / 52.1 in Table 4. Minor inconsistency. |
| C8: 10K preferences are "data-efficient" | ⭐ | No scaling curve over 1K / 5K / 10K is shown. |

### Honest read

1. **Fact-RLHF bundles three changes** — factual context + symbolic reward + length penalty — and the paper never ablates them independently. The "factual augmentation" attribution in the title is therefore weaker than the abstract implies.
2. **POPE F1 actually drops** from SFT+ to RLHF at both scales (82.7 → 81.5 at 7B; 82.8 → 81.9 at 13B). On the cleaner third-party hallucination benchmark, RLHF *regresses*.
3. **LLaVA-RLHF-7B underperforms LLaVA-SFT+-7B on MMBench** (51.4 vs 52.1). The abstract's positive framing hides this regression.
4. **MMHal-Bench stacks three optimism biases**: (i) it is **adversarially filtered against the baseline LLaVA-13B**, (ii) it is **judged by GPT-4 without the image**, (iii) GPT-4 is given category names and a human reference answer as hints. The three biases compound.
5. **94% human agreement = a single pairwise check** (LLaVA-13B vs IDEFICS-80B), generalised — but not measured — for every other model pair, including LLaVA-RLHF itself.
6. **"+60% MMHal" is loose** — it is roughly +20-25% over InstructBLIP-13B / IDEFICS-80B, or +128% over original LLaVA-13B. Neither is "+60%" cleanly. The headline number is a function of which baseline you pick.

The strongest contributions remain (C1) doing RLHF on an LMM at all, (C5) the explicit alignment-tax framing, and the engineering recipe (LoRA-on-everything, 13B reward for a 7B policy). The hallucination headline numbers should be read with the audit above stapled to them.

## Limitations

**Authors admit**:

- Alignment tax at small (7B) scale.
- Short/evasive answers can game MMHal-Bench → recommend always pairing it with LLaVA-Bench.
- Linear projection and mixture ratios are not tuned; scaling to bigger models is unexplored.
- Pre-training alignment is left open; a separate "Honest" reward model (Llama-2 style) is future work.

**Not addressed**:

- No variance / seed sensitivity reported anywhere.
- No isolation ablation of the three Fact-RLHF ingredients.
- Reward-hacking is shown qualitatively (Fig 1c) but not quantitatively — no curve of reward-vs-human-judgment over PPO steps.
- No inter-annotator agreement for the 10K preferences.
- MMHal-Bench is built adversarially against LLaVA-13B; cross-architecture difficulty is not controlled.
- POPE regression from SFT+ to RLHF is glossed over.
- No held-out medical / domain-shift evaluation. The "factual" channel is COCO captions, so the factual-augmentation idea is only validated on natural-image distributions — its transfer to clinical VLMs (where structured findings or radiology reports could play the same role) is unverified.
- Using a 13B reward to align a 7B policy is a confound — some of the gain may come from a bigger judge, not from factual augmentation per se.

## Why it matters

Every downstream multimodal-alignment paper — RLHF-V, POVID, mDPO, Silkie, and the early clinical-VLM RLHF lines — cites LLaVA-RLHF as the parent. The two transferable ideas are:

1. **Inject oracle textual facts into the reward model.** In medical imaging, this maps cleanly to feeding radiology reports / structured findings / ICD codes into a clinical-VLM reward model, blocking reward hacking without iterative re-annotation.
2. **Run a larger reward model than the policy.** Cheap to do with LoRA; consistently lifts LLaVA-Bench in their ablation.

The two things to bring forward but verify yourself:

- Per-component ablation of factual context vs symbolic reward vs length penalty.
- Hallucination evaluation on a benchmark *not* adversarially built against the baseline you are improving.

## References

- Sun, Shen, Cao et al., "Aligning Large Multimodal Models with Factually Augmented RLHF", arXiv:2309.14525, 2023.
- Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023.
- Dubois et al., "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback", NeurIPS 2023.
- Li et al., "Evaluating Object Hallucination in Large Vision-Language Models" (POPE), EMNLP 2023.
- Liu et al., "MMBench: Is Your Multi-modal Model an All-around Player?", 2023.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023.
