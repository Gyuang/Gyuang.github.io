---
title: "Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization"
excerpt: "HA-DPO trains LVLMs on style-consistent positive/negative caption pairs and lifts MiniGPT-4 POPE-Random accuracy 51.13 → 86.13 with only 2K images, while diagnosing exactly how vanilla DPO collapses into a style classifier."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/ha-dpo/
tags:
  - HA-DPO
  - DPO
  - LVLM
  - Hallucination
  - Preference-Learning
  - Style-Consistency
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- **HA-DPO reframes LVLM hallucination as preference selection** and trains the model via DPO on (positive, negative) detailed-caption pairs derived from Visual Genome and audited by GPT-4. The headline number is **MiniGPT-4-LLaMA2-7B POPE-Random accuracy 51.13 → 86.13 (+35.0)** and **MME total 932.00 → 1326.46 (+42.32% relative)** with only **2K images / ~16K pairs**.
- The strongest contribution is the *diagnosis*, not the recipe: Eq. 5 decomposes the DPO gradient into a reward-weighting term and a `[∇log π(y_pos) − ∇log π(y_neg)]` term, and Fig. 4 (log-prob + n-gram diversity), Table 8, and Fig. 7 (reward margin) all show — independently — that vanilla DPO with mismatched (LVLM-style negative, GPT-style positive) pairs collapses into a style classifier rather than a faithfulness signal. The fix is to have GPT-4 paraphrase **both** polarities (3 rewrites each) so length, tone, and structure match and only factual content differs.
- The weakest evidence is what the paper does **not** test. HA-DPO is never compared to vanilla DPO with a length penalty or length-normalized reward, even though the central thesis is "DPO is taking a length/style shortcut." The MiniGPT-4 +35 also rides a near-degenerate 98.66% Yes-Ratio baseline, gains on the already-strong LLaVA-1.5 are small (+0.93 POPE-Random), and **LLaVA-1.5 MME actually regresses** (Perception −8, Cognition −42) — directly undermining the "general capability improves" framing.

## Motivation

Mainstream LVLM hallucination mitigation has two failure modes. SFT-based methods (LRV, InstructBLIP, LLaVA-RLHF) need hundreds of thousands of annotated examples and still cap performance; post-hoc tool pipelines (Woodpecker) are bottlenecked by an external detector or tokenizer. HA-DPO instead targets a lightweight, data-cheap *preference-time* fix. The medical motivation is named explicitly in the introduction — a hallucinated medical caption "could lead to a misdiagnosis" — though, as we will see in Section *Limitations*, no medical-domain dataset is actually evaluated.

DPO is chosen over RLHF for its lack of an explicit reward model, but the authors immediately surface a problem that vanilla DPO ignores in the multimodal setting: the *style* of preferred vs. dispreferred responses in naturally collected pairs is systematically different (raw LVLM output vs. GPT-4 rewrite). Plain DPO therefore learns "sound like GPT-4" rather than "be faithful to the image." Every method choice that follows is in service of removing that style confound.

![HA-DPO pipeline](/assets/images/paper/ha-dpo/page_003.png)
*Figure 1: The four-stage HA-DPO pipeline — description generation → GPT-4 hallucination detection & correction → style-consistency augmentation → DPO training.*

## Core Innovation

- **Style-consistent pair construction.** After GPT-4 detects hallucinated sentences in LVLM-generated captions and rewrites them into corrected versions, GPT-4 is invoked a second time to **also rewrite the negative sample** and produce **3 paraphrases of each polarity**. Both sides now share GPT-4 prose with identical structure, length, and tone; only the factual claims differ.
- **Gradient-level justification for the fix.** Eq. 5 in the paper shows the DPO gradient decomposes into a reward-weighting term and a `[∇log π(y_pos) − ∇log π(y_neg)]` term. When the two sides come from different distributions, the second term dominates and the weighting factor cannot stabilize training — providing a *causal* explanation for the empirical instability that other DPO-for-LVLM papers report anecdotally.
- **Descriptive → QA recast and auxiliary SFT loss.** Descriptive pairs are reformatted into QA pairs (broadens the supervision signal beyond long-form generation), and an auxiliary SFT loss `L = L_dpo + λ L_aux` is added to stabilize training (λ = 0.5 for MiniGPT-4; λ = 0 for InstructBLIP and LLaVA-1.5).
- **Tiny data footprint.** 2,000 Visual Genome images → ~6K positive + 6K negative descriptive samples + ~10K QA pairs ≈ 16K preference pairs, trained via LoRA in ≤1-2 hours on 8×A100.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Reframing hallucination as preference learning via DPO meaningfully reduces hallucination across multiple LVLMs. | Tables 2, 3 across MiniGPT-4 / InstructBLIP / LLaVA-1.5. | POPE, SHR | ⭐⭐⭐ |
| C2 | Style-consistent pair construction is *necessary*; without it, DPO learns style not faithfulness and destabilizes. | Eq. 5 derivation, Fig. 4 (log-prob & n-gram), Fig. 7 (reward margin), Table 8 (n-gram diversity across β). | self-constructed | ⭐⭐⭐ |
| C3 | HA-DPO outperforms prior hallucination-mitigation methods (LRV 400K, LLaVA-RLHF 160K) at a fraction of the data. | Table 7 — beats both across all POPE splits. | POPE | ⭐⭐ (single benchmark, no variance reported) |
| C4 | Removing hallucinations *improves* general capability, not just safety/factuality. | Table 4 MME — MiniGPT-4 Perception +358, InstructBLIP +71. | MME | ⭐⭐ (LLaVA-1.5 *regresses*: Perception −8, Cognition −42) |
| C5 | SHR is a more reliable hallucination metric than POPE/HaELM. | Table 6: GPT-4 judgment accuracy 88.30 → 95.84% with factual augmentation; 200 VG images, ~250 sentences hand-audited. | self-constructed | ⭐⭐ (only 20 audited images; circular — SHR uses GPT-4, model trains on GPT-4 rewrites) |
| C6 | Headline +35% POPE-Random / +42.32% MME on MiniGPT-4. | Tables 2, 4. | POPE, MME | ⭐⭐⭐ for the number itself; ⭐ for generalizability (degenerate yes-saying baseline) |
| C7 | HA-DPO's style control does work that vanilla DPO + length penalty cannot. | Implicit; the paper compares only against DPO-without-style-control, **never** against length-normalized DPO. | — | ⭐ |

**Honest take.** The shortcut-diagnosis evidence (C2) is the strongest part of the paper. Eq. 5 supplies a clean gradient-decomposition argument for *why* style mismatch hijacks DPO, and Fig. 4 / Table 8 / Fig. 7 then provide three independent empirical signals — log-prob distribution overlap, n-gram diversity, and gradient/reward-margin smoothness — that all move in the same direction once style control is added. This level of triangulation is unusual for an alignment paper and is the part of HA-DPO most likely to outlive the specific recipe.

C1 and C6 (headline POPE/MME gains) are well-supported numerically but partially confounded by baseline quality. MiniGPT-4-LLaMA2-7B at POPE-Random has **98.66% Yes-Ratio** with ~51% accuracy — i.e., it is a near-degenerate "always answer Yes" classifier — so a substantial chunk of the "+35%" comes from simply teaching the model to ever say "No." Gains on the already-strong LLaVA-1.5 are much more modest (POPE +0.93 / +1.70 / +1.70 across Random / Popular / Adv; SHR −2.7), and **MME actually regresses on LLaVA-1.5** (Perception 1510.74 → 1502.58, Cognition 355.71 → 313.93) — directly undermining C4's framing that HA-DPO uniformly improves general capability.

C7 is the cleanest miss. The paper's whole story is that DPO takes a length/style shortcut, but the only ablation against this shortcut is HA-DPO vs. HA-DPO-without-style-control (Table 8) — which removes only the rewrite step while keeping a GPT-style positive vs. LVLM-style negative, *exactly* the length/style confound the paper warns about. A vanilla-DPO + length-penalty (or length-normalized reward, as in subsequent R-DPO / LongDPO work) baseline is never run. Without it, we cannot conclude that GPT-4 rewriting is doing more work than a much simpler length-balancing fix would.

C5 has a circularity problem: the model is trained on GPT-4 rewrites of VG annotations and graded by GPT-4 using VG annotations. The 95.84% GPT-4 judgment accuracy is itself measured on only 20 images / ~250 sentences, with no inter-annotator agreement and no human-only re-evaluation of the final SHR drop. Variance is not reported for any table.

## Method & Architecture

*Figure 2: HA-DPO pipeline. (1) Generate detailed captions from the target LVLM with `num_beams=5`, `temperature=1.0`, `do_sample=False`. (2) Feed VG region captions plus the LVLM caption to GPT-4 for per-sentence judgment (hallucination / correct / cannot-judge) and a rewrite of any hallucinated sentence. (3) Style-consistency augmentation — GPT-4 rewrites the negative as well, producing 3 paraphrases of each polarity in matched style. (4) DPO training with the standard objective plus an auxiliary SFT loss.*

The DPO objective is the standard one, conditioned on text and image inputs `[x_T, x_I]`:

$$\mathcal{L}_{dpo}= -\mathbb{E}_{(x_T,x_I,y_{pos},y_{neg})\sim D}\!\left[\log\sigma\!\left(\beta\log\tfrac{\pi_\theta(y_{pos}|[x_T,x_I])}{\pi_{ref}(y_{pos}|[x_T,x_I])} - \beta\log\tfrac{\pi_\theta(y_{neg}|[x_T,x_I])}{\pi_{ref}(y_{neg}|[x_T,x_I])}\right)\right].$$

The full training objective adds an auxiliary SFT loss on the original instruction-tuning data, `L = L_dpo + λ L_aux`, to keep the policy anchored to its instruction-following behavior.

![Why style mismatch breaks DPO](/assets/images/paper/ha-dpo/page_004.png)
*Figure 3: Why naive (LVLM-negative, GPT-positive) pairs send DPO down a shortcut. Cases (a)–(c) walk from raw, style-confounded pairs through partial rewrites to the fully style-matched pairs HA-DPO uses; only in (c) does the residual signal isolate factuality.*

The gradient decomposition (Eq. 5) is the formal version of the same picture: when `y_pos` and `y_neg` are drawn from different distributions, the `[∇log π(y_pos) − ∇log π(y_neg)]` term dwarfs the reward-weighting term, and the optimization tilts toward style separation rather than faithfulness. Empirically this is visible as the log-prob distributions for positives and negatives sliding apart (and the model's 4-gram diversity collapsing) without style control.

![Quantitative effect of style control](/assets/images/paper/ha-dpo/page_005.png)
*Figure 4: Without style-consistency control, positive and negative log-probabilities diverge and 4-gram diversity collapses; with style control, the distributions overlap cleanly and lexical diversity is preserved.*

**Training setup.** LoRA fine-tuning on `q_proj, k_proj, v_proj` for MiniGPT-4 (rank 64, α 16) and InstructBLIP-13B (rank 64, α 16); all linear layers for LLaVA-1.5-7B (rank 256, α 128). `β = 0.1` is the best setting across Tables 1 & 8 for all three models; β = 0.6 is the SHR sweet spot in the MiniGPT-4 ablation. Total compute is ≤1-2 hours on 8×A100 per model.

## Experimental Results

### POPE accuracy / F1 (Tables 2, 7)

| Model | Setting | Acc (Random) | F1 (Random) | Acc (Popular) | F1 (Popular) | Acc (Adv) | F1 (Adv) |
|---|---|---|---|---|---|---|---|
| MiniGPT-4-LLaMA2-7B | base | 51.13 | 67.13 | 51.46 | 67.72 | 51.26 | 67.16 |
| **MiniGPT-4-LLaMA2-7B** | **+ HA-DPO** | **86.13** | **84.96** | **79.50** | **79.25** | **75.66** | **76.29** |
| InstructBLIP-13B | base | 88.70 | 89.26 | 81.36 | 83.44 | 74.50 | 78.64 |
| **InstructBLIP-13B** | **+ HA-DPO** | **89.83** | **89.43** | **85.76** | **85.80** | **80.70** | **81.68** |
| LLaVA-1.5-7B | base | 89.60 | 89.70 | 86.20 | 86.79 | 79.76 | 81.75 |
| **LLaVA-1.5-7B** | **+ HA-DPO** | **90.53** | **90.25** | **87.90** | **87.81** | **81.46** | **82.54** |
| LRV (400K data) | — | 86.00 | 88.00 | 73.00 | 79.00 | 65.00 | 73.00 |
| LLaVA-RLHF (160K data) | — | 84.80 | 83.30 | 83.90 | 81.80 | 82.30 | 80.50 |

### SHR (Table 3) — lower is better

| Model | base SHR | +HA-DPO SHR | Δ |
|---|---|---|---|
| **MiniGPT-4-LLaMA2-7B** | 47.3 | **44.4** | **−2.9** |
| **InstructBLIP-13B** | 51.2 | **49.1** | **−2.1** |
| **LLaVA-1.5-7B** | 36.7 | **34.0** | **−2.7** |

### MME (Table 4)

| Model | Perception (base → +HA-DPO) | Cognition (base → +HA-DPO) |
|---|---|---|
| **MiniGPT-4-LLaMA2-7B** | 733.79 → **1092.18** | 198.21 → **234.28** |
| **InstructBLIP-13B** | 1344.91 → **1416.23** | 232.50 → 233.21 |
| **LLaVA-1.5-7B** | **1510.74** → 1502.58 | **355.71** → 313.93 |

Two patterns matter. First, the headline +35 POPE-Random on MiniGPT-4 is partly a story about *how broken the base model was* — it answered "Yes" 98.66% of the time. Second, the MME line on LLaVA-1.5 quietly contradicts the "improves general capability" framing: Perception drops 8 points and Cognition drops 42 points after HA-DPO. The paper mentions this in passing and does not analyze it; the most natural reading is that HA-DPO can erode capability on already-aligned models.

### Ablations

Table 1 sweeps β ∈ {0.3, 0.4, 0.5, 0.6, 0.8, 1.0} on MiniGPT-4 SHR. Too-small β destabilizes training (β = 0.3: SHR 57.2, 4-gram 87.9); too-large β over-constrains the policy (β = 1.0: SHR 56.7, 4-gram 94.6). β = 0.6 is the sweet spot (SHR 51.4).

Table 8 ablates style consistency at multiple β values: without style control, 1-gram diversity collapses to 17.9-57.2; with style control, it stays at 56.8-60.1 — close to the MiniGPT-4 reference (60.0). Figure 7 confirms the picture in gradient space: the reward margin remains smoothly bounded with style-consistent data and spikes wildly without it.

![SHR hallucination categories](/assets/images/paper/ha-dpo/page_006.png)
*Figure 5: SHR covers six hallucination categories — movement, emotion, spatial relation, nonexistent object, object attribute, physical state — broader than POPE's object-existence-only scope.*

### Qualitative

![Qualitative before/after](/assets/images/paper/ha-dpo/page_008.png)
*Figure 6: Baseline vs. HA-DPO outputs. Pre-HA-DPO outputs invent "a large stadium with seats and a scoreboard" or "a small island in the middle of the pool"; post-HA-DPO outputs drop those fabrications while keeping correct grounded content.*

## Limitations

**Authors admit.** SHR remains high after HA-DPO (LVLM hallucination is "still severe"); future work is needed to extend to real-world / specialized scenarios.

**Not addressed by the authors.**

- **No length-balanced DPO baseline.** The central thesis is that DPO is taking a length/style shortcut, yet HA-DPO is never compared to DPO with a length penalty or length-normalized reward. Without this experiment we cannot attribute the gains to *style* control specifically vs. a simpler length-balancing fix.
- **Degenerate baseline inflates MiniGPT-4 headline.** MiniGPT-4 POPE-Random base accuracy is 51.13% with 98.66% Yes-Ratio — a near-degenerate yes-saying classifier. A meaningful fraction of "+35" is teaching the model to ever say "No."
- **Tiny gains on strong models.** LLaVA-1.5 POPE-Random improves only +0.93; SHR drops are 2-3 points across the board with no variance / significance test.
- **LLaVA-1.5 MME regression.** Perception 1510.74 → 1502.58 (−8) and Cognition 355.71 → 313.93 (−42) directly contradict the "general capability improves" framing and are not analyzed.
- **Oracle / metric circularity.** The training data, the SHR judge, and a substantial part of the supervision share a single closed-source oracle (GPT-4). The 95.84% GPT-4-judgment accuracy is itself measured on only 20 images / ~250 sentences; no inter-annotator agreement is reported.
- **No external held-out benchmark.** AMBER, HallusionBench, and MMHal are not used; POPE / SHR / MME share authors, oracle, or image domain.
- **Cost transparency.** The paper celebrates "2K images" but the construction relies on heavy GPT-4 calls — three rewrites per sample over ~6K samples plus per-sentence judgments — which is not free in dollars or in dependency on a single closed model.
- **No medical-domain evaluation** despite the explicit medical-AI framing in the introduction.

## Why It Matters for Medical AI

For clinical deployment of LVLMs the issue is not whether a model can produce a fluent radiology report — it is whether the report invents findings that change patient management. HA-DPO is interesting to medical AI for two reasons. First, the diagnostic part of the paper (Eq. 5 + Fig. 4 + Table 8) is *transferable*: any team building a medical-LVLM alignment pipeline with DPO needs to know that vanilla DPO over (specialist-LVLM, expert-rewritten) pairs will likely learn "sound like the expert" rather than "be faithful to the scan," and the paper supplies a concrete protocol — paraphrase both polarities into matched style — to defuse that shortcut.

Second, the *practical* claim that hallucination can be cut with 2K images and ≤2 hours of LoRA training is exactly the budget a hospital research group might have. The caveats, however, port directly: a degenerate yes-saying baseline behaves very differently from a strong medical foundation model already tuned on CheXpert or MIMIC-CXR, and the MME regression on LLaVA-1.5 is a warning that HA-DPO can erode general capability on already-strong backbones. No medical dataset is evaluated in the paper itself, so anyone applying HA-DPO clinically should re-run the style-vs-length ablation on their own domain before trusting the recipe.

## References

- **Paper.** Zhao, Wang, Ouyang et al., *Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization*, CVPR 2025. arXiv: [2311.16839](https://arxiv.org/abs/2311.16839).
- **Code & data.** [opendatalab.github.io/HA-DPO](https://opendatalab.github.io/HA-DPO).
- **Related work.**
  - Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023 — the DPO objective HA-DPO instantiates.
  - Li et al., *Evaluating Object Hallucination in Large Vision-Language Models* (POPE), 2023.
  - Sun et al., *LLaVA-RLHF*, 2023 — RLHF baseline cited in Table 7.
  - Liu et al., *LRV-Instruction*, 2023 — large-scale SFT baseline cited in Table 7.
  - Yin et al., *Woodpecker*, 2023 — post-hoc tool-based hallucination correction.
