---
title: "Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment"
excerpt: "HALVA replaces sequence-level DPO with a phrase-level alignment loss on generatively augmented hallucination pairs, lifting AMBER discriminative F1 by 13.4 points (73.1 -> 86.5 on the 13B model) while keeping VQA-v2, MME and MM-Vet flat where HA-DPO and EOS regress."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/halva/
tags:
  - HALVA
  - DPA
  - MLLM
  - Hallucination
  - Preference-Alignment
  - LLaVA
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- **Problem and method.** Sequence-level preference methods (HA-DPO, EOS, RLHF-V) reduce MLLM object hallucination but drift the model away from its base behaviour and degrade general capabilities. HALVA replaces that with a *phrase-level* contrastive loss (DPA) that penalises only the hallucinated phrase spans inside a generatively augmented `{y_c, y_h}` pair, plus a token-wise *forward*-KL anchor against a frozen reference copy of the base MLLM.
- **Key innovation.** Visual Genome images are paired with Gemini Vision Pro captions (`y_c`); a closed-set co-occurrence pool and an open-set LLM pool generate minimally-edited hallucinated counterparts (`y_h`) by swapping objects, attributes, actions, or locations. The alignment term is computed *only* over the swapped phrase spans — unchanged scaffold tokens contribute zero gradient. Final training set is 21.5K pairs, no reward model, 1 epoch / 342 steps on 4xA100.
- **Headline result.** **AMBER discriminative F1 +13.4** (73.1 -> 86.5, HALVA-13B vs LLaVA-v1.5-13B), **AMBER hallucination rate -4.2** (36.4 -> 32.2, 7B), **CHAIR_s -8.6** (50.0 -> 41.4, 7B), **MME-Hall +31.7** (13B) — while VQA-v2 / TextVQA / MME / MM-Vet stay flat or improve, the regime where HA-DPO and EOS show statistically significant regressions.

## Motivation

The paper's framing is that object hallucination is a **local** failure mode: a specific phrase (`tooth-pick` near utensils, `tie` near a wedding cake) is wrong while the surrounding caption is fine. Existing finetuning fixes apply *sequence-level* losses to fix a *phrase-level* problem and consequently drift the model away from its base behaviour. Figure 2B in the paper makes this concrete: HA-DPO and EOS sit at high KL from LLaVA-v1.5 and simultaneously lose general-task accuracy, while their hallucination wins are not large enough to justify the regression.

A second, blog-relevant motivation is named explicitly: the authors argue detail preservation matters "particularly for tasks that require detailedness such as medical imaging analysis," citing ChatCAD and related work. That framing is rhetorical — **no medical benchmark is evaluated** — but it explains why caption-length collapse (EOS shrinks captions from 100.6 -> 79.7 tokens) is treated as a failure mode rather than as a feature.

![Figure 1](/assets/images/paper/halva/page_001.png)
*Figure 1: LLaVA-v1.5 hallucinates objects driven by co-occurrence statistics (tooth-pick next to a fork-and-knife setting, tie near a wedding cake); HALVA, finetuned with phrase-level alignment, does not.*

## Core Innovation

- **Phrase-level alignment loss `L_a`.** A Bradley-Terry-style log-softmax between the *phrase-span* probabilities of `y_c` (correct) and `y_h` (hallucinated). The sum runs only over the swapped phrase spans `[s_i^h, e_i^h]`; the unchanged scaffold ("A young man in a ... is ... in a ...") contributes zero gradient. This is the structural departure from DPO, which marginalises over the full sequence.
- **Token-wise forward-KL anchor `L_d`.** A frozen reference copy `pi_ref` of the *same* base MLLM acts as the anchor: `L_d = sum_j pi_ref(t_j) * [log pi_ref(t_j) - log pi_theta(t_j)]`. The KL direction is *forward* (`KL(pi_ref || pi_theta)`), not reverse — justified by the absence of on-policy rollouts and motivated by avoiding the mode-seeking / diversity collapse that reverse KL is known to induce.
- **Generative data augmentation with two negative pools.** Source images are from Visual Genome (108K images, dense annotations). Negatives are drawn from `O_cc` (closed-set, co-occurrence-driven — weaponising the very statistics that *cause* hallucination) and `O_oc` (open-set, LLM-generated). Yes/No samples use a direct `Yes <-> No` flip to target the positive-instruction bias documented in Liu 2023b / Bai 2024.
- **No reward model, no on-policy rollouts.** 21.5K pairs trained for 1 epoch with LoRA on the LLM, vision encoder and projector frozen, 1.5-3 hours on 4xA100.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets / Tables | Strength |
|---|---|---|---|---|
| C1 | Phrase-level alignment beats sequence-level (DPO-style) alignment for object hallucination | HALVA-7B vs HA-DPO-7B / EOS-7B on MME-Hall (665 vs 618.3 / 606.7), AMBER F1 (83.4 vs 78.1 / 75.6), MMHal score (2.25 vs 1.97 / 2.03) | Tables 1-5 (7B head-to-head) | ⭐⭐⭐ — wide benchmark coverage, same base model |
| C2 | DPA preserves general vision-language capability | Table 6: HA-DPO-7B and EOS-7B show statistically significant drops on VQA-v2 / TextVQA / MME; HALVA-7B is flat or improving across all five general benchmarks; pattern replicates at 13B and VILA-13B/384 | Table 6, Figure 2B | ⭐⭐⭐ — directly measured, significance reported, three base models |
| C3 | DPA reduces hallucination rate by up to 4.2% | AMBER hallucination rate 36.4 -> 32.2 for HALVA-7B = 4.2 points | Table 3 | ⭐⭐⭐ — matches the abstract number exactly |
| C4 | DPA improves hallucination-VQA F1 by up to 13.4% | AMBER discriminative F1 73.1 -> 86.5 for HALVA-13B = 13.4 points | Table 3 | ⭐⭐⭐ — matches the abstract number exactly |
| C5 | Forward-KL anchor is essential (not just any KL) | Theoretical argument (no rollouts, avoids mode-seeking); alpha sweep in Figure 5 controls divergence/alignment trade-off | Figure 5, Section 2 | ⭐⭐ — **alpha is ablated, but forward-KL vs reverse-KL is not directly compared in the main body**. The empirical separation of "phrase-level loss contribution" vs "forward-KL contribution" is not isolated |
| C6 | Phrase-level localisation (not just better data) is what causes the win | The head-to-head with HA-DPO is not a clean ablation because HA-DPO uses different training data. The cleanest comparison — sequence-level DPO on the *same* `{y_c, y_h}` pairs — is pointed to **Appendix C only** | Main body inconclusive; Appendix C cited | ⭐⭐ — readers should treat this as appendix-grade evidence, not main-body evidence |
| C7 | DPA transfers to non-object hallucinations (visual illusions, charts) | HallusionBench: +1.86 / +2.60 / +1.21 over base for 7B / 13B / VILA-13B/384 | Table 5 | ⭐⭐ — small effect sizes, single benchmark; authors themselves frame this as "curiosity driven" |
| C8 | Method generalises across base architectures | LLaVA-v1.5-7B (CLIP-ViT-L/14), LLaVA-v1.5-13B, VILA-v1.5-13B/384 (SigLIP-L-400M). Pattern holds for all three | All tables | ⭐⭐⭐ — three bases, two vision-encoder families |
| C9 | "Up to 13.4% F1" generalises to hallucination-VQA broadly | The 13.4 number is specifically AMBER discriminative F1, not MMHal or MME-Hall. The abstract phrasing slightly conflates AMBER F1 with hallucination-VQA in general | Table 3 only | ⭐⭐ — technically correct but reader may over-read |
| C10 | Method is more data-efficient than pretraining alternatives (>500K -> 21.5K) | LRV (Liu 2023b) uses >400K pretraining negatives; HALVA uses 21.5K finetuning pairs | Section 3 prose | ⭐⭐ — argued, not ablated like-for-like |
| C11 | HALVA matches GPT-4V on discriminative tasks | HALVA-13B AMBER F1 = 86.5; HALVA-13B/384 = 87.9; GPT-4V = 87.4 | Table 3 | ⭐⭐ — only on AMBER F1; on AMBER hallucination rate (54.2 vs 30.7) and MMHal score (2.58 vs 3.49), GPT-4V remains substantially ahead |

**Honest read.** The headline claims (C1-C4, C8) are well supported — three base models, five benchmarks each, three seeds with std reported. The genuinely novel scientific point is C2: HALVA is the first method (that I am aware of) that pins down *where* DPO-on-MLLM-hallucination breaks (Figure 2B's KL plot) and shows numbers that do not silently regress in the places competitors do.

The two weaknesses worth flagging up-front are C5 and C6. The "DPA" recipe bundles two design choices — *phrase-level loss* and *forward-KL anchor with a frozen reference* — and the main-body ablations only sweep alpha (the mixing coefficient), not the loss formulation itself. The cleanest possible ablation — same `{y_c, y_h}` data, but with sequence-level DPO or IPO — lives in **Appendix C**, not the main body. Readers should be aware that the phrase-level-vs-sequence-level claim is appendix-grade evidence in the published version, and that **forward vs reverse KL is not directly ablated anywhere**. The theoretical argument is plausible but the empirical separation of contributions is not isolated.

C7 is honestly framed by the authors as a "curiosity-driven" probe and the deltas (1-3 points) are modest — no overclaiming there. C11's "GPT-4V parity" only holds on AMBER discriminative F1; on AMBER hallucination rate and MMHal score, GPT-4V is still well ahead.

## Method & Architecture

![Figure 4](/assets/images/paper/halva/page_004.png)
*Figure 4: DPA training loop. The alignment loss L_a penalises hallucinated phrase spans relative to their correct counterparts under the trainable policy pi_theta; a forward-KL anchor L_d against a frozen reference copy pi_ref of the same base MLLM prevents drift away from base capability.*

**Generative data augmentation (the negative-sample generator).** Visual Genome images feed Gemini Vision Pro, which is prompted to produce a descriptive caption `y_c` ("Provide a one-sentence caption", "Describe the image in detail") or to pull non-descriptive Yes/No pairs from HA-DPO's data. The hallucinated counterpart `y_h` is constructed by swapping ground-truth concepts in `y_c` with concepts that are *not* present in the image. Swap categories: **objects, attributes, actions, locations**. The negative pool decomposes into `O_cc` (closed-set, drawn from co-occurrence statistics in an object-centric corpus — exactly the patterns that cause hallucinations) and `O_oc` (open-set, drawn by prompting an LLM directly). For Yes/No samples, `y_h` is simply the inverted answer, directly targeting positive-instruction bias.

![Figure 3](/assets/images/paper/halva/fig_p003_01.png)
*Figure 3: Generative data augmentation. The correct caption (man / white shirt / blue jeans / skateboarding / skate park) is rewritten into a hallucinated variant (woman / black dress / red sneakers / rollerblading / roller rink) with phrase-level swaps across object, attribute, action, and location categories.*

**Phrase-level alignment loss.** For each hallucinated span at token indices `[s_i^h, e_i^h]` and its aligned correct span at `[s_i^c, e_i^c]`:

$$
L_a = -\frac{1}{N} \sum_i \log \frac{\prod_j \pi_\theta(t_j^c \mid x, t_{<j}^c)}{\prod_j \pi_\theta(t_j^c \mid \cdot) + \prod_j \pi_\theta(t_j^h \mid \cdot)}
$$

This is a Bradley-Terry-style log-softmax between the *phrase-span* probabilities. Crucially, the sum runs only over hallucinated phrases, not all tokens. Unchanged scaffold tokens contribute zero gradient — the structural departure from DPO.

**Forward-KL anchor.** Using a frozen reference copy `pi_ref` of the same base MLLM and any reference sample `{x_r, y_r}`:

$$
L_d = \sum_j \pi_\mathrm{ref}(t_j^r \mid \cdot) \cdot \left[ \log \pi_\mathrm{ref}(t_j^r \mid \cdot) - \log \pi_\theta(t_j^r \mid \cdot) \right]
$$

This is `KL(pi_ref || pi_theta)`, the **forward** direction. The justification is that no on-policy rollouts of `pi_theta` are needed — the loss only sees `pi_ref`'s tokens — and forward KL avoids the mode-seeking / diversity collapse that reverse KL induces. Whether forward-vs-reverse KL is *empirically* what matters is, as flagged above, not directly ablated.

**Total objective.** `L_DPA = L_a + alpha * L_d`, with alpha = 0.4 chosen as the elbow of the alpha sweep in Figure 5 (Left): alpha = 0.01 lets the model drift far from base; alpha >= 1 over-anchors and `L_a` barely moves.

**Training setup.** Bases: LLaVA-v1.5-7B, LLaVA-v1.5-13B, VILA-v1.5-13B/384. Only LLM weights are trained via LoRA; vision encoder and projector are frozen. Effective batch 64, 1 epoch / 342 steps on 4xA100-80GB, 1.5-3 hours wall-clock. Each benchmark is evaluated 3x and averaged; GPT-4-as-judge runs additionally report std.

![Figure 2](/assets/images/paper/halva/page_002.png)
*Figure 2: The paper-defining trade-off. HALVA reduces object hallucination without diverging from the base MLLM (right panel: KL divergence from base), unlike HA-DPO and EOS which sit at high KL while simultaneously regressing on general benchmarks.*

## Experimental Results

Numbers below are exactly as reported. Bold marks HALVA variants — the paper's own method.

| Benchmark | Metric | LLaVA-v1.5-7B | **HALVA-7B** | HA-DPO-7B | EOS-7B | LLaVA-v1.5-13B | **HALVA-13B** | VILA-v1.5-13B/384 | **HALVA-13B/384** | GPT-4V |
|---|---|---|---|---|---|---|---|---|---|---|
| CHAIR (Table 1) | C_i ↓ | 15.4 | **11.7** | 11.0 | 12.3 | 13.0 | **12.8** | 9.2 | **8.4** | — |
| CHAIR | C_s ↓ | 50.0 | **41.4** | 38.2 | 40.2 | 47.2 | **45.4** | 33.0 | **30.0** | — |
| CHAIR | Length | 100.6 | **92.2** | 91.0 | 79.7 | 100.9 | **98.0** | 183.4 | **182.6** | — |
| MME-Hall (Table 2) | total ↑ (max 800) | 648.3 | **665.0** | 618.3 | 606.7 | 643.3 | **675.0** | 688.3 | **691.7** | — |
| AMBER (Table 3) | CHAIR ↓ | 7.8 | **6.6** | 6.7 | 5.1 | 6.6 | **6.4** | 9.9 | **9.1** | 4.6 |
| AMBER | Coverage ↑ | 51.0 | **53.0** | 49.8 | 49.1 | 51.9 | **52.6** | 63.3 | **63.9** | 67.1 |
| AMBER | Hall. rate ↓ | 36.4 | **32.2** | 30.9 | 22.7 | 30.5 | **30.4** | 56.1 | **54.2** | 30.7 |
| AMBER | F1 ↑ | 74.7 | **83.4** | 78.1 | 75.6 | 73.1 | **86.5** | 82.2 | **87.9** | 87.4 |
| MMHal (Table 4) | Score ↑ (0-6) | 2.11±0.05 | **2.25±0.09** | 1.97 | 2.03 | 2.37±0.02 | **2.58±0.07** | 2.58±0.02 | **2.58±0.06** | 3.49 |
| MMHal | Hall. rate ↓ | 0.54±0.01 | **0.54±0.01** | 0.60 | 0.59 | 0.50±0.00 | **0.45±0.02** | 0.46±0.01 | **0.45±0.01** | 0.28 |
| HallusionBench (Table 5) | Acc ↑ | 47.09±0.14 | **48.95±0.13** | 48.36 | 48.72 | 46.50±0.09 | **49.10±0.05** | 55.39±0.05 | **56.60±0.18** | 65.28 |
| HallusionBench | Yes/No bias (~0) | 0.31 | **0.17** | 0.26 | 0.29 | 0.38 | **0.20** | 0.19 | **0.02** | 0.06 |
| MME ↑ (Table 6) | | 1510.7 | **1527.0** | 1502.6 | 1424.4 (sig.) | 1530.1 | **1544.0** | 1569.6 | **1575.7** | — |
| VQA-v2 ↑ | | 78.5 | **78.5** | 77.6 (sig.) | 77.6 (sig.) | 80.0 | **80.0** | 82.8 | **82.8** | — |
| TextVQA ↑ | | 58.3 | **58.2** | 56.7 | 55.2 (sig.) | 61.2 | **61.2** | 65.0 | **64.8** | — |
| MM-Vet ↑ | | 31.1 | **32.1** | 30.7 | 31.4 | 36.1 | **37.8** | 44.3 | **44.3** | — |
| LLaVA-BW ↑ | | 65.4 | **67.2** | 66.2 | 65.8 | 72.5 | **72.7** | 80.8 | **82.4** | — |

The most informative reading is *vertical* — pairing each base with its HALVA variant — plus the side-by-side with HA-DPO-7B and EOS-7B that share LLaVA-v1.5-7B as base. HALVA never wins on every single hallucination metric (EOS-7B is the lowest on AMBER hallucination rate, 22.7), but it is the only method whose general-capability columns (VQA-v2 / TextVQA / MME) do not regress. EOS achieves its low hallucination rate by compressing captions from 100.6 -> 79.7 tokens; HALVA stops at 92.2 while AMBER coverage rises (53.0 vs EOS 49.1 vs base 51.0), which is the strongest argument that DPA is not colluding with degeneration.

The Yes/No bias result on HallusionBench is striking in isolation: HALVA-13B/384 drops the bias to 0.02 (essentially eliminated) from a base of 0.19, with the `Yes <-> No` augmentation directly targeting it.

![Figure 5 + 6](/assets/images/paper/halva/page_009.png)
*Figure 5 (left): alpha sweep. alpha = 0.4 is the elbow of the divergence/alignment trade-off; the L_a histogram (right of Fig. 5) shifts from a wide pre-training distribution centred near 0.6 to a sharp peak near 0.1 after training. Figure 6: qualitative four-panel — detail description, VQA, Yes/No answers, and visual illusion. The visual-illusion panel is where HALVA's transfer is real but small.*

## Limitations

**Acknowledged by the authors.**
- CHAIR is a limited metric (no object coverage, no detailedness signal) — they pivot to AMBER for richer measurement.
- HALVA is "not explicitly trained" for visual-illusion hallucination; Figure 6D shows partial success only.
- GPT-4-as-judge variance is reported on MMHal but not eliminated.

**Open questions (analyst's read).**
- *Forward KL vs reverse KL is not directly ablated.* The theoretical argument is plausible (no rollouts; avoid mode-seeking) but the empirical separation between "phrase-level loss" and "forward KL anchor" contributions is not isolated in the main body.
- *Phrase-level vs sequence-level DPO on the **same** `{y_c, y_h}` data is in Appendix C only.* The cleanest ablation that would prove phrase-level localisation (not data quality) is what drives the win lives outside the main body. Readers should treat the phrase-vs-sequence claim as appendix-grade evidence in the published version.
- *Does the win survive at >13B scale?* All experiments stop at 13B.
- *Does it transfer to MLLMs whose hallucinations are not driven by co-occurrence statistics?* Training negatives are constructed *from* co-occurrence — possibly self-limiting on OCR, chart, or medical-CT models.
- *Closed-generator dependence.* The augmentation pipeline requires Gemini Vision Pro for `y_c`; the released 21.5K pairs preserve downstream reproducibility, but extending the data needs API access.
- *No medical / out-of-distribution evaluation.* Despite the medical-imaging framing in the intro, no medical benchmark is run; everything is COCO/VG-adjacent imagery.
- *Silent label noise.* The pipeline assumes the LLM-generated `y_h` is genuinely absent from the image; false negatives (a "hallucinated" object that is actually present) are not measured.

## Why It Matters for Medical AI

The medical-AI framing in the introduction is honest about being aspirational: no medical benchmark is evaluated. What does transfer cleanly to medical MLLMs is the **scientific point**, not the model — that *sequence-level* preference alignment trades hallucination reduction for descriptive-detail collapse, and that *phrase-level* alignment plus a forward-KL anchor can avoid that trade-off on natural-image benchmarks. For medical reports where descriptive completeness is not optional (omitting an incidental finding is itself a clinical error), the EOS-style "stop early before hallucinating" recipe is a non-starter; HALVA's preservation of caption length and coverage is the property that would actually matter clinically.

The honest caveat is that medical hallucinations may not be **driven by co-occurrence in the first place**. HALVA's negatives are constructed from natural-image co-occurrence statistics (tooth-pick next to fork-and-knife); a chest X-ray MLLM hallucinating a pleural effusion that is not present is a different generative process. Whether the DPA loss formulation transfers when the augmentation pipeline cannot — i.e., when the negatives must come from a clinical knowledge graph rather than VG co-occurrence — is the open question that any medical-AI follow-up would need to answer first.

## References

- **Paper:** Sarkar, Ebrahimi, Etemad, Beirami, Arik, Pfister. *Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment.* ICLR 2025. arXiv:2405.18654v3 (28 Feb 2025).
- **Code & data:** [https://github.com/pritamqu/HALVA](https://github.com/pritamqu/HALVA) — checkpoints and the released 21.5K `{y_c, y_h}` pairs.
- **Related — sequence-level preference baselines:** HA-DPO (Zhao et al., 2023b), EOS (Yue et al., 2024), RLHF-V (Yu et al., 2024), LLaVA-RLHF (Sun et al., 2023).
- **Related — base MLLMs:** LLaVA-v1.5 (Liu et al., 2023a), VILA (Lin et al., 2024).
- **Related — evaluation suites:** CHAIR (Rohrbach et al., 2018), MME / MME-Hall (Fu et al., 2023), AMBER (Wang et al., 2023b), MMHal-Bench (Sun et al., 2023), HallusionBench (Guan et al., 2024).
- **Related — augmentation source:** Visual Genome (Krishna et al., 2017).
