---
title: "Silkie: Preference Distillation for Large Visual Language Models"
excerpt: "DPO on an 80,258-instruction GPT-4V preference set lifts Qwen-VL-Chat by +6.9 / +9.5 / +4.5 / +9.2 % on MME-P / MME-C / MMHal / MM-Vet — but only single-seed and with a judge-vs-human agreement of just 83.1%."
categories:
  - Paper
  - VLM-Alignment
  - LLM
tags:
  - Silkie
  - VLFeedback
  - DPO
  - RLAIF
  - Preference-Distillation
  - LVLM
  - GPT-4V
  - Qwen-VL-Chat
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
permalink: /paper/silkie/
---

## TL;DR
- VLFeedback is an **80,258-instruction, ~28K 4-way-comparison** GPT-4V-scored preference dataset for LVLMs, judged on helpfulness, visual faithfulness, and ethical considerations.
- DPO-distilling VLFeedback into Qwen-VL-Chat-v1.1 (7B) yields **Silkie**, which improves on the base by **+6.9 % MME-P, +9.5 % MME-C, +4.5 % MMHal-Bench, +9.2 % MM-Vet** — four benchmarks, consistent direction.
- The honest read: results are single-seed with no variance, GPT-4V-vs-human agreement is only **83.1 %** on 100 pairs (and Appendix C documents the judge mis-rating visual faithfulness), and despite 8.3K medical instructions in the training mix there is **no medical-domain evaluation**.

## Motivation

Open-source LVLMs are still trapped in a post-SFT regime where they hallucinate freely and produce text that is not grounded in the image. The text-only LLM community escaped a structurally identical problem with RLHF and then, more cheaply, with RLAIF (UltraFeedback): collect a large preference set, distil it with DPO. The multi-modal analogues that existed in late 2023 — LLaVA-RLHF and RLHF-V — each had a fundamental ceiling: LLaVA-RLHF needed expensive human raters and RLHF-V's full corpus is only **1.4K** pairs scoped narrowly to hallucination.

Silkie asks the obvious question: can GPT-4V *replace* the human rater for multi-aspect, broad-coverage VLM preference data? If so, the alignment recipe for LVLMs collapses to (i) sample diverse instructions, (ii) sample diverse model responses, (iii) ask GPT-4V to score, (iv) DPO. The medical-AI angle is incidental: LLaVA-Med (5.9K) and PMC-VQA (2.4K) are folded into the instruction pool, but no clinical evaluation is reported.

## Core Innovation

The contribution is the **pipeline**, not the model. There is no new architecture, no new objective beyond standard Rafailov DPO. What is genuinely new is the data-construction recipe:

- **A 12-model response pool** spanning closed (GPT-4V) and open (LLaVA-1.5 7B/13B, LLaVA-RLHF 7B/13B, Qwen-VL-Chat, IDEFICS-9B, Fuyu-8B, InstructBLIP-Vicuna 7B/13B, VisualGLM-6B, MMICL-13B) — for each instruction, 4 of the 12 are sampled, producing **~28K 4-way groups (~112K responses)**.
- **A three-aspect Likert rubric** (1–5) covering Helpfulness, Visual Faithfulness, Ethical Considerations, with rationales — scored by GPT-4V.
- **A scalar-collapse step**: the three per-aspect ratings are averaged into a single score s_i, then within each 4-response group the K(K-1)/2 = 6 ordered pairs are expanded (ties dropped), yielding the y_w / y_l pairs that DPO consumes.

The DPO objective itself is unchanged:

$$
\max_{\pi_\theta}\, \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right)
$$

with LoRA, 3 epochs, AdamW (β1=0.9, β2=0.98, eps=1e-6, wd=0.05), cosine LR (warmup 0.1, peak 1e-5), global batch 256, 16× A100, ~30 h per run.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | DPO on VLFeedback improves Qwen-VL-Chat across perception, cognition, hallucination, and general VQA. | Table 4: +6.9 / +9.5 / +4.5 / +9.2 % on MME-P / MME-C / MMHal / MM-Vet. | ⭐⭐⭐ — 4 benchmarks, consistent direction, but single seed and no CIs. |
| C2 | Silkie sets SoTA on MMHal-Bench at 3.02. | Table 4 row. | ⭐⭐ — single benchmark, single seed; 2.89 → 3.02 is small relative to GPT-4-judge noise. |
| C3 | Gains concentrate in fine-grained perception (OCR, artwork) and complex cognition (code, text-translation). | Fig 4 left, MME subtasks. | ⭐⭐⭐ — clean per-subtask breakdown. |
| C4 | AI-annotated VLFeedback beats human-annotated RLHF-V on a matched 1.4K-pair budget. | Fig 4 right. | ⭐⭐ — partly a coverage tautology: RLHF-V was *designed* for hallucination only, so losing on MME-C / MM-Vet is structural, not surprising. |
| C5 | GPT-4V is a reliable proxy for human preferences. | 83.1 % pairwise agreement on a 100-pair subset with 3 annotators. | ⭐⭐ — small sample, overall-quality only, no per-aspect breakdown; Appendix C documents concrete visual-faithfulness misjudgments. |
| C6 | Improvement is not driven by length-hacking or GPT-4V style mimicry. | Two controls — *Longest=Best* and *GPT-4V=Best* — both underperform Silkie (Table 4). | ⭐⭐⭐ — the strongest internal control in the paper. |
| C7 | Multi-aspect averaging improves visual faithfulness specifically. | MMHal-Bench 2.89 → 3.02. | ⭐⭐ — one benchmark, Δ=0.13, no significance test, no per-aspect ablation. |
| C8 | The dataset covers domain-specific (medical) instructions usefully. | Inclusion of LLaVA-Med (5.9K) + PMC-VQA (2.4K). | ⭐ — no medical-only eval, no domain-stratified result. |

## Method & Architecture

![VLFeedback annotation pipeline](/assets/images/paper/silkie/fig_p002_05.png)
*Figure 1: VLFeedback annotation pipeline — instructions are sampled from nine sources, 4 of 12 LVLMs generate responses, and GPT-4V scores each response on helpfulness, visual faithfulness, and ethical considerations with written rationales.*

The instruction pool (80,258 prompts over ~65K unique images) breaks down as: **LLaVA 19.6K, SVIT 22.8K, LLaVAR 13.8K, LRV 12.4K, LLaVA-Med 5.9K, PMC-VQA 2.4K, ComVint 2.4K, M3IT 0.7K, PCA-EVAL 0.4K**. Each instruction is routed to a random subset of 4 of the 12 generator models; GPT-4V then issues three 1–5 ratings per response. The scalar collapse — averaging the three aspect scores — is the load-bearing design decision: it makes DPO simple, but it also discards exactly the fine-grained signal that justified the three-aspect rubric in the first place.

![GPT-4V rating distributions](/assets/images/paper/silkie/page_005.png)
*Figure 2 (page 5): Per-aspect GPT-4V rating distributions across the 12 generator models. Ethical-considerations ratings collapse to 4–5 because the instruction pool is not red-team-style; helpfulness and visual-faithfulness distributions are nearly identical, a hint that the judge may not really be separating those two axes.*

## Experimental Results

**Main table (Table 4).** Backbone is Qwen-VL-Chat (7B) unless noted.

| Model | MME-P | MME-C | MMHal-Bench | MM-Vet |
|---|---|---|---|---|
| LLaVA | 807.0 | 247.9 | – | – |
| LLaVA-RLHF | – | – | 2.05 | – |
| LLaVA-v1.5 | 1510.7 | 316.1 | 2.42 | 30.5 |
| LLaVA-v1.5 + SFT (ShareGPT4V) | 1567.4 | 376.4 | 2.28 | 37.6 |
| Qwen-VL-Chat (base) | 1440.5 | 362.5 | 2.89 | 45.7 |
| Qwen-VL-Chat + DPO (Longest=Best) | 1393.8 | 355.4 | 2.59 | 44.5 |
| Qwen-VL-Chat + DPO (GPT-4V=Best) | 1460.9 | 353.6 | 2.81 | 45.9 |
| Qwen-VL-Chat + SFT (ShareGPT4V) | 1527.4 | – | – | 45.9 |
| **Silkie (Qwen-VL-Chat + DPO on VLFeedback)** | **1539.6** | **397.1** | **3.02** | **49.9** |
| Δ vs Qwen-VL-Chat | +6.9 % | +9.5 % | +4.5 % | +9.2 % |

**MME subtask analysis.** The gains are not uniform: Silkie clearly improves on **OCR, artwork, code-reasoning, and text-translation**, while basic existence / count / color / position barely move. This pattern is consistent with a preference-distillation story (the judge rewards detailed, grounded responses on harder subtasks) and inconsistent with a generic capability bump.

![MME subtask breakdown and VLFeedback-vs-RLHF-V comparison](/assets/images/paper/silkie/page_009.png)
*Figure 4 (page 9): Left — Silkie's MME subtask gains over Qwen-VL-Chat, concentrated on OCR, artwork, code, and translation. Right — 1.4K-matched-pair comparison vs RLHF-V: VLFeedback wins broadly on MME-P / MMHal / MM-Vet but ties on MME-C; RLHF-V is hallucination-focused by design, so the apparent "AI > human" sweep is partly a coverage artifact.*

**The two control experiments are the most informative table rows.** *Longest=Best* degrades all four metrics — the paper's evidence that length-bias is not the trick. *GPT-4V=Best* (always prefer GPT-4V's response when it is in the 4-tuple) improves MME-P marginally but degrades MME-C, and lands well below Silkie on MM-Vet. Both controls underperforming Silkie is the strongest evidence in the paper that the *quality of the multi-aspect preference signal* is what drives gains, not surface heuristics.

**GPT-4V–human agreement.** On a 100-pair subset rated by three human annotators, GPT-4V's pairwise preferences agree with the human majority **83.1 %** of the time. This is the load-bearing number for the entire RLAIF-for-LVLM premise, and it is reported only as an overall figure (no per-aspect breakdown).

![Qualitative case studies](/assets/images/paper/silkie/fig_p010_02.png)
*Figure 5: Side-by-side case studies on MMHal-Bench (left, visual grounding — Silkie correctly identifies the left stool that Qwen-VL-Chat hallucinates as "a vase with a red flower") and MM-Vet (right, scientific reasoning — Silkie produces the correct step-by-step solution).*

## Limitations

**Acknowledged.**
- Ethical-aspect supervision is degenerate — the instruction pool is not red-team-style, so the vast majority of ethical-considerations scores cluster at 4–5.
- LVLM and instruction coverage was limited to late-2023 models; the landscape moved fast.

**Not acknowledged.**
- **Single-seed, no variance.** Every headline number is reported without a confidence interval, multi-seed average, or significance test — this is the single largest weakness given that the headline deltas (e.g. MMHal 2.89 → 3.02) are well within plausible run-to-run noise for GPT-4-as-judge metrics.
- **Pool-judge collusion.** GPT-4V is *both* a generator (its outputs are in the 4-tuple) and the judge. The *GPT-4V=Best* ablation partially controls for this at the response-selection level but not at the per-pair scoring level — GPT-4V scoring its own outputs is not corrected for.
- **Aspect-collapse.** Averaging helpfulness + visual-faithfulness + ethics into a single scalar throws away the per-aspect signal that justified building the three-aspect rubric. No ablation isolates visual-faithfulness as a preference axis.
- **GPT-4V is a noisy visual judge.** Appendix C exhibits concrete cases where GPT-4V mis-grades visual faithfulness, undercutting the paper's strongest selling point. The 83.1 % agreement is on overall quality, not on visual faithfulness specifically.
- **No external hallucination evaluation.** POPE — the standard object-hallucination probe — is absent, as are HallusionBench, MMBench, ScienceQA, and SEED-Bench.
- **No medical evaluation.** Despite ~8.3K medical instructions in the training mix (LLaVA-Med + PMC-VQA), no medical-VQA or hallucination evaluation (e.g. PMC-VQA test, Med-HallMark) is reported.
- **Reproducibility is conditional on a proprietary, drifting annotator.** Re-running the pipeline against GPT-4V years later will not reproduce the same dataset.
- **API cost is undisclosed.** The economic argument for AI-feedback vs human-feedback rests on a number the paper does not give.

![GPT-4V visual-faithfulness misjudgment](/assets/images/paper/silkie/fig_p017_01.png)
*Appendix C example: an image where all three human annotators disagreed with GPT-4V's visual-faithfulness rating — a concrete instance of the judge unreliability the agreement-rate number averages over.*

## Why It Matters for Medical AI

The medical-AI relevance of Silkie is structural, not empirical. The pipeline shows that multi-aspect AI-feedback DPO is a plausible alignment recipe for *any* LVLM, including biomedical ones — but the paper itself supplies **no medical evidence**. Two cautions for anyone tempted to copy the recipe verbatim onto a med-VLM:

- **Visual faithfulness is the alignment axis that matters most clinically**, and it is precisely the axis on which Appendix C documents GPT-4V making mistakes. A med-VLM aligned against a noisy faithfulness judge inherits that judge's blind spots — directly.
- **GPT-4V is not a substitute for a clinician annotator.** The 83.1 % overall-quality agreement does not transfer to specialist-domain agreement; in radiology / pathology / dermatology the per-aspect agreement is the number that would need to be measured, and it is not measured in this paper.

The right reading is: VLFeedback validates the *engineering* of multi-aspect RLAIF for LVLMs; a medical deployment still needs a domain-grounded judge and a domain-stratified eval. The 8.3K medical instructions in VLFeedback are a starting point for analysis, not a finished medical alignment dataset.

## References

- **Paper.** Li, Xie, Li et al. *Silkie: Preference Distillation for Large Visual Language Models.* arXiv:2312.10665 (Dec 2023); ACL 2024 Findings. <https://arxiv.org/abs/2312.10665>
- **Code & dataset.** <https://vlf-silkie.github.io>
- **Related work.**
  - Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023) — the underlying DPO objective.
  - Cui et al., *UltraFeedback* (2023) — the text-only RLAIF blueprint Silkie multi-modalises.
  - Sun et al., *LLaVA-RLHF* (2023) — the human-rater LVLM RLHF baseline.
  - Yu et al., *RLHF-V* (CVPR 2024) — the 1.4K human-annotated hallucination-focused alternative compared against in Fig 4 right.
  - Liu et al., *LLaVA / LLaVA-1.5* — the open-source LVLM family Silkie's instruction mix and baselines draw from.
  - Bai et al., *Qwen-VL / Qwen-VL-Chat* — Silkie's reference backbone.
