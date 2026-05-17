---
title: "Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025)"
excerpt: "A regex-based bibliometric audit of 26,104 accepted abstracts that quantifies the VLM/Multimodal/LLM share rising from 16% (2023) to 40% (2025), with inside-VLM reasoning/instruction tasks up +11.5pp while grounding/referring collapses by -13.0pp."
categories:
  - Paper
  - LLM-Agents
tags:
  - VLM
  - Bibliometrics
  - Survey
  - Multimodal-LLM
  - Instruction-Tuning
  - LoRA
  - TF-IDF
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/vlm-26k-papers-survey/
---

## TL;DR
- A lexicon-driven audit of **26,104 CVPR/ICLR/NeurIPS abstracts (2023–2025)** measures how the field reorganized: **VLM/Multimodal/LLM share rises from 16% (2023) to 40% (2025)**, reaching 39.5% at CVPR 2025 and 40.7% at ICLR 2025.
- Inside VLM papers, the task mix flips: **Reasoning/Instruction jumps 13.5% → 25.0% (+11.5pp)** while **Grounding/Referring collapses by -13.0pp**, with parameter-efficient adaptation (Prompt/Prefix, Adapter/LoRA) and instruction tuning rising in lockstep.
- The contribution is **methodological transparency, not algorithmic novelty**: 35 hand-crafted regex categories + phrase-protected tokenization + TF-IDF slopes — *but the lexicon is never validated for recall*, so secondary cells should be read as a noise floor rather than evidence of decline.

## Motivation
The community widely *senses* that VLMs, diffusion, and 3D-Gaussian work now dominate top venues, but most evidence is anecdotal — Twitter screenshots, hand-picked accept lists, vendor blog posts. This paper argues for an auditable, primary-source measurement: parse the official accepted-paper metadata directly, share the lexicon, and let readers re-run or extend the analysis.

The medical-AI angle is light but present. "Medical / Biological Imaging" is explicitly tracked as one of the rising application categories (Fig. 3), and the author notes it rises *consistently* over the period. The more impactful signal for medical-AI readers, however, is the methodological shift inside VLMs (instruction tuning + LoRA on frozen backbones) that is now reshaping how clinical VLM papers should be framed.

## Core Innovation
This is not a model paper. The contribution is **measurement infrastructure**:

1. **Phrase-protected lexicon over abstracts.** Multi-word technical phrases (`gaussian_splatting`, `neural_radiance_fields`, `vision_language_model`) are protected as single tokens before lowercasing and stopword removal, so they survive normalization intact.
2. **35 regex topical categories** (Diffusion, Vision-Language/LLM, 3D, Video, Robustness, Efficiency, ...) with multi-label tagging. Prevalence in year $y$ = (# matched abstracts) / (# abstracts in $y$).
3. **Fine-grained sub-mining inside the VLM subset.** Separate lexicons for named model families (CLIP, BLIP/BLIP-2, LLaVA, Flamingo, DINO, Grounding DINO, ...), fusion mechanisms (Prompt, Adapter/LoRA, Cross-attention, Q-Former, ...), tasks, training paradigms, losses, datasets, and co-modalities.
4. **TF-IDF slope ranking.** Least-squares slope (pp/yr) over the per-year trajectory plus a 3-year endpoint delta (2025 − 2023, in pp).

No training, no learned classifier — the entire pipeline is deterministic regex + TF-IDF. The author releases the lexicon and code, framing the contribution as **auditable infrastructure rather than a one-shot ranking**.

## Claims & Evidence Analysis

| # | Claim | Evidence in the paper | Strength |
|---|-------|----------------------|----------|
| C1 | VLM/Multimodal/LLM share rises from **16% (2023) to 40% (2025)**. | §3, Fig. 1 (TF-IDF trajectory), Fig. 3 (largest slope bar). | ⭐⭐⭐ |
| C2 | Within VLM papers, the field pivots from grounding/referring toward instruction following and reasoning. | Table 4: **Reasoning +11.5pp**, **Grounding −13.0pp**; slopes +5.71 / −8.36 pp/yr. | ⭐⭐⭐ |
| C3 | Parameter-efficient adaptation (Prompt, Adapter/LoRA) dominates fusion design. | Table 3: Prompt/Prefix at 13–16%, Adapter/LoRA +2.8pp; heavy fusion (Encoder-Decoder) declining. | ⭐⭐⭐ |
| C4 | Contrastive objectives recede relative to CE/ranking and distillation. | Table 6: Contrastive **−5.7pp**, KL/Distill +0.3pp, CE/Focal **−0.1pp**, Triplet **−0.5pp**. | ⭐⭐ — the *recession* of contrastive is well-supported; the claimed *rise* of CE/ranking is not visible in the numbers, so this leg is over-stated. |
| C5 | Diffusion research consolidates around controllability, distillation, and speed. | Narrative only; no sub-mining table for diffusion sub-topics (unlike VLMs). | ⭐ |
| C6 | Composition shifts from NeRFs to Gaussian splatting. | Fig. 2 panel trends upward, but no within-panel split separating NeRF from Gaussian Splatting. | ⭐⭐ |
| C7 | CVPR has the strongest 3D footprint; ICLR has the highest VLM share. | CVPR 3D 23.1% vs. ICLR 7.8% (2025); ICLR VLM 40.7% vs. CVPR 39.5%. | ⭐⭐⭐ for 3D gap; ⭐ for VLM gap (within rounding noise). |
| C8 | Longitudinal signals are consistent across venues and years. | Asserted; no per-venue × per-year heatmap or stability statistic provided. | ⭐ |
| C9 | Medical/biological imaging rises consistently. | Narrative + small positive slope in Fig. 3; no per-year table. | ⭐⭐ |
| C10 | Explicit dataset mentions decline (COCO −3.0pp, ImageNet −1.5pp). | Table 7; author honestly flags this conflates "less popular" with "too obvious to name." | ⭐⭐ |

**Honest analysis.** The strong claims (**C1, C2, C3**) are well-supported because the lexicon hits are large in magnitude and *consistent across three independently constructed tables* (fusion, training, loss). The medium claims (**C4, C7, C10**) need the reader to look at the cells themselves — the abstract phrasing slightly oversells what the numbers show. The weak claims (**C5, C6, C8, C9**) are *plausible community knowledge* but the paper does not give them the same fine-grained sub-mining treatment it gives to VLMs; they remain essentially narrative.

Critically missing: (a) no variance / bootstrap on the prevalence ratios, so a "+0.6pp Trend" cell is indistinguishable from regex noise; (b) **no precision/recall evaluation of the lexicon** against a manually labeled subset, so we cannot judge how many true VLM papers are missed; (c) no statistical test for cross-venue gaps; (d) **NeurIPS 2025 absence is not corrected for** in the headline "16% → 40%" number, which mixes a 3-venue 2023 baseline against a 2-venue 2025 endpoint. None of this kills the headline trend (which is large enough to survive any reasonable recall correction), but it means secondary cells (Q-Former, Triplet, Chamfer) should be read as noise floor.

## Method & Architecture

![Direction trajectories across CVPR, ICLR, and NeurIPS 2022–2025](/assets/images/paper/vlm-26k-papers-survey/fig_p002_01.png)
*Figure 1: Direction trajectories across CVPR + ICLR + NeurIPS (2022–2025). VLM/Multimodal/LLM (orange) is the only direction with a near-vertical takeoff, climbing from ~0.05 to ~0.25 aggregated TF-IDF mass while diffusion and video understanding rise more gently. Source: paper Figure 1.*

### Pipeline

1. **Corpus assembly.** JSONL dumps from a Python spider: CVPR 2023–2025 (2,353 / 2,713 / 2,871), ICLR 2023–2025 (4,372 / 2,260 / 3,704), NeurIPS 2023–2024 (3,337 / 4,494). After empty-record removal: **N = 26,104**. An additional 8,424 papers from 2022 are kept *trend-only* and excluded from content analysis.
2. **Text normalization.** Unicode-normalize → lowercase → strip punctuation → phrase-protect multi-word technical phrases as single tokens → remove general stopwords and generic CV vocabulary.
3. **Topical labeling.** Match against 35 regex categories, multi-label allowed. Per-year prevalence is the matched fraction.
4. **Trend aggregation.** Per-year TF-IDF mass per direction; least-squares slope (pp/yr) and 3-year endpoint delta (2025 − 2023, in pp).
5. **Fine-grained VLM mining.** Inside the VLM-tagged subset, additional regex dictionaries for model families (Table 1), fusion mechanisms (Table 3), tasks (Table 4), training paradigms (Table 5), losses (Table 6), datasets (Table 7), co-modalities (Table 8).
6. **Cross-venue slicing.** Same lexicon re-applied per (venue, year).

![Small-multiples view of all 35 research directions, each on its own y-axis](/assets/images/paper/vlm-26k-papers-survey/fig_p003_01.png)
*Figure 2: All 35 directions, each panel on its own y-axis. Vision-Language/Multimodal/LLM, Diffusion & Generative, and NeRF/Gaussian Splatting curve up; self-supervised pretraining, GNNs, and meta-learning curve down. Source: paper Figure 2.*

The "math" that matters is the per-cell prevalence ratio and the linear slope across three (or four) years — there is no model to train.

## Experimental Results

There are no "method × baseline × metric" comparisons. The "results" are per-category prevalence tables. The headline macro shift:

| Direction | 2023 | 2024 | 2025 | Trend (pp) |
|---|---|---|---|---|
| **VLM / Multimodal / LLM (all venues)** | **16%** | — | **40%** | **+24** |
| VLM share at CVPR 2025 | — | — | 39.5% | — |
| VLM share at ICLR 2025 | — | — | 40.7% | — |
| Diffusion (all venues) | 8% | 14.9% | 19.2% | +11.2 |
| Diffusion at CVPR 2025 | — | — | 25.7% | — |

Inside VLM papers, the most informative shifts:

| Inside-VLM signal | 2023 | 2024 | 2025 | Trend | Slope (pp/yr) |
|---|---|---|---|---|---|
| **Reasoning/Instruction (task)** | **13.5%** | 22.3% | **25.0%** | **+11.5** | **+5.71** |
| **Grounding/Referring (task)** | **25.9%** | 14.5% | **12.9%** | **−13.0** | **−8.36** |
| Instruction Tuning (training) | 1.1% | 4.2% | 5.0% | +3.9 | +1.75 |
| LoRA/Adapters (training) | 1.3% | 4.0% | 4.1% | +2.8 | +1.26 |
| Contrastive/InfoNCE (loss) | 10.8% | 5.6% | 5.1% | −5.7 | −2.07 |
| LLaVA (named model) | 0.1% | 1.2% | 2.7% | +2.6 | +0.91 |
| ALIGN (named model) | 4.3% | 5.8% | 5.1% | −0.8 | +0.65 |

![Top rising directions ranked by least-squares slope](/assets/images/paper/vlm-26k-papers-survey/fig_p005_01.png)
*Figure 3: Top rising directions ranked by least-squares slope (2022→2025). VLM/Multimodal/LLM (~0.075) leads by nearly 2x over Diffusion & Generative (~0.037); medical/biological imaging appears at the bottom of the rising list. Source: paper Figure 3.*

The most striking *qualitative* finding is the **shape** of the shift, not any single number. The VLM literature is migrating from "build a new dual-encoder + contrastive loss" toward "freeze a strong backbone, attach a lightweight bridge (projector / LoRA / prompt), supervise with instruction data + KL distillation."

Three independently constructed tables corroborate this story:

- **Fusion (Table 3):** Prompt/Prefix +1.3pp and Adapter/LoRA +2.8pp rising; Encoder-Decoder −1.3pp and Dual-encoder −0.1pp falling.
- **Training (Table 5):** Pretrain+Finetune +5.2pp, Instruction Tuning +3.9pp, LoRA +2.8pp rising; Self/Weak/Semi-sup falls by **−6.1pp**.
- **Loss (Table 6):** Contrastive/InfoNCE falls by **−5.7pp** while KL/Distillation stays flat-positive.

**Cross-venue:** CVPR keeps the strongest 3D footprint (23.1% 3D-geometry in 2025 vs. 7.8% at ICLR) while ICLR leads the VLM share (40.7% vs. CVPR 39.5%).

## Limitations

**Author-acknowledged:**
- Lexicon-driven precision/recall tradeoff (high precision, possibly low recall on niche synonyms) — so absolute prevalences are *lower bounds*.
- Abstracts under-report training data and loss details, so Tables 6–7 are systematic underestimates.
- Percentages do not sum to 100% because of multi-label tagging; category boundaries (Grounding vs. Reasoning, Open-Vocab Det vs. Segmentation) are partially overlapping.
- Trend signals are 3-point (2023–2025), which limits slope reliability.
- NeurIPS 2025 is not in the corpus, creating an asymmetric venue panel.

**Not addressed:**
- **No lexicon validation set.** Precision/recall of the 35 regex categories against a human-labeled sample is never reported; raw prevalence cannot be translated into a confidence interval.
- **No correction for venue-size imbalance.** ICLR 2023 (4,372) vs. ICLR 2024 (2,260) — nearly a 2x swing in denominator — is treated as if random sampling.
- **No de-duplication of model families.** "ALIGN" and "CLIP" are separated despite the same architectural template; this inflates ALIGN's apparent dominance.
- **No abstract-vs-paper validation.** Papers that *use* LoRA may not say so in the abstract; non-random missingness pattern.
- **No author/affiliation bias check.** A handful of prolific industry labs could be disproportionately responsible for the VLM curve.
- **No mechanism for "why contrastive declines."** Is it because contrastive losses are now standard infrastructure (and therefore unmentioned in abstracts) or because the community has actually moved on? The data cannot distinguish these.
- **Medical-AI granularity is shallow.** "Medical/Biological Imaging" is one bar in Fig. 3; no breakdown of medical VLMs (MedCLIP/LLaVA-Med/RadFM) the way the general VLM section is mined.

## Why It Matters for Medical AI

The survey is general-purpose, but three findings carry over almost directly to medical VLM work:

- **The instruction-tuning + LoRA pivot is the new default.** If you are writing a clinical VLM paper in 2026 and your method section centers on a fresh dual-encoder + InfoNCE recipe, you are now swimming against a measurable current. The community-wide shift toward *frozen backbones + lightweight bridges + instruction data* is the implicit baseline reviewers will compare against.
- **Reasoning/instruction tasks outpace grounding/referring.** Medical equivalents — radiology report generation, multi-step CoT-style differential diagnosis, instruction-following CAD — are the same trend manifested in the clinical domain. Pure grounding/referring papers (organ localization, lesion pointing) are a shrinking share of the conversation.
- **Medical/biological imaging rises, but slowly.** Fig. 3 places it at the bottom of the *rising* list. The runway for medical VLM contributions is real, but the field is small relative to general VLM work — which is exactly why building on top of strong general backbones (rather than re-inventing the wheel) is the dominant strategy now.

What the survey *does not* give you: any breakdown of medical VLMs by family, no clinical benchmark coverage, no safety/governance axis. Use this paper as a **macro thermometer** for what the surrounding general-CV community is doing — and pair it with a medical-VLM-specific survey for clinical depth.

## References
- **Paper:** Lin, Fengming. "Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025)." arXiv:2510.09586v1 [cs.CV], October 13, 2025. [https://arxiv.org/abs/2510.09586](https://arxiv.org/abs/2510.09586)
- **Lexicon & code:** Released by the author alongside the preprint (see paper for current URL).
- **Companion VLM building blocks already on this blog:** CLIP, BLIP-2, LLaVA, LLaVA-Med, BiomedCLIP, CoOp/CoCoOp, SigLIP, CONCH, Virchow.
- **Related surveys:** LLM-Agents (Wang 2023; Xi 2023; Sumers 2023); Large Multimodal Agents (Xie 2024); Empowering MLLMs with Tools.
