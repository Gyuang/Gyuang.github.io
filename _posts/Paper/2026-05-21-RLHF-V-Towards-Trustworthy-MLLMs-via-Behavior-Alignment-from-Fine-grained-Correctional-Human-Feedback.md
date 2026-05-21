---
title: "RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback"
excerpt: "1.4K segment-level corrections plus Dense DPO drop Muffin's MHumanEval hallucination from 74.7% to 55.5% (-34.8% relative) and tie GPT-4V on Object HalBench mention rate (7.5 vs 7.3) with a 13B open backbone."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/rlhf-v/
tags:
  - RLHF-V
  - DDPO
  - DPO
  - MLLM
  - Hallucination
  - Alignment
  - Vision-Language
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- Existing MLLM RLHF pipelines rank entire long-form responses, which is ambiguous, label-noisy, and reward-hackable. RLHF-V instead collects **segment-level human corrections of hallucinated spans** (1.4K samples on the Muffin base) and trains with **Dense DPO (DDPO)**, which upweights the corrected token positions by gamma=5 inside the standard DPO loss.
- Headline number: **MHumanEval overall hallucination drops 74.7 -> 55.5 (-34.8% relative)** on the Muffin backbone, and **Object HalBench mention rate ties GPT-4V (7.5 vs 7.3)** with a 13B open model. Helpfulness on LLaVA Bench / VQAv2 does not regress.
- DDPO's standalone contribution is small (~2 absolute points over vanilla DPO on a single run); most of the gain comes from **IT-VQA SFT (74.7 -> 65.1)** and from **disabling image-crop augmentation**. The data recipe, not the loss, is doing the heavy lifting.

## Motivation

MLLMs hallucinate persistently even with strong LLM backbones - the authors' human eval finds GPT-4V hallucinates obviously in 45.9% of long-form image descriptions. The standard LLM-style RLHF pipeline (rank two long responses, then PPO over a learned reward) transfers poorly: annotators cannot reliably decide which long flawed response is "better," so the sparse credit signal has to flow back through linguistic variance and style bias before it can shape the actual hallucination behavior. This is exactly the regime where high-stakes deployments - medical CAD, clinical reporting, accessibility tools for visually impaired users - stall, so a data-efficient alignment recipe that *surgically removes hallucinated spans* is directly portable to those settings.

## Core Innovation

- **Preference data as edits, not rankings.** For each MLLM output `y_l`, annotators rewrite the hallucinated spans into a corrected `y_w`. By construction `y_w` is the optimal completion for that image (per the annotator's evidence), and the (y_w, y_l) pair differs *only* in the edited segments. Hallucination types corrected: objects 41.2%, positions 20.3%, numbers 16.5%, attributes 10.0%, actions 5.3%, misc 6.8% across 1.4K prompts, mean 64.4 words and 2.65 corrected segments per response.
- **Dense DPO (DDPO).** Split each response into corrected tokens `y_c` and unchanged tokens `y_u` and redefine the per-response log-likelihood as a length-normalized weighted sum: $\log \pi(y\mid x) = \frac{1}{N}\left[\sum_{y_i \in y_u} \log p(y_i\mid x, y_{<i}) + \gamma \sum_{y_i \in y_c} \log p(y_i \mid x, y_{<i})\right]$ with $N = |y_u| + \gamma|y_c|$. Plug this score into the unchanged DPO loss. They use gamma=5, beta=0.5.
- **Two orthogonal hygiene tricks.** (i) Intermediate **VQAv2 instruction tuning** to counter hallucinations baked into pretraining captions and GPT-4-generated visual instructions. (ii) **Disable image-crop augmentation**, because crops can remove an object the supervised caption mentions, training the model to confabulate.
- **Cost.** Muffin (BEiT-3 vision + 13B Vicuna v1.0), 7 epochs DDPO at 448 resolution, lr 5e-7, batch 32 - **< 1 hour on 8x A100** for the full DDPO stage.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Segment-level corrections outperform response-level rankings for hallucination reduction | Fig. 2 data-efficiency curve (RLHF-V vs LLaVA-RLHF preference data on the same Muffin base) + Table 3 ablation (vanilla DPO vs DDPO on same data) | MHumanEval, MMHalBench | ⭐⭐⭐ - controlled at fixed base + fixed loss; clean |
| C2 | DDPO (segment up-weight, gamma=5) beats vanilla DPO | Table 3: MHumanEval All 57.5 -> 55.5; MMHalBench Resp 54.2 -> 52.1 | MHumanEval, MMHalBench | ⭐⭐ - **single seed, 2-pt gap, no variance, no gamma sweep**; re-weighted log-lik is not a Bradley-Terry score for gamma != 1 |
| C3 | 1.4K RLHF-V samples beat LLaVA-RLHF's 10K preference + 72K augmented samples | Table 1: MHumanEval All 55.5 vs 72.6; ObjHalBench Resp 12.2 vs 38.1 | MHumanEval, ObjHalBench | ⭐⭐ - different backbones (Muffin/Vicuna-v1.0 vs Vicuna-v1.5); Fig. 2 controlled-base version is the stronger form of this claim |
| C4 | RLHF-V is more scene-robust than GPT-4V | Table 2 four scenes: avg delta +1.7 (RLHF-V) vs +5.0 (GPT-4V) | Object HalBench scene subsets | ⭐⭐ - only 4 COCO scenes, top-10 frequent objects scoring may flatter the method |
| C5 | Generalizes to other backbones (LLaVA, OmniLMM-12B) | LLaVA: -5.9 resp / -13.8 count hallucination. OmniLMM-12B: 4.5% mention on ObjHalBench, 1637 MME, 71.1 SeedBench-I | ObjHalBench, MHumanEval, MME, SeedBench | ⭐⭐ - two-model generalization; OmniLMM result from a separate codebase release |
| C6 | Hallucination reduction does not sacrifice helpfulness | LLaVA Bench Conv/Detail/Comp 93.1/75.3/91.6 within 5pp of Muffin; VQAv2 80.0 matches LLaVA 1.5 | LLaVA Bench, VQAv2 | ⭐⭐⭐ - clear in-table evidence across multiple helpfulness metrics |
| C7 | VQAv2 SFT alone reduces hallucination (orthogonal to RLHF) | Ablation "w/ IT-VQA only" beats Muffin 65.1 vs 74.7 on MHumanEval All | MHumanEval, MMHalBench | ⭐⭐⭐ - **clean single-axis ablation; IT-VQA carries most of the headline gain** |
| C8 | Disabling image-crop augmentation matters | Ablation "w/ untrust aug": MHumanEval 59.6 vs 55.5; VQAv2 also drops 77.1 vs 80.0 | MHumanEval, MMHalBench, VQAv2 | ⭐⭐ - consistent direction, modest magnitude; not isolated from RLHF-V |

**Honest read.** Three things deserve to be flagged because the abstract foregrounds the DDPO loss while the ablation tells a different story:

1. **The headline -34.8% relative drop on MHumanEval is a *system* result, not a DDPO result.** The full RLHF-V pipeline (1.4K segment corrections + IT-VQA SFT + crop-off + DDPO) takes Muffin from 74.7 -> 55.5 on MHumanEval All. The data recipe (segment-level corrections instead of rankings) and the SFT/augmentation hygiene are doing most of that work.
2. **IT-VQA SFT alone carries most of the gain.** Table 3 puts "w/ IT-VQA only" at 65.1 on MHumanEval All - 9.6 of the 19.2-point total improvement before any DPO variant is run. The remaining ~10 points are split between the segment-level preference data and the loss choice.
3. **DDPO over vanilla DPO is +2 points on a single run, with no gamma sweep and no variance.** 55.5 vs 57.5 on MHumanEval All is within plausible seed noise; without multiple seeds or a gamma ablation, the methodological contribution (which is what gives the paper its name) is the *least* evidenced piece. The redefined log-likelihood with gamma != 1 also is no longer a Bradley-Terry log-likelihood, so the DPO derivation no longer goes through cleanly - this is not flagged in the paper.

Other gaps: no out-of-domain evaluation (everything is COCO-derived; no medical / scientific / OCR-heavy stress test), no inter-annotator agreement or annotator-cost reporting (the "data-efficient" framing measures samples, not annotator-hours), no reward-hacking probe, and no head-to-head against other dense-credit methods (Wu et al. NeurIPS 2023 fine-grained RLHF; Gunjal et al. factually-augmented DPO).

## Method & Architecture

![RLHF-V framework overview](/assets/images/paper/rlhf-v/page_002.png)
*Figure 1: RLHF-V overview - fine-grained segment-level corrections of hallucinated spans feed Dense DPO (DDPO), producing the final aligned MLLM.*

The pipeline is three stages: (1) sample a response `y_l` from the base MLLM on a description prompt; (2) human annotator edits hallucinated spans into a corrected `y_w`; (3) optimize DDPO over the resulting (y_w, y_l) pair, with the corrected tokens weighted gamma=5x. The unchanged tokens contribute the dynamics that anchor the policy to the reference (preventing reward hacking on style / length), while the corrected tokens deliver the precise credit signal where the human actually disagreed with the model. IT-VQA SFT and crop-disabled augmentation are applied as pre/parallel-stages, not as part of DDPO itself.

## Experimental Results

![Main hallucination and helpfulness table](/assets/images/paper/rlhf-v/page_005.png)
*Table 1: Hallucination rates and helpfulness across five benchmarks. RLHF-V matches GPT-4V mention-level on Object HalBench (7.5 vs 7.3) using a 13B open backbone, and is the best or second-best open-source model on every hallucination column.*

Full numbers as printed in the paper (lower is better for all hallucination columns):

| Model | ObjHal Resp. | ObjHal Mention | MHumanEval Obj | MHumanEval Pos | MHumanEval Num | MHumanEval All | MMHal Info | MMHal Resp | LLaVA Conv | LLaVA Detail | LLaVA Comp | VQAv2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA | 63.0 | 29.5 | 46.6 | 21.2 | 19.9 | 80.8 | 31.9 | 70.8 | 85.4 | 74.3 | 96.3 | - |
| Muffin (base) | 50.5 | 24.5 | 33.6 | 16.4 | 26.0 | 74.7 | 33.4 | 68.8 | 89.3 | 79.7 | 97.7 | - |
| LRV | 32.3 | 22.3 | 43.2 | 11.6 | 19.2 | 82.9 | 22.2 | 78.1 | 61.7 | 47.3 | 55.0 | - |
| LLaVA-RLHF | 38.1 | 18.9 | 37.7 | 17.8 | 18.5 | 72.6 | 39.9 | 65.6 | 93.8 | 74.3 | 111.4 | - |
| InstructBLIP | 25.9 | 14.3 | 30.8 | 15.1 | 17.1 | 63.7 | 29.5 | 64.4 | 83.2 | 67.6 | 90.6 | - |
| Qwen-VL-Chat | 43.8 | 20.0 | 34.9 | 16.4 | 15.8 | 61.0 | 38.5 | 52.1 | 81.9 | 77.1 | 92.3 | 79.5 |
| LLaVA 1.5 | 46.3 | 22.6 | 30.8 | 17.8 | 17.1 | 61.0 | 39.2 | 52.1 | 81.6 | 75.5 | 95.2 | 80.0 |
| **RLHF-V** | **12.2** | **7.5** | **21.9** | **7.5** | **14.4** | **55.5** | 40.0 | 52.1 | 93.1 | 75.3 | 91.6 | 80.0 |
| GPT-4V | 13.6 | 7.3 | 22.6 | 12.3 | 11.0 | 45.9 | 47.6 | 31.3 | 96.0 | 102.5 | 106.7 | 77.2* |

The Object HalBench mention-rate tie with GPT-4V (7.5 vs 7.3) is the cleanest finding of the paper: a 13B open model with 1.4K human corrections matches a frontier commercial system on the most widely cited open-vocabulary hallucination metric.

![Over-generalization and data scaling](/assets/images/paper/rlhf-v/page_006.png)
*Table 2 + Figure 2: Scene-induced hallucination delta is +1.7 for RLHF-V vs +5.0 for GPT-4V across four scenes; data scaling shows 200 RLHF-V samples match LLaVA-RLHF's full 2.2K preference set on the same Muffin base, with no saturation through 2.2K.*

![Ablation table](/assets/images/paper/rlhf-v/page_007.png)
*Table 3: DDPO over vanilla DPO buys ~2 points, IT-VQA SFT alone closes most of the gap from Muffin, and disabling crop augmentation matters for both hallucination and VQA.*

Ablation read in absolute MHumanEval All / MMHalBench Resp / VQAv2 terms:

| Variant | MHumanEval All | MMHalBench Resp | VQAv2 |
|---|---|---|---|
| Muffin (base) | 74.7 | 68.8 | - |
| **RLHF-V full** | **55.5** | **52.1** | **80.0** |
| w/ vanilla DPO (no segment up-weight) | 57.5 | 54.2 | 80.0 |
| w/ IT-VQA only (no DDPO) | 65.1 | 58.3 | 80.0 |
| w/ untrustworthy aug (image crop on) | 59.6 | 54.2 | 77.1 |

Read this table carefully: from 74.7 -> 65.1 on MHumanEval All comes from IT-VQA SFT alone, 65.1 -> 57.5 comes from the segment-level preference data (under vanilla DPO), and only the last 57.5 -> 55.5 comes from DDPO. The "Dense DPO" branding overstates the loss's standalone contribution.

## Limitations

- **No variance reporting.** Every reported number is a single run. The 2-point DDPO-vs-DPO gap is within plausible seed noise; without multi-seed runs the methodological contribution is statistically unsupported.
- **No gamma sweep.** gamma=5 is chosen; sensitivity is not shown. The paper does not address that the gamma-reweighted score is no longer a Bradley-Terry log-likelihood, so the DPO derivation no longer cleanly applies.
- **Annotator quality opaque.** No inter-annotator agreement, no training/calibration protocol, no payment, no per-sample correction time. For a paper whose pitch is data quality, these omissions matter; the "data-efficient" framing also looks different if annotator-hours, not sample count, is the right unit.
- **Narrow evaluation surface.** All images are COCO-derived. The "more robust than GPT-4V" claim rests on 4 scenes and top-10 frequent objects per scene - a specific scoring choice that may flatter the method, and a setting that does not transfer obviously to medical or scientific imagery.
- **No reward-hacking probe.** The motivation pitches fine-grained correction as a hedge against reward hacking, but length / style / refusal-rate hacking after DDPO is not measured.
- **No head-to-head with other dense-credit methods.** Fine-grained RLHF (Wu et al., NeurIPS 2023) and factually-augmented DPO (Gunjal et al.) are dismissed in related work but not benchmarked.
- **Scale unknown.** The data-scaling curve in Figure 2 has not saturated through 2.2K samples; the paper does not push further.

## Why It Matters for Medical AI

Clinical CAD, radiology report generation, and any assistive system that turns a medical image into language is bottlenecked by exactly the failure mode RLHF-V targets: confidently asserting objects / positions / counts that are not in the image. The recipe is attractive for medical-AI labs for three reasons. (i) **1.4K corrections + < 1 hr on 8x A100** is well within the budget of a clinical-AI team that already pays radiologists for annotation - segment-level corrections of a draft report are arguably *easier* for a clinician than ranking two flawed reports, because corrections are how clinicians normally edit residents' drafts. (ii) The dominant gain comes from **IT-VQA SFT and augmentation hygiene**, both of which are trivially portable to any domain with a VQA-style supervised set (e.g. VQA-RAD, SLAKE, OmniMedVQA). (iii) The over-generalization claim (C4) is the medically relevant one - a CAD model that says "tumor" whenever it sees an MRI because that scene-class typically contains a tumor is the over-generalization failure mode; whether RLHF-V's scene-robustness transfers from COCO scene priors to modality-specific priors is the obvious next experiment.

Caveat: do not assume the COCO-based numbers transfer. Re-run the over-generalization test with modality-conditioned object sets before believing the "more robust than GPT-4V" story in a clinical setting.

## References

- Paper (arXiv): [arXiv:2312.00849](https://arxiv.org/abs/2312.00849) - Yu et al., *RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback*, CVPR 2024.
- Code & preference data: [https://github.com/RLHF-V/RLHF-V](https://github.com/RLHF-V/RLHF-V)
- Base model: Yu et al., *Muffin: Reformulating Vision-Language Pre-training as Multi-modal Feature Alignment*, 2023.
- Direct Preference Optimization: Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- LLaVA-RLHF (compared baseline): Sun et al., *Aligning Large Multimodal Models with Factually Augmented RLHF*, 2023.
- Fine-grained RLHF (related, not benchmarked): Wu et al., *Fine-Grained Human Feedback Gives Better Rewards for Language Model Training*, NeurIPS 2023.
- Object HalBench: Rohrbach et al., *Object Hallucination in Image Captioning*, EMNLP 2018.
- MMHal-Bench: Sun et al., 2023 (released alongside LLaVA-RLHF).
