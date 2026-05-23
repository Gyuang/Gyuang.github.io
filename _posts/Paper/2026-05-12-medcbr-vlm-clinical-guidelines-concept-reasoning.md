---
title: "Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning"
excerpt: "MedCBR pairs a guideline-conditioned CLIP fine-tune with a frozen Qwen3-8B reasoner to hit 94.2% AUROC on BUS-BRA and 84.0% on CBIS-DDSM, with 86.1% accuracy on CUB-200."
categories:
  - Paper
  - Pathology
  - LLM
permalink: /paper/medcbr-vlm-clinical-guidelines-concept-reasoning/
tags:
  - MedCBR
  - Concept Bottleneck Models
  - Vision-Language Models
  - CLIP
  - BI-RADS
  - Clinical Reasoning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- MedCBR rebuilds the concept bottleneck as a three-stage pipeline — an LVLM rewrites concept labels into guideline-conformant reports, CLIP is fine-tuned with InfoNCE + a diagnosis head + per-concept MLP heads, and a **frozen Qwen3-8B** reasoner consumes $(\hat y, \hat c, G)$ to emit BI-RADS-anchored explanations.
- Headline numbers: **94.2% AUROC on BUS-BRA, 84.0% AUROC on CBIS-DDSM, 86.1% accuracy on CUB-200** — beating CBM, CLIP CBM, P-CBM(-H), Label-free CBM, and AdaCBM, and edging CLIP ViT-L/14 LAION-2B by ~0.7-1.6 AUROC.
- The honest read is harsher than the abstract: the guideline-conditioning lift on the medical datasets is **only +0.2 to +1.3 AUROC**, mostly within std; MedCBR's **sensitivity sits below the radiologist on both medical datasets** (82.5 vs 93.9, 78.7 vs 90.8); the reasoning quality study is **n=20 cases graded by a single radiologist**; and the "guideline text" itself is a ChatGPT paraphrase of BI-RADS with no audited fidelity check.

## Motivation

Concept Bottleneck Models are the canonical interpretable-by-design recipe for medical imaging — predict human-readable BI-RADS descriptors, then run a linear head over them. Two practical problems block clinical reasoning. First, intermediate concepts like "irregular margin" or "posterior shadowing" are *risk indicators*, not deterministic diagnoses — a clinician integrates them against published guidelines (BI-RADS Atlas, ACR Appropriateness Criteria), but a CBM just sums them with a linear weight. Second, the human concept annotations themselves are noisy and incomplete due to inter-observer variability.

MedCBR positions itself as the "guideline-aware" CBM: it injects BI-RADS excerpts at training time (as the text side of the contrastive alignment) and at inference time (as a prompt to a downstream reasoning LLM). The aim is explanations that are not just textually fluent but **auditable against a named clinical standard**, with the reasoning step able to in principle override the concept predictor when concepts conflict with the guideline.

## Core Innovation

Three pieces, each modest on its own but tightly coupled:

1. **Guideline-driven concept enrichment.** A pretrained LVLM is given the image, the positive concept set $c^+$, the label $y$, and the relevant BI-RADS / Sibley section, and emits a free-text report $r$ that turns the binary concept vector into a guideline-grounded narrative. This $r$ becomes the text side of CLIP contrastive learning — not raw concept names, not radiology reports, but synthetic guideline-conformant text.

2. **Multi-task CLIP fine-tune.** Both encoders are pretrained CLIP, jointly trained with symmetric InfoNCE plus a linear diagnosis head and $N_c$ lightweight 2-layer MLP **per-concept adapters**. The bottleneck is no longer "linear head over concepts" but "shared visual feature with a diagnosis head and a bank of concept heads."

3. **Frozen reasoning LLM.** Qwen3-8B (no fine-tuning) is prompted with task instruction, the predicted label $\hat y$, the predicted concept confidences $\hat c$, and the guideline text $G$, and produces a step-by-step justification that explicitly cross-references concepts against $G$ and emits a BI-RADS category + next-step recommendation.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | MedCBR beats concept-based baselines on cancer detection | Table 1: +9.4 AUROC vs CBM on BUS-BRA, +4.4 vs CBM on CBIS-DDSM, +6.3 vs AdaCBM | ⭐⭐⭐ Multi-run 5-fold CV, tight std (~0.4-0.7), large margins over interpretable baselines. |
| C2 | Matches/beats strong vision-only CLIP encoders without sacrificing transparency | Table 1: +0.7 over CLIP ViT-L/14 LAION-2B on BUS-BRA, +1.6 on CBIS-DDSM | ⭐⭐ The margin is real but slim and the std bars on the strong CLIP rows (±1.2-2.5) overlap. |
| C3 | Generalizes beyond medical to natural images | Table 1: 86.1% on CUB-200, +10% over Label-free CBM | ⭐⭐ Single non-medical dataset, holdout split rather than CV. |
| C4 | Multi-task supervision is necessary (vs a CBL bottleneck) | Table 2: CLIP+CBL 91.8 → CLIP+MTL+Guideline 94.2 on BUS-BRA | ⭐⭐⭐ Consistent gain across all three datasets. |
| C5 | Guideline conditioning helps | Table 2: +0.2 to +5.0 AUROC depending on dataset | ⭐⭐ The headline-marketed contribution is also the weakest: medical gains are **0.2-1.3**, often inside std. The big lift (+3.8 to +5.0) is on CUB-200, not on ultrasound or mammography. |
| C6 | Higher per-concept AUROC than CBM / BiomedCLIP / LAION-2B CLIP | Fig. 3 bar plots | ⭐⭐ Visual evidence only; no per-concept numeric table. |
| C7 | Off-the-shelf medical VLMs are "fluent but ungrounded" | Table 3: MedGemma-4B / Qwen2.5VL-7B near-ceiling rubric scores but sensitivity 35-45% | ⭐⭐ Compelling pattern, but the rubric is graded by a single radiologist with no inter-rater statistic. |
| C8 | MedCBR matches a radiologist on F1 and exceeds on specificity | Table 3: F1 84.3 / 74.4 vs 75.1 / 70.9; Spec 94.3 / 76.0 vs 75.8 / 52.3 | ⭐⭐ True for the named metrics, but **sensitivity is below the radiologist on both datasets** (82.5 vs 93.9, 78.7 vs 90.8) — the clinically dangerous direction — and the paper does not foreground this. |
| C9 | Reasoning is "grounded and reliable" because it conditions on $G$ | Conceptual argument + qualitative Fig. 4/5 case studies | ⭐ No hallucination rate, no faithfulness metric; n=20 cases per dataset. |
| C10 | The reasoner can "challenge" the concept predictor when concepts conflict with the guideline | One qualitative Blue Jay example, Fig. 5 | ⭐ Single anecdote. |

**Honest summary.** The AUROC story (C1, C4) is genuinely well-supported — 5-fold CV, tight std, big margins over interpretable baselines. The matches-CLIP-ViT-L/14 claim (C2) is real but slim and inside overlapping std bars. The most-marketed contribution — guideline-grounded reasoning (C5, C7-C9) — is the weakest evidentially. The guideline ablation on the medical datasets adds **0.2-1.3 AUROC**, often inside the noise band; the bulk of the win comes from the multi-task CLIP fine-tune, and the paper does not cleanly separate "guideline helped" from "more text supervision helped." Worse, the rubric scoring of reasoning quality rests on **n=20 cases per dataset graded by a single radiologist** with no inter-rater agreement, no faithfulness/hallucination quantification, and the "clinical guideline" text itself is a ChatGPT paraphrase of the BI-RADS Atlas with no audit of paraphrase fidelity. Finally, the clinical-utility comparison is asymmetric: MedCBR's sensitivity is below the radiologist on both medical datasets, a fact the paper acknowledges briefly while leading with "exceeds radiologist specificity."

## Method & Architecture

![MedCBR three-stage pipeline](/assets/images/paper/vl-concept-reasoning-clinical/page_003.png)
*Figure 1: MedCBR's three stages. (a) LVLM rewrites concept labels into guideline-conformant reports; (b) CLIP is fine-tuned with InfoNCE + diagnosis head + per-concept MLP heads; (c) a frozen Qwen3-8B consumes $(\hat y, \hat c, G)$ and produces BI-RADS-anchored explanations.*

Formally, the dataset is $D = \{(x, c, y)\}$ with image $x$, binary concept vector $c \in \{0,1\}^{N_c}$, label $y \in \{\text{benign}, \text{malignant}\}$. Classical CBM predicts $\hat y = q(g(f(x)))$. MedCBR replaces this with

$$\hat y = q(h),\quad \hat c = g(h),\quad E = R(\hat y, \hat c, G)$$

where $h = f(x)$ is the shared visual feature and $G$ is the clinical guideline text.

1. **(a) Guideline-driven concept enrichment.** $r = T(x, c, G)$ with $T$ a pretrained LVLM. The prompt contains the image, the positive concept set $c^+ = \{c_j : c_j = 1\}$, the label $y$, and the relevant BI-RADS or Sibley section. The LVLM must (i) describe the visual findings tied to those concepts and (ii) summarize their diagnostic implications according to $G$.

2. **(b) Vision-language concept modelling.** Backbone is CLIP with vision encoder $f^v_\theta$ and text encoder $f^t_\phi$. For pair $(x, r)$, compute $h_v = f^v_\theta(x)$, $h_t = f^t_\phi(r)$, then symmetric InfoNCE:

   $$\mathcal{L}_{\text{CLIP} } = -\log \frac{\exp(\mathrm{sim}(h_v, h_t)/\tau)}{\sum_{r'} \exp(\mathrm{sim}(h_v, h_{t'})/\tau)}$$

   On top of $h_v$, two heads: a linear diagnostic head $W_y$ supervised by $\mathcal{L}_y = \mathcal{L}_{CE}(W_y h_v, y)$, and $N_c$ lightweight 2-layer MLP per-concept adapters $g_{\psi_i}$ supervised by $\mathcal{L}_c = (1/N_c)\sum_i \mathcal{L}_{CE}(g_{\psi_i}(h_v), c_i)$. Full objective:

   $$\mathcal{L}_{\text{MedCBR} } = \lambda \mathcal{L}_{\text{CLIP} } + \mu \mathcal{L}_y + \nu \mathcal{L}_c$$

3. **(c) Concept-based clinical reasoning.** A frozen Qwen3-8B is prompted with $\pi = (Q, \hat y, \hat c, G)$: task instruction $Q$, $\hat y = \sigma(f^y_\theta(h_v))$, concept confidences $\hat c_i = \sigma(f^{c_i}_\theta(h_v))$, and the relevant BI-RADS excerpt. The LRM is instructed to interpret each concept's contribution, cross-check against $G$, and emit a step-by-step justification with a BI-RADS category and next-step recommendation. **No fine-tuning** of the LRM.

**Training details.** PyTorch 2.1, single NVIDIA L40S (24 GB), AdamW, LR $1\times10^{-5}$, cosine schedule over 150 epochs with 10-epoch warmup. Images cropped to lesion ROIs, resized to 224×224, augmented with random translation, rotation, flip.

**Guideline construction (a soft point).** ChatGPT is used to paraphrase BI-RADS Atlas sections for BUS-BRA and CBIS-DDSM, and to condense the *Sibley Field Guide to Birds* for CUB-200. The paper does not audit these paraphrases against the verbatim BI-RADS text — so the "guideline conditioning" is really "conditioning on an LLM-paraphrased approximation of BI-RADS."

## Experimental Results

Main quantitative results (Table 1, 5-fold CV on the medical sets; CUB-200 is holdout). MedCBR row in bold.

| Method | BUS-BRA AUROC | BUS-BRA Bal.Acc | CBIS-DDSM AUROC | CBIS-DDSM Bal.Acc | CUB-200 Acc |
|---|---|---|---|---|---|
| CLIP RN50 | 87.4 ±1.4 | 80.9 ±1.7 | 73.3 ±1.7 | 67.7 ±1.6 | 60.1 ±1.1 |
| CLIP ViT-B/32 | 90.1 ±1.5 | 83.1 ±1.9 | 79.8 ±0.9 | 72.3 ±0.9 | 69.0 ±0.4 |
| CLIP ViT-L/14 (LAION-2B) | 93.5 ±1.2 | 88.1 ±1.7 | 82.4 ±2.5 | 75.5 ±2.4 | 85.7 ±0.2 |
| SigLIP | 90.8 ±0.8 | 85.0 ±1.5 | 82.3 ±1.4 | 74.6 ±1.3 | 78.6 ±0.6 |
| BiomedCLIP | 89.0 ±0.5 | 82.1 ±0.9 | 77.9 ±0.5 | 71.1 ±0.7 | — |
| CBM | 84.8 ±2.3 | 79.1 ±2.0 | 79.6 ±1.3 | 73.3 ±1.3 | 62.9 |
| CLIP CBM | 91.8 ±0.9 | 86.4 ±1.7 | 81.8 ±1.0 | 75.8 ±0.9 | 67.0 ±0.4 |
| P-CBM | 80.1 ±0.5 | 73.9 ±0.3 | 72.7 ±0.0 | 67.0 ±0.5 | 59.6 |
| P-CBMh | 87.0 ±1.7 | 75.0 ±3.0 | 77.2 ±1.8 | 69.3 ±1.0 | 61.0 |
| Label-free CBM | 60.0 ±2.8 | 59.1 ±2.1 | 70.0 ±0.4 | 65.3 ±0.5 | 74.3 ±0.3 |
| AdaCBM | 87.9 ±0.5 | 80.5 ±0.8 | 75.6 ±3.0 | 68.9 ±2.4 | 69.8 ±0.2 |
| **MedCBR** | **94.2 ±0.4** | **89.0 ±0.9** | **84.0 ±0.7** | **76.4 ±0.6** | **86.1 ±0.2** |

Two patterns stand out. (i) MedCBR has by far the **tightest std** of any row in the table (±0.4 on BUS-BRA AUROC, ±0.2 on CUB), suggesting the multi-task CLIP fine-tune is stabilizing. (ii) Strong vision-only CLIP encoders (ViT-L/14 LAION-2B, SigLIP) are already within striking distance — the interpretable bottleneck adds explainability without paying for it in headline accuracy, but it is **not** a step change over a well-tuned modern CLIP.

**Ablations (Table 2).** Removing the text branch and reverting to a CBL bottleneck $y = W_y \cdot g(h_v)$ drops BUS-BRA to 91.8 and CUB-200 to 67.0. Adding guideline-conformant reports recovers CBL on CUB-200 by +5.0. The full MTL+Guideline (MedCBR) gives the headline 94.2 / 84.0 / 86.1. **The largest guideline lift is on CUB-200 (+3.8 to +5.0), not on the medical datasets (+0.2 to +1.3)** — exactly the inverse of what you'd hope from a paper titled "Vision-Language Models Encode Clinical Guidelines."

**Clinical-utility study (Table 3, BUS-BRA / CBIS-DDSM).** MedCBR-8B reaches Sens. 82.5 / 78.7, Spec. 94.3 / 76.0, F1 84.3 / 74.4 vs. radiologist 75.1 / 70.9 F1; concept-reasoning rubric (CIntS / CIgS / BAS) 95.4 / 98.3 / 86.0 on BUS-BRA. Off-the-shelf medical VLMs (MedGemma-4B, Qwen2.5VL-7B) get near-ceiling guideline-conformity scores but only 35-45% sensitivity — fluent narratives that miss cancers. The headline F1 win is real but the sub-radiologist sensitivity on **both** datasets is the load-bearing caveat. The reasoning rubric (CIntS / CIgS / BAS) is also graded by a **single radiologist on 20 cases** per dataset, with no inter-rater agreement statistic.

**Per-concept performance (Fig. 3).** MedCBR > LAION-2B CLIP > BiomedCLIP > CBM on per-concept AUROC across both medical datasets. LAION-2B CLIP degrades specifically on modality-specific cues (echogenicity, posterior features, calcifications) — these are under-represented in natural-image web corpora, and the multi-task fine-tune is what recovers them.

**Qualitative reasoning examples.**

![Ultrasound BI-RADS 5 reasoning](/assets/images/paper/vl-concept-reasoning-clinical/page_007.png)
*Figure 2: BUS-BRA case — MedCBR weighs posterior shadowing, indistinct margins, and microlobulated margins against a benign regular shape and outputs BI-RADS 5 with a next-step recommendation.*

![CUB-200 reasoner contradicting concept predictor](/assets/images/paper/vl-concept-reasoning-clinical/page_008.png)
*Figure 5: CUB-200 — the reasoner contradicts a concept prediction ("underparts color red") when it conflicts with the Sibley field-guide entry for Blue Jay. This is the paper's strongest qualitative argument for guideline-grounded override, but it is a single anecdote.*

## Limitations

- **Sensitivity below radiologist** on both medical datasets (82.5 vs 93.9 on BUS-BRA, 78.7 vs 90.8 on CBIS-DDSM). This is the clinically dangerous direction (missed cancers), and the paper foregrounds the specificity and F1 wins instead.
- **Guideline-conditioning lift on the medical datasets is +0.2 to +1.3 AUROC**, often inside one std. The bulk of MedCBR's headline win comes from the multi-task CLIP fine-tune, not from the guideline text. The paper does not cleanly separate these two effects.
- **The "clinical guideline" is ChatGPT-paraphrased BI-RADS**, not verbatim Atlas text, and the paraphrase fidelity is not audited. Any clinical-grounding claim rests on the assumption that the LLM rewrite preserved the BI-RADS semantics — an assumption that should be checked, especially for a paper whose title is "Vision-Language Models Encode Clinical Guidelines."
- **Reasoning evaluation is single-rater, n=20.** The CIntS / CIgS / BAS rubric scores rest on one radiologist grading 20 cases per dataset. No inter-rater agreement statistic. No quantitative hallucination or faithfulness metric on the LRM outputs.
- **Two-class diagnosis only** (benign/malignant) on the medical datasets, no granular BI-RADS-category prediction or staging, despite the BI-RADS framing.
- **Single-site datasets per modality.** BUS-BRA is a single cohort, CBIS-DDSM is a US screening dataset; no cross-institution external test set.
- **Fixed reasoner.** Qwen3-8B is the only LRM studied; no ablation over reasoner size or family, so the reasoning quality numbers are entangled with Qwen3-8B's own behavior.
- **Concept-detection improvement** is shown only as Fig. 3 bar charts; no per-concept numeric table.

## Why It Matters for Medical AI

The structural lesson is useful even if the headline framing is over-sold. For interpretable medical imaging, MedCBR shows that (a) a multi-task CLIP fine-tune with **per-concept MLP heads instead of a single linear bottleneck** can close the explainable-vs-black-box gap on breast ultrasound and mammography while keeping concept-level transparency, and (b) handing the bottleneck off to a **frozen reasoning LLM** prompted with the guideline text produces narratives that are pleasant to read and at least sometimes correctly override the concept predictor. Both ingredients are cheap to graft onto existing CBM pipelines.

But the safety story is the dangerous part to copy. A system whose sensitivity sits below a radiologist on both medical datasets is not yet a clinical replacement; the paper's "matches/exceeds radiologist" framing is true only on F1 and specificity, and the rubric-graded reasoning quality rests on twenty cases and one radiologist. For teams considering this template in production, the right takeaway is: keep the multi-task fine-tune, keep the per-concept heads, keep the frozen-reasoner narrative — but invest the next research effort in a quantitative hallucination/faithfulness audit, multi-rater evaluation at meaningful sample size, paraphrase-fidelity validation of the guideline text against verbatim BI-RADS, and an external test set across institutions before any deployment claim.

## References

- Harmanani, Long, Guo, Wilson, Sabour, To, Fichtinger, Abolmaesumi, Mousavi. *Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning.* arXiv:2603.08921v1, March 2026. <https://arxiv.org/abs/2603.08921>
- Koh et al., *Concept Bottleneck Models.* ICML 2020.
- Yuksekgonul et al., *Post-hoc Concept Bottleneck Models (P-CBM, P-CBMh).* ICLR 2023.
- Oikarinen et al., *Label-free Concept Bottleneck Models.* ICLR 2023.
- Chowdhury et al., *AdaCBM.* MICCAI 2024.
- Radford et al., *CLIP — Learning Transferable Visual Models from Natural Language Supervision.* ICML 2021.
- Cherti et al., *OpenCLIP / LAION-2B reproductions.* CVPR 2023.
- Zhai et al., *SigLIP.* ICCV 2023.
- Zhang et al., *BiomedCLIP.* NEJM AI 2024.
- BUS-BRA breast ultrasound dataset [12]; CBIS-DDSM [29]; BrEaST [26]; CUB-200-2011 [38]; *Sibley Field Guide to Birds*.
- ACR BI-RADS Atlas — the verbatim guideline that the paper's training text paraphrases via ChatGPT.
