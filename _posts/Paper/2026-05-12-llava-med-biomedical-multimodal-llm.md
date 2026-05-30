---
title: "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day"
excerpt: "A two-stage curriculum on PubMed figure-caption pairs and GPT-4-distilled instruction data yields closed-set VQA SoTA (84.19 VQA-RAD, 91.21 PathVQA) and a heavily self-referential 50.2% chat score against GPT-4 — trained in under 15 hours on 8x A100."
categories:
  - Paper
  - LLM
permalink: /paper/llava-med-biomedical-multimodal-llm/
tags:
  - LLaVA-Med
  - Visual-Instruction-Tuning
  - Multimodal-LLM
  - Biomedical-VQA
  - Self-Instruct
  - GPT-4
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- LLaVA-Med adapts general-domain LLaVA to biomedicine with a two-stage curriculum: 600K PubMed image-caption pairs for vocabulary alignment (projection-only update), then 60K GPT-4-distilled multi-turn dialogues for instruction following (LM + projection update, vision encoder frozen). Total wall clock is **~15 hours on 8x A100**.
- After downstream fine-tuning, LLaVA-Med (BioMedCLIP + 7B LM) hits **VQA-RAD closed 84.19, SLAKE closed 86.78, PathVQA closed 91.21** — new SoTA on three closed-set columns. Open-set "SoTA" on SLAKE (87.11) is reported under **token recall**, not the classification accuracy used by prior baselines.
- The headline **50.2% chat score against language-only GPT-4** is structurally circular: GPT-4 generates the training data, GPT-4 generates the test questions, GPT-4 is the judge, and GPT-4-with-caption is the reference upper bound. The closed-set VQA numbers are real; the chat number is not a meaningful capability claim.

## Motivation

General-domain large multimodal models (LLaVA, multimodal GPT-4) behave like a layperson on biomedical images — refusing or hallucinating ("ribcage with wires coming out" on a chest X-ray). Prior biomedical VQA work side-steps this by casting the task as closed-set classification over a fixed answer vocabulary, which does not generalize to open-ended clinical inquiry. Two enablers had just arrived when this paper was written: PMC-15M, a 15M PubMed image-text corpus two orders of magnitude larger than MIMIC-CXR, and the visual-instruction-tuning recipe demonstrated by LLaVA. The gap LLaVA-Med fills is: can cheap, scalable PubMed supervision plus GPT-4-driven self-instruct bootstrap an open-ended biomedical visual assistant **without manual annotation**?

## Core Innovation

- **A self-instruct pipeline that hides the image from GPT-4.** Stage-2 dialogues are produced by feeding language-only GPT-4 the figure caption plus the in-line "citance" sentences from the PubMed article, with a system prompt telling it to write as if it can see the image. No human annotators, no image-aware LMM. This is the cheap-supervision trick — and, as we will see in the audit, the source of the paper's most circular evaluation.
- **A two-stage biomedical curriculum.** Stage 1 (concept alignment) updates only the linear projection on 600K caption pairs, treating Stage 1 as "expanding the vocabulary of aligned image-text tokens to biomedicine." Stage 2 (instruction tuning) unfreezes the LM and trains on 60K self-instruct dialogues balanced across CXR, CT, MRI, histopathology, and gross pathology.
- **Operational claim — one day on 8x A100.** 7h Stage-1 + 8h Stage-2 = ~15h wall clock. This is the most replicable thing in the paper.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | Trained in <15 hours on 8x A100 | Table 5: 7h Stage-1 + 8h Stage-2 | Self-reported, single run | ⭐⭐⭐ |
| C2 | Curriculum (Stage-1 then Stage-2) is necessary | Stage-1-only chat avg 23.3 < LLaVA 36.1; never reports "skip Stage-1, do Stage-2 only" — the necessary counterfactual | Chat eval + 3 VQA | ⭐⭐ |
| C3 | 60K-IM (with in-line mentions) > 60K without | Overall chat 50.2 vs 49.4 (Δ=0.8 pt); ablation 75.40 vs 58.60 confounds with different FT epochs | Chat + VQA | ⭐ |
| C4 | SoTA on certain VQA benchmarks | New closed-set SoTA on VQA-RAD (84.19) and PathVQA (91.21); SLAKE open-set 87.11. Not SoTA on VQA-RAD-Open (Prefix-T 84.30) or PathVQA-Open (Q2A 54.85). Open-set uses token recall vs prior classification accuracy. | VQA-RAD, SLAKE, PathVQA | ⭐⭐ |
| C5 | Reaches 50.2% of GPT-4 chat performance | Table 1 overall = 50.2 | 193-Q internal benchmark | ⭐ — **circular evaluation** (see Honest assessment) |
| C6 | Recipe generalizes to other vertical domains | None — speculative Discussion paragraph | — | ⭐ |
| C7 | Zero-shot Chinese understanding | Table 6: 2 hand-picked examples | SLAKE bilingual | ⭐ |
| C8 | BiomedCLIP vision encoder > CLIP | FT-9 avg 75.40 vs 73.90 (Δ=1.5) | 3 VQA | ⭐⭐ |
| C9 | Self-instruct yields diverse instructions | Figure 2 verb-noun cloud + modality pie | Descriptive | ⭐⭐ |
| C10 | Domain-specific instruction-tuning beats LLaVA | Large gaps across Table 4b vs LLaVA | 3 VQA + chat | ⭐⭐⭐ |

**Honest assessment.** The strongest claims are operational and closed-set: training cost (C1), the negative result that Stage-1 alone collapses instruction-following (C2 partially), and that domain instruction-tuning beats LLaVA's general-domain baseline on closed-set VQA (C10). The weakest claim is the abstract's headline — **C5: the 50.2% chat number is GPT-4 evaluating GPT-4-distilled outputs against a GPT-4 reference on GPT-4-generated questions.** Three measurement biases follow:

1. **Stylistic mimicry rewarded by the judge.** GPT-4 prefers GPT-4-shaped responses; the relative score advantages students who imitate GPT-4's hedging and verbosity over students with better visual grounding.
2. **Caption leakage masquerading as visual supervision.** Stage-2 answers are GPT-4 fabrications conditioned on the caption. Models learn to "see" things only the caption says — supervised hallucination. The Table 2 LLaVA-Med output for the CXR example exhibits exactly this: it confidently lists "endotracheal tubes, central venous catheters, and/or pulmonary artery catheters" while the GPT-4 reference (also working from caption) only says "endotracheal tube."
3. **No ground-truth visual labels.** GPT-4 never sees the image, so caption-image mismatch (common in PubMed) becomes silent label noise.

What is missing: variance across seeds, statistical significance on the small chat deltas (60K vs 60K-IM = 0.8 pt), an external chat eval not derived from self-instruct, blinded clinician evaluation, hallucination/grounding metrics, and the necessary "Stage-2-only from LLaVA" ablation.

## Method & Architecture

![LLaVA-Med two-stage curriculum and self-instruct data construction (page 5 of the paper)](/assets/images/paper/llava-med/page_005.png)
*Figure 1: The full Stage-1 -> Stage-2 -> Downstream pipeline lives on page 5 of the paper rather than as an isolated diagram. Stage 1 aligns 600K PubMed caption pairs through a frozen vision encoder and frozen LM, training only the linear projection W. Stage 2 unfreezes the LM and trains on 60K GPT-4-distilled multi-turn dialogues balanced across CXR, CT, MRI, histopathology, and gross pathology. The image montage shows representative samples from each of the five modalities used to balance Stage-2 sampling.*

### 1. Base model and frozen components

- Vision encoder: CLIP ViT-L/14, or **BioMedCLIP** in the best variant (frozen at all stages).
- Linear projection W from vision tokens to LM input space.
- Causal LM: LLaMA-7B / Vicuna-7B (also a 13B variant — gains are essentially flat, see Results).

### 2. Stage 1 — Biomedical Concept Alignment (600K pairs)

Sample 600K image-caption pairs from PMC-15M. For each pair $(X_v, X_c)$, draw a question $X_q$ from a fixed list (11 brief-description prompts for captions <30 words, 16 detailed prompts otherwise) and format as a single turn:

$$\text{Human: } X_q\, X_v\; \texttt{<STOP>} \quad \text{Assistant: } X_c\; \texttt{<STOP>}$$

Train with autoregressive next-token loss on the assistant span only. **Only the projection W is updated.** 1 epoch, batch 128, ~7h on 8x A100 40G. Framed as "expanding the vocabulary of aligned image-text tokens to biomedicine."

### 3. Stage 2 — Self-Instruct Biomedical Instruction-Following (60K dialogues)

Filter PMC-15M to single-plot figures, sample 60K image-text pairs balanced across the five modalities above. For each, prompt **language-only GPT-4** with the caption plus in-line citance sentences (the "60K-IM" variant) and few-shot examples. The system prompt instructs GPT-4 to **write as if it can see the image** and to avoid quoting caption-specific numbers. Output is 2-3+ turns of user/assistant dialogue. Three data variants are produced for ablation: 10K (no citance), 60K (no citance), and 60K-IM (with in-line citances — the best).

### 4. Stage 2 training

Unfreeze the LM and projection; vision encoder remains frozen. 3 epochs on 60K-IM, batch 128, ~8h on 8x A100. Total curriculum = ~15h.

### 5. Downstream fine-tuning

Continue training on each of VQA-RAD / SLAKE / PathVQA for 1, 3, 9, 15, or 18 epochs. Closed-set questions are prompted with the candidate answer list inline; open-set questions are unconstrained free-form generation.

### 6. Self-instruct figure example

![A multi-turn GPT-4-distilled instruction sample](/assets/images/paper/llava-med/page_004.png)
*Figure 3: A representative GPT-4-generated multi-turn instruction-following sample built from a single PubMed CT figure plus caption plus in-line citances. GPT-4 never sees the image — it writes as if it can.*

## Experimental Results

### Closed-set / open-set fine-tuned (Table 4a)

| Method | VQA-RAD Open / Closed | SLAKE Open / Closed | PathVQA Open / Closed |
|---|---|---|---|
| LLaVA (general) | 50.00 / 65.07 | 78.18 / 63.22 | 7.74 / 63.20 |
| LLaVA-Med (LLaVA init, CLIP) | 61.52 / 84.19 | 83.08 / 85.34 | 37.95 / 91.21 |
| LLaVA-Med (Vicuna init, CLIP) | 64.39 / 81.98 | 84.71 / 83.17 | 38.87 / 91.65 |
| **LLaVA-Med (BioMedCLIP, 7B)** | **64.75 / 83.09** | **87.11 / 86.78** | **39.60 / 91.09** |
| VL Encoder-Decoder (Bazi 2023) | 71.49 / 82.47 | — | 71.49 / 85.61 |
| Q2ATransformer (Liu 2023) | 79.19 / 81.20 | — | 54.85 / 88.85 |
| Prefix T. Medical LM (van Sonsbeek 2023) | 84.30 / 82.01 | — | 40.00 / 87.00 |
| PubMedCLIP | 60.10 / 80.00 | 78.40 / 82.50 | — |
| BiomedCLIP | 67.60 / 79.80 | 82.05 / 89.70 | — |
| M2I2 (Li 2022) | 66.50 / 83.50 | 74.70 / 91.10 | 36.30 / 88.00 |

New closed-set SoTA on **VQA-RAD (84.19)** and **PathVQA (91.21 / 91.65)**, and new open-set "SoTA" on **SLAKE (87.11)** — keeping in mind the open-set comparison is methodologically inconsistent: LLaVA-Med scores **token recall** against free-form generation while prior methods score **classification accuracy** over a fixed answer set. The authors explicitly flag this in §5.2. Not SoTA on VQA-RAD-Open (84.30 by Prefix-T) or PathVQA-Open (54.85 by Q2ATransformer).

### Chat relative score against language-only GPT-4 (Table 1)

| Variant | Conversation (143) | Description (50) | Overall (193) |
|---|---|---|---|
| LLaVA | 39.4 | 26.2 | 36.1 |
| LLaVA-Med Stage-1 only | 22.6 | 25.2 | 23.3 |
| LLaVA-Med Stage-1 + 10K | 42.4 | 32.5 | 39.9 |
| LLaVA-Med Stage-1 + 60K | 53.7 | 36.9 | 49.4 |
| **LLaVA-Med Stage-1 + 60K-IM** | **55.1** | 36.4 | **50.2** |

The 50.2 number is the abstract's headline. It is also — to repeat — GPT-4 vs GPT-4-distilled student on GPT-4-generated questions, with no human evaluation and no variance reporting.

### Ablations worth knowing (Table 4b)

- Stage-1 alone collapses instruction-following: average ~13-15 vs LLaVA's 35.2. Concept alignment without Stage-2 destroys chat utility — Stage 2 is doing the heavy lifting.
- Stage-2 epochs matter much more than Stage-2 data scale: 1 -> 3 epochs lifts no-FT avg from ~41 -> ~58; 10K -> 60K (fixed 3 epochs) only 35.45 -> 40.72.
- Downstream FT helps a lot: 60K-IM at FT-1 = 46.73, FT-3 = 65.30, FT-9 = 73.90, FT-15 = 73.88 — diminishing past 9 epochs.
- **Scaling LM 7B -> 13B is flat: 73.90 -> 74.05.** Paper frames this as positive; it actually argues against scaling for this recipe.
- **Vision encoder matters more than LM size: CLIP -> BioMedCLIP lifts 73.90 -> 75.40.**
- No seeds, no variance, no confidence intervals anywhere.

### Qualitative

![Hospital-day-2 post-intubation chest X-ray used in Table 2 chat comparison](/assets/images/paper/llava-med/page_007.png)
*Figure 4: Chest X-ray from the Table 2 chat comparison between LLaVA, LLaVA-Med, and language-only GPT-4. LLaVA-Med produces a clinically plausible description; LLaVA hallucinates wires. Note LLaVA-Med over-lists tubes ("ETT, central venous catheters, and/or pulmonary artery catheters") relative to the GPT-4 reference — likely caption-leakage-style supervised hallucination.*

![Abdominal CT for zero-shot Chinese question answering on SLAKE](/assets/images/paper/llava-med/page_010.png)
*Figure 5: Abdominal CT used in Table 6 to demonstrate zero-shot Chinese-question answering on SLAKE. The Chinese capability is attributed to LLaMA/Vicuna pretraining, not to any biomedical Chinese supervision in LLaVA-Med.*

![Coronal abdominal-pelvic CT used in Table 9 appendix](/assets/images/paper/llava-med/page_017.png)
*Figure 6: Coronal abdominal-pelvic CT used in Table 9 (appendix) for the "pelvic mass" chat comparison between LLaVA and LLaVA-Med.*

## Limitations

Authors acknowledge:

- LLaVA-Med exhibits hallucinations and weak in-depth reasoning.
- Open-set scores use token recall rather than classification accuracy — not directly comparable to the open-set columns reported by prior SoTA.
- Discriminative SoTAs have a structural advantage from constrained answer sets on classification VQA.

Authors do **not** address:

- The circular self-instruct evaluation (training data, test questions, and judge all derived from GPT-4-with-caption; reference upper bound is GPT-4 with the same caption).
- No human / clinician rating despite a clinical use case.
- No safety eval (refusal behavior, dangerous advice, dosage errors).
- No PHI / data-leakage analysis of PMC content used for training.
- No variance reporting — single-seed numbers throughout.
- The missing "Stage-2-only from LLaVA" baseline that would settle whether Stage-1 is necessary or merely cheap.
- LM 7B -> 13B is flat (74.05 vs 73.90 at FT-9, 60K-IM); the implication that scaling helps is not supported.
- Five-modality coverage skips ultrasound, dermatology, ophthalmology, endoscopy, and any volumetric 3D radiology.
- No comparison against Med-PaLM-M or Med-Flamingo on shared benchmarks (concurrent at preprint v1).

## Why It Matters for Medical AI

LLaVA-Med set the template for "self-instruct your way to a domain LMM": filter a domain image-text corpus, hand a text-only frontier model the captions, ask it to roleplay as if it can see the image, and train a student. That recipe has been copied widely. The medical-AI takeaway is bifocal:

- **Use the closed-set VQA SoTA results.** VQA-RAD 84.19, PathVQA 91.21 are real, replicable numbers from a 15-hour training run. The BiomedCLIP > CLIP > scale-the-LM ordering is a useful prior for anyone fine-tuning a biomedical LMM.
- **Do not use the 50.2% chat number as a capability claim.** Treat self-instruct chat scores against the same model family that generated the data as smoke tests, not evaluations. The community has since converged on this view, but the abstract still reads as if 50.2% means LLaVA-Med is "halfway to GPT-4."

For practitioners: the strongest signal here is that **domain vision encoders (BioMedCLIP) beat scaling the LM**, and that **Stage-2 instruction tuning carries the recipe** — both are cheap interventions worth keeping when you adapt this pipeline to other modalities.

## References

- Paper: [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day (arXiv:2306.00890)](https://arxiv.org/abs/2306.00890)
- Code & model: [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- Base model: [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- Domain vision encoder: [BiomedCLIP / PMC-15M (Zhang et al., 2023)](https://arxiv.org/abs/2303.00915)
- Benchmarks: [VQA-RAD (Lau 2018)](https://www.nature.com/articles/sdata2018251), [SLAKE (Liu 2021)](https://arxiv.org/abs/2102.09542), [PathVQA (He 2020)](https://arxiv.org/abs/2003.10286)
- Concurrent biomedical LMMs: [Med-PaLM-M](https://arxiv.org/abs/2307.14334), [Med-Flamingo](https://arxiv.org/abs/2307.15189)
