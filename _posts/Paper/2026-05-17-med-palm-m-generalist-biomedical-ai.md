---
title: "Towards Generalist Biomedical AI (Med-PaLM M)"
excerpt: "A single PaLM-E finetune (12B/84B/562B) covering 14 multimodal medical tasks beats prior SOTA on 5/12 -- with the 84B model preferred over the radiologist's own report 40.50% of the time -- but loses to specialists on MedQA, MIMIC-CXR AUC, DeepVariant Indel, and MIMIC-III summarization."
categories:
  - Paper
  - LLM-Agents
  - LLM
  - Pathology
permalink: /paper/med-palm-m-generalist-biomedical-ai/
tags:
  - Med-PaLM-M
  - PaLM-E
  - Generalist
  - Multimodal
  - MultiMedBench
  - Chest-X-ray
  - Report-Generation
  - VQA
  - Genomics
  - Instruction-Tuning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- **Med-PaLM M** is a single PaLM-E finetune (12B / 84B / 562B parameters) that handles 14 biomedical tasks -- text QA, radiology/pathology VQA, CXR classification and report generation, dermatology, mammography, and even DeepVariant pileup images for SNP/Indel calling -- all phrased as instruction-tuned text generation, without per-task architectures or adapters.
- The paper also introduces **MultiMedBench** (12 datasets / 14 tasks / ~1M samples), the canonical benchmark every subsequent generalist medical foundation model (LLaVA-Med, BiomedGPT, MedGemma) is compared against.
- Headline: **beats prior SOTA on 5/12 comparable tasks** -- MIMIC-CXR report generation Micro-F1-14 **+9.36** (44.20% -> 53.56%), Slake-VQA BLEU-1 **+14.1** (78.60% -> 92.70%), Path-VQA, VQA-RAD, VinDr-Mammo all positive -- and Med-PaLM M (84B) reports are preferred over the radiologist's own report in **40.50%** of 246 chest X-rays. But it **loses** to MedPaLM 2 on MedQA, to DeepVariant on Indel calling, to ParallelXNet on MIMIC-CXR classification AUC, and to clinical-T5 on MIMIC-III summarization. The 40.50% figure is the **84B** model, not the largest.

## Motivation

Clinical decision-making is inherently multimodal -- notes, labs, multiple imaging modalities, genomics -- yet virtually all deployed medical AI is unimodal and single-task (a mammography classifier that cannot ingest a patient's gene panel, MRI, or chart note). The authors argue the foundation-model paradigm gives, for the first time, a recipe to unify medical AI, but progress is blocked by two missing pieces: (i) a comprehensive multimodal medical benchmark and (ii) any actual generalist model competing with specialists on it. Med-PaLM M is offered as the proof of concept that closes both gaps, and it has since become the de facto reference for "generalist medical foundation model" in every downstream paper.

## Core Innovation

- **Base.** Med-PaLM M is an end-to-end finetune of PaLM-E (no adapters, no LoRA). PaLM-E wires a ViT vision encoder into a PaLM decoder; image tokens (256 per image) are projected into the language embedding space and interleaved with text. Three scales: 12B (PaLM 8B + ViT 4B), 84B (PaLM 62B + ViT 22B), 562B (PaLM 540B + ViT 22B). **The ViT does not scale from 84B to 562B** -- a load-bearing detail later.
- **Unified task spec.** Every task -- classification, VQA, report generation, summarization, even variant calling -- is reformulated as conditional text generation. Classification becomes multiple-choice QA with class labels rendered as `(A) Nevus (B) BCC ...` and the model emits the option text.
- **Text-only one-shot exemplar.** Each prompt = `instruction + 1 worked example (image replaced by the literal string "<img>") + actual question (with real image)`. The dummy-image trick keeps training memory at one image per sample while still teaching instruction-following.
- **Genomics-as-image hack.** DeepVariant pileup tensors `(100, 221, 6)` are reshaped to `(224, 224, 3)` grayscale-stacked-to-RGB. This is what lets a vision-language model do variant calling at all.
- **Training mixture.** MIMIC-CXR report generation dominates at **59.90%** of the mixture. MedQA is only 3.13%, VQA-RAD only 0.15%. Mixture ratios were tuned roughly proportional to dataset size with >=1 sample/task per batch.
- **MultiMedBench.** 12 open datasets, 14 tasks, 5 task types, ~1M samples: text QA (MedQA/MedMCQA/PubMedQA), summarization (MIMIC-III), VQA (VQA-RAD, Slake-VQA, Path-VQA), CXR report-gen + classification (MIMIC-CXR), dermatology (PAD-UFES-20), mammography (VinDr-Mammo, CBIS-DDSM), and genomics (PrecisionFDA Truth Challenge V2).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Med-PaLM M is competitive with or exceeds prior SOTA on all 14 MultiMedBench tasks. | Table 2: beats SOTA on 5/12 metrics-with-baseline, ties on several, **loses** to specialists on MedQA, MIMIC-CXR classification AUC, MIMIC-III summarization, and DeepVariant Indel. | All MultiMedBench | ⭐⭐ -- "near or exceeds" is fair, "competitive on all" is overstated. |
| C2 | Beats prior SOTA on MIMIC-CXR report generation by over 8% Micro-F1. | Table 2: 53.56 vs 44.20, Δ = **+9.36 abs.** | MIMIC-CXR | ⭐⭐⭐ |
| C3 | Outperforms prior SOTA on Slake-VQA by >10% BLEU-1 and F1. | Table 2: 92.70 vs 78.60 BLEU-1 (+14.1); 89.28 vs 78.10 F1 (+11.2). | Slake-VQA | ⭐⭐⭐ |
| C4 | Demonstrates positive task transfer from joint multi-task training. | Table 6 ablation: 53.56 -> 52.94 Micro-F1-14 when MIMIC-CXR classification is removed; effect = 0.62 abs. | MIMIC-CXR | ⭐ -- single comparison, single seed, no variance, tiny effect. |
| C5 | Zero-shot generalization to novel medical concepts (TB). | Table 4: 86.96 / 82.60 / 87.68% accuracy on Montgomery TB, n=138; specialist trained on all 138 still wins at 92.60%; **no AUC reported** (free-text decoding). | Montgomery County | ⭐⭐ -- single tiny dataset, specialist still better. |
| C6 | Emergent zero-shot multimodal CoT reasoning at scale. | Figure 3: two qualitative TB examples; 12B fails, 84B/562B succeed (with radiologist-noted errors). | Few qualitative CXRs | ⭐ -- "emergence" claimed from one binary success transition. |
| C7 | Generalizes to novel two-view CXR report generation. | Table 5: 50.54 Micro-F1-14 / 37.78 Macro / 28.30 F1-RadGraph, above single-view SOTA. | MIMIC-CXR two-view subset | ⭐⭐ -- real, but same dataset family used in training. |
| C8 | Radiologists prefer Med-PaLM M reports over reference in up to **40.50%** of cases. | Figure 4b: 84B 40.50%, 12B 34.05%, 562B 32.00%; n=246 cases, 4 raters. | MIMIC-CXR | ⭐⭐ -- India-based raters on US-ICU reports; "preferred" ≠ "correct"; **the 40.5% is the 84B, not the biggest model**. |
| C9 | Clinical error rate (0.25/report for 84B) is on par with human radiologists. | Figure 5 + comparison to Yu et al. [14]. | MIMIC-CXR | ⭐⭐ -- CIs partially overlap; no direct head-to-head with the same raters. |
| C10 | Medical finetuning is critical -- PaLM-E out-of-the-box is poor. | Table 2 PaLM-E 84B: MedQA 28.83%, MIMIC-III ROUGE-L 3.30%, Slake-VQA F1 24.53%. | All | ⭐⭐⭐ -- massive, consistent gap. |
| C11 | Scaling LLM helps language-heavy tasks; vision encoder is the bottleneck for image classification. | Table 3: monotone QA/VQA scaling; flat/non-monotone for VinDr/CBIS/PAD-UFES; **same 22B ViT shared by 84B and 562B**. | All | ⭐⭐⭐ -- internally well-controlled. |

**Honest read.** C2, C3, C10, C11 are the strong claims -- well-supported by hard numbers across consistent setups. C5--C9 are medium-strength: each rests on a single small evaluation set with no variance reporting, no multi-seed runs, no bootstrap on the specialist deltas. C4 (positive transfer ablation = sub-1-point effect) and C6 (emergence from a handful of qualitative TB cases) are weak. The most important systemic gap is **no statistical significance testing anywhere in Table 2** -- a 5/12 SOTA-beating count needs CIs to be credible. The press-quoted "AI beats radiologists 40% of the time" really means *one* of three model scales (the **84B**, not the largest) is preferred in 40.5% of single-image MIMIC-CXR snapshots judged by Indian thoracic radiologists who never saw priors or multi-view -- a long way from "clinically deployable."

## Method & Architecture

![Med-PaLM M hexagonal radar across 14 MultiMedBench tasks vs specialists](/assets/images/paper/med-palm-m/fig_p002_16.png)
*Figure 1: Med-PaLM M (shaded blue) matches or exceeds task-specific specialists (red) across the 14 tasks of MultiMedBench using a single set of weights. The radar style understates the losses on MedQA, MIMIC-CXR AUC, MIMIC-III summarization, and DeepVariant Indel calling -- see Table 2 below.*

**A. Architecture inheritance.** Med-PaLM M is end-to-end finetuned from PaLM-E at three scales:

- 12B = PaLM 8B + ViT 4B
- 84B = PaLM 62B + ViT 22B
- 562B = PaLM 540B + ViT 22B (ViT shared with 84B)

Each image is encoded into 256 tokens projected into the language embedding space and interleaved with text. All PaLM-E parameters are finetuned; no adapters, no LoRA.

**B. Unified text-generation interface.** Every task -- classification, VQA, summarization, report generation, variant calling -- is conditional text generation. Classification: render labels as MCQ options `(A) Nevus (B) Basal Cell Carcinoma ...` and decode the option text. For metrics like AUC the free-text label string must be matched back to the class list, which is why the TB zero-shot evaluation omits AUC.

**C. Instruction-tuning prompt template.** Each prompt is:

$$\text{prompt} = \text{instruction} \; + \; \text{1 worked example (image} \to \text{"<img>" string)} \; + \; \text{question (real image embedded)}$$

The text-only exemplar (i) keeps training memory at one image per sample and (ii) avoids cross-image cross-attention interference. CBIS-DDSM and PrecisionFDA use 0-shot; MedQA/MedMCQA use 2-shot text.

![Frontal CXR example used in the instruction-tuning prompt for MIMIC-CXR report generation](/assets/images/paper/med-palm-m/fig_p007_03.png)
*Figure 2 (example): a PA-view chest X-ray fed to the report-generation prompt; the model conditions on the indication text plus the text-only one-shot exemplar, then decodes the findings paragraph.*

![Skin-lesion image used in the dermatology MCQ example](/assets/images/paper/med-palm-m/fig_p007_06.png)
*Figure 2 (example): dermatology MCQ prompt -- patient history is prepended, class options are listed as A--F, and the model emits the option text. The same prompt template covers VQA, CXR classification, summarization, and DeepVariant variant calling.*

**D. Input preprocessing.** All images resized to 224x224x3 with aspect-preserving padding; grayscale stacked to 3 channels. DeepVariant's `(100, 221, 6)` pileup tensor (base / quality / strand / support) is reshaped to `(224, 224, 3)` grayscale-to-RGB -- this is the only reason a VLM can call variants at all.

**E. Training mixture (weighted, Table A.1).**

| Task | Mixture weight |
|---|---|
| **MIMIC-CXR report generation** | **59.90%** |
| MIMIC-CXR classification | 11.98% |
| MedMCQA | 6.25% |
| PAD-UFES-20 | 6.25% |
| MedQA | 3.13% |
| MIMIC-III summarization | 3.13% |
| Slake-VQA | 2.64% |
| Path-VQA | 1.90% |
| VinDr-Mammo | 1.56% |
| CBIS-DDSM | 1.56% |
| PrecisionFDA | 1.56% |
| VQA-RAD | 0.15% |

~60% of training signal is one task. Med-PaLM M is, structurally, **a CXR specialist that also does other things** -- worth keeping in mind when reading the "generalist" claim.

**F. Optimization (Table A.2).** Adafactor, β1=0.9, dropout 0.1, constant LR. LR = 5e-5 (12B/84B), 2.5e-5 (562B). Batch size 128 (12B/84B), 256 (562B). Max input 710 tokens, max output 256 tokens. Greedy decoding at inference; same prompt format as training.

## Experimental Results

**Table 2 -- Med-PaLM M vs SOTA specialists vs PaLM-E (no medical finetune).** Bold = Med-PaLM M actually beats prior SOTA.

| Task | Dataset | Metric | SOTA specialist | PaLM-E 84B | **Med-PaLM M (best)** |
|---|---|---|---|---|---|
| QA | MedQA | Accuracy | 86.50% (Med-PaLM 2) | 28.83% | 69.68% |
| QA | MedMCQA | Accuracy | 72.30% | 33.35% | 62.59% |
| QA | PubMedQA | Accuracy | 81.80% | 64.00% | 80.00% |
| Summ | MIMIC-III | ROUGE-L | 38.70% | 3.30% | 32.03% |
| Summ | MIMIC-III | F1-RadGraph | 40.80% | 8.00% | 34.71% |
| VQA | VQA-RAD | BLEU-1 | 71.03% | 59.19% | **71.27%** |
| VQA | Slake-VQA | BLEU-1 | 78.60% | 52.65% | **92.70%** |
| VQA | Slake-VQA | F1 | 78.10% | 24.53% | **89.28%** |
| VQA | Path-VQA | BLEU-1 | 70.30% | 54.92% | **72.27%** |
| VQA | Path-VQA | F1 | 58.40% | 29.68% | **62.69%** |
| Report Gen | MIMIC-CXR | Micro-F1-14 | 44.20% | 15.40% | **53.56%** |
| Report Gen | MIMIC-CXR | Macro-F1-14 | 30.70% | 10.11% | **39.83%** |
| Report Gen | MIMIC-CXR | F1-RadGraph | 24.40% | 11.66% | **26.71%** |
| Report Gen | MIMIC-CXR | BLEU-1 | 39.48% | 19.86% | 32.31% |
| Report Gen | MIMIC-CXR | CIDEr-D | 49.50% | 3.50% | 26.17% |
| Image Cls | MIMIC-CXR (5 conds) | Macro-AUC | 81.27% | 51.48% | 79.09% |
| Image Cls | VinDr-Mammo | Macro-AUC | 64.50% | 51.49% | **71.76%** |
| Image Cls | CBIS-DDSM (calc) | Macro-F1 | 70.71% | 11.37% | 67.86% |
| Genomics | PrecisionFDA | Indel-F1 | 99.40% (DeepVariant) | 53.01% | 97.04% |
| Genomics | PrecisionFDA | SNP-F1 | 99.70% | 52.84% | 99.35% |

The bolded rows give the paper's "5/12 SOTA-beating tasks" count. Note where it loses: text QA (MedPaLM 2), MIMIC-CXR AUC (ParallelXNet), MIMIC-III summarization (clinical-T5, possibly trained on MIMIC-III test data), and DeepVariant Indel. PaLM-E without medical finetuning is consistently weak -- the medical finetune is doing most of the work.

**Table 3 -- Scaling (12B / 84B / 562B).**

| Task | 12B | 84B | 562B | Pattern |
|---|---|---|---|---|
| MedQA Acc | 29.22 | 46.11 | **69.68** | Monotone (language) |
| MedMCQA Acc | -- | -- | **62.59** | Monotone (language) |
| Slake-VQA F1 | -- | -- | **89.28** | Monotone (VQA) |
| PAD-UFES-20 Macro-F1 | 78.42 | **84.32** | 77.03 | Plateaus at 84B |
| VinDr-Mammo Macro-F1 | 29.81 | **35.70** | 33.90 | Plateaus at 84B |
| MIMIC-CXR Micro-F1-14 | 51.41 | **53.56** | 51.60 | Non-monotonic |

Language-heavy tasks scale cleanly. Vision-heavy classification plateaus at 84B because **the 84B and 562B share the same 22B ViT**: the LLM grows but the vision bottleneck does not. CXR report generation is non-monotonic -- 84B beats 562B; authors speculate the 562B is undertrained and verbose.

**Table 4 -- Zero-shot TB (Montgomery County, n=138).**

| Model | Accuracy | AUC |
|---|---|---|
| Specialist (trained on all 138) | **92.60%** | reported |
| Med-PaLM M 12B (zero-shot) | 86.96% | -- |
| Med-PaLM M 84B (zero-shot) | 82.60% | -- |
| Med-PaLM M 562B (zero-shot) | **87.68%** | -- |

Encouraging but n=138 is tiny, the specialist still wins, and **AUC is not reported** (because free-text decoding doesn't yield probabilities).

**Table 5 -- Zero-shot two-view CXR report generation (trained on single-view only).**

| Setting | Micro-F1-14 | Macro-F1-14 | F1-RadGraph |
|---|---|---|---|
| Prior single-view SOTA | 44.20 | 30.70 | 24.40 |
| **Med-PaLM M two-view (zero-shot)** | **50.54** | **37.78** | **28.30** |

Above the prior single-view SOTA without training on two-view data. Genuine novel-task generalization, though still on the same MIMIC-CXR dataset family.

**Table 6 -- Positive task transfer.** Removing MIMIC-CXR classification from the mixture drops report-gen Micro-F1-14 from 53.56 -> 52.94 (Δ = 0.62) and F1-RadGraph from 26.71 -> 26.08. Real but tiny effect, single seed.

**Radiologist evaluation (n=246 CXRs, 4 raters).**

![Zero-shot CoT example on Montgomery TB CXR](/assets/images/paper/med-palm-m/fig_p013_08.png)
*Figure 3: zero-shot CoT on the Montgomery TB dataset. Med-PaLM M 84B / 562B localize the right-upper-lobe cavitary lesion; 12B fails to produce a coherent report. This is the entire empirical basis for the "emergent reasoning" claim -- a handful of qualitative cases, no quantitative comparison.*

- **Side-by-side ranking (Figure 4a).** Best-ranked rate: reference report 37.14%, Med-PaLM M 84B **25.78%**, 12B 19.49%, 562B 17.59%.
- **Pairwise preference vs reference (Figure 4b).** Med-PaLM M 84B preferred over the radiologist's own report in **40.50%** of cases (12B: 34.05%, 562B: 32.00%). **The headline 40.50% is the 84B model -- not the largest.**
- **Clinically significant error rate (Figure 5).** 84B = **0.25 errors/report** (95% CI 0.22--0.28), comparable to the Yu et al. human-radiologist baseline. 12B = 0.25 ish; 562B regresses to 0.29. Omission rates: 0.12 (12B/84B), 0.13 (562B).

![Qualitative report-generation comparison: same CXR, four reports (reference + 12B/84B/562B)](/assets/images/paper/med-palm-m/fig_p016_04.png)
*Figure 6: same chest X-ray, four reports (radiologist reference + Med-PaLM M 12B / 84B / 562B). Radiologists marked 12B with two clinically significant errors, 84B with zero, 562B with one insignificant error -- consistent with the quantitative error-rate ordering above (84B best, 562B worst).*

## Limitations

**Authors acknowledge.**

- MultiMedBench is small (~1M samples) and missing transcriptomics, proteomics, EHR time-series, ultrasound, MRI segmentation, ophthalmology.
- Vision encoder is the bottleneck for scaling multimodal medical models; medical data scarcity makes scaling harder than for text-only LLMs.
- Architecture not optimal for few-shot in-context learning.
- Real-world validation, safety, equity are out of scope.
- Model weights cannot be released.

**Additional gaps.**

- **No variance / no seeds.** Every Table 2 / Table 3 number is a single-run point estimate. No CIs, no bootstrap on the specialist deltas, no significance test anywhere -- a 5/12 SOTA-beating count is hard to evaluate without them.
- **No external-hospital / held-out-site evaluation** for the imaging tasks. Everything is the same MIMIC / CBIS / VinDr / PAD-UFES splits used in training.
- **No fairness or subgroup analysis** (sex, race, age, scanner type). Three of four radiologists for human evaluation are India-based; MIMIC-CXR is a Boston ICU -- known demographic and reporting-style mismatch.
- **No calibration / uncertainty.** Generative classifier emits a label string with no confidence; AUC is unavailable for several tasks (notably TB zero-shot, n=138).
- **Single-image inference only at training.** Multi-image reasoning is anecdotal.
- **Radiologist preference conflates "preferred" with "correct."** Preference can be driven by verbosity or style; the paper itself notes 562B is verbose.
- **No compute / efficiency numbers.** 562B inference cost vs a specialist is plausibly 1000x larger; whether the marginal accuracy is worth it is unaddressed.
- **MIMIC-CXR contamination risk.** PaLM-E pretraining corpora are not fully disclosed; pretrain-test leakage for any web-scraped medical content cannot be ruled out.
- **"Emergent CoT" claim** rests on a handful of qualitative TB examples with no quantitative reasoning benchmark.
- **PAD-UFES-20 has no canonical split**, so the 97.27% macro-AUC is not directly comparable to literature.

## Why It Matters for Medical AI

Med-PaLM M is the paper that made "generalist medical foundation model" a viable research program. The lasting contributions are real: (i) **MultiMedBench** -- a unified, multi-modality, instruction-tuned benchmark that every later paper (LLaVA-Med, BiomedGPT, Med-Gemini, MedGemma) is measured against; (ii) the **proof of concept** that a single set of weights can match or exceed specialists on 5/12 medical tasks without per-task architecture changes; (iii) the **text-only one-shot exemplar trick** that makes multimodal instruction tuning tractable on single-image training; (iv) the **genomics-as-image hack** that brings DeepVariant pileups into a VLM. The findings that should temper enthusiasm are equally important: scaling is **not monotonic** in vision-heavy tasks, the 22B ViT is the actual bottleneck, the 562B is **worse** than the 84B for CXR report generation and human preference, the headline 40.5% radiologist preference is a single-scale single-dataset snapshot, and on the tasks that matter most clinically -- USMLE-style QA, MIMIC-CXR AUC, MIMIC-III summarization, variant calling -- specialists still win. The honest framing two years on is: a remarkable demonstration that one model can do everything badly-to-acceptably, and most things worse than the dedicated specialist. The follow-up direction the paper itself points to (scale the vision encoder, add real CoT data, evaluate at external sites with statistical rigor) is what MedGemma, Med-Gemini, and successors have had to address.

## References

- Paper: Tu, Azizi, Driess, Schaekermann, Amin, Chang, Carroll, Lau, Tanno, Ktena, Mustafa, Chowdhery, Liu, Kornblith, Fleet, Mansfield, Prakash, Wong, Virmani, Semturs, Mahdavi, Green, Dominowska, Agüera y Arcas, Barral, Webster, Corrado, Matias, Singhal, Florence, Karthikesalingam, Natarajan. *Towards Generalist Biomedical AI.* arXiv:2307.14334v1 (26 Jul 2023); subsequently published in NEJM AI 2024.
- arXiv: <https://arxiv.org/abs/2307.14334>
- MultiMedBench: assembled from MedQA, MedMCQA, PubMedQA, MIMIC-III, MIMIC-CXR, VQA-RAD, Slake-VQA, Path-VQA, PAD-UFES-20, VinDr-Mammo, CBIS-DDSM, PrecisionFDA Truth Challenge V2.
- Predecessors: PaLM-E (Driess et al., 2023); PaLM 2; Med-PaLM 2 (Singhal et al., 2023).
- Successors / follow-ups compared against Med-PaLM M: LLaVA-Med, BiomedGPT, Med-Flamingo, Med-Gemini, MedGemma.
