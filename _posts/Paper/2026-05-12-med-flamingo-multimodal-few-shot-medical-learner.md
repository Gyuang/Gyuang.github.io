---
title: "Med-Flamingo: a Multimodal Medical Few-shot Learner"
excerpt: "Continued pretraining of OpenFlamingo-9B on a 0.8M-image medical textbook corpus delivers a +20% clinician-rated VQA-RAD bump on a custom split, while exposing 194 PathVQA test leaks and losing to the OpenFlamingo zero-shot baseline on pathology."
categories:
  - Paper
tags:
  - Med-Flamingo
  - Multimodal
  - Few-shot Learning
  - In-context Learning
  - Vision-Language Model
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- **Med-Flamingo continues OpenFlamingo-9B pretraining on a new interleaved corpus — MTB (~0.8M images, 584M tokens from 4,721 medical textbooks) plus PMC-OA — to enable real multimodal in-context learning in medicine.**
- Headline claim is **+20% relative clinician score on VQA-RAD (5.61 vs. 4.69)**, but this holds only on the authors' custom 90/10 split; on PathVQA few-shot Med-Flamingo (1.81) is **worse than OpenFlamingo zero-shot (2.16)**, and on Visual USMLE the gap to OpenFlamingo zero-shot is essentially zero (4.33 vs. 4.31).
- The most reusable contribution is the **leakage audit**: FAISS+ViT nearest-neighbour deduplication flags **194 / 6,700 PathVQA test images** as near-duplicates of pretraining content, and the authors document — though do not quantify — leakage in the official VQA-RAD splits.

## Motivation

Medicine is intrinsically multimodal and long-tailed: any deployable assistant has to recognise rare presentations from a handful of in-context exemplars rather than from millions of labelled examples. Yet at submission time no medical VLM supported interleaved few-shot prompting. BiomedCLIP and ChexZero are encoder-only and specialty-narrow. MedVINT supports only a single image per prompt and was not built for interleaved context. LLaVA-Med — the contemporaneous competitor — takes the **opposite** design path: GPT-4 self-instruct distillation followed by full LM fine-tuning. Med-Flamingo bets instead that **interleaved pretraining on authoritative textbooks**, with the LLaMA-7B decoder and CLIP ViT-L/14 vision encoder kept frozen, is what unlocks in-context behaviour. Neither paper benchmarks against the other, so the obvious community question — instruction tuning vs. interleaved pretraining for medical VLMs — remains open.

## Core Innovation

1. **MTB corpus.** 4,721 medical textbooks across 49 specialties, converted PDF→HTML with image tags preserved as `<image>` placeholders, segmented into 1–10-image chunks. Roughly 0.8M images and 584M tokens, 95/5 train/val.
2. **Frozen-backbone continued pretraining.** OpenFlamingo-9B style: frozen LLaMA-7B + frozen CLIP ViT-L/14, trainable Perceiver Resampler and gated cross-attention only (1.3B of ~8.3B parameters).
3. **Visual USMLE.** 618 open-ended cross-specialty problems rephrased from licensed Amboss content — a deliberate break from the radiology/pathology mould of existing medical VQA, with case vignettes and lab tables included.
4. **Leakage-aware evaluation.** ViT embeddings + FAISS nearest neighbour with a Euclidean-distance threshold (≈80) prune 194 PathVQA test images; the authors additionally build a custom VQA-RAD split that separately partitions both images and questions.
5. **Clinician-in-the-loop scoring.** A Streamlit app shows blinded shuffled generations to three medical doctors who score each 0–10 for clinical usefulness — the primary metric throughout the paper.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | First medical foundation model with multimodal in-context learning. | Architectural argument + qualitative few-shot examples (Figs. 1, 5, 6). | n/a (definitional) | ⭐⭐⭐ |
| C2 | MTB + PMC-OA continued pretraining improves generative medical VQA. | Tables 1–3 clinician scores on deduplicated splits. | VQA-RAD, PathVQA, Visual USMLE | ⭐⭐ |
| C3 | "Up to 20% improvement" in clinician rating over prior models. | VQA-RAD 5.61 vs. 4.69 = +19.6%, **but only on the custom split**. PathVQA few-shot (1.81) loses to OpenFlamingo zero-shot (2.16); Visual USMLE gap ≈0 (4.33 vs. 4.31). | VQA-RAD only | ⭐ |
| C4 | Best average rank (1.67) across three datasets in clinical evaluation. | Aggregated rank, not magnitude; no variance / no significance test. | All three | ⭐⭐ |
| C5 | Enables rationale generation, a capability not previously shown. | Figure 5 single qualitative example; no quantitative rationale evaluation; rationales unreliable when answer is wrong. | 1 anecdote | ⭐ |
| C6 | Textbook pretraining yields higher factual quality than web pretraining. | Asserted; no web-pretrained ablation. | — | ⭐ |
| C7 | Visual USMLE is more realistic than radiology/pathology VQA. | Specialty breadth (Fig. 8); open-ended phrasing; no inter-annotator agreement. | Visual USMLE | ⭐⭐ |
| C8 | PathVQA has real pretraining leakage that requires deduplication. | **194 / 6,700 test images flagged via FAISS + ViT.** | PathVQA | ⭐⭐⭐ |
| C9 | Official VQA-RAD splits suffer severe leakage. | Asserted, not numerically quantified. | VQA-RAD | ⭐⭐ |
| C10 | All evaluated VLMs (including Med-Flamingo) hallucinate. | Stated in abstract + Limitations; consistent with modest absolute scores (≤5.61/10). | All | ⭐⭐⭐ |

The single strongest contribution here is **C8 — the leakage audit**: a concretely quantified, reproducible methodology that should change how everyone splits medical VQA datasets. The headline efficacy claim (C3) is, by contrast, the weakest part of the paper — a single-dataset win on a custom split that disappears or reverses on the other two benchmarks. The average-rank framing (C4) papers over this by treating a 0.02-point Visual USMLE lead the same as a 0.92-point VQA-RAD lead.

## Method & Architecture

![Med-Flamingo three-stage overview](/assets/images/paper/med-flamingo/page_003.png)
*Figure 1: Three-stage Med-Flamingo overview — interleaved pretraining on MTB + PMC-OA with frozen LLaMA-7B and frozen CLIP ViT-L/14, glued by a trainable Perceiver Resampler and gated cross-attention layers; few-shot generative VQA prompting; clinician evaluation app.*

Med-Flamingo is OpenFlamingo-9B v1 with continued pretraining. Only the Perceiver Resampler (image-token compressor) and the gated cross-attention layers between LM blocks are trained — 1.3B of ~8.3B parameters. The joint loss mixes the paired (PMC-OA) and interleaved (MTB) distributions with $\lambda = 1$:

$$L = \mathbb{E}_{(x,y)\sim D_p}\Big[-\sum_{\ell=1}^L \log p(y_\ell \mid y_{<\ell}, x_{<\ell})\Big] + \lambda \cdot \mathbb{E}_{(x,y)\sim D_i}\Big[-\sum_{\ell=1}^L \log p(y_\ell \mid y_{<\ell}, x_{<\ell})\Big]$$

Training ran on 8× A100 80GB with DeepSpeed ZeRO Stage 2, 8-bit AdamW, and memory-efficient attention — 2,700 steps, ~6.75 wall-clock days, total batch size 400 (1 per device × 50 grad-accum × 8 GPUs).

![MTB specialty distribution](/assets/images/paper/med-flamingo/page_004.png)
*Figure 2: MTB's 4,721 textbooks across 49 specialty categories — heavily skewed toward neurology, radiology, oncology, surgery, and cardiology. Pathology and dermatology are under-represented, which the authors invoke to explain Med-Flamingo's weak PathVQA performance.*

Inference uses 6 in-context shots for VQA-RAD and PathVQA and 4 shots for Visual USMLE (longer prompts force context truncation). A `<image> Question: ... Rationale: ... Answer: ...` variant teaches the model — at prompt time, no fine-tuning — to emit a chain-of-thought-style rationale before the answer.

![Rationale prompting example](/assets/images/paper/med-flamingo/page_008.png)
*Figure 3: With `Question / Rationale / Answer` exemplars, Med-Flamingo emits a visually grounded rationale ("aorta visible as a circular shape ventral of the spine ... calcification of the aortic wall") before its final answer — the most distinctive qualitative result in the paper, though the authors note rationales are unreliable when the final answer is wrong.*

![Streamlit clinician evaluation app](/assets/images/paper/med-flamingo/page_007.png)
*Figure 4: The bespoke Streamlit app used for human evaluation — three medical doctors (incl. one board-certified radiologist) see the image, question, correct answer, and blinded shuffled generations from all systems, then score each 0–10 for clinical usefulness.*

## Experimental Results

The primary metric is mean clinician score (0–10); BERT-similarity and exact match are reported as automated checks.

| Model | VQA-RAD Clin.Eval / BERT / Exact | PathVQA Clin.Eval / BERT / Exact | Visual USMLE Clin.Eval / BERT |
|---|---|---|---|
| MedVINT zero-shot | 4.63 / 0.628 / 0.167 | 0.13 / 0.608 / 0.272 | 0.41 / 0.421 |
| MedVINT fine-tuned | 2.87 / 0.611 / 0.133 | 1.23 / 0.723 / 0.385 | — |
| OpenFlamingo zero-shot | 4.39 / 0.490 / 0.000 | **2.16** / 0.474 / 0.009 | **4.31** / **0.512** |
| OpenFlamingo few-shot | 4.69 / 0.645 / 0.200 | 2.08 / 0.669 / 0.288 | 3.39 / 0.470 |
| Med-Flamingo zero-shot | 3.82 / 0.480 / 0.000 | 1.72 / 0.521 / 0.120 | 4.18 / 0.473 |
| **Med-Flamingo few-shot** | **5.61** / **0.650** / 0.200 | 1.81 / **0.678** / 0.303 | **4.33** / 0.431 |

Three observations the paper does not foreground:

- **PathVQA is uniformly poor for everyone.** The best clinical score is OpenFlamingo zero-shot at 2.16/10 — even fine-tuned MedVINT, the only system that ingested ~20K pathology QA pairs at training, manages just 1.23. Med-Flamingo few-shot (1.81) is worse than the OpenFlamingo zero-shot baseline. The authors interpret this as evidence that classification-based metrics have severely overestimated pathology VLM capability.
- **Visual USMLE clinical score moves opposite to BERT-similarity.** Few-shot Med-Flamingo wins clinical evaluation (4.33) but loses BERT-sim (0.431 vs. 0.473) — the GPT-4-summarised few-shot prompts produce terser answers that diverge lexically from wordy ground truth.
- **No ablations isolate MTB vs. PMC-OA.** Despite MTB being the headline data contribution, there is no MTB-only or PMC-OA-only Med-Flamingo. Number of shots, $\lambda$, and inter-rater agreement are likewise not reported.

![Visual USMLE qualitative example](/assets/images/paper/med-flamingo/page_010.png)
*Figure 5: Visual USMLE case — Med-Flamingo correctly identifies smoking as the strongest risk factor for bladder cancer, while the OpenFlamingo baseline misdiagnoses as metastatic prostate cancer.*

![MTB image cluster annotation](/assets/images/paper/med-flamingo/page_014.png)
*Figure 6: Distribution of 100 manually annotated image clusters in MTB — anatomical illustrations, dermatology, surgical illustrations, X-ray, and MRI dominate the actual image-type composition.*

## Limitations

Authors acknowledge: not safe for clinical use; pretrained only (no instruction or preference tuning); hallucinations occur in all VLMs including Med-Flamingo; "book-smart" only — no EHR data, no 3D imaging, no video; few-shot rationales unreliable when the final answer is wrong.

Authors do not adequately address:

- **MTB vs. PMC-OA ablation.** Marginal value of the headline corpus is unmeasured — and MTB is the harder-to-redistribute half.
- **Shot-count sensitivity.** Why 6 / 4? No sweep.
- **Inter-rater agreement** among the three clinicians; no Cohen's κ, no per-rater breakdown, no confidence intervals on point estimates.
- **Custom VQA-RAD split.** Solves leakage but breaks comparability with prior published numbers — readers cannot directly map 5.61 onto existing VQA-RAD literature.
- **GPT-4 mediation in Visual USMLE construction.** Problems were rephrased and few-shot prompts summarised by GPT-4 with no distortion audit.
- **MTB redistribution.** Built from copyrighted textbooks; no license terms are provided, blocking external reproduction.
- **Head-to-head with instruction-tuned medical VLMs** (LLaVA-Med, Med-PaLM M) — explicitly deferred.

## Why It Matters for Medical AI

Two takeaways outlive the headline number. First, **the leakage audit is the actual contribution**: any future medical VLM paper that ignores FAISS-based pretraining/test deduplication should be regarded as untrusted. Second, Med-Flamingo and LLaVA-Med stake out **opposite hypotheses** about how to teach a generic VLM medicine — frozen-backbone interleaved pretraining on curated textbooks vs. GPT-4 self-instruct distillation plus full LM fine-tuning. Both papers report wins, neither benchmarks against the other, and the question of which path scales better with parameters, data, and instruction quality is still open. Anyone deciding between these recipes for a downstream medical application is currently choosing on prior beliefs rather than evidence.

## References

- Paper (arXiv): [Med-Flamingo: a Multimodal Medical Few-shot Learner (2307.15189)](https://arxiv.org/abs/2307.15189)
- Venue: Proceedings of Machine Learning for Health (ML4H) 2023
- Code (authors): [https://github.com/snap-stanford/med-flamingo](https://github.com/snap-stanford/med-flamingo)
- Backbone: Awadalla et al., *OpenFlamingo*, 2023
- Related — instruction-tuned counterpart: Li et al., *LLaVA-Med*, 2023
- Related — paired pretraining data: Lin et al., *PMC-OA*, 2023
- Related medical VLM baselines: *MedVINT*, *BiomedCLIP*, *ChexZero*
