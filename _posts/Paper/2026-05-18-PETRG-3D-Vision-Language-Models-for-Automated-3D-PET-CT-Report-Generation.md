---
title: "PETRG-3D: Vision-Language Models for Automated 3D PET/CT Report Generation"
excerpt: "Dual 3D ViT (PET + CT) + Perceiver Samplers + LoRA-Qwen3-8B with hospital-matched 'healthy report' templates beats PET2Rep/Qwen3-VL by +35.74 BLEU-4 / +30.63 ROUGE-L / +8.18 PT-All on PETRG-Lym — but the headline is partly an apples-to-oranges baseline, and external CT-CE metrics actually drop."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/petrg-3d/
tags:
  - PETRG-3D
  - PET-CT
  - Report-Generation
  - RadFM
  - Perceiver-Sampler
  - LoRA
  - Qwen3
  - Lymphoma
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- PETRG-3D is the first end-to-end framework that jointly consumes the full **3D PET *and* CT volumes** for whole-body report generation. Two frozen **RadFM ViT-3D** branches feed trainable **Perceiver Samplers** (128 latent queries → 128 visual tokens), which are wrapped in `<pet>` / `<ct>` special tokens and concatenated with a `[Center ID]` × `[Gender]`-matched **healthy-report template** (`<template>` tokens, SAMF) before being decoded by a **LoRA-tuned Qwen3-8B**.
- New benchmarks: **PETRG-Lym** (824 PET/CT-report pairs from 746 lymphoma patients across 4 Chinese hospitals; 245,509 paired slices; chronologically split 663/161 scans) and **AutoPET-RG-Lym** (135 external lymphoma cases derived from AutoPET with expert-written reports). A new **PETRG-Score** covers 24 anatomical regions × 5 PET uptake × 8 CT density labels, with Qwen3-Max as the label extractor.
- **Headline result vs PET2Rep / Qwen3-VL-8B-Sep on PETRG-Lym: +35.74 BLEU-4, +30.63 ROUGE-L, +8.18 PT-All, +1.10 CT-All** (PETRG-3D / Qwen3-8B: B-4 41.90, R-L 52.88, PT-All 32.06, CT-All 34.76). The ablation gains stack cleanly — **DSFE ~+5 B-4 over single-modality, SAMF another ~+5 B-4 internally and a striking +24.36 B-4 / +16.40 R-L on the external AutoPET hospital** — but the headline NLG gap is inflated by the fact that **baselines were never LoRA-fine-tuned**, all ablations are single runs without variance, and **external CT-CE actually collapses** (CT-All 34.76 → 17.14–22.79), a regression the abstract glosses over.

## Motivation

PET/CT is the workhorse of oncologic staging and treatment-response assessment, but report writing is bottlenecked by a global shortage of nuclear-medicine physicians. Compared with X-ray or CT alone, automated PET report generation (PETRG) is much harder: (i) metabolic patterns depend on tracer physiology and patient metadata (weight, injected dose, scan time), (ii) lymphoma is inherently a whole-body problem — per-region reading is insufficient, and (iii) paired image–report data are rare, expensive, and clinically heterogeneous across centers.

Prior PETRG work made two compromises. **PET2Rep** (AAAI'26) truncates the volume into a handful of 2D slices and pipes them into off-the-shelf VLMs (Qwen-VL, InternVL). **ViMed-PET** (NeurIPS'25) models PET *only*, dropping the CT half of the modality. Neither addresses inter-hospital style variability, and there is no public benchmark with expert-validated PET/CT reports. Standard NLG metrics (BLEU / METEOR / ROUGE) say nothing about whether the reported metabolic and structural findings are clinically correct. PETRG-3D attacks the **model**, **data**, and **evaluation** gaps in one paper.

## Core Innovation

- **Dual-Stream Volumetric Feature Encoding (DSFE).** Two parallel branches, each = `ViT-3D (frozen, RadFM-initialized)` → `Perceiver Sampler (trainable, 128 learnable latent queries)`. The PET branch is also initialized from CT-pretrained ViT — training PET encoder from scratch is shown to hurt all metrics.
- **Style-Adaptive Multimodal Fusion (SAMF).** Every sample carries `[Center ID]` (1–5) and `[Gender]` flags that index a curated dictionary of hospital-specific "healthy patient" templates written by senior NM physicians. The retrieved template is injected into the prompt as a `<template>…</template>` block. This is what lets the same model speak in four different hospital dialects without retraining.
- **Stop-token trick.** Training reports are terminated with an explicit `[end-of-report]` token; this kills the post-report hallucination caused by the fixed `max_new_tokens` buffer.
- **PETRG-Lym + AutoPET-RG-Lym + PETRG-Score.** First multi-center single-disease PETRG dataset, first expert-relabeled external PET/CT report benchmark, and a label-extraction metric that covers 24 anatomical regions × 5 PET uptake × 8 CT density classes (vs PET2Rep's 19 regions × 4 uptake).

## Claims & Evidence Analysis

| # | Claim | Evidence in paper | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Dual-branch 3D PET+CT encoding beats single-modality 3D. | Ablation Table 5 (Qwen3-8B): DSFE adds +4.78 B-4 over PET-only, +7.41 over CT-only; CE metrics also rise. | PETRG-Lym (val only). | ⭐⭐ |
| C2 | Style-adaptive prompting (SAMF) closes inter-hospital style gaps. | Suppl Table 3 per-center NLG with/without SAMF; **+24.36 B-4 / +16.40 R-L on external Center 5 (AutoPET)**; +6–11 B-4 on internal centers 2/3/4. | PETRG-Lym + AutoPET-RG-Lym. | ⭐⭐⭐ |
| C3 | PETRG-3D "substantially outperforms" existing methods (+31.49% R-L, +8.18% PT-All). | Tables 3–4 vs PET2Rep / LLaVA-Med / RadFM / M3D. | PETRG-Lym + AutoPET-RG-Lym. | ⭐⭐ — large gap, but baselines are off-the-shelf VLMs without LoRA fine-tuning. |
| C4 | CT-pretrained ViT transfers usefully to PET. | Fig. 4 — "PET Enc. from Scratch" worse on every metric. | PETRG-Lym (val). | ⭐⭐ |
| C5 | Explicit `[end-of-report]` stop token reduces post-report hallucination. | Fig. 4 + Suppl Fig. 8 qualitative. | PETRG-Lym. | ⭐⭐ |
| C6 | Regional decomposition *hurts* whole-body PETRG. | Fig. 4 — "w/ RI" and "w/ RI + FT Encoder" both regress. | PETRG-Lym (val). | ⭐⭐ |
| C7 | PETRG-Score is more clinically faithful than prior CE metrics. | Suppl Table 2 (coverage: 24 regions × 5 uptake × 8 density vs PET2Rep 19×4 / ViMed-PET 5×2). | Definitional only. | ⭐ — no inter-rater study, no human spot-check of Qwen3-Max extraction. |
| C8 | First end-to-end framework jointly modeling full 3D PET + CT. | Related-work Table 1: PET2Rep = 2D, ViMed-PET = PET-only. | n/a. | ⭐⭐⭐ |
| C9 | Released benchmark establishes a strong baseline for future PETRG. | AutoPET-RG-Lym + PETRG-Score public release. | AutoPET-RG-Lym. | ⭐⭐ — concrete dataset/protocol; "strong baseline" depends on community uptake. |
| C10 | Model is approaching clinical reliability. | Authors *deny* this — "diagnostic accuracy still below clinical requirements". | – | n/a (authors honest). |

**Honest read.** The strongest empirical evidence in the paper is **C2 (SAMF on the external hospital)**: a +24.36 BLEU-4 jump from a small architectural change, evaluated on a genuinely held-out scanner + country, is compelling. **C1 / C4 / C5 / C6** are reasonable but every ablation cell is a single run on a single internal validation split — no variance, no seeds, no significance testing. **C3 is inflated by the baseline choice.** LLaVA-Med, RadFM, and M3D are zero-shot generalist VLMs not fine-tuned on PET, and PET2Rep is itself a prompt-engineered pipeline; the only fair head-to-head would be PET2Rep *also* LoRA-fine-tuned on PETRG-Lym, which the paper does not run. Worse, SAMF literally *injects* the hospital template into the prompt, which the LLM can then partially copy back into the output — so the 6 → 42 B-4 leap on the internal set is dominated by stylistic match, not radiological reasoning (note that PT-Ab / CT-Ab improvements are far more modest at 19.5 / 28.8). **C7 is purely definitional**: PETRG-Score uses Qwen3-Max as an unverified extractor; the authors hand-wave that "extraction noise affects all baselines equally" but never quantify it.

The most important finding the paper *under-reports* is the **external CT degradation**: CT-All drops from 34.76 (internal) to 17.14–22.79 (external), and CT-Ab from 28.76 to 9.24–16.35. The abstract foregrounds favorable PET-CE numbers; the CT regression — arguably the more clinically actionable signal — gets one sentence. For a model intended to assist whole-body reporting, this overfitting to source-domain CT phrasing is a deployment-blocker that deserves an experiment, not a paragraph.

## Method & Architecture

![PETRG-3D framework: dual 3D ViT + Perceiver Sampler + SAMF + LoRA-Qwen3-8B decoder](/assets/images/paper/petrg-3d/page_005.png)
*Figure 2: PETRG-3D framework. **Blue** — Dual-Stream Volumetric Feature Encoding: two frozen RadFM ViT-3D branches each compress the PET / CT volume via a trainable Perceiver Sampler with 128 latent queries (output: 128 visual tokens × 768 dim). **Yellow** — Style-Adaptive Multimodal Fusion: the `[Center ID]` × `[Gender]` keys retrieve a healthy-patient template; PET / CT tokens are wrapped in `<pet>` / `<ct>` and the template in `<template>` special tokens. **Pink** — LoRA-adapted Qwen3-8B decoder (r=8, α=32, dropout 0.1) generates the report autoregressively with an explicit `[end-of-report]` stop token.*

**1) Preprocessing.** De-identify DICOM → extract metadata (weight, injected dose, scan time) → convert to NIfTI → resample PET and CT to **1.5 × 1.5 × 3 mm** in RAS → clip CT to **[−1000, +1000] HU** → convert PET to SUV via $\mathrm{SUV}(t) = c_{img}(t) \cdot \mathrm{BW} / \mathrm{ID}$ → remove scanner bed with TotalSegmentator → crop to upper-thigh level (trunk + 20% below pelvic floor, ≤ 50 slices).

![Preprocessing pipeline for PET/CT volumes and report text](/assets/images/paper/petrg-3d/page_001.png)
*Suppl. Fig. 1: Preprocessing pipeline. Image side — de-id → SUV conversion → bed removal (TotalSegmentator) → trunk crop. Text side — de-id → JSON parsing → normalization → keep only the "Findings" section.*

**2) Dual-Stream Volumetric Feature Encoding (DSFE).** Two parallel branches, each:

$$\mathrm{ViT\text{-}3D}_{\text{frozen, RadFM-init} } \;\rightarrow\; \mathrm{Perceiver\,Sampler}_{\text{trainable, 128 latents} }.$$

Sliding-window on the volume yields variable-length features; the Perceiver Sampler compresses them to a fixed sequence of **128 visual tokens × 768 hidden dim** per modality. Crucially the PET branch is also initialized from CT-pretrained ViT — Fig. 4 confirms that training the PET encoder from scratch hurts all metrics.

**3) Style-Adaptive Multimodal Fusion (SAMF).** Each sample carries `[Center ID]` ∈ {1…5} and `[Gender]` flags. These index a dictionary of hospital-specific healthy-patient templates curated by senior NM physicians at each institution. The retrieved template is injected into the prompt as `<template>…</template>`, while PET and CT tokens are wrapped in `<pet>` and `<ct>` special tokens. A linear projector aligns the 768-d visual tokens to the LLM's 4096-d hidden state.

**4) LoRA-adapted LLM decoder.** The full input — *system prompt + `<ct>` tokens + `<pet>` tokens + `<template>` block + generating instruction* — is fed to a Chinese-capable 7–9B LLM (default **Qwen3-8B**; also benchmarked: Llama2-7B, Mistral-7B-v0.3, Qwen2.5-7B, Gemma2-9B, GLM4-9B). LoRA config: **r = 8, α = 32, dropout 0.1** on the attention layers; the base LLM is frozen.

**5) Training & inference.** AdamW, LR **5×10⁻⁵** with 100-step linear warmup then constant, 30 epochs, **2× A800 (80 GB)**, effective batch size 16, max input 2,048 tokens. Inference: nucleus sampling, top-p 0.9, T 0.7, repetition penalty 1.05, max 1,024 new tokens.

## Experimental Results

### PETRG-Lym (internal validation)

| Method | LLM | B-1 | B-4 | MTR | R-L | PT-All | CT-All | PT-Ab | CT-Ab |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-Med | Mistral-7B-v0.2 | 3.21 | 0.40 | 2.46 | 4.84 | – | – | – | – |
| RadFM | Llama2-13B | 1.65 | 0.16 | 1.27 | 2.44 | – | – | – | – |
| M3D | Llama2-7B | 0.53 | 0.07 | 2.16 | 4.56 | – | – | – | – |
| PET2Rep-Sep | Qwen3-VL-8B | 39.60 | 6.16 | 27.06 | 22.25 | 23.88 | 33.66 | 11.53 | 28.41 |
| PET2Rep-Sep | InternVL-3.5-8B | 38.23 | 5.96 | 25.99 | 22.20 | 22.75 | 33.45 | 10.64 | 28.01 |
| PET2Rep-Fus | InternVL-3.5-8B | 37.79 | 5.83 | 25.69 | 21.92 | 23.88 | **33.88** | 12.22 | 28.62 |
| PETRG-3D | Qwen2.5-7B | 60.42 | 41.33 | 51.21 | 52.15 | 31.78 | 34.13 | 19.20 | 28.06 |
| **PETRG-3D** | **Qwen3-8B** | **60.78** | **41.90** | 51.16 | 52.88 | **32.06** | 34.76 | **19.53** | **28.76** |
| PETRG-3D | GLM4-9B | 58.73 | 41.55 | 50.80 | 53.74 | 29.06 | 31.65 | 15.72 | 25.07 |
| PETRG-3D | Mistral-7B-v0.3 | 53.18 | 36.07 | 43.20 | 48.60 | 27.25 | 30.24 | 13.52 | 23.55 |
| PETRG-3D | Llama2-7B | 28.19 | 14.90 | 22.60 | 27.19 | 26.10 | 25.45 | 11.99 | 17.97 |

### AutoPET-RG-Lym (external test)

| Method | LLM | B-4 | MTR | R-L | PT-All | CT-All | PT-Ab | CT-Ab |
|---|---|---|---|---|---|---|---|---|
| PET2Rep-Sep | Qwen3-VL-8B | 7.51 | 27.56 | 24.33 | 24.25 | 29.22 | 12.29 | 23.07 |
| PET2Rep-Sep | Qwen2.5-VL-Max | 5.83 | 28.78 | 21.21 | **26.76** | 25.94 | 14.00 | 18.46 |
| PET2Rep-Fus | InternVL-3.5-8B | 7.05 | 26.91 | 22.92 | 23.38 | **32.96** | 12.16 | **27.16** |
| **PETRG-3D** | **GLM4-9B** | **44.18** | **47.41** | **55.47** | **36.22** | 17.14 | **25.46** | 9.24 |
| PETRG-3D | Qwen3-8B | 32.47 | 41.50 | 40.97 | 33.48 | 22.79 | 22.81 | 16.35 |
| PETRG-3D | Mistral-7B-v0.3 | 35.03 | 40.31 | 44.20 | 23.63 | 16.94 | 11.64 | 9.58 |

PETRG-3D dominates NLG and PET-CE metrics on both splits, but **on the external set CT-CE collapses** (CT-All 17.14 / 22.79 vs PET2Rep ~33; CT-Ab 9–16 vs ~27). The authors hypothesize that CT description style varies more across centers than PET findings, so the model overfits source-domain CT phrasing.

### Ablations (Qwen3-8B, internal val)

| Configuration | B-4 | R-L | PT-All | CT-All |
|---|---|---|---|---|
| CT only | 29.13 | 41.62 | 28.81 | 32.96 |
| PET only | 31.76 | 43.61 | 26.01 | 32.95 |
| DSFE (both, no SAMF) | 36.54 | 46.93 | 28.38 | 33.27 |
| **DSFE + SAMF (full)** | **41.90** | **52.88** | **32.06** | **34.76** |

DSFE adds **~+5 B-4** over single-modality; SAMF adds **another ~+5 B-4** plus +3.7 PT-All. The per-center SAMF deltas are the most interesting cut: gains of +6 to +11 B-4 on internal centers 2/3/4, modest +0.22 on center 1, and a striking **+24.36 B-4 / +13.32 METEOR / +16.40 R-L** on the external Center 5 (AutoPET) — this is the central evidence for the cross-hospital transfer claim.

Other design choices (Fig. 4): training the PET encoder from scratch significantly degrades all metrics; removing the `[end-of-report]` stop token causes severe post-report hallucinations; region-wise decomposition (2,652 H&N / Chest / Abdomen / Pelvis pairs) *reduces* NLG performance, and additionally fine-tuning the visual encoder on regional data further harms NLG and CT-CE while only marginally helping PT-Ab. The simple **whole-body + frozen-encoder** recipe wins.

### Per-class stratification (Suppl Tables 4–5)

"Normal" dominates (F1 ≈ 0.82 PET, 0.77 CT). Abnormal classes all sit in the **8–45%** F1 range, worst at Calcification (8.33) and Wall/Membrane Thickening (11.43). The model is **far from clinical-grade for any individual abnormal label**, despite the rosy aggregate NLG numbers.

### Qualitative

![Chest-region qualitative comparison: PETRG-3D vs PET2Rep vs GT](/assets/images/paper/petrg-3d/fig_p007_03.png)
*Figure 3 (excerpt): Chest-region comparison. Colored spans mark anatomical areas; gray boxes mark incorrect findings. PET2Rep outputs are short, generic, or clinically irrelevant; PETRG-3D matches GT length and finding categories, with minor localization errors (e.g. "left upper lobe" vs "left lower lobe").*

![Ground-truth whole-body lymphoma report (Chinese with English translation)](/assets/images/paper/petrg-3d/fig_p021_01.png)
*Suppl. Fig. 5: A ground-truth whole-body lymphoma PET/CT report illustrating the anatomical breadth and length that a model has to reproduce — region-by-region descriptions of nodal involvement, SUVmax values, and CT density findings across head, neck, chest, abdomen, and pelvis.*

![Region-level decomposition example used for the 'w/ Regional Input' ablation](/assets/images/paper/petrg-3d/page_004.png)
*Suppl. Fig. 2 (excerpt): Region-level report decomposition used in the "w/ Regional Input" ablation — the whole-body report is split into Head&Neck / Chest / Abdomen / Pelvis-and-below pairs. Counterintuitively, this regional supervision **hurts** whole-body NLG performance, suggesting the model benefits from end-to-end whole-body context rather than per-region training signals.*

## Limitations

**Authors admit:**

- Diagnostic accuracy is below clinical requirements; the model "primarily captures stylistic patterns rather than performing detailed pathophysiological reasoning."
- Longitudinal scans (baseline vs post-treatment) are treated identically — no temporal modeling.
- Macro-F1 of abnormal classes is severely depressed by class imbalance ("Normal" dominates with F1 ≈ 0.77–0.82).

**Authors do not address:**

- **No variance, no seeds, no significance tests** anywhere in the ablation tables.
- **B-4 inflation from template injection** — SAMF literally puts a normal-case template into the prompt, which the model can partially copy back into the output. This plausibly drives most of the 6 → 42 B-4 leap on the internal set; note that PT-Ab / CT-Ab improvements are much smaller (19.5 / 28.8).
- **No quantification of Qwen3-Max extraction noise** for PETRG-Score; no human spot-check of extraction quality.
- **No fair head-to-head with PET2Rep under the same LoRA fine-tuning budget** — the headline gap is partly baselines being zero-shot or prompt-only.
- **No non-FDG tracers, no non-lymphoma cancers** — reproducibility outside this single disease / tracer combination is untested.
- **Chinese-only reports**: AutoPET's native German report style is not evaluated; the external "test" reports were re-authored by Chinese physicians, partially leaking source-domain stylistic conventions.
- **External CT-CE regression** (CT-All 34.76 → 17.14–22.79) is acknowledged in one sentence but not solved, despite being the most clinically consequential finding.
- **All ablations on internal validation only** — design choices may themselves be tuned to internal-set NLG behavior.

## Why It Matters for Medical AI

- **Concrete, releasable benchmark.** PETRG-Lym + AutoPET-RG-Lym + PETRG-Score together form the first public, multi-center, expert-validated PET/CT report-generation benchmark — useful regardless of one's view on PETRG-3D itself. The AutoPET-RG-Lym external set in particular finally lets PETRG papers report cross-country generalization numbers.
- **Style-adaptive prompting is a generalizable trick.** SAMF's `[Center ID]` × `[Gender]` template-retrieval is a lightweight, training-free way to absorb inter-hospital report-style variability. The +24.36 B-4 on the external center suggests this design pattern could transfer to other multi-center clinical generation tasks where each site has its own report dialect (radiology, pathology, oncology).
- **Honest about clinical readiness.** The authors explicitly state the model is not clinically deployable. The 8–45% per-class F1 on abnormal findings and the external CT-CE regression are real bottlenecks — they map directly to "the model is good at sounding like a hospital's reports, less good at being correct." This is the more useful framing for anyone planning to fine-tune on top of PETRG-3D.

## References

- arXiv: <https://arxiv.org/abs/2511.20145>
- Datasets: PETRG-Lym (4 Chinese hospitals, 824 scans), AutoPET-RG-Lym (135 cases derived from AutoPET; to be released)
- Related work: PET2Rep (AAAI'26), ViMed-PET (NeurIPS'25), RadFM, M3D-LaMed, LLaVA-Med, CT-CLIP
- Base models: RadFM ViT-3D encoder, Qwen3-8B / GLM4-9B / Qwen2.5-7B / Mistral-7B-v0.3 / Llama2-7B / Gemma2-9B decoders, Perceiver Sampler (Flamingo)
- Evaluation: PETRG-Score (24 regions × 5 PET uptake × 8 CT density), Qwen3-Max as label extractor
