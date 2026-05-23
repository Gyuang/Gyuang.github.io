---
title: "MedRegion-CT: Region-Focused Multimodal LLM for Comprehensive 3D CT Report Generation"
excerpt: "Region-centric R^2 Token Pooling, mask-driven visual extraction, and deterministic patient attributes drive a LLaMA-3.1-8B MLLM to SOTA on RadGenome-Chest CT (BLEU-4 0.290, CA-F1 0.450, GPT-4 48.837)."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/medregion-ct-region-focused-3d-ct-report/
tags:
  - MedRegion-CT
  - 3D-CT-Report-Generation
  - Multimodal-LLM
  - R2-Token-Pooling
  - Mask-Driven-Visual-Extractor
  - Patient-Specific-Attributes
  - Rad-DINO
  - LLaMA-3.1
  - SAT
  - RadGenome-Chest-CT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- MedRegion-CT mimics how a radiologist reads a chest CT — fast slice-wise overview plus per-organ deep dive — by feeding LLaMA-3.1-8B (LoRA) three token streams: slice-level **global tokens**, region-representative high-resolution tokens, mask-driven segmentation tokens, and a deterministic text block of patient-specific attributes.
- The headline trick is **R^2 (Region Representative) Token Pooling**, which compresses the 3D token budget from D x T to **D + T** while keeping inter-slice context — and, crucially, picks the representative slice per region as the one with the largest pseudo-mask pixel count rather than LITA's uniform temporal sampling.
- On RadGenome-Chest CT it sets SOTA versus M3D / MedBLIP / RadFM / CT2Rep at **BLEU-4 0.290, CA-F1 0.450, GPT-4 48.837**, with the cleanest piece of evidence being the R^2-vs-LITA isolation ablation: **CA-F1 0.405 vs 0.385, GPT-4 43.014 vs 42.293** under otherwise identical conditions.

## Motivation

RadGenome-Chest CT and CT-RATE have made large-scale 3D chest CT report generation tractable, but the existing 3D medical MLLMs (RadFM, M3D, MedBLIP, CT2Rep) compress the full volume into a small set of volume-level tokens. That works for the gist of a scan but loses the per-organ detail radiologists actually report on. Worse, there is no mature 2D-to-3D pretrained encoder for CT — 3D ViTs are typically trained from scratch and miss slice-level fidelity.

The authors' framing is procedural rather than architectural: they take the radiologist workflow (Fig. 1 — fast holistic scan, region-by-region analysis, integration with clinical metadata) and turn each of its three steps into a token stream. The LLM should see (a) every slice cheaply, (b) higher-resolution tokens at the slices that best represent each anatomical region, and (c) deterministic clinical attributes derived from per-organ pseudo-masks rather than re-derived by the LLM from pixels.

![Radiologist workflow that motivates MedRegion-CT](/assets/images/paper/medregion-ct/fig_p001_01.png)
*Figure 1: Radiologist workflow that motivates MedRegion-CT — fast holistic scan, per-region deep analysis, then clinical integration — directly maps to the model's three token streams.*

## Core Innovation

- **R^2 Token Pooling (D + T instead of D x T).** Per-slice average pooling produces D global tokens for cheap depth coverage; per-region adaptive pooling on a *single* representative slice (the one with the largest mask pixel count for that region) produces T region tokens. The two are concatenated. This is the key architectural difference from LITA-style uniform temporal sampling — region-centric selection rather than uniform slice strides.
- **Mask-Driven Visual Extractor — a 3D extension of MAIRA-SEG.** The same R^2 pooling is applied to pseudo-masks so they align with the multi-level visual tokens; mask pooling + linear projection yields *mask tokens*, flattening + linear projection of the masks yields *spatial tokens*. Two segmentation tokens per mask are placed at **fixed prompt positions regardless of mask positivity** — a deliberate change from MAIRA-SEG that lets the LLM autonomously decide whether a lesion mask carries signal.
- **Patient-Specific Attribute Extractor — deterministic text prompts from masks.** A RadGPT-style "vision translator" reads the SAT pseudo-masks and emits a fixed text template containing organ volumes (lung lobes, left/right atrium and ventricle, liver, kidneys) and lesion attributes (count, diameter in mm, spatial location for nodule / cyst / effusion). The LLM consumes this as part of the prompt, not as visual tokens.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | R^2 Token Pooling captures inter-slice context efficiently (D + T vs D x T) | Token-count math in Sec. 3.2; downstream metrics in Table 1. No wall-clock or memory benchmark. | RadGenome-Chest CT only | ⭐⭐ |
| C2 | Region-centric slice selection beats uniform LITA sampling | **Direct isolation ablation:** Ablation R^2 (CA 0.405, GPT-4 43.014) vs Ablation LITA (CA 0.385, GPT-4 42.293) | RadGenome-Chest CT | ⭐⭐ (strongest piece of evidence) |
| C3 | Mask-Driven Visual Extractor improves clinical accuracy | Ablation R^2+Mask (CA 0.430) > Ablation R^2 (CA 0.405) | RadGenome-Chest CT | ⭐⭐ |
| C4 | Patient-Specific Attributes improve clinical accuracy | Ablation R^2+Attr (CA 0.425) > Ablation R^2 (CA 0.405); full model (CA 0.450) > R^2+Mask (0.430) | RadGenome-Chest CT | ⭐⭐ |
| C5 | SOTA among 3D CT report generation models | Table 1 — best BLEU-4, CA-F1, GPT-4 vs CT2Rep / RadFM / MedBLIP / M3D | RadGenome-Chest CT | ⭐⭐ |
| C6 | Generated reports show "superior diagnostic accuracy and localization" | Single qualitative COVID-19 case (Fig. 3) | one case | ⭐ |
| C7 | The region-focused approach is "interpretable" | Asserted; pseudo-mask + per-region paragraph structure is the only mechanism | none specifically | ⭐ |
| C8 | Framework is "segmentation-model-agnostic" | Stated in Sec. 3.1, but only SAT is actually evaluated | none — claim untested | ⭐ |

**Honest read.** The internal ablations (C2, C3, C4) are the genuinely strong part of the paper — they isolate exactly what the abstract advertises. The SOTA claim (C5) is supported on one dataset only, with **no error bars, no multiple seeds, no significance tests**, and the BLEU-4 margin over M3D is razor-thin (0.290 vs 0.288). CA-F1 (+0.015) and GPT-4 (+1.6) are more meaningful gaps but a single GPT-4o Mini evaluator run is not statistically robust. The interpretability and "segmentation-model-agnostic" claims (C7, C8) are essentially unevaluated. Notably, none of the efficiency numbers (token count, inference time, memory) are quantified despite efficiency being a core motivation.

## Method & Architecture

![MedRegion-CT pipeline](/assets/images/paper/medregion-ct/fig_p004_01.png)
*Figure 2: MedRegion-CT pipeline — per-slice 2D Rad-DINO (ViT-B) encoding feeds R^2 pooling (global + region tokens); SAT pseudo-masks drive both the Mask-Driven Visual Extractor (mask + spatial tokens) and the deterministic Patient-Specific Attribute Extractor that emits text; all three streams enter LLaMA-3.1-8B with LoRA.*

The model emits a **structured 6-paragraph report** — Lung, Large airways, Mediastinum, Heart and great vessels, Osseous structures, Upper Abdomen — and ground-truth reports were pre-split into the same 6 regions for training. The full forward pass is

$$R = \mathrm{LLM}(T_{\text{vision} }, T_{\text{seg} }, T_{\text{attr} }, I)$$

with $I$ the instruction prompt that lists the 6 region tags and the attribute block.

**1. Pseudo-mask generation.** SAT (Zhao et al., 2023), a text-prompted universal medical segmentation model, produces pseudo-masks for the 6 anatomical regions plus the lesion classes (nodule, cyst, effusion). The framework is claimed to be segmentation-model-agnostic, but only SAT is evaluated.

**2. 2D slice encoder.** Each CT slice is independently encoded by a pretrained 2D image encoder (**Rad-DINO, ViT-B**). For a volume of D slices the raw output is D x T tokens.

**3. R^2 Token Pooling — global tokens $T_{\text{glob}}$.** For each of the D slices, average all visual tokens to obtain one global token per slice. This gives D global tokens at minimal cost — full depth coverage.

**4. R^2 Token Pooling — region representative tokens $T_{\text{reg}}$.** Select s representative slices (one per region in this work). Each is chosen as **the slice with the highest pseudo-mask pixel count for that region** — region-centric, *not* LITA's uniform sampling. Apply spatial adaptive pooling on each representative slice with a downsample factor of s, yielding T/s tokens per slice and T region tokens total.

**5. Concatenate.** $T_{\text{vision}} = \mathrm{Concat}(T_{\text{glob}}, T_{\text{reg}})$ — total token count **D + T** versus D x T for naive slice-wise approaches.

**6. Mask-Driven Visual Extractor → segmentation tokens $T_{\text{seg}}$.** A 3D extension of MAIRA-SEG / Osprey's mask-aware extractor. Pseudo-masks are themselves passed through R^2 pooling so they align with the multi-level visual tokens. Mask pooling + linear projection produces *mask tokens*; flattening + linear projection of the masks produces *spatial tokens*. **Two segmentation tokens per mask are inserted at fixed positions in the prompt regardless of whether the mask is positive** — the LLM autonomously decides whether a lesion mask is meaningful. Hyperparameters (projector, embedding dim, ViT-B output layer indices, token count) match MAIRA-SEG.

**7. Patient-Specific Attribute Extractor → text tokens $T_{\text{attr}}$.** A deterministic algorithm (RadGPT-style vision translator) extracts from the pseudo-masks: organ volumes (right/middle/left upper/lower lobes, left/right atrium, left/right ventricle, liver, left/right kidney) and lesion attributes (count, diameter in mm, spatial location for nodule / cyst / effusion). These are slotted into a fixed text template (Fig. 5) and tokenized as part of the prompt.

![Patient-Specific Attribute prompt template](/assets/images/paper/medregion-ct/fig_p013_01.png)
*Figure 5: Text-prompt template for Patient-Specific Attributes — pseudo-mask-derived organ volumes and lesion count/diameter/location are slotted into a fixed schema and concatenated with the region tags as input to the LLM.*

**8. LLM backbone.** **LLaMA-3.1-8B with LoRA.** The LLM consumes $T_{\text{vision}}$, $T_{\text{seg}}$, $T_{\text{attr}}$, and the instruction prompt $I$ and emits the per-region report. Optimization stack: AdamW (Loshchilov & Hutter, 2017) with ZeRO (Rajbhandari et al., 2019) for memory.

## Experimental Results

**Dataset.** RadGenome-Chest CT (Zhang et al., 2024), built on CT-RATE (Hamamci et al., 2024) — 25,692 non-contrast chest CT volumes from ~20,000 patients, organ-level segmentation across 197 anatomical regions (SAT-generated), and 665K grounded reports linking each sentence to specific anatomical regions. The 197 SAT regions were collapsed into 6 clinical regions and the corresponding sentences were merged into 6 region-specific ground-truth reports. Splits (official): 20,000 patients / 24,128 volumes for train+val, 1,304 patients / 1,564 volumes for test.

**Main comparison on RadGenome-Chest CT test set** (numbers verbatim from Table 1):

| Method | BLEU-4 | ROUGE-L | METEOR | CA-F1 | Green | GPT-4 |
|---|---|---|---|---|---|---|
| CT2Rep | 0.242 | 0.347 | 0.431 | 0.305 | **0.374** | 44.508 |
| RadFM | 0.262 | 0.365 | 0.489 | 0.350 | 0.234 | 46.178 |
| MedBLIP | 0.279 | 0.373 | 0.494 | 0.370 | 0.215 | 46.825 |
| M3D | 0.288 | **0.391** | 0.497 | 0.435 | 0.297 | 47.243 |
| **Ours (R^2 + Attr + Mask)** | **0.290** | 0.375 | 0.494 | **0.450** | 0.278 | **48.837** |
| Ablation R^2 + Mask | 0.280 | 0.367 | **0.498** | 0.430 | 0.264 | 44.457 |
| Ablation R^2 + Attr | 0.293 | 0.375 | 0.489 | 0.425 | 0.282 | 47.319 |
| Ablation R^2 only | 0.261 | 0.345 | 0.481 | 0.405 | 0.233 | 43.014 |
| Ablation LITA (uniform) | 0.268 | 0.343 | 0.479 | 0.385 | 0.231 | 42.293 |

CA-F1 is averaged across 18 chest-related abnormalities (RadBERT from CT-CLIP). GPT-4 score uses GPT-4o Mini with the M3D evaluation prompt (0–100). Green Score from Ostmeier et al. 2024.

**Key findings.**

1. **R^2-vs-LITA isolation (the cleanest evidence).** Comparing Ablation R^2 to Ablation LITA — same architecture otherwise — directly validates the region-centric slice selection: R^2 beats uniform LITA on every LM-based metric (CA 0.405 vs 0.385, Green 0.233 vs 0.231, GPT-4 43.014 vs 42.293). Picking "highest-mask-pixel" slices captures clinically informative content better than uniform temporal sampling.
2. **Both Mask and Attr help additively.** Each branch independently boosts CA-F1 by ~0.02–0.025 over R^2-alone, and combining all three is the only configuration that wins both BLEU-4 *and* CA-F1 simultaneously.
3. **A small honest inconsistency.** R^2+Attr alone has the highest BLEU-4 (0.293), slightly above the full model — the paper reports this without trying to explain it.
4. **CT2Rep's high Green Score (0.374)** is identified by the authors as a normal-case overfitting artifact, a useful piece of intellectual honesty about the metric landscape.

**Qualitative case (Fig. 3).** A COVID-19 pneumonia volume: MedRegion-CT correctly identifies bilateral GGO and explicitly notes absence of nodules; M3D over-localizes to lower lobes and incorrectly emphasizes "no nodules"; MedBLIP and RadFM omit GGO entirely; CT2Rep hallucinates emphysematous changes and linear atelectasis. The Ablation R^2 variant fails most dramatically — it reports "normal aeration" and misses GGO completely, which the authors use to argue that Mask + Attr are essential rather than decorative.

![Qualitative comparison on a COVID-19 pneumonia case](/assets/images/paper/medregion-ct/fig_p007_01.png)
*Figure 3: Qualitative comparison on a COVID-19 pneumonia case (green = correct clinically important findings, yellow = erroneous claims). Only the full MedRegion-CT cleanly identifies bilateral GGO and correctly excludes nodules; the R^2-only ablation misses GGO entirely.*

## Limitations

The authors implicitly acknowledge the unreliability of the Green metric (CT2Rep's normal-case overfitting) and the cost of 3D encoders (the architectural motivation). The more pointed gaps are not addressed:

- **No external dataset.** No out-of-distribution test, and notably no in-house Asan Medical Center validation despite the affiliation being a major hospital. Generalization is asserted, never measured.
- **No statistical significance.** No variance, no error bars, no multiple seeds. The BLEU-4 margin over M3D (0.290 vs 0.288) is well within noise for a single run, and the GPT-4o Mini evaluator is itself a single-pass black box.
- **No efficiency benchmarks.** The headline efficiency claim — D + T tokens vs D x T — is never benchmarked in wall-clock, FLOPs, or memory. Given that efficiency is the architectural motivation, this is a meaningful omission.
- **No SAT-error sensitivity analysis.** The entire mask + attribute pipeline depends on SAT pseudo-masks; if a mask is wrong, the LLM is fed *incorrect numerical claims as text* with no mechanism to detect or hedge. There is no analysis of how SAT failures propagate to report errors.
- **Only one segmentation backend evaluated.** The "segmentation-model-agnostic" claim is asserted but never tested.
- **The 197 → 6 region merge is a manual choice** with no sensitivity analysis (no 4-region or 8-region comparison).
- **Non-contrast chest CT only.** Generalization to contrast-enhanced or other anatomies is untested.
- **No comparison to Reg2RG**, the most directly related prior work (universal-segmentation-based 3D CT report generation), even though it is cited.
- **Single LLM backbone** (LLaMA-3.1-8B with LoRA). No scaling study.

## Why It Matters for Medical AI

For 3D radiology report generation specifically, MedRegion-CT shows that the right inductive bias is procedural — copy the radiologist's read pattern instead of trying to compress a volume into a single visual embedding. The R^2-vs-LITA ablation is the most transferable lesson: when budget forces you to drop slices, **pick slices by clinical content (mask pixel count) rather than by uniform temporal stride**. The deterministic "vision translator" pattern — convert pseudo-mask geometry to a text prompt the LLM consumes verbatim — is also a useful template for any task where you have reliable structured measurements you would otherwise force the LLM to re-derive from pixels. The catch is the dependency chain: SAT errors silently become text-prompt errors that the LLM is unlikely to question, and the paper does not yet tell us how often that happens.

## References

- Paper: Kyung et al., *MedRegion-CT: Region-Focused Multimodal LLM for Comprehensive 3D CT Report Generation*. arXiv:2506.23102v1 [eess.IV], 29 Jun 2025.
- Code: publicly available (per the abstract).
- Dataset: Zhang et al., *RadGenome-Chest CT*, 2024 — built on Hamamci et al., *CT-RATE*, 2024.
- Pseudo-mask backbone: Zhao et al., *SAT: Segment Anything in 3D Medical Images with Text Prompts*, 2023.
- 2D image encoder: Rad-DINO (ViT-B).
- Mask-aware extractor lineage: MAIRA-SEG; Osprey.
- Slice-sampling baseline: LITA (uniform temporal sampling).
- LLM backbone: LLaMA-3.1-8B with LoRA fine-tuning.
- Most related prior work (cited but not benchmarked against): Reg2RG.
