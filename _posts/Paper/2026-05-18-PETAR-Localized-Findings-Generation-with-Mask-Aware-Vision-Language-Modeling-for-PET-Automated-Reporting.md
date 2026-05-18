---
title: "PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting"
excerpt: "First public lesion-grounded PET/CT findings dataset (PETARSeg-11K, 11,356 lesions across 5,126 exams) and a mask-aware 3D VLM (PETAR-4B) that beats the strongest 3D baseline by +0.19 GREEN — but the ablation suggests the focal crop, not the mask, is doing most of the work."
categories: [Paper, CT-Report-Generation, Dataset]
permalink: /paper/petar/
tags:
  - PETAR
  - PET-CT
  - Report-Generation
  - 3D-VLM
  - Mask-Aware
  - Phi3
  - M3D-CLIP
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

![PETAR teaser](/assets/images/paper/petar/fig_p001_01.png)
*Figure 1: Lesion-mask conditioning lets PETAR produce site- and laterality-specific findings (left and right axillary lymph-node descriptions with SUVmax and slice numbers), while a generic 3D VLM hallucinates an unrelated lesion in the mandible.*

## TL;DR

- **Dataset:** PETARSeg-11K — first public PET/CT lesion-grounded findings corpus: **5,126 whole-body exams / 11,356 lesion-level findings**, four tracers (FDG, DOTATATE, fluciclovine, DCFPyL). Masks are **algorithmic, not human-annotated**.
- **Model:** PETAR-4B — M3D-CLIP 3D-ViT + Phi3-4B with element-wise additive mask conditioning on the PET stream only, plus a jittered cubic focal crop around the lesion. Three-stage training (projector → mask-PatchEmbed → full FT).
- **Headline:** **BLEU 0.535 / GREEN 0.257 vs M3D-RAD FT 0.485 / 0.071** on a 1,175-sample test set; 5 nuclear-medicine physicians rate outputs 3.7–3.9 vs 4.3–4.4 for human reports. But the ablation shows the **focal crop alone** lifts CIDEr 0.132 → 0.397 and GREEN 0.071 → 0.226 — the "mask-aware" framing is over-attributed.

## Motivation

Whole-body PET/CT reports are arguably the longest texts in radiology (up to 3× CT length, per Tie et al. 2024). They enumerate lesions with site / sub-site / laterality / morphology / metabolic activity — granularity that demands per-lesion grounding. Yet existing 3D medical VLMs (M3D, Merlin, CT2Rep, RadFM, Med3DVLM) are CT-only and rely on global pooling that erases sub-percent-volume lesions; in PETARSeg-11K, lesions occupy **0.01–2% of the volume, mean <0.1%**.

No prior public dataset combines whole-body PET/CT, free-text findings, and grounded 3D masks. autoPET / SegAnyPET have masks but no text; ViMed-PET and Pet2Rep have text but no masks. PETAR plugs that hole on both data and architecture sides.

## Core Innovation

Two ideas, one of which actually carries the paper:

1. **Mask-aware conditioning of PET tokens.** A separate (zero-initialized) `PatchEmbed` projects the binary lesion mask into the visual token space. Mask tokens are then added **element-wise** to PET tokens *only* — not cross-attention, not sequence concatenation:

   $$X_\text{PET} = T(Z_P + Z_M), \quad X_\text{CT} = T(Z_C)$$

   followed by `Concat` along the embedding dimension. The mask `PatchEmbed` is unlocked in a dedicated Stage 2 of training while everything else is frozen.

2. **Jittered cubic focal prompt.** A cube is centered on the lesion centroid $c$ with side $r$, then both $c$ and $r$ are perturbed by $\mathcal{U}(-0.2r, 0.2r)$ while keeping the mask fully contained. The focal branch supplies a high-resolution local view; the global branch keeps body context. Global and focal token streams are summed before being projected into the LLM.

This is a 3D adaptation of the Describe-Anything focal-prompt idea, applied to dual-modality (functional PET + anatomical CT) volumes.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | PETARSeg-11K is the first large-scale public PET/CT dataset with lesion-level mask↔text alignment | Table 1 vs RIDER-Lung, Head-Neck PET-CT, FDG-PET-CT-Lesions, SPADe, SegAnyPET, Pet2Rep, ViMed-PET | ⭐⭐⭐ |
| C2 | PETAR-4B outperforms all 2D/3D baselines on PETARSeg-11K | Table 2; +0.325 CIDEr and +0.186 GREEN over M3D-RAD FT | ⭐⭐ (single dataset, single seed, no variance) |
| C3 | Mask-guided conditioning and focal prompting cause the gain | Table 3 ablation | ⭐⭐ (focal prompt does almost all the work; mask-only contribution is barely measurable) |
| C4 | Element-wise mask conditioning beats bbox/coord prompting (ViP-LLaVA, Reg2RG) | Table 2: ViP-LLaVA 0.006 GREEN, Reg2RG-FT 0.031, PETAR 0.257 | ⭐⭐ (Reg2RG was designed for organ-level CT, so comparison is partially unfair) |
| C5 | Clinically useful per 5-physician reader study | Table 5: anat. 3.9, interp. 3.9, utility 3.7 vs human 4.4/4.4/4.3; ~60% preferred-or-tied | ⭐⭐⭐ (real cohort, sound methodology; IAA between readers not reported) |
| C6 | OOD generalization to autoPET | Table 5 external: 3.8/4.0/3.8 — within 0.1 of internal | ⭐⭐ (n=32, no automated metrics) |
| C7 | GREEN best correlates with physician judgment for PET RG (ρ=0.592) | Table 4 Spearman | ⭐⭐ (single study; moderate, not strong, correlations) |
| C8 | TotalSegmentator pretraining helps | Table 3 last two rows: GREEN 0.239 → 0.257 | ⭐⭐ (real but small) |
| C9 | PETARSeg-11K masks are reliable (98% contour accuracy) | Cites Huemann et al. 2025 sample study | ⭐ (**inherited**, not re-measured on PETARSeg-11K; masks are algorithmic, so there is no human IAA to report) |

## Method & Architecture

![PETAR architecture](/assets/images/paper/petar/fig_p005_01.png)
*Figure 2: PETAR-4B architecture. Global PET, CT, and mask tokens (left) plus a perturbed focal crop $(F_P, F_C, F_M)$ go through a shared M3D-CLIP ViT encoder; the mask projector adds its tokens element-wise to the PET stream only, PET and CT streams are concatenated along the embedding dim, spatially pooled, and fed to Phi3-4B with a fixed lesion-description prompt.*

**Inputs.** $P, C \in \mathbb{R}^{D\times W \times H}$ (PET, CT resampled to 3 mm isotropic, 192×192×352); binary mask $M \in \{0,1\}^{D \times W \times H}$. Goal: $y = f_\theta(P, C, M)$.

**Encoding pipeline.**
1. PET, CT, and mask are independently patchified into $K$ tokens of dim $d$. Mask uses a separate, zero-initialized `PatchEmbed`.
2. Mask tokens are added element-wise to PET tokens (CT is *not* touched). The shared 3D-ViT $T$ then encodes each stream: $X_\text{PET} = T(Z_P + Z_M)$, $X_\text{CT} = T(Z_C)$.
3. Streams are concatenated along the embedding dim: $X = \text{Concat}(X_\text{PET}, X_\text{CT})$.
4. The same pipeline runs on the focal crop $(F_P, F_C, F_M)$ to produce $\tilde X$. The two are summed: $T = X + \tilde X$, then spatially pooled and projected into the LLM space: $V = \text{Proj}(\text{SpatialPooler}(X + \tilde X))$.
5. Phi3-4B receives $(q, V)$ where $q$ is a fixed prompt ("Please describe the region highlighted in the PET image.") and autoregressively emits $y$.

**Three-stage training**, repeated first on TotalSegmentator pretraining and then on PETARSeg-11K:

- **Stage 1 — projector alignment.** Train only the visual→LLM projector. Mask-PatchEmbed weights are zero; encoder and LLM frozen.
- **Stage 2 — mask-PatchEmbed alignment.** Unfreeze only the mask `PatchEmbed`. Everything else frozen. This is the dedicated step where the additive mask path learns to do anything.
- **Stage 3 — full fine-tune.** End-to-end on PETARSeg-11K with autoregressive NLL.

**Compute.** 2× NVIDIA L40S, 10 epochs/stage, ~20 hours total, bf16.

## Dataset — PETARSeg-11K

![PETARSeg pipeline](/assets/images/paper/petar/fig_p003_01.png)
*Figure 3: PETARSeg-11K construction. An LLM ensemble (Mistral-7B + Mixtral-8×7B) extracts (SUVmax, slice) from each report sentence; iterative thresholding + region-growing (Jentzen 2015) segments the matching lesion in PET; Qwen3-30B-A3B then reformats the free-text finding into a structured (Region / Organ / Sub-site / SUVmax / Slice / Findings) record linked to the 3D mask.*

- **Source:** 33,000 PET/CT exams from a **single institution** (University of Wisconsin), IRB-waived, NLM-Scrubber de-identified. Filtered to 5,126 whole-body exams → **11,356 lesion-level findings**.
- **Lesion masks:** generated by Huemann et al. 2025's pipeline — LLMs extract `(SUVmax, slice)` from report sentences; PET volume is thresholded at the reported SUVmax; the connected component whose SUVmax matches within ±0.1 *and* intersects the reported axial slice is grown from the max-pixel until the contour stabilizes. **No manual segmentation.**
- **Mask quality:** the only number disclosed is **98% contour location accuracy, inherited from Huemann et al. 2025, not re-measured on PETARSeg-11K**. There is no inter-annotator agreement because there is no annotator — masks are algorithmic.
- **Reference findings:** Qwen3-30B-A3B (an LLM) reformats each clinical finding into the structured schema above. A layer of LLM rewriting therefore sits between original report and ground truth.
- **Tracers:** 18F-FDG, 68Ga-DOTATATE, 18F-fluciclovine, 18F-DCFPyL.
- **Pretraining corpus:** TotalSegmentator on the same CTs → ~100,000 mask–label pairs across 117 anatomical classes, framed as VQA.
- **Splits:** held-out test of **1,175** lesion-level samples; OOD = **32 manually curated autoPET cases** (no reports → reader scoring only).
- **License/release:** the paper says "publicly available" but the body does not surface a license, data-use agreement, or release URL. Worth waiting for the public drop before treating it as a benchmark.

**Construction biases to flag.** The mask pipeline keys on SUVmax + slice extracted from the report. Lesions that the radiologist describes qualitatively without those numbers are systematically excluded — the dataset is biased toward measurable, prominent, FDG-avid lesions and likely under-represents subtle or non-FDG-avid disease.

## Experimental Results

### Main comparison — PETARSeg-11K test (Table 2)

| Model | BLEU | ROUGE-L | METEOR | CIDEr | BERTScore | BARTScore | RaTE | GREEN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *2D zero-shot* | | | | | | | | |
| MedGemma-4B | 0.124 | 0.352 | 0.358 | 0.027 | 0.690 | −4.92 | 0.540 | 0.011 |
| HuatuoGPT-Vision-7B | 0.130 | 0.350 | 0.357 | 0.024 | 0.695 | −4.85 | 0.584 | 0.015 |
| InternVL3-8B | 0.137 | 0.355 | 0.356 | 0.035 | 0.694 | −4.72 | 0.614 | 0.030 |
| Qwen3-VL-8B | 0.375 | 0.343 | 0.364 | 0.051 | 0.690 | −4.74 | 0.612 | 0.022 |
| QoQ-Med | 0.364 | 0.317 | 0.368 | 0.025 | 0.680 | −4.64 | 0.505 | 0.006 |
| ViP-LLaVA (prompt-aware) | 0.064 | 0.301 | 0.308 | 0.009 | 0.651 | −5.96 | 0.503 | 0.006 |
| *2D fine-tuned* | | | | | | | | |
| Qwen3-VL-8B FT | 0.484 | 0.443 | 0.501 | 0.086 | 0.750 | −4.40 | 0.608 | 0.060 |
| MedGemma-4B FT | 0.495 | 0.454 | 0.510 | 0.119 | 0.754 | −4.36 | 0.613 | 0.086 |
| *3D* | | | | | | | | |
| Med3DVLM | 0.004 | 0.080 | 0.066 | 0.005 | 0.511 | −5.70 | 0.285 | 0.002 |
| M3D | 0.327 | 0.276 | 0.323 | 0.013 | 0.634 | −5.32 | 0.505 | 0.000 |
| M3D-RAD | 0.343 | 0.300 | 0.340 | 0.016 | 0.658 | −5.23 | 0.518 | 0.003 |
| Reg2RG (mask-aware) | 0.044 | 0.060 | 0.108 | 0.001 | 0.518 | −5.54 | 0.363 | 0.002 |
| Reg2RG FT | 0.478 | 0.416 | 0.487 | 0.055 | 0.732 | −4.58 | 0.532 | 0.031 |
| M3D-RAD FT | 0.485 | 0.446 | 0.501 | 0.132 | 0.750 | −4.34 | 0.627 | 0.071 |
| **PETAR-4B (Ours)** | **0.535** | **0.524** | **0.560** | **0.457** | **0.795** | **−4.00** | **0.713** | **0.257** |

Three observations.

1. **Off-the-shelf 3D medical VLMs are unusable on PET** — Med3DVLM/Reg2RG zero-shot score lower than weak 2D zero-shot baselines, confirming a wide PET domain gap.
2. **Fine-tuning closes most of the n-gram gap** (BLEU 0.485 vs PETAR 0.535) but leaves a **huge GREEN gap** (0.071 vs 0.257) and a non-trivial CIDEr gap (0.132 vs 0.457) — PETAR's wins are concentrated in metrics that reward clinically faithful phrasing.
3. **The closest architectural competitor — Reg2RG, the only other mask-aware 3D model — is beaten by ~0.06 BLEU and ~0.23 GREEN.** This is the paper's most defensible positive result, with the caveat below.

### Ablation (Table 3) — what's actually doing the work

| Mask | CT | Focal | TS | BLEU | CIDEr | GREEN |
|:---:|:---:|:---:|:---:|---:|---:|---:|
| × | × | × | × | 0.485 | 0.132 | 0.071 |
| ✓ | × | × | × | 0.480 | 0.137 | 0.088 |
| × | ✓ | × | × | 0.477 | 0.134 | 0.060 |
| × | × | ✓ | × | 0.528 | 0.397 | 0.226 |
| ✓ | × | ✓ | × | 0.525 | 0.381 | 0.232 |
| × | ✓ | ✓ | × | 0.517 | 0.428 | 0.234 |
| ✓ | ✓ | ✓ | × | 0.521 | 0.439 | 0.239 |
| ✓ | ✓ | ✓ | ✓ | **0.535** | **0.457** | **0.257** |

Read this row by row.

- **Focal prompt alone** lifts CIDEr 0.132 → 0.397 and GREEN 0.071 → 0.226. That single ingredient accounts for ~80% of the headline gain.
- **Mask only, no focal:** CIDEr 0.132 → 0.137, GREEN 0.071 → 0.088. Essentially flat.
- **CT only, no focal:** GREEN 0.071 → 0.060. Slight *regression*.
- **Mask + CT + focal + TS pretraining** is the configuration that finally gets to 0.257.

The honest reading: **the high-resolution lesion patch delivered by the focal crop is doing most of the work**, and the additive mask token becomes useful mostly as a region-disambiguation signal *once that patch is supplied*. The paper's "mask-aware" framing over-attributes credit; "focal-crop + mask disambiguation" would be a more accurate label.

### Metric vs. human correlation (Table 4, Spearman ρ)

GREEN 0.592 > RaTEScore 0.550 > BERTScore 0.511 > ROUGE 0.471 > METEOR 0.438 > CIDEr 0.421 > BLEU 0.214 > BARTScore 0.168.

n-gram metrics correlate **poorly** with physician judgment for PET findings. Even GREEN, the best proxy, sits at moderate ρ ≈ 0.59 — useful but not sufficient. Headline BLEU numbers in this domain should be read with appropriate skepticism.

### Reader study (Table 5)

5 board-certified nuclear-medicine physicians scored 116 internal + 32 external autoPET cases on a 5-point rubric (anatomical accuracy / interpretation / utility), excluding numeric measurements (they are LLM-hallucinated and post-overridden from the mask).

|  | Internal: Human | Internal: PETAR-4B | External (autoPET): PETAR-4B |
|---|---:|---:|---:|
| Anatomical | 4.4 ± 0.9 | 3.9 ± 1.2 | 3.8 ± 1.1 |
| Interpretation | 4.4 ± 0.8 | 3.9 ± 1.1 | 4.0 ± 1.0 |
| Utility | 4.3 ± 0.9 | 3.7 ± 1.1 | 3.8 ± 1.1 |

Pairwise preference on 116 internal cases: PETAR strictly preferred 15 / tie 54 / human preferred 47 — physicians preferred-or-tied PETAR in ~60% (69/116). External autoPET scores are within 0.1 of internal — the strongest robustness signal in the paper, even though the cohort is tiny.

![Qualitative comparison](/assets/images/paper/petar/fig_p008_01.png)
*Figure 4: Qualitative comparison. Red underlines mark incorrect anatomical descriptors, green correct ones — pre-FT M3D-RAD and MedGemma hallucinate distant anatomy; even after fine-tuning baselines mis-localize (e.g., an inguinal node called "left proximal femur"); PETAR consistently aligns text with the masked lesion.*

## Limitations

**Author-acknowledged:**
- Requires lesion masks at inference time. Proposed mitigations are clinician single-click tools (PETEdge+) or upstream segmentation networks — neither evaluated here.
- Numeric values (lesion size, SUVmax) are hallucinated by the LLM and intended to be overwritten from the mask in post-processing.

**Additional issues this review flags:**

1. **No human inter-annotator agreement on PETARSeg-11K.** Masks are algorithmic. The only quality number — 98% contour accuracy — is inherited from Huemann et al. 2025 and not re-measured on the new corpus. Reference findings are LLM-restructured by Qwen3-30B-A3B; any LLM rewriting error propagates into ground truth.
2. **No grounding metric despite "spatially grounded" framing.** No IoU/Dice between described lesions and ground-truth lesions; no penalty for talking about the wrong adjacent lesion. The mask is always supplied as input — the model is never asked to choose which of several lesions to describe. "Grounded" here means *conditioned on a mask*, not *grounding evaluated*.
3. **The mask ablation undercuts the central architectural claim.** Focal-prompt does almost all the work (CIDEr 0.132 → 0.397, GREEN 0.071 → 0.226). Mask-only without focal is essentially flat.
4. **No head-to-head with prior PET RG systems.** PETRG-3D, ViMed-PET, and Pet2Rep are the closest dedicated baselines — none compared. The PET-specific competitor set is limited to generic VLMs and Reg2RG (designed for CT organs, not PET lesions).
5. **No variance, no multi-seed, no confidence intervals** on any quantitative result. Single-run point estimates.
6. **OOD evaluation is 32 autoPET cases with 5 readers and no automated metrics.** Directionally encouraging, not statistically compelling.
7. **No mask-quality robustness curve.** Performance is never reported as a function of input mask IoU degradation, so the practical "PETAR-segmentation-in-the-loop" deployment story is untested.
8. **Single institution, restricted tracer set, public release terms not stated in body.**

## Why It Matters for Medical AI

Despite the caveats, this paper does two things the field needs.

- **A PET/CT lesion-grounded dataset, even an algorithmic one, is a real public good.** Until now PET report-generation work has been gated behind private institutional corpora; PETARSeg-11K — assuming the release actually materializes — gives the rest of the community a 1,175-sample test bed.
- **Focal-crop conditioning is the take-home recipe.** The ablation makes it clear that for sub-percent-volume lesions, dragging a high-resolution local patch into the encoder is the single highest-leverage change. The mask path is the credit-assigning narrative; the focal crop is the cause. Future 3D RG work on tiny anatomies (lung nodules, lymph nodes, brain mets) should reach for focal-prompt augmentation before architectural exotica.

What is *not* yet shown — and what the medical-AI community should ask for in v3 — is grounded-text evaluation (IoU/Dice between described and ground-truth lesions), mask-quality robustness, and a comparison against PETRG-3D / ViMed-PET / Pet2Rep on a shared evaluation protocol.

## References

- arXiv: [2510.27680](https://arxiv.org/abs/2510.27680) (v2, 30 Nov 2025)
- M3D / M3D-CLIP backbone: Bai et al., M3D — Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models, 2024
- Mask construction pipeline: Huemann et al., 2025 (cited for SUVmax/slice → iterative-threshold lesion extraction; source of the inherited 98% contour-accuracy figure)
- Iterative thresholding base: Jentzen, 2015
- Reference-finding restructuring: Qwen3-30B-A3B
- Closest 3D mask-aware competitor: Reg2RG (CT organ-level RG)
- Comparable PET-RG work not compared: PETRG-3D, ViMed-PET, Pet2Rep
- External dataset for OOD reader study: autoPET
