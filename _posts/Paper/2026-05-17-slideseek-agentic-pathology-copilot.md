---
title: "SlideSeek: Evidence-based Diagnostic Reasoning with a Multi-Agent Copilot for Human Pathology"
excerpt: "A supervisor-explorer agent loop driven by GPT-5-mini wraps PathChat+ (Qwen2.5-14B + CONCH v1.5) to autonomously navigate gigapixel WSIs, reaching top-1 0.860 / top-3 0.927 on the new in-house DDxBench (150 WSIs, 55 diseases)."
categories:
  - Paper
  - LLM-Agents
  - Pathology
permalink: /paper/slideseek-agentic-pathology-copilot/
tags:
  - SlideSeek
  - PathChat+
  - LLM-Agents
  - Multi-Agent
  - Pathology
  - Whole-Slide-Image
  - CONCH
  - Qwen2.5
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- **SlideSeek decouples reasoning from perception.** A GPT-5-mini *supervisor* maintains a differential-diagnosis hypothesis set and dispatches spatially-grounded tasks (`Examine x=1000-2000, y=3000-4000 at 20x`) to parallel *explorer* agents, which call PathChat+ for ROI captions through an `OpenSlide / TRIDENT` `navigate` tool. Reports are emitted with ROI-level citations and a binary HIGH/LOW confidence flag.
- **PathChat+ is the perception engine.** Qwen2.5-14B-Instruct + CONCH v1.5 ViT-L, joined by an attention-pool to 128 tokens + 2-layer MLP, AnyRes-tiled at 448x448 up to a 2x2 grid (effective 896x896), trained on **1.13M instructions / 5.49M Q&A turns / 624k images** in ~1,275 A100 GPU-hours.
- **Headline: top-1 0.860 [0.800, 0.907] and top-3 0.927 [0.880, 0.967] on DDxBench (150 in-house H&E WSIs, 55 oncotree codes).** The strongest ablation is the captioner swap - replacing PathChat+ with GPT-5-mini collapses top-1 to **0.427 (p < 0.001)** - which is the real lever in the system. The weakest is the "reasoning supervisor" claim: GPT-5-mini vs GPT-4.1 is +4.66 pp at **p = 0.142**, statistically not significant, and the paper bundles it under the same "multi-agent reasoning" headline.

## Motivation

A WSI at 20x is on the order of 10^9 pixels; DDxBench slides contain on average **1,020 +/- 783** candidate 20x ROIs. Every pathology MLLM evaluated to date - PathChat, Quilt-LLaVA, LLaVA-Med, and even frontier models like GPT-5 and Gemini 2.5 Pro - is benchmarked on **hand-cropped ROIs**, which (i) sidesteps the navigation problem that dominates real pathologist workflow and (ii) outsources the hardest decision (which regions to look at) to a human curator. Existing learned-navigation work is task-specific and lacks explicit multi-step reasoning.

SlideSeek targets that gap by putting region selection, magnification choice, hypothesis refinement, and report writing inside a single agentic loop, while keeping the visual understanding rooted in a domain-trained MLLM. The clinical pitch is an auditable copilot with region-grounded reasoning chains, including for rare cancers where labeled data is scarce.

## Core Innovation

- **Supervisor-explorer split.** A reasoning LLM (GPT-5-mini) plans; multiple parallel explorer agents (also GPT-5-mini) execute spatially-explicit tasks via two tools - `navigate(x, y, magnification, rationale)` and `submit_report(findings)` - with up to 10 non-overlapping navigation requests per turn. The supervisor decides `sufficient_evidence in {True, False}` after each round.
- **Pathology FM as the captioner.** All visual reasoning is routed through **PathChat+**, a Qwen2.5-14B + CONCH v1.5 MLLM with AnyRes tiling (448x448 base, up to 2x2 grid). When grid > 1, a 448x448 thumbnail is prepended as a global view; image-token count per ROI is **128-640**.
- **Final differential via fused ROIs.** The supervisor selects up to 10 non-overlapping ROIs discovered during exploration and sends them in a single multi-image call to PathChat+ for the primary diagnosis + 2 alternatives. A separate report prompt forces three sections (**Microscopic Findings**, **Differential Diagnosis**, **Critical Assessment**) with region-name citations like `region-mag_20-x_1000-y_2000`.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | SlideSeek reaches top-1 0.860 / top-3 0.927 on DDxBench | Figure 3B, Extended Data Table 11; 1,000-replicate bootstrap CIs | DDxBench (n=150, in-house) | ⭐⭐ - single in-house benchmark, no external validation, unblinded labeler |
| C2 | Beats general-purpose MLLMs by "up to 42%" on DDxBench | Figure 3C / Extended Data Table 12: PathChat+ 0.800 vs Gemini 2.5 Pro 0.513 on 10 expert-curated ROIs | DDxBench-ROIs (same 150 slides) | ⭐⭐ - true in the ROI-feed setting, but the "42%" pairs SlideSeek's full agent number to a non-agentic baseline |
| C3 | Strong rare-disease performance (top-1 0.818, top-3 0.899) | Figure 3B rarity stratification; unpaired permutation p < 0.05 for top-1 | DDxBench rare subset (n=51) | ⭐⭐ - rare set spans 41 classes with ~1.2 slides per class; per-class generalization untested |
| C4 | Multi-agent supervisor-explorer split is necessary | Figure 3D-E: top-1 0.860 -> 0.780 when supervisor is removed (p < 0.05) | DDxBench | ⭐⭐ - single ablation, no multi-seed variance; conflates parallelism + role-split |
| C5 | Reasoning supervisor (GPT-5-mini) beats non-reasoning (GPT-4.1) | Figure 3D-E: 0.860 vs 0.813, Delta = 4.66 pp | DDxBench | ⭐ - **p = 0.142, not significant**; top-3 identical (0.927). Abstract bundles it under "ablations confirmed the need for multi-agent reasoning" |
| C6 | A pathology-specific captioner is essential | Figure 3F-G: swapping PathChat+ for GPT-5-mini drops top-1 0.860 -> 0.427 (p < 0.001); swap to PathChat 1 -> 0.640 (p < 0.001) | DDxBench | ⭐⭐⭐ - large, clear effect, multiple captioner alternatives; the strongest claim in the paper |
| C7 | PathChat+ is SOTA across 10 ROI benchmarks vs 12 baselines | Figure 2, Extended Data Tables 1-9; paired permutation n = 1000 | PathMMU 5 subsets, PathQABench MCQ + Caption, BRACS, UniToPatho, HiCervix | ⭐⭐⭐ - broad, multi-dataset, multi-baseline, **10/10 wins** (one borderline at p = 0.099) |
| C8 | Efficient exploration: only ~156 of ~1,020 high-power ROIs inspected | Mean +/- SD reported | DDxBench | ⭐⭐ - descriptive only; no sliding-window control at matched compute |
| C9 | Confidence flag is calibrated (0.906 high vs 0.778 low, p < 0.05) | "Interpretable slide assessment" section | DDxBench | ⭐⭐ - binary buckets only; no reliability diagram, Brier, or ECE |
| C10 | System enables auditable, region-grounded reasoning | Figure 4 case study, Extended Data Figure 4 | DDxBench, anecdotal | ⭐ - qualitative only, no pathologist user study, no comparison to other report-generation systems |
| C11 | DDxBench is a useful new benchmark | Construction in Methods; 19 sites x 55 diagnoses | DDxBench (in-house) | ⭐ - **in-house only, not publicly downloadable**; reproducibility depends on IRB-mediated requests |

**Honest read.** The strongest evidence is the **captioner ablation (C6)** and the **PathChat+ ROI sweep (C7)**: both have multi-condition designs with proper statistical tests. The headline number (C1) is well-engineered but rests on a single 150-slide, in-house benchmark whose labels and predicted-vs-true correctness judgments come from the same institution that built the model - **no external validation**, no kappa, no second rater. C5 is **oversold**: the cited p is 0.142 and top-3 is unchanged. The bibliography cites **PathFinder** as the closest prior multi-agent diagnostic system, but there is **no head-to-head numerical comparison** despite the framing of SlideSeek as a successor to that line.

## Method & Architecture

![SlideSeek supervisor-explorer loop with PathChat+ training corpus](/assets/images/paper/slideseek/page_003.png)
*Figure 1: SlideSeek's supervisor-explorer loop wraps PathChat+ to autonomously navigate gigapixel WSIs and emit a visually grounded differential-diagnosis report. The instruction corpus that produced PathChat+ spans 1.13M instructions and 624k images across H&E, IHC, and cytology stains.*

**SlideSeek agentic loop.** A user supplies a WSI + system prompt with tissue site, patient sex, and the diagnostic task. The supervisor sees a low-resolution thumbnail plus a TRIDENT-generated slide description (dimensions, magnification, list of tissue bounding boxes) and emits an initial differential hypothesis set + a structured plan. Each task carries explicit `(region_id, x-range, y-range, magnification, target features, contextual rationale)`. Magnification tiers are categorized as **low (1.25-2.5x), medium (5-10x), high (20-40x)**. Explorers execute in parallel; each call to `navigate(...)` fetches an ROI through OpenSlide, forwards it to PathChat+ for a morphological caption, and returns the description + annotated thumbnail. The supervisor consumes findings, updates hypotheses with justifications, and decides `sufficient_evidence`. When done, it selects up to 10 non-overlapping ROIs and sends them in a single multi-image call to PathChat+ for the final differential.

**PathChat+ architecture.** Vision encoder = CONCH v1.5 (ViT-L); projector = attention-pool to 128 tokens followed by a 2-layer MLP with GeLU; LLM = Qwen2.5-14B-Instruct. AnyRes tiling pads each image to the nearest supported grid and splits into 448x448 tiles (max 2x2 -> 896x896). Two-stage training: (1) freeze LLM, train projector on a 500k caption subset; (2) unfreeze projector + LLM, loss only on answer tokens. Compute: pretraining on 8xA100-80GB, finetuning on 24xA100-80GB via DeepSpeed - total ~1,275 A100 GPU-hours. Inference on a 24 GB RTX 3090.

**Data curation.** Images <336x336 are dropped, a CONCH-encoder classifier rejects non-pathology images, lightweight LLMs (Qwen2-7B, Llama 3) filter low-quality / over-generic captions, and source captions are rewritten into pre-defined instruction formats vetted by board-certified pathologists. MS-COCO natural images are mixed in to teach refusal (`"Sorry I can only assist you with queries related to pathology."`). Final corpus: 1,133,241 instruction examples, 5.49M Q&A turns, 624k unique images (median 759x607 px), excluding 8,034 guardrail examples.

## Experimental Results

### Main DDxBench (Figure 3, Extended Data Table 11)

![DDxBench main results and ablations](/assets/images/paper/slideseek/page_006.png)
*Figure 3: On DDxBench (150 WSIs, 55 diseases), SlideSeek reaches top-1 0.860 / top-3 0.927. Ablating the supervisor-explorer split drops top-1 by 8 pp (p < 0.05); swapping PathChat+ for GPT-5-mini as captioner collapses top-1 to 0.427 (p < 0.001).*

| Configuration | Top-1 (n=150) | Top-3 | Notes |
|---|---|---|---|
| **SlideSeek (PathChat+ + GPT-5-mini supervisor, multi-agent)** | **0.860 [0.800, 0.907]** | **0.927 [0.880, 0.967]** | full system |
| SlideSeek, non-reasoning supervisor (GPT-4.1) | 0.813 | 0.927 | Delta = -4.66 pp, **p = 0.142 (n.s.)** |
| SlideSeek, single-agent (no supervisor/explorer split) | 0.780 | 0.880 | Delta = -8.00 pp, p < 0.05 |
| SlideSeek with PathChat 1 captioner | 0.640 | 0.807 | Delta = -22.0 / -12.0 pp, p < 0.001 |
| SlideSeek with GPT-5-mini captioner | 0.427 | 0.627 | Delta = -43.3 / -30.0 pp, p < 0.001 |
| PathChat+ on 10 expert-curated ROIs (no agent) | 0.800 | 0.920 | Delta vs full SlideSeek = -6.0 pp, p = 0.059 |
| PathChat 1 on 10 expert-curated ROIs | 0.720 | 0.887 | +8.0 pp gap to PathChat+, p < 0.05 |
| Gemini 2.5 Pro on 10 expert-curated ROIs | 0.513 | 0.713 | best non-pathology baseline |

**Rarity stratification.** Common 0.941 [0.882, 1.000] (n=99) vs rare 0.818 [0.737, 0.889] (n=51), unpaired permutation p < 0.05 for top-1; the top-3 gap (0.980 vs 0.899) is *not* significant (p = 0.095).

### PathChat+ ROI benchmarks (Figure 2)

![PathChat+ ROI benchmark sweep](/assets/images/paper/slideseek/page_005.png)
*Figure 2: Across VQA (PathMMU + PathQABench MCQ), classification (BRACS, UniToPatho, HiCervix), and captioning (PathQABench Caption), PathChat+ is statistically the strongest model in 10/10 tasks (paired permutation, n=1000), beating Gemini 2.5 Pro, GPT-5, Claude Sonnet 4, and prior pathology MLLMs.*

| Benchmark | PathChat+ | Best frontier MLLM | Best pathology MLLM | Delta |
|---|---|---|---|---|
| PathMMU (All-test, n=9,618) | **0.751** | Gemini 2.5 Pro / GPT-5 tied at 0.679 | PathChat 1 0.609 | +7.2 / +14.2 pp |
| PathMMU Atlas (test-all) | **0.840** | Gemini 2.5 Pro 0.752 | PathChat 1 0.753 | +8.8 / +8.7 pp |
| PathMMU PathCLS (test-all) | **0.748** | Gemini 2.5 Pro 0.561 | PathChat 1 0.452 | +18.7 / +29.6 pp |
| PathQABench MCQ (n=105) | **0.943** | Gemini 2.5 Pro 0.876 (p = 0.099) | PathChat 1 0.895 | +6.7 / +4.8 pp |
| BRACS MCQ (n=570) | **0.633** | Gemini 2.5 Pro 0.372 | PathChat 1 0.558 | +26.1 / +7.5 pp |
| UniToPatho MCQ | **0.522** | Gemini 2.5 Pro 0.401 | PathChat 1 0.506 | +12.1 / +1.6 pp |
| HiCervix MCQ (n=8,051) | **0.715** | GPT-5 0.456 | PathChat 1 0.368 | +25.9 / +34.7 pp |
| PathQABench Caption (METEOR) | **0.281** | Gemini 2.5 Pro 0.248 | PathChat 1 0.263 | +3.3 / +1.8 |

### Exploration cost and calibration

SlideSeek examines **194.5 +/- 102.8 regions per slide** (156.4 high-power + 31.1 medium + 6.9 low) out of ~1,020 candidate 20x ROIs - **~15% of high-magnification regions** are actually inspected. The number of examined regions does **not** correlate with accuracy. Self-reported confidence is mildly calibrated: **0.906 [0.844, 0.958]** on the 96 high-confidence cases vs **0.778 [0.667, 0.889]** on the 54 low-confidence cases (p < 0.05). The paper does not report a reliability diagram, Brier score, or ECE.

### Qualitative trace

![Follicular thyroid carcinoma trace](/assets/images/paper/slideseek/page_008.png)
*Figure 4: End-to-end SlideSeek trace on a follicular thyroid carcinoma case - the supervisor refines its hypothesis from 1.25x architecture to 20x cellular detail, identifies vascular invasion, and synthesizes a primary + two differential diagnoses with ROI-level evidence.*

### Failure modes (paper's own breakdown)

- **Tumor grading errors (5/11 misclassified, 45%)** - all astrocytoma; tumor type correct, WHO grade wrong (3 overcalls, 2 undercalls).
- **Missed small lesions (1/150)** - small foci of Merkel Cell Carcinoma missed amid benign tissue.
- **Modality hallucination (1/150)** - supervisor requested IHC; explorer hallucinated IHC findings on an H&E slide.

## Limitations

**Authors' acknowledged limitations.** Tumor grading is unreliable even when the tumor type is correct (the astrocytoma failure mode). Navigation can miss small clinically-decisive foci. The supervisor can hallucinate diagnostic tasks the input cannot support (IHC on H&E). Scope is single-modality, single-stain (H&E); multi-slide, multi-stain, and tumor-board emulation are sketched as future work.

**Issues the paper does not address head-on.**
- **No external validation.** DDxBench is BWH-only - no TCGA WSIs, no CPTAC, no slides from a non-Harvard institution. Domain shift across scanners, stain protocols, and tissue-prep is the dominant clinical failure mode for pathology AI and is untested here.
- **No multi-seed agent runs / no cost reporting.** The supervisor + explorers + PathChat+ make hundreds of LLM/MLLM calls per slide (~195 ROI evaluations + planning + differential + report). No token costs, no wall-clock per slide, no seed-to-seed variance.
- **Same-institution evaluator.** Open-ended differentials are judged by one board-certified pathologist implicitly part of the lab ecosystem; **no inter-rater kappa**.
- **PathFinder comparison missing.** PathFinder is the most directly comparable multi-agent diagnostic system; SlideSeek is not benchmarked against it.
- **Backbone-comparison gap.** "GPT-5 vs GPT-5-mini no significant difference" is asserted in prose without numbers - leaves open whether scaling the backbone would close the rare-disease gap.
- **Class-balance opacity.** 55 diseases / 150 slides means several classes are tested with n=1; per-class breakdowns are not in the main text.

## Why It Matters for Medical AI

The lesson that survives this paper is **architectural**: the captioner ablation (top-1 0.860 -> 0.427 when PathChat+ is swapped for GPT-5-mini) shows that for slide-level diagnosis the **pathology FM is the binding constraint** - not the reasoning LLM. A clinical copilot built on a frontier LLM alone, no matter how clever the agent scaffolding, leaves ~43 pp of accuracy on the table. Conversely, a strong domain MLLM with a thin agentic shell already gets within ~6 pp of the full SlideSeek system. For teams building computational-pathology assistants, the implication is to invest in the domain encoder + captioner first and the agent scaffolding second.

The honest caveat is that none of this has been shown on an external cohort yet. Until SlideSeek is run on slides from another institution, with a different scanner and stain protocol, and judged by a panel of independent pathologists, the 0.860 number is a strong proof of concept rather than a clinical claim.

## References

- Paper (arXiv): [Evidence-based diagnostic reasoning with multi-agent copilot for human pathology - arXiv:2506.20964](https://arxiv.org/abs/2506.20964) (v2, 26 Mar 2026)
- Related work cited here:
  - PathChat (the predecessor MLLM): Lu et al., Mahmood Lab.
  - CONCH v1.5 (vision encoder backbone): Lu et al., Nature Medicine.
  - TRIDENT (WSI preprocessing pipeline): Mahmood Lab.
  - PathFinder (closest prior multi-agent diagnostic system, not benchmarked head-to-head): Ghezloo et al.
  - Qwen2.5-14B-Instruct (LLM backbone): Qwen Team, Alibaba.
- Benchmarks referenced: PathMMU, PathQABench (MCQ + Caption), BRACS, UniToPatho, HiCervix; DDxBench is in-house and not publicly downloadable.
