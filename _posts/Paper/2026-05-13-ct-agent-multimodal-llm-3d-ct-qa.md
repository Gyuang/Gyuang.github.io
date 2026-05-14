---
title: "CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering"
excerpt: "An anatomy-decomposed LoRA-tool ensemble with hierarchical token compression that lifts CE-F1 from 0.221 to 0.420 on CT-RATE report generation."
categories:
  - Paper
  - CT-Report-Generation
  - LLM
tags:
  - CT-Agent
  - 3D-CT
  - Multimodal-LLM
  - LoRA
  - Token-Compression
  - Retrieval-Augmented-Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
permalink: /paper/ct-agent-multimodal-llm-3d-ct-qa/
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- CT-Agent decomposes 3D chest CT understanding into **ten anatomy-specific LoRA "tools"** coordinated by a Deepseek-v3 planner, with prediction-guided few-shot retrieval over a 20k-case corpus.
- A **hierarchical token compression** (MoE-based Global Token Aggregation + VisionZip-style Local Token Selection) compresses each slice from 256 to 64 tokens — a **~75% token reduction** — claimed to preserve cross-slice semantics.
- On CT-RATE report generation, **CE-F1 jumps from 0.221 (LLaVA-CT) to 0.420**, driven mainly by a CE-Recall surge (0.182 → 0.477). However, BLEU-4 actually regresses (0.231 vs LLaVA-CT 0.252 / M3D 0.245) and several QA regions get worse, undermining the abstract's "consistently outperforms" framing.

## Motivation

A 3D chest CT volume is hundreds of axial slices, which after a CLIP encoder explodes into tens of thousands of visual tokens — far beyond what current MLLMs can ingest. Existing 2D systems (R2Gen, MAIRA, XrayGPT, PMC-VQA) cannot model inter-slice spatial structure, and 3D-aware predecessors (CT2Rep, 3D-CT-GPT, MS-VLM, M3D) treat the volume as one homogeneous block, failing to (a) decompose anatomy and (b) keep the visual context manageable.

The paper's bet: an **agentic** decomposition — anatomy-specific tools dispatched by an LLM planner with retrieval-augmented exemplars — is the right way to make 3D CT QA both tractable in tokens and clinically faithful.

## Core Innovation

Three pieces, each addressing a separate bottleneck:

1. **Anatomy-Aware LoRA Ensemble.** Ten LoRA adapters (rank 16, α 16, dropout 0.05) sit on a shared, frozen LLaVA-Med-v1.5-mistral-7B backbone — one per anatomical region (lung, trachea & bronchi, mediastinum, heart, esophagus, pleura, bone, thyroid, breast, abdomen). A Deepseek-v3 planner routes the query to the right tool(s).
2. **Hierarchical Token Compression.** Per-slice tokens are compressed via a **Global Token Aggregation (GTA)** branch (token-wise MoE gate + slice-wise mean over all 240 slices) and a **Local Token Selection (LTS)** branch (CLS-attention top-K dominant tokens + key-similarity-merged contextual tokens). Final budget: K = 54 + M = 10 = **64 tokens per slice**.
3. **Prediction-Guided Few-Shot Retrieval.** The 10 region-tool outputs are concatenated, embedded with OpenAI `text-embedding-3-small`, and used to retrieve the top-3 most similar historical reports as in-context exemplars for the final report.

## Claims & Evidence Analysis

| Claim | Evidence | Strength |
|-------|----------|----------|
| **C1.** Hierarchical token compression cuts tokens ~75% while preserving semantics. | 256 → 64 tokens confirmed; Table 3 shows truncation at the same budget collapses lung-presence F1 to 0.465 vs CT-Agent's 0.952. **Audited only on Lung & Heart** — the easiest anatomies. | ⭐⭐ |
| **C2.** GTA materially improves cross-slice reasoning. | +0.051 F1 on lung-presence; only +0.007 / +0.005 on lung-abnormality / heart-abnormality — within likely run-to-run noise, no seeds reported. | ⭐ |
| **C3.** Anatomy-specific LoRA ensemble beats end-to-end fine-tune. | CE-F1 0.420 vs LLaVA-CT 0.221; +0.057 / +0.139 average F1 on presence / abnormality QA. | ⭐⭐ |
| **C4.** Prediction-guided retrieval beats static few-shot and zero-shot. | BLEU-4 0.136 → 0.210 → 0.231; METEOR 0.335 → 0.399 → 0.425. | ⭐⭐ |
| **C5.** State-of-the-art on 3D CT report generation. | Wins 7/9 metrics, but **loses BLEU-4** to LLaVA-CT (0.252) and M3D (0.245). | ⭐⭐ |
| **C6.** "Consistently outperforms" on QA. | Contradicted by Table 2: regressions on Mediastinum-Presence (-0.113), Esophagus-Presence (-0.027), Pleura-Abnormality (-0.019), Thyroid-Abnormality (-0.087). | ⭐ |
| **C7.** Trachea & Bronchi abnormality F1 = 0.790 vs baseline 0.018. | A 43× single-region jump with no error analysis or per-class confidence intervals. | ⭐ (suspect) |
| **C8.** Generalizable beyond chest CT. | No experiments outside chest; the 240-slice resampling, 10-region taxonomy, and predefined query templates are all chest-specific. | not supported |

**Honest read.** C1 and C3 are the strongest claims, but the token-compression validity audit is incomplete: Table 3 only ablates Lung and Heart and never tests compression on the regions where the method claims its biggest QA wins (Trachea & Bronchi, Mediastinum, Bone). The **ensemble overhead claim is essentially un-audited**: report generation requires 10 LoRA-tool forward passes of a 7B MLLM **plus** a Deepseek-v3 planner call **plus** an OpenAI embedding call **plus** vector retrieval per report — yet the paper never publishes wall-clock, GPU-second, or token-throughput numbers, despite "efficiency" being a banner claim. The CE metric is improved-Exact-Match against **GPT-generated** RadGenome labels, so a high CE may reflect stylistic alignment with the labeling distribution rather than clinical truth.

## Method & Architecture

![CT-Agent architecture overview](/assets/images/paper/ct-agent/fig_p005_01.png)
*Figure 1: CT-Agent's three-module architecture — the LLM planner routes a CT volume + query into either parallel all-region report generation or single-region QA, drawing from an Action Space (anatomy LoRAs, few-shot retrieval, query normalization) and a Memory module (query hub, embedding store, history log).*

**Pipeline (step by step):**

1. **Preprocessing (MONAI):** resample to (1.5, 1.0, 1.0) mm, crop foreground, convert to Hounsfield Units, decompose each volume into **T = 240 axial slices**. Each slice → frozen CLIP ViT-B/16 → 256 patch tokens × 1024 dim.
2. **Planning (Deepseek-v3):** classifies the query as Report Generation vs Region-Guided QA and identifies the involved anatomical regions. State transition $S_{t+1}=f(S_t, A_t, E_t)$.
3. **Anatomy-aware reasoning:** one LoRA per region on a shared LLaVA-Med-v1.5-mistral-7B backbone.
4. **Token compression** (see Figure 2 below).
5. **Few-shot retrieval** over a 20k-report corpus (see Figure 3 below).
6. **Report assembly** via the LLM with retrieved exemplars as in-context examples.

![Anatomy-aware reasoning tool with hierarchical token compression](/assets/images/paper/ct-agent/fig_p007_01.png)
*Figure 2: A frozen CLIP ViT-B/16 emits 256 tokens per slice; tokens flow through GTA (token-wise MoE + slice-averaging) and LTS (CLS-attention dominant selection + key-similarity contextual merging); the fused tokens are projected to 4096-dim and combined with the query before a LoRA-augmented MLLM produces the answer.*

**GTA (Global Token Aggregation).** For each slice $t$, every token $z_{t,i}$ is routed by a learned gate $\alpha_{t,i}=\mathrm{Softmax}(W_g z_{t,i}+b_g)\in\mathbb{R}^E$ to top-k of $E$ expert MLPs. Then slice-wise mean over all 240 slices: $Z_f=\frac{1}{T}\sum_t Z'_t \in \mathbb{R}^{256\times 1024}$ — i.e. 256 "global" tokens summarizing the entire volume.

**LTS (Local Token Selection).** Per slice, attention scores from the CLS token select top-K dominant tokens; the remaining tokens are merged into M contextual tokens by key-similarity argmax + average pooling. Final per-slice local matrix $Z_{\text{local}}\in\mathbb{R}^{(K+M)\times d}$ with K = 54, M = 10.

The two streams are concatenated, $Z_{\text{vision}}=[Z_{\text{global}}; Z_{\text{local}}]$, and projected by a learned linear $1024 \to 4096$ to enter the LLM embedding space.

![Prediction-guided exemplar retrieval](/assets/images/paper/ct-agent/fig_p010_01.png)
*Figure 3: The 10 anatomy-level outputs of the current case are concatenated and encoded with OpenAI text-embedding-3-small, matched against a vector index of historical reports, and the top-k cases are prepended as few-shot context for final report generation.*

**Training.** AdamW, lr = 2×10⁻⁴, batch 8, 2 epochs, 500-step warm-up, 8×A40 GPUs, FSDP + FP16 + grad accumulation. Loss is standard LM loss $\mathcal{L}=-\log P(Y\mid X_{\text{input}};\theta,\phi_{r_i})$, activating only the region's LoRA per sample.

## Experimental Results

### Datasets

- **CT-RATE** (Hamamci et al., 2024): 50,188 raw chest CT volumes (47,149 / 3,039) from 21,304 patients with paired radiology reports.
- **RadGenome-ChestCT** (Zhang et al., 2024): a CT-RATE subset with region-wise segmentation masks for 10 anatomical regions and **GPT-generated** pathology labels — 25,692 non-contrast scans, 665k grounded reports, 1.3M grounded VQA pairs.
- **Constructed QA dataset:** 2,033,648 QA pairs (1,914,448 / 119,200), templated from RadGenome's GPT annotations.

### Report Generation (Table 1)

| Method      | BL-1  | BL-2  | BL-3  | BL-4  | RL    | M     | CE-P  | CE-R  | CE-F1 |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 3D-CT-GPT   | -     | -     | -     | 0.133 | 0.145 | 0.140 | -     | -     | -     |
| CT2Rep      | 0.442 | 0.344 | 0.279 | 0.235 | 0.401 | 0.309 | 0.355 | 0.132 | 0.175 |
| MS-VLM      | -     | -     | -     | 0.232 | 0.438 | 0.396 | 0.222 | 0.329 | 0.261 |
| M3D         | 0.435 | 0.345 | 0.286 | 0.245 | 0.400 | 0.326 | 0.407 | 0.009 | 0.148 |
| LLaVA-CT    | 0.369 | 0.309 | 0.275 | **0.252** | 0.468 | 0.421 | 0.323 | 0.182 | 0.221 |
| **CT-Agent**| **0.502** | **0.374** | **0.290** | 0.231 | **0.490** | **0.425** | **0.423** | **0.477** | **0.420** |

CT-Agent dominates 7/9 columns; the **CE-Recall jump from 0.182 → 0.477** is the single largest delta and is the main driver of the F1 gap. **BLEU-4 regresses** versus LLaVA-CT and M3D — a fact the abstract elides.

### Region-Guided QA (Table 2)

| Region | Presence F1 (LLaVA-CT → CT-Agent) | Abnormality F1 (LLaVA-CT → CT-Agent) |
|--------|------------------------------------|----------------------------------------|
| Lung               | 0.937 → 0.952 | 0.521 → 0.566 |
| Trachea & Bronchi  | 0.070 → 0.205 | 0.018 → **0.790** |
| Mediastinum        | 0.690 → **0.577** | 0.523 → 0.758 |
| Heart              | 0.853 → 0.880 | 0.777 → 0.781 |
| Esophagus          | 0.259 → **0.232** | 0.207 → 0.230 |
| Pleura             | 0.345 → 0.366 | 0.279 → **0.260** |
| Bone               | 0.393 → 0.743 | 0.239 → 0.377 |
| Thyroid            | 0.899 → 0.948 | 0.540 → **0.453** |
| Breast             | 0.888 → 0.880 | 0.478 → 0.640 |
| Abdomen            | 0.556 → 0.679 | 0.345 → 0.462 |
| **Overall**        | **0.589 → 0.646** | **0.393 → 0.532** |

Wins are not uniform: Mediastinum, Esophagus, Pleura, and Thyroid each regress on at least one column; the **Trachea & Bronchi abnormality jump from 0.018 to 0.790** is a 43× single-region gain reported with no error analysis or per-class confidence intervals. The paper's "consistently outperforms" wording in Sec 5.2 is overstated.

### Ablations

- **Token compression (Table 3, Lung & Heart only):** truncation 0.465; random sampling 0.505; fixed-interval 0.523; CT-Agent w/o global 0.901; full CT-Agent **0.952** (lung-presence). Adding global tokens is +0.051 on lung-presence but only +0.005–0.007 on lung-abnormality and heart-abnormality.
- **Few-shot retrieval (Table 4):** zero-shot BLEU-4 / METEOR 0.136 / 0.335; static few-shot 0.210 / 0.399; prediction-guided 0.231 / 0.425.
- **Planning module:** removing region routing (i.e. plain LLaVA-CT) costs 0.199 CE-F1 on report gen and 0.057 / 0.139 average F1 on presence / abnormality QA.

![Qualitative comparison](/assets/images/paper/ct-agent/fig_p014_01.png)
*Figure 4: Sentence-level comparison of GT, LLaVA-CT, and CT-Agent reports — green = matching findings, blue = missed/extra, red = factual contradictions, yellow = internal contradictions; CT-Agent has fewer red/yellow spans and fewer hallucinated extras than LLaVA-CT.*

## Limitations

**Authors acknowledge** (Sec 6): no integration of multimodal clinical evidence (e.g. prior reports), no longitudinal scan analysis, no real-time physician feedback / interactive workflow.

**Not addressed by the authors:**

- **Inference cost.** No wall-clock, GPU-second, or token-throughput numbers despite "efficiency" being a banner claim. Report generation requires **10 LoRA-tool forward passes** of a 7B MLLM **plus** a Deepseek-v3 planner call **plus** an OpenAI embedding call **plus** retrieval per case — the net compute likely exceeds a single LLaVA-CT pass and directly contradicts the efficiency framing.
- **No variance / no statistical tests.** All results are single-run point estimates; no seeds, no confidence intervals.
- **GPT-labeled ground truth.** RadGenome's labels are GPT-generated, not radiologist-verified, so high CE-F1 may reflect **stylistic alignment with the labeling LLM** rather than clinical truth.
- **Token-compression ablation is only on Lung+Heart** — exactly the regions where the method is least challenged. Validity is not shown for Trachea & Bronchi, Mediastinum, Bone, etc., where the largest claimed gains occur.
- **No external validation.** Both training and test are CT-RATE / RadGenome (RadGenome is itself a CT-RATE derivative). No held-out hospital, no out-of-distribution scanner / protocol.
- **Hard-coded 240 slices and 10-region taxonomy** make the framework chest-specific despite generic claims in the title.
- **Planner failure modes unanalyzed.** The Deepseek-v3 planner can mis-route to the wrong anatomy; no recovery, fallback, or routing-error rate is reported.
- **Trachea & Bronchi 0.018 → 0.790 jump** gets a one-sentence mention with no qualitative analysis. The baseline F1 of 0.018 suggests the prior model essentially never predicted abnormality on this region — the gain may be from priors in the QA template rather than visual reasoning.
- **No comparison vs. a single-LoRA, full-context baseline.** The ablations isolate planner vs. compression vs. retrieval but never the **ensemble vs. monolithic-tool** question.

## Why It Matters for Medical AI

3D CT report generation is one of the harder medical multimodal tasks: real volumes are 200+ slices, real reports cover multiple organ systems in a single document, and clinical readers care about **completeness and faithfulness**, not BLEU. CT-Agent's CE-Recall jump (0.182 → 0.477) is the most clinically meaningful number in the paper — completeness has been the long-standing weak spot of 3D CT report generators. The agentic decomposition pattern (one tool per anatomy, planner on top, retrieval for grounding) is also a sensible template for other multi-organ imaging tasks (whole-body PET/CT, abdominal MRI).

That said, before deployment a buyer should demand: (a) latency / cost numbers for the 10-tool ensemble, (b) external-hospital validation outside CT-RATE, (c) a radiologist-rated subset to confirm CE gains are not just stylistic alignment with GPT-labeled ground truth, and (d) ablations on the harder regions (Trachea, Mediastinum) to verify the token compression actually preserves fine-grained findings.

## References

- Paper: Mao, Xu, Qin, Gao. *CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering.* arXiv:2505.16229v1 [cs.CV], 22 May 2025.
- Datasets:
  - Hamamci et al. *CT-RATE.* 2024. (HuggingFace, research-only)
  - Zhang et al. *RadGenome-ChestCT.* 2024.
- Backbones / tools: LLaVA-Med-v1.5 (Mistral-7B), CLIP ViT-B/16, Deepseek-v3, OpenAI `text-embedding-3-small`, MONAI, PEFT (LoRA).
- Related: CT2Rep, 3D-CT-GPT, MS-VLM, M3D, LLaVA-CT, R2Gen, MAIRA, XrayGPT, PMC-VQA, VisionZip.
