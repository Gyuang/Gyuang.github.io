---
title: "Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis"
excerpt: "Training-free LLM agents orchestrate off-the-shelf 3D brain MRI tools - GPT-5.1 single-agent hits inclusion 1.0 / accuracy 0.944 on longitudinal response."
categories:
  - Paper
tags:
  - Agentic-LLM
  - Neuroimaging
  - Medical-Image-Analysis
  - LLM-Agents
  - Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR
- A **training-free agentic pipeline** lets commercial LLMs (GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.5) drive off-the-shelf neuroimaging tools - SynthStrip, ANTsPy, BraTS-winner segmenters, SynthSeg, Panoptica, PyRadiomics - through end-to-end 3D brain MRI workflows (preprocessing, segmentation, volumetry, longitudinal response).
- The LLM **never sees voxels** - it interacts only with text and opaque object pointers, so all imaging stays local (privacy-preserving by design). Tool-selection precision lands at **0.838-0.998** and recall at **0.94-1.0**; output inclusion is **>= 0.96** and accuracy **0.82-0.97** across architectures.
- The architectural verdict is **less clean than advertised**: the orchestrator design is consistently the worst (Gemini orchestrator: **2.47 errors/run, 0.835 accuracy**), while on the hardest task **single-agent GPT-5.1 wins outright (1.0 inclusion / 0.944 accuracy)** - the "multi-agents beat monolith" claim only survives when averaging across LLMs.

## Motivation
Quantitative neuro-radiology is a chain of 3D operations - skull-strip, register, segment, measure, compare timepoints - and current LLMs have no native 3D spatial reasoning. Medical VLMs largely stop at 2D VQA, and prior agentic medical-imaging work (MedRAX on 2D chest X-ray, TissueLab on pathology, MedAgentSim on diagnostic dialogue) does not cover training-free orchestration of 3D neuroimaging pipelines. The question this paper asks is whether commercial LLMs, given only text/pointer interfaces to specialist tools, can autonomously assemble a correct clinical workflow - and which agent architecture is most reliable and cost-efficient.

## Core Innovation
Two pieces hold the contribution together:

1. **Privacy-first text/pointer interface.** Atomic tools (`load-image`, `register`, `segment`, ...) are exposed to the LLM as text-only handles. Voxel data never enters the prompt, no PHI leaves the host, and the atomic granularity stresses the agent's planning rather than hiding it inside bundled super-tools.
2. **Systematic comparison of four agent architectures** under a single multidimensional protocol - tool-call fidelity (order-invariant precision/recall + error counts), cost (tokens + cents), and output quality (inclusion + accuracy). The four designs are single-agent, agents-as-tools, handoffs, and orchestrator.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | A training-free agentic pipeline can execute end-to-end 3D brain MRI workflows. | Non-zero usable accuracy across all tasks/architectures/LLMs (Task 3 best 0.944, Task 1 perfect 1.0). | **3 stars** |
| C2 | Domain-specialized multi-agents are more robust and efficient than a monolithic agent. | Cost favours handoffs over single-agent for Claude (15.2 c vs 17.7 c) on Task 3 - but **single-agent GPT-5.1 still wins Task 3 outright (1.0 / 0.944)**. The claim holds only when averaging across LLMs. | **2 stars** (mixed) |
| C3 | Handoff-based shared context beats explicit agent-to-agent messaging and removes the need for an orchestrator. | Handoffs match or beat as-tools on cost (GPT 4.3 c vs 4.9 c); orchestrator is worst across all three LLMs (Gemini: 0.835 acc, 2.47 errors). | **2 stars** |
| C4 | Approach generalizes across LLMs (GPT-5.1, Gemini 3 Pro, Claude 4.5 Sonnet). | Architectural ordering is consistent on Task 3, but per-LLM winners flip between single-agent and handoffs. | **2 stars** |
| C5 | The released VQA benchmark enables rigorous evaluation of agentic neuro-radiology systems. | Dataset with per-architecture ground-truth tool sequences; URL released. | **2 stars** (small scale, BraTS-centric) |
| C6 | A near-perfect pipeline can still produce faulty results from a single wrong tool call. | Qualitative failure cases: Panoptica "unmatched" lesions mistaken for new lesions; missing localization with centroid-only output. | **2 stars** |

## Method & Architecture

![Agentic brain MRI pipeline overview](/assets/images/paper/agentic-neuroimaging/fig_p004_01.png)
*Figure 1: End-to-end agentic brain MRI pipeline - user query plus MRI volumes feed the agent, which plans tool calls and produces a final response audited by a judge LLM.*

**Tool palette.**
- *Preprocessing:* SynthStrip skull stripping; ANTsPy T1 co-registration and atlas registration (SRI24 / MNI152); resampling.
- *Segmentation:* BraTS challenge-winner models (glioma, metastases, meningioma, pediatric) via BraTS Orchestrator; SynthSeg for 32 anatomical regions.
- *Analysis:* connected-component lesion enumeration; Panoptica IoU-based longitudinal lesion matching; per-lesion sub-volume, bounding box, centroid; PyRadiomics morphological/textural features; USCLobes atlas for lesion localization.
- *Utilities:* registration verification, image loading, on-demand brain-region vocabulary retrieval (to cut prompt tokens), visualization.

**Four agent architectures.**

![Four agent architectures compared](/assets/images/paper/agentic-neuroimaging/fig_p005_01.png)
*Figure 2a: Single-agent design - one LLM holds the entire tool catalog and plans/executes alone.*

![Agents-as-tools design](/assets/images/paper/agentic-neuroimaging/fig_p005_02.png)
*Figure 2b: Agents-as-tools - preprocessing / segmentation / analysis experts expose themselves as callable tools and exchange structured `Request`/`Response` schemas (no free text).*

![Handoff design](/assets/images/paper/agentic-neuroimaging/fig_p005_03.png)
*Figure 2c: Handoffs - control plus full execution context/history is transferred between experts, preserving shared state across the workflow.*

![Orchestrator design](/assets/images/paper/agentic-neuroimaging/fig_p005_04.png)
*Figure 2d: Orchestrator - an additional central planner owns no tools and only assigns tasks to domain experts via structured schemas.*

Tool allocation: `registration-verification` is bound to the segmentation agent (colocate network-specific prerequisites); remaining utility tools live in the analysis agent; the orchestrator gets no tools.

**Evaluation harness.** LLM-as-a-judge with OpenAI o3. Tool-call fidelity is scored with order-invariant precision/recall plus error counts; cost in input/output tokens and cents/run; output quality as inclusion rate (fraction of queried fields present) and accuracy (correctness vs. ground truth). Temperature 0 throughout, OpenAI Agents SDK (Chat Completions API), batched via WandB Weave.

**Dataset.** Primarily public BraTS (12 pre-op glioma, 10 post-op glioma, 10 metastasis, 2 meningioma; T1/T2/T1ce/FLAIR) plus an in-house unprocessed subset (4 pre-op glioma + 3 metastasis) to exercise preprocessing. Longitudinal cohort: 8 / 4 pre-op glioma, 7 / 3 post-op glioma, 6 / 3 metastasis. VQA entries are free-text question + expected tool-call plan + expected keyword-value answer.

**Task split (increasing complexity).**
- **Task 1 - Segmentation:** 29 cases / 43 queries. Avg actions: single 2; as-tools / handoffs 3.45; orchestrator 4.9.
- **Task 2 - Single-timepoint assessment** (volumes, locations, up to 30 fields): 150 cases / 565 queries. Avg actions: 5.28 / 7.79 / 10.3.
- **Task 3 - Longitudinal response** (multi-timepoint volume change, new lesions): **36 cases** / 267 queries. Avg actions: 11.85 / 13.94 / 16.03.

## Experimental Results

### Tasks 1-2 with GPT-5.1 (Table 1)

| Task | Agent | #Err | Prec. | Rec. | #Act | In tok | Out tok | Cost (c) | Incl. | Acc. |
|------|-------|------|-------|------|------|--------|---------|----------|-------|------|
| T1 | as-tools | 0 | 0.800 | 0.982 | 4.09 | 11.0K | 517 | 1.9 | 1.0 | 1.0 |
| T1 | handoffs | 0 | 0.673 | 1.0 | 5.00 | 9.4K | 372 | 1.6 | 1.0 | 1.0 |
| T1 | orchestrator | 0 | 0.936 | 0.954 | 4.83 | 7.8K | 486 | 1.5 | 1.0 | 1.0 |
| **T1** | **single** | **0** | **0.941** | **1.0** | **2.16** | **9.4K** | **310** | **1.6** | **1.0** | **1.0** |
| T2 | as-tools | 0.01 | 0.990 | 0.978 | 7.71 | 16.7K | 694 | 2.9 | 0.995 | 0.929 |
| **T2** | **handoffs** | **0.06** | **0.991** | **0.998** | **7.95** | **15.5K** | **530** | **2.6** | **0.989** | **0.966** |
| T2 | orchestrator | 0.20 | 0.97 | 0.940 | 10.31 | 16.0K | 995 | 3.1 | 0.977 | 0.816 |
| T2 | single | 0.05 | 0.953 | 0.996 | 5.68 | 19.1K | 471 | 3.1 | 0.986 | 0.942 |

### Task 3 - Longitudinal response across LLMs (Table 2)

| LLM | Agent | #Err | Prec. | Rec. | #Act | In tok | Out tok | Cost (c) | Incl. | Acc. |
|-----|-------|------|-------|------|------|--------|---------|----------|-------|------|
| Claude 4.5 | as-tools | 0 | 0.996 | 0.98 | 13.71 | 42.2K | 2.2K | 15.9 | 0.985 | 0.925 |
| **Claude 4.5** | **handoffs** | **0** | **0.984** | **0.94** | **13.27** | **40.4K** | **2.1K** | **15.2** | **0.993** | **0.933** |
| Claude 4.5 | orchestrator | 0.06 | 0.991 | 0.96 | 15.68 | 42.5K | 2.7K | 16.9 | 0.899 | 0.820 |
| Claude 4.5 | single | 0 | 0.993 | 0.98 | 11.68 | 49.4K | 1.9K | 17.7 | 0.989 | 0.921 |
| **Gemini 3 Pro** | **as-tools** | **0.06** | **0.996** | **0.98** | **13.71** | **31.2K** | **1.4K** | **7.9** | **0.993** | **0.936** |
| Gemini 3 Pro | handoffs | 0 | 0.982 | 0.94 | 13.09 | 36.5K | 1.0K | 8.5 | 0.963 | 0.914 |
| Gemini 3 Pro | orchestrator | 2.47 | 0.838 | 0.98 | 22.27 | 51.8K | 2.1K | 13.0 | 0.899 | 0.835 |
| Gemini 3 Pro | single | 0 | 0.998 | 0.96 | 11.38 | 41.3K | 1.0K | 9.5 | 0.981 | 0.903 |
| GPT-5.1 | as-tools | 0.15 | 0.939 | 0.98 | 13.62 | 24.8K | 1.5K | 4.9 | 0.996 | 0.933 |
| GPT-5.1 | handoffs | 0.38 | 0.982 | 0.97 | 13.85 | 22.9K | 1.3K | 4.3 | 0.993 | 0.921 |
| GPT-5.1 | orchestrator | 0.35 | 0.952 | 0.96 | 15.94 | 22.1K | 2.2K | 5.1 | 0.959 | 0.918 |
| **GPT-5.1** | **single** | **0** | **0.983** | **0.98** | **11.79** | **28.3K** | **1.2K** | **5.0** | **1.0** | **0.944** |

**Qualitative findings.**
- Inclusion is uniformly high; failures are dominated by **abstentions** ("cannot find") rather than wrong values, traceable to a missed or mis-selected tool.
- Task-1 multi-agent precision dips are **benign** (extra handoffs or image loads, never extra analysis calls); precision normalizes in Tasks 2-3.
- Handoffs beat agents-as-tools because shared context preserves full state instead of being squeezed into structured `Request`/`Response` messages.
- The **orchestrator is worst on every LLM**; Gemini in particular accumulates 2.47 errors/run and 22.27 actions because the entry agent is no longer the analysis agent and intent gets garbled.
- Claude is most expensive per call (chatty outputs, different tokenizer, higher API pricing) but most robust to erroneous tool calls.
- Recurring failure modes: treating Panoptica "unmatched" lesions as truly new (ignoring nearby centroid evidence) and skipping lesion localization while still reporting centroids.

## Limitations
- **No variance, no significance tests.** Single-run, temperature-0 results hide API-side non-determinism, and 1-3 point accuracy gaps on 36 cases are easily inside the noise floor.
- **Evaluator conflict.** The LLM-as-judge is OpenAI o3, judging OpenAI GPT-5.1 outputs - a direct vendor conflict with no human spot-check or second judge.
- **Small samples.** Task 3 has only **36 cases**, meningioma is n=2, the in-house unprocessed cohort is n=7 - per-pathology generalization and the preprocessing branch are both thinly supported.
- **Architectural conclusion is overstated.** "Multi-agents beat single-agent" only holds when averaging across LLMs. On Task 3, **single-agent GPT-5.1 is the global best (1.0 / 0.944)** at essentially the same cost as handoffs (5.0 c vs 4.3 c) - the architecture is not the dominant factor.
- **No external baseline.** No comparison to MedRAX, TissueLab, M3D, E3D-GPT, or a deterministic "call every tool" pipeline; the marginal value of agentic reasoning over a scripted pipeline is therefore unknown.
- **No latency or per-tool failure-rate reporting** - both first-order concerns for clinical deployment.
- **Tool ceiling.** Overall accuracy is bounded by the underlying tools; cascaded decision systems (classify pathology then pick segmenter) are not implemented; code release is gated on acceptance.

## Why It Matters for Medical AI
The interesting bet here is not the LLM rankings - it is the **interface design**. By forcing the LLM to plan over opaque pointers, all imaging stays on the hospital host, sidestepping the PHI-leakage problem that has stalled clinical LLM adoption. If the architectural verdict turns out to be a wash (and the per-LLM numbers suggest it does), the practical takeaway is that a **well-engineered single-agent over a curated tool palette is good enough** for 3D neuro-radiology workflows today, and the engineering effort should go into tool quality and verification (Panoptica "new lesion" reasoning, localization correctness) rather than agent topologies. The benchmark - free-text questions paired with ground-truth tool sequences - is the more durable contribution; it gives the field a way to score future systems on what they actually do, not just what they output.

## References
- Paper: [arXiv:2604.16729](https://arxiv.org/abs/2604.16729)
- Dataset: [brain-mri-agents-dataset](https://anonymous.4open.science/r/brain-mri-agents-dataset-D165)
- Related: MedRAX (2D chest X-ray agents), TissueLab (pathology / radiology quantification), MedAgentSim (diagnostic dialogue), BraTS Orchestrator, SynthStrip, SynthSeg, Panoptica, PyRadiomics, ANTsPy.
