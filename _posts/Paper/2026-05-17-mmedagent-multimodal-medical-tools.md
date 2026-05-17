---
title: "MMedAgent: Learning to Use Medical Tools with Multi-modal Agent"
excerpt: "A LoRA-tuned LLaVA-Med planner that routes user queries to 6 medical tools across 7 tasks and 5 modalities — overall GPT-4-judge score 8.66 (rel. 109.48% of GPT-4o) on a 70-question internal eval, with 100% tool-selection accuracy on that same disjoint set."
categories:
  - Paper
  - LLM-Agents
  - Pathology
permalink: /paper/mmedagent-multimodal-medical-tools/
tags:
  - MMedAgent
  - LLaVA-Med
  - Medical-Agent
  - Tool-Use
  - Grounding-DINO
  - MedSAM
  - ChatCAD
  - Instruction-Tuning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- MMedAgent (EMNLP 2024 Findings) takes a **LLaVA-Med 7B backbone**, LoRA-tunes it (rank 128, 15 epochs, 2x A100, ~72h) on a **48K-sample GPT-4o one-shot synthesized instruction-tuning corpus**, and turns it into an action planner + result aggregator that routes user queries to 6 specialist medical tools (BiomedCLIP, medical-fine-tuned Grounding DINO, MedSAM, Grounding-DINO -> MedSAM chain, ChatCAD, ChatCAD+ RAG) across 7 tasks and 5 modalities.
- **Headline numbers:** overall GPT-4-judge absolute **8.66** vs LLaVA-34B **6.84** and LLaVA-Med **4.80**; relative score **109.48% of GPT-4o**, beating GPT-4o on organ grounding (102.29), disease grounding (125.89), and MRG (121.49). Tool-selection accuracy reaches **100%** after 15 epochs on the same evaluation set.
- **Caveats the abstract glosses over:** the entire Table 2 comparison is **70 questions** (10-20 per task) with a GPT-4 judge and no variance reported; the 100% tool-selection figure is measured on a disjoint task-per-template eval, not an ambiguous-routing stress test; Table 3 open-ended VQA shows a **-5.75 MRI regression** the paper never discusses; BiomedCLIP classification uses a hand-curated **11-label closed set** that masks real diagnostic granularity; and the medical-fine-tuned Grounding DINO is never standalone-benchmarked against medical detection baselines.

## Motivation

Medical MLLMs split into two camps. Specialists (LLaVA-Med, RadFM, Med-PaLM-M, PathAsst) own a modality or task but cannot generalize. Generalists are weaker than the specialists they purport to replace and cannot absorb new tools without retraining. In the general domain, **LLaVA-Plus** and **MM-React** sidestep this by treating the LLM as a planner that dispatches expert tools — but no equivalent existed for medicine spanning multiple modalities. PathAsst is the closest prior work, but is pathology-only. MMedAgent positions itself as the **first general-purpose multi-modal medical agent** covering MRI, CT, X-ray, histology, and gross pathology, across grounding, box-prompted segmentation, grounded segmentation, classification, medical report generation, RAG, and VQA.

## Core Innovation

- **A LLaVA-Med planner trained on 48K GPT-4o-synthesized tool-calling dialogues.** Each turn emits Thoughts / Actions (API Name + API Params) / Value, following the LLaVA-Plus dialogue schema; an auto-regressive loss is applied only to X_tool and X_answer.
- **Six tools, seven tasks, five modalities — including the first medical-fine-tuned Grounding DINO** (trained on WORD, FLARE2021, BRATS, MC, VinDr-CXR, Cellseg + COCO + Flickr30k; segmentation masks converted to outer-rectangle boxes) that fills the previously missing "medical detection" slot. Chaining Grounding DINO -> MedSAM yields **G-Seg**, a text-prompted medical segmentation pipeline.
- **A scalable adapter recipe.** Adding a new tool only requires a small instruction-tuning slice (LR 1e-6, batch 10, 1x A100). A "Pseudo Tool" experiment reaches 100% selection accuracy on the new tool within ~2K steps without regressing on the original tools — though only one simulated tool is tested.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|-------|----------------------|------------|----------|
| C1 | First general-purpose multi-modal medical AI agent. | Related-work survey (Section 2); contrasts with PathAsst (pathology-only) and LLM-only roleplay agents (AgentClinic, MedAgents). | Literature | ⭐⭐ — defensible "first" given the survey but quickly overtaken by MMedAgent-RL, MedRAX. |
| C2 | Outperforms open-source MLLMs and even GPT-4o on average across medical tasks. | Table 2 overall **rel 109.48%**, **abs 8.66** vs LLaVA-34B 6.84. | 70-Q internal eval, GPT-4-as-judge | ⭐ — large headline gap, but the eval set is **70 questions total**, no variance/CIs, no external benchmark replication. |
| C3 | Beats GPT-4o specifically on organ grounding, disease grounding, MRG. | Table 2: **102.29 / 125.89 / 121.49**. | Same 70-Q eval | ⭐⭐ — plausible: GPT-4o cannot draw bounding boxes natively, so any specialist detector wins. MRG margin is ChatCAD's MIMIC-CXR specialization talking. |
| C4 | Tool-selection accuracy reaches 100%. | Section 5.3, claim only. | 70-Q eval (same as Table 2) | ⭐⭐ — credible because tools and instruction templates are essentially disjoint per task; 100% on 70 disjoint tasks is the floor of difficulty, not the ceiling. No adversarial / ambiguous-instruction stress test. |
| C5 | Efficiently integrates new tools without forgetting old ones. | Fig. 4, "Pseudo Tool" reaches 100% within 2K steps. | 1 simulated tool, 30 eval Qs | ⭐ — single tool, no real distribution shift, no continual-learning stress with sequential additions. |
| C6 | Improves backbone (LLaVA-Med) on open-ended VQA. | Table 3: **51.42 vs 50.94 overall (+0.48)**. | 193-Q LLaVA-Med PMC-15M test set | ⭐ — effect size within noise; **MRI regresses by -5.75** and the paper does not address this. |
| C7 | Large gains on VQA benchmarks (RAD-VQA / SLAKE / PATH-VQA / PMC-VQA). | Table 4: ~2x absolute improvements vs LLaVA-Med. | Random 4K/300 splits per benchmark | ⭐⭐ — gains are real and large, but compared only against the **un-tuned** baseline. No comparison to LLaVA-Med fine-tuned on the same 4K samples, so the gain may largely be the fine-tuning, not the agent architecture. |
| C8 | Medical-fine-tuned Grounding DINO fills a previously missing tool. | Appendix Table 5 lists training sources. | — | ⭐ — the artifact exists, but its **standalone detection mAP is never compared** to medical detection baselines. |

**Honest summary.** The headline "beats GPT-4o" rests on a 70-question internal evaluation with a GPT-4 judge — provocative, not conclusive. The tool-selection and tool-extension experiments are framed as definitive ("100%", "within 2K steps") but use disjoint instruction templates and a single simulated new tool, which understates the real-world routing problem (ambiguous queries, overlapping tool capabilities). The genuinely strong contributions are reusable artifacts: (i) the **48K open-source instruction-tuning corpus**, (ii) the **medical-fine-tuned Grounding DINO**, (iii) the demonstration of **multi-tool chaining (G-Seg)** in medicine. Their value is independent of the contested benchmark scores.

## Method & Architecture

![MMedAgent four-step workflow](/assets/images/paper/mmedagent/fig_p003_01.png)
*Figure 1: MMedAgent's four-step workflow — the MLLM acts as both action planner (X_tool) and result aggregator (X_answer) around external medical tools. Round 1 emits Thoughts/Actions/Value to dispatch the tool; Round 2 consumes the tool output to write the user-facing answer.*

**A. Pipeline (Fig. 1).**

1. User supplies instruction $X_q$ and image $I_q$.
2. The MLLM emits a structured tool call $X_\text{tool}$ as **Thoughts / Actions {API Name, API Params} / Value**.
3. The dispatched tool runs on $I_q$, producing $X_\text{result}$ (text, bounding boxes, segmentation masks, classification labels, retrieved passages, ...).
4. The MLLM re-conditions on $(X_q, I_q, X_\text{result})$ and writes the final $X_\text{answer}$. Auto-regressive loss is applied only on $X_\text{tool}$ and $X_\text{answer}$.

**B. Instruction-tuning data (Appendix Fig. 5).**

GPT-4o with one-shot prompting synthesizes tool-calling dialogues per task. A fixed system prompt + an example conversation template are paired with each tool. Totals: **15K augmented VQA + 10K detection + 5K each for segmentation / classification / MRG / G-Seg + 3K RAG = 48K samples**.

![Instruction-tuning schema and tool-call example](/assets/images/paper/mmedagent/fig_p004_01.png)
*Figure 2: Training-data schema. Each turn emits Thoughts (need a tool?), Actions (API Name + Params, may be null), and Value (NL response). The first round dispatches the tool; the second round consumes the tool output to answer the user.*

**C. Tool roster (Table 1).**

| Task | Tool | Notes |
|------|------|-------|
| VQA | LLaVA-Med itself | No external tool — backbone answers directly. |
| Classification | **BiomedCLIP** | Cosine similarity over a closed label set L of **11 labels** covering modalities/tissues — granularity is hand-curated. |
| Grounding | **Medical Grounding DINO** | Pre-trained Grounding DINO further fine-tuned on WORD, FLARE2021, BRATS, MC, VinDr-CXR, Cellseg + COCO + Flickr30k. Boxes derived from segmentation masks (minimal outer rectangle). |
| B-Seg | **MedSAM** | Bounding-box-prompted SAM for medical images. |
| G-Seg | **Grounding DINO -> MedSAM** | Text-prompted medical segmentation by chaining. |
| MRG | **ChatCAD** | MIMIC-CXR-trained, X-ray only. |
| RAG | **ChatCAD+** | Over Merck Manual (1972 disease/procedure entries) + 1K GPT-4o-generated reports + 1K patient Qs; three modes: chest-X-ray analysis, general medical-report analysis, general medical advice. |

**D. Training.** LoRA rank 128, 15 epochs, batch 48, AdamW + cosine LR peaking at **2e-4**, on **2x A100 80GB for ~72h**. New-tool adaptation uses LR 1e-6, batch 10, 1x A100.

## Experimental Results

### Diverse tasks — Table 2 (relative score % of GPT-4o, last column absolute GPT-4 score)

| Model | Cell | Organ | Disease | Cls | MRG | RAG | Overall (rel) | Overall (abs) |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|
| Flamingo-Med | 13.11 | 15.87 | 15.33 | 23.56 | 16.59 | — | 14.68 | 1.16 |
| RadFM | — | — | — | 25.00 | 68.13 | — | 45.38 | 3.59 |
| LLaVA-Med (60K-IM) | 51.78 | 65.48 | 68.58 | 53.46 | 70.10 | 30.44 | 60.68 | 4.80 |
| Yi-VL-34B | 63.23 | 79.40 | 68.32 | 76.02 | 72.95 | 14.67 | 64.08 | 5.07 |
| LLaVA-Med (Tool in Test) | 45.32 | 52.77 | 67.91 | 57.53 | 74.34 | 67.55 | 65.31 | 5.17 |
| Qwen-VL-Chat | 61.34 | 65.90 | 62.38 | 88.40 | 73.41 | 78.80 | 76.21 | 6.03 |
| LLaVA-34B | 76.75 | 84.85 | 80.75 | 96.04 | 80.27 | 91.64 | 86.52 | 6.84 |
| **MMedAgent (ours)** | **97.50** | **102.29** | **125.89** | **81.11** | **121.49** | **85.55** | **109.48** | **8.66** |

The **+8.66 abs / 109.48% rel** headline is the single number the paper sells. It comes from **70 questions total** judged by GPT-4 with no variance reported and no clinician adjudication. The "LLaVA-Med (Tool in Test)" ablation — feeding $X_\text{result}$ to the un-tuned backbone — moves overall only from 60.68 -> 65.31, evidence that planner-level training matters far more than mere access to tool outputs.

### Open-ended dialogue — Table 3 (LLaVA-Med PMC-15M held-out test, 193 Qs)

| Model | Conv (143) | Desc (50) | X-ray | MRI | Histology | Gross | CT | Overall |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-Med | 53.30 | 38.90 | 56.58 | 40.84 | 54.71 | 48.47 | 50.68 | 50.94 |
| **MMedAgent** | **54.49** | **39.75** | **58.37** | **35.09** | **56.88** | **51.88** | **52.79** | **51.42** |

**+0.48 overall** is within noise. The interesting cell is **MRI: -5.75**, a non-trivial cross-modality regression that the paper never discusses. Possible explanations (mine, not theirs): the 48K instruction-tuning corpus is skewed toward modalities where the toolset works (X-ray for MRG, CT for grounding/segmentation), and the planner has learned to over-route MRI queries to tools that under-serve them.

### VQA benchmarks — Table 4 (after per-dataset 4K fine-tuning as new tools)

| Model | RAD Open | RAD Close | SLAKE Open | SLAKE Close | PATH Open | PATH Close | PMC Close |
|-------|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-Med | 28.23 | 61.40 | 39.17 | 52.16 | 12.30 | 54.05 | 27.48 |
| **MMedAgent** | **58.31** | **86.72** | **79.39** | **86.34** | **39.16** | **90.38** | **39.50** |

Gains are large, but compared only against the **un-tuned** baseline — no LLaVA-Med fine-tuned on the same 4K samples per benchmark is reported, so it is unclear how much of the gain is the agent architecture vs. the 4K fine-tuning.

### Qualitative & scalability

![Qualitative side-by-side](/assets/images/paper/mmedagent/fig_p008_01.png)
*Figure 3: LLaVA-Med (red) produces conversational responses without solving the request; MMedAgent (green) dispatches the correct specialist tool and visualizes the result across classification, grounding, segmentation, MRG, and RAG.*

![Scalability — Pseudo Tool](/assets/images/paper/mmedagent/fig_p009_01.png)
*Figure 4: Adding a simulated new tool: selection accuracy reaches 100% within ~2K fine-tuning steps without regressing on the original tool set. The experiment uses one simulated tool and 30 eval questions — supports the scalability claim qualitatively, not at production scale.*

## Limitations

**Author-acknowledged:** limited to 7 tasks, 5 modalities; more specialist tools needed; stronger generalist medical LLM backbones could improve performance.

**Auditor additions (not addressed in the paper):**

- The Table 2 eval is **70 questions** with a GPT-4 judge — no clinician adjudication, no inter-rater agreement, no significance testing, no variance/seeds.
- Routing is tested on disjoint task-per-template instructions. Real failure modes — **ambiguous queries** ("show me the lesion": grounding or segmentation?), **missing tools** (fallback policy unstated), **tool disagreement** (no aggregation conflict resolution shown) — are not stressed.
- **MRG bound to ChatCAD** => quality is bounded by MIMIC-CXR's US adult chest X-ray distribution; OOD CXRs (pediatric, non-US scanners) are untested.
- **BiomedCLIP classification uses an 11-label closed set** — this masks real diagnostic granularity. The agent will silently miss out-of-vocabulary findings.
- The **medical-fine-tuned Grounding DINO is never standalone-benchmarked** against medical detection baselines; only the end-to-end agent score is reported.
- The **MRI -5.75 regression in Table 3 is undiscussed**.
- The "Pseudo Tool" extension experiment uses **one simulated tool, 30 eval questions** — no real continual-learning workflow (e.g., five sequential additions across modalities) is tested.
- No latency / cost analysis: 7B model + 6 tools on one A100 is practically deployable, but throughput numbers are absent.
- The 48K instruction-tuning corpus inherits OpenAI's ToS restrictions (GPT-4o synthesized).
- No comparison with concurrent medical-agent work (MMedAgent-RL, MedRAX, AgentClinic).

## Why It Matters for Medical AI

What survives the audit are the **artifacts**, not the benchmark numbers. The **48K open-source GPT-4o-synthesized instruction-tuning corpus** is the first of its kind for medical tool-use and is directly reusable by any planner backbone. The **medical-fine-tuned Grounding DINO** plugs a hole that no public model previously filled and is itself worth re-evaluating in isolation. The **G-Seg chain (Grounding DINO -> MedSAM)** demonstrates that text-prompted medical segmentation can be assembled cheaply from existing specialist tools, without a new monolithic model. For practitioners building clinical decision-support, MMedAgent's value is as a **reference architecture and dataset** — the headline "beats GPT-4o" score should be treated as a small-sample signal, not a benchmark.

## References

- Paper: [arXiv:2407.02483](https://arxiv.org/abs/2407.02483) (v2, 5 Oct 2024) — EMNLP 2024 Findings.
- Code: <https://github.com/Wangyixinxin/MMedAgent>.
- Backbone: Li et al., *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine* (NeurIPS 2023).
- Planning template: Liu et al., *LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents* (2023).
- Tools: BiomedCLIP (Zhang et al. 2023), Grounding DINO (Liu et al. 2023), MedSAM (Ma et al. 2024), ChatCAD (Wang et al. 2023), ChatCAD+ (Zhao et al. 2023).
- Datasets: WORD, FLARE2021, BRATS, MC, VinDr-CXR, Cellseg, MIMIC-CXR, VQA-RAD, SLAKE, PATH-VQA, PMC-VQA, Merck Manual.
- Concurrent work to compare against: MMedAgent-RL, MedRAX, AgentClinic, MedAgents.
