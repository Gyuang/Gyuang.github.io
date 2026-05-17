---
title: "Empowering Multimodal LLMs with External Tools: A Comprehensive Survey"
excerpt: "A 21-page arXiv survey (An, Nie et al., Aug 2025) that maps ~360 tool-augmented MLLM works onto a 4-axis structure (Data / Tasks / Evaluation / Challenges) plus a 5-class tool taxonomy — useful as a navigation map, but with no quantitative cross-paper synthesis to back the headline claim that 'tools help'."
categories:
  - Paper
  - LLM-Agents
tags:
  - MLLM
  - Tool-Use
  - Survey
  - Taxonomy
  - MRAG
  - Hallucination
  - Multimodal-Agents
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/empowering-mllm-external-tools-survey/
---

## TL;DR
- The survey argues that **external tools** — knowledge bases, expert models, APIs, physical sensors, and program tools — systematically address the three structural weaknesses of MLLMs: scarce/low-quality multimodal data, weak performance on challenging tasks, and inadequate evaluation. It organizes roughly **360 cited works** into a **4-axis taxonomy** (Data, Tasks, Evaluation, Challenges) crossed with **6 challenging-task buckets** (MRAG, Reasoning, Hallucination, Safety, Agents, Video Perception) and a **5-class tool taxonomy**.
- The "tool" definition is deliberately broader than the LLM-era function-call view: anything external that improves an MLLM counts, which is how Stable Diffusion synthesizers, Grounding DINO detectors, GPT-4 annotators, and physical sensors all end up in the same framework.
- The single most important caveat is that there is **no quantitative cross-paper synthesis** anywhere in the paper. The only concrete metric extracted from the text is **WildVision's Spearman 0.94** correlation between GPT-4-as-judge and human preference on 500 samples — a single platform stat used to justify "MLLM-as-judge works," while §6 simultaneously flags fairness/bias concerns about that exact setup.

## Motivation
MLLMs (GPT-4V, Gemini, LLaVA, InternVL, Qwen-VL, DeepSeek-VL) inherit three structural problems from how they are built and evaluated:

1. **Data**: well-aligned multimodal corpora are expensive to curate, scrape pipelines produce noisy data, and synthetic data needs an oracle.
2. **Tasks**: knowledge-intensive VQA, hallucination, multi-step reasoning, jailbreak defense, and long-video understanding remain unsolved by scaling alone.
3. **Evaluation**: BLEU / CHAIR / ROUGE-class metrics do not capture generative quality, and human evaluation does not scale.

The authors take the "humans use tools to overcome physical limits" analogy and argue that tool augmentation is the right unifying frame for all three. They explicitly flag **healthcare** and **autonomous driving** as the high-stakes domains where reliability matters most — the medical-AI angle surfaces through MMed-RAG, MultiMedEval, HALLUGEN (radiology report grounding), and CoMT (physician-style reasoning chains), although none of these are evaluated against any clinical-validation criterion in the survey itself.

## Core Innovation
This is a taxonomy paper, not a measurement paper. Three contributions are durable:

1. **A broadened tool definition.** The LLM survey literature treats "tool" as "callable function." This paper widens it to any external means — including expert models like GroundingDINO and Stable Diffusion, knowledge bases, web APIs, physical sensors/robots, and program tools like Matplotlib and web crawlers — and partitions them into **five classes** (Knowledge Bases, Expert Models, APIs, Physical Tools, Program Tools).
2. **A 4-axis × 6-task organizing scheme.** Tools act on the Data axis (collection / synthesis / annotation / cleaning), the Tasks axis (MRAG / Reasoning / Hallucination / Safety / Agents / Video Perception), and the Evaluation axis (keyword / embedding / MLLM-as-judge / platforms), with a Challenges axis closing the loop.
3. **A 6-axis benefits frame** — Knowledge Acquisition, Expertise Enhancement, Efficiency, Adaptability, Interpretability, Robustness — used to motivate why each tool class is invoked in each sub-section.

The unified MLLM objective the authors use as the backdrop is the standard autoregressive form:

$$P(y \mid x_{mm}, x_{query}) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x_{mm}, x_{query}; \theta)$$

with the architecture decomposed into multimodal encoder (CLIP / Swin / CLAP) → projector (linear / Q-Former / Perceiver) → LLM backbone (LLaMA / Qwen / DeepSeek).

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | External tools comprehensively address the three core MLLM challenges (data, tasks, evaluation). | Conceptual mapping in §3–§5; no quantitative aggregation. | ⭐ |
| C2 | Tool integration meaningfully improves MLLM performance on challenging downstream tasks. | Cited per-paper claims; no cross-paper meta-analysis or controlled comparison. | ⭐ |
| C3 | The 5-class tool taxonomy (KB, Expert Models, APIs, Physical, Program) covers the field. | Authors' partitioning works on §4 examples, but "Expert Models" absorbs most cited tools and risks being a catch-all. | ⭐⭐ |
| C4 | Tool pipelines enable scalable high-quality multimodal data creation. | Strong corpus of examples (LAION-5B, DATACOMP, PaLI's 29B OCR pairs, MMC4 NSFW filtering @ p > 0.1, VideoHallucer CLIP threshold > 0.85). | ⭐⭐ |
| C5 | MRAG (retriever → reranker → utilization) is the dominant pattern for knowledge-grounded MLLMs. | Consistent 3-stage architecture across ≥30 cited systems; well-structured. | ⭐⭐ |
| C6 | External tools mitigate hallucination across cause-analysis, detection, and mitigation. | Largest section of the survey (§4.3); rich examples (Woodpecker, HALC, RLAIF-V, MMed-RAG) but **no quantitative comparison of mitigation magnitude**. | ⭐⭐ |
| C7 | MLLM-as-judge enables reliable evaluation; WildVision achieves 0.94 Spearman with humans. | Single benchmark statistic from [346]; no replication, no audit of judge-bias even though §6 flags it. | ⭐⭐ |
| C8 | Tool augmentation is critical for high-stakes domains (healthcare, autonomous driving). | Pointer claims via MMed-RAG, MultiMedEval, HALLUGEN, RAG-Driver, BadVLMDriver; no domain-specific outcome metrics aggregated. | ⭐ |
| C9 | Multimodal CoT + structured graphs (GoT, CCoT, CoTDet) improves reasoning benchmark performance. | Cited works' own benchmark gains; survey does not quantify. | ⭐ |
| C10 | Tool-augmented MLLMs face seven open challenges (efficiency, privacy, proactivity, trustworthiness, cost, fairness, diversity). | Authors' qualitative analysis in §6; consistent with field discourse but not empirically grounded. | ⭐⭐ |

The strong sections are **§3 (Data)** and **§4.3 (Hallucination)** — the citation density and the sub-categorization (grounding vs cognitive; detector-based vs LLM-based; input-augmented vs training-based vs calibration-based) actually buy the reader something beyond a bibliography. The weakest aspect is **C2**: the headline argument that tools help is never substantiated by a single aggregated table across cited works. The reader cannot tell whether SAM-based grounding outperforms CLIP-based decoding, or whether MMed-RAG retrieval beats vanilla MMed instruction tuning. The "Expert Models" tool class is also so broad that the taxonomy's discriminative power is weaker than it looks.

## Method & Architecture

![Survey overview wheel: six components anchored on five tool classes and six benefit axes](/assets/images/paper/empowering-mllm-tools-survey/page_002.png)
*Figure 1: The survey's identity figure — six components (Introduction, Preliminary, Data, Tasks, Evaluation, Challenges) anchored on the five tool classes (Knowledge Bases, Expert Models, APIs, Physical Tools, Program Tools) and the six benefit axes (Knowledge Acquisition, Expertise Enhancement, Efficiency, Adaptability, Interpretability, Robustness).*

The MLLM backbone the authors assume is the now-standard three-stage stack:

![Generic MLLM architecture: encoder + projector + LLM backbone](/assets/images/paper/empowering-mllm-tools-survey/page_003.png)
*Figure 2: Generic MLLM architecture — multimodal encoder (CLIP / Swin / CLAP) → projector (linear / Q-Former / Perceiver) → LLM backbone (LLaMA / Qwen / DeepSeek), trained under a unified autoregressive objective.*

The Data axis (§3) decomposes the tool-augmented data pipeline into collection, synthesis, annotation, and cleaning:

![Tool-augmented data pipeline taxonomy](/assets/images/paper/empowering-mllm-tools-survey/page_004.png)
*Figure 3: Data axis — representative methods organized by stage. Collection (Flume, cc2dataset, LAION, DATACOMP, MMC4, PaLI, Conceptual-12M); Synthesis (GPT-4 / ChatGPT / Stable Diffusion / DALLE-3 / Matplotlib); Annotation (GPT-4V direct labeling, tool chains like EasyDetect = Grounding DINO + GPT-4 + Google Search + MAERec); Cleaning (CLIP semantic filtering, CLD3, Perspective, Detoxify, MMC4's MLP NSFW filter @ p > 0.1, Spawning API).*

The Tasks axis (§4) covers the six challenging-task buckets — the section that does the most discriminative work in the paper:

![The six challenging multimodal tasks unified by this survey](/assets/images/paper/empowering-mllm-tools-survey/page_006.png)
*Figure 4: The six challenging multimodal tasks — MRAG (retriever → reranker → utilization), Reasoning (image / video / audio CoT and graph-CoT), Hallucination (cause analysis / detection / mitigation), Safety (image-based and prompt-based attacks; detection and mitigation defenses), Agents (closed-source vs open-source planners), and Video Perception (temporal grounding, dense captioning, video QA).*

![Per-task taxonomy of tool-enhanced methods](/assets/images/paper/empowering-mllm-tools-survey/page_007.png)
*Figure 5: Per-task taxonomy of representative tool-enhanced methods across the six challenging-task categories — useful as a navigation map when picking a starting paper for any specific sub-problem.*

## Experimental Results
A survey reports no original experiments. The paper also includes **no comparative result tables** across cited methods — no aggregated POPE F1, no HallusionBench accuracy roll-up, no MRAG retrieval recall comparison. The only numerical claims extractable from prose are individual cited results:

| Reported figure | Source method | Context |
|-----------------|---------------|---------|
| CLIP similarity > 0.85 | VideoHallucer [76] | Threshold for video segment retention in data cleaning |
| NSFW prob > 0.1 filtered | MMC4 [39] | MLP-based NSFW filtering threshold |
| 29B image–OCR pairs | PaLI [35] | Scale of OCR annotation tool output |
| 100M cross-modal pairs | Wukong [40] | Chinese multimodal pretraining benchmark |
| 200+ MLLMs / 80+ benchmarks | VLMEvalKit [345] | Coverage of evaluation toolkit |
| 50+ tasks / 10 models | LMMs-Eval [347] | Coverage of evaluation framework |
| 6 medical tasks / 23 datasets | MultiMedEval [348] | Medical MLLM benchmark coverage |
| **Spearman 0.94 vs human** | **WildVision [346]** | **GPT-4-as-judge over 500 samples — the only judge-vs-human correlation quoted in the entire paper** |

That is the entire quantitative content of the survey. There are no ablations, no robustness analyses, no qualitative side-by-side panels of the authors' own — all qualitative figures (Figs. 1–6) are taxonomy diagrams, not result panels.

## Limitations

The authors acknowledge seven open challenges in §6:

1. **Efficiency** — tool-call latency and deployment cost.
2. **Privacy** — API-based annotators / evaluators leak data.
3. **Proactivity** — MLLMs cannot autonomously decide when/which tools to call.
4. **Trustworthiness** — external tool failures propagate into MLLM outputs.
5. **Cost** — API and GPU spend on expert-model pipelines.
6. **Fairness / bias** — MLLM-as-judge inherits judge-model bias (directly in tension with C7).
7. **Diversity** — current tool zoo is narrow; physical/audio tools are under-developed.

What the authors **do not** address:

- **No quantitative comparison or meta-analysis** across cited methods — the reader cannot pick a winner within any of the six task buckets.
- **No reproducibility discussion** — most cited works rely on GPT-4 family snapshots; results drift with API updates.
- **No license / copyright audit** of pretraining datasets despite covering LAION-class corpora.
- **Medical / healthcare angle is never grounded in clinical-validation criteria** (FDA SaMD, prospective trials, robustness on out-of-distribution scanners) — MMed-RAG / MultiMedEval / HALLUGEN / CoMT are listed but never benchmarked against any clinical-utility standard.
- **No energy / carbon cost** discussion beyond a brief efficiency mention.
- **Taxonomy overlap** — Grounding DINO + ChatGPT pipelines appear in multiple sections without explicit cross-references.
- **No timeline analysis** — which tool patterns emerged when, which are converging, which are dying.
- **Audio tool gap** — the section is materially shorter than vision and the gap is never analyzed.

## Why It Matters for Medical AI
For a clinical MLLM team, this survey is best used as a **navigation map of capability building blocks**, not as evidence that any specific block actually works in a clinical setting.

- **Retrieval grounding (MRAG).** MMed-RAG and RULE (preference tuning with factuality risk control) are the two named medical entries; the broader MRAG section gives the design vocabulary (CLIP / BLIP / GME / Pic2word retrievers; MIS / CIDEr / LLM-as-reranker rerankers; ICL vs instruction-tuned utilization) that maps cleanly onto radiology report generation or pathology QA where retrieval over guidelines or prior cases is plausible.
- **Hallucination mitigation.** The §4.3 sub-taxonomy (grounding vs cognitive causes; detector vs LLM-based detection; input-augmented vs training-based vs calibration-based mitigation) is directly transferable. Woodpecker (ChatGPT + Grounding DINO + corrector) and HALC (Grounding DINO + OWLv2 + contrastive decoding) are concrete patterns worth replicating against MediHallDetector or HALLUGEN.
- **Annotation pipelines.** HALLUGEN's "NLP + ChatGPT on radiology reports" recipe is one of the few medical-specific tool-chain examples in the paper; for an institution with text-rich EHR or PACS reports, this is a defensible labeling pattern.
- **Evaluation.** MultiMedEval (6 medical tasks / 23 datasets) is the closest thing to a unified medical MLLM evaluation harness the survey names. The headline WildVision 0.94 Spearman result is **not** a medical claim — it is a general-domain platform stat that the survey nevertheless leans on to justify MLLM-as-judge. For medical use, the §6 fairness/bias caveat against MLLM-as-judge should dominate the §5 enthusiasm.

The harder truth: the survey lists every medical pointer needed to start, but never tells you which tool pattern survives prospective clinical validation — because none of the cited papers were filtered for that criterion.

## References

- **Paper**: An, W.*, Nie, J.*, Wu, Y., Tian, F., Lu, S., Zheng, Q. *Empowering Multimodal LLMs with External Tools: A Comprehensive Survey.* arXiv:2508.10955, August 14, 2025. [arXiv link](https://arxiv.org/abs/2508.10955).
- **Affiliations**: Xi'an Jiaotong University; Nanyang Technological University; Lenovo Research.
- **Related medical-AI works cited in the survey**:
  - MMed-RAG (medical multimodal RAG) — cited [126].
  - MultiMedEval (6 medical tasks / 23 datasets) — cited [348].
  - HALLUGEN (radiology report-grounded hallucination annotation) — cited [75].
  - CoMT (clinical chain-of-thought imitating physician reasoning) — cited [215].
  - RULE (preference tuning with factuality risk control for medical retrieval).
- **Related survey context on this blog**: *Large Multimodal Agents: A Survey* (`/paper/large-multimodal-agents-survey/`) and *LLM Agents in Medicine: Baymax Survey* (`/paper/med-llm-agents-survey-baymax/`).
