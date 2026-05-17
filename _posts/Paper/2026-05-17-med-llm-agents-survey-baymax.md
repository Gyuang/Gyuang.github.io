---
title: "A Survey of LLM-based Agents in Medicine: How far are we from Baymax?"
excerpt: "Taxonomy survey of 60 medical LLM-agent papers (2022–early 2024) organized along four architectural modules and four agent paradigms."
categories:
  - Paper
  - LLM-Agents
tags:
  - LLM-Agents
  - Survey
  - Taxonomy
  - Clinical-Decision-Support
  - Multi-Agent
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/med-llm-agents-survey-baymax/
---

## TL;DR
- The paper organizes the still-young field of LLM-based medical agents along **four architectural modules** (Profile, Clinical Planning, Medical Reasoning, External Capacity Enhancement) and **four canonical paradigms** (Single Agent, Sequential Task Chain, Collaborative Experts, Iterative Evolution).
- Its main contribution is a *taxonomy*, not an empirical study: 300 candidate papers were screened down to **60 included studies (2022–early 2024)** across decision support, documentation, training simulation, and healthcare service optimization.
- The "Baymax" framing in the title is rhetorical — the survey never operationalizes how far the field actually is from a Baymax-like agent, and its corpus screening falls short of PRISMA standards despite the v2 (May 2025) timestamp.

## Motivation
LLM-based agents — LLMs extended with tool use, memory, planning, and multi-agent coordination — have moved fast in medicine, but the literature is fragmented across CS venues, clinical journals, and preprint servers. General-purpose agent surveys (Xi et al. 2023; Wang et al. 2024a) do not cover the domain-specific requirements that medical deployment imposes: multimodal integration (text + imaging + labs), clinical collaboration with physician oversight, accuracy/reliability tied to patient outcomes, and transparency/traceability for regulation.

The authors argue a structured review is needed to (i) align disparate work under a common architectural vocabulary, (ii) catalogue what evaluation actually exists today, and (iii) surface the open challenges (hallucination, multimodal integration, ethics) that block clinical adoption — framed against the aspirational endpoint of "Baymax," a fully autonomous, safe, patient-centred medical agent.

## Core Innovation
Because this is a survey, the contribution is conceptual organization rather than a system. Three things are new:

1. **A four-module architectural taxonomy** (Profile / Clinical Planning / Medical Reasoning / External Capacity Enhancement) that can be applied uniformly to every cited medical agent.
2. **A four-paradigm typology** (Single Agent / Sequential Task Chain / Collaborative Experts / Iterative Evolution) that classifies how agents are wired together for clinical workflows.
3. **An evaluation typology** that splits benchmarks into Static Q&A, Workflow-based Simulation, and Automated Evaluation, and metrics into Exact-Match, Semantic Similarity, and LLM-based Evaluation.

No equations, no training procedure, no code release — the value is in the vocabulary.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | LLM-based agents are improving several clinical tasks (diagnosis, patient communication, education). | Narrative citation of Kim 2024a, Mukherjee 2024, Yu 2024 — no aggregated numbers, no meta-analysis. | ⭐⭐ |
| C2 | Four canonical agent paradigms cover medical agent systems. | Figure 1 + Table 1 classify each cited system into one paradigm. Internally consistent but several systems plausibly belong to multiple categories (e.g., MDAgents is both Adaptive Planning and Collaborative Experts). | ⭐⭐ |
| C3 | Hallucinations are a significant risk in medical agents. | Cites Huang 2024b, MedHallBench, HaluEval — no numbers in-paper. The BiasMedQA "precision below 80%" figure is for *bias*, not hallucination, so evidence does not match the claim cleanly. | ⭐⭐ |
| C4 | Multi-agent collaboration improves diagnostic accuracy. | Cites MDAgents, EHRAgent, MedAgents — but no head-to-head numbers (single-agent vs. multi-agent on the same benchmark) are reported in the survey. | ⭐ |
| C5 | Static Q&A benchmarks do not capture interactive, sequential clinical decision-making. | Conceptually sound and widely accepted; cited works (MedChain, AgentClinic) support it. | ⭐⭐ |
| C6 | 60 studies (2022–2024) form the corpus. | Appendix A.1 describes screening. No PRISMA flow diagram, no inter-rater reliability, no explicit inclusion/exclusion criteria beyond keywords. | ⭐ |
| C7 | Inference-time scaling improves complex tasks like differential diagnosis and treatment planning. | Cites Huang 2025 (O1 replication, part 3) — a single preprint, not yet broadly replicated clinically. | ⭐ |
| C8 | Integration with physical systems (e.g., nursing robots) is a viable direction. | One cited paper (Zhao 2025 IEEE RA-L). Aspirational. | ⭐ |
| C9 | BiasMedQA found medical-LLM precision can fall below 80%, some as low as 50%. | Faithful citation of Schmidgall 2024a; numbers are reproduced correctly as a motivating fact. | ⭐⭐ |

The taxonomy work (C2, C5) is the strongest contribution and is well-defended by Figure 1 and Tables 1–2. Clinical-impact claims (C1, C4) are largely assertive: the paper does not aggregate or compare numbers across systems, so a reader cannot verify that "multi-agent improves over single-agent" beyond accepting cited authors' self-reports. The corpus-selection methodology (C6) is unusually thin for a 2025 survey — no PRISMA flow chart, no list of excluded papers, no rater agreement, so the literature scope is not reproducible.

## Method & Architecture

![Conceptual architecture of LLM-based medical agents and the four canonical paradigms](/assets/images/paper/med-survey-baymax/fig_p003_01.png)
*Figure 1: The four-module architecture (Profile / Clinical Planning / Medical Reasoning / External Capacity Enhancement) and the four agent paradigms — Single Agent, Sequential Task Chain, Collaborative Experts, Iterative Evolution — that organize the entire taxonomy.*

The four modules unpack as follows:

- **Profile (Section 3.1).** Three prototypes — *Functional Modularization* (e.g., MEGDA), *Role Specialization* (surgical-OR simulation, Wu et al. 2024), *Departmental Organization* (MedAgents, Tang et al. 2024).
- **Clinical Planning (3.2).** *Task Decomposition* (Single Agent vs. Sequential Task Chain), *Multi-Agent Collaboration Across Departments* (Collaborative Experts), *Adaptive Planning Architecture* (e.g., MDAgents), *Iterative Self-Evolution* (e.g., Agent Hospital).
- **Medical Reasoning (3.3).** *Multi-Step Diagnostic Reasoning* (CoT, ToT), *Reflective Decision-Making* (ReAct-style), *Collaborative Group Reasoning* (consensus across specialist agents, e.g., KG4Diagnosis), *Memory-Enhanced Reasoning*.
- **External Capacity Enhancement (3.4).** *Perception* (EHR ingestion, OCR, CLIP-style image encoders), *Knowledge Integration* (medical KGs, drug-interaction DBs, guideline repositories), *Action* (medical calculators, EHR APIs, image-analysis tools; e.g., EHRAgent, MenTI).

The literature pipeline itself: systematic search of PubMed, ACM Digital Library, arXiv, and Google Scholar with keywords {large language model, medical agent, clinical decision support, healthcare AI} over 2022–2024. ~300 candidates → 80 after title/abstract screening → **60 included** after full-text review. English-language databases only.

## Experimental Results

This is a survey, so there are no original experiments. The paper produces three organizational artefacts plus the conceptual figure.

| Artefact | Page | What it shows |
|----------|------|---------------|
| Figure 1 | p.3 | Conceptual architecture (4 modules) and 4 paradigms |
| Table 1 | p.5 | Application × Functionality × Framework Type × Tool Use — 17 cited systems across 4 application areas |
| **Table 2** | **p.7** | **Benchmarks (Static Q&A / Workflow / Automated) and metrics (Exact-Match / Semantic / LLM-based)** |
| Table 3 | p.14 (appendix) | Duplicate of Table 2 (verbatim) |

Selected quantitative facts the paper *cites* from other works (not produced by the authors):

| Source | Fact |
|--------|------|
| MedMCQA | 194,000 questions across 21 subjects, 2,400 topics |
| MedChain | 12,163 cases across 19 specialties, 7,338 medical images |
| ClinicalLab | 24 departments, 150 diseases |
| **BiasMedQA** | **Medical-LLM precision can fall below 80%, "with some models performing as poorly as 50%"** |

No ablations, no robustness study, no qualitative analysis of any individual system are produced by the authors themselves.

## Limitations

Author-acknowledged (Section "Limitations"):

- Corpus restricted to 2022–early 2024, so post-cutoff architectures are not included.
- English-language databases only.
- Field evolves faster than the review.

Not acknowledged but visible:

- **No PRISMA-style methodology** — no flow diagram, no inter-rater reliability, no list of excluded papers; the literature search is not reproducible.
- **Corpus cutoff lag.** The v2 PDF is dated May 2025 but still bounds its corpus at "early 2024," so roughly 12–15 months of fast-moving work (most of the O1/R1-era medical-agent papers) sit outside the screened sample even though some are name-checked in Section 6.
- **"Baymax" framing not operationalized.** The title promises a readiness assessment; the paper never defines milestones, scorecards, or a roadmap for what would close the gap.
- **Tables 2 and 3 are duplicates** — a minor editorial issue but suggestive of light copy-editing.
- **Multi-agent vs. single-agent superiority is asserted, not tested** — no comparative numbers on shared benchmarks.
- **Safety/ethics is enumerative.** No concrete framework or scorecard is proposed for hallucination, bias, or privacy.
- **No regulatory discussion** (FDA SaMD, EU AI Act high-risk classification) despite the transparency/traceability framing in Section 2.2.
- **No deployment cost/latency analysis** — relevant given Section 6.3.2's "Resource Allocation Dilemma" is left unquantified.

## Why It Matters for Medical AI

The vocabulary the paper introduces — four modules, four paradigms, three benchmark families — is genuinely useful. If you are writing a related-work section for a medical-agent paper in 2026, classifying your system as, say, "Collaborative Experts with Adaptive Planning, Memory-Enhanced Reasoning, and EHR-API Action" gives a reader a much more precise picture than "an LLM-agent for diagnosis."

What this survey does *not* give you is empirical guidance: it will not tell you whether to bet on Sequential Task Chain or Collaborative Experts for your use case, what the accuracy ceiling of current systems is on shared benchmarks, or how the field's safety claims stack up under audit. For those questions, the survey is a map of the territory, not a compass.

## References
- Paper (arXiv): [2502.11211](https://arxiv.org/abs/2502.11211)
- Related work cited above: MDAgents, MedAgents, EHRAgent, Agent Hospital, KG4Diagnosis, MEGDA, MenTI, MedChain, AgentClinic, ClinicalLab, AI Hospital, MedHallBench, BiasMedQA
- General LLM-agent surveys for contrast: Xi et al. 2023; Wang et al. 2024a
