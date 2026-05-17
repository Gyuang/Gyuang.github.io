---
title: "A Survey on Evaluation of LLM-based Agents"
excerpt: "A 2026 survey organizing agent evaluation along five perspectives — capabilities, applications, generalist, benchmark dimensions, developer frameworks — whose strongest contributions are negative-existence gaps in safety, trajectory, self-reflection, and cost tooling."
categories:
  - Paper
  - LLM-Agents
tags:
  - LLM-Agents
  - Agent-Evaluation
  - Survey
  - Benchmarks
  - Tool-Use
  - Trajectory-Evaluation
  - Safety
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/llm-agents-evaluation-survey/
---

## TL;DR
- The paper provides the **first structured survey of LLM-agent evaluation**, organizing the field along five orthogonal perspectives: agent capabilities, application-specific benchmarks, generalist benchmarks, core benchmark dimensions, and developer frameworks.
- Its two consolidating artifacts are **Table 1** (eight flagship benchmarks compared on data / environment / interface / metric / safety — only **TAU-Bench** explicitly tags safety) and **Table 2** (eight evaluation frameworks across six capability axes — trajectory assessment is supported by only **3 of 8**: LangSmith, Vertex AI, AgentsEval).
- There is **no headline metric** because it is a survey; the durable contributions are *negative-existence gaps* — missing safety, missing trajectory tooling, missing self-reflection benchmarks, missing cost-efficiency schemas, and a missing LLM-vs-harness decoupling — all of which are directly visible in the paper's own tables.

## Motivation
LLMs are static text-to-text systems with fixed knowledge. LLM-based agents wrap the model in an *agent harness* (Yao et al., 2022) that adds multi-step planning, external tools, memory, and environment interaction — enabling autonomous execution rather than single-turn answering. This shift breaks legacy evaluation: agents must be measured on sequential decision-making in live environments, on tool-call schema adherence, on policy compliance, and on the *trajectory* of intermediate steps — none of which a one-shot text metric captures.

The survey is explicitly aimed at four audiences — developers, benchmark creators, practitioners, researchers — and maps both what exists and where the field is short. The medical-AI angle, however, is essentially **absent** from the paper itself: clinical agent benchmarks (AgentClinic, Agent Hospital, MedAgents, MMedAgent, MedRAX) are not in scope, despite a fast-growing literature. For a medical-AI reader, the framework transfers but the survey itself does not do the transfer.

## Core Innovation
This is a survey, so the "method" is an analytical scaffold rather than a model. Three contributions are new:

1. **A five-perspective taxonomy (Figure 1)** that separates *capability* evaluation (what an agent can do in isolation: planning, tool use, self-reflection, memory) from *application* evaluation (web, SWE, scientific, conversational) from *generalist* evaluation (multi-capability suites like GAIA, Gaia2, OSWorld, AppWorld, AgentBench, HAL) from *benchmark dimensions* (data / env / interface / metric / safety) from *developer frameworks* (LangSmith, Langfuse, Vertex AI, Arize, Galileo, Patronus, AgentsEval, Mosaic).
2. **A unified five-axis benchmark scaffold** (Table 1) that lets readers compare otherwise apples-to-oranges benchmarks on the same five qualitative columns and immediately see structural gaps (e.g. safety).
3. **A six-capability framework scaffold** (Table 2) — stepwise / monitoring / trajectory / human-in-loop / synthetic-data / A-B — that exposes industrial tooling gaps, most strikingly that trajectory evaluation (the headline ask for agents) is supported by only three of eight surveyed frameworks.

No equations or training procedure — the value is the vocabulary plus the two tables.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | "First comprehensive survey of evaluation methods for LLM-based agents." | 5 perspectives × ~25 pages × ~150 references including 2024–2026 work; no directly competing evaluation-focused survey cited. | ⭐⭐⭐ |
| C2 | The field is shifting toward more realistic, continuously updated benchmarks. | Concrete progression chains: WebShop → Mind2Web → WebArena → Online-Mind2Web (Web); HumanEval → SWE-bench → SWE-bench Verified → SWE-bench Pro → SWE-Lancer (SWE); BFCL v1 → v2 → v3 (tool use). | ⭐⭐⭐ |
| C3 | SOTA models still struggle with long-horizon planning. | Cites PlanBench, Natural Plan, FlowBench; no primary numbers re-tabulated here. | ⭐⭐ |
| C4 | SWE-bench Pro: SOTA models score "below 25% Pass@1." | Single secondary citation (Deng et al., 2025); no model/cost breakdown reproduced. | ⭐⭐ |
| C5 | SWE-Lancer: 1,400 freelance tasks with "over $1M in total payouts." | Single secondary citation (Miserendino et al., 2025). | ⭐⭐ |
| C6 | WebVoyager shows "over-optimistic performance estimates"; Online-Mind2Web is the harder replacement. | Single secondary citation (Xue et al., 2025); no reproduction. | ⭐⭐ |
| C7 | No standardized benchmark or methodology exists for **self-reflection**. | Authors enumerate LLM-Evolve, LLF-Bench as ad-hoc; explicit "remains a critical gap" statement. | ⭐⭐⭐ |
| C8 | Most frameworks lack built-in **safety / policy-compliance** evaluation. | Table 1: 7 of 8 benchmarks have Safety = No (only TAU-Bench is Yes); Table 2 has no safety column at all. | ⭐⭐⭐ |
| C9 | **Trajectory** assessment is poorly supported industry-wide. | Table 2: trajectory ✓ on only LangSmith, Vertex AI, AgentsEval — 5/8 frameworks do not support it. | ⭐⭐⭐ |
| C10 | Generalist evaluation needs **decoupling of LLM and harness**. | Conceptual argument backed by Harbor / Exgentic citations; no controlled study. | ⭐⭐ |

The survey's strongest claims are the *negative-existence* ones — C7 (self-reflection), C8 (safety), C9 (trajectory) — because each is supported directly by the paper's own tables and by the breadth of the references. The weakest are the application-specific SOTA numbers — C4 (SWE-bench Pro <25%), C5 (SWE-Lancer >$1M), C6 (WebVoyager over-optimism) — which inherit single-source citations without re-checking variance, model context, or reproduction. C1 ("first") is defensible in scope (prior surveys cover agent *systems*, not agent *evaluation*) but only mildly so.

## Method & Architecture

![The survey's five-perspective taxonomy of LLM-agent evaluation, with representative benchmarks under each branch](/assets/images/paper/llm-agents-evaluation-survey/page_002.png)
*Figure 1: The survey's five-perspective taxonomy — capability evaluation (planning, tool use, self-reflection, memory), application-specific evaluation (web, SWE, scientific, conversational), generalist evaluation (Gaia, OSWorld, AppWorld, AgentBench, HAL, Harbor, Exgentic), core benchmark dimensions (data / env / interface / metric / safety), and developer frameworks (LangSmith, Langfuse, Vertex AI, Arize, Galileo, Patronus, AgentsEval, Mosaic).*

The framework is decomposed as follows.

**Capability evaluation (§2)** surveys four core abilities. *Planning* — PlanBench, FlowBench, Natural Plan — reveals long-horizon failures with verifiable constraints. *Function calling / tool use* — a clear progression from single-step rule-matched benchmarks (ToolAlpaca, ToolBench, BFCL v1) to multi-turn stateful ones (BFCL v2/v3) to dependent-call (NESTFUL), implicit-parameter (ComplexFuncBench), and real-MCP-server (MCP Atlas, ToolDecathlon) setups. *Self-reflection* — only LLM-Evolve and LLF-Bench exist, and the authors explicitly flag the lack of a standard. *Memory* — StreamBench (episodic), MemBench / MemoryAgentBench (semantic); long-range consistency remains weak.

**Application-specific evaluation (§3)** traces four domains:
- *Web*: WebShop → Mind2Web (static traces) → WebArena (dynamic sandbox) → multimodal variants (WebVoyager, VisualWebArena) → safety variants (ST-WebAgentBench) → harder replacements (Online-Mind2Web).
- *SWE*: HumanEval/MBPP → SWE-bench → SWE-bench Verified (500 human-validated, containerized) → multilingual/multimodal SWE-bench → SWE-Lancer (1,400 Upwork tasks, >$1M payouts) and SWE-bench Pro (1,865 tasks, <25% Pass@1).
- *Scientific*: knowledge recall → SciRiff → full-pipeline benchmarks for ideation (Si et al.), experiment design (AAAR-1.0), code generation (SciCode, ScienceAgentBench, CORE-Bench, PaperBench), and peer review (Chamoun et al.).
- *Conversational*: TODS roots (MultiWOZ, ABCD) → τ-Bench (retail, airline; API + policy adherence) → τ²-Bench (shared dynamic Telecom env, compositional task generator) → IntellAgent, ALMITA.

**Generalist agents (§4)** splits into two camps: inherently broad-capability benchmarks (Gaia, Gaia2 with mobile + ambiguity + multi-agent collaboration, OSWorld, AppWorld, TheAgentCompany) and unification suites (AgentBench, HAL, plus the recent Harbor and Exgentic that push for *protocol unification* so the agent harness is held constant across environments).

**Core dimensions (§5)** is where Table 1 lives: data curation (hybrid dominates; GAIA is pure-human; pure-synthetic is rare), environment (static misses cascading errors; dynamic is the right default), interface (Code, Tools, GUI, or Mix), metric (unit tests, state match, answer match, action match — all binary outcome metrics, which the authors flag as insufficient), and safety (almost universally absent).

**Frameworks (§6)** distinguishes three evaluation granularities: *final response* (LLM-as-judge against faithfulness/politeness — fast but blind to intermediate steps), *stepwise* (per-step judges on tool choice, parameter schema, output usability — Arize Phoenix has stage-specific templates, Galileo adds a goal-progress *action-advancement* metric), and *trajectory* (reference-based with exact/partial/unordered/subset/graph matching in LangSmith, Vertex AI, AgentEvals; vs reference-free LLM-judge scoring of coherence, efficiency, goal-directedness). The trade-off is reproducibility (reference-based) vs flexibility (reference-free).

## Experimental Results
No experiments — survey paper. The two consolidating artifacts are reproduced below; the **bold** row in each is the entry most often cited as the field's reference point on the relevant axis.

**Table 1 — Comparative analysis of eight representative agent benchmarks.**

| Benchmark | Data | Env. | Interface | Metric | Safety |
|---|---|---|---|---|---|
| SWE-bench Verified | Hybrid | Dynamic | Code | Unit Tests | No |
| SWE-Lancer | Hybrid | Dynamic | Code | End-to-end | No |
| Mind2Web | Hybrid | Static | GUI | Action Match | No |
| WebArena | Hybrid | Dynamic | GUI | Mix | No |
| PaperBench | Hybrid | Dynamic | Code | End-to-end | No |
| **TAU-Bench** | **Hybrid** | **Dynamic** | **Tools** | **State Match** | **Yes** |
| AppWorld | Hybrid | Dynamic | Tools | State Match | No |
| GAIA | Human | Dynamic | Mix | Answer Match | No |

![Table 1 cropped from the survey: eight flagship benchmarks compared on data / environment / interface / metric / safety](/assets/images/paper/llm-agents-evaluation-survey/page_006.png)
*Figure 2: Table 1 in situ on page 6. The single column the reader's eye should fall on is **Safety** — only τ-Bench (TAU-Bench) ships an explicit safety tag. This is the survey's most concrete negative-existence finding.*

**Table 2 — Supported capabilities of eight major agent evaluation frameworks.**

| Framework | Stepwise | Monitoring | Trajectory | Human-in-Loop | Synthetic Data | A/B |
|---|---|---|---|---|---|---|
| **LangSmith (LangChain)** | **✓** | **✓** | **✓** | **✓** | **×** | **✓** |
| Langfuse | ✓ | ✓ | × | ✓ | × | ✓ |
| Google Vertex AI eval | ✓ | ✓ | ✓ | × | × | ✓ |
| Arize AI Evaluation | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| Galileo Agentic Eval | ✓ | ✓ | × | ✓ | × | ✓ |
| Patronus AI | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| AgentsEval (LangChain) | × | × | ✓ | × | × | × |
| Mosaic AI (Databricks) | ✓ | ✓ | × | ✓ | ✓ | ✓ |

![Table 2 cropped from the survey: eight frameworks across six capability axes](/assets/images/paper/llm-agents-evaluation-survey/page_008.png)
*Figure 3: Table 2 in situ on page 8. Trajectory assessment — arguably the defining ask for agent evaluation — is supported by only **3 of 8** frameworks (LangSmith, Vertex AI, AgentsEval). AgentsEval is the inverse extreme: trajectory-only and nothing else.*

Qualitative observations highlighted in the paper: (i) SWE-bench Pro stays below 25% Pass@1 even for SOTA models; (ii) WebVoyager exhibits over-optimistic performance estimates and Online-Mind2Web is proposed as the harder replacement; (iii) of all eight frameworks, only **LangSmith** supports stepwise + trajectory + monitoring + human-in-loop + A/B simultaneously, and **AgentsEval** is the inverse extreme (trajectory only).

## Limitations
**Authors' admitted limitations (§Limitations, p. 10):**
1. The field is exceptionally dynamic; the survey is a snapshot, mitigated by a living GitHub repo.
2. Benchmark/framework selection is curated for representativeness, so niche work is omitted.
3. Per-benchmark depth is shallow because breadth was prioritized.
4. Future-direction discussion is editorial — "foresight and interpretation," not measurement.

**Additional gaps this review identifies:**
- **Domain coverage is narrow.** Outside the four "representative" applications (Web, SWE, Scientific, Conversational), entire agent sub-fields are missing — *medical/clinical, legal, embodied/robotics, financial, security/red-team*. For a medical-AI reader this is the biggest omission; not a single clinical agent benchmark is in scope.
- **No quantitative cross-benchmark comparison.** Table 1 has five qualitative columns but no difficulty/cost/cardinality numbers, so the reader cannot order benchmarks by hardness or by cost-per-evaluation.
- **No discussion of evaluator-model bias** in LLM-as-judge methods (self-preference, length bias, position bias), despite §6 leaning on LLM judges throughout.
- **No treatment of reproducibility** — seeds, environment snapshots, dataset drift in live benchmarks — even though "live benchmarks" is a flagged trend.
- **Cost and efficiency metrics** are named as a future direction but no concrete schema (token-/$/latency normalization) is proposed.
- **Multi-agent evaluation** is mentioned once in passing (Hammond et al., 2025), no taxonomy depth, despite being a fast-growing sub-area.
- **Not a systematic review.** No PRISMA-style protocol, no inclusion/exclusion criteria, no appendix on the search procedure — this is a *narrative* survey.

## Why It Matters for Medical AI
The medical-AI angle is **essentially absent** from the paper, which is precisely why it matters here. The five-perspective scaffold transfers almost line-for-line to clinical agents:

- **Capability axis.** Planning maps to differential-diagnosis trees and care-pathway reasoning; tool use maps to imaging modalities, lab APIs, and EHR queries (the same schema-adherence problems BFCL studies); self-reflection maps to verification of clinical conclusions before they reach a patient (the gap the survey flags as critical is even more critical here); memory maps to longitudinal patient records.
- **Benchmark dimensions.** The *safety* column that is "No" everywhere except TAU-Bench is the most clinically urgent gap — medical agents need policy-compliance, contraindication, and harm-avoidance evaluation by construction, not as an afterthought. Clinical-agent benchmarks should default Safety = Yes.
- **Framework dimensions.** *Trajectory* evaluation — supported by only 3/8 frameworks — is exactly what clinical workflows need (was the right test ordered, was the right structure followed, did the agent justify each step), and yet most off-the-shelf tooling still cannot do it.
- **Application coverage.** AgentClinic, Agent Hospital, MedAgents, MMedAgent, MedRAX, ChatCAD+, Pathfinder, NOVA, SlideSeek, agentic neuroimaging — all reviewed elsewhere on this blog — are exactly the "fifth application domain" the survey did not write. A clinical analogue of Table 1 (medical benchmarks on data / env / interface / metric / safety) and Table 2 (medical eval frameworks across the six axes) is the natural follow-up.

In short, the survey is the right *vocabulary* for medical agent evaluation, but the *contents* still have to be supplied by the medical-AI community.

## References
- **Paper:** Yehudai, Eden, Li, Uziel, Zhao, Bar-Haim, Cohan, Shmueli-Scheuer. *A Survey on Evaluation of LLM-based Agents.* arXiv:2503.16416 (v2, 23 Apr 2026). [https://arxiv.org/abs/2503.16416](https://arxiv.org/abs/2503.16416)
- **Companion:** A GitHub repository tracking new agent evaluation work (cited in the abstract footnote of the paper).
- **Related on this blog:**
  - [Xi et al. — The Rise and Potential of LLM Agents: A Brain–Perception–Action Survey](/paper/xi-llm-agents-brain-perception-action-survey/)
  - [Large Multimodal Agents: A Survey](/paper/large-multimodal-agents-survey/)
  - [Baymax — A Survey of Medical LLM Agents](/paper/med-llm-agents-survey-baymax/)
  - Clinical agent papers: AgentClinic, Agent Hospital, MedAgents, MMedAgent, MMedAgent-RL, MedRAX, ChatCAD+, Pathfinder, NOVA, SlideSeek, agentic neuroimaging, VoxelPrompt, MDAgents.
- **Foundational references the survey leans on:** Yao et al., 2022 (ReAct / agent-harness framing); Stein et al., 2023 (PlanBench); Deng et al., 2025 (SWE-bench Pro); Miserendino et al., 2025 (SWE-Lancer); Xue et al., 2025 (Online-Mind2Web vs WebVoyager); Hammond et al., 2025 (multi-agent).
