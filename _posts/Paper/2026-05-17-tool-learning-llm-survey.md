---
title: "Tool Learning with Large Language Models: A Survey"
excerpt: "A 2024 survey that organizes 150+ tool-learning works into a four-stage pipeline (plan / select / call / respond) with a tuning-free vs tuning-based axis at every stage, plus a stage-tagged inventory of 33 benchmarks topped by ToolBench2 (16,464 tools / 126,486 instances)."
categories:
  - Paper
  - LLM-Agents
tags:
  - Tool-Learning
  - LLM-Agents
  - Survey
  - Tool-Use
  - Tool-Retrieval
  - Taxonomy
  - Benchmarks
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/tool-learning-llm-survey/
---

## TL;DR
- The paper organizes the post-2023 tool-learning literature into a **four-stage canonical pipeline** — *task planning → tool selection → tool calling → response generation* — and re-classifies every cited system by which stage(s) it targets and whether it is **tuning-free or tuning-based**.
- The single most useful deliverable is **Table 1**, a stage-tagged inventory of **33 benchmarks** spanning manual, RESTful, RapidAPI, and HuggingFace-hub tool sources; the largest is **ToolBench2 (16,464 tools / 126,486 instances)** and the only two multi-modal entries are **MLLM-Tool (932 tools / 11,642 instances)** and **m&m's (33 tools / 4,427 instances)**.
- This is a **narrative survey, not a meta-analysis** — no head-to-head numbers, no PRISMA protocol, no cost/latency comparisons; capability assertions such as "iterative > one-step" and "tool use improves interpretability" are stated rather than evidenced.

## Motivation
LLMs hallucinate, hold stale parametric memory, and cannot execute side-effectful actions. Tool learning — letting an LLM call external functions (search, calculator, Python, REST APIs, weather, maps, multi-modal models) — is the field's standard remedy, and 2023–2024 produced an explosion of systems: Toolformer, ReACT, ToolLLaMA, HuggingGPT, Gorilla, RestGPT, AnyTool, ConAgents, α-UMi, ToolkenGPT, and many more.

Existing surveys closest to this one (Mialon et al. 2023; Qin et al. 2023 "Tool learning with foundation models"; Wang et al. 2024) either pre-date the surge or sit at a more generic agents/RAG/reasoning altitude. This paper positions itself as a tool-learning-focused update, structured around an **implementable workflow** rather than a conceptual essay. Medical-AI relevance is acknowledged only in passing — §3.2 lists chemistry/biology/medicine/recommender systems as "Expertise Enhancement" use cases — but there is no medical case study, no clinical-tool benchmark, and no discussion of regulatory concerns.

## Core Innovation
This is a survey, so "innovation" means the organizing scheme it imposes. Three pieces are genuinely the authors' own:

1. **The double axis.** Each of the four pipeline stages is consistently sub-divided into *tuning-free* (CoT/ReACT/RestGPT/HuggingGPT/TPTU/AnyTool/ConAgents) and *tuning-based* (Toolformer/Gorilla/ToolLLaMA/ToolkenGPT/Themis/STE/Confucius/α-UMi). This second axis — not the pipeline itself — is the survey's clearest structural contribution.
2. **Stage-tagged benchmark inventory.** Table 1 lists 33 benchmarks and marks for each which of the four stages it evaluates, plus tool count, instance count, source, multi-tool flag, and executable flag.
3. **Cross-walking terminology.** The "one-step vs iterative" execution-paradigm split is aligned with Wang et al.'s "planning with feedback" and Huang et al.'s "interleaved decomposition," letting a reader move between vocabularies.

What is **not** original: the four-stage pipeline itself is aggregated from RestGPT, HuggingGPT, and TPTU; the COMP@K retrieval-completeness metric is from COLT, not this paper; and the "iterative > one-step" verdict is inherited from ReACT/DFSDT.

## Claims & Evidence Analysis

| # | Claim | Evidence in the paper | Strength |
|---|-------|------------------------|----------|
| C1 | First survey focused specifically on tool learning with LLMs (vs. broader agent / RAG / reasoning surveys). | §1 explicitly contrasts with Mialon 2023, Qin 2023, Wang 2024; >150 works covered as of Nov 2024 v3. | ⭐⭐⭐ |
| C2 | Tool-learning workflow decomposes into four stages: plan → select → call → respond. | Figure 3 + §4. Authors themselves state this is "adopted in numerous works" and cite RestGPT, HuggingGPT, TPTU — i.e. it is a *summary* of existing pipelines, not an original framework. | ⭐⭐ |
| C3 | Tool retrieval is necessary because real systems have too many tools to fit in context. | Cites ToolLLaMA, Confucius, ProTIP, ToolLens for context-length / latency issues; no head-to-head numbers in the survey itself. | ⭐⭐ |
| C4 | Iterative paradigm > one-step paradigm. | §4.1 asserts the shift "marks a significant advancement" but the survey runs no comparison; the claim is inherited from ReACT and DFSDT primary papers. | ⭐ |
| C5 | Tool learning improves interpretability (§3.5) and robustness to prompt rewording (§3.6). | Cites citation-grounding works and prompt-sensitivity surveys; no study that *quantifies* improved interpretability/robustness *because of* tool use. RoTBench — cited in the same survey — actually shows GPT-4 degrading badly under tool-input noise. | ⭐ |
| C6 | Tool learning is essential for high-stakes domains (aviation, healthcare, finance). | One sentence in §3.5 with two citations; no medical-AI deep dive, no clinical benchmark. | ⭐ |
| C7 | MLLM-Tool and m&m's extend tool learning into multi-modal. | Both benchmarks exist and are catalogued in Table 1; §6.7 acknowledges multi-modal "has not been extensively studied." | ⭐⭐ |

**Honest synthesis.** The durable contributions are (a) the §3 motivation taxonomy, (b) the §5.1 33-benchmark stage-tagged inventory, and (c) the consistent tuning-free vs tuning-based double axis. The weak parts are §3.5–§3.6 (interpretability/robustness asserted, not evidenced) and the "iterative > one-step" verdict (stated, not measured). The paper is visibly hand-curated — no PRISMA-style search protocol, no inclusion/exclusion criteria, no inter-rater categorization — typical of a narrative rather than systematic review.

## Method & Architecture

![Timeline of tool-learning works from 2022-12 to 2024-08, with publication-venue distribution](/assets/images/paper/llm-agents-tool-learning-survey/page_002.png)
*Figure 1: Tool-learning paper timeline (Dec 2022 → Aug 2024). The explosion runs from Toolformer (2023-02) through ReACT, HuggingGPT, ToolLLaMA, Gorilla, and AnyTool. Author-reported venue distribution: ~15% NeurIPS, ~15% ACL/COLING/AAAI, ~64% Others — the long tail reflects how much of this literature lives in workshops and preprints.*

![Full survey taxonomy tree — Why × How × Benchmarks × Challenges](/assets/images/paper/llm-agents-tool-learning-survey/page_004.png)
*Figure 2: The survey's full taxonomy — Why Tool Learning (six motivations) × How Tool Learning (four stages: Task Planning, Tool Selection, Tool Calling, Response Generation) × Benchmarks & Toolkits & Evaluation × Challenges.*

### The four-stage pipeline (§4)

1. **Task Planning (§4.2).** Decompose the user query into sub-questions and a dependency order. *Tuning-free:* CoT, ReACT, ART, RestGPT, HuggingGPT, TPTU, ToolChain*, ControlLLM, ATC, Sum2Act, DRAFT. *Tuning-based:* Toolformer, TaskMatrix.AI, Toolink, TPTU-v2, α-UMi, COA, DEER, SOAY, TP-LLaMA, APIGen.
2. **Tool Selection (§4.3).** Split first by *who selects*. **Retriever-based** (large catalog, top-K filter): term-based (TF-IDF, BM25 as in Gorilla) vs. semantic-based (Sentence-BERT, Contriever, CRAFT, ProTIP, COLT, ToolRerank). **LLM-based** (descriptions go into context, LLM picks): tuning-free (CoT, ReACT, DFSDT in ToolLLaMA, ChatCoT, ToolNet, AnyTool, GeckOpt) vs. tuning-based (ToolBench fine-tune, TRICE, ToolLLaMA, Confucius, ToolVerifier).
3. **Tool Calling (§4.4).** Extract correctly typed parameters and POST. Tuning-free: Reverse Chain, EasyTool, doc-compression, ConAgents. Tuning-based: Gorilla, GPT4Tools, ToolkenGPT (special "toolken" tokens), Themis, STE.
4. **Response Generation (§4.5).** Either **Direct Insertion** (slot tool output into a template — TALM, Toolformer, ToolkenGPT) or **Information Integration** (feed tool output back into the LLM — RestGPT schema-summarization, ToolLLaMA truncation, ReCOMP compressor, ConAgents schema-free extraction).

![Four-stage workflow with one-step vs iterative execution paradigms](/assets/images/paper/llm-agents-tool-learning-survey/page_008.png)
*Figure 3: The four-stage tool-learning workflow (left) and the one-step vs iterative execution paradigms (right). One-step (Toolformer, HuggingGPT) plans the full decomposition up front; iterative (ReACT, ToolLLaMA, RestGPT, Confucius) interleaves plan / call / observe so later steps can react to earlier tool outputs.*

### Tool retrieval framed as comparative screening
A worthwhile angle the survey draws out: tool retrieval is **not** classic top-1 document IR. The right objective is **set completeness** — the top-K must cover *all* tools the query needs, because a missing tool stalls the whole plan. This is operationalized as **COMP@K** (from COLT), an indicator that returns 1 only if the entire ground-truth tool set Φ_q sits inside the top-K Ψ_q^K:

$$\text{COMP}@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\bigl(\Phi_q \subseteq \Psi_q^K\bigr)$$

The recommended architecture is two-stage **retrieve-then-LLM-select**: an embedding retriever (Sentence-BERT / Contriever / ToolRerank) cheaply screens the catalog, then an LLM-based selector performs fine-grained comparison among the survivors. The "comparative screening" framing is the survey's clearest implementable recommendation, though the COMP@K metric itself is credited to COLT, not originated here.

### A note on what the typology actually is
The survey's typology is **two-dimensional: four pipeline stages × tuning-free vs tuning-based**. It is not a classical agent-architecture taxonomy (reactive vs deliberative, single- vs multi-agent, memory architecture, planner–executor separation). ConAgents and α-UMi are mentioned as multi-agent / planner–executor exemplars, but no separate "agent design" axis is constructed. For a reader looking for *agent design patterns* per se, this survey is narrower than its title suggests — it is really a **tool-use pipeline** survey.

## Experimental Results
No experiments — this is a survey, with no head-to-head benchmarking and no "method X on benchmark Y" tables. The substantive "result" is the benchmark inventory in §5.1 (Table 1), reproduced here in compact form:

| Benchmark | Stages covered | # Tools | # Instances | Source | Multi-tool | Executable |
|---|---|---|---|---|---|---|
| API-Bank | 1,2,3,4 | 73 | 314 | Manual | ✓ | ✓ |
| APIBench (Gorilla) | 2,3 | 1,645 | 16,450 | HF/TF/Torch hubs | ✗ | ✗ |
| ToolBench1 | 2,3 | 232 | 2,746 | Public APIs | ✗ | ✓ |
| ToolAlpaca | 2,3,4 | 426 | 3,938 | Public APIs | ✗ | ✗ |
| RestBench | 1,2,3 | 94 | 157 | RESTful APIs | ✓ | ✗ |
| **ToolBench2 (ToolLLM)** | **1,2,3,4** | **16,464** | **126,486** | **Rapid API** | **✓** | **✓** |
| MetaTool | 1,2 | 199 | 21,127 | OpenAI plugins | ✓ | ✗ |
| TaskBench | 1,2,3 | 103 | 28,271 | Public APIs | ✓ | ✓ |
| T-Eval | 1,2,3 | 15 | 533 | Manual | ✓ | ✓ |
| ToolEyes | 1,2,3,4 | 568 | 382 | Manual | ✓ | ✓ |
| UltraTool | 1,2,3 | 2,032 | 5,824 | Manual | ✓ | ✗ |
| API-BLEND | 2,3 | — | 189,040 | Existing datasets | ✓ | ✓ |
| Seal-Tools | 2,3 | 4,076 | 14,076 | Manual | ✓ | ✗ |
| ShortcutsBench | 2,3 | 1,414 | 7,627 | Public APIs | ✓ | ✓ |
| GTA | 2,3,4 | 14 | 229 | Public APIs | ✓ | ✓ |
| WTU-Eval | 1 | 4 | 916 | BMTools | ✓ | ✓ |
| AppWorld | 1,2,3 | 457 | 750 | FastAPI | ✓ | ✓ |
| ToolQA | QA | 13 | 1,530 | Manual | ✗ | ✓ |
| ToolEmu | Safety | 311 | 144 | Manual | ✗ | ✓ |
| ToolTalk | Conversation | 28 | 78 | Manual | ✗ | ✓ |
| VIoT | VIoT | 11 | 1,841 | Public models | ✗ | ✓ |
| RoTBench | Robustness | 568 | 105 | ToolEyes | ✓ | ✓ |
| **MLLM-Tool** | **Multi-modal** | **932** | **11,642** | Public models | ✓ | ✓ |
| ToolSword | Safety | 100 | 440 | Manual | ✓ | ✓ |
| SciToolBench | Sci-Reasoning | 2,446 | 856 | Manual | ✓ | ✓ |
| InjecAgent | Safety (prompt-injection) | 17 | 1,054 | Public APIs | ✗ | ✓ |
| StableToolBench | Stability | 16,464 | 126,486 | ToolBench2 | ✓ | ✓ |
| **m&m's** | **Multi-modal** | **33** | **4,427** | Public models | ✓ | ✓ |
| GeoLLM-QA | Remote sensing | 117 | 1,000 | Public models | ✓ | ✓ |
| ToolLens | Tool retrieval | 464 | 18,770 | ToolBench2 | ✓ | ✓ |
| SoAyBench | Academic search | 7 | 792 | AMiner | ✓ | ✓ |
| ToolSandbox | Conversation | 34 | 1,032 | Rapid API | ✓ | ✓ |
| CToolEval | Chinese | 398 | 6,816 | Public apps | ✓ | ✓ |

Authors flag a recurrent **data-quality problem**: tools in many benchmarks are "inaccessible or non-functional" (API rot), and queries are typically LLM-synthesized rather than from real users. **GTA, ShortcutsBench, and AppWorld** are highlighted as the cleanest exceptions that use real-world tools and user-driven queries.

The per-stage **evaluation metrics** the survey collates (§5.3) are: pass rate and tool-usage-awareness accuracy for planning (MetaTool, UltraTool, ToolLLM, T-Eval, RestGPT); Recall@K / NDCG@K / COMP@K for selection; parameter-spec conformance checks for calling (T-Eval, ToolEyes, UltraTool); BLEU / ROUGE-L / EM / F1 for response generation — i.e. metrics borrowed wholesale from text generation, which is itself a critique.

## Limitations

Authors-acknowledged (§6):
1. **Latency** — even simple ChatGPT-plugin queries take ~5 s.
2. **Evaluation rigor** — no comparative benchmarking framework; human eval is expensive and irreproducible; ToolEval is a cheap proxy but does not reflect real user preference.
3. **Tool quality** — tool sets are aggregated, often inaccessible/non-functional, and inconsistently formatted across sources.
4. **Safety** — RoTBench shows GPT-4 collapses under input noise, ToolSword flags safety-awareness gaps, InjecAgent demonstrates prompt-injection vulnerability.
5. **No unified framework** — most works tackle only one of the four stages.
6. **Synthetic queries** — most benchmarks use LLM-generated user queries, not real ones.
7. **Multi-modal underexplored.**

Not flagged by the authors but worth raising:
- **No quantitative meta-analysis** — 150+ works catalogued, zero aggregate numbers (no "fine-tuned methods improve pass rate by X over prompting on average").
- **No cost / FLOPs analysis** — tuning-based methods are repeatedly described as "more capable but expensive," with no training-cost or inference-cost numbers anywhere.
- **No coverage of production tool-calling APIs** — OpenAI function calling, JSON mode, Anthropic tool_use are absent despite being the dominant deployed paradigm.
- **No medical / clinical case study** despite naming healthcare as a beneficiary in §3.5.
- **Inconsistent "tool" definition** — §2 adopts "each API = one tool" but Table 1 mixes that convention with the alternative "tool = bundle of APIs" (ToolBench2 etc.); cross-benchmark tool counts are therefore not strictly comparable, and the survey does not warn the reader.
- **Multi-modal coverage is shallow** — exactly one short §3.4 paragraph, two catalogued benchmarks (MLLM-Tool, m&m's), and one §6.7 future-work paragraph admitting the gap. No discussion of visual tool calling (e.g. ViperGPT's executable scene programs, even though ViperGPT appears in the Figure 1 timeline), no multi-modal tool retrieval coverage, and no medical-imaging tools (MONAI, radiology APIs) anywhere.

## Why It Matters for Medical AI
For a medical-AI reader the value of this survey is **structural, not empirical**. The four-stage pipeline (plan → select → call → respond) and the tuning-free vs tuning-based axis are a clean scaffold for designing a clinical agent that has to choose among, say, a radiology-report retriever, a CT segmentation model, a drug-interaction database, and an EHR query API:

- **Task planning** maps to triaging a clinician's free-text request into ordered sub-tasks (image segmentation before measurement before reporting).
- **Tool selection** maps directly to deciding among many candidate models/APIs — and COMP@K's *completeness* framing is more clinically appropriate than top-1 retrieval, because skipping a required tool is a different failure mode than ranking it second.
- **Tool calling** is where structured-output / function-calling discipline matters most clinically; the survey under-covers production APIs here.
- **Response generation** must reconcile multi-tool outputs without hallucination, which is harder under the *Information Integration* style than the *Direct Insertion* style — relevant because most clinical reports require integration.

The survey itself, however, does **not** cover medical tools, medical-imaging APIs, or clinical safety/regulatory considerations. Anyone building a medical agent should treat this paper as a vocabulary reference and combine it with domain-specific surveys (e.g. multimodal medical-agent surveys, ChatCAD-line systems).

## References
- Paper: [Tool Learning with Large Language Models: A Survey, arXiv:2405.17935](https://arxiv.org/abs/2405.17935) — Qu, Dai, Wei, Cai, Wang, Yin, Xu, Wen (Renmin University of China, Baidu, ICT-CAS); Frontiers of Computer Science 2024, arXiv v3 4 Nov 2024.
- Companion paper list (no method release): [github.com/quchangle1/LLM-Tool-Survey](https://github.com/quchangle1/LLM-Tool-Survey)
- Pipeline ancestors: RestGPT, HuggingGPT, TPTU — the four-stage workflow is consolidated from these.
- Retrieval-completeness metric: COLT (COMP@K).
- Adjacent surveys: Mialon et al. 2023; Qin et al. 2023 *Tool learning with foundation models*; Wang et al. 2024 (agents/RAG/reasoning altitude).
- Multi-modal benchmarks catalogued: MLLM-Tool (932 tools / 11,642 instances), m&m's (33 tools / 4,427 instances).
- Cleanest real-world-tool benchmarks (per the survey): GTA, ShortcutsBench, AppWorld.
