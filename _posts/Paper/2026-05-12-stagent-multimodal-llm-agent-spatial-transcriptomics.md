---
title: "STAgent: Spatial transcriptomics AI agent charts hPSC-pancreas maturation in vivo"
excerpt: "Claude-3.7-Sonnet + GPT-4o orchestrated via LangGraph drives an end-to-end ST workflow on a 60K-cell STARmap dataset of hPSC-pancreas grafts — but the agent's contribution is demonstrated, not benchmarked."
categories:
  - Paper
tags:
  - STAgent
  - LLM-Agent
  - Spatial-Transcriptomics
  - LangGraph
  - Claude-3.7-Sonnet
  - GPT-4o
  - Code-RAG
  - STARmap
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- STAgent wraps **Claude-3.7-Sonnet and GPT-4o** behind a **LangGraph** state graph with **code-RAG** (Chroma over scraped bioinformatics GitHub repos), a **Python REPL** tool, **SerpAPI / Google Scholar** retrieval, and a multimodal context memory so a single one-line prompt drives Scanpy / Squidpy / STAligner / Tangram / CellChat end-to-end.
- A **visual-reasoning** loop lets the MLLM read rendered spatial cell-type maps directly — calling out α-mantle / β-core islet architecture, mesenchymal-network expansion, and graft-host revascularization — then grounds each observation in retrieved literature before writing a publication-structured report.
- Applied to a new **STARmap dataset of >60,000 cells across SC-pancreas grafts at 4 / 16 / 20 weeks** post-transplantation (154-gene dual-species panel, HUES8 hPSCs into SCID-Beige mice), STAgent recapitulates the conventional-ML pipeline and adds context-aware β-cell maturation themes — but there is **no quantitative agent benchmark, no ablation, no hallucination audit, no variance, no cost/latency table**. C1–C6 are demonstrations on one dataset.

## Motivation
The medical-AI hook is type-1 / type-2 diabetes cell-replacement therapy: hPSC-derived β-cells are a promising graft candidate, but in-vitro-differentiated SC-pancreas remains transcriptionally and functionally immature, and prior scRNA-seq studies of transplanted grafts lack the spatial resolution to see islet architecture form. The authors first modify the STARmap protocol (3-hour mRNA denaturation, 30% formamide hybridization) to push single-molecule sensitivity into pancreas tissue, then argue that the analysis bottleneck — spanning Scanpy, Squidpy, STAligner, Tangram, and CellChat scripting plus biological interpretation — is what actually limits scaling these studies. STAgent is positioned as the answer to that bottleneck, with the broader claim that emergent multimodal-LLM capabilities (reasoning, vision-language, code generation) can replace narrow task-specific ML for scientific exploration.

## Core Innovation
The methodological contribution is orchestration, not modeling — there is no fine-tuning objective, no novel architecture, no loss function. STAgent's distinctive ingredients are (i) a **5-step reasoning–action cycle** on a LangGraph state graph (observe multimodal data → reason about analytical approach → generate + execute Python → evaluate + retrieve literature → plan next step) with persistent state across long analyses; (ii) a **code-RAG** index built by scraping public bioinformatics GitHub repos, segmenting them into function-preserving chunks, and embedding them into Chroma so the agent retrieves template snippets it adapts at inference; (iii) a **visual-reasoning** path where the MLLM consumes rendered spatial maps directly rather than tabular gene matrices, then immediately issues a Google Scholar query to ground each visual observation; and (iv) a **multimodal context memory** typed over analytical objectives, intermediate quantitative results, generated visualizations, and conversation history — load-bearing for cross-timepoint reasoning. A Streamlit frontend with chat / voice / image input lets the user pick OpenAI, Anthropic, or Gemini as backbone.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | STAgent performs end-to-end ST analysis "without user intervention" beyond a one-line prompt. | Fig. 2 narrative + screenshots of generated code/plots; Fig. 5 generated report. | ⭐⭐ (one dataset, no failure-mode reporting, no inter-run variance) |
| C2 | STAgent reduces "weeks to minutes" of expert analysis. | Single qualitative comparison in Discussion; no timing table, no token cost, no junior-bioinformatician baseline. | ⭐ (assertion only) |
| C3 | Visual reasoning identifies tissue architectures (mantle-core islets, mesenchymal networks, graft-host interface). | Fig. 3a annotations; matches conventional-ML findings. | ⭐⭐ (consistent with author-annotated structure, but does not show discovery of unannotated structure) |
| C4 | Context-aware gene-set analysis avoids "biologically irrelevant pathways" from generic enrichment. | Fig. 4a: generic GO/KEGG returns "muscle contraction / glucose metabolism" (flagged); STAgent returns 4 β-cell themes. | ⭐⭐ (the baseline is run by the authors as a strawman; no objective metric is applied to either output) |
| C5 | Autonomous reports are "comparable to scientific publications" with citation-grounded biological implications. | Fig. 5d sample report excerpts; ~6 retrieved references in-text. | ⭐⭐ (only one report shown; no rater study; **no fact-check of inline citations**) |
| C6 | The system generalizes across spatial transcriptomics datasets. | Discussion only — no second dataset tested. | ⭐ (aspiration, not a result) |
| C7 | The modified STARmap protocol enables single-molecule sensitivity in pancreas. | Extended Data Fig. 1; 154-gene panel detection. | ⭐⭐ (no SNR / detection-efficiency benchmark vs. unmodified protocol) |
| C8 | hPSC-pancreas grafts mature into islet-like α-β-δ structures in vivo over 20 weeks. | Conventional-ML pipeline (Fig. 1g–i, Extended Data Fig. 5); consistent with prior scRNA-seq. | ⭐⭐⭐ (the cleanest result; validated by the non-agent pipeline) |

**Honest read**: the best-evidenced contributions are the **wet-lab protocol (C7)** and the **biological finding (C8)** — both validated independently of the agent. Everything *about* STAgent (C1–C6) is a proof-of-concept demonstration. There is no comparison to a human bioinformatician on the same task, no ablation of the four backend components (memory / code-RAG / literature / vision), no hallucination check on retrieved citations, no reproducibility test across non-deterministic LLM runs. C2 ("weeks to minutes") is a marketing claim with no wall-clock or token numbers. C4 is conceptually the most novel point but the comparison is constructed unfavorably for the baseline.

## Method & Architecture

![STAgent system schematic and SC-pancreas STARmap dataset](/assets/images/paper/stagent/fig_p017_01.png)
*Figure 1: STAgent architecture and the SC-pancreas STARmap dataset spanning three maturation timepoints — wet-lab pipeline (a), conventional ML pipeline (b), STAgent backend with LangGraph orchestration / code-RAG / Python REPL / Scholar retrieval (c), Streamlit frontend (d), and UMAP + spatial maps at 4 / 16 / 20 weeks (e–i).*

The backend pairs Claude-3.7-Sonnet and GPT-4o (used **without fine-tuning**, model selectable from the Streamlit sidebar) with a LangGraph state graph that exposes four tools: a Python REPL (CodeAct-style), a Chroma vector store of code snippets retrieved from public bioinformatics repos, a SerpAPI / Google Scholar wrapper, and a typed multimodal memory. The conventional-ML baseline pipeline the agent is compared against runs Scanpy preprocessing (`normalize_total → log1p → scale → pca → neighbors → umap → leiden`), STAligner (graph attention autoencoder on STAGATE embeddings + Louvain) for 6 cross-slice spatial domains, Tangram for whole-transcriptome imputation (mapping GSE151117 scRNA-seq onto the 154-gene STARmap data → ~15,000 imputed genes), and CellChat on the imputed data with the human-specific CellChatDB.

![STAgent autonomously generates and executes analysis code](/assets/images/paper/stagent/fig_p019_01.png)
*Figure 2: STAgent autonomously generates Scanpy/Squidpy code and produces UMAP, temporal cell-composition, spatial distribution, and neighborhood-enrichment analyses from a one-line natural-language prompt.*

Key hyperparameters and operational details that are **not reported** in the paper: temperature / top-p for either backbone; the specific OpenAI embedding model used in the code-RAG index; max-step budget; code-execution timeout; retry policy on failed Python execution; which GitHub repos populate the Chroma store; licensing handling for retrieved code. Cost and latency per analysis are reported only as "minutes".

## Experimental Results

There are **no quantitative comparisons of STAgent vs. baseline** — no agent-accuracy table, no code-correctness rate, no hallucination rate, no time-per-analysis, no report-quality score. The "results" are biological findings extracted from one dataset and presented in parallel by both pipelines.

| Finding | Conventional ML evidence | STAgent evidence | Agreement |
|---|---|---|---|
| α-mantle / β-core islet architecture by week 20 | Fig. 1g–i spatial maps; Ext. Data Fig. 5e neighborhood Z-scores | Fig. 3a visual-reasoning text: "α-cells reorganize into a mantle-peripheral arrangement around central β-cell cores" | Yes |
| Mesenchymal network expansion 4 → 20 weeks | Ext. Data Fig. 5d bar plots; CellChat Ext. Data Fig. 6c–e | Fig. 3a + Fig. 5d Biological Implications section | Yes |
| Endocrine-endocrine signaling strengthens over time | CellChat differential interactions (Ext. Data Fig. 6e) | STAgent neighborhood heatmaps (Fig. 2e) | Yes |
| Two-stage host revascularization (peripheral → infiltrating) | Inferred from spatial maps + CellChat | Fig. 5d, with retrieved citation to Pepper et al. 2013 | Agent adds literature grounding |
| **β-cell maturation themes** (FXYD2/SCG5/IAPP/CPE/NPTX2 up; GCG up → bihormonal plasticity; PARVB/CDH7 cell-matrix; β-cell heterogeneity) | Differential expression on 4 vs 20 week β-cells (Fig. 4b volcano) | Fig. 4d four functional themes via Google Scholar context | Agent provides thematic grouping the baseline does not |

![Visual reasoning over rendered cell-type maps with literature grounding](/assets/images/paper/stagent/fig_p020_01.png)
*Figure 3: STAgent reads rendered spatial maps directly, calls out architectural features at 4w vs 20w (α-mantle / β-core, mesenchymal network, graft-host interface), and pairs each observation with a Google Scholar retrieval to ground it in published literature.*

![Context-aware gene-set analysis returns β-cell-specific themes](/assets/images/paper/stagent/fig_p022_01.png)
*Figure 4: Where traditional GO/KEGG/Reactome enrichment on the same gene list surfaces irrelevant hits ("muscle contraction, glucose metabolism" — top, red flag), STAgent's literature-grounded reasoning returns four biologically coherent β-cell maturation themes (bottom).*

![End-to-end research report generation](/assets/images/paper/stagent/fig_p023_01.png)
*Figure 5: An autonomously generated research report covering transcriptomic profile evolution, cell composition dynamics, spatial organization, cell-cell signaling, islet architecture recapitulation, mesenchymal niche, vascular dynamics, and exocrine relationships — with inline citations to Arrojo e Drigo, Brereton, Hematti, Barachini, Sordi, and Pepper et al.*

### What is missing
- No GPT-4o-only, Claude-only, or no-RAG ablation; we cannot attribute gains to any backend component.
- No hallucination audit on retrieved literature — the cited papers are not checked against the report's attributed claims.
- No expert-rater study of report quality versus a human-written analysis of the same dataset.
- No statistics on code-execution failure / retry rate, no test-retest agreement across non-deterministic LLM runs.
- No comparison to other agentic bio frameworks (CRISPR-GPT, ChemCrow, AI Scientist) on a shared task.

## Limitations
Acknowledged by the authors: extensions to spatial proteomics / metabolomics / multiomics are future work; application to larger and more diverse datasets is future work; the current demonstration is on a single biological system.

Not addressed:
- **Hallucination** — the agent fetches papers and writes a report, but the paper does not check whether retrieved citations support the claims attributed to them.
- **Reproducibility** — running the same prompt twice on the same data with Claude-3.7-Sonnet is non-deterministic; no variance reporting.
- **Cost** — no token count, API spend, or wall-clock numbers. "Minutes" is the only quantitative latency claim.
- **Failure modes** — what does STAgent do when code execution fails, when literature retrieval returns nothing relevant, or when a spatial map contains an artifact?
- **Code-RAG provenance** — which GitHub repos are indexed, how licensing is handled, whether retrieved code is correct vs. silently outdated.
- The host is SCID-Beige (immunodeficient), so the agent cannot identify immune-graft interactions — precisely the variable that determines real transplant outcomes. The agent cannot see what is not in the panel.
- All three timepoints come from the same HUES8 line and same differentiation protocol; no inter-line or inter-batch variability.
- The system depends on closed proprietary LLMs (Claude, GPT-4o); reproducibility by other labs depends on those APIs being available and stable.
- The "Sptial" typo on Extended Data Fig. 5 is harmless but a reminder this is an unreviewed preprint.

## Why It Matters for Medical AI
If the rigor gap closes, STAgent points at a real workflow: a bioinformatician hands a multimodal LLM a spatial-omics dataset and a clinical question, and the agent runs the full Scanpy / STAligner / Tangram / CellChat stack, grounds each finding against the recent literature, and emits a publication-structured report. That is the right shape for clinical research-support tooling. But the contrast with [BioMaze]({{ site.baseurl }}{% post_url Paper/2026-05-12-biomaze-llm-biological-pathway-reasoning-benchmark %}) — which shipped a 5,100-question benchmark, accuracy tables across 7 backbones, and per-component ablations on 200 hand-labeled errors — is sharp. Both papers position LLMs as orchestrators of biological reasoning; only one of them *measures* it. For medical-AI deployment, what STAgent demonstrates is a UX; what it does not yet establish is that the UX is safe to act on. The cleanest contributions in this paper are the modified STARmap protocol and the biological finding that hPSC-pancreas grafts form native-like islet architecture by week 20 — both validated independently of the agent. Take those at face value; treat the agent as a compelling demo waiting on a benchmark.

## References
- Paper (bioRxiv, April 4 2025): <https://doi.org/10.1101/2025.04.01.646731>
- Code (promised at publication): <https://github.com/LiuLab-Bioelectronics-Harvard/STAgent>
- LangGraph: <https://langchain-ai.github.io/langgraph/>
- STARmap (Wang et al., 2018): in-situ 3D spatial transcriptomics protocol.
- Scanpy / Squidpy / STAligner / Tangram / CellChat — the analytical stack STAgent orchestrates.
- Pagliuca / Veres / Alvarez-Dominguez SC-β protocol — used to differentiate the HUES8 hPSCs.
- Reference scRNA-seq atlas: GEO GSE151117 (used by Tangram for whole-transcriptome imputation).
- Related agentic-bio frameworks (not compared in the paper): CRISPR-GPT, ChemCrow / Boiko et al., AI Scientist.
- Cross-reference: [BioMaze]({{ site.baseurl }}{% post_url Paper/2026-05-12-biomaze-llm-biological-pathway-reasoning-benchmark %}) — the benchmark-shaped counterpart to STAgent's demonstration-shaped contribution.
