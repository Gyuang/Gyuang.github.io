---
title: "Large Multimodal Agents: A Survey"
excerpt: "A 2024 survey that organizes ~34 multimodal LLM agents into a 4-type taxonomy (closed vs fine-tuned planner × with/without long memory) plus a multi-agent split, and flags evaluation fragmentation as the field's central open problem."
categories:
  - Paper
  - LLM-Agents
tags:
  - LMA
  - Multimodal-Agents
  - LLM-Agents
  - Survey
  - Taxonomy
  - Tool-Use
  - Memory
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/large-multimodal-agents-survey/
---

## TL;DR
- The paper proposes a **4-type taxonomy of single-agent LMAs** along two orthogonal axes — *closed-source vs fine-tuned open-source LLM planner* and *with vs without long-term memory* — plus a separate split for multi-agent collaboration.
- It indexes **~34 systems in Table 1** and a **tool zoo in Table 2** (BLIP/Grounding-DINO/SAM/Whisper/StyleTTS 2/Editly/OCR/diffusion APIs), and decomposes every LMA into four components: *perception, planning, action, memory*.
- There is **no headline metric and no benchmark comparison** — the survey's most defensible argument is that **evaluation is fragmented** (GAIA, SmartPlay, VisualWebArena are flagged as the closest steps toward a standard), which makes "Type IV > Type I" or "dynamic > static" claims unverifiable from the data the authors present.

## Motivation
LLM-driven *text-only* agents (ReAct, Toolformer) already match humans on narrow tasks but break down in real environments because the world is multimodal — web pages, GUIs, videos, robotic scenes. Existing agent surveys (Wang 2023; Sumers 2023; Xi 2023) cover language agents but barely touch multimodality.

This paper fills that gap by (i) cataloguing how *perception, planning, action, and memory* are actually implemented in 30+ recent multimodal systems, (ii) proposing a taxonomy so practitioners can choose a framework, and (iii) flagging the lack of standardized evaluation as the field's most urgent problem. The medical-AI angle is implicit — the perception/planning/action/memory decomposition transfers directly to medical agents — but, as flagged below, **zero medical LMAs are in the survey itself**.

## Core Innovation
This is a conceptual paper, not a model paper. Three things are new:

1. **A four-component view of any LMA** — every system is decomposed into *Perception*, *Planning (the "brain")*, *Action*, and *Memory*. Each component is further classified (e.g., Perception is "caption-everything" vs "sub-task tools" vs "visual-vocabulary retrieval"; Planning has 4 sub-axes including format and replanning).
2. **A 4-type single-agent taxonomy** (Figure 2): Type I closed-LLM planner / no memory; Type II fine-tuned open-LLM planner / no memory; Type III indirect (tool-mediated) long memory; Type IV native long memory.
3. **A two-sub-type multi-agent taxonomy** (Figure 3), again split by presence of shared long memory; roles include perceiver, patroller, planner, exploration, selection, deduction, recall (MP5, MemoDroid, AVIS, DiscussNav).

No equations or training procedure — the value is the vocabulary plus Table 1 and Table 2.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | LMAs face "more sophisticated challenges than language-only agents" because real-world inputs are multimodal. | Argued by example (web-search walkthrough, p.2). No quantitative gap measurement. | ⭐⭐ |
| C2 | Existing LMAs cluster into exactly four types along (closed vs fine-tuned planner) × (with/without long memory). | Figure 2 + Table 1 successfully sort every cited system into one of the four bins. | ⭐⭐⭐ |
| C3 | Multimodal long-term memory is essential for open-world tasks. | JARVIS-1 (Minecraft, 200+ tasks) and DEPS cited; no controlled memory-on vs memory-off ablation across types. | ⭐⭐ |
| C4 | Dynamic planning outperforms static planning. | Asserted; supported only by listing which strong systems use dynamic planning. No benchmark table. | ⭐ |
| C5 | Fine-tuned open-source planners (Type II) achieve "comparable" capability to closed-source GPT planners. | Asserted in §3, citing LLaVA-Plus, GPT4Tools, Tool-LMM. No head-to-head numbers reported in this survey. | ⭐ |
| C6 | Current evaluation methodologies are insufficient; GAIA / SmartPlay / VisualWebArena are steps forward. | §5 surveys metrics and benchmarks; the heterogeneity problem is well-argued and concrete. | ⭐⭐⭐ |
| C7 | Multi-agent collaboration "alleviates the burden on a single agent, thereby enhancing task performance." | Citations to MP5, MemoDroid, AVIS, DiscussNav. No aggregated quantitative comparison vs single-agent baselines. | ⭐⭐ |
| C8 | The survey covers the field comprehensively (Nov 2022 – Feb 2024). | Figure 1 timeline lists ~40 systems; the GitHub awesome-list is the canonical extended reference. | ⭐⭐ |

The taxonomy (**C2**) and the evaluation critique (**C6**) are the durable contributions and would justify citing this paper. Capability claims (**C3, C4, C5, C7**) are descriptive rather than evidenced — typical of a fast-moving survey written before benchmarks existed, but readers should *not* cite this paper as evidence that "dynamic > static" or "Type IV > Type I." The coverage claim (**C8**) is broad but not systematic: no PRISMA-style protocol, no inclusion/exclusion criteria, no appendix detailing the search.

## Method & Architecture

![The four types of single-agent LMAs along the closed-vs-fine-tuned and no-memory-vs-long-memory axes](/assets/images/paper/large-multimodal-agents-survey/page_004.png)
*Figure 1: The paper's identity figure — four LMA types. (a) **Type I**: closed-source LLM planner, no long-term memory. (b) **Type II**: fine-tuned open-source LLM planner, no long-term memory. (c) **Type III**: planner with indirect long-term memory (accessed via a retrieval tool). (d) **Type IV**: planner with native long-term memory that the LLM reads and writes directly.*

### Perception
Three implementation styles:

- **Caption-everything-to-text** via vision foundation models (Visual ChatGPT, MM-ReAct) — simple, but loses spatial detail.
- **Sub-task tools** that handle complex modalities directly (DoraemonGPT for video, ChatVideo) — preserves modality-specific information.
- **Visual-vocabulary retrieval** where CLIP-encoded environment states are matched to descriptive sentences (JARVIS-1 in Minecraft).

### Planning (the "brain")
Four sub-axes:

- **Model** — GPT-3.5/4 dominates; open-source planners use LLaMA, Vicuna, LLaVA, or custom decoders.
- **Format** — natural language (HuggingGPT), program code (ViperGPT, VisProg), or hybrid (AssistGPT).
- **Inspection & Reflection** — whether the planner can self-check and re-plan; memory-augmented systems (JARVIS-1, AVIS, DoraemonGPT) score better here, sometimes with Monte-Carlo expansion.
- **Planning method** — *static/fix* (CoT-style decomposition at the start, no replanning) vs *dynamic* (replan per step based on environment feedback).

### Action
Three categories:

- **Tool use** — VFMs, APIs, Python interpreters, web search.
- **Embodied** — robot/avatar/Minecraft actions.
- **Virtual** — mouse/keyboard/GUI clicks (Mobile-Agent, AppAgent, MM-Navigator).

Two learning approaches: prompt-based tool descriptions vs fine-tuning on action data (LLaVA-Plus, GPT4Tools, Auto-UI).

### Memory
Short-term memory suffices for simple settings; **long-term memory matters in open-world tasks**. Multimodal long memory is rare — most systems flatten everything to text. JARVIS-1's memory is the cited exception, storing key-value pairs of (multimodal state, successful plan) with CLIP-based retrieval:

$$p(t \mid x) \propto \text{CLIP}_v(k_t)^\top \text{CLIP}_v(k_x)$$

where $k_x$ is the encoded current visual state and $k_t$ is a stored experience.

![Figure 3 multi-agent frameworks and Table 2 tool catalogue by modality](/assets/images/paper/large-multimodal-agents-survey/page_005.png)
*Figure 2: Multi-agent collaboration patterns (top) and the per-modality tool catalogue (bottom): image (BLIP/BLIP2/InstructBLIP, Grounding-DINO, EasyOCR, InstructPix2Pix, SD/DALL·E 3, SAM/PaddleSeg), text (Bing Search, PyLint/PyChecker), video (Editly, OSTrack), audio (Whisper, StyleTTS 2).*

![Table 1 — the full taxonomy table indexing ~34 LMAs](/assets/images/paper/large-multimodal-agents-survey/page_003.png)
*Figure 3: Table 1 — ~34 LMAs indexed with 10 attribute columns (modalities supported, planner model, plan format, inspection, planning method, action type, action learning, multi-agent, long memory). This is the survey's most reusable artifact for practitioners selecting a framework.*

## Experimental Results
**There are no experiments.** The paper explicitly notes that heterogeneous evaluation makes a numerical comparison of LMAs impossible today, and frames this as the field's central open problem.

The closest substitute is the **taxonomy distribution** (counted from Table 1):

| Type | Defining trait | Representative systems | Long-term memory |
|------|----------------|------------------------|------------------|
| **I** | Closed-LLM planner via prompting | VisProg, ViperGPT, Visual ChatGPT, MM-ReAct, HuggingGPT, Chameleon, CLOVA, Cola, M3, AssistGPT, Mulan, Mobile-Agent, ControlLLM, CRAFT, LLaVA-Interactive, MusicAgent, AudioGPT, DEPS, GRID, DroidBot-GPT, ASSISTGUI | No |
| **II** | Fine-tuned open LLM planner | GPT4Tools (LLaMA), LLaVA-Plus (LLaVA), Tool-LMM (Vicuna), STEVE-13B, EMMA, Auto-UI, WebWISE, GPT-Driver | No |
| **III** | Indirect (tool-mediated) long memory | DoraemonGPT, ChatVideo, OS-Copilot | Yes |
| **IV** | Native long memory | JARVIS-1, OpenAgents, MEIA, AppAgent, MM-Navigator, DLAH, Copilot, WavJourney | Yes |
| **Multi-agent** | Multiple cooperating LMAs | AVIS, MP5, MemoDroid, DiscussNav | Mixed |

Qualitative observations the survey foregrounds:

- **Type I dominates by count**, reflecting that closed-source GPT-4 is the most reliable planner currently available.
- **Long-term multimodal memory remains rare** — only JARVIS-1 stores raw multimodal states rather than text projections.
- **Dynamic (replanning) planners cluster in Type III/IV and multi-agent systems**; static planners cluster in early Type I.

### Application landscape
The survey closes with eight application areas. The wide banner below is the at-a-glance view; the spotlights underneath show three of the most active deployment surfaces.

![Banner showing LMA application areas](/assets/images/paper/large-multimodal-agents-survey/fig_p009_05.png)
*Figure 4: LMA application areas at a glance — GUI automation, robotics/embodied AI, gaming, autonomous driving, video understanding, visual generation & editing, complex visual reasoning, audio editing.*

![Embodied/robotics applications](/assets/images/paper/large-multimodal-agents-survey/fig_p009_01.png)
*Figure 5: Embodied / robotics applications — Minecraft and other open-world environments where JARVIS-1, MP5, and STEVE operate.*

![GUI automation](/assets/images/paper/large-multimodal-agents-survey/fig_p009_02.png)
*Figure 6: GUI automation — LMAs manipulating mobile and web interfaces the way a human would (Mobile-Agent, AppAgent, MM-Navigator).*

![Autonomous driving](/assets/images/paper/large-multimodal-agents-survey/fig_p009_07.png)
*Figure 7: Autonomous driving — multimodal perception with LLM-based motion planners (GPT-Driver, DLAH).*

### Evaluation scaffolding
- **Subjective** — versatility, user-friendliness, scalability, value & safety.
- **Objective** — task-specific metrics plus three emerging benchmarks: **SmartPlay** (game suite), **GAIA** (466 hand-crafted Q&A), **VisualWebArena** (real-web visual tasks with Set-of-Marks).

## Limitations

**Authors admit:**
- Evaluation is fragmented; no standard benchmark for LMAs (§5).
- Multimodal long memory is under-developed (§2).
- Multi-agent coordination protocols, communication standards, and task distribution are open problems (§7).
- Real-world deployment needs stronger HCI grounding (§7).

**Authors do not address:**
- **Safety, value alignment, prompt injection** — a one-paragraph mention; no taxonomy of attack surfaces (visual prompt injection, tool misuse, GUI hijacking).
- **Cost, latency, and reliability** of multi-step tool chains — critical for any production deployment.
- **Reproducibility / open-source coverage** is uneven; no comparison of license, code availability, or community support.
- **Cutoff is Feb 2024** — by Visual Intelligence 2025 publication time, OS-Copilot, Mobile-Agent v2, AppAgent v2, AutoGen, and most agentic-RL training (GUI-R1) were already published and absent.
- **No PRISMA-style coverage methodology**, so the survey functions more as a curated map than a systematic review.

## Why It Matters for Medical AI
The biggest gap for medical-AI readers is also the simplest to state: **the survey contains zero medical LMAs**. LLaVA-Med, ChatCAD, ChatCAD+, MMedAgent, MedRAX, SlideSeek, MDAgents — none appear in Table 1, none in Table 2's tool zoo. That is striking for a survey nominally covering through Feb 2024, since LLaVA-Med (2023) and ChatCAD (2023) predate the cutoff.

What is still useful:

- **The 4-type taxonomy slots medical agents cleanly.** Most current medical LMAs are **Type I** (closed-LLM planner with VFM tools, no long memory): ChatCAD, ChatCAD+, MMedAgent, MedRAX. MDAgents pushes into the **multi-agent** quadrant with role specialization. There is essentially **no Type IV** medical system in production — a concrete research opportunity, since longitudinal patient memory is exactly the regime where native multimodal long memory should pay off.
- **The perception/planning/action/memory decomposition is directly transferable.** When evaluating a new clinical agent, ask: which of the three perception styles? Which planning format (NL, code, hybrid)? Static or dynamic replanning? Tool / embodied / virtual action? Short or long memory?
- **The evaluation critique applies even more strongly in medicine.** General-purpose benchmarks (GAIA, SmartPlay, VisualWebArena) are domain-irrelevant; medical equivalents (AgentClinic, MedAgentBench, MedChain) exist but are similarly fragmented.

What is *missing* and matters more in clinical use:

- **Safety / value alignment / prompt-injection taxonomy** — the one-paragraph treatment here is wholly inadequate for medical deployment.
- **Cost and latency of multi-step tool chains** — clinical workflows have hard time budgets and audit requirements that the general-purpose LMA literature ignores.
- **Grounding and traceability** — medical agents need to expose evidence chains, not just outputs. The taxonomy has no axis for this.

Use this paper as a **vocabulary primer** when reading or writing about medical LMAs; do not use it as evidence for any quantitative claim, and treat its complete omission of medical work as a sign that the medical-LMA community must build its own taxonomy on top of this scaffold.

## References
- **Paper:** Xie, Chen, Zhang, Wan, Li. "Large Multimodal Agents: A Survey." arXiv:2402.15116v1, Feb 2024 (later cited as Visual Intelligence, Springer, 2025). [https://arxiv.org/abs/2402.15116](https://arxiv.org/abs/2402.15116)
- **Companion list:** [https://github.com/jun0wanan/awesome-large-multimodal-agents](https://github.com/jun0wanan/awesome-large-multimodal-agents)
- **Related agent surveys:** Wang et al. (2023) "A Survey on LLM-based Autonomous Agents"; Sumers et al. (2023) "Cognitive Architectures for Language Agents"; Xi et al. (2023) "The Rise and Potential of LLM-Based Agents."
- **Benchmarks named:** GAIA (Mialon et al. 2023); SmartPlay (Wu et al. 2023); VisualWebArena (Koh et al. 2024); WebLINX (Lù et al. 2024).
- **Key cited LMAs:** Visual ChatGPT, MM-ReAct, HuggingGPT, ViperGPT, VisProg, AssistGPT, Chameleon, Mobile-Agent, GPT4Tools, LLaVA-Plus, Tool-LMM, JARVIS-1, DoraemonGPT, ChatVideo, OS-Copilot, OpenAgents, AppAgent, MM-Navigator, GPT-Driver, MP5, MemoDroid, AVIS, DiscussNav.
- **Medical LMAs absent from the survey but worth pairing on this blog:** LLaVA-Med, ChatCAD, ChatCAD+, MMedAgent, MedRAX, SlideSeek, MDAgents.
