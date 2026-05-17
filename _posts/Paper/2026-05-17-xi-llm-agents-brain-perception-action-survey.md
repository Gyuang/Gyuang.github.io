---
title: "The Rise and Potential of Large Language Model Based Agents: A Survey"
excerpt: "An 86-page Fudan NLP survey that relabels Russell-Norvig's sense-decide-act as Brain-Perception-Action and uses it to catalogue ~680 LLM-agent papers up to Sep 2023 — influential as a vocabulary, weak as an evaluation."
categories:
  - Paper
  - LLM-Agents
tags:
  - LLM-Agents
  - Survey
  - Taxonomy
  - Multi-Agent
  - Agent-Society
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/xi-llm-agents-brain-perception-action-survey/
---

![The envisioned LLM-agent society](/assets/images/paper/xi-llm-agents-survey/fig_p005_02.png)
*Figure 1 (Xi et al., 2023): The envisioned LLM-agent society — agents cook, perform, plan, trade, and collaborate with humans across a shared sandbox.*

## TL;DR
- The paper organizes the explosion of 2022-2023 LLM-agent work under a **Brain-Perception-Action (BPA)** trichotomy and uses it as the spine for a typology that covers single-agent, multi-agent, human-agent, and emergent "agent society" systems.
- The headline contribution is **lexical and organizational**: BPA is an acknowledged relabeling of Russell & Norvig's sense-decide-act, and the value lies in consolidating **~680 cited works** into one ontology with three typology trees (Figs. 3/4/5) and a 4-property society model (open, persistent, situated, organized).
- There is **no headline empirical metric** — no benchmark is run, no head-to-head comparison is reported, and the proposed evaluation dimensions in §6.2 are never executed. Treat the paper as a 2023 snapshot vocabulary document, not an evaluation.

## Motivation
By mid-2023, AutoGPT, BabyAGI, Voyager, Generative Agents, CAMEL, MetaGPT, ChatDev, AgentVerse, HuggingGPT, and ReAct-style harnesses had all surfaced within roughly six months. Each used "agent" to mean something different (planner+executor, role-played persona, ReAct loop, code-generating tool-user, simulacrum) and reused vocabulary inconsistently. The Fudan NLP group's pitch is that classical AI already settled this 30 years ago: an agent perceives, decides, and acts (Russell & Norvig), and the four classical agent properties — autonomy, reactivity, pro-activeness, and social ability (Wooldridge & Jennings, 1995) — are exactly what makes LLMs a natural fit for the "decide" slot.

Medical AI is barely addressed. Healthcare appears only as a passing example of a domain where trustworthiness matters and where human-agent partnership is useful. The relevance for medical readers is downstream: every subsequent medical-agent paper (MedAgents, MDAgents, AgentClinic, MMedAgent, CT-Agent, AutoRG-Brain, VoxelPrompt) borrows this paper's vocabulary, so the framework is worth knowing even if the source survey runs no clinical experiments.

## Core Innovation
The "method" is the taxonomy itself. There is no algorithm, no loss, no training procedure — only a labeled tree.

1. **Brain (§3.1)** — LLM-as-controller, decomposed into natural-language interaction, knowledge (parametric / commonsense / domain; editing and unlearning), memory (short-term context + external long-term with read/write/reflection), reasoning & planning (CoT, ToT/GoT, plan-and-solve, ReAct, Reflexion; with-feedback vs. without-feedback), and transferability/generalization (zero-/few-shot, continual learning, Voyager-style skill library).
2. **Perception (§3.2)** — input expansion beyond text. Visual via ViT/VQVAE encoders + Q-Former (BLIP-2, InstructBLIP, Flamingo) or projection layers (LLaVA, MiniGPT-4, PandaGPT); auditory via cascading (AudioGPT, HuggingGPT) or AST spectrogram-as-image; "other" (point/gesture in InternGPT, LiDAR, GPS, IMU, BCI).
3. **Action (§3.3)** — output expansion: textual output; tool use along three axes (learning tools — Toolformer/TALM; using tools — WebGPT/HuggingGPT/Visual ChatGPT/SayCan; making tools — LATM/CREATOR/SELF-DEBUGGING); embodied action (SayCan, EmbodiedGPT, PaLM-E, Voyager, AlphaBlock, LM-Nav, NavGPT).
4. **Single-agent deployment (§4.1)** — task-oriented (Mind2Web/WebGum, ALFWorld), innovation-oriented (Boiko et al. chemistry assistant), and lifecycle-oriented (Voyager / Minecraft survival).
5. **Multi-agent (§4.2)** — cooperative-disordered (ChatLLM-Net majority vote), cooperative-ordered (CAMEL, MetaGPT-as-waterfall, AgentVerse), and adversarial (Du et al. debate, ChatEval). The paper flags hallucination amplification under prolonged debate as a real failure mode.
6. **Human-agent (§4.3)** — instructor-executor (RLHF-style human-as-overseer) vs. equal-partnership (empathic collaborator, e.g., mental-wellness chatbots).
7. **Agent society (§5)** — four properties (open, persistent, situated, organized); behavior framed as positive / neutral / negative; personality measured via Big Five and MBTI; environments classified as text-based, virtual sandbox (Generative Agents, AgentSims, Minecraft), or physical.
8. **Open problems (§6.5)** — AGI debate (the paper does not commit), sim-to-real, collective intelligence, and Agent-as-a-Service (AaaS / LLMAaaS).

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | LLMs are "highly suited" to serve as the brain/controller of an AI agent because they satisfy autonomy, reactivity, pro-activeness, and social ability. | §2.3 — argument-by-property-matching against Wooldridge & Jennings (1995); cites GPT-4, FLAN, T0, ICL literature. No experiment. | ⭐ |
| C2 | The Brain-Perception-Action decomposition is a "general framework" suitable for diverse applications. | §3 — demonstrated by *fitting* ~680 papers into the three buckets. No falsification test, no example of a system that *fails* to fit. | ⭐ |
| C3 | LLM-based agents can exhibit reasoning/planning comparable to symbolic agents. | §2.2 — cites CoT and task-decomposition papers but does not benchmark LLM-agent vs. symbolic-agent on any shared task. | ⭐ |
| C4 | Multi-agent systems improve task quality through division of labor (Adam-Smith analogy). | §4.2 — cites MetaGPT, ChatDev, AgentVerse; no controlled ablation showing multi-agent > single-agent at matched inference cost. | ⭐⭐ |
| C5 | Simulated agent societies exhibit emergent social phenomena (relationship formation, propagation, organized cooperation). | §5.3 — cites Generative Agents (Park et al. 2023), S3, MetaGPT; described qualitatively. | ⭐⭐ |
| C6 | LLM-based agents are a "potential path to AGI." | §6.5 — explicitly framed as a proponents-vs-opponents debate; the paper does not commit. | ⭐ |
| C7 | Adversarial robustness, hallucination, bias, misuse, unemployment, and existential risks are real concerns. | §6.3 — well-sourced restatement of LLM-safety literature; no new attack/defense experiments specific to *agentic* deployment. | ⭐⭐ |
| C8 | The survey is "comprehensive." | Asserted in abstract/§1/§7; supported by length (86 pp.) and ~680 references. No PRISMA, no coverage methodology. | ⭐ |

**Honest assessment.** The Brain-Perception-Action framework is **analytically weak**: it is an acknowledged relabeling of sense-decide-act, and the paper itself notes the correspondence. The contribution is organizational/lexical rather than predictive — every system fits somewhere, which is exactly the standard critique of any tripartite taxonomy. What is genuinely useful is (a) the consolidation of the 2022-2023 explosion into one bibliography, (b) the typology trees in Figs. 3/4/5 which downstream papers reproduce, (c) the multi-agent taxonomy (cooperative-disordered/ordered vs. adversarial; instructor-executor vs. equal-partnership), and (d) the four-property society model. Everything else is aggregation of well-known systems (CoT, ReAct, Voyager, CAMEL, MetaGPT, Generative Agents, Toolformer, PaLM-E).

## Method & Architecture

![Page render of the Brain-Perception-Action master figure (Figure 2)](/assets/images/paper/xi-llm-agents-survey/page_010.png)
*Figure 2 (Xi et al., 2023): The proposed Brain-Perception-Action framework for LLM-based agents — perception expands inputs beyond text, the LLM brain handles knowledge/memory/reasoning/planning, and the action module externalizes via text, tools, or embodied control.*

The typology trees (Figs. 3, 4, 5 in the paper) are pure vector text that does not extract as PNG; the readable form is a nested list.

**Brain typology (Fig. 3 in the paper).**
- *Natural-language interaction* — multi-turn dialogue, multilingual, implicature handling.
- *Knowledge* — parametric (pretrained), commonsense, domain; editing and unlearning operations.
- *Memory* — short-term (context window) and long-term (vector stores, summary-based, episodic replay, RAG); read/write/reflection are explicit operations.
- *Reasoning & planning* — CoT, ToT, GoT, task decomposition, plan-and-solve, ReAct, Reflexion; categorized as *with-feedback* vs. *without-feedback*.
- *Transferability & generalization* — zero-/few-shot ICL, continual learning, catastrophic-forgetting mitigations (Voyager skill library).

**Perception typology (Fig. 4 in the paper).**
- *Textual* — the default.
- *Visual* — ViT/VQVAE encoders combined with Q-Former (BLIP-2, InstructBLIP, Flamingo) or projection layer (LLaVA, MiniGPT-4, PandaGPT).
- *Auditory* — cascading APIs (AudioGPT, HuggingGPT) or AST-style spectrogram-as-image.
- *Other* — point/gesture (InternGPT), LiDAR, GPS, IMU, BCI.

**Action typology (Fig. 5 in the paper).**
- *Textual output* — the default chat surface.
- *Tool use* — learning tools (Toolformer, TALM), using tools (WebGPT, HuggingGPT, Visual ChatGPT, SayCan), making tools (LATM, CREATOR, SELF-DEBUGGING).
- *Embodied action* — SayCan, EmbodiedGPT, PaLM-E, Voyager, AlphaBlock, LM-Nav, NavGPT.

**Multi-agent and society axes.**
- *Cooperative-disordered* — independent agents vote / aggregate (ChatLLM-Net).
- *Cooperative-ordered* — pipeline / waterfall roles (CAMEL, MetaGPT, AgentVerse).
- *Adversarial* — debate convergence (Du et al., ChatEval), with hallucination amplification flagged.
- *Human-agent instructor-executor* — humans give quantitative or qualitative feedback.
- *Human-agent equal-partnership* — agents as empathic collaborators.
- *Agent society four properties* — open, persistent, situated, organized.

## Experimental Results

**There are no experiments.** No table reports method × metric × dataset. §6.2 *proposes* four evaluation dimensions (utility, sociability, values, ability-to-evolve) and namechecks AgentBench, but the survey runs none of them and provides no leaderboard.

| Artefact | What it actually is |
|----------|---------------------|
| ~680 references | Cited works through ~Sep 2023 (companion GitHub list `WooooDyy/LLM-Agent-Paper-List`) |
| Fig. 1 (hero) | Pixel-art "envisioned agent society" — rhetorical opener |
| Fig. 2 | BPA master architecture diagram |
| Figs. 3 / 4 / 5 | Typology trees for Brain / Perception / Action |
| Figs. 7-12 | Scenario diagrams for single-agent, multi-agent (cooperative vs. adversarial), human-agent (instructor-executor vs. equal-partnership), and agent society |
| §6.2 | Proposed (but unexecuted) evaluation dimensions |

The qualitative findings worth flagging:

- **Hallucination amplification in long multi-agent debates** is identified as a real failure mode (Du et al. on debate converging to wrong consensus; MetaGPT internal observations).
- **Destructive emergent behavior** — agents destroying each other or the environment in pursuit of efficiency — is noted in AgentVerse.
- **The "Sydney"/FreeSydney petition** is used as the case study for emotional over-attachment and addictiveness risk.

## Limitations

Author-acknowledged (§6.5):

- LLM-agent → AGI is contested; the paper deliberately does not commit.
- Sim-to-real gap is large; physical-environment deployment faces hardware, generalization, and safety challenges.
- Collective intelligence is not guaranteed by scaling agent counts (groupthink, hallucination amplification).
- Agent-as-a-Service raises privacy, controllability, and migration challenges.

What the paper does not address adequately:

- **No coverage methodology.** No PRISMA-style flow diagram, no inclusion/exclusion criteria, no statement of how many candidate papers were screened.
- **No agent-system benchmarking.** §6.2 proposes evaluation dimensions but runs no head-to-head comparison.
- **Construct validity of personality measurement.** Applying Big Five / MBTI to LLM outputs is presented without caveats; later work shows these instruments are unreliable under prompt perturbation.
- **Cost analysis is absent.** Multi-agent systems are championed for "division of labor," but the inference-cost multiplier vs. single-agent CoT is never quantified.
- **Cutoff is ~Sep 2023.** Pre-dates AutoGen, LangGraph, CrewAI, GPT-4o, Devin-style coding agents, and benchmarks like SWE-bench-Agent, GAIA, AgentBoard, OSWorld.
- **No falsifiable predictions.** The framework cannot be wrong — every system fits somewhere.

## Why It Matters for Medical AI

The paper itself does very little medical work: healthcare surfaces only as a passing example of where trustworthiness matters and human-agent partnership is helpful. What matters is downstream uptake — the BPA vocabulary and the multi-agent taxonomy (instructor-executor vs. equal-partnership; cooperative-disordered vs. cooperative-ordered vs. adversarial) have been adopted, often without modification, by the medical-agent papers indexed elsewhere in this blog: MedAgents, MDAgents, AgentClinic, MMedAgent, CT-Agent, AutoRG-Brain, VoxelPrompt, MARCH, MedRAX, the Baymax survey. Reading this paper is useful as a lexicon source; reading it for clinical guidance is not.

The follow-on medical surveys had to *extend* Xi et al.'s framework with domain-specific roles (specialist, generalist, moderator, patient-simulator) and clinical-workflow paradigms (sequential task chain, collaborative experts, iterative evolution) that this 2023 paper does not anticipate.

## References

- Paper: Xi, Z. *et al.* (2023). *The Rise and Potential of Large Language Model Based Agents: A Survey.* arXiv:2309.07864v3. <https://arxiv.org/abs/2309.07864>
- Companion paper list: `WooooDyy/LLM-Agent-Paper-List` — <https://github.com/WooooDyy/LLM-Agent-Paper-List>
- Foundational antecedents: Russell & Norvig (sense-decide-act), Wooldridge & Jennings (1995, intelligent-agent properties).
- Representative systems referenced in the framework: AutoGPT, BabyAGI, Voyager, Generative Agents (Park et al. 2023), CAMEL, MetaGPT, ChatDev, AgentVerse, HuggingGPT, Toolformer, ReAct, Reflexion, PaLM-E, SayCan, EmbodiedGPT.
- Downstream medical extensions (this blog): MedAgents, MDAgents, AgentClinic, MMedAgent, CT-Agent, AutoRG-Brain, MARCH, MedRAX, and the Med-LLM-Agents "Baymax" survey.
