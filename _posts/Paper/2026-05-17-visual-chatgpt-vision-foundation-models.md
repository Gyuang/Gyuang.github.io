---
title: "Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models"
excerpt: "A frozen text-davinci-003 orchestrates 22 Visual Foundation Models via a Prompt Manager — a four-field tool schema, chained-UUID filenames, and a forced 'Thought:' suffix driving a ReAct loop — with zero quantitative benchmarks and only curated qualitative case studies."
categories:
  - Paper
  - LLM-Agents
  - LLM
permalink: /paper/visual-chatgpt-vision-foundation-models/
tags:
  - Visual-ChatGPT
  - Prompt-Manager
  - Tool-Use
  - ReAct
  - Vision-Foundation-Models
  - LangChain
  - ChatGPT
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- Visual ChatGPT keeps `text-davinci-003` frozen as the orchestrator and routes user requests to **22 off-the-shelf Visual Foundation Models (VFMs)** through a **Prompt Manager** that stringifies every visual signal into LLM-readable text. No fine-tuning, no joint training.
- The contribution is engineering schema, not learning: (a) a four-field VFM description (`Name / Usage / Inputs-Outputs / Example`), (b) a chained UUID filename convention (`{Name}_{Operation}_{Prev_Name}_{Org_Name}.png`) that lets the LLM trace image provenance from the path string, and (c) a forced "Thought: Do I need to use a tool?" suffix that anchors a ReAct-style reasoning loop.
- **Headline result is "the system runs."** Section 4 is titled "Experiments" but contains **zero benchmark numbers** — no success rate, no baseline comparison, no latency/cost. Evidence is one 16-round dialogue (Fig. 4) plus single curated failure-vs-success pairs for prompt ablations (Figs. 5-7). Deployment: 4x V100, 2k-token history window.

## Motivation

ChatGPT shipped in late 2022 with strong multi-turn language reasoning but zero visual I/O. Meanwhile a public zoo of strong but single-task VFMs (BLIP, Stable Diffusion, ControlNet, Pix2Pix, depth/edge/seg/pose detectors) was already mature. The authors reject the alternative of "train a giant native multimodal model" on cost and extensibility grounds — every new modality would demand retraining — and instead ask whether **prompting alone** can compose frozen experts into a conversational image-editing assistant.

The framing is general-purpose; medical imaging is not addressed in this paper. But the agentic tool-routing template laid down here became the scaffold for later systems like ChatCAD+ and HuggingGPT-style medical assistants, which is why it earns a place in a medical-AI reading list even though no medical claim is made.

## Core Innovation

- **Prompt Manager $\mathcal{M}$.** A structured prompt-engineering layer that stringifies six things into one LLM prompt every turn: system principle $P$, the VFM registry $\mathcal{F}$, dialogue history $H_{<i}$, the user query $Q_i$, the running reasoning history $R_i^{(<j)}$, and the most recent tool output $\mathcal{F}(A_i^{(j)})$.
- **Four-field VFM schema.** Each of the 22 tools is registered with `Name` (entry token, e.g. *Answer Question About The Image*), `Usage` (when to call it), `Inputs/Outputs` (call signature as text), and an optional `Example`. The ablation in Fig. 6 shows `Example` is often skippable while the other three are mandatory.
- **Chained UUID filenames.** Every new image is named `{Name}_{Operation}_{Prev_Name}_{Org_Name}.png` so the LLM can read provenance directly from the path token. The Fig. 7 demo where ChatGPT articulates the naming rule unprompted is the paper's most convincing single piece of evidence.
- **Forced "Thought:" suffix.** Every user query and every tool observation is suffixed with "Thought: Do I need to use a tool?" — this blocks hallucinated image descriptions and gives the regex parser a deterministic anchor. Loop terminates on "Thought: Do I need to use a tool? No".

![Visual ChatGPT architecture overview](/assets/images/paper/visual-chatgpt/page_001.png)
*Figure 1: Visual ChatGPT routes a user's compositional request ("generate a red flower conditioned on depth, then cartoonize it") through ChatGPT, which iteratively invokes VFMs via the Prompt Manager and chains intermediate images together.*

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Prompt Manager lets ChatGPT orchestrate 22 VFMs in zero-shot for generation, editing, and understanding | Fig. 4 16-round dialogue + assorted single cases. No success rate over a benchmark. | ⭐ |
| C2 | Each of the six System Principle clauses (filename sensitivity, format strictness, reliability, CoT, role, accessibility) is necessary | Fig. 5: one matched-pair qualitative example per component (only 4 of 6 shown). Cherry-picked. | ⭐ |
| C3 | Of `Name / Usage / Inputs-Outputs / Example`, only `Example` is optional | Fig. 6: one paired case per field. ChatGPT recovers from missing `Example` by summarizing history. | ⭐ |
| C4 | Chained UUID filenames let ChatGPT track provenance and infer the naming rule | Fig. 7 bottom-left: the model verbalizes the naming convention unprompted. One case, but unusually convincing. | ⭐⭐ |
| C5 | The system can ask clarifying questions on ambiguous queries (e.g. "replace the cup" with multiple cups) | Fig. 7 bottom-right: a single staged dialogue. No false-positive rate. | ⭐ |
| C6 | Multi-step compositional commands are auto-decomposed into VFM calls | Fig. 1 teaser + Fig. 4 depth-conditioned generation thread. Two instances. | ⭐ |
| C7 | Architecture is extensible — adding a VFM only requires writing a prompt block | Architectural assertion; not demonstrated with an add-a-tool case. | ⭐ |
| C8 | Reliability prompt prevents hallucinated image content | Fig. 5 bottom-left: one paired case (sunflower description). Single anecdote. | ⭐ |

**Honest assessment.** This is a **system / recipe paper that frames itself as experimental**. Section 4 ("Experiments") contains zero benchmarks — no success-rate counts even on the authors' own task suite, no comparison against a no-Prompt-Manager baseline, no comparison against concurrent tool-using LLMs (HuggingGPT, MM-ReAct, ViperGPT), no latency/cost numbers, no variance, no statistical tests. Every ablation in Figs. 5-7 is a single curated pair. The genuine contribution is the **engineering recipe** — the four-field VFM schema and chained-filename convention have proven influential in later agent frameworks — but the paper itself supplies almost no evidence of *how often* the system works on uncurated inputs. The only claim that rises above one-star is **C4**: the model articulating its own filename convention when asked is the rare ablation that constrains the explanation space.

## Method & Architecture

![Prompt Manager schematic](/assets/images/paper/visual-chatgpt/page_004.png)
*Figure 2: Prompt Manager (Fig. 3 in the paper). Converts system principles, VFM descriptions (Name / Usage / Inputs-Outputs / Example), dialogue history, user query, reasoning history, and tool outputs into a single LLM-readable prompt every turn.*

Visual ChatGPT defines a dialogue $S = \{(Q_1, A_1), \dots, (Q_N, A_N)\}$. Round $i$ may invoke multiple VFMs, producing intermediate answers $A_i^{(j)}$. The core recurrence (Eq. 1) is:

$$A_i^{(j+1)} = \text{ChatGPT}\big(\,\mathcal{M}(P),\, \mathcal{M}(\mathcal{F}),\, \mathcal{M}(H_{<i}),\, \mathcal{M}(Q_i),\, \mathcal{M}(R_i^{(<j)}),\, \mathcal{M}(\mathcal{F}(A_i^{(j)}))\,\big)$$

1. **System Principle $\mathcal{M}(P)$.** Six fixed clauses injected at the top of every turn: role declaration, VFM accessibility list, filename sensitivity (never fabricate file names), CoT encouragement, reasoning-format strictness (parseable by regex), and reliability (be loyal to tool observations).
2. **VFM registration $\mathcal{M}(\mathcal{F})$.** Each of the 22 tools is registered with the four-field schema above.
3. **User query handling $\mathcal{M}(Q_i)$.** Uploaded images get `image/{uuid}.png`; a fake QA pair is injected stating "image received" so subsequent rounds can reference it without re-feeding pixels. Every $Q_i$ is suffixed with the forced "Thought:" prompt.
4. **Intermediate output handling $\mathcal{M}(\mathcal{F}(A_i^{(j)}))$.** Every new image is named `{Name}_{Operation}_{Prev_Name}_{Org_Name}.png` (e.g. `ui3c_edge-of_o0ec_nji9dcgf.png` = canny edge of intermediate `o0ec`, original upload `nji9dcgf`). "Thought:" is re-appended so the LLM decides whether the chain is finished. When the query is ambiguous, the system is prompted to ask a clarifying question instead of guessing.
5. **Reasoning loop.** ReAct-style `Thought / Action / Action Input / Observation` template, parsed by elaborate regex; loop terminates on "Thought: Do I need to use a tool? No".
6. **Implementation.** LLM = OpenAI `text-davinci-003` via LangChain; VFMs from HuggingFace `diffusers`, Maskformer, and ControlNet repos; **4x Nvidia V100** for full deployment; chat history truncated at 2,000 tokens.

![Three-round dialogue with iterative VFM invocation flowchart](/assets/images/paper/visual-chatgpt/page_003.png)
*Figure 3: System overview (Fig. 2 in the paper). A three-round dialogue example with the iterative VFM-invocation flowchart and a detailed trace of one reasoning step.*

![Table of 22 supported VFMs](/assets/images/paper/visual-chatgpt/page_005.png)
*Figure 4: Table 1. The 22 Visual Foundation Models supported, spanning Stable Diffusion (text↔image), Pix2Pix and Maskformer-based editing (remove/replace), BLIP/BLIP-2 (VQA/captioning), and 8 ControlNet condition pairs (Edge/Line/Hed/Seg/Depth/NormalMap/Sketch/Pose) in both `Image-to-X` and `X-to-Image` directions.*

## Experimental Results

There is no main quantitative table. The evidence is exclusively qualitative.

| Evidence artifact | What it shows | Type | Quantification |
|---|---|---|---|
| Fig. 4 | 16-round dialogue: sketch → scribble2image → pix2pix watercolor → VQA color → remove apple → replace table → remove cup → replace background → depth → depth2image; separate generate-puppy-on-beach → replace-with-kitten → pencil-drawing thread | Capability demo | None |
| Fig. 5 | Ablation of $\mathcal{M}(P)$: removing Filename Sensitivity / Format Strictness / Reliability / CoT each breaks a specific failure mode | Qualitative ablation, 4 paired cases | None |
| Fig. 6 | Ablation of $\mathcal{M}(\mathcal{F})$: removing Name / Usage / Inputs-Outputs / Example, with Example noted as optional | Qualitative ablation, 4 paired cases | None |
| Fig. 7 | Ablation of $\mathcal{M}(Q_i)$ and $\mathcal{M}(\mathcal{F}(A_i^{(j)}))$: Unique Filename / Force VFM Thinking / Chained Filename / Ask for More Details | Qualitative ablation, 4 paired cases | None |
| §4.1 | Deployment configuration | Engineering fact | **22 VFMs, 4x V100, 2,000-token history** |

The most concrete operational fact in the paper is the deployment configuration. The §4.3 case-study text gives semi-quantitative failure characterizations ("Visual ChatGPT will guess [the tool name] many times until it finds an existing VFM, or encounters an error") but no success-rate counts.

![16-round multimodal dialogue](/assets/images/paper/visual-chatgpt/page_007.png)
*Figure 5: The 16-round multimodal dialogue (Fig. 4 in the paper) demonstrating generation, editing, VQA, and multi-step composition. This is the qualitative centerpiece of the paper.*

![System-principle ablations](/assets/images/paper/visual-chatgpt/page_008.png)
*Figure 6: System-principle ablations (Fig. 5). Removing each clause causes a specific failure mode: filename confusion, parser breakage, hallucinated content, or chain-of-thought collapse. One paired case per component.*

![VFM-description ablations](/assets/images/paper/visual-chatgpt/page_009.png)
*Figure 7: VFM-description ablations (Fig. 6). Among Name / Usage / Inputs-Outputs / Example, only Example is shown to be optional — ChatGPT can recover from a missing example by summarizing dialogue history.*

![User-query and output-handling ablations](/assets/images/paper/visual-chatgpt/page_010.png)
*Figure 8: Query/output ablations (Fig. 7). Unique filenames, the forced-VFM-thinking suffix, chained filenames, and clarifying-question prompts each prevent a distinct class of failure. The bottom-left panel — ChatGPT articulating the chained-filename rule unprompted — is the paper's single ⭐⭐ result.*

## Limitations

**Authors acknowledge (§5-6):**

- Heavy dependence on ChatGPT and on individual VFM quality — the orchestrator inherits every tool's failure modes.
- Heavy prompt engineering required, demanding both CV and NLP expertise.
- Limited real-time capability — multi-VFM chains are slow versus single specialist models.
- Token-length ceiling caps the number of VFMs that can be in-context-described; a pre-filter / retrieval router would be needed at scale.
- Security / privacy of plug-and-play foundation models, especially via remote APIs.
- No self-correction module — generation failures are not detected or repaired.

**Not addressed:**

- No success/failure rate on any held-out suite — readers cannot tell whether the system works 30% or 90% of the time on representative queries.
- No comparison to concurrent systems (HuggingGPT was on arXiv ~2 weeks later; ViperGPT and MM-ReAct were near-contemporaneous), so the Prompt Manager's relative contribution is not isolatable.
- No analysis of error propagation — when VFM #1 produces a wrong intermediate image, does ChatGPT recover or compound the error? Fig. 4 only shows successful runs.
- No measurement of how filename-sensitivity holds over long dialogues (the 2k-token truncation will silently drop older filenames).
- No medical / scientific imaging case despite the natural fit; subsequent work (ChatCAD+, BrainAdapter) had to do this lift separately.
- API/model drift: `text-davinci-003` was retired in 2024 — the reported behavior is no longer reproducible without prompt re-tuning for a different backbone.
- No $/dialogue figure despite obvious cost concerns.
- No regex-parser ablation — how brittle is the system when ChatGPT phrases "Thought" slightly differently?

## Why It Matters for Medical AI

Visual ChatGPT itself has no medical content, but the engineering schema it codifies — the four-field tool description, the forced-thought ReAct loop, the chained-filename convention — became the template downstream medical agent systems (ChatCAD+, HuggingGPT-medical variants, brain-imaging assistants) adopted almost verbatim. For medical-AI builders, the practical takeaway is twofold:

1. **The schema is reusable; the evidence is not.** The four-field VFM registry and chained-UUID provenance trick are sound engineering patterns that transfer to medical tool registries (segmentation models, registration models, report generators). But the qualitative-only evaluation style this paper normalized has been a recurring weakness in medical agent papers — readers should demand the success-rate measurements Visual ChatGPT itself never provided.
2. **The brittle parts transfer too.** A regex-anchored ReAct loop over a 2k-token window with no self-correction module is a stack of single points of failure. Any medical deployment that reuses this template needs to address error propagation, parser fragility, and tool-failure recovery — none of which Visual ChatGPT solves.

## References

- Paper: [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models (arXiv:2303.04671)](https://arxiv.org/abs/2303.04671)
- Code: [microsoft/visual-chatgpt](https://github.com/microsoft/visual-chatgpt)
- Related work: HuggingGPT (Shen et al., 2023), MM-ReAct (Yang et al., 2023), ViperGPT (Surís et al., 2023), ReAct (Yao et al., 2023), LangChain.
- Downstream medical follow-ups: ChatCAD+ (Zhao et al., 2024), BrainAdapter — agentic templates that reused the Prompt-Manager pattern in clinical contexts.
