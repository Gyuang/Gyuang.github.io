---
title: "MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action"
excerpt: "A 2023 Microsoft technical report that wraps ChatGPT (gpt-3.5-turbo, 4K ctx) around Azure Cognitive Services vision experts via a ReAct-style text-only prompt protocol — zero training, zero quantitative metrics, ~30 hand-picked demos across 11 capability buckets."
categories:
  - Paper
  - LLM-Agents
  - LLM
tags:
  - MM-ReAct
  - ChatGPT
  - ReAct
  - Tool-Use
  - Multimodal-Agents
  - LLM-Agents
  - LLM
  - Azure-Cognitive-Services
  - LangChain
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/mm-react-multimodal-reasoning-action/
---

## TL;DR
- MM-ReAct **prompts ChatGPT (gpt-3.5-turbo, 4,096-token context, Azure-hosted) to act as a tool-using orchestrator** that calls Azure Cognitive Services vision experts — captioning, image tagging, dense captioning, face detection, celebrity recognition, OCR, receipt understanding — plus Bing search, PAL math, and X-Decoder image editing. Built on LangChain. **No fine-tuning, no training data, no vision tokens injected into the LLM.**
- The interface is a pure-text protocol: images become a `<ImagePath>` placeholder; ChatGPT emits a thought followed by an action prefixed with the watchword **"Assistant,"**; a regex dispatcher parses the watchword + expert name + path argument; tool outputs are text-serialized (bounding boxes as `<obj, x1, y1, x2, y2>` tuples) and appended to history; the loop ends when no "Assistant," watchword appears in a response.
- **There is no quantitative table in the paper.** The authors openly concede this in Limitation 1. The "experiments" are ~30 hand-picked demo images and videos spread across 11 capability categories (n=1 per category in most cases), and the PaLM-E head-to-head re-uses PaLM-E's own qualitative figures with no failure-case symmetry. **Concurrent systems Visual ChatGPT and HuggingGPT are cited but never compared.**

## Motivation
The CV ecosystem is fragmented — image tagging, celebrity ID, OCR, dense captioning, receipt parsing each ship as separate services, and humans manually chain them per use case. The competing joint-finetuning route (Flamingo, PaLM-E, KOSMOS-1) costs enormous compute and annotated data, locks the system to whatever capabilities were trained in, and is hard to upgrade module-by-module.

The authors argue ReAct (Yao 2022) — which already lets an LLM interleave reasoning with tool calls (search, calculator) — should extend trivially to vision if visual signals can be **named by path** and tools can be **described in the prompt prefix**. There is no medical-AI angle in this paper; all demos are natural-image / receipt / chart / video tasks. The medical relevance is indirect: the same agentic pattern is what later medical CAD agents (ChatCAD+, MMedAgent) borrow.

## Core Innovation
MM-ReAct is a *prompt-engineering recipe*, not a model:

1. **File-path placeholders as visual handles.** Non-text inputs never enter the LLM — only their disk paths do. The path is a black-box identifier ChatGPT learns to pass forward.
2. **The "Assistant," watchword.** A literal lexical marker that turns "free-form chain of thought" into "tool dispatch." A regex on the orchestration side matches the watchword + expert name + path argument and routes the call.
3. **Text-serialized observations.** Detection / dense-captioning outputs become `<object name, x1, y1, x2, y2>` tuples with a one-line legend; captions and OCR are dumped as plain strings. Everything stays inside the dialogue history.
4. **Prompt-prefix tool registry.** Each expert is described in the system message — name, capability, input format, output schema, in-context dispatch examples. New tools plug in by adding a description block plus a regex handler; no retraining.

Implementation: LangChain (Chase 2023); ChatGPT via Azure `gpt-3.5-turbo` (4,096-token limit); vision experts from Azure Cognitive Services Vision APIs. No gradient updates anywhere in the pipeline.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | Text-only prompts can compose vision experts into a multimodal reasoning system without joint fine-tuning. | Working end-to-end traces in Figures 3, 18–22, 26 show the loop closes and produces coherent answers on cherry-picked inputs. | ⭐⭐ |
| C2 | MM-ReAct achieves "competitive results to PaLM-E." | Side-by-side qualitative panels (Figures 15–17) on 5–6 example tasks; **re-uses PaLM-E's own figure images; no held-out set; no failure cases shown; no benchmark score (OK-VQA, VQAv2, NLVR2) reported**. | ⭐ |
| C3 | Zero-shot capability across a "wide range" of advanced understanding tasks (multi-image, multi-hop document, open-world, video). | One figure per capability (Figures 4–14). **n=1 per capability** in most cases (4 receipts, 2 video clips). | ⭐ |
| C4 | Upgrading the LLM (ChatGPT → GPT-4 language-only) improves results. | Figures 23–24: GPT-4 solves a physics derivation gpt-3.5 fails; HTML example. **2 examples total, no systematic comparison.** | ⭐ |
| C5 | The system is extensible to new tools without retraining. | Figure 25 adds X-Decoder image editing via a new prefix block. | ⭐⭐ |
| C6 | Vision experts are reliably selected by ChatGPT given the prompt protocol. | Traces in Figures 18–22 show correct expert choice on the shown queries. **Dispatch error rate is never measured.** | ⭐ |
| C7 | Specialized experts (celebrity, dense caption) give advantages over generalist VLMs. | Anecdotal — Kobe Bryant celebrity recognition trace looks nicer than a captioning-only model would. **No A/B against a generalist VLM on any benchmark.** | ⭐ |
| C8 | Tool-augmented prompting matches joint fine-tuning while being more flexible. | Indirect — relies on C1+C2+C5 together. | ⭐ |

**Honest read.** Treat this as a **systems / demo paper** and it is informative: the recipe runs end-to-end (C1), the prompt-engineering tricks are reusable, and the design is genuinely extensible (C5). Treat it as a scientific claim about capability and the evidence is thin: zero benchmark numbers, n=1 per capability, hand-picked PaLM-E comparisons with no failure-case symmetry, **and Visual ChatGPT / HuggingGPT — both concurrent, both also tool-orchestration via LLM — are cited but never compared**. The abstract's "Zero-shot experiments demonstrate MM-REACT's effectiveness" overstates what was actually demonstrated; "effectiveness was illustrated on curated demos" would be more accurate.

## Method & Architecture

![MM-ReAct teaser collage: ChatGPT orchestrates Azure vision experts to handle receipts, math, memes, video, and spatial reasoning](/assets/images/paper/mm-react/page_001.png)
*Figure 1: MM-ReAct dispatches specialized vision experts via ChatGPT to solve compositional visual tasks — receipts, visual math, memes, video summarization, and spatial reasoning — without any fine-tuning.*

![MM-ReAct system flowchart: user input goes to ChatGPT which emits thought plus Assistant-prefixed action; regex dispatches to a vision expert; tool output is text-serialized back into the loop](/assets/images/paper/mm-react/page_003.png)
*Figure 2: System flow. User input → ChatGPT thought + action request with the "Assistant," watchword → regex dispatch to a vision expert → text-serialized observation → loop until ChatGPT emits a final answer (no watchword in the last turn).*

![Full eight-step Kobe Bryant trace: ChatGPT interleaves four calls to image captioning, dense captioning, object tagging, face detection, celebrity recognition, and Bing search to identify the subjects](/assets/images/paper/mm-react/page_005.png)
*Figure 3: A worked execution trace on the basketball photo. Eight numbered steps interleave four ChatGPT calls with image captioning, dense captioning, object tagging, face detection, celebrity recognition, and Bing search — the single most informative figure in the paper.*

The loop in detail:

1. **User input formatting.** Text reaches ChatGPT directly; images/videos are saved to disk and the file path is inserted as a `<ImagePath>` placeholder.
2. **Prompt prefix.** Hand-written system message enumerates each vision expert by name, capability, input format, and output schema, with in-context dialogue examples. **This consumes the bulk of the 4,096-token context.**
3. **Thought + action.** ChatGPT emits a chain-of-thought line, then `Assistant, what objects do you see in this image? <ImagePath>`.
4. **Regex dispatch.** The orchestrator regex-matches the watchword, parses out the expert name and path, invokes the tool.
5. **Observation serialization.** Detections / dense captioning → `<object, x1, y1, x2, y2>` tuples with a one-line coordinate legend; captions and OCR → plain text. Appended to history.
6. **Loop.** ChatGPT either issues another thought/action (e.g. tag → celebrity → Bing search) or emits a final user-visible answer. **Termination = absence of the watchword.**
7. **Multi-image and video.** Multiple images arrive as separate turns each with `<ImagePathK>`. Video = sampled frames + per-frame captions tagged with timestamps; reasoning happens over the timestamped text.
8. **Extensibility / LLM swap.** Swapping `gpt-3.5-turbo` for language-only GPT-4 changes only the LLM slot. New tools plug in via a prefix description block + a regex handler.

## Experimental Results

**There is no quantitative table in the paper.** The authors explicitly concede this in Limitation 1: "hard to systematically evaluate the performance with concrete accuracy numbers, due to a lack of annotated benchmarks." The "experiments" are an inventory of capabilities, each shown via one or two cherry-picked figures:

| Capability | Figure(s) | Tools chained | Sample size |
|---|---|---|---|
| Visual math & text reasoning | 1, 4 | OCR → ChatGPT arithmetic | 1 example |
| Visual-conditioned joke / meme | 1, 5, 18 | Caption + dense caption + OCR → ChatGPT | 1 example |
| Spatial / coordinate understanding | 1, 6 | Dense caption (boxes) → ChatGPT relational reasoning | 1 example |
| Visual planning & prediction (recipes) | 1, 6 | Caption + OCR → ChatGPT | 1 example |
| Multi-image reasoning (receipts) | 1, 7 | Receipt expert × N → ChatGPT aggregation | 4 receipts |
| Multi-hop doc QA — bar chart | 1, 8, 19 | Caption + dense caption + OCR → ChatGPT arithmetic | 1 example |
| Multi-hop doc QA — floorplan / flowchart / table | 9, 10, 11, 20 | Caption + OCR → ChatGPT | 1 example each |
| Open-world concept understanding | 1, 12 | Tagging + celebrity + Bing search | "morel mushrooms", celebrities |
| Video summarization / event localization | 1, 13, 14 | Frame caption + timestamps → ChatGPT | 2 clips (BLT video, Kobe video) |
| **PaLM-E head-to-head** (jokes, multi-image, scene-text, embodied) | 15, 16, 17 | Various | Hand-picked match-up; **re-uses PaLM-E's own demo images** |
| LLM upgrade (ChatGPT → GPT-4 language-only) | 23, 24 | Same pipeline, swapped LLM | 1 physics + 1 HTML example |
| New tool plug-in (X-Decoder image editor) | 25 | Editing expert added to pool | 1 demo |

**Ablations: none in any conventional sense.** No controlled comparison of (a) with vs. without chain-of-thought prefix, (b) with vs. without in-context examples, (c) GPT-3.5 vs GPT-4 across a task set (only the single physics example), (d) one expert removed at a time, (e) regex dispatch vs. function-calling, (f) effect of prompt prefix length on the 4K context budget.

![Multi-image receipt aggregation demo: four receipts processed sequentially and totaled by ChatGPT](/assets/images/paper/mm-react/page_012.png)
*Figure 7: Multi-image reasoning across four receipt images for travel-cost aggregation — the best showcase of compositional reasoning in the paper. Each receipt enters as its own `<ImagePathK>`; the receipt-understanding expert returns structured fields; ChatGPT performs the sum.*

![Side-by-side PaLM-E qualitative comparison: MM-ReAct shown to win on the hand-picked example](/assets/images/paper/mm-react/fig_p020_01.png)
*Figures 15–17 (panel shown): Side-by-side qualitative comparison with PaLM-E (joint-finetuned). Caveat — the demo images are re-curated from PaLM-E's own qualitative figures, no failure cases are shown for either system, and no benchmark score is reported.*

## Limitations

**Authors admit (Section 4.6):**
1. No annotated benchmark exists for wild-recognition compositional tasks, so they cannot report accuracy.
2. Capability is upper-bounded by the integrated expert pool; missing experts → missing capability; broken experts → broken answers.
3. Expert descriptions live in the prompt prefix, so the number of usable experts is bounded by ChatGPT's 4,096-token context.
4. Text serialization of vision signals is lossy.
5. Manual prompt engineering is required; no automation.

**The paper does not address (analyst notes):**
- **No quantitative evaluation at all** — not even on existing benchmarks the tool set would suggest (OK-VQA, ChartQA, DocVQA, ScienceQA, NLVR2, MSR-VTT, EgoSchema).
- **No reliability / dispatch metrics**: how often does the LLM call the *wrong* expert? How often does the regex parse fail? How often is a non-existent expert hallucinated?
- **No cost / latency budget**: 3–4 expert calls plus 3–4 ChatGPT calls per query (per Figure 3) is non-trivial to deploy and never reported.
- **Error compounding**: when caption / OCR is wrong, downstream reasoning silently inherits the error. No failure-cascade analysis.
- **Multi-image scaling**: the receipt demo uses 4 images; what happens at 20? At 100? Context saturates fast.
- **Video handling is shallow**: frames + timestamped captions miss audio, miss fine motion, and the BLT step-localization example may be overfit to a tutorial format with explicit narration.
- **Concurrent peers absent.** Visual ChatGPT (Wu et al., 2023) and HuggingGPT (Shen et al., 2023) — both also tool-orchestration via LLM — are cited but never compared. Only PaLM-E (joint-finetuned) is contrasted, and on PaLM-E's own home turf.
- **Reproducibility**: Azure Cognitive Services APIs evolve and are paid; the celebrity recognition model in particular has restricted access. The system as described cannot be reproduced openly.
- **Bias / safety**: celebrity recognition, receipt OCR with names and amounts, face detection — none discussed.

## Why It Matters for Medical AI
MM-ReAct has no medical content of its own — there are no demos on medical imaging, microscopy, satellite, or non-English OCR. The interest for medical AI is **architectural**: the same ReAct-with-vision-tools recipe is what later medical CAD agents (ChatCAD+, MMedAgent, agentic neuroimaging pipelines) inherit when they wire a generalist LLM around specialist segmentation / classification / report-drafting tools. If you are designing a medical agent, the durable lessons here are (i) **text-serialize specialist outputs aggressively** (boxes → tuples, segmentations → coordinate lists, lab values → strings) so the LLM can carry them across turns, (ii) **budget your 4K (or 32K, or 128K) context for tool descriptions first** because the registry dominates the prompt, and (iii) **expect the dispatch reliability question that MM-ReAct ducks to be the first thing a clinical reviewer asks** — measure it before you publish.

## References
- Paper: [MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action (arXiv:2303.11381)](https://arxiv.org/abs/2303.11381)
- Project page / code / demo / video: [multimodal-react.github.io](https://multimodal-react.github.io/)
- Related — ReAct (Yao et al., 2022): [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Related — Visual ChatGPT (Wu et al., 2023): [arXiv:2303.04671](https://arxiv.org/abs/2303.04671) (cited but never compared in MM-ReAct)
- Related — HuggingGPT (Shen et al., 2023): [arXiv:2303.17580](https://arxiv.org/abs/2303.17580) (cited but never compared in MM-ReAct)
- Related — PaLM-E (Driess et al., 2023): [arXiv:2303.03378](https://arxiv.org/abs/2303.03378) (the sole qualitative comparator)
- LangChain orchestration framework: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
