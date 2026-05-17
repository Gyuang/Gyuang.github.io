---
title: "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face"
excerpt: "A four-stage LLM-as-controller pipeline reaches 63.08% end-to-end human success and 52.62 single-task Acc using prompts alone — no fine-tuning, no training."
categories:
  - Paper
tags:
  - HuggingGPT
  - JARVIS
  - LLM-Agents
  - LLM
  - Tool-Use
  - Multimodal-Orchestration
permalink: /paper/hugginggpt-jarvis-task-orchestration/
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

![HuggingGPT overview](/assets/images/paper/hugginggpt/page_002.png)
*Figure 1: ChatGPT as a controller routing a multimodal user request to expert Hugging Face models (DETR, ViT, ControlNet, ViT-GPT2 captioner).*

## TL;DR

- **Single LLMs cannot natively perceive images/audio/video or orchestrate multi-step multimodal pipelines.** HuggingGPT (NeurIPS 2023) wires ChatGPT to the Hugging Face Hub as a prompt-only controller — no training, no fine-tuning.
- **A four-stage prompt pipeline — Task Planning → Model Selection → Task Execution → Response Generation — emits JSON tasks `{task, id, dep, args}` with a `<resource>-task_id` symbol that rewires upstream outputs into downstream arguments at launch time.** Model Selection shortlists candidates by **Hugging Face download count** and asks the LLM a single-choice question.
- **Headline numbers (GPT-3.5 controller):** **52.62 / 51.92 / 50.48** on single / sequential / graph planning splits, with a **63.08% end-to-end human success rate** across 130 requests. GPT-4 hits **41.36 Acc (sequential)** and **58.33 Acc (graph)** on the 46-example human-annotated set — dwarfing Alpaca / Vicuna baselines but still leaving more than 1 in 3 requests failing end-to-end.

## Motivation

LLMs are confined to text I/O; real user requests routinely span vision, speech, and video and require multiple expert models scheduled with dependencies. Prior tool-using systems either hard-wire a small foundation-model set (Visual ChatGPT, TaskMatrix) or rely on code-generation backbones (ViperGPT, Visual Programming). HuggingGPT instead treats the **entire Hugging Face Hub as a pluggable model registry**, using natural-language model cards as the interface and the LLM as the language-level "bus." There is no medical evaluation in the paper — the agent pattern is general-purpose.

## Core Innovation

Three ideas do the load-bearing work:

1. **A JSON task schema with explicit dependency wiring.** Each parsed task is `{"task": <type>, "id": <int>, "dep": [<prereq_ids>], "args": {"text": ..., "image": ..., "audio": ..., "video": ...}}`. The `<resource>-task_id` token is a deferred reference: at execution time it is rewritten into the actual upstream output (file path, bounding boxes, captions, …). This is what turns a flat task list into a DAG without the LLM having to emit code.
2. **Download-count shortlisting for Model Selection.** The Hub is huge; you cannot fit every candidate into a prompt. HuggingGPT first filters by task type, then ranks the survivors by HF download count, then drops the top-K into the prompt as `{model_id, metadata, description}` items. The LLM is forced to return `{"id": ..., "reason": ...}` — a single-choice question, not free-form generation.
3. **Hybrid local + Inference-API execution with a DAG scheduler.** Local endpoints (fast, limited coverage) are tried first; HF Inference API serves the long tail. Independent tasks dispatch in parallel; dependents serialise on the DAG.

There is no training and no model weights — the entire system is prompts plus orchestration code. Decoding uses `temperature = 0` with `logit_bias = 0.2` on `{` / `}` to bias toward valid JSON.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset(s) | Strength |
|-------|----------|------------|----------|
| C1: LLM can act as controller for multi-step multimodal AI tasks | Tables 3–5 planning F1/Acc; Figs 1, 2, 4–11 | GPT-4-annotated + 130-req human eval | ⭐⭐⭐ |
| C2: GPT-3.5/4 substantially beat open-source LLMs (Alpaca, Vicuna) as controller | Tables 3, 4, 5, 6, 8 — uniformly large gaps (52.62 vs 6.48 single Acc; 41.36 vs 0 sequential human Acc) | both planning datasets | ⭐⭐⭐ |
| C3: HuggingGPT solves a wide range of 24 task types across language/vision/speech/video | Table 13 task list; case studies Figs 4–11 | unlabeled qualitative | ⭐ |
| C4: Download-count shortlisting + in-context single-choice model selection is effective | Human eval: 93.89% passing / 84.29% rationality in model selection (Table 8) | 130 requests, 3 annotators | ⭐⭐ |
| C5: System reaches end-to-end success on real requests | Table 8: 63.08% success rate (GPT-3.5) | 130 human-rated requests | ⭐⭐ |
| C6: Global planning (one-shot full DAG) competitive with iterative planning (AutoGPT) | Appendix B discussion only — no head-to-head numbers | none | ⭐ |
| C7: Demonstration variety/number improves planning | Table 7 (variety 2→10 → graph F1 53.96→66.90); Figure 3 (saturates ~4 shots) | GPT-4-annotated | ⭐⭐ |
| C8: System is open and extensible — adding a new model only needs a description | Architectural claim; no quantitative add-a-model experiment | none | ⭐ |
| C9: Hybrid local + cloud endpoints improve stability/speed | Architectural claim; no latency/uptime numbers | none | ⭐ |
| C10: Achieves "impressive results" toward AGI | Aspirational; case studies only | – | ⭐ |

**Honest read.** C1, C2, and C7 are the genuine empirical core; the gap between GPT-3.5/4 and Alpaca/Vicuna is too large to be noise even without variance reporting. The weak spots are serious and worth naming up front:

- **GPT-4 is both the annotator and an evaluated controller.** The 3,497-example "GPT-4-annotated" split uses GPT-4 outputs as the labels, and GPT-4-Score is then used as a metric — a circular loop. The only honest test is the 46-example human-annotated set, on which GPT-4 reaches just 41–58% accuracy, well below the abstract's "impressive" framing.
- **No variance, no seeds.** Every number is a single deterministic run; with free-form upstream requests and an LLM judge, no confidence intervals or seed sweeps are reported.
- **No head-to-head agent comparison.** Visual ChatGPT, ViperGPT, Visual Programming, TaskMatrix, AutoGPT are mentioned in Related Work but never benchmarked against. The AutoGPT discussion (Appendix B, Table 12) is qualitative only.
- **C3's "24 tasks, wide range" is illustrated, not measured** — Figs 4–11 are hand-picked case studies, not per-task accuracy on standard benchmarks.
- **63.08% end-to-end success means ~37% of requests fail end-to-end** — a finding the abstract under-plays.

## Method & Architecture

![HuggingGPT four-stage pipeline](/assets/images/paper/hugginggpt/page_003.png)
*Figure 2: HuggingGPT's four-stage pipeline — Task Planning → Model Selection → Task Execution → Response Generation, with `<resource>-task_id` resolving cross-task dependencies.*

**Stage 1 — Task Planning (LLM only).** The user request is parsed into a list of JSON task objects. The prompt combines (a) *specification-based instruction* defining the four slots and the available task list, and (b) *demonstration-based parsing* with few-shot examples spanning single, sequential, and graph topologies. Multi-turn dialogues are supported by appending `{{ Chat Logs }}` so previously mentioned resources can be reused.

**Stage 2 — Model Selection (in-context task-model assignment).** For each parsed task, candidate HF models are filtered by task type, ranked by download count, and the top-K are inserted into the prompt as JSON `{model_id, metadata, description}`. The LLM returns `{"id": ..., "reason": ...}`. Download-based shortlisting is the key trick that keeps the prompt under the context limit.

**Stage 3 — Task Execution (hybrid endpoints).** Selected models run on a local endpoint (fast, limited coverage — local has priority) or the HF Inference API (broad coverage, slower). The `<resource>-task_id` symbol from Stage 1 is rewritten into the actual upstream output at launch time. Independent tasks dispatch in **parallel**; dependent ones serialise on the DAG.

**Stage 4 — Response Generation.** The LLM is given the full trace — original request, planned tasks, chosen models, structured inference outputs (bounding boxes, scores, file paths, captions) — and is asked to (i) directly answer the user, (ii) narrate the process, (iii) report file paths verbatim, and (iv) say "I can't make it" if results are empty.

**Decoding.** `gpt-3.5-turbo`, `text-davinci-003`, and `gpt-4` are the controllers; `temperature = 0`; `logit_bias = 0.2` on `{` and `}` to bias toward valid JSON. **Supported task vocabulary** (Table 13): 24 tasks across NLP, CV, Audio, Video — including text/token classification, summarization, translation, QA, conversation, image-to-text, text-to-image, VQA, segmentation, doc-QA, object detection, ControlNet variants, TTS, ASR, and text-to-video.

## Experimental Results

**Planning evaluation on the GPT-4-annotated test set** (one run each; the paper's own headline table):

| Task type | LLM | Acc / GPT-4 Score | Pre | Recall | F1 | Edit Dist. |
|-----------|-----|------------------:|----:|-------:|---:|-----------:|
| Single (Tab. 3)     | Alpaca-7b | 6.48 | 35.60 | 6.64 | 4.88 | – |
| Single              | Vicuna-7b | 23.86 | 45.51 | 26.51 | 29.44 | – |
| **Single**          | **GPT-3.5** | **52.62** | **62.12** | **52.62** | **54.45** | – |
| Sequential (Tab. 4) | Alpaca-7b | – | 22.27 | 23.35 | 22.80 | 0.83 |
| Sequential          | Vicuna-7b | – | 19.15 | 28.45 | 22.89 | 0.80 |
| **Sequential**      | **GPT-3.5** | – | **61.09** | **45.15** | **51.92** | **0.54** |
| Graph (Tab. 5)      | Alpaca-7b | 13.14 | 16.18 | 28.33 | 20.59 | – |
| Graph               | Vicuna-7b | 19.17 | 13.97 | 28.08 | 18.66 | – |
| **Graph**           | **GPT-3.5** | **50.48** | **54.90** | **49.23** | **51.91** | – |

**Human-annotated set** (Table 6, 46 examples — the only non-circular numbers in the paper):

| LLM | Sequential Acc ↑ | Sequential ED ↓ | Graph Acc ↑ | Graph F1 ↑ |
|-----|-----------------:|----------------:|------------:|-----------:|
| Alpaca-7b | 0 | 0.96 | 4.17 | 4.17 |
| Vicuna-7b | 7.45 | 0.89 | 10.12 | 7.84 |
| GPT-3.5 | 18.18 | 0.76 | 20.83 | 16.45 |
| **GPT-4** | **41.36** | **0.61** | **58.33** | **49.28** |

**Human evaluation on 130 requests** (Table 8). GPT-3.5 achieves **91.22% passing / 78.47% rationality** in planning, **93.89% / 84.29%** in model selection, and a **63.08% end-to-end success rate**; Vicuna-13b reaches only 15.64%, Alpaca-13b 6.92%.

**Ablations.** Table 7: scaling demonstration *variety* from 2 → 10 task types lifts GPT-4 graph F1 from **53.96 → 66.90** and sequential F1 from **55.13 → 60.80** — variety matters more than the authors emphasise. Figure 3: performance saturates by **~4 demonstrations**; 0-shot is meaningfully worse, but 5-shot ≈ 4-shot.

## Limitations

**Authors admit.** (i) Planning quality is bounded by the controller LLM; (ii) latency cost from multiple LLM round-trips; (iii) context-window limits on the candidate-model list; (iv) LLM nondeterminism breaking JSON contracts.

**Additional gaps the paper does not address.**
- **No tool-hallucination measurement** — the controller can name a non-existent model id or hallucinate arguments; failures of this kind are uncounted.
- **Download count as a quality proxy biases toward generic / older models.** Strong new models with few downloads are systematically excluded from the candidate shortlist.
- **No safety/security analysis.** The controller will route to arbitrary HF weights of untrusted license or behaviour.
- **No cost analysis** (USD per request, tokens per stage, latency budget).
- **No domain robustness study** — medical, legal, low-resource-language requests are absent from the eval set.
- **The 46-example human-annotated set is too small** for strong statistical claims, and no expansion plan is funded.
- **No plan repair on tool failure.** The workflow is single-shot global planning; if Stage 1 emits a bad DAG, the system cannot re-plan.
- **No head-to-head against Visual ChatGPT, ViperGPT, AutoGPT, TaskMatrix** — competing agent designs are discussed in Related Work but never measured against.

## Why It Matters for Medical AI

The paper does not run any medical experiment; relevance is indirect. The orchestration pattern — JSON-typed tasks, explicit dependency wiring, single-choice model selection from a curated shortlist, hybrid local/cloud execution — is the right shape for a medical agent that might route a single request between a CXR classifier, a CT segmentation network, a report generator, and a guideline-RAG retriever. But the paper's own evaluation already shows the load-bearing weaknesses: download-count ranking is a poor proxy in safety-critical settings (newer, validated medical models lose to popular generic ones), 37% end-to-end failure is unacceptable in clinical workflows, and there is no plan-repair loop when a downstream tool returns an empty or out-of-distribution result. Treat HuggingGPT as a reference architecture for the *control flow*, not a recipe for the *model-selection policy* — in any clinical adaptation the Stage 2 ranker needs to be replaced with something other than popularity, and a Stage 5 (verification / re-planning) needs to be added before deployment.

## References

- **Paper:** Shen, Song, Tan, Li, Lu, Zhuang. *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face.* NeurIPS 2023. [arXiv:2303.17580](https://arxiv.org/abs/2303.17580)
- **Code:** [microsoft/JARVIS](https://github.com/microsoft/JARVIS)
- **Related agent systems:** Visual ChatGPT (Wu et al., 2023); ViperGPT (Surís et al., 2023); Visual Programming (Gupta & Kembhavi, 2023); TaskMatrix.AI (Liang et al., 2023); Toolformer (Schick et al., 2023); AutoGPT (community, 2023).
