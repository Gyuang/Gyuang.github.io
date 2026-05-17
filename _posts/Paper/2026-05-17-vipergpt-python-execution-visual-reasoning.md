---
title: "ViperGPT: Visual Inference via Python Execution for Reasoning"
excerpt: "Codex writes unrestricted Python over a typed ImagePatch/VideoSegment API and CPython executes it — zero-shot 72.0 IoU on RefCOCO testA, 48.1 GQA, 51.9 OK-VQA, and beats supervised HiTeA on the NExT-QA Hard split (49.8 T / 56.4 C) with no training whatsoever."
categories:
  - Paper
  - LLM-Agents
  - LLM
permalink: /paper/vipergpt-python-execution-visual-reasoning/
tags:
  - ViperGPT
  - Code-Generation
  - Codex
  - Visual-Reasoning
  - Tool-Use
  - Neuro-Symbolic
  - GLIP
  - BLIP-2
  - X-VLM
  - MiDaS
  - LLM-Agents
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- ViperGPT (Surís, Menon, Vondrick — ICCV 2023) feeds Codex a Python API spec of vision modules (`ImagePatch`, `VideoSegment` with `find` / `exists` / `verify_property` / `compute_depth` / `simple_query` / `llm_query` backed by GLIP, X-VLM, MiDaS, BLIP-2, GPT-3) and lets it emit **unrestricted Python** — no DSL, no custom interpreter, just CPython executing the generated function. **Zero training, zero fine-tuning.**
- Zero-shot SOTA across the board: **RefCOCO testA 72.0 IoU, RefCOCO+ testA 67.0 IoU, GQA test-dev 48.1, OK-VQA 51.9**, plus **NExT-QA Hard Temporal 49.8 / Causal 56.4** — the Hard split actually beats supervised HiTeA (48.6 / 47.8). On the full NExT-QA set it slips to 60.0 vs. HiTeA's 63.1, an admission the paper makes only briefly.
- The headline claim is structurally sound but empirically thin: every benchmark is **single-split with no variance / no seeds**, the "+6% over best previous" OK-VQA framing **quietly excludes Flamingo** (closed model, +1.3 pt actual gap), there is **no head-to-head against concurrent VisProg** despite explicit textual differentiation, and **no code-execution failure rate** is reported anywhere — footnote 2 even admits error handling was cosmetically stripped from displayed code. Codex deprecation makes reproducibility an unaddressed hazard.

## Motivation

Two strands of prior work both stalled. **Neural Module Networks** (Andreas 2016; Hu 2017; Johnson 2017) needed expensive program supervision or RL, and once you jointly trained the program generator with the modules, the modules stopped being faithful to their named function — defeating the whole point of modular reasoning. **Monolithic end-to-end VLMs** (Flamingo, BLIP-2, KOSMOS) scale impressively but can't be audited step-by-step, can't reliably do explicit arithmetic, and improve only by adding more data and compute.

ViperGPT's wager is that LLMs trained on internet-scale code have already absorbed enough programming priors to act as a **training-free program generator**, while perception is delegated to off-the-shelf foundation models. Framing-wise this is Kahneman System 1 (perception modules) + System 2 (Python interpreter as deterministic reasoner). For the medical-AI audience this is the canonical reference behind ChatCAD, MMedAgent, MedRAX and other clinical agents that route across specialist perception models.

## Core Innovation

- **Codex emits unrestricted Python** — not a custom pseudocode DSL as in concurrent VisProg [Gupta & Kembhavi 2022]. The interpreter is standard **CPython**, so the model gets native `for`/`if`/`sort`/`len`/`math`/`datetime`/list slicing **for free**, plus Codex's massive code-on-the-internet prior.
- **API as prompt, not as implementation.** The Codex context window holds only the *signatures, type hints, and docstrings* of `ImagePatch` / `VideoSegment` plus a handful of query-to-code exemplars. The expected output is a string of the form `def execute_command(image): ...`.
- **Perception is a thin dispatcher.** `find` → GLIP-L, `verify_property` → X-VLM (chosen over CLIP per Bravo 2022), `compute_depth` → MiDaS DPT_Large median, `simple_query` → BLIP-2 fallback for any non-decomposable question, `llm_query` → GPT-3 as unstructured knowledge base, `select_answer` → GPT-3 multiple choice.
- **Faithfulness by construction.** The printed Python *is* the computation graph — no neural module ever silently overrides the named operation, in stark contrast to jointly-trained NMNs.
- **Video for free.** `VideoSegment.frame_iterator()` yields per-frame `ImagePatch` objects; the LLM is expected to write loops over them using image modules. No video-specific perception model is added — temporal reasoning is emergent from Python control flow.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Zero-shot SOTA on visual grounding. | Table 1: 72.0 / 67.0 IoU vs. ReCLIP 58.6 / 60.5. | RefCOCO, RefCOCO+ testA only | ⭐⭐ — large margin but single split each; no testB/val in the main paper. |
| C2 | Zero-shot SOTA on compositional VQA. | 48.1% vs. BLIP-2 44.7%. | GQA test-dev | ⭐⭐ — single split, +3.4 pt, no variance. |
| C3 | Zero-shot SOTA on knowledge VQA, "+6% over best previous". | 51.9 vs. Flamingo 50.6 (zero-shot) vs. PNP-VQA 35.9. | OK-VQA | ⭐⭐ — the +6% figure quietly **excludes Flamingo** as "non-public"; head-to-head against Flamingo is only +1.3 pt. Framing is inflated. |
| C4 | Matches/exceeds supervised models on video causal/temporal QA despite zero video training. | Hard split 49.8 / 56.4 vs. HiTeA 48.6 / 47.8. | NExT-QA only | ⭐⭐ — strongest qualitative claim. Holds on Hard split, **loses on full set 60.0 vs. 63.1**. Single benchmark, no MSRVTT-QA / ActivityNet-QA. |
| C5 | Code generation is "interpretable" and "faithful by construction". | Qualitative — generated code is the computation graph. | All | ⭐⭐⭐ — structural, not empirical, and the argument is sound: the printed Python is literally what executes. |
| C6 | Generating **unrestricted Python** (vs. VisProg's DSL) is materially better. | Qualitative + Fig. 7 showing `sort`/comparators/arithmetic matter. | RefCOCO | ⭐ — **no head-to-head against VisProg on a common benchmark** despite explicit textual differentiation. Plausible but unevidenced. |
| C7 | Every perception module is necessary; perception and Python ops both contribute. | Fig. 7 intervention bar chart (relative mIoU drop per nulled module). | RefCOCO | ⭐⭐ — single dataset, single split, no error bars; methodology is well-motivated. |
| C8 | "Training-free" and can absorb new modules without retraining. | Section 5.1 in-the-wild qualitative figure. | Anecdotal | ⭐⭐ — structurally true; quantitatively unbacked. |

**Honest read.** The strongest claim — programmatic decomposition with a Codex-written executor beats end-to-end zero-shot VLMs on diverse visual tasks — is well-supported in *direction* but every benchmark is single-split with no variance, no seeds, no significance tests. The most novel claim — emergent video reasoning without video training — shows on one benchmark and reverses on the full set. The intervention study (Fig. 7) is the most rigorous part and the most underused: it deserves to be the headline diagnostic but occupies one small panel. **No comparison against VisProg, no code-execution failure rate, no LLM-vs-vision attribution ablation.** Footnote 2 admits the displayed code has had error handling cosmetically stripped — implying the real generated code has `try`/`except` scaffolding that masks true robustness, the rate of which is never disclosed. Codex is now deprecated, leaving open-LLM reproducibility entirely unmeasured.

## Method & Architecture

![ViperGPT method overview](/assets/images/paper/vipergpt/fig_p003_01.png)
*Figure 2: ViperGPT method. The query and the Python API spec (signatures + docstrings of `ImagePatch` / `VideoSegment` plus a few exemplars) go to Codex, which returns a string `def execute_command(image): ...`. CPython compiles and executes the function, dispatching `find` / `verify_property` / `compute_depth` / `simple_query` / `llm_query` to GLIP, X-VLM, MiDaS, BLIP-2, and GPT-3 respectively. No gradient updates anywhere.*

**1. Formalism.** Given visual input $x$ and query $q$, synthesize program $z = \pi(q)$, then execute $r = \phi(x, z)$. Prior NMN work parameterized $\pi$ with an RL/supervised network over a DSL; ViperGPT replaces $\pi$ with **Codex** emitting raw Python and $\phi$ with the **CPython interpreter**.

**2. API as prompt.** The Codex prompt is the *specification* of `ImagePatch` / `VideoSegment` — signatures, type hints, docstrings, and a few in-context examples. The *implementation* is never shown to the LLM (saves context, decouples spec from impl).

**3. Module API (perception).**

- `find(image, noun) → list[ImagePatch]` — GLIP-L
- `exists(image, noun) → bool` — GLIP-L
- `verify_property(image, obj, attr) → bool` — X-VLM (stronger than CLIP on attributes per Bravo 2022)
- `best_image_match` / `best_text_match` — X-VLM / CLIP-style similarity
- `compute_depth(image) → float` — MiDaS DPT_Large median relative depth
- `simple_query(image, q) → text` — BLIP-2 fallback for non-decomposable visual questions
- `llm_query(text) → text` — GPT-3 as unstructured knowledge base (OK-VQA)
- `select_answer(info, q, choices) → str` — GPT-3 multiple choice (NExT-QA)
- `distance(p1, p2)` — pure Python on pixel coordinates

**4. Module API (reasoning).** No new operators. The LLM composes perception calls with built-in Python — `for`, `if/else`, `sort`, `len`, arithmetic, slicing, `datetime`, `math`. This is the load-bearing design choice that separates ViperGPT from VisProg's custom DSL.

**5. Program execution.** The Codex string is compiled and called against the input. A producer-consumer queue batches calls into the underlying GPU models for throughput.

**6. Video.** `VideoSegment.frame_iterator()` yields `ImagePatch` per frame with start/end timestamps. The LLM writes loops; no video-specific perception model is added.

**7. Contextual prompting (Sec. 5.3).** Extra knowledge can be injected as a Python comment (`# Context: picture taken in the UK`) and the generated program adjusts logic accordingly.

**8. Interventional explainability (Sec. 5.2).** For each module define a "null" default (e.g., `find` returns the whole image); rerun the same generated programs with that module nulled and measure mIoU drop. This isolates per-module contribution without confounding by which questions invoke which module.

![Compositional GQA example](/assets/images/paper/vipergpt/fig_p005_02.png)
*Figure 4: GQA-style compositional question. The generated program uses `find` to locate the bookcase, iterates child patches, and applies `verify_property` to identify the water bottle — perception and Python control flow interleaved.*

![Depth-based grounding example](/assets/images/paper/vipergpt/fig_p004_01.png)
*Figure 3: A "closest pizza" query, resolved by `find('pizza')` followed by `sort` on `compute_depth` — a pattern that monolithic VLMs cannot expose.*

## Experimental Results

| Task / Dataset | Metric | Best supervised baseline | Best prior zero-shot | **ViperGPT (zero-shot)** |
|---|---|---|---|---|
| RefCOCO testA (IoU) | Acc % | OFA 94.0; MDETR 90.4 | ReCLIP 58.6 | **72.0** |
| RefCOCO+ testA (IoU) | Acc % | OFA 91.7; MDETR 85.5 | ReCLIP 60.5 | **67.0** |
| GQA test-dev | Acc % | CRF 72.1; NSM 63.0 | BLIP-2 44.7 | **48.1** |
| OK-VQA | Acc % | PromptCap 58.8; REVIVE 58.0 | Flamingo 50.6; BLIP-2 45.9 | **51.9** |
| NExT-QA Hard — Temporal | Acc % | HiTeA 48.6; ATP 45.3 | — | **49.8** |
| NExT-QA Hard — Causal | Acc % | HiTeA 47.8; ATP 43.3 | — | **56.4** |
| NExT-QA Full Set | Acc % | HiTeA 63.1; VGT 56.9 | — | **60.0** |

ViperGPT is the top zero-shot method on every benchmark reported and beats the strongest supervised baselines on the NExT-QA Hard split (T and C). On full-set NExT-QA it trails HiTeA by 3.1 points but still beats VGT.

![Intervention study on RefCOCO](/assets/images/paper/vipergpt/fig_p008_02.png)
*Figure 7: Intervention study on RefCOCO. Nulling `find` (GLIP) causes a ~70% relative mIoU drop — by far the largest, confirming GLIP is the load-bearing perception module. Python operators (`sort`, comparators `>` / `<`, arithmetic) all show meaningful drops too: perception and Python control flow are **both** essential. The most rigorous panel in the paper — and the most underused.*

![Contextual programs](/assets/images/paper/vipergpt/fig_p008_01.png)
*Figure 8: Same query ("the car on the correct lane") yields **opposite** filters when prefixed with US vs. UK context. Codex incorporates the prior cleanly without retraining — the kind of test-time conditioning that a monolithic VLM cannot offer.*

![OK-VQA external-knowledge example](/assets/images/paper/vipergpt/fig_p006_01.png)
*Figure 5: OK-VQA example — bear-toy question resolved via `simple_query` to identify the object and `llm_query` to retrieve the hibernation fact from GPT-3.*

![NExT-QA temporal reasoning](/assets/images/paper/vipergpt/fig_p007_01.png)
*Figure 6: NExT-QA temporal reasoning. The generated program scans frames in a loop with a `simple_query` predicate ("is the boy dropping sparkles?") to localize the temporal anchor.*

![In-the-wild composition](/assets/images/paper/vipergpt/fig_p002_14.png)
*Figure 1 (excerpt): Multi-hop external-knowledge query — "which of these cars was founded by an angry ex-employee?" — chains `find` + `llm_query` to answer Lamborghini vs. Ferrari.*

## Limitations

**Authors acknowledge.**

- Performance is fundamentally bounded by the perception backbones (GLIP, BLIP-2, X-VLM, MiDaS). Improvements flow through; their failures cap the ceiling.
- Video inference is computationally expensive (Sec. 4.4) even with per-frame loops.
- Adding dedicated video-perception modules would further improve NExT-QA.

**Authors do *not* address.**

- **Code-execution failure rate.** What fraction of Codex outputs (a) fail to compile, (b) raise at runtime, (c) return wrong type, (d) silently return wrong answer? No numbers anywhere. Single biggest audit gap for a "code as reasoning" paper.
- **Error-handling overhead.** Footnote 2 (page 6) admits displayed examples have been **cosmetically stripped of comments and error handling** — meaning the real generated code has `try`/`except` scaffolding that masks robustness. The triggering rate is never disclosed.
- **LLM-vs-vision attribution.** Fig. 7 ablates vision modules only. Swapping Codex for a weaker code LLM, or in-context Python from a smaller model, is exactly the inverse ablation that would isolate how much of the win comes from the code-gen LLM itself. Absent.
- **API design ablation.** No sensitivity analysis on the few-shot exemplars in the API prompt or on the choice of methods (`verify_property` vs. arbitrary BLIP-2 attribute checks).
- **VisProg head-to-head.** The text explicitly differentiates from concurrent VisProg [Gupta & Kembhavi 2022] but never tables a same-benchmark comparison.
- **Cost and latency.** Multiple LLM calls per query (Codex + GPT-3 + perception models). No wall-clock or $-per-query.
- **Reproducibility.** Codex is **deprecated** by OpenAI; later open-source replications have observed substantial drops with open code LLMs. Not flagged in the paper.
- **Domain generalization.** Zero experiments outside natural-image benchmarks. Medical-imaging-agent papers that cite this (MedRAX, MMedAgent-RL, ChatCAD+) inherit *all* of these unknowns plus the risk that GLIP/X-VLM/MiDaS have never seen radiology.
- **No statistical significance, no seeds, no confidence intervals** on any reported number.

## Why It Matters for Medical AI

ViperGPT is the canonical reference for **code-as-reasoning visual agents**, and downstream clinical agent work — MedRAX, MMedAgent, ChatCAD+, and the broader wave of multi-tool radiology assistants — directly inherits the design pattern: a typed API of specialist perception modules, a code LLM as router/planner, and an interpreter as deterministic executor. The pattern is genuinely attractive for clinical use because the printed Python is auditable in a way that an end-to-end VLM forward pass is not.

But the unknowns that ViperGPT punts on become acute in medicine: **what fraction of generated programs silently fail?** is a patient-safety question, not a benchmark question. The lack of any open-LLM reproducibility, the deprecation of Codex, and the complete absence of domain-shift experiments mean every clinical paper that builds on this design is layering on top of an unmeasured failure-mode budget. The intervention methodology from Fig. 7 is the part worth borrowing: per-module ablation with explicit null defaults is exactly the diagnostic any medical-agent paper should be running before it claims interpretability.

## References

- Paper (arXiv): <https://arxiv.org/abs/2303.08128>
- Project page: <https://viper.cs.columbia.edu>
- Surís, D., Menon, S., Vondrick, C. **ViperGPT: Visual Inference via Python Execution for Reasoning.** ICCV 2023.
- Concurrent work: Gupta, T., Kembhavi, A. **Visual Programming: Compositional Visual Reasoning Without Training.** CVPR 2023.
- Perception backbones: GLIP (Li 2022), X-VLM (Zeng 2022), MiDaS (Ranftl 2020), BLIP-2 (Li 2023).
- Predecessor lineage: Neural Module Networks (Andreas 2016; Hu 2017; Johnson 2017).
- Downstream medical agents: ChatCAD+, MedRAX, MMedAgent.
