---
title: "Visual Programming: Compositional Visual Reasoning Without Training"
excerpt: "GPT-3 writes Python-like programs over a 20-module library executed by an interpreter that also emits an HTML rationale — +2.7 over ViLT-VQA on a sampled GQA testdev and 62.4% zero-shot on NLVRv2."
categories:
  - Paper
  - LLM-Agents
  - LLM
permalink: /paper/visprog-compositional-visual-reasoning/
tags:
  - VisProg
  - Neuro-Symbolic
  - LLM-as-Controller
  - In-Context Learning
  - Compositional Reasoning
  - GPT-3
  - CVPR 2023 Best Paper
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- VisProg (CVPR 2023 **Best Paper**) is a **train-free neuro-symbolic system**: GPT-3 emits a Python-like program of module calls from a few in-context (instruction, program) examples, and an interpreter executes the program over a shared 20-module library (OWL-ViT, DSFD, MaskFormer, CLIP, ViLT-VQA, Stable Diffusion + OpenCV/Python ops).
- The interpreter also auto-generates an **HTML visual rationale** of every intermediate output (boxes, masks, lists, images), turning errors into a debugging surface — users can rewrite the instruction to fix failing modules with no model weights touched.
- Headline numbers: **+2.7 acc on a sampled GQA testdev** (50.5 voting vs 47.8 ViLT-VQA), **62.4% zero-shot on NLVRv2 test** (vs 76.3 finetuned upper bound), **63.7 → 75.7 F1** on Knowledge Tagging and **59.8 → 66.4 accuracy** on Image Editing after instruction tuning on small in-house benchmarks.

## Motivation

End-to-end pretrain-then-multitask VL models (Flamingo, GPV, Unified-IO, Gato) need curated supervision per task and still cannot natively cover instructions that compose detection + external knowledge + image manipulation — e.g. "Tag the 7 main characters of Big Bang Theory" or "Replace Barack Obama with Barack Obama wearing sunglasses." Earlier Neural Module Networks decomposed tasks but relied on brittle semantic parsers or REINFORCE-trained layout generators and only emitted differentiable sub-graphs.

VisProg's bet is that GPT-3's in-context learning is already strong enough to *generate the layout itself* as high-level pseudocode, side-stepping training entirely and letting the system freely mix neural modules (CLIP, OWL-ViT, MaskFormer, ViLT, DSFD, Stable Diffusion) with non-differentiable symbolic ones (OpenCV ops, arithmetic, dict lookups, GPT-3 knowledge queries).

![VisProg overview](/assets/images/paper/visprog/fig_p001_01.png)
*Figure 1: VisProg overview — a natural-language instruction is fed to GPT-3, which emits a program; the interpreter executes module-by-module and assembles a final answer plus an HTML visual rationale (here, tagging the 7 main Big Bang Theory characters).*

## Core Innovation

Three things together, none of them alone:

1. **20-module shared library** with one Python class per module (`parse`, `execute`, `html`). Adding a tool = registering one class.
2. **Program generation by GPT-3 in-context learning** with a few hand-written (instruction → program) pairs as the prompt. The LLM never sees the image — descriptive module/argument/variable names act as type hints.
3. **HTML visual rationale** built incrementally by the interpreter, exposing every intermediate (bbox, mask, list, image, inpainted region) so users can localize a failure to a specific module and rewrite the natural-language instruction to fix it ("instruction tuning" with no weight updates).

![Task overview](/assets/images/paper/visprog/fig_p005_02.png)
*Figure 2: The four evaluation tasks — compositional VQA (GQA), zero-shot image-pair NLVR (NLVRv2), knowledge tagging, and image editing — and the modules each reuses (Loc, FaceDet, VQA are shared).*

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | VisProg solves compositional visual tasks without any task-specific training. | All four task tables (Tabs. 1–4) run with frozen GPT-3 + frozen modules; only prompts are crafted. | ⭐⭐⭐ |
| C2 | VisProg beats the underlying VQA model by +2.7 on compositional VQA. | Tab. 1: 50.5 (voting) vs 47.8 (ViLT-VQA). | ⭐⭐ Gain is real but on a **self-defined sampled slice** of GQA testdev, no comparison to NMN/MAC/MMN. |
| C3 | 62.4% zero-shot accuracy on NLVRv2 without training on image pairs. | Tab. 2 voting, 5 runs, 16 examples. | ⭐⭐ Full public test set, but "zero-shot" is conditional: ViLT-VQA is supervised on VQAv2; gap to finetuned upper bound is 13.9 pts. |
| C4 | Visual rationales let users substantially improve performance via instruction tuning. | Tab. 3 (+12.0 F1) and Tab. 4 (+6.6 acc). | ⭐⭐ Large gains but on **small in-house benchmarks (100/107 instructions) the same authors curated and inspected** — no held-out evaluator. |
| C5 | Accuracy rises monotonically with in-context examples; voting > random average. | Fig. 7 with 95% CIs from 5 seeds. | ⭐⭐⭐ Only result with explicit variance. |
| C6 | Shared modular framework flexibly handles 4 disparate tasks (20 modules). | Sec. 4.1–4.4 + Figs. 1, 4, 6. | ⭐⭐⭐ |
| C7 | Curated prompts ≈ voting at 5× less compute. | Tab. 1: 50.0 curated (1 run) vs 50.5 voting (5 runs). | ⭐ 0.5-point gap on a single sampled set with no error bars; curated prompt was built by inspecting val failure modes (leakage risk). |
| C8 | "Replacing ViLT-VQA could improve NLVR by up to 24%." | Fig. 8 error pies. | ⭐ This is an **oracle ceiling** that assumes every VQA-module error vanishes with a better model, not a measurement. |
| C9 | VisProg generalizes better than NMNs because layouts come from an LLM. | Conceptual argument only. | ⭐ **No head-to-head with NMN, MAC, MMN, or learned layout generators** despite the framing. |
| C10 | Compose 20 modules over text/image/image-pair/bbox/mask/generated-image modalities. | Module table + program figures. | ⭐⭐ Visibly true at the system level; qualitative. |

**Honest verdict.** The most defensible contributions are the *framework* (C1, C6, C10) and the *prompt-size / voting study* (C5). The headline numbers (C2, C3) live on a self-chosen GQA sample and a single benchmark for NLVR. No statistical tests anywhere outside Fig. 7. No reporting of cost, latency, or API-call budget per instruction. The instruction-tuning result is the most "wow"-inducing in the paper but is measured on tiny benchmarks the authors built *and* tuned against. Sensitivity to LLM choice is untested — only text-davinci-002 era GPT-3 is used.

## Method & Architecture

![Module library](/assets/images/paper/visprog/fig_p002_01.png)
*Figure 3: The 20 modules currently supported by VisProg — neural in red (OWL-ViT, DSFD, MaskFormer, CLIP, ViLT-VQA, Stable Diffusion) and symbolic / OpenCV-based in blue (Crop, CropAbove/Below/Left/Right/Front, Count, Eval, List, ColorPop, BgBlur, Emoji, Tag, Result).*

**Modules.** Each module is a Python class with three methods: `parse(step)` parses the program line, `execute(step, state)` runs the tool and writes the output variable back into the interpreter state, and `html(inputs, output)` returns the HTML snippet appended to the visual rationale. Registering a new tool means adding one class.

**Program generation.**

![Program generation](/assets/images/paper/visprog/fig_p004_01.png)
*Figure 4: Program generation — a task-specific preamble plus k hand-written (instruction, program) demonstrations plus the new instruction is fed to GPT-3, which emits a multi-line program where each line has the form `VAR = Module(arg=value, ...)`. The LLM never sees the image.*

**Interpreter.** Maintains a dict `state` mapping variable names → values (strings, ints, lists, PIL images, `{bbox, mask, category}` objects). Walks the program line-by-line, dispatches to the matching module's `execute`, and concatenates each module's `html()` snippet into the final visual rationale.

![Visual rationale](/assets/images/paper/visprog/fig_p005_01.png)
*Figure 5: Visual rationales — per-step HTML summaries are stitched into one interpretable trace. Top: image-editing rationale showing bbox, mask, and inpainted result. Bottom: NLVRv2 rationale showing per-image VQA queries and the boolean expression evaluated by `Eval`.*

**Per-task adaptation = swap the prompt examples.**

- **GQA (compositional VQA).** Up to 24 hand-written in-context examples. Programs decompose questions into `Loc → Crop(spatial) → Count / VQA → Eval`. Crucially, ViLT-VQA is invoked on simpler sub-queries about cropped regions, not the original compound question.
- **NLVRv2 (zero-shot image-pair).** 16 (12 after de-duplication) in-context statements. The statement is split into per-image VQA queries; answers combined with a Python boolean via `Eval`. The single-image ViLT-VQA never sees image pairs during training — this is what supports the "zero-shot" framing.
- **Knowledge Tagging.** 14 hallucinated in-context examples (no images). Pipeline: `List(query)` (GPT-3 returns a category list, capped by a `max` arg, default 20) → `Loc` or `FaceDet` → `Classify` (CLIP) → `Tag`.
- **Image Editing.** 10 in-context examples. Chains `FaceDet`/`Seg` → `Select` (CLIP region scoring) → `ColorPop` / `BgBlur` / `Emoji` / `Replace` (Stable Diffusion inpainting).

**Prompting strategies.** *random* (random subset, single run), *voting* (5 runs with different random subsets, majority vote — self-consistency analogue), *curated* (hand-selected subset, sometimes augmented with hallucinated failure-mode examples). **No weights are updated anywhere.**

## Experimental Results

| Task | Dataset (subset) | Method | Prompting | Ctx ex / run | Runs | Metric | Score |
|---|---|---|---|---|---|---|---|
| Compositional VQA | GQA testdev (sampled) | ViLT-VQA | — | — | 1 | Acc | 47.8 |
| Compositional VQA | GQA testdev (sampled) | VisProg | random | 24 | 1 | Acc | 48.2 |
| Compositional VQA | GQA testdev (sampled) | VisProg | curated | 20 | 1 | Acc | 50.0 |
| **Compositional VQA** | **GQA testdev (sampled)** | **VisProg** | **voting** | **24** | **5** | **Acc** | **50.5** |
| NLVR (zero-shot) | NLVRv2 test | ViLT-NLVR finetuned (upper bound) | — | — | 1 | Acc | 76.3 |
| NLVR (zero-shot) | NLVRv2 test | VisProg | curated | 12 | 1 | Acc | 61.8 |
| NLVR (zero-shot) | NLVRv2 test | VisProg | random | 16 | 1 | Acc | 61.3 |
| **NLVR (zero-shot)** | **NLVRv2 test** | **VisProg** | **voting** | **16** | **5** | **Acc** | **62.4** |
| Knowledge Tagging | 100 instr / 46 imgs | VisProg, original instructions | — | — | 1 | P / R / F1 | 69.0 / 59.1 / 63.7 |
| **Knowledge Tagging** | **100 instr / 46 imgs** | **VisProg, modified instructions** | **—** | **—** | **1** | **P / R / F1** | **77.6 / 73.9 / 75.7** |
| Knowledge Tagging (loc only) | same | VisProg, original | — | — | 1 | P / R / F1 | 87.2 / 74.9 / 80.6 |
| **Knowledge Tagging (loc only)** | **same** | **VisProg, modified** | **—** | **—** | **1** | **P / R / F1** | **87.4 / 82.5 / 84.9** |
| Image Editing | 107 instr / 65 imgs | VisProg, original | — | — | 1 | Manual acc | 59.8 |
| **Image Editing** | **107 instr / 65 imgs** | **VisProg, modified** | **—** | **—** | **1** | **Manual acc** | **66.4** |

(Tables 1–4, verbatim from the paper.)

![Qualitative results](/assets/images/paper/visprog/fig_p006_01.png)
*Figure 6: Qualitative gallery — image editing (top: color-pop, BG blur, replacement, de-identification) and knowledge tagging (bottom).*

![GQA prompt-size sweep](/assets/images/paper/visprog/fig_p007_01.png)
*Figure 7 (left): GQA val accuracy vs number of in-context examples; voting (5 runs) beats single-run mean, with 95% CI bars across seeds.*

![NLVR prompt-size sweep](/assets/images/paper/visprog/fig_p007_02.png)
*Figure 7 (right): NLVRv2 val accuracy vs in-context examples; saturates earlier than GQA (attributed by authors to fewer distinct modules being used).*

**Error taxonomy.**

![Error pies](/assets/images/paper/visprog/fig_p008_01.png)
*Figure 8 (GQA): dominant failure modes — incorrect program (16%) and VQA-module errors. (Pies for NLVR, Knowledge Tagging, and Image Editing show ViLT-VQA, `List`/`Select`, and `Select`/`Seg` as the respective bottlenecks; from manual inspection of ~100 rationales per task.)*

The "+24% if we swap ViLT" headline derived from these pies is an **oracle ceiling** — it assumes every VQA-module error disappears with a better model. Treat it as motivation for follow-up work, not as a measurement.

**Instruction tuning via visual rationales.**

![Instruction tuning examples](/assets/images/paper/visprog/fig_p009_01.png)
*Figure 9 (example 1): Inspecting the rationale reveals the `List` module returned a stale/wrong CEO; rewriting the instruction to "most recent CEO of IBM" flips the failure case to success.*

![Instruction tuning examples](/assets/images/paper/visprog/fig_p009_04.png)
*Figure 9 (example 2): A vague "item" query for `List` underspecifies the target; rewriting to "kitchen appliance that makes coffee" rescues the tag.*

The +12.0 F1 (Tagging) and +6.6 acc (Editing) gains come almost entirely from this human-in-the-loop loop, not from any model change. Caveat: same authors curated, ran, and "instruction-tuned" against the same small benchmarks they scored on.

## Limitations

**Acknowledged by the authors:**
- Sampled GQA testdev (not full leaderboard) — to cap GPT-3 spend.
- ViLT-VQA single-image limit propagates errors into NLVR.
- Stable-Diffusion visual artifacts are *not* penalized in the editing benchmark.
- Object replacement can fail catastrophically without producing a parseable rationale.

**Flagged here, not addressed in the paper:**
- **No comparison to NMN, MAC, MMN, MDETR**, or other learned-layout compositional baselines, despite framing the paper against them. Only ViLT-VQA is reported as a comparator.
- **No cost, latency, or per-instruction API-call budget** — central to whether this is practical at scale.
- **No statistical significance tests** on Tables 1–4; only Fig. 7 has CIs.
- **"Zero-shot" framing on NLVR** glosses over ViLT-VQA being supervised on VQAv2 and GPT-3 plausibly having seen NLVR-style text on the web.
- **Knowledge Tagging and Image Editing are tiny in-house benchmarks** (100 and 107 instructions) curated, inspected, and instruction-tuned against by the same authors — read the +12.0 / +6.6 gains as a ceiling of prompt engineering on a closed loop, not OOD generalization.
- **Sensitivity to LLM choice untested** — only text-davinci-002 era GPT-3 used; no Codex, no smaller LM, no open-weights ablation.
- **Program-syntax robustness** (fraction of invalid programs, missing modules) not broken out from "incorrect program."

## Why It Matters for Medical AI

Even though the paper itself is not medical, the recipe — frozen LLM as a controller emitting calls to a fixed pool of vision/symbolic tools, with a human-readable rationale per step — is precisely the template later medical-imaging agent systems (ChatCAD, MedRAX, MMedAgent, radiology-report agents) adopt. The interpretability win is especially relevant in clinical settings where every intermediate (segmentation mask, retrieved guideline, computed measurement) needs to be auditable. The honest caveats also transfer: tiny in-house eval sets, no cost/latency reporting, and oracle-ceiling claims are exactly the failure modes to watch for when reading medical-agent papers downstream.

## References

- Paper (arXiv): [2211.11559 — Visual Programming: Compositional Visual Reasoning Without Training](https://arxiv.org/abs/2211.11559)
- Project page: [prior.allenai.org/projects/visprog](https://prior.allenai.org/projects/visprog)
- CVPR 2023 — Best Paper Award
- Authors: Tanmay Gupta, Aniruddha Kembhavi (PRIOR @ Allen Institute for AI)
- Related: Neural Module Networks (Andreas et al. 2016), MAC (Hudson & Manning 2018), MMN (Chen et al. 2021), ViLT (Kim et al. 2021), CLIP (Radford et al. 2021), OWL-ViT (Minderer et al. 2022), MaskFormer (Cheng et al. 2021), Stable Diffusion (Rombach et al. 2022), Chain-of-Thought / Self-Consistency (Wang et al. 2022)
