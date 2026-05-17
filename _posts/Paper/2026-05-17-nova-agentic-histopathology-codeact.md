---
title: "NOVA: An Agentic Framework for Automated Histopathology Analysis and Discovery"
excerpt: "A CodeAct-style coding agent orchestrating 49 hand-engineered TRIDENT/LazySlide/HoVer-Net tools over 20 iterations scores 0.477 on SlideQuest vs 0.269 for LLM+PI+retries and 0.000 for LLM-only."
categories:
  - Paper
  - LLM-Agents
  - Pathology
permalink: /paper/nova-agentic-histopathology-codeact/
tags:
  - NOVA
  - SlideQuest
  - CodeAct
  - smolagents
  - LLM-Agents
  - Pathology
  - Whole-Slide-Image
  - TRIDENT
  - LazySlide
  - HoVer-Net
  - GPT-5
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- **NOVA is a non-fine-tuned coding agent for computational pathology.** A CodeAct/smolagents loop (Wang 2024; Roucher 2025) wraps a stock LLM (GPT-4.1 default) around 49 hand-engineered Python tools built on TRIDENT, LazySlide, and HoVer-Net. Per query, NOVA emits structured `thought`+`code` JSON, executes in a sandboxed Python 3.11 interpreter with a 10^7 operation cap, and iterates up to 20 steps. Memory is cleared between benchmark queries.
- **SlideQuest is the new benchmark: 90 pathologist- *and* biomedical-scientist-verified questions across DataQA / CellularQA / PatchQA / SlideQA on real gigapixel TCGA-BRCA WSIs.** Scoring uses Hungarian-matched JSON keys with a 15% numeric tolerance; no JSON output -> score 0. Unlike SlideBench-VQA (where LLM-only already hits 45%), SlideQuest's LLM-only baseline is **0.000** on every category - the benchmark genuinely requires code execution on raw pixels.
- **Headline (GPT-4.1, mean of 3 runs): NOVA 0.477 vs LLM+PI+retries 0.269 vs LLM+PI 0.154 vs LLM-only 0.000 - a +20.8 absolute-point margin.** Custom tools beat RAG-over-source-repos (0.477 vs 0.337) and self-generated tools (vs 0.326), which is the cleanest contribution. The "stronger LLMs win" narrative is undercut: **GPT-5-mini ($2.25/1M tok) averages 0.482, statistically indistinguishable from GPT-5 ($11.15/1M tok) at 0.498**.

## Motivation

A whole-slide image (WSI) at 20x is a pyramidal gigapixel file; answering even "what is the average nuclear eccentricity in this slide?" requires chaining tissue segmentation -> patch coordinate extraction -> HoVer-Net nuclei segmentation -> morphometric statistics, plus careful spatial calibration. Biomedical researchers without programming backgrounds are locked out. Two adjacent threads were ready to be combined - (i) coding agents (CodeAct, smolagents) and (ii) the recent open-source pathology stack (TRIDENT, LazySlide, HoVer-Net; foundation models like CONCH / TITAN / PRISM) - but no benchmark existed for dataset-scale agentic pathology. Existing benchmarks operate on static single ROIs (PathVQA, PathMMU), have answers an LLM can guess without an image (SlideBench-VQA: LLM-only 45%), or focus narrowly on diagnosis. NOVA + SlideQuest is the first attempt to evaluate computational-pathology agents on tasks that genuinely *require* code execution on raw WSI data.

## Core Innovation

- **CodeAct loop over a curated tool library.** The LLM emits `{thought, code}` JSON; the sandboxed interpreter runs it; stdout/errors/return values are appended as the next observation. Stops on declared completion or `max_steps=20` (200 for the case study). The dynamic system prompt is assembled per query from smolagents' `code_agent.yaml`, every tool's docstring + I/O signature, and user special instructions.
- **49 atomic, single-purpose tools across 7 categories** (Appendix E): ROI captioning / analysis, dataset processing checks, dataset processing pipelines, documentation retrievers, nuclei segmentation / contour analysis, single-WSI processing, and WSI classification (ABMIL train/test). Each tool has a standardized docstring (capability, inputs, outputs, prerequisites, side-effects) so the LLM can compose rather than call pre-built workflows.
- **SlideQuest benchmark.** 90 questions, 4 scales (DataQA n=25 / CellularQA n=25 / PatchQA n=25 / SlideQA n=15). Every question carries `id`, `data_type`, `dataset_relative_path`, `question`, `additional_instructions` (incl. enforced `seed=42`), `output_instructions` (JSON file `answer.json`), `columns_to_compare_and_tolerance`, plus `is_pathologist_verified` and `is_biomedical_scientist_verified` (both must be true for inclusion).

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | NOVA outperforms all coding-agent baselines on SlideQuest | Table I.1: 0.477 vs 0.269 (PI+retries) vs 0.154 (PI) vs 0.000 (LLM-only), 3-run mean with std-err | SlideQuest (90 Q) | ⭐⭐⭐ - large gap, variance reported, full ablation chain, differences far outside error bars |
| C2 | Custom hand-engineered tools beat RAG-over-source-repos and LLM-written tools | Table K.1: 0.477 (custom) vs 0.337 (RAG over TRIDENT/LazySlide/HoVer-Net) vs 0.326 (no tools); DataQA gap +24 pts | SlideQuest | ⭐⭐⭐ - clean 3-way comparison, RAG given the *actual* source repos so the comparison is fair |
| C3 | Stronger LLMs yield better results on harder categories | Table K.3: GPT-5 > GPT-4.1 on CellularQA (+0.023), PatchQA (+0.082), SlideQA (+0.079) | SlideQuest | ⭐⭐ - directionally consistent, but **DataQA declines (0.708 GPT-5 vs 0.777 GPT-4.1)**; GPT-5-mini matching GPT-5 on average undermines the "scale wins" framing |
| C4 | NOVA does not require instruction-fine-tuned models | All experiments use stock OpenAI endpoints with smolagents + tool docstrings | SlideQuest | ⭐⭐⭐ - methodologically demonstrated |
| C5 | SlideQuest measures computational ability, not linguistic knowledge | LLM-only baseline scores **0.000** on all 90 questions | SlideQuest | ⭐⭐⭐ - strongest possible evidence; contrasts cleanly with SlideBench-VQA where LLM-only hits 45% |
| C6 | NOVA can support scientific discovery (PAM50 case study) | Figure 4 + Appendix B markdown report; pathologist-verified to match Heng et al. 2017 | 4 H&E WSIs, 1 per PAM50 subtype | ⭐ - n=4 slides, qualitative sign-off only, no quantitative metric, no blind comparison vs human report. "Scalable discovery potential" is aspirational from this evidence |
| C7 | Pathologist + biomedical-scientist double-verification ensures clinical validity of SlideQuest | Every question carries two boolean verification flags | SlideQuest | ⭐⭐ - process documented but verifier identity, board certification, sub-specialty, and inter-rater agreement are not stated |
| C8 | 49 tools cover diverse histopathology tasks | Tables E.1-E.7 enumerate the tools | - | ⭐⭐ - genuine enumeration but no coverage analysis vs Table G.1's 33 capabilities |
| C9 | NOVA's failures stem from four named buckets | Section 6.2 + Appendix L (Figs L.1, L.2) | SlideQuest zero-scored cases | ⭐ - plausible taxonomy but **no per-bucket frequencies** |
| C10 | NOVA is faster than coding baselines on DataQA *while* scoring higher | Figure J.1: 2.76 h vs 4.20 h (PI+retries) on DataQA | DataQA only | ⭐⭐ - true for DataQA. **The §6.1 "runtime decreased with tools" framing cherry-picks DataQA**: NOVA takes 4.5 h vs 0.9 h on CellularQA, 8.7 h vs 1.2 h on PatchQA, **31 h vs 2 h on SlideQA** - 2-15x slower on three of four categories |

**Honest read.** The two strongest, multiply-supported contributions are **C1 (NOVA > coding baselines)** and **C2 (custom tools > RAG > LLM-written tools)** - both have full ablation chains, three-run variance, and large effect sizes robust to the GPT-4.1 vs GPT-5 swap. **C5 (the benchmark is computational, not linguistic)** is the cleanest piece of evidence: a 0.000 LLM-only baseline is hard to argue with. The **PAM50 case study (C6)** carries the abstract's "scalable discovery" headline but is n=4 slides with qualitative pathologist sign-off only - closer to a working demo than evidence. The **runtime story (C10)** is misrepresented: tools win on DataQA but NOVA is 2-15x slower than the retry baseline on the other three categories (it just gets more questions right within that time). The **GPT-5 vs GPT-4.1 narrative (C3)** is undercut by GPT-5-mini matching GPT-5, suggesting the binding constraint is tool availability and reasoning style, not raw LLM capability. Variance is reported well (3 runs everywhere, std-err in every table); no statistical significance testing on head-to-head margins, but with std-errs as small as ±0.027 the headline gaps clearly sit outside noise. Methodologically uncomfortable: the **200-step / memory-on case-study config is different from the 20-step / memory-off benchmark config** - the strongest qualitative result was achieved under a setting SlideQuest wouldn't measure.

## Method & Architecture

![NOVA framework: LLM emits thought+code JSON; Python interpreter executes against 49 pathology tools; observations feed back into the loop](/assets/images/paper/nova-pathology/fig_p004_01.png)
*Figure 1: NOVA framework. The core LLM (GPT-4.1 default; GPT-4.1-mini, GPT-5-mini, GPT-5 also benchmarked via Azure OpenAI) emits structured `thought` + `code` JSON blocks. A sandboxed Python 3.11 interpreter with a curated allowlist (numpy/scipy/pandas, openslide/skimage/cv2/PIL, torch, lifelines/sksurv, scanpy/spatialdata/geopandas/shapely; `os` and unrestricted-FS modules blocked; hard cap of 10^7 operations) executes against the 49-tool library. Observations are appended to agent memory and fed back for up to 20 iterations per query (200 for the case study). Memory is cleared between benchmark queries to prevent cross-task leakage.*

**Tool library (Appendix E, 49 tools across 7 categories).**

1. **ROI captioning / analysis** (E.1): `caption_single_histology_image_tool`, `score_single_histology_image_using_text_tool`, `encode_histology_roi_tool`.
2. **Dataset processing checks** (E.2): schema / existence checks for tissue masks, patch coords, patch features, slide features.
3. **Dataset processing pipelines** (E.3): dataset-wide tissue segmentation, patch coordinate extraction, patch-feature extraction with foundation models, slide-feature extraction with TITAN / MADELEINE / PRISM, score heatmap creation.
4. **Documentation retrievers** (E.4): RAG-style search over TRIDENT / LazySlide / HoVer-Net docs.
5. **Nuclei seg / contour analysis** (E.5): `segment_and_classify_nuclei_in_histology_roi_tool` (six classes via HoVer-Net) + classic contour primitives.
6. **Single-WSI processing** (E.6, the largest): text-prompt similarity visualization, zero-shot label prediction, PRISM report generation, single-WSI captioning, tile scoring by text, properties retrieval, tissue / tile / feature extraction, Leiden clustering, UMAP/PCA/t-SNE, Zarr access, top-k cluster patch retrieval, rectangle region read.
7. **WSI classification** (E.7): ABMIL train/test + split / metadata utilities.

Tools are atomic and single-purpose with standardized docstrings (capability, inputs, outputs, prerequisites, side-effects); the agent composes them rather than calling pre-built workflows. NOVA can also generate ad-hoc tools using the allowed data-science libraries when no provided tool fits, though the benchmark experiments use the fixed 49-tool set (the ad-hoc capability is asserted, not measured).

**Configuration (Table H.1).** `executor_type=local`, `use_structured_outputs_internally=True`, `planning_interval=null`, `verbosity_level=1`, `temperature=0` (GPT-4.1 family; GPT-5 requires `temperature=1`), `max_retries=20`.

## Experimental Results

### Main SlideQuest table (Table I.1, GPT-4.1, mean ± std-err over 3 runs)

| Baseline | DataQA (n=25) | CellularQA (n=25) | PatchQA (n=25) | SlideQA (n=15) | **Average** |
|---|---|---|---|---|---|
| LLM only | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000** |
| LLM + Python interpreter (PI) | 0.377 ± 0.053 | 0.058 ± 0.011 | 0.039 ± 0.022 | 0.133 ± 0.067 | **0.154** |
| LLM + PI + retries | 0.443 ± 0.019 | 0.152 ± 0.025 | 0.217 ± 0.012 | 0.259 ± 0.002 | **0.269** |
| **NOVA (49 tools)** | **0.777 ± 0.030** | **0.323 ± 0.017** | **0.335 ± 0.016** | **0.472 ± 0.027** | **0.477** |

![SlideQuest performance (left) and failure rate (right) across the four coding baselines on GPT-4.1](/assets/images/paper/nova-pathology/fig_p007_01.png)
*Figure 3: SlideQuest performance (left, higher is better) and failure rate (right, lower is better) for the four coding baselines on GPT-4.1, mean ± std-err over 3 runs. NOVA's 49 custom tools deliver a +20.8 absolute-point average score gain over LLM+PI+retries and cut the failure rate from 0.596 to 0.330.*

### Failure rate (Table I.2; 1.0 = every question scored 0)

| Baseline | DataQA | CellularQA | PatchQA | SlideQA | **Avg** |
|---|---|---|---|---|---|
| LLM only | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| LLM + PI | 0.580 | 0.773 | 0.947 | 0.867 | **0.783** |
| LLM + PI + retries | 0.507 | 0.627 | 0.613 | 0.667 | **0.596** |
| **NOVA** | **0.200** | **0.320** | **0.413** | **0.422** | **0.330** |

### Custom-tools ablation - the cleanest contribution (Table K.1, GPT-4.1)

| NOVA variant | DataQA | CellularQA | PatchQA | SlideQA | **Avg** |
|---|---|---|---|---|---|
| No custom tools (LLM writes from scratch) | 0.537 | 0.152 | 0.222 | 0.439 | 0.326 |
| With RAG over TRIDENT/LazySlide/HoVer-Net repos | 0.556 | 0.165 | 0.213 | 0.464 | 0.337 |
| **With custom tools (default)** | **0.777** | **0.323** | **0.335** | **0.472** | **0.477** |

Custom, docstring-documented tools beat **both** RAG over the actual source repos (+0.140) and pure LLM-written code (+0.151). This is the most defensible single contribution in the paper.

### Core-LLM ablation - non-monotonic (Table K.3)

| Core LLM | $ per 1M tok | DataQA | CellularQA | PatchQA | SlideQA | **Avg** |
|---|---|---|---|---|---|---|
| GPT-4.1-mini | $2.00 | 0.686 | 0.107 | 0.319 | 0.422 | 0.379 |
| GPT-4.1 | $10.00 | **0.777** | 0.323 | 0.335 | 0.472 | 0.477 |
| GPT-5-mini | $2.25 | 0.767 | 0.266 | 0.354 | **0.582** | 0.482 |
| GPT-5 | $11.15 | 0.708 | **0.346** | **0.417** | 0.551 | **0.498** |

GPT-5 wins on the *hard* categories (CellularQA, PatchQA, SlideQA) but **loses to GPT-4.1 on DataQA**. **GPT-5-mini at $2.25/1M tokens nearly matches full GPT-5 ($11.15) and beats GPT-4.1 ($10) overall**, while taking ~half the wall-clock on SlideQA (8.5 h vs 47.4 h for 15 questions). The "stronger LLMs win" framing in the paper is undercut by this row.

### Runtime (Appendix J) - the §6.1 cherry-pick

| Category | NOVA (h) | LLM+PI+retries (h) | NOVA / baseline |
|---|---|---|---|
| DataQA | 2.76 | 4.20 | 0.66x (NOVA faster) |
| CellularQA | 4.52 | ~0.9 | ~5x slower |
| PatchQA | 8.68 | ~1.2 | ~7x slower |
| SlideQA | 31.19 | ~2 | ~15x slower |
| GPT-5 on SlideQA (15 Q) | 47.4 | - | - |
| GPT-5-mini on SlideQA (15 Q) | 8.5 | - | - |

NOVA completes the full 90-question benchmark in ~40 hours on a single A100. The §6.1 sentence "run time increased substantially without tools but did not improve performance" is true for DataQA only and obscures that NOVA is **2-15x slower** than the retry baseline on the other three categories - it just answers more questions correctly within that time.

### Case study: PAM50 morphological exploration (§6.3, Figure 4, Appendix B)

Three sequential NOVA conversations on one H&E WSI per PAM50 subtype (Luminal A / B, Basal-like, HER2-enriched). The agent first web-searches the literature (3 steps, 30 s) to enumerate 5 morphological features per subtype, then runs a 10-step / 8m35s pipeline per subtype: `extract_tissue_in_wsi_tool` -> `extract_tissue_tiles_in_wsi_tool` -> `extract_patch_features_in_wsi_tool` -> `score_tiles_by_text_in_a_wsi_tool` -> `visualize_text_prompt_similarity_on_wsi_tool` -> top-k regions -> `read_rectangle_region_from_wsi_tool` -> `segment_and_classify_nuclei_in_histology_roi_tool`. A 2-step / 20s synthesis builds the cross-subtype comparison and the markdown report. The output recovers known associations from Heng et al. 2017 (Luminal A low-grade glandular structures + abundant connective tissue; Basal-like high inflammation + central necrosis; HER2-enriched comedo-type necrosis), pathologist-verified qualitatively. **n=4 slides, no quantitative metric, no blind comparison vs human-written report, no inter-rater agreement** - the "scalable discovery potential" headline rests on a working demo, not measurement. Case-study config uses `max_steps=200` with memory reset disabled, which is **different from the SlideQuest config** that produced the headline numbers.

![PAM50 case study: NOVA chains literature retrieval, per-subtype pipelines, and cross-subtype comparison into a pathologist-verified report](/assets/images/paper/nova-pathology/page_009.png)
*Figure 4: PAM50 case study. Three conversational turns chain literature retrieval -> per-subtype tissue-segmentation-through-nuclei-segmentation pipelines -> cross-subtype comparison into a markdown report that recovers known PAM50 morphological associations (Heng et al. 2017). Note the case study runs `max_steps=200` and disables memory reset - a different configuration than the SlideQuest benchmark.*

### Failure-mode taxonomy (§6.2, Appendix L)

Four buckets, no per-bucket frequencies reported: (i) **tool limitations** (HoVer-Net mis-classifies; text-image scoring gives wrong class - Figure L.1 on TCGA-AR-A2LR ductal vs lobular vs metaplastic); (ii) **framework limitations** (10^7 op cap aborts heavy CellularQA/PatchQA tasks -> agent retries with subsets -> incomplete answers); (iii) **ignoring existing tools / data** (Figure L.2: recomputing convexity from full tissue area when `extract_tissue_in_wsi_tool` already returns it without holes); (iv) **fabrications** (agent invents data when load fails; falls back to "darker nuclei are cancerous").

## Limitations

**Authors explicitly admit.**
- **Evaluation checks only final JSON outputs.** Intermediate fabrications, random guessing, and baseless tool calls go unpunished as long as the final answer falls within tolerance.
- The framework **cannot distinguish tool-implementation errors from agent-use-of-tool errors** (HoVer-Net mis-segmentation looks identical to NOVA mis-using HoVer-Net).
- **Reproducibility of agentic behaviour is an open challenge.** Even with `seed=42` enforced for numpy / torch / random, NOVA produces different pipelines across runs (visible in the std-err bands).
- **TCGA monoculture, BRCA-only.** All 90 questions are breast cancer; non-TCGA extensions are explicitly invited.

**Not addressed (but readers should notice).**
- **No human baseline.** A bioinformatician given the same Python environment + tools is not benchmarked - we don't know NOVA's score relative to a competent grad student.
- **No cost reporting per question.** Table K.3 gives $/1M tokens but the per-question $ that would matter for adoption is missing.
- **15% tolerance is the same for every numeric field.** A 15% slack on a cell count is very different from 15% on a p-value; the per-field rationale for tolerance values is not given.
- **Verifier identity unknown.** "Pathologist verified" - board-certified? Sub-specialty (breast pathology)? How many? Inter-rater agreement when multiple checked?
- **Tool composition complexity unmeasured.** The paper claims "modular tools can be composed flexibly" but never reports the average number of tools chained per correct answer, or whether NOVA discovers compositions no docstring example showed.
- **Self-tool-generation never quantified.** §3.2 mentions NOVA "can also create new tools ad hoc" but the benchmark uses the fixed 49-tool set.
- **No head-to-head vs PathFinder / SlideSeek / CPathAgent.** These are cited as "narrow / fine-tuned / VQA-focused" but no shared-task evaluation is run.
- **PatchQA failure rate 0.413 with full tools** - 41% of PatchQA questions still score 0; per-question failure distribution and which capabilities (Table G.1) those failures cluster around is not analyzed.
- **Single in-house benchmark.** Like PathFinder before it, NOVA's headline numbers come from a benchmark constructed by the authors. Community extension is explicitly requested.

## Why It Matters for Medical AI

NOVA's primary message for the medical-AI community is **C2**, not the headline number: when the goal is to give a non-coding biomedical researcher access to a multi-step pathology pipeline, **a small library of hand-engineered, docstring-documented tools beats both RAG-over-source-repos (0.477 vs 0.337) and pure LLM code generation (vs 0.326)** by a margin (~0.14) that survives a model swap from GPT-4.1 to GPT-5. The result generalizes a pattern already visible in MMedAgent and SlideSeek: the binding constraint on agentic medical-AI systems is rarely the LLM, almost always the perception / domain-tool layer wrapped around it. The GPT-5-mini ≈ GPT-5 result reinforces this - cost can drop ~5x with no quality loss when the tooling is right.

The honest caveats for clinical translation are larger than the headline suggests. SlideQuest is **TCGA-derived, breast-only, in-house, 90 questions, 15% tolerance, and JSON-output-only**. The PAM50 case study is **n=4 slides with qualitative pathologist sign-off** - a working demo, not evidence for "scalable discovery". The framework cannot tell whether a wrong answer came from HoVer-Net or from NOVA misusing HoVer-Net, and 41% of PatchQA questions still score 0 with everything turned on. NOVA is best read as a credible **template for tool-orchestration in computational pathology** plus a useful new benchmark that the community now needs to extend beyond TCGA-BRCA.

## References

- Paper (arXiv 2511.11324v1, 14 Nov 2025): https://arxiv.org/abs/2511.11324
- Code & benchmark: https://github.com/microsoft/nova-agent
- CodeAct: Wang et al., 2024 - https://arxiv.org/abs/2402.01030
- smolagents: Roucher et al., 2025 - https://github.com/huggingface/smolagents
- TRIDENT (Mahmood lab): https://github.com/mahmoodlab/TRIDENT
- LazySlide (Rendeiro lab): https://github.com/RendeiroLab/LazySlide
- HoVer-Net (Graham 2019): https://arxiv.org/abs/1812.06499
- CONCH visual-language pathology foundation model: https://www.nature.com/articles/s41591-024-02856-4
- PathFinder (multi-agent histopathology): https://arxiv.org/abs/2502.08916
- SlideSeek (agentic pathology copilot): see related post
