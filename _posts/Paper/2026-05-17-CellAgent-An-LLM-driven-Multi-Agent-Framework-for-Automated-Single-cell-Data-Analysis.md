---
title: "CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis"
excerpt: "GPT-4 + GPT-4V Planner / Tool-Selector+Code-Programmer / Evaluator pipeline reports 92% task completion (>2× direct GPT-4) and tops batch correction (0.684 vs scVI 0.642) and trajectory inference (0.496 vs Slingshot 0.473) — but the margins are narrow, every mechanism claim is unablated, and the best-of-3 self-iteration is not granted to baselines."
categories: [Paper, BioInformatics, LLM, LLM-Agents]
permalink: /paper/cellagent/
tags:
  - CellAgent
  - LLM-Agents
  - Multi-Agent
  - scRNA-seq
  - GPT-4
  - GPT-4V
  - Single-Cell
  - Bioinformatics
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- CellAgent wraps GPT-4 (+ GPT-4V) in a three-tier multi-agent loop — a **Planner** decomposes a natural-language scRNA-seq task into sub-steps, an **Executor** pair (Tool Selector + Code Programmer) retrieves tools from a curated library and emits code into a Jupyter sandbox, and an **Evaluator** scores multiple candidate runs and picks the winner.
- Headline number: **92% task completion across a >50-dataset benchmark, vs ~46% for direct GPT-4** (>2× claim). Best overall scores against five baselines on **batch correction (0.684 vs scVI 0.642)** and **trajectory inference (0.496 vs Slingshot 0.473)**, plus top mean cell-type annotation across five tissues.
- The win margins are narrow and every mechanism claim is unablated. The Evaluator's "best-of-3" candidate selection is not granted to baselines, contemporaneous scRNA foundation models (scGPT, scFoundation) are not compared, and the 92% completion headline has no per-task breakdown, no variance, and no failure-mode taxonomy.

## Motivation

A typical scRNA-seq workflow chains QC, HVG selection, normalization, batch correction, clustering, marker visualization, cell-type annotation, and trajectory inference, and the field has produced 1,400+ tools with strongly dataset-dependent hyperparameters. The skill barrier is double: Python plus enough single-cell biology to pick the right tool per step. Prior LLM-agent frameworks (MetaGPT, Copilot-style assistants) show that natural-language task decomposition works, but raw GPT-4 lacks specialized bio-tool knowledge, confuses overlapping concepts, and loses context over long multi-step pipelines. CellAgent's bet is that bio domain expertise can be packaged as (a) a curated tool library with standardized docstrings, (b) expert-authored role prompts, and (c) automatic visual evaluation through GPT-4V — without any fine-tuning.

## Core Innovation

- **Hierarchical agent split.** Planner handles task-level decomposition only; the Executor is further split into a Tool Selector (retrieval over the registered tool list) and a Code Programmer (docstring-grounded code generation). The Evaluator is a separate LLM call that scores candidate solutions.
- **Self-iterative optimization with MLLM-as-judge.** For batch correction and trajectory inference the Evaluator is GPT-4V scoring rendered UMAP plots; for annotation, GPT-4 aggregates outputs from multiple annotation tools. Default 3 candidate iterations per critical step.
- **Two-tier memory.** A *global* memory `M` stores only the final accepted code per step (compact, high-entropy). A *local* memory holds the per-subtask dialogue (errors, retries) and resets on subtask completion — keeping context windows lean across a long pipeline.
- **Code Sandbox via `nbconvert`.** Every generated code block executes inside a real Jupyter notebook; on exception the Code Programmer is re-prompted with the traceback until success or retry budget exhaustion.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | 92% task completion across the benchmark, >2× direct GPT-4 | Abstract + Sec. 2.1; no per-task breakdown, no per-dataset trial counts, no variance | ">50 datasets" composite (composition not enumerated) | ⭐ |
| C2 | Best batch correction overall (0.684) vs five baselines | Fig. 2a-c on 9 atlas datasets, weighted mean of 10 scIB-style metrics | 9 human organ atlases (celltypist) | ⭐⭐ |
| C3 | Best cell-type annotation mean accuracy across tissues | Fig. 3a bar chart; PBMC 17-cluster ~94% effective rate (10 full + 5 partial + 1 mismatch) | PBMC, liver, lung, pancreas, mouse retina | ⭐⭐ |
| C4 | Best trajectory inference overall (0.496 vs Slingshot 0.473) | Fig. 4a-b on 9 dynverse gold-standard datasets (Saelens et al. metrics) | 9 dynverse datasets | ⭐⭐ |
| C5 | The Planner / Executor / Evaluator separation is necessary for the gains | No ablation removing any agent role; only contrast is CellAgent vs raw GPT-4 | — | ⭐ |
| C6 | Self-iterative optimization with GPT-4V improves quality | No 0 vs 1 vs 3 iteration sweep; no GPT-4V vs human-judge agreement study | — | ⭐ |
| C7 | Tool library + docstring retrieval supplies bio-grounding | No ablation removing tool retrieval | — | ⭐ |
| C8 | GPT-4V can replace human visual judgment for batch correction / trajectory | Asserted in Discussion; no inter-rater study, no adversarial cases where GPT-4V picks a clearly worse UMAP | — | ⭐ |

**Honest read.** The benchmark wins (C2-C4) are credible because they reuse established public composites — scIB-style metrics for batch correction, the Saelens dynverse benchmark for trajectory, Hou & Ji 2024 scoring for annotation. But every margin is narrow (0.684 vs 0.642; 0.496 vs 0.473) and the paper reports no variance, no statistical tests, no random-seed repeats, and no confidence intervals. CellAgent's Evaluator picks the best of 3 candidates per critical step — an option the baselines are not granted, so this is not a like-for-like comparison. The 92% completion headline (C1) is the most-quoted number and the weakest evidenced: the numerator/denominator construction is not given, GPT-4's failure modes are not characterized, and the >50-dataset composition is not enumerated. Mechanism claims (C5-C7) lack ablations entirely — we cannot tell whether the three-agent split matters, or whether a single GPT-4 with the same tool list and sandbox would close most of the gap. C8 (GPT-4V replacing human judgment) is a load-bearing claim presented without a single agreement study. Finally, data-leakage risk is unaddressed: all benchmark datasets (PBMC 10x, Tabula Muris, celltypist atlases) are heavily described online and likely sit inside GPT-4's training corpus.

## Method & Architecture

![CellAgent per-subtask loop with Tool Selector, Code Programmer, Code Sandbox, and Evaluator](/assets/images/paper/cellagent/fig_p004_05.png)
*Figure 1: CellAgent's per-subtask loop — the Tool Selector retrieves candidates from the tool library, the Code Programmer reads docstrings and the global memory to emit executable code, the Code Sandbox runs it via `nbconvert`, and the Evaluator (GPT-4 / GPT-4V) scores and aggregates across iterations.*

![CellAgent top-level workflow from user task to Planner decomposition to aggregated result](/assets/images/paper/cellagent/fig_p004_04.png)
*Figure 2: Top-level workflow — natural-language task (e.g., "annotate the cell types of given single-cell data") is decomposed by the Planner into sub-steps `{t_1, ..., t_n}`, each handed to the Executor / Evaluator loop, and aggregated into the final result.*

### Inputs and Planner

The user provides an AnnData object `D`, a free-form task description `u_task`, and optional data and preference strings `u_D`, `u_req`. The Planner additionally receives the AnnData string representation `ψ(D)` so it can inspect shape and `obs`/`var` fields before decomposing:

$$\{t_1, \ldots, t_n\} \leftarrow A_p^{LLM}(p_{sys}^p, u_{task}, u_{req}, u_D, \psi(D))$$

The system prompt `p_sys^p` carries a role description, a JSON output spec, and pre-collected expert experience for scRNA-seq workflows.

### Executor

Each sub-task `t_i` is handled by two LLM calls. The **Tool Selector** retrieves a candidate subset from the registered tool list `T`:

$$T_{t_i} \leftarrow A_t^{LLM}(p_{sys}^t, u_{req}, T, t_i)$$

The **Code Programmer** reads the tool docstrings `Doc(T_{t_i})`, the global code memory `M`, and emits text analysis `w_i` plus executable code `c_i`:

$$(c_i, w_i) \leftarrow A_c^{LLM}(p_{sys}^c, u_{req}, u_D, M, t_i, Doc(T_{t_i}))$$

The Code Sandbox `E` executes `c_i` inside a Jupyter notebook. On exception the Code Programmer is re-prompted with `E(c_i)` until success or retry budget exhaustion.

### Evaluator and Self-Iteration

For critical steps (batch correction defaults to 3 candidate iterations, trajectory similar) the Executor produces multiple candidates `{c_i^j}`. The Evaluator picks the optimal one:

$$\bar{c}_i = A_e^{LLM}(p_{sys}^e, u_{req}, u_D, t_i, \{c_i^j\})$$

For batch correction and trajectory inference the Evaluator is **GPT-4V scoring rendered UMAP plots**; for annotation it is GPT-4 aggregating outputs across multiple annotation tools.

### Memory

Global memory `M ← {c̄_1, c̄_2, ...}` stores only the final accepted code per step. Local memory inside one subtask holds the dialogue (errors, retries) and resets when the subtask completes — a deliberate compactness choice to fit long pipelines into context.

### Tool Library

The curated, docstring-wrapped registry includes scVI / LIGER / Scanorama / Harmony / Combat (batch correction); CellMarker 2.0 / ACT / CellTypist / SCSA / ScType / GPT-4-as-annotator (annotation); Raceid stemid / Scorpius / PAGA / PAGA tree / Slingshot (trajectory). Each tool is wrapped as a Python class with a standardized docstring that the Code Programmer reads at generation time.

### Backbone

GPT-4 for textual reasoning, code generation, and annotation aggregation; GPT-4V for visual evaluation of UMAP plots. **No fine-tuning** — capability comes from prompts + tools + sandbox + iteration.

## Experimental Results

### Batch correction (9 human organ atlases, 10-metric composite)

![CellAgent batch correction across nine organ atlases](/assets/images/paper/cellagent/page_006.png)
*Figure 3: Batch correction across nine human organ atlases — CellAgent overall 0.684 vs scVI 0.642, leading rank on 4/9 datasets, and clean batch mixing on the heart UMAP.*

| Method | Overall score |
|---|---|
| **CellAgent** | **0.684** |
| scVI | 0.642 |
| Harmony | lower (not numerically given in main text) |
| Scanorama | lower |
| LIGER | lower |
| Combat | lowest cluster |

CellAgent ranks #1 on 4/9 datasets; median violin overall ≈ 0.69.

### Cell-type annotation (mean accuracy across 5 tissues)

![CellAgent cell-type annotation across PBMC, liver, lung, pancreas, mouse retina](/assets/images/paper/cellagent/fig_p008_01.png)
*Figure 4: Cell-type annotation on five tissues — CellAgent matches expert labels on 10/17 PBMC clusters (full match) and tops mean accuracy across PBMC, liver, lung, pancreas, and mouse retina.*

CellAgent reports highest mean accuracy across tissues; per-tissue numbers are shown only as Fig. 3a bars in the paper. On PBMC's 17 clusters: 10 full match + 5 partial match + 1 mismatch ≈ 94% effective annotation rate.

### Trajectory inference (9 dynverse gold-standard datasets)

![CellAgent trajectory inference on dynverse gold-standard datasets](/assets/images/paper/cellagent/page_009.png)
*Figure 5: Trajectory inference on nine dynverse gold-standard datasets — CellAgent 0.496 vs Slingshot 0.473, with recovered LT-HSC → ST-HSC → MPP lineage and monotonic CD48/MPO expression along pseudotime.*

| Method | Overall |
|---|---|
| **CellAgent** | **0.496** |
| Slingshot | 0.473 |
| PAGA / PAGA tree / Scorpius / Raceid stemid | lower (not enumerated in main text) |

CellAgent also leads on NMSE_rf, R²_rf, isomorphic, F1_milestones, and cor_features (radar in Fig. 4b).

### Overall completion

92% for CellAgent vs ~46% for direct GPT-4 ("more than doubling"; exact GPT-4 number not given in the main text).

### Qualitative

- **Fig. 2d (heart):** ventricle-enriched pericytes, myeloid, and stromal pericytes mix cleanly across batches while preserving cell-type structure.
- **Fig. 4c-d (aging-HSC Kowalczyk):** recovers the LT-HSC → ST-HSC → MPP lineage and shows monotonic CD48 / MPO increase along pseudotime.
- First reported use of GPT-4V as automatic visualization judge for batch correction / trajectory, and first use of GPT-4 to aggregate annotations across multiple tools.

## Limitations

**Acknowledged by the authors.**

- Self-evaluation depends on GPT-4V / GPT-4, capping the range of optimization goals to what an MLLM can score visually.
- Adding user-supplied tools to align with custom preferences is left as future work.

**Not addressed in the paper.**

- **No ablations** isolating the contribution of Planner, Evaluator, memory, sandbox, tool retrieval, or iteration count — the only contrast is CellAgent vs raw GPT-4.
- **No comparison with scRNA-seq foundation models** (scGPT, scFoundation, GenePT) or with the standalone Hou & Ji 2024 GPT-4 annotator beyond its inclusion as a sub-tool inside CellAgent.
- **No variance, CI, or statistical test** on any reported number; no random-seed repeats.
- **No cost / latency analysis.** GPT-4 + GPT-4V × 3 iterations × multiple tools per step is expensive and slow; no token-cost or runtime per dataset is reported.
- **Best-of-3 selection is unfair vs baselines.** The Evaluator picks the top of 3 candidate runs per critical step; scVI / Slingshot / Harmony are run once. A like-for-like comparison would re-run baselines with multiple hyperparameter draws and the same Evaluator.
- **Data-leakage risk.** All benchmark datasets are public and indexable (PBMC 10x, Tabula Muris, celltypist.org), so GPT-4 has likely seen them in pretraining; no held-out, previously-unseen dataset is evaluated.
- **No failure-case taxonomy.** In the 8% of failed runs, the paper does not break down whether the Planner mis-decomposes, the Code Programmer exhausts its retry budget, or the Evaluator picks a worse plot.
- **GPT-4V judging is unverified** against human-expert agreement on UMAP plots — the load-bearing C8 claim has no inter-rater study.
- **Task scope is narrow.** Spatial transcriptomics, multi-omics integration (CITE-seq, ATAC), differential expression, GSEA, and perturbation are out of scope.
- **Reproducibility.** No model-version pinning for GPT-4 / GPT-4V is reported in the main text — these endpoints drift in behavior over time.

## Why It Matters for Medical AI

If the framing holds up, automatic agents that can drive end-to-end scRNA-seq analysis from a natural-language brief would compress one of the more skill-bottlenecked steps in translational biomedical research — atlas-style annotation, batch-corrected disease comparisons, lineage tracing — into something a non-programmer clinician-scientist can run. The bigger architectural lesson for medical AI is the **MLLM-as-judge inside the loop**: visual scientific outputs (UMAPs, pseudotime plots, segmentation overlays) are exactly the artifacts a clinical workflow needs an agent to self-critique on, and CellAgent is one of the first systems to put GPT-4V in that judging seat at scale. The caveat is exactly the unaudited piece: until somebody runs a serious human-expert agreement study on the MLLM judge, every "the agent picked the best plot" claim is on credit.

## References

- Paper: Xiao, Liu, Zheng, Xie et al., *CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis*, bioRxiv 2024 (v2, May 17 2024). DOI: [10.1101/2024.05.13.593861](https://doi.org/10.1101/2024.05.13.593861)
- Benchmark sources: celltypist.org/organs (Conde et al. 2022, *Science*); Saelens et al. dynverse trajectory benchmark ([Zenodo 1443566](https://doi.org/10.5281/zenodo.1443566)); hemberg-lab pancreas; Tabula Muris; Allen mouse cortex+hippocampus.
- Related work: scGPT (Cui et al., 2024, *Nat. Methods*); scFoundation (Hao et al., 2024, *Nat. Methods*); MetaGPT (Hong et al., 2023); Hou & Ji, *Assessing GPT-4 for cell type annotation* (2024, *Nat. Methods*).
