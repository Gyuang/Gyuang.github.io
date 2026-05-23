---
title: "scMulan: A Multitask Generative Pre-trained Language Model for Single-Cell Analysis"
excerpt: "A 368M-parameter GPT-style decoder pretrained on 10M cells reframed as (entity, value) 'cell sentences' — headline zero-shot annotation Acc 0.917 / 0.927 / 0.873 on BoneMarrow / Simonson2023 / Su2022 — but the 'zero-shot' framing leaks cell-type labels through pretraining metadata tokens."
categories: [Paper, BioInformatics, LLM]
permalink: /paper/scmulan/
tags:
  - scMulan
  - Single-Cell
  - scRNA-seq
  - Foundation-Model
  - Cell-Sentence
  - Generative-Pretraining
  - GPT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- scMulan serializes each single cell as a "cell sentence" of `(entity, value)` tuples — gene symbols *and* metadata terms (organ, tissue, age, sex, sequencing tech, cell type) and task-prompt tokens — and trains a 368M-parameter GPT-style decoder on **hECA-10M** under a unified generative objective.
- Headline result on held-out cell-type annotation: **scMulan (zero-shot) reaches Acc / weighted-F1 = 0.917 / 0.937 on AHCA_BoneMarrow, 0.927 / 0.934 on Simonson2023, and 0.873 / 0.894 on Su2022**, beating fine-tuned scGPT, Geneformer, and Celltypist on every dataset.
- The "zero-shot" framing is partly confounded — cell-type labels were already present in pretraining as `W_M` metadata tokens, so scMulan saw the supervision Celltypist / scGPT / Geneformer had to be fine-tuned to learn. There is no marker-recall or hallucination test for the conditional-generation half of the paper, and the model's intestine zero-shot Macro-F1 collapses to 0.25 on an organ absent from pretraining.

## Motivation

Prior single-cell foundation models (scBERT, Geneformer, scGPT, scFoundation) treat pretraining as masked gene-expression prediction over numeric matrices, discarding the textual metadata — organ, donor demographics, sequencing protocol, cell-type label — that accompanies every public dataset. The Kedzierska et al. 2023 audit found that these models mostly behave as encoders requiring task-specific heads and fine-tuning, with weak zero-shot transfer.

scMulan's bet is the obvious one once you state it: serialize *both* gene expression and metadata into a single discrete-token "cell sentence", frame every downstream biology task as conditional generation under task-prompt tokens, and you should inherit the prompting behaviour NLP LLMs enjoy. The medical-AI payoff is direct — zero-shot cell-type annotation across organs (bone marrow, heart, liver) and conditional generation for augmenting rare disease cell populations.

## Core Innovation

- **Cell sentence (c-sentence).** Each cell becomes a sequence of `(entity, value)` tuples:
  - `W_G` = (gene_symbol, normalized_expression),
  - `W_M` = (metadata_term, 0) — organ, tissue, donor age/sex, sequencing tech, **cell type**,
  - `W_T` = (task_prompt, 0) — `<cell type annotation>`, `<cell generation>`, `<organ region prediction>`,
  - `W_S` = (special, 0), e.g. `#E#` end-of-sentence.
  All entities are natural-language terms ("Heart", "T cell", "CD3D"); non-gene words carry value 0.
- **Set-style generative objective.** Because gene order is biologically meaningless, the model predicts the *set of remaining entities* at each step (rather than a single next token in fixed order), genes are shuffled per epoch, and positional encodings are removed.
- **Dual entity + value embeddings; dual heads.** Input tokens are $h_0 = E_e(E_{\text{obs}}) + E_v(V_{\text{obs}})$; the decoder emits both an entity distribution (`head_e`, classification over the vocabulary) and a continuous expression value (`head_v`, MSE regression).
- **One model, three unified tasks.** Conditional cell generation, hierarchical (coarse + fine) cell-type annotation, and organ-region prediction are all framed as generation under different `W_T` prompts — no task-specific heads.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Precise zero-shot cell-type annotation across multiple organs | Table 1 — three held-out datasets, three metrics each | AHCA_BoneMarrow, Simonson2023, Su2022 | ⭐⭐ |
| C2 | Beats Celltypist / scGPT / Geneformer on annotation | Table 1; consistent margins on Acc / Prec / F1 | Same three | ⭐⭐ |
| C3 | Better fine-tuning sample efficiency than scGPT | Figure 3 — scaling curve from 20% → 100% | Intestine_HCL_55k only | ⭐ |
| C4 | Fast and accurate batch integration without fine-tuning | Figure 4 — AvgBIO / AvgBATCH and wall-clock | Lung (scIB), COVID-19 immune | ⭐⭐ |
| C5 | Can conditionally generate realistic single-cell expression profiles | Figure 5 — joint UMAP, Q-Q plots of expression and sparsity | hECA-10M-derived prompts vs hECA-10M reals | ⭐ |
| C6 | Generalizes to novel cell types not in pre-training | §3.1 paragraph + Figure S1/S2 commentary | Anecdotal — no quantitative novel-type recall | ⭐ |
| C7 | Learns biologically meaningful markers (interpretability) | §3.5 — list of canonical markers recovered | Figure S5 only | ⭐ |
| C8 | "First foundational model capable of multiple tasks simultaneously without fine-tuning or external layers" in single-cell | Discussion paragraph | None — definitional / framing claim | ⭐ |

**Honest read.**

- **C2 is partially confounded.** scMulan is "zero-shot" only in the sense that no further gradient steps occur on the test set — but its *pretraining* corpus (hECA-10M) explicitly contains the cell-type labels as `W_M` tokens under multiple task prompts. Celltypist / scGPT / Geneformer were fine-tuned on a *sampled subset* of hECA-10M, which differs in label distribution and volume. A fair head-to-head would have given baselines the same labels and supervision budget. That experiment is not run.
- **C3 is one-dataset evidence.** Sample-efficiency claims need at least 2–3 tissues to be credible. With only intestine, the curve could be tissue-idiosyncratic.
- **C1 is undermined by C3's other half.** Zero-shot Macro-F1 on intestine — an organ absent from hECA-10M — is only **0.25**. Zero-shot only works when the target organ is already in pretraining.
- **C4's wall-clock advantage is the cleanest result.** Single-digit minutes vs ~10 h fine-tune for scGPT is unambiguous.
- **C5 is qualitative.** No FID-equivalent, no train-classifier-on-synthetic baseline, no marker-gene recall on generated cells. Generated cells are admitted to be **systematically sparser** than real cells (premature `#E#` emission). Hallucination — whether prompting a rare organ-cell-type pair produces plausible but biologically wrong markers — is not measured.
- **C6 is essentially unsupported.** The paper concedes misannotations cluster on out-of-vocabulary cell types, which directly contradicts a strong zero-shot generalization claim.
- **C7 is anecdotal.** No quantitative overlap with PanglaoDB / CellMarker, no top-k marker recall.
- **No variance, no statistical tests.** Every reported number is a single run; no bootstrapped CIs on Table 1, no seed sweep on the fine-tuning curve.
- **scFoundation, from the same group, is cited in the introduction but never benchmarked.**

## Method & Architecture

![scMulan overview](/assets/images/paper/scmulan/fig_p003_01.png)

*Figure 1: scMulan overview. A cell is serialized into a c-sentence of (entity, value) tuples mixing gene expression, metadata, and task prompts; a GPT-style decoder with dual entity / value heads is pre-trained on hECA-10M under three generative tasks unified by the prompt token `W_T`.*

The transformer is a 24-layer decoder with ~368M parameters. Two architectural changes vs vanilla GPT:

- **Dual embedding layer.** Each input token sums an entity embedding and a value embedding: $h_0 = E_e(E_{\text{obs}}) + E_v(V_{\text{obs}})$. This lets the same `CD3D` token carry expression value 2.1 in one cell and 0 in another without two-stage tokenisation.
- **Dual prediction heads.** `head_e` is a classification MLP over the full vocabulary; `head_v` is an MSE regression head for the predicted expression value. Total loss is

  $$L = L_1 + \lambda L_2, \quad L_1 = -\sum_i \log P\!\left(\{E \setminus \{e_1,\dots,e_i\}\} \mid e_{\leq i}, v_{\leq i}; \theta\right), \quad L_2 = \frac{1}{|E_{\text{unobs} }|}\sum_{e_j \in E_{\text{unobs} }}\!\!\big(v_j^{\text{true} } - v_j^{\text{pred} }\big)^2.$$

  $L_1$ is the *set-prediction* analogue of next-token cross-entropy; genes are randomly shuffled each epoch to enforce order-insensitivity, and positional encodings are removed for input entities.

At inference, the model produces a distribution over the entity vocabulary, samples the next token from `head_e`, and reads its expression from `head_v`; generation terminates on `#E#`. Cell embeddings for batch integration are the final-layer hidden state when the input ends in `<cell type annotation>`.

**Pre-training corpus: hECA-10M.** A 10M-cell subset of hECA (Chen et al. 2022) covering human heart, brain, lung, liver, bone marrow, blood, and thymus. Composition is heavily skewed — brain and lung dominate, with bone marrow / blood / liver / heart / thymus far smaller — a bias the paper does not foreground.

## Experimental Results

### Zero-shot cell-type annotation

![Zero-shot annotation UMAPs](/assets/images/paper/scmulan/fig_p007_01.png)

*Figure 2: Zero-shot cell-type annotation on AHCA_BoneMarrow, Simonson2023, and Su2022. Predicted labels (bottom) closely match the manually curated labels (top) without any fine-tuning on the held-out sets.*

| Method | AHCA_BoneMarrow (Acc / Prec / F1) | Simonson2023 (Acc / Prec / F1) | Su2022 (Acc / Prec / F1) |
|---|---|---|---|
| **scMulan (zero-shot)** | **0.917 / 0.961 / 0.937** | **0.927 / 0.947 / 0.934** | **0.873 / 0.963 / 0.894** |
| Celltypist | 0.600 / 0.859 / 0.665 | 0.865 / 0.942 / 0.895 | 0.344 / 0.909 / 0.420 |
| scGPT (fine-tuned) | 0.686 / 0.928 / 0.740 | 0.877 / 0.938 / 0.898 | 0.792 / 0.878 / 0.830 |
| Geneformer (fine-tuned) | 0.793 / 0.798 / 0.796 | 0.588 / 0.939 / 0.635 | 0.742 / 0.911 / 0.760 |

scMulan tops every cell, sometimes by ~0.49 F1 over Celltypist on Su2022. But — once more — scMulan was pretrained on hECA-10M *with* the cell-type tokens; the baselines were fine-tuned on a different sampled subset. The gap measures both architecture/objective and supervision-volume difference.

### Fine-tuning on a held-out organ

![Fine-tuning sample efficiency](/assets/images/paper/scmulan/fig_p008_01.png)

*Figure 3: Fine-tuning sample efficiency on the held-out intestine (Intestine_HCL_55k). scMulan matches fully-fine-tuned scGPT / Celltypist baselines with only 40–60% of the training samples. Critically: its zero-shot Macro-F1 on this novel tissue is only 0.25 — zero-shot only works when the organ is in pretraining.*

At 100% training fraction scMulan reaches Macro-F1 ≈ 0.79 vs scGPT 0.70, Celltypist 0.65, Geneformer 0.62. With 40% of the data it already matches fully-fine-tuned scGPT on accuracy. Without any intestinal fine-tuning, Macro-F1 is **0.25** — a hard caveat against any "general-purpose zero-shot annotator" framing.

### Batch integration (and the speed argument)

![Batch integration](/assets/images/paper/scmulan/fig_p009_01.png)

*Figure 4: Batch integration on the COVID-19 immune dataset (Lotfollahi et al. 2022). scMulan (zero-shot, right column) achieves the highest AvgBIO score, with AvgBATCH comparable to fine-tuned scGPT, and runs in single-digit minutes versus the ~10 h fine-tune + inference required by scGPT.*

On the COVID-19 immune set scMulan (zero-shot) tops AvgBIO over fine-tuned scGPT, scANVI, and scVI; AvgBATCH matches scGPT_finetune. On the simpler Lung scIB set it lands third behind scGPT_finetune and scANVI. The robust win is wall-clock — single-digit minutes vs ~10 h for scGPT.

### Conditional cell generation

![Conditional generation](/assets/images/paper/scmulan/fig_p010_01.png)

*Figure 5: Conditional cell generation. 18,000 generated cells across 140 organ-cell-type prompts co-locate with 18,000 sampled real cells on a joint UMAP, and per-gene mean expression matches almost perfectly. But the sparsity Q-Q plot sits above the line of identity — generated cells are systematically sparser than real cells, which the authors attribute to premature `#E#` emission.*

Headline qualitative figure, but no marker-recall, no train-classifier-on-synthetic, no out-of-distribution prompt test (e.g., "neuron in bone marrow") to probe hallucination. The sparsity gap is named but unfixed.

### Interpretability

Saliency-gene analysis recovers canonical markers — ACTA2 / MYH11 → smooth muscle, DCN → fibroblast, NRXN3 → neuron, CD79A → B cell. No quantitative recovery metric (e.g., top-k overlap with PanglaoDB) is reported.

## Limitations

**Acknowledged:**

- Misannotations concentrate on cell types absent from hECA-10M (e.g., secretory cell in bone marrow).
- Conditional-generation cells have systematically higher sparsity than real cells.
- Zero-shot evaluation conventions in single-cell are still immature, so baselines aren't perfectly apples-to-apples.

**Not addressed:**

- **Human-only and organ-skewed pretraining.** No mouse, no kidney, pancreas, skin, GI.
- **Hallucination.** Prompting a rare or impossible organ-cell-type pair could still yield plausible-looking but biologically wrong markers — untested.
- **Quantitative generation quality.** No marker-gene recall on generated cells, no perturbation-response experiment, no synthetic-only classifier baseline.
- **Vocabulary growth.** Adding a new metadata category requires retraining the entity embedding — transfer cost unknown.
- **Label-alignment subjectivity.** Manual mapping of every test-set ontology onto hECA-10M's is a degree of freedom that could inflate metrics; the remapping is not blinded.
- **No variance reporting.** Single-run numbers throughout; no bootstrapped CIs, no seed sweep.
- **No comparison to Cell2Sentence** (Levine et al. 2023), the closest conceptual baseline — also a "cell-as-text" formulation.
- **Parameter-count discrepancy.** Abstract says 368M; discussion says "over 350M" — minor, but suggests light auditing.
- **bioRxiv preprint, not peer reviewed.** No external replication noted.

## Why It Matters for Medical AI

If the generative formulation holds up beyond the held-out hECA-10M organs, scMulan is a credible template for a *clinical* single-cell foundation model: zero-shot cell-type annotation across organs without per-tissue fine-tuning would meaningfully shorten the time from sequencing a patient sample to a labelled atlas. The wall-clock advantage on batch integration — minutes vs hours — directly addresses a real cost bottleneck in clinical scRNA pipelines. The conditional-generation half is more aspirational: rare-cell augmentation for disease atlases is the right north star, but a real medical deployment would need hallucination tests and marker-recall metrics that the paper does not run. Take the annotation results as a strong sufficient-conditions demonstration on in-distribution organs, and treat everything novel-organ or generation-quality as an open problem.

## References

- Paper (bioRxiv): <https://www.biorxiv.org/content/10.1101/2024.01.25.577152>
- DOI: `10.1101/2024.01.25.577152`
- Code & pre-trained weights: <https://github.com/SuperBianC/scMulan>
- hECA atlas (pretraining source): Chen et al. 2022, *iScience*
- Related work — Cell2Sentence (Levine et al. 2023), scGPT (Cui et al. 2024), Geneformer (Theodoris et al. 2023), scBERT (Yang et al. 2022), scFoundation (Hao et al. 2023)
- Zero-shot audit reference: Kedzierska et al. 2023
