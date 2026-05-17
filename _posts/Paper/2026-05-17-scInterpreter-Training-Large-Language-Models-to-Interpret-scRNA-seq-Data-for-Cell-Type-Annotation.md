---
title: "scInterpreter: Training Large Language Models to Interpret scRNA-seq Data for Cell Type Annotation"
excerpt: "Projects per-gene GPT-3.5 text embeddings of a cell's top-2048 expressed genes through an MLP into frozen LLaMA-13B as soft tokens for cell-type classification — reported to beat GenePT 'by a huge margin' on two undocumented in-house datasets, but with bar charts only, no numeric table, no ablations, and no random-init/no-LLM control."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/scinterpreter/
tags:
  - scInterpreter
  - scRNA-seq
  - Cell-Type-Annotation
  - LLaMA-13B
  - GenePT
  - Soft-Prompting
  - Single-Cell
  - LLM-for-Biology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- scInterpreter encodes every gene's NCBI description with OpenAI `text-embedding-ada-002`, projects the top-2048 expressed genes of each cell into LLaMA-13B's 5120-d hidden space through an MLP, prepends the instruction "what is the cell type of this given embedding?" plus a trainable class token, and reads out the class-token hidden state for cross-entropy classification — **LLaMA-13B stays frozen**; only the projector, class-token embedding, and classification head are trained.
- On two author-constructed datasets (**HUMAN-10k**: 10,000 cells, 61 types, 23,111 genes; **MOUSE-13k**: 13,000 cells, 37 types, 27,443 genes), scInterpreter is reported to outperform GenePT "with a huge margin" across accuracy / precision / recall / F1.
- **The headline gap is read off a bar chart.** The paper contains no numeric results table, no error bars, no seeds, no ablation, no comparison to scGPT / Geneformer / scFoundation / GeneCompass, and no held-out / cross-dataset test. Eyeballing Fig. 2 the gap is roughly **~0.30 → ~0.85 accuracy on HUMAN-10k** and **~0.30 → ~0.90 on MOUSE-13k**.

## Motivation

Single-cell foundation models that operate on raw counts — scGPT, scFoundation, Geneformer, GeneCompass — ignore the prior knowledge encoded in gene-description text and in general-purpose LLMs. **GenePT** (Chen & Zou, 2023) showed that gene embeddings derived from ChatGPT alone are a hard-to-beat baseline for single-cell tasks, but it uses the LLM only at the *embedding* stage: the LLM never actually reads the cell. The authors argue this leaves the LLM's relational/world knowledge on the table and propose to inject per-gene text embeddings *into* a frozen LLaMA-13B as soft prompts, letting the transformer aggregate them in-context before classification. The medical-AI relevance is indirect: cell-type annotation is the workhorse step for disease atlases, tumour-microenvironment analysis, and developmental-biology studies, so a stronger annotator feeds many downstream pipelines.

## Core Innovation

- **Per-gene text embeddings as soft tokens.** Each gene is represented by `ada-002(NCBI description)`, a 1536-d vector cached once. Cells become sequences of these gene embeddings — ordered by expression, top-2048 kept.
- **MLP into LLaMA's hidden space.** A trainable MLP maps each 1536-d gene embedding to 5120-d, the LLaMA-13B hidden size, giving an `n × h` matrix of soft tokens per cell.
- **Frozen LLM as in-context aggregator.** The full prompt — `instruction tokens ⊕ cell soft tokens ⊕ class token` — is run through frozen LLaMA-13B. The hidden state at the class-token position is the cell embedding fed to the classifier head.
- **Identical gene-embedding initialisation to GenePT.** Because the baseline and the proposed method share the same `ada-002` gene embeddings, any quality gap is attributed by the authors to the LLM's "common knowledge" rather than to the embedding source.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|-------|----------|------------|----------|
| C1 | LLMs can be adapted to *interpret and distinguish* cell types from scRNA-seq via the proposed architecture (abstract). | Method §2 + Fig. 1 pipeline; Figs. 2–4 show downstream classification works. | HUMAN-10k, MOUSE-13k | ⭐⭐ — architecture is plausible and a result exists, but "interpret" is overstated: LLaMA is frozen and the only trained components (projector + classifier head + class token) could plausibly work without the LLM. No ablation isolates the LLM's contribution. |
| C2 | scInterpreter outperforms GenePT "with a huge margin" on both datasets across accuracy / precision / recall / F1 (§3.2). | Fig. 2 bar charts. | HUMAN-10k, MOUSE-13k | ⭐ — bars only; no numeric table, no error bars, no significance test, single seed, single split. Visual gap ≈ +0.55 accuracy but unverifiable. |
| C3 | "Common knowledge from the LLM provides a better-supervised signal" — i.e. the gain comes from LLaMA's prior, not the rest of the architecture (§3.2). | Argument from shared gene embeddings: GenePT and scInterpreter use the same initial `ada-002` vectors, so the architectural delta is the frozen LLaMA. | HUMAN-10k, MOUSE-13k | ⭐ — confounded. scInterpreter also adds a trainable per-cell sequence model (LLaMA stack with class-token attention) and a trainable per-cell projector; GenePT pools gene embeddings and runs logistic regression. The missing control — swap LLaMA for a random-init transformer of the same size, or remove the LLM and classify from the projected sequence — is never run. |
| C4 | The confusion matrix shows "significant improvement … attributable to sophisticated feature encoding and contextual understanding of the LLM" (§3.2). | Fig. 3 left vs right (qualitative). | MOUSE-13k | ⭐⭐ — qualitatively striking, but no numeric per-class breakdown; attribution to "contextual understanding" of LLaMA inherits C3's confound. |
| C5 | Markedly superior clustering: intra-class similarity and inter-class divergence (§3.3). | Fig. 4 UMAPs. | MOUSE-13k | ⭐⭐ — UMAPs are visually cleaner but no ARI / NMI / ASW reported and UMAP hyperparameters are not given. |
| C6 | scInterpreter demonstrates "the potential of LLMs as effective tools for uncovering new biological insights" (abstract). | Conclusion paragraph; not directly evidenced. | none | ⭐ — aspirational. The task is closed-world supervised classification against existing labels; no new biology is uncovered. |

## Method & Architecture

![scInterpreter pipeline](/assets/images/paper/scinterpreter/fig_p002_01.png)
*Figure 1: scInterpreter pipeline. NCBI gene descriptions are embedded by GPT-3.5 `text-embedding-ada-002`; the top-2048 expressed genes of each cell are projected into LLaMA-13B's 5120-d hidden space, prepended with the instruction tokens and a trainable class-token embedding, and the class-token hidden state after a frozen LLaMA-13B forward pass is fed to a classifier MLP.*

**Pipeline, step by step.**

1. **Per-gene description retrieval.** For every gene in the panel, pull the textual description $T_{des}$ from NCBI Gene.
2. **Per-gene text embedding (frozen).** Encode each description with OpenAI `text-embedding-ada-002`:

   $$e = f_{gpt}(T_{des})$$

   This yields a 1536-d embedding per gene. Cached once and identical to GenePT's initial gene embeddings.
3. **Per-cell sequence construction.** Rank genes by expression and keep the top-`n = 2048`. Project each gene embedding through a trainable MLP to LLaMA-13B's hidden size `h = 5120`:

   $$C = \mathrm{MLP}_p(e_1 \oplus e_2 \oplus \cdots \oplus e_n), \quad C \in \mathbb{R}^{n \times h}$$
4. **Prompt assembly.** The input is `instruction_embedding ⊕ C ⊕ class_token_embedding`, where the instruction is "what is the cell type of this given embedding?".
5. **Frozen LLaMA-13B forward pass.** ReadOut is defined as the output hidden state at the class-token position, $\hat{e}_{cls}$.
6. **Classification head.** $\hat{y} = \mathrm{Softmax}(\mathrm{MLP}_c(\hat{e}_{cls}))$. Trained with cross-entropy.
7. **Trainables.** Only the projector $\mathrm{MLP}_p$, the class-token embedding, and the classifier $\mathrm{MLP}_c$. **LLaMA-13B is frozen** — this is not LLM fine-tuning despite the title's "Training Large Language Models."

**Operational details not reported in the paper.** Number of epochs, batch size, learning rate, optimiser, MLP depth/width, train/val/test split, GPU type, training compute, random seeds, count normalisation, treatment of dropout / zero counts, behaviour for cells with fewer than 2048 expressed genes, and whether 2048 soft tokens × LLaMA-13B fits without quantisation or gradient checkpointing — none are disclosed.

## Experimental Results

### Datasets

| Dataset | Cells | Cell types | Genes per cell |
|---------|------:|-----------:|---------------:|
| HUMAN-10k | 10,000 | 61 | 23,111 |
| MOUSE-13k | 13,000 | 37 | 27,443 |

Both datasets are described only as "We construct two scRNA-seq datasets." **No source atlas, donor / tissue / study citation, accession ID, curation protocol, license, or split ratio is given.** The MOUSE-13k cell-type list (Exe Ectoderm, Notochord, Caudal Neurectoderm, Primordial Germ Cell, Erythroid1/2/3, Forebrain/Midbrain/Hindbrain, …) is consistent with a mouse gastrulation / early-embryo atlas (e.g. Pijuan-Sala et al. 2019), but the paper does not say so. No preprocessing pipeline (CPM / log1p / HVG / QC) is described beyond "ranked top-2048 expressed genes."

### Main comparison (bar charts only — no numeric table in the paper)

| Dataset | Metric | GenePT (eyeballed) | **scInterpreter (eyeballed)** |
|---|---|---:|---:|
| HUMAN-10k | Accuracy  | ~0.30 | **~0.85** |
| HUMAN-10k | Precision | ~0.25 | **~0.85** |
| HUMAN-10k | Recall    | ~0.30 | **~0.85** |
| HUMAN-10k | F1        | ~0.25 | **~0.85** |
| MOUSE-13k | Accuracy  | ~0.30 | **~0.90** |
| MOUSE-13k | Precision | ~0.30 | **~0.90** |
| MOUSE-13k | Recall    | ~0.30 | **~0.90** |
| MOUSE-13k | F1        | ~0.30 | **~0.90** |

*All values are read off Fig. 2 to the nearest 0.05; the paper provides no numeric table. Exact numbers are not recoverable from the manuscript. Single split, single seed, no error bars.*

### Confusion matrices on MOUSE-13k

![GenePT confusion matrix on MOUSE-13k](/assets/images/paper/scinterpreter/fig_p003_09.png)
*Figure 3 (left): GenePT confusion matrix on MOUSE-13k — predictions scatter off-diagonal across many cell types, with strong off-diagonal blocks for Blood Progenitors 2, Primordial Germ Cell, Spinal Cord, and Nascent Mesoderm.*

![scInterpreter confusion matrix on MOUSE-13k](/assets/images/paper/scinterpreter/fig_p003_10.png)
*Figure 3 (right): scInterpreter confusion matrix on MOUSE-13k — predictions concentrate on the diagonal; a few rare types (e.g. Parietal Endoderm) still show weak diagonals.*

### UMAP of cell embeddings on MOUSE-13k

![Initial cell embeddings UMAP](/assets/images/paper/scinterpreter/fig_p003_12.png)
*Figure 4 (left): Initial cell embeddings — cell types are entangled in a single blob.*

![GenePT cell embeddings UMAP after training](/assets/images/paper/scinterpreter/fig_p003_11.png)
*Figure 4 (middle): GenePT after supervised training — some separation appears but many cell types remain interspersed.*

![scInterpreter cell embeddings UMAP](/assets/images/paper/scinterpreter/fig_p003_13.png)
*Figure 4 (right): scInterpreter cell embeddings — tighter intra-class clusters and clearer inter-class separation, though related Erythroid1/2/3 types still overlap. No quantitative cluster metric (ARI / NMI / ASW) is reported.*

### Ablations / robustness

**None.** The paper does not ablate: whether the frozen LLaMA-13B forward pass actually helps (no "remove the LLM, classify directly from $C$" control); the choice of LLM (no LLaMA-7B vs LLaMA-13B; no Mistral / GPT comparison); `n = 2048`; projector depth/width; prompt wording; necessity of the class token versus mean pooling; frozen versus LoRA / unfrozen LLaMA; statistical variance (no seeds, no error bars, no significance test).

## Limitations

**Acknowledged by the authors.** The conclusion calls this a "preliminary stride" — explicitly framed as a proof-of-concept.

**Not acknowledged but visible.**

- **The title overstates the contribution.** "Training Large Language Models" is misleading — LLaMA-13B is frozen end-to-end; only a projector, a class-token embedding, and a classifier head are trained. This is soft-prompt adaptation, not LLM training.
- **No dataset provenance.** HUMAN-10k and MOUSE-13k are described as "constructed by the authors" with no atlas citation, accession ID, donor/tissue split, license, or release. Reproducibility from the manuscript alone is effectively zero.
- **Single comparison only.** GenePT is the sole baseline. No comparison to scGPT, Geneformer, scFoundation, GeneCompass, or even a vanilla MLP / logistic-regression baseline on top of the same `ada-002` embeddings.
- **No standard benchmarks.** No PBMC, Tabula Muris, Pancreas-cross-platform, scIB, or OpenProblems — so the result is not comparable to any other published cell-type-annotation number.
- **No statistical evidence.** Single run, single split, single seed, no error bars, no significance test, no per-class metrics, no macro-vs-micro F1.
- **Missing critical control.** A randomly initialised transformer of equal parameter count, or removing LLaMA entirely and classifying from the projected gene sequence, is the experiment that would isolate "what the frozen LLM is doing." It is never run, so the central conceptual claim — that LLaMA's *common knowledge* drives the gain — is unsupported.
- **No cross-dataset, batch-effect, donor-holdout, zero-shot, or few-shot evaluation** despite the "foundation model" framing.
- **No compute disclosure.** 2048 soft tokens × LLaMA-13B per cell across 10–13k cells is non-trivial; training/inference cost is undisclosed.
- **No code or data release** mentioned in the manuscript.
- **GenePT baseline implementation is not described** — official pipeline or re-implementation? With a reported ~0.55 accuracy gap, baseline quality is load-bearing and unauditable.

## Why It Matters for Medical AI

Cell-type annotation is the gateway step for almost every downstream single-cell analysis in disease atlases, immuno-oncology, and developmental biology. A method that lets a *frozen* general-purpose LLM consume per-gene text embeddings as soft prompts is appealing because (a) it avoids retraining a domain-specific transformer from raw counts and (b) it can in principle inherit improvements as base LLMs evolve. The architectural sketch here is worth knowing for that reason. But the current empirical story does not justify deployment: the headline number is read off a bar chart, the datasets have no provenance, and the experiment that would tell us whether the LLM's prior is doing the work — versus the trainable projector + class-token + classifier head alone — is missing. For medical pipelines where cell-type miscalls propagate into clinical interpretation, the bar should be higher than a single in-house comparison.

## References

- **Paper (arXiv).** Li, Xiao, Wang, Feng, Li, Zhou. *scInterpreter: Training Large Language Models to Interpret scRNA-seq Data for Cell Type Annotation.* arXiv:2402.12405v1 (q-bio.GN), 18 Feb 2024. Also published as a 4-page Letters-track communication in *Frontiers of Computer Science*, 2024.
- **GenePT (the sole baseline).** Chen, Y. T., Zou, J. *GenePT: A simple but hard-to-beat foundation model for genes and cells built from ChatGPT.* bioRxiv, 2023-10.
- **Single-cell foundation models the paper does *not* compare to.** scGPT (Cui et al., 2023); scFoundation (Hao et al., 2023); Geneformer (Theodoris et al., *Nature*, 2023); GeneCompass (Yang et al., 2023).
- **Base LLM.** Touvron et al., *LLaMA: Open and efficient foundation language models*, arXiv:2302.13971, 2023.
- **Gene text embeddings.** OpenAI `text-embedding-ada-002`, used here for the per-gene NCBI description encoder.
- **Probable MOUSE-13k source (not cited in the paper; inferred from cell-type list).** Pijuan-Sala et al. *Mouse gastrulation atlas*, 2019.
