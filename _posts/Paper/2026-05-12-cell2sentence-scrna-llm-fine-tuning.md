---
title: "Cell2Sentence: Teaching Large Language Models the Language of Biology"
excerpt: "Fine-tuning GPT-2/Pythia on rank-ordered gene-symbol 'cell sentences' beats scGPT/scVI/scGen/scDiffusion on conditional cell generation (k-NN@10 0.2746, GW 54.30 on immune tissue) — but the load-bearing claim is NL pretraining, not rank encoding: NL+C2S doubles cell-type prediction accuracy (29.33 → 69.95) over C2S-only at the same parameter count."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/cell2sentence-scrna-llm-fine-tuning/
tags:
  - Cell2Sentence
  - C2S
  - scRNA-seq
  - LLM-for-Biology
  - GPT-2
  - Pythia
  - Single-Cell
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- C2S flattens a single-cell RNA-seq profile into a "cell sentence" — gene symbols rank-ordered by decreasing expression — and fine-tunes a vanilla GPT-2 / Pythia-160M on the result, so the same model handles cell generation, cell-type prediction, and abstract summarization without a custom tokenizer.
- Headline result on the immune-tissue benchmark (17,500 held-out cells, 5 sample repeats): **C2S (Pythia-160M) reaches k-NN@10 = 0.2746 ± 0.0073 and Gromov-Wasserstein distance = 54.30 ± 0.34**, beating scDiffusion (0.2368 / 72.02), scVI (0.2425 / 302.1), scGen (0.2377 / 315.9), and scGPT (0.1811 / 2989.8).
- The rank-symbol invertibility claim is real (log-linear regression recovers expression with **R² = 0.81 ± 0.07 across 127 datasets**), but it is a consequence of scRNA-seq's Zipf-like statistics — not unique to C2S. The downstream wins are most cleanly attributed to **natural-language pretraining**: NL+C2S roughly doubles direct cell-type prediction accuracy versus C2S-only at fixed parameter count (29.33 → 69.95 on GPT-2 small; 29.17 → 74.26 on GPT-2 medium).

## Motivation

Single-cell transcriptomics has its own zoo of specialized architectures (scVI, scGen, Geneformer, scGPT) that re-invent tokenization, attention, and training infrastructure on top of `c × n` count matrices. None of them inherit the rest of the language-model ecosystem — prompting, instruction tuning, Hugging Face tooling, natural-language metadata. The C2S premise is that if expression can be re-expressed as text **without a custom gene-ID tokenizer**, pretrained LLMs can absorb biology while still consuming and producing natural language. For medical AI that matters concretely: clinical labels, tissue annotations, and study abstracts arrive as text, so a single model that speaks both languages can do label prediction and report generation in one pass.

## Core Innovation

- **Rank-ordered gene symbols as a sentence.** Drop expression values; keep only the order of gene symbols sorted by decreasing log-normalized counts. The sentence is plain English to a tokenizer that already knows "CD8A" and "FOXP3" as multi-token strings.
- **Per-dataset log-linear inversion.** Fit `e_i = a_d · log(r_i) + b_d` on the training split. The saved `(a_d, b_d)` lets any generated sentence be decoded back to an expression vector without storing the original counts.
- **Twenty natural-language prompt templates** that interleave cell sentences with cell type, tissue, disease, and abstract text — turning every cell into instruction-style (prompt → response) training pairs.
- **A vanilla LLM fine-tune.** GPT-2 small/medium/large and Pythia-160M off Hugging Face, AdamW + cosine + FlashAttention-2, sequence length 1024 (GPT-2) or 9,200 (Pythia). No custom tokenizer, no domain pretraining objective.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Rank-ordered gene-symbol sentences carry enough signal that expression can be recovered with minimal loss | Fig. 3 scatter (Pearson R = 0.934, Spearman R = 0.815, R² = 0.815); Fig. 6 distribution R² = 0.81 ± 0.07 across 127 datasets; Fig. 7 per-dataset R² 0.68–0.95; Fig. 4 UMAP overlap | Immune tissue (Domínguez Conde) + 127 CellxGene datasets | ⭐⭐ |
| C2 | A vanilla LLM fine-tuned on cell sentences beats specialized scRNA generative models on conditional cell generation | Table 1: best k-NN@all-k and lowest GW (54.30 vs. next-best scDiffusion 72.02); scGPT and VAEs are 1–2 orders of magnitude worse on GW; 5 sample repeats with std | Immune tissue (single dataset) | ⭐⭐ |
| C3 | C2S generalizes to OOD bulk data (L1000, GTEx) for combinatorial label prediction | Table 2: partial-label ACC 0.631 (L1000), 0.575 (GTEx) — substantially above Geneformer / scGPT / XGBoost; 3 repeats with std. But full-label accuracy collapses to 15–20% across all methods, so the OOD claim holds only for partial credit | Cytokine PBMC + L1000 + GTEx | ⭐⭐ |
| C4 | C2S generates abstracts more aligned with ground truth than GPT-3.5 / Mixtral / Mistral baselines | Table 3: only C2S has p < 0.05 on T-test and KS test; MMD 0.198 (C2S) vs. 0.298 (GPT-3.5) vs. 0.639 (Mixtral) | Multi-tissue CellxGene, 19 held-out studies | ⭐⭐ |
| **C5** | **NL pretraining, not rank-symbol encoding per se, is the source of C2S's gains** | **Tables 5–6: NL+C2S vs. C2S-only at fixed parameter count — autoregressive cell-type k-NN@5 24.56 → 52.55 (GPT-2 small) and 26.36 → 54.67 (medium); direct cell-type prediction 29.33 → 69.95 (small) and 29.17 → 74.26 (medium)** | Immune subset (49,920 cells × 17 cell types) | ⭐⭐⭐ |
| C6 | Cell sentences enable "interpretation of single-cell data in natural language" / biological insight | Table 3 + qualitative Fig. 10. Statistical tests reach only p ≈ 0.003–0.02 on a 19-study eval; no expert evaluation, no factuality audit | Multi-tissue | ⭐ |

**The audit-worthy point.** The paper's pitch is that throwing away expression values and keeping only rank-ordered gene symbols is sufficient. Two lines of evidence support it — invertibility (R² = 0.81) and downstream wins versus expression-using baselines (scVI, scGen, scDiffusion) — but **neither is a direct head-to-head against using expression values under the same backbone**. The cleanest experiment ("fine-tune GPT-2 on (gene, expression-bin) pairs and compare to (gene-rank-only)") is not run. Per Tables 5–6 the dominant factor in cell-type performance is NL pretraining, not the rank-symbol encoding itself; the encoding is a feasibility result that lets the LLM in the door, but the work is being done by language-model priors over biology vocabulary.

## Method & Architecture

![Cell2Sentence overview: scRNA input flattened into rank-ordered gene symbol sentences, fine-tuned into a GPT-2/Pythia LLM, decoded back to expression](/assets/images/paper/cell2sentence/page_003.png)
*Figure 1: Cell2Sentence overview — single-cell expression profiles are flattened into gene-symbol sentences, fine-tuned into a GPT-2/Pythia LLM with text prompts, and decoded back to expression at inference.*

### 1. Preprocess counts

Drop cells with <200 genes expressed and genes detected in <200 cells; drop cells with >2,500 total counts or >20% mitochondrial reads (Scanpy defaults). Row-normalize to 10,000 and log-transform:

$$ C'_{i,j} = \log_{10}\!\left(1 + 10^4 \cdot \frac{C_{i,j} }{\sum_k C_{i,k} }\right) $$

### 2. Rank transform into a sentence

For each cell `i`, sort gene names by decreasing `C'_{i,j}` to get sentence `s_i`. Truncate to top-100 genes for GPT-2 (compute budget); Pythia-160M uses up to 9,200 tokens — effectively the full sentence.

### 3. Fit per-dataset invertibility model

With `r_i = log(rank(gene_i))` and `e_i = C'_{i,j}`, fit linear regression `e_i = a_d · r_i + b_d`. Save `(a_d, b_d)` per dataset so any generated sentence can be inverted to an expression vector. Duplicate generated genes get averaged ranks; unseen genes get zero.

### 4. Annotate with natural-language metadata

Prepend or append cell-type, tissue, disease, or abstract text using one of 20 prompt templates, producing instruction-style `(prompt, response)` pairs.

### 5. Fine-tune

GPT-2 small / medium / large and Pythia-160M from Hugging Face checkpoints. AdamW, cosine schedule, FlashAttention-2, batch 8 × grad-accum 16 (effective 128), learning rate 6e-4 with 1% warmup. Loss on labels only during fine-tuning; loss on prompt+label during from-scratch pretraining.

### 6. Inference and decoding

Sample with `top_p = 0.9`, `temperature = 0.7`. For conditional cell generation, prompt with cell type / tissue; for cell-type prediction or abstract generation, prompt with the cell sentence. To decode generated sentences, strip invalid gene symbols by regex, average ranks for duplicates while preserving positional ranks of invalid genes (a gene at position 4 stays at rank 4 even if position 3 was invalid), then apply `(a_d, b_d)` to recover expression.

![C2S pipeline detail showing rank-ordering, textual annotation, masked-LM-style training, three downstream task patterns, and the gene-expression reconstruction block](/assets/images/paper/cell2sentence/page_004.png)
*Figure 2: C2S pipeline detail. Rank-ordering converts each cell into a gene-symbol sentence; an LLM is fine-tuned autoregressively with mixed cell-sentence + metadata prompts; a per-dataset log-linear regression inverts generated sentences back to expression.*

## Experimental Results

**Experiment 1 — conditional cell generation (immune tissue, 500 cells/type × 35 types = 17,500; 5 sample repeats):**

| Model | k-NN@3 ↑ | k-NN@5 ↑ | k-NN@10 ↑ | k-NN@25 ↑ | GW dist ↓ |
|---|---|---|---|---|---|
| scGen | 0.2376 ± 0.0112 | 0.2330 ± 0.0093 | 0.2377 ± 0.0053 | 0.2335 ± 0.0041 | 315.95 ± 1.24 |
| scVI | 0.2436 ± 0.0062 | 0.2400 ± 0.0064 | 0.2425 ± 0.0034 | 0.2348 ± 0.0032 | 302.13 ± 0.93 |
| scDiffusion | 0.2335 ± 0.0125 | 0.2288 ± 0.0111 | 0.2368 ± 0.0067 | 0.2306 ± 0.0049 | 72.02 ± 0.39 |
| scGPT | 0.1838 ± 0.0086 | 0.1788 ± 0.0169 | 0.1811 ± 0.0149 | 0.1882 ± 0.0071 | 2989.81 ± 4.92 |
| **C2S (Pythia-160M)** | **0.2588 ± 0.0061** | **0.2565 ± 0.0060** | **0.2746 ± 0.0073** | **0.2715 ± 0.0070** | **54.30 ± 0.34** |

**Experiment 2 — combinatorial label prediction (3 repeats):**

| Model | Cytokine ACC | Cytokine AUROC | L1000 ACC | L1000 AUROC | GTEx ACC | GTEx AUROC |
|---|---|---|---|---|---|---|
| k-NN (partial) | 0.462 ± 0.0047 | 0.550 ± 0.0064 | 0.592 ± 0.0054 | 0.740 ± 0.0036 | 0.492 ± 0.0047 | 0.662 ± 0.0035 |
| XGBoost (partial) | 0.515 ± 0.0175 | 0.631 ± 0.0202 | 0.389 ± 0.0066 | 0.534 ± 0.0044 | 0.482 ± 0.0127 | 0.633 ± 0.0120 |
| Geneformer (partial) | 0.600 ± 0.0170 | 0.722 ± 0.0145 | 0.419 ± 0.0153 | 0.632 ± 0.0181 | 0.500 ± 0.0013 | 0.649 ± 0.0025 |
| scGPT (partial) | 0.419 ± 0.0001 | 0.500 ± 0.0000 | 0.334 ± 0.0076 | 0.500 ± 0.0000 | 0.270 ± 0.0522 | 0.500 ± 0.0000 |
| **C2S GPT-2 L (partial)** | **0.639 ± 0.0049** | **0.767 ± 0.0049** | **0.631 ± 0.0031** | **0.768 ± 0.0021** | **0.575 ± 0.0035** | **0.713 ± 0.0014** |
| **C2S GPT-2 L (full)** | **0.149 ± 0.0057** | 0.564 ± 0.0030 | **0.202 ± 0.0059** | 0.600 ± 0.0029 | **0.152 ± 0.0062** | 0.574 ± 0.0032 |

**Experiment 3 — abstract summary generation (embedding-space agreement with held-out abstracts):**

| Model | T-test ↑ | KS test ↑ | MMD ↓ | W ↓ |
|---|---|---|---|---|
| **C2S GPT-2 small** | **2.96, p=0.003*** | **0.35, p=0.023*** | **0.198 ± 0.004** | **0.414 ± 0.006** |
| **C2S GPT-2 large** | **2.49, p=0.013*** | **0.35, p=0.020*** | **0.202 ± 0.003** | **0.421 ± 0.004** |
| GPT-3.5-Turbo-1106 | 1.23, p=0.220 | 0.21, p=0.392 | 0.298 ± 0.004 | 0.490 ± 0.008 |
| Mixtral-8x7B-Instruct (AWQ) | −1.20, p=0.233 | 0.24, p=0.246 | 0.639 ± 0.016 | 0.544 ± 0.005 |
| Mistral-7B-Instruct | −8.64, p=0.384 | 0.23, p=0.299 | 0.754 ± 0.010 | 0.584 ± 0.004 |
| GPT-2 small (no FT) | 1.31, p=0.896 | 1.52, p=0.783 | 1.045 ± 0.009 | 0.752 ± 0.004 |
| GPT-2 large (no FT) | −1.44, p=0.885 | 1.81, p=0.581 | 0.939 ± 0.006 | 0.701 ± 0.016 |

**Ablations and robustness worth highlighting.** Tables 5–6 are the most informative panels in the paper: NL+C2S roughly doubles cell-type performance over C2S-only at the same parameter count, which is the strongest evidence that the LLM is doing more than memorizing rank statistics. Reconstruction quality across 127 datasets is Pearson R = 0.91 ± 0.04, Spearman R = 0.83 ± 0.05, R² = 0.81 ± 0.07. Gene-validity sanity check (Table 7): NL+C2S outputs are 99.6–99.7% valid HGNC symbols and 98.9–99.5% unique, so the LLM is generating plausible biological vocabulary, not gibberish. Generated-vs-real averaging (Table 4): GPT-2 small NL+C2S Pearson R = 0.984, R² = 0.949 — but this is class-averaged over 17 cell types, which suppresses per-cell variance.

## Limitations

Acknowledged by the authors:

- GPT-2 sequences are truncated to the top 100 genes "due to resource constraints" — Pythia-160M extends to 9,200 tokens but is still parameter-limited.
- 19% of expression variance is unaccounted for by the log-linear inversion.
- scGPT's conditional-generation code was unavailable; the authors reimplemented it from the paper text, which may underrate scGPT.

Noticed but not addressed:

- **No direct rank-vs-expression ablation under the same backbone.** The clean comparison — `(gene, expression-bin)` pairs vs. `(gene-rank-only)` — is missing, so we cannot tell whether discarding expression values helps, hurts, or is neutral.
- **No bottom-tail analysis.** The top-100 truncation removes rare and low-expressed genes whose rank↔expression relationship is also the noisiest; how much biological signal is lost (e.g., perturbation markers) is not quantified.
- **Generation evaluation is class-averaged.** Table 4 reports R² = 0.949 between averaged generated and averaged real cells per cell type — per-cell variance and intra-class heterogeneity are not reported.
- **No external test for cell generation.** Experiment 1 holds out cells from the same immune dataset C2S was trained on; cross-dataset / cross-tissue generalization of conditional generation is not measured.
- Variance is reported with only 3–5 repeats, and Table 2 std on scGPT accuracy is sometimes 0.0001 (with AUROC pinned at 0.500), consistent with a degenerate constant predictor — the C2S margin over Geneformer / XGBoost is the meaningful comparison, and that margin is 4–10 ACC points, not dominance.
- Per-dataset `(a_d, b_d)` must travel with each sentence to invert generation — an extra metadata coupling the "flexible / modular" framing glosses over.
- Gene-symbol drift across species, releases, and alias systems (HGNC vs. Ensembl) is not discussed.
- Significance in Table 3 is borderline (p = 0.003, 0.013, 0.020, 0.023) on a 19-study eval — fragile under resampling.

## Why It Matters for Medical AI

Clinical metadata, tissue labels, disease annotations, and full-text abstracts are all language; expression matrices are not. C2S is the first plausible demonstration that the same pretrained LLM can ingest both, fine-tuned only on cheap rank-ordered text, and emit either cells from text prompts or text summaries from cells. The pragmatic implication for medical-AI tooling: report-generation-style pipelines on scRNA-seq cohorts no longer need a bespoke encoder for the omics arm — a GPT-2-scale model with NL pretraining and a per-dataset linear decoder is competitive with specialist baselines on the tasks measured here. The caveat for clinical deployment is the same set of unaddressed questions above: no expert evaluation of generated abstracts, no factuality audit, no cross-cohort generalization test, and full-label accuracy that collapses on combinatorial OOD tasks. C2S is a strong feasibility result for "LLMs as scRNA backbones," not yet a production claim.

## References

- Paper: [Cell2Sentence: Teaching Large Language Models the Language of Biology (bioRxiv 2023.09.11.557287)](https://doi.org/10.1101/2023.09.11.557287)
- ICML 2024 version: published at ICML 2024 (Levine, Lévy et al., Yale CS / Yale Med / Penn / EPFL / USC Keck)
- Related: Cui et al., *scGPT* (Nature Methods 2024); Theodoris et al., *Geneformer* (Nature 2023); Lopez et al., *scVI* (Nature Methods 2018); Lotfollahi et al., *scGen* (Nature Methods 2019); Luo et al., *scDiffusion* (2024).
- Background on the rank↔expression Zipf-like law: Furusawa & Kaneko (2003); Qiu et al. (2013).
