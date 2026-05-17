---
title: "scELMo: Embeddings from Language Models are Good Learners for Single-cell Data Analysis"
excerpt: "GPT-3.5 text embeddings of gene/cell summaries, aggregated with an expression-weighted average, match scGPT on cross-source PBMC cell-type annotation (Acc 0.924 / F1 0.811) using only a light contrastive adaptor — no scRNA pre-training."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/scelmo/
tags:
  - scELMo
  - GenePT
  - GPT-3.5
  - Single-cell
  - Foundation-Model
  - LLM
  - Cell-Type-Annotation
  - Batch-Correction
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- scELMo replaces single-cell foundation-model pre-training with **GPT-3.5 text embeddings of LLM-written gene / protein / cell-type summaries**, aggregated into a cell embedding by `e_cells = AVG(X) e_f + e_c`. The novel ingredient over GenePT is the **expression-weighted average (`wa`)** `X_i / sum(X_i)` in place of GenePT's uniform mean (`aa`).
- On the cross-source **PBMC cell-type annotation benchmark (Table 1, fine-tuned)** scELMo+GPT-3.5 reaches **Accuracy 0.924 / F1 0.811**, vs scGPT 0.933 / 0.807 and Geneformer-finetuned 0.235 / 0.131 — parity with a full scRNA-seq foundation model using only a small contrastive adaptor on a 1× A5000.
- The same Table 1 contains the paper's most uncomfortable result: **scELMo + random gene embeddings hits F1 0.835 on PBMC, beating GPT-3.5 (0.811)**, and matches it on hPancreas / Aorta. The fine-tuning win is largely the adaptor — not the LLM. We will keep this in view throughout.

## Motivation

Single-cell "foundation models" (scGPT, Geneformer, scBERT) burn millions of cells of pre-training compute, and the authors' own prior work argues they rarely beat task-specific baselines once tuning is controlled. GenePT showed that an LLM's text embedding of a gene's NCBI summary already carries enough biological signal to cluster cells without sequencing pre-training. scELMo's pitch is that GenePT under-uses LLMs in three ways:

- **Text source is rigid.** NCBI summaries are sparse, inconsistently formatted, and missing for many features.
- **GenePT-s does not scale.** Embedding a per-cell ranked-gene sentence hits OpenAI rate limits and ignores scRNA sparsity structure.
- **Aggregation ignores magnitude.** GenePT averages gene embeddings uniformly per cell, throwing away expression weights.

The medical-AI angle is that the same pipeline plus a small contrastive adaptor can be reused for in-silico therapeutic-target screening (DCM/HCM, ascending aortic aneurysm) without retraining a transformer per disease.

## Core Innovation

- **LLM-as-feature-extractor for omics.** Query GPT-3.5 with `"Please summarize the major function of a gene: <SYMBOL>. Use academic language in one paragraph and include pathway information."`, embed the response → `e_f`. Same template (`cell type: ...`, `disease: ...`) for `e_c`.
- **Expression-weighted aggregation (`wa`).** `AVG_wa(X_i) = X_i / sum(X_i)` — each gene's contribution to a cell embedding is weighted by its normalized expression in that cell. GenePT = LLM-replaced-by-NCBI + uniform `aa`.
- **Light contrastive adaptor.** A ReLU MLP `T` trained with `L = L_classifier + λ·L_contrastive`, λ=100. Inputs are gene embeddings + raw expression; output is a cell embedding consumed by kNN. The same adaptor is reused unchanged for in-silico treatment scoring.
- **Embedding injection into existing tools.** No new architecture — `e_cells` / `e_f` are spliced into CINEMA-OT (replaces PCA), CPA (concatenated into CVAE latent via a learnable head), and GEARS (added to GEARS's native gene embeddings).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | scELMo handles clustering, batch correction, and cell-type annotation **without training a new model** (zero-shot) | Fig. 2c, Fig. 3, Table 1 zero-shot rows | hPancreas, Aorta, PBMC-3k, CITE-seq, CyTOF | ⭐⭐ — true for hPancreas/Aorta, **false for cross-source PBMC** where zero-shot collapses to Acc 0.190 |
| C2 | "Lighter structure and lower resource requirement but impressive performance" | "1× A5000 ≤30 GB" assertion; Aorta F1 0.946 vs scGPT 0.937 | hPancreas, Aorta, PBMC | ⭐⭐ — resource side asserted, not measured; performance side real on Aorta, tied on PBMC, lost on hPancreas zero-shot |
| C3 | `wa` > `aa` for downstream tasks | Fig. 2c (clustering), Fig. 3 (batch correction), Ext. Fig. 3a | 4 clustering + 3 batch-correction datasets | ⭐⭐ — supported for clustering/batch; **identical to `aa` on cell-type annotation** (Table 1) to 3 decimals |
| C4 | GPT-3.5 embeddings carry meaningful biological function | Ext. Fig. 2a,b — UMAP + kNN label propagation on Geneformer's gene-function set | gene-function set | ⭐⭐ — visual + one kNN classifier, no quantitative purity score, no held-out F1 |
| C5 | Fine-tuning identifies novel therapeutic targets (DCM/HCM, aortic aneurysm) | Fig. 4 + Supp. File 2 literature search; one CRISPR-validated gene (GSN) | Heart, Aorta | ⭐ — `Δcos` of 1e-4–4e-3 with **no permutation test or CI**; "validation" is post-hoc literature search; GSN was already Geneformer's anchor |
| C6 | scELMo improves CINEMA-OT / CPA / GEARS via embedding injection | Fig. 5a–d | ChangYe2021, perturbed PBMC, CPA example, Openproblems, Dixit, Adamson, Norman | ⭐⭐ — CINEMA-OT and CPA-on-Openproblems improve; **CPA-example median drops with GPT-3.5**; GEARS gain on Adamson/Norman "not significant" by authors' own words |
| C7 | LLM choice matters; GPT-3.5 is the right pick (vs GPT-2/4, LLaMA-2, Mistral, BioGPT, Claude 2, PaLM-2) | Fig. 2a,b "proportion of meaningful outputs" + query time, 20 genes + 20 proteins, manual grading | n=40 | ⭐ — sample size 40, grading rubric not given, no inter-rater agreement, no statistical test |
| C8 | scELMo extends GenePT for **multi-omic** data (Abstract & §2) | Appendix C / Ext. Fig. 6 | scRNA + scATAC unpaired | ⭐ — **Appendix C directly contradicts the abstract**: "scELMo is not capable of multi-omic data integration under the zero-shot learning framework" |
| C9 | The LLM embeddings are what drive the fine-tuning win | Table 1 fine-tuning rows including scELMo+random emb | hPancreas, Aorta, PBMC | ⭐ — **the random-embedding ablation falsifies a strong reading of this claim**: random vectors match (and on PBMC F1 *beat*) GPT-3.5 embeddings inside the same adaptor |

**Honest synthesis.** The two defensible contributions are (1) the `wa` aggregation, a clean fix to GenePT's most obvious weakness, and (2) the demonstration that a *tiny* contrastive adaptor over off-the-shelf LLM embeddings reaches parity with full scRNA-seq foundation models on cell-type annotation. Everything else is weaker than the abstract implies. The hallucination audit (n=40) is undersized for a paper whose pitch hinges on LLM choice. The "multi-omic" claim is contradicted by the paper's own appendix (C8). The in-silico-treatment effect sizes lack any statistical envelope around `Δcos` smaller than 1 part in 1,000 (C5). And the random-embedding ablation in Table 1 (C9) quietly says **the adaptor — not the LLM — is doing most of the work** in the fine-tuned regime; a paper titled "Embeddings from Language Models are Good Learners" should either retract or directly address this.

## Method & Architecture

![scELMo overall pipeline — zero-shot and fine-tuning frameworks built on GPT-3.5 text embeddings of gene/cell summaries](/assets/images/paper/scelmo/page_004.png)
*Figure 1: scELMo end-to-end pipeline. (a) Zero-shot route — LLM text embeddings × expression matrix → cell embeddings consumed directly by clustering, batch correction, and kNN annotation. (b) Fine-tuning route — a small contrastive adaptor is learned on top of the same embeddings and reused for cell-type classification, in-silico therapeutic-target screening, and perturbation analysis.*

The core computation is the cell-embedding aggregation:

$$ e_{\text{cells}} = \mathrm{AVG}(X)\, e_f + e_c, \quad \mathrm{AVG}_{aa}(X_i) = X_i / m,\ \mathrm{AVG}_{wa}(X_i) = X_i / \mathrm{sum}(X_i). $$

`wa` is the new ingredient; `e_c` is an optional cell-state term. GenePT corresponds to `LLM → NCBI text` plus `aa`.

**Pipeline steps.**

1. **LLM selection (Fig. 2a,b).** Benchmark GPT-2/3.5/4, LLaMA-2-70B, Mistral, BioGPT, Claude 2, PaLM-2 on 20 genes + 20 proteins by manually scoring text correctness against GeneCards/NCBI. GPT-3.5 wins the accuracy-vs-query-time trade-off; all downstream embeddings come from GPT-3.5's embedding endpoint. Repeat-embedding stability corr ≥ 0.9.
2. **Build `e_f` / `e_c`.** Prompt GPT-3.5 per gene/protein/cell-type/disease, embed the response.
3. **Aggregate to `e_cells`** with `wa` (or `aa`).
4. **Zero-shot consumers.** Leiden clustering, scIB `S_bio`/`S_batch` batch correction, kNN cell-type annotation (k=10, inherited from GenePT).
5. **Fine-tuning adaptor.** ReLU MLP trained with combined classifier + contrastive loss; λ=100. The output cell embedding feeds kNN.
6. **In-silico treatment.** For diseased vs control cells, `CS = cos(mean(e_disease), mean(e_control))`. Zero a gene's expression, recompute, take `Δ = CS_new − CS_old`; flag as a target if `Δ > 1e-4`. Restricted to top-10 DEGs per condition.
7. **Perturbation injection.** Splice `e_cells` / `e_f` into CINEMA-OT (replaces PCA), CPA (concat into CVAE latent), GEARS (concat with native gene embeddings). No retraining of these tools.
8. **Compute footprint.** 1× NVIDIA A5000 (≤30 GB) for fine-tuning. Dollar cost is OpenAI API calls (one per gene/cell-type). No GPU pre-training.

![scELMo LLM hallucination audit and clustering results — GPT-3.5 wins the accuracy/query-time trade-off and `wa` ranks first across clustering benchmarks](/assets/images/paper/scelmo/page_005.png)
*Figure 2: (a) Per-LLM proportion of "meaningful" gene/protein descriptions on the n=40 audit set; (b) per-LLM query time; (c) clustering NMI/ARI/ASW on hPancreas, Aorta, and PBMC-3k for `wa`, `aa`, GenePT, and PCA — `wa` ranks first overall.*

## Experimental Results

### Cell-type annotation (Table 1)

Numbers reproduced exactly from the paper. The paper's own method rows are bolded.

| Setting | Method | hPancreas Acc / F1 | Aorta Acc / F1 | PBMC Acc / F1 |
|---|---|---|---|---|
| zero-shot | GPT 2 query | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| zero-shot | GPT 4 query | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| zero-shot | scGPT (z) | 0.770 / 0.550 | 0.938 / 0.901 | 0.915 / 0.789 |
| zero-shot | Geneformer (z) | 0.500 / 0.270 | 0.860 / 0.620 | 0.126 / 0.095 |
| zero-shot | GenePT-w | 0.940 / 0.650 | 0.870 / 0.720 | 0.286 / 0.315 |
| zero-shot | GenePT-s | 0.890 / 0.560 | 0.860 / 0.620 | — / — |
| zero-shot | PCA | 0.633 / 0.357 | 0.929 / 0.910 | 0.436 / 0.285 |
| zero-shot | **scELMo / GPT 3.5 aa** | **0.933 / 0.629** | **0.877 / 0.719** | **0.190 / 0.230** |
| zero-shot | **scELMo / GPT 3.5 wa** | **0.933 / 0.629** | **0.877 / 0.718** | **0.190 / 0.231** |
| fine-tuned | scGPT | 0.970 / 0.741 | 0.962 / 0.937 | 0.933 / 0.807 |
| fine-tuned | Geneformer | 0.321 / 0.086 | 0.340 / 0.080 | 0.235 / 0.131 |
| fine-tuned | **scELMo + GenePT** | **0.963 / 0.680** | **0.958 / 0.946** | **0.919 / 0.801** |
| fine-tuned | **scELMo + random emb** | **0.968 / 0.680** | **0.951 / 0.882** | **0.903 / 0.835** |
| fine-tuned | **scELMo + GPT 3.5** | **0.966 / 0.731** | **0.956 / 0.938** | **0.924 / 0.811** |

Two reads matter here:

- **Headline.** In the fine-tuned regime, scELMo + GPT-3.5 ties scGPT on Aorta F1 (0.938 vs 0.937) and reaches PBMC F1 0.811 vs 0.807 — parity with a full sc foundation model using only the adaptor + LLM embeddings.
- **Random-embedding ablation (C9).** Within the same adaptor, **random gene vectors hit PBMC F1 0.835 — higher than GPT-3.5's 0.811** — and stay within noise on hPancreas/Aorta (0.968/0.680 and 0.951/0.882). This is the strongest evidence in the paper for what's actually doing the work, and it points away from the LLM.

### Batch correction

![scELMo batch correction on CITE-seq and CITE-seq × CyTOF — `wa` raises both `S_bio` and `S_batch`](/assets/images/paper/scelmo/page_006.png)
*Figure 3: Batch-correction UMAPs and scores on a single-protocol CITE-seq pair and a cross-protocol CITE-seq × CyTOF pair. The `wa` aggregation lifts `S_bio` and `S_batch` simultaneously, the regime where `wa` actually beats `aa`.*

### Where `wa` actually beats `aa`

![hPancreas UMAP comparison — only `wa` preserves cell-type structure](/assets/images/paper/scelmo/fig_p033_01.png)
*Ext. Fig. 5 (panel 1): hPancreas UMAP from GenePT embeddings — cell-type blocks blur.*

![hPancreas UMAP — `aa` aggregation](/assets/images/paper/scelmo/fig_p033_02.png)
*Ext. Fig. 5 (panel 2): Uniform-average (`aa`) aggregation on the same data — moderate structure.*

![hPancreas UMAP — `wa` aggregation](/assets/images/paper/scelmo/fig_p033_03.png)
*Ext. Fig. 5 (panel 3): Expression-weighted (`wa`) aggregation — visibly tighter cell-type blocks. The qualitative case for `wa` over `aa`.*

### Where scELMo fails — the multi-omic appendix

![scELMo on scRNA + scATAC multi-omic integration — neither `wa` nor `aa` removes the batch effect](/assets/images/paper/scelmo/fig_p034_01.png)
*Ext. Fig. 6 (panel 1): Raw multi-omic UMAP before integration — strong modality batch.*

![scELMo on scRNA + scATAC multi-omic integration — `wa` UMAP](/assets/images/paper/scelmo/fig_p034_02.png)
*Ext. Fig. 6 (panel 2): After `wa` aggregation, scRNA and scATAC clouds remain separated; iLISI stays at zero. Appendix C states this outright: "scELMo is not capable of multi-omic data integration under the zero-shot learning framework" — **directly contradicting the abstract's "multi-omic" claim (C8)**.*

### Other ablations worth flagging

- **`wa` vs `aa` on annotation.** Numerically identical to 3 decimals across hPancreas / Aorta / PBMC (Table 1). The "wa is better" headline generalizes only to clustering and batch correction.
- **In-silico treatment (Fig. 4).** `Δcos` of 1e-4 – 4e-3 with no permutation test; the 1e-4 threshold is asserted without sensitivity analysis; validation is literature search.
- **GEARS perturbation (Fig. 5d).** GenePT embeddings beat GPT-3.5 embeddings on Dixit / Adamson / Norman. The authors' own conclusion is that NCBI text is better than GPT-3.5 summaries for Perturb-seq — which itself contradicts the paper's framing that GenePT under-uses LLMs.

## Limitations

**Authors acknowledge.**
- LLMs evolve fast; GPT-3.5 embeddings will date.
- LLMs cannot describe recently-discovered genes (knowledge cutoff ~2021).
- Domain-specific fine-tuning of the LLM itself is infeasible on the authors' budget.
- Extending to large-feature modalities (GWAS, scATAC) is hard — and Appendix C says zero-shot multi-omic outright fails.

**Not addressed; we add.**
- **No seed/variance reporting** for any quantitative table. Single-run numbers everywhere.
- **No comparison to scFoundation, UCE, GenePT2**, or any 2024 follow-up — comparison set is essentially scGPT vs Geneformer vs PCA.
- **No cost / latency analysis** despite "lighter requirement" being a headline claim. OpenAI API calls have per-token cost and rate limits that should be quantified.
- **Adaptor architecture not described** in the main text — depth, hidden width, batch size, optimizer all missing, which matters because the adaptor is doing most of the work (C9).
- **Cross-species transfer untested.** All gene descriptions are human; mouse/zebrafish single-cell would expose how human-centric the symbol → GPT prompt is.
- **kNN `k=10` inherited from GenePT** without sensitivity analysis.
- **n=40 hallucination audit** is undersized for a paper whose pitch hinges on LLM choice (C7), with no rubric and no inter-rater reliability.
- **"Multi-omic" abstract claim is internally contradicted** by Appendix C (C8) and should be retracted or qualified.
- **Reproducibility gap.** "Details in Supplementary File 3" appears repeatedly; the main PDF contains no dataset-size table.

## Why It Matters for Medical AI

- **Lightweight clinical pipeline.** If parity with scGPT can be reached on a single A5000 with a small adaptor over off-the-shelf LLM embeddings, hospitals and core facilities that cannot afford foundation-model pre-training can still deploy modern cell-type annotation. This is the right scaling axis for translational labs.
- **In-silico target screening as triage, not evidence.** The DCM/HCM and aortic-aneurysm sweeps illustrate a workflow — `Δcos` scoring on gene knockouts — that is cheap enough to run before any wet-lab spend. But effect sizes of 1e-4 with no permutation test or replication mean the output is a *hypothesis generator*, not validation. Treat it like a database search hit.
- **An honest LLM-for-omics baseline.** The random-embedding row (C9) is the most useful gift this paper gives future work: any LLM-for-omics method should beat random-vector embeddings inside the same adaptor, on the same split, before claiming the LLM is what matters. scELMo itself does not pass that bar on PBMC F1.

## References

- **Paper (bioRxiv preprint, 2023-12-08 v1):** [10.1101/2023.12.07.569910](https://doi.org/10.1101/2023.12.07.569910)
- **Code:** [HelloWorldLTY/scELMo](https://github.com/HelloWorldLTY/scELMo) (MIT)
- **Closest predecessor:** GenePT — LLM text embeddings of NCBI gene summaries with uniform-average aggregation.
- **Compared foundation models:** scGPT (Cui et al. 2023), Geneformer (Theodoris et al. 2023), scBERT.
- **Integration tools the embeddings are spliced into:** CINEMA-OT, CPA, GEARS.
- **Benchmark datasets referenced:** hPancreas, Aorta, PBMC (cross-source), CITE-seq, CITE-seq × CyTOF, Heart (HCM/DCM), Ascending aortic aneurysm cohort, ChangYe2021, perturbed PBMC, CPA example, Openproblems, Dixit, Adamson, Norman.
