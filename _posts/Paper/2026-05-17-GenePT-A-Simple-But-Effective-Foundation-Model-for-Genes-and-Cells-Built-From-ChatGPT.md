---
title: "GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT"
excerpt: "Frozen GPT-3.5 embeddings of NCBI gene summaries match — and sometimes beat — Geneformer and scGPT on gene/cell benchmarks (e.g., AUC 0.82 vs 0.65–0.67 on Du et al.'s gene–gene interaction task), with zero pretraining."
categories: [Paper, BioInformatics, LLM]
permalink: /paper/genept/
tags:
  - GenePT
  - Single-cell
  - LLM-Embeddings
  - Foundation-Models
  - scRNA-seq
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- **Idea.** Skip self-supervised pretraining on tens of millions of cells. Instead, embed each gene's NCBI summary with GPT-3.5 `text-embedding-ada-002`, and build cell vectors by either (i) expression-weighted averaging of gene embeddings (**GenePT-w**) or (ii) embedding a sentence of gene names ordered by expression (**GenePT-s**).
- **Result.** Off-the-shelf logistic regression on GenePT embeddings reaches **96% accuracy on 15-class gene functionality** and **AUC 0.82 on Du et al.'s gene–gene interaction benchmark vs 0.65–0.67 for Gene2Vec / scGPT / Geneformer** with the same LR head. Cell-clustering wins are split 5-4 with scGPT across 9 tasks — competitive, not dominant.
- **Cost.** No training. No curated 30M-cell corpus. Just one OpenAI embedding call per gene (and per cell, for GenePT-s).

## Why It Matters

Geneformer and scGPT both poured tens of millions of single-cell transcriptomes into Transformer pretraining (30M and 33M respectively). That investment encodes co-expression statistics — but discards everything biologists already wrote about each gene in NCBI, OMIM, and GeneCards. GenePT asks the orthogonal question: if those summaries already capture the biology, can a frozen LLM embedding endpoint substitute for the entire pretraining pipeline?

The answer turns out to be "yes for gene-level tasks, roughly yes for cell-level tasks." That changes the calculus for any group that does not have 30M curated cells lying around — including most clinical scRNA-seq settings. The medical-AI tie-in is indirect (most downstream tasks here are clinical scRNA-seq atlases: pancreas, aorta, myeloid, cardiomyopathy, MS), but the representation is the contribution, not a diagnostic system.

## Method

![GenePT pipeline overview](/assets/images/paper/genept/fig_p003_01.png)
*Figure 1 — GenePT pipeline: NCBI gene summaries are embedded with GPT-3.5; cells are represented either by an expression-weighted average of gene embeddings (GenePT-w) or by embedding a sentence of gene names ordered by expression (GenePT-s).*

### Step 1: Gene vocabulary and text

- Union of Geneformer + scGPT gene lists plus downstream-dataset genes — ~33,000 genes (Ensembl IDs converted via `mygene` with >90% lookup success); ~60,000 aliases mapped via HGNC.
- For each gene, fetch the NCBI Gene summary; strip hyperlinks and date stamps. Mean length **73 words** (IQR 25–116).
- Three input variants tested (Appendix A): (i) name only, (ii) name + summary, (iii) full NCBI summary card.

### Step 2: Gene embedding

- Call OpenAI `text-embedding-ada-002` on each gene's text → **1,536-dim vector** per gene.
- A Llama-7B variant is also reported on the four binary gene-property tasks.

### Step 3: Cell preprocessing

Standard scanpy pipeline: row-normalize each cell to 10,000 transcripts, then $\log(1 + x)$.

### Step 4a: GenePT-w (weighted average)

![GenePT-w schematic](/assets/images/paper/genept/fig_p003_03.png)
*Figure 2 — GenePT-w: expression-weighted average of NCBI-derived gene embeddings, L2-normalized.*

$$\mathbf{c}_i = \mathrm{norm}\!\left(\sum_{g} \tilde{x}_{ig}\, \mathbf{e}_g\right)$$

where $\tilde{x}_{ig}$ is the normalized expression of gene $g$ in cell $i$ and $\mathbf{e}_g$ is the GPT-3.5 embedding of gene $g$'s NCBI summary.

### Step 4b: GenePT-s (sentence)

![GenePT-s schematic](/assets/images/paper/genept/fig_p003_02.png)
*Figure 3 — GenePT-s: each cell is serialized into a gene-name sentence ranked by expression, then embedded by GPT-3.5.*

For each cell, build a "sentence" of gene names ordered by descending normalized expression (zero-count genes dropped), then embed the entire string. Conceptually close to Cell2Sentence (Levine et al. 2023), but used for representation rather than generation.

### Step 5: No training

Both cell representations are zero-shot. Downstream tasks use off-the-shelf **L2-regularized logistic regression, random forest, k-means, k-NN** with default scikit-learn hyperparameters and 5-fold CV.

## Experiments

### Gene-level tasks (Table 1, 5-fold CV AUC ± SD)

| Model | Dosage sensitivity | Bivalent vs non-meth. | Bivalent vs Lys4-meth. | TF range |
|---|---|---|---|---|
| Geneformer (fine-tuned) | 0.91 ± 0.02 | 0.93 ± 0.07 | 0.88 ± 0.09 | 0.74 ± 0.08 |
| Gene2Vec + LR | 0.91 ± 0.03 | 0.66 ± 0.07 | 0.91 ± 0.04 | 0.83 ± 0.14 |
| BioLinkBert + RF | 0.87 ± 0.02 | 0.80 ± 0.06 | 0.85 ± 0.07 | 0.54 ± 0.23 |
| Random Embed + LR | 0.54 ± 0.04 | 0.59 ± 0.03 | 0.46 ± 0.07 | 0.36 ± 0.16 |
| GenePT (name only) + RF | 0.89 ± 0.02 | 0.90 ± 0.02 | 0.91 ± 0.04 | 0.58 ± 0.22 |
| **GenePT + LR** | **0.89 ± 0.03** | **0.91 ± 0.06** | **0.94 ± 0.03** | **0.73 ± 0.25** |
| **GenePT + RF** | **0.92 ± 0.02** | **0.92 ± 0.06** | **0.95 ± 0.04** | **0.64 ± 0.07** |
| GenePT (Llama-7B) + LR | 0.93 ± 0.04 | 0.88 ± 0.07 | 0.93 ± 0.05 | 0.67 ± 0.25 |

- **Gene functionality (15-class):** GenePT + LR → **96% accuracy**. On the 21k-gene subset where Gene2vec is defined: GenePT 0.95 vs Gene2vec 0.86.
- **Gene–gene interaction (Du et al.):** GenePT + LR **AUC 0.82** vs 0.65–0.67 for Gene2Vec / scGPT / Geneformer with the same LR head; Du et al.'s deep NN was 0.77; random embedding control 0.51.
- **PPI:** GenePT beats all baselines on Lit-BM, HuRI, and tissue-PPI.

![Gene-functionality UMAP](/assets/images/paper/genept/fig_p010_01.png)
*Figure 4 — UMAP of GenePT gene embeddings, colored by 15 functional classes; clusters separate protein-coding, miRNA, snRNA, pseudogene, lincRNA, and other classes without any supervised training.*

![Gene-functionality confusion matrix](/assets/images/paper/genept/fig_p010_02.png)
*Figure 5 — Confusion matrix for 15-class gene-functionality prediction using L2-regularized LR on GenePT embeddings (test accuracy 96%).*

![GGI ROC](/assets/images/paper/genept/fig_p010_03.png)
*Figure 6 — GenePT + LR reaches AUC 0.82 on Du et al.'s gene–gene interaction benchmark, beating Gene2Vec / scGPT / Geneformer (0.65–0.67) with the same classifier head and Du et al.'s deep NN (0.77).*

![Lit-BM PPI ROC](/assets/images/paper/genept/fig_p010_05.png)
*Figure 7 — PPI ROC on the Lit-BM literature dataset: GenePT outperforms expression-derived embeddings.*

![Gene-program heatmap](/assets/images/paper/genept/fig_p010_04.png)
*Figure 8 — Gene programs discovered by Leiden clustering on GenePT cosine-similarity graphs show coherent cell-type-specific activation in human immune tissue.*

### Cell-level clustering concordance (Table 2 summary)

Across 9 (dataset × annotation) tasks: **scGPT wins 5, GenePT (w or s) wins 4**. Geneformer is consistently the weakest pretrained option.

| Dataset / annotation | Best GenePT ARI | Geneformer ARI | scGPT ARI |
|---|---|---|---|
| **Pancreas — cell type** | **0.49 (GenePT-w)** | 0.04 | 0.21 |
| **Aorta — cell type** | **0.54 (GenePT-w)** | 0.21 | — |

### Cell-type annotation via 10-NN (Table C4)

| Dataset | GenePT-w | scGPT | Geneformer |
|---|---|---|---|
| **Pancreas** | **0.95** | 0.77 | 0.50 |
| **Bones** | **0.49** | 0.34 | 0.22 |
| Multiple Sclerosis | 0.38 | **0.76** | — |

Ensembling (GenePT-w + GenePT-s + scGPT) usually equals or beats any single embedding. GenePT-w is *worst* on MS — a failure that the paper does not discuss in prose.

### Batch effects (cardiomyocyte and Aorta)

![Aorta raw — phenotype](/assets/images/paper/genept/fig_p014_01.png)
*Figure 9(a) — Raw Aorta scRNA-seq (UMAP), colored by disease phenotype.*

![Aorta raw — cell type](/assets/images/paper/genept/fig_p014_02.png)
*Figure 9(b) — Raw data colored by annotated cell type: identical cell types split across patient-batch clusters.*

![Aorta raw — patient](/assets/images/paper/genept/fig_p014_03.png)
*Figure 9(c) — Same UMAP colored by patient ID — strong patient-level batch structure.*

![Aorta GenePT-s — phenotype](/assets/images/paper/genept/fig_p014_04.png)
*Figure 9(d) — GenePT-s embeddings: cells now cluster by disease phenotype rather than patient.*

![Aorta GenePT-s — cell type](/assets/images/paper/genept/fig_p014_05.png)
*Figure 9(e) — GenePT-s UMAP colored by cell type — coherent cell-type clusters.*

![Aorta GenePT-s — patient](/assets/images/paper/genept/fig_p014_06.png)
*Figure 9(f) — GenePT-s UMAP colored by patient ID: patient signal largely removed.*

- **Cardiomyocyte:** patient ARI drops from 0.33 (raw) to **0.07 (GenePT-s)**, 0.01 (Geneformer), 0.01 (scGPT) — GenePT-s is the *least* batch-robust of the three foundation models, but still removes most patient signal. Disease classification: GenePT-s and scGPT both 88% accuracy/precision/recall; Geneformer 71%.
- **Aorta:** GenePT-s yields 73% phenotype-prediction accuracy vs scGPT 75% vs Geneformer 69%.

### Ablation — input text content (Appendix A)

![Input-text ablation](/assets/images/paper/genept/fig_p017_01.png)
*Figure 10 — Sensitivity to NCBI input text: removing the summary text drops bivalent-vs-non-methylated from 0.91 → 0.85 (LR) and TF-range from 0.73 → 0.61. Gene names alone are weaker; the literature signal carries most of the gain.*

### Leakage audit

![Lit-BM full test](/assets/images/paper/genept/fig_p020_01.png)
*Figure 11(a) — Lit-BM PPI before removing pairs mentioned in NCBI summaries.*

![Lit-BM overlap removed](/assets/images/paper/genept/fig_p020_02.png)
*Figure 11(b) — After removing overlap (4% of positives): performance essentially unchanged — leakage is not the source of the signal.*

### Claims vs evidence (with ratings)

| # | Claim | Evidence | Rating |
|---|---|---|---|
| C1 | Matches/beats Geneformer on gene-property tasks without pretraining | Table 1, 5-fold AUC across 4 tasks; explicit random-embedding negative control | ⭐⭐⭐ |
| C2 | Beats expression-based embeddings on gene–gene interaction | Fig. 2(c) AUC 0.82 vs 0.65–0.67 with identical LR head | ⭐⭐⭐ |
| C3 | Competitive with scGPT on cell-type clustering, beats Geneformer | 9 tasks scoreboard scGPT 5 / GenePT 4; 10-NN annotation | ⭐⭐ |
| C4 | Robust to patient batch effects while preserving disease phenotype | Cardiomyocyte ARI 0.33→0.07; LR disease accuracy 88% | ⭐⭐ |
| C5 | Beats all baselines on PPI (Lit-BM, HuRI, heart tissue) | Fig. 2(d–f) ROC + Fig. B2 PR | ⭐⭐⭐ |
| C6 | Results not due to NCBI summary leakage | Temporal split argument + <1%–4% pair overlap, Lit-BM no-overlap test | ⭐⭐ |
| C7 | Lift over Geneformer/scGPT is a *fair* comparison | Asserted; not quantified | ⭐ |
| C8 | "Simpler and more efficient" than 30M-cell pretraining | No FLOPs / wall-clock / $ comparison reported | ⭐ |

## Honest Read

**What the paper proves convincingly.** Gene-level claims (C1, C2, C5) are the strongest contribution. Multiple datasets, an explicit random-embedding control, a text-content ablation that demonstrates the summary (not the tokenizer's handling of gene symbols) carries the signal, and a Llama-7B replication that shows the result is not an OpenAI-only artifact. The leakage audit on Lit-BM (4% positive overlap → no meaningful change after removal) is the right experiment to run.

**Where the abstract overshoots.** The cell-level story is weaker than the framing suggests:

- **Scoreboard is split.** Across 9 (dataset × annotation) clustering tasks, scGPT wins 5 and GenePT wins 4. The abstract reads as a clean victory; the table does not.
- **MS failure is buried.** On Multiple Sclerosis, GenePT-w drops to 0.38 10-NN accuracy vs scGPT's 0.76. MS is one of the more clinically interesting evaluations, and it is mentioned in a table without prose commentary.
- **Fairness of Table 1 is mixed.** Geneformer's Table 1 numbers come from a fine-tuned transformer head; GenePT uses LR/RF on frozen embeddings. The opposite-direction comparison (Geneformer-frozen + LR) is done only at the cell level. A fully matched protocol is missing.
- **No statistical tests.** Every comparison is mean ± SD over 5 folds, often with overlapping error bars (e.g., TF range: GenePT+LR 0.73 ± 0.25 vs Geneformer 0.74 ± 0.08 — indistinguishable). The Du et al. GGI delta is large enough not to need a test; the Table 1 deltas are not.

**Auditor-observed gaps the paper does not address:**

- **Token / context budget.** GenePT-s sentences can include thousands of gene names per cell. How is OpenAI's 8k-token context handled? Truncation policy is not specified — and truncation policy implicitly determines which genes the cell "sees."
- **Cost & reproducibility at scale.** Every cell requires an OpenAI embedding call. For a 1M-cell atlas this is on the order of thousands of dollars, on a closed and versioned API. No latency / $ table is given.
- **Information advantage.** GenePT-w explicitly weights by expression magnitude, while Geneformer's "rank of expression" tokenization discards magnitudes. GenePT-s sentences also bypass Geneformer's 2,048-token cap. The "no pretraining" win partly reflects access to richer per-cell input — that is fair to point out without diminishing the result.
- **Closed-model dependence.** `text-embedding-ada-002` is OpenAI-versioned and can change without notice. The Llama-7B variant exists but is reported only on the four binary gene-property tasks, not on cell-level benchmarks.
- **Failure-mode breakdown.** Pseudogene/lncRNA accuracy is not broken out, though Fig. 2(b) hints at confusion among lincRNA / lncRNA / processed-transcript classes — exactly where thin NCBI summaries would hurt.
- **Perturbation prediction is absent.** scGPT's headline downstream task — perturbation response prediction — is not evaluated. That is the most likely place where co-expression-pretraining beats a static literature embedding.
- **Stale literature, not just hallucination.** The authors argue that embedding (vs generation) limits hallucination; they do not address staleness of NCBI summaries for actively studied genes.

**Bottom line.** Take GenePT seriously as a strong, cheap baseline that any future "single-cell foundation model" paper has to beat. Do not yet take it as a replacement for scGPT/Geneformer in tasks they were actually designed for (perturbation response, atlas-scale integration with custom heads).

## References

- **Paper:** Chen, Y., & Zou, J. (2023/2024). *GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT.* bioRxiv. [https://doi.org/10.1101/2023.10.16.562533](https://doi.org/10.1101/2023.10.16.562533)
- **Code:** [https://github.com/yiqunchen/GenePT](https://github.com/yiqunchen/GenePT)
- **Related — Geneformer:** Theodoris et al., *Nature* 2023.
- **Related — scGPT:** Cui et al., *Nature Methods* 2024.
- **Related — Cell2Sentence:** Levine et al., 2023.
- **GGI benchmark:** Du et al., *Bioinformatics* 2019 (Gene2vec).
- **PPI benchmarks:** Luck et al. (HuRI) 2020; Rolland et al. (Lit-BM) 2014; Greene et al. (tissue PPI) 2015.
