---
title: "Understanding the LLM-based Gene Embeddings"
excerpt: "Per-dimension GSEA on OpenAI text-embedding-3-small recovers 49/50 Hallmark (98%) and 2,878/3,077 C2 (93.53%) MSigDB pathways with shuffled-null at 0.00%, but with no contamination audit, no specificity metric, no non-LLM baseline, and no downstream task the result is best read as a coverage upper bound on what an LLM trained on the public biomedical corpus can re-encode."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/llm-gene-embeddings/
tags:
  - GenePT
  - OpenAI-text-embedding-3-small
  - BioBERT
  - Gene-Embeddings
  - GSEA
  - MSigDB
  - LLM-for-Biology
  - Bioinformatics
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- A per-dimension GSEA framework over OpenAI's `text-embedding-3-small` 1,536-d gene vectors (GenePT pipeline, input = gene symbol + NCBI description) recovers **49/50 (98%) Hallmark** and **2,878/3,077 (93.53%) C2** MSigDB pathways at a Bonferroni-style per-dimension cutoff of p/d = 0.05 / 1,536; the shuffled-matrix null returns **0.00%**.
- Information is **distributed, not localized**: on average **84.36 dimensions/pathway** (Hallmark) and **73.52 dims/pathway** (C2) enrich each covered set; the only Hallmark miss is HALLMARK_NOTCH_SIGNALING (32 genes — the smallest set).
- Two robustness probes matter most. Symbol-only inputs (literally just `"TP53"`) still recover **64% Hallmark / 72.15% C2**, and **all 11 small LMs (23-110M params) exceed 86% coverage** once given the full NCBI description — **BioBERT-110M actually beats OpenAI on symbol-only Hallmark (70% vs 64%)**. The two together mean the corpus does most of the work; the LLM scale does less than the framing suggests.

## Motivation

Three families of LLM-for-gene-expression methods now coexist: chatbot prompting (e.g., ChatGPT for cell-type annotation), foundation models trained from scratch on scRNA-seq (scBERT, Geneformer, scGPT, scFoundation), and the **GenePT-style text-embedding route** — feed an NCBI gene description into an off-the-shelf text encoder, take the output vector as the gene's representation, and use it downstream like any embedding. The text-embedding route is cheaper, model-agnostic, and immediately compatible with classical ML; recent work has even ported it to EHR concepts. But the whole approach rests on an unverified premise: that a one- or two-paragraph NCBI blurb compressed into a 1,536-d vector actually contains *gene-regulatory* structure rather than literature-trivia. If it does not, all of the downstream perturbation-prediction, spatial-transcriptomics, and EHR-risk applications cited as motivation are pattern-matching artifacts.

![Motivation: NCBI descriptions of well-studied (TP53, BRCA1) and poorly characterized (A1BG) genes](/assets/images/paper/llm-gene-embeddings/page_004.png)
*Figure 1: NCBI gene descriptions referenced by the paper are short paragraphs of curated text; the central question is whether such terse text, once embedded, can reproduce pathway-level structure curated from decades of wet-lab biology.*

## Core Innovation

- **Per-dimension GSEA as the probe.** Instead of asking "does the full embedding look biological under UMAP?", treat each of the 1,536 axes as a candidate carrier of pathway-level signal: rank genes by their coordinate on dimension j, run pre-ranked fgsea against every MSigDB pathway, declare a pathway "reflected by j" if FDR-adjusted q < p/d with p = 0.05.
- **Pathway coverage as the headline metric.** Define
  $$\text{pathway coverage} = \frac{\#\{\text{pathways reflected by} \geq 1 \text{ dimension}\}}{\#\text{ total pathways}} \times 100\%.$$
  This is a union over dimensions — at-least-one-hit per pathway — which is generous on purpose; the headline number is therefore an *upper bound on what the canonical basis can show*, not an estimate of how cleanly the embedding factors biology.
- **Three control axes.** (i) Shuffle the matrix end-to-end and recompute coverage — should hit 0% under honest multiple-testing correction. (ii) Re-embed each gene under three input lengths — symbol only, half-description, full description — to disentangle "the LLM read the input text" from "the LLM remembered the symbol from pretraining". (iii) Repeat the whole pipeline with 11 small LMs (10 general-purpose 23-55M-param encoders + BioBERT-110M domain-specific) to test whether the recovered signal is OpenAI-specific or a generic property of text encoders.
- **Rotation-invariance caveat.** Per-dimension GSEA is not rotation-invariant: an orthogonal R preserves the geometry of A but redistributes column-wise signal in $\tilde A = AR$. The authors apply Crawford-Ferguson rotations (Quartimax / Varimax / Parsimax / FacParsim) via GPU gradient-projection and report coverage on Parsimax-rotated $\tilde A$ as a sanity check on the canonical-basis number.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | OpenAI embeddings recover **>93% of Hallmark and C2 pathways** under per-dimension GSEA | 49/50 Hallmark (98%), 2,878/3,077 C2 (93.53%) at p/d Bonferroni; shuffled-matrix null = 0.00% | MSigDB Hallmark + 6-subcategory C2 (3,077 sets) | ⭐⭐⭐ |
| C2 | Pathway signal is **distributed** across many dimensions, not localized | Mean 84.36 dims/pathway (Hallmark), 73.52 (C2); 95.55%+ of covered C2 pathways live in >1 dimension | MSigDB | ⭐⭐⭐ |
| C3 | Symbol-only embeddings still recover **>64% of pathways**, implying a pretraining-prior contribution | OpenAI symbol-only: 64% Hallmark, 72.15% C2 | OpenAI `text-embedding-3-small`, no input text other than the symbol | ⭐⭐ |
| C4 | Domain-specific SLMs perform best when **input is starved** | BioBERT symbol-only Hallmark 70% (best of all models, including OpenAI), C2 70.91% (2nd) | BioBERT-base-cased-v1.1 (110M params, PubMed+PMC pretraining) | ⭐⭐ |
| C5 | **All 11 SLMs approach OpenAI** when given full NCBI descriptions | All 11 SLMs >86% Hallmark and C2 with full description; >81% with half | 11 models × 3 inputs × 2 pathway collections = 66 numbers | ⭐⭐⭐ |
| C6 | Fine-tuning on **clinical text does not help** molecular pathway recovery | MedEmbed-small-v0.1 (BGE-small fine-tuned on synthetic PubMed Q-A) underperforms peers at symbol-only | 1 model, 1 comparison | ⭐ |
| C7 | Coverage is a **lower bound** because per-dimension GSEA is not rotation-invariant | Parsimax-rotated $\tilde A$ gives 94.00% / 92.79% (slightly below un-rotated 98.00% / 93.53%) | OpenAI matrix only | ⭐⭐ |
| C8 | LLM gene embeddings are "**lightweight, informative representations**" for downstream biological analysis | Abstract framing; no downstream task evaluated in this paper, all downstream evidence cited from prior work | — | ⭐ |

**Honest read.** Four issues materially weaken the headline narrative and the post-reviewer (and any practitioner about to drop these vectors into a downstream pipeline) should hold them in mind:

1. **Training-data contamination between OpenAI/BioBERT pretraining and MSigDB labels.** The "symbol-only ⇒ 64% Hallmark" result (C3) is *only interpretable as evidence of biology-encoding if the model did not see the labels during pretraining*. OpenAI's `text-embedding-3-small` and BioBERT (PubMed + PMC) have almost certainly ingested NCBI gene pages, GeneCards entries, and MSigDB-derived literature wholesale. Without a contamination-cleaned holdout — e.g., pathways added to MSigDB *after* the model training cutoff, or a synthetic pathway collection built post-cutoff from non-indexed sources — symbol-only "prior knowledge" cannot be distinguished from memorization of the test labels. The paper does not estimate or address this; it is the single largest threat to validity of C3 (and indirectly C4).
2. **Pathway coverage counts at-least-one-hit and ignores specificity.** A pathway is declared "covered" if any one of 1,536 dimensions enriches it under q < p/d. But the same dimension is allowed to "cover" many unrelated pathways simultaneously — the metric does not penalize promiscuous axes. The 84-dim/pathway redundancy headline can be read two ways: 84 independent corroborations of a true biological signal, or 84 chances per pathway to manufacture a spurious enrichment from a few hub-gene rankings. Without a specificity counterpart — e.g., per-dimension precision against a curated negative pathway list, or pathway-by-pathway purity — the 93%-98% number describes recall only.
3. **No non-LLM baseline.** A bag-of-words TF-IDF embedding over the same NCBI descriptions, a PubMed gene-cooccurrence vector, or a node2vec embedding over a PPI graph would each be cheap to compute and would directly answer "is this an LLM contribution or a 'text about gene X mentions pathway Y' co-occurrence contribution?". None are reported. The closest comparator is the 11-SLM grid (C5), but every SLM is also a text encoder — the experiment varies *which* LM, not whether an LM is needed at all.
4. **No downstream-task benchmark.** The abstract sells "lightweight informative representations for downstream biological analysis", yet no downstream task — cell typing, perturbation-response prediction, spatial-transcriptomics integration, EHR risk prediction, anything — is run inside this paper. Every downstream performance claim is inherited from cited prior work. C8 is, on the evidence in this paper, a framing claim, not an empirical claim.

The internally well-supported claims are C1, C2, C5, and C7: the methodology is simple, the null returns 0.00% to four significant figures, the FDR cutoff is aggressive, the SLM-grid is wide, and the rotation analysis is an unusually honest self-critique that the headline number depends on basis choice.

## Method & Architecture

![Per-dimension GSEA framework over the 1,536-d OpenAI embedding matrix](/assets/images/paper/llm-gene-embeddings/page_005.png)
*Figure 2: Per-dimension GSEA setup on the 1,536-d OpenAI embedding matrix. For each dimension j, rank all human protein-coding genes by their coordinate $A_{:,j}$, run pre-ranked fgsea against every MSigDB pathway, declare a pathway "reflected by j" if FDR-adjusted q < p/d with p = 0.05 and d = 1,536, then aggregate to pathway coverage as the fraction of pathways with at least one reflecting dimension.*

1. **Embedding source.** GenePT Zenodo release of human gene vectors produced by OpenAI `text-embedding-3-small`. Input per gene = official symbol concatenated with the NCBI description. Output = 1,536-d. The matrix is $A \in \mathbb{R}^{n \times p}$ with $n$ = human protein-coding genes, $p = 1{,}536$.
2. **Pathway databases.** Human MSigDB:
   - Hallmark: 50 sets, all retained.
   - C2 canonical: 3,917 sets total; restrict to BioCarta + KEGG Medicus + PID + Reactome + WikiPathways + KEGG Legacy (drop 19 "miscellaneous"); keep only protein-coding genes; drop sets with <10 such genes → **3,077 C2 sets** (220 BioCarta, 398 KEGG Medicus, 196 PID, 1,367 Reactome, 711 WikiPathways, 185 KEGG Legacy).
3. **Per-dimension GSEA.** For each $j \in \{1,\dots,1536\}$, rank genes by $A_{:,j}$, run pre-ranked `fgsea`, threshold at q < p/d ≈ 3.26e-5 (Bonferroni per dimension).
4. **Pathway coverage.** Union over dimensions per pathway, then average over pathways in the collection (definition reproduced in Core Innovation).
5. **Null control.** Shuffle every entry of $A$, rerun the full pipeline. Expectation: coverage ≈ 0.
6. **Input ablation.** Re-embed each gene with OpenAI under three inputs — symbol only, symbol + first half of NCBI description, symbol + full NCBI description — and rerun per-dimension GSEA on each.
7. **Cross-model ablation.** Re-embed all genes with 11 SLMs (Gan et al., 2025): 10 general-purpose 23-55M-param encoders (incl. `stella-base-en-v2` at 55M) plus BioBERT-base-cased-v1.1 (110M, pretrained on 4.5B PubMed abstract words + 13.5B PMC full-text words). MedEmbed-small-v0.1 (BGE-small fine-tuned on synthetic PubMed Q-A triplets) is the partial-domain-adapted control. Repeat the 3-input grid → 11 × 3 × 2 = 66 coverage numbers.

![MSigDB C2 filtering and the first Hallmark coverage result](/assets/images/paper/llm-gene-embeddings/page_006.png)
*Figure 3: MSigDB C2 is filtered to 3,077 protein-coding pathways across six sub-databases, and per-dimension fgsea on the OpenAI embedding matrix yields the first headline number: 49 of 50 Hallmark pathways (98%) are covered, the only miss being HALLMARK_NOTCH_SIGNALING (32 genes, the smallest set).*

8. **Rotation analysis.** Apply Crawford-Ferguson rotations to $A$ — try Quartimax, Varimax, Parsimax, FacParsim — optimize via GPU gradient projection of

   $$f(\tilde A) = (1-\kappa)\sum_{i}\sum_j\sum_{l\neq j}\lambda_{ij}^2\lambda_{il}^2 \;+\; \kappa\sum_j\sum_i\sum_{l\neq i}\lambda_{ij}^2\lambda_{lj}^2,$$

   select Parsimax by a distance-ratio sparsity metric, and rerun GSEA on $\tilde A$.

## Experimental Results

### Main quantitative results

| Embedding model | Input | Hallmark (50 sets) | C2 (3,077 sets) |
|---|---|---|---|
| **OpenAI `text-embedding-3-small`** | **Full NCBI description** | **98.00 (49/50)** | **93.53 (2,878/3,077)** |
| OpenAI `text-embedding-3-small` | Half NCBI description | 96.00 (48/50) | 92.10 |
| OpenAI `text-embedding-3-small` | Symbol only | 64.00 (32/50) | 72.15 |
| OpenAI, shuffled matrix (null) | Full | 0.00 | 0.00 |
| OpenAI, Parsimax-rotated $\tilde A$ | Full | 94.00 | 92.79 |
| BioBERT-base-cased-v1.1 (domain) | Symbol only | **70.00 (best, beats OpenAI)** | 70.91 (2nd to OpenAI's 72.15) |
| 11 SLMs (range) | Symbol only | 26 – 70 | 46.60 – 72.15 |
| All 11 SLMs | Full description | **>86 (all models)** | **>86 (all models)** |
| All 11 SLMs | Half description | >81 | >81 |

C2 per-sub-database non-detection rates: KEGG Legacy 0%, PID 2.55%, KEGG Medicus 3.02%, BioCarta / Reactome / WikiPathways 6-9%.

### Ablation and qualitative findings worth highlighting

![C2 per-category non-detection breakdown](/assets/images/paper/llm-gene-embeddings/page_007.png)
*Figure 4: Per-category C2 non-detection rates — the heavily-curated, mechanistically-oriented sub-databases (KEGG Legacy, PID, KEGG Medicus) are recovered almost completely, while community-curated WikiPathways and the smaller / more granular BioCarta and Reactome subsets contribute most of the missed pathways.*

- **Information is highly redundant.** Of the 49 covered Hallmark pathways, all but one are reflected by >1 dimension (mean 84.36 dims/pathway); for C2, 95.55% of covered pathways live in >1 dimension (mean 73.52). Single-dimension interpretability tools applied to GenePT-style embeddings would systematically under-report what is encoded.
- **The single uncovered Hallmark pathway is HALLMARK_NOTCH_SIGNALING**, the smallest Hallmark set with 32 protein-coding genes (vs ≥36 for all others) — coverage failure correlates with set size, not biological obscurity.
- **The model-prior contribution is non-trivial — but contamination-vulnerable.** Symbol-only embeddings (input = `"TP53"`, etc.) still recover 64% of Hallmark; the only place this signal can originate is the encoder's pretraining corpus, since the input contains no text beyond a 3-7 character token. Whether that "pretraining corpus" memorized the MSigDB labels themselves is the open question.

![Input-text ablation results](/assets/images/paper/llm-gene-embeddings/page_009.png)
*Figure 5: Input-text ablation. Symbol-only → 64.00% / 72.15%; half-description → 96.00% / 92.10%; full-description → 98.00% / 93.53% (Hallmark / C2). The gap between symbol-only and half-description is where the input text — as opposed to memorized pretraining priors — does its work.*

- **Domain pretraining beats scale at minimal input.** BioBERT (110M params, biomedical-only) beats OpenAI on symbol-only Hallmark (70% vs 64%) despite being orders of magnitude smaller — for this task, corpus matters more than parameter count when the input is starved.
- **Generic-domain fine-tuning can hurt.** MedEmbed-small-v0.1 (BGE-small fine-tuned on synthetic clinical Q-A triplets) underperforms its un-tuned siblings on gene-pathway recovery — clinical text and molecular-biology text are not interchangeable substrates.
- **`stella-base-en-v2` surprise.** The largest general-purpose SLM in the panel (55M params, ~200 GB contrastive pretraining including Wikipedia and long-form web articles) is second only to BioBERT among small models at symbol-only — long-form web text plus hard-negative contrastive training closes much of the gap to a domain-specialized encoder.

![11-SLM × 3-input × 2-pathway-collection comparison](/assets/images/paper/llm-gene-embeddings/page_010.png)
*Figure 6: 11 small language models versus OpenAI across three input regimes. With full NCBI descriptions every SLM clears 86% coverage on both Hallmark and C2; with symbol-only inputs BioBERT (domain-specialized) wins Hallmark; with half-descriptions every SLM clears 81% — together this says the full-description text is doing most of the work and OpenAI-scale is not strictly required.*

- **Rotation does not help.** Parsimax rotation yields 94.00% / 92.79% — slightly below the canonical-basis 98.00% / 93.53%. Interpretability-driven rotation is orthogonal to pathway recovery on this matrix, but importantly the un-rotated coverage is therefore a *lower* bound (for that basis choice).

## Limitations

**Authors acknowledge:**
- Per-dimension GSEA is not rotation-invariant; the canonical-basis coverage is a lower bound on what the geometry of A could in principle reflect.
- Parsimax rotation did not help, and the CF objective is not biology-aware.
- The hypothesis that OpenAI's encoder has "super-human cross-gene contrast capability" is "difficult to test directly" and is deferred.
- MSigDB pathway-set heterogeneity (BioCarta deprecation, Reactome hierarchical granularity, WikiPathways community curation) confounds cross-category comparisons.

**Not addressed (auditor's view):**
- **Training-data contamination.** `text-embedding-3-small` and BioBERT have almost certainly ingested NCBI gene pages, GeneCards, and MSigDB-derived literature. Without a contamination-cleaned holdout (pathways added to MSigDB after the model cutoff, or a synthetic post-cutoff pathway collection), "the model encodes biology" and "the model retrieved memorized facts" cannot be separated — this is the single largest threat to validity, and it applies especially to the symbol-only result (C3) and to the BioBERT-beats-OpenAI result (C4).
- **No non-LLM baseline.** A TF-IDF embedding over the same NCBI descriptions, a PubMed gene-cooccurrence vector, or a node2vec PPI embedding would each directly answer whether the recovered signal is an LLM-specific contribution or a generic property of "text about gene X mentions pathway Y". None are reported.
- **Coverage ≠ specificity.** Pathway coverage takes a union over dimensions per pathway — a pathway is covered if any one of 1,536 dimensions enriches it. Nothing penalizes a dimension that enriches dozens of unrelated pathways at once; 84 dims/pathway redundancy could equally be 84 false-positive opportunities. A per-dimension precision metric is missing.
- **Multiplicity is favorable to the result.** The p/d Bonferroni per dimension is stringent for individual false positives, but the union-aggregation step gives 1,536 chances to find each pathway; a joint pathway × dimension FDR (e.g., BH across the full grid) would be the honest correction.
- **No statistical variance.** Coverage numbers are point estimates — no bootstrap over genes, no permutation distribution beyond the single shuffled-matrix run, no per-dimension dependency correction beyond Bonferroni.
- **No downstream-task benchmark.** Despite framing the embeddings as "lightweight informative representations for downstream analysis", no cell-typing, perturbation-prediction, ST-integration, or EHR-risk task is run inside this paper.
- **No harder biology.** Tissue-specific pathways, disease-specific signatures, sex-specific gene programs, and non-human (mouse) gene sets are not tested. The entire result is human-protein-coding-only.

**Open questions raised by the work:**
- What is coverage on a holdout pathway collection released *after* the OpenAI / BioBERT training cutoffs?
- Does symbol-only coverage drop for genes whose Wikipedia / GeneCards pages are short (i.e., poorly characterized genes)?
- Would an embedding distilled from a non-text source (protein structure, gene-coexpression graph) outperform on coverage but underperform on biological-realism, or vice versa?

## Why It Matters for Medical AI

The downstream applications the paper invokes — perturbation-response prediction, spatial-transcriptomics integration, histopathology-multimodal frameworks, EHR risk prediction — are clinically motivated. For a practitioner about to drop GenePT-style vectors into one of these pipelines, this paper is genuinely useful in two ways and genuinely incomplete in one:

- Useful (1): the shuffled-null returning 0.00% means the canonical-basis pathway coverage is not an artifact of the test, and the redundancy result (84 dims/pathway) tells you that single-dimension interpretability tools applied to GenePT will under-report what is encoded.
- Useful (2): the SLM-grid means that for cost-sensitive deployments — on-premise hospital infrastructure, downstream tasks where API calls per patient are prohibitive — a small domain-specialized encoder (BioBERT-class) reaches >86% pathway coverage on full NCBI descriptions and beats the OpenAI baseline at symbol-only Hallmark. There is no clinical reason to call the closed-source API for this representation.
- Incomplete: without a downstream-task benchmark inside the paper and without a contamination audit, "the embedding is biologically informative" does not yet translate to "the embedding will help your clinical downstream task". Treat C8 as a hypothesis, not a result.

## References

- Cai Y., Gan D., Zhang H., Li J. **Understanding the LLM-based gene embeddings.** bioRxiv preprint, posted 22 December 2025. DOI: [10.64898/2025.12.19.695582](https://doi.org/10.64898/2025.12.19.695582). License: CC-BY-NC-ND 4.0.
- GenePT (gene embeddings via OpenAI `text-embedding-3-small`): Chen Y., Zou J. **GenePT: A simple but effective foundation model for genes and cells built from ChatGPT.** Code & data via the GenePT GitHub and Zenodo release.
- MSigDB Hallmark and C2 collections: Liberzon A. et al., *Cell Systems* 2015; Subramanian A. et al., *PNAS* 2005 (GSEA).
- fgsea (pre-ranked fast GSEA): Korotkevich G. et al., bioRxiv 2021.
- BioBERT: Lee J. et al. **BioBERT: a pre-trained biomedical language representation model for biomedical text mining.** *Bioinformatics* 2020.
- SLM panel (stella-base-en-v2, MedEmbed-small-v0.1, and 8 others): Gan D. et al., 2025 (referenced in the paper).
- Crawford-Ferguson rotation family: Crawford C. B., Ferguson G. A., *Psychometrika* 1970.
