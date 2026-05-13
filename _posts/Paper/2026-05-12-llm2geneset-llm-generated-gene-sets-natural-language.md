---
title: "Enhancing Gene Set Overrepresentation Analysis with Large Language Models (llm2geneset)"
excerpt: "GPT-4o-generated gene sets show 66-94% Bonferroni-significant overlap with KEGG, Reactome, and WikiPathways curated sets across 2,939 benchmark pathways — but no random-gene baseline and no MSigDB Hallmark/C2 test means this measures recall of training data, not discovery."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/llm2geneset-llm-generated-gene-sets-natural-language/
tags:
  - llm2geneset
  - GPT-4o
  - Gene-Set-Analysis
  - Overrepresentation-Analysis
  - LLM-for-Biology
  - Bioinformatics
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- llm2geneset is a two-step "propose pathways from DEGs, then generate the gene set from each description in a context-free second prompt" loop bolted onto a standard one-tailed hypergeometric overrepresentation test against a 19,846-gene human protein-coding universe.
- Across **2,939 curated gene sets** from KEGG_2021_Human, Reactome_2022, and WikiPathway_2023_Human, **66-94% of LLM-generated sets overlap the matching curated set at Bonferroni p < 0.01**; GPT-4o > GPT-4o-mini > GPT-3.5 on every database.
- The benchmark validates recall of pathway annotations the LLM has likely already seen during training (KEGG / Reactome / WikiPathways are heavily represented in PubMed) — **no random-gene-set null, no MSigDB Hallmark or C2, no non-OpenAI model, no held-out biology**. The headline number measures memorization, not discovery.

## Motivation

Gene set overrepresentation analysis (ORA) — the Fisher / hypergeometric test of an experimentally derived DEG list against a curated pathway library — is the workhorse for interpreting bulk and single-cell RNA-seq. The bottleneck is not the test; it is the library. KEGG, Reactome, WikiPathways, and MSigDB are hand-curated, go stale between releases, and are never tailored to the experimental context (cell type, perturbation, time point). The medical-AI hook in this paper is concrete: the authors apply ORA to bulk RNA-seq from iPSC-derived microglia treated with **AL002**, an investigational TREM2 agonist antibody currently in Phase 2 trials for early Alzheimer's disease, and ask whether an LLM prompted with "in vitro microglia treated with a TREM2 agonist antibody" can surface immune-relevant pathways that KEGG misses.

## Core Innovation

- **Decoupled proposal and generation prompts.** The proposal prompt sees the DEG list and emits N free-text pathway descriptions; the gene-set generation prompt sees only the description, not the DEGs. This decoupling is the only thing keeping the downstream hypergeometric test non-circular.
- **A standard hypergeometric ORA at the end.** Once a description has been converted to a gene set by the context-free `GetGenes` prompt, the rest of the pipeline is textbook ORA: hypergeometric survival function against a 19,846-gene universe, Benjamini-Hochberg or Bonferroni correction across the proposed N descriptions.
- **Three orthogonal prompt-engineering variants** — reasoning (per-gene rationale), confidence (low/med/high, keep only high), and ensembling (5 seeds, intersection) — that trade recall for precision.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | LLM-generated gene sets are significantly overrepresented in the matching curated set | Fig. 1C: 66-94% of sets pass Bonferroni p<0.01 hypergeometric test | KEGG (2021), Reactome (2022), WikiPathways (2023), 2,939 sets | ⭐⭐⭐ |
| C2 | Fraction overrepresented improves with LLM capability | Fig. 1C/1D: GPT-3.5 < GPT-4o-mini < GPT-4o on every database | Same 2,939 sets | ⭐⭐⭐ |
| C3 | Confidence and ensembling improve precision over the default prompt | Fig. 2C bar charts; "significantly improved" asserted with no specified test, p-value, or n | Same | ⭐⭐ |
| C4 | llm2geneset proposal beats single-prompt GSAI on biological-process recovery | Fig. 3B: higher mean unigram / bigram / cosine on GPT-4o across 3 DBs; cross-model results pushed to supplement | KEGG/Reactome/WikiPathways descriptions | ⭐⭐ |
| C5 | llm2geneset recovers two distinct processes from mixed gene sets where GSAI collapses to one | Fig. 3C: 50 mixed pairs per DB, three metrics | Synthetic mixes, pre-filtered to cosine > 0.7 on llm2geneset's *own* single-set outputs (test-set selection bias) | ⭐⭐ |
| C6 | Steerable context yields more on-target pathways for TREM2 microglia than KEGG | Fig. 4B qualitative ranking | One in-house RNA-seq, n=4 vs n=4, 30 DEGs | ⭐ (single experiment, sponsor's drug, no held-out validation) |
| C7 | Token cost is "minimal" | Fig. S1F: ~528K input + ~750K output tokens to generate all 2,939 sets with GPT-4o | Same | ⭐⭐ (numbers are reported; "minimal" is a value judgment) |

**Honest read.** C1 and C2 are robust at the level of "GPT-4o can name overlapping genes for a pathway description from KEGG / Reactome / WikiPathways" — 2,939 sets across three databases is a large benchmark and the capability gradient is consistent. The catch is what this benchmark *is*: KEGG, Reactome, and WikiPathways are extensively cited in PubMed, and GPT-4o's training corpus almost certainly contains the canonical gene lists for these pathways. The hypergeometric test controls for "would a random |G|-gene draw from 19,846 overlap |D∩G| by chance" — it does **not** control for the LLM's tendency to over-emit pathway-promiscuous hub genes (TP53, AKT1, MYC, STAT1) that would inflate overlap with any immune / signaling pathway. **A proper null — shuffling the mapping between LLM descriptions and curated descriptions and checking that overrepresentation collapses — is absent from the paper.** So is any benchmark against MSigDB Hallmark or MSigDB C2, despite the paper explicitly framing itself as an alternative to "static, human-curated gene set databases." Hallmark in particular would be the harder test because its sets are biology-motivated rather than literature-co-occurrence, and they cannot be exploited from training-data memorization as easily as named KEGG pathways. C6 (the AL002 case study) is a single qualitative experiment on the sponsor's drug with the LLM context manually written by the authors; there is no quantitative metric, no second context-string ablation, and no domain-expert held-out pathway list.

## Method & Architecture

![llm2geneset overview: proposal and context-free regeneration feeding a hypergeometric ORA](/assets/images/paper/llm2geneset/page_014.png)
*Figure 1: llm2geneset architecture and benchmark setup (see source paper for the figure-internal labels). The proposal prompt receives the DEG list and emits N pathway descriptions; a context-free regeneration prompt converts each description back to a gene set; a one-tailed hypergeometric test computes overrepresentation against a 19,846-gene universe.*

### 1. Gene set generation prompt `GetGenes`

A three-part fixed prompt: (i) system role "You are an expert in cellular and molecular biology"; (ii) query "List all the known genes directly and indirectly involved in the following biological process or cellular component `{descr}`"; (iii) JSON-schema formatting instruction with a `gene` field, HGNC symbols only. Parsing uses `json-repair`; on irrecoverable JSON the model is re-queried with a new seed. Duplicates are removed post hoc.

Three optional prompt variants are added to the same template:

- **Reasoning.** Schema extended so each gene comes with a one-sentence rationale.
- **Confidence.** Each gene tagged low / medium / high; only high-confidence genes are kept.
- **Ensembling.** Five independent generations with different OpenAI `seed` values; only genes present in all five are retained (strict intersection).

### 2. Pathway proposal prompt `GetPathwaysProcesses`

Given a DEG list `D`, a desired number of pathways `N`, and an optional context string `C` (e.g. "in vitro microglia treated with a TREM2 agonist antibody"), the model returns `N` non-overlapping pathway / process / component descriptions in JSON, with instructions to "Be as specific as possible" and "Do not include the gene names in the outputs". The latter is the critical hygiene step that keeps the downstream regeneration context-free.

### 3. The llm2geneset loop

```
P = GetPathwaysProcesses(D, N, C)
for pathway in P:
    G = GetGenes(pathway)                    # context-free regeneration
    p = hgsf(|D∩G|-1, B-|D|, |D|, |G|)       # one-tailed hypergeometric
    R.append((pathway, p))
return padjust(R)                            # FDR / Bonferroni
```

where `B = 19,846` is the human protein-coding gene universe (GRCh38.p14 via biomaRt `hsapiens_gene_ensembl`) and `hgsf` is `scipy.stats.hypergeom.sf`. The non-circularity guarantee rests entirely on `GetGenes` not seeing `D`.

### 4. Baseline: GSAI

The comparison method (Hu et al., 2023/2024) is a single-prompt baseline that asks the LLM to directly name the biological process for a list of genes with per-gene reasoning and confidence. llm2geneset replaces this with proposal + regeneration + ORA.

![Prompt-variant ablation: reasoning, confidence, ensembling trade recall for precision](/assets/images/paper/llm2geneset/page_015.png)
*Figure 2: Prompt-variant ablation (see source paper for the figure-internal labels). Default prompt has the highest fraction-overrepresented (recall); reasoning, confidence, and ensembling reduce recall but raise the fraction of returned genes that are in the curated set (precision). Authors recommend confidence over ensembling on token-cost grounds.*

![llm2geneset vs GSAI on biological-process recovery, single and multi-process](/assets/images/paper/llm2geneset/page_016.png)
*Figure 3: llm2geneset proposal vs GSAI baseline on single- and multi-process recovery (see source paper for the figure-internal labels). Note that the mixed-set benchmark in panel C pre-filters pairs to those where llm2geneset's own single-set output had cosine > 0.7 — a test-set construction that selects for cases where the method already does well.*

## Experimental Results

### Headline benchmark — fraction of LLM gene sets significantly overrepresented in the matching curated set (Bonferroni p < 0.01), Fig. 1C

| Database     | GPT-3.5 | GPT-4o mini | **GPT-4o** |
|--------------|---------|-------------|------------|
| KEGG         | 0.89    | 0.84        | **0.94**   |
| Reactome     | 0.66    | 0.67        | **0.81**   |
| WikiPathways | 0.74    | 0.78        | **0.84**   |

A clean capability gradient: GPT-4o ≥ GPT-4o-mini ≥ GPT-3.5 on every database, and KEGG is the easiest of the three (KEGG pathway gene lists are the most cited in PubMed, which is also the most plausible explanation for the gradient).

### Prompt-variant ablation (Fig. 2)

Default prompt has the highest recall (fraction overrepresented); reasoning, confidence, and ensembling all drop recall (e.g. GPT-3.5 KEGG: 0.89 default → 0.74 reasoning → 0.60 confidence → 0.59 ensembled). Precision (fraction of returned genes in the curated set) rises for confidence and ensembling but not for reasoning. The authors recommend confidence over ensembling because confidence costs roughly 1/5 the output tokens (Fig. S2A-B).

### Formatting / hygiene (Fig. S1C-E)

- HGNC adherence: ~95% (GPT-4o), ~88% (GPT-3.5, GPT-4o-mini).
- Duplicate gene rate per set: 13-24% (GPT-3.5) vs 5-6% (GPT-4o).
- Role prompt had negligible effect on overrepresentation.

### Token cost (Fig. S1F)

~528,666 input tokens and **750,278 output tokens** for GPT-4o to generate all 2,939 sets (~260 output tokens per set).

### Proposal-vs-GSAI (Fig. 3B, S3A) on GPT-4o

llm2geneset reports higher mean shared-unigrams, shared-bigrams, and `text-embedding-3-large` cosine similarity than GSAI on all three databases. Only GPT-4o is shown in the main figure; cross-model panels are in the supplement.

### Multi-process recovery (Fig. 3C)

50 pairs of high-similarity gene sets per database were combined and shuffled. llm2geneset outperforms GSAI on all three metrics. **Caveat:** pairs are pre-filtered to those where llm2geneset's own prior single-set output scored cosine > 0.7 — a test-set construction that throws away the hard cases where llm2geneset already fails.

### TREM2 / AL002 microglia case study (Fig. 4)

![AL002 / TREM2 microglia case study: context-steered llm2geneset vs KEGG ORA](/assets/images/paper/llm2geneset/page_017.png)
*Figure 4: AL002 / TREM2 agonist antibody microglia case study (see source paper for the figure-internal labels). KEGG ORA top hits are dominated by viral-defense-overlap diseases; GPT-4o llm2geneset with the microglia context produces "Regulation of inflammatory response", "Innate immune response", "Cellular response to TNF", "Leukocyte chemotaxis".*

- 30 DEGs at FDR 10% from DESeq2 on bulk RNA-seq of iMGs treated with AL002 (n=4) vs isotype IgG (n=4), 1 µg/ml, 24 h.
- **KEGG ORA** top hits: "Measles", "Rheumatoid arthritis", "Chagas disease", "Influenza A", "Coronavirus disease" — almost certainly driven by viral-defense gene overlap, not by actual viral biology in the experiment.
- **GPT-4o llm2geneset, no context:** "Defense response to virus", "Antiviral defense", "Type I interferon signaling", but also irrelevant "Bone resorption", "Gluconeogenesis", "Osteoclast differentiation" (driven by ACP5, DCSTAMP).
- **GPT-4o llm2geneset + microglia context:** "Regulation of inflammatory response" (FDR 1%), "Innate immune response", "Cellular response to TNF", "Leukocyte chemotaxis" — qualitatively far more on-target.

## Limitations

- **No random-gene-set negative control.** The hypergeometric test is not the same as a random-gene baseline — it controls for chance overlap given a fixed gene universe but not for the LLM's bias toward pathway-promiscuous hub genes (TP53, AKT1, MYC, STAT1) that would overlap any immune / signaling pathway. A proper null — shuffling the mapping between LLM descriptions and curated descriptions and checking that overrepresentation collapses — is not in the paper.
- **MSigDB Hallmark and MSigDB C2 are absent**, despite the paper positioning itself as an alternative to "static, human-curated gene set databases." Hallmark is the harder test (biology-motivated, not literature-co-occurrence) and would be much harder to satisfy via training-data memorization.
- **Training-data contamination.** KEGG, Reactome, and WikiPathways are heavily cited in PubMed; GPT-4o's training corpus almost certainly contains the canonical gene lists for these pathways. The benchmark is effectively "can GPT-4o recall pathway annotations from its training data," not "can it discover novel biology."
- **Test-set selection bias in Fig. 3C.** Multi-process recovery pairs are filtered to those where llm2geneset's own single-set output already scored cosine > 0.7, throwing away the hard cases.
- **"Significantly improved precision" in Fig. 2C** is stated without specifying a test, p-value, or n.
- **No variance / CIs across seeds for the main Fig. 1C benchmark.** OpenAI `seed` is best-effort, not deterministic, and only one generation per set per model is shown.
- **Single LLM family.** No Claude, Gemini, Llama, BioMedLM, or other non-OpenAI baseline.
- **AL002 case study** is one experiment, qualitatively assessed, on the sponsor's drug, with the LLM context string manually written by the authors. No held-out validation, no second context-ablation, no domain-expert pathway list. All authors are Alector employees.
- **Independence assumption inside the algorithm.** The proposal step emits N descriptions that are not statistically independent (LLM-correlated), which complicates the FDR correction applied to the resulting N p-values.

## Why It Matters for Medical AI

The end-to-end framing — bulk RNA-seq → DEG list → LLM-proposed pathways tailored to experimental context → standard ORA — is a plausible drop-in for the manual KEGG/Reactome/Hallmark workflow that bench biologists run after every differential-expression analysis. The AL002 case study makes the use-case concrete: KEGG returns viral-disease hits driven by shared interferon genes, while a microglia-context LLM prompt returns inflammatory-response and innate-immunity terms that match the known TREM2 biology of the drug under study. For translational programs where the gene list is small (here, 30 DEGs) and the experimental context is well-specified, dynamic pathway proposal could be useful even today.

That said, the medical-AI reader should not confuse "the LLM recovers the canonical KEGG gene list" with "the LLM understands the biology." This paper does not yet support the latter. The headline number measures how well GPT-4o memorized publicly available pathway databases — a useful property for an interactive interpretation tool, but not the same as the discovery claim implied by the paper's framing. Until the framework is tested against MSigDB Hallmark, a random-description null, and at least one non-OpenAI model, "LLM gene sets replace curated databases" should be read as a hypothesis, not a result.

## References

- Paper (bioRxiv preprint, posted 14 Nov 2024): [10.1101/2024.11.11.621189](https://doi.org/10.1101/2024.11.11.621189)
- Code: [github.com/Alector-BIO/llm2geneset](https://github.com/Alector-BIO/llm2geneset) — includes a Streamlit webapp
- GSAI baseline (Hu et al.): "Evaluation of large language models for discovery of gene set function"
- Curated gene set sources (Enrichr): `KEGG_2021_Human`, `Reactome_2022`, `WikiPathway_2023_Human` from [maayanlab.cloud/Enrichr](https://maayanlab.cloud/Enrichr/#libraries)
- AL002 clinical context: Alector / GSK TREM2 agonist antibody, Phase 2 INVOKE-2 trial in early Alzheimer's disease
- MSigDB Hallmark / C2 (not benchmarked in this paper but the natural next test): [gsea-msigdb.org](https://www.gsea-msigdb.org/gsea/msigdb)
