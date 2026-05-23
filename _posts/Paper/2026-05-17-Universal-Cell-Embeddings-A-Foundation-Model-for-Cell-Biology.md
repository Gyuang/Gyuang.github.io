---
title: "Universal Cell Embeddings: A Foundation Model for Cell Biology"
excerpt: "A 650M-parameter transformer trained on ~36M cells across 8 species produces a single zero-shot cell latent that beats Geneformer by +9.0% overall scIB on Tabula Sapiens v2 and transfers to a never-seen vertebrate class (chicken) — but ships with zero ablations on its load-bearing ESM2 gene-tokenization choice."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/uce/
tags:
  - UCE
  - Foundation-Model
  - Single-Cell
  - scRNA-seq
  - ESM2
  - Transformer
  - Zero-Shot
  - Cross-Species
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- A 33-layer, 650M-parameter transformer is pretrained self-supervised on ~36M scRNA-seq cells from 8 species (40 days on 24× A100 80GB) to produce a single 1280-d cell embedding that needs no fine-tuning on new datasets, tissues, or even unseen species.
- The load-bearing design is *protein-language-model tokenization of genes*: every gene is represented by the mean ESM2 embedding (d=5120) of its protein product, so any gene from any organism gets a usable token without a homolog mapping table.
- On Tabula Sapiens v2 (held-out, 581,430 cells, 162 cell types, 167 batches), UCE beats Geneformer zero-shot by **+9.0% overall scIB (+10.6% bio-conservation, +7.4% batch-correction)** and is "slightly better" than fine-tuned scVI/scArches — but the paper reports *zero ablations*, no variance, and no statistical tests on these headline numbers.

## Motivation

scRNA-seq atlases now span hundreds of datasets, dozens of tissues, and multiple species, but the standard integration tools (scVI, scArches, Harmony) require per-dataset retraining and assume a shared gene vocabulary. Cross-species transfer typically demands an explicit homolog map, which discards the long tail of species-specific genes. The authors frame this as cell biology's missing "foundation-model moment": one pretrained encoder that maps any cell into one universal latent without retraining — the cell-biology analogue of CLIP for vision or ESM2 for proteins.

The downstream medical-AI angle is concrete: a universal cell embedding enables cross-tissue, cross-disease search (the paper's Norn-cell case study reaches into IPF/COPD lungs) and zero-shot annotation of patient-derived scRNA-seq without per-study label curation.

## Core Innovation

- **ESM2-tokenized genes (species-agnostic).** Every protein-coding gene `g` is represented by the mean ESM2 embedding of all proteins it codes for. A never-before-seen gene from a new organism just gets its ESM2 embedding computed from its amino-acid sequence — no homolog lookup.
- **Chromosome-sorted "cell sentences."** Sample 1024 genes (with replacement) from a cell's expressed set, weighted by log-expression; group by chromosome, sort within chromosome by genomic location, delimit chromosomes with species-specific tokens, randomize chromosome order, prepend `CLS`.
- **Masked gene-presence prediction objective.** Mask 20% of expressed genes before sampling; predict their presence/absence from `[h_cell || MLP(ESM2 token)]` with binary cross-entropy. The cell-level supervisor is the existence of a gene's transcript, not its count value.
- **No HVG selection, no batch correction, no homolog mapping at inference.** A new dataset is preprocessed minimally (≥200 genes/cell, ≥10 cells/gene) and forward-passed once.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|-------|----------|------------|----------|
| C1 | UCE generates universal cell embeddings without fine-tuning | scIB metrics on a held-out atlas | TSv2 (held out) | ⭐⭐⭐ |
| C2 | UCE beats other zero-shot transformer foundation models | +9.0% overall / +10.6% bio / +7.4% batch vs. Geneformer; UCE wins 67–83% of cell-type silhouettes | TSv2 | ⭐⭐ — strong but single eval set, no variance, no significance test |
| C3 | UCE matches or exceeds fine-tuned methods (scVI, scArches) | "Slightly better" in scIB overall | TSv2 | ⭐⭐ — "slightly" is unquantified; no CI; could be within noise |
| C4 | UCE generalizes to species never seen in training (incl. chicken — new vertebrate class) | Nearest-centroid matching: 13/17 (green monkey), 17/24 (naked mole rat), 12/15 (chicken heart) | 3 independent species | ⭐⭐⭐ — strongest claim in the paper |
| **C5** | **ESM2 protein-LM tokenization is what enables cross-species transfer** | **Architectural argument only — no comparison to learned gene embeddings or one-hot+homolog baseline** | — | ⭐ — *no ablation*. Asserted, not shown. |
| C6 | Embedding geometry recovers Cell-Ontology structure | Monotonic distance vs. tree-distance up to 5 hops, one-sided t-test | TSv2 | ⭐⭐⭐ |
| C7 | UCE captures developmental lineage without supervision | >80% germ-layer classification on held-out cell types | TSv2 | ⭐⭐ — single classifier, single split |
| C8 | UCE enables discovery of novel cell biology (Norn cells in non-kidney tissues) | Marker-gene LFC patterns; COPD vs IPF Epas1:Egln1 p=0.035 | mouse kidney, human IPF/COPD lung | ⭐⭐ — single uncorrected p-value; no wet-lab follow-up |
| C9 | UCE handles datasets without preprocessing (no HVG, no batch correction) | Directly demonstrated by the zero-shot protocol | TSv2 | ⭐⭐⭐ |
| C10 | 650M + ESM2 + masked gene-prediction is the right recipe | None — no scaling curve, no compute-matched comparison | — | ⭐ |

**Honest read.** The cross-species transfer (C4) and ontology-structure (C6) results are the paper's genuinely strong contributions — they are hard to achieve by chance and span multiple independent datasets. The headline zero-shot benchmark numbers (C2/C3) are real but oversold: they live on a *single* held-out dataset (TSv2, the same Tabula Sapiens project as v1 which is in pretraining), there is no variance reporting, no error bars in the main quantitative comparisons, and no statistical test on the +9.0% delta. The most important missing experiment is an **ablation on ESM2 tokenization** (C5): the paper's central architectural claim — that protein-LM tokens are what makes UCE universal — is *asserted but never tested* against, for example, a learned gene embedding table or a one-hot-with-homolog-mapping baseline. The masking ratio, the chromosome-sorting positional scheme, the 1024-gene sampling budget, and the 650M-parameter scale are all similarly unablated. A compute footprint of 40 days × 24 A100s makes independent replication essentially infeasible, which makes the absence of ablations particularly costly. The Norn-cell case study (C8) is presented as discovery but is hypothesis generation — the wet-lab follow-up isn't done and the lone reported p-value (0.035) is unadjusted.

## Method & Architecture

![Universal Cell Embedding pipeline: gene expression → expression-weighted sampling → ESM2 protein tokens → chromosome-sorted transformer → CLS cell embedding](/assets/images/paper/uce/page_008.png)
*Figure 1a: Universal Cell Embedding pipeline — gene expression is converted into a chromosome-sorted "cell sentence" of ESM2 protein tokens, then encoded by a 33-layer transformer; the CLS output is the cell embedding.*

![Cell-sentence construction and masked gene-presence training objective](/assets/images/paper/uce/page_008.png)
*Figure 1a (detail): Cell-sentence construction and the masked gene-presence pretraining objective.*

### 1. Gene tokenization via protein LM

For every protein-coding gene `g`, pre-compute `p_g`, the mean ESM2 embedding (d_p = 5120) over all proteins gene `g` codes for. The dictionary is species-agnostic: a new organism's never-seen genes get their ESM2 embeddings computed from amino-acid sequence — no homolog table.

### 2. Cell → cell sentence

For cell `c_i` with expression vector `x_i`:

- Split genes into expressed `G_i^+ = \{g : x_g^i > 0\}` and non-expressed `G_i^-`.
- Sample a multiset `G_i^s` of **1024 genes with replacement** from `G_i^+`, weighted by log-normalized expression:

$$ P(g \mid c_i) = \frac{\log(x_g^i)}{\sum_{g' \in G_i^+} \log(x_{g'}^i)} $$

- Group sampled genes by chromosome, sort within chromosome by genomic location, delimit each chromosome with species-specific start/end tokens, randomize chromosome order, prepend `CLS`. The result is the cell sentence `S_i`.

### 3. Transformer encoder

Each gene's d_p = 5120 ESM2 token is compressed to d_emb = 1280 by a single-layer MLP, sinusoidal positional embeddings are added, and the sequence runs through a 33-layer transformer with multi-head self-attention. The cell embedding is the final-layer CLS output (optionally passed through a decoder MLP).

### 4. Self-supervised pretraining

- Mask 20% of expressed genes (`G_i^{M+}`) before sampling — held out from the cell sentence.
- Construct loss sets: `G_i^{L+}` (N_loss/2 drawn from masked-expressed) and `G_i^{L-}` (N_loss/2 from non-expressed).
- For every query gene `g` in `G_i^{L+} \cup G_i^{L-}`, concatenate the cell embedding with the (MLP-compressed) ESM2 token: `z_g^i = [h_\text{cell}^i \| \text{MLP}(p_g)]`, then predict expression indicator `y_g^i \in \{0,1\}` via an MLP with binary cross-entropy:

$$ \mathcal{L} = -\frac{1}{N} \sum_i \frac{1}{N_\text{loss}} \sum_j \Big[ y_j^i \log p(y_j^i) + (1 - y_j^i) \log(1 - p(y_j^i)) \Big] $$

### 5. Scale and compute

33 layers, ~650M parameters, trained on **>300 datasets / ~36M cells** for **40 days on 24× A100 80GB GPUs**. d_emb = 1280, d_p = 5120, sequence length 1024.

### 6. Inference (zero-shot)

Minimal preprocessing (≥200 genes/cell, ≥10 cells/gene; no HVG, no batch correction). Gene→protein mapping is computed (ESM2 embeddings looked up or computed for new genes), and cells are forward-passed once to obtain `h_cell`.

## Experimental Results

![UMAP of Tabula Sapiens v2 colored by cell type, comparing scVI / scArches / UCE / Geneformer / scGPT](/assets/images/paper/uce/fig_p025_01.png)
*Figure 2b: UMAPs on Tabula Sapiens v2 colored by cell type — UCE zero-shot recovers fine-grained clusters competitive with fine-tuned scVI/scArches.*

**Main zero-shot comparison (Tabula Sapiens v2, scIB).** Deltas are UCE-over-baseline as reported in the text (the paper does not present a single condensed table in the main body). UCE row is bolded as the paper's own method.

| Setting | Method | Bio-conservation | Batch-correction | Overall |
|---|---|---|---|---|
| Zero-shot | **UCE** | **best** | **best** | **best** |
| Zero-shot | Geneformer | UCE +10.6% | UCE +7.4% | UCE +9.0% |
| Zero-shot | scGPT | worse than Geneformer | — | — |
| Zero-shot | tGPT | worse than Geneformer | — | — |
| Fine-tuned | scVI | — | — | UCE "slightly better" |
| Fine-tuned | scArches | — | — | UCE "slightly better" |

**Per-cell-type silhouette (TSv2):** UCE has highest silhouette in **67%** of cell types; beats Geneformer on **80%**, tGPT on **73%**, scGPT on **83%**. For B cells specifically, UCE silhouette is **+93% over scGPT and +25% over Geneformer**.

![Cross-species transfer: zero-shot embedding of green monkey lymph node cells with predicted cell types aligning to ground truth](/assets/images/paper/uce/fig_p025_01.png)
*Figure 2d: A logistic classifier trained on human IMA UCE embeddings transfers zero-shot to green monkey lymph node; predicted and ground-truth cell types align across species.*

**Cross-species zero-shot (nearest-centroid cell-type matching, Extended Data Table 1).**

| Species (not in training) | Tissue | Correct top-1 | Correct top-3 |
|---|---|---|---|
| Green monkey | lymph node + lung | 13 / 17 | 17 / 17 (100%) |
| Naked mole rat | spleen + circulating immune | 17 / 24 | — |
| Chicken (bird — entirely new clade) | heart | — | 12 / 15 (top-2) |
| Chicken | retina | qualitative match (oligodendrocytes ↔ mouse lemur oligodendrocytes) | — |

**Within-species, within-tissue.** For human macrophages across 73 tissues, **72%** of tissue-specific macrophage centroids have another macrophage centroid as nearest neighbor; **93%** within top-3.

![UMAP of TSv2 lung cells in UCE space preserving both fine cell-type clusters and higher-level groupings](/assets/images/paper/uce/fig_p026_01.png)
*Figure 3a: UCE embedding of TSv2 lung cells preserves fine cell-type clusters and higher-level immune/epithelial/endothelial groupings simultaneously.*

![Embedding distance increases monotonically with Cell-Ontology tree distance](/assets/images/paper/uce/fig_p026_02.png)
*Figure 3b: Embedding distance increases monotonically with Cell-Ontology tree distance, up to 5 hops (one-sided t-test).*

![Per-tissue cell-type alignment accuracy between TSv2 and IMA across 27 tissues, UCE vs raw gene-expression space](/assets/images/paper/uce/fig_p026_03.png)
*Figure 3d: Per-tissue cell-type alignment accuracy (top-3 nearest centroids) between TSv2 and the IMA across 27 tissues — UCE vs. raw gene-expression space.*

**Cell-ontology structure.** Distance is monotonically increasing in Cell-Ontology tree distance up to 5 hops, statistically significant by one-sided t-test (Fig. 3b). A NN classifier predicts germ-layer of origin for held-out cell types with **>80% accuracy** (Supp. Fig. 7b).

**TSv2 → IMA cell-type alignment (Fig. 3d).** Average **56%** of tissues correctly matched in top-3 nearest centroids in UCE space; UCE is **60% more accurate at top-3** and **93% more accurate at top-1** than the raw-expression baseline.

![Norn-cell discovery workflow: cluster in one tissue, train logistic classifier on UCE embeddings, search across all IMA tissues](/assets/images/paper/uce/fig_p027_01.png)
*Figure 4a: Norn-cell discovery workflow — cluster in one tissue, train a logistic classifier on UCE embeddings, then run an unbiased search across all IMA tissues.*

![Log-fold-change of Norn marker genes across predicted Norn-like cells in kidney, lung, and heart](/assets/images/paper/uce/fig_p027_07.png)
*Figure 4c: Log-fold-change of Norn marker genes (Dcn, Lpar1, Col1a1, Cxcl12, Cfh) in predicted Norn-like cells across kidney, lung, and heart datasets.*

**Norn-cell case study.** A logistic regression on UCE embeddings, trained on mouse kidney, recovers Norn-cell clusters in 13 kidney datasets and additionally flags "Norn-like" cells in lung and heart that differentially express known Norn markers (Dcn, Lpar1, Col1a1, Cxcl12, Cfh). Applied to an IPF/COPD/control lung cohort, COPD predicted-Norn cells show higher Epas1:Egln1 than IPF predicted-Norn cells (p = 0.035, unadjusted), consistent with the clinical observation that COPD patients have higher serum Epo than IPF patients (n is small).

**Ablations.** *None reported.* The paper does not ablate (a) ESM2 vs. learned gene embeddings, (b) the chromosome-sorting positional scheme, (c) the 20% masking ratio, (d) the 1024-gene sampling budget, (e) the 650M model size. This is the single most important methodological gap.

## Limitations

**Authors acknowledge:**

- Benchmarks are limited to coarse cell-type labels; resolution-aware benchmarks don't exist.
- UCE (like all current scRNA-seq foundation models) discards transcript-level information — splicing, isoforms, genetic variants — when collapsing to gene-level counts.
- Future direction: "virtual cells" that incorporate transcript-level features.

**Not addressed by the authors:**

- **No ablations whatsoever.** ESM2 vs. learned gene embeddings, chromosome positional scheme, masking ratio, model size, sampling budget, sequence length — all untouched.
- **No variance or seed reporting** on the scIB benchmark numbers.
- **No statistical significance test** on the headline +9.0% improvement over Geneformer.
- **No compute-matched comparison** with baselines — UCE is ~10× larger than Geneformer.
- **Disease / clinical generalization is anecdotal.** The IPF/COPD work is exploratory hypothesis generation, not a benchmarked diagnostic application; the Norn workflow rests on small-n analyses.
- **TSv2 is same-project as the training set (TSv1).** "Zero-shot" is closer to "zero-shot within-distribution" than truly out-of-distribution clinical data.
- **Inference cost / memory footprint** for embedding new atlases is not characterized.
- **Distribution shift to non-CxG, non-academic data** (clinical biobank scRNA-seq, FFPE, lower-quality preps) is untested.
- **Cell-state vs. cell-type axis.** UCE is evaluated on cell-type identity; whether it captures continuous states (cell cycle, activation, perturbation response) is not assessed.

## Why It Matters for Medical AI

A single pretrained encoder that maps any cell — from any species, tissue, or study — into one comparable latent without retraining changes the unit economics of single-cell analysis in two ways relevant to clinical work. First, it makes **cross-cohort, cross-disease search** tractable: the Norn-cell case study shows the kind of unbiased "find cells like these elsewhere in the atlas" workflow that a universal embedding enables, even if the specific Norn/COPD/IPF findings are exploratory. Second, the **cross-species generalization** (especially to chicken, a new vertebrate class) is a credible signal that the ESM2-tokenized representation captures biology beyond the human/mouse training distribution — which matters for translational pipelines that move between model organisms and patient data. The big caveat is the same one that applies to every scRNA-seq foundation model right now: zero-shot performance is benchmarked on lab-grade atlases (Tabula Sapiens family), not on the messier clinical preps (FFPE, low-input, biobank-scale heterogeneity) where downstream clinical utility actually lives.

## References

- Paper: Rosen et al., "Universal Cell Embeddings: A Foundation Model for Cell Biology," bioRxiv preprint (2023). DOI: [10.1101/2023.11.28.568918](https://doi.org/10.1101/2023.11.28.568918)
- Code: [https://github.com/snap-stanford/uce](https://github.com/snap-stanford/uce)
- Related: Geneformer (Theodoris et al., Nature 2023); scGPT (Cui et al., Nature Methods 2024); tGPT (Shen et al.); scVI / scArches (Lopez et al.; Lotfollahi et al.); ESM2 (Lin et al., Science 2023); CellxGene Census v2023-07-10; Tabula Sapiens (TSP Consortium, Science 2022).
