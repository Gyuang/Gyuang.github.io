---
title: "Sparse Autoencoders Reveal Organized Biological Knowledge but Minimal Regulatory Logic in Single-Cell Foundation Models: A Comparative Atlas of Geneformer and scGPT"
excerpt: "TopK SAEs on every layer of Geneformer V2-316M and scGPT recover 82,525 + 24,527 features with 99.8% superposition, 29-59% ontology coverage, and median 2.36x causal specificity — yet only 6.2% (K562) -> 10.4% (multi-tissue) of TRRUST TFs get target-specific responses, isolating the model as the bottleneck."
categories:
  - Paper
  - BioInformatics
permalink: /paper/sparse-autoencoders-scgpt-geneformer-atlas/
tags:
  - Sparse-Autoencoders
  - Mechanistic-Interpretability
  - Single-Cell-Foundation-Models
  - Geneformer
  - scGPT
  - Superposition
  - TRRUST
  - CRISPRi
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- TopK sparse autoencoders (k=32, 4x overcomplete) are trained on the residual stream of every layer of **Geneformer V2-316M (18L)** and **scGPT whole-human (12L)**, producing the first comparative feature atlases for single-cell foundation models — **82,525 + 24,527 alive features**, two interactive websites, and a multi-tissue control.
- The findings split cleanly: scFMs **richly encode organized biology** — **99.8% superposition**, **29-59% ontology coverage**, **141 vs 76 co-activation modules**, **median 2.36x causal specificity**, **97-99.8% cross-layer information highways** — **but** they **fail at causal regulatory logic**.
- Headline negative result: TRRUST-anchored perturbation specificity is only **6.2% (3/48 TFs)** for K562-only SAEs and **10.4% (5/48 TFs)** for a balanced K562+Tabula Sapiens control, with **disjoint gain/loss sets (5 gained, 3 lost, 40 unchanged)** — non-systematic gain that pins the ceiling on the **model**, not on SAE training data.

## Motivation

Geneformer and scGPT have become default scFMs for cell-type annotation, perturbation prediction, and gene network inference, and the field routinely interprets attention edges as "regulatory" signal. A companion paper has already shown that attention captures co-expression, not regulation — but attention is one slice of a transformer; the residual stream is where information accumulates. Sparse autoencoders are the leading tool for resolving superposition in LLMs and have not been systematically applied to scFMs. The medical-AI stakes are direct: if scFMs are being proposed for perturbation prediction, drug response, and disease subtyping, these uses are only justified if the model has internalized **regulatory mechanism** rather than expression-pattern lookup. This paper asks that question with the right tool.

## Core Innovation

- **First systematic SAE interpretability for scFMs.** Identical TopK pipeline applied to two architecturally divergent models (next-token rank-value Geneformer vs masked continuous-expression scGPT), so qualitative-vs-quantitative convergence becomes a question one can actually pose.
- **Multi-tissue ablation isolates the bottleneck.** Retraining Geneformer SAEs on a balanced 1M-position pool (500K K562 + 500K Tabula Sapiens) keeps the model fixed and varies only SAE training data — separating "the SAE didn't see enough biology" from "the model doesn't encode regulation."
- **Cross-layer PMI graphs** define "information highways" by jointly encoding the same 500K positions through source- and target-layer SAEs, exposing **97-99.8%** functional connectivity in Geneformer and a **progressive bottleneck (95.7% -> 62.9%)** in scGPT.
- **TRRUST-anchored single-feature causal patching.** Zero out one SAE feature, decode, replace the residual stream, continue forward — specificity ratio is target-position |delta| / other-position |delta|. Median 2.36x; top feature 114.5x.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | scFMs use massive superposition; only 0.2% of SAE features align with top-50 SVD (cos > 0.7) | Table 1: 189/82,525 SVD-aligned across 18 Geneformer layers; replicated qualitatively in scGPT | ⭐⭐⭐ |
| C2 | Novel (non-SVD) features carry essentially all the biological signal | Annotation rate **52.5% (novel) vs 14.3% (SVD-aligned)**; 43,214 vs 27 annotated features | ⭐⭐⭐ |
| C3 | Geneformer and scGPT converge on qualitatively similar structure despite divergent design | Table 3 head-to-head; Fig 3 normalized-depth curves; modules/layer 7.8 vs 6.3; cell-type tiling both 99%+ | ⭐⭐⭐ |
| C4 | Features organize into 141 (Geneformer) / 76 (scGPT) biologically coherent co-activation modules | Section 2.6, Leiden + permutation-null PMI; module identities author-named from top genes | ⭐⭐ — no external pathway-clustering benchmark |
| C5 | U-shaped annotation profile reflects hierarchical abstraction (molecular -> abstract -> re-specialized -> prediction) | Fig 4; 57-59% L0-L1, trough 45.4% L8, recovery 55-56% L10-L11, decline 47-55% L15-L17 | ⭐⭐ — pattern is real, four-zone narrative is post-hoc |
| C6 | SAE features are causally necessary; median 2.36x specificity, top 114.5x | Table 5, Fig 6; single-feature ablation through L12-L17, 50 features x 200 cells | ⭐⭐ — single layer, single dataset, 50 hand-picked features; no negative-control distribution; scGPT analog (median 0.98x) is confounded by proxy expression |
| C7 | Feature-level causation succeeds where component-level (head/MLP) ablation fails | Contrast with companion attention paper's null results | ⭐⭐ — cross-paper comparison, not within-study |
| C8 | Models internalize co-expression but **not** causal regulatory logic | Table 6: **6.2% TRRUST specificity (3/48 TFs)** with 92% perturbation detection rate | ⭐⭐⭐ for "specificity is low" / ⭐⭐ for the absolute claim — TRRUST is incomplete, Geneformer only, single layer, no non-transformer baseline |
| C9 | The bottleneck is the model, not the SAE training data | Table 7: 6.2% (K562) -> 10.4% (K562+TS), 5 gained / 3 lost / 40 unchanged (disjoint sets); TF feature representation 64.5% -> 60.5% | ⭐⭐⭐ for ruling out SAE data diversity; leaves open SAE architecture/k |
| C10 | Features are layer-specific; biology is rebuilt every layer | 2-3% adjacent decoder-cosine persistence; persistent features annotate at 3.7% vs 59.6% for transient | ⭐⭐ — decoder-cosine only; functional cross-layer PMI tells the complementary story |
| C11 | Information flows through cross-layer functional connections (97-99.8% highways) | Section 2.8; cross-layer PMI between source/target-layer feature activations | ⭐⭐⭐ |
| C12 | Most unannotated features are not noise (95-98.5% co-active with annotated features in modules) | Section 2.11 | ⭐ — near-tautological given 6-12 modules cover 96-99.5% of features; standalone Jaccard clusters only 2-3% |

**Overall.** The strongest defensible results are **C1-C2 (superposition)**, **C3 (cross-model convergence)**, **C8-C9 (regulatory-specificity ceiling with multi-tissue control)**, and **C11 (information highways)**. The paper is unusually disciplined for an interpretability work — the **negative finding is the headline**, and the multi-tissue ablation is exactly the right control to isolate model-vs-data. Weak spots: (a) single-feature causal patching is shown only at Geneformer L11 on 50 hand-picked annotated features with no matched negative-control feature distribution and **no error bars on the 2.36x median**; (b) scGPT causal numbers are knowingly confounded by a uniform 1.0 expression proxy (continuous values were not preserved during extraction); (c) **TRRUST is a thin ground truth** — 48 TFs is a small panel and TRRUST-negative is not regulation-negative; (d) **no Norman / Adamson CRISPRi benchmark** to test whether the 6.2% ceiling generalizes; (e) no baseline-model anchor (a randomly initialized Geneformer or linear PCA null is missing); (f) the 4.2 pp K562-vs-multi-tissue swing depends on 2 TFs (3 -> 5 of 48) and has no bootstrap CI.

## Method & Architecture

![Geneformer SAE atlas across 18 layers — variance explained, dead features, annotation rate, SVD-aligned](/assets/images/paper/sparse-ae-scgpt-geneformer/page_004.png)
*Figure 1: Geneformer V2-316M SAE atlas across 18 layers — 82,525 alive features (0.5% dead), reconstruction declines with depth (81.7% mean variance explained) while only 0.2% of features align with the top-50 SVD axes.*

### 1. Activation extraction

Geneformer is run over **2,000 K562 control cells from the Replogle CRISPRi dataset** (mean 2,028 genes/cell, **4,056,351 token positions/layer**, 336.4 GB float32 total across 18 layers). scGPT is run over **3,000 Tabula Sapiens cells** (1k immune across 43 types + 1k kidney + 1k lung, **3,561,832 positions/layer**, ~82 GB total). PyTorch hooks save residual outputs as memory-mapped NumPy arrays.

### 2. TopK SAE

For input $x \in \mathbb{R}^d$ centered by the training-set mean $\mu$:

$$
h = W_{enc}(x - \mu) + b_{enc} \in \mathbb{R}^{4d}
$$

TopK sparsification keeps only the $k=32$ largest entries; the decoder reconstructs

$$
\hat{x} = W_{dec}\, h_{\text{sparse}} + \mu
$$

with **unit-normalized decoder columns after every gradient step**. Loss is plain $L = \lVert x - \hat{x} \rVert^2$. Geneformer: **18 SAEs x 4,608 features each**, 1M positions/layer subsample. scGPT: **12 SAEs x 2,048 features each**, full 3.56M positions/layer. Adam, lr $3\!\times\!10^{-4}$, batch 4,096, 5 epochs. Dead features = zero activations on 100K held-out positions.

### 3. Ontology annotation

For each alive feature, take the **top-20 genes by mean activation magnitude**, run **one-sided Fisher's exact + BH-FDR < 0.05** against **GO BP, KEGG, Reactome, STRING, TRRUST**. A feature is "annotated" if it has at least one significant enrichment. SVD comparison: top-50 singular vectors per layer; "SVD-aligned" if decoder cosine > 0.7 with any axis.

### 4. Co-activation modules and cross-layer highways

Pairwise **PMI** $\log_2 P(i,j)/[P(i)P(j)]$ across all positions (active = in that position's top-$k$), permutation $p<0.001$, then **Leiden community detection at resolution 1.0**. **Information highways** re-encode 500K positions through *both* source and target layer SAEs and measure cross-layer feature-activation PMI; an edge counts if PMI > 3.

### 5. Single-feature causal patching

At Geneformer L11, for each of **50 richly annotated features**: encode the hidden state through the SAE, **zero that feature only**, decode back, replace the hidden state, continue forward through L12-L17. Specificity ratio $= |\overline{\Delta_{\text{target}}}| / |\overline{\Delta_{\text{other}}}|$ where "target" = positions matching the feature's ontology genes. 200 cells per feature. scGPT L7 follows the same protocol but uses a documented **uniform 1.0 expression proxy** because original continuous values were not preserved during extraction.

### 6. Perturbation response mapping

**100 CRISPRi targets (48 TRRUST TFs + 52 non-TF genes)**, 20 perturbed cells each, vs 100K control positions. L11 SAE feature activations tested with **Wilcoxon rank-sum + BH-FDR**, effect $|\Delta|>0.5$. A TF is "specific" iff responding features' top genes are Fisher-enriched for that TF's TRRUST targets.

### 7. Multi-tissue control

Retrain Geneformer SAEs at L0/5/11/17 on a balanced 1M-position pool: **500K K562 + 500K Tabula Sapiens**. Identical architecture, identical downstream analysis. Holds the model fixed and varies only SAE training-data diversity.

## Experimental Results

### Cross-model atlas (Table 3, Figure 3)

![SAE vs SVD comparison — only 0.2% of features align with SVD, but novel features carry 98.7% of ontology hits](/assets/images/paper/sparse-ae-scgpt-geneformer/page_005.png)
*Figure 2: SAE vs SVD. 99.8% of SAE features are invisible to the top-50 SVD; the novel 99.8% carry 98.7% of ontology hits and explain 2.4x more variance than the SVD subspace.*

![Cross-model head-to-head table — Geneformer 18L/d=1152 next-token vs scGPT 12L/d=512 masked-expression](/assets/images/paper/sparse-ae-scgpt-geneformer/page_006.png)
*Table 3: Cross-architecture comparison under an identical SAE pipeline — Geneformer encodes more biology per feature (52.4% vs 31.0% annotation), scGPT reconstructs better (90.2% vs 81.7% variance explained).*

![Cross-model curves on normalized depth — variance, annotation, dead features, modules per layer](/assets/images/paper/sparse-ae-scgpt-geneformer/page_007.png)
*Figure 3: On normalized depth, scGPT shows higher reconstruction quality at every depth while Geneformer shows higher annotation rates — convergent qualitative structure with divergent quantitative profile.*

| Metric | Geneformer V2-316M | scGPT whole-human |
|---|---|---|
| Architecture | 18L, d=1,152, next-token | 12L, d=512, masked gene |
| Total alive features | **82,525** | **24,527** |
| Dead features | 419 (0.5%) | 49 (0.2%) |
| Mean variance explained | **81.7%** (76.8-85.3%) | **90.2%** (85.7-93.5%) |
| Mean annotation rate | **52.4%** (45.4-58.6%) | **31.0%** (28.7-33.9%) |
| Co-activation modules total / per layer | 141 / 7.8 | 76 / 6.3 |
| Module coverage of alive features | 96.0-99.5% | 96.3% mean |
| SVD-aligned (cos > 0.7) | 189 / 82,525 (**0.2%**) | "99.8% novel" same qualitative |
| Annotation rate, SVD-aligned vs novel | 14.3% vs 52.5% | — |
| SAE vs top-50 SVD variance | 77-85% vs 31-38% (**2.4x**) | comparable |

### Representative features and U-shape

![Representative SAE features at Geneformer L0 vs L11](/assets/images/paper/sparse-ae-scgpt-geneformer/page_008.png)
*Table 4: Representative SAE features — L0 captures molecular programs (cell cycle, DNA replication, MAPK/TGF-beta); L11 captures integrative programs (cell differentiation, ERAD, mitochondrial organization).*

![U-shaped per-ontology annotation profile across 18 Geneformer layers](/assets/images/paper/sparse-ae-scgpt-geneformer/page_009.png)
*Figure 4: U-shaped annotation profile — 57-59% at L0-L1, trough 45.4% at L8, recovery 55-56% at L10-L11, second decline 47-55% at L15-L17. The author reads this as four functional zones: Molecular (L0-4) / Abstract (L5-9) / Re-specialization (L10-12) / Prediction-focused (L15-17).*

### Causal patching (Geneformer L11)

![Co-activation modules across layers and Table 5 causal patching summary](/assets/images/paper/sparse-ae-scgpt-geneformer/page_010.png)
*Figure 5 + Table 5: Co-activation modules collapse near L0 to integrative machinery by L11; single-feature causal patching at L11 yields a mean specificity ratio of 8.98, median 2.36x, with 60% of features above 2x.*

| Metric | Value |
|---|---|
| Mean specificity ratio | **8.98** |
| Median specificity ratio | **2.36** |
| Features with > 2x specificity | 30/50 (60%) |
| Features with > 10x specificity | 6/50 (12%) |
| Mean target logit delta | -0.116 |
| Mean other logit delta | -0.005 (23x smaller) |
| Top feature (F2035, neg. reg. of cell differentiation) | **114.5x** |

scGPT L7 causal patching yields **median 0.98x, 0/50 features > 2x** — but the author flags the uniform 1.0 expression proxy as the likely confound, and the cross-layer PMI graph still shows scGPT features are interconnected.

### Perturbation response mapping — the negative result

![Perturbation detection (92%) vs regulatory specificity (6.2%) — features are causally necessary but not specifically regulatory](/assets/images/paper/sparse-ae-scgpt-geneformer/page_011.png)
*Figure 6 + Table 6: SAE features are causally necessary — ablating one feature disrupts associated targets at 2.36-114.5x specificity — but **only 3 of 48 TRRUST TFs (6.2%) produce target-specific responding features** under CRISPRi perturbation. 92% of targets produce *some* feature change; almost none of it is regulatorily specific.*

| Metric | Value |
|---|---|
| Perturbation targets with >= 1 SAE feature change | **92/100 (92%)** |
| TRRUST TFs with target-specific responding features | **3/48 (6.2%)** |
| Mean responding features / target | 2.54 |
| Mean target-specific features / target | 0.03 |

### Cross-layer information highways

![Adjacent-layer persistence (2-3%), L0 feature decay, and cross-layer information highways at 97-99.8% PMI > 3](/assets/images/paper/sparse-ae-scgpt-geneformer/page_012.png)
*Figure 7: 97-99.8% of features at each Geneformer layer have at least one target-layer partner at PMI > 3 (max-PMI mean 6.61-6.79; individual edges exceed PMI 10, e.g. L11->L17 protein-modification -> angiogenesis-regulation at 10.62). scGPT shows a progressive bottleneck: 95.7% (L0->L4) -> 78.6% (L4->L8) -> 62.9% (L8->L11).*

### Multi-tissue control: model is the bottleneck

![Multi-tissue SAE yields modest 6.2% -> 10.4% improvement with disjoint gain/loss sets at L11](/assets/images/paper/sparse-ae-scgpt-geneformer/page_014.png)
*Figure 9: Retraining Geneformer SAEs on a balanced K562 + Tabula Sapiens pool raises L11 TF specificity from 6.2% to 10.4% — but 5 TFs gain specificity (ATF5, BRCA1, GATA1, RBMX, NFRKB) while 3 lose it (MAX, PHB2, SRF) and 40 are unchanged. Disjoint sets, no L0 gain, second-decline at L17 — consistent with stochastic noise. TF-associated feature fraction actually drops from 64.5% to 60.5%.*

| SAE training data | Layer | TFs specific | Rate |
|---|---|---|---|
| K562-only | 11 | 3/48 | 6.2% |
| K562 + Tabula Sapiens | 0 | 0/48 | 0.0% |
| K562 + Tabula Sapiens | 5 | 4/48 | 8.3% |
| **K562 + Tabula Sapiens** | **11** | **5/48** | **10.4% (best)** |
| K562 + Tabula Sapiens | 17 | 1/48 | 2.1% |

### Geometry and cell-type tiling

UMAP of decoder vectors **collapses to a structureless point cloud** (mean pairwise cosine **0.0007**; within-module vs between-module Cohen's $d = 0.075$) — direct geometric evidence that 4,608 features pack into 1,152 dimensions by spreading near-uniformly on the hypersphere. **Cell-type tiling**: 2,028 / 2,048 scGPT L7 features (99.0%) enrich for at least one Tabula Sapiens cell type; Geneformer K562-trained SAEs generalize to Tabula Sapiens (immune -> T / B / macrophage features, kidney -> proximal tubule / podocyte, lung -> alveolar / endothelial).

## Limitations

**Author-acknowledged.** (1) TRRUST captures only a fraction of true TF->target relations. (2) Architecture fixed at k=32, 4x overcomplete — one point in SAE design space. (3) Causal patching is single-feature; combinatorial / multi-feature ablation untested. (4) Multi-tissue SAE used naive 50/50 K562 / Tabula Sapiens pooling. (5) scGPT causal patching used uniform 1.0 proxy expression. (6) Perturbation response mapping done only on Geneformer.

**Additional issues a careful reader will notice.**

- **No variance / confidence intervals** on the headline 6.2% / 10.4% / 2.36x numbers — single-run estimates. The 4.2 pp K562-vs-multi-tissue swing depends on 2 TFs (3 -> 5 of 48); a bootstrap CI on TF-level specificity is missing.
- **50 causal-patching features are hand-picked** as "richly annotated"; no matched negative-control distribution drawn from unannotated features. Selection bias on the 2.36x median is uncharacterized.
- **No external CRISPRi benchmark.** Norman, Adamson, or Replogle-essential would be a natural cross-validation of whether the 6.2% specificity ceiling generalizes beyond TRRUST's 48 TFs in the Replogle subset.
- **No baseline-model anchor.** A randomly initialized Geneformer or linear PCA null would have grounded "6.2% is low."
- **Module identities are author-named from top-gene gestalt**; no quantitative match to MSigDB hallmark sets or another external pathway clustering.
- **The 18 vs 12 layer comparison is normalized by relative depth**, but architecture-fair claims cannot be cleanly separated — next-token vs masked, rank vs continuous, depth, and width all vary simultaneously.
- **Single-author paper**, single companion citation doing the orthogonal attention analysis — independent replication of the regulatory-specificity ceiling on a third scFM (UCE, SCimilarity) would materially strengthen C8.

## Why It Matters for Medical AI

scFMs are being aggressively positioned for perturbation prediction, drug response, and disease subtyping — uses that all assume the model encodes regulatory mechanism rather than expression-pattern lookup. This paper, with the right interpretability tool and the right control, provides the **first systematic evidence that the regulatory assumption is wrong as currently trained**: massive superposition, organized biology, and information highways all exist, but the **6.2% -> 10.4% multi-tissue ceiling** — with disjoint gain / loss sets — pins the limitation on the **model**, not on the SAE. The natural follow-ups are perturbation-aware pre-training (does conditioning on CRISPRi response raise the ceiling, or only redistribute features?), SAE k-sweep / dictionary-size sweep (is 6.2% an SAE-architecture artifact?), and combinatorial feature ablation (do pairs / triples reveal regulatory programs single-feature patching misses?). Until these land, treat scFM-based perturbation predictions as **co-expression retrieval with a transformer prior**, not as causal regulatory inference.

## References

- Paper: [arXiv:2603.02952v1 — Sparse Autoencoders Reveal Organized Biological Knowledge but Minimal Regulatory Logic in Single-Cell Foundation Models](https://arxiv.org/abs/2603.02952) (March 2026, Ihor Kendiukhov, University of Tübingen)
- Code: [github.com/Biodyn-AI/bio-sae](https://github.com/Biodyn-AI/bio-sae)
- Interactive atlases: [Geneformer Feature Atlas](https://biodyn-ai.github.io/geneformer-atlas/) — 82,525 features, 18 layers; [scGPT Feature Atlas](https://biodyn-ai.github.io/scgpt-atlas/) — 24,527 features, 12 layers
- Companion attention-only study: Kendiukhov 2025 (cited as [3] in this paper)
- Related: Geneformer (Theodoris et al., *Nature*, 2023); scGPT (Cui et al., *Nat. Methods*, 2024); Replogle CRISPRi (Replogle et al., *Cell*, 2022); TRRUST v2 (Han et al., 2018); Tabula Sapiens (The Tabula Sapiens Consortium, *Science*, 2022); TopK SAEs (Gao et al., 2024); the Anthropic / DeepMind SAE-on-LLM line of work
