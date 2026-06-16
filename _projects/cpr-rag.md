---
title: "CPR-RAG: Clinical Prior-Regularized Retrieval for Anatomy-Aware 3D CT Report Generation"
authors: "Sungkyu Yang¹, Kang-Min Kim²·*, Mansu Kim¹·*"
affiliations: "¹ AI Graduate School, GIST &nbsp;&middot;&nbsp; ² Department of Software Convergence, Kyung Hee University"
venue: "ACL 2026"
date: 2026-04-15
order: 1
thumbnail: /assets/images/projects/cpr-rag/hero.png
hero_iframe: /assets/projects/cpr-rag/flow.html
hero_caption: "Interactive CPR-RAG model flow — three stages: anatomy-conditional representation via learnable organ queries → clinical-prior-regularized re-ranking using corpus co-occurrence statistics → normality description removal to maximize evidence density."
abstract: "Plug-and-play retrieval-augmented generation framework that integrates clinical priors and removes normal-finding boilerplate to improve organ-level grounding in 3D CT report generation."
links:
  - { label: "Paper (PDF)", url: "/assets/papers/cpr-rag.pdf" }
  - { label: "Code", url: "https://github.com/Gyuang/cpr-rag" }
  - { label: "BibTeX", url: "#bibtex" }
published: true
---

## Abstract

Generating radiology reports from 3D volumetric data remains challenging due to the difficulty of grounding fine-grained pathologies within high-dimensional scans. While retrieval-augmented generation (RAG) offers a potential solution, standard approaches struggle with **visual-semantic ambiguity** and often introduce irrelevant "normal" context that **dilutes pathological signals**. To address this, we introduce **CPR-RAG**, a model-agnostic RAG framework that enhances organ-level grounding by integrating clinical priors into the retrieval process.

Specifically, we propose a **clinical prior-regularized re-ranking module** that leverages corpus-derived co-occurrence statistics to align retrieved candidates with latent disease distributions, ensuring clinical consistency beyond mere visual similarity. We further employ **normality description removal** to selectively filter boilerplate normal descriptions, maximizing the information density of the evidence provided to the generator. Extensive experiments on **RadGenome-ChestCT** demonstrate that CPR-RAG significantly improves clinical efficacy across state-of-the-art radiology report generation models, and human evaluation confirms superior factual correctness, completeness, and utility.

## Motivation

3D CT report generation is a **section-structured, multi-organ reasoning task**. Conventional RAG over visually similar cases suffers from two failure modes:

1. **Visual-semantic ambiguity** — pneumonia and atelectasis can both present as pulmonary opacities yet require different clinical interpretations. Visually nearest neighbors are not pathologically nearest.
2. **Multi-organ entanglement** — retrieving a full report carries irrelevant descriptions of *other* organs (often templated normals), inflating the prompt with low-information content and triggering the "lost-in-the-middle" effect.

CPR-RAG addresses both by **operating at the organ level** and by **regularizing similarity with clinical co-occurrence priors**.

## Method

The framework has three stages:

**(1) Anatomy-conditional representation.** A learnable query matrix $Q = [q_1, \dots, q_{\lvert\Omega\rvert}]^\top$ with one query per anatomical region $\Omega = \{\text{Heart, Lung, Mediastinum, Pleura, Trachea/Bronchi}\}$ attends over the frozen 3D visual tokens $X$:

$$E = \mathrm{CrossAttn}(Q, X, X)$$

producing organ-specific embeddings $e_i$. We train these with an auxiliary multi-label classifier $C_i(\cdot)$ per organ:

$$\mathcal{L}_{\mathrm{aux}} = \sum_{i \in \Omega} \mathrm{BCE}(C_i(e_i),\, y_i)$$

![t-SNE of learned organ-region embeddings](/assets/images/projects/cpr-rag/organ-embedding-tsne.png)
*t-SNE of the learned organ embedding space $\mathcal{Z}$. The anatomy-conditional queries separate the five regions $\Omega$ into well-defined clusters, confirming that each query captures region-specific visual semantics.*

**(2) Clinical prior-regularized retrieval.** For each organ $i$, we combine visual affinity with a co-occurrence-derived clinical prior $\pi_i$ computed from the training corpus, re-ranking the top-$k$ visually retrieved candidates to enforce clinical consistency.

**(3) Normality description removal.** Boilerplate "no acute abnormality" segments are filtered out of the retrieved context, maximizing pathological signal density in the prompt.

We keep both the **visual encoder and the LLM frozen** — only lightweight projection modules and LoRA adapters are trained. This isolates gains to *how* retrieved context is used, not memorization.

## Results

![Performance comparison on RadGenome-ChestCT](/assets/images/projects/cpr-rag/results.png)
*Clinical efficacy and text-quality on RadGenome-ChestCT. CPR-RAG yields substantial recall improvements across all three backbones (M3D, RadFM, CT2Rep).*

Headline numbers on **RadGenome-ChestCT** (clinical-efficacy F1, micro / macro):

| Backbone | F1 (macro / micro) | Recall (macro / micro) |
|---|---|---|
| CT2Rep | 11.93 / 20.16 | 10.00 / 15.53 |
| CT2Rep **+ CPR** | **20.26 / 26.04** | **22.70 / 27.46** |
| M3D | 13.74 / 22.37 | 12.09 / 18.65 |
| M3D **+ CPR** | **29.06 / 32.73** | **42.93 / 46.49** |
| RadFM | 9.54 / 15.74 | 7.20 / 11.14 |
| RadFM **+ CPR** | **24.96 / 32.10** | **36.57 / 45.80** |
| RadFM + CPR (Oracle) | 37.72 / 45.15 | 37.73 / 45.39 |

Gains are largely **recall-driven** — CPR-RAG surfaces pathological findings that the base generator would otherwise default to "no acute abnormality" templates. The Oracle row (evidence selected by ground-truth labels) provides an empirical upper bound and is closely approached by RadFM+CPR.

![Qualitative examples](/assets/images/projects/cpr-rag/qualitative.png)
*Qualitative comparison — CPR-RAG retrieves clinically consistent cases and surfaces findings that base RAG misses.*

## Human Evaluation

A board-certified radiologist rated 100 randomly sampled cases on a **1–5 Likert scale**; statistical significance is assessed with the Wilcoxon signed-rank test.

| Model | Completeness | Correctness | Utility |
|---|---|---|---|
| RadFM | 2.48 | 2.30 | 2.76 |
| RadFM **+ CPR** | **2.93** | **2.45** | **3.09** |
| *p*-value | < 0.01 | 0.12 | < 0.01 |

CPR-RAG improves all three axes, with **Completeness** and **Utility** gains statistically significant (*p* < 0.01); **Correctness** shows a positive but non-significant trend (*p* = 0.12).

## BibTeX

```bibtex
@article{yang2026cprrag,
  title  = {CPR-RAG: Clinical Prior-Regularized Retrieval for Anatomy-Aware 3D CT Report Generation},
  author = {Yang, Sungkyu and Kim, Kang-Min and Kim, Mansu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year   = {2026}
}
```
