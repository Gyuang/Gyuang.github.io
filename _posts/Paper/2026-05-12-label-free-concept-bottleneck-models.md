---
title: "Label-Free Concept Bottleneck Models"
excerpt: "LF-CBM is the first CBM scaled to ImageNet, reaching 71.95% top-1 with a sparse interpretable final layer versus 74.35% for a sparse-standard ResNet-50 and 76.13% for the dense baseline."
categories:
  - Paper
  - LLM
permalink: /paper/label-free-concept-bottleneck-models/
tags:
  - LF-CBM
  - Concept-Bottleneck-Model
  - Interpretability
  - CLIP
  - GPT-3
  - CLIP-Dissect
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- LF-CBM turns any backbone into a Concept Bottleneck Model **without any labeled concept data** by chaining GPT-3 (per-class concept proposal), five concept filters, a differentiable `cos_cubed` CLIP-Dissect projection, and a GLM-SAGA sparse final layer.
- It is the **first CBM scaled to ImageNet**: **71.95% +/- 0.05% top-1** with a sparse interpretable final layer — comparable to a *sparse* ResNet-50 baseline (74.35%) but still 4.2% below the *dense* ResNet-50 (76.13%).
- The strongest evidence is a **13,515-rating MTurk study** rating LF-CBM CBL neurons more interpretable than ResNet-50 penultimate neurons (3.91 vs 3.65 on a 5-pt scale, with 74.4% / 78.8% preference over dense / sparse baselines); the weakest is a **5-edit manual-editing anecdote** that nudged ImageNet val accuracy from 71.98% to 72.02%.

## Motivation

Classical Concept Bottleneck Models (Koh et al., 2020) make a model's decision interpretable as a linear combination over named concepts, but pay two prices: per-image concept annotations are expensive and expert-bound, and the bottleneck consistently bleeds accuracy versus the standard backbone. Post-hoc CBM (Yuksekgonul et al., 2022) loosens the training cost but still needs labeled CAV data or a CLIP backbone, and it underperforms standard nets. LF-CBM's pitch is to outsource both burdens to foundation models — GPT-3 proposes the concept set, CLIP aligns images to concepts — so that an arbitrary backbone can be retrofitted into a CBM with zero concept labels. The authors are explicit in Appendix A.2 that this trick works *where CLIP works*; medical and other small-domain settings are flagged as still better served by labeled-concept CBMs.

## Core Innovation

- **Foundation models as a free concept supervisor.** GPT-3 (text-davinci-002) is prompted with three templates per class for important features, co-occurring objects, and superclasses; CLIP supplies an image-text similarity matrix $P$ that replaces per-image concept annotations.
- **A fully differentiable CLIP-Dissect similarity, `cos_cubed`.** The element-wise cube concentrates similarity on highly activating inputs, letting the Concept Bottleneck Layer $W_c$ be learned with Adam against a closed-form similarity loss rather than the non-differentiable SoftWPMI.
- **A sparse-by-construction final layer.** Elastic-net + GLM-SAGA produces 25-35 nonzero weights per output class (0.7-15% density), making each per-class decision rule readable.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | LF-CBM creates interpretable CBMs **without any labeled concept data**. | Pipeline (Sec 3); CUB-200 ignores its 312 available concept annotations yet reaches 74.31%. | All 5 | ⭐⭐⭐ |
| C2 | First CBM scaled to ImageNet. | Table 1 comparison; ImageNet row 71.95%. | ImageNet | ⭐⭐⭐ |
| C3 | LF-CBM retains accuracy "comparable to the original neural network". | Table 2: 71.95% vs sparse-standard 74.35% vs dense-standard 76.13% on ImageNet. | All 5 | ⭐⭐ — true only against the *sparse* baseline; vs the dense ResNet-50 the gap is real. |
| C4 | LF-CBM outperforms Post-hoc CBM. | Table 2: CIFAR-10 86.40% vs P-CBM-CLIP 84.50%; CIFAR-100 65.13% vs 56.00%; CUB 74.31% vs 59.60%. | CIFAR-10/100, CUB | ⭐⭐⭐ on CIFAR/CUB; P-CBM not run on Places/ImageNet, so the most interesting comparison is absent. |
| C5 | GPT-3 yields better concept sets than ConceptNet. | Appendix A.6, Table 5. | All 5 | ⭐⭐⭐ on CUB (74.31% vs 2.19% collapse); ⭐ elsewhere (0.1-1.5% gaps). |
| C6 | Concept filters improve interpretability. | Figure 7 qualitative comparison; Appendix A.4 shows no accuracy hit. | CIFAR-10, ImageNet | ⭐⭐ — interpretability gain argued qualitatively, no per-filter quantitative metric. |
| C7 | Decisions are explainable as linear concept combinations. | Figures 3, 4, Sankey + bar plots. | ImageNet, Places, CUB | ⭐⭐⭐ — entailed by architecture. |
| C8 | Manual editing of LF-CBM weights improves accuracy. | Section 5: 71.98% → 72.02% on ImageNet val over 5 edits. | ImageNet | ⭐ — net +0.04% over 1000 classes with cherry-picked edits, no statistical test. |
| C9 | `cos_cubed` matches non-differentiable SoftWPMI. | Appendix A.3, Table 3. | ImageNet final-layer probe | ⭐⭐ — mpnet-similarity 0.4523 vs 0.4525 ties, but top-1 neuron-naming accuracy is 63.33% vs 76.78%. |
| C10 | LF-CBM is more interpretable than the standard backbone (human-rated). | Appendix B; **13,515 MTurk ratings** on ImageNet; 3.91 vs 3.65 (5-pt); 74.4% / 78.8% preference after filtering. | ImageNet | ⭐⭐⭐ — largest, cleanest piece of evidence in the paper. |

**Honest take.** The "preserves accuracy" framing is anchored against a *sparse* standard baseline (which itself loses 2-10% versus dense). Genuinely strong claims are C1 (label-free works), C2 (first ImageNet CBM), C5 in the CUB special case, and C10 (the 13,515-rating MTurk study). The weakest are C8 (5-edit anecdote) and C3 read maximally. A subtle methodological issue: concept faithfulness is measured by CLIP-Dissect, which uses CLIP both as judge and as the feature provider that produced $P$. A neuron that learned a CLIP-shaped concept rather than a human-shaped concept can still pass Filter 5 — the faithfulness filter is partially circular, a point the paper does not address. Statistical reporting is also thin: only 3 training runs per dataset with std-dev, no significance tests against the sparse baseline (which on CIFAR-100 is 58.34% vs LF-CBM 65.13% — large delta but never formally tested).

## Method & Architecture

![LF-CBM four-step pipeline](/assets/images/paper/label-free-cbm/page_003.png)
*Figure 1: LF-CBM's four-step pipeline — (1) GPT-3 proposes a concept set per class, (2) CLIP builds the image-concept matrix $P$ and the backbone caches features, (3) the Concept Bottleneck Layer $W_c$ is trained by maximizing the `cos_cubed` similarity between neuron activations and concept columns of $P$, (4) a sparse final layer $W_F$ is learned with elastic-net GLM-SAGA.*

### Step 1 — Concept-set creation and filtering

For each class, GPT-3 (text-davinci-002) is prompted with three templates:

- "List the most important features for recognizing something as a {class}"
- "List the things most commonly seen around a {class}"
- "Give superclasses for the word {class}"

Two few-shot exemplars are shared across all datasets; each prompt is run twice and merged. Five filters then trim noise:

1. Concepts longer than 30 characters dropped.
2. Concepts with cosine similarity > 0.85 to any class name (CLIP ViT-B/16 + `all-mpnet-base-v2` ensemble) dropped.
3. Concepts with similarity > 0.9 to any other concept dropped.
4. Concepts whose top-5 CLIP activation falls below a dataset-specific cutoff (0.25-0.28) dropped.
5. After Step 3, concepts whose CBL neuron has CLIP-Dissect similarity < 0.45 dropped.

CIFAR-10 walk-through (Appendix A.5): 177 → 174 → 164 → 154 → 147 → 142 final concepts.

![GPT-3 concept proposal template](/assets/images/paper/label-free-cbm/page_020.png)
*Figure 2: One of the three GPT-3 prompt templates with example completions. Concepts generated by GPT-3 are highlighted in green.*

### Step 2 — Cache embeddings

For training set $\mathcal{D} = \{x_1, \ldots, x_N\}$ and concept set $\mathcal{C} = \{t_1, \ldots, t_M\}$, the CLIP image-text concept matrix $P \in \mathbb{R}^{N \times M}$ is computed once with $P_{i,j} = E_I(x_i) \cdot E_T(t_j)$, alongside backbone features $f(x_i) \in \mathbb{R}^{d_0}$. The cache is the bulk of pipeline wall time.

### Step 3 — Learn the Concept Bottleneck Layer $W_c$

Define $f_c(x) = W_c f(x)$ with $W_c \in \mathbb{R}^{M \times d_0}$. For neuron $k$ with activation pattern $q_k \in \mathbb{R}^N$ across the dataset, the differentiable `cos_cubed` similarity is:

$$
\operatorname{sim}(t_i, q_i) = \frac{\bar{q}_i^{\,3} \cdot \bar{P}_{:,i}^{\,3}}{\lVert \bar{q}_i^{\,3} \rVert_2 \cdot \lVert \bar{P}_{:,i}^{\,3} \rVert_2}
$$

where $\bar{q}$ is $q$ after mean-0 std-1 normalization, and the element-wise cube concentrates the similarity on highly activating inputs. The training objective is $\mathcal{L}(W_c) = -\sum_i \operatorname{sim}(t_i, q_i)$, optimized with Adam, early-stopped on validation similarity. Concepts whose final validation similarity falls below 0.45 are pruned (Filter 5).

### Step 4 — Learn the sparse final layer $W_F$

With $f$ and $W_c$ frozen, $W_F \in \mathbb{R}^{d_z \times M}$ is fit by elastic net via GLM-SAGA:

$$
\min_{W_F, b_F} \sum_i \mathcal{L}_{ce}\big(W_F f_c(x_i) + b_F, y_i\big) + \lambda R_\alpha(W_F), \qquad R_\alpha(W_F) = (1-\alpha) \tfrac{1}{2} \lVert W_F \rVert_F + \alpha \lVert W_F \rVert_{1,1}
$$

The authors use $\alpha = 0.99$ and tune $\lambda$ to leave **25-35 nonzero weights per output class** (0.7%-15% density depending on $M$).

### Backbones and scale

Backbones span CLIP-RN50 (CIFAR-10/100), ResNet-18 from imgclsmob (CUB), and ResNet-50 (ImageNet, Places365). Post-filter concept-set sizes: CIFAR-10 = 128, CIFAR-100 = 824, CUB-200 = 211, Places-365 = 2202, ImageNet = 4505. Training fits on a single Nvidia Tesla P100; post-cache training takes under 4 hours on every dataset.

## Experimental Results

### Main accuracy table

| Model | Sparse FC | CIFAR-10 | CIFAR-100 | CUB-200 | Places365 | ImageNet |
|---|---|---|---|---|---|---|
| Standard (dense) | No | 88.80% | 70.10% | 76.70% | 48.56% | 76.13% |
| Standard (sparse) | Yes | 82.96% | 58.34% | 75.96% | 38.46% | 74.35% |
| P-CBM | Yes | 70.50% | 43.20% | 59.60% | N/A | N/A |
| P-CBM (CLIP) | Yes | 84.50% | 56.00% | N/A | N/A | N/A |
| **LF-CBM (ours)** | Yes | **86.40% +/- 0.06%** | **65.13% +/- 0.12%** | **74.31% +/- 0.29%** | **43.68% +/- 0.10%** | **71.95% +/- 0.05%** |

The 71.95% ImageNet number is comparable to the *sparse* standard baseline (74.35%) but trails the *dense* ResNet-50 (76.13%) by 4.2%. Appendix A.7 shows that removing the elastic-net sparsity ("LF-CBM dense") recovers most of that gap (ImageNet 74.09%, Places365 48.25%, CIFAR-10 87.50%) — so the dense-baseline gap is largely the price of the sparse-interpretable bottleneck, not of the label-free pipeline.

### Global concept explanations

![Sankey diagrams of LF-CBM final-layer weights](/assets/images/paper/label-free-cbm/page_007.png)
*Figure 3: Final-layer concept weights as Sankey diagrams. LF-CBM separates Orange vs Lemon on ImageNet and Mountain vs Mountain-Snowy on Places365 via human-readable concept contributions.*

### Per-instance explanations

![Per-instance concept contributions](/assets/images/paper/label-free-cbm/page_008.png)
*Figure 4: Per-instance decision explanations for a CUB Red-headed Woodpecker and a Places365 junkyard. Each bar is a single concept's contribution to the prediction logit.*

### Concept-filter ablation

![CIFAR-10 automobile concept weights, with and without filters](/assets/images/paper/label-free-cbm/page_015.png)
*Figure 5: CIFAR-10 "automobile" final-layer weights with and without concept filters. Filters barely move accuracy (Appendix A.4) but visibly clean up the concept list — interpretability is the goal here, not accuracy.*

### Key ablations

- **Concept source (Appendix A.6).** GPT-3 vs ConceptNet is +0.1-1.5% on most datasets but **74.31% vs 2.19% on CUB-200** — ConceptNet cannot resolve fine-grained class names like "Groove billed Ani". This is the cleanest case for an LLM proposer.
- **Sparsity cost (Appendix A.7).** Removing elastic-net sparsity recovers ImageNet 74.09% / Places365 48.25%, but the top-10 concepts then explain only a tiny fraction of the logit (Figure 8) — sparsity is what makes the model human-readable.
- **Similarity function (Appendix A.3).** `cos_cubed` ties SoftWPMI on mpnet-similarity (0.4523 vs 0.4525) but loses 13% on top-1 neuron-naming accuracy (63.33% vs 76.78%). The paper's "performs as well" framing understates this.
- **Concept-set stochasticity (Appendix A.8).** Re-running GPT-3 yields 4380 vs 4523 concepts and 71.89% vs 71.95% accuracy. Aggregate is robust; per-image explanations can shift substantially (Figure 9).
- **Manual editing (Section 5).** Editing 10 final-layer weights moved ImageNet val from 71.98% to 72.02% (+38 correct, -17 incorrect across 5 edits over 10 affected classes — roughly +4.2% on the 500 affected validation examples but only +0.04% globally). The cute proof-of-concept frames a 5-edit anecdote as a contribution.

### Human evaluation

![MTurk human evaluation of LF-CBM neuron interpretability](/assets/images/paper/label-free-cbm/page_032.png)
*Figure 6: After filtering inconsistent raters, **74.4%** of MTurk evaluators preferred LF-CBM concept explanations over dense ResNet-50 explanations, and **78.8%** preferred LF-CBM over sparse ResNet-50. Notably, sparsity by itself did not improve perceived single-decision reasonableness — only LF-CBM-vs-dense, not sparse-vs-dense, moved the needle. With **13,515 ratings** across N=3 raters x 4,505 LF-CBM neurons + 3 x 2,048 ResNet-50 neurons, this is the strongest piece of evidence in the paper.*

## Limitations

**Authors acknowledge (Appendix A.2).**

- GPT-3 is stochastic and may miss important concepts.
- Automatic concept generation lacks domain expertise; pairing with a human expert is recommended.
- LF-CBM works "best where CLIP works well" — explicitly *not* a fit for medical or specialized small-data domains where labeled-concept CBMs remain preferable.

**Visible from the evidence but not addressed.**

- **Circularity in the faithfulness filter.** CLIP scores both define the image-concept alignment used to train $W_c$ (via $P$) and judge faithfulness in Filter 5 (CLIP-Dissect similarity >= 0.45). A neuron that learned a CLIP-shaped concept rather than a human-shaped concept can still pass — the judge is the projection.
- **Hyperparameter fragility undocumented.** Filter cutoffs (0.85 / 0.9 / 0.25-0.28 / 0.45) and the 25-35 nonzero weights-per-class target are tuned by trial and error; no sensitivity sweep.
- **Statistical reporting is thin.** Only 3 training runs per dataset with std-dev, no significance test against the sparse standard baseline.
- **Faithfulness gap.** Even after Filter 5, a CBL neuron's activation pattern is only "0.45 or higher" similar to its named concept — neuron-correlated-with-concept is not neuron-is-concept.
- **Concept-set stochasticity downstream.** Re-running GPT-3 changes both the concept set and per-image explanations; for a deployed interpretability system this is a real reproducibility concern that gets one paragraph.
- **No bias audit.** GPT-3 and CLIP both carry Western, English-centric, web-photo biases that can propagate into $W_F$; not audited.
- **Domain transfer untested.** The authors flag medical / specialized domains as out of scope but never run a single negative experiment to substantiate the caveat.

## Why It Matters for Medical AI

For medical-AI readers the takeaway is calibrated rather than enthusiastic. LF-CBM demonstrates that an LLM concept proposer plus a CLIP-based label-free projection can scale concept bottlenecks to ImageNet, and the 13,515-rating MTurk study is real evidence that the resulting explanations are more reasonable to humans than penultimate-layer neurons. But the authors explicitly tell you in Appendix A.2 not to apply this recipe directly to medical imaging — CLIP and GPT-3 do not cover specialized clinical vocabularies or modality-specific visual features, and the faithfulness filter that LF-CBM relies on is itself a CLIP-derived score. The practical guidance is: if you want concept-bottleneck interpretability in a medical setting, swap GPT-3 for a clinician-curated or ontology-derived concept set (RadLex, SNOMED CT subsets) and replace CLIP with a domain VLM (BiomedCLIP, CONCH, PathCLIP) before reusing the cos_cubed + GLM-SAGA backend. The architecture transfers; the foundation-model supervisor does not.

## References

- Paper: Oikarinen, Das, Nguyen, Weng. *Label-Free Concept Bottleneck Models.* ICLR 2023. [arXiv:2304.06129](https://arxiv.org/abs/2304.06129)
- Code: [https://github.com/Trustworthy-ML-Lab/Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM)
- Related: Koh et al., *Concept Bottleneck Models*, ICML 2020.
- Related: Yuksekgonul et al., *Post-hoc Concept Bottleneck Models*, ICLR 2023.
- Related: Oikarinen and Weng, *CLIP-Dissect*, ICLR 2023.
- Related: Wong et al., *Leveraging Sparse Linear Layers for Debuggable Deep Networks*, ICML 2021 (GLM-SAGA).
