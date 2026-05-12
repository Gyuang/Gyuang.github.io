---
title: "Improving Concept Alignment in Vision-Language Concept Bottleneck Models"
excerpt: "CLIP-based VL-CBMs hit 75.84/95.63/90.14% class accuracy on CUB/RIVAL/AwA2 with only 24.43/58.85/49.02% concept accuracy; a learnable concept-projection layer plus a Contrastive Semi-Supervised loss lifts concept accuracy by +39.10/+18.63/+32.11 pp using just 9/8/10 labels per class."
categories:
  - Paper
tags:
  - VL-CBM
  - CSS
  - Concept-Bottleneck-Model
  - CLIP
  - Interpretability
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- VL-CBMs that treat CLIP image-text similarity as a "concept score" look interpretable but are not — on CUB / RIVAL / AwA2 they hit **75.84 / 95.63 / 90.14% class accuracy** while only reaching **24.43 / 58.85 / 49.02% concept accuracy** against expert-defined concept ground truth. CLIP routes the right *label*, not the right *evidence*.
- The fix is structurally small: add a learnable linear **concept-projection layer** on top of CLIP's average-pooled patch tokens, then train it with a **Contrastive Semi-Supervised (CSS)** loss that uses sparse concept labels (9 / 8 / 10 per class) and propagates supervision through intra-class positive pairs and inter-class negatives.
- Headline gains are large: concept accuracy **+39.10 / +18.63 / +32.11 pp** on CUB / RIVAL / AwA2, plus **PLIP 42.56 → 64.19** attribute accuracy on the medical WBCAtt dataset. The faithfulness diagnosis is the strongest contribution; the medical and SOTA claims have real caveats — see the audit.

## Motivation

LLM-prompted concept generators (LaBo, Label-free CBM, CDL) bypass per-image concept labels but produce huge, noisy, weakly grounded concept sets. For trustworthy applications — especially **medical** ones where pathologist-defined morphological attributes already exist (e.g. WBCAtt for white blood cell morphology) — what you actually want is a CBM built on a *small, expert-defined* concept vocabulary. The authors do this and discover an awkward fact: when you grade a CLIP VL-CBM against expert concept ground truth, classification accuracy stays high but concept accuracy collapses below 25% on fine-grained CUB. The model is shortcutting the bottleneck. This paper closes that gap with a handful of labels per class — exactly the regime clinical annotation can afford.

## Core Innovation

- **Concept projection on average-pooled patch tokens.** CLIP's image-text cosine score is augmented by a learned linear map from the layer-normalized, average-pooled visual patch tokens: $C_i = E_I(x_i) \cdot E_T(T)^\top + \mathrm{LN}(E^*_I(x_i)) \cdot W_\mathrm{CP}$. Only $W_\mathrm{CP} \in \mathbb{R}^{768 \times c}$ and the linear class head $W_K$ are trained; CLIP itself stays frozen.
- **Contrastive Semi-Supervised (CSS) loss.** A pair-sampled mini-batch contributes two same-class images per anchor; the contrastive term pushes their concept vectors together and away from other classes, the cross-entropy term keeps classification competitive, and an L1 concept-anchor term is applied only on the few labeled images. The contrastive signal *propagates* the sparse concept anchor to unlabeled intra-class samples — no KNN pseudo-labels needed (the paper contrasts this with Hu et al. 2024, though without a head-to-head).
- **Class-level intervention via confounding-pair concept injection.** After CSS, the largest off-diagonal entries of the class-confusion matrix flag confounding pairs (California Gull ↔ Western Gull, Common Tern ↔ Arctic Tern). For those 4 classes the authors import the top-32 LaBo concepts per class (128 new concepts, scored by frozen CLIP), and train a small auxiliary classifier $W'_K$ that fires only on the confounding subset.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CLIP VL-CBMs have a large faithfulness gap: high class accuracy, low concept accuracy. | Table 1 — 24.43 / 58.85 / 49.02 concept vs. 75.84 / 95.63 / 90.14 class; Figure 2 qualitatives. | CUB, RIVAL, AwA2 | ⭐⭐⭐ |
| C2 | CSS substantially raises concept accuracy with very few labels. | Table 2 deltas +39.10 / +18.63 / +32.11 pp; Figure 3 sweep over {9,15,24,30} labels on CUB and {8,40,80,800} on RIVAL — CSS beats LaBo at every budget. | CUB, RIVAL, AwA2 | ⭐⭐⭐ |
| C3 | CSS also improves classification accuracy. | Table 2: +5.61 / +2.84 / +3.06 pp over CLIP VL-CBM; on CUB it beats the black-box ViT-B/16 (79.21 → 81.45). | 3 datasets | ⭐⭐ — single-seed, no variance reported. |
| C4 | CSS transfers to the medical domain. | WBCAtt: PLIP 42.56 → CSS 64.19 attribute accuracy at 50% labels. | WBCAtt only | ⭐⭐ — one non-diagnostic morphology dataset; no radiology / dermatology / WSI evaluation. |
| C5 | CSS is competitive with SOTA VL-CBMs using a much smaller expert concept set. | Table 3: 321 expert concepts, 83.89% CUB vs. LaBo (10k, 81.9), CDL (400, 83.4), LM4CV (400, 81.4). | CUB | ⭐⭐ — uses CLIP ViT-L/14 for this table; encoder change confounds the comparison. |
| C6 | CSS concept space is more truthful, sparser intra-class, more discriminative inter-class. | Table 5: truthfulness 3.22 vs. 14.04; intra-class σ 0.08 vs. 0.19; inter-class L2 7.04 vs. 3.28; Figure 4 t-SNE. | CUB only | ⭐⭐ |
| C7 | Class-level intervention reduces fine-grained errors. | Table 6: California Gull 27→14, Common Tern 21→11; overall 81.45 → 82.57. | CUB only | ⭐⭐ — +1.12 pp on 4 hand-picked classes. |
| C8 | Contrastive bootstrap beats KNN-pseudo-label approaches (Hu et al. 2024). | Argued in text. | — | ⭐ — no direct experiment. |
| C9 | CSS sets SOTA concept accuracy on AwA2. | **Refuted by Table 4**: ECBM 85.84 > CSS 81.3 on AwA2 concept accuracy. | AwA2 | ⭐ — over-claim contradicted by the authors' own table. |

**Honest take.** The faithfulness diagnosis (C1) is the load-bearing contribution and the numerical evidence is convincing across three datasets — a >50-point class-vs-concept gap on CUB is hard to explain away. The CSS gains (C2) are large enough that they are unlikely to be noise, and Figure 3 shows they hold across four label budgets. But the paper reports no random seeds, no standard deviations, and no significance tests, so every classification delta is a point estimate. The medical claim (C4) leans on a single morphology-annotation dataset (not a diagnostic outcome benchmark) and should not be read as evidence that CSS is ready for clinical CBM deployment. The CUB SOTA table (C5) switches to a stronger ViT-L/14 backbone, so the "small concept set wins" headline is partially a backbone story the paper acknowledges. And on AwA2 concept accuracy specifically, CSS (81.3) actually *loses* to ECBM (85.84) — a fact buried in Table 4 that the abstract narrative glosses over.

## Method & Architecture

![Figure 1: concept projection layer and Contrastive Semi-Supervised training](/assets/images/paper/vl-cbm-alignment/page_004.png)
*Figure 1: (a) The concept-projection layer adds a learned linear map from CLIP's average-pooled patch tokens (after LayerNorm) to the standard image-text similarity score. (b) The Contrastive Semi-Supervised scheme samples paired same-class images per anchor; intra-class concept vectors are pulled together while inter-class ones are pushed apart, with sparse ground-truth concept labels acting as anchors via an L1 term.*

### VL-CBM baseline

Given $k$ classes and a fixed set of $c$ expert text concepts $T = \{t_1, \dots, t_c\}$, the baseline VL-CBM scores each image as $C_i = E_I(x_i) \cdot E_T(T)^\top \in \mathbb{R}^{1 \times c}$ and predicts the class via a linear head $h \in \mathbb{R}^{c \times k}$. CLIP is frozen.

### Concept projection layer

The authors keep this and *add* a learnable projection from the average-pooled visual patch tokens $E^*_I(x_i) \in \mathbb{R}^{1 \times 768}$:

$$
C_i = E_I(x_i) \cdot E_T(T)^\top + \mathrm{LN}(E^*_I(x_i)) \cdot W_\mathrm{CP}, \qquad W_\mathrm{CP} \in \mathbb{R}^{768 \times c}.
$$

Class logits are $K_i = C_i \cdot W_K$. Only $W_\mathrm{CP}$ and $W_K$ are trained.

### CSS loss

Mini-batches are built by **pairwise sampling**: each of $n$ anchors contributes two same-class images $(x_i, x_j)$, with the other $2(n-1)$ samples from different classes. A sparse subset has ground-truth concept labels $G$. The total loss is

$$
\mathcal{L} = \mathcal{L}_\mathrm{contrastive} + \mathcal{L}_\mathrm{CE} + \mathcal{L}_\mathrm{concept}.
$$

The contrastive term over the $n$ positive pairs is

$$
\mathcal{L}_\mathrm{contrastive} = \frac{1}{n} \sum_{i,j} -\log \frac{\exp(\mathrm{sim}(C_i, C_j) / \tau)}{\sum_{m \neq i} \exp(\mathrm{sim}(C_i, C_m) / \tau)},
$$

$\mathcal{L}_\mathrm{CE}$ is standard cross-entropy on $K_l$, and $\mathcal{L}_\mathrm{concept} = \frac{1}{2n} \sum_l L_1(\gamma \cdot (s(C_l) - s(G_l)))$ applies only on concept-labeled samples, with $s$ a softmax normalization and $\gamma$ a scaling constant.

### Training details

- Backbone: OpenCLIP ViT-B/16 for CUB / RIVAL / AwA2; PLIP ViT-B/16 for medical WBCAtt; CLIP ViT-L/14 *only* for the CUB SOTA comparison in Table 3.
- Frozen CLIP + Adam optimizer; only $W_\mathrm{CP}$ and $W_K$ are trained.
- Concept-label budget: **CUB 9/class (30%), RIVAL 8/class (0.1%), AwA2 10/class (1.4%), WBCAtt 50%**.

### Class-level intervention (CUB)

Post-CSS, the test confusion matrix surfaces confounding class pairs (e.g. California Gull ↔ Western Gull). The authors import the top-32 LaBo concepts per confounding class (4 × 32 = 128 new concepts), score them with frozen CLIP, concatenate to form $C' \in \mathbb{R}^{440}$, and train a small auxiliary classifier $W'_K \in \mathbb{R}^{128 \times 4}$ whose zero-padded logits are summed onto $K_i$. $W_K$ and $W'_K$ are then jointly fine-tuned.

## Experimental Results

### Main faithfulness table (Table 2; class / concept accuracy, %)

| Model | CUB Class | CUB Concept | RIVAL Class | RIVAL Concept | AwA2 Class | AwA2 Concept | WBCAtt Attribute |
|---|---|---|---|---|---|---|---|
| ViT-B/16 (Black Box) | 79.21 | NA | 99.47 | NA | 94 | NA | 66.11 |
| CLIP VL-CBM | 75.84 | 24.43 | 95.63 | 58.85 | 90.14 | 49.02 | 42.56 |
| **CSS VL-CBM (Ours)** | **81.45** | **63.53** | 98.47 | **77.48** | 93.2 | **81.13** | **64.19** |

Concept-side deltas vs. CLIP VL-CBM are **+39.10 / +18.63 / +32.11 pp** on CUB / RIVAL / AwA2, plus **+21.63 pp** on WBCAtt.

### CUB SOTA comparison (Table 3, CLIP ViT-L/14)

| Method | # Concepts | Source | Class Acc |
|---|---|---|---|
| LaBo | 10,000 | LLM | 81.9 |
| SparseCBM | 926 | LLM | 80.02 |
| CDL | 400 | LLM | 83.4 |
| LM4CV | 400 | LLM | 81.4 |
| **CSS VL-CBM** | **321** | **Expert** | **83.89** |

CSS uses the smallest concept set on this table — but it also switches encoder from ViT-B/16 (used everywhere else) to ViT-L/14, so part of the 2-point lead over CDL is encoder, not method.

### RIVAL / AwA2 comparison (Table 4)

- RIVAL: CSS **98.86 class / 72.84 concept** vs. Text2Concept 95.38 (no concept reported).
- AwA2: CSS **93.2 / 81.3** vs. PCBM 88 / 71.9 and **ECBM 91.2 / 85.84**. CSS wins on classification but **loses concept accuracy to ECBM by 4.54 pp** — a caveat to the "improves concept alignment" framing.

### Distributional analysis (Table 5, CUB)

| Metric | CLIP | **CSS** |
|---|---|---|
| Truthfulness (L2 to GT, ↓) | 14.04 | **3.22** |
| Sparseness (intra-class σ, ↓) | 0.19 | **0.08** |
| Discriminability (inter-class L2, ↑) | 3.28 | **7.04** |

Figure 4's t-SNE corroborates: CSS concept vectors form tight, well-separated clusters near each class's GT mean, where CLIP's clusters overlap.

### Label-efficiency ablation (Figure 3)

![Figure 3: concept accuracy vs. labels-per-class on CUB and RIVAL](/assets/images/paper/vl-cbm-alignment/fig_p006_01.png)
*Figure 3: Concept accuracy as a function of concept-label budget. On CUB, CSS rises from 63.53% at 9 labels/class to 70.64% at 30; LaBo trails by 15-20 pp across the entire range. On RIVAL, CSS is near-saturated already at 8 labels/class.*

### Class-level intervention (Table 6, CUB)

For the four confounding-pair classes, total errors drop sharply (California Gull 27 → 14, Common Tern 21 → 11; the CG-misclassified-as-WG count alone goes 13 → 4). Overall CUB classification edges from **81.45 → 82.57** (+1.12 pp).

### Qualitative concepts (Figure 2)

![Figure 2: top-8 concepts per image, CLIP vs. CSS](/assets/images/paper/vl-cbm-alignment/page_005.png)
*Figure 2: Top-8 concept scores per image. CLIP collapses to the dominant color (every body part of a Baltimore Oriole becomes "orange"; a chimpanzee scores hooves/tusks/horns), whereas CSS recovers part-specific attributes (black-wing + black-nape + orange-breast for the oriole; bush/walks/bipedal for the chimpanzee).*

## Limitations

**Acknowledged by the authors.**
- *Ineffable concepts* — language cannot enumerate every visual subtlety (e.g. face recognition).
- *Unknown concepts* — VL-CBMs assume the salient concept set is known a priori; unrealistic in many domains.
- *Locality faithfulness* — concept scores may still rely on spurious spatial features (Raman et al. 2024); unexplored here.

**Not addressed but should be.**
- No random seeds, no standard deviations, no significance tests on *any* table. All deltas are single-point estimates.
- WBCAtt is the only medical experiment, and WBCAtt is a morphology-annotation benchmark — not a clinical / diagnostic outcome. No radiology, no dermatology, no WSI-scale histopathology, no external-site validation.
- No ablation isolating the contributions of (i) the projection layer, (ii) the contrastive loss, and (iii) the concept loss; we don't know which component is doing the work.
- The AwA2 concept-accuracy loss to ECBM (Table 4) is real and softens the headline.
- The Table 3 CUB SOTA result changes the encoder to ViT-L/14, confounding the small-concept-set claim with stronger image features.
- Hyperparameters $\gamma$ (L_concept scaling) and $\tau$ (contrastive temperature) are not tabulated; sensitivity unknown.
- The class-level intervention is a hand-curated 4-class patch on top of CSS, not a general procedure; comparisons to instance-level intervention baselines at matched effort are missing.
- The contrastive-vs-pseudo-label claim against Hu et al. (2024) is asserted but never tested.

## Why It Matters for Medical AI

Medical CBMs are exactly the regime where the WBCAtt-style setup makes sense: a small, expert-defined attribute vocabulary already exists (cell morphology, mammographic BI-RADS descriptors, radiology reporting templates), per-image concept annotation is expensive but a handful of labeled exemplars per class is realistic, and **faithfulness to clinical concepts is the whole point of the bottleneck**. Showing that CLIP VL-CBMs can score 90% class accuracy with sub-50% concept accuracy is a warning that any medical VL-CBM evaluated only by downstream prediction is probably hiding a faithfulness failure. The CSS recipe — frozen domain-specific VLM (here PLIP), thin trainable projection, sparse concept supervision — is well matched to clinical constraints.

The caveats are equally important. **No clinical or diagnostic dataset is evaluated.** WBCAtt asks "does this cell have a segmented nucleus?", not "does this slide indicate AML?" — attribute correctness is necessary but not sufficient for clinical CBM trust. Locality faithfulness (Raman et al. 2024) is also untested: a CSS-trained concept can score correctly while attending to the wrong spatial region, which in radiology means the right finding for the wrong reason. Before reading this as evidence for medical CBM deployment, downstream clinical evaluation, external-site validation, and locality testing are all needed.

## References

- Paper (arXiv): [Improving Concept Alignment in Vision-Language Concept Bottleneck Models, 2405.01825](https://arxiv.org/abs/2405.01825)
- Code: [github.com/NMS05/Improving-Concept-Alignment-in-Vision-Language-Concept-Bottleneck-Models](https://github.com/NMS05/Improving-Concept-Alignment-in-Vision-Language-Concept-Bottleneck-Models)
- Related: LaBo (Yang et al., 2023), Label-free CBM (Oikarinen et al., 2023), CDL, LM4CV, SparseCBM
- Related medical / baseline: PLIP (Huang et al., 2023), WBCAtt (Tsutsui, Pang, Wen, 2023), ECBM, PCBM
- Locality faithfulness: Raman et al., 2024
