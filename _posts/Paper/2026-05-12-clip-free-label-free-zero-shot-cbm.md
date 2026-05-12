---
title: "CLIP-Free, Label-Free, Zero-Shot Concept Bottleneck Models"
excerpt: "ZS-CBM converts any frozen classifier into a concept bottleneck by training only a small MLP and inserting a closed-form gram-matrix factor, reaching 86.4% ImageNet top-1 on ConvNeXtV2 versus 79.5% for the best CLIP-based supervised CBM."
categories:
  - Paper
tags:
  - ZS-CBM
  - TextUnlock
  - Concept-Bottleneck-Model
  - Interpretability
  - Zero-Shot
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- **TextUnlock + ZS-CBM** turns any frozen visual classifier into a concept bottleneck model without CLIP and without image-concept labels. A small MLP is trained to mimic the original classifier's soft distribution in the text-encoder space of class-name prompts; concepts are then attached by inserting a single gram-matrix factor $C^\top C$ with no further training.
- The headline number is **86.4% ImageNet top-1** with a ConvNeXtV2-Bpt@384 backbone, versus **79.5% for DN-CBM (CLIP-ViT-B/16)** and **75.4% for LF-CBM (CLIP-ViT-B/16)**. Across 40 backbones the TextUnlock alignment costs **~0.20 top-1 points** on average.
- The strongest evidence is the structural one — the concept-to-class matrix is genuinely closed form `W_con = C * U^T` — and the 40-backbone preservation sweep. The weakest is the headline comparison itself: ZS-CBM rides ImageNet-supervised backbones while every CLIP-CBM baseline rides unsupervised CLIP, so a non-trivial part of the 7-point gap is *label access*, not the new pipeline.

## Motivation

The label-free CBM literature has hardened into a CLIP monoculture. LF-CBM (Oikarinen et al., 2023), LaBo, DN-CBM, DCBM, CDM, and DCLIP all use CLIP either as the image encoder or as the image-concept alignment oracle (the $P$ matrix in LF-CBM). This forecloses interpretability for the entire universe of strong domain-specific classifiers: a radiology ResNet finetuned on CheXpert, a satellite-imagery EfficientNet, a DINO-pretrained microscopy ViT — none can be made into CBMs without either retraining as a CLIP-style contrastive model on a huge image-text corpus, or collecting expensive image-concept labels.

The paper's pitch is to do neither: distill the *original classifier's own class distribution* into a vision-language form by training a small MLP, then exploit the fact that the same text encoder defines both class and concept embeddings to construct the concept-to-class matrix in closed form. The medical-AI motivation is named explicitly in the framing — "legacy specialist models" — even though, as we will see, no medical dataset is actually evaluated.

## Core Innovation

- **TextUnlock as label-free distillation into text space.** A 2-layer MLP $\psi$ projects visual features $f$ into the text encoder's embedding space and is trained with cross-entropy *against the original classifier's softmax output $o$*, never against ground-truth labels. The objective is essentially $\mathrm{KL}(o \,\Vert\, \mathrm{softmax}(\tilde f \cdot U^\top))$, which preserves the full class distribution, not just the argmax.
- **Closed-form concept-to-class layer.** Once $\tilde f$ lives in the same space as class-name embeddings $U$ and concept embeddings $C$, the concept-to-class weights collapse to a text-to-text cosine matrix: $W_\mathrm{con} = C \cdot U^\top$. No GLM-SAGA, no elastic net, no training. The full logit becomes $\tilde f \cdot (C^\top C) \cdot U^\top$, and the gram matrix $C^\top C$ is the only structural difference between the CBM and the post-TextUnlock direct classifier.
- **CLIP-free by construction.** The image side uses a frozen specialist classifier; the text side uses a frozen sentence encoder (MiniLM SBERT in the paper). CLIP is never invoked, and the method consequently inherits whatever supervision the specialist classifier already encodes.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | TextUnlock is CLIP-free and label-free, applicable to any frozen visual classifier. | Method derivation (Sec 3.1); 40-backbone sweep in Table 1. | ImageNet | ⭐⭐⭐ |
| C2 | TextUnlock preserves original classifier accuracy (~0.2-point drop). | Table 1 across 40 models: avg $\Delta = -0.20$, max $-0.52$. | ImageNet | ⭐⭐⭐ |
| C3 | ZS-CBM outperforms all supervised CLIP-based CBMs. | Table 2: ConvNeXtV2-Bpt@384 86.4% vs best baseline DN-CBM 79.5%. | ImageNet | ⭐⭐ — true on numbers, but unfair backbone comparison (see honest take). |
| C4 | ResNet50 ZS-CBM beats CLIP-RN50 LF-CBM with 400x less image and 400,000x less text data. | Table 2: 73.9% vs 67.5%. | ImageNet | ⭐⭐ — framing hides that ZS-CBM still uses ImageNet class labels via the original classifier. |
| C5 | Method scales to domain-specific datasets without retraining the MLP. | Table 7: Places365 53.42 > CDM 52.70; EuroSAT 94.22 > CLIP 88.57; DTD 68.88 > CLIP 61.86. | Places365, EuroSAT, DTD | ⭐⭐⭐ |
| C6 | Concept-to-class layer is truly zero-shot (no training). | Eq. 2; concept filtering procedure (Appendix N). | All datasets | ⭐⭐⭐ — structurally true; $W_\mathrm{con} = C \cdot U^\top$ is closed form. |
| C7 | Method enables zero-shot image captioning beyond CLIP. | Table 3 COCO: CIDEr 17.9 / SPICE 6.9 vs ZeroCap 14.6 / 5.5 and ConZIC 12.8 / 5.2. | COCO Karpathy | ⭐⭐ — wins on CIDEr/SPICE only; BLEU-4 and METEOR are *worse* than baselines. |
| C8 | Concept interventions work for debugging biases. | Appendix E, Table 4 on Waterbirds-100 interventions. | Waterbirds-100 (curated, $N=140$) | ⭐⭐ — small self-curated split, no external standard. |
| C9 | Method is prompt-robust and text-encoder-robust. | Appendix H ($\Delta < 0.36$ across 20+ prompts); Appendix D ($\Delta \approx 0.07$ across 4 text encoders). | ImageNet | ⭐⭐⭐ |
| C10 | First CBM scaled to any backbone. | Implicit in Table 1's 40-backbone sweep vs LF-CBM's RN50/RN18 only. | ImageNet | ⭐⭐⭐ |

**Honest take.** The structural claims (C1, C2, C5, C6, C9, C10) are extremely well supported and the contribution is real: replacing a learned sparse linear probe with a closed-form gram-matrix factor is conceptually clean and *eliminates* the GLM-SAGA dependency that anchored LF-CBM's pipeline. The MLP-necessity ablation (Appendix C.1) is unusually rigorous — four different ablations (mean-feature, random-feature, random-weight, shuffled-feature) all collapse to 0.10-1.87% top-1.

The headline "outperforms supervised CLIP-based CBMs" claim (C3, C4) has a **serious backbone-supervision confound that the paper never directly addresses**. ZS-CBM uses backbones trained with full ImageNet class supervision (e.g., ConvNeXtV2-Bpt@384 was pretrained on ImageNet-21K and finetuned on ImageNet-1K with labels), while every LF-CBM baseline uses CLIP, which is not trained on ImageNet labels at all. So the 86.4% vs 79.5% delta is not apples-to-apples: ZS-CBM piggybacks on a strong *supervised* backbone that already encodes ImageNet's 1000-class structure, while CLIP must achieve everything zero-shot. A genuine head-to-head would put CLIP-ViT-L/14 *finetuned on ImageNet* in the comparison, and that experiment is missing. A non-trivial fraction of the 7-point gap is label access, not the new pipeline.

A second elephant in the room is **interpretability density**, which the paper sidesteps. LF-CBM's selling point was that the final layer is sparse and human-readable — **25-35 nonzero concepts per class** (0.7-15% density). ZS-CBM's $W_\mathrm{con} = C \cdot U^\top$ is **dense by construction** with all 20,000 concepts active per prediction. The paper compares accuracy across CBMs but never reports how many concepts dominate a typical ZS-CBM prediction, nor whether 20K dense contributions are more or less interpretable to a human than 25-35 sparse ones. For a method whose entire point is interpretability, this is a load-bearing gap.

## Method & Architecture

![TextUnlock training and inference pipeline](/assets/images/paper/clip-free-label-free-cbm/page_003.png)
*Figure 1: TextUnlock training (left) and inference (right). The MLP $\psi$ projects visual features $f$ into the text encoder's space, and is trained with cross-entropy against the original classifier's soft distribution $o$. At inference, the linear classifier weights are replaced by the text-encoded class prompts $U$ — the text encoder doubles as a frozen weight generator.*

### Stage 1 — TextUnlock

Let $F = W \circ F_v$ be a frozen classifier with visual encoder $F_v(I) = f \in \mathbb{R}^n$ and linear head $W \in \mathbb{R}^{n \times K}$. Let $T(\cdot)$ be a frozen text encoder (MiniLM SBERT in the paper) mapping text to $\mathbb{R}^m$. The original classifier's soft distribution is $o = \mathrm{softmax}(f \cdot W) \in \mathbb{R}^K$.

1. **Text-space class basis.** For each of $K$ classes, build $\ell_i = $ "an image of a $\{$class$_i\}$" and encode $u_i = T(\ell_i)$. Stack into $U \in \mathbb{R}^{K \times m}$. Both $U$ and the prompts are frozen.
2. **MLP $\psi$ projecting into text space.** Define $\tilde f = \mathrm{MLP}(f) \in \mathbb{R}^m$ and logits $s_i = \tilde f \cdot u_i$. Per Appendix C.2 the best config is 2 layers with `dim_out_factor = 2` (e.g., for ViT-B/16: $768 \to 1536 \to 384$). Only the MLP is trainable; $F_v$, $W$, and $T$ are all frozen.
3. **Distill the original distribution (no labels).** Train with

$$
\mathcal{L} = -\sum_i o_i \cdot \log \frac{\exp(s_i)}{\sum_j \exp(s_j)},
$$

equivalent to $\mathrm{KL}(o \,\Vert\, \mathrm{softmax}(s))$. No labels are ever used — the objective preserves the full original distribution.

### Stage 2 — ZS-CBM construction (no training)

![ZS-CBM construction with the closed-form concept-to-class matrix](/assets/images/paper/clip-free-label-free-cbm/page_005.png)
*Figure 2: ZS-CBM. (a) Concept activations $a = \tilde f \cdot C^\top$ are cosine similarities to a fixed 20K-concept set encoded by the same text encoder. (b) The concept-to-class matrix $W_\mathrm{con} = C \cdot U^\top$ is purely text-to-text similarity and requires no training. (c) The full CBM logit is $\tilde f \cdot (C^\top C) \cdot U^\top$.*

4. **Concept discovery.** Take a fixed concept set $Z$ (the paper uses the 20K most-common English words from `google-10000-english` after Appendix-N filtering: drop terms equal to the class name, constituents of the class name, parent/subparent classes, sibling species, and synonyms). Encode each $z_k$ with the *same* text encoder: $c_k = T(z_k)$, stack into $C \in \mathbb{R}^{Z \times m}$. For an image $I$, concept activations are $a = \tilde f \cdot C^\top \in \mathbb{R}^Z$.
5. **Closed-form concept-to-class.** Because $U$ and $C$ share a space,

$$
W_\mathrm{con} = C \cdot U^\top \in \mathbb{R}^{Z \times K}, \qquad S_\mathrm{cn} = (\tilde f \cdot C^\top)(C \cdot U^\top) = \tilde f \cdot (C^\top C) \cdot U^\top.
$$

If $C^\top C = I$ you recover the original classifier exactly; the gram matrix is the only structural perturbation.

6. **Training cost.** Only the MLP is trained, on ImageNet-1K with the soft-target cross-entropy. The original classifier is queried once to cache $o$.

## Experimental Results

### Main accuracy comparison on ImageNet (Table 2, page 7)

| Method | Type | Backbone | Top-1 |
|---|---|---|---|
| LF-CBM | Supervised, CLIP-based | CLIP RN50 | 67.5 |
| LF-CBM | Supervised, CLIP-based | CLIP ViT-B/16 | 75.4 |
| LaBo | Supervised, CLIP-based | CLIP ViT-B/16 | 78.9 |
| CDM | Supervised, CLIP-based | CLIP ViT-B/16 | 79.3 |
| DCLIP | Supervised, CLIP-based | CLIP ViT-B/16 | 68.0 |
| DN-CBM | Supervised, CLIP-based | CLIP ViT-B/16 | 79.5 |
| DCBM-SAM2 | Supervised, CLIP-based | CLIP ViT-L/14 | 77.9 |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **ResNet50** | **73.9** |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **ResNet50v2** | **78.1** |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **ViT-B/16v2** | **83.2** |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **BeiT-L/16** | **86.2** |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **ViT-L/16v2** | **86.3** |
| **ZS-CBM (ours)** | **Zero-shot, CLIP-free** | **ConvNeXtV2-Bpt@384** | **86.4** |

Read with care: the ZS-CBM rows use backbones trained *with* ImageNet class supervision, while every baseline row uses unsupervised CLIP. The ResNet50 comparison (73.9 vs LF-CBM's 67.5) is the most honest because of similar capacity, but even there ZS-CBM's RN50 is ImageNet-supervised and LF-CBM's CLIP-RN50 is not.

### TextUnlock accuracy preservation

Averaged over the 40 backbones in Table 1, the post-TextUnlock top-1 is **0.20 points** below the original classifier (range: $-0.52$ on BeiT-B/16 to $+0.18$ on DINOv2-B). Examples: ResNet50 75.80 vs 76.13 ($-0.33$); ViT-L/16v2 87.61 vs 88.06 ($-0.45$); ConvNeXtV2-Bpt@384 87.34 vs 87.50 ($-0.16$). The MLP genuinely preserves the original decision distribution.

### Cross-dataset CBM transfer (Table 7)

| Dataset | Best Supervised CLIP CBM | ZS-CBM (ours, ImageNet-trained only) |
|---|---|---|
| Places365 | CDM CLIP-ViT-B/16: 52.70 | **DenseNet161: 53.42** |
| EuroSAT | CLIP-ViT-B/16: 88.57 | **ResNet50: 94.22** |
| DTD | CLIP-ViT-B/16: 61.86 | **ResNet50: 68.88** |

The ImageNet-trained MLP transfers cleanly to scenes, satellite, and texture domains without retraining. The EuroSAT gap (+5.65) and DTD gap (+7.02) are large.

### Zero-shot captioning on COCO (Table 3)

ConvNeXtV2-Bpt@384 reaches **CIDEr 17.9 / SPICE 6.9** versus ZeroCap CIDEr 14.6 / SPICE 5.5 and ConZIC CIDEr 12.8 / SPICE 5.2. The paper acknowledges that BLEU-4 and METEOR are *lower* than ZeroCap and attributes this to n-gram-overlap bias against the method's caption style; a compositional in-context-learning variant ("@384com") recovers BLEU-4 to 4.40 versus baseline 2.6. The framing of "outperforming existing methods" only holds on 2 of 4 standard metrics.

### Qualitative concept attribution

![Qualitative ZS-CBM concept-attribution examples](/assets/images/paper/clip-free-label-free-cbm/page_008.png)
*Figure 3: Qualitative ZS-CBM explanations. Top concepts driving predictions like "scorpion" (claws, venomous, desert) and the documented "dumbbell" bias where the model latches onto the lifting arm because training images always show one.*

### Key ablations

- **MLP necessity (Appendix C.1).** Mean-feature, random-feature, random-weight, and shuffled-feature ablations of the MLP drop top-1 to 0.10-1.87% across ResNet101v2, ConvNeXt-Base, BeiT-L/16, and DINOv2-B. The MLP is genuinely learning a non-trivial projection.
- **MLP design (Appendix C.2).** 2 layers + `dim_out_factor = 2` is optimal (75.80% vs 1L/1x = 72.48%). Δ over the worst config is 3.32 points — non-trivial design sensitivity.
- **Text encoder choice (Appendix D).** Swapping MiniLM → DistilRoberta / MPNet-Base / MPNet-Base-MultiQA moves accuracy 75.73-75.80% — essentially insensitive.
- **Prompt robustness (Appendix H).** ViT-B/16 ImageNet top-1 ranges from 80.70 ("an image of a {}") to 80.34 (worst prompt) — Δ < 0.36 across 20+ prompts.
- **Concept intervention on Waterbirds-100 (Appendix E).** Zeroing bird concepts drops accuracy as expected; keeping only bird concepts improves it. Confirms the CBL is causally responsible for predictions, on a small ($N=140$) self-curated split.

### Polysemy failure mode

![Polysemy failure on the drake (bird) class](/assets/images/paper/clip-free-label-free-cbm/fig_p015_01.png)
*Figure 4: Top-detected concepts for a duck image labeled "drake" are "drake, dylan, cory, rihanna, robbie, thug, lyric, duck". The model latches onto Drake the rapper before Drake the bird — a striking failure mode for a system whose pitch is human-readable explanations.*

![Polysemy failure on the african grey parrot class](/assets/images/paper/clip-free-label-free-cbm/fig_p015_02.png)
*Figure 5: Same failure on "african grey" parrot — top concepts are "ethiopian, tanzania, arabidopsis, turquoise, purple, blues, greens". The 20K-word concept set drags in geographic and chromatic associations that have nothing to do with the bird.*

## Limitations

**Authors acknowledge (Appendix B).**

- Wrong semantic-association failures with polysemous class names (drake the bird vs Drake the rapper; "cock" yielding sexual associations).
- Partially mitigated but not eliminated by switching from the 20K-word set to LF-CBM's curated set.
- The same issue occurs in CLIP-based CBMs — true, but not a defense.

**Visible from the evidence but not addressed.**

- **Backbone-supervision confound in the headline comparison.** ZS-CBM uses ImageNet-supervised backbones; every LF-CBM baseline uses unsupervised CLIP. A non-trivial part of the 7-point gap is label access, not the new pipeline. A CLIP-ViT-L/14 finetuned on ImageNet — the obvious head-to-head — is missing.
- **No medical or specialist-domain evaluation despite being the explicit motivation.** Places365, EuroSAT, and DTD are still general-vision benchmarks. The "legacy specialist" pitch invited a radiology or histopathology classifier and the paper provides none.
- **Dense $W_\mathrm{con}$ versus sparse LF-CBM $W_F$ never compared on interpretability.** ZS-CBM's concept-to-class matrix is dense by construction with all 20K concepts active per prediction; LF-CBM deliberately sparsifies to 25-35 nonzeros per class. The paper compares accuracy but never reports how many concepts dominate a typical ZS-CBM prediction, nor whether 20K dense contributions are more or less interpretable to a human than 25-35 sparse ones.
- **No statistical reporting.** All numbers are point estimates; no std-dev, no significance tests vs CLIP-CBM baselines.
- **Concept-set filtering is dataset-specific.** The "filter out parent/subparent/synonyms" procedure (Appendix N) requires class-hierarchy knowledge — i.e., something domain-specific even if not labels.
- **Distillation faithfully preserves the original classifier's biases.** The MLP is trained to mimic $o$, so any systematic bias in the original classifier is encoded into the projection space. Inference-time concept interventions can correct biases, but the underlying $o$ distillation guarantees the bias is there.
- **No domain-specific text encoder tested.** All four sentence encoders in the ablation are general English encoders. A BiomedBERT or domain encoder — exactly what a "legacy specialist" classifier would pair with — is never tested.
- **Concept-set size never swept.** The paper uses 20K English words and reports no scaling study; LF-CBM's empirical result was that filtering down to ~150-4500 concepts works best.

## Why It Matters for Medical AI

The pitch is tailored to medical AI but the evidence is not. The motivation passage repeatedly invokes "legacy specialist classifiers" — a radiology ResNet, a microscopy ViT, a histopathology backbone — yet every experiment is run on general-vision benchmarks (ImageNet, Places365, EuroSAT, DTD, Waterbirds-100). For a clinical reader the interesting question is whether the MLP can preserve the *medically meaningful* class distribution of, say, a CheXpert-finetuned ResNet, and whether the 20K English-word concept set has anything to say about pneumonia, consolidation, or pleural effusion. The paper offers no answer.

The practical adaptation path is roughly: (1) swap the frozen text encoder for a clinical one (BiomedBERT, ClinicalBERT, BioMedCLIP's text tower); (2) replace the 20K-word concept set with a curated clinical vocabulary (RadLex, SNOMED CT subsets, or the concept lists already curated for medical CBM work); (3) audit polysemy carefully — the drake / Drake failure mode is exactly the kind of error that becomes unsafe when the concepts are "consolidation" or "infiltrate". The closed-form $W_\mathrm{con} = C \cdot U^\top$ should still work, and the TextUnlock distillation is genuinely backbone-agnostic, so the architecture transfers. What does not transfer is the validation: until someone runs the experiment on real clinical backbones, the medical pitch remains a hypothesis.

## References

- Paper: Anonymous. *CLIP-Free, Label-Free, Zero-Shot Concept Bottleneck Models.* ICLR 2026 (under review). [OpenReview 9YpbmkPmuT](https://openreview.net/forum?id=9YpbmkPmuT)
- Related: Oikarinen et al., *Label-Free Concept Bottleneck Models*, ICLR 2023.
- Related: Koh et al., *Concept Bottleneck Models*, ICML 2020.
- Related: Yuksekgonul et al., *Post-hoc Concept Bottleneck Models*, ICLR 2023.
- Related: Yang et al., *Language in a Bottle (LaBo)*, CVPR 2023.
- Related: Rao et al., *Discover-then-Name CBM (DN-CBM)*, ECCV 2024.
- Related: Reimers and Gurevych, *Sentence-BERT*, EMNLP 2019.
