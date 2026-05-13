---
title: "Towards a Visual-Language Foundation Model for Computational Pathology (CONCH)"
excerpt: "A CoCa-style pathology VLM trained on 1.17M curated image-caption pairs delivers zero-shot NSCLC subtyping balanced accuracy 0.900 and matches CTransPath on supervised SICAP quad-kappa (0.846 vs 0.835)."
categories:
  - Paper
tags:
  - CONCH
  - CoCa
  - Vision-Language
  - Computational-Pathology
  - Foundation-Model
  - Zero-Shot
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- CONCH curates **1,170,647 human-histology image-caption pairs** from PubMed Open Access and educational sources via a three-stage YOLOv5 + GPT-splitter + CLIP-alignment pipeline, then trains a CoCa-style model (ViT-B/16 image encoder, 12-layer GPT text encoder, 12-layer multimodal decoder) with contrastive + captioning losses for 40 epochs on 8x A100.
- Across **13 benchmarks** (slide-level classification, ROI classification, cross-modal retrieval, zero-shot segmentation, captioning) CONCH beats PLIP, BiomedCLIP, and OpenAI CLIP — usually with p<0.01 in two-sided permutation tests with 1,000-sample bootstrap CIs.
- Headline numbers: **NSCLC zero-shot balanced accuracy 0.900** (PLIP 0.787), **BRCA 0.840** (PLIP 0.507), **SICAP quadratic-kappa 0.711** (BiomedCLIP 0.553), mean text-to-image retrieval recall 0.440 vs BiomedCLIP 0.267, and supervised **SICAP quad-kappa 0.846 matching CTransPath 0.835** — i.e., the CoCa-trained encoder reaches state-of-the-art SSL pathology backbone quality without being SSL-only.

## Motivation

Diagnostic pathology spans thousands of entities and is heavily verbal: reports, textbooks, and journal figures encode most of the working knowledge. The dominant Computational Pathology (CPath) paradigm — one supervised model per cohort — does not scale to rare diseases or open-set recognition, and prior pathology vision-language models (PLIP from Twitter scraping, BiomedCLIP from broad PubMed, MI-Zero) are limited by either small or impure pretraining data and have only been demonstrated on ROI-tile classification. WSI-scale zero-shot classification, segmentation, retrieval, and captioning are largely unaddressed. A single task-agnostic backbone that can be steered with natural-language prompts would cut annotation burden across the heterogeneous pathology workflow — that is the gap CONCH targets.

## Core Innovation

- **A data-curation pipeline that turns >18M raw PubMed figures into 1.17M usable pathology image-caption pairs.** A YOLOv5 detector extracts single-panel histology sub-images from multi-panel figures; a PubMed-pretrained GPT splits compound captions at `"Next caption: "`; a CLIP model trained on cleaned data aligns sub-images to sub-captions by cosine similarity.
- **A pathology-specific CoCa configuration.** The vision tower is warm-started with iBOT on 16M in-house WSI tiles; the language tower is warm-started by next-token prediction on 550k+ MGH reports plus 400k+ pathology PubMed abstracts; the bottom 12 LM layers become the text encoder, the top 12 layers + LM head become the multimodal decoder.
- **Joint contrastive + captioning training** with two attentional poolers on the image side — one query for the contrastive global token, 256 queries for fine-grained caption tokens — making one backbone serve classification, retrieval, segmentation, and captioning simultaneously.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CONCH beats prior pathology VLMs at zero-shot tile/slide classification "often by a wide margin" | Tables 1-7; p<0.01 permutation test on most metrics; prompt-ensembled | TCGA BRCA/NSCLC/RCC, DHMC, SICAP, CRC100k, WSSS4LUAD | ⭐⭐⭐ |
| C2 | The 1.17M-pair corpus is what enables the lift over BiomedCLIP | Compared at matched downstream protocol, but architecture and data differ simultaneously; only intra-CONCH ablations in Extended Fig. 8 | Internal pretraining splits | ⭐⭐ |
| C3 | Captioning loss helps representation quality over plain CLIP on the same data | Extended Fig. 8a: CoCa avg > CLIP avg on classification; CLIP variant edges ahead on retrieval | 7 classification + 3 retrieval datasets | ⭐⭐ |
| C4 | Strong zero-shot WSI-level performance, not just ROI | BRCA 0.840 / NSCLC 0.900 / RCC 0.893 balanced acc, DHMC kappa 0.236; p<0.01 vs next-best | 4 slide-level test sets, n=150-225 | ⭐⭐⭐ |
| C5 | Zero-shot WSI segmentation approaches supervised quality | Dice 0.601 (SICAP) / 0.615 (DigestPath) vs PLIP 0.549 / 0.426; no supervised reference reported | SICAP n=31, DigestPath n=250 | ⭐⭐ |
| C6 | 8x label efficiency vs baselines in few-shot supervised learning | Figure 3, Extended Fig. 6 with n=5 runs per nc | BRCA, NSCLC, RCC, SICAP, CRC100k | ⭐⭐ |
| C7 | Retrieval substantially outperforms baselines | Mean recall +17.3% over BiomedCLIP averaged; p<0.01 on 5/6 directions; TCGA LUAD t2i gain not significant (p=0.22) | Source A/B (private), TCGA LUAD | ⭐⭐ |
| C8 | First pathology VLM with image captioning | METEOR 0.193 vs GIT-large 0.125; only 558 fine-tuning pairs; authors note verbatim regurgitation | Source A test, n=162 | ⭐ |
| C9 | Image encoder matches CTransPath (a state-of-the-art SSL pathology backbone) | SICAP supervised quad-kappa **0.846 vs 0.835**; CRC100k 0.930 vs **0.938** | SICAP, CRC100k | ⭐⭐⭐ |
| C10 | Heatmaps provide meaningful interpretability of WSI predictions | Figure 2e + Extended Figs. 3-5 — qualitative pathologist agreement only | TCGA cases | ⭐ |

**Honest read.** The headline classification claims (C1, C4) travel furthest: 7 tasks across diverse organ systems, bootstrap CIs, and explicit p-values. The CTransPath-matching encoder result (C9) is genuinely strong because it pits CONCH head-to-head with a rigorous SSL baseline on the two tasks where CTransPath does not have a TCGA leakage advantage. Where framing exceeds evidence: (a) **data attribution (C2)** mixes architecture, unimodal pretraining, and captioning loss — the iBOT pretraining on 16M in-house tiles is an advantage no baseline has, and the 1.17M-pair corpus is never isolated against a same-scale BiomedCLIP variant. (b) **TCGA LUAD retrieval (C7)** gain over BiomedCLIP is only 5.3% and *not* statistically significant (p=0.22); the "wide margin" framing rests on Source A/B, both of which are private and unverifiable. (c) **Segmentation (C5)** at n=31 slides for SICAP with no supervised reference makes "approaches supervised quality" implied rather than shown. (d) **Captioning (C8)** is a proof of concept — METEOR 0.193 is well below natural-image baselines, 558 fine-tuning pairs is tiny, and authors flag verbatim regurgitation. (e) **Single-seed supervised training** for all main supervised tables (only test-set bootstrap CIs; only few-shot uses n=5 runs). (f) **No external slide-level validation** outside TCGA/DHMC, and **possible TCGA leakage into PubMed pretraining** goes unexamined despite the same authors aggressively avoiding CTransPath on TCGA tasks for the inverse reason.

## Method & Architecture

![CONCH overview: data curation, CoCa training, and downstream radar](/assets/images/paper/2307.12914_CONCH/page_004.png)
*Figure 1: CONCH overview — automated curation of 1.17M pathology image-caption pairs (a-b), CoCa-style image-encoder / text-encoder / multimodal-decoder training (c), and downstream performance radar across 13 tasks (d).*

### 1. Pretraining-data curation (1.17M pairs)

Two source pools: **EDU** (~45k manually cleaned educational pairs) and **PMC OA** (~18M raw figures from PubMed Central Open Access). The pipeline runs three stages.

- **Histopathology detection.** A YOLOv5 detector is trained on synthetic multi-panel images assembled from clean EDU singles, then iteratively refined on PMC OA misses. It outputs single-panel sub-images from compound figures.
- **Caption splitting.** A PubMed-pretrained GPT-style LM is fine-tuned on EDU original-vs-split pairs and used as a causal LM that emits sub-captions delimited by the literal token `"Next caption: "`.
- **Image-caption alignment.** A CLIP model trained on cleaned EDU + single-figure PMC OA assigns each detected sub-image to the split caption with highest cosine similarity in its aligned latent space.

The pipeline outputs **1,786,362** raw pairs; filtering to human-only yields **1,170,647** — the default CONCH pretraining set. An H&E-only subset (457,372) is worse on average (Extended Fig. 8).

### 2. Unimodal warm-starts

- **Vision tower:** ViT-B/16 trained with iBOT on **16M 256x256 tiles at 20x** from 21,442 in-house WSIs covering >350 OncoTree cancer subtypes (80 epochs, batch 1024, 4x A100).
- **Language tower:** 24-layer GPT-style autoregressive Transformer (embed 768, hidden 3072, vocab 32k, seq len 512) trained by next-token prediction on **550k+ MGH final-diagnosis reports** (regex de-identified) + **400k+ pathology PubMed abstracts** + educational text.
- After unimodal pretraining, layers 1-12 of the LM initialize the unimodal text encoder; layers 13-24 + LM head initialize the multimodal decoder.

### 3. CoCa joint training

Three modules:

- **Image encoder** $f(\cdot; \theta)$: ViT-B/16 + two attentional poolers — $f_\text{contrast}$ with **1 query** (global token used for contrast), $f_\text{caption}$ with **256 queries** (fine-grained tokens for the decoder).
- **Text encoder** $g(\cdot; \phi)$: 12-layer GPT-style with appended `<CLS>`.
- **Multimodal decoder** $h(\cdot; \psi)$: 12-layer GPT-style with cross-attention to the 256 caption tokens after each self-attention block, plus the LM head.

Loss — image-text contrastive + text-image contrastive + autoregressive captioning, equal weights:

$$
\mathcal{L} = -\frac{1}{2M}\sum_{i=1}^{M}\log\frac{\exp(\tau u_i^\top v_i)}{\sum_{j=1}^{M}\exp(\tau u_i^\top v_j)} -\frac{1}{2M}\sum_{j=1}^{M}\log\frac{\exp(\tau v_j^\top u_j)}{\sum_{i=1}^{M}\exp(\tau v_j^\top u_i)} -\frac{1}{M}\sum_{i=1}^{M}\sum_{t=1}^{T+1}\log p(w_{i,t}\mid w_{i,0:t-1}, x_i; \theta,\phi,\psi)
$$

with $(u_i, v_i)$ the L2-normalized image/text embeddings and $\tau$ a learned temperature.

Training config: 8x A100 80GB, local batch 48 with gradient accumulation 4 (effective global batch **1536**), image size **448x448**, max caption length 128, **40 epochs**, AdamW (0.9, 0.999), weight decay 0.2, peak LR 1e-4 cosine, 250 warm-up steps, fp16 mixed precision. Built on `open_clip` 2.14.0.

### 4. Downstream usage

- **Zero-shot tile classification.** Prompts `"{template} {classname}."`; L2-normalize text embeddings as a linear head; argmax cosine similarity. Both single-prompt (median over 50 random samples) and prompt-ensembled (mean text embedding) results are reported.
- **Zero-shot WSI classification.** Tile at 10x via CLAM segmentation, embed, score against each class prompt, aggregate with **top-K pooling** (best K in {1, 5, 10, 50, 100}).
- **Zero-shot segmentation.** Tile at 224x224 with 75% overlap; assign each pixel the argmax class label; average overlapping predictions.
- **Supervised slide-level.** CONCH tile embeddings (the 512-d contrastive-pooler output) into **ABMIL** (gated attention, hidden 384, dropout 0.25, 20 epochs, AdamW LR 1e-4 cosine, class-balanced sampler).
- **Supervised ROI.** Linear probing via logistic regression with L2 $\lambda = 100/(MC)$, LBFGS, max 800 iters.
- **Captioning fine-tuning.** Set the contrastive weight to 0; train up to 40 epochs with early stopping (patience 10) on validation METEOR; top-K sampling (K=50) at inference.

## Experimental Results

### Zero-shot classification (prompt ensembled)

| Task (metric) | n | **CONCH** | PLIP | BiomedCLIP | OpenAICLIP |
|---|---|---|---|---|---|
| BRCA subtyping (bal. acc.) | 150 | **0.840** | 0.507 | 0.553 | 0.500 |
| RCC subtyping (bal. acc.) | 225 | **0.893** | 0.804 | 0.791 | 0.347 |
| NSCLC subtyping (bal. acc.) | 150 | **0.900** | 0.787 | 0.780 | 0.553 |
| DHMC LUAD (Cohen's kappa) | 143 | **0.236** | 0.079 | 0.009 | 0.004 |
| SICAP (quadratic kappa) | 2,122 | **0.711** | 0.187 | 0.553 | 0.107 |
| CRC100k (bal. acc.) | 7,180 | **0.791** | 0.674 | 0.553 | 0.271 |
| WSSS4LUAD (bal. acc.) | 4,693 | **0.719** | 0.624 | 0.616 | 0.296 |

CONCH leads every entry, with the most dramatic gap on SICAP (Gleason grading) where PLIP collapses to 0.187 quadratic kappa.

![Zero-shot and supervised classification comparison](/assets/images/paper/2307.12914_CONCH/page_007.png)
*Figure 2: Zero-shot (c) and supervised (d) classification of CONCH vs PLIP / BiomedCLIP / OpenAICLIP across 7 tasks; an example IDC slide and its CONCH cosine-similarity heatmap (e).*

### Label efficiency

In BRCA subtyping, CONCH+ABMIL with **8 labels/class** already beats every baseline using **64 labels/class** — an 8x labeling reduction. CONCH **zero-shot** beats PLIP/BiomedCLIP **supervised** up to 64 labels/class on BRCA and up to 128 on NSCLC.

![Slide-level few-shot ABMIL boxplots](/assets/images/paper/2307.12914_CONCH/page_009.png)
*Figure 3: Slide-level few-shot ABMIL boxplots (n=5 runs per nc). The dotted line is CONCH zero-shot performance — it outperforms baseline few-shot supervised training up to 64 labels/class on BRCA and 128 on NSCLC.*

### Cross-modal retrieval (mean recall, text->image / image->text)

| Dataset | n | **CONCH** | PLIP | BiomedCLIP | OpenAICLIP |
|---|---|---|---|---|---|
| Source A (t2i / i2t) | 797 | **0.688 / 0.692** | 0.187 / 0.202 | 0.373 / 0.397 | 0.049 / 0.044 |
| Source B (t2i / i2t) | 1,755 | **0.390 / 0.366** | 0.076 / 0.075 | 0.239 / 0.236 | 0.032 / 0.025 |
| TCGA LUAD (t2i / i2t) | 165 | **0.240 / 0.182** | 0.096 / 0.071 | 0.187 / 0.166 | 0.062 / 0.048 |

Caveat: TCGA LUAD t2i gain over BiomedCLIP is only 5.3 points and **not statistically significant (p=0.22)**, and Source A/B are private.

![Zero-shot retrieval bars and TCGA LUAD examples](/assets/images/paper/2307.12914_CONCH/page_010.png)
*Figure 4: Zero-shot cross-modal retrieval — text-to-image and image-to-text Recall@K on Source A/B and TCGA LUAD (a), retrieval schematic (b), and example top-5 retrievals on TCGA LUAD (c).*

### Zero-shot segmentation (Dice / Precision / Recall)

| Dataset | **CONCH** | PLIP | BiomedCLIP | OpenAICLIP |
|---|---|---|---|---|
| SICAP (n=31) | **0.601 / 0.672 / 0.751** | 0.549 / 0.605 / 0.644 | 0.484 / 0.536 / 0.557 | 0.367 / 0.599 / 0.605 |
| DigestPath (n=250) | **0.615 / 0.663 / 0.709** | 0.426 / 0.526 / 0.541 | 0.446 / 0.581 / 0.601 | 0.367 / 0.492 / 0.511 |

The main-text Methods quotes DigestPath Dice 0.569 / Recall 0.684 / Precision 0.644 while Extended Data Table 27 reports 0.615 / 0.709 / 0.663 — the discrepancy is unexplained in-text; we use Table 27.

![Zero-shot segmentation pipeline and qualitative masks](/assets/images/paper/2307.12914_CONCH/page_012.png)
*Figure 5: Zero-shot segmentation pipeline (a), Dice / Precision / Recall on SICAP and DigestPath (b-c), and qualitative tumor masks vs ground truth (d-e).*

### Supervised classification — does the encoder match CTransPath?

| Task | Metric | **CONCH** | PLIP | BiomedCLIP | OpenAICLIP | CTransPath / RN50 |
|---|---|---|---|---|---|---|
| BRCA (n=150) | bal. acc. | **0.847** | - | - | - | RN50: +8.0% lower |
| RCC (n=225) | bal. acc. | **0.942** | - | - | - | RN50: -4.9% |
| NSCLC (n=150) | bal. acc. | **0.927** | - | - | - | RN50: 0.840 |
| SICAP (n=2,122) | quad-kappa | **0.846** | 0.762 | 0.716 | 0.704 | CTransPath: 0.835 |
| CRC100k (n=7,180) | bal. acc. | 0.930 | 0.879 | 0.896 | 0.884 | **CTransPath: 0.938** |

On the two tasks where CTransPath does not have a TCGA leakage advantage (SICAP, CRC100k), CONCH and CTransPath are within ~1 percentage point of each other in either direction. Given that CTransPath is a hand-tuned SSL pathology encoder, getting equivalent linear-probe behavior out of the **contrastive-pooler head of a CoCa model** is the strongest unsexy result in the paper.

### Captioning

| Model | METEOR | ROUGE |
|---|---|---|
| **CONCH** | **0.193** | **0.215** |
| GIT-base | 0.122 | 0.135 |
| GIT-large | 0.125 | 0.153 |

Read as a proof-of-concept rather than a deployment-ready captioner — only 558 fine-tuning pairs, 162 test, and the authors flag verbatim regurgitation in the failure analysis.

![Captioning bars plus a success and a clear-cell RCC vs fat-necrosis failure](/assets/images/paper/2307.12914_CONCH/page_014.png)
*Figure 6: Captioning METEOR/ROUGE vs GIT-base/large (a); a successful caption example (b); and a clear-cell-RCC-vs-fat-necrosis confusion that the authors flag as a typical failure mode (c).*

### Pretraining-data ablation

Human-only CoCa > full unfiltered > H&E-only on the **classification** average, and human-only CoCa > human-only **CLIP** on classification — but the CLIP variant edges ahead on retrieval. This means the captioning loss helps representation learning more on classification than on retrieval.

![Pretraining-data ablation across classification and retrieval](/assets/images/paper/2307.12914_CONCH/page_036.png)
*Extended Figure 8: Pretraining-data ablation across 7 classification benchmarks (a) and 3 retrieval datasets (b) — human-only CoCa wins on average classification; the CLIP variant on the same data edges ahead on retrieval.*

## Limitations

**Acknowledged by the authors.**
- Pretraining scale (1.17M) is small versus billion-scale general VLMs; further scaling is expected to help.
- Image-level / tile-level focus only. Fine-grained tasks — mitosis detection, sub-cellular features, cell counting — are out of scope.
- Captioning quality is weak in absolute terms and sometimes regurgitates training text.
- Educational pretraining data is publisher-copyrighted and not redistributable.

**Not addressed.**
- **No external slide-level validation** outside TCGA / DHMC.
- **Possible TCGA leakage into PubMed pretraining** is never audited, even though the curation pipeline does not deduplicate against TCGA test slides. The authors aggressively avoid CTransPath on TCGA for the same reason in reverse — but do not apply that same scrutiny to themselves.
- **Source A / Source B unnamed and not released**, so 2/3 of retrieval evaluation and 100% of captioning evaluation cannot be reproduced.
- **Single-seed supervised training** for Tables 15-19; only test-set bootstrap CIs.
- **Prompt sensitivity** is studied, but the prompt pools were authored by the team's pathologists.
- **Fairness / demographic robustness** (race, age, scanner vendor, stain protocol) is not analyzed.
- **Compute and carbon costs** are not reported.
- **Weights are not openly released at preprint time** — "may be requested upon institutional permission and case by case approval."

## Why It Matters for Medical AI

Pathology is one of the few medical domains where natural-language supervision is both abundant (reports, textbooks, journal articles) and tightly coupled to image features. CONCH demonstrates that with careful curation a single backbone can credibly serve **WSI classification, ROI classification, retrieval, segmentation, and captioning** — collapsing what has historically been five separate supervised pipelines into one task-agnostic foundation. The C9 result (CoCa-trained encoder matching CTransPath on SICAP and CRC100k) is the load-bearing one for medical deployment: it says a single VLM training run can produce a vision tower that is competitive with the best dedicated SSL pathology encoder, while *also* getting the text alignment and zero-shot capability for free. The honest caveats — single-institution test sets, possible TCGA leakage, weights not openly released — temper the "foundation model for pathology" framing, but the direction is clearly correct.

## References

- Lu, M. Y., Chen, B., Williamson, D. F. K., et al. *Towards a Visual-Language Foundation Model for Computational Pathology.* arXiv:2307.12914v2, 25 Jul 2023. Later published as Lu et al., *Nature Medicine* 2024, "A visual-language foundation model for computational pathology."
- Paper PDF: <https://arxiv.org/abs/2307.12914>
- Code / weights (gated; Nature Medicine release): <https://github.com/mahmoodlab/CONCH>
- Related work: PLIP (Huang et al., Nature Medicine 2023), BiomedCLIP (Zhang et al., 2023), MI-Zero (Lu et al., CVPR 2023), CoCa (Yu et al., 2022), iBOT (Zhou et al., ICLR 2022), CTransPath (Wang et al., MedIA 2022), CLAM (Lu et al., Nature BME 2021), ABMIL (Ilse et al., ICML 2018).
