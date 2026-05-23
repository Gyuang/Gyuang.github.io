---
title: "MFTrans: A Multi-Resolution Fusion Transformer for Robust Tumor Segmentation in Whole Slide Images"
authors: "Sungkyu Yang¹, Woohyun Park²·³, Kwangil Yim³·*, Mansu Kim¹·*"
venue: "WACV 2025"
date: 2025-02-18
order: 2
thumbnail: /assets/images/projects/mftrans/hero.png
hero: /assets/images/projects/mftrans/hero.png
hero_caption: "Dual-branch architecture — high-magnification local CNN branch (ConvNeXt) and low-magnification global ViT branch are fused stage-by-stage via cross-attention + Token Fusion Blocks, mirroring how pathologists alternate between zoom levels during diagnosis."
abstract: "Dual-branch CNN + Transformer that fuses high- and low-magnification WSI patches via stage-wise cross-attention, achieving SOTA tumor segmentation on Camelyon16, PAIP2019, and the Catholic Uijeongbu dataset."
links:
  - { label: "Paper (PDF)", url: "/assets/papers/mftrans.pdf" }
  - { label: "Poster", url: "/assets/papers/mftrans-poster.pdf" }
  - { label: "Code", url: "https://github.com/aimed-gist/MFTrans" }
  - { label: "BibTeX", url: "#bibtex" }
published: true
---

<p style="text-align:center; color:#666; font-size:0.9em; margin-top:-0.5em;">¹ AI Graduate School, GIST &nbsp;&middot;&nbsp; ² Data Science, The Catholic Univ. of Korea &nbsp;&middot;&nbsp; ³ Hospital Pathology, College of Medicine, The Catholic Univ. of Korea<br><span style="font-size:0.85em;">* corresponding authors</span></p>

## Abstract

Accurate tumor segmentation in whole slide images (WSI) is essential for histopathological diagnosis, but manual analysis is labor-intensive and shows up to **24% inter-pathologist discrepancy**. Existing AI models typically operate at a **single magnification**, which limits the diagnostic information available.

We propose **MFTrans**, a multi-resolution fusion transformer that integrates **high- and low-magnification** images through a dual-branch architecture. A **global token transformer + cross-attention mechanism** fuses hierarchical features stage-by-stage, mirroring how expert pathologists alternate between low-power (context) and high-power (cell morphology) views during diagnosis. MFTrans achieves SOTA segmentation on **Camelyon16, PAIP2019, and the Catholic Uijeongbu St. Mary's Hospital** datasets under both balanced and imbalanced setups.

## Motivation

![Pathologist diagnostic workflow vs single-magnification models](/assets/images/projects/mftrans/motivation.png)
*Pathologists assess tumor structures at low magnification and inspect cell morphology at high magnification. Single-magnification models lose half of this diagnostic signal.*

## Architecture

The model has three coupled branches:

1. **Local branch** — ConvNeXt encoder on high-magnification patches, captures fine-grained cellular features via depth-wise convolutions + locality bias.
2. **Global branch** — ViT-style global token transformer on low-magnification crops, models long-range tissue context.
3. **Fusion branch** — at each stage, a **Token Fusion Block** uses the global token as query and local features as key/value via cross-attention, then concatenates and refines via inverted bottleneck convolutions before passing to the image decoder.

![Detailed architecture of the Global and Fusion branches](/assets/images/projects/mftrans/architecture.png)
*Sub-figures (a), (b), (c): Global Token Transformer Block, Image Decoder Block, and Token Fusion Block. Multi-stage fusion lets low-mag global context guide high-mag local prediction (and vice versa) at every resolution level.*

**Training objective** combines BCE + Dice over the segmentation head, multi-stage auxiliary segmentation losses, and a binary tumor-presence classification loss:

$$\mathcal{L}_{\text{total}} = \alpha\,\mathcal{L}(M, \text{head}(f_0)) + \beta\,\mathcal{L}(M, \text{head}(f_2)) + \gamma\,\mathcal{L}(M, \text{head}(f_4)) + \omega\,\mathcal{L}_{\text{bce}}(CM, C(G_4))$$

where $M$ is the GT mask and $CM$ is a binary tumor-presence label.

## Results

Headline numbers (Dice / Jaccard / Accuracy):

| Dataset | MFTrans | 2nd best |
|---|---|---|
| **Camelyon16** | **0.920 / 0.851 / 0.925** | 0.918 / 0.848 / 0.925 (Transfuse) |
| **PAIP2019** | **0.818 / 0.692 / 0.938** | 0.810 / 0.681 / — (Transfuse) |
| **Catholic Uijeongbu** | SOTA | — |

MFTrans wins on the harder **PAIP2019** liver-cancer set (more complex tumor morphology) by a clearer margin, while Camelyon16 is a near-tie. The Catholic Uijeongbu hospital dataset — annotated end-to-end by collaborating pathologist **K. Yim** — validates the model on real-world clinical workflow rather than curated challenge data.

![Qualitative segmentation comparison](/assets/images/projects/mftrans/qualitative.png)
*Qualitative comparison vs competing methods on Camelyon16 / PAIP2019. MFTrans recovers fine tumor boundaries that single-magnification models smear.*

## Poster

![WACV 2025 poster](/assets/images/projects/mftrans/poster.png)
*WACV 2025 poster summarizing motivation, architecture, and results. [Download full PDF](/assets/papers/mftrans-poster.pdf).*

## BibTeX

```bibtex
@inproceedings{yang2025mftrans,
  title     = {MFTrans: A Multi-Resolution Fusion Transformer for Robust Tumor Segmentation in Whole Slide Images},
  author    = {Yang, Sungkyu and Park, Woohyun and Yim, Kwangil and Kim, Mansu},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  pages     = {4595--4604}
}
```
