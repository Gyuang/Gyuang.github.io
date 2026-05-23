---
title: "BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs"
excerpt: "Scraping PubMed Central into 15.28M figure-caption pairs (PMC-15M) and retraining CLIP with PubMedBERT lifts biomedical text-to-image R@1 from 1.00% to 69.60%."
categories:
  - Paper
  - Pathology
permalink: /paper/biomedclip-general-biomedical-vision-language-pretraining/
tags:
  - BiomedCLIP
  - PMC-15M
  - CLIP
  - PubMedBERT
  - Vision-Language
  - Foundation-Model
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- BiomedCLIP scrapes 4.4M PubMed Central Open Access articles into **PMC-15M = 15,282,336 figure-caption pairs** spanning 30+ image categories (radiology, pathology, microscopy, plus a long tail of charts and diagrams) — more than 100x larger than MIMIC-CXR (377k) and 175x larger than ROCO (87k).
- The recipe is OpenCLIP with three coordinated domain swaps: **PubMedBERT** text encoder + WordPiece tokenizer, context length **77 -> 256 tokens** (covers 90% of PMC captions), and ViT-B/16 at 224x224 (384-px and larger ViTs were tried and rejected for hurting downstream accuracy).
- Headline retrieval gap on the PMC-15M test split (n=725,739): **txt->img R@1 = 69.60%** for BiomedCLIP vs **8.48%** for PubMedCLIP and **1.00%** for OpenAI CLIP. On RSNA pneumonia linear-probing, BiomedCLIP at 10% labels already matches fully-supervised BioViL despite seeing fewer chest X-rays during pretraining.

## Motivation

Public biomedical multimodal corpora before 2023 were small (<=377k pairs), access-restricted (MIMIC-CXR), and overwhelmingly chest-X-ray-shaped. General-domain CLIP transfers poorly: on the PMC-15M retrieval benchmark it scores R@1 = 1.00% (txt->img). The bet behind BiomedCLIP is that a single broad-domain foundation model trained on every image type PubMed ships — radiology + pathology + microscopy + dermatology + diagrams — can outperform a modality-specific specialist through positive transfer. That is the exact opposite of the pathology-only design philosophy of PLIP and CONCH, and the paper rests its narrative on the surprise that *more diverse* data beats *more in-domain* data.

## Core Innovation

- **PMC-15M pipeline.** All 4.4M PubMed Central Open Access full-text articles (snapshot June 15, 2022) parsed via PubMed Parser on Azure Databricks (Spark) to yield 15.28M figure-caption pairs with PMID/PMCID provenance. Two orders of magnitude larger than any prior biomedical multimodal corpus.
- **Domain-tuned text branch.** Replace CLIP's GPT-2/BPE text tower with PubMedBERT (Gu et al. 2021, domain-pretrained on PubMed) + WordPiece, extend context 77 -> 256 tokens. Ablation isolates this as the single most impactful axis: val loss 0.6626 -> 0.4807, img->txt R@1 64.53 -> 73.50.
- **Resolution discipline.** ViT-B/16 at 224x224 wins downstream despite 384x384 yielding better validation loss — a clean val-vs-downstream tradeoff where the authors picked the right side and ablate it transparently.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | PMC-15M is two orders of magnitude larger than prior biomedical multimodal datasets | Direct count: 15.28M vs MIMIC-CXR 377k, CheXpert 224k, ROCO 87k, ARCH 7.5k | - | ⭐⭐⭐ |
| C2 | BiomedCLIP outperforms general-domain CLIP on biomedical tasks by a wide margin | txt->img R@1 69.60 vs 1.00; zero-shot mean across 5 classification sets | PMC-15M test + 5 classification | ⭐⭐⭐ |
| C3 | BiomedCLIP substantially outperforms prior biomedical VLMs (PubMedCLIP, MedCLIP) | Retrieval and zero-shot tables | PMC-15M, PCam, LC25000, TCGA-TIL, RSNA | ⭐⭐ — unfair to PubMedCLIP: data scale, not just architecture, differs |
| C4 | BiomedCLIP beats radiology-specialist BioViL on RSNA pneumonia despite less radiology pretraining | Fig. 3D linear probe; BiomedCLIP 10% ~= BioViL 100% | RSNA only | ⭐⭐ — single dataset, single baseline, no replication on CheXpert/ChestX-ray14, no variance |
| C5 | Diverse multi-modality pretraining yields positive transfer to specialty tasks | Same RSNA result + zero-shot wins | 5 classification + 2 VQA | ⭐⭐ — supportive but the specialty pillar is one benchmark |
| C6 | Domain text encoder (PubMedBERT) > general (GPT-2) for biomedical CLIP | Suppl. Table 1: val loss 0.6626 -> 0.5776 with PubMedBERT swap alone | PMC-15M val | ⭐⭐⭐ — clean A/B |
| C7 | 256-token context helps because captions are long | Suppl. Table 1: val loss 0.5776 -> 0.4807 | PMC-15M val | ⭐⭐⭐ |
| C8 | Larger ViT (B vs S/M) helps | Suppl. Table 2: R@1 69.45 / 71.85 / 73.50 | PMC-15M val | ⭐⭐ — val only, downstream effect not isolated |
| C9 | 384-px input hurts downstream despite better val loss | Suppl. Tables 4-5: val loss 0.3819 -> 0.3406 but zero-shot mean 75.52 -> 70.37 | PMC-15M val + 5 zero-shot sets | ⭐⭐⭐ |
| C10 | BiomedCLIP can serve as a privacy-preserving proxy for proprietary CXR via PMC-15M retrieval | Fig. 5B Recall/F1 on 4 CheXbert labels, n=980 Providence images | Providence in-house CXR | ⭐ — single internal cohort, 4 of 13 labels, **Cardiomegaly INVERTED (CLIP 36.24 > BiomedCLIP 6.99)** |
| C11 | New SOTA on VQA-RAD and SLAKE | Fig. 4A bar charts | VQA-RAD, SLAKE | ⭐⭐ — true vs legacy MAML/PubMedCLIP, but **LLaVA-Med (which uses BiomedCLIP as its vision tower) and BiomedGPT-B already score higher overall**. The "SOTA" framing is outdated. |

**Honest read.** Four claims (C1, C2, C6, C7, C9) are rock-solid: counts, retrieval gaps, and clean A/B ablations carry them. The headline narrative — "diverse data beats specialty data" (C4, C5) — rests on **a single specialty benchmark (RSNA) against a single specialty baseline (BioViL)**, with no replication on CheXpert / NIH ChestX-ray14 / Padchest, no variance reporting, and no multi-seed runs anywhere in the paper. The privacy-proxy section (C10) is interesting framing but **Cardiomegaly retrieval is actually inverted — CLIP 36.24% Recall@1 beats BiomedCLIP 6.99%** — and only 4 of 13 CheXbert labels are analyzed on a single 980-image internal cohort. The VQA "new SOTA" claim (C11) is dated: LLaVA-Med (itself built on BiomedCLIP) and BiomedGPT-B exceed it on overall accuracy. And for the pathology vertical specifically, **CONCH** (CoCa loss, 1.17M curated pathology pairs, in-house iBOT vision warm-start, 448-px input) reclaimed the SOTA shortly after — illustrating that "broad scale beats curation" is regime-dependent.

## Method & Architecture

![BiomedCLIP overview: PMC-15M pipeline plus contrastive image/text encoder](/assets/images/paper/biomedclip/page_006.png)
*Figure 1: BiomedCLIP system overview — PMC-15M is built from 4.4M PubMed Central full-text articles; ViT-B/16 + PubMedBERT are trained contrastively and applied to retrieval, classification, and VQA downstream tasks.*

### 1. PMC-15M dataset construction

Every public PMC OA article in the June 2022 snapshot is downloaded, XML-parsed for figure-caption pairs, and de-duplicated by PMID/PMCID. The pipeline runs on Azure Databricks. Yield: **15,282,336 image-caption pairs from >3M distinct articles**, split into 13.9M train / 13.6k dev / 725.7k test. A separate PMC-Fine-Grained-46M pipeline OCR-splits composite figures into sub-panels but is *not* used for BiomedCLIP pretraining — only for image-type frequency analysis.

### 2. Contrastive objective

Standard symmetric InfoNCE on N pairs per batch:

$$
\mathcal{L} = -\frac{1}{2N}\left( \sum_{i=1}^{N}\log\frac{e^{\cos(I_i, T_i)/\tau} }{\sum_{j=1}^{N}e^{\cos(I_i, T_j)/\tau} } + \sum_{i=1}^{N}\log\frac{e^{\cos(I_i, T_i)/\tau} }{\sum_{j=1}^{N}e^{\cos(I_j, T_i)/\tau} } \right)
$$

with learnable temperature tau and linear-projected encoder outputs $I_i, T_i$. Sharded contrastive loss (Cherti et al. 2022) handles memory; built on OpenCLIP 2.x.

### 3. Text-side adaptation (the highest-impact swap)

Replace GPT-2 with PubMedBERT, BPE -> WordPiece (30k domain vocab), and bump max context **77 -> 256 tokens**. Suppl. Table 1 ablation on PMC-15M val:

| Config | Val loss | img->txt R@1 |
|---|---|---|
| GPT-2 / BPE / 77 | 0.6626 | 64.53 |
| PubMedBERT / WP / 77 | 0.5776 | 69.03 |
| **PubMedBERT / WP / 256** | **0.4807** | **73.50** |

### 4. Image-side adaptation

ViT-B/16 beats ViT-S/16 and ViT-M/16 (R@1 69.45 / 71.85 / 73.50 in Suppl. Table 2). ImageNet warm-start essentially ties random init (82.90 vs 83.15) but is chosen for downstream stability. **Resolution stays at 224**: 384 cuts val loss further (0.3819 -> 0.3406) but degrades zero-shot mean accuracy 75.52% -> 70.37% — likely because PCam (96-px native) suffers upsampling artifacts.

### 5. Batch schedule

A 4k->64k progressive batch reaches better val R@1 (87.32 vs 83.98), but downstream gain plateaus past 4k. Final: **constant batch 4k for 40 epochs**, AdamW (beta1=0.9, beta2=0.98), peak LR 5e-4, weight decay 0.2, cosine schedule, 2000 warmup steps, 16x A100 (or V100) with bf16 AMP + gradient checkpointing.

## Experimental Results

### Cross-modal retrieval on PMC-15M held-out test (n = 725,739)

| Method | txt->img R@1 | txt->img R@5 | txt->img R@10 | img->txt R@1 | img->txt R@5 | img->txt R@10 |
|---|---|---|---|---|---|---|
| OpenAI CLIP | 1.00 | 2.51 | 3.59 | 0.79 | 2.13 | 3.08 |
| PubMedCLIP | 8.48 | 16.20 | 20.10 | 7.91 | 15.50 | 19.50 |
| BiomedCLIP (GPT-2) | 59.60 | 80.40 | 85.70 | 60.00 | 80.10 | 85.30 |
| **BiomedCLIP (PubMedBERT)** | **69.60** | **86.30** | **90.20** | **70.10** | **86.40** | **90.20** |

PubMedCLIP being only marginally better than CLIP — and worse than BiomedCLIP with GPT-2 — is the main empirical evidence that **data scale dominates architectural choice** for biomedical CLIP variants.

### Zero-shot classification (BiomedCLIP-224 row from Suppl. Table 5)

| Dataset | BiomedCLIP zero-shot acc |
|---|---|
| PCam | 73.41 |
| LC25000-Lung | 65.23 |
| LC25000-Colon | 92.98 |
| TCGA-TIL | 67.04 |
| RSNA pneumonia | 78.95 |
| **Mean** | **75.52** |

PLIP is competitive on pathology benchmarks (except PCam, where it under-performs because OpenPath Twitter data has few lymph-node tiles); PubMedCLIP only beats CLIP on RSNA. No prompt-ensembling variance is reported, and the Suppl. Table 10 templates are minimal — a known sensitivity unaddressed in the paper.

### Few-/full-shot linear probing on RSNA pneumonia

BiomedCLIP at **10% labels** matches fully-supervised BioViL at 100% labels, and at 100% labels exceeds both BioViL and GLoRIA. This is the load-bearing evidence for the "positive transfer from diverse data" claim — and it stands or falls on a single dataset.

### Medical VQA (accuracy: Open / Closed / Overall)

| Method | VQA-RAD Open | Closed | Overall | SLAKE Open | Closed | Overall |
|---|---|---|---|---|---|---|
| MAML | 56.00 | 77.90 | 69.20 | 76.80 | 80.60 | 78.30 |
| CLIP | 59.90 | 79.40 | 71.30 | 78.60 | 81.00 | 79.50 |
| PubMedCLIP | 60.10 | 80.00 | 72.10 | 78.40 | 82.50 | 80.10 |
| **BiomedCLIP** | **67.00** | 76.50 | 72.70 | **84.30** | **88.90** | **86.10** |
| BiomedGPT-B | 60.90 | 81.30 | 73.20 | 84.30 | 89.90 | 86.50 |
| Med-PaLM M (562B) | - | - | - | - | - | 87.00 |
| LLaVA-Med (uses BiomedCLIP) | 64.75 | 83.09 | **75.80** | 87.11 | 86.78 | **87.00** |

LLaVA-Med — which uses BiomedCLIP as its frozen vision tower — exceeds BiomedCLIP on overall accuracy on both benchmarks. That is downstream evidence of transferability rather than competition, but the paper's abstract phrasing of "new state-of-the-art" predates LLaVA-Med and BiomedGPT-B and is now outdated.

### Privacy-preserving proxy retrieval (Providence in-house CXR, n=980)

![Fig. 5B: CheXbert-label agreement between Providence CXR and PMC-15M retrieved images](/assets/images/paper/biomedclip/page_019.png)
*Figure 5B: Recall@1 and F1@1 of CheXbert-label agreement between Providence proprietary CXR and the top-1 retrieved PMC-15M image. BiomedCLIP dominates on Lung Opacity, Atelectasis, and Pleural Effusion but **loses to CLIP on Cardiomegaly (6.99 vs 36.24)** — captions in PMC don't mention "cardiomegaly" the way structured radiology reports do.*

| Label | BiomedCLIP Recall | PLIP Recall | CLIP Recall |
|---|---|---|---|
| Lung Opacity | **88.80** | 79.10 | 22.76 |
| Atelectasis | **95.04** | 84.40 | 51.77 |
| Pleural Effusion | **25.35** | 8.45 | 0.00 |
| Cardiomegaly | 6.99 | 0.44 | **36.24** |

## Limitations

**Authors admit.**
- ~50% of PMC figures are composite multi-panel — a single embedding represents many sub-images. PMC-Fine-Grained-46M splits them but is not used for BiomedCLIP pretraining itself.
- In-line text references to figures are not incorporated.
- Image encoder capped at ViT-B; resolution capped at 224 (336 flagged as future work even though 75% of PMC images are >336 native).
- Methodology not yet extended to gene-expression or sequence modalities.

**Unaddressed.**
- **No variance reporting anywhere** — no error bars, no multi-seed, no significance tests on Fig. 2/3/4.
- **No out-of-distribution radiology evaluation** (CheXpert, NIH ChestX-ray14, Padchest) — the entire "positive transfer" pillar rests on RSNA alone vs one baseline (BioViL).
- **No pretraining-compute report** (16x A100, but no GPU-hours), limiting reproducibility.
- **No data-quality ablation** — what fraction of the 15M is informative biomedical content vs chart/flowchart/table noise? Given the long tail in Fig. 1B, this matters.
- **No prompt-ensembling variance** — Suppl. Table 10 templates are minimal and zero-shot accuracy is known to be template-sensitive.
- **For pathology specifically, CONCH (1.17M curated pairs, CoCa loss, 448-px, in-house iBOT warm-start) is the better choice today** — the "diverse data beats specialty" framing applies to the radiology surprise on RSNA, not to pathology where careful curation reclaimed SOTA.

## Why It Matters for Medical AI

BiomedCLIP's lasting contribution is **the dataset, not the architecture**: PMC-15M became the public foundation that subsequent work (LLaVA-Med, BiomedGPT, downstream pathology and radiology models) builds on. It also provided the first empirical refutation of the assumption that radiology specialists must train on radiology data — at least under the loose "match BioViL on RSNA" bar. For practitioners the practical implications are:

- Use BiomedCLIP as a **broad biomedical embedder** for retrieval, zero-shot screening, and as a vision tower for VQA / multimodal LLMs (LLaVA-Med does exactly this).
- For **pathology-specific tasks**, prefer CONCH or PathChat; BiomedCLIP is generalist, not specialist.
- For **CXR-specific tasks with structured reports**, the Cardiomegaly inversion is a cautionary tale — domain captions matter, and PMC's loose figure captions cannot substitute for clinical-report text on every condition.
- For **proprietary clinical data**, the "retrieve a public proxy" idea is interesting but the evidence is too thin (n=980, 4 of 13 labels, one label inverted) to deploy as-is.

## References

- Paper: Zhang et al., *BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs*, arXiv:2303.00915v3, 8 Jan 2025.
- Release: [aka.ms/biomedclip](https://aka.ms/biomedclip)
- Related: OpenAI CLIP (Radford et al. 2021), PubMedBERT (Gu et al. 2021), PubMedCLIP (Eslami et al. 2023), BioViL (Boecking et al. 2022), GLoRIA (Huang et al. 2021), MedCLIP (Wang et al. 2022), PLIP (Huang et al. 2023), CONCH (Lu et al. 2024), LLaVA-Med (Li et al. 2023), BiomedGPT (Zhang et al. 2024), OpenCLIP (Cherti et al. 2022).
