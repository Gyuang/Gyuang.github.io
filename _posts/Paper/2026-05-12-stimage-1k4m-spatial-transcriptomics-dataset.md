---
title: "STimage-1K4M: A Histopathology Image-Gene Expression Dataset for Spatial Transcriptomics"
excerpt: "1,149 ST slides and 4,293,195 sub-tile / gene-expression pairs across 10 species and 50 tissues — the broadest paired ST+H&E catalog to date, with caveats."
categories:
  - Paper
tags:
  - STimage-1K4M
  - Spatial Transcriptomics
  - Computational Pathology
  - Dataset
  - Contrastive Learning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- Existing pathology image-text datasets (ARCH, OpenPath, Quilt-1M) caption an entire slide with one short sentence, which is blind to sub-tile heterogeneity. STimage-1K4M replaces text with **whole-transcriptome gene expression per sub-tile**, aggregating 1,149 spatial-transcriptomics (ST) slides from ST v1, Visium, and Visium HD.
- **Headline scale: 4,293,195 sub-tile / gene-expression pairs (15k-30k dims per spot) across 10 species, 50 tissues, with pathologist annotations for 71 slides over 9 studies.**
- Proof-of-concept fine-tuning of CLIP (ViT-B/32) and PLIP (ViT-L/14) on 12 DLPFC slides beats their own zero-shot baselines on macro-F1, but **still trails zero-shot UNI** — a caveat the headline phrasing in the Discussion section glosses over.

## Motivation
Multimodal pathology datasets (OpenPath, Quilt-1M, ARCH) anchor each H&E slide to a single Twitter/YouTube/textbook caption — fine for slide-level concepts ("cancer slide") but useless for sub-tile heterogeneity (tumor vs. stroma vs. lymphoid). Meanwhile, sequencing-based ST already delivers whole-transcriptome readouts at spot-level resolution (100 µm for ST v1, 55 µm for Visium, 8 µm bins for Visium HD) bundled with H&E images.

The authors' argument is direct: gene expression is the right "text" for sub-tile pathology — ~20,000 quantitative dimensions per spot beats one ~50-token caption per slide. The bottleneck is curation, not capability. Prior ST databases (SpatialDB, STOmicsDB, SODB, SPASCER, SOAR) catalog expression data but **do not ship aligned histopathology images** in a uniform schema. STimage-1K4M is positioned as the missing paired-modality resource.

## Core Innovation
- **Unified paired-modality curation.** GEO trawl (856 datasets from 121 studies) + 62 datasets from 10x Genomics + 233 manually scraped slides from 10 publications, filtered to sequencing-based ST that ships with H&E images.
- **Per-spot output schema.** Each spot stores a cropped sub-tile image, center (x, y) coordinate, spot radius, and a 15k-30k-dim gene-expression vector. Per dataset, the abstract / title / keywords are kept as paper-level metadata.
- **ST v1 coordinate recovery.** ST v1 typically needs CytAssist images to map spot coordinates to the image — rarely public. Only datasets with both mapped and unmapped coordinates are kept, with radii recomputed via the SpatialTranscriptomicsResearch pipeline.
- **Manual verification at scale.** Every dataset is hand-checked for coordinate-image alignment.
- **Pathologist labels.** 71 slides across 9 studies receive spot-level domain annotations (DLPFC layers, breast tissue regions, prostate, kidney, mouse brain).

![STimage-1K4M overview](/assets/images/paper/stimage-1k4m/page_004.png)
*Figure 1: STimage-1K4M overview — (a) sources span GEO, Spatial Research, 10x Genomics, and publications; (b) sequencing-based ST technologies (ST v1, Visium, Visium HD) and their spot/bin geometry; (c) 1,149-slide / 4.3 M-spot composition across technology, species, and 50 tissues.*

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | 1,149 slides, 4,293,195 sub-tile / gene-expression pairs, 3 ST techs, 10 species, 50 tissues. | Section 3 + Figure 1c (151+994+4=1,149 slides; 60,145+2,336,306+1,896,744=4,293,195 spots). | ⭐⭐⭐ Arithmetically consistent; reproducible from the GitHub manifest. |
| C2 | The most exhaustive paired ST+H&E collection to date. | Related-work prose listing SpatialDB, STOmicsDB, SODB, SPASCER, SOAR, etc. | ⭐⭐ Plausible — none of these ship uniform paired images — but no side-by-side count table, and the concurrent HEST-1k reports 1,229 paired slides. |
| C3 | Sub-tile gene expression is richer supervision than slide-level captions. | Section 1 prose only. | ⭐ No head-to-head experiment vs. PLIP/CONCH/Quilt-Net. |
| C4 | Fine-tuning CLIP/PLIP on STimage-1K4M improves macro-F1 over zero-shot CLIP/PLIP. | Figure 3a, 5-seed linear probing on 12 DLPFC slides; HVG vs. overlap-HVG ablation. | ⭐⭐ Within-encoder comparison with multi-seed bars, but only one tissue (DLPFC); no numerical table, only a figure. |
| C5 | "Pre-training using STimage-1K4M ... outperforms larger state-of-the-art models such as CLIP and PLIP" (Section 6). | Figure 3a, DLPFC only. | ⭐ **Overstated.** True only against *zero-shot* CLIP/PLIP. Fine-tuned variants do not beat zero-shot UNI in the same figure, and the Limitations section concedes it. |
| C6 | Gene-conditioned embeddings separate WM from cortical layers better than zero-shot image-only embeddings. | Figure 3b boxplots (Silhouette / CH / DB across 12 DLPFC slides) + Figure 3d t-SNE on slide 151675. | ⭐⭐ Three metrics across 12 slides + qualitative t-SNE, but no significance test. |
| C7 | HVG beats overlap-HVG as a gene-selection strategy. | Figure 3a bars. | ⭐⭐ Differences are visually small; overlap-HVG has lower variance and is the more principled cross-slide choice. |
| C8 | The dataset enables gene-expression prediction, super-resolution, deconvolution, nuclei segmentation. | Figure 2 schematic; no demonstrations for super-resolution, deconvolution, or nuclei segmentation. | ⭐ Schematic-level claim, not experimental evidence. |
| C9 | Cross-tissue fine-tuning fails due to batch effects. | Limitations prose without numbers. | ⭐ Anecdotal; no log or table provided. |
| C10 | Gene expression can serve as positional encoding for vision transformers. | Speculative Section 6 discussion. | ⭐ Hypothesis, not evidence. |

**Honest summary.** C1 (the dataset counts) is solid. C6 is the most defensible *modeling* claim. The headline "outperforms larger state-of-the-art models" (C5) collapses to "fine-tuned ViT-B/32 and ViT-L/14 beat their own zero-shot baselines on one DLPFC cohort, while still trailing zero-shot UNI." Read the dataset contribution at face value, but discount the modeling section.

## Method & Architecture

![Tasks enabled by STimage-1K4M](/assets/images/paper/stimage-1k4m/page_006.png)
*Figure 2: Four canonical tasks STimage-1K4M targets — gene-expression super-resolution, image-gene contrastive representation learning, image-driven spatial clustering, and deconvolution + nuclei segmentation.*

### Proof-of-concept training recipe
- **Backbones.** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) and PLIP ViT-L/14 (`vinid/plip`), fp32, following CLIP repo issue #83.
- **Architecture.** Keep the image encoder; replace the text encoder with a **single FC layer that compresses gene expression into a 32-dim latent**. The image encoder is also projected to 32-dim via one FC layer. InfoNCE-style contrastive loss over the 32-dim embeddings.
- **Training data.** Only the 12 DLPFC slides from Maynard et al. 2021 (47,681 spots, human brain) — chosen to dodge cross-species gene-name mismatch and inter-study batch effects.
- **Gene-set strategies.**
  - **HVG:** Per slide, scanpy `pp.highly_variable_genes` selects the top-128 HVGs; HVGs are concatenated across slides by rank, **ignoring gene names** (so column $i$ for slide A and slide B may correspond to different genes).
  - **Overlap-HVG:** Intersect gene names across all slides first, then run `experimental.pp.highly_variable_genes` on the merged matrix → top-100 HVGs.
- **Optimization.** 15 epochs, single A100, hyperparameters inherited from CLIP defaults.
- **Linear probing.** sklearn `SGDClassifier` on 32-dim fine-tuned embeddings or 512-dim zero-shot embeddings; 80/10/10 split; L2 with α ∈ {1, 0.1, 0.01, 0.001, 0.0001}; 5 seeds; macro-F1 selection on validation.

## Dataset H2 — composition and HEST-1k comparison

### Headline counts

| Property | Value |
|---|---|
| Whole-slide images (slides) | **1,149** |
| Sub-tile / gene-expression pairs (spots) | **4,293,195** |
| ST technologies | Spatial Transcriptomics (ST v1), Visium, Visium HD |
| Genes per spot | 15,000-30,000 |
| Species | 10 (predominantly human + mouse) |
| Tissues | 50 |
| Pathologist-annotated slides | 71 (across 9 studies) |
| Cancer-related slides | 456 (39.7%) |
| License | "Permissible license for research-based use" via Google Form; code MIT |
| Distribution | FTP server; first author maintains |

### Composition breakdowns (Figure 1c)
- **By slide (technology):** ST v1 13.1% (151), Visium 86.5% (994), Visium HD 0.3% (4).
- **By spot (technology):** ST v1 1.4% (60,145), Visium 54.4% (2,336,306), Visium HD 44.2% (1,896,744). Visium HD dominates spot counts despite being only 4 slides — its 8 µm bins yield median 469,728 spots/slide vs. 2,261 (Visium) and 359 (ST v1).
- **By species:** human 673, human + mouse 9, mouse 418, other 49.
- **By tissue:** brain 251 (21.8%), breast 205 (17.8%), then heart, kidney, liver, ovary, pancreas, skin, spleen with single- to low-double-digit counts.
- **Median genes per slide:** ST v1 16,075 / Visium 32,285 / Visium HD 18,572.

### STimage-1K4M vs. HEST-1k (concurrent NeurIPS 2024 D&B paper)

| Axis | STimage-1K4M | HEST-1k |
|---|---|---|
| **Paired ST+H&E samples** | **1,149 slides** | 1,229 slides |
| Sub-tile / patch pairs | **4,293,195 spots** (raw ST spots, full transcriptome 15-30k genes) | 2.1M patches (224×224 px @ 20×, top-50 HVG targets per task) |
| ST technologies | ST v1, Visium, Visium HD | ST v1, Visium, Visium HD, **Xenium** |
| Species | 10 (mostly human + mouse) | 2 (human + mouse) |
| Tissues / organs | **50 tissues** | 26 organs |
| Cancer samples | 456 slides (39.7%) | 367 samples / 25 OncoTree subtypes |
| Pathologist annotation | 71 slides / 9 studies (spot-level) | per-sample OncoTree + sample type + 76.4M CellViT nuclei (instance-level) |
| Image preprocessing | center coordinate + radius per spot, raw whole slide | YOLOv8 fiducial re-alignment, DeepLabV3 tissue mask, 224×224 @ 20× patches, CellViT nuclei |
| Toolkit | Scripts on GitHub | **HEST-Library** Python package + HEST-Benchmark (9 tasks, 11 encoders) |
| **License** | "Permissible license for research-based use" (Google Form request) | **CC BY-NC-SA 4.0** (HuggingFace Datasets) |
| Distribution | FTP | HuggingFace ~1 TB |
| Curation policy | Breadth-first; manual coordinate-image verification; keeps raw whole transcriptome | Benchmark-first; aggressive QC (drop µm/px > 1.15), realignment, pseudo-Visium Xenium patches; HVG-only at benchmark time |
| Complementary strength | **Higher spot count, more species, more tissues, raw transcriptome retained** | **Standardized patches, ready-to-use benchmark, includes Xenium, mature toolkit** |

The two datasets are genuinely complementary: STimage-1K4M is the broader catalog (more species, tissues, raw expression dim retained), HEST-1k is the benchmark-engineered subset with stricter image QC, Xenium coverage, and a published 9-task evaluation.

**License caveat.** STimage-1K4M's "permissible license for research-based use" — distributed via a Google Form (https://forms.gle/3Waa4FQnqpK8UGSY7) — is opaque relative to HEST-1k's clean **CC BY-NC-SA 4.0** on HuggingFace. For anyone planning to redistribute a derived model or dataset, HEST-1k's terms are far less ambiguous, and STimage-1K4M users may need to negotiate per-source-dataset rights.

![Pathologist annotation coverage](/assets/images/paper/stimage-1k4m/page_023.png)
*Figure 4: Spot-level pathologist annotations span 71 slides / 9 studies across human brain, breast, prostate, kidney, and mouse brain — the labeled subset for clustering benchmarks.*

## Experimental Results

The paper's only experiments are the DLPFC linear-probing demo (Figure 3). This is a dataset paper, not a model paper.

![DLPFC proof-of-concept results](/assets/images/paper/stimage-1k4m/page_007.png)
*Figure 3: DLPFC proof-of-concept — (a) macro-F1 across encoders; (b) cluster-quality boxplots; (c) pathologist-annotated brain reference; (d) t-SNE panels per model.*

### Brain-layer (L1-L6, WM) linear-probing macro F1

No numerical table is provided in the paper. Visual estimates from Figure 3a:

| Encoder | Macro F1 (visual estimate from Figure 3a) |
|---|---|
| CLIP (zero-shot) | ≈0.35 |
| **fine-tune CLIP HVG** | **≈0.42 (highest among CLIP/PLIP variants)** |
| **fine-tune CLIP overlap-HVG** | **≈0.40** |
| PLIP (zero-shot) | ≈0.40 |
| **fine-tune PLIP HVG** | **≈0.42** |
| **fine-tune PLIP overlap-HVG** | **≈0.40** |
| UNI (zero-shot) | ≈0.45 (visibly the tallest bar) |

The authors' verbal summary: fine-tuned CLIP and PLIP with HVG beat their zero-shot counterparts, but **none beat zero-shot UNI**, which they attribute to UNI being a far larger / better-pretrained foundation model they could not afford to fine-tune. In other words: the "outperforms CLIP/PLIP" framing in the Discussion is only true against **zero-shot** CLIP/PLIP; a fine-tuned UNI would almost certainly still win.

### Cluster quality (Figure 3b)
Silhouette / Calinski-Harabasz / Davies-Bouldin against pathologist layers, per slide, over the 12 DLPFC slides. Fine-tuned CLIP/PLIP variants achieve higher Silhouette and CH and lower DB than zero-shot CLIP/PLIP on most slides; UNI is competitive or better on Silhouette/CH but the fine-tuned variants visibly separate WM from L1-L6 more cleanly. No numerical table; spread across the 12 slides is wide.

### t-SNE qualitative (Figure 3c-d)
DLPFC sample 151675: fine-tuned CLIP/PLIP embeddings produce more compact, layer-coherent clusters than the zero-shot variants. UNI also produces visibly clean clusters.

## Limitations

### Authors' admitted limitations
- Fine-tuned CLIP/PLIP still underperform zero-shot UNI — no compute to fine-tune UNI.
- Gene-expression encoder is a single FC layer projecting up to 128 HVGs into 32 dims — explicitly described as "simplistic".
- Cross-tissue fine-tuning failed; batch effects dominate.
- Brain + breast over-representation (39.6% combined) biases full-set training.

### Unaddressed gaps
- **No benchmark protocol.** Unlike HEST-Benchmark's 9 tasks × 11 encoders × patient-stratified k-fold, STimage-1K4M ships no canonical evaluation. Each downstream user invents a protocol.
- **Opaque license.** "Permissible license for research-based use" via a Google Form — no SPDX identifier, no redistribution clause. HEST-1k's CC BY-NC-SA 4.0 is far more downstream-friendly.
- **No FTP URL in the paper.** Distribution depends on first-author maintenance of an unspecified FTP server.
- **No per-gene QC.** Median gene counts are reported but UMI thresholds, mitochondrial filtering, and doublet handling are left to the user.
- **No Xenium coverage.** Imaging-based ST (Xenium, MERFISH, STARmap, CosMx) is excluded by design — a deliberate scope choice but a real ceiling that HEST-1k partially closes.
- **No statistical significance tests.** 5 seeds × 3 cluster metrics × 12 slides could easily support Wilcoxon signed-rank tests; none are reported.
- **Ablations missing.** Latent dim is 32 only (no 64/128/256); HVG count is 128 only; only InfoNCE; no ResNet/ViT-L/DINO/MAE comparison.
- **No retrieval/zero-shot demonstrations.** A CLIP-style paired dataset should at minimum show spot→image and image→spot retrieval.
- **No batch-effect correction baselines** (Harmony, ComBat, scVI) — only the observation that batch effects hurt.
- **Coarse pathologist annotations.** 71 slides / 9 studies, layer-level for brain and region-level for breast/prostate/kidney — fine for cluster benchmarking, insufficient for cell-type or fine pathology classification.

## Why It Matters for Medical AI
For computational-pathology researchers who want to move past slide-level captions, STimage-1K4M is the broadest paired ST + H&E catalog publicly available — wider species and tissue coverage than HEST-1k, with raw whole-transcriptome vectors retained per spot. It is the right *training-corpus* choice when scale and diversity dominate over benchmark cleanliness; HEST-1k is the right *evaluation* choice when you need a standardized, license-clean protocol with Xenium coverage. Most serious teams will end up using both: STimage-1K4M to pretrain image-gene contrastive encoders, HEST-Benchmark to evaluate them under controlled conditions. The license ambiguity is the real friction point — model releases trained on STimage-1K4M will need careful per-dataset audit before redistribution.

## References
- Paper (arXiv): https://arxiv.org/abs/2406.06393
- Code: https://github.com/JiawenChenn/STimage-1K4M (MIT license for code)
- Data access form: https://forms.gle/3Waa4FQnqpK8UGSY7
- HEST-1k (concurrent NeurIPS 2024 D&B): Jaume et al., "HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis"
- DLPFC source cohort: Maynard et al., "Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex," Nature Neuroscience 2021
- Related foundation encoders: CLIP (Radford et al. 2021), PLIP (Huang et al. 2023), UNI (Chen et al. 2024)
