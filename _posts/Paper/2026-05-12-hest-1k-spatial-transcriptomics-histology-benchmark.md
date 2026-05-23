---
title: "HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis"
excerpt: "1,229 paired ST+WSI samples across 153 cohorts, plus a 9-task gene-expression benchmark where H-Optimus-0 edges UNIv1.5 by just 0.0056 Pearson (well inside per-fold sigma)."
categories:
  - Paper
  - Dataset
  - Spatial-Transcriptomics
permalink: /paper/hest-1k-spatial-transcriptomics-histology-benchmark/
tags:
  - HEST-1k
  - Spatial-Transcriptomics
  - Computational-Pathology
  - Foundation-Models
  - Benchmark
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- HEST-1k unifies **1,229 paired spatial-transcriptomics (ST) + H&E whole-slide images** from 153 cohorts (26 organs, 2 species, 367 cancer samples / 25 OncoTree subtypes), producing **2.1M expression-morphology patch pairs** and **76.4M CellViT-classified nuclei** under a single CC BY-NC-SA 4.0 release.
- The release ships three things at once: the dataset, a Python toolkit (HEST-Library) that re-aligns legacy Visium/STv1/Xenium data via YOLOv8 fiducial detection and DeepLabV3 tissue segmentation, and a 9-task gene-expression-prediction benchmark (HEST-Benchmark) on the top-50 most variable genes per task.
- Headline benchmark number: **H-Optimus-0 mean Pearson 0.4146 vs. UNIv1.5 0.4090** across 9 tasks (PCA+Ridge, patient-stratified k-fold). The lead is **0.0056 absolute** with per-fold sigma of 0.04-0.09 and **no significance test reported** — treat the leaderboard as a tier list, not a ranking. A model-scaling correlation of **R=0.81, p<0.01** holds across 11 foundation models; the data-scaling correlation is **R=0.48, p=0.13** (explicitly not significant).

## Motivation

Spatial transcriptomics finally lets us read gene expression with sub-millimeter spatial coordinates, but every public ST cohort uses a different file format, alignment convention, and image resolution. The result: deep-learning studies on paired ST+H&E have been confined to a handful of patients per paper. Meanwhile, computational pathology has been flooded with foundation models (UNI, CONCH, GigaPath, Virchow, H-Optimus-0, UNIv1.5) whose Camelyon/Gleason numbers are saturating and increasingly hard to discriminate.

HEST-1k attacks both problems with one release: a unified ST+WSI corpus large enough to support multimodal research, plus a hard new task — predicting the top-50 most variable genes from a 112-µm H&E patch — that re-stratifies the foundation-model zoo on a clinically interesting morpho-molecular target.

## Core Innovation

- **A reproducible legacy-format ingestion pipeline.** Convert any Visium / Visium HD / Xenium / STv1 cohort to pyramidal TIFF + `AnnData`, fine-tune YOLOv8 on 119 hand-annotated fiducial regions to auto-align Visium spots, re-estimate pixel resolution from inter-spot distance, and reject anything coarser than 1.15 µm/px.
- **HEST-Library** wraps the pipeline as a Python package with one-call dataset download, metadata filters, tissue patching, CellViT-based nuclear segmentation, and hooks for batch-effect mitigation (ComBat / Harmony / MNN).
- **HEST-Benchmark.** 9 patient-stratified k-fold regression tasks (IDC, PRAD, PAAD, SKCM, COAD, READ, ccRCC, LUAD, LYMPH-IDC). Target = log1p-normalized top-50 highly variable genes per task; input = 224x224 px at 20x (~112 µm); regressor = PCA(256) + Ridge.
- **Honest scaling-law reporting.** The model-size correlation (R=0.81, p<0.01) is real; the data-size correlation (R=0.48, p=0.13) is explicitly flagged as not significant — a rare moment of restraint in foundation-model papers.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | HEST-1k unifies 1,229 paired ST-WSI samples across 153 cohorts, 26 organs, 2 species, 367 cancer samples / 25 subtypes, yielding 2.1M patch-expression pairs and 76.4M classified nuclei. | Appendix Tables A1-A10; Figure 1.a counts; Section 3.2-3.3 aggregate statistics. | All HEST-1k sources. | ⭐⭐⭐ |
| C2 | HEST-Library reproducibly handles legacy ST formats and auto-aligns Visium spots via a YOLOv8 fiducial detector. | Figure 5 qualitative example (6.5x6.5 mm and 11x11 mm Visium); 119 hand-annotated fiducial regions. | Visium subset of HEST-1k. | ⭐⭐ |
| C3 | Larger pathology foundation models do better on HEST-Benchmark (model scaling law). | Figure 2.a: R=0.81, p<0.01 across 11 encoders. | 9-task HEST-Benchmark. | ⭐⭐ |
| C4 | Data scaling (number of pretraining patches) yields only weak gains. | Figure 2.b: R=0.48, p=0.13. | Same 11 encoders. | ⭐⭐ |
| C5 | **H-Optimus-0 is the best foundation model for gene-expression prediction from histology.** | **Table 1 avg 0.4146 vs. UNIv1.5 0.4090; Table A14 XGBoost similarly.** | **HEST-Benchmark (9 tasks, 2-24 patients each).** | **⭐ (gap of 0.0056 absolute is ~1/10 of per-fold sigma; no significance test, no per-task significance, very small per-task patient counts)** |
| C6 | Domain-specific pretraining matters: ResNet50 (IN) is the worst encoder. | Table 1; ~9 absolute points behind the best. | HEST-Benchmark. | ⭐⭐⭐ |
| C7 | Performance saturates / varies by task — some cancers are harder to predict expression for. | Per-task spread: H-Optimus-0 SKCM 0.6432 vs. READ 0.2292. | HEST-Benchmark. | ⭐⭐ |
| C8 | Nuclear size in neoplastic cells correlates with cancer-relevant gene expression (GATA3, FOXA1, FLNB, TPD52). | Figure 3.e (R=0.47, p<10^-4 GATA3); Appendix Figure 6 (FLNB R=0.45, TPD52 R=0.47, FOXA1 R=0.47). | Two IDC Xenium samples. | ⭐ |
| C9 | Expression-guided fine-tuning of CONCH (CONCH-FT) yields a better breast-cancer encoder than CONCH. | Table 2: ER/PR/HER2 AUC +0.003 / +0.008 / +0.009 on BCNB n=1,058. | 5 Xenium breast cases (train) -> BCNB (test). | ⭐ |
| C10 | HEST-Benchmark is diverse and challenging vs. saturated benchmarks like Camelyon. | Average Pearson 0.22-0.40 leaves headroom. | HEST-Benchmark. | ⭐⭐ |

**Honest read.** The dataset and tiering claims (C1, C2, C6, C10) are well-supported and stand independently — HEST-1k is exactly what it advertises and per-source counts are independently checkable. The **headline ranking claim (C5) is the weakest part of the paper**: H-Optimus-0's 0.0056 absolute lead over UNIv1.5 is far inside the typical per-fold sigma (0.04-0.09 across tasks), and the paper reports no significance test on the average, no paired per-task win/loss test, and no bootstrap intervals on the means. With 5 of the 9 tasks relying on 2-3 patients, per-fold variance dominates the signal. The right way to read Table 1 is as a coarse three-tier list — legacy ImageNet < legacy domain-specific < modern DINOv2-scale models — and the within-tier ordering should be treated as undetermined. The biomarker claims (C8) are exploratory single-slide observations with no multiple-testing correction; CONCH-FT (C9) is a feasibility demo on n=5 patients.

## Method & Architecture

![HEST-1k overview: 1,229 paired ST+WSI samples, 26 organs, and three downstream applications](/assets/images/paper/2406.16192_HEST-1k/page_002.png)
*Figure 1: HEST-1k overview — 1,229 paired ST+WSI samples across 26 organs, plus three downstream applications (HEST-Benchmark, biomarker exploration, multimodal representation learning).*

The pipeline has eight stages, all wrapped in the HEST-Library Python package.

1. **Source scraping.** Pull paired ST+WSI cohorts from 10x Genomics, Mendeley, Spatial-Research, Zenodo, NCBI, GitHub, Human Cell Atlas, BioStudies, HTAN, plus 3 internal cohorts.
2. **Format unification.** Every WSI (OME.TIFF, JPG, BigTIFF, ...) becomes a pyramidal generic TIFF compatible with OpenSlide/QuPath; every expression matrix (CSV, MEX, h5, ...) becomes an `AnnData` object.
3. **Tissue segmentation.** A DeepLabV3 + ResNet50 head, fine-tuned on hand-annotated regions covering pen marks, fiducials, multiple stains, and artifacts.
4. **Spot-image realignment.** For Visium, 119 hand-annotated fiducial regions train a YOLOv8 (COCO-pretrained) detector for the four corner fiducials; spot coordinates are derived if at least 3 of 4 corners are found. STv1 cohorts fall back to provided spot files; Xenium uses VALIS to register the DAPI image (intrinsically aligned with transcripts) to H&E.
5. **Resolution re-estimation.** Pixel size in µm/px is computed from inter-spot pixel distance vs. known physical spacing, cross-checked against self-reported magnification; anything coarser than 1.15 µm/px is dropped.
6. **Patching.** 224x224-px patches at 20x (~112x112 µm) centered on each spot. Xenium transcripts are pooled into pseudo-Visium 55x55-µm patches without spacing. Output: **2.1M patch-expression pairs.**
7. **Nuclear segmentation.** CellViT (PanNuke-pretrained) yields instance masks plus a 5-class classification — neoplastic / non-neoplastic epithelial / inflammatory / stromal / necrotic. Aggregate: **76.4M nuclei** (17.6M neoplastic, 21.5M stromal, 4.9M normal epithelial, 15.4M inflammatory, 76k necrotic; mean 62.1k per slide).
8. **Metadata.** Per-sample: license/source/year, species, OncoTree code + organ, sample type, ST technology, gene count, spot count, reads, spot size & spacing, image resolution, magnification, FFPE vs. frozen.

### Visium spot auto-alignment

The hardest engineering step is recovering spot coordinates without trusting the published metadata. Visium slides have four corner fiducial markers, and YOLOv8 trained on 119 annotated regions detects them reliably enough to derive spot positions whenever at least three corners are visible.

![Raw Visium capture area with corner fiducials](/assets/images/paper/2406.16192_HEST-1k/page_015.png)
*Figure 3 (left): Raw 6.5x6.5 mm Visium capture area with corner fiducial markers — the input the YOLOv8 detector consumes.*

### HEST-Benchmark evaluation protocol

- **9 tasks:** IDC breast (Xenium), PRAD prostate (Visium), PAAD pancreas (Xenium), SKCM skin (Xenium), COAD colon (Visium), READ rectum (Visium), ccRCC kidney (Visium), LUAD lung (Xenium), LYMPH-IDC axillary nodes (Visium). Cohort sizes 2-24 patients.
- **Target:** top-50 most variable genes per task (log1p-normalized, dropping genes nonzero in fewer than 10% of spots).
- **Input:** 224x224 px patches at 20x (~112x112 µm).
- **Splits:** patient-stratified k-fold where k = number of patients (k/2 for ccRCC).
- **Regressors:** PCA(256) + Ridge with $\lambda = 100/(M \cdot C)$ where $M$ is embedding dim and $C=50$ (main); plain Ridge (Appendix A13); XGBoost (Appendix A14).
- **Metric:** Pearson correlation, mean ± std across folds. Hardware: single NVIDIA RTX 3090.
- **Encoders compared (11):** ResNet50 (IN, 25M), CTransPath (28M), Phikon (86M), CONCH (86M), REMEDIS (232M), UNI (307M), Virchow / Virchow2 (632M), GigaPath / H-Optimus-0 / UNIv1.5 (1.13B).

## Dataset

| Property | Value |
|---|---|
| Paired ST+WSI samples | **1,229** |
| Cohorts | 153 (public + 3 internal) |
| Organs | 26 |
| Species | 2 (Homo sapiens, Mus musculus) |
| Cancer samples | 367 across 25 OncoTree subtypes |
| Technology mix | Visium ~48.0% (n=547), Visium HD 0.8% (n=10), Xenium 5.3% (n=65), STv1 44.9% (n=552) |
| Patch-expression pairs | 2.1 M (224x224 px @ 20x) |
| Nuclei segmented | 76.4 M (CellViT 5-class) |
| Resolution split | 224 px / 112 µm: 52.8% (n=649) vs. 47.2% (n=580) other |
| Sample types | Pathological 453, Cancer 367, Healthy 345, Treated 53, Tumor 7, Genetically modified 4 |
| License | CC BY-NC-SA 4.0 (dataset, library, benchmark) |
| Distribution | HuggingFace Datasets; ~1 TB raw |

**Source breakdown (Appendix Table A1):** NCBI 696 samples (43 datasets, 298 GB), Spatial-Research 139 (4 datasets), Mendeley 118 (9), Miscellaneous 114, 10x Genomics 112 (87 datasets), Internal 28 (3), Zenodo 21 (4).

**Organ distribution.** Heavily skewed toward brain (n=318) and spinal cord (n=211), then breast (n=125), bowel (n=94), skin (n=88), kidney / heart (n=70 each), prostate (n=62), lung (n=60), with single-digit counts for placenta, embryo, ovary, lymph node, cervix, bladder, pancreas, etc. Brain + spinal cord alone make up ~43% of the corpus.

**Known limitations of the dataset itself:**

- Staining/compression artifacts, variable acquisition protocols, non-uniform image quality across cohorts.
- Significant batch effects on both imaging and transcriptomic axes — HEST-Library exposes hooks (ComBat / Harmony / MNN) but the paper does **not** quantify them head-to-head.
- Severe organ imbalance; cancer-organ coverage in HEST-Benchmark is narrower than the full corpus suggests.
- Many cohorts have very few patients — Tasks 1-8 use 2-3 patients each, only Task 7 (ccRCC) has 24.
- Nuclear classifications inherit CellViT / PanNuke biases. The authors are explicit that the segmentation is **not a clinical gold standard**.

## Experimental Results

### Main HEST-Benchmark — PCA + Ridge (Table 1)

Pearson correlation per task; bold = best, _italic_ = second-best per column.

| Encoder | Params | IDC | PRAD | PAAD | SKCM | COAD | READ | ccRCC | LUAD | LYMPH-IDC | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 (IN) | 25M | 0.4741 | 0.3075 | 0.3889 | 0.4822 | 0.2528 | 0.0812 | 0.2231 | 0.4917 | 0.2322 | 0.3260 |
| CTransPath | 28M | 0.5110 | 0.3427 | 0.4378 | 0.5106 | 0.2285 | 0.1100 | 0.2279 | 0.4985 | 0.2353 | 0.3447 |
| Phikon | 86M | 0.5327 | 0.3420 | 0.4432 | 0.5355 | 0.2585 | 0.1517 | 0.2423 | 0.5468 | 0.2373 | 0.3656 |
| CONCH | 86M | 0.5363 | 0.3548 | 0.4475 | 0.5791 | 0.2533 | 0.1674 | 0.2179 | 0.5312 | 0.2507 | 0.3709 |
| REMEDIS | 232M | 0.5290 | 0.3471 | 0.4644 | 0.5818 | 0.2856 | 0.1145 | 0.2647 | 0.5336 | 0.2473 | 0.3742 |
| GigaPath | 1.13B | 0.5508 | 0.3708 | 0.4768 | 0.5538 | 0.3010 | 0.1860 | 0.2391 | 0.5399 | 0.2493 | 0.3853 |
| UNI | 307M | 0.5702 | 0.3140 | 0.4764 | 0.6254 | 0.2630 | 0.1762 | 0.2427 | 0.5511 | 0.2565 | 0.3862 |
| Virchow | 632M | 0.5702 | 0.3309 | 0.4875 | 0.6088 | 0.3110 | 0.2019 | 0.2637 | 0.5459 | 0.2594 | 0.3977 |
| Virchow2 | 632M | 0.5922 | 0.3465 | 0.4661 | 0.6174 | 0.2578 | 0.2084 | 0.2788 | 0.5605 | 0.2582 | 0.3984 |
| UNIv1.5 | 1.13B | 0.5989 | 0.3645 | 0.4902 | _0.6401_ | 0.2925 | _0.2240_ | 0.2522 | 0.5586 | _0.2597_ | _0.4090_ |
| **H-Optimus-0** | **1.13B** | _0.5982_ | **0.3850** | **0.4932** | **0.6432** | 0.2991 | **0.2292** | 0.2654 | 0.5582 | 0.2595 | **0.4146** |

Per-task standard deviations across folds are large — IDC sigma ~0.08-0.09, PAAD sigma ~0.04-0.07, COAD sigma ~0.01-0.06. **The gap from best (H-Optimus-0, 0.4146) to second (UNIv1.5, 0.4090) is 0.0056 absolute — roughly 1/10 of typical per-fold sigma.** No significance test on the average, no paired per-task test, no bootstrap intervals on the means. The C5 leaderboard rating earns its ⭐: the *tiering* is robust (legacy ImageNet < legacy domain-specific < modern DINOv2-scale), the *within-tier ordering* is not.

### Scaling laws

![Model and data scaling on HEST-Benchmark](/assets/images/paper/2406.16192_HEST-1k/page_008.png)
*Figure 5: Scaling laws across 11 encoders. (a) Model scaling: log(#trainable params) vs. avg Pearson correlation, R=0.81, p<0.01. (b) Data scaling: log(#pretraining patches) vs. avg correlation, R=0.48, p=0.13 — explicitly not significant.*

- **Model scaling law:** R=0.81, p<0.01 between log(#trainable params in the vision encoder) and average benchmark performance. CONCH, UNIv1.5, and H-Optimus-0 sit above the regression line and are flagged as "parameter-efficient" outliers.
- **Data scaling law:** R=0.48, p=0.13 between log(#pretraining patches) and average performance. The authors attribute the weak signal to ignoring per-WSI patch correlations and morphological diversity. To their credit, they label the result as not significant rather than hand-wave a "law" out of it — but the section heading still calls it one.
- **Per-task spread (H-Optimus-0):** SKCM 0.6432 down to READ 0.2292. The authors interpret low-performing tasks as weak morphology-expression coupling or noisier cohorts; both READ and ccRCC are confounded with very small patient counts.

### Cross-head consistency

H-Optimus-0 again wins the XGBoost mean (Appendix A14), beating UNIv1.5 by 0.69% — same caveats about significance apply. CTransPath-to-best gap is 7.0% under PCA+Ridge and 4.8% under XGBoost. ResNet50 (IN) is consistently last across both heads.

### Biomarker exploration (Section 6)

On a single IDC Xenium sample (n=168,033 detected neoplastic nuclei), GATA3 expression spatially co-localizes with high-nuclear-area regions; Pearson R between GATA3 expression and neoplastic-cell nuclear area = **0.47 (p<10^-4)**.

![GATA3 expression heatmap on IDC Xenium WSI](/assets/images/paper/2406.16192_HEST-1k/page_009.png)
*Figure 6: GATA3 expression heatmap overlaid on an invasive ductal carcinoma Xenium WSI; high expression visibly aligns with tumor regions.*

In a second IDC sample (n=342,018 nuclei, Appendix Figure 6) nuclear size correlates with FLNB (R=0.45), TPD52 (R=0.47), and FOXA1 (R=0.47). Across the 12 hand-crafted nuclear features, size-related features dominate; shape/topology features stay below R=0.2. There is no multiple-testing correction, no null/permutation baseline, and only two slides — read this as a case study, not as inference.

### Multimodal alignment fine-tuning (Section 7, Table 2)

CONCH-FT (CONCH ViT-B fine-tuned on 5 Xenium IDC/ILC samples with InfoNCE) vs. base CONCH, evaluated on BCNB (n=1,058 WSIs, mean-pool patch embeddings -> logistic regression):

| Encoder | Rank | ER AUC | ER Bal.acc | PR AUC | PR Bal.acc | HER2 AUC | HER2 Bal.acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| CONCH | 144.66 | 0.881 | 0.745 | 0.810 | 0.698 | 0.715 | 0.624 |
| **CONCH-FT** | **146.47** | **0.884** | **0.752** | **0.818** | **0.714** | **0.724** | 0.615 |

CONCH-FT wins 5 of 6 metrics; HER2 balanced accuracy regresses (0.624 -> 0.615). Improvements are tiny (Delta-AUC <= 0.009) and Table 2 reports only point estimates — no fold variance, no significance test. With n=5 training patients this is a feasibility demonstration, not a validated method.

## Limitations

**Acknowledged by the authors:**

- ST measurements are inherently noisy (dropout, spillover, low sensitivity); HEST does not denoise expression.
- Batch effects across cohorts and technologies are significant but unquantified; the library exposes hooks (ComBat / Harmony / MNN) but no head-to-head comparison is reported.
- HEST-Library is a "blueprint," not a complete legacy-format coverage solution.
- Nuclear segmentation inherits CellViT / PanNuke biases; not a clinical reference.
- Genetic and demographic representativeness is unaudited.

**Not addressed (analyst-noted):**

- **No statistical significance testing for the encoder leaderboard.** No bootstrap, no paired tests, no Bonferroni across the 9 tasks. The "winner" depends on rounding.
- **Severe cohort imbalance:** 5 of 9 HEST-Benchmark tasks rely on 2-3 patients. The patient-stratified k-fold protocol is honest, but the resulting confidence intervals are unavoidably wide.
- **No train/test contamination audit.** Encoders pretrained on TCGA / proprietary data might overlap with HEST cohorts.
- **Per-task top-50 HVG target selection** is not held-out at the gene level — different encoders are effectively scored on slightly different gene panels per task, complicating cross-task comparison.
- **Xenium pseudo-Visium pooling at 55 µm** is presented but not validated against a real Visium acquisition of the same tissue.
- **Biomarker analysis** is two-slide exploratory work with no inferential statistics across patients and no multiple-testing correction.
- **CONCH-FT is trained on n=5 patients** with no ablation isolating contrastive alignment vs. additional breast-cancer image exposure.
- **No latency / inference-cost comparison** across the 11 encoders, despite the "parameter-efficient" framing.
- Mouse vs. human balance is unclear in the headline numbers; HEST-Benchmark itself is human-only.

## Why It Matters for Medical AI

HEST-1k changes the marginal cost of multimodal morpho-molecular research from "scrape five papers and re-implement five alignment pipelines" to "one HuggingFace download and a Python call." That alone is a structural contribution to computational pathology — the same way ImageNet or BraTS were less interesting as algorithms than as common ground.

The benchmark is the more delicate piece. As a *tiering* instrument for foundation models, HEST-Benchmark works: it cleanly separates ResNet50, the early domain-specific Swin/iBOT models, and the modern DINOv2-scale family. As a *ranking* instrument, it doesn't — and the paper's own numbers (0.0056 lead, per-fold sigma 0.04-0.09, 2-3 patients per task) make that visible to anyone willing to look. The healthy way to use HEST-Benchmark in a medical-AI paper is to report it alongside variance, declare a tier rather than a winner, and treat the top-50 HVG task as one signal among several — not as a Camelyon-style discriminator.

Finally, the CONCH-FT proof-of-concept is genuinely interesting as a direction. Using paired Xenium data to fine-tune a vision-language encoder for a tissue-type-specific downstream task is exactly the kind of cross-modality bootstrapping that ST is uniquely positioned to enable. The n=5 fine-tuning cohort here is too small to call a method, but the framework is real and HEST-1k makes it cheap to scale.

## References

- **Paper:** Jaume et al., "HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis," NeurIPS 2024 Datasets & Benchmarks (Spotlight). arXiv: [2406.16192](https://arxiv.org/abs/2406.16192) (v2, 2 Nov 2024).
- **Code & library:** [https://github.com/mahmoodlab/hest](https://github.com/mahmoodlab/hest)
- **Data:** HuggingFace Datasets (CC BY-NC-SA 4.0).
- **Foundation models compared:** UNI / UNIv1.5 (Chen et al.), CONCH (Lu et al.), GigaPath (Xu et al.), Virchow / Virchow2 (Vorontsov et al.), H-Optimus-0 (Saillard et al.), Phikon (Filiot et al.), CTransPath (Wang et al.), REMEDIS (Azizi et al.).
- **Toolkit dependencies:** CellViT (PanNuke), DeepLabV3 + ResNet50, YOLOv8, VALIS, OpenSlide, QuPath, scanpy, AnnData.
