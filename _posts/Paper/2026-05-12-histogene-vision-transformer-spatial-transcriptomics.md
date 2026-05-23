---
title: "HisToGene: Leveraging Information in Spatial Transcriptomics to Predict Super-Resolution Gene Expression from Histology Images in Tumors"
excerpt: "A whole-section Vision Transformer with one-hot coordinate embeddings predicts 785-gene expression from H&E patches and beats ST-Net's per-section median Pearson r in all 32 HER2+ breast cancer sections at ~2.5x lower compute — but every absolute number is modest and the headline super-resolution claim leans on N=6 sections with patient leakage."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/histogene-vision-transformer-spatial-transcriptomics/
tags:
  - HisToGene
  - Vision-Transformer
  - Spatial-Transcriptomics
  - Histology
  - Super-Resolution
  - HER2-Breast-Cancer
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- HisToGene swaps ST-Net's per-spot CNN regressor for a **whole-section Vision Transformer** that ingests every spot patch of a tissue section as one sequence, with **one-hot (x, y) grid coordinate embeddings**, and regresses 785 log-normalized gene expression values per spot.
- A **dense-grid resampling trick at inference time** — predict over patches shifted by half a spot diameter and average four overlapping predictions per quarter-patch — yields a free 4x super-resolution gene expression map and a re-aggregated spot-level estimate (HisToGene*) that the authors find usually beats the single-pass prediction.
- On the Andersson HER2+ breast cancer ST cohort (32 sections / 7 patients / 9,612 spots / 785 genes, leave-one-section-out), HisToGene beats ST-Net's per-section median Pearson r in **all 32 sections**, with patient-B medians peaking ~**0.25-0.30** (vs. ST-Net medians clustering near 0), and trains at **~11 min/fold vs. ~27 min/fold** for ST-Net on a single V100.

## Motivation

Spatial transcriptomics (ST) reads out a spatially resolved transcriptome from a tissue section, but the assay is slow, costly, and only routinely available at well-resourced centres. H&E slides, by contrast, are produced for nearly every solid-tumour biopsy on Earth. If a model could turn H&E into a virtual transcriptome, the cost asymmetry would be huge. ST-Net (He et al., 2020) showed a per-spot CNN regressor can do this in principle, but its design has two structural weaknesses HisToGene is built to attack:

1. **Each spot is regressed independently.** Spatial smoothness of expression and shared microenvironment between neighbouring spots are simply thrown away.
2. **Per-spot i.i.d. modelling exposes the network to patient batch effects** when train and test sections come from different subjects.

Vision Transformers had just landed (Dosovitskiy et al., 2021) and the authors saw a clean fit: an ST section *is* a variable-length sequence of patches with 2-D positions — exactly what self-attention was designed for.

## Core Innovation

- **Self-attention over an entire tissue section.** Every spot can attend to every other spot in the same section; neighbouring spots co-determine each prediction. ST-Net's per-spot i.i.d. assumption is broken.
- **One-hot (x, y) array-coordinate embeddings.** Each axis index is one-hot encoded (up to 30 columns / rows on HER2+) and linearly projected to 1024-D. Coordinate awareness is baked into the token, not learned implicitly.
- **Free super-resolution at inference time.** Because the transformer is sequence-length-agnostic, the same trained model can score overlapping half-spot-shifted patches and average the four predictions per quarter-patch — a 4x finer map and a re-aggregated HisToGene* spot estimate, with no extra labels. ST-Net's per-spot CNN cannot do this.

## Method & Architecture

![HisToGene workflow: modified ViT over an entire tissue section with one-hot coordinate embeddings and dense overlapping inference for super-resolution](/assets/images/paper/histogene/page_027.png)
*Figure 1: HisToGene workflow. (a) A modified ViT consumes a sequence of H&E spot patches with one-hot (x, y) coordinate embeddings; (b) at inference, dense overlapping patch sampling lets the same trained model emit a 4x super-resolution gene expression map by averaging predictions across overlapping spot-tokens.*

### 1. Patch extraction

For each tissue section, crop a $112 \times 112$ pixel H&E patch around every spot centre (112 px matches the original ST array spot diameter on HER2+) and stack into $\mathbf{F}_I \in \mathbb{R}^{N \times (3 \cdot W \cdot H)}$, where $N$ varies per section.

### 2. Image-token embedding

A single learnable linear layer $\mathbf{W}_I$ projects $\mathbf{F}_I$ to $\mathbf{E}_I \in \mathbb{R}^{N \times 1024}$. No CNN backbone, no ViT-style 16x16 sub-patching — each whole spot patch becomes one token via a raw-pixel linear projection. This is an unusually shallow front-end relative to later pathology-FM-based successors.

### 3. Positional encoding

Spot grid coordinates $(x, y)$ are one-hot encoded *separately per axis* and projected to 1024-D by learnable $\mathbf{W}_x$, $\mathbf{W}_y$:

$$\mathbf{E} = \mathbf{E}_I + \mathbf{E}_x + \mathbf{E}_y.$$

The encoding is discrete and grid-tied, not sinusoidal — which makes the model coordinate-aware but also brittle to platforms with different geometries (Visium HD, Slide-seqV2, Stereo-seq).

### 4. Transformer encoder

Eight layers, 16 heads, $d_\text{model} = 1024$, dropout 0.1. Standard scaled dot-product attention,

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\tfrac{Q K^\top}{\sqrt{d_k}}\right) V,$$

with $Q = K = V$ taken over all $N$ spot-tokens of the section. Self-attention is **global within a section** — every spot can attend to every other spot. Output $\mathbf{H} \in \mathbb{R}^{N \times 1024}$.

### 5. Regression head and loss

A linear layer maps each row of $\mathbf{H}$ to the 785-D expression vector. Loss is MSE against $\log(\text{TP10M} + 1)$ counts (UMI per spot / total spot UMI x $10^6$, then natural log).

### 6. Training recipe

PyTorch, learning rate $10^{-5}$, 100 epochs, one section per forward pass (batch size = 1 *section*, $N$ spots inside it). Single V100 32GB. ~11 min/fold versus ~27 min/fold for the authors' re-implementation of ST-Net.

### 7. Super-resolution inference (HisToGene_SR / HisToGene*)

At test time, slide the 112-px patch window in half-spot steps so the section is tiled by overlapping patches. Every 56x56 sub-patch is then covered by exactly 4 different spot-tokens, and per sub-patch the 4 predictions are averaged:

$$E_\text{sub-patch} = \tfrac{1}{4}\!\left(E_\text{top-left} + E_\text{top-right} + E_\text{bot-left} + E_\text{bot-right}\right).$$

This produces (a) a 4x-finer SR map (`HisToGene_SR`) and (b) when the four sub-patches of an original spot are re-summed, a spot-level estimate `HisToGene*` that usually correlates better with the observed expression than the single-pass prediction.

![Attention maps reveal shallow-to-deep behaviour](/assets/images/paper/histogene/page_031.png)
*Figure 5: Attention maps from layers 1, 4, and 8 of the modified ViT at original and super-resolution settings. Shallow layers attend to the target spot; deep layers attend to distant tumour-related spots. Single-example qualitative figure — treat as illustration, not evidence.*

## Claims & Evidence Analysis

| # | Claim (abstract / discussion) | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | "HisToGene accurately predicts gene expression" | Per-section median Pearson r on 785 genes; best sections (patient B) reach ~0.25-0.30 medians; top genes mean r 0.26-0.32 | HER2+ only | ⭐⭐ — strictly relative to ST-Net (~0). Absolute r is modest; "accurately" is a stretch. |
| **C2** | **"HisToGene outperforms ST-Net in gene expression prediction"** | **Fig 2a: HisToGene > ST-Net per-section median in all 32 sections; IQRs do not overlap in most sections** | **HER2+** | **⭐⭐⭐ — visually overwhelming margin; comparison is against authors' re-implementation, which is a minor caveat, not fatal.** |
| C3 | "Outperforms ST-Net on clustering tissue regions using predicted expression" | Fig 4 ARI: HisToGene* beats ST-Net in 5/6 sections | HER2+ (6 annotated sections) | ⭐⭐ — small N=6; no variance reporting; deltas (e.g. C1: 0.10 vs. 0.11) within noise. |
| C4 | "Super-resolution predicted expression leads to higher clustering accuracy than observed expression" | Fig 4 ARI: HisToGene* > observed in 4/6 sections | HER2+ | ⭐ — N=6 sections, no variance, no patient hold-out. Patient B has 6 sections so LOO-by-section leaks the patient in 5/6 test sections. The strongest abstract claim is the weakest experimentally. |
| C5 | "First method for super-resolution gene expression prediction from histology" | Architecture supports variable-length input; SR trick presented in Methods | HER2+ (qualitative SR maps, Fig 3) | ⭐⭐ — true at time of writing (Nov 2021); SR is interpolation with **no ground truth at the 56-µm sub-spot scale**. |
| C6 | "Robust to patient heterogeneity" | Fig 2a: HisToGene produces non-zero correlations across multiple patients; ST-Net does not | HER2+ | ⭐⭐ — relative to ST-Net only. The 0.00 (patient A/E) vs. 0.25-0.30 (patient B) median spread is itself huge patient-level variability. "Less catastrophic than ST-Net" is fair; "robust" overstates. |
| **C7** | **"Computationally fast"** | **~11 min vs. ~27 min per fold on V100; single forward pass per section vs. per-spot inference for ST-Net** | **HER2+, 31 train sections** | **⭐⭐⭐ — clean wall-clock comparison with a sound architectural reason.** |
| C8 | Top predicted genes are biologically meaningful (GSEA enrichment) | Fig 2b: top-100 list enriched in TSH-signalling, energy metabolism, fatty-acid biosynthesis pathways | HER2+ | ⭐ — GSEA on a correlated, post-selected gene list with no FDR control across pathways. Rhetorical, not statistical. |
| C9 | Attention maps reveal interpretable shallow-to-deep behaviour | Fig 5: shallow layer attends to target spot; deep layer to distant "tumour-related" spots | HER2+, one example | ⭐ — single-example qualitative figure; "tumour-related" not formally tested. |

**Overall rating: ⭐⭐.** C2 and C7 are the load-bearing wins and would survive aggressive peer review — HisToGene beats ST-Net's per-section median Pearson r in every one of 32 sections and trains in ~40% of the time. C1's "accurately" is relative-only: 0.05-0.30 medians are exactly what BLEEP, EGN, STPath, and OmiCLIP/Loki cite when they later claim large gains over HisToGene. C4 is the abstract's headliner *and* its weakest claim — N=6 sections, no variance, no held-out patients, and patient B (6 sections) sits in the training fold for 5 of those 6 tests, so "predicted expression beats observed for clustering" cannot be cleanly separated from within-patient leakage and K-means initialisation noise. What is missing across the board: variance reporting on Pearson r and ARI, gene-level p-value adjustment, an out-of-cohort dataset as a primary result, an ablation isolating positional encoding from architectural choice, and a non-deep baseline (e.g. ridge regression on PCA-of-patches).

## Experimental Results

The paper reports no single summary table; numbers are read off Figures 2 and 4.

### Per-section Pearson correlation, 785 genes (Fig 2a)

![Per-section Pearson correlation across 785 genes on HER2+ and GSEA on top-100 predicted genes](/assets/images/paper/histogene/page_020.png)
*Figure 2: Per-section Pearson correlation across 785 genes on the HER2+ breast cancer dataset. (a) HisToGene* (yellow) > HisToGene (blue) > ST-Net (grey) in nearly every section; ST-Net medians cluster around 0. (b) GSEA on the top-100 predicted gene list shows enrichment in breast-cancer-relevant pathways for HisToGene/HisToGene*.*

| Method | Metric | Dataset | Number (as plotted) |
|---|---|---|---|
| ST-Net (re-impl.) | per-section median Pearson r, 785 genes | HER2+, 32 sections | ~0.00 for almost every section (grey boxes centred on zero) |
| HisToGene | per-section median Pearson r | HER2+, 32 sections | ~0.00-0.25; patient-B sections (B1-B6) peak ~0.20-0.25 |
| **HisToGene\*** (SR-averaged) | per-section median Pearson r | HER2+, 32 sections | **~0.05-0.30; B1-B6 medians ~0.25-0.30. Beats HisToGene in 19/32 sections, loses in 6/32** |

Top gene mean Pearson r across 32 sections (HER2+):

| Gene | HisToGene | HisToGene* |
|---|---|---|
| GNAS | 0.32 | - |
| MYL12B | 0.27 | - |
| FASN | 0.27 | 0.24 |
| CLDN4 | 0.26 | - |
| FN1 | 0.22 | 0.24 |

### Qualitative super-resolution maps (Fig 3)

![Spatial expression maps for top predicted genes including 4x super-resolution](/assets/images/paper/histogene/page_029.png)
*Figure 3: Spatial expression maps for top predicted genes (GNAS, MYL12B, FASN, CLDN4 in (a); GNAS, FN1, MYL12B, FASN in (b)) compared across observed, ST-Net, HisToGene, HisToGene*, and 4x super-resolution HisToGene_SR. Per-section Pearson r annotated. SR maps reveal fine-grained patterns absent in spot-level predictions but are not independently validated.*

### K-means clustering vs. pathologist annotation (Fig 4, ARI)

![K-means clustering ARI on 6 pathologist-annotated HER2+ sections](/assets/images/paper/histogene/page_021.png)
*Figure 4: K-means clustering (k=4) of predicted gene expression on the 6 pathologist-annotated HER2+ sections. HisToGene* yields the highest ARI in 4/6 sections (B1: 0.30; D1: 0.26; F1: 0.22; C1: 0.11) and beats clustering of observed expression in 4/6 — the paper's most-cited but most-fragile claim, given N=6 and no variance.*

| Section | Observed expr. | ST-Net | HisToGene | **HisToGene\*** | Winner |
|---|---|---|---|---|---|
| B1 | 0.29 | 0.02 | 0.22 | **0.30** | HisToGene* |
| C1 | 0.07 | 0.10 | 0.03 | **0.11** | HisToGene* |
| D1 | 0.16 | 0.12 | 0.22 | **0.26** | HisToGene* |
| E1 | 0.03 | **0.12** | 0.04 | 0.06 | ST-Net |
| F1 | 0.04 | 0.02 | 0.11 | **0.22** | HisToGene* |
| G2 | **0.17** | 0.06 | 0.15 | 0.10 | Observed |

HisToGene* wins 4/6 sections and beats clustering of observed expression in 4/6 — but absolute ARIs of 0.06-0.30 with no variance reporting are essentially within K-means initialisation noise, and ST-Net's single win (E1) where it edges out observed expression too is a tell that these are unstable comparisons rather than vindication of any method.

### Compute

| Method | Wall-clock per LOO fold (V100 32GB) |
|---|---|
| ST-Net (re-impl.) | ~27 min |
| **HisToGene** | **~11 min** |

### Ablations

The paper does **not** run a proper ablation. No run with positional embeddings removed; no run varying layers / heads; no run replacing one-hot coordinates with learnable / sinusoidal; no run with a frozen CNN backbone in front of the linear patch projector. HisToGene > ST-Net is implicitly taken as evidence for both coordinate awareness and global attention, but that conflates several design choices.

### Generalisation: cSCC supplementary

A leave-one-section-out run on the Ji et al. cSCC dataset (12 sections, 4 patients, mixed ST + Visium platforms, 6,630 spots, 134 genes) gives weak results for both methods; HisToGene still wins. The authors openly acknowledge neither model is reliable there. This is the closest thing to an external test and arguably should be in the abstract — the most honest result in the paper.

## Limitations

**Acknowledged by the authors:**

- Deep learning needs lots of training data; cSCC results are weak due to small N and platform heterogeneity.
- Performance depends on training-set size; future ST scaling will help.
- ST-Net comparison uses a re-implementation because the official code was unmaintained.

**Not acknowledged but material:**

- **No patient-held-out evaluation on HER2+.** Patient B has 6 sections; LOO-by-section is not LOO-by-patient. The best-performing test results (B1-B6) are the ones most contaminated by within-patient leakage.
- **No variance reporting.** Every Pearson r and ARI number is a single-run point estimate. No re-seed, no cross-validation variance.
- **Per-section HVG selection** is a form of test-set peeking — the gene set is chosen using information from all sections including the test section.
- **One-hot coordinate embeddings assume a fixed array geometry (max 30 x 30).** Visium HD, Slide-seqV2, Stereo-seq do not fit; the architecture is brittle for the multi-platform future.
- **Linear patch projector.** Projecting raw 112x112x3 = 37,632-D pixel vectors through one linear layer to 1024-D is wildly under-parameterised by modern standards. This is exactly why successors swap in CNN / ViT / pathology-FM backbones (CTransPath, Hibou, UNI, Phikon, Virchow) and report big gains.
- **No clinical-task evaluation.** ARI on 4 K-means clusters vs. hand-drawn tissue boundaries is a soft proxy for any diagnostic outcome.
- **Super-resolution is not validated as super-resolution.** There is no ground truth at the 56-µm sub-spot scale; SR is validated only by re-aggregating to spot scale. A real validation would require Stereo-seq or Visium HD on the same tissue.

## Why It Matters for Medical AI

Almost every solid-tumour biopsy on Earth generates an H&E slide; almost none generates spatial transcriptomics. A model that converts cheap, abundant H&E into virtual ST is therefore a high-leverage piece of clinical infrastructure — the same pitch every later foundation-model successor (BLEEP, EGN, STPath, OmiCLIP, HEST) inherits from HisToGene. The durable contribution of HisToGene is the architectural template: **coordinate-aware self-attention over a whole section**, with the spot as the token. Every method that came afterwards either kept that template (THItoGene, EGN, Hist2ST) or replaced the linear patch projector with a pretrained pathology FM and added cross-attention to it (BLEEP, OmiCLIP, STPath). The absolute Pearson r of 0.05-0.30 medians is exactly the baseline number that 2024-era methods cite when they claim large gains; in that sense HisToGene's most lasting role in 2026 is as a sanity-check baseline on benchmarks like HEST-Bench, not as a deployable tool.

## References

- Pang, M., Su, K., Li, M. *Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors*. bioRxiv 2021.11.28.470212. DOI: [10.1101/2021.11.28.470212](https://doi.org/10.1101/2021.11.28.470212)
- Code: [github.com/maxpmx/HisToGene](https://github.com/maxpmx/HisToGene)
- Related work:
  - He, B. et al. *Integrating spatial gene expression and breast tumour morphology via deep learning* (ST-Net), Nature Biomedical Engineering 2020.
  - Andersson, A. et al. *Spatial deconvolution of HER2-positive breast cancer*, Nature Communications 2021 — the HER2+ ST dataset used here.
  - Ji, A. et al. *Multimodal analysis of composition and spatial architecture in human squamous cell carcinoma*, Cell 2020 — the cSCC supplementary cohort.
  - Dosovitskiy, A. et al. *An image is worth 16x16 words* (ViT), ICLR 2021.
  - Xie, R. et al. *BLEEP: Bi-modal Embedding for Expression Prediction*, NeurIPS 2023 — retrieval-based successor.
  - Jaume, G. et al. *HEST-1k*, NeurIPS 2024 — modern H&E-ST benchmark on which HisToGene appears as a baseline.
