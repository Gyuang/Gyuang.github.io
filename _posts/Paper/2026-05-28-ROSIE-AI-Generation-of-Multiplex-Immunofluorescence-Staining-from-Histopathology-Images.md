---
title: "ROSIE: AI Generation of Multiplex Immunofluorescence Staining from Histopathology Images"
excerpt: "A 50M-parameter ConvNeXt-Small patch regressor, trained on the largest paired H&E+CODEX corpus to date (1,342 samples / 16M cells), predicts 50 protein markers from H&E alone — headline Pearson R 0.319 on Stanford-PGC, but drops to 0.218 cross-site and walks back cross-platform generalization."
categories:
  - Paper
  - Spatial-Proteomics
  - Pathology
permalink: /paper/rosie/
tags:
  - ROSIE
  - Virtual-Staining
  - Spatial-Proteomics
  - CODEX
  - Multiplex-Immunofluorescence
  - ConvNeXt
  - Computational-Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- ROSIE is a patch-wise **ConvNeXt-Small (~50M params)** regressor — *not* a GAN, *not* a diffusion model, *not* a UNet — that maps a 128×128 px H&E patch to a 50-vector of mean CODEX biomarker expressions over the centered 8×8 px tile, trained with a single **masked MSE loss**. Tiles are then stitched into a whole virtual mIF image.
- Trained on the **largest co-stained H&E+CODEX corpus to date**: **1,342 samples, 18 studies, 13 disease types, ~16M cells, 134M patches** — co-stained on the same physical slice (CODEX first, then H&E), not adjacent sections.
- Headline numbers (Stanford-PGC in-distribution holdout): **Pearson R = 0.319, Spearman R = 0.386, sample-level C-index = 0.694** across all 50 biomarkers; downstream **SCGP tissue-structure ARI = 0.475**, **TIL-count Pearson R = 0.805**. Cross-site/cross-disease generalization **drops to Pearson R = 0.218 (Ochsner-CRC) / 0.265 (Tuebingen-GEJ)**, and cross-platform Orion-CRC is honestly framed as a partial failure.

## Motivation

H&E is universal, cheap, and information-dense for morphology — but blind to specific protein markers that drive precision oncology decisions (B vs T cell discrimination, TIL scoring, PD-L1 / Ki67 status). Multiplex IF (CODEX/PhenoCycler, Orion) gives ~30–50 proteins per slide but is slow, expensive, and absent from routine clinical workflows.

Prior virtual-staining work has been limited to 1–10 markers, single tissue types, GAN architectures validated visually rather than biologically, and small (often single-sample) cohorts. ROSIE scales the paired-dataset paradigm by 1–2 orders of magnitude — 16M cells, 13 diseases, 10 body sites — and validates *downstream biological tasks* (cell phenotyping, tissue-structure discovery, TIL/LNE quantification). The pitch is to turn the ubiquitous H&E slide into a screening tool for the tumor immune microenvironment.

## Core Innovation

**Scale-plus-simplicity.** Rather than wrestle with the well-known instabilities of GAN-based virtual staining at panel scale, ROSIE deliberately picks the most boring possible architecture (a mid-size CNN with a regression head) and instead invests in *data*: 1,342 co-stained samples, 18 studies, 148 unique biomarkers reconciled to the 50 most prevalent, with a single masked MSE that gracefully handles per-study panel heterogeneity.

Three deliberate design choices follow:

1. **No GAN, no diffusion, no perceptual loss** — only masked MSE on per-patch mean expression. The paper explicitly contrasts this with a pix2pix baseline that is reported as unstable and underperforming.
2. **CNN over ViT foundation models.** ConvNeXt-Small (50M params, ImageNet pre-trained) beats a >300M-param ViT-L/16 histopathology foundation model — attributed to local inductive bias and foundation-model overfitting plateaus on this regression task.
3. **QC heuristics from H&E alone.** Two cheap proxies — predicted-channel dynamic range and Wasserstein-1 distance from the test H&E histogram to the training distribution — flag low-quality predictions *before* any ground truth is available, lifting mean Pearson R from 0.285 to 0.336 by median-thresholding.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | ROSIE predicts up to 50 protein biomarkers from H&E alone | Table 2, Fig. 2; per-biomarker Pearson R bars in Fig. 2B and Supp. Figs. 1–3 | Stanford-PGC + 3 holdouts (CODEX) | ⭐⭐⭐ |
| C2 | The model generalizes to clinical sites and disease types unseen during training | Table 2 rows for Ochsner-CRC and Tuebingen-GEJ (Pearson 0.218 / 0.265, dropped from 0.319 in-domain) | 2 cross-site CODEX cohorts | ⭐⭐ — clear generalization, but with a measurable accuracy drop the abstract underplays |
| C3 | ROSIE-derived biomarkers enable accurate cell phenotyping (incl. B vs T cells, not separable on H&E alone) | Fig. 3 F1 scores (N=817,765); Supp. Fig. 4 confusion matrix | Stanford-PGC primarily; Ochsner-CRC F1=0.411 | ⭐⭐⭐ for B/T separation; ⭐⭐ overall (Macrophages F1=0.18, Neutrophils F1=0.26 are weak) |
| C4 | ROSIE outperforms a SOTA H&E cell-phenotyping model (CellViT++) | Supp. Fig. 9 on 6 reconciled classes | Stanford-PGC, ~220K reconciled cells | ⭐⭐ — single dataset, single CellViT++ checkpoint trained on a different label set; comparison is structurally favorable to ROSIE |
| C5 | ROSIE outperforms GAN-based virtual staining (pix2pix) | Supp. Table 1, Supp. Fig. 11 | Stanford-PGC | ⭐⭐ — only one GAN variant, only one dataset; no comparison to VirtualMultiplexer, Multi-V-Stain, HEMIT, 7-UP, DeepLIIF on a shared benchmark |
| C6 | A 50M-param ConvNeXt-Small beats ViT-L/16 histopathology foundation models | Supp. Table 1 | unclear (likely Stanford-PGC) | ⭐⭐ — interesting and consistent with the task-specific-vs-FM literature, but evidence is one row in a supplementary table with no variance |
| C7 | Discovered tissue structures match CODEX-derived structures | Fig. 4: ARI 0.475, F1 0.624 (N=635,649) | Stanford-PGC | ⭐⭐⭐ |
| C8 | ROSIE quantifies clinically meaningful TIL/LNE neighborhoods | Fig. 5: TIL Pearson R = 0.805, LNE Pearson R = 0.598 | Stanford-PGC | ⭐⭐ — high TIL Pearson is partly driven by a few large-count samples; **Spearman 0.329** for TILs is far weaker than the Pearson advertises |
| C9 | Simple dynamic-range and W1 heuristics filter low-quality predictions ahead of ground truth | Supp. Fig. 6: median-cut raises Pearson R 0.285 → 0.312 / 0.336 | 4 eval sets | ⭐⭐⭐ |
| C10 | Method generalizes across mIF platforms (CODEX → Orion) | Supp. Fig. 8: robust on CD45, fails on ECad and others | Orion-CRC (5 WSIs, 50 crops, 17 markers) | ⭐ — authors honestly walk this back in the Discussion; cross-platform generalization is **not** established |
| C11 | "Immunologically hot vs cold" tissues are correctly distinguished | Stanford-PGC 20.0%/21.1% T-cells; Ochsner-CRC 40.1%/30.6% | 2 cohorts | ⭐⭐ — direction right; CRC overpredicted by ~10 pts (≈30% relative) |

**Honest read.** C1, C3, C7 are well supported on a credibly large, multi-site CODEX corpus — this is the most rigorous virtual-mIF paper to date by sample count and panel breadth. The cross-site generalization story (C2) is real but the abstract's "robustly predict and spatially resolve" understates a Pearson R drop from 0.319 to 0.218 on truly held-out colorectal data. Comparative claims against ViT foundation models, GANs, and CellViT++ (C4–C6) each rest on a single supplementary row with no variance and no shared benchmark — they are suggestive, not definitive. The cross-platform Orion experiment (C10) is the closest the paper comes to a true generalization stress test, and it is acknowledged as a partial failure (to the authors' credit, but it means clinical-deployment framing should be tempered). Variance is reported as 95% bootstrap CIs on F1 plots but is **absent from Table 2's headline correlation numbers**, and the patch-level vs sample-averaged metric mixing — Results text claims **Pearson R = 0.285** while Table 2 reports **0.246** — is never reconciled.

## Method & Architecture

![ROSIE overview: paired H&E+CODEX corpus and patch-wise ConvNeXt regressor](/assets/images/paper/rosie/fig_p002_01.png)
*Figure 1: ROSIE overview — 1,342 co-stained H&E+CODEX samples across 18 studies feed a patch-wise ConvNeXt-Small that predicts 50 markers per 8×8 px CODEX tile from a 128×128 px H&E patch and stitches the tiles into a whole virtual mIF image.*

**Pipeline.**

1. **Data acquisition.** FFPE tissue is first stained and imaged with CODEX (Akoya PhenoCycler, 20×, custom barcoded antibodies, autofluorescence subtraction via blank channels), then re-stained with H&E on a MoticEasyScan Pro 6N at 40×. H&E and CODEX are therefore on the **same physical slice**, not adjacent sections.
2. **Registration.** CODEX DAPI vs grayscale H&E (CLAHE-enhanced); SIFT + RANSAC produces a partial affine transform (translation, rotation, uniform scale). Fallbacks: deconvolved-H&E nuclear channel, individual color channels. This is the QC bottleneck — the authors flag fraying artifacts as an unmitigated upper bound on prediction accuracy.
3. **Patch construction.** Each sample is tiled into non-overlapping 8×8 px CODEX patches; for each, a 128×128 px H&E patch centered on it is cropped. Target = mean expression of each of the 50 markers over the 8×8 CODEX patch. The 128 px context covers ~30 cells (~3-hop neighborhood) at 40×.
4. **Model.** ConvNeXt-Small (~50M params), ImageNet-pretrained, with a regression head emitting a 50-vector per patch. ViT-L/16 histopathology foundation models (>300M params) were tried and underperformed.
5. **Loss.** Masked MSE — per-study panels differ across the 18 cohorts (148 unique biomarkers total; model emits the 50 most prevalent), so loss is computed only over markers actually present in that sample. No adversarial loss, no perceptual loss.
6. **Training.** 4× V100, batch 256, Adam, lr = 1e-4 halved every 30k iterations; early-stopped after 75k steps without improvement on Pearson R / SSIM validation; augmentations = H/V flips, brightness/contrast/saturation/hue jitter, normalization.
7. **Inference / stitching.** Sliding window at 8 px stride produces a 3.02 µm/pixel virtual mIF (an 8× downsample of native). Optional 1 px stride yields native 0.3775 µm/pixel at 64× compute cost. Stitched output is saved as TIFF in the same layout as real CODEX so any downstream CODEX pipeline runs unchanged.
8. **Cell-level expression.** Upsample predicted image to native res, then average per-cell using a DAPI-derived DeepCell segmentation mask.
9. **QC heuristics for OOD samples.** Dynamic range (99th – 1st percentile per channel) and Wasserstein-1 distance between test-H&E and per-training-H&E 256-bin histograms — both computable **without ground truth**. Median-thresholding lifts mean Pearson R from 0.285 to 0.312 (W1) or 0.336 (dynamic range).
10. **Interpretability.** Grad-CAM over the final conv feature maps per biomarker — nuclear markers (DAPI, Ki67, PCNA) light up at patch center; contextual markers (CD68, PanCK, ECad) light up diffusely across the cellular neighborhood.

## Experimental Results

### Main biomarker prediction (50 biomarkers, Table 2)

![Per-biomarker Pearson R bars and qualitative ROSIE-vs-CODEX comparisons](/assets/images/paper/rosie/fig_p004_01.png)
*Figure 2: Per-biomarker performance and qualitative results — ROSIE recapitulates structural markers (PanCK, EpCAM, Vimentin, ECad) more reliably than immune markers and degrades gracefully on Ochsner-CRC, Tuebingen-GEJ, and UChicago-DLBCL (median-percentile samples shown).*

| Method | Dataset | Pearson R | Spearman R | C-index (sample) |
|---|---|---:|---:|---:|
| H&E expression (intensity proxy) | Stanford-PGC | 0.007 | 0.013 | 0.504 |
| Cell-morphology MLP | Stanford-PGC | 0.072 | 0.081 | 0.574 |
| **ROSIE** | **Stanford-PGC (in-distribution holdout)** | **0.319** | **0.386** | **0.694** |
| ROSIE | Ochsner-CRC (unseen site & disease) | 0.218 | 0.276 | 0.597 |
| ROSIE | Tuebingen-GEJ (unseen site & disease) | 0.265 | 0.289 | 0.668 |
| ROSIE | UChicago-DLBCL (same study, different coverslip) | 0.254 | 0.327 | 0.820 |
| ROSIE | Average across 4 eval sets | 0.246 | 0.297 | 0.695 |

There is an unexplained discrepancy in the paper's own narrative: the Results text reports a four-dataset average of "Pearson R 0.285, Spearman R 0.352, C-index 0.706" while Table 2 reports **0.246 / 0.297 / 0.695**. The gap appears to be patch-level vs sample-averaged aggregation, but it is never reconciled inline — and the headline figure most readers will quote is the higher number.

### Cell phenotyping (Stanford-PGC, N = 817,765 cells)

![Per-cell-type F1 scores and predicted vs true cell-type maps](/assets/images/paper/rosie/fig_p005_01.png)
*Figure 3: ROSIE-derived expressions enable cell phenotyping on 817,765 Stanford-PGC cells, beating morphology and bulk-proportion baselines across all seven types. B vs T cell separation (F1 ≈ 0.31 vs ≈ 0.56) is the clinically interesting gain over H&E alone — but Macrophages F1 = 0.18 and Neutrophils F1 = 0.26 expose serious gaps on low-prevalence immune types.*

| Cell type | ROSIE F1 | Morphology baseline F1 |
|---|---:|---:|
| Epithelial | 0.61 | ~0.45 |
| T cells | 0.56 | low |
| Fibroblasts | 0.55 | low |
| Endothelial | 0.31 | 0.00 |
| B cells | 0.31 | low |
| Neutrophils | 0.26 | low |
| Macrophages | 0.18 | 0.00 |

Cross-site (Ochsner-CRC) mean F1 = **0.411** vs **0.507** on Stanford-PGC — degrades but stays usable. Importantly, the paper notes phenotyping is restricted to the **top-24 markers** by prevalence; the F1 numbers above are an upper bound on what a clinician should trust from this output.

### Tissue-structure discovery (SCGP, Stanford-PGC)

![SCGP clusters from ground-truth vs ROSIE vs morphology, F1 per structure, ARI boxplot](/assets/images/paper/rosie/fig_p007_01.png)
*Figure 4: Unsupervised tissue-structure discovery via SCGP on ROSIE-imputed mIF reaches ARI 0.475 and F1 0.624 vs ground-truth CODEX — far above the morphology baseline at ARI 0.105.*

### TIL / LNE quantification (Stanford-PGC)

![TIL and LNE neighborhood maps and scatter plots](/assets/images/paper/rosie/fig_p008_01.png)
*Figure 5: Sample-level TIL counts (Pearson R = 0.805) and lymphocyte-neighboring-epithelial-cell proportions (Pearson R = 0.598) from virtual mIF track ground truth — though TIL Spearman 0.329 reveals weaker mid-range ordering than the Pearson alone implies.*

### Other experiments

- **Immune "hot vs cold" sanity check.** Stanford-PGC (pancreatic, expected cold): 20.0% T cells predicted vs GT 21.1%. Ochsner-CRC (CRC, expected hot): 40.1% predicted vs GT 30.6%. Direction is right; CRC overpredicts by ~10 absolute points (≈30% relative).
- **GAN comparison.** pix2pix (256×256 patches, 50-channel output, 50 epochs) reportedly "significantly underperforms" with border artifacts and training instability (Supp. Table 1, Fig. 11). One dataset, one GAN variant.
- **ViT vs CNN.** ViT-L/16 histopathology foundation model (>300M params) is beaten by ConvNeXt-S (50M, ImageNet pre-trained); attributed to local inductive bias and foundation-model overfitting.
- **Cross-platform Orion-CRC.** Robust on some markers (CD45), poor on others (ECad explicitly called out). Authors honestly frame this as a limit of generalization.
- **QC heuristics.** Median-thresholding on W1 lifts mean Pearson R 0.285 → 0.312; on dynamic range, 0.285 → 0.336.

## Limitations

**Authors acknowledge:**

- H&E/CODEX registration is susceptible to fraying / tissue-distortion artifacts that bound prediction accuracy.
- Biomarker imbalance in training data degrades performance on rare markers; phenotyping is silently restricted to the **top-24 markers** by prevalence.
- 128×128 context window (~30 cells) is the operating point; 256×256 was tried with no gain.
- All training H&E and CODEX is from one in-house pipeline (single H&E scanner, single PhenoCycler) — cross-platform generalization (Orion) is partial.
- ROSIE is *"not an appropriate replacement for a full-panel CODEX experiment"* — the panel-wide training is for representation sharing and marker discoverability, not equivalence.
- Not blinded; no power calculation.

**Not adequately addressed:**

- **No per-biomarker hallucination-risk analysis for rare / low-prevalence markers.** The paper reports average Pearson R but does not call out which markers are unsafe to consume — e.g., what is the false-positive rate of "predicted PD-L1 high" cases? Given Macrophages F1 = 0.18 and Neutrophils F1 = 0.26, this is a real downstream-deployment gap.
- **No statistical-significance tests on the headline Table 2 numbers**; no confidence intervals on the headline Pearson R values.
- **No survival / treatment-response / prognosis endpoint** — clinical-utility framing in the discussion is aspirational, not demonstrated.
- **Pix2pix is the only generative-model baseline.** No diffusion baseline, no comparison with concurrent virtual-mIF works (HEMIT, Multi-V-Stain, 7-UP, VirtualMultiplexer) on a shared benchmark.
- **Headline-metric inconsistency:** Results text claims **Pearson R = 0.285** while Table 2 reports **0.246** — patch-level vs sample-averaged aggregation, but never reconciled inline.
- "Coverslip-level" splitting is used; for Stanford-PGC and UChicago-DLBCL the same study contributes to both train and eval — not patient-stratified or institution-stratified for those two.
- Demographic / sex / age / ethnicity reporting absent.
- Limited interpretability: Grad-CAM on a few markers is a qualitative gesture; no quantitative attribution validation.

## Why It Matters for Medical AI

ROSIE is the first virtual-mIF effort that is *plausibly* big enough — 16M cells, 13 diseases, 50 markers — for the question "can H&E pre-screen which slides deserve a real CODEX run?" to become an honest empirical question rather than a single-tissue demo. The downstream tasks the authors prioritize (cell phenotyping, SCGP, TIL/LNE) are exactly the right ones: they are how an oncology workflow would actually consume the output.

That said, three medical-AI deployment risks deserve to be loud:

1. **Hallucination on rare immune markers.** Macrophage F1 = 0.18 and Neutrophil F1 = 0.26 mean any clinician reading a ROSIE-predicted slide for these cell types is reading mostly noise. The phenotyping-only-on-top-24-markers compromise is the right one, but it is buried.
2. **Cross-site/cross-disease drop.** Pearson R 0.319 → 0.218 on unseen colorectal data is a real generalization gap. Any clinical pilot needs to be staged at the deploying institution before believing the in-distribution number.
3. **Cross-platform is not established.** The Orion-CRC experiment is honest about its partial-failure status — anyone deploying ROSIE on a non-PhenoCycler stack should treat the model as untested there.

If you read past the abstract's "robustly predict and spatially resolve" framing, this paper actually delivers a clearer message than its own headline: virtual mIF is now a credible *screening* tool, but it is not a CODEX replacement, and the rare-marker hallucination question is unanswered.

## References

- Wu, E., Bieniosek, M., Wu, Z., Thakkar, N., Charville, G. W., Makky, A., Schurch, C. M., Huyghe, J. R., Peters, U., Li, C. I., Li, L., Giba, H., Behera, V., Raman, A., Trevino, A. E., Mayer, A. T., & Zou, J. (2025). **ROSIE: AI generation of multiplex immunofluorescence staining from histopathology images.** *Nature Communications* 16:7633. DOI: [10.1038/s41467-025-62346-0](https://doi.org/10.1038/s41467-025-62346-0)
- Code & docs: [https://gitlab.com/enable-medicine-public/rosie](https://gitlab.com/enable-medicine-public/rosie) (weights on request)
- Related virtual-staining work referenced: pix2pix (Isola et al. 2017); CellViT++ (Hörst et al.); 7-UP, HEMIT, Multi-V-Stain, VirtualMultiplexer, DeepLIIF (not benchmarked head-to-head here).
- Co-stained acquisition platform: Akoya CODEX / PhenoCycler. Cross-platform validation: Lunaphore Orion.
- Downstream methods reused: DeepCell (segmentation), SCGP (tissue-structure clustering).
