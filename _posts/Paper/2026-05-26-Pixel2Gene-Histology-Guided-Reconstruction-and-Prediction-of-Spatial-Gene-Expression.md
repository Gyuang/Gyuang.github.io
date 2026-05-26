---
title: "Pixel2Gene enables histology-guided reconstruction and prediction of spatial gene expression"
excerpt: "A per-gene 4-layer MLP on hierarchical-ViT superpixel features, trained only on high-confidence ST spots, doubles as a denoiser, full-tissue extrapolator, and out-of-sample H&E-to-expression predictor on Visium HD, Xenium, and CosMx — with no head-to-head baseline and no real molecular ground truth for unmeasured regions."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/pixel2gene/
tags:
  - Pixel2Gene
  - Spatial-Transcriptomics
  - Visium-HD
  - Xenium
  - CosMx
  - Histology
  - MLP
  - iStar
  - Denoising
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- Pixel2Gene is a **per-gene 4-hidden-layer MLP (256 units, LeakyReLU, Adam, 1,200 epochs)** that regresses spatial expression from hierarchical ViT features of H&E superpixels (8 x 8 µm, 579-dim, encoder reused from iStar), trained on the subset of superpixels that survive a tissue mask plus a per-K-means-cluster bottom-quantile UMI filter (e.g. drop the lowest ~30 % per cluster).
- The paper sells "denoising" and "prediction" as separate capabilities, but **mechanistically it is one MSE-trained model deployed at three locations**: measured-and-retained spots (denoising), measured-but-filtered-out spots (in-sample enhancement), and unmeasured spots inside or outside the capture window (full-tissue / out-of-sample). The "denoising" effect is just the model regressing low-confidence spots toward their high-confidence neighbors because the low ones were excluded from training.
- **Headline result reported only as boxplots, no numbers in text**: on Visium HD CRC-P2 top-300 highly expressed genes, Pixel2Gene improves Pearson/Spearman/cosine and reduces RMSE/MAE against a Gamma-shrinkage (mean 0.1, sd 0.03) downsampled input, and raises Moran's I at 16/32/64 µm above the downsampled and (sometimes) the original data. **No head-to-head against BLEEP, ST-Net, HisToGene, Hist2ST, GHIST, iStar, MISO, SpatialScope, THItoGene, or sCellST appears in any figure or table.**

## Motivation

High-resolution ST platforms (Visium HD, Xenium, Xenium Prime 5K, CosMx) have transformed tissue biology but remain bottlenecked by cost (Visium HD ≈ $1k+/slide), restricted capture areas (Visium HD 6.5 x 6.5 mm), dropout sparsity, and fragmented fields of view (CosMx ≈ 200 x 0.5 mm FOVs covering < 5 % of a slide). Routine H&E is cheap (~$10/slide), fast (< 24 h), and whole-slide.

Prior work either (a) denoises ST without histology, or (b) predicts expression de novo from histology (ST-Net, HisToGene, Hist2ST, BLEEP, GHIST), throwing away the high-confidence ST signal that already exists. Pixel2Gene's pitch is to **unify denoising and prediction** by training on filtered measured spots and then applying the same model to noisy spots, unmeasured spots in the same slide, or entirely new H&E slides. The clinical motivation is direct: enable whole-slide spatial molecular maps for clinical cohorts (CRC, gastric, breast, kidney, lung) at scale.

## Core Innovation

**One model, one loss, three deployment positions.** The pipeline is:

1. Tile H&E into 16 x 16 px ≈ 8 x 8 µm superpixels (1 px = 0.5 µm after rescaling).
2. For each superpixel, extract a 579-dim hierarchical ViT feature (multi-level features + raw RGB), following iStar.
3. Build the training set by filtering superpixels: drop background/debris/folds with **HistoSweep** (same group's preprint), then K-means cluster superpixels in feature space and drop the bottom-quantile UMI tail within each cluster. This is the operational definition of "high-confidence ST".
4. For each gene $k$, train an independent 4-hidden-layer MLP $g_k(\cdot)$ (256 units/layer, LeakyReLU $\alpha=0.1$) under plain MSE on the retained pairs:

$$\mathcal{L} = \sum_{k=1}^{K}\sum_{(m,n) \in P}\big(X_{kmn} - g_k(h_{mn})\big)^2$$

   Adam, batch 128, 1,200 epochs, learning rate scaled to training-set size.
5. **Three deployment modes — same trained model.**
   - *In-sample enhancement:* predict at every measured superpixel, including those filtered out of training. The model regresses these toward their high-confidence neighbors; this is what the paper calls "denoising".
   - *Full-tissue prediction:* apply to every superpixel in the slide, including the unprofiled region outside the Visium HD 6.5 x 6.5 mm capture window.
   - *Out-of-sample prediction:* apply to a different patient's H&E (CRC-P1, CRC-P5) without any new ST.

There is **no separate denoising loss**, no two-stage training, and no joint multi-gene model — for a 5,000-gene Xenium Prime sample that is 5,000 independently trained MLPs.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Pixel2Gene denoises and reconstructs sparse ST better than the raw degraded input. | Fig. 2b boxplots: Pearson/Spearman/cosine/RMSE/MAE/Moran's I improve vs. Gamma-shrinkage downsampled. | CRC-P2 Visium HD top-300 genes; mouse brain in supp. | ⭐⭐ — only vs. "do nothing"; no denoising-method baseline (MAGIC, scVI, kNN-on-H&E). |
| C2 | Recovers coherent spatial patterns from technical noise in transcriptome-wide Visium HD. | Fig. 3a CDCA7/SDCBP2 visualization; clustering by 0-20%, 20-40%, 40-60% expression bins. | CRC-P2 only | ⭐ — visual only; the "reference" is adjacent Xenium of the same tissue block, not biological GT. |
| C3 | Generalizes across Xenium (gastric), Xenium 5K (breast), CosMx (kidney). | Figs. 4-5 marker-gene visualization + clustering vs. pathologist annotation. | Gastric n=1, breast n=1, kidney n=2 | ⭐⭐ — multi-platform breadth but n=1 per platform; no quantitative agreement metric; "clearer" is subjective. |
| C4 | Enables full-tissue prediction outside the Visium HD capture window. | Fig. 6a COL1A1 across the whole CRC-P2 slide; clusters resemble pathologist's tumor core / invasive front / mucosa. | CRC-P2 only | ⭐⭐ — visually compelling for one gene/sample, but the unmeasured region has no molecular GT; the claim is unfalsifiable as stated. |
| C5 | Generalizes across patients (out-of-sample prediction). | Fig. 6b-c: train on CRC-P2, predict on CRC-P1, CRC-P5; ANO7 "closely mirrors" the training pattern. | 3 CRC patients, same disease, same platform | ⭐⭐ — qualitative pattern match on selected genes; no per-gene quantitative agreement on held-out patients; no cross-disease test. |
| C6 | Outperforms joint Xenium + Visium HD integration for degraded post-Xenium VHD. | Fig. 7: Pixel2Gene-on-VHD clustering "more coherent" than joint Xenium + VHD clustering. | Single lung cancer sample | ⭐ — single sample, qualitative cluster comparison, no NMI/ARI vs. pathologist annotation in main text. |
| C7 | Robust to staining variability and morphological heterogeneity via superpixel + multiscale ViT. | None — asserted in Discussion. | — | (no evidence) |
| C8 | Architectural advantage over GHIST (segmentation-free, hierarchical, multi-platform). | Discussion text only; no head-to-head experiment. | — | (no evidence) |

**Honest read.** This is a *demonstration* paper, not a benchmark paper. Five structural concerns dominate:

1. **Denoising vs prediction is one model, not two stages.** The abstract frames reconstruction and prediction as separate capabilities; mechanistically it is plain MSE on a filtered training set, evaluated at different spatial locations. The "denoising" effect is exclusion-induced smoothing — superpixels with bottom-quantile UMI per cluster are kept out of training, so the model regresses them toward retained neighbors. That is conceptually clean but it is not a denoising method in the MAGIC / scVI sense.
2. **The "ground truth" for unmeasured regions is unverifiable.** All quantitative benchmarks use **synthetic Gamma-shrinkage downsampling** (mean 0.1, sd 0.03) applied only to the top-300 highly expressed genes on CRC-P2. Predicted-unmeasured-region accuracy (Fig. 6a) is judged against a pathologist H&E annotation — which the model already saw as input — or against adjacent Xenium of the same tissue block. **Never against a true held-out molecular ground truth.**
3. **No head-to-head baselines anywhere.** GHIST, BLEEP, sCellST, ST-Net, HisToGene, iStar, MISO, SpatialScope, THItoGene are discussed verbally in the Introduction and Discussion. None appears in any figure or table. For a method whose stated contribution is improving on histology-to-expression and ST denoising, this is the missing experiment.
4. **No ablations.** Nothing on the UMI quantile threshold, hierarchical-ViT vs flat encoder, per-gene vs joint model, MLP depth/width, or K-means cluster count for the filter.
5. **No variance, no significance tests, n=1 per non-CRC platform.** Out-of-sample is limited to 3 CRC patients on the same platform. The showcased genes (KRT8, SPARC, PLAC8, CDCA7, ACTA2, CD3D, EPCAM, KRT20, ERBB2, NPHS2, AQP2 ...) are all genes with strong known histological correlates; no test on genes whose expression is *not* expected to track morphology, and no reporting on the fraction of genes the method fails on — which is the standard failure mode of histology-to-expression models per BLEEP.

## Method & Architecture

![Pixel2Gene workflow: H&E plus filtered single-cell ST drive a hierarchical ViT feature extractor and per-gene MLPs deployed for in-sample enhancement, full-tissue prediction, and out-of-sample inference](/assets/images/paper/pixel2gene/fig_p035_01.png)
*Figure 1: Pixel2Gene workflow — H&E plus filtered single-cell ST feed a hierarchical ViT (iStar encoder, 579-dim per superpixel) and an independent 4-layer MLP per gene, deployed at three spatial locations using the same trained model.*

Step-by-step:

1. **Image rescaling and superpixel grid.** H&E rescaled so 1 px = 0.5 x 0.5 µm; tiled into 16 x 16 px ≈ 8 x 8 µm superpixels (single-cell scale). Padded so dimensions divide 256 for hierarchical extraction. ST coordinates co-rescaled; misaligned/overlapping Visium HD bins dropped.
2. **Expression alignment.** Visium HD: take the 8 µm bin directly. Xenium / Xenium 5K / CosMx: aggregate molecule counts within each superpixel via transformed transcript coordinates.
3. **Hierarchical histology features.** ViT extracts multi-scale features from local patches up through larger contexts; final per-superpixel feature vector length 579 (multi-level features + raw RGB), following iStar.
4. **Two-stage training-set filtering.**
   - Tissue mask: **HistoSweep** removes background, debris, folds, low-content regions via stain density + texture.
   - Profiling-quality mask: K-means cluster superpixels in feature space, then drop superpixels below a within-cluster UMI quantile (e.g. bottom 30 %).
5. **Per-gene MLP regression.** As above — independent model per gene, MSE on retained pairs only.
6. **Three inference modes, same model.** In-sample enhancement, full-tissue prediction, out-of-sample prediction.
7. **Optional cell-level estimates.** Weighted sum over superpixels overlapping a segmented cell, following iStar.
8. **Downstream clustering.** MiniBatch K-means + iterative hierarchical merging (merge clusters < 0.5 % of total superpixels).

## Experimental Results

The paper reports **no quantitative table**. All comparisons are boxplots or qualitative visualizations.

| Setting | Metrics | Baseline | Reference |
|---|---|---|---|
| **CRC-P2 Visium HD downsampling** | Pearson, Spearman, cosine, RMSE, MAE on top-300 genes; Moran's I @ 16/32/64 µm | Downsampled input only | Fig. 2b |
| CRC-P2 clustering | NMI vs. observed clustering; Sankey cluster correspondence | Observed vs. downsampled vs. enhanced | Fig. 2c |
| Mouse brain Visium HD | Same metrics | Downsampled input | Supp. Figs. 1, 2b |
| CRC-P2 transcriptome-wide | Visual + clustering across 0-20%, 20-40%, 40-60% expression bins | Observed VHD; adjacent Xenium as "reference" | Fig. 3 |
| Gastric cancer Xenium | Marker visualization, TLS detection, pathologist annotation overlap | Observed Xenium | Fig. 4a-c |
| Breast cancer Xenium 5K | Marker visualization + clustering | Observed Xenium 5K | Fig. 4d-e |
| Kidney CosMx (healthy + T2D) | Marker visualization + clustering + glomerulus identification | Observed CosMx | Fig. 5 |
| CRC full-tissue prediction | Visualization + clustering | None (no measurement in those regions) | Fig. 6a |
| CRC out-of-sample (P1, P5) | Visualization vs. observed Xenium/VHD on the same patient | Observed sample as "comparison" (not held-out molecular GT) | Fig. 6b-c |
| Post-Xenium lung Visium HD | Clustering vs. joint Xenium + VHD integration | Observed post-Xenium VHD; joint integration | Fig. 7 |

![CRC-P2 downsampling: KRT8, SPARC, PLAC8, SELENOP, TAGLN observed vs downsampled vs Pixel2Gene-enhanced](/assets/images/paper/pixel2gene/fig_p036_01.png)
*Figure 2a: CRC-P2 downsampling — five marker genes (KRT8, SPARC, PLAC8, SELENOP, TAGLN) observed vs. downsampled vs. Pixel2Gene-enhanced at 8 µm.*

![Pearson, Spearman, cosine, RMSE, MAE and Moran's I improvements after Pixel2Gene enhancement, with NMI and Sankey cluster alignment to pathologist annotation](/assets/images/paper/pixel2gene/fig_p036_02.png)
*Figure 2b-c: Boxplot improvements on top-300 highly expressed genes (Pearson, Spearman, cosine, RMSE, MAE, Moran's I @ 16/32/64 µm) and clustering vs. pathologist annotation with NMI and Sankey alignment.*

![Low-confidence genes CDCA7 and SDCBP2 on Visium HD CRC-P2 observed vs Pixel2Gene vs adjacent Xenium reference](/assets/images/paper/pixel2gene/fig_p037_01.png)
*Figure 3a: Low-confidence genes (CDCA7, SDCBP2) on Visium HD CRC-P2 — observed vs. Pixel2Gene vs. adjacent Xenium as a "reference" (not biological ground truth).*

![Unsupervised clustering stratified by expression tier 0-20%, 20-40%, 40-60% showing Pixel2Gene recovers structure that observed expression loses in mid and low tiers](/assets/images/paper/pixel2gene/fig_p037_02.png)
*Figure 3b: Unsupervised clustering stratified by expression bin (0-20%, 20-40%, 40-60%) — Pixel2Gene recovers structure that observed expression loses in mid/low tiers.*

![Gastric cancer Xenium with ACTA2 CD3D EPCAM KRT20 enhancement and clustering against pathologist annotation](/assets/images/paper/pixel2gene/fig_p038_01.png)
*Figure 4a-b: Gastric cancer Xenium — ACTA2/CD3D/EPCAM/KRT20 enhancement and clustering vs. pathologist annotation.*

![Xenium 5K breast cancer enhancement of ERBB2 CD44 CD4 MS4A1 plus tertiary lymphoid structure detection](/assets/images/paper/pixel2gene/fig_p038_02.png)
*Figure 4c-e: Xenium 5K breast cancer — ERBB2 / CD44 / CD4 / MS4A1 enhancement, clustering, and tertiary lymphoid structure detection.*

![Healthy kidney HK3039 CosMx sparse FOV coverage reconstructed and AQP2 TAGLN NPHS2 EMCN restored with glomerulus identification](/assets/images/paper/pixel2gene/fig_p039_01.png)
*Figure 5a-c: Healthy kidney HK3039 CosMx — sparse FOV coverage reconstructed; AQP2 / TAGLN / NPHS2 / EMCN restored; glomerulus identification from enhanced expression.*

![Diabetic kidney HK2844 CosMx where Pixel2Gene compensates for pathology-driven disorganization in raw CosMx](/assets/images/paper/pixel2gene/fig_p039_02.png)
*Figure 5d-f: Diabetic kidney HK2844 — Pixel2Gene compensates for pathology-driven disorganization in raw CosMx.*

![Visium HD CRC-P2 full-tissue prediction of COL1A1 and PIGR with clustering revealing tumor core invasive front and mucosa matching pathologist annotation](/assets/images/paper/pixel2gene/fig_p040_01.png)
*Figure 6a: Visium HD CRC-P2 full-tissue prediction of COL1A1 / PIGR — clustering reveals tumor core, invasive front, and mucosa matching pathologist annotation (which the model already saw as H&E input).*

![Cross-patient transfer: Xenium and Visium HD models trained on CRC-P2 applied to CRC-P1 and CRC-P5 for ANO7 and COL1A1](/assets/images/paper/pixel2gene/fig_p040_02.png)
*Figure 6b-c: Cross-patient transfer — Xenium and Visium HD models trained on CRC-P2 applied to CRC-P1 and CRC-P5 (ANO7, COL1A1). Qualitative pattern match only.*

![Post-Xenium lung Visium HD clustering: observed VHD vs joint Xenium plus VHD integration vs Pixel2Gene-only](/assets/images/paper/pixel2gene/fig_p041_01.png)
*Figure 7a: Post-Xenium lung Visium HD clustering — observed VHD vs. joint Xenium + VHD integration vs. Pixel2Gene-only on a single sample.*

![Per-region H&E and clustering comparison showing Pixel2Gene extrapolates beyond the Visium HD capture area where joint integration cannot](/assets/images/paper/pixel2gene/fig_p041_02.png)
*Figure 7b-d: Per-region H&E + clustering comparison — Pixel2Gene extrapolates beyond the Visium HD capture area where joint integration cannot.*

**Ablations / robustness.** None reported. There is no ablation of the UMI-quantile filtering threshold, the hierarchical vs. flat feature extractor, per-gene independent models vs. a joint multi-gene model, the MLP depth/width, or the K-means cluster count for the filtering mask. No variance/error-bar reporting beyond boxplot IQR over gene panels. No statistical significance tests for any improvement claim.

**Head-to-head with prior methods.** None. Pixel2Gene is never benchmarked against ST-Net, HisToGene, Hist2ST, BLEEP, GHIST, sCellST, iStar, MISO, SpatialScope, or THItoGene in any quantitative table or figure.

## Limitations

Authors acknowledge that performance depends on training-set quality and representativeness, that no real ground truth exists for unprofiled regions, and that benchmarks rely on synthetic downsampling that may not match real-world degradation. Listed future work: better training-set selection, self-supervised / multimodal objectives, active sampling.

Beyond what the authors flag:

- **No baseline comparisons** with any prior histology-to-expression model (BLEEP, ST-Net, HisToGene, Hist2ST, GHIST, iStar, MISO, SpatialScope, THItoGene, sCellST). For a method whose entire pitch is improving on these, this is a load-bearing omission.
- **No ablations** on the UMI-quantile cutoff, K-means cluster count, hierarchical ViT vs. flat features, per-gene vs. joint MLPs, or MLP capacity.
- **Per-gene independent MLPs.** For 5,000-gene Xenium Prime that is 5,000 trained models per sample; no scalability / compute numbers (GPU hours, training time, memory).
- **No variance reporting** across reruns; no per-gene Pearson distribution beyond a boxplot summary; no statistical significance test for any "improvement" claim.
- **CRC-dominant generalization story.** Out-of-sample is tested only across 3 CRC patients on the same platform; cross-disease transfer (e.g. train on CRC, test on gastric) is not attempted — which is the more interesting generalization claim.
- **Selection bias in showcased genes.** Every highlighted gene (KRT8, SPARC, PLAC8, CDCA7, SDCBP2, ACTA2, CD3D, EPCAM, KRT20, ERBB2, AQP2, NPHS2, COL1A1 ...) has a strong known histological correlate. No test on genes whose expression is *not* expected to track morphology; no reporting on what fraction of genes the method fails on.
- **HistoSweep dependency** (same group's preprint) is not evaluated against alternative tissue masks.
- **n=1 per non-CRC platform** (gastric n=1, breast n=1, lung n=1, kidney n=2).
- **The unmeasured-region "ground truth" problem.** Full-tissue accuracy (Fig. 6a) is judged against a pathologist H&E annotation that the model already saw, and out-of-sample accuracy (Fig. 6b-c) against observed expression of the same patient — never against a held-out molecular ground truth. The strongest claims of the paper are unfalsifiable as evaluated.

## Why It Matters for Medical AI

If accepted as a *capability demonstration*, Pixel2Gene is the kind of pragmatic tool clinical translational labs actually want: an H&E-only inference path to whole-slide spatial molecular maps across Visium HD, Xenium, Xenium 5K, and CosMx, with newly released kidney CosMx data attached. The per-gene MLP is trivially trainable on a single GPU, the iStar feature extractor is open-source, and the three deployment modes cover the three things you actually do with ST data clinically (denoise what you measured, extrapolate the slide, transfer to new patients).

If read as a *benchmark paper*, it does not yet support its stronger claims. The absence of head-to-head comparison with GHIST / BLEEP / iStar / HisToGene means we do not know whether Pixel2Gene is the right histology-to-expression model — only that it is *a* histology-to-expression model that beats doing nothing on synthetic downsampling. The reasonable path forward for the field is to fold Pixel2Gene into the standard benchmark suite alongside those methods on real held-out molecular ground truth (e.g. cross-tissue-section validation, or train on Visium HD and validate against held-out Xenium of the same block), with ablations on the filter threshold and a joint multi-gene variant.

## References

- Yao S., Schroeder A., Jiang S., Im S., Park J.H., Dumoulin B., Hwang T.H., Susztak K., Li M. *Pixel2Gene enables histology-guided reconstruction and prediction of spatial gene expression.* bioRxiv, 23 Feb 2026. DOI: [10.64898/2026.02.21.707168](https://doi.org/10.64898/2026.02.21.707168)
- Code: <https://github.com/yaosicong1999/Pixel2Gene>
- Zhang D. *et al.* iStar: Inferring super-resolution tissue architecture by integrating spatial transcriptomics with histology. *Nature Biotechnology*, 2024. — source of the hierarchical ViT feature extractor reused here.
- Schroeder A. *et al.* HistoSweep (preprint, 2026, same group) — tissue mask used in step 4.
- Related prior work referenced verbally but **not benchmarked**: ST-Net (He 2020), HisToGene (Pang 2021), Hist2ST (Zeng 2022), BLEEP (Xie 2023), GHIST (Wang 2024), MISO (2024), SpatialScope (Wan 2024), THItoGene (2024), sCellST (2024).
