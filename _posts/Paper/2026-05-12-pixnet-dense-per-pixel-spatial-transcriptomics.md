---
title: "From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images"
excerpt: "PixNet reframes ST prediction as dense per-pixel regression, hitting PCC@M 0.325 on breast Visium HD vs. SGN's 0.226 (+43.8%) and being the only method that survives 100 µm → 2 µm scale shift (PCC 0.198 vs. iStar 0.126)."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/pixnet-dense-per-pixel-spatial-transcriptomics/
tags:
  - PixNet
  - Spatial-Transcriptomics
  - Dense-Prediction
  - UNI2-h
  - Visium-HD
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

![PixNet hero — spots vs. pixels reformulation](/assets/images/paper/spatially-dense-ge/page_001.png)
*Figure 1: PixNet reframes spatial gene expression prediction. Top — spot-level baselines crop a fixed-resolution H&E patch and regress one expression vector per spot. Bottom — PixNet predicts a dense per-pixel expression map and aggregates within an arbitrary disk to score any region.*

## TL;DR

- PixNet abandons the "one expression vector per fixed-size spot" template that every prior method (ST-Net → SGN → MERGE) inherited, and instead **decodes a dense per-pixel H×W×M gene expression map** from a WSI, with a differentiable disk-aggregation operator that lets training spot radius differ from inference spot radius.
- The architecture is a UNI2-h pyramidal encoder feeding a hybrid decoder (depth-to-space upsampling in deep stages, bilinear in shallow stages) fused by Separable Attention Fusion Blocks, supervised only at the spots where ground truth was measured.
- Headline numbers: **breast Visium HD PCC@M 0.325 vs. SGN 0.226 (+43.8%)**, Her2ST 0.453 vs. TRIPLEX 0.404 (+12.1%), and — the real story — **train on 100 µm STNet, test on 2 µm Visium HD: PCC 0.198 vs. iStar 0.126 (+57.1%)**, the only method whose accuracy survives the scale shift.

## Motivation

Spatial transcriptomics is the bottleneck biology wants AI to solve: it is expensive, low-throughput, and limited by the spotting platform's physical resolution. A long line of work has tried to predict expression directly from H&E images: ST-Net (2020), HisToGene (2021), Hist2ST (2022), BLEEP (NeurIPS 2023), TRIPLEX (CVPR 2024), EGGN (PR 2024), SGN (MICCAI 2024), MERGE (CVPR 2025), ScstGCN (2025), iStar (Nat Biotech 2024). They all share a primitive: crop a fixed-size H&E patch around a spot center, regress one M-dim expression vector.

Two consequences make that primitive painful in 2025–2026.

First, a 100 µm Visium spot contains many cells of multiple types — the spot's expression is already an in-platform average. A model trained against that label can never see cellular heterogeneity below 100 µm; the output granularity is wrong for cell-type-aware diagnostics.

Second, the model is geometrically locked to the spot size it was trained on. **Visium HD** — the platform that actually matters going forward — delivers 2 µm bins (roughly single-cell), about 20 px in image space. Every spot-level baseline tested in this paper collapses to PCC < 0.14 on 2 µm Visium HD when trained at 100 µm. The crop-and-regress recipe simply does not transfer.

PixNet's response is architectural: move the prediction primitive from "crop" to "pixel," and let the user pick the integration window at inference.

## Core Innovation

The reframing is the contribution. Given an H×W H&E slide, predict $G \in \mathbb{R}^{H \times W \times M}$ — a dense expression map — and define a spot's predicted expression as the **sum of $G$ over the disk** of the spot's center and radius:

$$\hat y_n = \sum_{(\Delta x, \Delta y)\,:\,(\Delta x - x_n)^2 + (\Delta y - y_n)^2 \le r_n^2} G(\Delta x, \Delta y)$$

This single equation is the pivot. It is differentiable, so the model trains end-to-end with the standard MSE + PCC loss applied at measured spots only. And because $r_n$ is a free parameter at inference time, a model trained at 100 µm radii can be queried with 2 µm radii on a different platform — exactly the cross-scale transfer that spot-cropping methods structurally cannot do.

The rest of the architecture is the dense-prediction machinery needed to make a high-resolution $G$ feasible from a pathology foundation encoder: U-Net-style hybrid upsampling and a separable-attention fusion block. Both are standard ingredients in segmentation; the novelty is putting them in service of a spatial-transcriptomics primitive nobody had reformulated before.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|
| C1: Recast ST prediction as a dense per-pixel task with disk aggregation. | Method §3, Eq. 9 — architectural by construction. | — | ⭐⭐⭐ |
| C2: Beats SOTA on 4 ST datasets across all 5 metrics. | Tables 1–2 vs. 13 baselines, 5-rerun ±std. | STNet, Her2ST, breast Visium HD, mouse-brain Visium HD | ⭐⭐⭐ |
| C3: +43.8% PCC@M over previous best on breast Visium HD. | Table 1: 0.325 vs. SGN 0.226. | Breast Visium HD | ⭐⭐ (only 2 WSIs in that dataset) |
| C4: +12.1% PCC@M over TRIPLEX on Her2ST. | Table 2: 0.453 vs. 0.404. | Her2ST (36 WSIs) | ⭐⭐ |
| C5: 100 µm → 2 µm transfer: PCC 0.198 vs. iStar 0.126 (+57.1%). | Table 5; every spot baseline < 0.14. | STNet → breast Visium HD | ⭐⭐ (single direction; 2 WSI target) |
| C6: Sparse supervision suffices for dense prediction. | §3.4 loss; no contrast with a dense-imputation baseline. | — | ⭐ |
| C7: Multi-scale training helps. | Table 6: all-three-sizes 0.325 > 16+8 µm 0.303 > 2 µm-only 0.299. | Breast Visium HD | ⭐⭐ |
| C8: SAFB > ResNet18 > ViT > Conv decoder. | Table 3: 0.325 / 0.213 / 0.188 / 0.169 PCC@M. | Breast Visium HD | ⭐⭐ (ViT < ResNet18 is unexpected, unexplained) |
| C9: UNI2-h is the best foundation encoder. | Figure 4 bar chart only — no numeric table. | Breast Visium HD | ⭐ |
| C10: PCC is the diagnostically meaningful metric. | Field-standard framing; PCC@M is the headline. | All | ⭐⭐ |

The two ⭐⭐⭐ claims (dense reformulation + multi-dataset SOTA) are well-supported. The two ⭐ claims (sparse supervision and encoder choice) are the weakest links: neither is contrasted against a competing strategy with full numeric reporting.

## Method & Architecture

![PixNet architecture](/assets/images/paper/spatially-dense-ge/page_003.png)
*Figure 2: PixNet pipeline. A UNI2-h pyramidal encoder produces multi-level features $F_l$; the decoder upsamples with depth-to-space (sub-pixel) blocks in deep stages and bilinear interpolation in shallow stages; Separable Attention Fusion Blocks fuse skip-connected encoder features with upsampled decoder features via depthwise-conv refinement and 1×1 Q/K/V attention. The final 1×1 conv yields $G \in \mathbb{R}^{H \times W \times M}$, and Eq. 9 sums $G$ over each spot's disk to compute the supervised prediction.*

**Encoder.** UNI2-h, a ViT-L pathology foundation model (Chen et al., Nat Med 2024). Intermediate transformer groups (e.g., layers 2, 4, 6) are tapped to produce L levels of pyramidal feature maps $F_l$.

**Decoder — two upsamplers by depth.** The authors split the upsampling strategy by stage:

- *Deep stages — Depth-to-Space Upsampling Block (DSUB).* Sub-pixel convolution (Shi et al. 2016): compute $K = C_{F_l} \cdot 2^d$ filters with $d=2$, then D2S reshapes channels into higher spatial resolution. Argument: D2S preserves the spatial information learned by the encoder, which they argue dominates expression-prediction quality.
- *Shallow stages — bilinear interpolation.* Shallow features already carry fine spatial detail; D2S there over-transforms them, so a conv-block sandwich with bilinear upsample is used instead.

**Fusion — SAFB.** Each level fuses the upsampled decoder feature $U_{l-1}$ (or its bilinear variant $\hat U_{l-1}$) with the skip-connected encoder feature $F_{l-1}$. The encoder feature is first residually refined with a depthwise conv + SiLU + LN + BN block; the refined feature is concatenated with the decoder feature, projected to Q/K/V via 1×1 conv, and self-attention is applied. The fused output becomes $D_{l-1}$.

**Output head and supervision.** A 1×1 conv on $D_1$ yields $G \in \mathbb{R}^{H \times W \times M}$ with $M = 250$ (top-250 mean-expression genes per dataset, following ST-Net). Loss:

$$\mathcal{L} = \mathcal{L}_{\text{mse}} + \lambda \cdot \mathcal{L}_{\text{pcc}}, \quad \lambda = 0.5$$

where $\mathcal{L}_{\text{pcc}}$ is a batch-wise Pearson correlation loss. The MSE+PCC combination is ablated in Table 4 and beats either alone.

**Training.** Single NVIDIA RTX A6000, AdamW lr $5 \times 10^{-4}$, weight decay $10^{-4}$, 200 epochs, seed 42, trained from scratch (decoder), with the foundation encoder presumably initialised from UNI2-h weights (the text does not state freezing). Decoder channels $[64, 128, 256, 512, 512, 512]$. Per-spot total-count normalisation + log transform follow TRIPLEX. Every experiment averaged over 5 reruns with reported standard deviations.

## Experimental Results

### Main comparison

**Breast cancer Visium HD (Table 1)** — best in bold:

| Method | MSE ↓ | MAE ↓ | PCC@F ↑ | PCC@S ↑ | PCC@M ↑ |
|---|---|---|---|---|---|
| STNet | 0.269±0.03 | 0.482±0.05 | -0.009±0.05 | -0.001±0.06 | -0.001±0.06 |
| HistoGene | 0.265±0.02 | 0.438±0.05 | 0.089±0.07 | 0.159±0.10 | 0.157±0.10 |
| EGN | 0.264±0.03 | 0.413±0.03 | 0.102±0.03 | 0.166±0.10 | 0.161±0.08 |
| EGGN | 0.241±0.02 | 0.423±0.05 | 0.143±0.02 | 0.209±0.05 | 0.200±0.06 |
| BLEEP | 0.247±0.05 | 0.435±0.05 | 0.157±0.03 | 0.216±0.08 | 0.205±0.08 |
| HGGEP | 0.267±0.10 | 0.446±0.12 | 0.118±0.03 | 0.191±0.03 | 0.187±0.04 |
| Junayed et al. | 0.306±0.06 | 0.511±0.08 | 0.047±0.10 | 0.133±0.12 | 0.134±0.12 |
| TRIPLEX | 0.259±0.04 | 0.432±0.08 | 0.122±0.03 | 0.200±0.04 | 0.199±0.07 |
| iStar | 0.285±0.03 | 0.479±0.06 | 0.119±0.05 | 0.213±0.02 | 0.216±0.03 |
| SGN | 0.230±0.03 | 0.358±0.06 | 0.173±0.07 | 0.227±0.04 | 0.226±0.02 |
| BG-TRIPLEX | 0.255±0.03 | 0.448±0.07 | 0.109±0.03 | 0.187±0.04 | 0.190±0.06 |
| ScstGCN | 0.229±0.03 | 0.352±0.04 | 0.159±0.08 | 0.231±0.10 | 0.225±0.14 |
| MERGE | 0.290±0.06 | 0.458±0.03 | 0.125±0.07 | 0.227±0.04 | 0.221±0.02 |
| **PixNet (Ours)** | **0.153±0.02** | **0.274±0.05** | **0.196±0.02** | **0.313±0.03** | **0.325±0.02** |

PixNet wins all five metrics by margins that exceed reported standard deviations against every baseline. On the brain-cancer (mouse brain) Visium HD split, the same pattern holds: PCC@M 0.304 vs. SGN 0.195. On STNet (100 µm breast): PCC@M 0.409 vs. BG-TRIPLEX 0.357. On Her2ST: PCC@M 0.453 vs. TRIPLEX 0.404.

### Cross-scale generalization (the key result)

Train on STNet (100 µm); test on breast cancer Visium HD at 2 / 8 / 16 µm:

| Method | 2µm PCC@M | 8µm PCC@M | 16µm PCC@M |
|---|---|---|---|
| STNet | 0.000 | 0.002 | 0.005 |
| HistoGene | 0.088 | 0.093 | 0.097 |
| EGN | 0.091 | 0.087 | 0.100 |
| EGGN | 0.087 | 0.088 | 0.102 |
| BLEEP | 0.115 | 0.127 | 0.125 |
| TRIPLEX | 0.092 | 0.100 | 0.099 |
| iStar | 0.126 | 0.133 | 0.137 |
| SGN | 0.118 | 0.136 | 0.123 |
| MERGE | 0.112 | 0.127 | 0.139 |
| **PixNet** | **0.198** | **0.219** | **0.226** |

This is the table that justifies the dense-prediction reformulation. Every spot-level method tops out below PCC@M 0.14 at 2 µm; PixNet sits at 0.198 — a +57.1% relative improvement over iStar, the next best. It is also the one structural property of the architecture that competing methods cannot match by simply swapping encoders.

### Qualitative

![Qualitative comparison on cancer-related genes](/assets/images/paper/spatially-dense-ge/page_007.png)
*Figure 3: Predicted vs. ground-truth gene expression for APP and FASN at 2 µm Visium HD and XBP1 and GNAS at 100 µm STNet — all cancer-related markers. PixNet produces tissue-coherent maps at single-cell resolution; SGN's spot-wise predictions remain blocky.*

### Ablations

- **Decoder architecture (Table 3, breast Visium HD).** Plain Conv 0.169 → ResNet18 0.213 → ViT 0.188 → SAFB **0.325** PCC@M. The ViT decoder underperforms ResNet18 — striking and unexplained.
- **Loss (Table 4).** MSE-only 0.293; PCC-only 0.319; combined **0.325**. Pearson loss alone already beats MSE alone, confirming PCC is the limiting signal.
- **Foundation encoder (Figure 4).** UNI2-h beats ResNet18, CLIP, Virchow2, H-Optimus-0, and UNI — but the comparison is reported only as a bar chart; exact deltas are not in the text.
- **Training spot size (Table 6, breast Visium HD).** 16 µm only: 0.244; 8 µm only: 0.288; 2 µm only: 0.299; 16+8 µm: 0.303; all three: **0.325**. Multi-scale matters, and 2 µm-only is already near the ceiling.

![Foundation encoder ablation — MSE](/assets/images/paper/spatially-dense-ge/page_008.png)
*Figure 4a: MSE across foundation encoders on breast Visium HD.*

## Limitations

**Author-acknowledged.** The dense map is still supervised only at measured spots; the paper proposes large-scale self-supervised WSI pretraining as future work.

**Unaddressed in the paper:**

- **Tiny biological sample size on Visium HD.** Both Visium HD datasets have only 2 WSIs each. The massive spot counts (18.7M, 14.2M) make the standard deviations look tight, but they measure intra-slide spot variability, not inter-patient variance. A two-slide test set cannot estimate generalisation across patients.
- **Foundation-encoder ablation is graphical only.** Figure 4 reports no numbers for UNI2-h vs. UNI vs. H-Optimus-0 vs. Virchow2. For a SOTA claim that depends on encoder choice, this is unusually thin.
- **No significance tests.** No paired t-test against SGN/TRIPLEX, no confidence intervals beyond ±std over 5 reruns.
- **ViT decoder < ResNet18 decoder.** This is counter-intuitive and unexplained; no hyperparameter sweep is shown for the ViT variant.
- **Inference-time radius generalisation is never isolated.** Eq. 9 lets $r_n$ at inference differ from $r_n$ at training, but Table 5 conflates radius change with platform change (STNet → Visium HD has different staining, scanners, and tissue). A controlled within-Visium-HD radius-swap ablation is absent.
- **Memory cost of the dense map at WSI scale.** Generating $G \in \mathbb{R}^{H \times W \times 250}$ on 50K×50K slides is not trivial; tiling strategy and memory footprint are not discussed.
- **Code is promised but not released as of v4.** Thirteen baselines were "retrained in the same local setup" but exact configs/seeds are not in the paper.
- **Cross-tissue generalisation is not tested.** Every train/test pair is in-tissue; breast → brain (or vice versa) is never attempted.

## Why It Matters for Medical AI

The cell-type-aware diagnostics that pathology AI keeps promising require sub-spot resolution. Visium HD is the platform that delivers it, and spot-cropping methods cannot use it. PixNet is the first paper that breaks the "one crop, one expression vector" assumption and shows the resulting model **transfers across the scale shift that matters for clinical adoption** — 100 µm research data → 2 µm Visium HD. If the dense-prediction primitive holds up under more rigorous evaluation (more WSIs, statistical tests, code release), it is a candidate architectural shift for the entire ST-from-histology subfield, not just a tactical SOTA bump.

## References

- Paper: *From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images*, Zhang, Yang, Pan. arXiv:2503.01347v4 (25 Nov 2025).
- Code: announced as forthcoming; no repository URL in v4.
- Foundation encoder: Chen et al., *Towards a General-Purpose Foundation Model for Computational Pathology* (UNI / UNI2), Nat Med 2024.
- Closest spot-level baselines: SGN (MICCAI 2024), TRIPLEX (CVPR 2024), MERGE (CVPR 2025), iStar (Nat Biotech 2024), BLEEP (NeurIPS 2023).
- Dense-prediction precedents: Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016 (depth-to-space).
- Datasets: STNet (He et al., Nat Biomed Eng 2020), Her2ST (Andersson et al., bioRxiv 2020), 10x Visium HD Breast Cancer and Mouse Brain.
