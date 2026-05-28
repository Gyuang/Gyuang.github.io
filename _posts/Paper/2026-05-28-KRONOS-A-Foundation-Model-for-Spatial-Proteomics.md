---
title: "KRONOS: A Foundation Model for Spatial Proteomics"
excerpt: "A ViT-S/16 trained with DINO-v2 on 47M single-marker fluorescence patches that handles arbitrary multiplex panels via a shared per-marker tokenizer and a non-learnable sinusoidal marker encoding — ablating the marker encoding alone costs ~37 points of balanced accuracy."
categories:
  - Paper
  - Spatial-Proteomics
  - Pathology
permalink: /paper/kronos/
tags:
  - KRONOS
  - Spatial-Proteomics
  - Foundation-Model
  - DINO-v2
  - Self-Supervised
  - CODEX
  - Multiplex-Imaging
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- KRONOS is a ViT-S/16 (~22M params) pretrained with DINO-v2 on **SPM-47M** — 47M single-marker patches drawn from 3.67M multiplex 256x256 patches, 175 markers, 16 tissues, 8 fluorescence platforms (CODEX, CellDive, COMET, MxIF, Orion, IBEX, Vectra, CosMx). It is the first general-purpose backbone that ingests arbitrary, variable multiplex marker panels through a single shared tokenizer.
- The load-bearing trick is a **non-learnable sinusoidal marker encoding** added per channel: ablating it drops cHL balanced accuracy from **0.689 to 0.337** (-37 pts) and DLBCL-1 from **0.686 to 0.315**. Token size, overlap, and CLS-vs-marker-average pooling are near-interchangeable by comparison.
- Headline numbers: DLBCL-2 9-class cell phenotyping balanced accuracy **0.7969 +/- 0.0125** (UNI 0.5511, DINO-v2 0.2980, CA-MAE 0.5503); prostate tumour vs non-tumour AUC **0.91** vs 0.68 mean-marker baseline; ccRCC ICI response AUC **0.7895**. The "treatment response" numbers are real but rest on n=14-28 patients with CIs over random splits, not patients — read with caution.

## Motivation

Spatial proteomics platforms (CODEX, MIBI, IBEX, COMET, MxIF, ...) produce tissue images with tens to >100 protein channels per slide. The standard analysis pipeline is brittle: (i) cell segmentation with Mesmer or Cellpose, then (ii) per-marker mean-intensity thresholding for phenotyping. Markers are treated independently, spatial context is discarded, and artefacts in crowded tissue propagate downstream. Existing vision foundation models (DINO-v2, UNI, Virchow) are pinned to 3-channel RGB; the few channel-agnostic backbones (CA-MAE, ChAda-ViT) were pretrained on cell-profiling data (RxRx, JUMP-CP) that looks nothing like multiplex tissue. The medical-AI motivation is concrete — spatial proteomics underlies TME analysis, immune-infiltration biomarkers, and ICI response prediction — but per-cohort datasets are tiny and panels are inconsistent across labs, so an annotation-efficient, panel-agnostic backbone fills a real gap.

## Core Innovation

- **Shared per-marker tokenizer.** A single 16x16 conv patch-embed is applied independently to each marker channel, producing N spatial tokens per each of M markers. Weight sharing across markers is what lets the model handle unseen markers at inference.
- **Sinusoidal marker encoding (non-learnable).** Each marker index j gets a fixed sinusoidal vector m_j; because it is data-independent, any new marker at test time just maps to an unused index. This is the single most important design choice (-37 pts when removed).
- **Learnable 2D spatial encoding with interpolation.** Standard ViT positional encoding, interpolated to accept arbitrary input sizes (64x64 cell crops, 256x256 patches, 96x96 um regions).
- **DINO-v2 SSL with a memory hack.** Each training image is sub-sampled to **3 channels** — DAPI + 2 random protein markers — purely to fit on 8x A100-80GB with FSDP. The model never sees a full 50-plex patch during pretraining; all higher-order marker interactions are extrapolated from triples.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | First foundation model for fluorescence spatial proteomics that handles variable marker panels and unseen markers | Architecture (shared conv + sinusoidal marker encoding); ablation Fig S6B (+37.4 pts from marker encoding) | Architecture + cHL/DLBCL ablation | ⭐⭐⭐ |
| C2 | KRONOS outperforms DINO-v2 / UNI / CA-MAE on cell phenotyping | Fig 2A, Table S6: 0.7358 / 0.7402 / 0.7969 vs best baseline 0.62 / 0.51 / 0.55 across 3 cohorts | cHL, DLBCL-1, DLBCL-2 | ⭐⭐⭐ |
| C3 | KRONOS generalises across cohorts | Cross-dataset transfer Fig 2D: 0.79 / 0.75 bal-acc on held-out cohort, both directions | DLBCL-1 <-> DLBCL-2 only | ⭐⭐ — only two cohorts with matched panels, both CODEX DLBCL from related collaborators; cross-platform transfer not tested |
| C4 | KRONOS is label-efficient (few-shot, human-guided, zero-shot) | Fig 2E (100 vs 1000), Fig S3B/C, ARI clustering Fig S3D | cHL, DLBCL-1, DLBCL-2 | ⭐⭐⭐ — 100-shot KRONOS beats 1000-shot baselines on all three cohorts |
| C5 | Segmentation-free patch-level phenotyping is a new useful paradigm | Fig 3B: KRONOS beats marker-expression baseline at 5 thresholds; visual map Fig 3C | DLBCL-1 only | ⭐⭐ — paradigm is real but the comparison is to a **non-spatial mean-intensity baseline**, not Mesmer+linear-probe / MAPS / CellSighter. At the clean threshold (100%) the gap is 0.72 vs 0.69 — modest. |
| C6 | KRONOS predicts treatment response better than DINO-v2 / UNI | Fig 5B-D: ccRCC AUC 0.79, CTCL-B 0.65, CTCL-P 0.59 | 3 cohorts, n=14-28 patients | ⭐ — small cohorts; CIs are over 100 random 80/20 splits, **not over patients**; no comparison to the original Phillips 2021 anti-PD-1 motif analysis or Schurch 2020 CLR/DII pipeline. CTCL-P AUC 0.59 on n=14 patients is sold as "state-of-the-art" in the abstract |
| C7 | KRONOS captures biologically meaningful tissue organisation | Fig 4B cluster maps match germinal centres / mantle / T-cell zones; CRC clusters C1/C4 enriched by Mann-Whitney p <= 0.001 | cHL, tonsil, FL, CRC | ⭐⭐ — qualitatively convincing but authors judge cluster-histology correspondence themselves |
| C8 | KRONOS enables image reverse-search across institutions | Fig 6C: EBV+/- case retrieval acc 0.82; Fig 6D 3/4 cross-dataset queries match EBV status | 29 EBV-annotated cores total | ⭐⭐ — directionally right, "image search engine" framing is aspirational at this n |
| C9 | KRONOS mitigates technical batch effects | Silhouette 0.0124 vs 0.0608 across 276 pairs (Fig S7) | 24-section serial dataset | ⭐⭐ — quantitative but single tissue source |
| C10 | Marker-aware encoding is essential | Fig S6B: 0.689 -> 0.337 (cHL), 0.686 -> 0.315 (DLBCL-1) without it | cHL, DLBCL-1 | ⭐⭐⭐ |
| C11 | KRONOS achieves SOTA across 11 independent cohorts | Six tasks x multiple cohorts; KRONOS best in all reported tables | 11 cohorts | ⭐⭐ — "11 cohorts" really means **"best of 3 generic backbones (UNI, DINO-v2, CA-MAE) on 11 cohorts"**. No comparison to MAPS, CellSighter, ChAda-ViT, or the original cohort papers' bespoke predictors. |

**Honest read.** The mechanical contributions are genuinely strong: the shared per-marker tokenizer plus fixed sinusoidal marker encoding plus DINO-v2 on SPM-47M produce a backbone whose cell-phenotyping, cross-dataset, few-shot, and ablation results (C1, C2, C3 within DLBCL, C4, C10) hold up cleanly. Three caveats deserve to be read alongside the abstract:

1. **"Segmentation-free" is real-paradigm-but-oversold.** The head-to-head comparison is to a non-spatial mean-marker baseline (0.72 vs 0.69 at the strictest threshold), not to Mesmer+linear-probe, MAPS, or CellSighter. The cell-phenotyping main pipeline still uses Mesmer-derived cell masks to crop cells; "segmentation-free" applies only to the Fig 3 patch experiment.
2. **Therapy response is the weakest part of the paper.** n=14 (CTCL-P) / 27 (ccRCC) / 28 (CTCL-B); confidence intervals are over 100 random splits, not over patients; no comparison to the original Phillips 2021 spatial-motif analysis on CTCL-P or to Schurch 2020 CLR/DII on CRC. An AUC of 0.59 on n=14 patients labelled "SOTA" in the abstract is a stretch.
3. **"11 cohorts SOTA" decodes to "beat 3 generic backbones on 11 cohorts."** Task-specific SOTA models (MAPS on cHL phenotyping, CellSighter, ChAda-ViT, Wenckstern et al. 2025 virtual tissues, and the bespoke per-cohort predictors from the original studies) are not compared.

Curation and access caveats worth flagging up front: **IMC and MIBI (ion-based modalities) are excluded** from pretraining — stated as "may require architectural adjustments" but no empirical experiment is run, so this is a curation choice rather than a justified architectural limit; IMC/MIBI dominate cohorts like Bodenmiller's Basel breast-cancer atlas and Angelo/Keren's TNBC. Pretraining data is **44% private + partial-access ImmunoAtlas tracks** (UFlorida 38.1%, BIDMC 36.8%, Stanford 5.4% together = 80% of patches), so SPM-47M cannot be reproduced externally. Weights are released "for academic research purposes" only, with a provisional patent filed with MGB and senior authors holding equity in Modella AI / Elucidate Bio.

## Method & Architecture

![KRONOS overview: SPM-47M dataset and ViT-S/16 with per-marker tokenizer plus sinusoidal marker encoding](/assets/images/paper/kronos/page_005.png)
*Figure 1: KRONOS overview — SPM-47M pretraining corpus (47M single-marker patches, 3.67M multiplex patches, 175 markers, 8 fluorescence platforms) and ViT-S/16 with shared per-marker tokenizer + non-learnable sinusoidal marker encoding + learnable 2D spatial encoding.*

The model treats a multiplex image as X in R^{HxWxM} with M variable per study (3 to 58 during pretraining). Each channel is tiled into N = (H/P)(W/P) non-overlapping PxP tokens (P=16). A single shared conv patch-embed f_embed produces $\bar{x}^j_i = f_{\text{embed}}(x^j_i) \in \mathbb{R}^D$ for each (marker j, position i). The final token combines three terms:

$$z^j_i = \bar{x}^j_i + m_j + \phi_i$$

where m_j is the **non-learnable sinusoidal marker encoding** with $m_{j,d} = \sin(j / 10000^{2d/D})$ and $m_{j,d+D/2} = \cos(j / 10000^{2d/D})$, and phi_i is a learnable 2D spatial encoding interpolated to arbitrary input sizes. The N*M tokens plus a CLS token go through 12 standard Transformer blocks. Three output views are extracted: (a) CLS token for patch-level summaries, (b) marker-averaged features $\bar{z}^j = (1/N)\sum_i \bar{z}^j_i$ concatenated to an M*D vector (default for downstream), (c) token-specific features for dense prediction.

**Pretraining.** DINO-v2 (self-distillation CLS loss + iBOT-style masked-image-modelling) with EMA teacher and multi-crop (global + local + masked-global). The released "KRONOS" is the teacher. Each training image is reduced to 3 channels (DAPI + 2 random markers) as a GPU-memory compromise — the model **never sees a full 50-plex patch during SSL**, which leaves all higher-order marker co-occurrences to extrapolation. 125k iterations, batch 1024, 8x A100-80GB with FSDP. Stratified sampling across 23 tissue-x-technology subsets handles the 38.1%/36.8%/5.4% UFlorida/BIDMC/Stanford imbalance and the CODEX(45.9%)/MIBI(0%) platform skew.

![DINO-v2 pretraining pipeline with channel-wise token embedding](/assets/images/paper/kronos/fig_p047_01.png)
*Figure S2: Conventional segmentation -> mean-expression -> clustering pipeline (top) versus KRONOS workflows (bottom) — cell-, patch-, and case-level analyses share a single backbone.*

## Experimental Results

### Cell phenotyping (linear probe, 4-fold, 1,000 cells/class)

| Dataset (classes, cells) | KRONOS | UNI | DINO-v2 | CA-MAE |
|---|---|---|---|---|
| cHL (16-class, 134k) bal-acc | **0.7358 +/- 0.0089** | 0.5570 +/- 0.0136 | 0.6210 +/- 0.0121 | 0.5331 +/- 0.0123 |
| DLBCL-1 (9-class, 361k) bal-acc | **0.7402 +/- 0.0309** | 0.5077 +/- 0.0333 | 0.2664 +/- 0.0201 | 0.5041 +/- 0.0314 |
| DLBCL-2 (9-class, 1.1M) bal-acc | **0.7969 +/- 0.0125** | 0.5511 +/- 0.0377 | 0.2980 +/- 0.0226 | 0.5503 +/- 0.0368 |
| DLBCL-2 avg precision | **0.8007 +/- 0.0462** | 0.4985 +/- 0.0244 | 0.2432 +/- 0.0103 | 0.4946 +/- 0.0300 |

![Cell phenotyping comparison: KRONOS vs UNI / DINO-v2 / CA-MAE on cHL and DLBCL with predicted vs ground-truth cell maps](/assets/images/paper/kronos/fig_p008_01.png)
*Figure 2: Cell-phenotyping benchmark on cHL / DLBCL-1 / DLBCL-2. KRONOS (orange) beats UNI, DINO-v2, and CA-MAE at every cohort; the right panel shows predicted cell-type map versus expert annotation on a representative DLBCL-1 region.*

### Cross-dataset transfer

| Direction | KRONOS | UNI | DINO-v2 | CA-MAE |
|---|---|---|---|---|
| DLBCL-1 -> DLBCL-2 bal-acc | **0.7896 +/- 0.0072** | 0.5629 +/- 0.0108 | 0.2928 +/- 0.0022 | 0.5651 +/- 0.0074 |
| DLBCL-2 -> DLBCL-1 bal-acc | **0.7505 +/- 0.0100** | 0.5164 +/- 0.0074 | 0.2632 +/- 0.0021 | 0.5116 +/- 0.0086 |

Both DLBCL cohorts are CODEX with overlapping panels and overlapping senior authors — true cross-platform transfer (e.g., CODEX -> MxIF) is not measured.

### Few-shot (DLBCL-2, 100 cells/class)

| Method | bal-acc @ 100 cells/class |
|---|---|
| **KRONOS** | **0.765** |
| UNI | 0.550 |
| CA-MAE | 0.504 |
| DINO-v2 | 0.298 |

KRONOS with 100 samples/class beats every baseline trained with 1,000 — the cleanest label-efficiency result in the paper.

### Segmentation-free patch phenotyping (DLBCL-1, 32x32 px patches ~16 um)

| Majority-class threshold | KRONOS bal-acc | Mean marker expression |
|---|---|---|
| 20% | 0.5510 +/- 0.0538 | 0.4704 +/- 0.0469 |
| 100% | **0.7183 +/- 0.0759** | 0.6885 +/- 0.0759 |

The framing ("no segmentation needed") is bolder than the magnitude (~3 pts at the clean end), and the comparator is a non-spatial mean-intensity baseline rather than Mesmer+linear-probe or MAPS.

![Segmentation-free patch phenotyping, prostate region classification, and artefact detection](/assets/images/paper/kronos/fig_p012_10.png)
*Figure 3: Segmentation-free patch phenotyping (B,C), prostate tumour vs non-tumour region classification (AUC 0.91 vs 0.68 mean-marker baseline), and artefact detection on DAPI (precision 0.98 vs 0.74).*

### Region and artefact detection

- Prostate tumour vs non-tumour AUC: **KRONOS 0.91 +/- 0.03** vs marker-expression 0.68 +/- 0.07 (3 channels: DAPI + AR + cytokeratin).
- DAPI artefact detection (3-class, DLBCL-1): mean precision **0.98 +/- 0.03** vs baseline 0.74 +/- 0.02.

### CRC patient stratification (CLR vs DII, K-means + kNN)

| K | KRONOS | DINO-v2 | UNI | CA-MAE |
|---|---|---|---|---|
| 8 | **0.7584 +/- 0.0162** | 0.6564 +/- 0.0159 | 0.6665 +/- 0.0191 | 0.6952 +/- 0.0157 |

![CRC patient stratification with K-means on KRONOS embeddings](/assets/images/paper/kronos/fig_p014_09.png)
*Figure 4C-E: CRC patient stratification — KRONOS cluster distributions separate Crohn-like Reaction (CLR) from Diffuse Inflammatory Infiltration (DII) at bal-acc 0.758 (K=8); four clusters reach Mann-Whitney p <= 0.001.*

### Therapy-response prediction (MIL, PCA-64/256)

| Dataset (R/NR, n patients) | KRONOS AUC | DINO-v2 | UNI |
|---|---|---|---|
| ccRCC (PCA-256, n=27) | **0.7895 +/- 0.0123** | 0.6890 +/- 0.0157 | 0.6723 +/- 0.0158 |
| CTCL-B (PCA-64, n=28) | **0.6500 +/- 0.0525** | 0.5487 +/- 0.0602 | 0.5750 +/- 0.0576 |
| CTCL-P (PCA-64, n=14) | **0.5900 +/- 0.0793** | 0.4400 +/- 0.0850 | 0.3550 +/- 0.0804 |

CIs are over 100 random 80/20 splits, not over patients; no comparison to Phillips 2021 spatial-motif analysis (CTCL-P) or Schurch 2020 CLR/DII pipeline (CRC). The ccRCC gap is the strongest of the three.

### Ablations (cHL / DLBCL-1, balanced accuracy)

| Choice | cHL | DLBCL-1 |
|---|---|---|
| No marker encoding | 0.315 | 0.337 |
| **+ sinusoidal marker encoding (default)** | **0.689** | **0.686** |
| 16x16 tokens, no overlap (default) | 0.689 | 0.686 |
| 4x4 tokens, no overlap | 0.683 | 0.712 |
| 16x16 tokens, 50% overlap | 0.688 | 0.713 |
| CLS-token (image) embedding | 0.688 | 0.713 |
| **Marker-specific embedding (default)** | **0.736** | **0.740** |
| 12.5k iters | 0.502 | 0.600 |
| 125k iters | 0.689 | 0.686 |

The marker-encoding row is the single most consequential result in the paper — without it, KRONOS collapses to near-random on these tasks.

### Batch-effect mitigation

Silhouette score across 276 pairs of differently-processed serial sections: KRONOS **0.0124 +/- 0.0307** vs mean-marker-expression 0.0608 +/- 0.0386 (lower = less batch clustering). Only one tissue source, so "less affected by stain/HIER variation" is the calibrated claim, not platform-invariance.

## Limitations

**Acknowledged by the authors.** IMC and MIBI (ion-based modalities) are excluded from pretraining and listed as future work; pretraining uses only 3 channels per image as a memory compromise; longitudinal / multimodal extensions deferred.

**Not addressed but worth flagging.**

- **3-channel pretraining bottleneck.** All inter-marker context during SSL is restricted to triples. The model never sees a full 50-plex patch; cross-marker attention beyond 3-way interactions is implicitly extrapolated. The paper does not analyse whether marker triples cover the marker co-occurrence graph adequately.
- **Single-patient cHL.** The most-cited phenotyping benchmark is one patient's tissue; 4-fold cross-val is across spatial quadrants of the same FOV. Cell types are not patient-disjoint.
- **Private data dominate pretraining.** 44% in-house plus partial-access ImmunoAtlas tracks — three of the five largest sources (UFlorida 38.1%, BIDMC 36.8%, Stanford 5.4%) sum to ~80% of patches. SPM-47M cannot be replicated externally.
- **Patent + commercial entanglement.** Code/weights released "for academic research purposes" only; provisional patent filed with Mass General Brigham; senior authors hold equity in Modella AI and Elucidate Bio.
- **No comparison to task-specific SOTA.** MAPS on cHL, CellSighter, Schurch 2020's CLR/DII pipeline, Phillips 2021's anti-PD-1 motif on CTCL-P — none appear as baselines. The comparison set is the same three generic backbones (UNI, DINO-v2, CA-MAE) across all 11 cohorts.
- **No paired statistical tests** on the headline phenotyping comparisons — only mean +/- std across 4 folds.
- **CTCL-P n=14 patients.** AUC 0.59 +/- 0.08 reported as "significantly higher" with no test statistic; CIs are over random splits, not patients.
- **"Segmentation-free" oversells.** The main cell-phenotyping pipeline still uses Mesmer-derived cell masks for cell-crop extraction; segmentation-free applies only to the Fig 3 patch experiment, and even there the baseline is not a segmentation-based one.
- **IMC/MIBI exclusion is curation, not architecture.** Stated as needing "architectural adjustments" but no experiment is run to support that.
- **DLBCL-2 cell-count typo.** Page 32 reports "31,098,443 cells" for DLBCL-2 while the main text consistently writes ~1.1M — likely transcription error worth checking against Supplementary Table S3.

## Why It Matters for Medical AI

Spatial proteomics is rapidly becoming a substrate for TME analysis, immune-cell-state mapping, and ICI response prediction in oncology, but per-cohort datasets remain tiny (tens of patients) and panels are inconsistent across labs. A panel-agnostic backbone that transfers across tissues with linear probes and beats specialist generalist backbones at few-shot phenotyping is genuinely useful for translational workflows — particularly the artefact-detection, batch-correction, and reverse-search applications, which are the under-sold parts of the paper. The therapy-response framing should be read in the context of n=14-28 patients and no published-baseline comparisons; the cell-phenotyping, region-classification, and few-shot results are the parts most likely to hold up in independent replication. For groups working with CODEX / CellDive / COMET / MxIF panels (i.e. fluorescence multiplex, not IMC/MIBI), KRONOS embeddings are a reasonable drop-in for downstream linear probes — provided you can live with the academic-only licence.

## References

- Paper (arXiv 2506.03373v1): <https://arxiv.org/abs/2506.03373>
- Code / weights (academic research only): <https://github.com/mahmoodlab/KRONOS>
- DINO-v2 (Oquab et al., 2023): <https://arxiv.org/abs/2304.07193>
- UNI pathology foundation model (Chen et al., 2024, Nat Med): <https://www.nature.com/articles/s41591-024-02857-3>
- CA-MAE channel-agnostic masked autoencoder (Kraus et al., 2024): <https://arxiv.org/abs/2404.11399>
- ChAda-ViT channel-adaptive ViT (Bourriez et al., 2024): <https://arxiv.org/abs/2311.15264>
- MAPS spatial-proteomics cell phenotyping (Shaban et al., 2024): <https://www.nature.com/articles/s41467-024-46989-z>
- Schurch et al. 2020, CRC CLR/DII spatial neighbourhoods (Cell): <https://www.cell.com/cell/fulltext/S0092-8674(20)30870-9>
- Phillips et al. 2021, CTCL anti-PD-1 spatial motif: <https://www.nature.com/articles/s41591-021-01521-4>
