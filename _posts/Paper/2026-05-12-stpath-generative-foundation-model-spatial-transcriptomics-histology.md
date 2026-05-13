---
title: "STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images"
excerpt: "A geometry-aware Transformer pretrained by masked gene-expression regression on 983 WSIs reaches PCC 0.266 on HEST-Bench (+6.9% over UNI, +34.4% over TRIPLEX), with in-context prompting adding +0.257 absolute PCC on CCRCC at 5% prompted spots."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/stpath-generative-foundation-model-spatial-transcriptomics-histology/
tags:
  - STPath
  - Spatial-Transcriptomics
  - Foundation-Model
  - Computational-Pathology
  - Masked-Modeling
  - Equivariant-Transformer
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- STPath is a **generative foundation model** that predicts spatially resolved gene expression from H&E whole-slide images (WSIs) without per-dataset fine-tuning, pretrained on **983 WSIs across 17 organs, 4 ST technologies, and a 38,984-gene vocabulary**.
- Each spot becomes a multi-modal token (Gigapath visual embedding + multi-hot gene vector + organ/tech metadata); the slide is processed by an **E(2)-invariant frame-averaging spatial Transformer** trained with **masked gene-expression regression** under tailored slide/mask/gene schedules.
- On HEST-Bench (10 cancer cohorts, top-50 HVGs, zero-shot), STPath reaches **average PCC 0.266 vs. 0.198 for TRIPLEX (+34.4%) and ~0.249 for UNI (+6.9%)**, with **+0.257 absolute PCC on CCRCC using only 5% prompted spots**. Slide-level downstream gains (survival C-index +5.6–6.1%, mutation AUC +5.0–5.6%) are real but rely on concatenation with Gigapath/UNI features and lack per-cohort variance reporting.

## Motivation

Spatial transcriptomics (ST) maps gene expression onto tissue morphology and is increasingly central to characterizing the tumor microenvironment. The assays, however, are low-throughput, expensive, and patchy in coverage. Prior image-to-ST methods (BLEEP, TRIPLEX, HEST-Bench probes on UNI/Gigapath) train one model per organ or per gene panel, so they cannot generalize across the 17 organs, 4 ST technologies, and tens of thousands of gene symbols that compose the public WSI–ST corpus.

STPath argues for a single generative foundation model — pretrained on the union of HEST-1K + STImage-1k4m — that can (i) zero-shot predict ST for new slides from H&E alone, (ii) use a few labeled spots as in-context prompts to improve hard cohorts, and (iii) feed predicted ST back into downstream pathology tasks (survival, mutation status) where slide-level labels are spatially diffuse.

## Core Innovation

- **Generative multi-modal tokenization of a slide.** Each ST spot is one token built from three tracks: a Gigapath patch embedding, a learned bag-of-genes representation over a 38,984-gene vocabulary, and learned organ/sequencing-technology metadata embeddings.
- **E(2)-invariant frame-averaging spatial attention.** Rather than positional encodings, STPath averages projections of pairwise spot coordinates over a PCA-derived frame set, producing an attention bias invariant to rotations, reflections, and translations of slide coordinates.
- **Masked gene-expression regression with three custom schedules.** A continuous spot region (size Uniform(64, 1024)) is sampled per step; the mask ratio is drawn from `Beta(10, 1)` so ~90% of spots are masked; 80% of the loss is over highly variable genes (HVGs), 20% over all measured genes; the sequencing-tech embedding is dropped 80% of the time to handle missing-tech inference.
- **A single forward pass handles three regimes.** Zero-shot (all spots masked), in-context prompting (some spots visible), and feature extraction for downstream weakly-supervised tasks (predicted biomarker expression fed into ABMIL).

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|-------|----------|----------|----------|
| C1 | STPath is the **first** foundation model for WSI ↔ ST integration with a generative pretraining paradigm. | Conceptual; baselines (UNI, Gigapath, TRIPLEX, BLEEP) are all single-task or discriminative. A concurrent flow-matching paper by the same authors is cited but not benchmarked. | n/a | ⭐⭐ |
| C2 | STPath outperforms baselines on HEST-Bench by **+6.9% PCC over UNI** and **+14.4% AMI over TRIPLEX**, with p < 0.001 (two-sided Wilcoxon). | Figure 2(c, d); per-cohort PCC via Appendix Table 4; bootstrap CIs; **wins 8/10 on PCC, 9/10 on AMI**. | HEST-Bench (10 cohorts) | ⭐⭐⭐ |
| C3 | In-context prompting with as few as **5% spots** yields large gains, e.g., **+0.257 absolute PCC on CCRCC** (0.035 → 0.292). | Figure 2(g); Appendix Table 4 full per-cohort × per-prompt-% sweep with 95% CIs. | HEST-Bench | ⭐⭐⭐ |
| C4 | STPath-predicted ST clusters align with pathologist annotations better than baselines on 5 annotated cohorts (+0.121 AMI on brain, +29.6% on a 10x sample). | Figure 3 bar plots + qualitative cluster maps. | Maynard brain, GSE213688, Chen, Andersson, Erickson (42 samples) | ⭐⭐ (no numeric table; 1 cohort has only 1 slide) |
| C5 | STPath improves slide-level **survival** prediction by **+6.1% C-index (UNI)** and **+5.6% (Gigapath)** averaged over 4 cohorts; HNSC UNI 0.665 → 0.771. | Figure 5(b); 7/8 settings improved. | MBC, SURGEN, HNSC, LUAD | ⭐⭐ (no per-cohort error bars in main text) |
| C6 | STPath improves slide-level **gene mutation** prediction by **+5.6% AUC (UNI)** and **+5.0% (Gigapath)** over 24 cohort×biomarker settings (gains in 18/24). | Figure 5(c). | CPTAC BRCA/COAD/LUAD/GBM/LSCC/HNSC × 8 biomarkers | ⭐⭐ |
| C7 | STPath enables **unsupervised** mutation prediction via spot-variance of predicted biomarker expression — TP53-BRCA AUC 0.73; **+13.9% avg** over best baseline. | Figure 5(d). | CPTAC BRCA, COAD, LUAD | ⭐⭐ (only 5 (cohort,biomarker) combos; PIK3CA-BRCA AUC 0.60 is near chance) |
| C8 | The geometry-aware (E(2)-invariant) Transformer is a key design enabler. | Section 4.2 derives the architecture; **no ablation removes the spatial bias.** | — | ⭐ |
| C9 | The `Beta(10, 1)` high-mask-rate schedule outperforms low-mask BERT-style (15%) masking. | Section 4.3 motivates it analytically; **no empirical mask-rate sweep.** | — | ⭐ |
| C10 | STPath generalizes across organs, genes, and sequencing technologies. | 17 organs, 38,984 genes, 4 ST techs in pretraining. **But zero-shot PCC is strong on only 3/10 HEST-Bench cohorts (IDC 0.751, SKCM 0.602, LUNG 0.559) and weak on 6/10 (PRAD 0.298, READ 0.300, CCRCC 0.035, HCC 0.104, LYMPH_IDC 0.070).** | HEST-Bench | ⭐⭐ |

**Honest read.** The gene-expression and in-context prompting claims (C2, C3) are unusually well-supported for a preprint — per-cohort numbers, bootstrap CIs, Wilcoxon p-values, and a harder no-finetune protocol than HEST-Bench's leave-one-out setting. These earn the 3-star rating.

The downstream-task claims (C5–C7) are weaker. The comparison is "Gigapath/UNI + STPath features vs. Gigapath/UNI alone" — really an evaluation of whether STPath's predicted expression adds information on top of an already-strong PFM, not a head-to-head with other ST-prediction foundation models. The 5–6% C-index/AUC gains are real but modest, and Figure 5 omits per-cohort confidence intervals. The unsupervised TP53-by-variance trick (C7) is clever but rests on 5 positive examples with PIK3CA-BRCA barely above chance.

The "first foundation model for WSI–ST" framing (C1) is partially undermined by the authors' own concurrent flow-matching paper, which is cited but not benchmarked. **The architecture/schedule claims (C8, C9) receive zero ablation evidence — the single most fixable gap in the manuscript.** The 1-star rating on these is deliberate and is retained from the analyzer's calibration; readers should not treat the geometry-aware attention or the `Beta(10, 1)` mask schedule as empirically validated design choices yet.

## Method & Architecture

![STPath pretraining overview with multi-track spot tokenization and masked-gene-expression objective](/assets/images/paper/stpath_2025.04.19.649665/page_003.png)
*Figure 1: STPath pretraining overview — multi-track spot tokenization (Gigapath image embedding, multi-hot gene vector over a 38,984-gene vocabulary, and organ/sequencing-tech metadata), geometry-aware spatial Transformer with frame-averaging coordinate bias, and masked gene-expression generative objective. Bottom inset shows the downstream applications.*

### Tokenization of a WSI as a spot sequence

A slide is segmented into `N` spots with coordinates `x_coord ∈ R^{N×2}`, spot images `x_img ∈ R^{N×3×H×W}`, and gene expression vectors `x_gene ∈ R^{N×G}` over a curated vocabulary of `G = 38,984` gene symbols. Each spot becomes one token via three projections summed together:

- **Visual:** `z_img = Linear(f_PFM(x_img))` with `f_PFM` = Gigapath patch encoder.
- **Gene:** `z_gene = x_gene · W_gene` — a log1p-normalized multi-hot vector multiplied by a learned matrix, i.e., a weighted bag-of-gene-embeddings.
- **Meta:** `z_meta = embed(x_organ) + embed(x_tech)` — learned organ and sequencing-technology embeddings.

The final spot token is `z = z_img + z_gene + z_meta`, with `z ∈ R^{N×d}` and `d = 512`.

### E(2)-invariant spatial attention

For each spot pair, the relative direction `x^{i→j}_coord = x_coord^{(i)} − x_coord^{(j)}` is computed. PCA on the pairwise direction cloud yields a frame set `F(x'_coord) = {[±u_1, ±u_2]}` (4 frames). Averaging linear projections of relative coordinates across this frame set produces a scalar bias `z_coord ∈ R^{N×N}` invariant to rotations, reflections, and translations of slide coordinates. Self-attention adds the bias:

$$
A = \mathrm{softmax}\!\left(\frac{z_Q z_K^\top}{\sqrt{d}} + z_{\mathrm{coord}}\right)
$$

### Masked gene-expression generative objective

A subset `m` of spots is masked, and the model regresses the gene expression of masked spots conditional on visible spots and metadata. Gene expression is continuous, so the loss is **MSE** (the paper writes the log-likelihood form in Eq. 9 but states MSE in implementation):

$$
\mathcal{L} = \frac{1}{|m|} \sum_{i \in m} \lVert x_{\mathrm{gene}}^{(i)} - \hat x_{\mathrm{gene}}^{(i)} \rVert^2
$$

### Three pretraining schedules (the load-bearing design)

- **Slide sampling schedule.** At each step, sample a continuous spot region of size `Uniform(64, 1024)` from the slide — both an efficiency mechanism and an augmentation.
- **Masked noise schedule.** Mask ratio drawn from `Beta(10, 1)` (~90% in expectation); 95% of masked spots receive a `[MASK]` embedding, 5% receive 50% gene dropout. The sequencing-tech embedding is replaced by a padding token 80% of the time to handle missing-tech at inference.
- **Gene target schedule.** 80% of the time, the loss is computed only over highly variable genes; 20% over all measured genes — designed to down-weight housekeeping genes.

### Training and inference modes

The model has **4 Transformer layers, hidden dim 512, 4 attention heads, GeLU, dropout 0.1**. Optimization: Adam, lr `5e-4`, grad-norm clip 1.0, batch size 2 WSIs, early stopping (20-epoch patience). Hardware: 8 × NVIDIA L40S-48GB; ~100 epochs over 928 WSIs (986 total, 5% held out). The same forward pass supports three regimes:

1. **Zero-shot** — mask all spots, predict from image + meta only.
2. **In-context spot imputation** — expose a few spots as prompts; predict the rest with the *same* forward pass.
3. **Downstream weakly-supervised** — STPath-predicted expressions for 14 biomarkers (PIK3CA, TP53, BAP1, PBRM1, KRAS, EGFR, CASP8, BRAF, GATA3, MYBPC1, ARID1A, KEAP1, STK11, SMAD4) are concatenated with Gigapath/UNI visual features and fed into an ABMIL head.

### Six downstream tasks evaluated

1. **Zero-shot gene-expression prediction** on HEST-Bench (10 cohorts, top-50 HVGs).
2. **In-context spot imputation** at 0/5/10/20/30/40/50% prompted spots.
3. **Spatial clustering** (Leiden on predicted expression) on 5 expert-annotated cohorts (42 samples).
4. **Biomarker correlation** for 6 cancer biomarkers across 4 HEST-Bench cohorts.
5. **Weakly-supervised survival and mutation prediction** via ABMIL with concatenated STPath features (8 CPTAC + MBC + SURGEN cohorts).
6. **Unsupervised mutation prediction** via spot-variance of predicted biomarker expression — no training at all.

## Experimental Results

### Gene expression prediction on HEST-Bench

![HEST-Bench gene expression prediction with in-context prompting curves](/assets/images/paper/stpath_2025.04.19.649665/page_004.png)
*Figure 2: HEST-Bench gene-expression prediction. (a/b) Evaluation protocol: per-fold finetuning (HEST-Bench original) vs. STPath's single-model zero-shot setup. (c, d) STPath wins 8/10 cohorts on PCC and 9/10 on AMI for top-50 HVGs. (g) In-context prompting curve: 5–50% prompted spots monotonically improves PCC, with the largest gains on hard cohorts (CCRCC, HCC, PRAD).*

Zero-shot PCC on top-50 HVGs (no per-dataset fine-tuning), reproduced from Figure 2(c) and Appendix Table 4 (0% prompt row).

| Method | Avg PCC | Wins (of 10) | IDC | SKCM | LUNG | PRAD | PAAD | COAD | READ | CCRCC | HCC | LYMPH_IDC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BLEEP | excluded ("pretraining not stable") | — | — | — | — | — | — | — | — | — | — | — |
| TRIPLEX | 0.198 | 0 | — | — | — | — | — | — | — | — | — | — |
| UNI | ~0.249 (next-best; +6.9% gap) | — | — | — | — | — | — | — | — | — | — | — |
| Gigapath | < UNI | — | — | — | — | — | — | — | — | — | — | — |
| **STPath (zero-shot)** | **0.266** | **8 / 10 (PCC); 9 / 10 (AMI)** | **0.751** | **0.602** | **0.559** | 0.298 | 0.471 | 0.521 | 0.300 | 0.035 | 0.104 | 0.070 |

The paper does not publish a side-by-side numeric table covering all baselines × all cohorts in the main text — only average gap statements (+6.9% PCC over UNI, +14.4% AMI over TRIPLEX, p < 0.001 two-sided Wilcoxon) and Figure 2(c, d) bar charts. The per-cohort STPath numbers above come from Appendix Table 4 (0% prompt row).

### In-context spot imputation (Appendix Table 4)

Averages over 10 HEST-Bench cohorts on masked spots:

| Prompt % | 0% | 5% | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|---|
| Avg PCC | 0.371 | 0.435 | 0.450 | 0.463 | 0.470 | 0.476 | 0.481 |
| Avg AMI | 0.526 | 0.538 | 0.536 | 0.546 | 0.547 | 0.554 | 0.558 |

Per-cohort highlights at 5% prompt: **CCRCC 0.035 → 0.292 (+0.257 absolute)**, HCC 0.104 → 0.179 (+72% relative), PRAD 0.298 → 0.386 (+29% relative). At 20% prompt: LYMPH_IDC 0.070 → 0.293 (+318%). The in-context capability is the most distinctive practical advantage of the generative paradigm here.

### Spatial clustering

![Spatial clustering on STPath-predicted expression vs. pathologist annotations](/assets/images/paper/stpath_2025.04.19.649665/page_005.png)
*Figure 3: Leiden clustering on STPath-predicted gene expression aligns with pathologist annotations across 5 cohorts — Maynard brain layers, GSE213688 breast subtypes, Chen tumor regions, Andersson HER2+ cell types, and Erickson prostate benign/malignant. One sample shown per cohort.*

STPath claims best AMI and Homogeneity on all 5 cohorts; the largest reported gaps are **+0.121 AMI on the human-brain (Maynard) cohort** and **+29.6% on a 10x sample**. No table of numerical scores per cohort is included in the manuscript body — only bar plots. The Andersson cohort averages only ~435 spots per slide, well below the others, so comparable AMI across organs partly reflects spot density rather than model quality.

### Biomarker correlation

![Per-biomarker PCC and marker-gene matrix plots from predicted ST](/assets/images/paper/stpath_2025.04.19.649665/page_006.png)
*Figure 4: (a) Per-biomarker PCC across 6 cancer biomarkers in 4 HEST-Bench cohorts; STPath beats best baseline by +9.3% on average (p < 0.001). (b–e) logFC matrix plots of top-10 marker genes recovered by Wilcoxon rank-sum testing on STPath-predicted ST in 4 annotated samples — recovers known markers including SPAG5, MMP11, GSDMD, CCL25, IFNA2.*

The headline finding is **PCC > 0.95 on GATA3 and MYBPC1 in IDC**. Both are top-ranked HVGs in those slides, so the very high correlation partly reflects strong morphology→gene coupling for canonical breast markers rather than a generic ability to predict any biomarker.

### Slide-level survival and mutation

![Survival C-index, mutation AUC, and unsupervised mutation via spot-variance](/assets/images/paper/stpath_2025.04.19.649665/page_007.png)
*Figure 5: (a) ABMIL with concatenated STPath ST features. (b) Survival C-index gains across 4 cohorts (HNSC: UNI 0.665 → 0.771). (c) Weakly-supervised mutation AUC across 24 settings, +5–6% avg. (d) Unsupervised mutation via spot-variance of predicted biomarker expression — TP53 BRCA AUC 0.73. (e, f) BRCA / COAD high-expression patch visualizations for predicted biomarkers.*

- **Survival (C-index).** 4 cohorts × 2 PFMs = 8 settings; gains in 7/8. Averages: **UNI +6.1%, Gigapath +5.6%**. HNSC standout: UNI 0.665 → 0.771; Gigapath 0.591 → 0.698.
- **Weakly-supervised mutation (AUC).** 24 cohort × biomarker settings; gains in 18/24. Averages: **UNI +5.6%, Gigapath +5.0%**.
- **Unsupervised mutation prediction.** Variance of STPath-predicted biomarker expression across spots used directly as a mutation score, no training: BRCA PIK3CA AUC 0.60; BRCA TP53 AUC 0.73; COAD TP53 AUC 0.73; COAD KRAS AUC 0.66; LUAD EGFR AUC 0.66. Average gain over best baseline: +13.9%. The 0.60 PIK3CA result is barely above chance and sits next to the headline 0.73 in the same figure.

### Ablations

The only ablation-adjacent analysis in the main text is the pretraining curve in Figure 2(f) showing STPath > UNI > Gigapath > TRIPLEX on validation PCC over training steps. **There is no formal ablation of (i) the frame-averaging spatial Transformer vs. plain self-attention, (ii) the `Beta(10, 1)` mask schedule vs. fixed-rate masking, (iii) the HVG-vs-all gene-target schedule, or (iv) removal of organ/tech meta embeddings.** This is the single biggest experimental gap and the basis for the ⭐ ratings on C8 and C9.

Variance reporting is uneven: 95% bootstrap CIs are present throughout Appendix Table 4 (gene-expression panels), but absent from Figures 3 and 5 in the main text.

## Limitations

**Acknowledged by the authors.**

- Zero-shot PCC remains low on under-represented organs (CCRCC explicitly), attributed to pretraining-data scarcity.
- Future work targets additional modalities such as scATAC-seq.

**Not addressed.**

- **No ablations.** Frame-averaging spatial attention, the `Beta(10, 1)` mask schedule, the HVG-vs-all gene-target schedule, the 80% sequencing-tech dropout, and the three-track input fusion are proposed without any experimental comparison to simpler alternatives.
- **No HEST-Bench leaderboard comparison.** STPath evaluates in a no-finetune zero-shot setting that is incomparable to HEST-Bench's official leave-one-out protocol — so "+6.9% over UNI" is non-trivially apples-to-oranges relative to the community benchmark.
- **No baseline for in-context prompting.** A trivial baseline like "average expression of k nearest visible spots" or kNN on Gigapath features is not reported, leaving open how much of the prompting gain is genuinely model-driven.
- **No external validation cohort.** All downstream tasks reuse CPTAC; no truly held-out hospital cohort.
- **No site/scanner-effect analysis.** Pathology foundation models are known to leak site identity; STPath's gains could partly stem from CPTAC-specific patterns already memorized by Gigapath.
- **Uneven variance reporting.** Bootstrap CIs exist for gene-expression panels but are missing from Figures 3 and 5.
- **BLEEP exclusion** ("pretraining not stable") is footnoted but never quantified.
- **Sub-50-gene focus.** All HEST-Bench metrics are on top-50 HVGs; the full 38,984-gene vocabulary is asserted as an advantage but never quantitatively evaluated.
- **Zero-shot generalization is uneven.** STPath is strong on 3/10 HEST-Bench cohorts (IDC, SKCM, LUNG) and weak on 6/10 (PRAD, READ, CCRCC, HCC, LYMPH_IDC, and partially COAD).

## Why It Matters for Medical AI

Image-only pathology foundation models hit a ceiling at the macroscale phenotypes their supervision signals describe. Spatial transcriptomics carries microscale molecular signal but is hard to scale clinically. STPath is a credible argument that a single generative model — pretrained across organs, technologies, and tens of thousands of genes — can predict ST from H&E well enough to serve as a feature extractor for downstream weak labels (survival, mutation status) and as a zero-shot ST imputer when only a handful of measured spots are available.

The strongest evidence is squarely on gene-expression PCC/AMI on HEST-Bench and the in-context prompting curve, where the protocol is harder than HEST-Bench's own leave-one-out setup and per-cohort bootstrap CIs are reported. The downstream story (survival/mutation) is more modest than the abstract suggests — these are concatenation studies on top of strong PFMs without per-cohort variance reporting — and the "first foundation model for WSI–ST" framing is partially undermined by the authors' own concurrent flow-matching work. Most importantly, **the architecture- and schedule-level claims (geometry-aware attention, `Beta(10, 1)` masking) are presented without ablations**, so practitioners cannot yet tell which of STPath's design choices matter.

## References

- Paper (bioRxiv preprint): <https://www.biorxiv.org/content/10.1101/2025.04.19.649665>
- Code: <https://github.com/Graph-and-Geometric-Learning/STPath>
- Venue: bioRxiv preprint (posted April 24, 2025); subsequently published as *npj Digital Medicine* 2025
- Related work:
  - HEST-Bench — Jaume et al., NeurIPS 2024
  - HEST-1K, STImage-1k4m — pretraining corpus sources
  - Gigapath — Xu et al., Nature 2024 (pathology foundation model used as patch encoder)
  - UNI — Chen et al., Nature Medicine 2024 (pathology foundation model baseline)
  - TRIPLEX, BLEEP — image-to-ST baselines
  - Patho-Bench — Vaidya et al. (downstream label source for CPTAC cohorts)
  - CPTAC — The Cancer Imaging Archive (downstream cohorts)
