---
title: "BLEEP: Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning"
excerpt: "A CLIP-style joint embedding of H&E patches and Visium spots lifts Pearson r on held-out liver to 0.217 (MG) / 0.175 (HEG) / 0.173 (HVG), beating ST-Net and HisToGene by 39-120% — but every number comes from one tissue and one donor cohort."
categories:
  - Paper
tags:
  - BLEEP
  - Contrastive-Learning
  - Spatial-Transcriptomics
  - Vision-Language
  - Retrieval
  - Visium
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- BLEEP reframes image-to-expression prediction as **query-reference imputation in a CLIP-style joint embedding** of H&E patches and Visium spot expression, rather than per-gene supervised regression. The 3,467-gene profile of a query patch is the **average of the K=50 nearest reference spot expressions** in the shared 256-dim space.
- A **similarity-smoothed contrastive loss** softens the standard CLIP target with intra-modal similarity, preventing biologically similar in-batch spots — common in repetitive tissue zonation — from being pushed apart.
- On a held-out human liver Visium slice, BLEEP attains Pearson **r = 0.217 +/- 0.002 (MG)**, **0.175 +/- 0.016 (HEG)**, **0.173 +/- 0.011 (HVG)** versus ST-Net (0.099 / 0.126 / 0.091) and HisToGene (0.097 / 0.072 / 0.071) — **+120% / +39% / +90%** over the second-best baseline.

## Motivation

Spatial transcriptomics (Visium, MERFISH, seqFISH+, Slide-seqV2) bridges histology and molecular profiling, but the platforms are low-throughput, expensive, and equipment-bound. H&E histology, by contrast, is ubiquitous in clinical practice. The natural goal is to predict spatially resolved expression directly from H&E. Prior supervised regression baselines — ST-Net, HisToGene, Hist2ST — suffer three pathologies:

1. **Ill-posed mapping.** Morphology cannot determine expression of every gene.
2. **Curse of dimensionality.** Predicting thousands of genes via per-gene heads collapses to mean expression, destroying variance.
3. **Visium artifact sensitivity.** Spot quality varies within and across samples, polluting regression targets.

BLEEP attacks these by switching the output primitive: instead of decoding a gene vector, retrieve real measured expression profiles from a contrastively aligned image-expression embedding and average them.

## Core Innovation

- **Retrieval as the output head.** Predictions are convex combinations of **real reference spot profiles**, so outputs are bounded to the data manifold — no decoder, no per-gene regression, and variance is inherited from the reference distribution.
- **Similarity-smoothed CLIP loss.** Replacing CLIP's identity target with a softmax over averaged intra-modal similarities reduces the penalty for in-batch spots that are genuinely close in either morphology or expression — important when tissue zonation produces many near-duplicate spots per batch.
- **ResNet50 image encoder beats ViT variants** in this small-data regime (Supplementary Table 1) — authors attribute the gap to ViT memorization.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | BLEEP outperforms HisToGene and ST-Net on MG / HEG / HVG correlation | Table 1: 0.217 / 0.175 / 0.173 vs. <= 0.126; 3 replicates with reported deviation; gaps far exceed deviations | Single held-out liver slice (slice #3 of 4) | ⭐⭐ |
| C2 | Improvements of 39-120% over the second-best method | Table 1 (120% on MG, 39% on HEG, 90% on HVG vs. ST-Net) | Same | ⭐⭐ |
| C3 | BLEEP preserves biological heterogeneity / variance better than regression baselines | Figure 2 (variance plots); Figure 3 (CYP3A4 spatial fidelity); Figure 4 (GGC heatmaps) | Same slice | ⭐⭐ |
| C4 | Query-reference imputation is the right design vs. supervised regression | Implicit via C1 + Table 3 row 3 (top-1 = 0.079 MG, terrible); no head-to-head with regression head on same encoder | Same slice | ⭐ |
| C5 | Smoothed (similarity-adjusted) contrastive objective helps | Table 3 last two rows: 0.217 vs. 0.209 (MG); 0.175 vs. 0.165 (HVG); 0.173 vs. 0.170 (HEG); within or near reported deviation on HEG | Same slice, 3 replicates | ⭐ |
| C6 | K=50 is a sensible default; K=10 hurts, higher K marginal | Table 3 K-sweep (10, 50, 100) | Same | ⭐⭐ |
| C7 | Robust to within-slice experimental artifacts (low-quality regions) | Figure 5: all methods reconstruct expression over red low-quality regions | Same | ⭐⭐ (but the property is shared by all image-based methods, not specific to BLEEP) |
| C8 | Robust to cross-sample batch effects | Supplementary Figure 1 + qualitative claim that predictions stay in-distribution because they are convex combinations of reference profiles | Same 4-slice cohort | ⭐ |
| C9 | BLEEP recapitulates and denoises gene-gene correlations | Figure 4 GGC heatmaps (qualitative) | Same | ⭐ (no quantitative GGC similarity metric reported; "denoising" vs. "over-smoothing" not adjudicated) |
| C10 | ResNet50 outperforms ViTs as image encoder | Supplementary Table 1 | Same | ⭐ (relies on supplement; small-data caveat, not externally validated) |
| C11 | First bi-modal embedding-based framework for expression prediction from histology | Related-works review; novelty claim | n/a | ⭐⭐ (plausible at NeurIPS 2023 — predecessors HE2RNA, ST-Net, HisToGene, Hist2ST are all regression-based) |
| C12 | Reduces time/cost of gene expression profiling for clinical/research use | Forward-looking; no cost/throughput experiment | n/a | ⭐ (aspirational) |

**Overall rating: ⭐⭐.** The numerical claims (C1, C2, C6) are well-supported *within the scope tested* — gaps of 0.217 vs. 0.099 with deviation <= 0.020 are not noise. But **every result is from a single tissue type (liver), single platform (Visium), single donor cohort, single held-out slice**. There is no cross-patient split, no cross-tissue evaluation (breast / brain / tumor), no cross-platform check. "3 replicates" means repeated training runs on the same split — it controls training stochasticity, not data variance. Absolute correlations remain low (0.17-0.22); the authors acknowledge this. The conceptually distinctive contribution is C3 — variance preservation as a free byproduct of retrieval-based imputation, with Figure 2 as the strongest evidence. C5 (smoothed loss) is borderline within reported uncertainty on HEG. C7's artifact-robustness is shared by any image-only method. C8 and C9 lean on qualitative figures with no quantitative metric (no Frobenius distance between predicted and reference GGC matrices, no quantitative batch-effect score, no p-values vs. baselines). A leave-one-patient-out evaluation plus at least one external dataset (e.g., the ST-Net breast cohort) would meaningfully change the rating.

## Method & Architecture

![BLEEP pipeline: contrastive joint embedding and query-reference imputation](/assets/images/paper/bleep/page_004.png)
*Figure 1: BLEEP pipeline — (a) contrastive joint embedding from paired H&E patches and Visium expression, (b) query patches projected and k-NN retrieved in the joint space, (c) imputation as a weighted average of retrieved reference expression profiles.*

### 1. Inputs and pairing

From each Visium spot, extract a **224x224 H&E patch** (roughly 55 µm physical side) centered on the spot, paired with that spot's normalized expression vector. Genes: the union of the top 1,000 highly variable genes per slice (Scanpy-computed), giving **3,467 genes total**. Per-spot total-count + log normalization; **Harmony** batch correction across the 4 slices. Slice #3 is held out for test; slices #1, #2, #4 form the reference.

### 2. Encoders

- **Image encoder $f_\text{img}$:** pretrained **ResNet50** + projection head -> 256-dim embedding $H_v$.
- **Expression encoder $f_\text{expr}$:** FCN that also serves as the projection head -> 256-dim embedding $H_x$.

### 3. Similarity-smoothed contrastive loss

For batch size $B = 512$:

$$
\text{sim}(H_v, H_x) = H_x H_v^T
$$

Intra-modal similarities $\text{sim}(H_v, H_v) = H_v H_v^T$ and $\text{sim}(H_x, H_x) = H_x H_x^T$. The soft target is

$$
\text{target} = \text{softmax}\Big(\frac{\text{sim}(H_x, H_x) + \text{sim}(H_v, H_v)}{2} \cdot \tau\Big)
$$

and the loss is the symmetric cross-entropy

$$
L = \text{mean}\big(\text{ce}(\text{sim}(H_v, H_x), \text{target}) + \text{ce}(\text{sim}(H_v, H_x)^T, \text{target}^T)\big).
$$

The smoothing prevents in-batch spots with similar morphology or expression — common in tissue with repetitive zonation — from being pushed apart, which would otherwise destroy retrieval quality. Borrowed from Shariatnia's "Simple CLIP" implementation.

### 4. Query-reference imputation (Algorithm 1)

1. Tile the query whole-slide H&E into patches $V'$.
2. Embed $Q'_v = f_\text{img}(V')$.
3. For each $q' \in Q'_v$, compute L2 distances to all reference image embeddings $H_v$.
4. Take the **top-K** nearest reference indices; retrieve their **expression** profiles $X[\text{indices}]$.
5. Predict expression as the **simple average** of those K profiles (default; alternatives ablated in Table 3).

### 5. Training setup

4x NVIDIA V100, AdamW, batch size 512, lr 1e-3, 150 epochs. Inference defaults: K = 50, smoothed objective, uniform averaging.

Crucially, **there is no decoder back to expression**. Reference expression vectors are the prediction primitives, so imputation is a linear combination of real measured profiles — predictions stay on the data manifold by construction.

## Experimental Results

### Main quantitative comparison (Table 1)

Held-out slice #3, average across 3 replicates with maximum deviation from mean.

| Method | MG (8 marker genes) | HEG (top-50 highly expressed) | HVG (top-50 highly variable) |
|--------|---------------------|-------------------------------|------------------------------|
| HisToGene | 0.097 +/- 0.015 | 0.072 +/- 0.018 | 0.071 +/- 0.011 |
| ST-Net | 0.099 +/- 0.020 | 0.126 +/- 0.005 | 0.091 +/- 0.007 |
| **BLEEP** | **0.217 +/- 0.002** | **0.175 +/- 0.016** | **0.173 +/- 0.011** |

Top-5 individual genes (Table 2, representative replicate). On canonical liver zonation markers BLEEP achieves CYP3A4 r = 0.741 (vs. 0.549 for HisToGene / ST-Net), CYP1A2 r = 0.681 (vs. 0.542 / 0.532), CYP2E1 r = 0.675 (vs. 0.330 / 0.530), GLUL r = 0.656 (vs. 0.488 / 0.463), FABP1 r = 0.503 (vs. 0.328 / nm). These are biologically meaningful — CYP3A4 / CYP1A2 / CYP2E1 / GLUL are pericentrally zonated; FABP1 is periportally zonated.

### Ablations (Table 3)

| Smoothed Obj. | K | Aggregation | MG | HVG | HEG |
|---------------|----|-------------|-----|-----|-----|
| Yes | 10 | average | 0.179 +/- 0.020 | 0.146 +/- 0.008 | 0.148 +/- 0.022 |
| Yes | 100 | average | 0.215 +/- 0.011 | 0.180 +/- 0.012 | 0.181 +/- 0.015 |
| Yes | - | simple (top-1) | 0.079 +/- 0.032 | 0.075 +/- 0.016 | 0.084 +/- 0.023 |
| Yes | 50 | weighted (sim-weighted avg) | 0.186 +/- 0.018 | 0.161 +/- 0.015 | 0.157 +/- 0.026 |
| No | 50 | average | 0.209 +/- 0.017 | 0.165 +/- 0.007 | 0.170 +/- 0.005 |
| **Yes (default)** | **50** | **average** | **0.217 +/- 0.002** | **0.175 +/- 0.016** | **0.173 +/- 0.011** |

Findings:

- **K=1 (top-1 retrieval) is catastrophic** — averaging is essential to denoise Visium intrinsic noise.
- **K=100 marginally improves HEG/HVG** but plateaus on MG and increases variance-flattening risk.
- **Similarity-weighted averaging is worse than uniform averaging** — counter-intuitive; the paper does not explain why. If more-similar neighbors are not more informative, what is being retrieved?
- **The smoothed objective gives a small but consistent gain** (0.217 vs. 0.209 on MG); on HEG the gain is within reported deviation.

### Variance preservation

![Predicted per-gene mean and variance vs. ranked gene index](/assets/images/paper/bleep/page_007.png)
*Figure 2: Per-gene predicted mean (upper) and per-gene predicted variance (lower) vs. ranked gene index. BLEEP roughly tracks the reference variance curve while HisToGene and ST-Net collapse to near-flat variance — i.e., the regression baselines learn mean as a shortcut.*

This is the conceptually distinctive evidence in the paper. Per-gene predicted variance is not something a supervised regressor optimizes for, but it is exactly the property a retrieval-based predictor inherits for free from its reference distribution.

### Spatial fidelity

![CYP3A4 predicted vs. original spatial maps](/assets/images/paper/bleep/page_007.png)
*Figure 3: CYP3A4 (pericentral marker) predicted vs. original — variable-scale (top) and fixed-scale (bottom) color maps overlaid on the held-out slice. BLEEP visibly recovers pericentral zonation; the regression baselines do not.*

### Gene-gene correlation structure

![Gene-gene correlation heatmaps for original vs. each method](/assets/images/paper/bleep/page_008.png)
*Figure 4: Gene-gene correlation heatmaps for original vs. BLEEP / HisToGene / ST-Net predictions. BLEEP qualitatively recapitulates and sharpens the block structure — the authors interpret this as denoising; an alternate read is over-smoothing toward the reference structure.*

No quantitative GGC similarity metric (e.g., Frobenius distance to the original) is reported, so the denoising-vs-over-smoothing dispute is unresolved.

### Artifact robustness

![Leiden cluster overlays for low-quality Visium regions](/assets/images/paper/bleep/page_009.png)
*Figure 5: Leiden clusters overlaid on H&E for original vs. each method. Image-based prediction reconstructs expression over low-quality (red) Visium regions for all methods — this is a property of the image-only task, not of BLEEP specifically.*

## Limitations

**Acknowledged:**

- Absolute correlations are low (0.17-0.22) — many genes are poorly correlated with morphology, and Visium itself under-detects many transcripts.
- BLEEP's predicted variance is underestimated relative to the original (Figure 2, lower row).
- Averaging may smooth away genuine abrupt biological signal, particularly for genes weakly tied to image features.
- Larger pretrained ViT encoders did worse than ResNet50 — attributed to memorization on a small training set.

**Not addressed but visible to a careful reader:**

- **No cross-patient / cross-cohort evaluation.** All 4 slices are from a single donor cohort and likely highly correlated; the train/test split is by slice not by subject, which inflates performance.
- **No cross-platform generalization.** Visium-only; transfer to Slide-seqV2, Stereo-seq, Visium HD, or in-situ methods (MERFISH / seqFISH+) is untested.
- **No same-encoder regression baseline.** A controlled "swap retrieval head for a linear regression head on the same ResNet50" ablation is missing; the regression-vs-retrieval comparison is confounded by encoder choice.
- **Similarity-weighted aggregation is worse than uniform averaging**, unexplained — undermines the "retrieval is meaningful" narrative.
- **No reference-size scaling curve.**
- **No per-prediction uncertainty quantification**, even though retrieval naturally provides one (distance to the K-th neighbor).
- **No statistical significance test** against ST-Net / HisToGene; gaps are clearly outside reported deviations but no p-values are given.
- **Computational cost of L2 search** at WSI scale or atlas scale is not discussed.

## Why It Matters for Medical AI

BLEEP is the canonical "CLIP-without-text" baseline for spatial transcriptomics: a clean demonstration that a CLIP-style joint embedding plus retrieval can substitute for supervised regression when the output modality has its own native vector space. This positions it as the natural baseline against which captioned-LLM bridge work (OmiCLIP, Loki, SpotWhisperer) should be measured — those methods add natural-language captions on top of essentially this paradigm. For clinical translation, the headline argument is decoupling spatial molecular insight from the cost of running Visium itself; but until cross-patient, cross-tissue, and cross-platform evaluations land, the result should be read as a single-tissue feasibility study, not a deployment-ready predictor.

## References

- Paper: [arXiv:2306.01859v2 — Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning](https://arxiv.org/abs/2306.01859) (NeurIPS 2023)
- Code: [bowang-lab/BLEEP](https://github.com/bowang-lab/BLEEP)
- Data: GEO accession [GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429)
- Related: ST-Net (He et al., *Nat. Biomed. Eng.*, 2020), HisToGene (Pang et al., 2021), Hist2ST (Zeng et al., 2022), HE2RNA (Schmauch et al., *Nat. Commun.*, 2020), CLIP (Radford et al., 2021), Shariatnia "Simple CLIP" implementation.
