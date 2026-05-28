---
title: "Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference"
excerpt: "SpaHGC wires target spots to reference-slide spots through a 3-edge heterogeneous graph (TS/CS/RS) and a complementary masked contrastive objective, reporting +4.71 to +9.36 PCC(%) over the strongest prior baseline on all 7 ST datasets."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/spahgc/
tags:
  - SpaHGC
  - Spatial-Transcriptomics
  - Heterogeneous-Graph
  - Contrastive-Learning
  - Cross-Slice-Transfer
  - UNI
  - GraphSAGE
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- SpaHGC (CVPR 2026) attacks the H&E to spatial-transcriptomics regression problem by giving the model access to *other* slides via a heterogeneous graph: target spots (UNI image embedding only) and reference spots (UNI image embedding **concatenated with ground-truth gene expression**) are wired together through three edge types — intra-target spatial (TS, K=5), cross-slice image-similarity (CS, K=7), and intra-reference joint-feature (RS, K=7). Expression literally leaks from reference to target through node features and CS edges feeding into a cross-node dual-attention encoder.
- The "masked" half of the name is **complementary feature-channel masking** (two views with binary masks satisfying $M_1+M_2=1$, ratios $\alpha=0.8$ target / $\beta=0.9$ reference), not node/edge dropout. The "contrastive" half is **BYOL/SimSiam-style cosine alignment with stop-grad** (no negatives), which is a slight terminology stretch.
- Headline number: **+4.71 to +9.36 PCC(%) over the strongest prior baseline on every one of seven datasets** (HER2+, cSCC, Alex, Visium BC, Lymph Node, Pancreas1, Pancreas2), with Wilcoxon signed-rank $p<0.001$ on all 63 pairwise per-slice comparisons. Ablations confirm CS edges are the most load-bearing component (−2.4 to −3.5 PCC when removed).

## Motivation

Spatial transcriptomics (10x Visium, legacy ST) answers "which genes are expressed where" in tissue, but a single slide costs upward of \$1k and the readout is sparse and noisy. Matched H&E is essentially free, so image-to-ST regressors have been an active line of work — STNet, HisToGene, Hist2ST, BLEEP, mclSTExp, M2OST. SpaHGC's diagnosis is that this entire literature has two blind spots:

1. **Single-slide tunnel vision.** Regression-family methods model intra-slide spatial structure with CNNs or local graphs and never look at other slides; this leaves cross-patient shared expression programs on the table.
2. **Flat retrieval pools.** CLIP-retrieval-family methods (BLEEP, mclSTExp) do see a reference bank, but treat it as a bag of contrastive negatives with no graph topology between reference spots.

The fix proposed here is a heterogeneous graph that explicitly wires a target spot to (a) its 2D spatial neighbours within its own slide, (b) its morphologically nearest reference-slide spots across the entire reference bank, and (c) the global semantic scaffold of the reference cohort. Crucially, **reference nodes carry the ground-truth gene expression as part of their node feature**, so cross-slice attention is the actual mechanism by which expression gets transferred to an unlabeled target spot.

## Core Innovation

There are three moves that distinguish SpaHGC from prior image-to-ST work:

1. **Asymmetric node design.** Target node $v_t^{(i)}$ carries only the image feature $z_t^{(i)} \in \mathbb{R}^{1024}$ (UNI). Reference node $v_r^{(j)}$ carries the concatenation $h_r^{(j)} = [z_r^{(j)} \,\|\, y_r^{(j)}] \in \mathbb{R}^{1024+M}$ of image feature and the known gene-expression vector. This is the leak channel.
2. **Three edge types in one heterogeneous graph.** TS (5 spatial KNN within target), CS (Top-7 image-cosine target-to-reference, across *all* reference slides), RS (Top-7 joint-feature within and across reference slides). The CS edges are the cross-patient knowledge channel; ablations confirm they are the most important edge type.
3. **Complementary feature-channel masking + cosine alignment.** Two views are generated with binary masks $M_t^{(1)}+M_t^{(2)}=1$ and $M_r^{(1)}+M_r^{(2)}=1$ — *features kept in view 1 are exactly those masked in view 2*. One view is run with stop-grad as a stable target; the consistency loss is $\mathcal{L}_\text{con} = \tfrac{1}{N}\sum_j(2 - 2\cos(\hat L^1_j, \hat L^2_j))$, computed separately for target and reference nodes. No InfoNCE, no explicit negatives — this is BYOL/SimSiam in graph form.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|
| C1: SpaHGC outperforms 9 SOTA methods on every dataset and metric | Table 1 (PCC + RMSE on all 7) + Supp. Table S2 (Wilcoxon $p<0.001$ on all 63 method × dataset comparisons) | All 7 | ⭐⭐⭐ |
| C2: Cross-slice (CS) edges are the most important inductive bias | Table 2 ablation: removing CS drops PCC by 2.4–3.5 (vs TS 1.3–2.4, RS 0.4–1.3) | Lymph Node, Pancreas1, Pancreas2 | ⭐⭐⭐ |
| C3: Complementary masking beats no-mask and random-mask | Table 4: 32.59 → 34.11 → 35.02 on Lymph Node | 3 datasets | ⭐⭐ |
| C4: CNDA and CNAP both contribute | Table 5: w/o CNDA −1.3 to −2.1 PCC, w/o CNAP −0.3 to −1.0 PCC | 3 datasets | ⭐⭐ |
| C5: Predictions enable better tumor-region clustering (ARI) | Supp. Figs S9–S10: ARI 0.48 vs 0.28 on Visium BC; 0.46 vs 0.43 on Alex | 2 datasets | ⭐⭐ on Visium BC, ⭐ on Alex (gap = 0.03) |
| C6: Predicted expression is *more biologically meaningful* than raw (higher KEGG/GO p-values) | Supp. Fig S11 on Alex only, no null-shuffle control | 1 dataset, qualitative | ⭐ |
| C7: Method enables knowledge transfer from reference to target slides | Headline narrative + CS-edge ablation; protocol is within-dataset LOOCV with no cross-cohort or leave-patient-out test | Same-dataset LOOCV only | ⭐⭐ |
| C8: Method is computationally efficient (fastest + 2nd-smallest model) | Supp. Fig S12: 0.17 h on Pancreas2 vs M2OST 10.5 h; 5.3 M params (only M2OST is smaller at 2.3 M) | cSCC, Visium BC, Pancreas2 | ⭐⭐⭐ |
| C9: Pathology-specific UNI is necessary for the gain | Table 3: UNI 24.48 vs ResNet18 20.25 on Pancreas1 — ~half the headline gain over M2OST disappears | 3 datasets | ⭐⭐ |

## Method & Architecture

![SpaHGC architecture — Figure 2 full pipeline](/assets/images/paper/spahgc/fig_p003_02.png)
*Figure 2: SpaHGC pipeline. (A) UNI-based patch encoding and heterogeneous graph construction with TS / CS / RS edges. (B) GraphSAGE + CNDA backbone driven by complementary feature-channel masking. (C) CNDA performs bidirectional cross-attention between target and reference nodes. (D) CNAP is a 4-head unidirectional attention pool from target queries to reference keys/values, with an MLP head emitting the predicted gene-expression vector. (E) Downstream analyses (clustering ARI, KEGG/GO enrichment).*

![Figure 1B — heterogeneous-graph teaser](/assets/images/paper/spahgc/fig_p001_06.png)
*Figure 1B: SpaHGC wires target spots (image-only) to reference-slide spots (image + GT expression) via three edge types — TS (intra-target), CS (cross-slice morphology), RS (intra-reference joint feature). Expression flows from reference to target through both the concatenated reference node feature and the CS edges feeding into CNDA.*

### Pipeline, step by step

1. **Patch encoding.** UNI (a pathology foundation model) encodes both target patches $P_t \in \mathbb{R}^{n\times224\times224\times3}$ and reference patches $P_r$ to 1024-dim features. Ablation Table 3 shows this matters: UNI > Virchow > ViT-Base > DeiT-Base > DenseNet121 > ResNet18.
2. **Heterogeneous-graph construction.**
   - **TS edges:** within target slide only, $Q=5$ nearest spots by 2D Euclidean spot coordinates.
   - **CS edges:** per target patch, Top-$K=7$ reference patches across *all* reference slides by cosine similarity of UNI embeddings.
   - **RS edges:** within and across reference slides, Top-$K=7$ peers by cosine similarity of the joint $h_r = [z_r \| y_r]$ feature.
3. **Complementary feature-channel masking.** Two views with binary masks satisfying $M_t^{(1)} + M_t^{(2)} = 1$ and $M_r^{(1)} + M_r^{(2)} = 1$, ratios $\alpha=0.8$ target and $\beta=0.9$ reference. Each view sees ~10–20% of the features and the two views together cover everything exactly once.
4. **Encoder per view.** Two parallel GraphSAGE stacks process TS and RS edges; a CNDA module processes CS edges. CNDA does bidirectional cross-attention: target queries attend to reference K/V ($\bar L_t = \mathrm{softmax}(Q_t K_r^\top/\sqrt{d'}) V_r$), and reference queries symmetrically attend to target K/V. Local and cross-attentional embeddings are merged with `scatter_mean`. 4 layers total.
5. **Contrastive alignment.** One view runs with `stop-grad`; the other is trained with $\mathcal{L}_\text{con}^i = \tfrac{1}{N}\sum_j (2-2\cos(\hat L^1_j, \hat L^2_j))$, separately for target and reference nodes. BYOL/SimSiam-style; no negatives.
6. **Decoder = CNAP.** 4-head unidirectional multi-head attention from merged target representation $\hat L_t$ (queries) to merged reference representation $\hat L_r$ (keys/values), with a residual: $\hat y_t = \mathrm{MLP}(\hat L_t + \hat H_t)$. This layer emits the predicted gene-expression vector.
7. **Total loss:** $\mathcal{L}_\text{total} = \mathcal{L}_\text{mse} + \mathcal{L}_\text{pcc} + \mathcal{L}_\text{con}^t + \mathcal{L}_\text{con}^r$ — equal-weighted, no $\lambda$ search reported.
8. **Optimisation:** Adam, lr $1\times10^{-4}$, weight decay $1\times10^{-4}$, 200 epochs, single RTX 4090 (24 GB), PyTorch 2.1.2 / Python 3.9.

## Experimental Results

### Datasets

Seven matched H&E + ST datasets. Three of the five non-headline datasets (Lymph Node, Pancreas1, Pancreas2) are HEST-1k subsets — the natural narrative anchor, although the paper does not lean into the HEST-1k story.

| Dataset | Tissue / subtype | Platform | # slices | Source |
|---|---|---|---|---|
| cSCC | Cutaneous squamous-cell carcinoma | legacy ST | 12 | Ji et al. 2020 |
| HER2+ | Breast cancer, HER2+ | legacy ST | 36 (8 patients) | Andersson et al. 2020 |
| Alex | Breast cancer, TNBC | Visium | 4 | Wu et al. 2021 |
| Visium BC | Breast cancer, HER2+ | Visium | 3 | Janesick et al. 2023 |
| Lymph Node | Lymph node | Visium | 4 | HEST-1k |
| Pancreas1 | Pancreatic cancer | Visium | 3 | HEST-1k |
| Pancreas2 | Pancreatic cancer | Visium | 4 | HEST-1k |

Preprocessing: scanpy top-1000 HVGs per slice, intersected across slices within a dataset — the shared gene panel is "approximately 200 to 800 genes" (varies per dataset, which makes absolute PCC numbers non-comparable across datasets). CPM-normalised then $\log(1+x)$. Patches are 224×224 around spot centroids.

**Evaluation protocol — Nested LOOCV.** Outer LOOCV holds out one slide as test; remaining slides form the reference bank; an inner LOOCV inside the training fold handles hyperparameter tuning. Top-K reference neighbours for each target slide are drawn exclusively from the corresponding fold's training reference bank, with the test/validation slides excluded. **Important caveat: this is leave-one-slide-out *within the same dataset*, not leave-one-patient-out.** For HER2+ (36 slides / 8 patients) the reference bank for a held-out slide can include other slides from the same patient.

### Main quantitative table (Table 1; PCC(%) ↑; LOOCV mean)

| Method | HER2+ | cSCC | Alex | Visium BC | Lymph Node | Pancreas1 | Pancreas2 |
|---|---|---|---|---|---|---|---|
| STNet | 5.61 | 9.20 | 3.20 | 2.80 | 3.40 | 2.24 | 31.56 |
| HisToGene | 7.89 | 17.56 | 1.11 | 1.16 | 19.24 | 1.65 | 26.13 |
| Hist2ST | 14.43 | 19.23 | 11.94 | 14.63 | 9.34 | 2.33 | 13.67 |
| EGGN | 17.98 | 16.22 | 3.25 | 12.26 | 24.78 | 13.81 | 30.10 |
| THItoGene | 17.26 | 18.15 | 10.69 | 2.34 | 10.82 | 2.16 | 18.97 |
| HGGEP | 19.68 | 20.13 | 2.55 | 7.79 | 6.20 | 2.30 | 5.65 |
| BLEEP | 18.56 | 23.56 | 5.04 | 4.09 | 24.83 | 2.29 | 28.55 |
| mclSTExp | 23.15 | 31.88 | 6.77 | 5.13 | 21.64 | 9.22 | 31.61 |
| M2OST | 18.24 | 24.88 | 15.13 | 6.52 | 30.97 | 15.12 | 38.35 |
| **SpaHGC (Ours)** | **27.86** | **38.79** | **17.19** | **20.08** | **35.02** | **24.48** | **41.36** |
| Δ vs best baseline | +4.71 | +6.91 | +2.06 | +5.45 | +4.05 | +9.36 | +3.01 |

RMSE numbers similarly favour SpaHGC (e.g. cSCC 0.15 vs best baseline 0.17, Pancreas1 0.17 vs 0.21). LOOCV std-devs are ~0.003–0.028 across folds, much smaller than the inter-method gaps. Supp. Table S2 reports Wilcoxon signed-rank $p<0.001$ across all 63 method × dataset comparisons (per-slice PCC).

### Qualitative

![Figure 3 — predicted spatial expression maps](/assets/images/paper/spahgc/fig_p005_01.png)
*Figure 3: Predicted spatial expression maps for marker genes SPINK5 (cSCC) and CEACAM5 (Pancreas1). SpaHGC reaches PCC 0.84 / 0.59 vs the next-best baselines mclSTExp (0.76) and M2OST (0.41).*

### Ablations (Lymph Node / Pancreas1 / Pancreas2)

| Component removed | Effect (ΔPCC, range across 3 datasets) |
|---|---|
| CS edges | **−2.4 to −3.5** (largest hit; the cross-slice premise) |
| TS edges | −1.3 to −2.4 |
| RS edges | −0.4 to −1.3 |
| CNDA module | −1.3 to −2.1 |
| CNAP module | −0.3 to −1.0 |
| Complementary mask → random mask | −0.9 to −2.2 |
| Complementary mask → no mask | −1.5 to −3.4 |
| UNI → ResNet18 (Pancreas1 only) | 24.48 → 20.25 (≈ half the headline gain) |

Hyperparameter sweeps (Supp. Tables S3–S6) show Q=5, K=7, α=0.8, β=0.9 are flat optima.

### Compute (Supp. Fig S12)

Despite the heterogeneous graph, SpaHGC is the *fastest* model on all three benchmark datasets (0.17 h on Pancreas2 vs M2OST 10.5 h) and the second-smallest in trainable parameters (5.3 M; only M2OST is smaller at 2.3 M).

### Downstream tasks

- **Tumor-region clustering (Supp. Figs S9–S10).** k-means on predicted expression vs pathologist annotation. ARI on Visium BC: SpaHGC 0.48 vs EGGN 0.28 vs HGGEP 0.13. ARI on Alex: SpaHGC 0.46 vs M2OST 0.43 vs mclSTExp 0.35 — the 0.03 gap on Alex is much narrower than the PCC gap and is the weakest of the downstream claims.
- **KEGG / GO:BP enrichment on Alex (Supp. Fig S11).** DEGs from SpaHGC-predicted expression yield more statistically significant enrichments than DEGs from the raw ST. Presented as denoising, but a model trained to minimise MSE on log-CPM will naturally over-smooth, which mechanically improves enrichment p-values; the paper does not control for this with a permutation or null-shuffle baseline. ⭐ at most.

![Supp. Fig S11A — pathologist annotation on Alex](/assets/images/paper/spahgc/fig_p024_01.png)
*Supplementary Figure S11A: pathologist-annotated Alex slide used as the ground-truth tumour mask for downstream enrichment analysis.*

![Supp. Fig S11A — SpaHGC predicted expression heatmap](/assets/images/paper/spahgc/fig_p024_02.png)
*Supplementary Figure S11A: SpaHGC's predicted gene-expression heatmap on the same Alex slide, fed into DEG identification and KEGG/GO:BP enrichment.*

## Limitations

**The authors admit:**

- Parameter count is slightly larger than M2OST (5.3 M vs 2.3 M).
- Shared-HVG panel size varies across datasets (200–800 genes), making absolute PCC numbers not directly comparable across datasets.

**The authors do not address — and these are the limits worth knowing:**

1. **"Cross-slice" is weaker than the title implies.** Despite the name "Cross-Slice Knowledge Transfer", the evaluation protocol is leave-one-slide-out *within the same dataset*. For HER2+ (36 slides / 8 patients) a held-out slide's reference bank can include other slides from the same patient — so SpaHGC is partly learning a within-patient consistency prior. There is no leave-patient-out, no cross-dataset transfer (e.g. cSCC → HER2+), and no cross-platform transfer (legacy ST ↔ Visium). The HEST-1k 3+4+4-slide subsets would have been the natural setting for a leave-cohort-out experiment.
2. **The most relevant recent baselines are missing.** TRIPLEX, Gene-DML, and DKAN are not in the comparison set. These are the natural head-to-heads and their absence is the most glaring gap — "SOTA across 7 datasets" should be read against a slightly weaker baseline list (STNet / HisToGene / Hist2ST / THItoGene / HGGEP / M2OST / EGGN / BLEEP / mclSTExp). Several included baselines (STNet on Alex/Visium BC, HisToGene on Pancreas1) score near-zero PCC and may be under-tuned on the smaller-panel Visium datasets, inflating the headline gaps.
3. **Roughly half of the headline gain on Pancreas1 is the encoder swap.** Replacing UNI with ResNet18 drops Pancreas1 from 24.48 to 20.25 PCC — vs M2OST at 15.12, so ~4 of the ~9 PCC gain is attributable to the foundation-model encoder rather than to the graph or contrastive design. This is honest of the paper to report but worth foregrounding.
4. **No batch-effect or platform-stratified analysis.** Three datasets are legacy ST and four are Visium; spot diameter, library prep, and capture efficiency differ. There is no Harmony / Combat-style correction, no platform-balanced experiment, and no analysis of where the Top-7 CS neighbours actually come from (same patient? same anatomical region? cross-platform?). The implicit batch-effect story leans entirely on frozen UNI + masking + CNDA's selective attention.
5. **"Contrastive" is terminologically loose.** Eq. 13 is cosine alignment with `stop-grad` and no explicit negatives — BYOL/SimSiam-style, not InfoNCE. Calling this "contrastive learning" throughout is a small stretch.
6. **Biological-relevance claim (C6) is one paragraph on one dataset** without a null-shuffle control, and a model trained to minimise MSE on log-CPM will mechanically over-smooth in a way that improves enrichment p-values.
7. **Variance reporting is asymmetric.** Std-devs reported for PCC; no confidence intervals on the ARI clustering numbers or on per-gene visualisations like "PCC=0.84 on this slide".
8. **No per-gene PCC breakdown** vs gene-expression magnitude — the field has shown predictors often only do well on the top few hundred most-variable genes, and this paper does not disaggregate.

## Why It Matters for Medical AI

The argument that "image-to-ST should look at other slides, not just the current one" is correct and clinically motivated — sequencing every slide is impractical, and a reference bank of fully-profiled slides is exactly the asset hospitals can accumulate over time. SpaHGC's specific contribution — making the reference bank a structured graph with morphological + joint-feature edges, and letting a cross-node attention encoder shuttle GT expression from reference to target — is a clean and well-ablated instantiation of that intuition.

For practitioners, the headline takeaways are:

- The CS edge ablation (−2.4 to −3.5 PCC) is the cleanest evidence yet that cross-slide reference retrieval is a meaningful prior on top of intra-slide CNN/GNN regression.
- The compute-efficiency numbers (5.3 M params, 0.17 h on Pancreas2) make this a reasonable drop-in baseline for downstream clinical pipelines despite the heterogeneous-graph complexity.
- The honest limit: until someone runs leave-patient-out or cross-cohort splits, the "knowledge transfer" claim should be read narrowly as *within-dataset, partially-within-patient* transfer. That is still useful, but not the cross-cohort generalisation the title might suggest.

## References

- arXiv: 2603.22821 (cs.CV), v1 — *Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference*
- Code: https://github.com/wenwenmin/SpaHGC
- Venue: CVPR 2026 (accepted; per preprint footer)
- Key baselines: STNet (Nat. Biomed. Eng. 2020), HisToGene (Nat. Commun. 2022), Hist2ST (Brief. Bioinform. 2022), BLEEP (NeurIPS 2023), mclSTExp (Brief. Bioinform. 2024), M2OST (AAAI 2025), EGGN, HGGEP, THItoGene
- Backbone: UNI (Chen et al., *Nature Medicine* 2024) — pathology foundation model
- Benchmark anchor: HEST-1k (Jaume et al. 2024) — Lymph Node + Pancreas1 + Pancreas2 are HEST-1k subsets
- Notable missing baselines (would have strengthened the comparison): TRIPLEX, Gene-DML, DKAN
