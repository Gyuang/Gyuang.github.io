---
title: "Multi-view deep learning of highly multiplexed imaging data improves association of cell states with clinical outcomes"
excerpt: "hmiVAE (Campbell lab, bioRxiv 2025) concatenates four IMC views into a single VAE latent and reports higher in-sample Cox C-index than FlowSOM/Louvain on three breast/melanoma TMA cohorts, but skips the leave-one-view-out ablation that would actually justify the 'multi-view' headline."
categories:
  - Paper
  - Spatial-Proteomics
permalink: /paper/hmivae/
tags:
  - hmiVAE
  - Imaging-Mass-Cytometry
  - Multi-View-VAE
  - Spatial-Proteomics
  - Survival-Analysis
  - Cox-PH
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- **hmiVAE is a vanilla concatenation-style multi-view VAE for Imaging Mass Cytometry (IMC).** Four per-cell views — mean marker expression `Y`, pixel-wise Pearson correlation against the DNA-stain mean as a nuclear-co-localization proxy `S`, five morphology features `M`, and a degree-normalised neighbour mean of the other three views as a spatial-context vector `C` — pass through per-view MLP encoders, get concatenated, and project to a single integrated latent `z`. **No product-of-experts, no cross-attention, no graph convolution** — the "spatial" view is literally a `D · [Y;S;M]` neighbour average, and information sharing across views happens only through the one integrated hidden layer after concat.
- **Headline result.** Leiden clustering on the integrated `z` achieves the **highest per-cluster Cox-PH concordance index** across three IMC TMA cohorts (Jackson-BC, Ali-BC, Hoch-Melanoma) when cluster prevalence is measured per mm² of tissue, with significance over FlowSOM/Louvain reported on the Ali-BC cohort for *both* prevalence metrics (Fig. 6B; numbers transcribed from bar charts, no numeric table in the paper).
- **Honest caveats the post will defend.** (1) Cox fits are **in-sample**, per cluster, with **no held-out patient split** and no multi-seed variance, so the survival headline is the most over-claimed result. (2) The "multi-view" claim itself is **the weakest of the lot** — integrated rarely beats expression-only on the breast cohorts and there is **no leave-one-view-out ablation** tied to a downstream metric. (3) The most defensible single claim is **cell-type recovery on the 500-cell manual Jackson-BC subset**, the only place a *manual* ground truth exists.

## Motivation

Highly multiplexed imaging (IMC, MIBI, cycIF) quantifies dozens of proteins per cell with sub-cellular spatial context, but the dominant analytical workflow still collapses each cell to a vector of mean marker intensities and clusters it with FlowSOM, PhenoGraph or Louvain. That throws away three things cancer biology says should matter: where the protein sits inside the cell (nuclear vs cytoplasmic vs membrane), the cell's shape (lymphocyte-round vs fibroblast-elongated), and what its neighbours look like (tumour-immune interface vs immune-cold stroma).

Multi-modal VAEs have already shown gains in adjacent omics modalities — totalVI for CITE-seq, DeepST and SpatialDIVA for spatial transcriptomics — by learning a joint embedding rather than concatenating engineered features. No equivalent existed for IMC. The medical-AI angle is direct: **if the learned embedding stratifies patients better than FlowSOM on mean expression, IMC-derived cell-state biomarkers gain prognostic value.** That is exactly the claim hmiVAE tries to support.

## Core Innovation

There are three design decisions that define hmiVAE, and each one is more conservative than the title suggests:

1. **Four hand-engineered per-cell views, not learned per-view representations.** Expression `Y ∈ ℝ^{N×P}` (per-cell mean of per-pixel counts), nuclear co-localization `S ∈ ℝ^{N×P}` (per-cell pixel-wise Pearson correlation of each protein channel against the mean of two Iridium DNA channels — so this is a correlation scalar per protein, not a learned subcellular map), morphology `M ∈ ℝ^{N×5}` (area, perimeter, eccentricity from `scikit-image` plus DeepCell concavity/asymmetry), and spatial context `C ∈ ℝ^{N×(2P+5)}`. The spatial view is built by z-scoring `Y, S, M`, concatenating them, then left-multiplying by a sparse 10-nearest-neighbour adjacency `D ∈ {0,1}^{N×N}` and degree-normalising — **so "spatial" is a single-hop neighbour mean of the other three views, with no graph convolution and no attention.**
2. **Concatenation-style multi-view VAE with conditional batch covariates `b`.** Each of `{Y, S, M, C}` plus `b` (per-cell one-hot of sample ID, plus per-sample background-stain statistics) goes through a small MLP encoder; the four view embeddings are concatenated, passed through one shared hidden layer, and projected to `(μ, log σ²)` of an integrated latent `z`. The decoder takes `[z ; b]`, runs symmetric integrated layers, then splits into four per-view decoder heads. **This is not product-of-experts (no per-view posteriors combined multiplicatively), not mixture-of-experts, and there is no explicit cross-view alignment loss.** Joint information sharing happens exactly once — at the post-concat hidden layer.
3. **KL warm-up over constant β, equal-weight per-view reconstruction, in-sample Leiden + per-cluster Cox-PH for downstream.** The ELBO sums per-view reconstruction likelihoods with equal weight (no learned β_j per view, even though `P ≈ 36–46` ≫ `M = 5`), with β annealed from 0 by +0.1/epoch — the paper's Fig. 1C shows warm-up beats constant β on reconstruction. Downstream, scanpy `neighbors (k=100)` + Leiden over the integrated `z` produces cluster labels; clusters are then summarised per patient as (a) proportion of cells and (b) cells per mm² of tissue, and each cluster is fit *individually* with a `lifelines` Cox-PH model **stratified on disease stage**.

## Claims & Evidence Analysis

| Claim | Evidence in paper | Dataset(s) | Strength |
|---|---|---|---|
| C1: hmiVAE integrated clusters give better survival prediction than FlowSOM/Louvain on the same engineered features | Fig. 6B — bar plot of per-cluster Cox C-index, significance stars on prevalence/mm² metric; significant on **both** metrics for Ali-BC | Jackson-BC, Ali-BC, Hoch-Melanoma | ⭐⭐ |
| C2: Multi-view integration beats single-view (expression-only) for clinical-variable association | Fig. 5B — integrated vs expression-only bars; paper itself states differences are **not significant** between hmiVAE-integrated and hmiVAE-expression on the breast cohorts | Jackson-BC, Ali-BC | ⭐ |
| C3: hmiVAE expression-only embedding gives better cell-type clusters than Louvain/FlowSOM | Fig. 4C ARI vs published labels; Fig. 3 heatmaps; per-cell-type precision/recall on the 500-cell **manual** subset (Geuenich 2021) of Jackson-BC | Jackson-BC (manual GT), partial Ali-BC | ⭐⭐ on Jackson-BC, ⭐ on Ali-BC (FlowSOM was used to *make* the Ali-BC labels — self-fulfilling baseline) |
| C4: Nuclear co-localisation and spatial context capture information mean expression cannot | Fig. 2 t-value heatmaps — some integrated clusters differ on co-loc or neighbourhood rather than expression | All three | ⭐⭐ (interpretive only — **no leave-view-out ablation tied to a downstream metric**) |
| C5: hmiVAE generalises across IMC cohorts via latent-space projection | Fig. 7B–E ARI; **mixed result** — baseline-KNN beats hmiVAE-KNN on the full feature set and nuclear co-localization; hmiVAE wins on expression-only and spatial-only; only same-panel Jackson↔Ali transfer tested | Jackson↔Ali only | ⭐ — paper itself admits this is not straightforward |
| C6: KL warm-up improves reconstruction over constant β | Fig. 1C box plots | All three | ⭐⭐ (reconstruction proxy, not a downstream-task gain) |
| C7: Identified clusters correspond to known biology (Ali-BC hypoxic tumour core, Hoch-Melanoma immunosuppressive niche) | Fig. 6A + Supp. Fig. 7/10 — image inspection; no quantitative orthogonal validation | Ali-BC, Hoch-Melanoma | ⭐ (illustrative anecdotes) |

**Honest synthesis.** The strongest claim is **C3 on Jackson-BC** because that is the only place in the entire paper where the ground truth is *manually* labelled (500 cells from Geuenich 2021) rather than copy-pasted from the original publications' own clustering. The headline survival claim (**C1**) is moderately supported — concordance gains are consistent in *direction* across three cohorts and statistically significant in places, but: (a) the comparison is to FlowSOM and Louvain on the **same hand-engineered feature set**, with no neural baseline and no IMC-specific deep model (Mosna, CellSighter, SpaceFlow); (b) no multi-seed C-index distributions are reported; (c) clusters are post-hoc selected per dataset, inflating the multiple-testing burden across (datasets × prevalence-measure × cluster-id). The "multi-view" claim (**C2**) is genuinely weak: the integrated embedding rarely beats expression-only on the breast cohorts, and **no view-removal ablation** is performed — which is the single biggest methodological gap given that "multi-view" is in the title. The transfer claim (**C5**) is openly negative-mixed, which the discussion to its credit acknowledges.

**The hidden weakness most likely to break replication: survival eval is in-sample.** Leiden resolutions per view are picked by the authors and reported as fixed (1.0, 0.5, 1.0, 0.1, 0.5 for integrated/expression/co-loc/morphology/spatial). There is no held-out patient split for the Cox model — every patient is fit and scored on the same data. C-index is in-sample. No cross-validated or external-cohort survival C-index is reported. **No 3-star claim survives this constraint**, which is why the audit table caps out at ⭐⭐.

## Method & Architecture

![hmiVAE architecture and hyperparameter sweep](/assets/images/paper/imc-multiview/fig_p005_01.png)
*Figure 1: hmiVAE takes four per-cell views (expression `Y`, nuclear co-localization `S`, morphology `M`, spatial context `C`) plus batch covariates `b` into per-view MLP encoders, concatenates them into an integrated latent `z`, and reconstructs each view through a symmetric per-view decoder. Panels C-G report the hyperparameter sweep: KL warm-up beats constant β on reconstruction, latent dim and hidden depth matter little, and smaller batches plus wider hidden layers improve reconstruction quality.*

The four views and the conditioning vector are constructed entirely from segmentation masks plus the per-pixel marker tensor, with no learned upstream embedder. Concretely, per cell `i`:

- $Y_i \in \mathbb{R}^P$ — mean per-pixel intensity over the cell's segmentation pixels for each of `P` proteins.
- $S_i \in \mathbb{R}^P$ — for each protein channel, the pixel-wise Pearson correlation against the mean of the two Iridium DNA channels over the cell's pixels. A scalar per protein, **not** a learned subcellular localisation map.
- $M_i \in \mathbb{R}^5$ — area, perimeter, eccentricity from `regionprops` plus DeepCell-style concavity and asymmetry.
- $C_i \in \mathbb{R}^{2P+5}$ — degree-normalised mean of `[Y; S; M]` over the 10 nearest spatial neighbours of cell `i`. So `C = D_norm · [Y; S; M]` where `D` is the binary 10-NN spatial adjacency. **No graph convolution, no learned aggregator.**

Per-cell conditioning vector `b` concatenates a one-hot of sample ID with per-sample technical covariates: mean background-stain channel (e.g. ArAr80) and its correlation with each protein channel (a non-specific-binding proxy), mean per-image pixel intensity, and mean background intensity. When background channels are unavailable, "background" defaults to the mean of each protein over mask-value-0 pixels.

The training objective is the standard ELBO with equal-weight per-view reconstruction:

$$\mathcal{L} = \sum_{j \in \{Y, S, M, C\}} \sum_{i=1}^{N} \mathbb{E}_{q(z|X)}\!\left[\log p(x_{i,j} \mid z)\right] - \beta \, \mathrm{KL}\!\left(q(z \mid X) \,\|\, p(z)\right).$$

Hyperparameter grid (seed ∈ {0,1,42,123,1234}, hidden layers ∈ {1,2}, hidden dim ∈ {8,32,64}, latent dim ∈ {10,20}, β schedule ∈ {warm-up, constant}, batch ∈ {40000,…,4000}) is searched by **best test-set reconstruction likelihood**. Note that selection is on reconstruction, not on any downstream task — so the hyperparameter choice is not optimised for survival or cell-type recovery.

![Per-cluster feature-driver t-values across four views](/assets/images/paper/imc-multiview/fig_p007_01.png)
*Figure 2: Per-cluster t-value heatmaps for each view across the three datasets. Some integrated clusters share an expression profile but separate on nuclear co-localization or neighbourhood context — the paper's main interpretive argument that the additional views carry distinct information. Note that this is correlational evidence: no quantitative leave-one-view-out experiment is run to show that removing co-loc or spatial actually degrades a downstream metric.*

## Experimental Results

The paper does not publish numeric tables for the main results — every comparison is read off bar charts. The qualitative summary below transcribes those bars; readers should treat the absence of numbers as itself a finding.

| Result | Setting | Outcome (as reported in figures) |
|---|---|---|
| Cell-type recovery (ARI vs published labels), Fig. 4C | Jackson-BC | **hmiVAE expression-only > Louvain and FlowSOM** |
| Cell-type recovery (ARI), Fig. 4C | Ali-BC | hmiVAE expression-only ≈ FlowSOM (FlowSOM was used in the original Ali publication — baseline self-recovery) |
| Per-cell-type precision/recall, Fig. 4 | Jackson-BC 500-cell manual subset | **hmiVAE higher precision on B cells, macrophages, endothelial and stromal cells** |
| Clinical-variable association, Fig. 5B | Jackson-BC, Ali-BC | hmiVAE integrated has higher mean association but **not statistically significant** vs Louvain/FlowSOM |
| Clinical-variable association, Fig. 5B | Hoch-Melanoma | **hmiVAE expression-only significantly higher** than FlowSOM/Louvain on cluster-proportions metric |
| Cox C-index (per cluster, in-sample), Fig. 6B | All three cohorts | **hmiVAE integrated highest** on both prevalence metrics across all 3 datasets |
| Cox C-index significance over baselines, Fig. 6B | Ali-BC | Significant on **both** cluster-proportion and cells/mm² metrics |
| Cox C-index significance over baselines, Fig. 6B | Jackson-BC, Hoch-Melanoma | Significant most consistently for cells/mm² |
| Cross-dataset projection ARI, Fig. 7B-E | Jackson↔Ali | **Mixed** — baseline-KNN beats hmiVAE-KNN on full feature + co-loc; hmiVAE wins on expression-only + spatial-only |
| KL warm-up vs constant β, Fig. 1C | All three | **Warm-up wins on reconstruction** (downstream gain not tested) |

![Cell-type cluster heatmaps for hmiVAE vs Louvain vs FlowSOM](/assets/images/paper/imc-multiview/fig_p010_01.png)
*Figure 3: Expression-only embeddings from hmiVAE recover distinct B-cell, T-cell, macrophage and basal/luminal epithelial clusters on Ali-BC and Jackson-BC, where FlowSOM and Louvain tend to merge multiple immune subtypes into broader "general immune" bins.*

![ARI vs published labels and per-cell-type precision/recall](/assets/images/paper/imc-multiview/fig_p011_01.png)
*Figure 4: hmiVAE expression-only clustering achieves the highest ARI against published cell labels on both breast cohorts and higher precision for several cell types on the 500-cell manual ground-truth subset of Jackson-BC. This is the only place in the paper where a manual (non-self-fulfilling) ground truth exists, and it is consequently the strongest single piece of evidence.*

![Clinical-variable association across methods](/assets/images/paper/imc-multiview/fig_p013_01.png)
*Figure 5: Cluster prevalence (proportion and per-mm²) tested against clinical variables; hmiVAE integrated clusters tend to higher mean association, but the gap over FlowSOM/Louvain is only statistically significant on Hoch-Melanoma with expression-only embeddings — not on either breast cohort with the integrated embedding.*

### Ablation and robustness — what is missing

- **No leave-one-view-out ablation** on any downstream metric (survival, clinical association, cell-type recovery). The paper interprets Fig. 2 t-values as evidence that nuclear co-localization and spatial context add information, but never reports "hmiVAE w/o spatial" vs "hmiVAE w/o morphology" C-indices. Given that "multi-view" is in the title, this is the single biggest methodological gap.
- **No multi-seed variance** on downstream metrics. Random seed is searched as a hyperparameter for picking the best reconstruction model, but downstream tables show one point estimate.
- **No held-out patient split** for the Cox model. Every cluster is fit and C-index is reported on the same data the embedding was trained on.
- **No comparison to existing IMC-specific deep models** (Mosna, CellSighter, SpaceFlow, SpatialDIVA). The only baselines are classical clustering on the same engineered features.

### Qualitative biology vignettes

- **Ali-BC integrated cluster 0**: luminal epithelial cells co-expressing CAIX (hypoxia marker) and Ki67, depleted in myoepithelial neighbours; the paper interprets this as a **hypoxic tumour core** and high prevalence associates with worse survival. (Fig. 6A; Supp. Fig. 10.)
- **Hoch-Melanoma integrated cluster 10**: tumour cells embedded in an IDO+/FOXP3+/CD11b+ neighbourhood — an **immunosuppressive niche** — with significant survival association on both prevalence metrics. (Supp. Fig. 7.)

These are well-told stories on individual patients but are not quantitatively validated against orthogonal stains or external cohorts.

## Limitations

**Author-admitted:**

- Hyperparameter sweep did not include learning rate or per-view embedding sizes.
- Latent-space projection across cohorts is fragile; integrated latent and nuclear-co-loc embeddings do not transfer well even with a shared antibody panel.
- Distance-based KNN over learned latents is questionable; paper cites the well-known critique.
- No standard quantitative benchmark for HMI deep models exists; manual ground-truth annotations are scarce and expensive.

**Audit additions:**

- **No view-ablation downstream.** "Multi-view" is the headline but no experiment isolates each view's marginal contribution to a clinical metric.
- **No multi-seed variance** on downstream tables; only on reconstruction during the HP sweep.
- **No held-out / cross-validated survival evaluation.** Cox is fit in-sample, one cluster at a time.
- **Multiple-testing burden** across (cluster × prevalence-measure × clinical-variable) is not corrected for in the headline claims; BH is applied only within the per-cluster hazard ratios.
- **No deep-model baselines** for IMC — only FlowSOM/Louvain on the same engineered features.
- **No CODEX/MIBI data.** The method is IMC-only, despite cross-platform integration being a natural follow-up.
- **The spatial view is a degree-normalised neighbour mean**, not a graph-conv aggregator; higher-order topology that a GNN would capture is discarded.
- **Per-view reconstruction weighting is uniform**, even though `P ≈ 36–46` ≫ `M = 5`, so the morphology view's gradient signal is plausibly drowned out by sheer reconstruction-loss magnitude.
- **The cross-cohort projection uses a randomly assigned sample one-hot** for the query (the totalVI trick from Gayoso et al. 2021), which makes the conditional-VAE conditioning meaningless on the query and likely contributes to the weak transfer result.

## Why It Matters for Medical AI

For medical AI, the relevance is narrow but real. Tumour IMC TMA cohorts are accumulating fast (Jackson 2020, Ali 2020, Hoch 2022, IMMUCan), and the *de facto* analytical pipeline still ends at FlowSOM-on-mean-expression. hmiVAE is a concrete demonstration that even a vanilla concat-VAE that doesn't touch the spatial graph beyond a 1-hop mean can produce cluster labels with **better in-sample Cox-PH concordance for patient stratification than FlowSOM on the same engineered features**, on three independent cohorts. That is a useful signal — it tells the IMC community that the engineered-feature ceiling is below what a learned joint embedding can do, which justifies investing in stronger architectures (graph-conv spatial views, PoE/MoE multi-view posteriors, learned per-view weighting, real cross-validated survival eval) rather than tuning FlowSOM harder. As a deployable biomarker pipeline, however, the in-sample Cox evaluation and missing leave-one-view-out experiment mean hmiVAE itself should be read as a strong proof-of-concept rather than a prognostic tool ready for any clinical workflow.

## References

- Paper: Ayub, Jackson, Selega, Campbell. *Multi-view deep learning of highly multiplexed imaging data improves association of cell states with clinical outcomes.* bioRxiv 2025.03.14.643377 (preprint, posted 17 Mar 2025). DOI: [`10.1101/2025.03.14.643377`](https://doi.org/10.1101/2025.03.14.643377).
- Code: [`github.com/camlab-bioml/hmiVAE`](https://github.com/camlab-bioml/hmiVAE) and the manuscript repo [`github.com/camlab-bioml/hmiVAE_manuscript`](https://github.com/camlab-bioml/hmiVAE_manuscript).
- IMC cohorts: Jackson et al., *Nature* 578, 615–620 (2020); Ali et al., *Nature Cancer* 1, 163–175 (2020); Hoch et al., *Sci. Immunol.* 7 (2022).
- Manual 500-cell IMC ground truth: Geuenich et al., *Cell Systems* 12, 1173–1186.e5 (2021).
- Related multi-modal VAEs cited as motivation: totalVI (Gayoso et al., *Nat. Methods* 2021), DeepST (Xu et al., *NAR* 2022), SpatialDIVA (Sukhinin et al., *Nat. Commun.* 2024).
- Baselines compared against: FlowSOM (Van Gassen et al., *Cytometry A* 87, 636–645 (2015)); Louvain via scanpy (Wolf et al., *Genome Biology* 19, 15 (2018)).
