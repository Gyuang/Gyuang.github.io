---
title: "Molecularly informed analysis of histopathology images using natural language"
excerpt: "SpotWhisperer chains DeepSpot and CellWhisperer to query H&E patches by free text, reaching weighted AUROC 0.717 on a lung-cancer Visium benchmark vs PLIP 0.554 / CONCH 0.478."
categories:
  - Paper
  - Spatial-Transcriptomics
permalink: /paper/spotwhisperer-molecularly-informed-histopathology-natural-language/
tags:
  - SpotWhisperer
  - DeepSpot
  - CellWhisperer
  - Spatial-Transcriptomics
  - Vision-Language
  - Computational-Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- Pathology VLMs (PLIP, CONCH) only see macroscale phenotype labels, so they are blind to microscale molecular signal. SpotWhisperer fixes this without any new joint training by chaining **DeepSpot (H&E -> spatial transcriptome)** with **CellWhisperer (transcriptome <-> text)** so that an H&E patch can be queried by free text through an inferred molecular profile.
- The contribution is twofold: (1) a training-free two-stage pipeline and (2) a newly curated lung-cancer Visium benchmark with two label tracks (expert region annotations and atlas-derived cell types) that enables spot-level zero-shot evaluation.
- On 5 NSCLC Visium samples (16,032 spots), region annotation reaches an overall weighted **AUROC of 0.717 (SEM 0.049)** vs PLIP 0.554 and CONCH 0.478. Cell-type annotation reaches **0.656 (0.047)** vs 0.586 / 0.555. SpotWhisperer wins 3 of 4 region types and 3 of 4 cell types.

## Motivation

Routine cancer diagnosis still leans on H&E morphology, which is cheap but blind to gene expression. Pathology VLMs like PLIP and CONCH inherit that blindness because their training labels are themselves macroscale ("tumor", "inflamed tissue"). Spatial transcriptomics (ST) provides the microscale molecular readout, but cost and technical demand keep it out of the clinic.

The authors argue that two recent ingredients - DeepSpot, which regresses ST from H&E, and CellWhisperer, a transcriptome-text contrastive model - can be composed to graft a molecular intermediate onto pathology image analysis and expose it to clinicians as a natural-language search interface, with no ST assay needed at inference time.

## Core Innovation

- **Training-free composition.** SpotWhisperer itself trains nothing new. The only training is a tissue-specific DeepSpot regression head; CellWhisperer and the VLM baselines are all frozen and zero-shot.
- **Molecular intermediate as the routing layer.** Instead of aligning the image directly to text, the pipeline routes H&E -> inferred 5,000-HVG expression profile -> CellWhisperer joint embedding. Text queries are matched against molecular signal, not morphology.
- **Microscale benchmark.** A new lung-cancer Visium benchmark is released for spot-level zero-shot evaluation, with both expert pathologist region labels and Human Lung Cell Atlas-derived cell-type labels.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | SpotWhisperer outperforms pathology VLMs on a newly curated benchmark for spatially resolved H&E annotation. | Table 1 (overall 0.717 vs 0.554/0.478) and Table 2 (0.656 vs 0.586/0.555); winning 3/4 region types and 3/4 cell types. | 5 NSCLC Visium samples, 16,032 spots. | ⭐⭐ |
| C2 | The method enables interactive natural-language exploration of microscale tissue biology with access to inferred gene expression. | Fig. 1b ("T cells" query overlaps TLS); Fig. 1c (LTB gene map matches the same regions); web UI + supplementary video. | 1 sample, qualitative. | ⭐ |
| C3 | Inferred (vs. measured) spatial transcriptomes are still meaningful enough for CellWhisperer to interpret. | Fig. 3 UMAP shows clustering by expert annotation on DeepSpot-predicted profiles. | Lung evaluation set; n=5 samples pooled. | ⭐⭐ |
| C4 | Cell-type signatures, being transcriptome-defined, are harder for pure-image VLMs but easier for SpotWhisperer. | Table 2 - SpotWhisperer wins 3/4 cell types and overall by 0.07 AUROC. | Same 5 samples; cell-type labels are themselves model-derived. | ⭐⭐ |
| C5 | The advantage is not explained by VLMs being given too narrow a context window. | Table 5 patch-size sweep (80 vs 224 px) shows no meaningful improvement. | 5 samples, region task only. | ⭐⭐ |
| C6 | VLMs retain advantages on macroscale-defined classes such as TLS, suggesting complementarity. | Table 1 row "tertiary lymphoid structures": PLIP 0.881, CONCH 0.886 vs SpotWhisperer 0.801. | Same 5 samples. | ⭐⭐ |
| C7 | The approach is a path to "routine integration" of microscale molecular information into clinical histopathology. | Conceptual / aspirational, supported only by web-UI demo and lung-cancer numbers. | None beyond the lung benchmark. | ⭐ |
| C8 | DeepSpot's recent improvements (5,000-gene prediction, foundation-model backbone) enable this pipeline. | Cited from Nonchev et al., 2025; not re-validated here. | External. | ⭐ (in this paper) |

The headline benchmark result (C1, C4) is real but narrow: a single tissue type, n=5 evaluation samples, no statistical-significance testing reported (only mean ± SEM, with overlapping error bars in some rows - e.g., Stroma: CONCH 0.630±0.069 vs SpotWhisperer 0.526±0.060). A paired Wilcoxon across samples or per-spot AUC bootstrap would have made the claims much firmer. C5 (the patch-size control) is an underrated piece of evidence - it heads off the most obvious counter-explanation. C2 and C7 are the weakest: the "interactive natural-language exploration" claim leans almost entirely on one anecdote (T-cells/LTB over a TLS region) and a video, with no systematic user study and no false-positive measurement over a query catalog.

## Method & Architecture

The whole pipeline is a frozen, training-free composition of two pre-existing models plus an evaluation harness; the only new training is a lung-specific DeepSpot regressor.

![SpotWhisperer pipeline and web UI demo](/assets/images/paper/spotwhisperer_2025.07.14.664402/page_002.png)
*Figure 1 - SpotWhisperer pipeline (a) and web UI demo (b-c): a free-text query for "T cells" highlights tertiary-lymphoid-structure regions identified by experts, and the same regions show enriched LTB expression, while existing VLMs (d) link only to macroscale labels.*

The pipeline breaks down into five steps:

1. **Tile extraction.** Each whole-slide H&E image is subdivided into tiles centered on Visium spot positions. Each spot gets a bag of sub-tiles (zoomed-in morphology) and a bag of neighbor tiles (context), following the DeepSpot input format (Appendix B).
2. **H&E -> spatial transcriptome (DeepSpot).** A pre-trained pathology foundation model, **H-optimus-0** (Saillard et al., 2024), embeds every tile. A trainable DeepSpot regression head then predicts log-normalized expression for the **top 5,000 highly variable genes** per spot. The lung-specific DeepSpot head is trained on **36 independent 10x Visium lung-cancer samples** from De Zuani et al. (2024), disjoint from the 5 evaluation samples (Appendix A.1).
3. **Transcriptome -> joint embedding (CellWhisperer).** The 5,000-gene predicted profile per spot is fed into CellWhisperer's frozen transcriptome encoder, yielding a spot-level embedding in the same space as CellWhisperer's text encoder (CLIP-style contrastive training on transcriptome-text pairs; Schaefer et al., 2024a/b).
4. **Zero-shot scoring.** Each candidate label (e.g., "tumor cells", "immune cells") is encoded by CellWhisperer's text encoder. Per-spot class probabilities are computed as **softmax over cosine similarities** between the spot transcriptome embedding and the class text embeddings (Appendix B.1). VLM baselines (PLIP, CONCH) are scored analogously, with softmax over text-image cosine similarities on the spot-centered patch (Appendix B.2).
5. **Web UI.** The CellWhisperer web app is extended so a user types a free-text query ("T cells", or a gene symbol like "LTB") and receives a spatial heat-map overlay on the H&E slide; both the matching-spot map and the predicted gene expression map are accessible (Fig. 1b-c).

Beyond the standard contrastive softmax there is no load-bearing formula:

$$
p(c \mid x) = \mathrm{softmax}_c\!\left( \langle e_x, e_c \rangle / \tau \right)
$$

Key setup choices:

- **No fine-tuning** of CellWhisperer or the VLM baselines - everything downstream of DeepSpot is zero-shot.
- **No prompt-ensembling** for any model; CONCH's prompt ensemble is deliberately disabled to mimic a single-query user (B.2).
- **Patch-size sweep** for VLMs (80 px vs 224 px, Table 5) showed no meaningful gain.
- DeepSpot is **tissue-specific**; cross-tissue generalization is not tested.

### Benchmark construction

The conclusion up front: the evaluation set is small (5 samples) but it is the first spot-level benchmark with both expert region labels and atlas-derived cell-type labels.

![Benchmark construction with dual label tracks](/assets/images/paper/spotwhisperer_2025.07.14.664402/page_004.png)
*Figure 2 - Construction of the lung-cancer microscale benchmark: ground-truth labels are derived two ways (bioinformatics cell types and pathologist region annotations), then SpotWhisperer and the PLIP/CONCH baselines are scored zero-shot via per-class AUROC.*

- 5 lung-cancer Visium samples from Dawo et al. (Zenodo 14620362, 2025).
- **16,032 spot-centered patches** at 20x magnification.
- Two label tracks per spot:
  - **Region labels (expert pathologist):** TUM, NOR, TLS, INFL - natural-language renderings of the abbreviations are listed in Table 3.
  - **Cell-type labels (transcriptome-derived):** Epithelial / Endothelial / Immune / Stroma - assigned by a Human Lung Cell Atlas logistic regression on the *measured* ST data. Deliberately coarse because Visium has multi-cell resolution.
- No held-out fine-tuning split; all 5 samples are used for zero-shot evaluation. Mean / SEM computed across n=5 samples.

## Experimental Results

### Region-level annotation (Table 1; AUROC, mean (SEM) across 5 samples)

SpotWhisperer wins 3 of 4 region types and clears the overall weighted AUROC by a wide margin.

| Region type | PLIP | CONCH | SpotWhisperer (Ours) |
|---|---|---|---|
| infiltrating cells | 0.399 (0.036) | 0.550 (0.069) | **0.648 (0.089)** |
| normal cells | 0.453 (0.068) | 0.380 (0.057) | **0.658 (0.077)** |
| tertiary lymphoid structures | 0.881 (0.059) | **0.886 (0.031)** | 0.801 (0.050) |
| tumor cells | 0.627 (0.039) | 0.506 (0.048) | **0.760 (0.048)** |
| **Overall (weighted)** | 0.554 (0.026) | 0.478 (0.025) | **0.717 (0.049)** |

Overall weighted AUROC pushes +0.163 over PLIP and +0.239 over CONCH. The exception is TLS, where both VLMs are stronger - macroscale, morphologically well-defined classes still favor direct image matching, which the authors read as a complementarity signal (C6).

### Cell-type annotation (Table 2; AUROC, mean (SEM) across 5 samples)

Cell types are transcriptome-defined, so they should be intrinsically harder for VLMs - and that is what shows up everywhere except Stroma.

| Cell type | PLIP | CONCH | SpotWhisperer (Ours) |
|---|---|---|---|
| Endothelial | 0.584 (0.025) | 0.527 (0.039) | **0.622 (0.048)** |
| Epithelial | 0.565 (0.025) | 0.549 (0.013) | **0.670 (0.052)** |
| Immune | 0.622 (0.022) | 0.564 (0.023) | **0.659 (0.035)** |
| Stroma | 0.521 (0.041) | **0.630 (0.069)** | 0.526 (0.060) |
| **Overall (weighted)** | 0.586 (0.022) | 0.555 (0.009) | **0.656 (0.047)** |

The Stroma row goes to CONCH, but both error bars are large and the gap is not decisive.

### Ablations / robustness

The paper's only ablation is a **patch-size sweep for the VLM baselines** (Appendix B.3, Table 5). Expanding the patch context from 80 px to 224 px moves PLIP from 16.18% to 16.20% accuracy and CONCH from 9.36% to 12.62% on region annotation - nowhere near closing SpotWhisperer's lead, which rules out the "VLMs just need more context" counter-explanation.

What is missing is any ablation of **SpotWhisperer itself**: no test of measured ST vs DeepSpot-inferred ST, no swap of H-optimus-0 for another backbone, no swap of CellWhisperer for an alternative transcriptome-text model (e.g., LangCell), no gene-set size variation, no justification of the 5,000-HVG choice.

### Does the molecular bottleneck preserve signal?

![UMAP of CellWhisperer embeddings on DeepSpot-inferred profiles](/assets/images/paper/spotwhisperer_2025.07.14.664402/page_006.png)
*Figure 3 - UMAP of CellWhisperer embeddings on DeepSpot-inferred 5,000-HVG profiles for the lung evaluation set; tumor and normal regions cluster cleanly while infiltrating cells remain diffuse, indicating that the molecular intermediate retains expert-distinguishable structure.*

Qualitatively, Fig. 1b shows the "T cells" query map overlapping expert TLS contours, and Fig. 1c shows an interpretable LTB gene expression map at the same locations. Useful as narrative evidence for biological plausibility - but it is a single qualitative example.

## Limitations

Authors acknowledge:

- Only one tissue type (lung cancer) tested; cross-tissue generalization deferred to future work.
- Cell-type evaluation is restricted to coarse classes (Immune / Epithelial / Endothelial / Stroma) because Visium spots cover multiple cells; finer-grained cell-type prediction awaits higher-resolution ST.
- No prompt-ensembling for any model; the authors note prompt tuning could lift both VLMs and SpotWhisperer.
- A tri-modal joint embedding (image + transcriptome + text) and a pan-tissue model are listed as future directions.
- Code, data, and web-UI access are gated behind "archival publication" - i.e., not yet released with the preprint.

Reviewer-side critique:

- **No statistical-significance testing.** No DeLong's test for paired AUROCs and no sample-level paired Wilcoxon - so in rows with overlapping SEMs (notably Stroma), it is impossible to tell which gaps are noise.
- **No ablation isolating the contribution of the molecular intermediate from the H-optimus-0 image features.** SpotWhisperer's lift could come from DeepSpot's image features rather than from the transcriptome bottleneck. A control where the H-optimus-0 CLS embedding is fed directly to a small text-aligned head would test whether the molecular intermediate is genuinely doing work. None of that is reported.
- **No comparison to oracle / measured ST.** It is unknown how much of SpotWhisperer's lift would survive if DeepSpot were perfect, or how much it loses relative to using actual Visium counts at inference.
- **No simple-baseline comparison.** Even a minimal linear-probe fine-tune of CONCH/PLIP on the lung benchmark is missing, so it is unclear how much of the gap is a zero-shot artefact.
- **No robustness to stain / scanner / vendor.** Performance on other ST platforms (Visium HD, Xenium, Stereo-seq) is untested. Training (De Zuani 2024) and evaluation (Dawo 2025) are both 10x Visium NSCLC, so domain shift is mild by design.
- **TLS underperformance (0.801 vs 0.886) is not diagnosed** - is it a CellWhisperer text-encoder vocabulary issue, a DeepSpot regression artefact in lymphoid regions, or a fundamental microscale/macroscale mismatch?
- **No failure-mode catalog for the chat UI** - which queries trigger hallucinations is unstudied.
- **n=5 is a narrow statistical base.** SEMs are computed across 5 samples, so confidence intervals are wide and any single-row gap inside ~0.10 AUROC should be read as suggestive, not conclusive.

## Why It Matters for Medical AI

The pathology VLM line is capped at whatever phenotypes its labels already see - which becomes a ceiling the moment clinicians want molecular information from the image. SpotWhisperer offers a different idea: instead of training a new joint image-text model, splice in a molecular intermediate as the routing layer. The appeal is that it recombines existing foundation-model assets without new joint training, and it gives microscale molecular querying without an ST assay at inference time.

That said, the current evidence is pinned to a single domain (NSCLC Visium, 5 evaluation samples) and there is no ablation separating the contribution of the molecular bottleneck from the H-optimus-0 image features. The "routine clinical integration" framing in the abstract is aspirational; a real clinical-integration argument would need clinician-in-the-loop evaluation and cross-tissue / cross-platform validation.

## References

- Paper (bioRxiv preprint): <https://doi.org/10.1101/2025.07.14.664402>
- Project page (web UI): <http://spotwhisperer.bocklab.org>
- Venue: ICML 2025 Workshop on Foundation Models for the Life Sciences (FM4LS)
- License: CC-BY-NC 4.0 (preprint)
- Related work:
  - DeepSpot - Nonchev et al., 2025 (H&E -> spatial transcriptome regression)
  - CellWhisperer - Schaefer et al., 2024a/b (transcriptome <-> text contrastive)
  - H-optimus-0 - Saillard et al., 2024 (pathology foundation model)
  - PLIP, CONCH - pathology VLM baselines
  - Human Lung Cell Atlas - Sikkema et al., 2023
  - Training ST data - De Zuani et al., Nat. Commun. 2024 (NSCLC Visium)
  - Evaluation ST data - Dawo et al., Zenodo 14620362, 2025
