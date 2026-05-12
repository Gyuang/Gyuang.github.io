---
title: "PathGen: Generating Crossmodal Gene Expression from Cancer Histopathology Improves Multimodal AI Predictions"
excerpt: "A DDPM that synthesises 6-group bulk transcriptomic embeddings from WSI patches lifts TCGA-GBMLGG survival C-Index from 0.842 to 0.861 and grade AUC from 0.823 to 0.890 over WSI-only — significant by Wilcoxon, still trailing WSI+real (0.866 / 0.907) on every cohort."
categories:
  - Paper
tags:
  - PathGen
  - Diffusion-Models
  - Crossmodal-Generation
  - Computational-Pathology
  - Survival-Analysis
  - Conformal-Prediction
  - MCAT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- PathGen is a **conditional DDPM in a 6-group gene-embedding space** that synthesises bulk transcriptomic features from H&E WSI patch embeddings (UNI, frozen), so that an MCAT-style multimodal head (MCAT_GR) can still ingest "gene-like" inputs at inference when no real RNA assay is available. This is **WSI-level bulk transcriptomic synthesis**, not spot-level spatial transcriptomics.
- The reverse step is itself crossmodal: three rounds of MCAT-style **genomic-guided co-attention** (genes query, WSI patches key/value) plus standard transformer encoder layers — the same co-attention skeleton is reused in the MCAT_GR grade + survival head, so real and synthesised gene embeddings can be swapped without retraining.
- On TCGA-GBMLGG, WSI+synth reaches **C-Index 0.861 / AUC 0.890** vs WSI-only **0.842 / 0.823** (Wilcoxon p<0.05) and WSI+real **0.866 / 0.907** (gap not rejected, p>0.05). The pattern replicates on TCGA-KIRC/UCEC/BRCA and external CPTAC-GBM, but CPTAC-UCEC AUC stalls at **0.593** and every absolute number still favours real genes.

## Motivation

H&E WSIs are the routine substrate of cancer pathology, but the molecular tests that genuinely improve grading and prognosis — bulk RNA panels, MammaPrint-style assays — are gated by cost and infrastructure, especially in public-sector healthcare. Prior multimodal models (Pathomic Fusion, MCAT) reach state-of-the-art only when paired transcriptomics is available at inference, which is exactly the bottleneck blocking clinical translation.

PathGen reframes the problem: at deployment, **synthesise the bulk transcriptomic vector from the WSI itself** and feed it to the same multimodal head. The clinical framing is explicitly that of a screening tool — synth genes plus conformal coverage are pitched as a way to decide *which patients actually warrant the real assay*, not as a sequencing replacement.

## Core Innovation

- **Diffusion in gene-embedding space, not raw gene space.** Each of the 6 MCAT gene groups (tumour suppressors, oncogenes, protein kinases, cell differentiation markers, transcription factors, cytokines & growth factors) is encoded to a 1024-dim embedding via a small linear-ELU stack; DDPM is run over these 6 x 1024 embeddings, which keeps the diffusion target dimensionality cohort-independent (raw gene counts range 3,144-7,817 across TCGA cohorts).
- **The denoiser is crossmodal-attended.** The reverse step is a transformer with three rounds of MCAT-style co-attention between the noisy gene embedding and the WSI patch bag, rather than a plain conditional MLP. This produces co-attention maps that are 0.85-0.998 Spearman-correlated between real and synth genes, which the authors invoke to explain why downstream metrics survive a moderate fidelity gap.
- **Architectural reuse end-to-end.** Both PathGen and the downstream MCAT_GR head share the genomic-guided co-attention skeleton, so the same risk/grade predictor accepts real or synth gene embeddings without retraining. MCAT_GR adds a parallel global-attention-pooling branch for grade prediction; the risk head ensembles co-attended pathomic features with gene transformer embeddings, the grade head uses only the co-attended features (chosen empirically).
- **Conformal wrapper on top.** Split-conformal prediction sets are produced for both grade (classification score $s_i = 1 - p_{i,y_i}$) and survival (residual-on-risk score with time-bin intersection); α=0.1; stratified conditional conformal is also run per demographic group.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Synthesised transcriptomes are "highly correlated" with real | Spearman 0.436-0.717 over all gene groups (Fig 2.a.i; Suppl. Table 5) | 4 TCGA + 2 CPTAC cohorts | ⭐⭐ ("highly" is generous; UCEC at 0.436 is moderate) |
| C2 | WSI+synth significantly improves grade + survival vs WSI-only | Wilcoxon p<0.05, ANOSIM p=0.001 over Fig 2.b table | All 6 cohorts | ⭐⭐⭐ (consistent direction incl. external CPTAC; magnitudes ~0.01 on KIRC/UCEC C-Index) |
| C3 | WSI+synth is statistically indistinguishable from WSI+real | Wilcoxon p>0.05, ANOSIM p>0.05 (Suppl. Table 6) | All 6 cohorts | ⭐⭐ (non-rejection, not equivalence; deltas 0.866→0.861, 0.907→0.890, 0.697→0.681 all favour real; tests underpowered at n=18-273 cases) |
| C4 | SOTA among real-gene multimodal models | Quoted numbers vs GSCNN, DeepAttnMISL, MCAT, Pathomic Fusion, PathGen-X | TCGA-GBMLGG/KIRC/UCEC/BRCA | ⭐⭐ (beats SOTA on GBMLGG/UCEC/BRCA; loses to Pathomic Fusion on KIRC C-Index 0.697 vs 0.720; SOTA numbers taken from original papers, not re-run on matched splits) |
| C5 | Co-attention maps are clinically interpretable; oncogenes/TFs drive prediction | Real-vs-synth co-attention Spearman 0.85-0.998 (Fig 5.b); per-group contribution donuts (Fig 5.c); qualitative heatmaps (Figs 3, 4) | All cohorts | ⭐⭐ (internal real-vs-synth consistency strong; clinical interpretability admittedly outside scope) |
| C6 | Conformal predictions are well-calibrated across demographics | Marginal + stratified conditional coverage (Suppl. Tables 9-13); Beta coverage-slack derivation | All cohorts | ⭐⭐ (aggregate coverage OK; several subgroup Munc → 1.0, i.e. uninformative — survival Munc = 1.0 on TCGA-UCEC and CPTAC-UCEC) |
| C7 | PathGen synth predictions are fair across gender/age/censorship | Wilcoxon p>0.05 on per-group uncertainty deltas | TCGA cohorts | ⭐ (fairness = "no diff in uncertainty between real and synth", not performance parity across groups) |
| C8 | Can serve as a low-cost screening tool for who needs a real assay | 21.15 s/WSI gene synthesis + 0.71 s grade/risk; conformal coverage | — | ⭐ (aspirational; no prospective study identifies the decision-flip subpopulation) |

**Overall rating: ⭐⭐.** The cleanest result is C2 (WSI+synth > WSI-only) with the co-attention-similarity argument (C5 internal part) doing real explanatory work. The headline equivalence-to-real-genes claim (C3) is *non-rejection* of H0, not positive equivalence — and the deltas always favour real genes, so framing real and synth as "interchangeable" is stronger than the evidence supports. No TOST or equivalence-margin analysis. No run-to-run variance reporting (single training run per setting). **Critically, no simpler baseline:** a frozen-UNI → linear/MLP regression to gene-group expression is the obvious sanity check, and without it we cannot tell whether the diffusion machinery is doing work beyond what a deterministic regressor would. No diffusion ablations either (timesteps, three rounds of co-attention, embedding-space vs raw-gene-space). The external CPTAC-UCEC AUC of **0.593** is barely above chance and should temper the "generalises well across independent datasets" framing.

## Method & Architecture

![PathGen pipeline: WSI patches to UNI embeddings to diffusion gene synthesis to MCAT_GR](/assets/images/paper/pathgen/fig_p018_01.png)
*Figure 1: PathGen pipeline — WSI patches → frozen UNI embeddings → DDPM-based gene-group embedding synthesis → MCAT_GR multimodal grade/risk head with split-conformal uncertainty.*

### 1. WSI preprocessing and patch embedding

- Diagnostic slides tiled into **224×224 patches** at a randomly chosen magnification level (1/2/3) — magnification-agnostic training, no stain normalisation.
- Patches with normalised mean intensity > 0.8 dropped as background.
- Each patch is fed to the pre-trained **UNI** foundation model (Chen et al. 2024) → 1×1024 embedding. UNI is frozen throughout.

### 2. Gene grouping and encoding

- Following MCAT, genes are partitioned into **6 functional groups**: tumour suppressors, oncogenes, protein kinases, cell differentiation markers, transcription factors, cytokines & growth factors. Counts per group are cohort-specific (84-3712 genes/group; total 3,144-7,817).
- z-score normalised per gene.
- Per-group **gene encoder** = 4 linear layers with ELU activation → 1×1024 embedding. The diffusion target is this 6×1024 embedding tensor.

### 3. Forward diffusion

Standard DDPM noise schedule over T=1000 steps:

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\, I).$$

### 4. PathGen transformer (reverse step)

At each timestep t, the noisy gene embedding $x_t$ and the WSI patch bag are fed into the PathGen transformer:

1. **Three rounds of MCAT-style genomic-guided co-attention** — genes are queries; WSI patches are keys/values. The thrice-applied design is justified empirically as "ensuring correspondence to the input WSI" but is not ablated.
2. Standard transformer encoder layers on the co-attended representation.
3. Per-group **gene decoder** (linear-ELU stacks; final layer linear) projects denoised embeddings back to gene expression vectors of the original cardinality.

Training loss is plain DDPM only:

$$L_\text{diff} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\, \|\epsilon - \epsilon_\theta(x_t, t)\|^2\, \right].$$

Learning rate 1e-4, A100 GPU, no auxiliary supervised loss on the decoded gene values — supervision flows only through the embedding-space target.

### 5. Inference

T=1000 DDPM denoising steps from $x_T \sim \mathcal{N}(0, I)$ conditioned on the WSI bag. Reported wall-clock **21.15 s / slide** for gene synthesis + **0.71 s** for grade and risk.

### 6. MCAT_GR head

A modified MCAT (Chen 2021) adding a parallel global-attention-pooling branch for grade prediction.

- **Risk head:** consumes both the co-attended pathomic features and the gene transformer embeddings (learnable ensemble).
- **Grade head:** uses *only* the co-attended features — gene embeddings are excluded (chosen empirically; Suppl. Table 4).
- **Joint loss:** $L_{\text{MCAT\_GR}} = \lambda L_\text{grade} + (1-\lambda) L_\text{risk}$ with BCE for grade and negative-log-likelihood survival for risk; **λ = 0.3** picked from a grid scan as the joint optimum (Suppl. Fig 8 / Suppl. Table 4).

### 7. Conformal prediction

- **Grade:** split-conformal with score $s_i = 1 - p_{i,y_i}$; prediction set thresholded at $1 - \hat{q}$.
- **Survival:** scores are absolute residual on risk; conformal sets are time-bins whose risk band intersects $[\hat{r}-\hat{q},\, \hat{r}+\hat{q}]$.
- α = 0.1. Stratified conditional conformal is also computed per demographic group.

### Architecture detail

![PathGen architecture detail: diffusion model, MCAT_GR head, transformer, co-attention, encoder/decoder](/assets/images/paper/pathgen/fig_p024_01.png)
*Figure 2: Architectural breakdown — (a) PathGen diffusion model; (b) MCAT_GR head; (c) PathGen transformer with three rounds of co-attention; (d) gene encoder; (e) genomic-guided co-attention block; (f) gene decoder.*

## Experimental Results

### Datasets

- **TCGA-GBMLGG:** 912 WSIs / 745 cases (183 GBM + 562 LGG); 532/56/75/82 cases train/val/cal/test.
- **TCGA-KIRC** (renal): 485 WSIs / 462 cases; 337/34/45/46.
- **TCGA-UCEC** (uterine): 294 WSIs / 267 cases; 191/19/30/27.
- **TCGA-BRCA:** 1010 WSIs / 946 cases; 676/78/97/95. No grade annotation — survival only.
- **CPTAC-GBM** (external): 242 WSIs / 62 cases — calibration + test only (32/30).
- **CPTAC-UCEC** (external): 364 WSIs / 71 cases — 38/33 cal/test.

Transcriptomics from cBioPortal; gene-group lists from the MCAT signatures.csv. Cases missing grade, survival, or transcriptomic data are dropped. Code under **CC-BY-NC-ND 4.0** — non-commercial, no-derivatives, which is unusual for a "general crossmodal baseline."

### Main quantitative table (Fig 2.b / Suppl. Table 6)

| Cohort | Modality | C-Index (risk) | AUC (grade) |
|---|---|---:|---:|
| TCGA-GBMLGG | WSI only | 0.842 | 0.823 |
| **TCGA-GBMLGG** | **WSI + synth (PathGen)** | **0.861** | **0.890** |
| TCGA-GBMLGG | WSI + real | 0.866 | 0.907 |
| TCGA-KIRC | WSI only | 0.671 | 0.714 |
| **TCGA-KIRC** | **WSI + synth (PathGen)** | **0.681** | **0.773** |
| TCGA-KIRC | WSI + real | 0.697 | 0.778 |
| TCGA-UCEC | WSI only | 0.663 | 0.796 |
| **TCGA-UCEC** | **WSI + synth (PathGen)** | **0.673** | **0.821** |
| TCGA-UCEC | WSI + real | 0.680 | 0.828 |
| TCGA-BRCA | WSI only | 0.603 | — |
| **TCGA-BRCA** | **WSI + synth (PathGen)** | **0.720** | — |
| TCGA-BRCA | WSI + real | 0.720 | — |
| CPTAC-GBM (external) | WSI only | 0.547 | 0.736 |
| **CPTAC-GBM (external)** | **WSI + synth (PathGen)** | **0.565** | **0.865** |
| CPTAC-GBM (external) | WSI + real | 0.564 | 0.863 |
| CPTAC-UCEC (external) | WSI only | 0.518 | 0.508 |
| **CPTAC-UCEC (external)** | **WSI + synth (PathGen)** | **0.530** | **0.593** |
| CPTAC-UCEC (external) | WSI + real | 0.533 | 0.593 |

WSI+synth beats WSI-only on every row (Wilcoxon p<0.05; ANOSIM p=0.001) and is never significantly worse than WSI+real (Wilcoxon p>0.05). Note however that the WSI+real number is numerically higher on 8 of the 11 settings where the comparison is non-degenerate — the equivalence claim is non-rejection of H0, not a positive equivalence test.

### Synthesis fidelity and main results panel

![Fidelity, downstream gain, and conformal coverage across cohorts](/assets/images/paper/pathgen/fig_p019_01.png)
*Figure 3: (a) Real-vs-synth gene-group correlation (Spearman) and nMAE across cohorts; (b) significant gain of WSI+synth over WSI-only and statistical closeness to WSI+real for both risk (C-Index) and grade (AUC); (c) conformal coverage and uncertainty profiles.*

**Spearman / nMAE per cohort (Suppl. Table 5):** GBMLGG 0.713 / 0.141 · KIRC 0.717 / 0.160 · UCEC 0.436 / 0.173 · BRCA 0.642 / 0.155 · CPTAC-GBM 0.662 / 0.167 · CPTAC-UCEC 0.669 / 0.178. TCGA-UCEC at 0.436 is the weakest — flagged but not investigated.

### Comparison vs prior real-gene SOTA (paper-quoted)

On TCGA-GBMLGG C-Index: GSCNN 0.781, DeepAttnMISL 0.734, MCAT 0.817, Pathomic Fusion 0.826, **PathGen w/ real 0.866 / w/ synth 0.861**. Pathomic Fusion grade AUC 0.906 vs PathGen 0.907 (real) / 0.890 (synth). On **TCGA-KIRC C-Index, Pathomic Fusion 0.720 beats PathGen real 0.697 / synth 0.681** — though Pathomic Fusion uses pathologist-annotated ROIs while PathGen does not. PathGen-X (Krishna 2024) gets 0.81 C-Index on TCGA-GBM grade IV only. On TCGA-UCEC and TCGA-BRCA C-Index, PathGen tops DeepSets / DeepAttnMISL / MCAT (0.673-0.720 vs 0.522-0.622). These SOTA numbers are taken from the original papers, **not re-run on matched splits**, so the comparison is informal.

### Co-attention fidelity (the real internal sanity check)

![Distributed-vs-pooled parity, real-vs-synth co-attention correlation, per-group contribution](/assets/images/paper/pathgen/fig_p022_01.png)
*Figure 4: (a) Per-patch vs whole-bag grading parity (Wilcoxon p>0.05) and true-vs-false-grade attention contrast; (b) real-vs-synth co-attention map Spearman per gene group (0.85-0.998); (c) per-cohort percentage co-attention contribution by gene group.*

The real-vs-synth co-attention correlation (0.85-0.998 Spearman; nMAE 0.013-0.063) is **substantially tighter than the underlying gene-level correlation (0.436-0.717)** — this is the mechanism the authors invoke to explain why downstream metrics survive the gene-fidelity gap. Plausible, but not the same as showing the diffusion model is necessary: a deterministic regressor producing low-fidelity gene embeddings could in principle also yield faithful co-attention maps.

### Qualitative case (TCGA-LGG, grade III, male 58)

![TCGA-LGG case heatmaps and co-attention maps for real vs synth genes](/assets/images/paper/pathgen/fig_p020_01.png)
*Figure 5: TCGA-LGG case — WSI, distributed per-patch grade/risk heatmaps, and per-gene-group co-attention maps for real vs synthesised gene expression. Real and synth co-attention concentrate on the same anatomical regions despite the gene-level correlation being moderate.*

### Fairness / demographics

![Per-patient risk, uncertainty, and demographic strata sorted by predicted risk](/assets/images/paper/pathgen/fig_p023_01.png)
*Figure 6: Per-patient predicted risk, uncertainty, and demographic strata sorted by predicted risk across TCGA-GBMLGG, CPTAC-GBM, TCGA-KIRC, TCGA-UCEC, and CPTAC-UCEC.*

Wilcoxon p>0.05 on per-group uncertainty deltas between real and synth across gender, age, censorship, time-bin, and magnification. Some subgroups undercover the 0.9 conformal target but fall within the Beta(n+1-l, l) coverage-slack window for the small calibration-set sizes. **Survival Munc = 1.0 on TCGA-UCEC and CPTAC-UCEC** means the conformal set contains every time-bin — i.e. the model cannot distinguish anything on those cohorts.

### λ ablation

![Lambda sweep on TCGA-GBMLGG and TCGA-KIRC](/assets/images/paper/pathgen/fig_p026_01.png)
*Suppl. Figure 8: AUC and C-Index vs λ on TCGA-GBMLGG and TCGA-KIRC. λ = 0.3 is the joint sweet spot, but the C-Index surface is fairly flat across 0.1-0.6; AUC drops sharply only at λ ≥ 0.8.*

### Other ablations (Suppl. Table 4)

Four feature-usage settings — (1) random attention, (2) co-attention only, (3) co-attention + risk ensemble (chosen), (4) co-attention + risk + grade ensemble. Setting 3 is best in *mean* AUC/C-Index across two cohorts; ANOSIM p<0.05 is reported but pairwise effect sizes are not, and including gene embeddings in the grade head *hurts* performance — hence the asymmetric MCAT_GR design.

## Limitations

**Acknowledged in the paper:**

- Only 6 functional gene groups are synthesised, not the full transcriptome.
- Less abundant, clinically important genes would need domain expertise to evaluate.
- PathGen does not replace real transcriptomic assays.
- Conformal coverage slack is unavoidable on small calibration sets.
- The screening-tool framing requires a dedicated clinical follow-up.

**Not addressed (and load-bearing):**

- **No simpler baseline.** A frozen-UNI → linear/MLP regression to gene-group embeddings is the obvious deterministic counterpart; without it, we cannot tell whether the DDPM machinery is doing real work beyond what a regressor would.
- **No variance / multi-seed reporting.** Table 6 numbers appear to be single training runs per setting.
- **No diffusion ablations.** T=1000, three rounds of co-attention, embedding-space vs raw-gene-space diffusion, classifier-free guidance — none ablated.
- **Equivalence vs non-rejection.** "p>0.05 between WSI+synth and WSI+real" is treated as equivalence; no TOST or pre-specified equivalence margin. WSI+real is numerically better on every cohort.
- **CPTAC-UCEC AUC = 0.593** is barely above chance and is not foregrounded; the strong external-generalisation framing leans almost entirely on CPTAC-GBM.
- **STPath / Stem / spatial-resolved methods are not compared.** PathGen targets bulk WSI-level gene-group features, which is a different problem from spot-level ST prediction, but the implicit claim that bulk synthesis is sufficient for downstream prediction is not benchmarked against spatially-resolved synthesis.
- **CC-BY-NC-ND 4.0 licence.** Non-commercial, no-derivatives is unusual for a generic crossmodal baseline and limits community reuse.

## Why It Matters for Medical AI

If the synth pathway holds up under a stricter equivalence test and a simpler regression baseline, the clinical implication is concrete: a hospital running only H&E can still feed an MCAT-style multimodal grade/risk model the gene-channel it needs, then use the conformal set widths to triage which patients should actually be sequenced. That is a more realistic deployment story than "use a multimodal model that requires a bulk RNA assay every clinician already knows they can't afford." The honest version of the contribution is closer to "WSI+synth > WSI-only by a small but consistent margin, with subgroup-stratified uncertainty," not "synth ≈ real."

Two open questions matter for medical AI specifically: (i) whether the diffusion design is necessary or whether a deterministic regressor would close most of the WSI-only-to-WSI-real gap, and (ii) whether the screening-tool framing survives a retrospective decision-flip analysis on a cohort where real assays are available — i.e. for what fraction of patients would the synth-driven prediction set actually differ from the real-driven one, and where is that fraction clinically actionable.

## References

- Paper: [arXiv:2502.00568v4](https://arxiv.org/abs/2502.00568) (13 Jan 2026)
- Code: [github.com/Samiran-Dey/PathGen](https://github.com/Samiran-Dey/PathGen) (CC-BY-NC-ND 4.0; Zenodo v1.0.0)
- UNI foundation model: Chen et al., *Nat. Med.* 2024
- MCAT: Chen et al., ICCV 2021 — *Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images*
- Pathomic Fusion: Chen et al., *IEEE TMI* 2020
- PathGen-X: Krishna et al., [arXiv:2411.00749](https://arxiv.org/abs/2411.00749)
- Datasets: TCGA-GBMLGG / KIRC / UCEC / BRCA (TCGA); CPTAC-GBM / CPTAC-UCEC (external); transcriptomics from cBioPortal
