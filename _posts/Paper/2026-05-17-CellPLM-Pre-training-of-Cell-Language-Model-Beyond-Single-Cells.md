---
title: "CellPLM: Pre-training of Cell Language Model Beyond Single Cells"
excerpt: "A cells-as-tokens transformer with a Gaussian-mixture VAE prior, jointly pretrained on 9M scRNA-seq + 2M SRT cells, cuts PBMC denoising RMSE to 0.725 vs scVI 0.777 and runs ~100x faster than scGPT at inference."
categories: [Paper, BioInformatics, LLM]
tags:
  - CellPLM
  - Single-Cell
  - Foundation-Model
  - Spatial-Transcriptomics
  - VAE
  - Flowformer
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
permalink: /paper/cellplm/
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- CellPLM flips the dominant single-cell foundation-model recipe: instead of treating genes as tokens within a single cell, it treats **cells as tokens within a tissue**, so a transformer can model inter-cellular context directly.
- It jointly pretrains on **9M scRNA-seq + 2.7M SRT cells** (11.4M total) in a single encoder, integrating spatial cells via 2D sinusoidal positional encodings and constraining the latent space with a **16-component Gaussian-mixture (GMM) prior** (not a mixture-of-experts).
- Headline numbers: PBMC 5K denoising **RMSE 0.725** vs scVI 0.777 and scGPT 0.901; SRT imputation Lung2 correlation **0.318** vs the best baseline 0.227; cell-type annotation MS F1 **0.766** vs scGPT 0.703; embedding inference **0.85 s vs 129 s (scGPT) / 428 s (Geneformer)** on 48,082 cells.

## Motivation

The first wave of single-cell foundation models (scBERT, Geneformer, xTrimoGene, scGPT, tGPT) ported the NLP recipe wholesale: each cell is a "sentence" of gene tokens, and masked-language modeling teaches the transformer intra-cell gene relations. The CellPLM authors argue this is biologically wrong on three counts:

1. **scRNA-seq is a bag-of-genes, not a sequence.** There is no inherent order to exploit, so positional encodings over gene tokens do nothing useful.
2. **Cell-cell communication and shared lineage are first-class biological signals** that no gene-token model can see — the receptive field is one cell wide by construction.
3. **Single-cell corpora are 2-3 orders of magnitude smaller and noisier than text corpora** (HCA <50M cells vs CC-Net 32B sentences), so the model needs a stronger inductive bias than "predict the next gene."

For medical AI the framing matters: most downstream tasks the community actually runs (denoising, batch correction, perturbation response, spatial imputation for biopsies) depend on cellular *context* in tissue, not on intra-cell gene grammar.

## Core Innovation

Three design choices, each load-bearing:

1. **Cells-as-tokens construction.** Per cell, a learnable per-gene embedding $h_j$ is summed weighted by the log1p expression $X_{i,j}$, yielding a single $d=1024$ vector per cell. The transformer's self-attention then runs **over cells**, not over genes — a tissue / FOV becomes the sequence.
2. **Joint scRNA-seq + SRT pretraining via 2D sinusoidal PE.** Spatial cells inject their $(x, y)$ coordinates through a 2D sinusoidal positional encoding (half the dimensions for $x$, half for $y$); scRNA-seq cells get a single shared learnable placeholder PE. Both modalities flow through the same Flowformer encoder. The reconstruction loss is restricted to **only the genes measured by that cell's platform**, so a 1,000-gene SRT cell is never forced to hallucinate the remaining 12,500 scRNA-seq genes.
3. **Gaussian-mixture latent prior (16 components), not MoE.** The VAE prior is replaced from $\mathcal{N}(0, I)$ to a 16-component mixture of Gaussians, on the inductive-bias intuition that cell populations are intrinsically multi-modal (types/states). The ablation in Table 10 shows this is the single most important architectural choice.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | First pre-trained transformer encoding inter-cell relations | Architectural; supported by transformer-encoder ablation on SRT imputation (Table 10) | Lung2 / Liver2 | ⭐⭐ |
| C2 | Joint scRNA-seq + SRT pretraining helps downstream SRT | "From scratch" CellPLM collapses on SRT (Lung corr 0.058) vs fine-tuned 0.318 (Table 3) | Lung2, Liver2 | ⭐⭐ |
| C3 | GMM prior is necessary vs isotropic Gaussian | Ablation w/o MoG: Lung corr 0.318 → 0.258; hPancreas F1 0.749 → 0.711 (Table 10) | MS, hPancreas, Lung2, Liver2 | ⭐⭐⭐ |
| C4 | ~100x faster inference than scGPT / Geneformer | Table 1: 0.85 s vs 129.19 s / 428.24 s on 48,082 cells (A100 40GB) | one dataset, single run, no std | ⭐⭐ |
| C5 | Beats scGPT on cell-type annotation | MS F1 0.766 vs 0.703; hPancreas F1 0.749 vs 0.718 (Table 4) | MS, hPancreas | ⭐⭐ |
| C6 | Best denoising over 8 baselines incl. scGPT | PBMC 5K RMSE 0.725 vs DCA 0.775, scVI 0.777; Jurkat 0.391 vs scVI 0.416 (Table 2) | PBMC 5K, Jurkat | ⭐⭐⭐ |
| C7 | Better zero-shot clustering than scGPT / Geneformer / PCA | ARI 0.867 vs 0.836 / 0.461 / 0.843 (Section 4.1) | **single dataset (Li 2020 aortic), no variance** | ⭐ |
| C8 | Beats GEARS + scGen on perturbation prediction | Figure 7 bar chart (no numerical table) | Adamson, Norman | ⭐⭐ |
| C9 | Pretrained knowledge transfers to new datasets (RQ1) | Pretrain-vs-scratch deltas are large and consistent across Tables 2, 3, 4 | 4 datasets | ⭐⭐⭐ |

**Honest read.** C3 (GMM prior), C6 (denoising), and C9 (transfer gap) are the well-supported claims — each has a clean ablation or a multi-dataset comparison with reported variance. C7 is the weakest: zero-shot clustering rests on a *single* held-out dataset (Li 2020 aortic) with no variance, and the extreme gap over Geneformer (ARI 0.867 vs 0.461) more likely reflects Geneformer's rank-only tokens being mismatched to clustering than CellPLM superiority. C4's 100x speedup is real but the framing hides a confound — scGPT and Geneformer are batched to 64 / 256 cells (OOM-limited) while CellPLM is batched to all 48,082, so the comparison conflates architecture and memory efficiency.

## Method & Architecture

![CellPLM pretraining pipeline](/assets/images/paper/cellplm/fig_p004_02.png)
*Figure 1: CellPLM pretraining. Cells from scRNA-seq and SRT are embedded into single-vector tokens via a weighted sum of learnable gene embeddings, optionally augmented with 2D sinusoidal positional encodings for spatial cells, processed by a 4-layer Flowformer encoder, projected into a 16-component Gaussian-mixture latent space, and decoded back to gene expression with a per-sample batch embedding.*

The model is an encoder-decoder VAE-style transformer with four modules.

**1. Gene Expression Embedder.** For cell $i$ with log1p-normalized expression $X_i \in \mathbb{R}^k$ (k = 13,500 inner-join genes), assign a learnable embedding $h_j \in \mathbb{R}^d$ per gene and aggregate:

$$E_i = \sum_{j=1}^{k} X_{i,j} h_j$$

Implemented with sparse ops (zero-rate up to 90%). Output: one $d=1024$ vector per cell.

**2. Positional Encoding.**
- *SRT cells:* 2D sinusoidal PE over $(x, y)$ tissue coordinates, normalized/truncated to $[0, 100)$; half the dimensions encode $x$, half $y$.
- *scRNA-seq cells:* a single shared learnable placeholder PE $p'$ (no coordinates exist).

Input to the encoder is $H^{(0)} = E + P$. This is the entire mechanism by which the two modalities are "jointly" trained — they pass through the same transformer, differing only in what PE carries.

**3. Transformer Encoder.** $L=4$ stacked **Flowformer** (Wu et al., 2022) layers — a linear-complexity transformer, necessary because $N$ (cells per FOV) can reach ~10,000. Hidden dim 1024. Self-attention is over cells.

**4. Gaussian-Mixture Latent Space.** Instead of $\mathcal{N}(0, I)$, the prior is a mixture of Gaussians with $L=16$ components:

$$p(y_i; \pi) = \text{Multinomial}(\pi), \quad p(z_i | y_i) = \prod_l \mathcal{N}(\mu_{y_i,l}, \text{diag}(\sigma_{y_i,l}^2))$$

The inference network outputs $\hat\mu_i, \hat\sigma_i^2$ and a soft cluster assignment $\hat\pi_i$.

**5. Batch-aware Decoder.** A 2-layer MLP (hidden 1024) taking $h^{(0)} = z + b$ with a per-sample learnable batch embedding $b$ (scVI-style), absorbing batch effects so the latent stays "batch-free."

**6. Pretraining objective.** Mask **75% of cells**, and within each selected cell mask **25% of measured genes**. Loss is MSE on masked entries plus the two KL terms of the denoising VLB:

$$\mathcal{L}_{pretrain} = \mathcal{L}_{MSE} + \mathcal{L}_{cond} + \mathcal{L}_Y$$

Crucially, MSE is computed **only on genes measured by that cell's platform** — SRT (~1,000-gene panels) and scRNA-seq (~13,500 genes) are not forced to predict each other's unmeasured genes.

**Compute.** 82M parameters, 8x V100 16GB, <24 h.

![scRNA-seq vs SRT input representation](/assets/images/paper/cellplm/fig_p015_03.png)
*Figure 2: Input representation. scRNA-seq cells are bag-of-genes vectors with no spatial coordinates; SRT cells carry $(x, y)$ tissue coordinates that drive the 2D sinusoidal positional encoding.*

## Experimental Results

**Pretraining corpus (11.4M cells).** scRNA-seq from HTCA (4.7M), HCA (1.4M), GEO (2.6M) plus deduplicated atlases; SRT from NanoString CosMx FFPE NSCLC (2.7M cells, 1,000 genes). Gene set frozen to the **13,500-gene inner-join** across scRNA-seq datasets.

**Task 1 — scRNA-seq denoising (Table 2, RMSE / MAE lower is better).**

| Method | PBMC 5K RMSE | PBMC 5K MAE | Jurkat RMSE | Jurkat MAE |
|---|---|---|---|---|
| DeepImpute | 1.168 ± 0.018 | 1.051 ± 0.025 | 0.786 ± 0.006 | 0.557 ± 0.003 |
| scGNN 2.0 | 1.376 ± 0.015 | 1.237 ± 0.019 | 1.001 ± 0.016 | 0.917 ± 0.021 |
| SAVER | 0.884 ± 0.001 | 0.748 ± 0.001 | 0.569 ± 0.001 | 0.472 ± 0.001 |
| DCA | 0.775 ± 0.002 | 0.621 ± 0.002 | 0.423 ± 0.001 | 0.351 ± 0.001 |
| scVI | 0.777 ± 0.005 | 0.623 ± 0.004 | 0.416 ± 0.001 | 0.344 ± 0.002 |
| MAGIC | 0.793 ± 0.001 | 0.639 ± 0.001 | 0.424 ± 0.001 | 0.351 ± 0.002 |
| scGPT (fine-tuned) | 0.901 ± 0.001 | 0.565 ± 0.001 | 0.711 ± 0.001 | 0.498 ± 0.001 |
| CellPLM (zero-shot) | 0.854 ± 0.001 | 0.692 ± 0.000 | 0.517 ± 0.001 | 0.426 ± 0.000 |
| CellPLM (from scratch) | 0.761 ± 0.009 | 0.571 ± 0.011 | 0.395 ± 0.003 | 0.320 ± 0.003 |
| **CellPLM (fine-tuned)** | **0.725 ± 0.001** | **0.551 ± 0.001** | **0.391 ± 0.001** | **0.320 ± 0.001** |

**Task 2 — SRT imputation (Table 3, Corr / Cosine higher is better).**

| Method | Lung2 Corr | Lung2 Cosine | Liver2 Corr | Liver2 Cosine |
|---|---|---|---|---|
| SpaGE | 0.227 ± 0.011 | 0.352 ± 0.015 | 0.253 ± 0.014 | 0.376 ± 0.005 |
| stPlus | 0.177 ± 0.021 | 0.360 ± 0.014 | 0.224 ± 0.010 | 0.399 ± 0.012 |
| gimVI | 0.130 ± 0.010 | 0.325 ± 0.010 | 0.163 ± 0.019 | 0.338 ± 0.010 |
| Tangram | 0.123 ± 0.005 | 0.285 ± 0.008 | 0.168 ± 0.024 | 0.309 ± 0.008 |
| CellPLM (zero-shot) | 0.119 ± 0.024 | 0.327 ± 0.011 | 0.141 ± 0.013 | 0.322 ± 0.145 |
| CellPLM (from scratch) | 0.058 ± 0.020 | 0.370 ± 0.013 | 0.024 ± 0.039 | 0.352 ± 0.011 |
| **CellPLM (fine-tuned)** | **0.318 ± 0.015** | **0.481 ± 0.011** | **0.328 ± 0.011** | **0.481 ± 0.010** |

**Task 3 — Cell type annotation (Table 4, macro F1 / Precision).**

| Method | MS F1 | MS Precision | hPancreas F1 | hPancreas Precision |
|---|---|---|---|---|
| CellTypist | 0.667 ± 0.002 | 0.693 ± 0.001 | 0.708 ± 0.023 | 0.736 ± 0.025 |
| ACTINN | 0.628 ± 0.012 | 0.634 ± 0.009 | 0.705 ± 0.005 | 0.709 ± 0.006 |
| SingleCellNet | 0.637 ± 0.001 | 0.700 ± 0.001 | 0.739 ± 0.006 | 0.761 ± 0.004 |
| TOSICA | 0.578 | 0.664 | 0.656 | 0.661 |
| scBERT (fine-tuned) | 0.599 | 0.604 | 0.685 | 0.699 |
| scGPT (fine-tuned) | 0.703 | 0.729 | 0.718 | 0.735 |
| CellPLM (from scratch) | 0.709 ± 0.007 | 0.732 ± 0.015 | 0.689 ± 0.034 | 0.682 ± 0.037 |
| **CellPLM (fine-tuned)** | **0.766 ± 0.007** | **0.803 ± 0.008** | **0.749 ± 0.010** | **0.753 ± 0.010** |

**Inference speed (Table 1, 48,082 cells, A100 40GB, embedding only).** CellPLM **0.85 s**, scGPT 129.19 s, Geneformer 428.24 s. Note the comparison is not batch-matched — scGPT and Geneformer hit OOM and run at batch 64 / 256, while CellPLM processes the whole set at once.

**Zero-shot clustering (Section 4.1, Li 2020 aortic, single dataset).** CellPLM ARI **0.867** / NMI **0.823**; PCA 0.843 / 0.812; scGPT 0.836 / 0.818; Geneformer 0.461 / 0.586.

![Zero-shot clustering on Li 2020 aortic — CellPLM](/assets/images/paper/cellplm/fig_p007_01.png)
*Figure 3a: CellPLM zero-shot UMAP, ARI 0.867 / NMI 0.823.*

![Zero-shot clustering on Li 2020 aortic — PCA](/assets/images/paper/cellplm/fig_p007_02.png)
*Figure 3b: PCA baseline, ARI 0.843 / NMI 0.812 — competitive without any learned model.*

![Zero-shot clustering on Li 2020 aortic — scGPT](/assets/images/paper/cellplm/fig_p007_03.png)
*Figure 3c: scGPT zero-shot, ARI 0.836 / NMI 0.818.*

![Zero-shot clustering on Li 2020 aortic — Geneformer](/assets/images/paper/cellplm/fig_p007_04.png)
*Figure 3d: Geneformer zero-shot, ARI 0.461 / NMI 0.586 — rank-only tokens hurt clustering.*

Patient-batch correction (same dataset, color = donor):

![Patient batch — CellPLM](/assets/images/paper/cellplm/fig_p007_06.png)
*Figure 4a: CellPLM mixes patients without fine-tuning.*

![Patient batch — PCA](/assets/images/paper/cellplm/fig_p007_07.png)
*Figure 4b: PCA — donor structure still visible.*

![Patient batch — scGPT](/assets/images/paper/cellplm/fig_p007_08.png)
*Figure 4c: scGPT — partial mixing.*

![Patient batch — Geneformer](/assets/images/paper/cellplm/fig_p007_09.png)
*Figure 4d: Geneformer — donor structure persists.*

**Ablations (Table 10) — three studies on cell-type classification and SRT imputation.**

- **w/o Mixture of Gaussian** (replace GMM with isotropic Gaussian): largest drop — Lung corr 0.318 → 0.258, Liver corr 0.328 → 0.232, hPancreas F1 0.749 → 0.711.
- **w/o Latent Distribution** (deterministic AE instead of VAE): intermediate drop.
- **w/o Transformer Encoder** (MLP, 85M → 50M params): big hit on SRT imputation (Lung corr 0.318 → 0.244), small hit on hPancreas F1 — confirms the transformer's value is inter-cell modeling, which only spatial data fully exposes.

![scVI clustering comparison](/assets/images/paper/cellplm/fig_p023_02.png)
*Figure 5: Appendix H.1 — scVI zero-shot clustering (ARI 0.843 / NMI 0.823) for context against the page-7 panel.*

**Qualitative — gene embedding structure.**

![Gene embedding UMAP](/assets/images/paper/cellplm/fig_p024_01.png)
*Figure 6: UMAP of learned gene embeddings — HLA Class I (red) and Class II (orange) cluster separately, suggesting the model picks up gene-family structure unsupervised.*

**Qualitative — cell-cell communication (speculative).**

![Liver CosMx attention stream plot](/assets/images/paper/cellplm/fig_p021_01.png)
*Figure 7: Attention-derived "cell-cell communication" stream plot in a Liver CosMx FOV. Purely qualitative — no quantitative validation against CellChat / NicheNet.*

## Limitations

**Authors acknowledge:**
- Gene set frozen at the 13,500-gene inner-join — limits future extensibility.
- Pretraining used only 11.4M cells (vs ~50M in HCA) — they call for scaling.
- Privacy / over-reliance societal risks (Appendix C).

**Authors do not address — flag these:**
- **Zero-shot clustering rests on one dataset.** Section 4.1 reports a single held-out dataset (Li 2020 aortic) with no variance. The headline ARI 0.867 vs Geneformer 0.461 is the most striking number in the paper, but a single-dataset, no-variance, like-vs-unlike-objective comparison is the thinnest evidence in the table.
- **No scFoundation / UCE / GeneCompass baselines.** All contemporaneous single-cell foundation models with comparable or larger pretraining corpora are absent. The baseline set is essentially scGPT + Geneformer + scBERT plus task-specific tools.
- **Single SRT platform.** Pretraining SRT is *only* NanoString CosMx NSCLC FFPE — 10x Visium, MERFISH, Stereo-seq, slide-seq are absent. Every "joint scRNA-seq + SRT" claim is therefore conditioned on tumor-microenvironment-shaped spatial knowledge from one platform.
- **Strong oncology bias** in the scRNA-seq pretraining mix (HTCA + multiple GEO tumor studies) — unclear performance on healthy/developmental atlases.
- **Cell-cell communication claim (Figure 6) is purely qualitative** — no quantitative validation against known ligand-receptor databases (CellChat, NicheNet).
- **Scaling laws unexplored.** Only one model size (82M) trained — no evidence the architecture scales to scGPT's 50M→650M or scFoundation's 100M regime.
- **75% cell mask / 25% gene mask is reported but not ablated** — these are aggressive rates and almost certainly matter.
- **Cross-species transfer untested** (Geneformer and UCE both demonstrate this).
- **The "tissue = sentence" framing assumes co-located cells at inference.** For a query mixing cells from unrelated samples, attention behavior is undefined.

## Why It Matters for Medical AI

The clinically-relevant single-cell tasks — denoising of low-depth biopsies, batch correction across patients/sites, imputing missing genes in spatial panels, and predicting perturbation response — all depend on *cellular context within tissue*, not on intra-cell gene grammar. CellPLM is the first foundation model to put that context directly into the attention pattern, and the ablation shows the inductive bias pays off most exactly where it should: SRT imputation collapses without the transformer encoder. For oncology specifically (where the pretraining corpus is concentrated), CellPLM offers a substantially faster encoder for cohort-scale analyses — 0.85 s vs 129 s for ~48k cells is the difference between an interactive workflow and an overnight job.

The honest caveat is dosage: with one SRT platform and one tumor type in pretraining, the "spatial" half of the claim is narrower than the framing suggests, and clinical deployment will need either re-pretraining on the target platform/tissue or a serious evaluation campaign first.

## References

- Paper: [CellPLM: Pre-training of Cell Language Model Beyond Single Cells (ICLR 2024, OpenReview)](https://openreview.net/forum?id=BKXvPDekud)
- Code: [github.com/OmicsML/CellPLM](https://github.com/OmicsML/CellPLM)
- Related foundation models: scBERT (Yang et al., 2022), Geneformer (Theodoris et al., 2023), scGPT (Cui et al., 2024), scFoundation (Hao et al., 2023), UCE (Rosen et al., 2023), GeneCompass (Yang et al., 2024)
- Linear-complexity transformer used: Flowformer (Wu et al., ICML 2022)
- Mixture-of-Gaussians VAE prior origin: VaDE (Jiang et al., 2017), GMVAE (Dilokthanakul et al., 2017)
- Closest comparison for the SRT half: scVI (Lopez et al., 2018), Tangram (Biancalani et al., 2021), gimVI (Lopez et al., 2019)
