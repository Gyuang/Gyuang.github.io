---
title: "DeepCell Types: Generalized Cell Phenotyping for Spatial Proteomics with Language-Informed Vision Models"
excerpt: "A channel-wise transformer that fuses per-marker CNN features with DeepSeek-R1-distilled Llama-3.3-70B text embeddings reaches zero-shot leave-one-dataset-out F1 of 0.551 across 9 imaging platforms — but the headline excludes rare cell types and leaves the key ablations unrun."
categories:
  - Paper
  - Spatial-Proteomics
  - LLM
permalink: /paper/deepcell-types/
tags:
  - DeepCell-Types
  - Spatial-Proteomics
  - Cell-Phenotyping
  - Channel-wise-Transformer
  - LLM-Embeddings
  - Gradient-Reversal
  - CODEX
  - MIBI
  - IMC
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-28
last_modified_at: 2026-05-28
---

## TL;DR

- **DeepCell Types (DCT)** is a single checkpoint that phenotypes cells across **9 multiplexed-imaging platforms** (CODEX, MIBI, IMC, IBEX, MICS, CellDIVE, CyCIF, InSituPlex, Vectra) by tokenizing **each marker channel** with a CNN image embedding plus a frozen LLM **text embedding** of the marker name, then fusing channel tokens with a 5-layer transformer (no positional encoding) and predicting cell type via CLIP-style contrastive matching to text embeddings of cell-type names.
- The language source is **not BioBERT / CLIP** — it is a two-stage **DeepSeek-R1-distilled Llama-3.3-70B-Instruct** pipeline: an "explainer" produces a JSON-templated description per marker / cell type, and its companion embedder yields an 8192-d vector that is linearly projected to 256.
- Headline: zero-shot leave-one-dataset-out (LODO) **aggregated F1 = 0.551 ± 0.08** on 13 held-out datasets vs. XGBoost 0.458 ± 0.172 and MAPS 0.427 ± 0.117 — but **rare cell types (<25% of experiments or <500 samples) are excluded** from this aggregate, and the in-distribution gap to XGBoost is only ~0.008.

## Motivation

Spatial-proteomics platforms (IMC, MIBI, CODEX, IBEX, MICS, CyCIF, InSituPlex, Vectra, CellDIVE) each ship idiosyncratic antibody panels — different markers, different channel counts, different naming conventions. Each new dataset has historically required either (a) re-labeling and re-training a per-panel classifier (CellSighter, MAPS, STELLAR), or (b) hand-engineered prior cell-type ↔ marker maps (Celesta, Astir). Existing transfer learning works only when source and target panels overlap heavily.

The HuBMAP-scale opportunity is direct: millions of cells across heterogeneous panels that today's per-panel classifiers cannot consolidate into a single phenotyping model. DCT reframes panel variability as a **vocabulary problem** solvable by letting an LLM tell the vision model what each channel *means*.

## Core Innovation

- **Per-channel tokenization with language fusion.** A token is built per marker channel as `token_c = CNN(image_c) + Linear(LLM_text_emb(marker_name_c))`. The vision branch is marker-agnostic (shared CNN weights across all channels); the meaning of each channel enters through the text embedding, not through a per-marker head.
- **Channel-wise self-attention, no positional encoding.** A 5-layer transformer attends over the channel tokens with a learnable `[CLS]`. The absence of positional encoding plus a padding mask makes the architecture **panel-length and order invariant** — one model can ingest panels of any size up to `C_max = 75`.
- **CLIP-style cell-type prediction.** Cell-type names are embedded by the same LLM stack; cosine similarity between the cell's `[CLS]` and each candidate cell-type name embedding gives the prediction. This decouples the inference-time prediction set from the training-time label set — at inference, only valid cell types for a given tissue need be considered.
- **Marker-positivity as attention.** The `[CLS] → marker` attention weights in the final layer, normalized to `[0, 1]`, double as **per-marker positivity scores** — no extra head.
- **Modality invariance via gradient reversal.** A modality classifier is attached to `[CLS]` through a Gradient Reversal Layer (Ganin & Lempitsky 2015) to push the encoder toward platform-invariant features, with its weight ramped by the quartic root of epoch count.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | A single model phenotypes cells across 9 heterogeneous platforms without per-panel retraining | Same checkpoint evaluated across 9 platforms and 13 LODO splits (Fig. 2e, 2g) | Expanded TissueNet | ⭐⭐⭐ |
| C2 | Language-informed vision improves zero-shot generalization over mean-intensity baselines | Aggregated F1 0.551 vs. XGBoost 0.458 vs. MAPS 0.427 on 13 LODO datasets (Fig. 2g) | Expanded TissueNet LODO | ⭐⭐ |
| C3 | The channel-wise transformer is panel-length and order invariant | Architectural argument (no positional encoding + padding mask); panels of varying counts up to `C_max=75` are trained | All training panels | ⭐⭐ (architectural; no explicit channel-permutation or unseen-marker test reported) |
| C4 | The model focuses on biology, not modality | GRL + adversarial modality classifier; t-SNE colored by cell type (Fig. 2d) vs. modality (Fig. S4b) shows cell-type structure and modality intermixing | Expanded TissueNet | ⭐⭐ (visual + adversarial loss; **no quantitative invariance metric, no GRL-ablated comparison**) |
| C5 | Better marker-positivity than Nimbus, including under domain shift | Fig. 2f (authors' dataset) + Fig. S4f (Nimbus's native dataset): F1 / Precision / Recall / Accuracy bars | Both | ⭐⭐ (clear on authors' dataset; on Nimbus's dataset DCT *underperforms* the native model — only the *relative* drop is smaller) |
| C6 | Dynamic inference-time prediction-set binding improves accuracy | Fig. S4e: 0.528 → 0.551 aggregated F1 (Static → Dynamic) | LODO | ⭐⭐ |
| C7 | Robust performance across cell types, tissues, modalities | Per-bin F1 plots (Fig. 2e, S4d) | In-distribution | ⭐⭐ (tail bins have wide variance; **Fig. 2g shows several categories where DCT ties or loses to MAPS**) |
| C8 | Expert-in-the-loop labeling produces high-quality, consistent labels for 9.8 M cells | DeepCell Label interface description (Fig. S3); manual gating protocol | Internal | ⭐ (**no reported inter-annotator agreement, no label-noise audit**) |
| C9 | The approach generalizes field knowledge into a single, continuously-improvable model | Discussion claim | — | ⭐ (forward-looking, not experimentally tested) |

**Honest read.** C1 — one checkpoint truly does run across 9 platforms with one interface — is the genuinely well-supported headline. C2 holds, but the effect size is moderate: a 0.551 vs. 0.458 absolute F1 gap is real, yet both numbers indicate that **zero-shot cross-panel phenotyping is still a hard, unsolved problem**, and the abstract's "superior accuracy and generalizability" reads stronger than the numbers warrant. Several pieces of due-diligence are missing:

1. **"Panel-agnostic" is architectural + dropout, not directly tested.** The claim rests on (a) no positional encoding and (b) random dropout of 8 channels per training sample. There is no channel-permutation experiment at inference and no **unseen-marker** generalization test — LODO splits hold out *datasets*, not specific markers.
2. **Baselines are thin.** Mesmer is *used upstream for segmentation*, not benchmarked as a phenotyping baseline. The actual phenotyping baselines are XGBoost, MAPS, Nimbus. **STELLAR (graph-NN) and CellSighter (CNN) are cited but absent from LODO**, even though both are the obvious cross-panel comparators.
3. **Rare cell types are excluded from the headline.** The 0.551 LODO aggregate explicitly drops categories present in <25% of experiments or <500 samples — so the headline says nothing about the **long tail**, and the per-category bars in Fig. 2g show DCT tying or losing on several tail categories.
4. **No inter-annotator agreement** on Expanded TissueNet labels. For a model whose ceiling is set by label quality across 9.8 M cells across 18+ source datasets, this is a meaningful omission.
5. **Missing ablations**: (i) vision-only DCT (no LLM, e.g. channel-name one-hots) would isolate how much the LLM is doing vs. the architecture; (ii) no-GRL ablation would test whether modality invariance is actually GRL or just the contrastive loss; (iii) LLM-source sensitivity (BioBERT, ClinicalBERT, GPT-4 descriptions vs. DeepSeek-R1-distilled-Llama-3.3) is not run.
6. **In-distribution gap is small** (DCT 0.903 vs. XGBoost 0.895 vs. MAPS 0.883). All the headline value lives in the LODO zero-shot setting, not in beating baselines on a single panel.

## Method & Architecture

![DeepCell Types architecture and benchmarks](/assets/images/paper/deepcell-types/fig_p016_01.png)
*Figure 1: DeepCell Types architecture — per-channel image + LLM-derived language embeddings fused by a channel-wise transformer, contrastive cell-type prediction, modality-invariant latent space via GRL, and zero-shot LODO gains over XGBoost and MAPS.*

### 1. Per-cell input tensor

After Mesmer whole-cell segmentation, for each cell extract a **64×64** patch from each of `C` marker channels, plus a self-mask (central cell) and a neighbor-mask (other cells in patch). Stack to shape `(C, 3, 64, 64)`; zero-pad to `C_max = 75` with a binary padding mask.

### 2. Image encoder

An **11-layer 2D-CNN** (kernel 3, padding 1, interleaved strides 1/2, SiLU + BN), applied independently per channel by reshaping `(B, C_max, 3, 64, 64) → (B·C_max, 3, 64, 64)`; output reshaped to `(B, C_max, 256)`. The CNN is **marker-agnostic** — all channels share weights.

### 3. Language encoder (two-stage, both frozen)

- **Explainer**: DeepSeek-R1-distilled **Llama-3.3-70B-Instruct**, prompted per marker (or cell type) for "functionality, cell-type association, alternative names" in a fixed JSON schema. Cell-type prompts additionally inject the dataset's channel list and the dataset's cell-type list.
- **Embedder**: the companion text-embedding model from the same distilled stack, producing an **8192-d vector** that is **linearly projected to 256** to match the image-token dim.

This pipeline is explicitly **not BioBERT and not CLIP-text** — it is a single LLM stack used as both verbalizer and embedder.

### 4. Channel-wise transformer

For each channel,
$$
\text{token}_c = \text{img\_emb}_c + \text{Linear}(\text{text\_emb}_c).
$$
A learnable `[CLS]` is prepended. A **5-layer encoder** (hidden 256, FFN 512, padding mask hides dummy channels) attends over channel tokens. **No positional encoding** — this is what makes the model panel-length / order invariant.

The `[CLS] ↔ marker` attention weights in the final layer, normalized to `[0, 1]`, are used as per-marker **positivity scores** with no extra head.

### 5. Cell-type prediction (CLIP-style)

The cell type's text embedding (same language encoder, cell-type prompt) is the target. Cosine similarity between the cell's `[CLS]` embedding and the *correct* cell-type-name embedding is maximized against all *incorrect* names in the batch. At inference, the candidate set can be restricted per tissue (**dynamic prediction-set binding**, Fig. S1c / S2d), giving full training-set / inference-set decoupling.

![Per-channel input tokenization and attention with padding mask, GRL modality head, prediction-set decoupling](/assets/images/paper/deepcell-types/fig_p018_01.png)
*Figure 2: Supplementary S2 — per-channel input (raw + self-mask + neighbor-mask), channel-wise attention with padding mask, gradient-reversal modality head, and training/inference prediction-set decoupling.*

### 6. Losses (Algorithm 1)

- **Focal-CLIP contrastive loss** ($\gamma = 2.0$) on the symmetric image ↔ text similarity matrix, for cell-type ID.
- **BCE with label smoothing 0.2** on normalized attention weights vs. expert-gated marker-positivity labels.
- **Cross-entropy with label smoothing 0.01** for the modality classifier, attached to `[CLS]` via a **Gradient Reversal Layer**; its weight ramps with the quartic root of epoch count so the encoder learns cell-type discrimination first and only later unlearns modality cues.

### 7. Training

20 epochs, RAdam, lr 1e-4. Augmentations: random flip / rotation / resize; **random drop of 8 marker channels per sample** (training-time analogue of the panel-agnostic claim); Gaussian noise ($\sigma = 0.005$) added to marker and cell-type name embeddings.

### 8. Baselines

XGBoost and MAPS receive only **mean intensity per channel** (no images). XGBoost: NaN for absent markers, 160 epochs, default params. MAPS: zero for absent markers + cell-size feature, 500 epochs. Marker-positivity comparison is against **Nimbus** (pretrained, run via its official inference repo).

## Dataset — Expanded TissueNet

![Expanded TissueNet composition](/assets/images/paper/deepcell-types/fig_p015_01.png)
*Figure 3: Expanded TissueNet — 9.8 M cells across 9 imaging platforms, 17 tissues, and 8 lineages spanning 48 specific cell types, assembled via human-in-the-loop labeling in DeepCell Label.*

- **Scale**: ~9.8 M labeled cells.
- **Sources**: 18+ published datasets + unpublished HuBMAP-deposited data. A few datasets are held under pre-publication agreements and excluded from the public release.
- **Coverage**: 9 imaging platforms (IMC, CODEX, MIBI dominate; IBEX, MICS, CellDIVE, CyCIF, InSituPlex, Vectra are tail), 17 tissue types, **8 broad lineages → 48 specific cell types** with HuBMAP Cell-Ontology IDs (e.g. `Treg → CL:0000815`). "Tumor" is listed without a CL ID.
- **Markers**: 262 unique proteins; **average 24.6 markers per dataset**.
- **Preprocessing**: resample to 0.5 µm/pix, 99th-percentile threshold per FOV-per-channel, min-max normalize, zarr storage. Whole-cell masks from Mesmer. 64×64 patches centered on each cell.
- **Labeling**: human-in-the-loop via the (extended) DeepCell Label browser tool. Marker positivity is established by **manual gating of mean signal intensity per marker per cell type per dataset**.

![Long-tailed distribution and sparse cell-type × dataset presence](/assets/images/paper/deepcell-types/fig_p017_01.png)
*Figure 4: Supplementary S1 — cell-type abundance is long-tailed; the dataset × cell-type presence matrix is sparse (most panels resolve only a subset of the 48 cell types); per-tissue valid cell-type sets enable inference-time prediction-set binding.*

**Detectable biases**: severe platform skew (IMC / CODEX / MIBI dominate, the other 6 platforms hold ~0.06 M of 9.8 M cells); long-tailed tissue and lineage distribution; the "48 cell types" is the union of panels, not what any one dataset can resolve. No inter-annotator agreement reported.

## Experimental Results

| Setting | Dataset | Metric | XGBoost | MAPS | Nimbus | **DeepCell Types** |
|---|---|---|---|---|---|---|
| In-distribution test set (Fig. S4d) | Expanded TissueNet held-out | Aggregated F1 | 0.895 ± 0.046 | 0.883 ± 0.061 | — | **0.903 ± 0.051** |
| Zero-shot LODO, 13 large datasets, common cell types (Fig. 2g) | Expanded TissueNet LODO | Aggregated F1 | 0.458 ± 0.172 | 0.427 ± 0.117 | — | **0.551 ± 0.08** |
| Marker positivity, authors' data (Fig. 2f) | Expanded TissueNet | F1 / P / R / Acc | — | — | lower across all four | **higher across all four** |
| Marker positivity, Nimbus's native data (Fig. S4f) | Nimbus dataset | F1 / P / R / Acc | — | — | higher (native) | **lower, but degrades less than Nimbus on Expanded TissueNet** |
| Inference-time prediction-set binding (Fig. S4e) | LODO zero-shot | Aggregated F1 | — | — | — | Static 0.528 ± 0.100 → **Dynamic 0.551 ± 0.08** |
| Inference cost (Fig. S4c) | — | ms / cell / channel | — | — | — | **0.24 ± 0.07** (linear in cell count) |

![In-distribution confusion, modality-colored latent, per-bin F1, dynamic binding, cross-domain marker-positivity](/assets/images/paper/deepcell-types/fig_p020_01.png)
*Figure 5: Supplementary S4 — in-distribution confusion matrix, modality-colored latent space (intermixing), linear inference cost (~0.24 ms / cell / channel), per-tissue / lineage / modality F1 vs. baselines, dynamic prediction-set binding gain (0.528 → 0.551), and cross-domain marker-positivity vs. Nimbus.*

### Qualitative findings worth flagging

- **In-distribution gap is small.** DCT (0.903) beats XGBoost (0.895) by ~0.008 — within the reported deviation. All the value lives in the LODO setting.
- **Dynamic prediction-set binding contributes ~+0.023 F1** of the zero-shot gain — meaningful but not dominant; the rest comes from the language-informed architecture itself.
- **Latent space (Fig. 2d / S4b)**: NCA → t-SNE shows cell-type clusters; modality recolor shows intermixing. This is a *visualization*, not a quantitative invariance metric.
- **Confusion matrix (Fig. S4a)** confirms the expected hard pairs (lymphocyte subtypes; macrophage / monocyte / DC); no off-diagonal mass on biologically unrelated pairs is reported.
- **Marker-positivity cross-domain** (Fig. 2f / S4f): both Nimbus and DCT drop on the other group's data; the paper claims DCT degrades less. Delta sizes are shown only in bar charts.

## Limitations

**Authors admit:**

- Out-of-domain degradation persists; new modalities / tissues may still need labeling and finetuning.
- Generalization *across cell types* (vs. across panels) is unsolved — explicitly called an open challenge given tissue-specific cell types.
- 64×64 patch crops cap the receptive field; multi-scale models with global tissue context are future work.
- Some datasets are held under pre-publication agreements and not in the public release.

**Authors do not address:**

- **No inter-annotator agreement / label-noise audit** on Expanded TissueNet — both the ceiling and the LODO numbers are anchored on labels of unmeasured quality.
- **No language-encoder ablation** (vision-only baseline, alternative LLMs, explainer vs. embedder split).
- **No GRL ablation**, despite gradient reversal being one of three central training-time ingredients.
- **No comparison to STELLAR (graph-NN) or CellSighter (CNN)** on the LODO benchmark, even though both are in related work; phenotyping baselines are limited to mean-intensity classifiers (XGBoost, MAPS) and marker-positivity (Nimbus). Mesmer is *used* upstream for segmentation, not benchmarked.
- **"Panel-agnostic" is not stress-tested** by channel-order permutation at inference, marker-name paraphrase robustness, or unseen-marker generalization — LODO splits hold out *datasets*, not specific markers (training-time dropout implicitly trains for missing-marker robustness).
- **"Tumor"** has no CL ontology ID and is one of the largest lineages — its definition is implicitly heterogeneous across cancer types.
- Single training seed; no run-to-run variance reported.
- Cellpose / Mesmer segmentation errors propagate into Expanded TissueNet labels but are not characterized.

## Why It Matters for Medical AI

Cell phenotyping is the foundational step that turns multiplexed-imaging pixels into biological measurements that downstream clinical and translational analyses (immune-microenvironment characterization, treatment-response correlates, spatial-statistics readouts) actually consume. The status quo — a separate classifier per consortium, per platform, per panel — is what is currently bottlenecking projects like HuBMAP from delivering a unified tissue atlas at scale.

DCT's contribution is genuinely structural: by routing marker semantics through a frozen LLM and using channel-wise self-attention with no positional encoding, the same checkpoint runs across CODEX, MIBI, IMC, IBEX, MICS, CellDIVE, CyCIF, InSituPlex, and Vectra without retraining. For clinical-research consumers who want to compare cell densities across hospital cohorts that happen to use different panels, that is the right primitive — provided one keeps the headline (0.551 LODO F1, **excluding rare cell types**) honest, and treats the model as a strong prior to be corrected, not as a closed answer for any new panel.

## References

- Paper (bioRxiv preprint, v3 2025-08-22): [https://doi.org/10.1101/2024.11.02.621624](https://doi.org/10.1101/2024.11.02.621624)
- Code: [https://github.com/vanvalenlab/deepcell-types](https://github.com/vanvalenlab/deepcell-types)
- Data + weights: [https://vanvalenlab.github.io/deepcell-types](https://vanvalenlab.github.io/deepcell-types)
- Figure reproduction: [https://github.com/vanvalenlab/DeepCellTypes-2024_Wang_et_al](https://github.com/vanvalenlab/DeepCellTypes-2024_Wang_et_al)
- Related: Mesmer (Greenwald et al., 2022) — segmentation backbone used upstream
- Related baselines: MAPS, CellSighter, STELLAR, Nimbus, Celesta, Astir
- Gradient Reversal Layer: Ganin & Lempitsky (2015), *Unsupervised Domain Adaptation by Backpropagation*
- LLM stack: DeepSeek-R1 distilled Llama-3.3-70B-Instruct (explainer + embedder)
