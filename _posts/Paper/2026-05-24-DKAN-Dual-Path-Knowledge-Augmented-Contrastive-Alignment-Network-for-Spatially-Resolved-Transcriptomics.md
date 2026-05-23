---
title: "DKAN: Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics"
excerpt: "An LLM-summarized NCBI gene-function prior is injected as a third modality and used as the cross-attention query over image and expression features, sweeping all 6 metrics x 3 ST datasets vs 10 baselines."
categories: [Paper, Spatial-Transcriptomics]
tags:
  - DKAN
  - Spatial-Transcriptomics
  - Cross-Attention
  - Contrastive-Learning
  - Knowledge-Prior
  - LLM
  - BioBERT
  - UNI
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-24
last_modified_at: 2026-05-24
permalink: /paper/dkan/
---

## TL;DR

- DKAN predicts spatially resolved gene expression from H&E whole-slide images by injecting an **LLM-summarized NCBI gene-function prior** (GPT-4o -> BioBERT -> Transformer) as a *third* modality, used as the **query** in two parallel cross-attentions over the image and expression features so the heterogeneous modalities are never aligned directly.
- The training loss combines a CLIP-style contrastive objective on the two knowledge-conditioned representations with a multi-branch supervised + intermediate-distillation MSE, all balanced by an adaptive reciprocal-magnitude weighting; inference uses only the image and text paths (no exemplar bank, no retrieval).
- **Headline result: best on all 6 metrics x 3 ST datasets (HER2+, STNet, cSCC) vs 10 baselines under patient-disjoint cross-validation**, e.g. on cSCC PCC-ALL **0.407 vs TRIPLEX 0.363** (+12.1% rel.) and on HER2+ PCC-ALL **0.330 vs TRIPLEX 0.304**.

## Motivation

Histology slides (H&E) are cheap and routine; spatial transcriptomics (ST) is expensive and low-throughput. A growing line of work (ST-Net, HisToGene, Hist2ST, EGN/EGGN, BLEEP, mclSTExp, TRIPLEX, M2OST, ST-Align) regresses per-spot gene expression from WSI patches to lower the cost of spatially resolved cancer biomarker maps. The authors identify three recurring gaps:

1. Image features capture low-level morphology (color, texture) but miss the **biological semantics** of each target gene (function, pathway, phenotype).
2. Retrieval / exemplar-based contrastive methods (BLEEP, EGN, mclSTExp, ST-Align) build a reference bank and pull neighbors at inference, adding storage and complexity.
3. Multi-scale fusion tends to align heterogeneous image and expression features **directly**, which the authors argue washes out cross-modal structure.

DKAN's pitch is to introduce a *third* modality that mediates: an LLM-summarized gene-function prior that lives in text space and is used as the shared query across image and expression cross-attentions.

## Core Innovation

The single conceptual move is using gene-functional text as a **dynamic cross-modal coordinator** rather than another vector to be averaged in. Concretely:

- For each target gene, the NCBI Gene summary is fed into GPT-4o with a structured prompt (role / task / output spec) to produce a keyword-only `{symbol, functionality, phenotype}` JSON under 120 words.
- That text is embedded by **BioBERT** and refined by a Transformer to give `f_text`.
- Two parallel cross-attentions then use `f_text` as **Query** over the image features (`f_img`) and the expression features (`f_exp`) as Key/Value:
  - Image path: a "functional query instruction" that filters morphology regions relevant to each gene.
  - Expression path: a "distribution correction factor" that nudges the expression representation toward biological-pathway logic.
- The image and expression modalities are **never aligned to each other directly** -- alignment happens only through the shared knowledge-conditioned queries.

This is paired with a CLIP-style contrastive loss on the two knowledge-conditioned embeddings, plus multi-branch supervised + intermediate-distillation MSE, all balanced by adaptive **reciprocal-magnitude** weighting (the smaller-magnitude objective gets the higher weight). Inference uses only the image + text paths.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | DKAN beats SOTA on spatial gene-expression prediction | Table 1: best on **all 6 metrics x 3 datasets** vs 10 baselines, patient-disjoint CV with std. devs. reported | HER2+, STNet, cSCC | star star star |
| C2 | LLM-summarized NCBI gene-functional prior improves prediction | Table 4 "w/o text" drops STNet PCC-ALL 0.219->0.210, HER2+ 0.330->0.311, cSCC 0.407->0.326 | All three | star star (real and reproduced, but on STNet sits inside ~1 sigma; *largest* contributor is actually multi-scale) |
| C3 | One-stage paradigm removes the need for exemplar retrieval | Architecturally true (no reference bank, no NN search at inference); beats BLEEP, EGN, mclSTExp on every metric; inference 0.158 s/spot vs BLEEP 0.461, mclSTExp 0.587 | All three | star star star |
| C4 | "Over-reliance on exemplar retrievals" is a real limitation of prior work | **Asserted but never operationalized** -- no retrieval-failure experiment, no bank-size sensitivity, no perturbation of BLEEP/EGN exemplars | -- | star (assertion, not measurement) |
| C5 | Dual-path alignment beats direct image<->expression alignment | "Text-as-KV" ablation: STNet 0.219 vs 0.216, HER2+ 0.330 vs 0.330 (tie), cSCC 0.407 vs 0.368. Q-vs-KV is tested; the bigger "no dual-path at all" baseline is missing | STNet, HER2+, cSCC | star star (partial) |
| C6 | BioBERT is the best text encoder | Tables 2, S1-S3: BioBERT best on all three datasets, but margin <= 0.008 PCC vs Conch / PLIP / BioGPT -- well inside std. dev. | All three | star (numerically best, statistically indistinguishable) |
| C7 | UNI is the best image encoder | Same tables: UNI best by 0.01-0.03 PCC vs Conch; larger gap vs ResNet/PLIP | All three | star star |
| C8 | Adaptive loss weighting beats fixed weights | Table 5 / S1 / S2: PCC-ALL gains of 7-8 points (HER2+ 0.330 vs 0.258, STNet 0.219 vs 0.148) -- large and consistent | All three | star star star |
| C9 | Multi-scale (WSI + region + patch) is necessary | Ablations: single largest drop -- STNet 0.219->0.117 (**~50% collapse**), HER2+ 0.330->0.210, cSCC 0.407->0.300 | All three | star star star |
| C10 | DKAN is "computationally efficient" | Table S4 (cSCC only): training 3.7 h vs ST-Net 4.1, TRIPLEX 4.8, **but** 15.8 GFLOPs vs ST-Net 2.9 and 87 M params vs ST-Net 7.2 | cSCC only | star star (true for wall-clock, misleading for FLOPs/params; GPT-4o API cost not reflected) |

## Method & Architecture

![DKAN framework](/assets/images/paper/dkan/page_003.png)
*Figure 2 (page 3): DKAN framework -- (a) gene-semantic prior from NCBI + GPT-4o + BioBERT, (b) gene expression encoder, (c) multi-level image encoder with UNI (WSI / region) and ResNet18 (patch), (d) dual-path contrastive alignment using text as cross-attention query.*

The pipeline has four modules feeding one composite loss.

**1. Gene Semantic Representation (knowledge prior).** For each of the `N_g = 250` spatially variable target genes, the NCBI Gene summary is retrieved and fed to GPT-4o with a structured prompt (role / task / output spec). For breast-cancer datasets the role is "expert in genomics specializing in human breast-cancer genes"; output is a JSON `{gene symbol, functionality, phenotype}` under 120 words, keyword-only, no adjectives. The text is embedded by **BioBERT** (`d_t = 1024`) and refined by a Transformer (plus linear projection) into `f_text in R^{N_g x d}`.

**2. Gene Expression Embedding.** A 2-layer MLP with GELU, dropout, residual + LayerNorm projects the per-spot expression vector `Y in R^{N_p x N_g}` to `f_exp in R^{N_p x d}`.

**3. Multi-level Image Embedding.** Three hierarchical encoders:

- WSI-level `f_wsi`: **UNI** foundation model (frozen, `d_h = 1024`) + trainable Transformer head.
- Region-level `f_region`: UNI over the `k = 25` nearest-neighbor patches (5x5 grid around the spot), frozen + Transformer adapter.
- Patch-level `f_patch`: **ResNet18** pretrained per Ciga et al. 2022, fully trainable, final pooling/FC removed (`d_p = 512`).

Two cross-attentions then fuse {WSI <-> region} and {WSI <-> patch} with WSI as query; results are summed into `f_img`.

**4. Dual-Path Contrastive Alignment (the core).** Two parallel cross-attention blocks both use `f_text` as **Query** and the modality features as Key/Value:

$$e_{ti} = \text{CrossAttn}(Q=f_{text}, K=V=f_{img}), \quad e_{te} = \text{CrossAttn}(Q=f_{text}, K=V=f_{exp})$$

The image and expression modalities never see each other directly; alignment is mediated by the knowledge query.

**5. Losses.**

- **Contrastive (CLIP-style):** positive = same gene, negative = different genes within the batch, temperature `tau = 0.1` (HER2+, STNet) or `0.08` (cSCC):

$$L_{cont} = -\sum_i \log \frac{\exp(\text{sim}(e_{ti}^i, e_{te}^i)/\tau)}{\sum_j \exp(\text{sim}(e_{ti}^i, e_{te}^j)/\tau)}$$

- **Supervised with intermediate distillation:** each branch's intermediate prediction `Y_hat_d` for `d in {img, patch, wsi, region}` is regressed against both ground truth `Y` and the final prediction `Y_hat`, weighted by `lambda`: `L_d = lambda * ||Y_hat_d - Y_hat||^2 + (1 - lambda) * ||Y_hat_d - Y||^2`, plus a final-output MSE.
- **Adaptive composite:** `L = w_sup * L_sup + w_cont * L_cont`, with `w_sup`, `w_cont` set as normalized reciprocals of the current loss magnitudes so the smaller-magnitude objective gets the higher weight.

**6. Inference.** Only the image and text paths are used -- the expression encoder is contrastive-loss-only at training time, and there is no reference bank or NN lookup.

**7. Hyperparameters.** Patches 224x224, `k = 25`, `N_g = 250` spatially variable genes, Adam `lr = 1e-4`, StepLR (step=50, gamma=0.9), batch size 128, NVIDIA A800. CV: 8-fold patient-disjoint for STNet, leave-one-patient-out for HER2+ (8 folds) and cSCC (4 folds).

## Experimental Results

![Main comparison table](/assets/images/paper/dkan/page_005.png)
*Table 1 (page 5): DKAN vs 10 baselines across HER2+, STNet, cSCC -- best on all 6 metrics x 3 datasets with patient-disjoint cross-validation.*

### Main comparison (Table 1). Bold = best.

| Dataset | Model | MAE down | MSE down | PCC-ALL up | PCC-HPG up | PCC-HEG up | PCC-HVG up |
|---|---|---|---|---|---|---|---|
| **HER2+** | ST-Net | 0.432 +/- 0.05 | 0.311 +/- 0.07 | 0.150 +/- 0.13 | 0.287 +/- 0.19 | 0.115 +/- 0.11 | 0.090 +/- 0.08 |
|  | BLEEP | 0.401 +/- 0.03 | 0.277 +/- 0.05 | 0.151 +/- 0.11 | 0.277 +/- 0.16 | 0.246 +/- 0.09 | 0.261 +/- 0.07 |
|  | EGN | 0.366 +/- 0.04 | 0.229 +/- 0.05 | 0.204 +/- 0.12 | 0.364 +/- 0.16 | 0.152 +/- 0.09 | 0.120 +/- 0.05 |
|  | mclSTExp | 0.398 +/- 0.04 | 0.272 +/- 0.05 | 0.163 +/- 0.11 | 0.289 +/- 0.16 | 0.114 +/- 0.08 | 0.091 +/- 0.06 |
|  | HisToGene | 0.388 +/- 0.06 | 0.253 +/- 0.07 | 0.150 +/- 0.09 | 0.295 +/- 0.15 | 0.099 +/- 0.07 | 0.079 +/- 0.05 |
|  | Hist2ST | 0.417 +/- 0.07 | 0.293 +/- 0.08 | 0.193 +/- 0.10 | 0.360 +/- 0.17 | 0.126 +/- 0.07 | 0.109 +/- 0.03 |
|  | TRIPLEX | 0.364 +/- 0.05 | 0.234 +/- 0.06 | 0.304 +/- 0.14 | 0.491 +/- 0.18 | 0.271 +/- 0.10 | 0.260 +/- 0.06 |
|  | M2OST | 0.446 +/- 0.10 | 0.340 +/- 0.15 | 0.147 +/- 0.12 | 0.313 +/- 0.19 | 0.098 +/- 0.09 | 0.090 +/- 0.06 |
|  | **DKAN** | **0.361 +/- 0.04** | **0.224 +/- 0.06** | **0.330 +/- 0.13** | **0.531 +/- 0.15** | **0.317 +/- 0.09** | **0.304 +/- 0.07** |
| **STNet** | TRIPLEX | 0.342 +/- 0.02 | 0.200 +/- 0.02 | 0.194 +/- 0.07 | 0.344 +/- 0.10 | 0.160 +/- 0.06 | 0.224 +/- 0.07 |
|  | HisToGene | 0.326 +/- 0.02 | 0.180 +/- 0.02 | 0.103 +/- 0.04 | 0.217 +/- 0.11 | 0.060 +/- 0.02 | 0.074 +/- 0.03 |
|  | **DKAN** | **0.322 +/- 0.02** | **0.179 +/- 0.02** | **0.219 +/- 0.07** | **0.387 +/- 0.09** | **0.200 +/- 0.06** | **0.244 +/- 0.07** |
| **cSCC** | TRIPLEX | 0.415 +/- 0.06 | 0.278 +/- 0.08 | 0.363 +/- 0.07 | 0.476 +/- 0.07 | 0.276 +/- 0.07 | 0.272 +/- 0.06 |
|  | EGN | 0.438 +/- 0.05 | 0.303 +/- 0.06 | 0.278 +/- 0.06 | 0.388 +/- 0.06 | 0.194 +/- 0.06 | 0.180 +/- 0.06 |
|  | **DKAN** | **0.383 +/- 0.05** | **0.239 +/- 0.06** | **0.407 +/- 0.08** | **0.508 +/- 0.08** | **0.346 +/- 0.09** | **0.321 +/- 0.08** |

Improvements over the second-best baseline (TRIPLEX) range from +1.3% relative (STNet MSE 0.179 vs 0.200) to **+12.1% relative on cSCC PCC-ALL (0.363 -> 0.407)** and **+25.4% relative on cSCC PCC-HEG (0.276 -> 0.346)**. SGN, THItoGene, and M2OST collapse across all three datasets.

### Ablations and qualitative visualizations

![Encoder ablations and qualitative biomarker maps](/assets/images/paper/dkan/page_006.png)
*Table 2 (encoder ablation) and Figure 3 (qualitative cancer-biomarker visualization with per-gene PCC) -- DKAN's spatial maps match the ground truth more closely than every baseline on FN1, ERBB2, GNAS, HSP90AB1, SPARC.*

- **Encoders (Tables 2, S1-S3).** BioBERT marginally best as text encoder (STNet PCC-ALL 0.219 vs 0.211 Conch / 0.217 PLIP / 0.217 BioGPT -- all inside 1 sigma). UNI wins as image encoder more clearly (STNet PCC-ALL 0.219 vs 0.202 ResNet18, 0.190 PLIP).
- **Prompt (Table 3).** Full prompt (summary + functionality + phenotype) beats "w/o text constraint" 0.219 vs 0.206 and "w/o summary" 0.219 vs 0.199 on STNet PCC-ALL -- both within ~1 sigma.
- **LLM choice (Table 3).** GPT-4o > DeepSeek-V3 ~ LLaMA2 ~ DeepSeek-R1 (STNet PCC-ALL 0.219 vs 0.202 / 0.193 / 0.198). Notably the reasoning model (R1) underperforms a non-reasoning one for short factual gene summaries.
- **Module ablation (Table 4).** w/o multi-scale collapses STNet PCC-ALL 0.219 -> 0.117 (largest drop, **~50%**); w/o text 0.219 -> 0.210; w/o contrastive 0.219 -> 0.209; text-as-KV 0.219 -> 0.216. Curiously, **w/o contrastive achieves a *better* MAE (0.320 vs 0.322)** -- the contrastive term trades pointwise error for correlation. Pattern repeats on HER2+ (multi-scale drop 0.330 -> 0.210).
- **Fusion / loss design (Tables 5, S1, S2).** Cross-attention is best overall but on STNet PCC-ALL **Sum.+Trans. is actually marginally *better* (0.221 vs 0.219)** -- the fusion ablation is essentially a tie there. Adaptive weighting beats fixed weights by a wide margin (HER2+ PCC-ALL 0.330 vs 0.258, STNet 0.219 vs 0.148); intermediate distillation gives a smaller but consistent boost.
- **Compute (Table S4, cSCC only).** 15.84 GFLOPs, 87.2 M params, 3.7 h training, 0.158 s/spot inference -- heavier than ST-Net or TRIPLEX in FLOPs / params but faster training than EGN, HisToGene, M2OST, mclSTExp. **The GPT-4o API call for gene-summary generation is not in this table.**

### Cancer-biomarker qualitative results

![HER2+ biomarker visualizations](/assets/images/paper/dkan/page_012.png)
*Figure S2 (page 12): HER2+ cancer-marker visualization -- DKAN per-spot PCC of 0.898 (ERBB2), 0.673 (GNAS), 0.661 (HSP90AB1) vs best baseline TRIPLEX 0.704 / 0.445 / 0.606. Cherry-picked but consistent.*

## Limitations

The paper's own conclusion does not enumerate limitations. From the audit:

- **"Exemplar over-reliance" framing is rhetorical.** The paper motivates against retrieval-based baselines but never measures retrieval failure modes -- no bank-size sensitivity, no exemplar perturbation, no test where the retrieval explicitly fails. C4 is an assertion, not a measurement.
- **No external dataset, no significance tests.** All three benchmarks are public legacy ST datasets with small cohorts (4-23 patients). No HEST-1k, no STImage-1K4M, no held-out scanner / site. Improvements over TRIPLEX sometimes overlap in std. dev. (STNet MAE 0.322 vs 0.342) with no paired-fold tests.
- **GPT-4o dependency is hidden.** Best results require GPT-4o; switching to open LLMs (DeepSeek-V3 / LLaMA2 / R1) costs ~0.02 PCC on STNet. This API cost is missing from the compute table, and there is no ablation removing the *cancer-context* phrase from the prompt -- a potential knowledge-contamination route.
- **Cross-attention vs Sum.+Trans. fusion is a tie on STNet** (0.219 vs 0.221 PCC-ALL). The fancier fusion is not clearly justified on every dataset.
- **The biggest contributor in ablations is multi-scale**, by a large margin: removing it ~halves PCC-ALL on STNet (0.219 -> 0.117) and drops HER2+ from 0.330 -> 0.210. Adaptive weighting is the second-largest. The gene-semantic prior gain is real and reproduced across all three datasets, but on STNet sits within ~1 sigma -- the headline novelty is not the dominant performance driver.
- **Restricted to 250 spatially variable genes** in carcinoma tissue (breast + skin); whole-transcriptome and normal-tissue generalization untested.
- **Patch encoder (ResNet18) trainable, UNI frozen.** No ablation on unfreezing UNI or trying CONCH / Virchow as the trainable patch backbone.
- **No failure-case analysis** (which genes / spots DKAN gets wrong, and why).

## Honest Read

DKAN's headline (C1, star star star) is genuinely well-supported: a clean sweep of 6 metrics across 3 datasets vs 10 baselines under patient-disjoint CV, std. devs. reported. The conceptual novelty -- text-as-query, gene-semantic prior as a third modality -- is well-motivated and the dual-path design is elegant.

But the ablations tell a more sober story than the abstract:

1. The motivational complaint about "over-reliance on exemplar retrieval" is **never operationalized** (C4, star). DKAN winning over BLEEP / EGN / mclSTExp shows it is *competitive*; it does not show the retrieval mechanism is what fails.
2. The largest single contributor by a wide margin is **multi-scale fusion** (C9, star star star) -- removing it collapses STNet PCC-ALL by ~50%. The second is **adaptive loss weighting** (C8, star star star). The gene-semantic prior (C2, star star) helps consistently but on STNet sits inside 1 sigma.
3. **Cross-attention vs Sum.+Trans. fusion is essentially a tie on STNet** (0.219 vs 0.221) -- the fancier fusion is not clearly justified on every dataset.
4. The encoder choices (BioBERT, UNI) are presented as ablation wins, but the text-encoder margins are inside std. dev. (C6, star); only UNI vs ResNet-class encoders shows a real gap (C7, star star).
5. The **GPT-4o dependency is hidden** in the compute table; open-LLM substitutes cost ~0.02 PCC on STNet and the API cost is unreported.
6. No external dataset, no paired-fold significance tests, no normal-tissue or failure-case analysis.

The "knowledge prior as a query that mediates two modalities" is a useful pattern and worth borrowing. The headline number is real. The argument that this specific architectural piece is the dominant contributor is weaker than the paper suggests -- multi-scale and adaptive weighting are doing more of the work, and the exemplar-retrieval critique that motivates the whole design is never tested directly.

## Why It Matters for Medical AI

Predicting spatial transcriptomics from cheap, routine H&E lowers the cost of spatially resolved cancer biomarker mapping (HER2+ breast, cSCC) by an order of magnitude. The mechanistic idea here -- treat curated, LLM-summarized domain knowledge as a **third modality used as a query**, rather than another vector to concatenate -- is portable to other medical-AI multimodal tasks (radiology-report-grounded segmentation, pathology grading conditioned on guidelines). The caveat for clinical translation is that DKAN inherits the GPT-4o dependence and has not been validated outside legacy ST cohorts.

## References

- Paper (arXiv): [arXiv:2511.17685](https://arxiv.org/abs/2511.17685) -- "Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics", Zhang, Chu, Liu, Tong, Li (City University of Hong Kong), AAAI 2026.
- Code: [github.com/coffeeNtv/DKAN](https://github.com/coffeeNtv/DKAN)
- Datasets: HER2+ (Andersson et al. 2021), STNet (He et al. 2020), cSCC (Ji et al. 2020).
- Related baselines: ST-Net (He et al. 2020), HisToGene (Pang et al. 2021), Hist2ST (Zeng et al. 2022), EGN/EGGN (Yang et al. 2023), BLEEP (Xie et al. 2023), mclSTExp (Min et al. 2024), TRIPLEX (Chung et al. 2024), M2OST (Wang et al. 2024).
- Foundation models used: UNI (Chen et al. 2024) for WSI / region encoding, BioBERT (Lee et al. 2020) for gene-summary embedding, GPT-4o (OpenAI 2024) for gene-summary generation, ResNet18 pretrained per Ciga et al. 2022 for patch encoding.
