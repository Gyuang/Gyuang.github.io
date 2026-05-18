---
title: "Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging"
excerpt: "Swapping contrastive 3D CT encoders for a wavelet + causal-conv + LFQ tokenizer (K = 262,144) trained with a 3-stage curriculum lifts CT-RATE report-generation F1 by 40% over CT-CHAT and cuts text-to-CT FID by 76.5%."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/btb3d/
tags:
  - BTB3D
  - 3D-Tokenizer
  - Wavelet
  - Causal-Convolution
  - LFQ
  - CT-Report-Generation
  - Text-to-CT
  - NeurIPS-2025
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- BTB3D replaces contrastive 3D CT encoders with a **causal 3D convolutional encoder-decoder + lookup-free quantizer (LFQ, K = 2^18 = 262,144)** trained via a 3-stage reconstruction curriculum, producing compact wavelet-domain tokens that scale to 512x512x241 volumes without retraining.
- The architectural recipe is three coordinated swaps: a **3D Haar wavelet front-end** for 8x free compression, **causal axial convolutions + overlapping temporal tiling** so a model trained on 9-slice subvolumes generalizes to 300+ slices, and **LFQ** that eliminates the embedding table entirely.
- Headline numbers (single run): on CT-RATE report generation the 16x16x8 variant lifts clinical F1 to **0.258 (+40% rel. over CT-CHAT's 0.184)**; on text-to-CT synthesis the 8x8x8 variant cuts mean **FID 9.51 -> 2.24 (-76.5%)** and halves FVD_CT-Net (7.66 -> 3.96).

## Motivation

3D radiology VLMs lag behind their 2D counterparts because the bottleneck is the *vision* side, not the language side. Contrastive pretraining was inherited from 2D CLIP, but in radiology multiple reports can describe the same scan very differently, so penalizing unmatched (image, text) pairs degrades semantics. Worse, contrastive losses need large batch sizes that are infeasible at full CT resolution, forcing shallow encoders or aggressively downsampled inputs that silently drop findings like small nodules. Existing text-to-CT generators (GenerateCT, MedSyn) hit the same compute wall and resort to cascaded 2D upsampling, producing inter-slice artifacts. The authors argue what is missing is a *unified, scalable, reconstruction-based tokenizer* for CT — better tokens, not bigger LLMs, are what 3D medical VLMs need.

## Core Innovation

- **3D Haar wavelet front-end.** Decompose the volume into 8 sub-bands (1 low + 7 high frequency), giving 8x compression as a free preprocessing step and exposing high-frequency anatomical detail as channels. Same transform handles 2D slices when D = 1, so the architecture is mode-agnostic.
- **Causal factorized 3D convolutions.** Each residual block is a 1xkxk spatial conv followed by a kx1x1 temporal conv with (k-1) zeros padded *only in the past*. Token t depends strictly on slices <= t — matching the prefix-decoding nature of downstream LLMs and enabling overlapping tile-by-tile inference on arbitrarily long scans.
- **Lookup-Free Quantization at K = 262,144.** Encoder output of dimension d = 18 is binarized element-wise by sign and packed to integers; the "codebook" is implicit, with zero embedding table to maintain. An entropy regularizer is added to push uniform code usage.
- **Three-stage curriculum.** Stage 1 trains on 9-slice subvolumes; Stage 2 introduces overlapping temporal tiling on 201-slice windows; Stage 3 freezes the encoder + codebook and fine-tunes the decoder on full 241-slice volumes. Stage 2 is the load-bearing transition (+14 dB PSNR).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Causal conv + LFQ tokens beat contrastive 3D encoders for report generation | Table 3: F1 0.258 vs 0.184 (+40% rel.); Table 5 wins on 13/18 abnormalities | CT-RATE | ⭐⭐⭐ |
| C2 | Improvements generalize OOD | Table 2: RAD-ChestCT F1 0.266 vs CT-CHAT 0.182 (+46%) | RAD-ChestCT (single external network) | ⭐⭐ |
| C3 | 76.5% FID reduction and halved FVD for text-to-CT | Table 4: FID 2.236 vs 9.512; FVD_CT-Net 3.955 vs 7.659 | CT-RATE | ⭐⭐⭐ on CT-RATE only |
| C4 | Three-stage curriculum is necessary; naive training fails | Table 1 stage-wise PSNR (9.35 -> 23.98 -> 28.17) | CT-RATE | ⭐⭐ — stage ablation only; no compute-matched "end-to-end on long volumes" baseline |
| C5 | Causal design enables unified 2D/3D training | Architectural argument + wavelet handles D = 1 | — | ⭐ — no quantitative 2D transfer experiment |
| C6 | Compression-quality trade-off (8^3 for synthesis, 16^2 x 8 for reports) | Tables 3 + 4 show split winners | CT-RATE | ⭐⭐ — only two operating points |
| C7 | LFQ + entropy reg avoids codebook collapse | Cites prior work; no codebook utilization / perplexity reported | — | ⭐ — asserted, not measured |
| C8 | 40% F1 improvement headline | 0.258 vs 0.184 = 40.2% on CT-RATE; 0.266 vs 0.182 = 46.2% on RAD-ChestCT | CT-RATE, RAD-ChestCT | ⭐⭐⭐ on the number, ⭐ on robustness (single run, no CIs) |

**Honest read.** The reconstruction and synthesis gains are large enough to survive the missing variance reporting — a >14 dB PSNR jump or a 4x FID drop is unlikely to be noise. The report-generation story is more nuanced: F1 is the only metric where Ours-16 dominates by a clear margin, and **the gain is recall-driven, not precision-driven**. Precision actually *drops* from CT-CHAT's 0.450 to BTB3D's 0.260, while recall jumps from 0.158 to 0.260. In other words, BTB3D writes longer reports that find more findings — including, plausibly, more hallucinated ones — and F1 here is dominated by recall. The paper does not separate hallucination from sensitivity; a precision-controlled comparison or a factuality metric like RadGraph-F1 would be needed. Three further weaknesses are worth flagging upfront:

1. **No ablation isolates individual components.** Every "why this works" claim — wavelet vs no wavelet, causal vs non-causal, LFQ vs VQ-VAE vs FSQ, entropy reg vs none — rests on the stage table alone. There is no component-level decomposition.
2. **Codebook utilization is asserted, not measured.** For a K = 262,144 codebook, the absence of any utilization or perplexity statistic is the single most surprising omission.
3. **Compute gates replication.** 64 H100s for the tokenizer, 40 for report-gen, 16 for synthesis. The authors themselves answer **No** to NeurIPS checklist Q7 (statistical significance), citing compute. The open-source promise is real but the replication promise is industry-only.

## Method & Architecture

![BTB3D architecture: wavelet front-end, causal 3D convolutional encoder/decoder, LFQ, and overlapping temporal tiling](/assets/images/paper/btb3d/fig_p003_01.png)
*Figure 1: BTB3D architecture. (a) Stage 1 — a 9-slice subvolume goes through 3D Haar wavelet decomposition, causal stride-2 spatial-then-temporal convolutions, and lookup-free quantization (K = 2^18) before a symmetric causal decoder reconstructs the volume in wavelet domain. (b) Stage 2 — overlapping temporal tiling: from each 9-slice window keep only the second token, which integrates all 9 slices, then decode the concatenated sequence in a single pass over 201 slices.*

### 1. Wavelet front-end

The input x in R^{DxHxW} is decomposed into W(x) in R^{D/2 x H/2 x W/2 x 8} (1 low-frequency + 7 high-frequency sub-bands). High-frequency anatomy lives as channels rather than pixels, and 8x compression comes for free before any learned operation. The same op handles 2D slices when D = 1.

### 2. Causal factorized encoder

Each residual block is a 1xkxk spatial conv followed by a kx1x1 temporal conv. Temporal padding is applied **only in the past direction** — token at axial index t depends only on slices <= t. Two variants:

- **8x8x8** — two stride-2 spatial convs plus the 2x wavelet gives 8x total compression. Used for synthesis where spatial fidelity matters.
- **16x16x8** — extra spatial stride-2 -> 16x spatial / 8x temporal. Coarser but memory-friendlier; used for report generation.

### 3. Lookup-Free Quantization

Encoder output y in R^{...xd} with d = 18 is binarized element-wise to b in {-1, +1}^18 and packed to integers in {0, ..., 2^18 - 1}. There is **no embedding table**: codes are addressed implicitly. An entropy regularizer L_entropy is added to push uniform code usage and discourage collapse. The total objective is:

$$
\mathcal{L} = \mathcal{L}_\text{rec} + \lambda_\text{adv}\,\mathcal{L}_\text{adv} + \mathcal{L}_\text{vq}
$$

with L_rec an L1 reconstruction in wavelet domain, L_adv a 3D adversarial loss in CT domain (not wavelet), and L_vq the standard VQ commitment. **VGG/LPIPS perceptual losses are deliberately omitted** as a mismatch for grayscale CT.

### 4. Three-stage curriculum

- **Stage 1** — end-to-end on 9-slice subvolumes (batch 8 or 40 for 2D), 150k iterations. Learns local spatio-temporal structure.
- **Stage 2** — overlapping temporal tiling on 201-slice windows, batch 1, 60k iterations. From each window keep only the second token (which has integrated all 9 slices in its causal receptive field), then concatenate to form [z^1_1, z^1_2, z^2_2, ..., z^T_2]. This stage delivers the **biggest reconstruction jump (+14 dB PSNR)**.
- **Stage 3** — freeze E and codebook, fine-tune G only on full 241-slice volumes, 50k iterations. Sharpens fine boundaries (fissures, vessels).

### 5. Downstream heads

- **Report generation.** Tokens linearly projected into LLaMA-3.1-8B input space with LoRA (r = 64 / alpha = 128 for 8^3; r = 128 / alpha = 256 for 16^2 x 8). For 8^3 the token count is 4x larger so embeddings are merged 72 -> 1. 40k iters, AdamW lr 2e-5, DeepSpeed ZeRO-3 on 40 H100s.
- **Text-to-CT synthesis.** A 12-layer transformer (1024 d, 16 heads) with [7,7,7] windowed self-attention and [2,2,2] patching, trained with flow-matching loss (SiT-style) and a T5v1.1-base text encoder. 1500 epochs on 16 H100s.

## Experimental Results

### Reconstruction across curriculum stages (Table 1, CT-RATE)

| Stage | Compression | PSNR ↑ | SSIM ↑ | MSE ↓ |
|---|---|---|---|---|
| Stage 1 | 8^3 | 9.350 | 0.206 | 0.117 |
| Stage 2 | 8^3 | 23.980 | 0.697 | 0.005 |
| **Stage 3** | **8^3** | **28.166** | **0.760** | **0.001** |
| Stage 1 | 16^2 x 8 | 11.067 | 0.353 | 0.079 |
| Stage 2 | 16^2 x 8 | 23.808 | 0.700 | 0.005 |
| **Stage 3** | **16^2 x 8** | **26.750** | **0.749** | **0.002** |

Stage 2 -> Stage 1 is the +14 dB jump; Stage 3 adds another ~3 dB. The curriculum table is doing all the architectural justification work in the paper.

![Per-stage reconstruction for both BTB3D variants across three planes](/assets/images/paper/btb3d/page_007.png)
*Figure 2: Reconstruction quality of the 8x8x8 model improves dramatically from Stage 1 (blurry, loses structure past 9 slices) through Stage 2 (coherent long-range anatomy) to Stage 3 (sharp fissures and vessels).*

### CT-RATE report generation (Table 3)

| Model | F1 | Precision | Recall | CRG | BLEU-1 | BLEU-mean | METEOR |
|---|---|---|---|---|---|---|---|
| CT2Rep | 0.160 | 0.435 | 0.128 | 0.359 | 0.372 | 0.280 | 0.197 |
| Merlin | 0.160 | 0.295 | 0.112 | 0.352 | 0.231 | 0.154 | 0.148 |
| CT-CHAT | 0.184 | **0.450** | 0.158 | 0.368 | 0.373 | 0.272 | 0.215 |
| Ours-8 | 0.187 | 0.260 | 0.150 | 0.357 | 0.411 | 0.295 | 0.220 |
| **Ours-16** | **0.258** | 0.260 | **0.260** | **0.370** | **0.439** | **0.305** | **0.223** |

Look carefully at the Precision column: CT-CHAT keeps the precision crown at 0.450 while BTB3D-16 sits at 0.260. The F1 gain is entirely recall-driven (0.260 vs 0.158). The 40% F1 headline is real but cannot be read as a clean improvement in clinical correctness.

### RAD-ChestCT external generalization (Table 2)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| CT2Rep | 0.133 | 0.299 | 0.139 |
| Merlin | 0.182 | 0.271 | 0.149 |
| CT-CHAT | 0.182 | **0.382** | 0.171 |
| Ours-8 | 0.192 | 0.269 | 0.165 |
| **Ours-16** | **0.266** | 0.272 | **0.329** |

Same recall-driven pattern transfers OOD. Encouraging that the F1 gap holds; the precision floor stays the same.

### Text-to-CT synthesis (Table 4, CT-RATE)

| Model | FID (mean) ↓ | FVD_CT-Net ↓ | FVD_I3D ↓ | CLIP text-img ↑ |
|---|---|---|---|---|
| GenerateCT | 9.512 | 7.659 | 1512.5 | 23.625 |
| MedSyn | 12.592 | 13.927 | 725.81 | 23.571 |
| **Ours-8** | **2.236** | **3.955** | **325.51** | **24.270** |
| Ours-16 | 5.011 | 4.020 | 429.34 | 23.322 |

FID drop of 76.5% and FVD halved. Synthesis is the cleanest part of the result set — the numbers move enough that the missing CIs are not the binding constraint.

![Text-to-CT generation comparison: MedSyn, GenerateCT, BTB3D-16, BTB3D-8 across three planes](/assets/images/paper/btb3d/page_009.png)
*Figure 3: Text-to-CT generation for the prompt "63 y/o male: no pneumonia, mild sequelae, atherosclerotic changes, hepatosteatosis, hiatal hernia." MedSyn and GenerateCT show inter-slice artifacts and parenchymal blur; BTB3D-8 produces sharp anatomy in axial, coronal, and sagittal views.*

### Per-abnormality breakdown (Table 5)

BTB3D-16 wins on 13 of 18 abnormalities (mean F1 0.258 vs CT-CHAT 0.187). Notable jumps: cardiomegaly 0.305 vs 0.207, pleural effusion 0.308 vs 0.199, mosaic attenuation 0.183 vs 0.094, peribronchial thickening 0.125 vs 0.043. CT-CHAT and Merlin still win on calcifications and emphysema — patterns where dense / high-contrast signals favor a contrastive encoder.

![Per-abnormality F1 breakdown on CT-RATE](/assets/images/paper/btb3d/fig_p028_01.png)
*Appendix Figure: Per-abnormality F1 on CT-RATE. BTB3D-16 wins 13/18; CT-CHAT and Merlin retain the lead on calcifications and emphysema.*

## Limitations

**Authors admit:**

- Chest CT only. No validation on MRI, PET, abdomen, or whole-body.
- No statistical significance / repeated runs (NeurIPS checklist Q7 = No), justified by compute cost.
- No clinical reader study, no prospective evaluation.
- 2D/3D unification is designed-in but not empirically demonstrated; "2D mode" is only used on 2D slices extracted from CT-RATE.
- No uncertainty or causal-reasoning module.

**Additionally unaddressed (this review's flags):**

- **No component-isolating ablation.** Wavelet vs no-wavelet, causal vs non-causal, LFQ vs VQ-VAE vs FSQ, entropy reg on vs off — none of these are isolated. Every "why this works" claim rests on the stage table.
- **No codebook utilization or perplexity reported.** For a K = 262,144 codebook, the absence of any utilization statistic is the most surprising omission given that collapse avoidance is an explicit claim.
- **Recall-driven F1 gain.** Precision drops from 0.450 to 0.260 on CT-RATE; the paper does not separate sensitivity from hallucination. A factuality metric like RadGraph-F1 or an explicit precision-controlled comparison would settle this.
- **Token-count trade-off shown at only two operating points** (8^3 and 16^2 x 8). The curve between them is unknown.
- **No fVLM comparison** (the most relevant 3D contrastive baseline), justified by "weights unavailable" but no re-implementation attempted.
- **No inference latency / throughput numbers** for downstream generation, despite "memory efficient" being a stated benefit.
- **Compute gates replication.** 64 H100s for the tokenizer alone — replication is industry-only.

## Why It Matters for Medical AI

If the wavelet + causal-conv + LFQ recipe holds up under independent ablation, it shifts the locus of progress in 3D medical VLMs from the LLM side back to the *tokenizer* side. Two practical implications worth tracking:

1. **Causal tokenization aligns with LLM prefix decoding**, so the same tokenizer can plug into different language backbones without re-aligning representation spaces — a useful property as 3D-aware LLMs proliferate.
2. **A reconstruction-trained tokenizer decouples vision pretraining from paired text**, which matters for modalities (MRI, PET, whole-body CT) where large paired report corpora do not yet exist.

The cautions: the recall-driven F1 framing means clinical adoption should not be inferred from the headline; the missing codebook utilization data leaves open whether K = 262,144 is actually being used; and the compute footprint means this is a methodology paper for industry labs first and an academic recipe second.

## References

- **Paper**: Hamamci, I. E., Er, S., Shit, S., Reynaud, H., Yang, D., Guo, P., Edgar, M., Xu, D., Kainz, B., Menze, B. *Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging.* NeurIPS 2025. arXiv:2510.20639v1, 23 Oct 2025.
- **Code**: https://github.com/ibrahimethemhamamci/BTB3D
- **Dataset (CT-RATE)**: Hamamci et al. 2024. 25,692 chest CT scans / 21,304 patients with paired findings + impression. CC BY-NC-SA 4.0.
- **Dataset (RAD-ChestCT)**: Draelos et al. 2021. External multi-label benchmark. CC BY-NC-ND 4.0.
- **Related: tokenizers** — MagViT-2 (Yu et al. 2024) for LFQ; Cosmos-Tokenizer (NVIDIA 2024) for wavelet + causal-conv design lineage.
- **Related: CT report generation** — CT2Rep (Hamamci et al. 2024), Merlin (Blankemeier et al. 2024), CT-CHAT (Hamamci et al. 2024).
- **Related: text-to-CT synthesis** — GenerateCT (Hamamci et al. 2023), MedSyn (Xu et al. 2024).
