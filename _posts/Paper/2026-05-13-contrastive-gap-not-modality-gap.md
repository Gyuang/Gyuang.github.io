---
title: "It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap"
excerpt: "A unimodal, no-cone, identical-pair control still produces 100% linearly-separable CLIP embeddings — so the 'modality gap' is really a contrastive gap, and adding uniformity + alignment terms cuts centroid distance 0.66 -> 0.13 at d=128."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/contrastive-gap-not-modality-gap/
tags:
  - CLIP
  - Modality-Gap
  - Contrastive-Learning
  - Uniformity-Alignment
  - Representation-Geometry
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-14
last_modified_at: 2026-05-14
---

## TL;DR

- Replace CLIP's text encoder with a *second image encoder*, align the two random-init cones, feed *identical-image positive pairs*, and the gap **still appears**: linear separability climbs from 0.50 -> **1.00** (Table 1, 512D, 2048 imgs, 1200 epochs). The "modality gap" is really a property of contrastive loss in high dimensions.
- The fix is the rigorous part: adding Wang & Isola-style in-modal uniformity + alignment, plus a new cross-modal uniformity term (`L_CUAXU = L_CLIP + L_Uniform + L_XUniform + L_Align`) cuts centroid distance from **0.66 -> 0.13** and linear separability from **1.00 -> 0.83** at d=128 — measured across d ∈ {32, 64, 128} with 3 seeds and ±1 SE (Tables 7-9).
- **The "closes gap -> better downstream" framing is overgeneralized.** I↔T retrieval is statistically unchanged (Table 4, all within 1 SE); zero-shot gains are 1-7 pp on Caltech101 / CIFAR-100; the headline +18% on SIMAT (36.02 -> 42.47) has no error bars and uses a single λ.

## Motivation

Liang et al. (2022) showed CLIP-style two-tower contrastive models leave image and text embeddings in disjoint regions of the unit hypersphere — the "modality gap" — and attributed it to (1) the cone effect at random init, (2) mismatched image-caption pairs, and (3) insufficient training. Subsequent work added local-minima and hard-negative explanations. **None of these prior accounts ever tested whether modality is actually necessary** for the gap to form. If the gap survives in a setting where modality is removed, then the prevailing causal story is wrong, and downstream attempts to close the gap by translation or projection treat a symptom rather than the cause. The authors' control experiment is exactly the missing test.

The medical-AI relevance is indirect but real — clinical CLIP variants (CONCH, BiomedCLIP, MedCLIP) inherit this geometry, and any claim that aligning radiology images with text "fixes" the gap rests on an assumption this paper challenges.

## Core Innovation

- **Unimodal control experiment.** Two image encoders + paired-identical images + cone-aligned init = a setup that *simulates the absence of a modality*. After 1200 epochs the encoder embeddings are still 100% linearly separable (Table 1). This is the paper's strongest intellectual move.
- **Multimodal uniformity + alignment loss.** A multi-modal adaptation of Wang & Isola (2020): in-modal `L_Uniform` (image and text), cross-modal `L_XUniform`, and `L_Align`, combined as `L_CUA` and `L_CUAXU`.
- **Dimensionality framing.** The gap *grows with d* for vanilla CLIP (centroid distance 0.10 -> 0.31 -> 0.66 from d=32 -> 64 -> 128), supporting the "contrastive geometry in high dimensions" story over a modality-specific story.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | The modality gap is not modality-specific; it persists in a unimodal, no-cone, identical-pair control. | Table 1: linear-sep 0.50 -> 1.00, centroid 0.00 -> 0.06. | MS COCO 2048-img subset, 512D, 1 run. | ⭐⭐ — conceptually decisive but a *single seed* on a tiny subset; no variance, no sweep over d, batch size, or τ. |
| C2 | The gap is caused by low uniformity / low-d manifold collapse in high-d space. | Figs. 2/4 PCA explained variance + Table 5 image uniformity values. | MS COCO val. | ⭐⭐ — correlational. Uniformity is *directly optimized*, so its drop is mechanical, not causal isolation. |
| C3 | Adding `L_Uniform + L_Align (+ L_XUniform)` reduces the gap. | Tables 2/7/8/9 over d ∈ {32, 64, 128}, 3 seeds, ±1 SE. | MS COCO val. | ⭐⭐⭐ — multi-d, multi-seed, with confidence intervals. Cleanest part of the paper. |
| C4 | Closing the gap maintains image-text retrieval. | Table 4/10, ±1 SE. | MS COCO val. | ⭐⭐⭐ — explicit error bars; differences within noise. Note: this *undermines* the "better representation" story. |
| C5 | Closing the gap improves zero-shot classification "consistently". | Fig. 3 + Table 11, 3 seeds, ±1 SE. | CIFAR-10/100, ImageNet1k, DTD, Caltech101. | ⭐⭐ — real on Caltech101 / CIFAR-100, marginal on CIFAR-10 / ImageNet, essentially flat or worse on DTD. "Consistently" is generous. |
| C6 | Closing the gap improves multi-modal arithmetic by 18%. | Table 6 SIMAT @ λ=1. | SIMAT. | ⭐ — single λ, no seeds, no SE. The biggest headline number rests on the weakest reporting. |
| C7 | "Closing the contrastive gap improves downstream performance." | C5 + C6. | as above. | ⭐⭐ — overgeneralized. Retrieval is unchanged; gains are small on small/easy datasets *after* fine-tuning on COCO with a 32-128D head, not standard CLIP at 512D. |

**Honest read.** The reframing (C1) is the strongest intellectual contribution — the unimodal control is exactly the right experiment, and it lands. But it is *one number per cell*: a serious version would replicate across seeds, batch sizes, and dimensionalities to show the gap-formation curve as a function of d (the paper hints d matters but never sweeps it in the control). The fix (C3) is genuinely well-supported. The "improves downstream" framing (C5-C7) leans on a +18% SIMAT bump with no variance, while the more rigorously-measured retrieval task shows *no* improvement. The fine-tuning regime also confounds two things: (a) reducing the projection head from 512D -> {32, 64, 128}D, and (b) adding new loss terms — the baseline `L_CLIP` at d=128 is *also* a re-projected, fine-tuned variant, so we never see an apples-to-apples comparison against full-strength 512D OpenAI CLIP.

## Method & Architecture

![3D CLIP training, epoch 0: image and text embeddings sit in disjoint cones](/assets/images/paper/contrastive-gap/fig_p005_04.png)
*Figure 1d: Epoch 275 of the 3D demo — embeddings finally cover the unit sphere and only here does I->T retrieval jump to 0.87. The trajectory disjoint cones -> arcs -> rings -> uniform sphere is the visual core of the paper's "uniformity, not alignment alone, drives downstream quality" claim.*

### 1. Gap metrics (Section 3.1)

- **Centroid distance**: $\|C_I - C_T\|_2$ where $C_M = \tfrac{1}{N}\sum_j E_j^M$ (range 0-2).
- **Linear separability**: train a linear classifier on 80% of {image-emb, text-emb} labels, report test accuracy on 20%. 50% means perfectly overlapping; 100% means disjoint.

### 2. The unimodal control (Section 3.2)

The experiment that carries the paper:

1. Replace CLIP's text encoder with a **second copy of the image encoder** (different random init).
2. Compute a fixed translation matrix that **overlaps encoder-2's init cone onto encoder-1's**, removing initialization-time gap.
3. Use **identical-image pairs** as positives, eliminating mismatched-pair noise.
4. Train to convergence (loss = 0) on 2048 MS COCO images, batch 64, 512D, ~20k steps / 1200 epochs.

If any of Liang et al.'s three causes (cone, mismatched pairs, undertraining) were necessary, the gap should not appear here. It does.

### 3. CLIP loss baseline

$$
L_{CLIP} = -\frac{1}{2N}\sum_{j} \log \frac{\exp(\langle E_j^I, E_j^T\rangle/\tau)}{\sum_k \exp(\langle E_j^I, E_k^T\rangle/\tau)} - \frac{1}{2N}\sum_{k} \log \frac{\exp(\langle E_k^I, E_k^T\rangle/\tau)}{\sum_j \exp(\langle E_j^I, E_k^T\rangle/\tau)}
$$

### 4. New loss terms (Section 4)

- **In-modal uniformity**: $L^I_{Uniform} = \log\!\left(\tfrac{1}{N}\sum_{j,k} \exp(-2\|E_j^I - E_k^I\|^2)\right)$, similarly for text; combined as $L_{Uniform} = \tfrac{1}{2}(L^I_{Uniform} + L^T_{Uniform})$.
- **Cross-modal uniformity**: $L_{XUniform} = \log\!\left(\tfrac{1}{N}\sum_{j}\sum_{k\neq j} \exp(-2\|E_j^I - E_k^T\|^2)\right)$.
- **Alignment**: $L_{Align} = \tfrac{1}{N}\sum_j \|E_j^I - E_j^T\|^2$.

Two combined losses studied: `L_CUA = L_CLIP + L_Uniform + L_Align` and `L_CUAXU = L_CUA + L_XUniform`.

### 5. Fine-tune protocol (Section 5 + Appendix B.3)

Pre-trained OpenAI CLIP ViT-B/32; **final projection layer reinitialized to project to d ∈ {32, 64, 128}**; τ = 0.01 (fixed); 9 epochs on MS COCO 2017 train (118k); batch 64; LR 1e-6; AdamW (β=0.9/0.99, wd=0.1); single A5000, ~3.5h/run; **3 seeds**.

A meaningful caveat: only the *first* of MS COCO's 5 captions is used per image — discarding ~80% of caption supervision and not ablated.

![Cumulative PCA explained variance at d=128](/assets/images/paper/contrastive-gap/fig_p007_01.png)
*Figure 2: At d=128, `L_CUAXU` (red) needs the most principal components to reach 0.8 cumulative variance — embeddings spread along a higher-dimensional manifold than vanilla `L_CLIP`. The gap between curves widens as d grows (Fig. 4).*

## Experimental Results

### Idealized unimodal experiment (Table 1, 512D, 2048 imgs, 1200 epochs, 1 run)

| Stage | Centroid distance | Linear separability | Contrastive loss |
|---|---|---|---|
| Init | 0.00 | 0.50 | 4.83 |
| **After training** | **0.06** | **1.00** | **0.00** |

Even with one modality, cone-aligned init, and zero mismatched pairs, encoders end perfectly linearly separable. **No seed variance, no d sweep** — the central reframing rests on a single number per cell.

### Gap metrics on MS COCO val, 128D (Tables 2 / 9, mean ± 1 SE over 3 seeds)

| Loss | Linear-sep ↓ | Centroid dist ↓ |
|---|---|---|
| L_CLIP | 1.00 ± 0.00 | 0.66 ± 0.03 |
| L_CUA | **0.73 ± 0.01** | **0.08 ± 0.02** |
| **L_CUAXU** | **0.83 ± 0.02** | **0.13 ± 0.01** |

Same pattern at d=32 (1.00 -> 0.59 LS; 0.10 -> 0.07 CD) and d=64 (1.00 -> 0.65 LS; 0.31 -> 0.07 CD). The gap also **grows with d** for vanilla CLIP — the paper's main piece of indirect evidence that this is high-dimensional contrastive geometry.

### MS COCO retrieval, 128D (Tables 4 / 10, ±1 SE)

| Loss | I→T@1 | I→T@5 | I→T@10 | T→I@1 | T→I@5 | T→I@10 |
|---|---|---|---|---|---|---|
| L_CLIP | 0.28 | 0.57 | 0.70 | 0.27 | 0.56 | 0.69 |
| L_CUA | 0.26 | 0.54 | 0.68 | 0.25 | 0.54 | 0.66 |
| **L_CUAXU** | **0.27** | **0.56** | **0.69** | **0.27** | **0.56** | **0.69** |

All values within 1 SE of each other — **closing the gap does not improve retrieval**. This is the cleanest negative result in the paper and it is in tension with the abstract's "closes gap -> better downstream" framing.

### Zero-shot classification, 128D (Table 11, ±1 SE)

| Loss | CIFAR-10 | CIFAR-100 | ImageNet | DTD | Caltech101 |
|---|---|---|---|---|---|
| L_CLIP | 0.76 ± 0.02 | 0.32 ± 0.01 | 0.13 ± 0.00 | 0.10 ± 0.01 | 0.42 ± 0.01 |
| L_CUA | 0.78 ± 0.01 | 0.33 ± 0.00 | 0.14 ± 0.00 | 0.10 ± 0.01 | 0.48 ± 0.00 |
| **L_CUAXU** | **0.77 ± 0.02** | **0.35 ± 0.01** | **0.15 ± 0.00** | **0.09 ± 0.01** | **0.49 ± 0.01** |

Improvements are 1-7 pp, strongest on Caltech101 (+7), real on CIFAR-100 (+3) and ImageNet (+2), statistically marginal on CIFAR-10, and **flat or worse on DTD**.

### SIMAT multi-modal arithmetic (Table 6, λ=1)

| Loss | SIMAT ↑ |
|---|---|
| L_CLIP | 36.02 |
| L_CUA | 42.18 |
| **L_CUAXU** | **42.47** |

The headline +18% relative — **but no error bars, no seed variance, single λ, no per-subset breakdown**. The biggest claimed downstream win is the most weakly evidenced.

![Average zero-shot accuracy vs CLIP dimensionality](/assets/images/paper/contrastive-gap/fig_p007_02.png)
*Figure 3: Average zero-shot accuracy across CIFAR-10/100, ImageNet, DTD, Caltech101 versus dimensionality — uniformity-augmented losses widen their lead over vanilla CLIP as d grows from 32 -> 128, mirroring the gap-grows-with-d pattern in centroid distance.*

## Limitations

**Authors admit (§C).** Only fine-tuning on MS COCO, no from-scratch large-scale runs (so claims do not transfer to WIT-400M-scale training). Image-text retrieval is unchanged despite better uniformity/alignment, hinting that "better latent space" ≠ "better retrieval". Other latent-space properties may be more predictive of representational quality than uniformity/alignment.

**Not addressed but visible.**
- The unimodal control (Table 1) has **no seed variance, no d sweep, no batch-size sweep**. The central reframing rests on one number per cell.
- The fine-tuning protocol re-initializes the final projection from 512D to {32, 64, 128}D, so all reported `L_CLIP` baselines are *also weakened* relative to publicly-released CLIP. **No apples-to-apples comparison with stock 512D CLIP.**
- **Temperature τ is fixed at 0.01** throughout. Temperature is well known to control uniformity in contrastive losses (Wang & Isola 2020 §4.2; Wang & Liu 2021); fixing it eliminates a natural alternative explanation.
- **Only the first of MS COCO's 5 captions is used.** This changes the alignment ceiling and is not ablated.
- **SIMAT** is reported at a single λ with no SE — biggest claim, weakest reporting.
- **No medical, satellite, or other-domain CLIP variants tested**, despite the introduction citing medical/video/protein contrastive models.
- No side-by-side benchmark against CyCLIP (Goel et al. 2022) or modality-gap geodesic mixup (Oh et al. 2023).
- The "low uniformity -> gap" story is correlational. A principled test would directly enforce a higher-dimensional manifold via a different mechanism (e.g., whitening, BarlowTwins-style decorrelation) and check whether the gap closes without the specific uniformity loss.

## Why It Matters for Medical AI

Clinical CLIP variants (CONCH, BiomedCLIP, MedCLIP) inherit this geometry. Two practical takeaways:

- **Stop attributing the gap to "image vs report" being fundamentally different.** The unimodal control says the geometry forms even when both encoders see identical natural images. So translation/projection tricks that "align modalities" in clinical embedding spaces are likely closing a contrastive-loss artifact, not a modality distinction.
- **But do not assume closing the gap will help your downstream task.** The cleanest measurement in the paper — image-text retrieval with error bars — shows *no improvement* from `L_CUAXU`. For a clinical retrieval setting (find related cases for a query report) this paper provides no evidence that uniformity-augmented CLIP would help. The wins are on small zero-shot classification benchmarks (1-7 pp) and SIMAT arithmetic (+18%, no variance) — neither is a chest-X-ray-grade evidence regime.

The honest summary: this is a representation-geometry paper with a sharp conceptual contribution (the control experiment) and a well-measured fix to a property nobody has yet shown is the property that matters. Worth knowing about; not yet worth deploying clinically.

## References

- Paper: Fahim, Murphy, Fyshe, *It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap*, arXiv:2405.18570v3, 6 Jun 2024 (preprint, under review).
- Background: Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, NeurIPS 2022.
- Loss-design ancestors: Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020; Wang & Liu, *Understanding the Behaviour of Contrastive Loss*, CVPR 2021.
- Related fixes: Goel et al., *CyCLIP: Cyclic Contrastive Language-Image Pretraining*, NeurIPS 2022; Oh et al., *Geodesic Multi-Modal Mixup for Robust Fine-Tuning*, NeurIPS 2023; Welle, *Modality Gap Analysis* (2023).
- Evaluation: Couairon et al., *Embedding Arithmetic for Text-driven Image Transformation* (SIMAT), arXiv:2112.03162.
- Datasets: MS COCO 2017 (Lin et al. 2014); CIFAR-10/100; ImageNet1k; DTD; Caltech101.
