---
title: "VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis"
excerpt: "A single LLaMA-architecture decoder emits Python that drives a jointly-trained 3D UNet at native voxel resolution, beating MoME by +12.53 mean Dice on four unseen lesion datasets and matching FreeSurfer's longitudinal AD effect size 3.8x10^5 times faster."
categories:
  - Paper
  - LLM-Agents
  - Pathology
permalink: /paper/voxelprompt-3d-medical-vision-language-agent/
tags:
  - VoxelPrompt
  - LLM-Agent
  - 3D-Segmentation
  - Neuroimaging
  - UNet
  - LLaMA
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- VoxelPrompt is a single end-to-end agent that takes a natural-language prompt plus any number of 3D MRI/CT volumes and has a **LLaMA-architecture decoder emit executable Python** that drives a jointly-trained 3D UNet — replacing the manual chaining of segmentation + measurement + reporting tools that neuroimaging workflows currently require.
- One model covers **185 bilateral anatomical structures + 14 pathology classes** across MRI (T1w/T2w/FLAIR/PD/GRE/DWI) and CT, matches or beats **13/17 single-task specialists**, and beats MoME by **+12.53 mean Dice** on four genuinely unseen lesion datasets.
- Longitudinal Alzheimer's analysis matches FreeSurfer's hippocampal-atrophy effect size (-1.21 vs -1.20) at a **3.8x10^5x runtime speedup** (21.4 hrs -> 0.2 sec); native-resolution processing also delivers 2x runtime and 2.4x memory savings vs isotropic resampling.

## Motivation

Real neuroimaging questions — "how much has *this specific* lesion grown since last visit?", "hippocampal asymmetry normalized by cranial size", "isolate the parietal-lobe hypointensity" — require chaining brittle, fixed-target tools and writing custom post-processing. There is no end-to-end interface that takes a free-form question over a multi-volume scan session and returns a quantitative, traceable answer. Existing medical VLMs (LLaVA-Med, RadFM) are largely 2D and text-only; existing 3D segmenters (SynthSeg, MoME, SAT, BiomedParse v2) are fixed-target and cannot be language-prompted. VoxelPrompt's pitch is that the LLM should **plan and execute**, not hallucinate answers, while a co-trained vision network supplies the spatial primitives.

## Core Innovation

- **Agent emits Python, not answers.** A 16-block LLaMA-architecture decoder (d=512, FFN 2048, 32 heads, LLaMA-2 tokenizer) trained **from scratch** interleaves code tokens with latent embeddings phi that condition the vision net. Code runs in a persistent Python interpreter; the agent reads variables back as feedback embeddings and continues, producing a traceable program + intermediate segmentations + final answer.
- **Co-trained vision UNet at native voxel resolution.** Instead of resampling to 1 mm^3 isotropic, in-plane spacing x and slice spacing y are tracked explicitly and downsampled with explicit anisotropy control (y is only equalized to x once r = y/x <= 2). Yields 2x runtime / 2.4x inference-memory / 2.2x training-memory savings; with 5 input volumes, isotropic conformation **OOMs on 90% of batches on an 80 GB A100**.
- **Attention-based stream-interaction module.** Per-layer voxel features across S streams are concatenated with stream-specific phi, projected, and attended over (dim b=32) before being split back per stream. This lets the vision net consume an arbitrary number of co-registered volumes per forward pass, and is the only fusion that holds up under corrupted inputs (-0.6% Dice vs -4.6% max / -5.4% mean).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | A single jointly-trained agent + vision net matches or exceeds task-specific specialists | Fig 4A — 13/17 targets p>0.05 or favorable; pathology +4.3 +/- 5.7%, anatomy -0.1 +/- 0.3% | 638 held-out subjects, 17 targets | ⭐⭐⭐ — clean ablation, variance reported, paired tests |
| C2 | Only zero-shot lesion segmenter that generalizes across diverse abnormality types | +12.53 mean Dice over MoME; only method competitive on all four sets | BraTS-MEN, EPISURG, PediMS, BHSD (MRI + CT) | ⭐⭐⭐ — four unseen datasets, two modalities, strong margin |
| C3 | Beats SOTA SynthSeg on whole-brain anatomical segmentation | +1.1 +/- 2.3% mean Dice; sig on 23/45 ROIs (p<0.05) | 108 held-out volumes | ⭐⭐ — small absolute gain, within one std; authors themselves walk it back |
| C4 | Native-resolution processing yields 2x runtime / 2.4x memory gain | Fig 4B over 500 sampled geometries | Synthetic geometry distribution (Appendix C.3) | ⭐⭐ — large N but author-chosen geometry distribution, no clinical workload benchmark |
| C5 | Attention stream-interaction is more robust under corrupted inputs | -0.6% vs -4.6% / -5.4% Dice under corruption | 500 synthetic brains (OASIS-derived) | ⭐⭐ — synthetic corruption protocol only, no real low-quality clinical scans |
| C6 | Matches longitudinal FreeSurfer effect size with ~10^5x speedup on AD detection | Hippocampus -1.21 vs -1.20; ventricles 1.14 vs 1.12; 21.4 hr -> 0.2 s | ADNI, 100 subjects (50 AD / 50 control), 2-yr follow-up | ⭐⭐ — single cohort, two anatomical structures |
| C7 | Free-form natural-language workflow generation works in real use-cases | Figs 1, 3A — qualitative examples only | Authors' held-out scans | ⭐ — **explicitly qualitative; no code-correctness metric, no Pass@k, no inter-rater study** |
| C8 | Pathology VQA matches single-task classifiers and fine-tuned RadFM | 89.0 +/- 3.6% vs 89.3 +/- 4.2% vs 87.1 +/- 7.9% | 102 cases, 5 VQA tasks | ⭐⭐ — small set, exact-string-match metric; **RadFM was crippled to 8 LLM layers for memory**, which weakens "matches SOTA" framing |
| C9 | Training from scratch is a sound choice vs finetuning a pretrained LLM | None presented | — | ⭐ — implicit claim; authors flag the missing ablation as future work |

**Honest read.** C1, C2 are the strongest contributions and well-supported by clean specialist comparisons across many targets and four genuinely unseen datasets. C5, C6 are convincing but rest on synthetic corruption protocols or a single ADNI cohort. The biggest gap is C7: the headline pitch — *free-form workflow generation* — is shown almost entirely qualitatively. The paper never quantifies how often the agent emits correct Python on unseen prompt phrasings, the error modes when prompts are out-of-template, or end-to-end correctness on multi-step compositional tasks beyond hand-picked Figure 1 examples. The C8 RadFM parity claim is also weaker than it reads, because the RadFM baseline was truncated to its first 8 LLM layers to fit memory.

## Method & Architecture

![VoxelPrompt agent + vision net co-training overview](/assets/images/paper/voxelprompt/page_004.png)
*Figure 2: The agent alpha emits Python code into a persistent environment Omega that calls a jointly-trained vision network m_enc / m_gen, conditioned by latent embeddings phi. Bottom: a two-visit tumor-growth query unrolls across three code steps with variables persisting between them.*

### 1. State and agent loop

Given a natural-language prompt p and a set V of 3D volumes (any number, any acquisition, any resolution), the initial state mu_1 encodes p and per-volume acquisition-date metadata. The agent alpha is a decoder-only transformer with the LLaMA architecture, randomly initialized: 16 blocks, hidden d = 512, FFN width 2048, 32 heads, LLaMA-2 tokenizer (gamma = 32,000). At step i it emits an embedding sequence phi_i = phi_c (+) phi_phi. phi_c is projected to token probabilities P(c_i) and argmax-decoded into code c_i; phi_phi tokens follow special `<MOD>` markers and are projected by a SiLU FC layer to vision conditioners phi in R^32 that condition the vision net for that specific instruction.

### 2. Persistent execution environment

A Python interpreter Omega holds variables (encodings, segmentations, scalars) across steps. c_i runs in Omega; `read(...)` operations pull a variable's value back into a feedback embedding z_i which is concatenated into the next state:

$$
\mu_{i+1} = \mu_i \oplus \phi_i \oplus z_i
$$

The loop terminates when the agent emits a stop token.

### 3. Vision network

A 6-level UNet split into encoder m_enc(V, phi) producing multi-scale features E and a generator m_gen(E, phi) producing volumes W. 3D conv kernels 3^3, 32 channels at the top, 96 at lower levels, SiLU + groupnorm(4), max-pool down / trilinear up. The conditioning phi is mixed into every level. Sigmoid output for binary segmentation.

### 4. Stream-interaction module

For each layer's voxel feature a_s in R^c in stream s, concatenate with stream-specific phi_s, FC -> a'_s, stack across S streams to A' in R^{S,c}, then attend with dim b=32:

$$
B = f(\mathrm{softmax}(QK^\top/\sqrt{b})\, V) + A',
$$

then split back per stream. This is what lets the vision net consume a variable number of volumes per forward pass.

### 5. Native-resolution processing

Spacings (x, y) are explicit. **Downsampling:** x_{n+1} = 2 x_n; y_{n+1} is updated to x_{n+1} only when anisotropy r_n = y_n / x_n <= 2. **Upsampling:** spacings inferred from skip connections. Cross-stream attention temporarily resamples to a common geometry, then returns features to original space.

### 6. Loss and training

L = L_ce(P(c), c*) + lambda * sum_j L_img(W_j, W*_j), with L_img = soft Dice and lambda = 0.1. Adam, lr 1e-4, batch size 1 with 10 gradient-accumulation steps on a single A100; halve lr after 10^5 steps without val improvement; stop after 4 reductions. Prompts are synthesized by combinatorial templates (per task tau, sample template from P_tau, recursively fill placeholders) to broaden terminology, syntax, and tense coverage.

## Experimental Results

![VoxelPrompt main quantitative results: zero-shot lesion segmentation, longitudinal AD, whole-brain Dice](/assets/images/paper/voxelprompt/page_007.png)
*Figure 3: (A) language-targeted ROI selection; (B, C) zero-shot lesion segmentation across 4 unseen datasets; (D) longitudinal Alzheimer's effect size vs FreeSurfer with ~10^5x speedup; (E) whole-brain anatomical Dice vs SynthSeg.*

### Zero-shot lesion segmentation (Dice %, Figure 3C)

| Dataset (target) | VoxelPrompt | MoME | SAT | BiomedParse v2 |
|---|---|---|---|---|
| BraTS-MEN (meningioma, n=30) | **87.1** | 81.5 | 41.9 | 60.2 |
| EPISURG (resection cavity, n=35) | **74.7** | 50.5 | 7.2 | 15.3 |
| PediMS (pediatric MS, n=9) | **74.1** | 53.2 | 73.9 | 25.5 |
| BHSD (hemorrhage, n=36) | **63.5** | 64.1 | 48.1 | 34.2 |
| **Mean** | **74.85** | 62.33 (Delta -12.53) | 42.78 | 33.80 |

VoxelPrompt is the only method competitive on all four datasets; the second-best baseline varies dataset-to-dataset, indicating that none of the specialists generalize.

### Multi-task vs single-task specialists (Figure 4A, 638 held-out subjects, 17 targets)

- Pathology (7 targets): mean Dice **+4.3 +/- 5.7%** *better* than per-target specialists.
- Anatomy (10 targets): **-0.1 +/- 0.3%** (essentially tied).
- Matches or beats 13/17 specialists; asterisked diffs are p < 0.05.

### Whole-brain anatomical segmentation vs SynthSeg v2 (Figure 3E)

- Mean Dice improvement **+1.1 +/- 2.3%** over SynthSeg across 45 ROIs on 108 held-out volumes.
- Statistically significant improvement on **23/45 ROIs** (p < 0.05).
- Authors explicitly frame this as "we are not trying to win at segmentation, only to retain it while gaining prompting flexibility."

### Longitudinal Alzheimer's analysis vs longitudinal FreeSurfer (Figure 3D, ADNI, 100 subjects)

| Metric | FreeSurfer | VoxelPrompt |
|---|---|---|
| Hippocampal-atrophy effect size | -1.20 (p = 2.5e-7) | **-1.21 (p = 1.9e-7)** |
| Lateral-ventricle-expansion effect size | 1.12 (p = 2.6e-6) | **1.14 (p = 1.8e-6)** |
| Runtime | 21.4 +/- 1.4 hrs | **0.2 +/- 0.0 sec (~3.8x10^5x faster)** |

### Ablations

![VoxelPrompt ablations: multi-task vs specialists, native-resolution efficiency, stream-interaction robustness](/assets/images/paper/voxelprompt/page_008.png)
*Figure 4: (A) Single multi-task VoxelPrompt vs 17 single-task specialists; (B) native-resolution vs isotropic resampling — 2x runtime / 2.4x memory savings; (C) attention vs mean/max stream interaction under corrupted inputs.*

**Native-resolution efficiency (Figure 4B, avg over 500 samples).** 2x inference-runtime reduction, 2.4x inference-memory reduction, 2.2x training-memory reduction vs isotropic conformation. With 5 input volumes, isotropic conformation OOMs on 90% of batches on an 80 GB A100.

**Stream-interaction ablation (Figure 4C).** On uncorrupted multi-volume inputs, all three reductions tie (attention 86.9 / max 87.1 / mean 86.7 at 3 images). On corrupted inputs, attention degrades only -0.6 +/- 3.4% Dice vs -4.6 +/- 4.5% (max) and -5.4 +/- 4.0% (mean). This is the actual justification for the attention block.

**Pathology characterization (Appendix D, 102 cases x 5 VQA tasks).** VoxelPrompt 89.0 +/- 3.6% vs single-task classifiers 89.3 +/- 4.2% vs fine-tuned RadFM 87.1 +/- 7.9%. Note that **RadFM was truncated to its first 8 LLM layers** to fit memory, which weakens the parity framing.

### Dataset and lesion synthesis

![Procedural lesion synthesis pipeline](/assets/images/paper/voxelprompt/fig_p018_03.png)
*Figure 5: Procedural lesion synthesis — Brownian noise -> thresholded shape -> anatomically constrained placement -> Perlin-texture in-painting. Supports negative examples and intra-lesional heterogeneity.*

Training data: 6,925 3D brain MRI/CT scans aggregated from 15 public cohorts (4,852 train / 213 val / 1,860 test). Anatomy labels from FreeSurfer/FastSurfer/SynthStrip pipelines plus Adil 2021 and Pauli 2018 atlas annotations; pathology from BraTS, ISLES, ATLAS, WMH plus 101 manually-annotated Radiopaedia cases reconstructed from 2D slices via affine registration. Preprocessing: intensity normalize to [0, 1], RAS reorientation, 20 mm FOV cropping around cranial cavity, SynthMorph co-registration. Heavy augmentation: random affine, bias-field, k-space corruption, anatomical masking, and **random slice-separation sparsification in [1, 6] mm at 50% probability** to control voxel throughput.

## Limitations

**Authors admit.**
- Training on template-synthesized prompts limits generalization to truly unseen phrasings.
- Brain-only — no body, cardiac, or musculoskeletal evaluation.
- Training from scratch is heavy; a finetuned pretrained LLM might generalize better and is flagged as future work.

**Unaddressed.**
- **No code-generation correctness rate.** No Pass@k metric, no inter-rater scoring of emitted Python, no quantitative evaluation of whether the agent emits *semantically correct* code (e.g. measures the right ROI) on out-of-template prompts.
- **No safety / hallucination analysis** for syntactically valid but semantically wrong code.
- **No radiologist agreement study** with VoxelPrompt's free-form quantitative outputs.
- **RadFM baseline was crippled** (only first 8 LLM layers retained) to fit memory, which makes the VQA parity claim less clean than it reads.
- **No baseline against a frozen pretrained LLM + tool-calling setup** (e.g., MMedAgent is cited but never quantitatively compared).
- **Total training wall-clock not reported** ("single A100" only).
- Pediatric and CT representation in training data is sparse; rare-pathology labels come from a single annotator pool.

## Why It Matters for Medical AI

VoxelPrompt is the cleanest existing demonstration that a **single 3D vision agent can replace a brittle chain of fixed-target tools** while remaining traceable: the emitted Python is auditable, intermediate segmentations are inspectable, and the final answer is a real number rather than an LLM hallucination. The C1/C2 evidence — matching 13/17 specialists with a single model and +12.53 Dice over MoME on four unseen lesion datasets — is strong enough to take seriously as a template for end-to-end neuroimaging workflows. The 3.8x10^5x speedup over FreeSurfer at parity AD effect size also reframes longitudinal cohort analysis from a multi-hour batch job into an interactive query.

For practitioners the practical implications are:
- Use VoxelPrompt where **free-form, multi-step quantitative neuroimaging queries** are the bottleneck — longitudinal cohort analyses, ad-hoc ROI measurements, and language-prompted lesion isolation.
- Do **not** rely on it for safety-critical autonomous reporting yet — the headline workflow-generation claim (C7) has no quantitative correctness evaluation, and the C8 RadFM-parity result is built on a crippled baseline.
- For pathology-specific or whole-body 3D tasks, the architecture is plausible but the paper provides no evidence; the published model is brain-only.
- The native-resolution + attention-fusion design is the most reusable component — it is the part that makes "any number of co-registered volumes at arbitrary acquisition geometry" tractable on a single 80 GB A100.

## References

- Paper: Hoopes, Dey, Butoi, Guttag, Dalca, *VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis*, arXiv:2410.08397v2, 15 Oct 2025.
- Related work: SynthSeg (Billot et al. 2023), SynthMorph (Hoffmann 2024), MoME (Zhang et al. 2024), SAT (Zhao et al. 2024), BiomedParse (Zhao et al. 2024), FreeSurfer / FastSurfer, RadFM (Wu et al. 2023), LLaVA-Med (Li et al. 2023), MMedAgent (Li et al. 2024).
- Datasets: BraTS, ISLES, ATLAS, WMH, EPISURG, PediMS, BHSD, ADNI, OASIS, Radiopaedia.
