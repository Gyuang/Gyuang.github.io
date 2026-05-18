---
title: "Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D"
excerpt: "Carreira-style I3D inflation of MedSigLIP plus a three-stage alignment with MedGemma 1.5-4B lifts brain-MRI Clinical Pathology F1 from 0.413 (2D slice baseline) to 0.951 on a 468-subject BraTS2020+Brainlife benchmark — though the headline gap conflates 3D encoding with three stages of in-domain fine-tuning."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/brain3d/
tags:
  - Brain3D
  - MedSigLIP
  - MedGemma
  - I3D-Inflation
  - LoRA
  - Vision-Language
  - 3D-MRI
  - Report-Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- Brain3D inflates a pretrained 2D medical encoder (**MedSigLIP**) into a native 3D encoder via Carreira-style I3D depth-replication, then aligns it with **MedGemma 1.5-4B-IT** through a three-stage protocol — contrastive grounding, projector warmup, and LoRA fine-tuning of LLM attention layers (r=16, alpha=32).
- Decomposed positional embeddings — learnable along depth + broadcast pretrained 2D spatial — let the inflated encoder retain MedSigLIP's spatial inductive biases while learning volumetric structure from a 468-subject corpus.
- Headline: **Clinical Pathology F1 = 0.951 (95% CI [0.91, 0.98]) vs 0.413 for MedGemma 1.5 2D slice-based baseline** on a single 20% in-distribution test split (n approx. 94) — but the comparison is confounded by three stages of in-domain training that the 2D baseline never receives.

## Motivation

Brain MRI interpretation needs coherent 3D spatial reasoning: hemispheric laterality, tumor infiltration patterns, periventricular signal changes. Most medical VLMs — Med-Flamingo, LLaVA-Med, MedGemma — are natively 2D and process volumes as slice stacks, which the authors argue produces lateralization errors and false lesion attribution. Generalist 3D medical VLMs (M3D-LaMed, Med3DVLM) exist but are CT-centric and lack neuroradiology grounding; training a 3D foundation from scratch is data- and compute-prohibitive on the BraTS-scale corpora available. Brain3D's bet is that **inflating a domain-specialized 2D encoder and progressively aligning it with a medical LLM** is the cheap, clinically-faithful path to brain-MRI report automation.

## Core Innovation

- **I3D inflation of a medical encoder.** Replicate the 2D MedSigLIP patch-embedding kernel along the depth axis (naive Carreira inflation, no learning), then normalize weights to preserve activation scale. The paper does not ablate inflated init vs random 3D init, so the value of inflation specifically is asserted rather than measured.
- **Decomposed positional embeddings.** P_3D(z,y,x) = P_depth(z) + P_spatial(y,x), where P_depth is learnable and P_spatial reuses pretrained 2D embeddings broadcast along depth — explicitly preserves the pretrained 2D spatial prior while injecting fresh depth structure.
- **Three-stage alignment.** Phase 1 contrastive InfoNCE (only inflated patch embedding + projector + scalar gate trainable) → Phase 2A projector warmup with masked next-token prediction → Phase 2B LoRA fine-tuning of LLM attention layers (r=16, alpha=32). The staged ablation is the cleanest result in the paper.
- **Soft-prompt conditioning, no cross-attention.** K=32 pooled visual tokens are simply prepended to text embeddings — a deliberately minimalist injection that keeps MedGemma's text path intact.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Inflating a 2D medical encoder into native 3D outperforms slice-based 2D processing for brain-MRI report generation. | Table I: Brain3D Path F1 0.951 vs MedGemma 2D 0.413; Lat F1 0.689 vs 0.526; Anat F1 0.691 vs 0.461. | BraTS2020 + Brainlife controls, single 20% test split (n approx. 94). | ⭐⭐ |
| C2 | Native 3D encoding is "necessary" for diagnostic factualness (+130% over 2D baseline). | Same Table I comparison. | Same single split. | ⭐⭐ — confounded: Brain3D differs from MedGemma in encoder, training schedule, AND LoRA fine-tuning. There is **no fair 2D-vs-3D ablation** where MedGemma 1.5 gets the same three-stage in-domain training. |
| C3 | Staged alignment is essential — each phase contributes. | Table I bottom block: Phase 1 → 2A → 2B monotonic gains on every metric except CIDEr. | Same split. | ⭐⭐⭐ — the cleanest, best-supported claim in the paper. |
| C4 | Generalist 3D pretraining alone is insufficient for neuroradiology. | Table I: Med3DVLM Path F1 0.119, near-zero CIDEr 0.007. | Same split, Med3DVLM evaluated **zero-shot with no in-domain fine-tuning**. | ⭐ — unfair comparison the paper does not acknowledge. |
| C5 | Brain3D achieves "perfect specificity on healthy scans." | Abstract and conclusion prose only. | 99 healthy controls, ~20 in the test split. | ⭐ — **no per-cohort specificity table, no confusion matrix, no number reported beyond the prose claim**. |
| C6 | Phase 2B shifts output from verbose captions toward structured clinical reports. | CIDEr drops 0.504 → 0.293 between 2A and 2B; qualitative Table II. | Same split. | ⭐⭐ |
| C7 | Lateralization errors are the dominant residual failure (~15%). | Prose in IV-F Error Analysis; Fig. 3 LIME overlays. | Pathological subset of test set (~74 cases). | ⭐⭐ — qualitative, no error matrix or per-class breakdown. |
| C8 | Inflation preserves pretrained inductive biases. | Implicit — no ablation comparing inflated vs randomly-initialized 3D encoder. | — | ⭐ — **no controlled experiment isolating the inflation step**. |

**Honest read.** The staging ablation (C3, C6) is the strongest result — Phase 1 → 2A → 2B gives clean monotonic gains and is well-supported by Table I. The headline "2D vs 3D" comparison (C1, C2) is **confounded**: MedGemma 1.5 is evaluated zero-shot on slice stacks while Brain3D receives three stages of supervised training on BraTS+TextBraTS, so the +130% gap measures the joint effect of inflation, staged alignment, in-domain data, and LoRA — **not "3D vs 2D" in isolation**. A fair ablation would fine-tune MedGemma 1.5 with the same Phase 2A/2B pipeline on slice-stacked inputs; this is missing. C4 (Med3DVLM at 0.119 Path F1) is similarly unfair — a generalist CT-centric model evaluated cold against an in-domain fine-tuned competitor. The "perfect specificity on healthy scans" claim (C5) is asserted but never quantified — no number, no table, no confusion matrix. No prior brain-MRI report-generation systems (R2GenGPT, brain-specific CNN-LSTM baselines) are benchmarked despite being cited in Related Work. And critically, **the "reports" are TextBraTS templates** auto-generated from segmentation labels rather than radiologist prose — a caveat the authors do not flag — so the 0.951 Path F1 should be read as "the model matches a rule-based extractor over four descriptors on tag-derived captions on a single in-distribution split," not "clinically deployable report generation."

## Method & Architecture

![Brain3D end-to-end architecture: inflated 3D ViT encoder, adaptive pooling, MLP projector, and MedGemma soft-prompt conditioning](/assets/images/paper/brain3d/fig_p003_01.png)
*Figure 1: Brain3D architecture — skull-stripped 3D MRI → inflated 3D ViT encoder (MedSigLIP weights, Carreira-style depth replication) → adaptive pooling to K=32 visual tokens → MLP projector with learnable scalar gate s → soft-prompt prepended to MedGemma 1.5-4B-IT for autoregressive report generation.*

### 1. Task formulation

Given a volumetric MRI $X \in \mathbb{R}^{C \times D \times H \times W}$ and a fixed instruction prompt $Y_{\text{prompt}}$, generate report $Y$ autoregressively:

$$
p(Y \mid X, Y_{\text{prompt}}) = \prod_{t=1}^{|Y|} p(y_t \mid X, t_{1:S}, y_{<t})
$$

### 2. Preprocessing

Skull-stripping → RAS reorientation → 1st-99th percentile intensity clipping to [0,1] → resample to a fixed grid of **64 x 128 x 128** voxels (single channel).

### 3. Inflated 3D Vision Encoder

The backbone is **MedSigLIP** (MedGemma family), a 2D Transformer pretrained on medical image-text pairs. The 2D patch-embedding kernel $W_{2D}$ is inflated into $W_{3D}$ by collapsing RGB channels into a single channel and **replicating the kernel along the depth axis** — Carreira-style I3D inflation [Carreira & Zisserman 2017] — then **normalizing the inflated weights** to preserve activation scale. Inflation is purely a weight-init scheme; nothing about the inflation itself is learned.

Positional embeddings are decomposed:

$$
P_{3D}(z, y, x) = P_{\text{depth}}(z) + P_{\text{spatial}}(y, x)
$$

where $P_{\text{depth}}$ is learnable and $P_{\text{spatial}}$ reuses pretrained 2D embeddings broadcast along the depth axis. Output: $Z_{\text{enc}} \in \mathbb{R}^{N \times d_v}$.

### 4. Visual token compression

Adaptive average pooling along the sequence dimension reduces $N$ volumetric tokens to **K = 32** compressed visual tokens $Z_{\text{pool}} \in \mathbb{R}^{K \times d_v}$, decoupling volume resolution from LLM context length.

### 5. Vision-Language projection

Two-layer MLP with GELU maps $d_v \to d_{\text{llm}}$, scaled by a learnable scalar gate $s$ (`vis_scale`):

$$
Z_{\text{vis}} = s \cdot \text{MLP}(Z_{\text{pool}})
$$

Low initial $s$ enables gradual visual conditioning.

### 6. Soft-prompt LLM conditioning

No cross-attention — visual tokens are simply prepended to text embeddings:

$$
Z_{\text{in}} = \text{Concat}(Z_{\text{vis}}, Z_{\text{txt}}) \in \mathbb{R}^{(K+T) \times d_{\text{llm}}}
$$

Then autoregressively decoded by the causal LLM (**MedGemma 1.5-4B-IT**).

### 7. Staged Vision-Language Alignment

![Three-phase training pipeline: contrastive grounding, projector warmup, LoRA fine-tuning](/assets/images/paper/brain3d/fig_p004_01.png)
*Figure 2: Staged alignment protocol — Phase 1 contrastive InfoNCE grounding (LLM and ViT frozen, only inflated patch embedding + projector + scalar gate trainable) → Phase 2A projector warmup with masked next-token prediction (only projector and scalar gate trainable) → Phase 2B joint projector + LoRA (r=16, alpha=32) fine-tuning of LLM attention layers.*

- **Phase 1 - Contrastive Grounding.** LLM and vision backbone frozen. Trainable: inflated patch embedding $P_{3D}$, MLP projector $\theta_{\text{proj}}$, scalar $s$. Symmetric InfoNCE on L2-normalized global visual/textual embeddings:

$$
\mathcal{L}_{P1} = \tfrac{1}{2}(\mathcal{L}_{v \to t} + \mathcal{L}_{t \to v})
$$

No prompt is prepended to the report during this phase.

- **Phase 2A - Projector Warmup.** Vision encoder and LLM frozen. Trainable: $\theta_{\text{proj}}$, $s$. Masked next-token prediction over $U = \text{Concat}(Z_{\text{vis}}, Z_{\text{txt}}, Z_Y)$; loss applied only on report tokens (visual and prompt tokens masked to −100).

- **Phase 2B - Linguistic LoRA fine-tuning.** 3D vision encoder frozen. Trainable: $\theta_{\text{proj}}$ + **LoRA adapters injected into LLM attention layers** (rank r=16, alpha=32). Same masked next-token-prediction objective as Phase 2A. The paper specifies LoRA targets "LLM attention layers" but does not enumerate Q/K/V/O modules further.

### 8. Inference

Conservative sampling: temperature T=0.1, top-p=0.9, repetition penalty 1.2, trigram blocking. Fixed canonical prompt: *"Generate a radiology report for this brain MRI FLAIR scan."*

### 9. Training infrastructure

Single NVIDIA A100 (64GB), PyTorch + HuggingFace, bfloat16 mixed precision, AdamW with linear warmup + cosine decay, effective batch size 128 via gradient accumulation, early stopping after 15 epochs without validation improvement.

## Dataset

- **Total:** N = 468 subjects, subject-level stratified split **70 / 10 / 20** train/val/test (stratified by class and lesion laterality).
- **Pathological cohort (BraTS):** 369 FLAIR volumes from **BraTS2020 training set**; structured reports (location, edema, necrosis) derived from **TextBraTS** segmentation-tag templates [Shi et al. 2025, arXiv:2506.16784]. Laterality distribution: 42.5% left, 40.7% right, 14.6% bilateral, 2.2% undefined.
- **Healthy controls:** 99 healthy brain MRIs (21.2% of dataset) from **OpenNeuro / Brainlife** sourced via the MPI-Leipzig "mind-brain-body" dataset (Babayan et al. 2019).
- **Modality coverage:** FLAIR only. The conclusion lists T1/T2/FLAIR multi-sequence as future work — the model is not validated on multi-sequence inputs.

The "reports" are not radiologist prose but auto-generated from segmentation labels via TextBraTS templates, and the healthy controls come from an unrelated young/old volunteer dataset rather than age- and scanner-matched BraTS controls — both are plausible sources of trivial healthy-vs-tumor shortcut learning.

## Experimental Results

### Main comparison (Table I, mean with 95% CI)

| Model | B-1 | B-4 | R-L | METEOR | BERTScore | CIDEr | Lat F1 | Anat F1 | **Path F1** |
|---|---|---|---|---|---|---|---|---|---|
| Med3DVLM (3D generalist, zero-shot) | 0.051 | 0.005 | 0.083 | 0.055 | 0.836 | 0.007 | 0.300 | 0.225 | 0.119 |
| MedGemma 1.5 (2D slice-based, zero-shot) | 0.245 | 0.024 | 0.189 | 0.190 | 0.859 | 0.029 | 0.526 | 0.461 | 0.413 |
| Brain3D Phase 1 (contrastive only) | 0.122 | 0.005 | 0.113 | 0.128 | 0.800 | 0.003 | 0.243 | 0.141 | 0.211 |
| Brain3D Phase 2A (projector warmup) | 0.280 | 0.099 | 0.285 | 0.250 | 0.884 | **0.504** | 0.658 | 0.503 | 0.711 |
| **Brain3D Phase 2B (projector + LoRA)** | **0.302** | 0.098 | **0.289** | **0.253** | **0.898** | 0.293 | **0.689** | **0.691** | **0.951** |

Phase 2B is best on every metric except CIDEr, where Phase 2A's longer "caption-like" outputs win 0.504 vs 0.293. The authors interpret the CIDEr drop as a desirable shift from verbose captions to terse structured reports.

### Staging observations

- Phase 1 alone is essentially incapable of generating useful reports (B-4 = 0.005, Path F1 = 0.211) — it sets up the contrastive embedding space but is not a generative checkpoint.
- Phase 2A is where descriptive fluency peaks (CIDEr 0.504, ~17x over Phase 1).
- Phase 2B trades descriptive verbosity for clinical precision (Path F1 0.711 → 0.951, +34% relative over 2A; CIDEr 0.504 → 0.293).
- The paper claims **"perfect specificity on healthy scans"** in the abstract and conclusion, but **no per-cohort breakdown, confusion matrix, or numerical specificity is reported** — only the prose claim of "negligible hallucinations on healthy scans" in IV-F.

### Interpretability and error analysis

![3D LIME attribution over SLIC supervoxels showing tumor-hemisphere concentration with diffuse contralateral activation](/assets/images/paper/brain3d/fig_p005_01.png)
*Figure 3: 3D LIME over SLIC supervoxels on a representative case — red supports the generated report, blue opposes; the tumor-bearing hemisphere dominates attribution but diffuse contralateral activation foreshadows the ~15% laterality-inversion failure mode.*

The dominant residual failure is **laterality inversion** (~15% of pathological cases). Diffuse gliomas suffer from under-reported peripheral infiltration. Under uncertainty, the model regresses toward frequent anatomical phrases ("left parietal and occipital lobes") — a frequency-bias artifact of the small templated training corpus.

![Generated report comparison across Brain3D, MedGemma 1.5, and Med3DVLM](/assets/images/paper/brain3d/fig_p006_01.png)
*Figure 4: Qualitative report comparison on a representative test case — Brain3D recovers laterality and pathology categories, MedGemma 1.5 (2D slice-based) hallucinates the wrong hemisphere, and Med3DVLM returns a generic "normal" template (note: this crop is narrow; verify the figure shows the qualitative report table).*

## Limitations

**Author-acknowledged.**

- Laterality inversion (~15% of pathological cases) is the dominant error.
- Diffuse glioma peripheral infiltration is under-reported.
- Distributional bias toward frequent anatomical phrases under uncertainty.
- LIME interpretability is "exploratory and not intended as a definitive attribution study."
- Future work: anatomically-informed positional embeddings, DPO/RLHF for spatial accuracy, multi-sequence MRI (T1/T2/FLAIR).

**Not addressed by the authors (reviewer-identified).**

- **Reports are TextBraTS templates, not radiologist prose.** TextBraTS auto-generates "reports" from segmentation labels — the model is learning to caption segmentation tags, not real clinical narratives. The authors do not flag this caveat. The 0.951 Path F1 should be read as "matches a rule-based extractor over four descriptors on tag-derived reports," not as a clinical deployability number.
- **Confounded 2D-vs-3D ablation.** The +130% gap vs MedGemma 1.5 conflates the 3D encoder with three stages of in-domain training on BraTS+TextBraTS. MedGemma is evaluated zero-shot and slice-stacked; no fair comparison fine-tunes the 2D baseline through the same Phase 2A/2B pipeline.
- **No inflation-vs-random-3D-init ablation.** Inflated weights vs randomly-initialized 3D encoder vs from-scratch 3D pretraining is not shown — so the value of *inflation specifically* (vs simply having any 3D encoder) is asserted, not measured.
- **No prior brain-MRI report-generation baselines.** Comparisons are only against MedGemma 1.5 (2D generalist) and Med3DVLM (3D generalist, CT-centric). Brain-specific report-generation work (R2GenGPT, CNN-LSTM systems) is cited in Related Work but not benchmarked.
- **"Perfect specificity on healthy" is asserted without a number.** No per-cohort table, no false-positive rate, no confusion matrix — only prose.
- **No external test set.** All numbers come from a single 20% in-distribution split of BraTS2020+Brainlife (test n approx. 94, healthy subset n approx. 20). No held-out scanner/site, no other BraTS years (2021/2023, UPenn-GBM, UCSF-PDGM, BraTS-Africa), no cross-validation, no multi-seed variance.
- **Healthy cohort source mismatch.** Controls come from MPI-Leipzig "mind-brain-body" volunteers — a young/old healthy cohort unmatched in age, scanner, and acquisition to BraTS2020, a known source of trivial healthy-vs-diseased separability.
- **No LoRA target-module ablation.** "LoRA on LLM attention layers" with r=16, alpha=32 — no enumeration of Q/K/V/O targets, no rank sweep, no comparison to MLP-targeting LoRA.
- **+130% framing.** Arithmetically correct (0.951 / 0.413 ≈ 2.30) but rhetorically inflated for a metric bounded in [0, 1].

## Why It Matters for Medical AI

The staging protocol (contrastive grounding → projector warmup → LoRA on LLM attention) is the genuinely transferable contribution — it gives a clean recipe for adapting any pretrained 2D medical encoder to a volumetric modality without retraining a foundation model from scratch. Inflation as a *cheap* path from 2D to 3D in medical imaging is also a useful demonstration, even if the inflation-vs-random-init ablation is missing. For practitioners building report-generation systems on small in-house volumetric datasets, the recipe is reusable: take MedSigLIP-class encoder, inflate, train a projector to match the LLM, then LoRA the LLM. What this paper does *not* demonstrate — and should not be cited as having demonstrated — is that Brain3D is clinically deployable for brain-MRI reporting. The training target is templated, the evaluation is in-distribution, the "specificity" claim is unquantified, and no real radiologist prose appears anywhere in the pipeline.

## References

- arXiv: [Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D](https://arxiv.org/abs/2602.22098)
- Code: [PRAISELab-PicusLab/BrainGemma3D](https://github.com/PRAISELab-PicusLab/BrainGemma3D)
- I3D inflation: Carreira & Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset", CVPR 2017.
- MedGemma / MedSigLIP: Google Health AI, MedGemma model family.
- BraTS2020: Menze et al. 2015, "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)".
- TextBraTS: Shi et al. 2025, arXiv:2506.16784.
- Healthy controls: Babayan et al. 2019, MPI-Leipzig "mind-brain-body" dataset (OpenNeuro).
- LoRA: Hu et al. 2022, "LoRA: Low-Rank Adaptation of Large Language Models".
