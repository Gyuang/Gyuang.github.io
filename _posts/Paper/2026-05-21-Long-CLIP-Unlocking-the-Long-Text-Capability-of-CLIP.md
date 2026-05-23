---
title: "Long-CLIP: Unlocking the Long-Text Capability of CLIP"
excerpt: "Knowledge-preserved positional-embedding stretching plus a primary-component matching loss lifts CLIP's text cap from 77 to 248 tokens with flat zero-shot performance — but the headline 25% long-text gain is partly inflated by a self-built eval set."
categories: [Paper, VLM-Alignment, Multimodal-Alignment, LLM]
permalink: /paper/long-clip/
tags:
  - Long-CLIP
  - CLIP
  - Positional-Embedding
  - Contrastive-Learning
  - Vision-Language
  - PCA
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- CLIP's text branch is hard-capped at 77 tokens and **empirically effective only up to ~20 tokens**; Long-CLIP fine-tunes CLIP on ~1M long captions with two ideas: (1) a *knowledge-preserved stretch* (KPS) of the positional embedding that lifts the cap to **248 tokens**, and (2) a *primary-component matching* (PCM) loss that re-aligns a coarse, PCA-reduced image feature to the short caption.
- Headline numbers (ViT-L/14): Urban-200 long-text I2T R@1 **47.0 -> 81.5**, COCO short-text T2I R@1 **35.4 -> 46.3**, ImageNet zero-shot **75.5 -> 73.5** (versus 58.4 for naive fine-tuning) — and the model trains in **~0.25 GPU-hour x 8 GPUs** on ShareGPT4V.
- Long-CLIP is engineered as a *drop-in* replacement for CLIP's text encoder, including in Stable Diffusion 1.5 and SDXL. The SD result is **qualitative only** (no FID, CLIPScore, or T2I-CompBench), and the "+25% long-text R@1" headline collapses a +34.5-pt gap on the authors' own Urban-200 with a +11.6-pt gap on ShareGPT4V held-out.

## Motivation

CLIP underwrites zero-shot classification, retrieval, and the text branch of every CLIP-conditioned diffusion model (SD-v1.5, SDXL, DALL-E 2). But its absolute positional embedding is hard-capped at 77 BPE tokens, and the authors show the *effective* length is closer to 20: web-scraped (image, caption) pairs are short, so position embeddings beyond ~20 never see enough supervision. Two failure modes follow:

- **Fine-grained retrieval breaks** when two near-identical images differ only in adverbial or relational detail (their "the lemon is purple" counterexample).
- **Long-prompt T2I generation breaks** because SD's text encoder truncates everything past position 77.

Pre-training a new text encoder from scratch is expensive. Naive linear interpolation + fine-tuning destroys short-text alignment (ImageNet drops 13.1 pts, COCO T2I R@1 drops 14.4 pts). Long-CLIP is engineered so the existing CLIP ecosystem swaps in the new encoder *without retraining the downstream model*.

![Long-CLIP teaser: fine-grained retrieval and long-prompt generation](/assets/images/paper/long-clip/fig_p002_01.png)
*Figure 1: Long-CLIP captures fine-grained attributes that CLIP misses on the same image, lifting both retrieval and text-to-image generation.*

![Stable Diffusion conditioned on Long-CLIP](/assets/images/paper/long-clip/fig_p002_05.png)
*Figure 2: Stable Diffusion conditioned on Long-CLIP retains attributes (colour, position) that are lost when the prompt is truncated to 77 tokens.*

## Core Innovation

- **Diagnose effective length empirically.** Sweep the truncation length on Urban-200 and observe that CLIP's R@1 plateaus past ~20 tokens — therefore "20" is the well-trained prefix to preserve.
- **Knowledge-Preserved Stretching (KPS).** Freeze positions 1-20, only interpolate positions 21-77 with a larger ratio lambda_2 = 4. Maximum length becomes 20 + 57 * 4 = **248 tokens**.
- **Primary Component Matching (PCM).** Decompose the image feature with PCA (top-32 eigenvectors), keep only the principal components, and contrastively align this *coarse* feature to the short caption — while the *fine-grained* full feature aligns to the long caption.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CLIP's effective text length is ~20 tokens | Fig. 2a R@1-vs-length plateau | Urban-200 (self-built) | ⭐⭐ single-dataset, self-built; would be stronger with the same probe on COCO/Flickr/ShareGPT4V |
| C2 | KPS-only beats fixed-ratio interpolation on short retrieval / classification | Tab. 4 row 3 vs row 1 (ImageNet 55.1 -> 65.6; COCO T2I R@5 43.4 -> 64.3) | ImageNet, CIFAR-100, COCO, Flickr | ⭐⭐⭐ clean ablation, large gaps |
| C3 | PCM is needed on top of KPS to fully recover ImageNet / Flickr | Tab. 4 row 4 vs row 3 (66.8 vs 65.6; 71.4 vs 70.4) | same | ⭐⭐ gaps are small (1.2 / 1.0 pts), no variance, single seed |
| C4 | +25% R@1 on long-text image retrieval vs CLIP | Tab. 1 Urban-200 L/14: 47.0 -> 81.5 (+34.5); ShareGPT4V L/14: 84.0 -> 95.6 (+11.6) | ShareGPT4V (intra-distribution), Urban-200 (self-built) | ⭐⭐ **inflated headline** — Urban actually +34, ShareGPT only +11-14; both eval sets are not arm's-length from training |
| C5 | +6% on short-text image retrieval | Tab. 2: COCO L/14 T2I R@1 35.4 -> 46.3 (+10.9); Flickr L/14 T2I R@1 28.0 -> 41.2 (+13.2) | COCO, Flickr30k (full 30k) | ⭐⭐⭐ genuine gain; note Flickr uses the 30k version, not the standard 1k test split |
| C6 | No decay on zero-shot classification | Tab. 3 avg 66.12 -> 66.14 (B/16); 69.92 -> 69.78 (L/14) | ImageNet + 4 others | ⭐⭐⭐ well-supported; the 2-pt ImageNet/V2 drop on L/14 is honestly disclosed |
| C7 | Plug-and-play replacement in SD without further training | Fig. 6 qualitative; supp. Fig. 1 SDXL | - | ⭐ **purely qualitative** — no FID, CLIPScore, T2I-CompBench, no user study |
| C8 | PCM beats alternative short-text-preserving strategies | Tab. 5 | 5 metrics | ⭐⭐ margins of 0.3-4.6 pts; "bounded encoding" actually wins Urban T2I 80.0 vs 79.0 (paper does not dwell on this) |
| C9 | Strategy generalises to DeCLIP | Supp. Tab. 3: Urban I2T 25.5 -> 51.0 | DeCLIP B/32 | ⭐⭐ only one other backbone — no OpenCLIP / SigLIP / EVA-CLIP |

**Honest read.** The two strongest claims are **C2 (KPS works)** and **C5/C6 (short retrieval up, zero-shot flat)** — multi-dataset, two backbones, clean ablation. The weakest are **C4 (the "25%" headline glosses over a +34.5-pt gap on a self-built eval set and a +11.6-pt gap on an intra-distribution held-out)** and **C7 (the entire plug-and-play SD pitch is qualitative)**. No variance / multi-seed runs are reported anywhere. There is one minor internal inconsistency: **main text says batch 2048, supplementary Tab. 4 says 1024**. The CIFAR/ImageNet drop on L/14 (75.5 -> 73.5 ImageNet, 69.9 -> 67.9 V2) is genuine and partially papered over by averaging with CIFAR-100, where Long-CLIP gains. The supplementary "competitors" (X-VLM, X^2-VLM, PTP-BLIP) are fine-grained-region methods, **not contemporaneous long-text-CLIP variants** (DCI-CLIP, FILIP, AltCLIP, EVA-CLIP-L are absent) — convenient strawmen.

## Method & Architecture

![Long-CLIP training pipeline](/assets/images/paper/long-clip/fig_p007_03.png)
*Figure 3: Long-CLIP training — the fine-grained image feature aligns with the long caption; a PCA-reduced coarse feature aligns with the short summary. Positional embeddings are stretched with KPS (freeze positions <=20, interpolate the rest with ratio 4).*

### Step 1. Diagnose the effective length

![CLIP's R@1 saturates at ~20 tokens](/assets/images/paper/long-clip/fig_p005_01.png)
*Figure 4: CLIP's R@1 saturates at ~20 tokens (left) and fails on relational attributes (right) — the two failures Long-CLIP targets.*

The truncation sweep on Urban-200 (Fig. 2a) shows R@1 plateauing past ~20 tokens. The authors take 20 as the empirical "well-trained prefix length." This anchors the KPS rule below — and is also Claim C1's only piece of evidence, which is why I rate it ⭐⭐.

### Step 2. Knowledge-Preserved Stretching (KPS) of positional embeddings

Vanilla linear interpolation with a single ratio lambda_1 damages the well-trained low positions:

$$
\mathrm{PE}^*(\mathrm{pos}) = (1-\alpha)\cdot\mathrm{PE}(\lfloor \mathrm{pos}/\lambda_1 \rfloor) + \alpha\cdot\mathrm{PE}(\lceil \mathrm{pos}/\lambda_1 \rceil), \quad \alpha = (\mathrm{pos} \bmod \lambda_1)/\lambda_1
$$

KPS replaces this with a piecewise rule that **freezes positions <=20** and interpolates only positions 21-77 with a larger ratio lambda_2 = 4, yielding a maximum input length of 20 + 57 * 4 = 248 tokens:

$$
\mathrm{PE}^*(\mathrm{pos}) =
\begin{cases}
\mathrm{PE}(\mathrm{pos}), & \mathrm{pos} \le 20 \\
(1-\alpha)\cdot\mathrm{PE}(\lfloor \mathrm{pos}/\lambda_2 \rfloor) + \alpha\cdot\mathrm{PE}(\lceil \mathrm{pos}/\lambda_2 \rceil), & \mathrm{otherwise}
\end{cases}
$$

Ablation (Tab. 4) shows KPS alone lifts ImageNet from 55.1 -> 65.6 and COCO T2I R@5 from 43.4 -> 64.3 versus naive fine-tuning.

### Step 3. Primary Component Matching (PCM)

Fine-tuning only with (long caption, image) pushes the image encoder to encode *every* detail into one vector, blurring the short-caption alignment. PCM adds a second contrastive objective on a *coarse* version of the image feature:

$$
(v_1, i_1), \ldots, (v_n, i_n) = F(I_{\mathrm{fine} }) \quad \text{(decomposition)}
$$

$$
I_{\mathrm{coarse} } = F^{-1}\big(E(F(I_{\mathrm{fine} }))\big)
$$

Concretely **F is the eigendecomposition of the (batch) covariance** of image features, **E keeps the top-32 eigenvectors** by eigenvalue, and F^-1 is the linear projection back. The total loss is the symmetric InfoNCE between (a) I_fine <-> T_long and (b) I_coarse <-> T_short.

![Long-CLIP pseudo-code](/assets/images/paper/long-clip/fig_p008_01.png)
*Figure 5: NumPy-style pseudo-code for the joint fine-grained / coarse-grained contrastive objective.*

### Step 4. Training

1 epoch on ShareGPT4V (~1M (image, long-caption) pairs; avg 826 chars), AdamW LR 1e-4, weight decay 1e-2, 200 warm-up iterations, ~0.25 h on 8 GPUs. **Main text says batch 2048; supplementary Tab. 4 says 1024 — flagged as a minor internal inconsistency in the paper.** Backbones: CLIP ViT-B/16 and ViT-L/14; also DeCLIP ViT-B/32 in the supplement.

### Step 5. Inference and plug-and-play

Replace the CLIP text encoder in SD-v1.5 and SDXL with no further training. For SDXL the OpenCLIP-bigG branch only receives KPS (no PCM) because retraining it would be prohibitive.

## Dataset

![Urban-200 sample](/assets/images/paper/long-clip/fig_p010_02.png)
*Figure 6: Urban-200 sample — two visually near-identical street scenes whose long captions differ only in marked attributes. GPT-4V-captioned, manually verified.*

| Role | Dataset | Size | Notes |
|---|---|---|---|
| Training | ShareGPT4V | ~1M (image, long-caption) pairs; avg 826 chars | Captions generated by GPT-4V on diverse web images. 1k pairs held out as eval. |
| Long retrieval (held-out) | ShareGPT4V held-out 1k | 1,000 | Same distribution as training — **risk of intra-distribution optimism**. |
| Long retrieval (curated) | **Urban-200 (new)** | 200 visually similar urban scenes from Visual Genome, captions by GPT-4V, manually verified | Avg 101 words. **Built by the same team; the "+25%" headline rests on this set.** |
| Long retrieval (supp.) | **Urban-1k (new)** | 1,000 images, ~107 words/caption | Released on HF: BeichenZhang/Urban1k. |
| Short retrieval | COCO2017 (5k val), Flickr30k (full 30k) | standard | Authors use the **whole Flickr30k** rather than the conventional 1k test split — hurts apples-to-apples comparisons vs prior literature. |
| Zero-shot classification | ImageNet-1k, ImageNet-V2, ImageNet-O, CIFAR-10, CIFAR-100 | standard | 80-prompt CLIP template averaging. |

**Biases to flag.** (1) ShareGPT4V captions are GPT-4V-generated — Long-CLIP learns to match a particular VLM's caption style. (2) Urban-200/1k are GPT-4V-captioned *and* curated to be hard — the headline 25% gap is partly a property of an eval set the same team built. (3) No medical, scientific, document, or non-Western-urban distribution is tested.

## Experimental Results

### Long-caption retrieval (Tab. 1, R@1)

| Backbone | Method | ShareGPT4V I2T | ShareGPT4V T2I | Urban-200 I2T | Urban-200 T2I |
|---|---|---:|---:|---:|---:|
| B/16 | CLIP | 78.2 | 79.6 | 46.5 | 46.0 |
| B/16 | Direct FT | 94.1 | 93.6 | 78.5 | 78.0 |
| **B/16** | **Long-CLIP** | **94.6** | 93.3 | **79.5** | **79.0** |
| L/14 | CLIP | 81.8 | 84.0 | 47.0 | 47.0 |
| L/14 | Direct FT | 95.3 | 95.4 | 78.0 | 76.5 |
| **L/14** | **Long-CLIP** | **95.8** | **95.6** | **81.5** | **81.5** |

### Short-caption retrieval (Tab. 2, R@1)

| Backbone | Method | COCO I2T | COCO T2I | Flickr I2T | Flickr T2I |
|---|---|---:|---:|---:|---:|
| B/16 | CLIP | 51.8 | 32.7 | 44.1 | 24.7 |
| B/16 | Direct FT | 37.4 | 21.8 | 25.7 | 17.9 |
| **B/16** | **Long-CLIP** | **57.6** | **40.4** | **46.8** | **34.1** |
| L/14 | CLIP | 56.1 | 35.4 | 48.5 | 28.0 |
| L/14 | Direct FT | 37.9 | 23.1 | 26.0 | 17.9 |
| **L/14** | **Long-CLIP** | **62.8** | **46.3** | **53.4** | **41.2** |

### Zero-shot classification (Tab. 3, top-1)

| Backbone | Method | ImageNet | ImageNet-O | ImageNet-V2 | CIFAR-10 | CIFAR-100 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| B/16 | CLIP | 68.4 | 42.2 | 61.9 | 90.8 | 67.3 | 66.12 |
| B/16 | Direct FT | 55.1 | 31.7 | 44.8 | 83.9 | 59.2 | 54.94 |
| **B/16** | **Long-CLIP** | 66.8 | **42.7** | 61.2 | 90.7 | **69.3** | **66.14** |
| L/14 | CLIP | **75.5** | 31.9 | **69.9** | **95.5** | 76.8 | **69.92** |
| L/14 | Direct FT | 58.4 | 29.2 | 52.7 | 92.7 | 68.7 | 60.30 |
| **L/14** | **Long-CLIP** | 73.5 | **33.7** | 67.9 | 95.3 | **78.5** | 69.78 |

**Ablation (Tab. 4).** KPS and PCM are individually necessary. KPS alone recovers most of the short-caption performance (ImageNet 55.1 -> 65.6, COCO T2I R@5 43.4 -> 64.3); PCM alone is much weaker (ImageNet 58.8). Combined: 66.8 / 69.3 / 65.8 / 71.4 / 79.0.

**Alternative strategies (Tab. 5).** Against "undistinguished" (single feature aligned to both texts), "mixed-length text" (10% short-text swap), and "bounded encoding" (SmoothL1 to frozen CLIP short-text feature), PCM wins 4/5 metrics — and **ties / loses on Urban-200 T2I (PCM 79.0 vs bounded encoding 80.0)**, a detail the paper does not dwell on.

**Plug-and-play generation.** SD-v1.5 and SDXL with the Long-CLIP text encoder swap in with no retraining; for prompts >77 tokens SD now generates content from the truncated tail (**qualitative only — no FID, CLIPScore, T2I-CompBench**).

![Long-prompt generation comparison](/assets/images/paper/long-clip/fig_p013_03.png)
*Figure 7: On a >77-token prompt, only Long-CLIP retains the late-appearing attributes (swan, weeping willows).*

![Long-CLIP swapped into SDXL](/assets/images/paper/long-clip/fig_p020_01.png)
*Figure 8: Long-CLIP swapped into SDXL — prompts beyond 77 tokens regain the truncated attributes (orange) with little quality loss. Qualitative only.*

**Generalisability (supp. Tab. 3).** Applied to DeCLIP ViT-B/32, Long-DeCLIP improves Urban I2T R@1 25.5 -> 51.0 with no loss on average classification (63.6 -> 63.9).

## Limitations

Authors' own:

- Hard upper bound stays — 248 tokens is just a bigger constant. RoPE-style relative encodings would not have this ceiling.
- Only 1M (image, long-text) pairs were used; the scaling behaviour is unexplored.

Not addressed by the authors (and the more important set):

- **No statistical significance, no seed variance, no confidence intervals.** Some headline gaps are 1-2 pts on a single seed.
- **No comparison against contemporaneous long-text CLIP variants** (DCI-CLIP, FILIP, AltCLIP, EVA-CLIP-L, BLIP-2 text encoders). The supplementary "competitors" are fine-grained-region methods (X-VLM, X^2-VLM, PTP-BLIP) — not the right reference class for long-text retrieval.
- **PCM keeps top-32 eigenvectors with no hyperparameter ablation.**
- **Behaviour beyond 248 tokens is uncharacterised** — graceful truncation? hard cliff?
- **Plug-and-play diffusion has no quantitative metric.** Given the paper pitches itself as a CLIP drop-in, the absence of FID / CLIPScore / T2I-CompBench / user study is a real omission.
- **Domain shift is untested** — medical reports, legal text, code, multilingual long captions. Likely degrades, since training was English ShareGPT4V only.
- **The "effective length = 20" claim hinges on one self-built dataset.** The same probe on COCO/Flickr/LAION-COCO/CC12M-LC is missing.
- **Internal inconsistency.** Main text reports batch 2048; supplementary Tab. 4 reports 1024. Small, but it propagates into any replication study.

## Why It Matters for Medical AI

The same length / effective-length pathology is exactly why medical CLIPs (BiomedCLIP, PMC-CLIP) underperform on free-text radiology reports that routinely exceed 77 BPE tokens. BiomedCLIP already extended context 77 -> 256 by **retraining from scratch on PMC-15M**; Long-CLIP suggests a much cheaper path — KPS + PCM on top of an existing medical CLIP — with the bulk of the short-text alignment preserved. The PCM idea (coarse PCA projection aligned to a short summary, full feature aligned to a long report) maps naturally onto the "findings + impression" split of radiology reports. The honest caveat: none of this is tested in the paper, ShareGPT4V is GPT-4V-style web captions (not medical prose), and the Urban-200 caution applies double in medicine — any evaluation set you build yourself is by construction the one your method wins on.

## References

- Paper: [Long-CLIP: Unlocking the Long-Text Capability of CLIP, arXiv:2403.15378](https://arxiv.org/abs/2403.15378)
- Code: [github.com/beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP)
- Urban-1k benchmark: [huggingface.co/datasets/BeichenZhang/Urban1k](https://huggingface.co/datasets/BeichenZhang/Urban1k)
- Related: CLIP (Radford et al. 2021), DeCLIP (Li et al. 2021), ShareGPT4V (Chen et al. 2023), Stable Diffusion (Rombach et al. 2022), SDXL (Podell et al. 2023)
