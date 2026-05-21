---
title: "Sigmoid Loss for Language Image Pre-Training"
excerpt: "Replacing CLIP's softmax-over-batch with a per-pair sigmoid loss removes the global all-gather, reaches 83.2% ImageNet 0-shot with SO-400M, and shows contrastive performance saturates at 32k batch and degrades past 98k."
categories: [Paper, VLM-Alignment, Multimodal-Alignment, LLM]
permalink: /paper/siglip/
tags:
  - SigLIP
  - SigLiT
  - Sigmoid-Loss
  - CLIP
  - Contrastive-Learning
  - Vision-Language
  - Foundation-Model
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- Replace CLIP's softmax InfoNCE — which forces a global all-gather over the full B×B similarity matrix — with a **per-pair sigmoid binary classification loss** plus a learnable bias `b` (init −10) that absorbs the heavy negative-prior imbalance. Each (image, text) pair becomes an independent positive/negative judgement, removing the need for any global partition function across the batch.
- A chunked device-permutation implementation replaces two all-gathers with D collective permutes, so SigLiT B/8 reaches **79.7% ImageNet 0-shot in 1 day on 4 TPU-v4 chips** and SigLiT g/14 hits **84.5% in 2 days**. Full SigLIP SO-400M/14 (729 patches) reaches **83.2% ImageNet 0-shot, 82.9% ObjectNet, 70.2/52.0 COCO I→T/T→I R@1** — beating EVA-CLIP-E (5B params, 12.5× the SO-400M parameter count) on every metric.
- The 1M-batch run finally settles the "bigger is better" dogma: contrastive performance **peaks around 32k and degrades past ~98k** (sigmoid 73.4 → 73.0 → 71.6 at 32k / 98k / 307k for 9B-seen SigLIP B/16; mSigLIP avg retrieval 34.9 → 32.7 from 32k → 240k).

## Motivation

Softmax-CLIP's row/column normalization requires every device in a data-parallel job to see the full set of embeddings, demanding an all-gather and a |B|² similarity matrix that becomes the dominant cost as batches scale. Naïve softmax is also numerically unstable — implementations subtract per-row max, which is a *second* full-batch pass. The community had assumed huge batches (≥64k) were essential for contrastive quality, locking image-text pre-training out for groups without hundreds of accelerators.

The paper asks whether a loss that decomposes additively over pairs can (a) match softmax quality, (b) free up memory for larger effective batches on fewer chips, and (c) let the authors actually probe the batch-size scaling curve up to 1M to test the large-batch dogma — three questions softmax cannot cleanly answer because the optimization, the memory, and the engineering trick are tangled together.

## Core Innovation

- **Per-pair sigmoid loss with learnable bias.** $\mathcal{L} = -\frac{1}{|B|}\sum_i\sum_j \log\sigma(z_{ij}\cdot(t\cdot x_i\cdot y_j + b))$ with $z_{ij}=+1$ for matched pairs and $-1$ otherwise. The −10 init on `b` starts the model near the negative prior, preventing the |B|²−|B| negatives from producing a giant initial gradient (Table 4: removing `b` drops INet 0-shot 63.0 → 62.0; pathological init `b=0, t'=log 1` collapses to 53.7%).
- **Chunked device-permutation kernel.** Each device computes its local b×b block, then permutes its text chunk to the next device and accumulates b×b negative blocks; after D−1 permutes every pair has been scored. Memory peak per device is b² (not |B|²), all-gathers replaced by D collective-permutes. This is the load-bearing engineering trick that enables a 1M-batch run.
- **β₂=0.95 stabilization.** Default Adam/Adafactor `β₂=0.999` produces gradient-norm spikes at large B. Lowering to `β₂=0.95` damps the second-moment memory so the optimizer recovers within a few steps — Fig. 5 shows smooth curves at 0.95 vs catastrophic spikes at 0.999.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Sigmoid loss outperforms softmax at small batches (<16k) | Fig. 2 left/middle; Table 5 (4k: 65.3 vs 63.8; 8k: 68.6 vs 66.6; 16k: 72.3 vs 71.7); Table 7 (5-seed σ ±0.1–0.2pp) | INet 0-shot (SigLiT, SigLIP) | ⭐⭐⭐ — multi-seed across SigLiT/SigLIP/mSigLIP, two pre-training setups, consistent monotone gap of 1.6–2.0pp at B∈{4k,8k,16k} |
| C2 | Performance saturates around 32k batch and *degrades* beyond ~98k | Fig. 2 (all 3 panels); Table 5 (sigmoid 9B: 32k=73.4, 98k=73.0, 307k=71.6); Table 8 (SigLiT 1M=84.7 ties 256k=84.7) | INet 0-shot, XM3600 36-lang | ⭐⭐⭐ — repeated across SigLiT/SigLIP/mSigLIP including 1M batch; unique smoking gun in the literature |
| C3 | Sigmoid is more memory-efficient → larger B on fewer chips | 4 TPU-v4 fits B=4096 sigmoid vs 2048 softmax for Base SigLIP | qualitative resource note in §4.2 | ⭐⭐ — anecdotal: one configuration, no peak-memory profiling table, no comparison vs gradient-checkpointed softmax |
| C4 | SigLiT 79.7% INet 0-shot in 1 day on 4 TPU-v4 chips | Table 1 row 1 (B/8 frozen + 12-layer L text) | INet | ⭐⭐ — single run, but reproducible (frozen public B/8 ckpt, public LiT data, recipe in App. A) |
| C5 | SigLIP SO-400M is SOTA at its parameter count, beats EVA-CLIP-E 5B | Table 3: 83.2 INet vs EVA-CLIP-E (5B params) 82.0 | INet/v2/ReaL/ObjectNet/COCO | ⭐⭐⭐ — beats a 12.5× larger model across 5 of 6 metrics, ties the 6th |
| C6 | Sigmoid is more robust to label/data noise | Fig. 7, 5 corruption regimes, p up to 0.5 | INet 0-shot, M/16 model | ⭐⭐ — only M/16, single dataset; consistent gap but not stress-tested across model scales |
| C7 | The negative imbalance (~16k:1) is not the problem; hard negatives matter | Fig. 6 masking study | SigLiT B=16k, 900M | ⭐⭐ — single batch size; "matched-pairs hard" only marginally exceeds full negatives |
| C8 | Disabling weight decay on pre-trained encoder is necessary for fine-tune SigLIP | Fig. 4 top/bottom (10-shot linear shows fine-tune-with-WD ≈ from-scratch) | INet 0-shot + 10-shot linear | ⭐⭐ — strong qualitative curves, no table of numbers |
| C9 | β₂=0.95 stabilizes large-batch transformer training | Fig. 5 loss/gradnorm/Δw curves | qualitative | ⭐ — one figure, no quantitative downstream comparison; relies on prior MAE/iGPT results |
| C10 | Bias term b=−10 init is critical | Table 4: 63.0 vs 62.0 (no bias), 53.7 (b=0, t'=log 1) | INet/Pet/CIFAR100 0-shot, B=8k 900M | ⭐⭐ — three datasets, but only one batch size and one schedule |

**Honest read.** The two core scientific claims (C1: sigmoid wins at small B; C2: scaling saturates at 32k) are extremely well-supported — three independent setups (SigLiT, SigLIP, mSigLIP) trace the same curve, and the 1M-batch run is unique evidence in the literature. C5 (SO-400M 83.2 beats EVA-CLIP-E 5B at 82.0 on INet) is also clean. What's weaker:

- **No variance bars on the headline Fig. 2 curves.** Table 7 reports σ≈0.1–0.2pp at one batch size only; given the 32k→98k drop is just 0.4pp (sigmoid 73.4→73.0), some "saturation curve" wiggle is within noise.
- **C3 (memory) is anecdotal** — one config, no profiling, no comparison vs gradient-checkpointed softmax or activation-recompute baselines.
- **C6 (noise robustness) is extrapolated from one M/16 model.** The supervised-classification prior for sigmoid+noise (Beyer et al. 2020) is real, but contrastive-with-batch-misalignment is a different setting and deserved more sizes.
- **No external/medical evaluation.** All eval is INet/COCO/XM3600. No domain-shift benchmark beyond ObjectNet, no medical or satellite or scientific transfer.
- **Reproducibility — WebLI is private.** Models are released on `big_vision`, but anyone wanting to *retrain* must use a substitute dataset; LiT is the only public corpus among the pre-training data.
- **Modality gap never measured.** The paper does not check whether sigmoid-trained image and text embeddings still occupy disjoint cones in feature space (the Liang et al. 2022 phenomenon). Pairwise sigmoid removes the *normalization* coupling between modalities but the L2-normalized dot-product geometry is unchanged, so a modality gap of similar magnitude to CLIP's is the expected null hypothesis — yet not verified here. This is the cleanest open follow-up.

## Method & Architecture

![SigLIP loss as 6 lines of pseudocode](/assets/images/paper/siglip/page_002.png)
*Algorithm 1: the entire SigLIP loss in 6 lines — L2-normalize image and text embeddings, dot-product with learned temperature `t` and bias `b`, log-sigmoid against ±1 labels. The whole contribution fits on a screen.*

### 1. Embedding setup

Image tower $f(\cdot)$ (ViT-B/L/SO-400M) and text tower $g(\cdot)$ (transformer of matched size); both produce L2-normalized $d$-dim vectors $x_i = f(I_i)/\|f(I_i)\|$, $y_i = g(T_i)/\|g(T_i)\|$.

### 2. Sigmoid loss vs softmax (Eq. 1–2)

$$
\mathcal{L}_{\text{sig}} = -\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{j=1}^{|B|} \log\sigma\!\left(z_{ij}\cdot(t\cdot x_i\cdot y_j + b)\right)
$$

with $z_{ij}=+1$ for matched pairs ($i=j$) and $-1$ otherwise. $t = e^{t'}$ is a learnable temperature; $b$ is a learnable bias. **Initialization that matters:** $t' = \log 10$ (so $t\approx 10$), $b = -10$.

The contrast with softmax is structural, not just numerical:

- **Softmax-CLIP** requires the global partition function over the batch — once per row, once per column — so each device must materialize the full |B|² matrix and run an all-gather. Implementations subtract the per-row max for stability, doubling the full-batch passes.
- **Sigmoid** drops cross-pair normalization. Each $(i,j)$ becomes an independent binary decision. This collapses to a chunked b² memory footprint and D collective-permutes — no global statistics, no max-subtract, no all-gather.

### 3. Chunked distributed implementation

With per-device batch $b = |B|/D$, device $d_i$ holds $b$ images + $b$ texts. Each device computes its local b×b block (positives + b−1 local negatives), then permutes its text chunk to the next device and accumulates b×b negative blocks; after D−1 permutes every pair has been scored. Memory peak per device is $b^2$ (not $|B|^2$); all-gathers are replaced by D collective-permutes (faster than 2 all-gathers across D devices). This is the engineering pivot that turns the loss into a deployable training recipe.

### 4. SigLiT vs SigLIP vs mSigLIP

- **SigLiT** uses a *frozen* pre-trained image tower (LiT recipe) and trains only the text tower. With precomputed image embeddings, compute is near-trivial — that is how 4 TPU-v4 chips suffice for a 1-day, 79.7% INet run.
- **SigLIP** trains both towers; the "fine-tune w/o enc.wd" recipe disables weight decay on the pre-trained vision encoder, which is the only setting that doesn't degrade visual quality (Fig. 4: 10-shot linear stays high vs. fine-tune-with-WD which drops to from-scratch level).
- **mSigLIP** runs the same loss on full multilingual WebLI; a 250k-token vocabulary with a K=96 bottleneck projection (N×K · K×W instead of N×W) keeps embedding tables tractable, costs ~0.5pp.

### 5. Optimizer recipe

ScalingViT-Adafactor by default; LION used only for the 4-chip SigLiT 1-day run (LR 1e−4 peak, WD 1e−7, 6.5k step warmup, cosine to 0 over 65k steps at B=32k). Default LR=1e−3, WD=1e−4 is best or near-best across B∈{8k,16k,32k} (App. C, Tables 6–7).

## Experimental Results

### Main comparison (Table 3, "overtraining" recipe, 40B examples seen)

| Method | Backbone | Patches | INet val | INet v2 | INet ReaL | ObjectNet | COCO I→T | COCO T→I |
|---|---|---|---|---|---|---|---|---|
| CLIP | B | 196 | 68.3 | 61.9 | – | 55.3 | 52.4 | 33.1 |
| OpenCLIP | B | 196 | 70.2 | 62.3 | – | 56.0 | 59.4 | 42.3 |
| EVA-CLIP | B | 196 | 74.7 | 67.0 | – | 62.3 | 58.7 | 42.2 |
| **SigLIP** | **B** | **196** | **76.2** | **69.6** | **82.8** | **70.7** | **64.4** | **47.2** |
| SigLIP | B | 1024 | 79.2 | 73.0 | 84.9 | 74.7 | 67.6 | 50.4 |
| CLIP | L | 256 | 75.5 | 69.0 | – | 69.9 | 56.3 | 36.5 |
| EVA-CLIP | L | 256 | 79.8 | 72.9 | – | 75.3 | 63.7 | 47.5 |
| **SigLIP** | **L** | **256** | **80.5** | **74.2** | **85.9** | **77.9** | **69.5** | **51.1** |
| SigLIP | L | 576 | 82.1 | 75.9 | 87.0 | 81.0 | 70.6 | 52.7 |
| OpenCLIP | G (2B) | 256 | 80.1 | 73.6 | – | 73.0 | 67.3 | 51.4 |
| EVA-CLIP | E (5B) | 256 | 82.0 | 75.7 | – | 79.6 | 68.8 | 51.1 |
| **SigLIP** | **SO (400M)** | **729** | **83.2** | **77.2** | **87.5** | **82.9** | **70.2** | **52.0** |

SigLIP SO-400M (400M params) beats EVA-CLIP-E (5B params, 12.5× larger) on INet, INet-v2, INet-ReaL, ObjectNet, and COCO I→T, and ties on COCO T→I — the cleanest "smaller and better" result in the contrastive literature.

### Batch-size sweep — the central scientific result

![Batch-size sweep: sigmoid wins at small batch, both saturate ~32k, very large batches hurt](/assets/images/paper/siglip/page_004.png)
*Figure 2: across SigLiT (left), SigLIP (middle), and 36-lang mSigLIP (right), sigmoid leads softmax at small batches; the curves cross around 32k; both losses peak by ~32k and degrade past ~98k. No variance bars are drawn on the headline curves.*

| Setup | 4k | 8k | 16k | 32k | 98k | 256k | 1M |
|---|---|---|---|---|---|---|---|
| SigLiT sigmoid (18B seen) | – | 83.6 | 84.2 | 84.6 | – | 84.7 | **84.7** |
| SigLiT softmax (18B seen) | – | 83.1 | 84.1 | 84.4 | – | 84.6 | – |
| **SigLIP sigmoid (9B, B/16)** | **68.4** | **70.6** | **72.3** | **73.4** | 73.0 | 71.6 (307k) | – |
| SigLIP softmax (9B, B/16) | 66.6 | 69.4 | 71.7 | 72.9 | **73.2** | 72.6 (307k) | – |

The 1.6–2.0pp small-batch advantage is real and replicated across both SigLiT and SigLIP setups. The 1M-batch SigLiT run (84.7) merely *ties* the 256k run — a clean falsification of the large-batch dogma. mSigLIP 36-lang avg retrieval shows the same pattern more dramatically, dropping from 34.9 at 32k to 32.7 at 240k.

### Negative-ratio masking (Fig. 6, SigLiT B=16k, 900M)

Random masking of negatives degrades; keeping only **hard** negatives preserves quality almost fully; "easy only" collapses. The ~268M negatives in a 16k batch are almost all uninformative — an efficient hard-negative miner could in principle match full-batch quality at a fraction of the compute. The paper does not build such a miner; it merely points out the design space.

### Noise robustness (Fig. 7, M/16 at B=16384, 3.6B examples)

Sigmoid stays above softmax across image-corruption, text-corruption, batch-misalignment, and combined corruptions up to p=0.4–0.5. The gap widens as corruption increases. Caveat: only the M/16 model size is reported — robustness scaling with model capacity is not stress-tested.

### Bias-term ablation (Table 4)

| Config | INet 0-shot |
|---|---|
| **Default (b=−10, t'=log 10)** | **63.0** |
| No bias | 62.0 |
| b=0, t'=log 1 (pathological) | 53.7 |

The −10 init is worth ~1.0pp on INet over no-bias and ~9pp over the pathological init — small but consistent across INet/Pet/CIFAR-100.

## Limitations

**Authors acknowledge.**
- Imbalance between positives and negatives could be addressed by smarter hard-negative mining (Fig. 6) — the design space is left open.
- Large batches lead to gradient-norm spikes; β₂=0.95 mitigates but doesn't eliminate (Fig. 5 still shows a spike at step 2B).
- mSigLIP coverage of low-resource languages is poor — XM3600 te (Telugu) at 0.3–8% R@1, mi (Maori) ~0%, vs de/fr/it ~70%. "100 languages" is a generous accounting.

**Unaddressed by the paper.**
- **Modality gap not measured.** The paper never quantifies whether sigmoid-trained image and text embeddings still occupy disjoint cones in feature space (the Liang et al. 2022 phenomenon). Pairwise sigmoid removes the *normalization* coupling between modalities, but the L2-normalized dot-product geometry is unchanged, so a modality gap of similar magnitude to CLIP's is the expected null hypothesis — yet not verified.
- **Why does softmax close the gap at large B but sigmoid doesn't *gain* further?** A natural conjecture: softmax's per-row partition function gives an effective hard-negative weighting that sigmoid lacks; sigmoid's flat per-pair treatment becomes a wash. The hard-negative-masking experiment (Fig. 6) is consistent but not framed this way.
- **Memory claim is anecdotal** (C3) — one config, no profiling, no comparison vs gradient-checkpointed softmax.
- **Noise robustness only at M/16** — no scaling study across model sizes.
- **No variance bars on Fig. 2 headline curves** — Table 7 reports σ at one batch size, the rest the reader must assume.
- **Downstream segmentation/detection transfer not benchmarked** — only classification + retrieval.
- **Calibration of the learned bias and temperature post-training** — what does $b \to \sim -12$ mean for using SigLIP scores as similarity probabilities? Not analyzed.
- **No comparison vs decoupled-softmax variants** (DCL, SogCLR) that also reduce the global-normalization pain.
- **WebLI is private** — only LiT (also Google) is public among the pre-training corpora; retraining requires a substitute dataset.

## Why It Matters

SigLIP's lasting contribution is **the loss is the budget unlock**: dropping global normalization changes contrastive pre-training from a hundred-chip job to a four-chip job, and the chunked-permute kernel turns an algorithmic argument into a deployable training recipe. Two practical consequences:

- **For practitioners with small clusters** (≤16 chips), sigmoid is a free 1.6–2.0pp lift on top of identical data and architecture, with no engineering downside — the 1-day, 4-chip 79.7% SigLiT recipe is reproducible from public LiT data and a public B/8 checkpoint.
- **For scaling research**, the 1M-batch experiment closes the "bigger batches = better contrastive" conjecture. The community should now treat ~32k as the practical contrastive sweet spot and stop spending compute past 98k unless a new mechanism (hard-negative mining, decoupled softmax) is introduced.

The SO-400M shape — 400M params, 729 patches, 83.2 INet — is also the reference visual encoder shape that subsequent multimodal LLMs (PaLI-3, downstream LLaVA-style stacks) adopted, so SigLIP weights have outlived the SigLIP paper as a default vision tower for VLM/LLM pipelines.

## References

- Paper: Zhai, Mustafa, Kolesnikov, Beyer. *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023. arXiv:2303.15343.
- Code & checkpoints: [google-research/big_vision](https://github.com/google-research/big_vision)
- Related: CLIP (Radford et al. 2021), LiT (Zhai et al. CVPR 2022), OpenCLIP (Cherti et al. 2022), EVA-CLIP (Sun et al. 2023), Modality-gap (Liang et al. NeurIPS 2022), DCL / SogCLR (decoupled-softmax variants).
