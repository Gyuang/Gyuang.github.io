---
title: "Representation Learning with Contrastive Predictive Coding"
excerpt: "Origin of InfoNCE: a single objective transfers across speech, vision, text, and RL — ImageNet linear probe 48.7% top-1 (+9.1 over Colorization same backbone), with an MI lower bound capped at log(N)."
categories:
  - Paper
  - Multimodal-Alignment
tags:
  - CPC
  - InfoNCE
  - Self-Supervised
  - Mutual-Information
  - Contrastive-Learning
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
permalink: /paper/cpc-infonce-contrastive-predictive-coding/
---

## TL;DR
- A single self-supervised recipe — predict *future latents* with a log-bilinear scorer, train with the **InfoNCE** softmax over 1 positive + N−1 in-batch negatives — transfers across speech, vision, text, and RL.
- The MI argument: at the optimum `f_k(x_{t+k}, c_t) ∝ p(x_{t+k} | c_t) / p(x_{t+k})`, and `I(x_{t+k}; c_t) ≥ log(N) − L_N`. The bound is **capped at log(N)** and is only tight when the log-bilinear scorer can realize the true density ratio — two assumptions the framing glosses over.
- Headline numbers (single seed, no error bars): LibriSpeech phone linear probe **64.6%** (vs MFCC 39.7%, supervised 74.6%); LibriSpeech speaker **97.4%** (vs supervised 98.5%); ImageNet linear probe **48.7%** top-1 / **73.6%** top-5 (+9.1 absolute over Colorization on the same ResNet backbone); TREC **96.8%**; CPC auxiliary loss improves 4 of 5 DeepMind Lab tasks over A2C.

## Motivation

Generative unsupervised models (PixelCNN-style `p(x | c)`) burn capacity reconstructing pixel noise instead of slow, semantically useful structure (phonemes, objects, story arcs). The contrastive/predictive-coding literature already had isolated wins — word2vec for text, colorization and jigsaw for images, triplet losses for video — but each was modality-specific. The gap was a *single* objective whose induced features transfer across modalities, with the optimization made tractable by negative sampling rather than full density estimation. CPC's framing — predict latents, not pixels, and discriminate the positive among negatives — is the move that later seeds SimCLR, MoCo, CLIP, and every downstream medical-VLM contrastive loss. The paper itself has no medical content, but every later medical contrastive method (BiomedCLIP, CONCH, BLEEP, OmiCLIP, SigLIP) inherits this objective.

## Core Innovation

The architecture has three pieces:

1. **Encoder** `z_t = g_enc(x_t)` — strided convs (audio), ResNet-v2-101 patch encoder (vision), conv + mean-pool (text).
2. **Autoregressive context** `c_t = g_ar(z_{≤t})` — GRU(256) for audio, row-autoregressive PixelCNN-style aggregator for image patches, GRU(2400) over sentences for text.
3. **Density-ratio scorer** `f_k(x_{t+k}, c_t) = exp(z_{t+k}^\top W_k c_t)` — a *separate* log-bilinear projection `W_k` per prediction step k. `f` is unnormalized; only its ratio matters.

The objective for a sample set `X` of size `N` (one positive drawn from `p(x_{t+k} | c_t)`, `N−1` negatives drawn from a proposal `p(x_{t+k})`):

$$
L_N = -\mathbb{E}_X \Big[ \log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \Big]
$$

The optimum (Eq. 5) is `f_k^\star(x_{t+k}, c_t) \propto p(x_{t+k} | c_t) / p(x_{t+k})`, independent of N. Substituting back and using Jensen plus `N − 1 \le N` gives

$$
I(x_{t+k}; c_t) \ge \log(N) - L_N
$$

which is the MI lower bound that gives InfoNCE its name.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | InfoNCE is a *valid* lower bound on MI: `I ≥ log(N) − L_N`. | §2.3 + Appendix A.1 derivation. | ⭐⭐⭐ as math, **conditioned on** the encoder realizing the optimal density ratio. The bound is also capped at log(N), so any reported "MI" is at most log(batch). |
| C2 | A single mechanism transfers across **four** modalities. | Tables 1, 3–5 + Figure 6. | ⭐⭐⭐ — uncommon at the time to test 4 modalities with one recipe. |
| C3 | "+9% absolute top-1 / +4% top-5 over SOTA on ImageNet." | Table 3 (48.7 vs Colorization 39.6) and Table 4 (73.6 vs MS+Ex+RP+Col ensemble 69.3). | ⭐⭐ — single seed, no variance, and architectures differ across baselines (AlexNet-era vs ResNet-v2). The same-backbone delta vs Colorization (+9.1) is real. |
| C4 | Predicting *multiple* future steps is important. | Table 2: k=2 → 28.5; k=4 → 57.6; k=8 → 63.6; **k=12 → 64.6**; k=16 → 63.8. | ⭐⭐⭐ — clear monotonic-then-saturating curve on phones. |
| C5 | The negative-sample distribution matters / hard negatives help. | Table 2 second block; spread 57.3 → 65.5. | ⭐⭐ — only ~8 pt spread, one task, no statistical test. |
| C6 | CPC as auxiliary loss accelerates RL. | Figure 6: 4/5 DM Lab tasks improve over A2C. | ⭐⭐ — single hyperparameter-searched run per task, no seed variance, lasertag is unchanged. |
| C7 | The same `c_t` captures speaker identity AND phonetic content. | Table 1: 97.4% speaker + 64.6% phone from one representation, two linear probes. | ⭐⭐⭐ — large absolute gaps over MFCCs. |
| C8 | InfoNCE is more stable than MINE for MI estimation. | Appendix A.1 derivation + a single sentence that "MINE was unstable when prediction was easy." | ⭐ — no quantitative comparison provided. |

**Honest read.** The theoretical core (C1) is the paper's most-cited contribution and is correct, but two practical caveats are not surfaced: the bound is tight only at the optimum, and a log-bilinear `z^\top W_k c` cannot in general realize an arbitrary `p(x | c) / p(x)`; the bound also cannot exceed `log(N)`, period (Poole et al. 2019; McAllester & Stratos 2020). The implicit framing — "we maximize mutual information" — is theoretically misleading: Tschannen et al. 2019 showed that estimators with *worse* MI bounds can yield *better* representations, and the practical success of CPC/SimCLR is better explained by alignment-and-uniformity geometry and hard-negative dynamics than by any quantity related to MI magnitude.

## Method & Architecture

![Figure 1](/assets/images/paper/cpc-infonce/fig_p002_01.png)
*Figure 1: CPC architecture. The encoder `g_enc` maps observations `x_t` to latents `z_t`; an autoregressive `g_ar` summarises history into a context `c_t`; the model predicts future latents `z_{t+k}` for k=1..K and is trained with InfoNCE against in-batch negatives.*

Concrete instantiations across modalities:

- **Audio**: 5 strided 1-D convs (strides [5,4,2,2,2], filters [10,8,4,4,4], 512 units) downsample 16 kHz PCM by 160 → one 512-d feature per 10 ms; GRU(256) context; predict K=12 future latents (200 ms horizon).
- **Vision**: ResNet-v2-101 (no BatchNorm), 3rd-block output spatial-mean-pooled to a 1024-d vector per 64×64 patch on a 7×7 grid (32-px overlap) of greyscale 256×256 ImageNet images; PixelCNN-style row-autoregressive context over the 7×7 grid; predict up to 5 rows below.
- **NLP**: 1-D conv + ReLU + mean-pool → 2400-d sentence embedding; GRU(2400) context over BookCorpus sentences; predict 3 future sentences.
- **RL**: CPC as auxiliary loss on a batched-A2C agent (no replay), unroll length 100, predict up to 30 steps.

Training is Adam with `lr = 2e-4`. Negatives are drawn from the same minibatch — i.e. an empirical proposal distribution, not the true marginal `p(x_{t+k})`. The bias from this approximation is not analysed, and is later flagged by the debiased-CL and hard-negative-CL line of work.

## Experimental Results

| Domain | Benchmark / Metric | Random init | Prior best | **CPC** | Supervised |
|---|---|---|---|---|---|
| Speech | LibriSpeech phone acc (41-way, linear) | 27.6 | MFCC 39.7 | **64.6** (72.5 w/ 1 hidden) | 74.6 |
| Speech | LibriSpeech speaker acc (251-way, linear) | 1.87 | MFCC 17.6 | **97.4** | 98.5 |
| Vision | ImageNet top-1 linear (ResNet-v2 family) | — | Colorization 39.6 / RelPos 36.2 / Exemplar 31.5 / MS 27.6 | **48.7** | — |
| Vision | ImageNet top-5 linear | — | Ensemble (MS+Ex+RP+Col) 69.3 / Colorization 62.5 | **73.6** | — |
| NLP | TREC | — | Skip-thought 91.4 | **96.8** | — |
| NLP | MR / CR / Subj / MPQA | — | Skip-thought+LN 79.5 / 82.6 / 93.4 / 89.0 | 76.9 / 80.1 / 91.2 / 87.7 | — |
| RL | DeepMind Lab (5 tasks) | A2C baseline | — | improves 4/5; lasertag unchanged | — |

**Ablations (Table 2, LibriSpeech phones).** Steps predicted: k=2 → 28.5, k=4 → 57.6, k=8 → 63.6, **k=12 → 64.6**, k=16 → 63.8. Negative-sample distribution: mixed-speaker 64.6, same-speaker 65.5, mixed-speaker excl. current sequence 57.3, same-speaker excl. current sequence 64.6, current-sequence-only 65.2. Drawing negatives only from *other* speakers is *worse* — the first empirical hint that easy negatives are uninformative, a theme later developed by MoCo/SimCLR/hard-negative-CL.

ImageNet beats Colorization on the same ResNet-v2 backbone by +9.1 top-1, but the comparison to AlexNet-era baselines (Exemplar, RelPos, Jigsaw) is partly an architecture-versus-method confound that the paper does not fully untangle. Every number is a single training run; **there are no error bars anywhere in the paper**.

![Figure 2](/assets/images/paper/cpc-infonce/fig_p005_01.png)
*Figure 2: t-SNE of CPC speech contexts `c_t` for 10 LibriSpeech speakers — points cluster cleanly by speaker, confirming speaker identity is recoverable from the unsupervised representation.*

![Figure 3](/assets/images/paper/cpc-infonce/fig_p005_02.png)
*Figure 3: positive-vs-negative discrimination accuracy in the InfoNCE loss as a function of prediction horizon (10 ms per step). Accuracy decays smoothly from k=1 to k=20, showing the prediction task is non-trivial and gets harder farther into the future.*

![Figure 5 row](/assets/images/paper/cpc-infonce/fig_p007_01.png)
*Figure 5 (selected row): image patches that most activate one CPC neuron — illustrative of part-like / object-like feature organisation learned without labels.*

![Figure 5 row](/assets/images/paper/cpc-infonce/fig_p007_05.png)
*Figure 5 (selected row): a different CPC neuron's top-activating patches; qualitatively distinct from the row above.*

![Figure 6](/assets/images/paper/cpc-infonce/fig_p009_01.png)
*Figure 6: reinforcement-learning curves on 5 DeepMind Lab tasks. Black = A2C baseline; red = A2C + auxiliary CPC loss. CPC improves 4 of 5 tasks; lasertag (rightmost) is unchanged.*

## Limitations

Several gaps are not addressed in the paper and matter for anyone citing it:

- **The MI bound is capped at `log(N)`.** Any "mutual information" reported via InfoNCE is upper-bounded by the log of the batch size, regardless of the true MI (Poole et al. 2019; McAllester & Stratos 2020). The paper's narrative implicitly suggests larger N gives a *more accurate* MI estimate; in fact larger N raises the *ceiling* of an estimator that is biased low.
- **Encoder-capacity assumption.** The proof assumes `f_k` can realise the optimal density ratio. With a log-bilinear `z^\top W_k c`, this is a strong assumption — the bound is loose whenever `p(x | c)` is multimodal or non-bilinear-separable. The paper does not characterize this looseness empirically.
- **Minibatch negatives bias the proposal.** Negatives are drawn from the same minibatch (empirical `p(x)`), not the true marginal. The induced bias is not analysed; later work (debiased contrastive, hard-negative contrastive) makes this central.
- **MI is not what matters for downstream.** Tschannen et al. 2019 showed that InfoMax estimators with *worse* MI estimates can yield *better* representations. The "maximise MI" framing is intuitive but theoretically misleading — alignment-and-uniformity geometry (Wang & Isola 2020) explains the empirics better.
- **No variance / no seeds.** Every reported number is a single training run. The +9% ImageNet headline could lie inside the seed-variance band of the strongest baselines, and the RL curves are single runs per task.
- **No comparison to a non-contrastive predictive baseline** (e.g. predict `z_{t+k}` directly with MSE) on the same setup. The contrastive choice is justified intuitively, not ablated against the natural alternative.
- **MINE comparison is qualitative only** — one sentence asserting instability, no learning curves.
- **Author-acknowledged caveats.** Window/context size strongly affects speech results (only ~1 sec context tested); NLP transfer benchmarks are small and bag-of-words is competitive; phone accuracy jumps 64.6 → 72.5 with a single hidden layer, so "not all information is linearly accessible"; lasertag is "purely reactive" and does not benefit.
- **Vision encoder choice (no BatchNorm)** is unexplained, and downstream baseline comparison mixes architectures.
- **Compute cost** is never reported (8–32 GPUs implied; no wall-clock or sample-efficiency comparison).
- **Dataset caveats.** LibriSpeech is read English audiobook speech, not conversational; ImageNet patches are deliberately greyscaled to remove colour cues (which favours CPC over the colorization baseline it is compared against); BookCorpus has well-documented licensing/copyright issues, not raised.

The paper aged well as an *engineering recipe* — predict in latent space, use many in-batch negatives, multi-step prediction. Its *theoretical framing* (MI maximization) was sharpened and partly corrected by 2019–2020 follow-ups; the 2020 alignment-and-uniformity reformulation is the more accurate vocabulary for what InfoNCE actually optimizes.

## Why It Matters for Medical AI

CPC has no medical content, but every medical vision-language and spatial-omics contrastive method downstream of 2018 inherits InfoNCE almost verbatim — CLIP, BiomedCLIP, CONCH, BLEEP, SigLIP, OmiCLIP, the spatial-transcriptomics contrastive line. Two reading notes for those papers:

1. The "we maximize mutual information between modalities" rationale routinely appears in medical-VLM motivation paragraphs and is theoretically loose: the MI estimate is capped at `log(batch)` and biased low; success is better explained by alignment + uniformity geometry and hard-negative dynamics. Citing CPC as a *mutual-information* result is overclaiming; citing it as the canonical *contrastive predictive recipe* is fair.
2. The minibatch-negatives proposal is a real bias — in medical settings where batches are dominated by intra-patient or intra-site samples, the empirical `p(x)` differs strongly from any clinically meaningful marginal, and the theoretical MI bound provides no guidance about what the resulting representation actually captures.

## References

- Paper: [arXiv:1807.03748v2](https://arxiv.org/abs/1807.03748) — van den Oord, Li, Vinyals (DeepMind, 2018; v2 Jan 2019)
- Open implementations (third-party, unofficial): [davidtellez/contrastive-predictive-coding](https://github.com/davidtellez/contrastive-predictive-coding), [jefflai108/Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)
- MI critique: Tschannen et al. 2019, ["On Mutual Information Maximization for Representation Learning"](https://arxiv.org/abs/1907.13625); Poole et al. 2019, ["On Variational Bounds of Mutual Information"](https://arxiv.org/abs/1905.06922); McAllester & Stratos 2020, ["Formal Limitations on the Measurement of Mutual Information"](https://arxiv.org/abs/1811.04251)
- Geometric reformulation: Wang & Isola 2020, ["Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"](https://arxiv.org/abs/2005.10242)
- Hard-negative / debiased CL: Chuang et al. 2020 ("Debiased Contrastive Learning"); Robinson et al. 2021 ("Contrastive Learning with Hard Negative Samples")
- Direct successors: SimCLR (Chen et al. 2020), MoCo (He et al. 2020), CLIP (Radford et al. 2021)
