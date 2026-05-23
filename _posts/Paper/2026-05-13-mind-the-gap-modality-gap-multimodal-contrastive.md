---
title: "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning"
excerpt: "CLIP-style image and text embeddings live in two disjoint cones on the hypersphere; the cone effect at initialization plus low-temperature contrastive loss with mismatched pairs holds the gap in place — and translating CLIP embeddings along the gap vector lifts ViT-B/16 zero-shot CIFAR10 from 0.9013 to 0.9081."
categories:
  - Paper
  - Multimodal-Alignment
permalink: /paper/mind-the-gap-modality-gap-multimodal-contrastive/
tags:
  - Modality-Gap
  - Contrastive-Learning
  - CLIP
  - Cone-Effect
  - Representation-Learning
  - Fairness
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- In CLIP-style multi-modal contrastive models, image and text embeddings occupy **two disjoint cones** on the shared hypersphere — the **modality gap** — and this is *not* an artifact of architecture, modality, or data. The same gap appears in CLIP, VideoCLIP, ConVIRT (medical), and CLASP (amino acid), and is already present at random initialization with two identical encoders fed identical data.
- Two mechanisms produce and preserve the gap: (i) a **cone effect at initialization** caused by ReLU-style nonlinearity + depth (formalized as Theorem 1: each layer contracts cosine similarity), and (ii) the **contrastive loss at low temperature with mismatched pairs**, whose loss landscape probe shows the *current* CLIP gap distance $\|\vec\Delta_{\text{gap}}\| = 0.82$ is the global minimum at $\tau = 1/100$.
- Translating embeddings along the gap vector — no fine-tuning — gives small but statistically significant zero-shot gains: **CIFAR10 0.9013 -> 0.9081** ($p = 3.5\text{e-6}$), **CIFAR100 0.6658 -> 0.6737** ($p = 8.7\text{e-3}$), **EuroSAT 0.5410 -> 0.5645** ($p = 7.0\text{e-6}$); on FairFace, increasing the gap from 0.82 to 0.97 reduces denigration bias for every race in the sum metric (e.g., White 15.7% -> 13.7%) at a 0.0008 top-1 cost.

## Motivation

CLIP framed contrastive vision-language pretraining as learning a *shared* embedding space where image and text live together. The authors observe that "shared" is misleading: paired image and text embeddings sit in completely disjoint regions of the unit sphere. Worse, the natural attributions — different data distributions, different encoder families — are insufficient. The gap appears even with two identical randomly initialized encoders fed identical Gaussian noise. So neither the data nor the architecture asymmetry can be the root cause.

For medical AI this matters because **ConVIRT** — the precursor that BiomedCLIP, MedCLIP, and most CXR-text contrastive models inherit from — exhibits exactly the same gap. Any zero-shot diagnosis pipeline built on a contrastive medical foundation model inherits this geometry, and as the paper shows, *modifying* the gap can change downstream performance and fairness. This is the discovery paper for what is now a heavily-cited subfield (gap-aware fine-tuning, gap-aware retrieval, gap mitigation in domain adaptation).

## Core Innovation

This is an analysis paper, not a method paper. The contribution is a **three-part causal account** of the modality gap:

- **Cone effect at initialization.** In a deep ReLU-style net, the average pairwise cosine similarity of last-layer embeddings is far from 0 — i.e., embeddings live in a narrow cone, not on the whole sphere. Theorem 1 proves each layer contracts cosine similarity under standard Gaussian init.
- **Different encoders land in different cones.** Theorem 2 shows that for deep nets the variance of intermediate outputs is dominated by the initialization seed, not the data. So two encoders, even with identical inputs, will land in *two different* cones — and their embeddings sit in disjoint regions before any training happens.
- **Contrastive loss preserves the gap.** An *embedding-shift* probe — translate every image embedding by $-\lambda\vec\Delta_{\text{gap}}$ and every text embedding by $+\lambda\vec\Delta_{\text{gap}}$, renormalize, plot loss vs. gap distance — reveals that at CLIP's learned $\tau = 1/100$, the loss minimum coincides with the *current* gap. A toy 3D simulation isolates *mismatched pairs at low temperature* as the mechanism that creates this repulsive structure.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | A modality gap exists in pre-trained multi-modal contrastive models | UMAP (Fig. 1b) + SVD (Fig. 4) on CLIP, VideoCLIP, ConVIRT, CLASP | MSCOCO, HowTo100M (proxy), MIMIC-CXR (proxy), CLASP corpus | ⭐⭐⭐ |
| C2 | The gap exists at random initialization, not just after training | Fig. 1c UMAP + Fig. 2c (25 random seeds form 25 distinct cones) | MSCOCO + random noise | ⭐⭐⭐ |
| C3 | Cone effect is caused by ReLU-style nonlinearity + depth | Fig. 2b sweep over activations and depths; Theorem 1 | Synthetic Gaussian inputs | ⭐⭐⭐ |
| C4 | Each layer contracts cosine similarity (theorem) | Theorem 1 + Appendix D proof | Analytical | ⭐⭐ — covers single ReLU-MLP layer with Gaussian weights; does not formally cover residual streams, attention, or LayerNorm |
| C5 | Variance of intermediate outputs is dominated by initialization, not data | Theorem 2 + Fig. 2c (25 seeds -> 25 distinct cones) | Empirical + analytical | ⭐⭐⭐ |
| C6 | Contrastive loss with low $\tau$ has global minimum at the *current* gap | Fig. 3b: CLIP loss landscape probe at $\tau = 1/100$ | MSCOCO val (5,000 pairs) | ⭐⭐ — single CLIP checkpoint, single batch, but the result is striking |
| C7 | Repulsive structure is temperature-dependent | Fig. 3b–d at $\tau \in \{1/100, 1/50, 1\}$ + Supp. Fig. 8 fine-tuning | MSCOCO | ⭐⭐⭐ |
| C8 | Mismatched pairs are the mechanism | 3D toy simulation with vs. without mismatch (Fig. 3e–g vs. Supp. Fig. 9) | Synthetic | ⭐⭐ — clean simulation; no real-CLIP ablation that varies the mismatch rate |
| C9 | Both initialization *and* optimization contribute to the final gap | From-scratch CLIP with Procrustes-aligned init: gap 0.04 -> 0.75 after training | MSCOCO, batch 64, 3 seeds | ⭐⭐⭐ |
| C10 | Modifying the gap improves zero-shot performance | Table 1 (5 datasets), Table 5 ($\chi^2$ $p$-values) | CIFAR10/100, EuroSAT, SVHN, HatefulMemes | ⭐⭐ — gains real and significant on 3 of 5; absolute lifts are small (0.0007–0.0235); single CLIP backbone; per-task $\lambda$ search |
| C11 | Modifying the gap reduces denigration bias for *all* races | Table 2 FairFace (gap 0.82 -> 0.97) | FairFace eval | ⭐⭐ — consistent on the *sum* metric; White goes 0.2% -> 0.4% on non-human; gap 0.97 is post-hoc tuned |
| C12 | Cone effect is a general inductive bias of deep nets | Random weights on random noise still produce cones (Fig. 2a) | MSCOCO + Gaussian/integer noise | ⭐⭐⭐ — directly refutes the prior "unbalanced word frequency" explanation |

**Overall: ⭐⭐⭐ for the diagnosis, ⭐⭐ for the intervention.** The cone-effect-at-init evidence (C1–C3, C5, C12) is rock-solid: multiple architectures, multiple input regimes (real data, random noise), multiple visualization methods (UMAP, PCA, SVD), and a theorem that matches the empirical curves. The temperature/landscape story (C6, C7) is also clean. The from-scratch ablation (C9) does the right thing methodologically — multiple seeds, confidence intervals, and a *real* counterfactual: a Procrustes-aligned init that opens the gap back up to ~57% of the unmitigated value, demonstrating that contrastive optimization actively re-creates the gap.

The downstream interventions (C10, C11) are weaker than the abstract suggests. The zero-shot gains are sub-percent on three of five datasets, are obtained by per-task hyperparameter search over the shift magnitude $\lambda$, and the authors explicitly disclaim being a "method to close the gap" — readers should interpret Table 1 as an *existence proof of the lever*, not a SOTA improvement. The fairness result uses a single tuned shift; Supp. Table 4 reveals that pushing the gap further breaks both fairness and accuracy badly (gap 1.29 -> Black non-human 40.5%, accuracy 0.4083), so the relationship is U-shaped, not monotone. Cross-modality generalization (CLIP vs. VideoCLIP vs. ConVIRT vs. CLASP) is **visualization-only** — quantitative gap distances and downstream effects are CLIP-only.

## Method & Architecture

![The modality gap across CLIP, VideoCLIP, ConVIRT, and CLASP, both pre-trained and at random initialization](/assets/images/paper/mind-the-gap/page_002.png)
*Figure 1: The modality gap in CLIP, VideoCLIP, ConVIRT, and CLASP — visible both in pre-trained (b) and randomly initialized (c) models. The gap is not a quirk of any single architecture or modality.*

### 1. Measure the gap

Define the modality gap vector

$$\vec\Delta_{\text{gap} } = \frac{1}{n}\sum_i x_i - \frac{1}{n}\sum_i y_i$$

over L2-normalized image and text embeddings $\{x_i\}, \{y_i\}$. For pre-trained CLIP on 5,000 MSCOCO pairs, $\|\vec\Delta_{\text{gap}}\| = 0.82$.

### 2. Establish the cone effect

Extract 5,000 last-layer embeddings from ResNet-18 (ImageNet pre-trained), CLIP Vision Transformer, and CLIP Text Transformer. Compute cosine similarity over all $\binom{5000}{2} \approx 12.5\text{M}$ pairs. Average cos-sim is **0.56 / 0.47 / 0.51** (pre-trained, real data); **0.99 / 0.72 / 0.67** (random weights, real data); **0.999 / 0.94 / 0.41** (random weights, random noise). For reference: an angular region with cos-sim 0.56 occupies less than $1/2^{512}$ of the unit hypersphere surface in 512-D — an extraordinarily narrow cone.

![Cone effect: cosine-similarity histograms, depth-and-nonlinearity sweep, and 25 random initializations forming 25 distinct cones](/assets/images/paper/mind-the-gap/page_003.png)
*Figure 2: The cone effect — cosine similarities of embeddings are far from zero (a), grow with depth under nonlinearity (b), and 25 random initializations form 25 distinct cones on the same data (c).*

### 3. Pin the cone effect on nonlinearity + depth

Sweep MLPs with various activations, all $512 \times 512$ layers initialized $\mathcal{N}(0, 1/512)$, on Gaussian inputs. Without activation: no cone. With nonlinearity: average cos-sim grows rapidly with depth and saturates near 1. Sigmoid hits 0.99 at 2 layers; ReLU saturates by ~10 layers.

### 4. Theorem 1 (monotonicity of cosine similarity)

For fixed $u, v \in \mathbb{R}^{d_{\text{in}}}$ with $\|u\| = r\|v\|$, random Gaussian $W$ and bias $b$ with elements $\sim \mathcal{N}(0, 1/d_{\text{out}})$, if $\cos(u, v) < \big(\tfrac{1}{2}(r + 1/r)\big)^{-1}$, then with probability $1 - O(1/d_{\text{out}})$:

$$\cos\big(\phi(Wu + b),\, \phi(Wv + b)\big) > \cos(u, v).$$

The Appendix-D proof uses a rectified-Gaussian decomposition + Chebyshev, splitting $(Wu + b)_k(Wv + b)_k$ into max/min cross-terms and exploiting weight-distribution symmetry to bound the inner product (Lemma 3).

### 5. Theorem 2 (variance is dominated by initialization)

$$\frac{\operatorname{Var}\big[\mathbb{E}[h_\Theta(U) \mid \Theta]\big]}{\operatorname{Var}[h_\Theta(U)]} \geq \beta,$$

where $\beta$ is the average cos-sim of the previous layer. Since Theorem 1 drives $\beta \to 1$ for deep nets, the variance ratio is dominated by the choice of random init — which is exactly why two encoders land in two different cones.

### 6. Embedding-shift loss-landscape probe

Translate every image embedding by $-\lambda\vec\Delta_{\text{gap}}$ and every text embedding by $+\lambda\vec\Delta_{\text{gap}}$, renormalize to the sphere, sweep $\lambda$, and plot the resulting CLIP contrastive loss vs. Euclidean gap distance. **At $\tau = 1/100$ (CLIP's learned temperature), the loss minimum sits at $\|\vec\Delta_{\text{gap}}\| = 0.82$ — the original gap is the global minimum.** As $\tau$ increases (1/50, 1), the repulsive structure flattens and closing the gap becomes optimal.

![Embedding-shift loss-landscape probe at three temperatures and the toy 3D simulation](/assets/images/paper/mind-the-gap/page_006.png)
*Figure 3: Embedding-shift loss-landscape probe. At CLIP's learned $\tau = 1/100$ (b), the loss minimum sits at the original gap distance 0.82 — the contrastive loss actively prefers the gap. As temperature rises (c, d), this repulsive structure flattens. The toy 3D simulation (e–g) reproduces the same temperature-dependent repulsive structure with two intentionally mismatched pairs.*

### 7. Init-vs-optimization ablation

Train two CLIP-from-scratch models on MSCOCO (batch 64, $\tau = 1/100$, 3 seeds, 95% CI):

| Model | Initial gap | Final gap |
|---|---|---|
| Standard init | 1.1891 ± 0.0017 | 1.2991 ± 0.0389 |
| **Procrustes-amended init** ($W = \arg\min_{W \in O_D} \|X - YW\|$) | **0.0388 ± 0.0351** | **0.7457 ± 0.0633** |

Even with a near-zero initial gap, training opens it back up to ~57% of the unmitigated value — confirming **both initialization and optimization contribute**.

## Experimental Results

### Cone-effect statistics (pretraining is not necessary for the cone)

| Setting | ResNet | Vision Transformer | Text Transformer |
|---|---|---|---|
| Pretrained, real data (MSCOCO) | 0.56 (min 0.23) | 0.47 (min 0.05) | 0.51 (min 0.01) |
| Random weights, real data | 0.99 | 0.72 | 0.67 |
| Random weights, random noise | 0.999 | 0.94 | 0.41 |

Even the *minimum* cos-sim across ~25M pairs is positive in many settings — the entire embedding cloud sits on one side of the origin.

### Zero-shot accuracy with embedding shift (CLIP ViT-B/16, no fine-tuning)

| Dataset | Original gap acc. | Modified gap acc. | Direction | $p$-value (Table 5) |
|---|---|---|---|---|
| **CIFAR10** | **0.9013** | **0.9081** | increase gap | 3.476e-06 |
| **CIFAR100** | **0.6658** | **0.6737** | decrease gap | 8.701e-03 |
| **EuroSAT** | **0.5410** | **0.5645** | decrease gap | 7.020e-06 |
| SVHN | 0.5389 | 0.5396 | increase gap | (not reported) |
| HatefulMemes | 0.5800 | 0.5811 | increase gap | (not reported) |

Table 5 evaluates on the whole dataset (no fine-tuning involved). Table 1 and Table 5 differ very slightly because they use different subsets — e.g., CIFAR10 0.9013 vs. 0.9026.

### Denigration-bias reduction on FairFace (CLIP ViT-B/32, gap 0.82 -> 0.97)

| Race | Crime-related (orig -> mod) | Non-human (orig -> mod) | Sum (orig -> mod) |
|---|---|---|---|
| Black | 1.0% -> 0.8% | 0.1% -> 0.1% | 1.1% -> 1.0% |
| **White** | **15.5% -> 13.2%** | 0.2% -> 0.4% | **15.7% -> 13.7%** |
| Indian | 1.2% -> 1.1% | 0.0% -> 0.0% | 1.2% -> 1.1% |
| Latino | 2.8% -> 1.9% | 0.1% -> 0.1% | 2.8% -> 2.0% |
| Middle Eastern | 6.3% -> 5.2% | 0.0% -> 0.0% | 6.3% -> 5.2% |
| Southeast Asian | 0.5% -> 0.3% | 0.0% -> 0.0% | 0.5% -> 0.3% |
| East Asian | 0.7% -> 0.6% | 0.0% -> 0.0% | 0.7% -> 0.6% |

Top-1 accuracy drops only from 0.5817 to 0.5739 (−0.0008 in the headline; full-evaluation Appendix B.2 reports −0.0078). Pushing the gap *too far* in either direction backfires: at gap distance $d = 0.07$, White crime-related inflates to 23.0% and accuracy drops to 0.5599; at $d = 1.29$, Black non-human inflates catastrophically to 40.5% and accuracy crashes to 0.4083 (Supp. Table 4). The trade-off is **U-shaped in both fairness and accuracy**, not monotone.

### Qualitative findings worth highlighting

- **Cross-modal generalization is purely visualization-based.** Figure 1b/c and Figure 4 (SVD) show CLIP, VideoCLIP, ConVIRT, and CLASP all exhibit a visible gap under both pre-trained and random-weight regimes — but no quantitative gap distances are reported per model. Only CLIP gets distance measurements.
- **Embedding-dimension robustness (Supp. Fig. 16):** training CLIP from scratch on Conceptual Captions 3M with embedding dims {64, 128, 256, 512} gives roughly constant gap distance — the gap is not an artifact of dimensionality.
- **High-temperature fine-tuning closes the gap (Supp. Fig. 8):** $\tau \in \{1/10, 1\}$ shrinks the gap monotonically with $\tau$, consistent with the landscape probe.

## Limitations

**Acknowledged by the authors:**

- The paper is *not* proposing a method to close the gap — they explicitly say it is unclear whether closing the gap is desirable.
- Investigating *how much* multimodal data misalignment drives the gap is left to future work.
- Whether closing the cone effect in non-language modalities improves ML performance (analogous to the BERT post-processing literature) is open.

**Not addressed (or only weakly):**

- **Scale.** All from-scratch CLIP training is at batch 64 on MSCOCO. Production CLIP uses batch 32,768 on 400M pairs; whether the "contrastive loss preserves the gap" finding holds at production scale is not characterized.
- **Architecture coverage of the theory.** Theorem 1 covers ReLU-MLP layers with i.i.d. Gaussian weights. Modern multi-modal models use attention, residual streams, and LayerNorm — the gap persists empirically, but the theoretical result does not formally cover these.
- **Why $\tau$ converges where it does.** CLIP learns $\tau$ end-to-end and ends up at 1/100. The paper treats $\tau$ as an exogenous knob but does not investigate why optimization drives it down to a value that preserves the gap. Is the gap doing useful work?
- **No mechanism for downstream gains.** The paper shows changing the gap changes accuracy, but offers no explanation for *why* a larger gap helps CIFAR10 while a smaller gap helps CIFAR100 / EuroSAT.
- **Single CLIP backbone for all interventions.** Table 1 has no error bars across CLIP checkpoints (only one) and no error bars across the choice of $\lambda$ either. Table 5's $\chi^2$ $p$-values partially compensate.
- **Medical-imaging downstream.** ConVIRT is shown to have a gap, but no zero-shot or retrieval result on medical data is reported — exactly the question medical-AI readers want answered.
- **Procrustes-init experiment is silent on downstream.** The amended model's gap re-opens, but the paper does not report whether amended-then-trained CLIP has better or worse downstream performance than standard-init CLIP.

## Why It Matters for Medical AI

ConVIRT — the precursor of BiomedCLIP, MedCLIP, GLoRIA, and most CXR-text contrastive backbones — is one of the four models the paper visualizes (Fig. 1b/c). It exhibits the same gap as CLIP, both pre-trained and at random initialization. Three concrete implications for medical foundation models:

1. **Zero-shot pneumonia / cardiomegaly classifiers built on a contrastive CXR-text backbone inherit a non-shared embedding geometry**, and the embedding-shift trick is a near-zero-cost knob to probe whether closing or opening the gap improves clinical metrics. The paper does not run this experiment — it would be a clean follow-up.
2. **The cone-effect-at-init result implies that even random-init medical encoders sit in narrow cones**, so any "from-scratch" pretraining curriculum that uses a CLIP-style contrastive objective will start from disjoint cones regardless of how well-curated the medical corpus is. Curriculum or warm-start strategies that explicitly align cones at init (analogous to the Procrustes amendment) are an underexplored design space for medical contrastive pretraining.
3. **Fairness in medical AI** is increasingly evaluated by demographic-subgroup performance. The U-shaped fairness/accuracy trade-off shown on FairFace is a cautionary note: gap-shifting is a real lever but not a free win, and any clinical-deployment fairness audit should sweep $\lambda$ rather than report a single tuned value.

## References

- Paper: [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning (NeurIPS 2022)](https://arxiv.org/abs/2203.02053)
- Code: [github.com/Weixin-Liang/Modality-Gap](https://github.com/Weixin-Liang/Modality-Gap)
- Documentation: [modalitygap.readthedocs.io](https://modalitygap.readthedocs.io/)
- Models analyzed: CLIP (Radford et al., 2021), VideoCLIP (Xu et al., 2021), ConVIRT (Zhang et al., 2020), CLASP (OpenBioML)
- Related: cone effect in BERT representations (Ethayarajh, 2019); BERT-flow / BERT-whitening post-processing literature
