---
title: "Probabilistic Concept Bottleneck Models"
excerpt: "Per-concept Gaussian embeddings with paired +/- anchors deliver calibrated concept uncertainty on CUB/AwA2 while leaving classification accuracy tied with deterministic CBM (CUB 0.680 vs 0.670; AwA2 0.880 vs 0.877)."
categories:
  - Paper
tags:
  - ProbCBM
  - CBM
  - Concept-Learning
  - Uncertainty-Estimation
  - Probabilistic-Embeddings
  - Interpretability
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- ProbCBM (Kim et al., ICML 2023) replaces CBM's deterministic per-concept logit with a **Gaussian embedding** $\mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c))$ per concept, plus two trainable anchors $z_c^+, z_c^-$ for "exists" and "does not exist". Concept and class predictions become distance-to-anchor matching problems solved by Monte-Carlo sampling.
- On CUB-224 and AwA2 the headline accuracy numbers are basically tied with CBM (**CUB class 0.680 +/- 0.004 vs 0.670 +/- 0.006**; **AwA2 class 0.880 +/- 0.002 vs 0.877 +/- 0.004**); the "w/o prob" ablation reaches **0.677** on CUB, so the architecture, not the probabilistic modelling, drives the small accuracy gain.
- The real win is **calibrated concept uncertainty**: after a 64x64 occlusion, ProbCBM detects uncertainty increases on a much larger fraction of samples than CBM, CBM+MC-dropout, and CEM (Fig 6), and uncertainty-ordered intervention reaches 100% class accuracy faster than random-order intervention (Fig 11) — the actual practical payoff.

## Motivation

CBM-Koh (ICML 2020) frames each concept as a hard binary classification, but **class-level concept annotation** — every American Goldfinch in CUB gets the same 112-attribute vector — plus standard augmentation (random crop, color jitter) routinely creates inputs where the concept's visual evidence is *not actually present*: a cropped image no longer shows the "belly: yellow" region; a color-jittered wing no longer matches its labelled colour. CBM's sigmoid head has no way to say "I cannot tell whether this concept is present in *this* image."

ProbCBM's pitch is that **reliability is what makes a concept explanation worth showing to a human**, and reliability requires a per-concept "how sure am I?" attached to every prediction. The medical relevance is implicit but direct: the canonical CBM-Koh KLG-from-knee-x-ray pipeline routinely operates on x-rays where joint space narrowing or osteophytes are partially occluded or off-frame — precisely the partially-visible-concept regime where a clinician needs the model to hedge.

![Same Least Tern class but tail, belly and color partially missing across images](/assets/images/paper/probabilistic-cbm/page_001.png)
*Figure 1: The motivating problem. All four images are labelled "Least Tern" with the same class-level concept vector, but the relevant visual evidence is partial or absent in each — exactly the ambiguity ProbCBM targets with a per-concept variance.*

## Core Innovation

- **Probabilistic concept embeddings.** Each concept is a Gaussian $\mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c))$ predicted by a per-concept Probabilistic Embedding Module (PEM) head on top of a shared ResNet-18 backbone. This generalises Hedged Instance Embeddings (HIB; Oh et al., 2019) from instances to concepts.
- **Two anchors per concept.** Trainable $z_c^+, z_c^-$ represent "concept present" and "concept absent" in the same embedding space. Existence probability is the MC-averaged sigmoid of $a \cdot (\|z - z^-\| - \|z - z^+\|)$. Appendix A argues two anchors (vs. only $z^+$) are necessary to keep $\sigma_c$ a meaningful uncertainty signal — though no empirical single-anchor ablation is provided.
- **Analytic class uncertainty.** Because the class-embedding map is a linear projection of concatenated concept samples, $h \sim \mathcal{N}(w^\top \mu + b, w^\top \Sigma w)$ where $\Sigma = \mathrm{diag}(\sigma_{c_1}, \dots, \sigma_{c_{N_c}})$. Class uncertainty is reported as $\det(w^\top \Sigma w)$ — i.e. **class uncertainty is a closed-form function of concept uncertainties**, giving the "explainable class uncertainty" claim actual mathematical content rather than vibes.

![Concept embedding space, class projection, and uncertainty as ellipse size](/assets/images/paper/probabilistic-cbm/page_002.png)
*Figure 2: Each image becomes an ellipse in concept embedding space; the same ellipses are then linearly projected to class embedding space. Wider ellipses = larger $\sigma$ = more uncertain. The two anchors $z_c^+, z_c^-$ define where "exists" and "does not exist" live for each concept.*

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|-------|----------|----------|----------|
| C1 | Probabilistic concept modelling does not hurt prediction performance vs CBM | Table 1: concept accuracy tied; class +0.01 on CUB-224, tied on AwA2 — 3-run std reported | CUB, AwA2 | ⭐⭐⭐ |
| C2 | The variance head $\sigma_c$ is a meaningful, calibrated concept uncertainty | Fig 4-5 qualitative on MNIST + Fig 6 quantitative occlusion + Fig 10 monotone uncertainty-vs-accuracy | MNIST-synth, CUB, AwA2 | ⭐⭐⭐ |
| C3 | Class uncertainty is decomposable into concept uncertainties | Closed-form $w^\top \Sigma w$ is exact; Fig 11 shows class uncertainty drops as concepts are fixed by intervention. No quantitative decomposition-faithfulness check. | CUB | ⭐⭐ |
| C4 | ProbCBM beats CBM+MC-dropout and CEM at uncertainty estimation | Fig 6 occlusion proportion. Single occlusion protocol, single image size (299x299), no OOD, no ECE | CUB only | ⭐⭐ |
| C5 | Two anchors are necessary; a single positive anchor breaks $\sigma$ | Appendix A verbal/geometric argument only — no ablation table | — | ⭐ |
| C6 | Uncertainty-guided intervention is faster than random-order intervention | Fig 11: ordered run vs mean +/- std over 5 random orders | CUB | ⭐⭐ |
| C7 | Augmentation-induced ambiguity is the source of concept uncertainty | Fig 8 cutout & hue sweeps show monotone increase in $\sigma$ — confounded with raw classification difficulty | CUB, AwA2 | ⭐⭐ |
| C8 | Sampling-free inference ($z_c = \mu_c$) is essentially free in accuracy | Table 1 "w/o sampling" rows; differences <= 0.001 within std | CUB, AwA2 | ⭐⭐⭐ |

**Honest read.** The strongest contribution is *not* the accuracy table — the "w/o prob" deterministic-embedding ablation reaches **0.677** class accuracy on CUB, only 0.003 below full ProbCBM and *above* CBM's 0.670. So C1, as framed in the abstract, is over-attributed to probabilistic modelling: the embedding-anchor architecture is doing most of the work. What probabilistic modelling actually buys is C2 + C4 + C6 + C8 — a cheap, calibrated, intervention-relevant uncertainty signal that CBM does not have.

The weakest move is **C5**: the two-anchor design is presented as load-bearing but is defended only geometrically in Appendix A. A single-anchor probabilistic CBM ablation would have been the cleanest experiment in the paper and is missing. **C4** rests on one occlusion protocol with a single patch size — no naturally OOD inputs, no expected calibration error (ECE), no reliability diagram. And despite the medical-AI framing of "reliability for partially-visible concepts," **there is zero medical-imaging experiment** — the CBM-Koh OAI KLG setup is right there and untested. Variance is reported (3 runs, std) only for Table 1; Figures 6/8/10/11 are single-seed.

## Method & Architecture

![PEM heads produce per-concept Gaussians; samples are matched to z+/z- anchors; concatenated samples are projected to class anchors](/assets/images/paper/probabilistic-cbm/page_003.png)
*Figure 3: Backbone → per-concept PEM (mean + variance heads) → MC samples $z_c^{(n)}$ → distance to concept anchors $z_c^+, z_c^-$ → concatenated samples → linear projection → distance to class anchors $g_k$.*

### Setup

Training triples $(x, C, y)$ with $C \in \{0,1\}^{N_c}$ over $N_c$ concepts. As in CBM, $\hat y = f(g(x))$, but embeddings replace scalar concept activations.

### Probabilistic Embedding Module (PEM)

A shared ResNet-18 backbone (2-layer CNN for the synthetic MNIST setup) produces a feature map. For each concept $c$, an independent PEM head — FC → BN → ReLU → self-attention-based mean and variance heads (architecture borrowed from Chun et al., 2021) — outputs $(\mu_c, \sigma_c) \in \mathbb{R}^{D_c} \times \mathbb{R}^{D_c}$, giving

$$p(z_c \mid x) \sim \mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c)).$$

### Two-anchor concept prediction

With $N_s$ MC samples $z_c^{(n)} \sim p(z_c \mid x)$ and learnable scalar $a > 0$,

$$p(c=1 \mid z_c^{(n)}) = \sigma\!\left(a\,(\|z_c^{(n)} - z_c^-\|_2 - \|z_c^{(n)} - z_c^+\|_2)\right),$$

and the existence probability is the MC average. Appendix A is the load-bearing argument: with only a positive anchor, $\sigma_c$ of negative samples blows up to push their mean far from $z^+$ without any constraint, breaking the variance-as-uncertainty interpretation.

### Probabilistic class prediction

Per-sample concept embeddings are concatenated and linearly projected to a class embedding,

$$h^{(n)} = w^\top [z_{c_1}^{(n)}, \dots, z_{c_{N_c}}^{(n)}] + b,$$

and matched against trainable class anchors $g_k$ by negative squared distance softmax with learnable scale $d > 0$:

$$p(y_k \mid x) \approx \tfrac{1}{N_s} \sum_n \frac{\exp(-d\|h^{(n)} - g_k\|_2)}{\sum_{k'} \exp(-d\|h^{(n)} - g_{k'}\|_2)}.$$

### Training loss

Concept predictor: BCE on the MC-averaged concept probability plus a KL prior to prevent variance collapse,

$$\mathcal{L}_{\text{concept}} = \mathcal{L}_{BCE} + \lambda_{KL}\sum_c \mathrm{KL}(\mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c)) \,\|\, \mathcal{N}(0, I)).$$

The class predictor is trained **separately** with cross-entropy (sequential training in the CBM-Koh taxonomy).

### Anchor mix-in (the regularisation trick)

When training the class head, each sampled $z_c^{(n)}$ is, with probability $p_{\text{replace}}$, swapped for the ground-truth anchor ($z_c^+$ if the GT concept is 1, else $z_c^-$). This prevents the class head from over-fitting to whatever wrong concept embeddings $g$ happens to produce during training — and the authors credit this trick with the small class-accuracy gain over deterministic embeddings.

### Analytic class uncertainty

Since concatenation + linear projection is linear, $h \sim \mathcal{N}(w^\top \mu + b, w^\top \Sigma w)$ where $\Sigma = \mathrm{diag}(\sigma_{c_1}, \dots, \sigma_{c_{N_c}})$. **Concept uncertainty** is reported as the geometric mean of $\sigma_c$ (volume of the per-concept Gaussian); **class uncertainty** as $\det(w^\top \Sigma w)$. No Monte-Carlo needed for either.

### Inference modes

Two paths: with MC sampling or with $z_c = \mu_c$ (no sampling). Table 1 shows the gap is <= 0.001 on every metric, so the sampling-free path is the default.

## Experimental Results

![Page 6: Table 1 accuracy parity with CBM, and Figure 6 bar chart where ProbCBM detects uncertainty increases on the most samples after occlusion](/assets/images/paper/probabilistic-cbm/page_006.png)
*Figure 4: Table 1 + Figure 6 on the same page — accuracy is on par with CBM, but ProbCBM detects post-occlusion uncertainty increases on a much larger fraction of samples than CBM, CBM+MC-dropout, and CEM.*

### Main quantitative comparison (Table 1)

| Dataset | Image size | Method | Concept acc | Class acc |
|---|---|---|---|---|
| CUB | 224x224 | CBM | 0.950 +/- 0.001 | 0.670 +/- 0.006 |
| CUB | 224x224 | **ProbCBM** | **0.949 +/- 0.001** | **0.680 +/- 0.004** |
| CUB | 224x224 | ProbCBM w/o prob | 0.950 +/- 0.001 | 0.677 +/- 0.004 |
| CUB | 224x224 | ProbCBM w/o sampling | 0.949 +/- 0.001 | 0.679 +/- 0.003 |
| CUB | 299x299 | CBM | 0.956 +/- 0.001 | 0.708 +/- 0.006 |
| CUB | 299x299 | CEM | 0.954 +/- 0.001 | 0.759 +/- 0.002 |
| CUB | 299x299 | **ProbCBM** | **0.956 +/- 0.001** | **0.718 +/- 0.005** |
| AwA2 | — | CBM | 0.975 +/- 0.001 | 0.877 +/- 0.004 |
| AwA2 | — | **ProbCBM** | **0.975 +/- 0.000** | **0.880 +/- 0.002** |
| AwA2 | — | ProbCBM w/o prob | 0.975 +/- 0.000 | 0.880 +/- 0.002 |
| AwA2 | — | ProbCBM w/o sampling | 0.975 +/- 0.000 | 0.880 +/- 0.001 |

**Reading.** ProbCBM beats CBM by ~1 pt class accuracy on CUB-224 and is statistically indistinguishable on AwA2. The "w/o prob" row is the diagnostic that matters: stripping out probabilistic embeddings but keeping the embedding-anchor architecture lands at **0.677** on CUB and identical 0.880 on AwA2 — so the small accuracy edge over CBM is the architecture, not the probability head. **CEM still wins CUB-299 class accuracy by 4 points (0.759 vs 0.718)**, which the authors attribute to CEM's joint training and to its concept embeddings carrying task information beyond existence. Honestly framed: ProbCBM's accuracy story is "tied with CBM," not "best of breed."

### Uncertainty estimation under occlusion (Figure 6, the actually-novel comparison)

After occluding a 64x64 patch from each CUB-299 image, the authors measure the **fraction of samples whose estimated uncertainty increases**:

- ProbCBM detects uncertainty increase on a visibly larger fraction of samples than CBM, CBM+MC-dropout, and CEM, on both concept and class uncertainty.
- CEM is "diverged significantly" — its entropy-based uncertainty does not behave like ProbCBM's variance-based one, despite both using concept embeddings.
- MC-dropout helps CBM somewhat but is dominated by ProbCBM.

### Secondary results

![Cutout/hue sweeps monotonically increase uncertainty; uncertainty correlates with prediction accuracy](/assets/images/paper/probabilistic-cbm/page_008.png)
*Figure 5: (Fig 8) Increasing cutout patch size or hue shift monotonically increases concept and class uncertainty on CUB and AwA2. (Fig 9-10) PCA of the leg-color embedding space; test images bucketed by uncertainty into 10 groups show monotonically decreasing accuracy in higher-uncertainty buckets — so ProbCBM's uncertainty is useful as a failure predictor.*

![Uncertainty-ordered intervention reaches 100% class accuracy faster than random-order intervention on CUB](/assets/images/paper/probabilistic-cbm/fig_p009_02.png)
*Figure 6 (paper Fig 11): On CUB, intervening on concepts in descending order of estimated uncertainty reaches 100% class accuracy faster than random-order intervention (mean +/- std over 5 random orders). This is the cleanest practical payoff of having calibrated $\sigma_c$ — uncertainty becomes a queue for the radiologist's attention.*

## Limitations

**Authors acknowledge.**
- Inherits CBM's dependence on quality of human concept annotations — ProbCBM claims to *reduce* but not eliminate this dependency by absorbing label noise into $\sigma$.
- Does not address **concept leakage** (Mahinpei et al., 2021), explicitly flagged as an open problem for the whole CBM family including ProbCBM.

**Not addressed.**
- **No medical dataset** despite the reliability framing. The CBM-Koh OAI KLG knee-x-ray setup is the natural test for partially-visible-concept uncertainty and is not run.
- **Single-anchor ablation missing** — Appendix A's two-anchor argument deserves an empirical knock-out experiment.
- **No standard calibration metric** — no expected calibration error (ECE), no reliability diagram, no proper-scoring-rule comparison. C2 rests on occlusion-detection proportion plus accuracy-vs-uncertainty buckets, which are useful but not the standard battery.
- **Concept dim $D_c$** is a free hyperparameter set per dataset in the appendix only; no sensitivity analysis in the body.
- **CUB's class-level concept labels are the very noise source ProbCBM is designed for**, making the headline experiment slightly circular: of course modelling per-image ambiguity helps when the labels are intentionally per-class.
- **CEM still beats ProbCBM by 4 points class accuracy at 299x299**, framed as a difference in training objective but worth flagging.
- **The "ambiguity in diverse visual contexts" claim** (Fig 1, Fig 7) is illustrated only on cherry-picked examples — no quantitative measurement of how often visual-context ambiguity actually causes CBM concept errors in the wild.

## Why It Matters for Medical AI

ProbCBM is technically a natural-image paper but addresses a failure mode that is more pronounced in clinical imaging than in CUB. A KLG knee x-ray frequently shows the joint at an angle, partially occluded by overlapping bone, or imaged at a resolution where osteophytes are sub-pixel — exactly the regime where a deterministic-binary concept predictor will produce a confident answer with no warning. The relevant comparisons among CBM-family papers reviewed on this blog:

- **vs CBM-Koh (ICML 2020).** ProbCBM is strictly downstream of CBM-Koh's *sequential* regime (concept head first, class head second). The qualitative leap is from $\hat c \in \mathbb{R}^k$ (a vector of probabilities/logits) to $\{(\mu_c, \sigma_c)\}_c$ (a distribution per concept). Practically, CBM-Koh's intervention overwrites $\hat c_j$ with the oracle value; ProbCBM's intervention additionally **collapses $\sigma_{c_j} \to 0$**, so the same edit also reduces class uncertainty rather than only the point prediction. The accuracy comparison is essentially a wash; the value of ProbCBM is the uncertainty axis CBM-Koh does not have.
- **vs Bayesian-CBM-LLM (Feng et al., NeurIPS 2025; on this blog).** Both prepend "probabilistic/Bayesian" to CBM but operate at completely different levels. BC-LLM is **Bayesian over the concept set itself** — using a Metropolis-within-Gibbs sampler with an LLM as proposal/prior to *discover* which concepts belong in the bottleneck from a (potentially infinite) candidate space; the classifier on top is plain logistic regression on binary extractions. ProbCBM is **probabilistic over embeddings of a fixed, human-given concept set** — concepts are taken as known a priori, but each concept's presence in *this* image is a Gaussian rather than a hard bit. The two are largely orthogonal and stackable in principle (BC-LLM to discover the concept set, then ProbCBM-style embeddings to model per-image ambiguity), and neither paper notes this combination. BC-LLM's win is on medical (MIMIC-IV) accuracy/AUC/Brier; ProbCBM's win is on calibration-style proxies on natural images.

The cleanest framing: CBM-Koh gave radiologists an editable intermediate variable, BC-LLM lets the concept set itself be data-driven, and ProbCBM puts a calibrated "how sure am I?" on every concept slot. If you want the *intervention-as-queue* behaviour shown in Figure 11 of ProbCBM transferred to clinical imaging, you would build it on top of a CBM-Koh-style pipeline trained with the ProbCBM head — that experiment has not been run.

## References

- Kim, E., Jung, D., Park, S., Kim, S., Yoon, S. *Probabilistic Concept Bottleneck Models.* ICML 2023 (PMLR vol. 202). arXiv:2306.01574.
- Paper PDF: <https://arxiv.org/abs/2306.01574>
- Code: <https://github.com/ejkim47/prob-cbm>
- Related work: CBM-Koh et al. (ICML 2020, arXiv:2007.04612); Concept Embedding Models / CEM (Espinosa Zarlenga et al., NeurIPS 2022); Hedged Instance Embeddings / HIB (Oh et al., ICLR 2019); Probabilistic Embeddings for Cross-modal Retrieval (Chun et al., CVPR 2021); concept leakage in CBMs (Mahinpei et al., 2021); Bayesian-CBM-LLM (Feng et al., NeurIPS 2025).
