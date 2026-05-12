---
title: "Concept Bottleneck Models"
excerpt: "Modern CNNs routed through a human-concept layer match end-to-end accuracy on OAI (RMSE 0.418 vs 0.441) and enable test-time interventions that cut OAI RMSE from >0.4 to ~0.3 after correcting just 2 of 10 concepts."
categories:
  - Paper
tags:
  - CBM
  - Interpretability
  - Concept-Learning
  - Test-Time-Intervention
  - OAI
  - CUB
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- Concept Bottleneck Models (CBMs) revive the classical two-stage pipeline `x -> c -> y`: take a modern backbone (ResNet-18 for OAI knee x-rays, Inception-v3 for CUB birds), resize one layer to exactly **k concept units**, add a concept loss, and train under one of three regimes — Independent, Sequential, or Joint.
- On OAI, joint CBMs match the standard end-to-end model on task accuracy (**KLG RMSE 0.418 +/- 0.004 vs standard 0.441 +/- 0.006**) while supporting a unique inference-time affordance: replacing the predicted concept vector `c_hat` with the radiologist's true value `c` and recomputing `y_hat = f(c_hat)`. **Intervening on just 2 of 10 concepts drops OAI RMSE from >0.4 to ~0.3.**
- The honest read: the "no accuracy/interpretability tradeoff" framing is contradicted by the paper's own Fig 2-Left on CUB (where joint is 2.4 pts worse than standard and independent/sequential trail by ~7 pts), and the dramatic CUB intervention numbers are inflated by class-level concept denoising — every bird of the same species is *defined* to share a concept vector, so an oracle intervention leaks class identity.

## Motivation

End-to-end deep classifiers map pixels directly to labels and offer no native handle for the high-level vocabulary that clinicians actually use — "joint space narrowing", "bone spurs", "calcification". Existing interpretability is *post-hoc*: TCAV-style linear probes and Network Dissection can detect concepts in activations but cannot intervene on them, and may fail outright if the model never internalised the concept in the first place.

The medical-AI angle is unusually explicit. Knee osteoarthritis grading via the Kellgren-Lawrence scale (KLG) is *normatively defined* in terms of standard radiographic concepts, so a model whose intermediate representation aligns with those concepts is both auditable and editable by a radiologist at test time. The paper argues the historical accuracy-vs-interpretability tradeoff that killed early concept models (Kumar 2009; Lampert 2009) no longer holds when the concept layer is embedded inside a modern CNN.

## Core Innovation

- **A unified architectural recipe.** Take any backbone, resize one of its layers to have exactly `k` neurons matching the `k` human-annotated concepts, then put a small head `f` on top mapping `c_hat -> y_hat`. The recipe is agnostic to backbone choice and to whether the task is regression (OAI KLG) or 200-way classification (CUB species).
- **Three training regimes as a design knob.** *Independent* trains `g` and `f` separately, `f` on the true concepts. *Sequential* trains `g` first, then `f` on `g`'s predicted concepts. *Joint* minimises a combined loss with weight `lambda` (standard model is `lambda -> 0`; sequential is `lambda -> infinity`). Each occupies a different point on the intervenability-vs-base-accuracy frontier.
- **Test-time concept intervention as a first-class operation.** Unlike post-hoc interpretability, a CBM lets a human at inference time replace `c_hat_j` with the oracle `c_j` and recompute `y_hat = f(c_hat)`. The paper is the first systematic study of this as a knob a user can pull at inference, including a workaround for the classification case where `f` is trained on logits rather than sigmoided probabilities.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CBMs are competitive with standard end-to-end models on task accuracy | Table 1: OAI joint 0.418 +/- 0.004 vs standard 0.441 +/- 0.006 (better); CUB joint 0.199 vs standard 0.175 (worse by 2.4 pts); CUB independent/sequential ~7 pts worse | OAI, CUB | ⭐⭐⭐ on OAI; ⭐⭐ on CUB |
| C2 | CBMs achieve much higher concept accuracy by direct training than post-hoc linear probes | Table 2: OAI 0.53 vs 0.68 RMSE; CUB 0.03 vs 0.09 error; SENN probe also ~0.68 | OAI, CUB | ⭐⭐⭐ |
| C3 | Test-time intervention substantially improves task accuracy | Fig 4-Left: OAI RMSE drops from ~0.43 to ~0.15 (Independent, 10 concepts); CUB error drops from ~0.24 to <0.05 (Independent, all groups); 2 OAI interventions halves RMSE | OAI, CUB | ⭐⭐⭐ for Independent + nonlinear `c -> y`, with caveats |
| C4 | There is no fundamental tradeoff between task and concept accuracy | Fig 2-Left frontier + Tables 1-2 | OAI, CUB | ⭐⭐ — overstated; CUB shows a visible tradeoff |
| C5 | CBMs are more data-efficient (esp. on OAI) | Fig 2-Right: sequential bottleneck reaches standard accuracy at ~25% of OAI data; CUB advantage is small to zero | OAI, CUB | ⭐⭐ |
| C6 | CBMs are more robust to background covariate shift | Table 3 TravelingBirds: 0.482 vs 0.627 error | TravelingBirds (synthetic) | ⭐⭐ — one synthetic dataset, no GroupDRO/IRM baseline |
| C7 | The training regime is a meaningful tradeoff knob between intervenability and base accuracy | Fig 4: Independent wins at full intervention, loses without; Joint is the reverse | OAI, CUB | ⭐⭐⭐ |
| C8 | Intervention can fail when joint `lambda` is too small (control) | Fig 4-Left control curve: true-`c` substitution *increases* RMSE at `lambda=0.01` | OAI | ⭐⭐ |
| C9 | Linear-regression analysis suggests CBMs help when `k << d` and `sigma_C^2 << sigma_Y^2` | Sec 8 / App C theory | Theory only | ⭐ |

**Honest read.** The strongest, most reproducible claims are C2, C3 (with caveats), and C7. Direct concept supervision *does* yield much better concept-axis interpretability than probing; intervention *does* reliably improve accuracy when the architecture supports it; and the three training regimes *do* give a meaningful design knob.

Where framing exceeds evidence: **C1** is presented as a clean win, but on CUB the standard model is 2.4 pts better than the best CBM and 7 pts better than independent/sequential — "competitive accuracy" papers over this. **C4** ("no tradeoff") is directly contradicted by the paper's own Fig 2-Left for CUB, where the joint frontier is visible. **C6** rests on a single synthetic dataset with no dedicated robust-training baseline. Most importantly, **the CUB intervention numbers are inflated by class-level concept denoising**: the authors replace per-image binary attributes with class-level majority vote, so two birds of the same species are *defined* to share an identical concept vector. An oracle intervention then leaks class identity, not just concept knowledge — a known optimistic assumption acknowledged in Sec 6.2 but easy to miss when reading the headline curves. The intervention story is genuine, but on CUB the magnitudes are partly an artifact of the dataset preprocessing.

Variance is reported as +/- 2SD across random seeds (small seed count, not specified in the body); no significance tests; no external/multi-site replication; OAI is the only medical dataset and is single-cohort with single-rater concept labels.

## Method & Architecture

![CBM pipeline: x -> c -> y for OAI knee x-rays and CUB birds](/assets/images/paper/cbm-koh-2020/fig_p001_01.png)
*Figure 1: The canonical CBM pipeline — knee x-ray (OAI) routed through 10 clinical concepts (joint space narrowing, bone spurs, calcification, ...) to a KLG severity score, and bird photo (CUB) routed through 112 binary attributes to a 200-way species prediction.*

### 1. Setup

Training data are triples $(x^{(i)}, c^{(i)}, y^{(i)})$ with $c \in \mathbb{R}^k$ a vector of $k$ human-annotated concepts. The model factorises as $\hat{y} = f(g(x))$ where $g: \mathbb{R}^d \to \mathbb{R}^k$ predicts concepts and $f: \mathbb{R}^k \to \mathbb{R}$ predicts the target. *Concept accuracy* is how well $g(x)$ matches $c$; *task accuracy* is how well $f(g(x))$ matches $y$.

Architectural recipe: take a backbone (ResNet-18 for OAI; Inception-v3 for CUB), resize one layer to exactly $k$ neurons, and treat that layer's activations as $\hat{c}$. A small head $f$ — a 3-layer MLP on OAI, a single linear / logistic layer on CUB — maps $\hat{c} \to \hat{y}$.

### 2. Three training regimes

- **Independent.** Train $\hat{g}$ on $(x, c)$ pairs and *separately* train $\hat{f}$ on the **true** concepts $(c, y)$. At test time $\hat{f}$ receives $\hat{g}(x)$ — there is a train/test distribution shift for $\hat{f}$.
- **Sequential.** Train $\hat{g}$ as above, then train $\hat{f}$ on the **predicted** concepts $\hat{g}(x)$. No distribution shift, but $\hat{f}$ adapts to $\hat{g}$'s errors.
- **Joint.** Minimise

$$
\sum_i \big[\, L_Y(f(g(x^{(i)})); y^{(i)}) + \sum_j \lambda\, L_{C_j}(g(x^{(i)}); c^{(i)})\, \big]
$$

for fixed $\lambda > 0$. The standard end-to-end model is the $\lambda \to 0$ limit; sequential is $\lambda \to \infty$. Chosen $\lambda = 1$ on OAI, $\lambda = 0.01$ on CUB.

For sequential/joint with binary concepts, $f$ takes the **logits** $\hat{\ell} = \hat{g}(x)$ (not the sigmoided $\hat{c}$). This matters for intervention.

### 3. Loss functions

- **OAI:** MSE for both $L_Y$ (KLG regressed on 0-3 ordinal target) and $L_{C_j}$ (concepts are ordinal/continuous).
- **CUB:** cross-entropy over 200 species for $L_Y$; binary cross-entropy per attribute for $L_{C_j}$.

### 4. Test-time intervention

- **OAI (regression).** Replace $\hat{c}_j$ with oracle $c_j$, then recompute $\hat{y} = f(\hat{c})$. The ordering over the 10 concepts is computed once on a held-out validation set — input-independent.
- **CUB (classification).** Cannot directly overwrite $\hat{c}_j$ because $f$ uses logits $\hat{\ell}_j$. Workaround: set $\hat{\ell}_j$ to the **5th percentile** (if $c_j = 0$) or **95th percentile** (if $c_j = 1$) of $\hat{\ell}_j$ over the training distribution. Attributes are grouped (e.g. all wing-colour attributes form one group) and the oracle returns the true group value in one query. A "from sigmoid" variant trains $f$ on $\sigma(\hat{\ell})$ — easier to intervene, harder to optimise (CUB task error 0.224 vs 0.199).

### 5. Datasets

| Dataset | Modality | n | Task | Concepts | Granularity |
|---|---|---|---|---|---|
| OAI | Knee x-ray | 36,369 | KLG (4-level ordinal, first two grades merged -> regression on 0-3) | k=10 ordinal clinical attributes annotated by radiologists | **Instance-level** |
| CUB-200-2011 | Bird photos | 11,788 | 200-way species classification | k=112 binary attributes | **Class-level** — denoised by majority vote so all crows share a wing colour |
| TravelingBirds | CUB composited onto Places backgrounds | derived from CUB | 200-way species | same 112 attributes | Train: one fixed background per class; Test: backgrounds shuffled |

The CUB class-level denoising is the load-bearing preprocessing step. It is convenient but inflates intervention effectiveness, because the oracle is essentially returning class identity in disguise. OAI concept labels are predominantly **single-rater** (not consensus), so the "radiologist + model" intervention narrative also rests on optimistic ground truth.

## Experimental Results

### Main task & concept accuracy

| Model | OAI y RMSE | CUB y error | OAI c RMSE | CUB c error |
|---|---|---|---|---|
| Independent | 0.435 +/- 0.024 | 0.240 +/- 0.012 | 0.529 +/- 0.004 | 0.034 +/- 0.002 |
| Sequential | 0.418 +/- 0.004 | 0.243 +/- 0.006 | 0.527 +/- 0.004 | 0.034 +/- 0.002 |
| **Joint** | **0.418 +/- 0.004** | **0.199 +/- 0.006** | 0.543 +/- 0.014 | **0.031 +/- 0.000** |
| Standard | 0.441 +/- 0.006 | 0.175 +/- 0.008 | 0.680 +/- 0.038 (probe) | 0.093 +/- 0.004 (probe) |
| No bottleneck | 0.443 +/- 0.008 | 0.173 +/- 0.003 | — | — |
| Multitask | 0.425 +/- 0.010 | 0.162 +/- 0.002 | — | — |
| SENN [probe] | — | — | 0.676 +/- 0.026 | — |

Average Pearson correlation between predicted and true concepts is >=0.87 for all CBMs on OAI; F1 ~0.92 on CUB vs 0.77 for the linear probe baseline. The "no tradeoff" claim travels on OAI; on CUB the standard model is 2.4 pts better than joint and 7 pts better than independent/sequential.

![Task-vs-concept frontier and data efficiency](/assets/images/paper/cbm-koh-2020/fig_p005_01.png)
*Figure 2: (Left) Task-vs-concept-error frontier on OAI and CUB — essentially no tradeoff on OAI and a visible mild tradeoff on CUB. (Middle) Per-concept accuracy histograms. (Right) Data-efficiency curves where the OAI sequential CBM matches the standard model at ~25% of the training data.*

### Test-time intervention (Figure 4)

- **OAI nonlinear `c -> y`**: intervening on all 10 concepts drops Independent task RMSE to ~0.15; Sequential/Joint plateau around ~0.23. After only ~2 interventions, all bottleneck models drop below the standard model's RMSE.
- **OAI linear `c -> y`**: intervention is markedly less effective despite similar pre-intervention RMSE — Independent only reaches ~0.32 with all 10 concepts replaced. Authors flag this as unexplained.
- **OAI joint with `lambda=0.01`** (control): replacing $\hat{c}$ with true $c$ actually *increases* test error — concepts have drifted from their human meaning.
- **CUB**: intervention reduces error monotonically; Independent drops from ~0.24 to ~0.03 at full intervention (28 concept groups). Joint plateaus around ~0.10. "Joint from sigmoid" intervenes more naturally and catches up to Sequential, but starts from worse base error (0.224).

![Qualitative intervention examples](/assets/images/paper/cbm-koh-2020/fig_p007_01.png)
*Figure 3: Qualitative test-time intervention. Flipping a single concept value — joint-space narrowing on OAI; under-tail or throat colour on CUB — flips an incorrect prediction to the correct class.*

![Intervention sweep curves](/assets/images/paper/cbm-koh-2020/fig_p007_02.png)
*Figure 4: Intervention sweeps. (Left) OAI with nonlinear c -> y — Independent RMSE collapses to ~0.15 at full intervention; the joint control with lambda=0.01 actually gets worse, exposing concept misalignment. (Middle) OAI with linear c -> y — intervention much less effective (open puzzle). (Right) CUB — Independent reaches near-zero error at full intervention; "Joint from sigmoid" rescues intervenability at a cost to base accuracy.*

### Background-shift robustness (TravelingBirds)

| Model | y error | c error |
|---|---|---|
| Standard | 0.627 +/- 0.013 | — |
| **Joint** | **0.482 +/- 0.018** | 0.069 +/- 0.002 |
| Sequential | 0.496 +/- 0.009 | 0.072 +/- 0.002 |
| Independent | 0.482 +/- 0.008 | 0.072 +/- 0.002 |

All CBMs cut error by ~14-15 points vs the standard model on shuffled-background test data. But TravelingBirds is a *synthetic* composite, and there is no comparison to dedicated robust-training methods (GroupDRO, IRM) — so this is one data point, not a general claim.

![TravelingBirds background-shift illustration](/assets/images/paper/cbm-koh-2020/fig_p009_01.png)
*Figure 5: TravelingBirds — the same Black-billed Cuckoo class shifts from a forest-path background at train to a coffee-shop background at test, motivating the robustness experiment in Table 3.*

### Data efficiency

On OAI, the sequential bottleneck reaches standard-model accuracy with ~25% of the training data. On CUB the standard and joint models are slightly better in low-data regimes; sequential/independent lag. Again, the OAI advantage is real; the CUB advantage is not.

### Post-hoc probing

Linear probes on a standard ResNet/Inception cannot recover the concepts as accurately as a CBM reads them off the bottleneck: OAI concept RMSE **0.68 (probe) vs 0.53 (CBM)**; CUB concept error **0.09 vs 0.03**. SENN's representations are also hard to probe (OAI 0.68). This is a *necessary-but-not-sufficient* argument for training with concept supervision rather than recovering concepts post-hoc — and it is the cleanest result in the paper.

## Limitations

**Authors acknowledge.**
- CBMs require concept annotations at training time (extra labelling cost).
- The CUB intervention setting assumes humans never make mistakes and that class members share concepts.
- The linear `c -> y` failure on OAI is unexplained.
- TravelingBirds robustness depends on the *choice* of concept set — no guarantee an arbitrary concept set helps.
- Incomplete concept sets create a "side channel" problem: adding $x \to y$ shortcuts breaks clean intervention semantics.
- No human studies — claims about "radiologist + model > radiologist or model alone" are inferred from concept-replacement experiments only.

**Not addressed.**
- The OAI intervention ordering is **input-independent** (a fixed ranking on validation). No comparison to adaptive / per-example querying, which is flagged as future work but not run.
- No analysis of *which* concepts drive the gains, nor of correlated/redundant concepts in the OAI set of 10.
- The Independent model's advantage at full intervention but disadvantage at zero intervention is a covariate shift in the input to $f$ — noted but not fixed (e.g. by training $\hat{f}$ with noise injected on $c$).
- No comparison to attention-based or prototype-based interpretable models other than SENN; ProtoPNet, B-cos, etc. are absent.
- Concept supervision is treated as ground truth even though OAI concepts are single-rater and CUB concepts are denoised to class level. The very thing CBMs make auditable — concept alignment — is itself measured against noisy labels.
- No **adversarial intervention** test: what if a clinician occasionally edits a concept incorrectly?
- **No leakage analysis.** Joint training with logits can let $f$ exploit information in $\hat{\ell}$ that does not correspond to the binary concept value. Margeloiu et al. (2021) later show this is a serious issue with CBMs; it is not addressed here.

## Why It Matters for Medical AI

KLG knee osteoarthritis grading is *normatively defined* in terms of standard radiographic concepts, which makes it almost uniquely suited to a CBM: the human-concept layer matches the way radiologists actually score the image. The OAI result — joint CBM beats the standard model on RMSE while letting a radiologist override individual concepts at inference — is the strongest medical-AI argument in the paper, even though it relies on single-rater concept labels.

In the broader context of the medical-AI literature reviewed on this blog, CBMs occupy a different point in the design space from most concept-aware works that have come since. Post-CLIP medical VLMs (PLIP, BiomedCLIP, CONCH on this blog) try to *recover* human-aligned representations from natural-language supervision, with no intervention semantics. Prompt-tuning work (CoOp/CoCoOp, Biomed-DPT) inherits CLIP's frozen representation and edits only the prompt distribution. Prototype models (ProtoPNet) project to learned prototypes rather than human-named concepts. **CBMs are the only architecture in this lineage where a clinician can edit a named intermediate variable at inference and re-derive the prediction by direct forward pass through $f$.**

That said, the foundational CBM has well-known sequels (Margeloiu et al. 2021 on concept leakage; Yuksekgonul et al. on post-hoc CBMs; label-free CBMs using LLMs as concept proposers) that all attack weaknesses this paper introduces or sidesteps. Read this as the canonical reference for the *idea*, not as the final word on its implementation.

## References

- Koh, P. W.\*, Nguyen, T.\*, Tang, Y. S.\*, Mussmann, S., Pierson, E., Kim, B., Liang, P. *Concept Bottleneck Models.* ICML 2020 (PMLR vol. 119). arXiv:2007.04612 (v3, 29 Dec 2020).
- Paper PDF: <https://arxiv.org/abs/2007.04612>
- Code: <https://github.com/yewsiang/ConceptBottleneck>
- OAI dataset (gated NIH application): <https://nda.nih.gov/oai/>
- Related work: TCAV (Kim et al., ICML 2018), Network Dissection (Bau et al., CVPR 2017), SENN (Alvarez-Melis & Jaakkola, NeurIPS 2018), GroupDRO (Sagawa et al., ICLR 2020), Margeloiu et al. 2021 (concept leakage in CBMs), Yuksekgonul et al. 2023 (post-hoc CBMs), label-free CBMs (Oikarinen et al., ICLR 2023).
