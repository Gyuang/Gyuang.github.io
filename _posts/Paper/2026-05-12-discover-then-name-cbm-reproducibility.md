---
title: "Revisiting Discover-then-Name Concept Bottleneck Models: A Reproducibility Study"
excerpt: "Reproducing DN-CBM (ECCV 2024) shows two of three original claims hold; an activation-weighted cosine-alignment penalty lifts mean concept-name alignment from 0.146 to 0.540 at 1-4pp accuracy cost, validated by a 203-person Wilcoxon study."
categories:
  - Paper
tags:
  - DN-CBM
  - Reproducibility
  - Concept-Bottleneck-Models
  - Sparse-Autoencoder
  - CLIP
  - Interpretability
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- This is a **TMLR June 2025 reproducibility study** of Rao et al.'s DN-CBM (ECCV 2024), not the original. The Amsterdam team confirms C1 (SAE-based concept discovery) and C3 (competitive classification accuracy) but **only partially reproduces C2** — beyond the top-aligned cherry-picked neurons, mid- and low-aligned SAE neurons routinely fail to match their assigned names.
- The authors' own contribution is an **activation-weighted, magnitude-normalized cosine-alignment penalty** added to the SAE objective (Eq. 7). It pushes each *firing* dictionary vector toward its assigned CLIP text embedding without rewarding dead neurons or activation inflation.
- Headline numbers at `C = 10^-4`: **mean cosine similarity 0.146 -> 0.540**, with accuracy costs of **ImageNet -2.18pp, CIFAR-10 -2.83pp, CIFAR-100 -4.12pp, Places365* -0.66pp**. A **203-participant Wilcoxon signed-rank study** confirms perceived interpretability rises significantly for mid- and low-aligned concepts (p < 0.001) while top-aligned ratings stay flat (p = 0.176).

## Motivation

DN-CBM is, on paper, the cleanest answer to "where do CBM concepts come from?": no human annotation (Koh et al. 2020), no GPT-3 list (Label-Free CBM), no prompt sweep — just sparse autoencoders over CLIP features plus a 20k-unigram dictionary lookup. The original ECCV 2024 paper reports strong accuracy and visually convincing neurons.

The Amsterdam group asks the unglamorous but critical question: when you train this end-to-end yourself, do the discovered neurons actually align with the words you slap on them? It matters because every downstream consumer of a DN-CBM treats those names as ground truth — if half the neurons have weakly-aligned labels, the explanations the model surfaces are mostly decorative. No medical-AI angle is evaluated; the experiments span CC3M / ImageNet / Places365* / CIFAR-10/100 / Waterbirds-100.

## Core Innovation

- **Faithful reproduction of DN-CBM** trained on CC3M (CLIP ResNet-50, SAE latent `h = 8192`, sparsity `lambda_1 = 3e-5`), with three probe configurations (v1 README-default, v2 standard classification, v3 concept intervention) and three independent probe runs for variance.
- **Alignment-aware fine-tuning loss (Eq. 7)** that augments the SAE objective with `-C * (phi(f(a)) / ||phi(f(a))||_2) * v`, where `v` is the per-neuron cosine score. Activation scaling means dead neurons get no bonus; L2 normalization prevents activation inflation from gaming the metric.
- **Two user studies**: a 22-person replication of Rao et al.'s Figure 6, and a purpose-built **203-person Wilcoxon comparison** stratified by alignment tertile on classification-relevant concepts. The second study is the strongest piece of new evidence in the paper.

## Claims & Evidence Analysis

| # | Claim | Source | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|---|
| C1 | SAE discovers human-understandable concepts without pre-selection | Rao et al. | Cosine-score distribution shape matches Rao Fig 2a | CC3M + 4 probe sets | ⭐⭐⭐ |
| C2 | Dictionary vectors align with their assigned text-embedded names | Rao et al. | Top-aligned look good; **low-aligned ("zealand", "silhouette", "commissioned", "sauna") visibly mismatched**; local-explanation neighbors include garbage even on correct predictions | 4 probe sets, 22-person Study 1 | ⭐ |
| C3 | DN-CBM is competitive in classification | Rao et al. | ImageNet 72.65 vs 72.9; CIFAR-10 86.71 vs 87.6; CIFAR-100 68.51 vs 67.5; Places365* 49.96 vs 53.5 | 4 probe sets | ⭐⭐⭐ |
| **E1** | **Activation-weighted cosine penalty raises mean alignment** | **Authors** | **Mean cosine 0.146 -> 0.540 at C=1e-4; 7-value sensitivity sweep** | **CC3M** | **⭐⭐⭐** |
| **E2** | **Fine-tuning improves perceived interpretability with statistical significance** | **Authors** | **Wilcoxon p=0.000 mid and low alignment; p=0.176 top; n=203 paired ratings** | **Places365*** | **⭐⭐⭐** |
| E3 | Accuracy trade-off is bounded ("slight decrease") | Authors | -2 to -4pp on ImageNet/CIFAR; Waterbirds **84.24 -> 67.39** before intervention | 4 probe sets + Waterbirds | ⭐⭐ |

**Honest read.** The reproduction is unusually well-executed: probe trained three times for variance, two user studies (one a faithful replication, one purpose-built), and a sensitivity sweep over the new hyperparameter `C`. The most important contribution is the *negative result on C2* — by quantifying low-aligned neurons that the original paper visualized but did not analyze, the Amsterdam team punctures the implicit "uniformly interpretable concepts" narrative. The extension is a credible patch, but: (a) `C` is selected qualitatively, not optimized; (b) the user study covers Places365* only — alignment generalization to CIFAR/ImageNet is shown only qualitatively (Appendix B); (c) top-aligned fine-tuned concepts drift into abstract tokens ("judaism") that never fire at inference, so the headline mean-cosine 0.540 overstates how aligned the *predictively active* neurons actually are; (d) only one SAE training run is reported, so all downstream analysis depends on which neurons CC3M happened to surface this time.

## Method & Architecture

![DN-CBM reproduction overview — ranked cosine distribution alongside top- and low-aligned task-agnostic concept grids](/assets/images/paper/discover-then-name-cbm/page_007.png)
*Figure 1: Page 7 of the paper — ranked cosine-similarity distribution (Fig. 2) alongside high- and low-aligned task-agnostic concept grids (Figs. 3-4). The visual gap between "plaid"/"burgundy" (top) and "zealand"/"sauna" (bottom) is the core piece of evidence against claim C2.*

### 1. Original DN-CBM (reproduced verbatim)

CLIP ResNet-50 image and text encoders, embedding dim `d = 1024`. The SAE is a Bricken et al. 2023-style dictionary learner with linear encoder `f(.)` (weights `W_E in R^{d x h}`), ReLU `phi`, and linear decoder `g(.)` (weights `W_D in R^{h x d}`), latent dim `h = 8192` (8x over-complete). The self-supervised loss is

$$
\mathcal{L}_{\text{SAE}}(a) = \lVert \text{SAE}(a) - a \rVert^2 + \lambda_1 \lVert \phi(f(a)) \rVert_1
$$

with `lambda_1 = 3e-5`. Training: CC3M for 200 epochs, batch 4096, lr 0.1 Adam, resample frequency 10.

**Naming step.** Vocabulary `V` = Google 20k unigrams. For neuron `c`, the dictionary vector is `p_c = [W_D]_{c,:}`. Assigned label:

$$
s_c = \arg\max_{v \in V} \cos\bigl(\angle(p_c, T(v))\bigr)
$$

and the cosine score `v_c = cos(angle(p_c, T(s_c)))` is the paper's main interpretability proxy.

**CBM construction.** Linear probe `h(.)` on top of `phi o f o I`: `t(x) = (h o phi o f o I)(x)`. The SAE encoder is frozen; the probe minimizes `CE(t(x), y) + lambda_2 ||omega||_1`. Three probe configs:
- **v1** (README defaults): lr 1e-2, `lambda_2 = 0.1`, batch 512, 200 epochs.
- **v2** (default per-dataset classification): lr 1e-3 (1e-2 for CIFAR-100), `lambda_2 = 1`.
- **v3** (concept intervention): lr 1e-1, `lambda_2 = 10`, plus top-5 weight pruning per class.

### 2. Alignment-aware fine-tuning (the paper's own contribution)

The fine-tuning loss adds an activation-weighted, magnitude-normalized cosine bonus:

$$
\mathcal{L}_{\text{FSAE}}(a) = \lVert \text{SAE}(a) - a \rVert^2 + \lambda_1 \lVert \phi(f(a)) \rVert_1 - C \cdot \frac{\phi(f(a))}{\max(\lVert \phi(f(a)) \rVert_2, \epsilon)} \cdot v
$$

where `v in R^h` is the per-neuron cosine score from the naming step. Two design choices matter: activation scaling by `phi(f(a))` so rarely-firing neurons are not pushed toward CLIP-aligned vectors they never use, and L2 normalization that prevents the optimizer from inflating activations to game the cosine bonus.

`C` is swept over `{1e-6, 1e-5, ..., 1e0}` with 30 extra fine-tuning epochs on CC3M each. The authors select `C = 1e-4` qualitatively — best Places365* validation accuracy and mean cosine 0.540. They explicitly note that no principled quantitative rule exists.

### 3. Evaluation protocols

- **Study 1 (replicating Rao Fig. 6):** 22 participants, 12 top-activating images per concept, 1-5 ratings for naming accuracy and consistency.
- **Study 2 (novel, evaluating the extension):** 203 participants. Classification-explanation concepts (appearing >5x as top-5 contributors on Places365* test) are ranked by cosine score and stratified into top-40 / middle-40 / bottom-40. Each participant rates 12 (concept, 5-image) pairs from 0-5. Paired comparison via **Wilcoxon signed-rank test**.
- **Waterbirds-100 intervention (v3).** Top-5 retained weights per class are manually flagged bird vs. non-bird; two interventions (keep-only-bird, remove-bird) are scored on worst-group accuracy.

### Hyperparameter summary

| Block | Hyperparam | Value |
|---|---|---|
| Encoders | text / image | CLIP ResNet-50 |
| SAE | latent `h` / embed `d` | 8192 / 1024 |
| SAE | `lambda_1` (sparsity) | 3e-5 |
| SAE | lr / batch / epochs | 0.1 / 4096 / 200 |
| Vocab | source / size | Google 20k unigrams / 20,000 |
| Probe v2 | lr / `lambda_2` | 1e-3 (CIFAR-100 1e-2) / 1 |
| Fine-tune | `C` (selected) | 1e-4 |
| Compute | total / CO2 | **91.91 A100-hours / 10.87 kg CO2-eq** |

## Experimental Results

### Reproducing original DN-CBM (Table 3)

| Model | ImageNet | Places365* | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| DN-CBM (Rao et al., original) | 72.9 | 53.5 | 87.6 | 67.5 |
| **DN-CBM (reproduced)** | **72.65 +/- 0.02** | **49.96 +/- 0.04*** | **86.71 +/- 0.02** | **68.51 +/- 0.04** |
| DN-CBM Fine-tuned (`C = 1e-4`) | 70.47 +/- 0.04 | 49.30 +/- 0.03* | 83.88 +/- 0.02 | 64.39 +/- 0.16 |

ImageNet, CIFAR-10 and CIFAR-100 reproduce within ~1pp. The Places365* -3.5pp gap is plausibly attributable to the 10% stratified subsample (the largest deviation from Rao et al.'s setup). The fine-tuned row shows a bounded but real accuracy cost — biggest on CIFAR-100 (-4.12pp).

### Task-agnostic concept grids (the C2 stress test)

![Top-aligned task-agnostic concepts: plaid, burgundy, turquoise, sweater](/assets/images/paper/discover-then-name-cbm/page_007.png)
*Figure 2: Top-activating images across ImageNet / CIFAR-10 / CIFAR-100 / Places365* for the **four highest-cosine concepts** ("plaid", "burgundy", "turquoise", "sweater"). The cherry-picked success case — visually coherent and matching their assigned names.*

![Low-aligned task-agnostic concepts: zealand, silhouette, commissioned, sauna](/assets/images/paper/discover-then-name-cbm/page_007.png)
*Figure 3: Same layout for **low-cosine concepts** ("zealand", "silhouette", "commissioned", "sauna"). Top-activating images are visibly incoherent and rarely match the assigned name — the central piece of evidence against claim C2.*

### Local explanations on original DN-CBM

![Local explanation, original DN-CBM, coherent case](/assets/images/paper/discover-then-name-cbm/page_008.png)
*Figure 4: A Places365 image where the reproduced DN-CBM surfaces thematically coherent concepts (greenhouse case).*

![Local explanation, original DN-CBM, failure case](/assets/images/paper/discover-then-name-cbm/page_008.png)
*Figure 5: A failure case where DN-CBM surfaces "kayaking", "dams", "trivium" for an escalator — concrete evidence that "explanations" can be incoherent even on correct classifications.*

### Cosine-score distribution after fine-tuning

Reproduced DN-CBM cosine scores range roughly -0.01 to 0.42 with mean **0.146**. After fine-tuning at `C = 1e-4`, the range expands to roughly 0 to 0.8 with mean **0.540** (Fig. 8a). At `C >= 1e-3` the top ~6000 neurons saturate at cosine ~ 1 but activations diffuse across many weakly-firing neurons — the meaningful trade-off curve sits at `C = 1e-4`.

### Extension qualitative results

![Fine-tuned task-agnostic concepts — the same four labels now show coherent images](/assets/images/paper/discover-then-name-cbm/page_011.png)
*Figure 6: After fine-tuning, the same four low-aligned concepts ("zealand", "silhouette", "commissioned", "sauna") now produce visibly coherent top-activating images — "zealand" -> New-Zealand-like landscapes, "silhouette" -> dark shadowy shapes, "commissioned" -> artworks and statues. Sauna remains uneven outside Places365*.*

### Side-by-side local explanations

![Original vs fine-tuned local explanations, Places365 rainforest](/assets/images/paper/discover-then-name-cbm/page_012.png)
*Figure 7: Local explanations for a Places365* "rainforest" image — original DN-CBM neighbors include junk concepts; the fine-tuned model surfaces "meadow", "fields", "flower".*

![Original vs fine-tuned local explanations, CIFAR-10 horse](/assets/images/paper/discover-then-name-cbm/page_012.png)
*Figure 8: CIFAR-10 horse classification — original DN-CBM lists "arnold", "cosmos", "labrador", "eleven", "pelican", "michigan", "aaliyah", "busty", "elephants"; fine-tuned DN-CBM lists "horses", "equine", "horseback", "meadow", "fields", "flower". Direct evidence that activation-weighted alignment fixes specifically the predictively active neurons.*

### Study 2: n=203 Wilcoxon comparison

The headline figure (Fig. 9) was not extracted from the PDF. The numerical result, summarized:

| Alignment group | Original DN-CBM (reproduced) | Fine-tuned (`C = 1e-4`) | Wilcoxon p |
|---|---|---|---|
| Top-40 (high alignment) | similar | similar | 0.176 (n.s.) |
| **Middle-40** | lower | higher | **0.000** |
| **Bottom-40 (low alignment)** | lower | higher | **0.000** |

The effect is concentrated on **mid- and low-aligned concepts** — precisely the regime where the original DN-CBM's claim C2 was weakest. Top-aligned fine-tuned concepts often drift into abstract tokens ("judaism") that never fire at inference, an artifact the authors flag honestly.

### Concept interventions on Waterbirds-100 (Table 2)

Reproduced DN-CBM (v3):

- Before intervention: overall **84.24%**, worst groups L.Bird@W **76.30%**, W.Bird@L **58.27%**.
- Keep-only-bird concepts: overall **88.07% (+3.8)**, W.Bird@L **82.73% (+24.5)** — large worst-group recovery.
- Remove-bird concepts: overall **59.47% (-24.8)**, W.Bird@L **15.83% (-42.4)** — concepts are causally used.

Fine-tuned DN-CBM (v3): starts at **67.39%** overall (fine-tuning hurts Waterbirds more than other datasets, since v3's top-5 retention does not match the more diffuse fine-tuned representation), but keep-only-bird intervention still reaches **83.24% (+15.8)**, W.Bird@L **75.54% (+56.1)**.

### Study 2 user-survey interface

![Example user-study survey UI for concept "Tomb"](/assets/images/paper/discover-then-name-cbm/page_015.png)
*Figure 9: Example survey question from Study 2 — participants rate how well a single name (here "Tomb") describes a 5-image set, from 0 to 5. The pairing structure for the Wilcoxon test is at the concept level (random sampling without replacement), not the participant level.*

## Limitations

**Authors flag:**
- Places365 reproduction uses a **10% stratified subsample** (`Places365*`), so the -3.5pp accuracy gap is not strictly comparable.
- **CC3M link-rot** means the actual SAE training set differs from Rao et al.'s — an uncontrolled drift in the SAE itself.
- `C` is selected qualitatively; no principled quantitative criterion is proposed.
- Cosine similarity may be a poor alignment proxy in high dimensions (Aggarwal 2001).
- Fine-tuned models under-perform v3 Waterbirds interventions because the top-5 retention rule does not match the more diffuse representation.
- Fine-tuned explanations sometimes mirror the class name directly (e.g., "greenhouse" explains "greenhouse"), reducing explanatory value. Excluding class-name-similar concepts from the probe is hypothesized but not tested.

**Not addressed (this reviewer's notes):**
- **No medical or specialized-domain evaluation.** Vocabulary is Google 20k English unigrams; no clinical vocabulary, no radiology or pathology imagery. The same limitation Label-Free CBM flagged.
- **Single SAE training run.** Every downstream conclusion hinges on which neurons CC3M happened to surface this time — no SAE-seed variance is reported.
- **No comparison to LLM-driven CBMs** (LF-CBM, LaBo) at equal probe complexity — so the "competitive with state-of-the-art" portion of C3 is not retested.
- **Fairness / vocabulary audit.** Some assigned names (e.g., "busty", "aaliyah") indicate the unigram list contains problematic content that leaks into "explanations" — not audited.
- **Wilcoxon test pairing structure.** Participants rate different concepts per group (random sampling without replacement), so the pairing is at the concept level — defensible but should have been spelled out.

## Why It Matters for Medical AI

This reproduction does not evaluate medical data, but the methodological lesson generalizes uncomfortably. Many recent medical-AI explainability pipelines treat SAE neurons or CBM concepts as labeled atoms — exposing them in dashboards, radiology read-room overlays, or audit reports as if the assigned name were a faithful description. The Amsterdam team's finding is that **roughly half of DN-CBM neurons are mid- or low-aligned and their names visibly do not match what activates them**, even on general-domain CC3M with a generous 20k-unigram vocabulary. In a clinical setting — where vocabularies are smaller, image distributions are narrower, and downstream consumers literally include physicians — that drift becomes a deployment risk: the explanation displayed beside a prediction may have nothing to do with the model's actual feature.

The activation-weighted alignment fix is also instructive. It does not improve interpretability uniformly; it improves it **specifically for the neurons that actually fire at inference**. For medical CBM designs, this is exactly the right invariant — the explanations users see are derived from active features, not from dead dictionary atoms — and is a cleaner target than "make the whole dictionary CLIP-aligned in cosine".

## References

- Byrman, Kasteleyn, Kuipers, Uyterlinde. *Revisiting Discover-then-Name Concept Bottleneck Models: A Reproducibility Study.* TMLR, June 2025. OpenReview ID `946cT3Jsq5`.
- Rao, Mahajan, Bohle, Schiele. *Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery.* ECCV 2024.
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Anthropic, 2023.
- Oikarinen et al. *Label-Free Concept Bottleneck Models.* ICLR 2023.
- Koh et al. *Concept Bottleneck Models.* ICML 2020.
- Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020 (Waterbirds).
- Aggarwal et al. *On the Surprising Behavior of Distance Metrics in High Dimensional Space.* ICDT 2001.
