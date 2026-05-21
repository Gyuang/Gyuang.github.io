---
title: "MFPO: Modality-Fair Preference Optimization for Trustworthy MLLM Alignment"
excerpt: "A two-headed DPO objective (text-side + region-perturbed image-side) plus a chosen-anchor margin and entropy-binned curriculum drops LLaVA-v1.5-7B Object HalBench CHAIRs from 53.6 to 13.4 — though 'modality fairness' is operationalised qualitatively from reward bars rather than as a formal metric, and V-DPO is conspicuously missing from the comparison."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/mfpo/
tags:
  - MFPO
  - DPO
  - mDPO
  - RLHF-V
  - Hallucination
  - Preference-Optimization
  - LLaVA
  - SAM
  - Diffusion-Noise
  - Curriculum-Learning
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- **MFPO is two parallel DPO heads sharing one policy:** $L_\text{text}$ contrasts chosen vs rejected captions on the *original* image, $L_\text{image}$ contrasts the original vs a *region-perturbed* image while holding the chosen response fixed — forcing the model to ground answers in pixels rather than text shortcuts.
- **Image negatives are not random crops.** A multipartite-graph PageRank picks salient keywords from the chosen caption, SAM grounds each keyword to a region, and diffusion noise (best at **500 steps**) is applied *only* to those regions. A chosen-anchor margin loss $L_\text{margin}$ (best $\eta=0$) and an easy-to-hard semantic-entropy curriculum complete the recipe.
- **Headline:** LLaVA-v1.5-7B + MFPO drops Object HalBench CHAIRs **53.6 → 13.4** and lifts MMHal-Bench score **2.07 → 2.69**, beating GPT-4V on 5/8 metrics — but GPT-4V still wins MMHal Score (3.49 vs 2.69), AMBER Coverage (67.1 vs 55.7) and HalRate (0.28 vs 0.49), and the "fairness" claim rests on a *qualitative* reward-gap visualisation, not a formal metric.

## Motivation

The authors run a striking diagnostic: take a text-DPO-aligned LLaVA-v1.5 and replace the input image with **heavily diffusion-noised** versions. Across 100 samples the answer changes only **9%** of the time, and visual-token attention drifts onto background pixels rather than the queried object (Figure 1c). The "trustworthy MLLM" reports in the DPO / RLHF-V / mDPO / HA-DPO / POVID line are, at least in part, memorisation of *text-side* preference patterns — the model has learned to match the chosen response, not to read the image.

![MFPO motivation: text-DPO models still answer correctly on heavily noised images and attend off-target](/assets/images/paper/mfpo/fig_p001_01.png)
*Figure 1: After standard text-only DPO, LLaVA-v1.5 still answers correctly on heavily noised images (only 9% gap across 100 samples), and visual-token attention drifts off the queried object — the authors' evidence that current preference optimization is memorising text shortcuts. Sample size is small (n=100, single backbone) and no confidence intervals are reported.*

MFPO's bet is that the fix is *modality fairness*: the preference objective has to receive gradient from the image branch with comparable magnitude to the text branch. The medical-AI relevance is indirect but real — the same shortcut behaviour is exactly what makes radiology MLLMs unreliable when image regions are subtly altered.

## Core Innovation

- **Region-perturbed image negatives.** Keywords are mined from the chosen caption via a multipartite-graph PageRank that combines positional influence, semantic cosine similarity, and a positional decay term; SAM grounds them to regions; diffusion noise is restricted to those regions. This is the differentiator vs mDPO's random crops and POVID's global-noise negatives.
- **Two parallel DPO heads, one policy.** $L_\text{text}$ varies the response with the image fixed; $L_\text{image}$ varies the image with the *chosen* response fixed. Same backbone, same $\pi_\text{ref}$, two loss terms.
- **Chosen-anchor margin $L_\text{margin}$.** Best $\eta = 0$, which collapses the "margin" to a pure regulariser on the chosen log-ratio — the framing as "margin" is therefore somewhat misnamed.
- **Easy-to-hard curriculum.** Per-sample semantic entropy bins the training set into thirds; the model is fine-tuned sequentially easy → medium → hard.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Current text-DPO MLLMs barely use the image — only 9% answer gap between clean and heavily noised inputs. | Figure 1b/c on 100 sampled images + attention visualisation. | Sample not specified beyond "100 images." | ⭐⭐ — striking but small n, single backbone (LLaVA-v1.5), no CI. |
| C2 | Fine-grained region perturbation > random 20% crop > global noise. | Table 8: CHAIRs 13.4 vs 16.5 vs 21.7. Figure 2 reward bars. | MMHal + ObjHal, 7B only. | ⭐⭐⭐ — clean monotone ablation across three conditions, replicated across two benchmarks. |
| C3 | $L_\text{text}$, $L_\text{image}$, $L_\text{margin}$ all contribute. | Table 4: drop $L_\text{image}$ → CHAIRs 21.9 (vs 13.4); drop $L_\text{text}$ → MMHal 2.34; drop $L_\text{margin}$ → 2.61. | MMHal + ObjHal. | ⭐⭐ — single-seed but deltas (esp. CHAIRs) are large enough to be credible. Note Table 4 row 2 ($L_\text{image}$ only) **underperforms full** — the joint objective is what matters, not $L_\text{image}$ in isolation. |
| C4 | Easy-to-hard curriculum beats end-to-end joint training. | Table 7: MMHal 2.69 vs 2.53; CHAIRs 13.4 vs 16.0. | MMHal + ObjHal. | ⭐⭐ — directionally supported but the gap (2.69 vs 2.53) is within plausible seed variance — no variance reported. |
| C5 | LLaVA-v1.5-7B + MFPO matches/exceeds 13B, 34B, and GPT-4V on 5/8 metrics. | Table 1. | MMHal, ObjHal, AMBER. | ⭐⭐ — true on the listed metrics but cherry-picked framing. GPT-4V still leads MMHal Score (3.49 vs 2.69), AMBER Coverage (67.1 vs 55.7), and HalRate (0.28 vs 0.49). |
| C6 | 40-point absolute drop in Object HalBench CHAIRs (53.6 → 13.4 on 7B; 46.3 → 11.4 on 13B). | Table 1. | ObjHal. | ⭐⭐⭐ — numbers check out; the abstract's "40% improvement" phrasing is loose (CHAIRs is an absolute-point metric, not a percentage). |
| C7 | MFPO improves general capability, not just hallucination. | Table 3: LLaVA-Bench conversation 53.3 → 65.8; captioning 53.4 → 60.0. | LLaVA-Bench only. | ⭐⭐ — one small GPT-4-judge eval, prone to variance. |
| C8 | Modality fairness is the *mechanism*: text and image rewards converge. | Figure 2 reward gap closing 2.03/0.35 → 2.87/1.13 → **4.05/3.80**. | One backbone, one dataset. | ⭐⭐ — visually compelling but **"fairness" is operationalised qualitatively as a reward-gap visualisation; the paper defines no formal fairness metric**. Reward magnitudes also depend on $\beta$/normalisation choices not isolated. |
| C9 | Keyword selection is 86% accurate; SAM grounding is 90%+. | Manual check, n=100 each. | Sampled from training data. | ⭐ — single annotator, no inter-rater agreement, no CI; the 14%/10% failure modes are reported as "still helpful" without verification. |

**Honest read.**

- The two strongest claims are **C2** (region-level > random > global, with three-way ablation across two benchmarks) and **C6** (raw CHAIRs drops are unusually large and consistent across MMHal, ObjHal, AMBER).
- **C1 is the motivation and the weakest evidentiary leg**: n=100, no confidence interval, single backbone, no seed variance. The whole framing of the paper rests on this number.
- **C8 — "modality fairness" — is qualitative.** Figure 2 shows the text/image reward gap closing from (2.03, 0.35) → (4.05, 3.80) as image negatives get finer-grained. No formal fairness metric is defined (no Wasserstein gap on reward distributions, no per-modality calibration score). "Fair" here means "the bars in Figure 2 look closer together," not a definition the field can test.
- **V-DPO is conspicuously absent from Table 1**, which is the obvious head-to-head visual-DPO baseline. The paper compares against DPO, mDPO, HA-DPO, HALVA, POVID, RLHF-V — but the closest visual-preference competitor is missing. Without V-DPO, the "ours is the modality-fair one" framing has a hole.
- **HA-DPO underperforms vanilla DPO** on MMHal Score (1.97 vs 2.14), which weakens the narrative that adding hallucination-aware structure to DPO automatically helps — the paper does not engage with this.
- **Table 9 (Silkie + RLHF-V) is the real strength against mDPO** — beating mDPO on *two* preference-source datasets isolates the contribution of fine-grained negatives + margin + curriculum, not just a better text source.
- **Table 4 row 2** ($L_\text{image}$ only) underperforms the full objective: the contribution is the joint optimisation, not the image branch alone.
- **Table 5 best $\eta=0$**: $L_\text{margin} = -\log\sigma(\beta\log\pi_\theta(y_w|t,m)/\pi_\text{ref}(y_w|t,m) - \eta)$ with $\eta=0$ degenerates to a pure chosen-anchor regulariser. Calling it a "margin loss" overstates what the term does.
- **C9 audits** (86% keyword accuracy, 90%+ SAM accuracy) are single-annotator with no seed variance.
- **No medical or OOD evaluation; no POPE, no HallusionBench, no MME-Hallucination.** The benchmark selection is narrow and the "5/8 metrics beat GPT-4V" headline is selective.

## Method & Architecture

![MFPO framework: parallel text-DPO and image-DPO losses sharing one policy, with margin loss and entropy-binned curriculum](/assets/images/paper/mfpo/fig_p003_01.png)
*Figure 2: MFPO framework. (a) $L_\text{text}$ — standard DPO on chosen vs rejected captions with the original image fixed. (b) $L_\text{image}$ — DPO on original vs region-perturbed image with the *chosen* response fixed. (c) $L_\text{margin}$ — chosen-anchor regulariser. (d) Easy-to-hard curriculum binned by per-sample semantic entropy.*

### 1. Multipartite-graph keyword selection

From the chosen response build a word graph $G = (V, E)$. Edge weight combines:

- **Positional influence:** $\theta_{ij} = \sum 1/(1 + |l_i - l_j|^\phi)$
- **Semantic similarity:** $S(k_i, k_j) = \frac{v(k_i)\cdot v(k_j)}{\|v(k_i)\| \|v(k_j)\|}$ weighted by $\gamma$
- **Positional decay:** $\kappa_{ij} = e^{-\lambda l_i}$

A weighted PageRank with damping $\alpha_\text{page}$ is iterated to convergence; the top-$K$ words are taken as image-relevant keywords. Manual audit on n=100 samples reports 86% precision (page 6–7).

### 2. Region mapping + diffusion perturbation

![MFPO image-negative construction: PageRank → SAM → region-only diffusion noise](/assets/images/paper/mfpo/fig_p004_01.png)
*Figure 3: Image-negative construction. Multipartite-graph PageRank selects top-K keywords from the chosen caption; SAM grounds each keyword to a region; diffusion noise is applied only inside those regions, producing fine-grained dispreferred images. When SAM fails (~10% of cases), global noise is used as fallback; LISA can raise mapping accuracy to 95%+ but is not the default.*

Each top-$K$ keyword is fed to SAM (Kirillov et al., 2023) to get a region $R_i$. Diffusion noise is applied *only inside* $R_i$:

$$
m' = \sqrt{\alpha_{\text{diff},t}} \cdot R_i + \sqrt{1 - \alpha_{\text{diff},t}} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

with $\alpha_{\text{diff},t} = \prod_{j=0}^{t} \beta_{\text{diff},j}$. Best schedule: **500 diffusion steps**.

### 3. Joint loss

Total objective $L_\text{total} = L_\text{text} + L_\text{image} + L_\text{margin}$.

**Text DPO** (Eq. 5), standard on $(y_w, y_l)$ given $(t, m)$:

$$
L_\text{text} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|t,m)}{\pi_\text{ref}(y_w|t,m)} - \beta \log \frac{\pi_\theta(y_l|t,m)}{\pi_\text{ref}(y_l|t,m)}\right)\right]
$$

**Image DPO** (Eq. 6), on (original $m$ vs perturbed $m'$) holding chosen $y_w$ fixed:

$$
L_\text{image} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|t,m)}{\pi_\text{ref}(y_w|t,m)} - \beta \log \frac{\pi_\theta(y_w|t,m')}{\pi_\text{ref}(y_w|t,m')}\right)\right]
$$

**Margin** (Eq. 7), SimPO-style chosen anchor:

$$
L_\text{margin} = -\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|t,m)}{\pi_\text{ref}(y_w|t,m)} - \eta\right)
$$

Best $\eta = 0$ (Table 5), which means $L_\text{margin}$ degenerates to a pure chosen-anchor pull-up — naming it "margin" is therefore loose. Best loss ratio $L_\text{text} : L_\text{image} : L_\text{margin} = 1:1:1$ (Table 6).

### 4. Easy-to-hard curriculum

Per-sample semantic entropy $H(P) = -\sum p_i \log p_i$ bins the training set into easy / medium / hard thirds. The model is fine-tuned sequentially: easy → medium → hard. Beats end-to-end joint training (MMHal 2.69 vs 2.53; CHAIRs 13.4 vs 16.0; Table 7).

### 5. Training stack

Backbone LLaVA-v1.5-7B/13B and LLaVA-v1.6-7B. Stages 1–2 are standard LLaVA pretraining/instruction-tuning; MFPO is stage 3. Text preference source: **RLHF-V** (Silkie used for cross-dataset robustness in Table 9). Image preference data is *constructed* from RLHF-V via the keyword → SAM → diffusion pipeline. Specific $\beta$, learning rate, batch size deferred to supplementary section 4 (not in main text).

## Experimental Results

Main trustworthiness comparison (Table 1). Lower is better for HalRate / CHAIR / Cog; higher for Score / Cover. MFPO rows are **bold**.

| Model | MMHal Score↑ | MMHal HalRate↓ | ObjHal CHAIRs↓ | ObjHal CHAIRi↓ | AMBER CHAIRs↓ | AMBER Cover↑ | AMBER HalRate↓ | AMBER Cog↓ |
|---|---|---|---|---|---|---|---|---|
| LLaVA-v1.5-7B | 2.07 | 0.59 | 53.6 | 25.2 | 7.8 | 51.0 | 36.4 | 4.2 |
| + DPO | 2.14 | 0.65 | 49.0 | 13.0 | 6.5 | 55.5 | 34.5 | 2.3 |
| + mDPO | 2.39 | 0.54 | 35.7 | 9.8 | 4.4 | 52.4 | 24.5 | 2.4 |
| + HA-DPO | 1.97 | 0.59 | 39.9 | 19.9 | 6.7 | 49.8 | 30.9 | 3.3 |
| + HALVA | 2.08 | 0.60 | 46.6 | 53.0 | 6.6 | 53.0 | 33.2 | 3.4 |
| + POVID | 2.08 | 0.56 | 48.1 | 24.4 | — | — | — | — |
| **+ MFPO (7B)** | **2.69** | **0.49** | **13.4** | **6.6** | **4.1** | **55.7** | **22.5** | **1.9** |
| LLaVA-v1.6-7B + MFPO | 2.89 | 0.45 | 10.6 | 5.1 | 3.1 | 58.8 | 18.7 | 1.1 |
| GPT-4V | 3.49 | 0.28 | 13.6 | 7.3 | 4.6 | 67.1 | 30.7 | 2.6 |
| LLaVA-v1.5-13B | 2.42 | 0.53 | 46.3 | 22.6 | 7.8 | 51.0 | 36.4 | 4.2 |
| + RLHF-V (13B) | 2.81 | 0.49 | 12.2 | 7.5 | 6.3 | 46.1 | 25.1 | 2.1 |
| + HALVA (13B) | 2.84 | 0.48 | — | — | 6.4 | 52.6 | 30.4 | 3.2 |
| **+ MFPO (13B)** | **2.94** | **0.42** | **11.4** | **4.6** | **3.4** | **56.1** | **19.4** | **1.4** |

**Ablations.**

- *Loss composition* (Table 4): dropping $L_\text{image}$ collapses CHAIRs from 13.4 to 21.9; dropping $L_\text{text}$ drops MMHal score to 2.34; dropping $L_\text{margin}$ to 2.61. **Note:** $L_\text{image}$ alone (Table 4 row 2) underperforms full — the joint objective is needed.
- *Image-preference construction* (Table 8): global noise → CHAIRs 21.7; random 20% region → 16.5; fine-grained keyword+SAM → **13.4**.
- *Modality reward balance* (Figure 2): text/image reward bars close from (2.03, 0.35) [text-only] → (2.87, 1.13) [text + cropped image] → **(4.05, 3.80)** [text + fine-grained image]. MMHal climbs 2.07 → 2.39 → 2.63. This is the *qualitative* "fairness" evidence.

![Diffusion-step sweep on CHAIR metrics: 500 steps is the sweet spot](/assets/images/paper/mfpo/fig_p007_01.png)
*Figure 4: CHAIRs and CHAIRi vs diffusion noise steps. 500 steps is the sweet spot — too little noise leaves rejected images indistinguishable, too much erases content and removes the preference signal.*

- *Curriculum* (Figure 5, Table 7): easy 2.37 → medium 2.55 → hard 2.69 (monotone); end-to-end alternative 2.53.
- *Cross-dataset robustness* (Table 9): trained on Silkie instead of RLHF-V, MFPO still beats mDPO (MMHal 2.67 vs 2.39; CHAIRs 14.7 vs 35.7). This is the cleanest evidence that the *MFPO objective* — not just the text-preference source — carries the gain.
- *General capability* (Table 3): LLaVA-Bench conversation 53.3 → 65.8, captioning 53.4 → 60.0 — alignment did not sacrifice perception.

## Limitations

**Acknowledged.**

- SAM fails on ~10% of keywords (mitigated by global-noise fallback or LISA at >95% accuracy, but LISA is not the default).
- Keyword extraction fails on ~14% of cases.

**Not addressed.**

- **No seed variance or confidence intervals on any table** — Tables 1, 4, 5, 6, 7, 8 are single-seed. Several reported deltas (curriculum 2.69 vs 2.53; margin η ablation 2.61 → 2.69) are within plausible seed noise.
- **"Modality fairness" is not formally defined.** The mechanism claim (C8) rests on Figure 2 reward bars closing — no Wasserstein distance, no per-modality calibration, no statistical test. Reward magnitudes also depend on $\beta$ and reference-normalisation choices that are not isolated.
- **V-DPO is missing from Table 1**, despite being the closest visual-DPO baseline. The "ours is the modality-fair one" framing has a hole until V-DPO is benchmarked head-to-head.
- **HA-DPO underperforming vanilla DPO** on MMHal Score is not discussed.
- **No POPE, HallusionBench, MME-Hallucination, or any medical/OOD benchmark.** The "5/8 metrics beat GPT-4V" headline is selective — GPT-4V still leads MMHal Score, AMBER Coverage, and HalRate.
- **C9 audits (86%/90%) are single-annotator** — no inter-rater agreement, no CI, no seed variance.
- **The "margin" loss with best $\eta=0$ is misnamed** — it reduces to a pure chosen-anchor regulariser, not a margin in the SimPO sense.
- **No compute-cost analysis.** SAM + multipartite graph + diffusion noise per training sample is expensive; GPU-hours not reported.
- **No human eval of generated outputs** — all trustworthiness metrics are automatic (CHAIR rule-based / GPT-4 judge).
- **Keyword graph is text-only.** If the chosen caption is short or wrong, the visual grounding signal degrades — not stress-tested.

## Why It Matters for Medical AI

The diagnostic in Figure 1 — *a text-DPO MLLM answers correctly on heavily noised images* — is exactly the failure mode that makes generalist VLMs unsafe for radiology. A model that has memorised text-side preference patterns will confidently report findings on a corrupted or wrong scan. MFPO's contribution (forcing the image branch into the DPO objective with region-level negatives) is the right direction for clinical trust, even if the paper itself never evaluates on medical data. Re-running MFPO with RLHF-V replaced by RadFM or MIMIC-CXR preference data, and with image negatives constructed by perturbing radiologist-annotated ROIs, is the obvious next step the authors do not take.

## References

- Paper: *Modality-Fair Preference Optimization for Trustworthy MLLM Alignment*, Jiang et al. arXiv:2410.15334 (v2, 6 Jun 2025); labeled CVPR 2025 by user.
- Related: DPO (Rafailov et al., 2023), mDPO (Wang et al., 2024), HA-DPO (Zhao et al., 2023), HALVA (Sarkar et al., 2024), POVID (Zhou et al., 2024), RLHF-V (Yu et al., 2024a), Silkie (Sun et al., 2023), SimPO (Meng et al., 2024).
- Image construction: SAM (Kirillov et al., 2023), LISA (Lai et al., 2023).
- Benchmarks: Object HalBench (Rohrbach et al., 2018), MMHal-Bench (Sun et al., 2023), AMBER (Wang et al., 2023a), LLaVA-Bench (Liu et al., 2023).
- Code repo: not stated in main text; supplementary referenced.
