---
title: "Scaling Medical Imaging Report Generation with Multimodal Reinforcement Learning"
excerpt: "UniRG fine-tunes Qwen3-VL-8B with two-stage SFT+GRPO and a RadCliQ-weighted clinical reward, topping the ReXrank chest X-ray leaderboard on all four boards (1/RadCliQ-v1 up to 5.14 on IU-Xray)."
categories:
  - Paper
  - CT-Report-Generation
  - LLM
permalink: /paper/scaling-medical-imaging-report-generation-multimodal-rl/
tags:
  - UniRG
  - Radiology-Report-Generation
  - GRPO
  - Reinforcement-Learning
  - Qwen3-VL
  - RadCliQ
  - CheXprompt
  - Chest-X-Ray
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- **UniRG** fine-tunes Qwen3-VL-8B-Instruct on ~563k chest X-ray studies from 80+ institutions with a two-stage **SFT + GRPO** recipe and a clinically grounded composite reward (BLEU + BERTScore + SembScore + RadGraph-F1 + CheXprompt).
- The reward design is the headline: **Step 1** weights BLEU/BERTScore/SembScore/RadGraph-F1 with the published RadCliQ-v1 regression coefficients (0, 0.370, 0.253, 0.377); **Step 2** adds an LLM-judge clinical-error reward $1/(\text{\#CheXprompt errors}+1)$ at weight 0.5 with KL=0.03 against the Step 1 policy.
- **UniRG-CXR sets a new ReXrank SOTA on all four boards** — 1/RadCliQ-v1 of **1.62 / 1.23 / 4.80 / 1.16** (Findings) and **1.59 / 1.07 / 5.14 / 0.70** (Findings + Impression) on ReXGradient / MIMIC-CXR / IU-Xray / CheXpert Plus, with IU-Xray and ReXGradient gains exceeding 50% over the prior best.

## Motivation

Radiology report generators trained with SFT alone overfit institution-specific lexical templates and over-optimize n-gram metrics that correlate weakly with radiological correctness. For safety-critical deployment that brittleness is unacceptable: a model has to generalize across institutions, demographics, and longitudinal follow-up workflows. UniRG's bet is that switching the optimization target from token-level cross-entropy to a **clinically grounded composite reward** — and adding RL on top of SFT — directly addresses both the cross-institution generalization gap and the metric-overfitting gap. The framing is medical-AI first: rewards are picked because RadCliQ regresses against radiologist judgments, and the second-stage error reward is gated by an LLM that scores explicit clinical mistakes.

A note on the title: **"scaling" here means scaling across institutions and metrics, not neural scaling laws.** There is no model-size sweep, no data-scaling curve, no compute-vs-quality plot in the paper.

## Core Innovation

- **Two-step GRPO with a curriculum of rewards.** Step 1 optimizes a RadCliQ-weighted linear combination of BLEU/BERTScore/SembScore/RadGraph-F1 with no KL term and a higher clip threshold (DAPO-style) to keep entropy from collapsing. Step 2 initializes from the Step 1 checkpoint, adds a CheXprompt-derived $1/(\text{errors}+1)$ reward at weight 0.5, and re-introduces a small KL=0.03 against the Step 1 policy.
- **Composite reward beats any single-metric RL on RadCliQ.** Single-metric RL trades off other metrics (BLEU RL: RadGraph $-2.1\%$; RadGraph RL: BLEU $-1.6\%$). The combined reward delivers the highest 1/RadCliQ-v1 gain ($+15.3\%$ over the SFT baseline) on MIMIC.
- **Universal across four leaderboard datasets.** A single training run produces a model that tops all four ReXrank boards in both Findings and Findings+Impression settings, plus a longitudinal variant that uses the prior CXR + prior report as additional context.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | UniRG-CXR sets a new ReXrank SOTA across all four datasets in both Findings and Findings+Impression. | Figure 1d, Figure 2a — leads on all 4 datasets x 2 settings on the public leaderboard. | ReXGradient, MIMIC-CXR, IU-Xray, CheXpert Plus | ⭐⭐⭐ |
| C2 | The combined reward avoids metric overfitting and improves metrics it was not trained for (e.g., RaTEScore). | Figure 2a (RaTEScore values shown for UniRG-CXR vs baselines); Figure 2b ablation showing single-metric RL hurts other metrics. | MIMIC (ablation); 4 datasets (RaTEScore) | ⭐⭐ |
| C3 | The RL step is what produces the gain. | **Partially supported.** Supp. Table 1: RL-alone (1/RadCliQ 0.950) actually *underperforms* SFT-alone (0.962); only **SFT+RL (1.110)** wins. Figure 2b deltas all start from the SFT checkpoint, so they are SFT+RL minus SFT, not pure RL minus SFT. | MIMIC | ⭐⭐ |
| C4 | Cross-institution / zero-shot generalization beats prior models. | Figure 4a: top across BLEU/BERTScore/SembScore/RadGraph/1/RadCliQ on held-out IU-Xray (IU train data excluded) and proprietary PD. | IU-Xray (zero-shot variant), PD | ⭐⭐ (no variance bands, one zero-shot fold) |
| C5 | Disease-level diagnostic F1 leads on prevalent and long-tail conditions. | Figure 4b–c: CheXbert-derived F1 across 14 MIMIC and 6 PD conditions, including Pneumothorax (n=81) and Pleural Other (n=106). | MIMIC, PD | ⭐⭐ |
| C6 | Robust and equitable across gender, age, race subgroups. | Figure 4d: subgroup means on CheXpert Plus exceed second-best dashed line. | CheXpert Plus | ⭐ (training data is 71% race-unknown, 35% age-unknown; no fairness test, no CIs) |
| C7 | Longitudinal training uses prior reports/images, not as a copy-prior shortcut. | Figure 3a: longitudinal > no-longitudinal; Figure 3b: clearly above grey copy-prior baseline; Figure 3c: **** significance markers across 4 temporal categories. | MIMIC test with priors | ⭐⭐⭐ |
| C8 | The CheXprompt error-awareness reward is what drives clinical-error reduction. | Figure 2d: training-step #errors/report drops monotonically with the error reward, flat without it; Figure 2f: 14.8% of UniRG-CXR reports have $\geq$4 errors vs. MedVersa 32.3% and MedGemma 43.5%. | MIMIC val (1024); MIMIC test | ⭐⭐ |
| C9 | A single model is universal across four datasets and many metrics. | Figure 2a radar plots across 4 datasets. | 4 ReXrank datasets | ⭐⭐ |

**Honest read.** The headline ReXrank SOTA (C1) and the longitudinal result (C7) are unambiguously well supported because the leaderboard provides apples-to-apples comparisons that the authors did not run themselves and the longitudinal panel includes significance markers and a copy-prior baseline. The most important methodological claim — that **RL is the active ingredient** — is the weakest part of the audit. Supp. Table 1 directly contradicts that reading: RL alone (0.950) is *worse* than SFT alone (0.962); only the combined SFT+RL (1.110) wins, and even then by a margin comparable to the SFT-only delta from baseline (0.625 → 0.962). Every Figure 2b ablation is a delta on top of an already-strong SFT checkpoint, so the paper does not isolate "what would have happened with more SFT epochs / more SFT data" as a control — there is **no SFT-scaling baseline** to rule out the simpler "more SFT" hypothesis. The "scaling" framing in the title is also under-evidenced: no model-size sweep, no data-scaling curve, no compute-vs-quality plot. Variance bands and significance tests are absent everywhere except the longitudinal markers in Figure 3c (test/method unspecified). Demographic robustness rests on a training mix that is 71% race-unknown.

## Method & Architecture

![UniRG overview: training data, GRPO+composite-reward pipeline, and ReXrank leaderboard headline](/assets/images/paper/scaling-medimg-rg-rl/page_003.png)
*Figure 1: UniRG overview — training data composition (~563k studies, 80+ institutions), the two-stage SFT + GRPO pipeline with a clinically grounded composite reward, and the ReXrank leaderboard headline showing UniRG-CXR topping all four boards.*

1. **Backbone.** Initialize from open-source **Qwen3-VL-8B-Instruct**. Inputs are a 512x512 frontal CXR plus context (indication + comparison text) and, when training the longitudinal variant, the most recent prior frontal CXR plus its report.
2. **Task.** Generate the `Findings:` and `Impression:` sections via a fixed prompt template. Inference temperature is 0.
3. **Stage 1 — SFT.** Train on the union of MIMIC-CXR, CheXpert Plus, ReXGradient, and IU train splits. Grid search over $\text{lr} \in \{1e\text{-}5, 5e\text{-}5\}$ and batch size $\in \{128, 256, 512\}$; best config lr=5e-5, bs=256, 3 epochs.
4. **Stage 2 — GRPO RL.** Group-relative advantages, no value head. Two DAPO-style tweaks: higher clip threshold to keep entropy from collapsing, and dropping the KL penalty in Step 1. Hyperparameters: lr=5e-6, global batch=256, 16 rollouts/query.
5. **Step 1 reward — RadCliQ-oriented.** Composite reward
   $$R_1 = 0\cdot\text{BLEU} + 0.370\cdot\text{BERTScore} + 0.253\cdot\text{SembScore} + 0.377\cdot\text{RadGraph-F1}$$
   matching the published RadCliQ-v1 regression coefficients. One epoch. BLEU's coefficient is literally zero — the model is never *directly* rewarded for BLEU yet still reports BLEU gains downstream.
6. **Step 2 reward — error-reduction.** Initialize from the best Step 1 checkpoint, add
   $$R_{\text{err} } = \frac{1}{\#\text{CheXprompt errors} + 1}$$
   at weight 0.5 alongside the Step 1 reward, and re-introduce a KL term with coefficient 0.03 against the Step 1 policy. One additional epoch. CheXprompt is GPT-4-backed.
7. **No format reward** is needed because SFT already produces well-structured `Findings/Impression` outputs.
8. **Longitudinal variant** is trained by adding the prior CXR + prior report into the context window during both SFT and RL — no architectural change.

**Training corpus.** Aggregate **563,494 studies / 785,687 images / 229,161 patients / 80+ institutions** from MIMIC-CXR, CheXpert Plus, ReXGradient-160k, and IU-Xray, with a separate proprietary PD set used only for evaluation. The training mix is 23% white / 4% asian / 2% black / 71% unknown race; 35% age unknown. All four corpora are US- or US-style.

## Experimental Results

![UniRG-CXR results across four datasets, single- vs combined-reward ablation, training dynamics, and clinical-error histogram](/assets/images/paper/scaling-medimg-rg-rl/page_005.png)
*Figure 2: Universal metric gains and reward-design ablations — radar plots across the four ReXrank datasets, single-metric vs combined-reward RL on MIMIC, training dynamics for RadCliQ and clinical-error count, and the CheXprompt error histogram against MedVersa and MedGemma.*

### ReXrank leaderboard (1/RadCliQ-v1, higher is better)

| Setting | ReXGradient | MIMIC-CXR | IU-Xray | CheXpert Plus |
|---|---|---|---|---|
| **UniRG-CXR (Findings)** | **1.62** | **1.23** | **4.80** | **1.16** |
| Prev best (Findings) | 1.35 (RadPhi4VisionCXR) | 1.10 (MedVersa) | 1.92 (MoERad-IU) | 1.05 (CheXOne-R1) |
| **UniRG-CXR (Findings+Impression)** | **1.59** | **1.07** | **5.14** | **0.70** |
| Prev best (F+I) | 0.98 (MedVersa) | 0.92 (MedVersa) | 1.45 (MedVersa) | 0.68 (CheXOne-R1) |

Per-metric on ReXGradient findings: BLEU 0.29, BERTScore 0.54, SembScore 0.58, RadGraph 0.30, **RaTEScore 0.62**. RaTEScore is *not* in the reward, so the concurrent gain is the paper's main "no metric overfitting" evidence.

### Reward ablation on MIMIC (% gain over SFT baseline)

| Variant | BLEU | SembScore | RadGraph | 1/RadCliQ-v1 |
|---|---|---|---|---|
| SFT baseline | 0.0 | 0.0 | 0.0 | 0.0 |
| + BLEU RL | +13.2 | +2.1 | $-2.1$ | +5.3 |
| + SembScore RL | +12.3 | +7.2 | +3.5 | +12.2 |
| + RadGraph RL | $-1.6$ | +15.7 | +9.7 | +13.1 |
| **Combined reward RL (UniRG-CXR)** | **+5.1** | **+6.1** | **+15.3** | **+15.3** |

### SFT vs RL vs SFT+RL on MIMIC findings+impression (Supp. Table 1)

| Method | BERTScore | SembScore | RadGraph-F1 | 1/RadCliQ |
|---|---|---|---|---|
| Baseline (no fine-tune) | 0.293 | 0.242 | 0.102 | 0.625 |
| SFT alone (3 ep) | 0.421 | 0.426 | 0.236 | 0.962 |
| RL alone (3 ep) | 0.437 | 0.435 | 0.198 | **0.950** |
| **SFT + RL (3 + 2 ep)** | **0.449** | **0.478** | **0.267** | **1.110** |

The single most important number in this table is that **RL alone (0.950) underperforms SFT alone (0.962)** on RadCliQ. The full pipeline wins, but the paper does not run an SFT-scaling control (more SFT epochs / more SFT data) to rule out the simpler hypothesis that the gain from 0.962 to 1.110 could have been partly bought with extra SFT.

### Clinical-error distribution on MIMIC (CheXprompt)

| Model | $\leq$1 err | 2 err | 3 err | $\geq$4 err |
|---|---|---|---|---|
| MedGemma | 8.1% | 21.0% | 27.4% | 43.5% |
| MedVersa | 16.1% | 21.0% | 30.6% | 32.3% |
| **UniRG-CXR** | **21.3%** | 32.8% | 31.1% | **14.8%** |

### Longitudinal report generation

![Longitudinal report generation: UniRG-CXR vs Maira-2 / GPT-5 / MedGemma / copy-prior across encounter buckets and temporal change categories](/assets/images/paper/scaling-medimg-rg-rl/page_006.png)
*Figure 3: Longitudinal evaluation on MIMIC studies that include a prior CXR + prior report. UniRG-CXR (longitudinal) > UniRG-CXR (no longitudinal) > Maira-2 (longitudinal) > Maira-2 (no longitudinal) > GPT-5 > MedGemma > GPT-4o ≈ copy-prior baseline, holding across 1st–5th+ encounter buckets and four temporal change categories (new development / no change / progression / regression). Significance markers are **** in panel (c).*

### Cross-institution generalization

![Zero-shot IU-Xray and PD generalization, condition-level F1 on 14 MIMIC and 6 PD conditions, and demographic-subgroup stability on CheXpert Plus](/assets/images/paper/scaling-medimg-rg-rl/page_007.png)
*Figure 4: Generalization study — zero-shot performance on held-out IU-Xray (IU training data excluded) and proprietary PD, condition-level CheXbert-derived F1 across 14 MIMIC and 6 PD conditions including long-tail Pneumothorax (n=81) and Pleural Other (n=106), and gender / age / race subgroup means on CheXpert Plus.*

## Limitations

**Authors acknowledge**

- Future work needed for instruction-following / interactive use.
- Future work needed for richer multimodal patient context (labs, prior imaging, clinical notes).
- Frontal-only inputs at 512x512; lateral views are ignored.

**Not addressed by the authors**

- **Title misnomer.** "Scaling" in the title refers to *scaling across institutions and metrics*, not the neural-scaling-law sense. There is no model-size sweep, no data-scaling curve, no compute-vs-quality plot.
- **No SFT-scaling control.** RL alone (0.950) underperforms SFT alone (0.962) on RadCliQ; the paper never runs more-SFT-epochs or more-SFT-data baselines to rule out the simpler hypothesis that part of the SFT+RL gain (0.962 → 1.110) is just additional SFT compute.
- No variance, no seed averaging, no statistical tests on the headline numbers (the only **** markers appear in the longitudinal panel without specifying a test).
- CheXprompt rewards use **GPT-4 as judge**; no analysis of judge bias or robustness to GPT-4 drift.
- RadCliQ-oriented Step 1 weights BLEU at 0, yet BLEU still improves downstream — a sanity-check ablation isolating the source of that BLEU gain would be informative.
- Demographic fairness analysis uses overlapping-distribution language but no equality-of-opportunity / TPR-gap test, on a training mix that is 71% race-unknown and 35% age-unknown.
- Reproducibility: all RL hyperparameters and reward weights depend on private CheXprompt cost; the proprietary PD set is eval-only, but the GPT-4 judge dependency is structural.
- **Inference cost of GPT-4-backed CheXprompt during RL training is not discussed** — this is a non-trivial scaling bottleneck for any group that wants to reproduce the recipe.

## Why It Matters for Medical AI

UniRG is the cleanest demonstration so far that a clinically grounded composite reward, optimized with GRPO on top of a strong SFT checkpoint, can move radiology report generation across an entire public leaderboard at once — including a long-tail condition slice and a longitudinal evaluation with a copy-prior baseline. The two practical takeaways for medical-AI practitioners are: (i) **mix metrics in your reward** rather than picking a favorite, because single-metric RL trades off other metrics that radiologists care about; and (ii) **structure RL as a curriculum** — first match a regression-derived clinical metric (RadCliQ), then add an LLM-judge error reward with a small KL anchor against the previous policy. The honest caveats are equally important: the SFT-only baseline already does most of the work on RadCliQ, RL alone underperforms SFT alone, the paper carries no SFT-scaling control, and the RL recipe depends on a GPT-4-backed reward whose cost and drift are not discussed. A reproducer should budget for the CheXprompt inference cost and add the SFT-scaling control the paper omits before declaring RL the active ingredient.

## References

- **Paper:** Liu, Q.\*, Zhang, S.\*, Qin, G.\*, Gu, Y., Jin, Y., Preston, S., Xu, Y., Kiblawi, S., Yim, W.-w., Ossowski, T., Naumann, T., Wei, M.†, Poon, H.† *Scaling Medical Imaging Report Generation with Multimodal Reinforcement Learning.* arXiv:2601.17151v1 [cs.CV], 23 Jan 2026. Microsoft Research.
- **Backbone:** Qwen3-VL-8B-Instruct.
- **RL algorithm:** GRPO (DeepSeekMath); DAPO-style clip-higher and KL-drop tweaks.
- **Reward components:** BLEU; BERTScore; SembScore; RadGraph-F1; RadCliQ-v1 (regression weights for the Step 1 composite); CheXprompt (GPT-4-backed clinical-error judge for Step 2).
- **Datasets:** MIMIC-CXR; CheXpert Plus; ReXGradient-160k; IU-Xray; proprietary PD (eval-only).
- **Leaderboard:** ReXrank.
- **Compared models:** MedVersa, MedGemma, Maira-2, RadPhi4VisionCXR, MoERad-IU, CheXOne-R1, GPT-4o, GPT-5.
