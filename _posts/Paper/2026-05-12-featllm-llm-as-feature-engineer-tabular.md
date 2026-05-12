---
title: "FeatLLM: Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning"
excerpt: "An LLM emits per-class rules once at training time; a tiny non-negative linear model scores them at inference, lifting average AUC from 70.26 to 77.86 at 4-shot across 11 tabular benchmarks."
categories:
  - Paper
tags:
  - FeatLLM
  - Tabular Learning
  - Few-Shot
  - LLM Feature Engineering
  - Rule Extraction
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- FeatLLM uses an LLM **only at training time** to produce per-class human-readable rules, parses them into binary features, and trains a tiny non-negative linear scorer on top — so inference no longer touches the LLM.
- Across 11 standard tabular datasets at 4/8/16 shots it averages **77.86 / 79.31 / 80.70 AUC**, vs TabLLM's 70.26 / 72.76 / 76.22. The "~10% on average" headline is real at 4-shot (+7.6) but **shrinks to +4.5 AUC by 16-shot**.
- Inference is **0.006 ms/sample** (matching XGBoost) vs 335–463 ms for per-sample LLM baselines — a four-to-five-orders-of-magnitude win that is the paper's most durable contribution.

## Motivation
LLMs carry rich domain priors ("older patients have higher disease risk") that tree-based tabular learners cannot access. Prior LLM-based tabular methods (TabLLM, TABLET, in-context prompting) exploit this but pay three costs: (a) one LLM forward pass *per test sample*, (b) parameter-efficient finetuning that requires an open checkpoint, (c) prompt overflow on wide tables. Communities (103 features) and Myocardial (111 features) both break TabLLM/TABLET outright. FeatLLM's bet is that you can extract the LLM's priors **once**, freeze them as discrete rules, and run inference on something the size of a logistic regression.

## Core Innovation
Three moves stacked together:
1. **Two-stage prompt.** Step 1 elicits causal/common-sense reasoning *without* the few-shot examples, so the LLM commits to feature-task relationships from prior knowledge alone. Step 2 then conditions on Step 1 + the $k$ examples to emit 10 rules per class in a fixed schema (`[Feature] >= [Value]`, `[Feature] is in [List]`, etc.).
2. **LLM-as-parser.** Rather than regex over messy rule text, a second prompt asks the LLM to write a Python function `df_input -> df_output` that materializes each rule as a binary column. The function is `exec()`'d.
3. **Non-negative linear scorer + bagging.** Logits are $\max(w_k, 0) \cdot z^i_k$ so each rule can only *support* its class. Repeat T=20 trials with temperature 0.5, feature/sample bagging, shuffled few-shot order; average probabilities. Bagging is what keeps Communities/Myocardial inside the context window.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Beats LLM-based tabular baselines by ~10% on average | Table 1: 77.86 vs TabLLM 70.26 at 4-shot (+7.6); 80.70 vs 76.22 at 16-shot (+4.5). Headline rounds up. | ⭐⭐ |
| C2 | Inference matches conventional ML, unlike per-sample LLM methods | Table 5: 0.006 ms vs 335–463 ms. Structurally obvious from the linear scorer at test time. | ⭐⭐⭐ |
| C3 | Works on >100-feature datasets where TabLLM/TABLET/In-context fail | Table 3: baselines N/A on Communities and Myocardial; FeatLLM produces numbers via bagging. | ⭐⭐ |
| C4 | Two-step reasoning suppresses spurious correlations | Fig. 5: FeatLLM degrades least when random Adult columns are appended to Heart. | ⭐⭐ |
| C5 | Robust to spurious correlations *in general* | Contradicted by Fig. 12: when noise comes from a related-domain dataset (Myocardial), noisy-rule ratio jumps from 1–2% to 7–8%. | ⭐ (over-generalized) |
| C6 | Method is "LLM-agnostic" | Table 15: only GPT-3.5 and PaLM 2 tested; results swing per dataset (Adult −4 to −5 AUC on PaLM; Myocardial +5 to +7). Overclaimed. | ⭐ |
| C7 | Ensembling/bagging is essential | Table 4: −Ensemble = −6.49 average AUC, the single biggest ablation. | ⭐⭐⭐ |
| C8 | LLM-generated rules are interpretable | Table 16 qualitative (`Glucose<100 / BMI<25 → No diabetes`). No clinician evaluation, no quantitative interpretability metric. | ⭐⭐ |
| C9 | 4-shot FeatLLM matches 32-shot conventional baselines | Fig. 4 curves cross around 32 shots; baselines catch up by 64. | ⭐⭐ |
| C10 | LLM is extracting priors, not memorizing labels | Post-Sept-2023 datasets (Cultivars, NHANES): mixed — STUNT beats FeatLLM on Cultivars at 4-shot. | ⭐⭐ |
| C11 | Non-negative weight projection is principled | Asserted, never ablated against a free linear layer. | ⭐ |
| C12 | Parsing failures are minor; GPT-4 would fix them | Table 13 shows **up to 40.33% parsing failure on Credit-g**, 35% on Bank. No GPT-4 ablation in main paper. | ⭐ |

## Method & Architecture

![FeatLLM pipeline](/assets/images/paper/featllm/fig_p003_01.png)
*Figure 1: FeatLLM pipeline — a bagged sample is serialized into a prompt; the LLM emits per-class rules; rules are parsed by a second LLM call into binary features; a non-negative linear model scores classes; T=20 trials are ensembled.*

The linear scorer (Eq. 2 in the paper):

$$\text{logit}^i_k = \max(w_k, 0) \cdot z^i_k, \qquad p^i = \mathrm{Softmax}([\text{logit}^i_1, \ldots, \text{logit}^i_c])$$

where $z^i_k \in \{0,1\}^R$ is the binary rule-hit vector for class $k$ ($R=10$). Trained with Adam (lr 0.01, 200 epochs) and 2- or 4-fold CV for early stopping. Backbone is `gpt-3.5-turbo-0613`; about 30 API calls total for 16-shot training on Adult.

## Experimental Results

**Main comparison (Table 1 averages, 11 datasets, AUC %).**

| Method | 4-shot | 8-shot | 16-shot |
|---|---:|---:|---:|
| LogReg | 65.47 | 72.03 | 76.33 |
| XGBoost | 50.00 | 60.52 | 69.72 |
| SCARF | 58.22 | 62.18 | 71.69 |
| TabPFN | 62.93 | 69.53 | 74.37 |
| STUNT | 62.36 | 67.47 | 69.72 |
| In-context (GPT-3.5) | 68.44 | 70.41 | 72.72 |
| TABLET | 68.69 | 70.53 | 73.02 |
| TabLLM (T0 + IA3) | 70.26 | 72.76 | 76.22 |
| **FeatLLM (Ours)** | **77.86** | **79.31** | **80.70** |

Note the XGBoost row: **50.00 AUC at 4-shot on every dataset** indicates a degenerate baseline collapsing to a single class. The paper does not re-run XGBoost with class weights and never benchmarks CatBoost, LightGBM, or a tuned GBDT — the obvious modern competitors. The "few-shot win" is real; the "feature engineering replaces tabular models" framing is not established.

![Shot scaling](/assets/images/paper/featllm/fig_p008_01.png)
*Figure 2: Averaged AUC across all 13 datasets vs number of training shots. FeatLLM holds 76–82 AUC across 4–64 shots while XGBoost and RandomForest only close the gap around 32–64 shots.*

**Wide-feature datasets (Table 3).** TabLLM, TABLET, TabPFN, and in-context are N/A on Communities (103 feat) and Myocardial (111 feat) because the prompt overflows. FeatLLM survives via feature bagging. But on **Myocardial at 16 shots FeatLLM scores 55.32 vs STUNT 61.22 and LogReg 60.00** — a loss on a clinical dataset that the abstract never mentions.

**Robustness — and its caveat.**

![Spurious columns from Adult](/assets/images/paper/featllm/fig_p009_01.png)
*Figure 3: Heart dataset AUC as 0–5 random unrelated columns from Adult are appended. FeatLLM degrades least; per-sample LLM baselines collapse.*

![Spurious columns from Myocardial](/assets/images/paper/featllm/fig_p022_01.png)
*Figure 4 (Appendix I): When the appended noise columns come from a domain-related dataset (Myocardial → Heart), FeatLLM becomes unstable and the share of rules drawing on noisy columns jumps from 1–2% to 7–8%. The robustness claim of Fig. 3 is conditional on noise being out-of-domain.*

**Ablations.**

![Rules per class](/assets/images/paper/featllm/fig_p009_02.png)
*Figure 5: AUC vs rules-per-class. R=10 is the sweet spot across all shot counts; R=30 overfits in the 4-shot regime.*

![Ensemble size](/assets/images/paper/featllm/fig_p017_01.png)
*Figure 6: AUC vs ensemble size T. Gains plateau around T=20, the default in the main experiments.*

From Table 4 (deltas vs full FeatLLM, averaged over 13 datasets): **−Ensemble = −5.4 to −7.4 AUC** (largest single ablation, confirming C7); −Tuning = −1.4 (4-shot) → −5.8 (32-shot); −Reasoning Step 1 = −5.0 at 4-shot, only −1.5 at 16-shot; −Description = −0.3 to −1.8.

**Inference cost (Table 5, Adult).** FeatLLM 0.006 ms/sample, same order as XGBoost; TabLLM 335 ms; In-context 463 ms. Training is 860 s, mostly LLM API calls, vs TabLLM's 251 s of GPU finetuning.

**Parsing reliability (Table 13).** Up to **40.33% of LLM-generated parsing functions fail on Credit-g**, 35% on Bank, 35% on Sequence-type. The paper waves at GPT-4 as a fix but never runs the experiment.

## Limitations
- **No modern GBDT baseline.** No CatBoost, no LightGBM, no tuned XGBoost — and the reported XGBoost is degenerate (50.00 AUC at 4-shot). The cleanest "are LLM features actually better than raw features in a strong tabular model" comparison is missing.
- **Headline overclaim.** The "~10% on average" gap is +7.6 at 4-shot but only **+4.5 by 16-shot**, and FeatLLM loses on Car at every shot count (72.69 / 73.26 / 79.43 vs TabLLM 85.82 / 87.43 / 88.65) and on Myocardial at 16-shot (55.32 vs STUNT 61.22).
- **Parsing brittleness.** Up to 40% parser-function failures on Credit-g; the proposed GPT-4 fix is unverified.
- **"LLM-agnostic" is two backbones.** GPT-3.5 vs PaLM 2 swings results by ±5 AUC per dataset — the claim is weak.
- **Robustness is conditional.** When spurious columns come from a domain-related dataset, the rule-noise ratio jumps 4–7×.
- **No data-contamination test** for Adult / Bank / Diabetes / Heart / Credit-g (all pre-2023 UCI staples).
- **No significance testing**, only 3 seeds with large stds (XGBoost ±20+ at 4-shot).
- **No clinician evaluation** of extracted rules despite four clinical datasets and an explicit medical pitch.
- **No regression, no >4-class classification, no calibration analysis.**
- The non-negative weight constraint is never ablated against a free linear layer.

## Why It Matters for Medical AI
Four of the 13 datasets (Diabetes, Heart, Myocardial, NHANES) are clinical, and the extracted rules — `Glucose < 100`, `Age >= 50 & ExerciseAngina = Y`, `BMI < 25 & Glucose < 100 → no diabetes` — read like textbook heuristics rather than learned regularities. This is the appeal: an auditable, interpretable feature layer for low-shot clinical prediction where neither raw GBDTs nor per-sample LLMs are practical. The honesty caveats matter more here than elsewhere, though: (a) the Myocardial regression (55.32 AUC) is on a clinical task, (b) the rules are never reviewed by clinicians, and (c) GPT-3.5 has plausibly seen Heart and Diabetes during pretraining, so the "extracts prior knowledge" framing slides toward "memorizes labels" without a contamination probe. As a template for "freeze LLM priors as features, then run cheap, auditable inference," the pattern generalizes; as a clinical pipeline, it still needs an external held-out site, a domain-expert rule audit, and a comparison to manually engineered clinical features.

## References
- Paper: Han, Yoon, Arik, Pfister. "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning." ICML 2024 (PMLR 235). arXiv:2404.09491.
- Backbone: `gpt-3.5-turbo-0613`; secondary: PaLM 2 Text-Unicorn.
- Closest baselines: TabLLM (Hegselmann et al., 2023), TABLET (Slack & Singh, 2023), TabPFN (Hollmann et al., 2023), STUNT (Nam et al., 2023).
