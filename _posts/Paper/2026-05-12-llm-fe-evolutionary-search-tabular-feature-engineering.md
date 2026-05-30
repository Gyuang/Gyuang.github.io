---
title: "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers"
excerpt: "A FunSearch-style island-evolutionary loop turns the LLM into an implicit crossover operator over Python feature-transformation programs, lifting XGBoost classification mean rank to 1.42 across 19 datasets and 1.10 across 10 regression datasets."
categories:
  - Paper
  - LLM
permalink: /paper/llm-fe-evolutionary-search-tabular-feature-engineering/
tags:
  - LLM-FE
  - Tabular Learning
  - Evolutionary Search
  - LLM Feature Engineering
  - Program Synthesis
  - FunSearch
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- LLM-FE casts tabular feature engineering as **program search**: the LLM emits Python `modify_features(df_in) -> df_out` programs, an evaluator trains XGBoost/MLP/TabPFN on the augmented table, and high-scoring programs are fed back as in-context examples through a 3-island Boltzmann-sampled memory — FunSearch lineage applied to AutoML.
- Across 29 datasets with a **20-LLM-call budget**, LLM-FE reaches **mean rank 1.42 (19 classification)** and **1.10 (10 regression)** vs CAAFE 3.47 / OpenFE 3.63 / OCTree 4.05 / FeatLLM 5.11, with **10/10 regression Wilcoxon p < 0.001** and a clean ablation that isolates evolutionary refinement (−0.100) as a larger factor than domain knowledge (−0.061).
- The honesty caveats matter: **5/15 classification datasets show no significant gain** (blood, cdc-diabetes, cmc, heart, communities), the operator-bias mitigation is partly engineered via a curated complex-operator prompt that baselines do not receive, and post-cutoff gains shrink ~10× relative to plausibly-contaminated benchmarks.

## Motivation
LLM-driven tabular AutoML is crowded but each existing family has a structural ceiling. **FeatLLM** (ICML 2024) extracts class-conditional binary rules over the *original* feature space — atomic conditions, no composition, no feedback. **CAAFE** and **OCTree** iteratively refine a single program/rule chain — feedback but no diversity, so they get stuck in local optima. Küken et al. (2024) separately show that LLMs prompted for feature engineering disproportionately emit simple arithmetic (`+`, `−`, `×`, `/`, `abs`); domain knowledge alone does not push them toward complex operators.

LLM-FE's bet is that all three gaps close at once if you (a) widen the output space from rules to *executable transformation programs*, (b) replace single-candidate refinement with a *population* maintained across islands, and (c) bias the LLM with high-scoring programs as in-context examples so successful operators propagate.

## Core Innovation
Three moves stacked on top of the FeatLLM-style "LLM-as-feature-extractor" template:
1. **Program-space output, not rule-space.** The LLM completes `def modify_features(df_input) -> pd.DataFrame:` — arbitrary Python, executable, free to compose `GroupByThenMean`, `Residual`, `Sigmoid`, `Log` etc. FeatLLM's binary-rule output cannot construct new continuous features; LLM-FE's program output can.
2. **FunSearch-style multi-island memory.** Population is partitioned across $m=3$ islands seeded with identical simple programs; per iteration one island is picked, programs inside it are clustered by validation-score signature, a cluster is **Boltzmann-sampled** ($\tau_c$ anneals from $T_0=0.1$ as more programs accumulate), $k=3$ exemplars are formatted as `modify_features_v0/_v1/_v2`, and the LLM is asked to complete `_v3`.
3. **LLM as implicit crossover operator.** No explicit recombination — the LLM, conditioned on $k=3$ high-scoring parents, performs prompt-conditioned generation that empirically mixes features from multiple parents (Meyerson et al. 2024 "LM crossover" lineage). This is what lets LLM-FE keep improving for 20 iterations where the single-trajectory ablation plateaus by iteration 5.

A nuance the paper does not flag loudly: in 50% of calls an *alternate* instruction prompt explicitly enumerates a complex-operator list (`GroupByThenMin/Max/Mean/Median/Std/Rank`, `Frequency`, `Residual`, `Sigmoid`, `Square`, `Logarithm`, `Combine`, `CombineThenFreq`). The CAAFE/OCTree baselines do not receive a matching hint, so the operator-frequency comparison in Figure 4 partly measures prompt engineering rather than search.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | LLM-FE consistently outperforms SOTA FE baselines | Tables 2/3: mean rank 1.42 (classif) / 1.10 (regression); wins outright on 9/10 regression datasets, top-2 on most classification. | ⭐⭐⭐ |
| C2 | Evolutionary refinement is the dominant component | Ablation: removing it drops aggregate accuracy 0.687 → 0.587 (−0.100), larger than removing domain knowledge (−0.061) or data examples (−0.043). Figure 6 trajectory: greedy ablation plateaus by iter 5; full method climbs for 20. | ⭐⭐⭐ |
| C3 | Domain knowledge (feature names) is critical | Anonymizing features to `C1, C2, ...` drops to 0.626; with names, LLM-FE writes `Log_Cholesterol`, `proliferation_activity = Normal_Nucleoli × Mitoses`. | ⭐⭐⭐ |
| C4 | Method is LLM-agnostic | Table 11: GPT-3.5, Qwen2.5-72B, GPT-4o-mini, Gemini-2.5-Flash all hit ~0.836-0.842 aggregate XGBoost accuracy. Meaningful step up from FeatLLM's 2-LLM test. | ⭐⭐⭐ |
| C5 | Generalizes across downstream tabular models | XGBoost +0.020, MLP +0.046, TabPFN +0.011 over base; CatBoost +0.022. | ⭐⭐⭐ |
| C6 | Handles wide-feature datasets where baselines fail | AutoFeat fails (>12h) on 8 large datasets; FeatLLM fails on arrhythmia, eucalyptus; LLM-FE produces numbers everywhere. | ⭐⭐ |
| C7 | Mitigates LLM bias toward simple operators | Figure 4 shifts mass to `GroupByThen*`, `Residual`, `Sigmoid`; ~45% "complex" features vs negligible for CAAFE/OCTree. **But the alternate prompt explicitly enumerates these operators while baselines do not get the same hint — partly engineered.** | ⭐⭐ |
| C8 | Improvements survive past GPT-3.5 pretraining cutoff | Table 4 post-cutoff datasets: small consistent gains (+0.003 to +0.005). **~10× smaller than on plausibly-contaminated benchmarks (jungle_chess +0.100).** | ⭐⭐ |
| C9 | Statistically significant improvements over base | Wilcoxon: 10/10 regression p<0.001; 8/15 classification p<0.05. **5/15 classification not significant (blood, cdc-diabetes, cmc, heart, communities), directly contradicting "consistently outperforms"**; blood-transfusion regression is actually 0.747 → 0.743 (small loss). | ⭐⭐ |
| C10 | Features are interpretable and clinically meaningful | Figure 11: `Log_Cholesterol`, `proliferation_activity`; SHAP: 16.67% top-10 / 62.96% top-50. **No clinician evaluation; rediscovered features are textbook transformations any first-year resident would write.** | ⭐⭐ |
| C11 | More efficient than competitors at equal performance | Figure 3 Pareto: ~600 s, ~0.86 accuracy; CAAFE ~700 s, ~0.85; OCTree ~300 s, ~0.84. Note **b=3 programs/call → 60 candidate programs from 20 LLM calls**, vs 20 for CAAFE/OCTree. | ⭐⭐ |
| C12 | Features transfer across predictors | Table 13: XGBoost-discovered features improve MLP / TabPFN, but underperform native discovery (MLP native 0.791 vs transferred 0.763). | ⭐⭐ |
| C13 | Multi-island design is essential | Asserted but the 1-island vs 3-island number is buried in the appendix, not in the main paper. | ⭐ |
| C14 | Comparison is fair | The alternate prompt hardcodes OpenFE's advanced operator list for LLM-FE only; CAAFE/OCTree do not receive a matched operator hint. | ⭐ |

## Method & Architecture

![LLM-FE pipeline](/assets/images/paper/llm-fe/page_002.png)
*Figure 1: The LLM-FE loop. (a) Prompt builder picks one of 3 evolutionary islands, Boltzmann-samples a cluster, takes $k=3$ high-scoring `modify_features` programs as in-context examples, and asks the LLM to complete `modify_features_v3`. (b) Each of the $b=3$ generated programs is sandbox-`exec()`'d (30 s, 2 GB caps); survivors transform $X_\mathrm{tr}$. (c) XGBoost/MLP/TabPFN is trained on $T(X_\mathrm{tr})$ and scored on $T(X_\mathrm{val})$. (d) The scored program is appended to the *same* island only if it beats that island's current best — preserving island independence and elitism. 20 LLM calls × 3 programs/call = 60 candidate programs.*

The bilevel objective the search approximates:

$$\max_T \; E\!\left(f^*(T(X_\mathrm{val})), Y_\mathrm{val}\right) \quad \text{s.t.}\quad f^* \in \arg\min_f L_f(f(T(X_\mathrm{tr})), Y_\mathrm{tr}),$$

with $T \sim \pi_\theta(p_t)$ where $\pi_\theta$ is the LLM and $p_t$ is the iteration-$t$ prompt. The prompt itself has five blocks: instruction (with a 50% chance of injecting the complex-operator enumeration), dataset spec (feature names + descriptions + a few serialized rows in the TabLLM/FeatLLM format), the literal `evaluate()` Python function so the LLM "knows" what is being optimized, $k=3$ exemplars, and the `modify_features` stub. Hyperparameters: $T=20$ LLM calls, $m=3$ islands, $b=3$ programs/call, $k=3$ exemplars, temperature 0.8, Boltzmann $T_0=0.1$ / $N=10{,}000$. Default backbone: GPT-3.5-Turbo.

## Experimental Results

**Main classification (Table 2, XGBoost accuracy, mean over 5 splits).**

| Dataset | Base | AutoFeat | OpenFE | CAAFE | FeatLLM | OCTree | **LLM-FE** |
|---|---:|---:|---:|---:|---:|---:|---:|
| adult | 0.873 | — | 0.873 | 0.872 | 0.842 | 0.870 | **0.874** |
| balance-scale | 0.856 | 0.925 | 0.986 | 0.966 | 0.800 | 0.882 | **0.990** |
| breast-w | 0.956 | 0.956 | 0.956 | 0.960 | 0.967 | 0.969 | **0.970** |
| blood-transfusion | 0.742 | 0.738 | 0.747 | 0.749 | **0.771** | 0.755 | 0.751 |
| car | 0.995 | 0.998 | 0.998 | **0.999** | 0.808 | 0.995 | **0.999** |
| communities (103 ft) | 0.706 | — | 0.704 | 0.707 | 0.593 | 0.708 | **0.711** |
| covtype | 0.870 | — | **0.885** | 0.872 | 0.554 | 0.832 | 0.882 |
| credit-g | 0.751 | 0.757 | 0.758 | 0.751 | 0.707 | 0.753 | **0.766** |
| heart | 0.858 | 0.857 | 0.854 | 0.849 | 0.865 | 0.852 | **0.866** |
| jungle_chess | 0.869 | — | 0.900 | 0.901 | 0.577 | 0.869 | **0.969** |
| myocardial (111 ft) | 0.784 | — | 0.787 | **0.789** | 0.778 | 0.787 | **0.789** |
| tic-tac-toe | 0.998 | **1.000** | 0.994 | 0.996 | 0.653 | 0.997 | 0.998 |
| vehicle | 0.754 | **0.788** | 0.785 | 0.771 | 0.744 | 0.753 | 0.769 |
| **Mean rank (19 sets)** | 3.95 | 5.11 | 3.63 | 3.47 | 5.11 | 4.05 | **1.42** |

**Regression (Table 3, RMSE).** LLM-FE wins outright on 9 of 10 datasets; loses only insurance (5.069 vs OCTree 4.969). Mean rank 1.10 vs OpenFE 2.80. FeatLLM and CAAFE are excluded from the regression sweep because their open-source codebases are classification-only — honest disclosure, but it means the regression comparison silently drops two LLM-based competitors.

**Ablation and efficiency.**

![Ablation and Pareto plot](/assets/images/paper/llm-fe/page_008.png)
*Figure 2: (left) Classification ablation — w/o-Evolutionary-Refinement drops aggregate accuracy to 0.587, larger than removing domain knowledge (0.626) or data examples (0.644). (right) Pareto plot on the 8 large classification datasets — LLM-FE at ~600 s / ~0.86 accuracy dominates CAAFE (~700 s / ~0.85) and OCTree (~300 s / ~0.84). Caveat: the 20-LLM-call budget gives LLM-FE 60 candidate programs (b=3 per call) vs 20 for the baselines, so the API-call axis is matched but the program-count axis is not.*

**Domain-knowledge isolation.**

![Heart dataset](/assets/images/paper/llm-fe/page_022.png)
*Figure 3: Heart-Disease (XGBoost accuracy). LLM-FE (0.866) with domain knowledge beats LLM-FE without (0.856), OpenFE (0.854), and AutoFeat (0.857). The 1-point gap from anonymizing feature names is what isolates the domain-knowledge contribution — but Wilcoxon on Heart is not statistically significant, qualifying the win.*

**LLM-agnosticism (Tables 5, 10, 11).** XGBoost: base 0.820 → LLM-FE-Llama-3.1-8B 0.832 / LLM-FE-GPT-3.5 0.840. MLP: 0.745 → 0.768 / 0.791. TabPFN: 0.852 → 0.856 / 0.863. Across GPT-3.5, Qwen2.5-72B, GPT-4o-mini, Gemini-2.5-Flash the XGBoost-LLM-FE numbers are essentially equivalent (~0.836-0.842). This is the strongest cross-LLM evidence in the LLM-tabular literature so far.

**Operator-bias evidence (Figure 4).** Base LLM and CAAFE concentrate 75%+ of their operators on `multiply`, `divide`, `add`, `subtract`, `abs`. LLM-FE shifts mass to `GroupByThenMean`, `Residual`, `Sigmoid`, `Log`. The appendix claims ~45% "complex" features under Küken et al. 2024's taxonomy — but the alternate prompt explicitly enumerates these operators for LLM-FE only, so the comparison is partly engineered.

**Memorization control (Table 4).** Post-Sept-2021 datasets: kidney-stones 0.761 → 0.761 (no gain), health-insurance 0.756 → 0.759, pharyngitis 0.655 → 0.660, fico 0.715 → 0.719, acs-income 0.807 → 0.809. The gains are real but **~10× smaller** than on plausibly-contaminated jungle_chess (+0.100) — strongly suggestive that part of the headline win on UCI staples is contamination.

**Wilcoxon (Table 14).** 10/10 regression p<0.001 — clean. 8/15 classification p<0.05; 2/15 marginal; **5/15 not significant (blood, cdc-diabetes, cmc, heart, communities)**. The abstract's "consistently outperforms" is overstated; on Heart specifically — the flagship clinical interpretability example (Figure 11) — the improvement is not significant.

**HPO interaction (Table 9).** With Optuna 100-trial HPO, LLM-FE still beats base XGBoost on 3/5 difficult datasets — but on vehicle, OpenFE+HPO (0.810) beats LLM-FE+HPO (0.780). Feature engineering and HPO interact non-monotonically.

## Limitations
- **5/15 classification datasets show no statistically significant improvement** — blood, cdc-diabetes, cmc, heart, communities — directly contradicting "consistently outperforms".
- **Operator-bias mitigation is partly engineered**: the alternate prompt hardcodes OpenFE's advanced operator list for LLM-FE while CAAFE/OCTree do not receive a matched hint. Figure 4 partly measures prompt engineering, not search algorithm.
- **Post-cutoff gains are ~10× smaller** than on plausibly-contaminated UCI staples — the contamination signal is real even though the paper waves at it.
- **Clinical interpretability is anecdotal** — no clinician evaluation, no rule audit; the rediscovered "clinical" features (`Log_Cholesterol`, `BMI/Age`) are textbook transformations.
- **Multi-class evaluation uses accuracy, not AUROC or macro-F1**, biasing comparisons on imbalanced sets like myocardial (78:22), bank (88:12), cdc-diabetes (84:16).
- **No clean "tuned modern GBDT alone" baseline**: on vehicle, OpenFE+HPO beats LLM-FE+HPO. The question "does LLM-FE beat a well-tuned CatBoost/LightGBM with no feature engineering" is not answered.
- **20 LLM calls × 3 programs/call = 60 candidate programs**; CAAFE/OCTree get 20. The "matched API budget" framing is fair but the "matched candidate count" framing is not.
- **No analysis of program execution failure rate** under the 30 s / 2 GB sandbox filter — how many of the 60 candidates per dataset survive?
- **No cost reporting** in dollars/tokens — only wall-clock seconds.
- **FeatLLM comparison is modified** (XGBoost downstream with full training set, not FeatLLM's published linear scorer) — explained as fair but means the FeatLLM numbers in Table 2 are not directly comparable to FeatLLM's own paper.
- **Multi-island necessity** (C13) is asserted but not ablated in the main paper.
- **No fairness / subgroup analysis** on Heart, Diabetes, ACS-Income despite an explicit LLM-bias acknowledgment in the Impact Statement.

## Why It Matters for Medical AI
The medical pitch is honest but narrow: 7+ of the 29 datasets are clinical (Heart, Myocardial, CDC Diabetes, Breast-W, Pharyngitis, Kidney Stones, Health-Insurance), but they are all small UCI-style tabular sets — no time-to-event, no imaging, no real EHR cohort, no clinician evaluation of the discovered features. The natural sequel to the already-shipped [FeatLLM]({% post_url Paper/2026-05-12-featllm-llm-as-feature-engineer-tabular %}) post is to frame LLM-FE as "FeatLLM + evolutionary search + program-space output instead of rule-space output" — FeatLLM's binary rules cannot compose new continuous features (`Log_Cholesterol`, `proliferation_activity = Normal_Nucleoli × Mitoses`), LLM-FE's programs can. The template is genuinely useful for hypothesis generation in low-shot clinical tabular tasks where neither raw GBDTs nor per-sample LLMs are practical, with three caveats that matter more in medicine than elsewhere: (a) Heart, the flagship clinical example for the interpretability narrative, is not Wilcoxon-significant; (b) the rediscovered clinical features are textbook, so this is "LLM as a fast scribe for known transformations" rather than novel discovery; (c) GPT-3.5 has plausibly seen Heart/Diabetes/Breast-W during pretraining, and the order-of-magnitude shrinkage of gains on the post-cutoff control set is consistent with that contamination. As an auditable hypothesis generator that funnels into a tiny downstream model, the pattern is right; as a clinical deployment recipe, it still wants external-site validation, a clinician rule audit, and a head-to-head against manually engineered clinical features.

## References
- Paper: Abhyankar, Shojaee, Reddy. "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers." TMLR (accepted May 2026). arXiv:2503.14434v3. OpenReview: `qvI35hkpOO`.
- Backbones: `gpt-3.5-turbo` (default), Llama-3.1-8B-Instruct, GPT-4o-mini, Qwen2.5-72B-Instruct, Gemini-2.5-Flash.
- Closest predecessors: FeatLLM (Han et al., ICML 2024), CAAFE (Hollmann et al., NeurIPS 2023), OCTree (Nam et al., 2024), OpenFE (Zhang et al., ICML 2023), AutoFeat (Horn et al., 2020), FunSearch (Romera-Paredes et al., Nature 2024).
- Memorization controls: Bordt et al. 2024; Hollmann et al. 2024.
- Operator-bias reference: Küken et al. 2024.
