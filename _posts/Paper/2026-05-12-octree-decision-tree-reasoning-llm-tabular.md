---
title: "OCTree: Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning"
excerpt: "An LLM-as-optimizer loop that feeds CART decision-tree reasoning back as natural-language if-else feedback, cutting XGBoost relative error by ~5.0% across 19 anonymized Grinsztajn datasets."
categories:
  - Paper
tags:
  - OCTree
  - Tabular Learning
  - LLM Feature Engineering
  - Decision Tree Reasoning
  - LLM-as-Optimizer
  - Automated Feature Engineering
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- OCTree treats automated feature engineering as a black-box optimization problem and uses an LLM as the optimizer — at each step the LLM proposes a new column-generation rule, the prediction model is retrained, and a **CART (max depth 4) decision tree is serialized as natural-language if-else syntax** and fed back as structured feedback for the next iteration.
- Headline numbers: **~5.0% average XGBoost relative error reduction** across the 19 anonymized Grinsztajn et al. classification datasets, and **25.12% mean error with OCTree+Llama-2-7B vs 25.43% for CAAFE+GPT-4o** on six descriptive datasets — a rhetorically loud "7B beats GPT-4o" framing that **conflates optimizer design with LLM scale**, since CAAFE is not re-run with the same Llama-2 backbone.
- Honesty caveats that matter: the DT-reasoning ablation — the paper's *central* methodological contribution — runs on only **4 datasets and reports no significance tests**, with the Disease jump (1.7% → 6.8%) sitting well inside the ±7.9 baseline std; medical-AI scope is narrow (one small Kaggle Disease dataset + one Clinical Trial set, nothing MIMIC-scale); there is a plausible CART-feedback leakage chain (CART fit on `D_train ⊕ r_t`, validation score selects the rule) that goes unprobed.

## Motivation
Tree-based models (XGBoost, CatBoost) still beat deep nets on most tabular tasks, and practitioners squeeze further gains via manual feature engineering. The automated-FE literature has three structural ceilings: **AutoFeat / OpenFE** restrict themselves to a hand-coded arithmetic search space; **CAAFE** requires linguistic column descriptions and falls back to a greedy single-shot evaluator; none of the three exploit trajectory information beyond a single validation score.

OCTree's pitch is that an LLM can act simultaneously as (i) a context-aware rule proposer that exploits column names when available, (ii) a black-box optimizer that consumes a *trajectory* of past rules + scores + decision-tree reasoning, and (iii) a general engine that still works on anonymized (context-agnostic) tables where CAAFE cannot run. The medical-AI relevance is direct: clinical and financial tables are routinely de-identified, so methods that depend on descriptive column names are inapplicable, and CART-style reasoning is a familiar interpretable surrogate in clinical decision support.

## Core Innovation
Three moves stacked together:
1. **LLM-as-optimizer over rules, not over a fixed operator vocabulary.** The LLM emits a rule $r: X \to X'$ (a Python function for context-agnostic data, or a natural-language rule converted to Python in the descriptive case). No predefined arithmetic search space, no rule template — the rule can use `abs()`, `np.sin`, conditional branching, anything.
2. **Decision-tree reasoning as structured feedback.** At each iteration, a CART tree of max depth 4 is fit on the augmented training set $D_\mathrm{train} \oplus r_t$ and serialized as natural-language if-else statements. This is the paper's central methodological bet: telling the LLM *which* features and *which* thresholds the predictor relied on is more informative than a scalar validation score.
3. **Trajectory prompt sorted by score ascending.** The LLM optimizer prompt $p_\mathrm{gen}(T_t, C, c_\mathrm{target})$ contains the full history $\{(s_i, d_i, r_i)\}_{i=0}^t$, *sorted from worst to best* because LLMs tend to copy the last item — so the best rule is placed last to seed the next proposal. The LLM is also instructed that the new rule must differ from every prior $r_i$.

Multi-column extension: once a rule is selected, the new column is appended to the input space and the loop restarts to add a second column (e.g., "Smoking Status" → "Physical Activity Level"). Feature transfer: rules optimized on cheap XGBoost are reused on expensive MLP / HyperFast — a runtime trick the authors explicitly lean on.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | OCTree improves baseline prediction models across diverse tabular tasks | Tables 1, 3, 13 across XGBoost / MLP / HyperFast; 6 descriptive + 19 anonymized datasets, 3 seeds | ⭐⭐⭐ |
| C2 | OCTree beats competing AutoFE methods (AutoFeat, OpenFE, CAAFE) | Tables 2 and 5; 22 datasets aggregated | ⭐⭐⭐ |
| C3 | A 7B open LLM beats GPT-4-class baselines (CAAFE+GPT-4o) | Table 2: 25.12% (OCTree+Llama-2) < 25.43% (CAAFE+GPT-4o) on 6 descriptive | ⭐⭐ — **cross-design comparison**; CAAFE is not re-run with the same fine-tuned Llama-2-7B, so the win cannot be attributed to optimizer design vs LLM scale |
| C4 | Decision-tree reasoning as feedback is the critical innovation | Table 6 ablation: full > no-DT > baseline on 4 datasets | ⭐⭐ — **only 4 datasets**; on Disease the absolute lift (≈1.4 pp) is well within the ±7.9 baseline std; no paired significance test |
| C5 | Method works without language descriptions | Table 3 across 19 anonymized Grinsztajn datasets, ~5.0% avg gain for XGBoost and MLP | ⭐⭐⭐ |
| C6 | Features transfer from cheap model (XGBoost) to expensive ones (MLP, HyperFast) | Table 7, 4 datasets | ⭐⭐ — 1 of 8 cells reports N/I (Clinical→MLP); modest sample |
| C7 | LLM-suggested columns correspond to real-world predictive features | Table 8 (Cough > Cholesterol ranking) and Figure 3 (real US-NLM Age imputation lifts accuracy) | ⭐ — two anecdotal probes, not a systematic evaluation |
| C8 | OCTree is complementary to OpenFE; stacking yields further gains | Table 5: 4.6% → 7.9% for XGBoost | ⭐⭐ |
| C9 | The feedback loop handles hallucinations | Qualitative §4.3 + appendix examples | ⭐ — no quantitative measurement of hallucination rate or recovery |

**Honest analysis.** C1, C2, C5 are the load-bearing claims and they hold up — the breadth of datasets and multiple seeds are real. The weak pillars are C3, C4, C7. C3 is rhetorically powerful but rests on cross-design comparison: OCTree+Llama-2-7B vs CAAFE+GPT-4o is *not* an apples-to-apples ablation of optimizer design, and the missing experiment (CAAFE+Llama-2-7B, OCTree+GPT-4o, both backbones for both methods) is the one that would have settled the framing. C4 — the central methodological contribution — is ablated on only 4 datasets with no significance tests, and the headline +5.1 pp jump on Disease (1.7% → 6.8%) is mathematically inside one standard deviation of the XGBoost baseline (28.09 ± 7.9). C7 generalizes from two cherry-picked qualitative examples to a broad "validity" claim. Missing throughout: confidence intervals on the "X%" relative improvements, paired significance tests, wall-clock / token-cost comparisons vs CAAFE and OpenFE (despite repeated GPT-4o calls), and any analysis of the failure modes on jannis / wine / bank-marketing where OCTree reports no improvement.

## Method & Architecture

![OCTree pipeline overview](/assets/images/paper/octree/page_002.png)
*Figure 1: OCTree overview. The LLM proposes a column name and an initial rule, the prediction model is trained on the augmented table, a CART decision tree (max depth 4) is fit on the augmented training set and serialized as natural-language if-else feedback, and the entire trajectory $\{(s_i, d_i, r_i)\}$ — sorted by score ascending — is fed back to the LLM for the next iteration.*

The bilevel objective:

$$\min_{r}\; \mathcal{L}_{f^*}(\mathcal{D}_{\text{val}} \oplus r) \quad\text{s.t.}\quad f^* = \arg\min_f \mathcal{L}_f(\mathcal{D}_{\text{train}} \oplus r),$$

solved iteratively by an LLM optimizer with feedback. The pipeline:

1. **Column name (Step 0).** $p_\mathrm{col}(C, c_\mathrm{target})$ → propose a new column name $c_\mathrm{new}$ (e.g., "Trading Volume"). Uses linguistic context when available.
2. **Initial rule (Step 1, t=0).** $p_\mathrm{init}(C, c_\mathrm{new})$ → produce $r_0$. For context-agnostic datasets, $r_0$ is hand-set as the product of the two columns with highest XGBoost importance (e.g., $x_6 = x_1 \cdot x_5$).
3. **Generate the new column (Step 2).** Apply $r_t$ to produce $D \oplus r_t$.
4. **Train + score (Step 3a).** Train $f^*$ on $D_\mathrm{train} \oplus r_t$, compute $s_t = \mathcal{L}_{f^*}(D_\mathrm{val} \oplus r_t)$.
5. **Extract DT reasoning (Step 3b).** Fit CART (max depth 4) on $D_\mathrm{train} \oplus r_t$, serialize as natural-language if-else.
6. **Optimization step.** Build trajectory $T_t = \{(s_i, d_i, r_i)\}_{i=0}^t$ sorted by score ascending; prompt $p_\mathrm{gen}(T_t, C, c_\mathrm{target})$ to emit $r_{t+1}$ that differs from every prior rule.
7. **Repeat** for a fixed iteration budget (not specified in the main text — Appendix-only).
8. **Multi-column extension.** Append $c_\mathrm{new}$ to $X$ and restart; stop when validation no longer improves.
9. **Feature transfer.** Rules optimized on XGBoost are reused on MLP / HyperFast.

LLM backbones evaluated: GPT-4o (closed) and Llama-2-7B fine-tuned on UltraChat (their default open model). Llama-2-Chat-7B and Code-Llama-7B appear in Table 4 ablations.

## Experimental Results

**Main context-aware comparison (Table 1, descriptive datasets, lower is better).**

| Method | LLM | Tesla† MAE×10⁻³ | Enefit† MAE×10⁻³ | Disease* err% | Clinical* err% | Academic* err% |
|---|---|---:|---:|---:|---:|---:|
| XGBoost baseline | — | 6.61 | 8.00 | 28.09±7.9 | 46.27±5.0 | 14.15±0.6 |
| **OCTree + XGBoost** | **Llama-2-7B** | **5.56 (15.9%)** | **8.00 (0.0%)** | **26.19±7.2 (6.8%)** | **45.07±4.1 (2.6%)** | **14.11±0.5 (0.3%)** |
| **OCTree + XGBoost** | **GPT-4o** | **5.48 (17.1%)** | **7.82 (2.3%)** | **25.72±6.6 (8.4%)** | **43.75±4.4 (5.4%)** | **13.74±0.1 (2.9%)** |
| MLP baseline | — | 7.41 | 33.53 | 38.10±3.6 | 41.77±1.7 | 14.41±0.8 |
| **OCTree + MLP** | **GPT-4o** | **5.01 (32.4%)** | **21.68 (35.3%)** | **30.95±5.8 (18.8%)** | **39.25±0.5 (6.0%)** | **14.22±0.5 (1.3%)** |
| HyperFast baseline | — | N/A | N/A | 28.57±10.0 | 43.64±1.1 | 14.67±0.7 |
| **OCTree + HyperFast** | **GPT-4o** | N/A | N/A | **27.14±3.8 (5.0%)** | **42.00±1.5 (3.8%)** | **14.49±0.5 (1.2%)** |

†Regression (MAE×10⁻³, single time split, no variance). *Classification error % across 3 seeds. Parenthetical numbers are relative improvement over the baseline. Note the Disease ±7.9 baseline std — most reported gains sit inside one standard deviation.

**Head-to-head vs AutoFE baselines (Table 2, 6 descriptive datasets).**

| Method | w/o desc. | w/ desc. | LLM | Avg. err. (%) | Δ vs XGBoost |
|---|---|---|---|---:|---:|
| XGBoost baseline | — | — | — | 25.87±2.2 | — |
| AutoFeat | yes | no | — | 25.76±2.1 | 0.4% |
| OpenFE | yes | no | — | 26.44±1.7 | N/I |
| CAAFE | no | yes | GPT-4o | 25.43±2.2 | 1.7% |
| **OCTree** | **yes** | **yes** | **Llama-2-7B** | **25.12±1.9** | **2.9%** |
| **OCTree** | **yes** | **yes** | **GPT-4o** | **24.53±1.9** | **5.2%** |

OCTree+Llama-2-7B (25.12) edges out CAAFE+GPT-4o (25.43) — the paper's "7B beats GPT-4o" framing. But the comparison is cross-design: a fair ablation would also report CAAFE+Llama-2-7B and OCTree+GPT-4o-vs-OCTree+Llama-2 head-to-head, isolating optimizer design from LLM scale.

![Method / multi-column extension](/assets/images/paper/octree/page_005.png)
*Figure 2: Multi-column extension. Rules are applied sequentially to introduce additional features (e.g., Smoking Status → Physical Activity Level) once the previous column stops improving validation.*

**Anonymized 19 datasets (Table 3).** ~5.0% average XGBoost relative error reduction across the Grinsztajn et al. benchmark. Biggest wins: **electricity 20.1%, rl 18.2%, compass 17.6%, covertype 12.5%**. Three "no improvement" datasets (jannis, wine, bank-marketing) are reported but unanalyzed. MLP shows large wins on MiniBooNE (24.1%) and rl (11.6%). **Integrating OCTree with OpenFE (Table 5) boosts the XGBoost gain to 7.9%** — the methods are complementary.

**DT-reasoning ablation (Table 6).** Generating a new feature alone gives most of the lift on language-described data (Disease 1.7%, Clinical 1.4%), and adding DT reasoning roughly doubles the gain (Disease 6.8%, Clinical 2.6%; electricity 17.2% → 20.1%, kddCup09 2.0% → 4.0%). **This is the central methodological claim and the ablation runs on only 4 datasets with no significance tests** — on Disease the absolute lift (~1.4 pp) sits inside the ±7.9 baseline std.

**Feature transfer (Table 7).** XGBoost-optimized rules transfer to MLP (Disease 7.5%, electricity 3.9%, kddCup09 3.4%) and HyperFast (Disease 5.8%, electricity 3.2%). One miss: Clinical → MLP gives no improvement.

**LLM choice (Table 4, 19 anonymized datasets, XGBoost).** Baseline 16.53 → Llama-2-Chat-7B 16.32 → Code-Llama-7B 15.83 → **authors' UltraChat-tuned Llama-2-7B 15.71**. The default open-LLM is not off-the-shelf — it is fine-tuned for the task.

**Feature validity probes (Table 8, Figure 3).** LLMs correctly rank `Cough > Cholesterol` for the Disease task (consistent with XGBoost importance). On Clinical Trial the LLM suggests an "Age" column; imputing real US-NLM age data lifts accuracy by several points. Two anecdotal probes; not a systematic audit.

## Limitations
- **Headline cross-LLM comparison is cross-design.** OCTree+Llama-2-7B vs CAAFE+GPT-4o conflates optimizer design and LLM scale. CAAFE is never re-run with the same fine-tuned Llama-2-7B.
- **DT-reasoning ablation is narrow.** Only 4 datasets, no paired significance tests; on Disease the central effect sits inside the baseline standard deviation. The headline justification for the method's core idea deserves the full benchmark.
- **No statistical significance tests.** Many gains on the 19-dataset anonymized benchmark are <2% and well within 1 std. No Wilcoxon or paired t-tests anywhere.
- **No wall-clock or token-cost comparison** vs CAAFE / OpenFE despite repeated GPT-4o calls — efficiency claims (Pareto) are absent.
- **Iteration budget is not specified in the main text**; sensitivity to budget is unstudied.
- **CART-feedback leakage risk.** CART is fit on $D_\mathrm{train} \oplus r_t$; its serialized rules go back to the LLM, which proposes $r_{t+1}$ evaluated on $D_\mathrm{val}$. The selection chain is plausible — no held-out probe.
- **Failure modes unanalyzed.** wine, bank-marketing, jannis report no improvement; the paper does not diagnose why.
- **Default Llama-2-7B is fine-tuned on UltraChat**, not off-the-shelf — Table 4 shows the gap between Llama-2-Chat-7B (16.32) and the UltraChat-tuned version (15.71) is meaningful, so "open-LLM competitive with GPT-4o" requires that specific finetune.
- **Medical scope is narrow.** One Kaggle Disease dataset and one Clinical Trial dataset; nothing like MIMIC, eICU, or a real EHR cohort.
- **No regression / multi-class calibration analysis**; no fairness or subgroup analysis on the clinical datasets.

## Why It Matters for Medical AI
OCTree fits naturally between [FeatLLM]({% post_url Paper/2026-05-12-featllm-llm-as-feature-engineer-tabular %}) and [LLM-FE]({% post_url Paper/2026-05-12-llm-fe-evolutionary-search-tabular-feature-engineering %}) in the LLM-as-feature-engineer lineage. FeatLLM emits class-conditional binary rules *once* at training time over the original feature space — no composition, no feedback. OCTree adds **single-trajectory iterative refinement with structured DT-reasoning feedback** — composition via rule chaining, but a single optimization path. LLM-FE (TMLR 2026, the chronological successor) keeps the program-space output and adds **multi-island evolutionary search** to escape local optima — explicitly motivated by OCTree's single-trajectory plateau. Crucially, the LLM-FE paper benchmarks OCTree at mean rank **4.05** on the same 19-dataset classification suite vs LLM-FE's 1.42, so OCTree's "headline win" on the 6 descriptive sets does not generalize to the broader 19-dataset benchmark when matched against the next-generation method.

For medical AI specifically, the appeal is real but conditional: CART-style decision-tree feedback in natural language is exactly the interpretable surrogate clinicians read in CDS literature, and OCTree's context-agnostic regime (the 19 Grinsztajn datasets) is the one mode where CAAFE cannot run — relevant for de-identified clinical and financial tables. The caveats matter more in medicine than elsewhere: (a) only one small Kaggle Disease dataset and one Clinical Trial set anchor the medical claim, neither with clinician review of generated columns; (b) the "Age column imputation lifts accuracy" probe is a single qualitative example, not a systematic feature-validity audit; (c) the CART-feedback leakage chain is plausible and unprobed — a real worry when selection bias affects model deployment. As an *auditable hypothesis generator* for de-identified clinical tables that funnels into a downstream tree or MLP, the pattern is sound; as a clinical deployment pipeline it still needs an external held-out cohort, a clinician audit of LLM-proposed columns, and a comparison against manually engineered clinical features.

## References
- Paper: Nam, Kim, Oh, Tack, Kim, Shin. "Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning." NeurIPS 2024. arXiv:2406.08527 (v2, 18 Nov 2024).
- Code: [github.com/jaehyun513/OCTree](https://github.com/jaehyun513/OCTree)
- Backbones: GPT-4o (closed); Llama-2-7B fine-tuned on UltraChat (default open); Llama-2-Chat-7B, Code-Llama-7B (ablation).
- Closest predecessors: CAAFE (Hollmann et al., NeurIPS 2023), OpenFE (Zhang et al., ICML 2023), AutoFeat (Horn et al., 2020).
- Closest successors covered on this blog: [FeatLLM]({% post_url Paper/2026-05-12-featllm-llm-as-feature-engineer-tabular %}) (Han et al., ICML 2024); [LLM-FE]({% post_url Paper/2026-05-12-llm-fe-evolutionary-search-tabular-feature-engineering %}) (Abhyankar et al., TMLR 2026).
- Benchmark: Grinsztajn, Oyallon, Varoquaux. "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022.
