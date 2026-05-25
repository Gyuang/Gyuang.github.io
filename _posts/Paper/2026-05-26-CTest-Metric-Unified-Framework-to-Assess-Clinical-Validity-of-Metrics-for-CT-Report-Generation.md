---
title: "CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation"
excerpt: "Auditing eight CT-RRG metrics across seven CT-CLIP+LLM generators on CT-RATE: GREEN Score correlates best with experts (Spearman rho = 0.70), CRG anti-correlates (-0.27), and BERTScore-F1 is nearly blind to factual errors (Delta approx -0.02)."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/ctest-metric/
tags:
  - CTest-Metric
  - CT-Report-Generation
  - Metric-Evaluation
  - GREEN-Score
  - F1-RadGraph
  - CRG
  - BERTScore
  - CT-RATE
  - CT-CLIP
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- CTest-Metric is a **three-module audit framework** (WSG style-rephrasing, SEI graded synthetic error injection, MvE expert-correlation on disagreement cases) that stress-tests eight CT-report-generation metrics over seven CT-CLIP-backed LLM generators on CT-RATE. **No new metric is proposed** — the methodology itself is the contribution.
- Headline correlations on 175 expert-rated disagreement cases: **GREEN Score Spearman rho = 0.70**, F1-RadGraph 0.53, NLG cluster (BLEU/ROUGE/METEOR/BERTScore-F1) in 0.26-0.35, **CRG anti-correlated at -0.27**.
- Factual-sensitivity result: at the multi-error level, GREEN drops -0.6053 and F1-RadGraph -0.3970, while **BERTScore-F1 barely moves (Delta approx -0.02)** — embedding-similarity metrics can mask factual errors entirely.

## Motivation

CT report-generation (RRG) work (Med3DVLM, CT-AGRG, Med-2E3) keeps reporting BLEU/ROUGE/METEOR/BERTScore/F1-RadGraph/CheXpert numbers inherited from the chest X-ray pipeline. CT reports are longer, multi-organ, and use a denser anatomical vocabulary than 2D X-ray reports, so the X-ray-tuned metric stack is not obviously transferable. Two prior audits (Yu et al. 2023; Banerjee et al. ReXamine-Global 2024) showed these metrics correlate poorly with experts, but only on X-ray. No equivalent framework exists for CT, which means CT model selection is currently grounded in metrics that may not reflect clinical fidelity. The paper aims to fill that gap with a reusable, modular test suite rather than a new metric.

## Core Innovation

- **Disagreement-driven expert audit.** Instead of a random sample, MvE deliberately picks the 175 patient cases where the eight metrics maximally disagree (top-25 per LLM ranked by per-patient inter-metric std, across seven LLMs). This concentrates clinician annotation effort on contested reports.
- **Two LLM-driven perturbation modules.** WSG uses LLaMA-3.1-8B-Instruct to *rephrase* model predictions while preserving clinical semantics (style robustness), and SEI uses the same model to *inject* 1 / 2 / multiple factual errors into ground-truth reports (factual sensitivity). Both are zero-shot prompts.
- **Generator-side breadth, encoder-side monoculture.** Seven decoders (Distilgpt, GPT2, GPT2-Medium, LLaMA-3.2-1B, R2GenGPT, R2GenGPT+BioGPT-Large, R2GenGPT+LLaMA-3.2-1B) span an order of magnitude of capacity, but all share the **CT-CLIP** image encoder — a deliberate design choice that doubles as the framework's biggest blind spot.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | First unified metric-assessment framework for CT RRG. | Differentiated from Yu 2023 / Banerjee 2024 (X-ray-only). Definitional. | n/a | ⭐⭐ — true as scoped; "first" is narrow since the methodology mostly ports X-ray-era frameworks to a new corpus. |
| C2 | Lexical NLG metrics are highly sensitive to stylistic variations. | Fig. 2a: BLEU drops 37-48% on six of seven models after rephrasing. | CT-RATE val | ⭐⭐⭐ — consistent direction across models; the BioGPT-Large outlier does not break the trend. |
| C3 | GREEN Score aligns best with expert judgments (rho = 0.70). | Fig. 2b on 175 disagreement cases. | CT-RATE val | ⭐⭐ — single dataset, no CI, no second clinician panel; sample is **selected for disagreement**, which can inflate winners and bury losers. |
| C4 | CRG shows negative correlation (rho = -0.27). | Fig. 2b on the same 175 cases; CRG also anti-correlates with every other metric. | CT-RATE val | ⭐⭐ — real on this slice, but disagreement-only sampling can invert correlations; on a random sample CRG might still carry useful signal. |
| C5 | BERTScore-F1 is least sensitive to factual error injection. | Fig. 3: Delta(S_M, S_G) approx -0.02 at the multi-error level. | CT-RATE GT + LLM-injected errors | ⭐⭐ — clear effect, but "factual error" is whatever LLaMA-3.1-8B-Instruct decides to inject; edit fidelity is **not independently validated**. |
| C6 | Framework is a robust pathway for future metric developers. | Three modular tests, eight metrics, seven LLMs. | CT-RATE only | ⭐ — one dataset, one image encoder, one rephraser LLM; framework robustness is asserted, not measured. |

**Honest read.** C2 and C5 are well-supported descriptive findings — metric brittleness under rephrasing and embedding-metric blindness to factual error are convincingly demonstrated. The headline story C3+C4 ("GREEN wins, CRG fails") is the weakest part of the paper:

- **MvE samples by construction inflate metric disagreement.** Top-25 per-patient highest-std cases per LLM x 7 LLMs = 175 maximally contested reports. CRG operates on a different feature scale (multi-label presence/absence of clinical entities) than the other seven, so std-normalization may systematically place CRG on the "disagreeing" side. Its negative rho here does not necessarily reflect a problem with CRG on random reports.
- **Only two reviewers, no IAA, no CIs, no significance test.** The Spearman ranking is presented as a clean leaderboard with no statistical infrastructure to support pairwise claims (GREEN > F1-RadGraph, etc.).
- **GREEN may win partly by shared philosophy, not clinical validity.** GREEN is an LLM-as-judge metric that parses error counts via regex; the expert protocol is also "count clinical errors." That GREEN aligns with experts may partly reflect a shared *evaluation philosophy* rather than superior clinical grounding. F1-RadGraph (rho = 0.53), which also adopts explicit-entity logic, supports this read.
- **Encoder monoculture.** All seven generators share CT-CLIP. If CT-CLIP has systematic failure modes (e.g., weak grounding for specific pathologies), every report inherits those failures and any metric aligned with CT-CLIP's blind spots will look good. No non-CT-CLIP control generator (Merlin, M3D-CLIP, etc.) is included, so GREEN's rho = 0.70 cannot be cleanly separated from CT-CLIP compatibility.
- **Single dataset and single rephraser.** Only CT-RATE (one institution, non-contrast chest CT), only LLaMA-3.1-8B-Instruct for both rephrasing and error injection. Generalization to other body regions, contrast phases, or rephraser models is untested.

## Method & Architecture

![CTest-Metric framework overview: WSG rephrasing, SEI graded error injection, MvE expert correlation on disagreement cases](/assets/images/paper/ctest-metric/fig_p002_01.png)
*Figure 1: CTest-Metric framework. (a) Three test modules — WSG (rephrasing preserves clinical content while perturbing style), SEI (LLM injects 1 / 2 / multiple errors into ground truth), MvE (top-25 highest-std cases per LLM are flagged for clinician review). (b) Example reports illustrating one-error, two-error, and multi-error injections.*

The framework has three modules layered on a fixed metric pool and generator pool.

1. **Backbone setup.** Seven generators share the **CT-CLIP** image encoder (Hamamci et al., 2024) and pair it with seven LLM decoders: Distilgpt, GPT2, GPT2-Medium, LLaMA-3.2-1B, R2GenGPT (frozen LLM, shallow alignment), R2GenGPT+BioGPT-Large, R2GenGPT+LLaMA-3.2-1B. All trained 10 epochs on one A100, hyperparameters per public repos.
2. **Metric pool.** M = {BLEU, ROUGE, METEOR, BERTScore-F1, F1-RadGraph, RaTEScore, GREEN Score, CRG}. First four are NLG-style; last four are clinical-efficacy (CE)-style.
3. **WSG (Writing Style Generalizability).** LLaMA-3.1-8B-Instruct rephrases each predicted report P -> P' via a zero-shot prompt that asks for clinical-semantic preservation. Both versions are scored under every metric; Delta% = (S' - S) / S is reported per (model, metric) cell. Low |Delta%| means style-robust.
4. **SEI (Synthetic Error Injection).** Same LLM-prompt machinery, but errors (1, 2, or "multiple") are injected into the *ground-truth* report, yielding S_1, S_2, S_M. These are compared against the ideal S_G (GT vs GT) and the metric's Delta-curve is plotted across error levels. Large negative slope means good factual sensitivity.
5. **MvE (Metrics-vs-Expert).** Per LLM: compute per-patient scores under all 8 metrics, normalize, take per-patient std across metrics, pick top-25 highest-std (most contentious). Repeat for all 7 LLMs to get 7 x 25 = 175 disagreement cases. Two clinical reviewers rate them (second opinion only on ambiguous cases). Spearman rho is computed pairwise across all 8 metrics plus the expert column.

## Experimental Results

### WSG — % deviation after rephrasing (Fig. 2a)

![WSG heatmap: percent deviation after rephrasing across 7 generators x 8 metrics](/assets/images/paper/ctest-metric/fig_p004_01.png)
*Figure 2a: BLEU / ROUGE / METEOR / F1-RadGraph collapse 25-48% under rephrasing; BERTScore-F1 holds within +/- 2%; CRG and GREEN are essentially flat. The R2GenGPT+BioGPT-Large row goes positive across every metric, indicating the rephraser cleaned up an unusually low-quality baseline.*

| Model | BLEU | ROUGE | METEOR | BERTScore-F1 | RaTEScore | F1-RadGraph | GREEN | CRG |
|---|---|---|---|---|---|---|---|---|
| Distilgpt | -41.12 | -28.89 | -25.58 | -1.18 | -6.39 | -39.10 | -1.97 | 0.09 |
| GPT2 | -45.35 | -31.28 | -27.82 | -1.28 | -6.82 | -41.11 | -4.08 | 0.00 |
| GPT2-Medium | -38.92 | -28.83 | -25.71 | -1.38 | -5.75 | -35.92 | -0.03 | 0.00 |
| LLaMA-3.2-1B | -37.29 | -27.88 | -24.38 | -1.34 | -5.90 | -35.02 | 0.39 | 0.03 |
| R2GenGPT | -47.76 | -28.12 | -27.70 | -1.98 | -6.72 | -42.29 | 1.22 | -0.21 |
| R2GenGPT + BioGPT-Large | **+20.73** | **+8.36** | **+10.60** | **+5.86** | **+32.06** | **+68.95** | **+30.45** | 0.51 |
| R2GenGPT + LLaMA-3.2-1B | -48.82 | -25.39 | -26.34 | -1.16 | -5.35 | -41.46 | 6.80 | -0.68 |

NLG metrics swing -48.82% to -1.16% on rephrasing; BERTScore-F1 is the most stable NLG metric (|Delta| < 2%) because contextual embeddings absorb synonym swaps. F1-RadGraph is unexpectedly fragile (-42.29% to +68.95%) — graph-entity matching is sensitive to how the rephraser renames entities. CRG and GREEN are the two style-robust metrics; CRG is essentially flat (|Delta| <= 0.68%) because it only checks presence/absence of multi-label clinical entities. The R2GenGPT+BioGPT-Large row is a clear outlier — every metric goes *up* on rephrasing, a strong sign that the baseline reports were unusually low quality and the rephraser improved them rather than perturbing them. The paper does not flag this.

### SEI — Delta from ideal vs error level (Fig. 3)

![SEI line plot: Delta from ideal score across error levels for each metric](/assets/images/paper/ctest-metric/fig_p004_03.png)
*Figure 3: Synthetic Error Injection. BERTScore-F1 is nearly flat (Delta approx -0.02 at multi-error), GREEN drops most steeply (-0.6053), F1-RadGraph second (-0.3970). All metrics are indistinguishable at 1 error; separation only emerges at the multi-error level.*

| Metric | Delta(S_1, S_G) | Delta(S_2, S_G) | Delta(S_M, S_G) |
|---|---|---|---|
| BERTScore-F1 | ~0 | ~0 | **-0.02** (least sensitive) |
| **GREEN** | -0.0812 | -0.1277 | **-0.6053** (most sensitive) |
| F1-RadGraph | -0.0891 | -0.1647 | **-0.3970** |
| CRG | small | small | ~-0.27 |
| BLEU / ROUGE / METEOR / RaTEScore | small at 1-2 errors | diverge only at "Multiple" | moderate (~-0.05 to -0.15) |

BERTScore-F1's near-zero response is the paper's warning shot — embedding-similarity metrics can mask factual errors entirely. Note that "Multiple" is unspecified (could be 3, could be 10), and the F1-RadGraph vs GREEN gap at this point may depend on how the rephraser-prompt expands "multiple."

### MvE — Spearman rho on 175 disagreement cases (Fig. 2b)

![MvE correlation matrix: Spearman rho across 8 metrics + Expert on 175 disagreement cases](/assets/images/paper/ctest-metric/fig_p004_02.png)
*Figure 2b: Spearman correlations on 175 disagreement cases. GREEN <-> Expert rho = 0.70 (best); F1-RadGraph 0.53; NLG cluster 0.26-0.35; CRG anti-correlated at -0.27. Inter-NLG rho is 0.82-0.92 — they agree with each other, not with experts.*

| Metric | rho vs Expert |
|---|---|
| **GREEN Score** | **0.70** |
| F1-RadGraph | 0.53 |
| RaTEScore | 0.47 |
| BLEU | 0.35 |
| METEOR | 0.35 |
| ROUGE | 0.28 |
| BERTScore-F1 | 0.26 |
| **CRG** | **-0.27** |

Inter-NLG rho ranges 0.82-0.92 (they agree with each other, not with experts). GREEN <-> F1-RadGraph rho = 0.60. CRG anti-correlates with everything, including all other CE metrics — a result that has to be read together with the disagreement-only sampling.

### Dataset

CT-RATE (Hamamci et al., 2024): 50,188 non-contrast chest CT volumes paired with radiology reports; official train/val split. One institution, one body region, one contrast phase. Both training and evaluation share the CT-RATE writing style, so the rephrased reports likely still live closer to CT-RATE language than to truly out-of-distribution radiologist prose. License: open-access, ethics waiver invoked, no IRB needed.

## Limitations

**Author-acknowledged.**

- Only two clinical reviewers with tiebreaker on ambiguous cases — no inter-annotator agreement reported (no kappa, no %-agreement).
- WSG / SEI rely on LLaMA-3.1-8B-Instruct prompting; the rephrasings and error injections primarily produce laterality and negation flips, and edit fidelity was not independently audited.
- CT-RATE is the only dataset; generalization to other institutions, contrast phases, or body regions is unverified.

**Not addressed.**

- *Why GREEN wins.* GREEN is itself an LLM-as-judge metric that parses error counts via regex; expert reviewers also score by counting clinical errors. The high rho may partly reflect a *shared evaluation philosophy* rather than GREEN being more clinically valid. F1-RadGraph (rho = 0.53), which also uses explicit-entity logic, supports this read. The paper does not separate "GREEN is clinically valid" from "GREEN and experts both adopt error-count semantics."
- *Sampling bias from disagreement-only cases.* The 175 cases are by construction maximally inconsistent. CRG's negative rho on this slice does not imply it would be negative on a random sample — CRG may diverge from the other seven in a direction experts happen not to share *on contested reports specifically*.
- *Encoder monoculture.* All seven generators use CT-CLIP. No ablation with a different image encoder (Merlin, M3D-CLIP) to test whether metric rankings hold when generators do not share an encoder's failure modes.
- *No statistical inference.* No bootstrapped CIs on rho, no significance test for rho(GREEN) > rho(F1-RadGraph), no power analysis for n = 175.
- *"Multiple Errors" undefined.* The error count for the multi-error bucket is unspecified, so the Delta gap between F1-RadGraph (-0.397) and GREEN (-0.605) at this point may be partly a function of how the prompt expands "multiple."
- *No release timeline.* The v1 preprint promises code plus rephrased / error-injected reports but does not link the repo.

## Why It Matters for Medical AI

The CT-RRG field is in a familiar trap: a benchmark culture has formed around X-ray-era metrics that may or may not measure what we actually care about. CTest-Metric is a useful diagnostic — its *descriptive* findings (NLG metrics collapse under rephrasing; BERTScore-F1 cannot see factual errors) are real and should change defaults. Two implications:

- **Stop reporting BERTScore-F1 alone for CT-RRG.** A 0.02 spread on a metric that does not move with factual errors is not a signal. Pair it with at least one entity-aware metric (F1-RadGraph) and one LLM-judge metric (GREEN) and disclose the spread between them.
- **Treat GREEN's rho = 0.70 as a *useful prior*, not a coronation.** Until the same audit is repeated on a non-CT-CLIP encoder, a different institution, a random (not disagreement-curated) sample, and with reported IAA + CIs, the ranking is provisional. The contribution is the audit recipe, not the final ranking.

The *prescriptive* claim — that improving on these three modules yields a better metric — is asserted, not demonstrated, and should be approached skeptically until follow-on work designs a metric against this benchmark and shows it generalizes.

## References

- **Paper:** Sharma, Bejar, Durak, Bagci. *CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation.* ISBI 2026. arXiv:2601.11488v1.
- **Related:**
  - CT-RATE / CT-CLIP — Hamamci et al., 2024. The shared image encoder and benchmark dataset.
  - F1-RadGraph — Jain et al., *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports*, NeurIPS 2021.
  - GREEN Score — Ostmeier et al., *GREEN: Generative Radiology Report Evaluation and Error Notation*, EMNLP 2024.
  - CRG — Clinical Report Generation metric for multi-label entity presence.
  - ReXamine-Global — Banerjee et al., 2024. Prior X-ray metric audit; closest methodological cousin.
  - Yu et al., 2023 — Earlier X-ray metric audit motivating disagreement-based sampling.
  - R2GenGPT — Wang et al., 2023. One of the seven generator backbones.
