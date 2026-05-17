---
title: "MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning"
excerpt: "A training-free 5-stage role-played consultation lifts GPT-4 zero-shot medical QA to 86.7% average across 9 benchmarks — but the +0.8 MedQA win over 5-shot CoT+SC has no variance, and a buried ablation shows removing the least-relevant expert actually helps."
categories:
  - Paper
permalink: /paper/medagents-collaborators-zero-shot-medical/
tags:
  - MedAgents
  - LLM-Agents
  - LLM
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- MedAgents is a **training-free, 5-stage role-played consultation** (Expert Gathering → Analysis Proposition → Report Summarization → Collaborative Consultation → Decision Making) that simulates a multi-disciplinary clinical meeting on top of a frozen GPT-3.5 / GPT-4.
- **Headline result:** GPT-4 + MedAgents reaches **86.7% average across 9 medical QA benchmarks**, beating GPT-4 zero-shot-CoT (80.8%) and GPT-4 5-shot CoT+SC (85.4%); on MedQA specifically it hits **83.7%** — only **+0.8** over the 5-shot CoT+SC baseline (82.9%), reported without variance.
- The MedQA ablation shows the **Analysis Proposition stage carries +7 of the +12 total gain** — the per-domain role-played analyses do the work; the unanimity Consultation loop adds polish, not magic.

## Motivation

LLM performance in medicine is bottlenecked by (i) scarce / private high-quality medical training data, and (ii) the need for both deep domain knowledge and multi-hop clinical reasoning. The community has answered with tool-augmented retrieval (Almanac, GeneGPT, KARD) or instruction-tuning on medical corpora (Med-PaLM 2, MedAlpaca, DISC-MedLLM). Both routes are expensive and brittle, and plain CoT has been shown to *hurt* zero-shot medical accuracy by surfacing fluent but wrong rationales. MedAgents proposes a third path: keep the LLM frozen, but elicit its latent medical knowledge through a structured *multi-disciplinary consultation* mimicking how hospitals actually deliberate on complex cases. The evaluation is squarely clinical — MedQA, MedMCQA, PubMedQA, and 6 MMLU medical subtasks, all board-style multiple-choice.

## Core Innovation

- **Two disjoint expert pools.** Question-domain experts ($m$) reason about the clinical scenario; option-domain experts ($n$) arbitrate between answer choices. Defaults: $m=5, n=2$ (PubMedQA: $m=4$).
- **Information routing inside Analysis Proposition.** Option experts are *conditioned on* the question analyses before they opine — a soft information-routing step that distinguishes MedAgents from generic debate frameworks.
- **Unanimity (not majority) stop criterion.** The Collaborative Consultation loop iterates vote-and-modify until *all* experts vote yes, capped at 5 rounds. This is the paper's strongest design departure from majority-vote ensemble methods.
- **"Replaces RAG."** The authors claim the role-played consultation removes the need for retrieval augmentation. The claim is qualitative — no RAG baseline is run.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | MedAgents outperforms zero-shot baselines "by a large margin" | Table 2: +4.3 avg over GPT-3.5 zero-shot, +6.1 avg over GPT-4 zero-shot | 9 datasets, n=300 each | ⭐⭐ |
| C2 | MedAgents matches or beats few-shot CoT+SC baselines in the zero-shot setting | Table 2: GPT-4 86.7 > 5-shot CoT+SC 85.4; GPT-3.5 72.1 > 71.6 | 9 datasets, n=300 each | ⭐⭐ |
| C3 | All 5 stages contribute non-trivially | Table 3 ablation showing +7/+3/+2 incremental gains | MedQA only | ⭐ |
| C4 | CoT can *hurt* medical QA via hallucination | Table 2: GPT-3.5 ZS-CoT 58.3 < ZS 67.8 avg; GPT-4 ZS-CoT 80.8 ≈ ZS 80.6; Appendix A patella example | 9 datasets + 1 anecdote | ⭐⭐ |
| C5 | Role-playing replaces the need for RAG | Appendix B case studies + Section 4.7 | Qualitative only | ⭐ |
| C6 | Optimal agent counts ($m=5, n=2$) generalize across datasets | Figure 3, Table 5 | 4 datasets | ⭐⭐ |
| C7 | Diverse domain agents > same-domain agents | Table 7: 64.1 vs 59.2 with 6 agents | MedQA, MedMCQA | ⭐⭐ |
| C8 | Performance is robust to domain perturbation | Table 6 "minor variance" | n=20 only | ⭐ |
| C9 | 77% of errors are knowledge (not reasoning) issues | Figure 4 (40 sampled errors) | MedQA+MedMCQA | ⭐⭐ |
| C10 | MedAgents framework "extends reasoning abilities" of LLMs | Aggregate Table 2 gains | 9 datasets | ⭐ |

**Honest read.** The aggregate gains are real and the engineering is clean, but the audit reveals load-bearing weaknesses. The MedQA +0.8 win over 5-shot CoT+SC (C1, C2) is reported with temperature 1.0 on a 300-example subsample and **no variance, no significance test, no reruns** — well within plausible run-to-run noise. The "CoT hurts" framing (C4) is mostly a GPT-3.5 artifact: GPT-3.5 ZS-CoT collapses to 58.3 (worse than plain ZS at 67.8), but on GPT-4 ZS-CoT 80.8 ≈ ZS 80.6, so the dramatic hallucination story softens at the frontier model. The most over-stated claim is C5: the paper says role-playing "remarkably bypasses the need for RAG" without ever running a RAG baseline. And buried in Table 6 (n=20) is the most uncomfortable finding — see Method & Architecture below for what removing the *least* relevant expert does to accuracy.

## Method & Architecture

![MedAgents 5-stage pipeline overview](/assets/images/paper/medagents/page_002.png)
*Figure 1: The MedAgents 5-stage consultation pipeline — Expert Gathering → Analysis Proposition (Question + Option domains) → Report Summarization → Collaborative Consultation (unanimity vote, up to 5 rounds) → Decision Making.*

### 1. Expert Gathering

Two disjoint expert pools are recruited via prompting:

$$QD = \mathrm{LLM}(q, r_{qd}, \text{prompt}_{qd}), \quad OD = \mathrm{LLM}(q, op, r_{od}, \text{prompt}_{od})$$

with $|QD| = m$ question-domain experts and $|OD| = n$ option-domain experts. Defaults: $m = 5, n = 2$ on MedQA / MedMCQA / MMLU and $m = 4, n = 2$ on PubMedQA (whose option space is Yes/No/Maybe).

### 2. Analysis Proposition

Each expert generates a free-text analysis. Question analyses are independent: $qa_i = \mathrm{LLM}(q, qd_i, r_{qa}, \text{prompt}_{qa})$. Option analyses are *conditioned on the prior question analyses* and the option set: $oa_i = \mathrm{LLM}(q, op, od_i, QA, r_{oa}, \text{prompt}_{oa})$. This is the stage that the MedQA ablation (Table 3) credits with the largest single gain: **49 → 55 (CoT) → 62 (+Analysis, +7)**.

### 3. Report Summarization

A "medical report assistant" role distills $QA \cup OA$ into a structured report with two fields, *Key Knowledge* and *Total Analysis*:

$$\text{Repo} = \mathrm{LLM}(QA, OA, r_{rs}, \text{prompt}_{rs}).$$

### 4. Collaborative Consultation

![Worked VSD case: question/option expert split, initial report, unanimous final report](/assets/images/paper/medagents/page_003.png)
*Figure 2: A 3-month-old infant VSD case — question and option domain experts are split, an initial report is drafted, then the vote-and-revise loop runs until all experts vote yes. The loop is capped at 5 attempts.*

Each round, every expert returns a binary vote on the current report; any "no" yields modification suggestions $Mod_i$, and the union is fed back to produce $\text{Repo}_j = \mathrm{LLM}(\text{Repo}_{j-1}, Mod_j, \text{prompt}_{mod})$. The loop terminates on **unanimous** approval or after $t = 5$ attempts.

### 5. Decision Making

A final "decision maker" prompt selects one option given the consensus report: $ans = \mathrm{LLM}(q, op, \text{Repo}_f, \text{prompt}_{dm})$.

**Hyperparameters that matter.** Temperature 1.0, top_p 1.0 (high stochasticity, no reported variance over reruns). Self-consistency baselines use 5 samples at temperature 0.7. Cost: **~\$1.41 per 100 questions (~1.4¢ / question), ~40 s / example** — roughly an order of magnitude more expensive than zero-shot CoT.

## Experimental Results

### Main accuracy (300-example subsample per dataset, accuracy %)

| Method | MedQA | MedMCQA | PubMedQA | Anat. | Clin. Know. | Coll. Med. | Med. Gen. | Prof. Med. | Coll. Bio. | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Flan-PaLM 5-shot CoT | 60.3 | 53.6 | 77.2 | 66.7 | 77.0 | 83.3 | 75.0 | 76.5 | 71.1 | 71.2 |
| Flan-PaLM 5-shot CoT+SC | 67.6 | 57.6 | 75.2 | 71.9 | 80.4 | 88.9 | 74.0 | 83.5 | 76.3 | 75.0 |
| GPT-3.5 Zero-shot | 54.3 | 56.3 | 73.7 | 61.5 | 76.2 | 63.6 | 74.0 | 75.4 | 75.0 | 67.8 |
| GPT-3.5 Zero-shot CoT | 44.3 | 47.3 | 61.3 | 63.7 | 61.9 | 53.2 | 66.0 | 62.1 | 65.3 | 58.3 |
| GPT-3.5 Zero-shot CoT+SC | 61.3 | 52.5 | 75.7 | 71.1 | 75.1 | 68.8 | 76.0 | 82.3 | 75.7 | 70.9 |
| GPT-3.5 5-shot CoT+SC | 62.1 | 58.3 | 73.4 | 70.4 | 76.2 | 69.8 | 78.0 | 79.0 | 77.2 | 71.6 |
| **MedAgents (GPT-3.5)** | **64.1** | **59.3** | 72.9 | 65.2 | **77.7** | **69.8** | **79.0** | 82.1 | **78.5** | **72.1** |
| GPT-4 Zero-shot | 73.0 | 69.0 | 76.2 | 78.5 | 83.3 | 75.6 | 90.0 | 90.0 | 90.0 | 80.6 |
| GPT-4 Zero-shot CoT | 61.8 | 69.0 | 71.0 | 82.1 | 85.2 | 80.8 | 92.0 | 93.5 | 91.7 | 80.8 |
| GPT-4 Zero-shot CoT+SC | 74.5 | 70.1 | 75.3 | 80.0 | 86.3 | 81.2 | 93.0 | 94.8 | 91.7 | 83.0 |
| GPT-4 5-shot CoT+SC | 82.9 | 73.1 | 75.6 | 80.7 | 90.0 | 88.2 | 90.0 | 95.2 | 93.0 | 85.4 |
| **MedAgents (GPT-4)** | **83.7** | **74.8** | **76.8** | **83.5** | **91.0** | 87.6 | 93.0 | **96.0** | **94.3** | **86.7** |

**MedQA margins.** GPT-4 + MedAgents beats GPT-4 zero-shot CoT by +21.9 (61.8 → 83.7) — the headline win is mostly that CoT collapses on MedQA. Against GPT-4 zero-shot CoT+SC the gap is +9.2 (74.5 → 83.7); against GPT-4 5-shot CoT+SC it is only **+0.8** (82.9 → 83.7), with no variance reported in either cell.

**Re-read the CoT-hurts story carefully.** On GPT-3.5, zero-shot CoT (58.3) is worse than plain zero-shot (67.8) — supports the hallucination narrative. On GPT-4, zero-shot CoT (80.8) is essentially tied with plain zero-shot (80.6) — the "CoT hurts" framing weakens at the frontier model and the MedAgents win against CoT becomes a more ordinary ~6-point gap.

### Results page (Table 2 + ablations)

![Main results table, ablation, agent count, and domain studies](/assets/images/paper/medagents/page_006.png)
*Table 2 + Table 3 + Table 4 + Table 5 + Table 6: 9-benchmark main comparison plus the MedQA ablation, open-source-model comparison, agent-count sweep, and the buried domain-variation study.*

### Ablation (MedQA, GPT-3.5)

- Direct prompt 49.0 → CoT 55.0 → **+Analysis Proposition 62.0 (+7)** → +Summary 65.0 (+10) → +Consultation 67.0 (+12).
- The single biggest jump is from CoT to Analysis Proposition. The unanimity Consultation loop contributes the last +2. The per-domain role-played analyses do the heavy lifting; the consensus loop adds polish.

### Agent count and domain composition

![Agent-count sweep and same-domain vs diverse-domain comparison](/assets/images/paper/medagents/page_007.png)
*Figure 3 + Table 7: accuracy across question-agent counts (relatively flat after $m = 4$), and same-domain vs diverse-domain comparison (64.1 with 6 diverse agents vs 59.2 with 6 same-domain agents — diversity matters more than count).*

### The buried Table 6 anomaly (n = 20)

On MedQA / MedMCQA (GPT-3.5, n=20 samples):

- Baseline: 63.8 / 58.9
- Remove the **most** relevant expert: 60.5 / 55.4 (drops, as expected)
- Random removal: 62.2 / 56.3
- **Remove the *least* relevant expert: 66.2 / 61.5 — accuracy *improves*.**

If unanimity-based consensus worked as advertised, irrelevant experts should be neutral; instead they hurt. The paper labels this "minor variance" and moves on. n=20 is too small for firm conclusions, but the direction of the effect undermines the "diverse perspectives" story (C7) and is left unexplained.

### Error analysis

![Error category breakdown — 77% knowledge failures](/assets/images/paper/medagents/fig_p008_01.png)
*Figure 4: 40 sampled errors from MedQA + MedMCQA — 45% mis-retrieval of domain knowledge, 32% lack of domain knowledge, 15% consistency, 8% CoT. 77% of residual errors are knowledge failures, hinting at a parametric-knowledge ceiling.*

## Limitations

**Authors acknowledge:** parametric medical knowledge in LLMs goes stale and needs updating; different LLMs at different pipeline stages could be worth exploring (not tried); evaluation is English-only; the method is materially more expensive than CoT or direct prompting (footnote).

**Not addressed:**

- **No variance, no significance tests, no reruns.** Temperature 1.0 plus a 300-example subsample plus a +0.8 MedQA win presented as decisive.
- **300-sample subsets across all 9 datasets**, including 500-example PubMedQA and 6.1K-example MedMCQA. On PubMedQA the GPT-3.5 MedAgents result (72.9) is actually *worse* than zero-shot CoT+SC (75.7); not flagged in the main text.
- **No RAG baseline.** The "replaces RAG" claim rests on Appendix-B case studies, not a head-to-head.
- **Unanimity stop criterion has no ablation.** Majority vote? Single round? Not compared.
- **Table 6 anomaly unexplained.** Removing the least-relevant expert *helps*.
- **No prompt-perturbation robustness study** despite the entire method being prompt-engineered.
- **No human / clinician evaluation** of the synthesized reports — interpretability is asserted, not measured.
- **Only multiple-choice MCQA.** No open-ended clinical text, no EHR.

## Why It Matters for Medical AI

MedAgents is the most cited demonstration that a *structured*, training-free agent protocol can wring meaningful zero-shot gains out of a frozen general-purpose LLM on medical QA — without retrieval, without fine-tuning, and without specialist models. The pattern that the Analysis Proposition stage carries most of the gain (the +7 of +12 in the MedQA ablation) is the load-bearing finding: role-playing as per-domain experts elicits more useful clinical reasoning than vanilla CoT. The Consultation loop is more cosmetic than its narrative suggests. For deployment, the unresolved questions are the ones the paper sidesteps — Does it actually beat a real RAG pipeline? Does the +0.8 over few-shot CoT+SC hold across reruns? Why does removing the least-relevant expert *help*? Until those are answered, MedAgents is best read as a strong prompt-engineering recipe rather than a paradigm shift.

## References

- Paper: [arXiv 2311.10537 (v4, 4 Jun 2024)](https://arxiv.org/abs/2311.10537) — ACL 2024 Findings
- Code: [github.com/gersteinlab/MedAgents](https://github.com/gersteinlab/MedAgents)
- Related: Med-PaLM 2 (Singhal et al. 2023), MedAlpaca (Han et al. 2023), DISC-MedLLM (Bao et al. 2023), Almanac (Zakka et al. 2024), GeneGPT (Jin et al. 2024), KARD (Kang et al. 2023), MDAgents (Kim et al. 2024)
