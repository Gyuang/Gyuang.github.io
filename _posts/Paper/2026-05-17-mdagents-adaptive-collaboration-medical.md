---
title: "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making"
excerpt: "An LLM moderator triages each medical query into PCC / MDT / ICT pathways, winning 7 of 10 medical benchmarks with up to +4.2% over the best prior method and ~half the API calls of MedAgents."
categories:
  - Paper
permalink: /paper/mdagents-adaptive-collaboration-medical/
tags:
  - MDAgents
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

- MDAgents inserts an LLM moderator that classifies each medical query as **low / moderate / high** complexity and routes it to a **Primary Care Clinician (PCC)**, a **Multi-Disciplinary Team (MDT)**, or an **Integrated Care Team (ICT)** — mirroring real clinical triage instead of forcing a fixed agent committee on every case.
- **Headline result:** best on **7 of 10** medical benchmarks (text, image+text, video+text) using GPT-4(V), with up to **+4.2% (p < 0.05)** over the best prior method, while using **~9.3 API calls vs MedAgents' 20.3** at N=3 agents.
- The moderator-review + MedRAG ablation lifts average accuracy from **71.8% to 80.3% (+11.8%)**; on MedQA, the classifier selects the optimal complexity tier ~**81% (±29%)** of the time — the load-bearing claim of the framework.

## Motivation

Medical decision-making is intrinsically tiered. A primary-care physician handles routine cases alone, but complex multi-organ cases get escalated to multidisciplinary teams or sequential specialist consultations. Existing LLM systems for medicine either run a single model with prompt engineering (Medprompt, Ensemble Refinement) or a fixed-size committee of agents (MedAgents, ReConcile, AutoGen, DyLAN). Both ignore the clinical reality that not every question needs a tumor board. The authors argue this static design wastes compute on easy cases and under-resources hard ones, and propose a framework whose collaboration topology is selected per-query by an LLM moderator — paralleling ED triage.

## Core Innovation

- **Adaptive triage by an LLM moderator.** A GP-prompted LLM reads the query and classifies it as `low`, `moderate`, or `high` complexity using clinical constructs — acuity, comorbidity, severity of illness.
- **Three collaboration topologies routed per query.** `low → PCC` (single agent, 3-shot), `moderate → MDT` (N specialists in a round-table with GP-moderator), `high → ICT` (sequential sub-teams whose written reports condition the next team).
- **Moderator review + MedRAG.** The moderator can break ties, critique each agent, and inject retrieval-augmented medical knowledge. The two together drive the +11.8% gain in the ablation — a synergistic pairing rather than a single dominant component.
- **Training-free orchestration.** Pure inference-time scaffolding over frozen GPT-4(V) / Gemini-Pro(Vision); no fine-tuning, no medical foundation model required.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | MDAgents outperforms prior solo and group methods on 7/10 medical benchmarks. | Table 2 main results, counted column-wise. | All 10 benchmarks. | ⭐⭐⭐ — directly tabulated, std over 3 seeds. |
| C2 | "Up to 4.2% (p < 0.05)" improvement over the best prior method. | Abstract + §4.2; matches MedQA delta vs CoT-SC (88.7 − 83.9 = 4.8) and PMC-VQA vs Medprompt (56.4 − 53.4 = 3.0). | Per-benchmark. | ⭐⭐ — significance procedure not detailed; per-dataset p-values not broken out. |
| C3 | The classifier LLM picks the optimal complexity tier ~80% of the time. | Figure 3, a = 0.81 ± 0.29. | **MedQA only** (25 problems × 10 reps). | ⭐⭐ — large stdev and single-benchmark estimation; the 80% headline may not transfer. |
| C4 | Moderator review + MedRAG yields +11.8% average accuracy. | Table 3. | Averaged across all datasets. | ⭐⭐ — averaging masks per-dataset variance; no per-dataset breakdown. |
| C5 | Adaptive is more compute-efficient (9.3 vs 20.3 API calls) while remaining more accurate. | §4.4 + Figure 6a-b. | Aggregate. | ⭐⭐ — believable but compares MDAgents N=3 vs MedAgents N=5; not like-for-like. |
| C6 | MDAgents is more robust to temperature than Solo / Group. | Figure 6c at T=0.3 / 1.2. | Aggregate. | ⭐ — only two temperature points; "robustness" is over-claimed without a sweep. |
| C7 | Adaptive converges to consensus across modalities. | Figure 7 entropy curves, N=30/dataset, Gemini-Pro(V). | 3 modality groups. | ⭐⭐ — clean qualitative trend; no statistical test on convergence rate. |
| C8 | The framework emulates real clinical triage and benefits from it. | Conceptual PCC/MDT/ICT mapping + Appendix F human-physician comparison. | Conceptual + qualitative. | ⭐ — analogy is suggestive but not falsifiable; physician comparison is sequestered in the appendix. |

**Honest read.** The 7-of-10 wins (C1) are well supported and the table is transparent about losses (PubMedQA, MedVidQA). The compute-efficiency story (C5) is real but slightly stacked — N=3 vs N=5 is not a fixed compute budget. The complexity-classifier accuracy (C3) is the *load-bearing* claim of the whole paper, since the framework collapses if the moderator misclassifies, yet it is evidenced on a single benchmark with a one-sigma spread (±0.29) that covers a wide range. The +11.8% ablation (C4) is impressive but reported only as an aggregate — without per-dataset numbers, we cannot tell whether MedRAG or the moderator review carries most of the gain. Significance is asserted (p < 0.05) but the procedure is undescribed in the main text, and **50 samples × 3 seeds** is a thin basis for statistical claims when much larger test sets exist. External validity beyond English USMLE-style MCQA is untested.

## Method & Architecture

![MDAgents pipeline overview](/assets/images/paper/mdagents/page_002.png)
*Figure 1: The MDAgents framework — complexity check → recruitment → analysis/synthesis → final decision, with single PCC / MDT / ICT pathways selected per query by an LLM moderator.*

### 1. Medical complexity check

A moderator LLM (acting as a GP) reads the medical query *q* and classifies it as `low`, `moderate`, or `high`. The prompt grounds the classifier in clinical constructs: *acuity* (low), *comorbidity* and *case-management complexity* (moderate), *severity of illness* (high).

### 2. Expert recruitment

A recruiter LLM enlists agents based on the complexity label:

- **Low → PCC:** a single Primary Care Clinician.
- **Moderate → MDT:** N specialists (typically N=5: Pathologist, Radiologist, Surgeon, Oncologist, Endocrinologist, with a GP moderator) at a round table.
- **High → ICT:** multiple sequential sub-teams (Initial Assessment Team → diagnostic teams → Final Review & Decision Team), each producing a written report consumed by the next team.

### 3. Analysis & synthesis

![MDT case from PMC-VQA: collaborative shift from fibrosis to hemorrhage](/assets/images/paper/mdagents/page_004.png)
*Figure 2: An MDT case from PMC-VQA. Five specialists initially split (3 fibrosis, 2 hemorrhage); after collaborative discussion and moderator review, consensus shifts to hemorrhage.*

- **Low:** `ans = Agent(Q)` via 3-shot prompting; no iteration.
- **Moderate:** iterative discussion of up to **R rounds** with **T turns/round**. Each round, agents declare participation and preferred interlocutors; messages are exchanged; consensus is parsed by comparing opinions. If consensus fails, the moderator critiques each agent and a new round opens with retained history. Termination is consensus-driven.
- **High:** each sub-team generates a report; reports accumulate; later teams condition on prior reports.

### 4. Decision-making

A decision-maker agent fuses outputs differently by complexity: directly use PCC answer (low), use full interaction history (moderate), or use accumulated reports (high). Temperature ensembles add robustness. Inference time per query: **14.7 s (low) / 95.5 s (moderate) / 226 s (high)**.

**Hyperparameters that matter:** N (peak accuracy at N=3 = 83.5%), temperature (Adaptive uniquely *benefits* from T=1.2 vs T=0.3), max rounds R, turns/round T_turn.

## Experimental Results

### Main accuracy (GPT-4(V); MedVidQA uses Gemini-Pro(V)). Bold = best per column.

| Setting | Method | MedQA | PubMedQA | Path-VQA | PMC-VQA | MedVidQA | DDXPlus | SymCat | JAMA | MedBullets | MIMIC-CXR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Solo | Zero-shot | 75.0±1.3 | 61.5±2.2 | 57.9±1.6 | 49.0±3.7 | 37.9±8.4 | 70.3±2.0 | 88.7±2.3 | 62.0±2.0 | 67.0±1.4 | 40.0±5.3 |
| Solo | CoT-SC | 83.9±2.7 | 58.7±5.0 | 61.2±2.1 | 50.5±5.2 | 49.2±8.2 | 52.1±6.4 | 83.3±3.1 | 68.0±2.8 | 76.0±2.8 | 51.7±4.0 |
| Solo | Medprompt | 82.4±5.1 | 51.8±4.6 | 59.2±5.7 | 53.4±7.9 | 44.5±2.0 | 59.5±17.7 | 87.3±1.2 | 70.7±4.3 | 71.0±1.4 | 53.4±4.3 |
| Group | MedAgents | 79.1±7.4 | 69.7±4.7 | 45.4±8.1 | 39.6±3.0 | 51.6±4.8 | 62.8±5.6 | 90.0±0.0 | 66.0±5.7 | 77.0±1.4 | 43.3±7.0 |
| Group | Meta-Prompting | 80.6±1.2 | 73.3±2.3 | 55.3±2.3 | 42.6±4.2 | — | 52.6±6.1 | 77.3±2.3 | 64.7±3.1 | 49.3±1.2 | 42.0±4.0 |
| Group (multi-model) | ReConcile | 81.3±3.0 | **79.7±3.2** | 57.5±3.3 | 31.4±1.2 | — | 68.4±7.4 | 90.6±2.5 | 60.7±5.7 | 59.5±8.7 | 33.3±3.4 |
| Group (multi-model) | AutoGen | 60.6±5.0 | 77.3±2.3 | 43.0±8.9 | 37.3±6.1 | — | 67.3±11.8 | 73.3±3.1 | 64.6±1.2 | 55.3±3.1 | 43.3±4.2 |
| Group (multi-model) | DyLAN | 64.2±2.3 | 73.6±4.2 | 41.3±1.2 | 34.0±3.5 | — | 56.4±2.9 | 75.3±4.6 | 60.1±3.1 | 57.3±6.1 | 38.7±1.2 |
| Adaptive | **MDAgents (Ours)** | **88.7±4.0** | 75.0±1.0 | **65.3±3.9** | **56.4±4.5** | 56.2±6.7¹ | **77.9±2.1** | **93.1±1.0** | **70.9±0.3** | **80.8±1.7** | **55.9±9.1** |

¹ Weighted Voting reports 57.8% on MedVidQA, marginally beating MDAgents (56.2%) — one of 3 benchmarks where MDAgents is *not* the top entry (the others: PubMedQA, where ReConcile leads at 79.7%, and a razor-thin margin on JAMA).

### Why adaptive routing works

![Complexity-classification quality and per-benchmark Solo vs Group vs MDAgents](/assets/images/paper/mdagents/page_008.png)
*Figure 3 + Figure 4: complexity-classification quality on MedQA and a per-benchmark bar showing MDAgents beating Solo and Group settings.*

- **Complexity classifier hits the optimal tier ~81% of the time on MedQA** (a = 0.81 ± 0.29, mid 0.11 ± 0.28, min 0.08 ± 0.16) — but the ±0.29 stdev is large, and this is a single-benchmark estimate.
- **Adaptive beats every fixed complexity level** — text-only 81.2% vs low 64.2% / moderate 71.6% / high 65.8%. Routing patterns are modality-dependent: 64% of text-only queries get classified `high`, 55% of image+text get `low`, 87% of video+text get `low`.

### Ablations: moderator review + MedRAG

![Static vs adaptive routing and the +11.8% moderator+MedRAG ablation](/assets/images/paper/mdagents/page_009.png)
*Figure 5 + Table 3: static complexity assignments vs adaptive routing across modalities, and the +11.8% boost from moderator review + MedRAG (71.8% → 75.2% → 77.6% → 80.3%).*

The two components are synergistic, but the paper reports only the average — there is no per-dataset MedRAG breakdown to tell us whether retrieval dominates the gain or the moderator review does.

### Robustness and consensus dynamics

![Number of agents, temperature robustness, and consensus entropy across modalities](/assets/images/paper/mdagents/page_010.png)
*Figure 6 + Figure 7: number-of-agents trade-off (peak at N=3 = 83.5%), temperature robustness (T=0.3 vs T=1.2), and consensus entropy decline across modalities (video+text converges fastest, text-only slowest).*

- **More agents ≠ better.** Peak accuracy at N=3 (83.5%); API calls: Solo (5-shot CoT-SC) 6.0; MedAgents (N=5) 20.3; MDAgents Adaptive (N=3) **9.3** — roughly half MedAgents' cost at higher accuracy.
- **Temperature robustness.** Adaptive improves at T=1.2 vs T=0.3 while Solo and Group regress — but with only two temperature points, "robust" is over-claimed.

## Limitations

**Authors acknowledge:** no use of medical foundation models (Med-Gemini, AMIE, Med-PaLM 2) that could share medical vocabulary across agents; MCQ-only scope with no patient/caregiver in the loop; risk of medical hallucination; MedVidQA contains relatively easy content.

**Not addressed but visible in the work:**

- **Sample size.** 50 examples per benchmark × 3 seeds is small for claims at p < 0.05; the full test sets exist and were not used.
- **Cost reporting is partial.** Total token cost (vs API call count) is not given, and **226 s/case at high complexity** is concerning for deployment.
- **Closed-source dependency.** Most main-text numbers depend on GPT-4(V); open-model results are sequestered in the appendix.
- **Recruiter ablation is missing.** Only the complexity classifier is ablated; we do not know what happens if the recruiter picks the *wrong* specialists for an MDT.
- **No specialist-tuned medical LLM comparison.** No head-to-head with Med-PaLM 2 or Med-Gemini, only with general LLMs wrapped in prompting/agent scaffolds.
- **Equity / bias.** No analysis of whether complexity classification or final accuracy varies by patient demographics encoded in vignettes.
- **No external clinical validation.** No real-world deployment, no large-scale physician-panel comparison on the same MCQs.

## Why It Matters for Medical AI

MDAgents is the first multi-agent LLM framework for medicine that takes triage seriously as an *architectural* choice rather than a prompt-engineering tweak. The PCC/MDT/ICT routing is a direct translation of how care actually escalates, and the demonstration that an LLM moderator can pick the right tier ~80% of the time (on MedQA) makes adaptive collaboration look like a real lever for inference-time compute allocation in clinical decision support. The honest reading is narrower: the framework wins on MCQA in English with frozen GPT-4(V), with thin sample sizes, no medical-foundation-model comparison, and a single benchmark validating the load-bearing classifier claim. For deployment, the gap between "wins 7/10 USMLE-style MCQs" and "safe clinical assistant" remains the entire field.

## References

- Paper: [arXiv 2404.15155](https://arxiv.org/abs/2404.15155) — NeurIPS 2024
- Code: [github.com/mitmedialab/MDAgents](https://github.com/mitmedialab/MDAgents)
- Related: MedAgents (Tang et al. 2024), ReConcile (Chen et al. 2024), AutoGen (Wu et al. 2023), DyLAN (Liu et al. 2024), Medprompt (Nori et al. 2023), MedRAG (Xiong et al. 2024)
