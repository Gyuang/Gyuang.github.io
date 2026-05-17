---
title: "AgentClinic: A Multimodal Agent Benchmark to Evaluate AI in Simulated Clinical Environments"
excerpt: "An OSCE-style four-agent simulator (doctor / patient / measurement / moderator) across 5 sub-benchmarks where Claude-3.5 Sonnet leads (62.1% on AgentClinic-MedQA) and frontier MedQA scores collapse 1.5-2x when forced into sequential dialogue."
categories:
  - Paper
permalink: /paper/agentclinic-multimodal-benchmark-clinical-agents/
tags:
  - AgentClinic
  - LLM-Agents
  - Dataset
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- AgentClinic reformats static medical MCQ benchmarks into a four-agent sequential simulator — **doctor / patient / measurement / moderator** — covering 5 sub-benchmarks (MedQA 215, MIMIC-IV 200, NEJM 120, Spec 260, Lang 749), 9 specialties, 7 languages, 13 image types, 24 cognitive/implicit bias prompts, and a 6-tool Agent Toolbox (CoT variants, two Adaptive-RAG sources, a persistent Notebook).
- **Headline result:** **Claude-3.5 Sonnet leads** on AgentClinic-MedQA at **62.1% +/- 3.3** and on MIMIC-IV at 42.9 +/- 3.3; **o1-preview reaches 80.6 +/- 5.6** on the MedQA subset (Appendix G). On NEJM multimodal, Claude-3.5 leads at 37.2 +/- 2.2 with images pre-loaded.
- The abstract's "drop to a tenth" headline applies **only to the worst models** (Llama-2-70B-chat collapses to 4.5%); frontier models drop **~1.5-2x** (GPT-4 ~90% MedQA -> 51.6% AgentClinic-MedQA). The strongest mechanistic explanation is the coverage analysis in Appendix F.3: GPT-4 extracts only **67% of relevant info** from the original vignette through sequential questioning.

## Motivation

LLM medical evaluation has been dominated by USMLE-style MCQ. MedQA accuracy rose from 38.1% (2021) to 90.2% (Nov 2023), yet residency literature shows USMLE scores are weakly predictive of real clinical performance (Lombardi 2023). Clinical work is sequential, partial-information, modality-mixed, and dialogue-driven — none of which an MCQ measures. Existing dialogue benchmarks (AMIE, CRAFT-MD, SAPS) are closed-source and miss multimodality, tool use, multilinguality, specialist coverage, and bias simulation. AgentClinic targets exactly that gap as a reproducible, OSCE-style agent harness.

## Core Innovation

- **Role-partitioned four-agent simulator.** Each role is a separately prompted LLM whose information is strictly scoped: the doctor sees only an objective string, the patient never sees the diagnosis, the measurement agent owns lab/exam values, and the moderator judges only the final free-text diagnosis against ground truth.
- **5 sub-benchmarks across modalities, specialties, languages.** MedQA / MIMIC-IV / NEJM-multimodal / Spec (9 specialties) / Lang (7 languages) — broader contemporary coverage than any peer dialogue benchmark.
- **Cognitive + implicit bias injection with patient-perception metrics.** 6 cognitive + 6 implicit biases each on patient and doctor sides, with the patient agent rating doctor Confidence/Compliance/Consultation on 1-10 scales after diagnosis.
- **Agent Toolbox.** Six pluggable tools selectable in the doctor's system prompt — Zero/One/Reflection CoT, two Adaptive-RAG sources (18 medical textbooks; PubMed + StatPearls + Wikipedia), and a 1000-character cross-case persistent **Notebook** that the doctor must rewrite each turn.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | "Solving MedQA in sequential format drops accuracy to below a tenth of original." | Llama-2-70B-chat 4.5% on AgentClinic-MedQA vs much higher MedQA. | MedQA -> AgentClinic-MedQA (Fig. 3) | **⭐⭐** — true only for the *worst* models; for GPT-4 the drop is ~90% -> 52%, i.e. ~1.7x, not 10x. |
| C2 | Claude-3.5 outperforms other LLMs in most settings. | Fig. 2 / Table 4 / Table 6 / §F.4 — wins MedQA, MIMIC-IV, Lang, most tools, NEJM, multi-agent. | All 5 sub-benchmarks | **⭐⭐⭐** — broad, consistent, well documented. |
| C3 | MedQA accuracy is not predictive of AgentClinic-MedQA accuracy. | Fig. 3 scatter. | MedQA vs AgentClinic-MedQA | **⭐⭐** — visual claim only; no R^2, Spearman, or p-value; n ~ 10 models is underpowered. |
| C4 | Llama-3 gains up to 92% relative from the Notebook tool. | Table 6: 19% -> 41.1% with Notebook + Reflection. | AgentClinic-MedQA only | **⭐⭐** — single benchmark, single model; not replicated on MIMIC-IV / NEJM. |
| C5 | Cognitive and implicit biases reduce diagnostic accuracy. | §3.2, Fig. 4 — Mixtral drops to 78-89% normalized. | MedQA only, GPT-4 + Mixtral only | **⭐⭐** — only 2 doctor models tested; GPT-4 effect within noise (max 4% abs); GPT-4 verbosely *refuses* 25/215 bias prompts, contaminating measurement. |
| C6 | Patient agents report lower confidence / compliance / consultation under bias. | Fig. 4 bottom panels. | MedQA | **⭐⭐** — coherent signal, but the rater is itself a GPT-4 patient agent, so this measures *LLM simulation of patient reaction*, not human patient reaction. Authors acknowledge. |
| C7 | AgentClinic supports realistic multimodal diagnosis. | §3.7, 120 NEJM cases with images. | NEJM only | **⭐⭐** — small benchmark; even best (Claude-3.5) at 37.2%; per-image-type cells (Table 1) include n <= 6 with no uncertainty bars. |
| C8 | Tools differentially help models. | Table 6. | MedQA | **⭐⭐** — clear pattern but single-dataset, single-run per condition; no seed variance reported per cell. |
| C9 | The LLM moderator is reliable. | Citation to Zheng 2023 (>80% LLM-human agreement). | — | **⭐** — no AgentClinic-specific moderator-vs-clinician validation reported. |
| C10 | Coverage gap explains the MedQA <-> AgentClinic gap. | §F.3 — GPT-4 extracts 67% avg (72% correct dx vs 63% incorrect). | MedQA subset, GPT-4 only | **⭐⭐** — the most mechanistic explanation in the paper; sample size unspecified, one doctor model only. |

**Honest take.** The methodological contribution — operationalizing OSCE-style sequential evaluation with role-partitioned agents, a real toolbox, and serious multilingual + specialist coverage — is genuinely valuable. The "drop to a tenth" framing is true for the *worst* models; for frontier LLMs the drop is closer to 1.5-2x, still meaningful but not 10x. The **bias section is the weakest leg**: only two doctor models, GPT-4 refusing 25/215 bias prompts, and patient-perception scores produced by LLMs rating LLMs — a closed loop that may primarily measure GPT-4's persona-following consistency. Variance reporting is uneven (95% CIs on main accuracy tables, missing on most tool-use cells). **The biggest unaddressed risk is training-data leakage**: GPT-4 / Claude-3.5 may have been trained on MedQA, MIMIC-IV, MedMCQA, and NEJM case texts. The authors raise this and offer Fig. 3 non-predictivity as mitigating evidence, but Fig. 3 is also consistent with leakage of vignette content while the *sequential format* still degrades performance.

## Method & Architecture

![AgentClinic four-agent simulator with dialogue example](/assets/images/paper/agentclinic/fig_p003_01.png)
*Figure 1: AgentClinic four-agent loop — the doctor agent queries the patient and measurement agents under a turn budget, then the moderator scores the final diagnosis against ground truth.*

### 1. Four agents, scoped information

- **Doctor agent (system under test)** — sees only an objective string ("Evaluate patient with chest pain..."); must ask questions or `Request Test: [name]` within `N` turns; ends with `Diagnosis Ready: [...]`.
- **Patient agent** — conditioned on history and symptoms, never on the diagnosis; instructed not to volunteer the disease name.
- **Measurement agent** — conditioned on physical exam findings and lab values from the OSCE template; returns results in a formatted block (e.g., CBC panel in Appendix J.2). Falls back to "Normal Readings" if the requested test is not in the template.
- **Moderator agent** — given the ground-truth diagnosis and the doctor's free-text answer, emits Yes/No. Justified by appeal to Zheng et al. 2023 (>=80% LLM-judge agreement).

### 2. OSCE template and case construction

Cases are serialized to JSON with strict fields (Demographics, History, Primary/Secondary Symptoms, PMH, Social Hx, ROS, Vital Signs, exam findings, Test Results, Correct Diagnosis); each field is fed only to the agent that should own it (Appendix I). GPT-4 populates the JSON from MedQA / MIMIC-IV / MedMCQA / NEJM source text, then **humans manually validate every case**. Multilingual cases are GPT-4-translated and then **native-speaker corrected**.

### 3. Five sub-benchmarks

| Sub-benchmark | Size | Source | Modality | Access |
|---|---|---|---|---|
| AgentClinic-MedQA | 215 | USMLE via MedQA (Jin 2021) | text | MedQA-derived |
| AgentClinic-MIMIC-IV | 200 | MIMIC-IV — single-dx patients from first 200 of ~6k | text | PhysioNet credentialed |
| AgentClinic-NEJM | 120 | NEJM Case Challenges (sampled from 932) | text + image | NEJM paywall |
| AgentClinic-Spec | 260 | MedMCQA — 9 specialties | text | MedMCQA license |
| AgentClinic-Lang | 749 | AgentClinic-MedQA translated to 6 non-English languages | text | derivative |

### 4. Bias, tools, multimodality

- **Bias injection** (Appendices A.2, L.5, L.6): static system-prompt snippets implement 6 patient cognitive, 6 doctor cognitive, 6 patient implicit, and 6 doctor implicit biases (paper variously says 23-24). Reported as normalized accuracy `Acc_bias / Acc_NoBias`.
- **Agent Toolbox** (Appendix K): Zero-shot CoT, One-shot CoT, Reflection CoT, Adaptive RAG over 18 medical textbooks (Jin 2021), Adaptive RAG over PubMed + StatPearls + Wikipedia, and a **Notebook** — a 1000-character cross-case persistent memory that the doctor must rewrite each turn to retain prior notes.
- **Multimodal input (NEJM)**: image given at initialization vs. image returned by the measurement agent on request. 13 image-type categories (Physical, CT, Derm, Path, X-ray, Ophth, MRI, Biopsy, Surgery, Instrument, ECG, Echo, US).
- **Patient-centric metrics** (§3.3): after diagnosis, the patient agent rates the doctor 1-10 on Confidence, Compliance, Consultation.
- **Defaults**: `N=20` doctor turns (~40 dialogue lines); for MedQA-doctor evaluation, *patient*, *measurement*, and *moderator* are all GPT-4 (gpt-4-0613) to standardize.

## Experimental Results

### Main accuracy (N=20; GPT-4 acts as patient / measurement / moderator)

| Model | AgentClinic-MedQA (% ± 95% half-width) | AgentClinic-MIMIC-IV (% ± 95%) |
|---|---|---|
| **Claude-3.5-Sonnet** | **62.1 ± 3.3** | **42.9 ± 3.3** |
| OpenBioLLM-70B | 58.3 ± 4.2 | 38.1 ± 3.2 |
| Human physicians (n=3) | 54 ± 28.5 | — |
| GPT-4 (0613) | 51.6 ± 3.3 | 34.0 ± 3.1 |
| Mixtral-8x7B | 37.1 ± 3.1 | 29.5 ± 3.1 |
| GPT-3.5 | 36.6 | 27.5 ± 3.0 |
| GPT-4o (2024-05-13) | 34.2 ± 3.4 | 24.0 ± 2.9 |
| MedLlama3-8B | 31.4 ± 2.9 | 29.7 ± 2.6 |
| Meditron-70B | 29.1 ± 2.4 | 25.5 ± 2.43 |
| PMC-Llama-7B | 23.6 ± 2.1 | 34.3 ± 3.0 |
| Llama-3-70B-Instruct | 19.0 ± 2.5 | 8.5 ± 1.9 |
| Llama-2-70B-chat | 4.5 ± 1.3 | 13.5 ± 2.2 |
| **o1-preview** (Appendix G) | **80.6 ± 5.6** | n/a (cost) |

![Doctor-model accuracy across AgentClinic-MedQA, patient-LLM ablation, and AgentClinic-MIMIC-IV](/assets/images/paper/agentclinic/fig_p004_01.png)
*Figure 2: Doctor-model accuracy on AgentClinic-MedQA (left), patient-LLM ablation (middle), AgentClinic-MIMIC-IV by model (right). Claude-3.5 leads on both text-only sub-benchmarks.*

### Static vs sequential — the headline collapse

![MedQA-vs-AgentClinic-MedQA scatter](/assets/images/paper/agentclinic/fig_p005_01.png)
*Figure 3: Static-MCQ MedQA accuracy is only weakly predictive of AgentClinic-MedQA accuracy; reformatting matters. Note no R^2 / Spearman is reported and n ~ 10 is underpowered.*

The famous "drop to a tenth" only describes the worst models (Llama-2-70B-chat 4.5%). Frontier models drop closer to 1.5-2x: GPT-4 ~90% -> 51.6%; the **coverage analysis in §F.3** is the most mechanistic explanation in the whole paper — GPT-4 extracts only **67% of relevant info** through sequential questioning (72% on correct diagnoses, 63% on incorrect).

### AgentClinic-NEJM (multimodal, 120 cases)

| Model | Image at init (%) | Image on request (%) |
|---|---|---|
| **Claude-3.5-Sonnet** | **37.2 ± 2.2** | 35.4 ± 2.4 |
| GPT-4 | 27.7 ± 2.0 | 25.4 ± 2.1 |
| GPT-4o | 21.4 ± 1.7 | 19.1 ± 1.4 |
| GPT-4o-mini | 8.0 ± 1.2 | 6.1 ± 1.2 |

![AgentClinic-NEJM accuracy by image delivery mode](/assets/images/paper/agentclinic/fig_p011_01.png)
*Figure 6: AgentClinic-NEJM accuracy when images are pre-loaded (pink) vs requested via the measurement agent (blue); Claude-3.5 leads by ~10 points.*

### Bias and patient-perception metrics

![Bias normalized accuracy and patient-perception bars](/assets/images/paper/agentclinic/fig_p007_01.png)
*Figure 4: Top — normalized accuracy under cognitive / implicit biases for GPT-4 vs Mixtral. Bottom — patient-agent self-reported Confidence, Compliance, Consultation under each bias.*

- GPT-4 minimally affected (avg ~97%, max 4% absolute drop) — but the paper notes GPT-4 verbosely **refuses 25/215 bias prompts**, contaminating the measurement.
- Mixtral-8x7B drops to **78.4%** normalized under doctor cognitive bias (37% -> 29% absolute).
- Patient-perception ratings drop sharply under implicit bias even when accuracy holds — education bias hits Confidence / Compliance / Consultation; self-diagnosis cognitive bias drops Confidence by 4.7 points.
- Caveat: the rater is itself a GPT-4 patient agent. This measures **LLM simulation of patient reaction**, not human patient reaction.

![Sample dialogues showing cognitive-bias prompts in doctor and patient turns](/assets/images/paper/agentclinic/fig_p021_01.png)
*Figure 7: Sample dialogues illustrating how cognitive-bias prompts surface in doctor and patient turns.*

### Multilingual and specialty coverage

![Accuracy across 7 languages, 9 specialties, and 6 agent tools](/assets/images/paper/agentclinic/fig_p009_01.png)
*Figure 5: Accuracy across 7 languages (left), 9 specialties (middle), and 6 agent tools (right).*

- **Claude-3.5 Sonnet 48.4% avg across 7 languages — more than 2x the next best** (GPT-4 at 20.9%). English best for all; Chinese remains hardest for everyone except Claude (GPT-4 drops to 11.21% Chinese vs 40.18% English).
- MedAgents + Claude-3.5: **65.2 ± 3.6** in the multi-agent setting (Table 3) — only marginally above single-agent Claude-3.5 (62.1).

### Agent Toolbox deep dive

![Per-model absolute and relative accuracy across the six Agent Toolbox tools](/assets/images/paper/agentclinic/fig_p041_01.png)
*Figure 11: Per-model absolute and relative-to-baseline accuracy across the six Agent Toolbox tools.*

![Llama-3 70B experiential learning curve with the Notebook tool](/assets/images/paper/agentclinic/fig_p041_02.png)
*Figure 12: Llama-3 70B accuracy rises with cumulative Notebook updates across cases — the experiential-learning effect.*

- **Llama-3-70B + Notebook -> 41.1%** from ~19% baseline (the largest tool gain; cited as up to **92% relative**, but single-benchmark, single-model).
- Claude-3.5 best overall with tools (avg 51.3%, peak 56.1% with Notebook).
- GPT-4 peak 43.9% with Adaptive RAG (Web); 42.2% with Reflection CoT.
- GPT-3.5 *loses* up to 27.1% with Adaptive RAG (Book) — tools are not uniformly helpful.

### Clinician realism reader study

![Three MD raters scored 20 dialogues on doctor / patient / measurement realism and empathy](/assets/images/paper/agentclinic/fig_p023_01.png)
*Figure 8: Three MD raters scored 20 dialogues for doctor / patient / measurement realism and empathy on a 1-10 scale.*

Average scores: Doctor 6.2, Patient 6.7, Measurement 6.3, Empathy 5.8. Common faults: doctor agent jumps to symptoms without an opening greeting and fixates on diagnoses; patient agent over-verbose / parrots; measurement agent occasionally returns partial lab panels.

## Limitations

**Authors acknowledge:**

- LLM moderator (mitigated only by Zheng 2023, no in-domain clinician validation on the actual AgentClinic dialogues).
- Measurement agent may hallucinate values; SQL/database back-end suggested as future work.
- Possible training-data leakage of MedQA / MIMIC-IV / NEJM in proprietary models.
- Cross-model patient-doctor pairing may disadvantage non-matched LLMs (Panickssery 2024 self-recognition effect).
- Simulated patients != real patients; perception ratings are persona-LLM artifacts.
- Cohort missing nurses, families, administrators, insurers, embodied constraints.

**Not addressed / under-addressed:**

- **Training-data leakage is the central unaddressed risk.** Fig. 3 non-predictivity is offered as mitigation, but is equally consistent with vignette-content leakage plus a real sequential-format penalty. No release-date gating between commercial-model knowledge cutoffs and NEJM case publication dates.
- No multi-seed variance for tool-use cells; per-cell CIs missing in Table 6.
- No inter-moderator agreement (LLM vs clinician) on the actual AgentClinic dialogues — would directly validate C9.
- MIMIC-IV cohort filtered to **single-diagnosis patients**, sidestepping the comorbidity complexity that defines real EHR work.
- No specialty- or image-type failure-mode analysis beyond raw accuracy (Table 1 cells with n <= 6 reported without uncertainty).
- "92% relative improvement" framing for Llama-3 + Notebook obscures that the *absolute* still trails much smaller numbers from Claude / GPT-4.
- Demographic-bias study restricted to race summary stats; no per-race accuracy slices.
- Translation-quality confound never quantified for the Lang sub-benchmark.

## Why It Matters for Medical AI

AgentClinic is the most coverage-broad open dialogue-style clinical evaluation harness to date — five sub-benchmarks, nine specialties, seven languages, real multimodality, an honest toolbox, and patient-perception metrics that no MCQ benchmark can produce. The reframing it forces — that **MedQA scores overstate clinical readiness** because sequential information extraction is a separate skill — is the right reframing for the field, and the §F.3 coverage analysis (GPT-4 extracts only 67% of the relevant info) gives a concrete mechanism rather than a vibe. Two warnings for readers: the abstract's 10x collapse is a worst-case framing (frontier models drop 1.5-2x), and the bias section is the weakest leg of the paper (two doctor models, GPT-4 refusing prompts, LLM-judging-LLM perception loops). For deployment, AgentClinic is an evaluation harness, not a safety case — leakage remains the elephant in the room, and the closed-loop patient simulator is a persona, not a person.

## References

- Paper: [arXiv:2405.07960v5](https://arxiv.org/abs/2405.07960) (25 May 2025) — npj Digital Medicine track
- Code / cases: project repository linked from the paper
- Source benchmarks: MedQA (Jin et al. 2021), MIMIC-IV (Johnson et al. 2023), MedMCQA (Pal et al. 2022), NEJM Case Challenges
- Related: MedAgents (Tang et al. 2024), AMIE (Tu et al. 2024), CRAFT-MD (Johri et al. 2024), Medprompt (Nori et al. 2023), MedRAG (Xiong et al. 2024), LLM-as-judge (Zheng et al. 2023)
