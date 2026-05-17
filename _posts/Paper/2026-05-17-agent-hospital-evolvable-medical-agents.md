---
title: "Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents"
excerpt: "A 16-room LLM-agent hospital where doctors evolve via two RAG memories — 92.22% on MedQA with GPT-4o (96.15% with o1-preview), but margins are <1pp over Medprompt without variance reporting."
categories:
  - Paper
permalink: /paper/agent-hospital-evolvable-medical-agents/
tags:
  - Agent-Hospital
  - MedAgent-Zero
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

- A simulacrum of a 16-area hospital is populated by LLM-powered patient, nurse, and doctor agents; doctors evolve by treating tens of thousands of synthetic patients and reformatting textbooks into Q-A pairs, all without manually labeled training data.
- The training recipe — **MedAgent-Zero / SEAL** — keeps the base LLM frozen and grows two RAG-backed memories: a **medical case base** (successful Q-A pairs) and an **experience base** (natural-language rules distilled from failures and self-validated).
- **Headline:** MedQA-USMLE **92.22%** with GPT-4o (Hybrid **92.77%**), **96.15%** with o1-preview — but the margin over Medprompt is **<1pp** with no variance, no seeds, and no significance test.

## Motivation

LLM-based medical AI has focused on phase 1 of becoming a doctor — absorbing textbook knowledge via pre-training and fine-tuning (Med-PaLM, etc.) — while ignoring phase 2: gaining expertise through years of clinical practice. Real-world apprenticeship is slow and feedback-poor (most patients never come back to tell you whether the prescription worked). Existing LLM medical-agent work (MedAgents, MDAgents, Medprompt) focuses on multi-agent collaboration at inference time and has no mechanism for cumulative experience. The authors propose to bypass these constraints by constructing a closed-loop synthetic hospital where (a) feedback always exists, (b) time runs orders of magnitude faster than the real world, and (c) patient cohorts can be generated on demand — letting one frozen LLM "evolve" via accumulated memory rather than gradient updates.

## Core Innovation

- **Simulacrum as training ground.** A 16-area sandbox (triage, registration, consult rooms, exam, pharmacy, follow-up, library) built with Tiled + Phaser, inspired by Smallville. 42 doctor agents across 21 clinical departments + 11 non-clinical departments, covering **339 diseases**.
- **Closed-cycle care.** Eight event types drive the loop: Disease Onset → Triage → Registration → Consultation → Medical Examination → Diagnosis → Medicine Dispensary → Convalescence (+ optional re-visit, + "Reading Books" for off-hours learning).
- **MedAgent-Zero evolution.** Base LLM is frozen. Two external memories grow during practice — a **case base** (success memory) and an **experience base** (failure-derived natural-language rules, validated against held-out Q-A before admission).
- **Textbook-as-Q-A.** Documents are reformatted by an LLM into MCQ form so they fit the same schema as patient cases — sidestepping parametric fine-tuning entirely.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | Doctor agents acquire expertise through practice in a closed-loop simulacrum without labeled training data. | Fig. 5a-b, Fig. 9, Table 3 — accuracy rises with #patients across many departments. | Self-generated synthetic patients. | ⭐⭐ — train and test are both LLM-generated from the same KB, i.e. in-distribution self-evaluation. |
| C2 | Skills learned in the virtual world transfer to the real world (MedQA). | Fig. 5b red trend curve; Table 4. | MedQA respiratory subset + full MedQA. | ⭐⭐ — trend is small and the red scatter is wide; "Hybrid" variant explicitly leaks MedQA training Q-A pairs. |
| C3 | MedAgent-Zero outperforms SOTA medical-agent methods on MedQA. | Table 4. | MedQA. | ⭐⭐ — single-run accuracies, no CIs; margins ≤0.79 pp on GPT-4o / o1-preview over Medprompt. |
| C4 | Method exhibits **scaling laws in evolution** (analogous to Kaplan et al.). | Fig. 5b virtual-world curve to 50k patients. | Respiratory department only. | ⭐ — one curve, one department; ref 49 mis-cites the Kaplan paper; no power-law fit reported. |
| C5 | Both case base and experience base contribute, with experience base more important. | Fig. 12 ablation (92.22 vs 91.36 vs 90.65 vs 88.22). | MedQA. | ⭐⭐ — clean ablation, single base model, no variance. |
| C6 | Generalizes across **all** 32 departments / 339 diseases. | Table 3 (21 clinical depts) + Fig. 7 (top-5 diseases per remaining dept). | Synthetic patients. | ⭐⭐ — broad coverage shown, but tested on the same generator that produced training cases. |
| C7 | "Human doctors take several years"; the simulacrum compresses time. | Narrative only. | — | ⭐ — rhetorical, not measured. |

**Honest read.** The paper's most interesting claim — that experience accumulated in a synthetic world transfers to a real-world benchmark — rests on a single set of numbers (Table 4 MedQA) where MedAgent-Zero beats the strongest baseline by **less than 1pp** on GPT-4o and o1-preview, with **no confidence intervals, no seeds, no significance tests**. The "Hybrid" variant that posts the headline 92.77% explicitly mixes MedQA training data into the case base — which directly undermines the "Zero / no labeled data" framing. The in-simulator gains (Examination **66.14 → 98.76%**, Diagnosis **76.98 → 95.31%**) are dramatic but the test patients come from the same LLM generator as the training patients, so "ground truth" is "the disease the generator was told to simulate" — the model is being graded by itself. Fig. 8 acknowledges an **OpenAI API update at ~30k patients**, meaning the underlying base model silently changed mid-run — the longitudinal "scaling" curve is on a non-stationary base. Fig. 9 shows visible **memory poisoning** in Cardiology between 12k and 14k patients ("unhelpful experience accumulated") with no mitigation beyond the post-hoc validation pass. Reference 49 cites the Kaplan scaling-laws arXiv ID with the wrong title, suggesting the scaling-law analogy is rhetorical rather than analytical.

## Method & Architecture

![Agent Hospital simulacrum — 16-area sandbox with LLM-powered patients, nurses, and doctors](/assets/images/paper/agent-hospital/fig_p002_01.png)
*Figure 1: The Agent Hospital simulacrum — a 16-area sandbox where LLM-powered patient, nurse, and doctor agents go through the full closed cycle from disease onset to follow-up.*

### 1. Simulacrum construction

- **Sandbox**: 16 functional areas built with Tiled (map editor) + Phaser (HTML5 game engine), explicitly inspired by Smallville (Park et al., UIST 2023).
- **Populations**: 42 doctor agents across **21 clinical departments** + 11 non-clinical departments (32 total, covering **339 diseases**), 4 nurse agents; medical staff are assumed never to fall ill.
- **Disease knowledge** scraped from **Baidu Health Encyclopedia** + common-disease lists from **DXY** (Chinese medical site), supplemented by GPT-4 for under-represented departments. Note: a Chinese-language KB underlies an English benchmark evaluation, and the paper does not analyze the coverage mismatch.

### 2. Closed-cycle treatment

![One closed-cycle treatment episode in Agent Hospital](/assets/images/paper/agent-hospital/fig_p004_01.png)
*Figure 2: One closed-cycle treatment episode — triage → registration → consultation → exam → diagnosis → pharmacy → recovery → optional re-visit.*

### 3. Patient generation (5 sequential LLM calls per patient)

1. **Disease selection** from the curated DXY + Baidu KB list.
2. **Basic demographics** biased by KB priors (e.g., Herpes Zoster more common >50).
3. **Medical history** (e.g., prior chickenpox → herpes zoster risk).
4. **Symptoms** conditioned on disease + history.
5. **Exam reports** generated from KB reference standards — never naming the target disease.

A separate **quality-control agent** screens generated patients against the KB before they enter the simulator.

### 4. MedAgent-Zero evolution (frozen LLM)

Two external memories grow during practice:

1. **Medical case base** (success memory). Two sources:
   - **Patient-doctor interaction**: every correct diagnosis Q→A is appended (one separate case base **per task per department**: examination selection, diagnosis, treatment).
   - **Textbook learning**: documents reformatted by an LLM into MCQ form so they fit the same Q→A schema — no parametric fine-tuning.
2. **Experience base** (failure memory) — three sub-routines:
   - **Reflection**: on a wrong answer, the agent compares its answer to ground truth and writes a natural-language rule (à la tuning-free rule accumulation, Yang et al. EMNLP 2023).
   - **Validation**: each rule is tested against exemplar Q-A pairs; survives only if it produces correct diagnoses, otherwise discarded.
   - **Refinement**: rules reformatted to a canonical template; at inference an extra LLM judgment filters top-K retrieved rules for relevance.
3. **Inference prompt** (4 blocks): Instruction (role) · Patient Information (symptoms ± exam results) · Candidate Choices · Personal Experience (top-n cases + top-k rules from RAG).
4. **Retrieval**: cosine similarity over `text-embedding-ada-002` vectors. Hyperparameters: **top-3 cases + top-4 experiences** are optimal on MedQA (Fig. 11).
5. **Training scale**: 20,000 patients per department for training, 200 held-out per department for evaluation; respiratory department curve shown up to 50,000.

## Experimental Results

### MedQA-USMLE (Table 4)

| Method | GPT-3.5 | GPT-4 | GPT-4o | o1-preview |
|---|---|---|---|---|
| Direct | 58.29 | 78.16 | 88.22 | 95.05 |
| CoT | 64.02 | 83.11 | 90.42 | — |
| MedAgents | 66.30 | 84.45 | 89.24 | — |
| Medprompt\* | 71.09 | 88.30 | 91.12 | 94.50 |
| Medprompt | 73.76 | 89.47 | 91.52 | 95.36 |
| **MedAgent-Zero** | **74.31** | **89.71** | **92.22** | **96.15** |
| **MedAgent-Zero (Hybrid)** | **76.83** | **91.20** | **92.77** | **96.15** |

Gains over the strongest baseline (Medprompt) are **+0.55** (GPT-3.5), **+0.24** (GPT-4), **+0.70** (GPT-4o), **+0.79** (o1-preview) — small in absolute terms and reported without confidence intervals. The Hybrid variant injects MedQA training Q-A pairs directly into the case base, which contradicts the "Zero" framing.

### In-simulator results (Table 3, 21 clinical departments, 200 held-out test patients/department)

| Metric | Before evolution | After 20k patients | Δ |
|---|---|---|---|
| **Examination accuracy** | 66.14% | **98.76%** | +32.62 pp |
| **Diagnosis accuracy** | 76.98% | **95.31%** | +18.33 pp |

Extreme cases: Psychiatry examination **23.5 → 99.5** (+76 pp), Cardiology diagnosis **52.5 → 96.0**. Per-department evolved diagnosis ≥ 87.5% in every department. The catch: test patients are drawn from the same generator as training patients, so this is in-distribution self-evaluation — gains may reflect learning the generator's quirks rather than medicine.

### Scaling & dynamics

![Virtual-to-real transfer: respiratory department, 0 to 50k patients](/assets/images/paper/agent-hospital/fig_p008_01.png)
*Figure 5b: As doctor agents treat more synthetic patients (0 → 50k, respiratory), both virtual-world diagnostic accuracy (blue) and MedQA respiratory-subset accuracy (red, ~88.2 → 92.2) trend upward — the paper's main "virtual → real transfer" evidence. The red scatter is wide; only the trend curve is monotone.*

![Per-department top-5 disease accuracy before vs. after MedAgent-Zero evolution](/assets/images/paper/agent-hospital/fig_p008_02.png)
*Figure 5a (excerpt): Diagnostic accuracy on the top-5 disease categories per department before (blue) and after (orange) evolution — e.g., rheumatic heart disease in Cardiology jumps from 9% to 82%.*

- **Fig. 8** shows validated experiences growing sub-linearly while errors grow super-linearly; a discontinuity around 30k patients is attributed to an **"OpenAI API update"** — the underlying base model was silently changed mid-run.
- **Fig. 9** shows Cardiology segment accuracy dropping noticeably between 12k and 14k patients, attributed to **"unhelpful experience accumulated"** — i.e., the memory can poison itself, and the post-hoc validation pass does not fully prevent it.
- **Ablation (Fig. 12, GPT-4o)**: full system **92.22** · w/o experience base **91.36** · w/o case base **90.65** · Direct **88.22**. Removing the experience base hurts more than removing the case base, but the largest jump is Direct → either memory alone.
- **Hyperparameter (Fig. 11)**: peaked at top-3 cases / top-4 experiences; degrades on either side (e.g., top-8 experiences = 91.28 vs. top-4 = 92.22).

## Limitations

**Authors acknowledge:**
- Base LLM is frozen, not evolvable.
- Doctors only recommend high-level treatment plans (not dosages, scheduling).
- No cross-department consultation.
- Bias inheritance / ethical concerns are flagged at a discussion level only.

**Not addressed (audit-level concerns):**
- **Circular evaluation.** Synthetic train + synthetic test → "correct diagnosis" = "the disease the patient generator was instructed to simulate." Whether evolved doctors are learning *medicine* or *the generator's quirks* is not separated.
- **No external clinical validation.** No chart review, no human-clinician audit of generated-patient realism, no second medical benchmark (MedMCQA, PubMedQA, MMLU-medical). MedQA is the only outside-world checkpoint.
- **No variance reporting anywhere.** All numbers are single-run.
- **Memory poisoning is visible** (Cardiology 12k → 14k drop, Fig. 9) but no mitigation beyond the post-hoc validation pass.
- **"Hybrid" variant injects MedQA training data** into the case base — directly undermining the "Zero" framing for the headline 92.77%.
- **Mid-experiment model drift.** The OpenAI API update at ~30k patients (Fig. 8) makes the "scaling" curve a moving target.
- **Cost & latency.** 20,000 patients × 21 departments via GPT-4-class APIs is presumably enormous; no cost numbers given.
- **Reproducibility.** Simulator code, patient corpora, prompts, and case/experience bases are not released in the PDF.
- **Mis-citation.** Ref 49 (Kaplan scaling laws) has the wrong title in the bibliography — copy-paste error that weakens the scaling-law analogy.

## Why It Matters for Medical AI

Agent Hospital is one of the cleanest formulations to date of an idea that has been in the air since Smallville: that an LLM can keep getting better at a task without gradient updates if it lives inside a feedback-rich sandbox and remembers what worked. For medical AI specifically, the architectural separation between a **frozen base model** and **two RAG-grown memories** (validated experience rules + case-based exemplars) is portable to other domains where labels are scarce but a generator is cheap. The honest takeaway, though, is that the evidence here mostly tests whether the recipe lets a model fit its own synthetic distribution — which it clearly does — rather than whether evolution in a Chinese-KB-grounded simulacrum produces transferable clinical skill on real English-language USMLE questions. A sub-1pp gain over Medprompt with no error bars is not yet that evidence. The right next step is a held-out clinical benchmark the generator has never seen, run with seeds and CIs, and ideally a chart-level review of generated-patient realism by clinicians.

## References

- Paper (arXiv v3, 17 Jan 2025): [https://arxiv.org/abs/2405.02957](https://arxiv.org/abs/2405.02957)
- Code / simulator: not released as of v3
- Related work:
  - Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (Smallville), UIST 2023
  - Tang et al., *MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning*, 2023
  - Nori et al., *Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine* (Medprompt), 2023
  - Kim et al., *MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making*, NeurIPS 2024
  - Yang et al., *Tuning-free Rule Accumulation* (cited for the reflection/rule mechanism), EMNLP 2023
  - Jin et al., *What Disease Does This Patient Have? A Large-scale Open-domain Question Answering Dataset from Medical Exams* (MedQA), 2021
