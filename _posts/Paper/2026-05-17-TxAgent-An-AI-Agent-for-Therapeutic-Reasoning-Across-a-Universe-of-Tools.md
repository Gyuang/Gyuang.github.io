---
title: "TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools"
excerpt: "Llama-3.1-8B fine-tuned into a ReAct-style agent over a 211-tool biomedical toolbox; hits 92.1% open-ended on DrugPC vs. GPT-4o 66.3% on FDA-2024 drugs."
categories:
  - Paper
  - BioInformatics
  - LLM
  - LLM-Agents
permalink: /paper/txagent/
tags:
  - TxAgent
  - ToolUniverse
  - ToolRAG
  - LLM-Agents
  - Tool-Use
  - ReAct
  - openFDA
  - Open-Targets
  - Precision-Therapy
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR
- TxAgent fine-tunes **Llama-3.1-8B-Instruct** into a ReAct-style agent over **TOOLUNIVERSE (211 tools)** spanning openFDA, Open Targets, and Monarch/HPO, with a **TOOLRAG** retriever that pulls in extra tools on demand so the toolbox can scale past the LLM's context window.
- Training data — **TXAGENT-INSTRUCT, 378K step-wise SFT samples decomposed from 85,340 GPT-4o-generated reasoning traces** (177,626 steps, 281,695 function calls) — is built by a TOOLGEN → QUESTIONGEN → TRACEGEN multi-agent pipeline grounded in FDA labels (pre-2024) + Open Targets + PrimeKG, with explicit filters against hallucinated IDs and unverified-knowledge answers.
- On benchmarks built from **2024-FDA-approved drugs only** (post-base-model cutoff): **DrugPC open-ended 92.1% vs GPT-4o 66.3%**, **TreatmentPC open-ended 75.0% vs DeepSeek-R1-671B 67.5%**, and brand/generic/description variance of **0.00667 vs GPT-4o 9.96**.

## Motivation
Precision therapy needs evidence-grounded recommendations that respect comorbidities, drug-drug interactions, age/pregnancy populations, mechanism-of-action constraints, and an FDA label set that updates continuously. Pure LLMs hallucinate and freeze at training cutoff. Vanilla RAG depends on a static vector index that quickly stales as new drugs are approved. Existing tool-use LLMs (ToolACE-8B, WattTool-8B, Gorilla OpenFunctions) generate single-shot function calls, can only hold a few dozen tool descriptions in context, and break on multi-step retrieval. The authors' thesis is that **goal-directed tool selection plus iterative reasoning over verified APIs** produces an answer whose every fact is traceable to FDA or Open Targets — a structurally different safety/auditability story from RAG-over-PubMed or domain-fine-tuned biomedical LLMs (Med-PaLM, MEDITRON, Llama-3-Med). The cited adversarial data-poisoning result on medical LLMs (Alber et al., *Nat Med* 2025) is the explicit safety motivation for moving information lookup out of model weights.

## Core Innovation
Three pieces, jointly:
1. **TOOLUNIVERSE + TOOLRAG.** A uniform-spec toolbox of 211 verified biomedical APIs over openFDA (Elasticsearch), Open Targets (GraphQL), and Monarch (REST), plus a `gte-Qwen2-1.5B-instruct`-based retriever fine-tuned with multiple-negatives ranking that returns top-k tool descriptions when the agent declares "I need a tool that does X". Retrieval is treated as a first-class action so the agent never needs all 211 descriptions in its prompt.
2. **A multi-agent synthetic-data pipeline (TOOLGEN → QUESTIONGEN → TRACEGEN).** GPT-4o agents respectively (a) draft + test tool specs against real API payloads, (b) build drug-, disease-, and tool-chain-centered questions with a 3-axis (grounding/answerability/reasonableness) GPT-4o evaluator, and (c) generate reasoning traces by a HELPER-conditioned, ground-truth-aware SOLVER, then filter for tool-call correctness, no-hallucinated-IDs, and no repeated thoughts/calls.
3. **An inference architecture with explicit thoughts and a [FinalAnswer] token.** Each step emits a thought $T_i$, a set of parallel JSON function calls $C_i$, and consumes evidence $E_i$; a separate summarization prompt $F_S$ compresses long tool outputs so the context window survives multi-step traces.

## Claims & Evidence Analysis

| # | Claim | Evidence in paper | Strength |
|---|---|---|---|
| C1 | TxAgent beats GPT-4o and Llama-3.1-70B on drug knowledge using only 8B params | DrugPC open-ended 92.1 vs GPT-4o 66.3 vs Llama-70B 52.8 over 3,168 Qs on 2024-only drugs | ⭐⭐⭐ |
| C2 | Beats fine-tuned tool-use LLMs (ToolACE-8B, WattTool-8B) with the same toolbox | Failure-rate analysis: tool-use baselines emit invalid answers 56–63% of the time | ⭐⭐⭐ (more a comment on tool-use-LLM brittleness than a like-for-like win) |
| C3 | Robust to brand/generic/description rewordings | DrugPC/BrandPC/GenericPC 93.6–93.8; **variance 0.00667 vs GPT-4o 9.96** | ⭐⭐⭐ |
| C4 | Outperforms DeepSeek-R1-671B on personalized treatment | TreatmentPC open-ended **75.0 vs 67.5**; MCQ 86.8 vs 76.5 | ⭐⭐ (TxAgent has structural priors R1 lacks at inference; benchmark built by the authors) |
| C5 | Real tools beat LLM-as-tool | Figure 3c: DrugPC 93.8 → 72.7 with GPT-4o-as-tool; TreatmentPC 86.8 → 67.1 | ⭐⭐⭐ |
| C6 | Explicit thoughts > function-calls-only | Figure 3e: −22.3 (DrugPC) / −21.5 (TreatmentPC) absolute | ⭐⭐⭐ |
| C7 | Multi-step reasoning is essential at train and test | Step-1 training → −22 pts; step-1 inference → −13 pts (Fig 3f,g) | ⭐⭐⭐ |
| C8 | Continually updated knowledge generalizes to brand-new approvals | 3 qualitative traces (Bizengri, Kisunla, Alyftrek) | ⭐ — no quantified post-2024 vs in-distribution split outside DrugPC itself |
| C9 | Transparent traces are clinically auditable | Figures 1f–i, 4d–g | ⭐ — no clinician scoring, no faithfulness/groundedness metric |

The most load-bearing piece of evidence in the paper is the **LLM-as-tool ablation (C5)**: it shows the grounded API responses do the work, not the prompt format. The brand/generic robustness (C3) is the cleanest delta. The headline DrugPC and TreatmentPC numbers (C1, C4) are large but should be read alongside the benchmark-construction caveats below.

![TxAgent inference loop and TOOLUNIVERSE](/assets/images/paper/txagent/fig_p023_08.png)
*Figure 1: At inference TxAgent emits a reasoning thought, optionally calls TOOLRAG to retrieve more tools, issues parallel JSON function calls into TOOLUNIVERSE (211 tools over openFDA, Open Targets, Monarch), consumes the responses, and loops until it emits `[FinalAnswer]`. The example tool spec shown is `get_associated_targets_by_disease_efoID`.*

![TOOLUNIVERSE category wheel](/assets/images/paper/txagent/page_008.png)
*Figure 2: TOOLUNIVERSE breakdown — 211 tools across ~15 categories, dominated by adverse events / risks / safety and drug-target-disease association lookups.*

## Method & Architecture

### Inference loop
A reasoning step $R_i = (T_i, C_i, E_i)$:

- Thought $T_i = F_{TX}(Q, R_{<i}, P_i)$ — autoregressive, includes the `[FinalAnswer]` token when ready.
- Function calls $C_i = \{C_i^1, \dots, C_i^k\} = F_{TX}(Q, R_{<i}, T_i, P_i)$ — parallel JSON calls, each `{name, args}` matching a TOOLUNIVERSE spec.
- Evidence $E_i$ — the TOOLUNIVERSE backend translates each call into the actual openFDA Elasticsearch / Open Targets GraphQL / Monarch REST request and returns the result; optionally summarized.

### TOOLUNIVERSE and TOOLRAG
- Coverage: adverse events / risks / safety, addiction & abuse, drug usage in specific populations (pregnancy, pediatric, geriatric, nursing), administration & handling, pharmacology (PK / PD / mechanism), ID & labeling, clinical annotations, patient/caregiver info, disease-phenotype-target-drug associations, biological annotation, publications, search, target characterization.
- Uniform spec: `name`, free-text `description`, typed `arguments` (required/optional), and an argument → API-field mapping rule.
- **TOOLRAG** sits on top of `gte-Qwen2-1.5B-instruct` (multiple-negatives ranking fine-tune), encodes the agent's free-text "what tool do I need" string, and returns top-k tool descriptions by cosine similarity. Retrieved tools are appended to $P_i$; the agent can call them on subsequent steps without ever pre-loading all 211 descriptions.

### TOOLGEN — automated tool construction
A 3-agent + human pipeline over API documentation:

1. **SUMMARIZER** (GPT-4o) reads the schema and lists capabilities.
2. **TOOL GENERATOR** (GPT-4o) drafts each spec — name, description, arguments, API mapping.
3. **TOOL CHECKER** (GPT-4o) synthesizes test queries with concrete drug/disease/target IDs and validates that the call returns useful payloads; failing tools are dropped.
4. **Human verification** by domain experts before admission to TOOLUNIVERSE — no inter-rater agreement, reviewer count, or rejection rate reported.

### QUESTIONGEN — therapeutic-question construction
Three families:

- **Drug-centered:** sample a 2024-excluded drug from FDA labels, sample a section (indications, contraindications, PK, ...), generate questions; descriptive paraphrases support DescriptionPC.
- **Disease-centered (specialized treatment):** sample disease from PrimeKG, pull associated phenotypes / targets / candidate drugs; an INFORMATION EXTRACTOR does side-by-side drug comparisons across indications/contraindications/warnings/population restrictions; QUESTION GENERATOR builds MCQs whose distractors are drugs *indicated for the disease but ruled out by a patient-specific factor* — feeds TreatmentPC.
- **Tool-chain-centered:** sample a path in the tool graph (edges inferred by LLM judgment) so the ground-truth solution requires that chain.

A 3-axis GPT-4o evaluator filters on knowledge-grounding, answerability, reasonableness.

### TRACEGEN — reasoning-trace construction

- **HELPER** (GPT-4o, has access to ground truth $G$ + explanation $X$ + prior trace) emits a step hint $H_{i+1}$ that nudges the SOLVER but is forbidden from revealing the answer; on $A \neq G$ it triggers self-reflection and re-reasoning.
- **TOOL PROVIDER** seeds $\hat P_0$ from the question's reference info, then queries the iteratively trained TOOLRAG for new tools when the SOLVER asks.
- **SOLVER** (GPT-4o) runs the ReAct-like loop with real or "virtual" TOOLRAG calls — virtual calls are later rewritten as real ones so training and inference traces look identical.
- **Trace evaluator** — two-axis filter:
  - **Correctness:** MCQ → exact option match; open-ended → GPT-4 as judge. Function calls checked for correct tool name, argument names/types, presence of required arguments.
  - **Behavior:** rejects traces with hallucinated IDs (IDs appearing in calls that never appeared in tool feedback), arbitrary answers (final answer relies on internal LLM knowledge instead of retrieved evidence), or repeated thoughts / identical-arg repeated calls.

The net effect: **TxAgent is distilling a GPT-4o-orchestrated, ground-truth-conditioned ReAct policy into Llama-3.1-8B**. Surviving traces become 378,027 step-wise SFT rows; each trace $R = \{R_1, \dots, R_M\}$ is unrolled into $M$ training examples whose input is $(S, Q, R_{<i}, P_i)$ and output is $(T_i, C_i)$ (or $(T_M, C_M, A)$ for the last step).

### Training recipe & augmentations
- LoRA fine-tune of Llama-3.1-8B-Instruct, FSDP across 4× H100, **9.93 GPU-days** for TxAgent-8B. Standard autoregressive cross-entropy on output tokens only. TRL + Alignment Handbook + Transformers + DeepSpeed + PyTorch.
- **Tool-description rephrasing:** each tool rewritten in **20 variants** (name, function description, arg names, arg descriptions); training samples randomly draw a variant — prevents memorization of literal names.
- **Tool-set extension:** $P_i$ is padded with TOOLRAG-retrieved candidates and random TOOLUNIVERSE tools so the model learns to filter rather than pick blindly.
- **Tool shuffling:** order randomized to defeat positional bias.
- Long tool results replaced by summarization output when the sample exceeds the context.
- **Leakage control:** all drugs FDA-approved after 2023 are excluded from training.

![TXAGENT-INSTRUCT construction pipeline](/assets/images/paper/txagent/fig_p027_01.png)
*Figure 4: TOOLGEN (211 tools) + QUESTIONGEN (85,340 questions) + TRACEGEN (85,340 traces / 177,626 steps / 281,695 function calls) feed the 378K-sample TXAGENT-INSTRUCT corpus. Inset shows the tool-spec / API-mapping schema and an example reasoning trace for a 60-year-old MI patient avoiding Etodolac ER.*

## Experimental Results

All five benchmarks are constructed by QUESTIONGEN over 2024-FDA-approved drugs and human-reviewed (no reported reviewer count, inter-rater agreement, or rejection rate). Open-ended scoring works by free-generating an answer, then re-prompting the same model to pick the matching option from the MCQ key.

### DrugPC — 3,168 MCQs across 11 FDA-label task categories

| Model | MCQ | Open-ended |
|---|---|---|
| Llama-3.1-70B-Instruct | 75.1 | 52.8 |
| GPT-4o | 76.4 | 66.3 |
| ToolACE-8B (full TOOLUNIVERSE access) | ~31 | ~32 |
| WattTool-8B (full TOOLUNIVERSE access) | ~34 | ~37 |
| **TxAgent-8B** | **93.8** | **92.1** |

Per-task breakdown (Figure 2b,c) puts TxAgent first on every single one of the 11 DrugPC tasks in both settings. **Tool-use baselines emit invalid (un-parseable) answers on 58.9% MCQ / 56.6% open-ended for WattTool-8B and 63.1% / 60.7% for ToolACE-8B** — the paper attributes the failures to inability to fit 211 tool descriptions in context, single-shot function-call habit, and no reasoning between calls.

![DrugPC headline results](/assets/images/paper/txagent/fig_p023_09.png)
*Figure 5: DrugPC accuracy bars — TxAgent 92.1% open-ended vs GPT-4o 66.3% vs Llama-3.1-70B-Instruct 52.8% (Figure 1d), and TxAgent vs tool-use LLMs ToolACE/WattTool (Figure 1e), where the baselines collapse below 40%.*

### Brand / Generic robustness — 3,168 questions each, drug names rewritten

| Model | DrugPC | BrandPC | GenericPC | Variance |
|---|---|---|---|---|
| GPT-4o | ~73.4 | ~70 | 77.3 | 9.96 |
| Llama-3.1-70B-Instruct | 75.1 | 73.0 | ~71 | 2.42 |
| ToolACE-8B | low | low | low | 1.05 |
| WattTool-8B | low | 40.2 | 31.5 | 13.07 |
| **TxAgent-8B** | **93.8** | **93.6** | **93.7** | **0.00667** |

### DescriptionPC — 626 questions, drug names replaced by mechanism + indication + contraindication + interactions paragraphs
- **Answer-only:** TxAgent 90.4 / GPT-4o 85.9 / Llama-70B 85.3.
- **Drug identification alone:** TxAgent 60.1 / GPT-4o 55.8 / Llama-70B 23.6.
- **Two-step (identification AND answer):** **TxAgent 56.5 / GPT-4o 48.2 / Llama-70B 20.1.**

The two-step metric is the honest one; once the model has to first identify the drug, Llama-70B loses ~65 absolute points, while TxAgent loses ~34 but stays first.

### TreatmentPC — 456 patient-specific MCQs (pregnancy / age / comorbidity / co-medication constraints)

| Model | MCQ | Open-ended |
|---|---|---|
| Llama-3.1-8B-Instruct (base) | 56.1 | 33.1 |
| Llama-3.1-70B-Instruct | 70.4 | 49.6 |
| GPT-4o | 74.1 | 61.4 |
| ToolACE-8B + TOOLUNIVERSE | ~30 | 13.4 |
| WattTool-8B + TOOLUNIVERSE | 18.2 | low |
| DeepSeek-R1-Llama-8B (distilled) | 50.7 | 40.1 |
| DeepSeek-R1-Llama-70B (distilled) | 67.5 | 60.3 |
| DeepSeek-R1 (full, 671B) | 76.5 | 67.5 |
| **TxAgent-8B** | **86.8** | **75.0** |

Notable: **TxAgent's open-ended 75.0 beats GPT-4o's MCQ 74.1** — i.e., TxAgent without options outperforms GPT-4o with options. It beats DeepSeek-R1-671B by +10.3 (MCQ) and +7.5 (open-ended) at ~84× smaller scale, and beats the distilled R1-Llama-8B sibling by +36.1 / +34.9. Caveat: this measures "tool-augmented small model vs un-tooled big model" on a benchmark whose distractor logic mirrors TxAgent's fine-tuning targets — see *Limitations*.

### Ablations
- **TOOLUNIVERSE vs LLM-as-tool (Fig 3c):** swapping the real backend for GPT-4o-as-tool drops DrugPC 93.8 → 72.7 and TreatmentPC 86.8 → 67.1; Llama-3.1-8B-as-tool gives 68.7 / 74.8. The grounded API responses are doing the work, not the prompt format.
- **Toolbox-size scaling (Fig 3d):** monotone improvement from 10% → 100% of tools — DrugPC 78.4 → 93.8, TreatmentPC 71.7 → 86.8.
- **Thought ablation (Fig 3e, Algorithm 3):** removing $T_i$ collapses DrugPC 93.8 → 71.5 (−22.3) and TreatmentPC 86.4 → 64.9 (−21.5).
- **Training-trace-length effect (Fig 3f):** capping training reasoning steps at 1 drops DrugPC 93.8 → 71.6 and TreatmentPC 86.8 → 66.9.
- **Inference-step limit (Fig 3g):** TreatmentPC accuracy at 1 step = 73.5; plateaus around 85+ at 5 steps.

### Case studies

![Case study — Prozac × Xolremdi CYP2D6 contraindication](/assets/images/paper/txagent/fig_p029_03.png)
*Figure 6 (TreatmentPC 4e): TxAgent identifies the bidirectional CYP2D6 problem (Prozac is both inhibitor and substrate; Xolremdi is contraindicated with CYP2D6-cleared drugs) and recommends against the combination.*

![Case study — pediatric DMD avoiding steroids and exon-skipping](/assets/images/paper/txagent/fig_p029_04.png)
*Figure 7 (4d): Pediatric DMD case. TxAgent retrieves DMD drugs via `get_drug_names_by_indication`, filters to Duvyzat, fails on a first pediatric-use call, re-retrieves a pediatric-use tool, and confirms safety in children ≥6 yo — illustrative of graceful recovery on a no-result tool call.*

![Case study — 70-year-old Cobenfy max dose](/assets/images/paper/txagent/fig_p029_07.png)
*Figure 8 (4f): Geriatric dosing — parallel calls to dosage + geriatric-use tools return 100 mg / 20 mg BID with a urinary-retention rationale.*

![Case study — hypertension with 2nd-degree AV block](/assets/images/paper/txagent/fig_p029_08.png)
*Figure 9 (4g): Enumerates hypertension drugs then filters via `get_drug_name_by_contraindication("AV block")`, returning ACE-inhibitor / CCB / ARB candidates.*

## Limitations

**Authors admit (Discussion):** TOOLUNIVERSE has coverage gaps; questions outside those gaps cannot be answered. There is no uncertainty quantification on internal knowledge or on tool feedback. The system is text-only — no pathology images, EHR time-series, lab values, or genomic variants. Extended memory for patient histories and multi-modal integration are listed as future work.

**Honest read — what the numbers do and do not show:**

- **All five benchmarks are GPT-4o-generated and GPT-4o-graded.** Questions, distractors, traces, and judges all originate from the same model family, with a human-review step whose size, agreement, and rejection rate are not reported. The headline gains are reported on benchmarks whose construction the authors control end-to-end.
- **TreatmentPC distractor logic mirrors TxAgent's fine-tuning targets.** Distractors are "drugs indicated for the disease but ruled out by a patient-specific factor pulled from the same FDA-label fields TxAgent later queries". Because TxAgent is trained on traces from the same generator using the same FDA fields, there is a real shortcut risk: the model can learn to "look up the field QUESTIONGEN used to make the distractor wrong" rather than reason clinically. The paper does not address this evaluation circularity.
- **No post-training tool-call accuracy is reported.** During trace generation the paper checks tool name, argument names/types, and ID grounding — but for the *deployed* model there is no table reporting how often TxAgent picks the right tool, calls it with the right values, or recovers from an empty/wrong response. End-to-end answer accuracy conflates "right tool / right answer", "right tool / wrong answer", and "wrong tool / lucky answer".
- **TOOLRAG retrieval quality is not quantified.** No recall@k, no MRR, no comparison to BM25 / ColBERT / bge-large. The "extend $P$ with random tools" augmentation is itself an implicit admission that retrieval errors propagate.
- **No clinician scoring, no uncertainty quantification, text-only inputs.** "Open-ended" scoring is still MCQ-via-second-pass; no pharmacist or physician has read a sample of TxAgent's open-ended answers for safety or correctness; the system has no formal way to say "I don't know".
- **Partial leakage control.** Drugs post-2023 are excluded from training and 2024-only drugs anchor the benchmarks, but Open Targets disease-target associations, PrimeKG disease descriptions, and HPO phenotype labels are *not* time-sliced — a TreatmentPC question whose answer turns on a pre-2024 contraindication mechanism is fully in-distribution at the disease level.
- **No adversarial / out-of-distribution probing.** No tests on non-FDA regulators (EMA / PMDA), off-label use, recreational/abuse scenarios, dosing in renal/hepatic impairment beyond labeled detail, pediatric dosing where labels say "safety not established", or pregnant patients where the model may inherit label conservatism inappropriately.
- **No latency / cost / token-budget table.** Multi-step reasoning with parallel API calls has real wall-clock and inference-cost implications for clinical deployment; not reported.
- **Base-model coupling.** TxAgent is fine-tuned on Llama-3.1-8B-Instruct (cutoff Dec 2023). The "newly approved drugs" win is partly an artifact of choosing a base whose cutoff is recent enough to leak via base-model knowledge but old enough to miss 2024 approvals — a different base with a 2024 cutoff would show smaller relative gains over baselines.
- **No RAG-over-DailyMed / RAG-over-PubMed baseline.** The "tools vs LLM-as-tool" ablation makes the case against an in-weights baseline; it does not address the simpler "static retrieval over the same corpora" baseline.

## Why It Matters for Medical AI
The strongest structural argument in this paper is **C5**: replacing the real APIs with GPT-4o-as-tool destroys ~20 absolute points on DrugPC. That ablation is the cleanest evidence in the literature so far that for therapeutic-knowledge tasks, the source-of-truth retrieval is doing the heavy lifting — not the prompt scaffold, not the chain of thought, not the model scale. Combined with the brand/generic invariance (variance 0.00667 vs GPT-4o's 9.96), it points to a deployment pattern where small fine-tuned agents over curated, typed, auditable biomedical tools beat un-tooled frontier LLMs on tasks where every fact has to be traceable. That is exactly the regime — drug labels, drug-drug interactions, contraindications — where hallucination is most dangerous.

The bottom line on safety is more conservative. TxAgent is structurally safer than a vanilla LLM for the narrow class of questions answerable purely from FDA labels and Open Targets, because every fact in its answer is traceable to a tool response. But the headline numbers should not be read as a safety guarantee: the benchmarks are co-generated by the same agent system that trains the model, the eval is end-to-end answer-correctness only, and there is no clinical, adversarial, or external-distribution validation. The case studies are convincing existence proofs of the architecture, not coverage proofs of clinical reliability.

## References
- **Paper:** Gao, Zhu, Kong, Noori, Su, Ginder, Tsiligkaridis, Zitnik. *TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools.* arXiv:2503.10970v1, 14 Mar 2025. <https://arxiv.org/abs/2503.10970>
- **Project page:** <https://zitniklab.hms.harvard.edu/TxAgent>
- **Agent code:** <https://github.com/mims-harvard/TxAgent>
- **Toolbox (TOOLUNIVERSE):** <https://github.com/mims-harvard/ToolUniverse>
- **Weights:** <https://huggingface.co/collections/mims-harvard/txagent-67c8e54a9d03a429bb0c622c>
- **Related — data sources:** openFDA <https://open.fda.gov>; Open Targets <https://platform.opentargets.org>; Monarch / HPO <https://monarchinitiative.org>; PrimeKG (Chandak et al., *Sci Data* 2023).
- **Related — tool-use baselines:** ToolACE (Liu et al., 2024); WattTool / Granite-WattTool (IBM, 2024); Gorilla OpenFunctions (Patil et al., 2023).
- **Related — biomedical LLMs:** Med-PaLM (Singhal et al., *Nature* 2023); MEDITRON-70B (Chen et al., 2023); Llama-3-Med.
- **Safety motivation:** Alber et al., *Medical large language models are vulnerable to data-poisoning attacks*, *Nature Medicine* 2025.
