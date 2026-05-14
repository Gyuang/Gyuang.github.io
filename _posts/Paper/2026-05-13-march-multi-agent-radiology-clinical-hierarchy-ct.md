---
title: "MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation"
excerpt: "A Resident -> Fellow -> Attending multi-agent stack lifts CE-F1 from 0.253 (Reg2RG) to 0.399 on RadGenome-ChestCT, but the ablation arithmetic shows retrieval -- not the consensus protocol the paper foregrounds -- is doing most of the work."
categories:
  - Paper
  - CT-Report-Generation
  - LLM
permalink: /paper/march-multi-agent-radiology-clinical-hierarchy-ct/
tags:
  - MARCH
  - Multi-Agent
  - 3D-CT
  - Radiology-Report-Generation
  - Retrieval-Augmented-Generation
  - LLaMA-2
  - GPT-4.1
  - GPT-4o
  - RadGenome-ChestCT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- MARCH replaces a single end-to-end 3D-CT report generator with a three-stage **Resident -> Fellow -> Attending** agent hierarchy: a 3D-ViT + LoRA-LLaMA-2-7B Resident drafts a region-structured report, GPT-4.1 Fellows revise it with retrieval-augmented evidence (image-image, image-text, logits-logits), and a GPT-4o Attending runs stance-based multi-round consensus.
- On RadGenome-ChestCT (1,564-volume test split) MARCH reaches **CE-F1 0.399 vs 0.253 for Reg2RG (+58% relative)**, with BLEU-1 0.482 and ROUGE-L 0.383 -- the headline jump is real and large.
- But the ablation arithmetic tells a different story than the paper's narrative: retrieval (Resident 0.219 -> SR-SA 0.332, **+0.113 CE-F1**) is the dominant lever; the Attending consensus protocol the paper sells as the key contribution adds only **+0.037**, and is never cleanly isolated.

## Motivation

3D chest CT report generation is a clinical bottleneck. Pathological findings are sparse in volumetric data, end-to-end VLMs hallucinate freely, and there is no built-in verification loop. Real radiology read-out sessions (Seah 2021; Waite 2017) are explicitly hierarchical -- a Resident drafts, a Fellow cross-checks against priors and similar cases, an Attending arbitrates -- precisely because cross-checking reduces interpretive errors. Existing automated systems collapse this into a single black-box generator. MARCH's bet is that explicitly modelling the *organisational* hierarchy (not just averaging outputs) is what closes the clinical-fidelity gap, especially for subtle findings (hiatal hernia, pericardial effusion) that single-pass models routinely miss.

## Core Innovation

- **Stage 1 -- Resident.** A SAT (Segment Anything with Text) module partitions each CT into 10 anatomical subregions (abdomen, bone, breast, esophagus, heart, lung, mediastinum, pleura, thyroid, trachea/bronchi); a frozen RadFM-pretrained dual-stream ViT3D extracts per-region + global features; LoRA-tuned LLaMA-2-Chat-7B emits a region-structured draft.
- **Stage 2 -- Retrieval + Fellow.** Three retrieval modes return top-3 neighbours from the training database -- image-to-image (volume similarity), image-to-text (paired reports), and logits-to-logits (an 18-class abnormality head produces a diagnostic profile, neighbours are reports with similar profiles). Retrieved evidence is concatenated and a GPT-4.1 Fellow validates and revises the draft.
- **Stage 3 -- Attending consensus.** N Fellow revisions are aggregated by a GPT-4o Attending into an initial consensus T^(0). Each subsequent round, every Fellow emits a *stance* (agree/disagree + 1-3 confidence + reasoning + cited evidence); the Attending refines T^(t) -> T^(t+1) and decides whether to continue using a four-case rule (all-agree, weak-disagreement, strong-evidence-disagreement, majority-disagreement). Stops at consensus or T_max rounds.
- **Why it is interesting.** Retrieval gives the Fellows a grounded evidence pool that monolithic VLMs simply do not have access to; the stance-based protocol replaces blind voting with structured argumentation. The architecture is the contribution -- ablations across four GPT backbones (Table 3) show all of them beat all baselines.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | MARCH outperforms SOTA on RadGenome-ChestCT across all metrics. | Table 1: leads on all 9 columns (BLEU-1..4, METEOR, ROUGE-L, CE-P/R/F1). | RadGenome-ChestCT only | ⭐⭐ |
| C2 | Each component contributes to overall performance. | Table 2 ablation: monotone CE-F1 0.219 -> 0.332 -> 0.352 -> 0.362 -> 0.399. | RadGenome-ChestCT | ⭐⭐ |
| C3 | "The most notable drop is observed when removing the consensus-driven Finalization." | Table 2 arithmetic actually shows MR-MA -> Full = +0.037, while Resident -> SR-SA = +0.113. | RadGenome-ChestCT | ⭐ -- the text overstates Stage-3; retrieval is the largest jump. |
| C4 | The multi-agent hierarchy reduces clinical hallucinations. | Figure 2 per-abnormality F1 + Appendix F qualitative cases. **No** RadFact / RadGraph-F1 / hallucination-specific metric. | RadGenome-ChestCT | ⭐ -- asserted, not measured. |
| C5 | Robust across LLM backbones. | Table 3: GPT-4.1 0.399 ~ GPT-4o 0.392 ~ GPT-5 0.391 ~ GPT-4.1-mini 0.393 -- all beat all baselines. | RadGenome-ChestCT | ⭐⭐ |
| C6 | Retrieval-augmented revision improves grounding. | Table 2 SR-SA jump (+0.113 CE-F1 over Resident) is the single largest ablation effect. | RadGenome-ChestCT | ⭐⭐⭐ |
| C7 | Consensus-driven discourse resolves diagnostic discrepancies. | Single qualitative case study (Fig. 3 + Appendix C); no quantitative agreement-rate analysis. | 1 case | ⭐ |
| C8 | Generalises the human read-out workflow. | Conceptual mapping in the introduction; no empirical workflow study. | None | ⭐ -- architectural inspiration, not tested. |
| C9 | Better detection of subtle findings (hiatal hernia, pericardial effusion). | Figure 2 per-abnormality bars; recall on common cardiovascular calcifications > 0.8. | RadGenome-ChestCT | ⭐⭐ |

**Honest read.** The headline number (CE-F1 0.253 -> 0.399 against Reg2RG) is real and large -- not a noise-level gain. But the paper's narrative oversells the consensus protocol. The four-row ablation actually credits **retrieval** as the dominant lever, the Attending discourse adds a marginal +0.037, and there is no clean "Attending vs no-Attending" row that isolates the stance-based protocol from everything else changing in the "Full" condition. Add to that single-dataset evaluation, no variance / significance reporting, no hallucination metric despite the framing, and a baseline set that never gets the same retrieval database wired in -- and the experimental case for "modelling the organisational hierarchy is what matters" is much weaker than the abstract suggests.

## Method & Architecture

![MARCH overview: three-stage Resident -> Fellow -> Attending pipeline](/assets/images/paper/march/page_002.png)
*Figure 1: MARCH overview. Stage 1 (Resident): SAT-based 10-region segmentation -> dual-stream ViT3D -> LoRA-LLaMA-2-7B draft. Stage 2 (Retrieval + Fellow): three retrieval modes (image-image, image-text, logits-logits) feed a GPT-4.1 Fellow that revises the draft. Stage 3 (Attending consensus): N Fellow revisions -> GPT-4o Attending stance-based multi-round refinement until convergence or T_max rounds.*

**A. Stage 1 -- Resident agent A_res.**

1. Input: 3D chest CT volume $I$.
2. **Multi-region segmentation** with SAT (Zhao et al. 2025) partitions $I$ into 10 anatomical subregions.
3. A **frozen dual-stream ViT3D** (RadFM, Wu et al. 2025) extracts per-region + global features.
4. **LoRA-tuned LLaMA-2-Chat-7B** generates the region-structured draft $T = A_{\text{res}}(I; \theta_{\text{res}})$. The prompt template explicitly slots in `<image_*>` global tokens followed by per-region tokens (Prompt 1, Appendix A).

![Multi-region anatomical segmentation module](/assets/images/paper/march/fig_p002_22.png)
*Figure 2: Multi-region anatomical segmentation. SAT partitions the chest CT into the 10 subregions used by the dual-stream ViT3D; per-region features are concatenated with the global feature before the LLaMA-2 decoder.*

**B. Stage 2 -- Retrieval agent A_ret + Fellow agent A_fel.**

5. **Image-to-Image:** the 3D vision encoder retrieves visually similar CT volumes (top-3).
6. **Image-to-Text:** retrieves the reports paired to those volumes.
7. **Logits-to-Logits:** a classification head on top of $A_{\text{res}}$ predicts 18 canonical abnormalities (pleural effusion, atelectasis, etc.); reports with similar diagnostic logit profiles are retrieved.
8. Retrieved evidence $R = A_{\text{ret}}(I, D)$ is concatenated and passed to a GPT-4.1 Fellow which validates and revises:
   $T' = A_{\text{fel}}(T, R)$.

**C. Stage 3 -- Attending agent A_att + N Fellows.**

9. **Round 1 (consensus synthesis):** $T^{(0)} = A_{\text{att}}(\{T'_i\}_{i=1..N})$.
10. **Round t+1 (iterative refinement):** each Fellow reviews $T^{(t)}$ and emits a stance $S_i^{(t)} = A_{\text{fel},i}(T'_i, T^{(t)})$ (agree/disagree + 1-3 confidence + reasons + evidence; Prompt 4).
11. $T^{(t+1)} = A_{\text{att}}(T^{(t)}, \{S_i^{(t)}\})$.
12. The Attending (GPT-4o) decides whether to continue using four cases: all-agree -> stop; weak disagreement -> stop; strong evidence-backed disagreement -> continue; majority disagreement -> continue. Terminate at consensus or $T_{\max}$ rounds.

**Training / hyper-parameters.** Resident + Retrieval trained on 1x H100, AdamW, lr 1e-5, batch 1, 10 epochs (~40 h). LLM agents are GPT-4.1 (Fellow) and GPT-4o (Attending), temperature 0. Default $N = 3$ Fellow agents -- chosen for cost (Appendix D), even though Table 5 finds 5-10 to be optimal.

## Experimental Results

**Dataset.** RadGenome-ChestCT (Wu et al. 2025): 25,692 volumes from 21,304 patients, official 24,128 / 1,564 train/test split, 10 region annotations, 18 RadBERT-derived abnormality labels. Strong class imbalance (lung nodule n=11,696 vs interlobular septal thickening n=2,006; breast n=1,138, thyroid n=1,144). De-identified, IRB-approved, public.

![Tables 1-3: SOTA comparison, ablation, LLM sensitivity](/assets/images/paper/march/page_004.png)
*Figure 3: Quantitative results on RadGenome-ChestCT. Table 1 (top): MARCH leads on all nine metrics. Table 2 (middle): ablation across pipeline stages. Table 3 (bottom): LLM-backbone sensitivity -- all four backbones beat all baselines.*

**Main comparison (Table 1, n=1,564).** The MARCH row is bold.

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CE-Precision | CE-Recall | CE-F1 |
|---|---|---|---|---|---|---|---|
| R2GenPT (2023b) | 0.433 | 0.242 | 0.399 | 0.323 | 0.340 | 0.066 | 0.110 |
| MedVInT (2024) | 0.443 | 0.246 | 0.404 | 0.326 | 0.377 | 0.148 | 0.212 |
| CT2Rep (2024) | 0.444 | 0.236 | 0.402 | 0.310 | 0.317 | 0.089 | 0.139 |
| M3D (2024) | 0.436 | 0.245 | 0.400 | 0.326 | 0.407 | 0.090 | 0.148 |
| RadFM (2025) | 0.442 | 0.237 | 0.399 | 0.315 | 0.382 | 0.131 | 0.195 |
| Reg2RG (2025b) | 0.473 | 0.249 | 0.441 | 0.367 | 0.423 | 0.181 | 0.253 |
| **MARCH (Ours)** | **0.482** | **0.257** | **0.456** | **0.383** | **0.495** | **0.335** | **0.399** |

The CE-Recall jump is especially dramatic (0.181 -> 0.335, +85% relative); precision also climbs from 0.423 to 0.495. Linguistic metrics improve too but by far smaller margins (BLEU-1 +0.009, ROUGE-L +0.016) -- the gain is concentrated in *clinical entity coverage*, not surface fluency.

**Ablation (Table 2) and the role-contribution audit.**

| Variant | BLEU-1 | BLEU-4 | METEOR | CE-F1 | Delta CE-F1 |
|---|---|---|---|---|---|
| Resident-only | 0.469 | 0.246 | 0.435 | 0.219 | -- |
| SR-SA (1 round, 1 Fellow, with retrieval) | 0.476 | 0.250 | 0.447 | 0.332 | **+0.113** |
| SR-MA (1 round, multi-Fellow) | 0.475 | 0.251 | 0.454 | 0.352 | +0.020 |
| MR-MA (multi-round, multi-Fellow) | 0.479 | 0.255 | 0.456 | 0.362 | +0.010 |
| **Full MARCH** | **0.482** | **0.257** | **0.456** | **0.399** | +0.037 |

Read this row by row: the **+0.113 CE-F1 jump from adding retrieval + a single Fellow** is by a wide margin the largest effect in the table. Adding more Fellows is +0.020; multi-round is +0.010; whatever changes between MR-MA and "Full" is +0.037. The paper's prose -- "the most notable drop is observed when removing the consensus-driven Finalization" -- is just inconsistent with the table. Stage 2 (retrieval-augmented revision) is doing most of the work; the Attending stance protocol is a useful but secondary refinement, and is never isolated cleanly (the 4-row ablation conflates the Attending agent with everything else that differs in the "Full" condition).

**LLM sensitivity (Table 3).** GPT-4.1 (CE-F1 0.399) > GPT-4o (0.392) ~ GPT-5 (0.391) ~ GPT-4.1-mini (0.393). All four backbones beat all baselines. This is consistent with the architecture-not-the-LLM hypothesis.

**Fellow count (Table 5, n=100 subset).** 1 -> 0.323; 3 -> 0.330; 5 -> 0.335; 10 -> 0.337; 20 -> 0.327. Saturates at 5-10; 20 degrades. The main runs use $N=3$ for budget reasons -- so the headline 0.399 was produced with **fewer agents than the optimum** found on the subset. The reported number is plausibly a lower bound on what MARCH can achieve.

![Case study: Resident draft -> Fellow revision -> Attending consensus](/assets/images/paper/march/page_011.png)
*Figure 4: Case study (Figure 3 in the paper). Resident draft is revised by Fellows using retrieved evidence; the Attending arbitrates stance-based disagreements into the final consensus report. A single illustrative case -- not a quantitative agreement-rate study.*

## Limitations

**Acknowledged by the authors.** Only GPT-series LLMs tested (no open-source / domain-specific medical LLMs). No long-term memory (no longitudinal patient history, no cross-case learning). No human-in-the-loop interface despite the workflow framing.

**Not addressed but worth flagging.**

- **Single dataset.** Every result is on RadGenome-ChestCT. No CT-RATE, Argus, or M3D-Cap external test. Claims about "modelling human-like organisational structures enhances reliability" need at least one out-of-distribution split.
- **No variance / no statistical significance.** All numbers are single-run. Temperature 0 makes the LLM stages deterministic, but seed sensitivity for the trained Resident is not reported.
- **Stage-3 attribution is muddled.** As above: the prose credits the Attending consensus, the table credits retrieval. The paper does not include an "Attending-removed" row.
- **Hallucination is asserted, not measured.** No RadFact, no RadGraph-F1, no hallucination-specific metric. CE-F1 against RadBERT labels is a partial proxy and inherits RadBERT's labelling biases.
- **Configuration mismatch.** Table 5 finds 5-10 Fellows optimal; the headline runs use 3.
- **Baselines never get retrieval.** All compared methods are end-to-end VLMs. A fair comparison would wire the same retrieval database into Reg2RG or RadFM -- which would isolate the multi-agent contribution from the retrieval contribution.
- **Cost is invisible.** No reporting of API tokens / dollars / latency per report. With $N=3$ Fellows and multi-round Attending discourse, inference cost almost certainly dwarfs single-pass baselines, but this trade-off is not quantified.
- **Backbone asymmetry.** The Resident is a frozen ViT3D + LoRA LLaMA-2-7B -- older than all three GPT versions used downstream. How much of the gain comes from GPT-4.1 simply being a strong reviser of any draft?
- **Round count distribution unreported.** How many cases actually need >1 round? Does the Attending stop at round 1 most of the time? This would clarify whether the multi-round mechanism is doing real work or is mostly a fallback.

## Why It Matters for Medical AI

CT report generation is one of the more clinically consequential generation tasks in medical AI -- a missed pulmonary embolism or unmentioned pleural effusion has direct downstream patient impact. MARCH is useful evidence for two specific positions:

1. **Retrieval-augmented revision is a high-leverage intervention** for 3D radiology generation. The +0.113 CE-F1 jump from adding retrieval + one revising LLM to the Resident draft is, in this paper, the single largest gain in the entire pipeline -- and it is achieved without any additional training. Anyone building a CT or MRI report system today should be wiring in image-image / image-text / logits-logits retrieval before reaching for fancier multi-agent protocols.
2. **The "human workflow as architecture" framing is rhetorically powerful but empirically thin.** The paper sells the Resident -> Fellow -> Attending hierarchy as the contribution, but the ablation arithmetic does not support that emphasis. For medical-AI readers, this is a useful cautionary example of how strong narrative framing can drift from what the numbers actually say.

The takeaway is therefore mixed: a real and large CE-F1 improvement on a non-trivial 3D-CT benchmark, but with attribution issues, a missing hallucination metric, and a single-dataset evaluation that leaves the workflow-generalisation claim unsubstantiated.

## References

- **Paper:** *MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation*, Lin, Ding, Wu, Peng. arXiv:2604.16175v1 (cs.AI), 17 Apr 2026.
- **Dataset:** RadGenome-ChestCT (Wu et al., 2025) -- 25,692 chest CT volumes, official 24,128 / 1,564 split.
- **Backbones:** RadFM ViT3D (Wu et al. 2025); SAT segmentation (Zhao et al. 2025); LLaMA-2-Chat-7B (Touvron et al. 2023); GPT-4.1, GPT-4o (OpenAI).
- **Closest baselines:** Reg2RG (2025b), RadFM (2025), CT2Rep (2024), M3D (2024), MedVInT (2024), R2GenPT (2023b).
- **Workflow inspiration:** Seah et al., 2021 (radiology read-out hierarchy); Waite et al., 2017 (perceptual error in radiology).
