---
title: "MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning"
excerpt: "Curriculum multi-agent RL with a per-stage entropy bonus trains the attending physician to break away from wrong specialist consensus — 73.3% avg on 5 medical-VQA benchmarks, +23.6% over SFT on OOD but only ~6 pt over AFlow in-domain."
categories:
  - Paper
  - LLM-Agents
  - LLM
permalink: /paper/mmedagent-rl-multi-agent-collaboration/
tags:
  - MMedAgent-RL
  - Multi-Agent
  - Reinforcement-Learning
  - GRPO
  - Curriculum-Learning
  - Medical-VQA
  - Qwen2.5-VL
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- **Replace the hand-coded GP roles in a medical multi-agent pipeline with RL-trained Qwen2.5-VL-7B policies** so the system *learns* both how to route to specialists and how to reconcile their disagreement — instead of bolting clinical workflow on top of frozen LVLMs (MedAgents, MDAgents, AFlow).
- **Curriculum-based Multi-Agent RL (C-MARL).** Training samples are stratified by specialist accuracy into Easy (consensus correct) / Medium (disagreement) / Hard (consensus *wrong*); a stage-dependent entropy bonus $\gamma_s$ is injected into the GRPO objective so the attending physician *exploits* consensus on easy cases and *explores against* it on hard cases.
- **Headline numbers.** 73.3% avg across 5 medical-VQA benchmarks on a 7B base (76.6% with majority-vote TTS); the abstract's "**+23.6% over SFT**" is measured on the OOD benchmarks only — versus the next-best multi-agent method (AFlow) the ID average gap is closer to **~6 pt**, and on Hard cases the gain is **51 vs 42 for SFT** (the cleanest evidence in the paper).

## Motivation

Single-model Med-LVLMs (LLaVA-Med, HuatuoGPT-Vision, MedVLThinker) cannot cover the breadth of medical sub-specialties from one weight set, and prior multi-agent systems (Agent Hospital, MDAgents, MedAgents) wrap clinical workflow around frozen specialists with hand-coded routing — no agent in the loop is *trained on the collaboration signal*, so they cannot recover when a specialist is wrong. The paper's bet is that the "attending physician" role is where reasoning failure compounds, and that RL with curriculum-shaped entropy is the right knob to keep that agent from blindly echoing flawed expert opinions while still benefiting from correct ones. The medical angle matters because the reliability variance across specialty LVLMs is large and the cost of confidently echoing a wrong expert is high.

## Core Innovation

- **Two RL-trained GP agents (triage + attending) wrapping a frozen specialist panel.** The triage GP picks one of $k = 7$ specialties from $\{$Pathologist, Radiologist, Surgeon, Oncologist, Endocrinologist, Ophthalmologist, Dermatologist$\}$; the attending GP integrates the panel's independent opinions into a final answer. Both are Qwen2.5-VL-7B trained with GRPO.
- **Simple rule reward.** $R = R_{\text{format}} \in \{0, 0.5\} + R_{\text{acc}} \in \{0, 1\}$, with output wrapped in `<think>…</think><answer>…</answer>`. No learned reward model, no preference data.
- **Curriculum stratification by specialist accuracy.** For each sample, compute $s = \mathrm{Acc}(y_d, y^*)$ over the $e = 3$ specialist responses and bin into $D_{\text{easy}}$ ($s = 1$), $D_{\text{medium}}$ ($0 < s < 1$), $D_{\text{hard}}$ ($s = 0$, **adversarial expert consensus**).
- **Per-stage entropy bonus.** $J_{\text{C-MARL}}(\theta) = \mathbb{E}\big[J_{\text{GRPO}}(\theta) + \gamma_s \, H_t(\pi_\theta^{\text{attend}})\big]$ with $\gamma_{\text{easy}} \approx 0$ (exploit consensus), $\gamma_{\text{medium}} > 0$, $\gamma_{\text{hard}} \gg \gamma_{\text{medium}}$ (break away from misleading consensus).
- **Per-stage KL coefficient.** $\beta$ rises with difficulty: 1e-3 (easy), 4e-3 (medium), 1e-2 (hard) — staying anchored to the reference when exploration is highest.

## Claims & Evidence Analysis

| # | Claim (as stated by paper) | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | RL-driven dynamic multi-agent collaboration beats static GP→Specialist→GP pipelines | Table 1: 73.3 ID avg vs MedAgents 65.6 / MDAgents 66.8 / AFlow 67.5; +17-23 pt OOD avg | 5 medical-VQA benchmarks | ⭐⭐⭐ |
| C2 | "Average performance gain of 23.6% over strong baselines" (abstract) | Sec. 5.2 attributes the 23.6% to **SFT specifically on OOD**, not "strong baselines" broadly | OmniMedVQA, MMMU | ⭐⭐ (the abstract overstates the comparator) |
| C3 | C-MARL is the dominant contributor | Ablation Table 2: removing C-MARL drops 4-dataset avg by ~17 pt (MMMU 71.9 → 50.2) | 4 datasets | ⭐⭐⭐ |
| C4 | Curriculum-stage entropy schedule (low→high $\gamma$) is *necessary*, not just curriculum order | Figure 4 + KL sweep show per-stage $\beta$ ordering matches exploit/explore | 4 datasets | ⭐⭐ — **no curriculum-vs-$\gamma$-schedule ablation** is run, so the entropy schedule is supported by proxy, not isolated |
| C5 | Triage RL improves performance | Ablation: +3 pt avg from triage, +12.6 pt on MMMU when added | 4 datasets | ⭐⭐⭐ |
| C6 | Curriculum order has formal convergence advantage over SGD on the same objective | Theorem 4.1 + Appx. B proofs assume local convexity of GRPO loss (Asm. B.4) | Theoretical | ⭐⭐ — not validated against a same-data SGD baseline |
| C7 | Method generalizes OOD | Table 1 OOD avg 72.6 vs base 58.7 | OmniMedVQA, MMMU-Med | ⭐⭐⭐ |
| C8 | The system mitigates "specialist hallucination" on Hard cases | Figure 4 Hard: **51 (ours) vs 42 (SFT) vs ~47 (frozen pipeline)** | 4 datasets aggregated | ⭐⭐ — the "hallucination metric" is **adversarial-expert robustness MC accuracy**, not an independent hallucination measure |
| C9 | "Aha moment" reasoning | A single qualitative case study (Figure 5) | $n = 2$ cases | ⭐ — authors themselves caveat "lacking the 'aha moment' observed in humans" |

**Honest read.** The core engineering claim — that an RL-trained attending physician beats hand-coded multi-agent pipelines — is well-supported (C1, C3, C5, C7). Three caveats matter:

1. **The 23.6% headline is over SFT on OOD only.** The gap over the next-best multi-agent method (AFlow) on the ID benchmarks is **~6 pt average, not 23.6** — closer in spirit to "competitive" than "dominant" in-distribution.
2. **The entropy-schedule claim is supported by proxy.** The paper never runs "curriculum order without $\gamma$-schedule" vs "curriculum + $\gamma$-schedule" head-to-head, so we cannot disentangle "curriculum helps" from "entropy bonus helps". The KL coefficient sweep is suggestive but not the controlled ablation.
3. **The "hallucination" framing is loose.** Hard cases are operationalized as "all specialists wrong" — that is **adversarial robustness to bad expert input**, not hallucination in the LVLM ungrounded-generation sense. Figure 4 measures multiple-choice accuracy on a difficulty-binned subset, not factuality or grounding.

Also missing across the board: **no variance / seed reporting**, no statistical significance test, no external clinical benchmark beyond OmniMedVQA / MMMU, and the upper bound is gated by specialist quality — Figure 3 makes clear that the RL delta shrinks once you can already afford 3×o3 as your specialist pool, a cost most labs cannot.

## Method & Architecture

![MMedAgent-RL framework: triage GP, frozen specialist panel, attending GP with C-MARL curriculum](/assets/images/paper/mmedagent-rl/page_003.png)
*Figure 1 (paper Fig. 2, page 3): MMedAgent-RL framework. **Stage 1 — Triage GP (RL-trained):** given $(x_v, x_t)$, choose one of $k = 7$ specialties; trained from Qwen2.5-VL-7B with GRPO using the dataset modality label as ground truth (PathVQA → Pathologist, etc.). **Stage 2 — Specialist panel (frozen, role-played):** $e = 3$ proprietary LVLMs (best config: 3×o3) produce **independent** opinions $y_d$ — no inter-specialist debate, no majority voting, so minority opinions survive into aggregation. **Stage 3 — Attending GP (RL-trained with C-MARL):** sees $(x_v, x_t, y_d)$ and emits final reasoning + answer; this is the paper's main contribution.*

### Simple rule reward

$$R = R_{\text{format} } + R_{\text{acc} }, \quad R_{\text{format} } \in \{0, 0.5\}, \quad R_{\text{acc} } \in \{0, 1\}.$$

Format reward fires when the output is wrapped in `<think>…</think><answer>…</answer>`; accuracy reward fires when the answer string matches the multiple-choice ground truth. No learned reward model.

### Difficulty stratification (Algorithm 1)

For every training sample compute $s = \mathrm{Acc}(y_d, y^*)$ over the $e$ specialist responses and bin:

- $s = 1 \to D_{\text{easy}}$ (all specialists correct)
- $0 < s < 1 \to D_{\text{medium}}$ (specialists disagree)
- $s = 0 \to D_{\text{hard}}$ (**specialist consensus is wrong** — adversarial)

### Three-stage curriculum with per-stage entropy and KL

Train sequentially $D_{\text{easy}} \to D_{\text{medium}} \to D_{\text{hard}}$ with GRPO, augmenting the objective with a token-level entropy bonus:

$$J_{\text{C-MARL} }(\theta) = \mathbb{E}\Big[\, J_{\text{GRPO} }(\theta) + \gamma_s \cdot H_t(\pi_\theta^{\text{attend} }) \,\Big], \qquad H_t = -\sum_{j} p_{t,j} \log p_{t,j},$$

with $p_t = \mathrm{softmax}(z_t/\tau)$ over the vocabulary $V$. The coefficient schedule encodes the curriculum:

| Stage | $\gamma_s$ | KL $\beta$ | Intuition |
|---|---|---|---|
| Easy | $\approx 0$ | 1e-3 | Exploit specialist consensus, no exploration reward |
| Medium | $> 0$ | 4e-3 | Moderate exploration under conflict |
| Hard | $\gg \gamma_{\text{medium}}$ | 1e-2 | Strong exploration bonus to *break away* from misleading consensus |

A theorem (Sec. 4 / Appx. B) argues curriculum order yields better convergence than vanilla SGD on the same objective because per-stage optimal-policy distances $\|\theta^*_j - \theta^*_{j+1}\|$ are small, giving warm starts — but the assumption of local convexity of the GRPO loss (Asm. B.4) is not validated empirically. **Crucially, no ablation isolates curriculum-vs-$\gamma$-schedule.**

### Optimization details

GRPO with $G = 8$ rollouts/sample, rollout & train batch size 128, lr 1e-6, sampling temperature $\tau = 1.0$, base Qwen2.5-VL-7B. All evaluation is multiple-choice accuracy, so the accuracy reward is well-defined and free-form generation quality is **not measured**.

## Experimental Results

Training and in-domain test on VQA-RAD, SLAKE, PathVQA (official splits); OOD on OmniMedVQA and MMMU (Health & Medicine subset). Difficulty labels are training-time-only; the paper applies the test-split stratification post-hoc to avoid leakage. **All numbers are single-run accuracy %; no seeds, no confidence intervals.**

### Table 1 — Main medical-VQA results (Table 1, page 7)

| Model | VQA-RAD | SLAKE | PathVQA | ID Avg | OmniMedVQA | MMMU-Med | OOD Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o | 61.0 | 75.5 | 69.4 | 68.6 | 68.5 | 69.7 | 69.1 |
| LLaVA-Med-7B | 51.4 | 48.6 | 56.8 | 52.3 | 44.1 | 36.9 | 40.5 |
| HuatuoGPT-Vision-7B | 63.0 | 77.2 | 58.7 | 66.3 | 74.6 | 51.0 | 62.8 |
| Qwen2.5-VL-7B (base) | 61.8 | 64.7 | 60.5 | 62.3 | 60.8 | 56.6 | 58.7 |
| MedVLThinker-7B | 63.7 | 67.8 | 65.2 | 65.6 | 62.4 | 57.0 | 59.7 |
| MedAgents | 65.6 | 67.9 | 63.2 | 65.6 | 55.8 | 49.7 | 52.6 |
| MDAgents | 66.8 | 68.2 | 65.4 | 66.8 | 58.2 | 52.3 | 55.1 |
| AFlow | 67.3 | 68.9 | 66.4 | 67.5 | 59.6 | 53.6 | 56.6 |
| **MMedAgent-RL (7B)** | **71.5** | **76.2** | **72.3** | **73.3** | **73.3** | **71.9** | **72.6** |
| **+ TTS (majority vote)** | **73.9** | **80.1** | **74.3** | **76.1** | **79.6** | **73.5** | **76.6** |

The headline 73.3 ID avg is **~6 pt over AFlow (67.5)** — meaningful but well short of "+23.6%". The OOD numbers are where the gap explodes: AFlow drops to 56.6 OOD avg while MMedAgent-RL holds 72.6, a clean +16 pt that survives without TTS.

### Hard-case robustness — the cleanest evidence

![Per-difficulty bars: SFT degrades on Hard, C-MARL recovers](/assets/images/paper/mmedagent-rl/fig_p008_02.png)
*Figure 2 (paper Fig. 4, page 8): Per-difficulty average accuracy across the four medical-VQA benchmarks. On **Hard cases** (specialist consensus wrong), the frozen GPT-4o → Qwen2.5-VL pipeline scores ~47, **SFT actually degrades to ~42** (it overfits to bad expert advice), and **C-MARL (Hard) recovers to ~51**. On Easy cases, C-MARL climbs to ~73 vs ~60 baseline. The gap is largest exactly where prior methods break — although note that 51 / 100 on a 4-way multiple-choice subset is **above random but not dramatic**, and this is the paper's own "adversarial-expert robustness MC accuracy" — not an independent hallucination measure.*

### Ablation (Table 2, page 8) — what each component buys

Removing components from MMedAgent-RL on the four-dataset average:

| Variant | VQA-RAD | SLAKE | PathVQA | MMMU |
|---|---:|---:|---:|---:|
| **Full MMedAgent-RL** | **71.5** | **76.2** | **73.3** | **71.9** |
| w/o Triage GP | 66.3 | 69.9 | 66.2 | 59.3 |
| w/o Specialists (skip panel) | 65.8 | 67.8 | 64.4 | 54.2 |
| w/o C-MARL (RL attending w/o curriculum + entropy) | 63.5 | 65.5 | 57.9 | 50.2 |

Removing C-MARL drops the MMMU score by ~22 pt — the headline ablation that justifies the curriculum + entropy machinery as a single bundle. **However, no row isolates "curriculum order alone" from "curriculum + $\gamma$-schedule"**, so the marginal contribution of the entropy schedule (as opposed to the data-ordering benefit) remains unmeasured.

### Specialist composition (Figure 3, page 8)

3×o3 (73.0) > mixed > single specialist; a single Qwen2.5-VL specialist yields only 60.7 avg. Performance is monotone in specialist quality — which is the expected pattern but also the main reason to read the +23.6% headline with caution: a non-trivial fraction of the gain comes from access to 3×o3 inference, which most labs cannot afford to call three times per query.

### KL coefficient sweep (Appx. G.3.2, page 27)

![Per-stage KL sweep: easy prefers low beta, hard prefers high beta](/assets/images/paper/mmedagent-rl/fig_p027_01.png)
*Figure 3 (paper appendix): Per-stage KL coefficient sweep. **Easy** stage prefers low $\beta \approx 0.001$, **Hard** stage prefers high $\beta \approx 0.1$ — consistent with the exploit-then-explore curriculum story (low-KL/low-entropy when consensus is trustworthy, higher-KL/higher-entropy when it is adversarial). **Medium** is essentially flat around $\beta = 0.005$. This is **indirect** evidence for the per-stage $\gamma_s$ schedule, not a direct ablation against fixing $\gamma$.*

## Limitations

**Authors acknowledge:**

- Final performance depends on specialist quality (Figure 3) — the RL delta shrinks as you can already afford strong specialists.
- The model does not exhibit a true "aha moment" (Sec. 5.3 case study).
- LLM (Gemini 2.5 Pro) used for language polishing only (Appx. A).

**Not addressed (my read).**

- **No seed variance / confidence intervals on any number** — 5-benchmark averages with single-run accuracies are fragile, and the +6 pt ID-avg gap over AFlow is exactly the regime where seed noise matters.
- **No curriculum-vs-$\gamma$-schedule ablation** — entropy regulation's marginal contribution over plain curriculum-ordered GRPO is uncertain.
- **The "hallucination metric" is adversarial-expert robustness MC accuracy**, not an independent hallucination / factuality / grounding score.
- **No evaluation on free-text medical QA or report generation** — multiple-choice accuracy does not probe whether the `<think>` traces are clinically faithful.
- **Cost analysis missing.** Each inference requires $e = 3$ specialist calls and the best config uses 3×o3; comparisons to single-model baselines do not match inference-token budget.
- **No safety / red-teaming on adversarial medical prompts** — the "Hard" stratification is naturally occurring, not synthesized.
- **Triage label is derived from dataset modality** — a proxy that conflates "image type" with "appropriate specialist" (a pathology slide could go to an oncologist, for example).
- **Theorem assumes local convexity of the GRPO loss** (Asm. B.4); not verified empirically.

## Why It Matters for Medical AI

Medical multi-agent pipelines (Agent Hospital, MedAgents, MDAgents, AFlow) have so far been *workflow engineering on top of frozen LVLMs*: routing, voting, and aggregation are all hand-coded, no part of the pipeline is trained on the collaboration signal, and as soon as a specialist confidently echoes a wrong opinion the whole tower collapses. MMedAgent-RL is the first credible attempt to make the attending-physician role a *learned* policy with a curriculum that explicitly targets the failure mode — adversarial expert consensus — that matters most clinically. The simple rule reward ($R_{\text{format}} + R_{\text{acc}}$) and the entropy-schedule recipe are easy to drop into any GRPO-based agent stack, and the Hard-case result (51 vs 42 for SFT) is the kind of robustness gain that translates directly to clinical settings where specialist LVLMs disagree. That said, the practical lift over the best static multi-agent baseline is ~6 pt in-domain, not 23.6%; the cost of three o3 specialist calls per query is non-trivial; and the paper's "hallucination" framing should be read as adversarial-expert MC robustness, not factuality. The right way to use this work in a medical-AI stack is as **evidence that the attending role is the lever** — and as a recipe for training that role — rather than as a turnkey replacement for the underlying specialist models.

## References

- Paper (arXiv 2506.00555v3, 26 Jan 2026): https://arxiv.org/abs/2506.00555
- Venue: ICLR 2026
- Authors: Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Zihan Dong, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Linjun Zhang, Shujie Liu, Yan Lu, Huaxiu Yao (UNC-Chapel Hill, Microsoft Research, CMU, Rutgers, Yale)
- Related work: Tang et al. 2023 (MedAgents); Kim et al. 2024 (MDAgents); Zhang et al. 2024 (AFlow); Li et al. 2024 (Agent Hospital); Shao et al. 2024 (GRPO / DeepSeekMath); Bai et al. 2025 (Qwen2.5-VL); Li et al. 2024 (LLaVA-Med); Chen et al. 2024 (HuatuoGPT-Vision); Pan et al. 2025 (MedVLThinker).
