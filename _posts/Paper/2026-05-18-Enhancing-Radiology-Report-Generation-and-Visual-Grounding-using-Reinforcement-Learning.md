---
title: "Enhancing Radiology Report Generation and Visual Grounding using Reinforcement Learning"
excerpt: "GRPO with a clinical -RadCliQ composite reward and a Hungarian soft-F1 grounding reward pushes RadVLM from RadCliQ 1.14 to 0.86 on MIMIC-CXR and lifts grounding mAP@0.5 by 2-7 points — all on chest X-ray, not CT."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/ct-rl-grounding/
tags:
  - RadVLM
  - GRPO
  - Reinforcement-Learning
  - RadCliQ
  - Hungarian-Soft-F1
  - Visual-Grounding
  - Qwen3-VL
  - Chest-X-Ray
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- The authors post-train **RadVLM (Qwen3-VL-8B-Instruct backbone)** with **GRPO** using two task-specific continuous rewards: `-RadCliQ` (BERTScore + CheXbert similarity + RadGraph-F1 composite) for free-text findings, and a **DETR-style Hungarian-matched soft-F1** for bounding-box grounding.
- Headline on MIMIC-CXR / RadVLM splits: **RadCliQ 1.14 -> 0.86**, GREEN 29.1 -> 32.7, CheXbert-micro 49.0 -> 57.0; grounding **mAP@0.5 improves by +2-7 points** across Anatomy / Abnormality / PhraseMS / PhrasePad (best: 84.5 / 45.9 / 87.9 / 63.0).
- Explicit chain-of-thought ("Thinking") **consistently fails to help** under matched GRPO and often slightly hurts grounding; the held-out GREEN judge also moves up, although the authors themselves note GREEN is length-confounded. **Despite the `ct-report-generation/` filing, this paper does not run a single CT experiment** — generalization to CT is speculative.

## Motivation

SFT trains medical VLMs to imitate next tokens, but radiology cares about two sequence-level objectives that next-token loss does not see: (i) the *clinical correctness* of a free-text findings paragraph (entity overlap, factuality, conciseness) and (ii) the *geometric overlap* of bounding boxes with reference lesions. Prior medical-RL work either tackled close-ended VQA/classification (Med-R1, MedVLM-R1, ChestX-Reasoner) or used **offline DPO** for reports (CheXalign, CheXPO). The closest online-GRPO attempt for open-ended CXR reporting (DeepMedix-R1) used lexical-only rewards and bundled chain-of-thought with RL without disentangling them. This paper isolates three questions left open by that literature: how to *design* rewards that mix lexical and clinical signal; whether thinking *adds* anything once GRPO is in place; and whether GRPO can substitute for in-domain SFT when starting from a general-domain VLM.

## Core Innovation

- **Clinically-grounded report reward.** `-RadCliQ` (Yu et al. 2023) is a calibrated composite of BERTScore + CheXbert vector similarity + RadGraph-F1 that aligns with radiologist judgements better than any single component. The authors negate it so GRPO maximizes a normally-negative scalar (floor -3 on a missing answer) and run 32 RadCliQ workers in parallel to hide the RadGraph-XL latency.
- **Continuous grounding reward via Hungarian matching.** For each image, compute pairwise IoU between predicted and reference boxes, solve a one-to-one Hungarian assignment (as in DETR; Carion 2020), credit matched pairs by their IoU as "soft true positives", and compute a soft-F1. This is **deliberately decoupled from the mAP@0.5 evaluation metric**, so reported gains are not tautological.
- **Matched ablations across four model variants.** Qwen3-VL +/- Thinking and RadVLM +/- Thinking are each GRPO-trained independently per task, letting the paper read off the effects of in-domain SFT vs. chain-of-thought vs. RL on a single grid.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | GRPO with task-aligned rewards consistently improves both report generation and grounding over SFT-only RadVLM. | Table 1 (all 7 report metrics improve, RadCliQ 1.14 -> 0.86); Table 2 (all four grounding splits improve, e.g. Anatomy 82.1 -> 84.5, PhrasePad 59.0 -> 63.0). | MIMIC-CXR (reports); Chest-Imagenome, VinDr-CXR, MS-CXR, PadChest-GR (grounding). | ⭐⭐⭐ |
| C2 | The held-out GREEN judge (not used as a reward) also improves, suggesting robustness vs reward hacking. | Table 1 GREEN 29.1 -> 32.7 for RadVLM -> RadVLM+RL; Qwen3-VL also moves 21.9 -> 25.8. | MIMIC-CXR-derived test. | ⭐⭐ (single dataset, no variance reported; authors themselves note GREEN is length-sensitive — partially confounded). |
| C3 | RadVLM+RL is SOTA among reported baselines on most metrics. | Bold cells in Tables 1 and 2; beats DeepMedix-R1, MAIRA-2, CheXagent-2, LLaVA-Rad on RadCliQ, ROUGE-L, BERTScore, RadGraph-F1, GREEN, and most grounding splits. | Same as C1. | ⭐⭐ (no statistical tests, no CIs, single seed implied; RadVLM+RL still trails MAIRA-2 / LLaVA-Rad on **CheXbert-macro** 33.4 vs 35.8 / 39.2 — not flagged in the abstract). |
| C4 | Explicit chain-of-thought ("Thinking") gives at best marginal gains and often slightly underperforms direct-answer variants under matched RL. | Table 1: RadVLM+RL vs RadVLM-Thinking+RL essentially tied (RCQ 0.86 vs 0.87). Table 2: Thinking *loses* on Abnormality (45.9 -> 43.9), PhraseMS (87.9 -> 86.4), PhrasePad (63.0 -> 60.5); wins only Anatomy by 0.4. Qwen3-VL-Thinking+RL clearly worse than Qwen3-VL+RL on grounding (e.g. Anatomy 79.0 -> 68.8). | Same. | ⭐⭐⭐ (consistent direction across 5+ comparisons; aligns with Li 2025a, Lai 2025, and Qwen3-VL's own findings). |
| C5 | GRPO alone (no in-domain SFT) lets a general-domain VLM approach in-domain SFT-only models. | Qwen3-VL+RL beats RadVLM-SFT on BERTScore (55.8 > 53.3), RadGraph-F1 (22.5 > 19.0), RadCliQ (1.05 < 1.14); nearly matches on grounding. | Same. | ⭐⭐ (mixed: clearly true on report metrics but Qwen3-VL+RL still loses on CXb-macro 22.3 vs 30.5 and on all grounding splits — "approach" is fair, "substitute" is not). |
| C6 | Reward design is critical; RadGraph-F1 alone causes reward hacking via length collapse, while RadCliQ / BERTScore / GLEU broadly improve metrics. | Table 3: ANC collapses 207 -> **93 chars** with RGF1-C; only RGF1 improves while CXb-micro falls 49.0 -> 34.1 and GREEN 29.1 -> 23.5. | MIMIC-CXR test split. | ⭐⭐⭐ (vivid, reproducible-looking demonstration of the failure mode predicted by prior DPO/RL CXR work). |
| C7 | Hungarian soft-F1 is a better RL signal than threshold-gated mAP because it gives continuous credit. | Asserted by analogy to DETR / Panoptic Quality. **No ablation against a hard-mAP reward is reported in the body.** | — | ⭐ (architectural argument, not empirically isolated). |
| C8 | Findings generalize across "domain-specific" and "general-domain" VLMs. | Matched experiments on Qwen3-VL and RadVLM, each +/- Thinking, all GRPO-trained. | Same eval splits. | ⭐⭐ (one *backbone family*, one modality — CXR; not tested on a non-Qwen backbone, not on CT, not on a non-MIMIC report test set). |

**Honest read.** The strongest claims (C1, C4, C6) are the paper's three durable contributions. The weaker claims are around generality: there is **no external/out-of-distribution test set for reports** (everything funnels through the MIMIC-CXR-derived RadVLM split, on which RadVLM was *trained*), **no variance reporting** (single runs, no seeds, no CIs, no significance tests), **no PPO or online-DPO baseline** to isolate "GRPO specifically" from "any policy-gradient post-training", and **no ablation on soft-F1 vs hard-mAP reward** (C7). The CheXbert-macro gap vs MAIRA-2 / LLaVA-Rad is candidly disclosed in the tables but never explained — it likely reflects label-balance issues the chosen reward does not address. The abstract's "state-of-the-art" framing should be read as *SOTA among the eight baselines in the authors' evaluation pipeline*, not a public-leaderboard claim.

## Method & Architecture

![RadVLM-GRPO worked example: cardiomegaly + pacemaker CXR used to illustrate both -RadCliQ and Hungarian soft-F1](/assets/images/paper/ct-rl-grounding/fig_p002_01.png)
*Figure 1: Case CXR (cardiomegaly with a left-sided pacemaker) that the paper's Figure 1 uses to illustrate both reward channels — the same image drives the `-RadCliQ` reward over the generated findings paragraph and the Hungarian-matched soft-F1 reward over predicted boxes.*

The pipeline has three stages, all sharing the Qwen3-VL-8B-Instruct backbone:

1. **SFT on the RadVLM instruction corpus** — 2 epochs, lr 1e-5, effective batch 8 x 64 = 512 on 64 GPUs via LLaMA-Factory. Covers free-text findings, abnormality classification, anatomical / abnormality / phrase grounding, and GPT-4o multi-turn dialogues (~1 M pairs).
2. **Optional cold-start "Thinking" SFT** — ~28 k chain-of-thought trajectories *retro-rationalized* by Qwen3-VL-235B-Instruct conditioned on image + ground-truth answer; second SFT stage on a mix of these CoTs and direct-QA data to avoid catastrophic forgetting.
3. **GRPO post-training** (Shao et al. 2024) via the `verl` library: 300 steps, batch = 512 prompts, **8 rollouts per prompt** (154 k prompt-image pairs total), KL = 0.01, **asymmetric clipping** (Yu et al. 2025). Reward is computed on the *final answer only* — for Thinking models, the answer is text after `</think>`; missing answer yields 0 (or -3 for `-RadCliQ`). Four variants are GRPO-trained independently per task: Qwen3-VL+RL, Qwen3-VL-Thinking+RL, RadVLM+RL, RadVLM-Thinking+RL.

### Report reward — `-RadCliQ`

RadCliQ is a composite of BERTScore + CheXbert vector similarity + RadGraph-F1, calibrated to align with radiologist judgements better than any single component. The authors use the RadEval implementation with RadGraph-XL. Since lower RadCliQ is better, they reward `-RadCliQ`. To prevent RadGraph (no batch inference) from becoming a bottleneck, 32 RadCliQ workers run in parallel.

### Grounding reward — Hungarian soft-F1

For each image, compute pairwise IoU between predicted and reference boxes, solve a one-to-one Hungarian assignment, count matched pairs as "soft true positives" weighted by IoU, with unmatched predictions as FP and unmatched references as FN. Final reward is a soft-F1 over those counts. Unlike threshold-gated mAP@0.5, this is *continuous* — gradients persist near the threshold, giving GRPO a smoother signal. Inspired by Panoptic Quality (Kirillov 2019).

## Experimental Results

### Report generation (Table 1, MIMIC-CXR derived test set)

| Model | Size (B) | R-L | B-S | CXb-micro | CXb-macro | RGF1 | GRN | RCQ ↓ |
|---|---|---|---|---|---|---|---|---|
| MedGemma-pt | 4 | 20.7 | 47.7 | 49.8 | 32.5 | 15.5 | 21.9 | 1.37 |
| MedGemma-it | 27 | 15.9 | 31.3 | 47.0 | 31.5 | 12.0 | 23.3 | 1.79 |
| MAIRA-2 | 7 | 17.7 | 46.6 | 52.1 | 35.8 | 12.9 | 21.3 | 1.42 |
| CheXagent-2 | 3 | 22.5 | 37.4 | 54.5 | 38.7 | 20.1 | 29.9 | 1.45 |
| DeepMedix-R1 | 7 | 22.3 | 52.8 | 48.2 | 28.3 | 18.6 | 30.0 | 1.23 |
| LLaVA-Rad | 7 | 22.2 | 48.9 | 53.3 | **39.2** | 16.8 | 28.6 | 1.34 |
| Qwen3-VL | 8 | 14.0 | 42.0 | 35.3 | 20.4 | 10.8 | 21.9 | 1.67 |
| Qwen3-VL-Thinking | 8 | 15.0 | 43.1 | 34.6 | 18.5 | 10.9 | 18.5 | 1.63 |
| Qwen3-VL+RL | 8 | 24.7 | 55.8 | 47.4 | 22.3 | 22.5 | 25.8 | 1.05 |
| Qwen3-VL-Thinking+RL | 8 | 23.3 | 55.0 | 44.4 | 22.4 | 21.0 | 24.7 | 1.12 |
| RadVLM (SFT) | 8 | 26.0 | 53.3 | 49.0 | 30.5 | 19.0 | 29.1 | 1.14 |
| **RadVLM+RL** | 8 | **29.9** | **59.2** | **57.0** | 33.4 | **25.8** | 32.7 | **0.86** |
| **RadVLM-Thinking+RL** | 8 | 30.0 | 59.0 | 56.3 | 33.6 | 25.7 | **32.9** | 0.87 |

### Visual grounding (Table 2, mAP@0.5 in %)

| Model | Anatomy | Abnorm. | PhraseMS | PhrasePad |
|---|---|---|---|---|
| MAIRA-2 | 19.8 | 11.3 | 80.1 | 38.8 |
| Qwen3-VL | 11.1 | 5.1 | 19.0 | 10.7 |
| Qwen3-VL-Thinking | 8.7 | 1.2 | 14.6 | 10.7 |
| Qwen3-VL+RL | 79.0 | 36.0 | 79.4 | 55.3 |
| Qwen3-VL-Thinking+RL | 68.8 | 27.6 | 76.1 | 42.0 |
| RadVLM (SFT) | 82.1 | 44.2 | 84.6 | 59.0 |
| **RadVLM+RL** | 84.5 | **45.9** | **87.9** | **63.0** |
| RadVLM-Thinking+RL | **84.9** | 43.9 | 86.4 | 60.5 |

### Reward-choice ablation (Table 3) and dynamics

- **RadGraph-F1-only training is empirically reward-hacked.** Training RadVLM with `+RGF1-C` collapses report length from ANC = 207 chars to **93 chars** and *only* improves RadGraph-F1 itself (20.1) while dropping every other metric (CXb-micro 49.0 -> 34.1, GREEN 29.1 -> 23.5). RadCliQ, BERTScore and GLEU all improve metrics broadly; RadCliQ wins **4 of 7** evaluation metrics.
- **Training dynamics.** Qwen3-VL+RL on report generation collapses response length within the first few steps — direct evidence that the RadCliQ signal suppresses excess verbosity. Thinking variants are notably more stable in length but slower to converge in reward.

![GRPO critic-score curves: RadVLM converges fastest, RadVLM-Thinking variants lag behind](/assets/images/paper/ct-rl-grounding/fig_p029_01.png)
*Figure 2: Reward trajectories during GRPO. RadVLM (in-domain SFT start) leads throughout, while RadVLM-Thinking variants converge more slowly — consistent with the paper's claim that explicit thinking does not add value once GRPO is in place.*

![Mean response length during GRPO: thinking variants stay long, RadVLM stays compact](/assets/images/paper/ct-rl-grounding/fig_p030_01.png)
*Figure 3: Mean response length during GRPO. Thinking variants stay long and stable, while non-thinking RadVLM stays compact — useful context for the RadCliQ-driven length suppression observed on Qwen3-VL+RL.*

- **GREEN improves with RL despite not being the reward** — the authors highlight this as evidence against reward gaming, but caveat that GREEN is itself length-sensitive (Hein et al. 2024), so its gains on BERTScore- and GLEU-trained variants may be partly length-driven.
- **No PPO or online-DPO baseline.** The body cites offline DPO (CheXalign, CheXPO) and online GRPO (DeepMedix-R1) but does not re-train these inside its own evaluation pipeline, so the gap between *GRPO specifically* and *any online RL post-training* is not isolated.

## Limitations

**Authors acknowledge.** Cold-start CoT was generated with ground truth visible, biasing rationales toward the target answer; alternative no-GT / filter-by-correctness strategies (Appendix I) were reportedly worse. Thinking may underperform because of cold-start degradation, uninformative traces, the absence of vision inside the CoT, or Qwen3-VL's pre-training corpus emphasizing math/code over radiology. LLM-as-judge rewards (e.g. GREEN) need explicit reward-hacking defenses and were not used during training.

**Reviewer-noticed gaps.**

- **No CT evaluation.** Despite the parent agent's `ct-report-generation/` filing, this paper is entirely about **chest X-ray** (a 2-D modality). RadCliQ's RadGraph entity space is trained on CXR reports; a CT port would need a CT-trained RadGraph or substitute. Hungarian soft-F1 generalizes to 3-D IoU but bbox supervision in CT is far rarer (most CT datasets use segmentation masks or anatomical labels). Any CT generalization claim from this paper is speculative.
- **Reward-hacking shown for one channel only.** The vivid length-collapse demo for RadGraph-F1-only training (ANC 207 -> 93) is great, but no analogous audit is run on the GREEN signal that the paper highlights as held-out — and GREEN gains are themselves potentially length-mediated.
- **Thinking consistently fails to help.** RadVLM+RL ≈ RadVLM-Thinking+RL on reports (0.86 vs 0.87 RCQ), and Thinking actually *loses* on Abnormality (45.9 -> 43.9), PhraseMS (87.9 -> 86.4) and PhrasePad (63.0 -> 60.5). On Qwen3-VL the gap widens (Anatomy 79.0 -> 68.8). This aligns with parallel literature (Li 2025a, Lai 2025, Qwen3-VL's own ablations) and deserves louder framing than the paper gives it.
- **No PPO baseline.** The "is GRPO actually the right RL choice?" question is left unanswered.
- **No variance, no seeds, no significance tests.** 64 GPUs x 300 RL steps x multiple variants — the budget clearly allowed at least bootstrap CIs over the test set.
- **No soft-F1-vs-hard-mAP ablation** isolating the continuous-reward argument (C7) — it is justified only by analogy to DETR / Panoptic Quality.
- **No external / OOD test for reports.** Everything funnels through the MIMIC-CXR-derived RadVLM split, on which RadVLM was trained.
- **Single frontal view only.** No lateral, no priors ("comparison to prior" sentences are pre-stripped from references), no longitudinal evaluation by construction.
- **CheXbert-macro persistently trails** MAIRA-2 (35.8) and LLaVA-Rad (39.2): RadVLM+RL lands at 33.4 — disclosed in the tables but not flagged in the abstract. RadCliQ's BERTScore + CheXbert-similarity + RadGraph-F1 mix may underweight rare-disease label balance.

## Why It Matters for Medical AI

The durable methodological contribution is **reward design**, not GRPO itself. The paper makes two concrete, transferable claims that practitioners can use today:

1. *Composite clinical rewards beat any single lexical metric in isolation.* The Table 3 length-collapse demonstration for RadGraph-F1-only training is a textbook reward-hacking case study — anyone training a medical VLM with online RL should pre-register a composite reward and an independent length/abstention monitor.
2. *Continuous geometric rewards beat threshold-gated ones for grounding.* The Hungarian soft-F1 recipe (one-to-one matching, IoU as soft credit) ports directly to any task with bounding boxes and a fixed-cost matcher.

The caveat is equally important: the paper's CXR-only evaluation means clinicians should *not* read "SOTA on MIMIC-CXR" as "ready for CT". A CT port requires (i) a CT-trained entity recognizer in place of RadGraph-XL and (ii) box supervision that most CT datasets do not provide. The "no PPO baseline + no variance reporting + length-confounded held-out GREEN" stack also means the comparative *advantage* of GRPO over alternative online-RL methods on CXR is still an open question.

## References

- Paper: [Enhancing Radiology Report Generation and Visual Grounding using Reinforcement Learning](https://arxiv.org/abs/2512.10691) (arXiv 2512.10691v1, Dec 2025)
- Code: [uzh-dqbm-cmi/RadVLM-GRPO](https://github.com/uzh-dqbm-cmi/RadVLM-GRPO)
- Models / data: PhysioNet release planned (credentialed access)
- Related — RadVLM and CXR VLMs: RadVLM (Deperrois 2025a/b), MAIRA-2, CheXagent-2, LLaVA-Rad, MedGemma.
- Related — medical RL: Med-R1, MedVLM-R1, ChestX-Reasoner, DeepMedix-R1 (online GRPO with lexical rewards); CheXalign, CheXPO (offline DPO).
- Related — reward signals: RadCliQ (Yu et al. 2023), RadGraph-XL, CheXbert, GREEN (held-out judge), Panoptic Quality (Kirillov 2019), DETR Hungarian matcher (Carion 2020).
- Related — RL infrastructure: GRPO (Shao et al. 2024), asymmetric clipping (Yu et al. 2025), `verl` library.
