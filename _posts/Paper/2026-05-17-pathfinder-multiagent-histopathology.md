---
title: "PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology"
excerpt: "PathFinder chains four agents (Triage U-Net, T5-conditioned U-Net Navigator, Quilt-LLaVA Describer, GPT-2+LoRA Diagnoser) to iteratively pan-and-zoom through WSIs; 0.74 accuracy on the 35-case M-Path melanoma test set vs 0.66 ABMIL+CONCH/UNI2-h and 0.65 average pathologist."
categories:
  - Paper
  - LLM-Agents
  - Pathology
permalink: /paper/pathfinder-multiagent-histopathology/
tags:
  - PathFinder
  - Multi-Agent
  - Whole-Slide-Image
  - Quilt-LLaVA
  - U-Net-Navigator
  - LoRA
  - Pathology
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- PathFinder decomposes WSI diagnosis into four cooperating agents — **Triage** (QuiltNet + PPEG transformer + multi-scale convs + SE block), **Navigation** (4-layer U-Net conditioned on T5-encoded prior patch descriptions, outputs an importance heatmap for probabilistic ROI sampling), **Description** (Quilt-LLaVA 7B, instruction-tuned on 102K GPT-4-condensed Quilt-1M findings), and **Diagnosis** (GPT-2 + LoRA `r=8` text-only classifier over trajectories).
- On the M-Path 4-class melanoma test set (35 balanced WSIs, mean of 10 runs of 5-trajectory majority vote) PathFinder hits **0.74 accuracy / 0.74 F1**, +8 pts over the strongest baseline (ABMIL with CONCH or UNI2-h at 0.66) and +9 pts over the 0.65 average pathologist accuracy reported in prior work.
- The strongest result is structural: Triage gate and text-conditioned Navigator each add ~10–16 pts in clean ablations (no-Triage 0.58, vision-only nav 0.64, CLIP-text nav 0.62, imitated viewport sampling 0.63). The headline pathologist-beating and GPT-4o-equivalence claims rest on n=35 with no significance test, no external cohort, and a 2-rater × 25-ROI forced-concise description study.

## Motivation

Standard computational pathology shreds a gigapixel WSI into thousands of independent patches and aggregates them with MIL or hierarchical transformers, throwing away the iterative, multi-scale, evidence-accumulating reasoning real pathologists use. On melanoma grading — where the class label hinges on a few diagnostically critical regions — that aggregation ceiling shows up as both lower accuracy and lost interpretability. The medical framing is sharp: average pathologist accuracy on the M-Path 4-class task is only ~65%, so an interpretable system that exposes *which* patches it looked at and *why* matters as much as the raw number.

## Core Innovation

- **A text-conditioned visual Navigator.** A lightweight U-Net takes the masked WSI plus the *averaged T5 embedding of all previously generated patch descriptions* and outputs an importance heatmap. Patches are then probabilistically sampled proportional to heatmap intensity — closing a visual-textual feedback loop instead of asking an LLM to predict grid coordinates directly (an alternative the authors tried first; their LLaVA-style U-Net + LLaMA-7B coordinate regressor collapsed to always-center under their data scale).
- **A Triage gate as a malignancy-bias absorber.** Quilt-LLaVA's instruction data is malignancy-skewed, so without a front-end binary gate the downstream agents over-call disease; the Triage Agent (PPEG transformer + multi-scale 3×3 / 5×5 / 7×7 convs + Squeeze-Excitation + CLS-token transformer) separates Class 1 (mild/moderate dysplastic nevi) from Classes 2–4 before the navigator runs.
- **Trajectory-as-text aggregation.** The Diagnosis Agent is a plain GPT-2 LoRA-tuned on 100k synthetic trajectories (5–10 patches each, descriptions paraphrased with LLaMA-3.1-Instruct), classifying via vocab-logit head on a fixed prompt. Cheap to train and replaceable — the authors flag that swapping in a stronger LLM is a likely free win.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | PathFinder beats SOTA WSI methods by ~8 pts on M-Path melanoma | Table 1: 0.74 vs 0.66 (ABMIL + CONCH / UNI2-h), mean of 10 runs | M-Path 4-class, 35-case balanced test | ⭐⭐ — single dataset, n=35, no significance test, but variance bands in Fig. 3 don't visibly overlap with the 8-pt margin at 5 trajectories |
| C2 | "First AI to surpass average pathologist accuracy" (+9 pts) | 0.74 vs 0.65 human baseline from a separate study | M-Path | ⭐⭐ — true for the cited number, but "first AI" is restricted to this dataset and this 4-class task; humans in [9] worked under different conditions, not head-to-head |
| C3 | Multi-agent iterative pipeline is necessary (not just better patches/backbones) | Vision-only nav 0.64, Imitated viewport sampling 0.63, CLIP-text nav 0.62, ABMIL-top-patch variants ≤0.54, Exhaustive search 0.68 — all below 0.74 | M-Path | ⭐⭐⭐ — best-supported claim; ablations span both the text-feedback axis and the navigator-architecture axis |
| C4 | T5 beats CLIP for navigator conditioning because of token limits | 0.74 (T5) vs 0.62 (CLIP); when descriptions are *low quality* (LLaVA-Med) T5 *underperforms* CLIP (0.56 vs 0.60) | M-Path | ⭐⭐ — interesting asymmetry, plausible explanation, but no direct token-truncation analysis |
| C5 | Triage Agent is essential | Removing Triage drops 0.74 → 0.58; Appendix Table 1 shows Triage > TransMIL/DSMIL/AMIL on Class-1 vs rest | M-Path | ⭐⭐⭐ — large effect, clean isolation, named failure mode (Quilt-LLaVA malignancy bias) |
| C6 | Description Agent is comparable to GPT-4o | Pathologist preference roughly tied between PathFinder-7B and GPT-4o; LLaVA-Med trails both | 25 ROIs × 2 raters, forced ≤20-word prompts | ⭐ — small n, no inter-rater agreement reported, no statistical test, descriptions forced outside both systems' natural verbosity |
| C7 | 5 trajectories × 10 patches is near-optimal | Plateau curves in Fig. 3 | M-Path | ⭐⭐ — solid for the test set; no external validation |
| C8 | Pipeline is "fully explainable" | Qualitative trajectory: heatmap → patch → description → mask → next heatmap (Fig. 2 right) | Qualitative only | ⭐ — structural interpretability is real, but no user study shows pathologists actually verify or override using these outputs |

**Honest read.** The defensible contributions are the *system design* (C3, C5) and the *text-conditioned U-Net Navigator* — both are backed by multiple ablation rows in Table 1 plus a clean failure-mode story (the abandoned LLaVA-coordinates navigator in Appendix §2). The headline numbers (C1, C2) are real *for this benchmark* but rest on 35 balanced test WSIs, one site, no external cohort, no per-class breakdown beyond aggregate accuracy/F1, and no significance test on the 8-pt margin (which corresponds to roughly ±3 cases at n=35). The "comparable to GPT-4o" description claim (C6) is the weakest: 2 raters × 25 cases × forced-concise prompts is more anecdote than evidence. The interpretability claim (C8) is structurally genuine — you do get a trajectory, per-patch text, and a final diagnosis — but is not validated by a human-factors study. There is also a quiet leakage concern: the Navigator's ground-truth heatmaps come from the same M-Path pathologist-viewport cohort whose held-out split the system is then evaluated on, so some of the 0.74 may reflect study-specific viewport priors rather than transferable ROI signal.

## Method & Architecture

![PathFinder pipeline: WSI → Triage gate (Benign / At Risk) → Navigation–Description loop on critical patches → trajectory of patches and descriptions → Diagnosis Agent](/assets/images/paper/pathfinder/fig_p001_02.png)

*Figure 1: PathFinder's four-agent pipeline — Triage gates the WSI into benign-vs-at-risk, Navigation and Description iterate to collect evidence patch by patch, and the Diagnosis Agent integrates the resulting trajectories into a final class label.*

**(1) Triage Agent — binary risk gate.** 512×512 patches at 10× are background-filtered (saturation ≥15; resample to ≥150 patches if short). QuiltNet features `(N, 768)` are linearly projected to `(N, 512)`, reshaped to a square grid (zero-padded by `H²−N`), refined by a transformer block, PPEG positional encoding, a second transformer block, multi-scale 3×3 / 5×5 / 7×7 convs with an SE block, then flattened and classified through a CLS-token transformer + MLP head. Trained with BCE, lr `2e-4`, weight decay `1e-5`, batch 1 / grad accum 32, ≤100 epochs, early-stop after 30 stagnant epochs. Job: split Class 1 (mild/moderate dysplastic nevi) from Classes 2–4. Without this gate downstream agents drift toward malignancy because Quilt-LLaVA's training data is malignancy-skewed; removing it costs 16 pts (0.74 → 0.58).

**(2) Navigation Agent — text-conditioned U-Net.** 4-layer-encoder / 4-layer-decoder U-Net `f_Nav`. At iteration `t` it consumes the masked WSI `I^(t)` (previously sampled patches blanked) and the aggregated text embedding `E^(t-1) = (1/(t−1)) Σ T5_text(D^(k))` of all prior descriptions, and emits an importance heatmap `M^(t) = f_Nav(I^(t), E^(t-1))`. Patches are sampled with `p^(t)_{i,j} ∝ M^(t)_{i,j}` and the high-res crop is handed to the Description Agent. Trained with pixel-wise BCE against pathologist-viewport ground-truth heatmaps from M-Path; auxiliary descriptions are produced by Quilt-LLaVA and paraphrased for augmentation.

![Left: Navigation Agent U-Net takes masked WSI + T5-encoded descriptions and outputs an importance heatmap, which is gridded into 16×16 cells for probabilistic sampling. Right: a three-step trajectory example with patch crops, descriptions, and masking between steps.](/assets/images/paper/pathfinder/fig_p003_01.png)

*Figure 2: (left) the text-conditioned U-Net Navigator turns the masked WSI and accumulated T5-embedded descriptions into a sampling heatmap; (right) one trajectory iteratively samples a patch, describes it with Quilt-LLaVA, and masks it out before the next step.*

**(3) Description Agent — Quilt-LLaVA 7B captioner.** Quilt-LLaVA 7B instruction-tuned on 102K samples for one epoch, where GPT-4 condensed Quilt-1M captions into concise finding lists. Each sampled patch yields a brief natural-language description that feeds back into the Navigator and forward to the Diagnosis Agent.

![Appendix Figure 1: Triage Agent architecture — QuiltNet features, square padding, transformer + PPEG, multi-scale convs + SE block, CLS-token transformer head.](/assets/images/paper/pathfinder/fig_p013_01.png)

*Appendix Figure 1: Triage Agent architecture — QuiltNet features are reshaped into a square grid, refined by transformer + PPEG positional encoding and multi-scale convolutions with an SE block, then classified via a CLS-token transformer head.*

**(4) Diagnosis Agent — GPT-2 + LoRA over trajectories.** GPT-2 with a linear head over vocab logits → 3 classes (Classes 2–4; Class 1 is filtered by Triage). Training trajectories are generated by running the navigator on a sub-sampled 512×512 WSI, gridding to 16×16 cells of 32×32 patches, scoring each cell by mean heatmap intensity, then doing 10 iterations of weighted sampling at 10× with masking. 5 trajectories of 10 patches per training WSI; 20 trajectories per test WSI; LLaMA-3.1-Instruct paraphrases for diversity. Training set resampled to 20k cases / 100k trajectories; trajectory length randomized 5–10 and order shuffled. Prompt: *"The image descriptions below are extracted from different patches from the same whole slide image (WSI); please tell me which class the image belongs to: {descriptions}"*. LoRA `r=8`, `α=8`, dropout 0.1, lr `5e-5`, weight decay `1e-3`, batch 16, cross-entropy.

**Inference.** 5 sampled trajectories per test WSI → 5 Diagnosis-Agent predictions → majority vote. The whole evaluation is repeated 10 times with different 5-of-20 trajectory subsets; Table 1 reports the mean.

## Experimental Results

Main result on M-Path 4-class melanoma, balanced 35-WSI test set, mean of 10 runs of 5-trajectory majority vote unless noted. `*` ABMIL rows are single-run, no voting.

| Method | Accuracy | F1 |
|---|---|---|
| Human Experts [9] | 0.65 | 0.65 |
| ScAtNet [52] | 0.62 | 0.62 |
| ScAtNet + ROI Heatmap | 0.63 | 0.63 |
| ScAtNet + SAG | 0.60 | 0.60 |
| ABMIL* | 0.46 | 0.47 |
| ABMIL w/ CONCH* | 0.66 | 0.60 |
| ABMIL w/ UNI2-h* | 0.66 | 0.66 |
| ABMIL w/ QuiltNet* | 0.61 | 0.63 |
| BioMistral-7B (LLM-only) | 0.43 | 0.43 |
| Mistral-Nemo-Instruct-2407 | 0.41 | 0.41 |
| GPT-4o (LLM-only over PathFinder descriptions) | 0.49 | 0.49 |
| Meta-Llama-3-8B-Instruct | 0.31 | 0.31 |
| LLaVA-Med-v1.5-Mistral-7b | 0.43 | 0.43 |
| Quilt-LLaVA-v1.5-7b | 0.29 | 0.29 |
| PathFinder + ABMIL/UNI2-h top-patch selection | 0.46 | 0.46 |
| PathFinder + ABMIL/QuiltNet top-patch selection | 0.46 | 0.46 |
| PathFinder + ABMIL/CONCH top-patch selection | 0.54 | 0.54 |
| PathFinder + T5 nav + **No Triage** | 0.58 | 0.58 |
| PathFinder + T5 nav + LLaVA-Med descriptions | 0.56 | 0.56 |
| PathFinder + CLIP nav + LLaVA-Med descriptions | 0.60 | 0.60 |
| PathFinder + Imitated Sampling (pathologist viewport stats) | 0.63 | 0.63 |
| PathFinder + Vision-Only Navigator (no text feedback) | 0.64 | 0.64 |
| PathFinder + CLIP navigator | 0.62 | 0.62 |
| PathFinder + Exhaustive search over all FG patches | 0.68 | 0.68 |
| **PathFinder + T5 Text-Conditioned Visual Navigator (full)** | **0.74** | **0.74** |

Auxiliary findings worth flagging:

- **Triage Agent ablation (Appendix Table 1):** 0.91 overall accuracy / 0.95 Non-Class-1 F1 / **0.57 Class-1 F1** vs TransMIL 0.83 / 0.90 / 0.40 — better than MIL baselines but still ~43% F1 miss rate on dysplastic-nevi cases that then never reach the rest of the pipeline.
- **Trajectory scaling (Fig. 3):** majority-vote accuracy plateaus by ~5 trajectories of 10 patches at ~0.74; with 5 trajectories, accuracy climbs from ~0.60 at length 2 to ~0.74 at length 10.
- **Encoder choice:** T5-conditioned navigator beats CLIP-conditioned by 12 pts (0.74 vs 0.62), attributed to CLIP's 77-token truncation discarding aggregated context.
- **Description preference (Fig. 4):** PathFinder's fine-tuned 7B describer ties GPT-4o under double-blind pathologist preference on 25 ROIs (2 raters), both dominating LLaVA-Med; preference reasons cluster on correctness, not detail (both forced to ≤20 words).
- **Selective vs exhaustive:** iterative navigation (0.74) beats exhaustively analyzing all non-background patches (0.68) — *which* patches you skip matters.

## Limitations

**Authors acknowledge:**

- Heavy reliance on pre-existing curated datasets and substantial compute, limiting low-resource deployment.
- Navigator decision-making is opaque — it's a U-Net, so the heatmap is the only window in.
- Description Agent occasionally hallucinates, which propagates into Diagnosis.
- GPT-2 was chosen for resource reasons; the authors expect a stronger LLM would help.

**The reader should also notice:**

- **No external validation.** All numbers are on M-Path melanoma, n=35 test cases (≈9 per class after balancing). Generalization to other cancers, stains, scanners, or sites is untested.
- **No statistical significance test** on the 8-pt margin over CONCH-ABMIL; at n=35 that margin corresponds to roughly ±3 cases.
- **No per-class results** in Table 1 — only aggregate accuracy/F1. With Triage Class-1 F1 at 0.57, ~43% of dysplastic-nevi cases may be mis-routed; their downstream fate is unreported.
- **Pathologist comparison is not head-to-head.** The 0.65 baseline comes from a separate study with different conditions, not the same 35 WSIs read under matched protocols.
- **Cost / latency unbenchmarked.** 5 trajectories × 10 patches × (U-Net forward + T5 encode + Quilt-LLaVA 7B caption + paraphrase) × 10 evaluation runs has no wall-clock or GPU-hour budget reported.
- **Hallucination → diagnosis pathway is unquantified.** Description-Agent hallucinations are flagged but not measured; there's no audit of whether wrong descriptions correlate with wrong diagnoses.
- **Ground-truth heatmap leakage.** The Navigator is trained on pathologist viewports from the same M-Path study whose split defines the eval; how much of 0.74 is genuine ROI signal vs study-specific viewport priors?
- **Triage as bottleneck.** The Diagnosis Agent only ever sees Classes 2–4; end-to-end joint training is not explored.
- **GPT-4o description comparison** (C6) is 25 ROIs × 2 raters × forced ≤20-word prompts with no inter-rater agreement and no significance test — closer to anecdote than benchmark.

## Why It Matters for Medical AI

Most multi-agent medical-LLM systems are essentially LLM ensembles wrapped around a fixed retrieval set. PathFinder is one of the first to couple a *visual* policy (the Navigator's where-to-look-next heatmap) with a *textual* memory (the running T5-encoded description aggregate) inside a closed feedback loop, with a domain-specific safety gate (Triage) sitting in front. That architectural pattern — small task-specific encoders for gating and routing, mid-size open MLLMs for description, plain LMs for trajectory aggregation — is portable to other gigapixel-evidence tasks (radiology multi-slice CT, longitudinal EHR review) where iterative evidence accumulation matters more than a single forward pass. The benchmark caveats are real, though: until PathFinder is run on TCGA, CPTAC, or a multi-site external melanoma cohort with significance testing, the "first AI to beat pathologists" framing should be read as a hypothesis the paper is well-positioned to test next, not a settled result.

## References

- Paper: [PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology (arXiv:2502.08916)](https://arxiv.org/abs/2502.08916)
- Project page / code / demo: [pathfinder-dx.github.io](https://pathfinder-dx.github.io/)
- Quilt-LLaVA describer backbone: Seyfioglu et al., *Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos* (CVPR 2024)
- QuiltNet / Quilt-1M feature encoder: Ikezogwo et al., *Quilt-1M: One Million Image-Text Pairs for Histopathology* (NeurIPS 2023)
- M-Path melanoma dataset and pathologist baseline: Elmore et al., *Pathologists' diagnosis of invasive melanoma and melanocytic proliferations*, BMJ 2017
- CONCH / UNI baselines: Lu et al., *A visual-language foundation model for computational pathology* (Nature Medicine 2024); Chen et al., *Towards a General-Purpose Foundation Model for Computational Pathology* (Nature Medicine 2024)
- ScAtNet baseline: Mehta et al., *Scale-Aware Transformer for Histopathological Image Classification* (TMI 2023)
- ABMIL: Ilse et al., *Attention-based Deep Multiple Instance Learning* (ICML 2018)
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022)
