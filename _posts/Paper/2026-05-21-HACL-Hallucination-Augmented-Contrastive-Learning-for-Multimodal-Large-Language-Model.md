---
title: "HACL: Hallucination Augmented Contrastive Learning for Multimodal Large Language Model"
excerpt: "An auxiliary InfoNCE loss on EOS-token LLM hidden states with GPT-4-fabricated hallucinative captions as image-to-text hard negatives, applied only in stage-1 pretraining with the LLM frozen. Headline LLaVA POPE F1 jumps 66.71 -> 83.82 and Yes-ratio collapses 99.55% -> 44.33% — but the gains shrink to within-noise on LLaVA-1.5."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/hacl/
tags:
  - HACL
  - Hallucination
  - Contrastive-Learning
  - InfoNCE
  - MLLM
  - LLaVA
  - MiniGPT-4
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- **HACL** adds an auxiliary cross-modal InfoNCE loss during stage-1 MLLM pretraining, computed on the EOS-token last-layer hidden state inside the LLM itself, with **GPT-4-fabricated hallucinative captions used as image-to-text hard negatives**. Stage 2 (instruction tuning) is unchanged.
- The strongest evidence is the LLaVA POPE Random split: **F1 jumps from 66.71 to 83.82**, and **Yes-ratio collapses from 99.55% to 44.33%** (target 50%). The MMHal-Bench overall score goes 1.55 -> 2.08 on LLaVA and 1.39 -> 1.80 on MiniGPT-4.
- The weakest evidence is the universality claim: on the already-strong LLaVA-1.5 baseline, all gains shrink to within roughly $\pm 1.4$ points (POPE F1 87.31 -> 88.70; MME +19.40; VQAv2 +0.6). The headline percentage improvements anchor on the weakest MLLMs in the leaderboard.

## Motivation

The authors reframe hallucination not as a decoding problem (top-$p$, beam, contrastive decoding) and not as an RLHF problem (LLaVA-RLHF), but as a **representation-alignment problem inside the LLM**. They visualize the EOS-token last-layer hidden state for (image, GT caption, GPT-4-hallucinative caption) triples from a trained LLaVA and report two observations: (i) the projected visual tokens form their own cluster, far from any text cluster — a modality gap that survives the learnable interface; (ii) GT captions and hallucinative captions are entangled in the same text cluster, so the LLM has no embedding-level signal to prefer the GT over the hallucinative variant.

![Modality gap and HACL effect on EOS-token representations, plus MMHal-Bench and MME bar charts](/assets/images/paper/hacl/fig_p001_01.png)
*Figure 1: (a) Without HACL, projected visual tokens (purple) sit far from text tokens (green), and GT captions and hallucinative captions are mixed inside the text cluster. (b) With HACL, the modality gap narrows and the hallucinative cluster separates from the GT cluster. (c) MMHal-Bench overall score and MME on LLaVA / MiniGPT-4 baselines.*

The thesis follows directly: if the learnable interface is pushed to (i) pull visual reps toward GT-caption reps and (ii) push them away from hallucinative-caption reps, the LM should be less prone to emit the hallucinative variants at decode time. There is no medical-AI angle in the paper — all benchmarks are natural-image (COCO, POPE, MMHal-Bench).

## Core Innovation

- **Contrastive loss computed inside the LLM, not in a CLIP-style external head.** An `<EOS>` token is appended to (a) the projected visual token sequence $S_v$ and (b) the text token sequence $S_t$. Each is passed through the *same* frozen LLM, and the last-layer hidden state at the EOS position becomes the global representation $\hat e_v$ / $\hat e_t$ on which InfoNCE is computed.
- **GPT-4 hallucinative captions as paired hard negatives in the image-to-text direction.** For each GT caption, GPT-4 rewrites it with fine-grained (attribute, count, location) or coarse-grained (object existence) errors of comparable length. This paired negative goes into the InfoNCE denominator only for the $v \to t$ direction; the $t \to v$ direction is left unmodified.
- **Stage-1-only, frozen-LLM design.** HACL is applied during caption pretraining with the vision encoder *and* the LLM frozen — only the projector/Q-former is trained. Stage 2 instruction tuning is unchanged. Table 6 shows that unfreezing the LLM during stage 1 collapses MME, so this is not a free design choice.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | There is a modality gap between projected visual tokens and text tokens *inside the LLM*, and GT vs. hallucinative captions are entangled. | Figure 1(a), Figure 4(a) — PCA of EOS-token hidden states. | 200 COCO val2017 pairs. | ⭐⭐ — qualitative PCA only; no CKA, no cosine-distance distribution, no significance test. |
| C2 | HACL reduces the modality gap and separates hallucinative from GT text reps. | Figure 1(b), Figure 4(c). | Same 200 COCO pairs. | ⭐⭐ — qualitative; PCA projection axes are not fixed across panels. |
| C3 | HACL improves MMHal-Bench overall by 34.66% / 29.5% over MiniGPT-4 / LLaVA. | Table 1. | MMHal-Bench (GPT-4-as-judge). | ⭐⭐ — single benchmark, single seed, no variance bars, GPT-4-as-judge bias unaddressed. The absolute deltas (+0.41, +0.53) are real on the *weak* baselines. |
| C4 | HACL reduces object hallucination on POPE. | Table 2. | POPE Random/Popular/Adversarial on COCO. | ⭐⭐⭐ on LLaVA (F1 66.71 -> 83.82 on Random; Yes-ratio 99.55% -> 44.33% indicates real calibration). ⭐ on LLaVA-1.5 (+0.42 to +1.39 F1 — within noise). |
| C5 | HACL also improves general visual comprehension, not just hallucination. | Tables 3-4 (VQAv2, GQA, VizWiz, TextVQA, ScienceQA, MME, MMBench, MM-Vet, SEED). | 9 benchmarks. | ⭐⭐ — signs are consistent but magnitudes on LLaVA-1.5 are 0.2-2 pt, within the multi-modal noise floor. The LLaVA MME +60 likely reflects yes/no calibration fixing, not new visual skill. |
| C6 | Hallucinative captions (HC), not contrastive learning alone (CL), drive the gains. | Table 5 ablation. | POPE/MMHal/VQA/MME. | ⭐⭐⭐ on LLaVA (CL alone: POPE 69.23; +HC: 78.31, a 9-point lift). ⭐ on LLaVA-1.5 (+HC adds ~1 pt over CL). |
| C7 | HACL requires a frozen LLM in stage 1; activating the LLM is catastrophic. | Table 6. | POPE/MMHal/VQA/MME. | ⭐⭐ — supports the design but limits applicability to architectures where the LLM is updated during stage-1 pretraining (e.g., InternVL-style). |
| C8 | Hard-negatives in the image-to-text direction are sufficient; the text-to-image direction does not need them. | Implicit in Eq. 5. | None. | ⭐ — asymmetric design is **unablated**. No experiment justifies the asymmetry. |

**Honest take.** Two things are clearly true. First, on the weak baselines (LLaVA-7B and MiniGPT-4) the POPE Yes-ratio collapsing from 99.55% to 44.33% is a real, large, calibration-level effect — those models were emitting "yes" to essentially every object-presence query, and HACL teaches the projector to encode object identity instead of memorising co-occurrence priors. That is a genuine contribution. Second, on LLaVA-1.5 the picture is very different: every metric moves by $\le 1.4$ points, which is well inside the run-to-run noise floor that this paper never quantifies. So the headline 29-35% gains are real but they live on the weakest models in the leaderboard, and the universality claim is much softer than the abstract suggests.

The single most striking omission is **CHAIR — the standard caption-level free-form hallucination metric**. HACL's framing is generation-side ("MLLMs hallucinate when they describe images"), but every generation-side number in the paper is GPT-4-as-judge on the small MMHal-Bench. POPE is a binary yes/no probe and doesn't measure free-form caption hallucination. The fact that the central object-hallucination benchmark for captioning isn't in the paper, despite the framing, is hard to read as anything other than a choice.

Other gaps worth flagging: there is **no quality audit of the GPT-4-generated hallucinative captions** — false-positive rate (cases where the "hallucination" is actually plausible or a paraphrase), and the distribution over hallucination types (object-swap vs. attribute vs. count vs. relation) are never reported. There are **no variance bars / no multi-seed numbers**, so we cannot tell which deltas on LLaVA-1.5 are signal and which are noise. The **asymmetric design** (hard negatives only in $v \to t$) is unablated. And there is **no comparison to simpler alternatives that use the same negatives** — e.g., a margin / DPO-style loss against the GPT-4 hallucinative captions — so we cannot tell whether contrastive InfoNCE specifically is necessary, or whether any margin-based objective against these negatives would suffice.

## Method & Architecture

The HACL framework attaches to a standard MLLM stack (vision encoder $V_\theta$ -> learnable interface $F_\alpha$ -> LLM $L_\beta$) and adds an auxiliary contrastive loss during stage-1 pretraining only.

### Per-modality global embeddings inside the LLM

Append `<EOS>` to (a) the projected visual token sequence $S_v$ and (b) the text token sequence $S_t$. Run each through the same LLM separately. Take the last-layer hidden state at the EOS position:

$$
\hat e_v = \mathrm{LLM}([S_v, \texttt{<EOS>}])_{\mathrm{EOS}}, \qquad \hat e_t = \mathrm{LLM}([S_t, \texttt{<EOS>}])_{\mathrm{EOS}}.
$$

Critically, $\hat e_v$ and $\hat e_t$ live in the LLM's hidden-state space, not in a separate CLIP-style projection head. This is what lets HACL act on the modality gap that the visualization in Figure 1 diagnoses.

### Bidirectional InfoNCE (without hallucinations)

For a batch of $N$ (image, caption) pairs, with $f(x, y) = \exp(\mathrm{sim}(x, y) / \tau)$:

$$
\mathcal{L}_{t \to v} = -\mathbb{E}_i \log \frac{f(\hat e_t^i, \hat e_v^i)}{f(\hat e_t^i, \hat e_v^i) + \sum_{k \ne i} f(\hat e_t^i, \hat e_v^k)}
$$

and the symmetric $\mathcal{L}_{v \to t}$. So far this is ALBEF/CLIP-style image-text contrastive on EOS-token embeddings inside the LLM.

### Hallucination-augmented image-to-text loss

For each anchor image $i$, GPT-4 produces a paired hallucinative caption whose EOS-embedding is $\dot e_t^i$. Add it to the $v \to t$ denominator as an extra (paired, hard) negative:

$$
\mathcal{L}_{v \to t} = -\mathbb{E}_i \log \frac{f(\hat e_v^i, \hat e_t^i)}{f(\hat e_v^i, \hat e_t^i) + f(\hat e_v^i, \dot e_t^i) + \sum_{k \ne i} f(\hat e_v^i, \hat e_t^k)}.
$$

The $t \to v$ direction is unchanged — there is no "hallucinative image" anchor on the other side. This asymmetry is a design choice the paper does not ablate.

### Stage-1 joint objective

The total stage-1 objective is

$$
\mathcal{O}_\alpha = \arg\min \mathcal{L}_G + \tfrac{1}{2}(\mathcal{L}_{v \to t} + \mathcal{L}_{t \to v}),
$$

where $\mathcal{L}_G$ is the standard caption-generation NLL, and **only $\alpha$ (the interface) is trained**. Both the vision encoder and the LLM are frozen. Stage 2 (instruction tuning) is unchanged from each baseline and does **not** use the contrastive term.

### GPT-4 hallucinative caption generation

GPT-4 is prompted with a GT caption plus few-shot exemplars and asked to rewrite it inserting hallucination errors of comparable length. Errors are either coarse-grained (object existence) or fine-grained (attribute / count / location). Examples:

![GPT-4-generated hallucinative caption: bus with fabricated colors and patterns](/assets/images/paper/hacl/fig_p004_01.png)
*Figure 2: A coarse-grained attribute hallucination — the rewrite inserts color and pattern attributes that are absent from the image.*

![GPT-4-generated hallucinative caption: horse with non-existent silver sword](/assets/images/paper/hacl/fig_p004_02.png)
*Figure 3: A coarse-grained object-presence hallucination — GPT-4 introduces a "silver sword" that does not appear in the image.*

![GPT-4-generated hallucinative caption: object swap from elephant to zebra](/assets/images/paper/hacl/fig_p004_03.png)
*Figure 4: An object-swap hallucination — one of two elephants is rewritten as a zebra.*

The paper provides no quantitative quality check on these generations — false-positive rate (rewrites that are not actually hallucinations) and hallucination-type distribution are not reported.

### Practical knobs

- **Negative queue.** 16,384-entry MoCo/ALBEF-style queue compensates for small per-GPU batch size.
- **Compute.** 16 x A100-80G. Batch size 64 / 32 / 8 for LLaVA / LLaVA-1.5 / MiniGPT-4 respectively.
- **Coverage of hallucinative negatives.** For MiniGPT-4, only ~10% of the ~100M pretraining pairs (~10M) get a GPT-4 hallucinative caption — pure cost limit. For LLaVA(-1.5), all 558K pretraining pairs are paired with a hallucinative negative.

## Experimental Results

### MMHal-Bench (Table 1)

| Model | Overall score | Hallucination rate |
|---|---|---|
| MiniGPT-4-7B | 1.39 | — |
| **MiniGPT-4-7B + HACL** | **1.80** (+0.31, paper text claims +34.66% rel.) | — |
| LLaVA-7B | 1.55 | 0.76 |
| **LLaVA-7B + HACL** | **2.08** (+0.53) | **0.62** (-0.15) |
| LLaVA-RLHF-7B | 2.05 | — |
| LLaVA-1.5-7B | 2.08 | — |
| **LLaVA-1.5-7B + HACL** | **2.13** (+0.05) | — |

HACL on LLaVA-7B reaches 2.08 — essentially matching LLaVA-RLHF-7B (2.05) without any RLHF infrastructure. On LLaVA-1.5-7B the gain is +0.05, indistinguishable from noise.

### POPE — Random split, COCO (Table 2)

| Model | F1 | Yes-ratio (target 50%) |
|---|---|---|
| LLaVA-7B | 66.71 | 99.55 |
| **LLaVA-7B + HACL** | **83.82** (+17.11) | **44.33** (delta -55.22) |
| LLaVA-1.5-7B | 87.31 | — |
| **LLaVA-1.5-7B + HACL** | **88.70** (+1.39) | — |

This is the cleanest piece of evidence in the paper. Vanilla LLaVA-7B was saying "yes" to almost every object-presence query (99.55%); after HACL it calibrates toward 50% (44.33%). That is consistent with the projector learning to actually encode object identity. On LLaVA-1.5, where the starting calibration is already much better, HACL adds 1.39 F1 — real but small.

### MME and VQAv2 (Tables 3-4)

| Model | MME | VQAv2 |
|---|---|---|
| MiniGPT-4-7B | 581.67 | — |
| **MiniGPT-4-7B + HACL** | **653.94** (+72.27) | — |
| LLaVA-7B | 502.82 | 71.3 |
| **LLaVA-7B + HACL** | **562.58** (+59.76) | **73.3** (+2.0) |
| LLaVA-1.5-7B | 1510.70 | 78.5 |
| **LLaVA-1.5-7B + HACL** | **1530.10** (+19.40) | **79.1** (+0.6) |

The LLaVA MME of 502 is unusually low for a strong MLLM; the +60-point gain likely reflects fixing the yes/no calibration that the same model exhibits on POPE, not new visual capability. On LLaVA-1.5, where MME is already 1510, the gain shrinks to +19, again within typical run-to-run variance.

### Ablation: CL vs. CL+HC (Table 5)

| Setting | LLaVA POPE Random F1 | LLaVA-1.5 POPE Random F1 |
|---|---|---|
| Baseline | 66.71 | 87.31 |
| + CL (contrastive only, no hallucinative captions) | 69.23 | 86.31 |
| + CL + HC (full HACL) | **78.31** | **87.26** |

On the weak baseline, contrastive learning alone (+CL) gives a 2.5-point bump and hallucinative-caption hard negatives add another 9 points — i.e., HC is doing most of the work. On the strong baseline, CL+HC adds only ~1 point over CL alone. The hard-negative effect is sharply diminishing as the underlying model gets stronger, which limits the "universal" framing.

### Stage-1 training paradigm (Table 6)

Activating the LLM during stage-1 HACL training is catastrophic on MME (jumps from 562 down to as low as 324 depending on the row). The paper interprets this as: HACL's auxiliary InfoNCE loss interferes with LLM-internal representations when the LLM is updated. Practical consequence: HACL is not a drop-in for architectures (e.g., InternVL-style) that update the LLM during stage 1.

## Limitations

**Authors acknowledge.**

- GPT-4 generation cost forces the MiniGPT-4 hallucinative-caption coverage down to ~10% of pretraining pairs.
- Activating the LLM during stage 1 degrades scores; HACL is only validated with a frozen LLM in stage 1.

**Visible from the evidence but not addressed.**

- **No CHAIR-style free-form captioning hallucination evaluation.** The method is framed as fixing generation hallucination, but every generation-side number is GPT-4-as-judge on the small MMHal-Bench. CHAIR is the standard caption-level hallucination metric; its absence is conspicuous.
- **Headline gains anchor on the weakest baselines.** The 29-35% MMHal improvements are computed against LLaVA-7B (POPE F1 66.71) and MiniGPT-4-7B — both known to be heavily yes-biased. On LLaVA-1.5 the same loss adds $\le 1.4$ F1 on POPE and $+0.05$ on MMHal, undermining universality.
- **No variance or multi-seed reporting.** All numbers are point estimates. The within-noise LLaVA-1.5 deltas cannot be distinguished from run-to-run noise.
- **No quality audit of the GPT-4-generated hallucinative captions.** False-positive rate (cases where the "hallucination" is plausible or a paraphrase) and the distribution over hallucination types (object-swap vs. attribute vs. count vs. relation) are not reported. The negatives that drive the loss are themselves opaque.
- **Asymmetric hard-negative use is unablated.** Hallucinative captions enter only in the $v \to t$ direction. The paper does not test a symmetric variant or a $t \to v$-only variant.
- **No comparison to simpler margin / DPO baselines using the same negatives.** Is contrastive InfoNCE specifically necessary, or would any margin-based objective against GPT-4-generated negatives suffice? The paper does not answer.
- **No non-COCO / non-natural-image evaluation.** Every hallucination benchmark is COCO-distributed. No medical, document, chart, or other domain-shifted evaluation — so we cannot tell whether HACL fixes hallucinations or whether it fixes COCO-flavored hallucinations.
- **Frozen-LLM stage-1 requirement constrains applicability.** Architectures that update the LLM in stage 1 (InternVL-style) cannot use HACL as-is, per Table 6.

## References

- Paper: Jiang, Xu, Dong, Chen, Ye, Yan, Ye, Zhang, Huang, Zhang. *Hallucination Augmented Contrastive Learning for Multimodal Large Language Model.* CVPR 2024. [arXiv:2312.06968v4](https://arxiv.org/abs/2312.06968).
- Code: [github.com/X-PLUG/mPLUG-HalOwl/tree/main/hacl](https://github.com/X-PLUG/mPLUG-HalOwl/tree/main/hacl).
- Related: Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023.
- Related: Liu et al., *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*, CVPR 2024.
- Related: Zhu et al., *MiniGPT-4*, ICLR 2024.
- Related: Li et al., *POPE: Evaluating Object Hallucination in Large Vision-Language Models*, EMNLP 2023.
- Related: Sun et al., *Aligning Large Multimodal Models with Factually Augmented RLHF (LLaVA-RLHF)*, ACL Findings 2024.
- Related: Rohrbach et al., *Object Hallucination in Image Captioning (CHAIR)*, EMNLP 2018.
- Related: He et al., *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)*, CVPR 2020.
- Related: Li et al., *Align before Fuse (ALBEF)*, NeurIPS 2021.
