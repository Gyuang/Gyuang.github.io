---
title: "Read Like a Radiologist: Efficient Vision-Language Model for 3D Medical Imaging Interpretation"
excerpt: "MS-VLM replaces 3D-patch encoders with per-slice 2D DINO [CLS] tokens aggregated by a Big Bird Z-former, reaching ROUGE-L 0.438 / METEOR 0.396 / CA-F1 0.261 on CT-RATE — but losing BLEU-4, Precision, and the GPT-judge Hallucination column to CT2Rep."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/read-like-radiologist-efficient-3d-medical-vlm/
tags:
  - MS-VLM
  - Z-former
  - Masked Embedding Modeling
  - DINO
  - Big Bird Sparse Attention
  - Perceiver Resampler
  - Vicuna LoRA
  - CT-RATE
  - Rectal MRI
  - 3D Medical VLM
  - Radiology Report Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- 3D medical VLMs (RadFM, CT2Rep, M3D-LaMed) inherit 3D-patch encoders from video models that over-correlate adjacent slices and fix the z-length; **MS-VLM treats each axial slice as a separate 2D DINO ViT-B/16 [CLS] token and lets a Big Bird "Z-former" sparse-attention transformer aggregate them**, trained with Masked Embedding Modeling (mask ratio 0.3, L1 reconstruction).
- The construction is variable-length by design — Z-former + a 32-query Perceiver Resampler bridge into Vicuna-7B with LoRA, so the model accepts the *original* slice count $L$ rather than CT2Rep's hard-coded $z=240$, and it can stack multi-view × multi-phase rectal MRI (5 planes × T1/T2/DWI) as one sequence.
- On CT-RATE (3,039-volume test): **ROUGE-L 0.438, METEOR 0.396, CA-F1 0.261** — beating CT2Rep on 4 of 6 NLG/CA columns, but losing **BLEU-4 (0.232 vs 0.252)**, **Precision (0.222 vs 0.355)**, and the **GPT-4o-mini Hallucination judge (0.127 vs 0.186)**. On in-house rectal MRI (43-patient test), the all-views/all-phases configuration wins 6 of 7 findings (T-stage 0.581 vs T2-axial-only 0.279) — but $n=43$, single hospital, no CIs.

## Motivation

3D medical VLMs face two structural problems. First, paired 3D-volume / report data is scarce — CT-RATE has 25k volumes versus millions of CXR pairs. Second, the 3D vision encoders the field has imported from video understanding (3D ViT, 16×16×16 patches) over-correlate the z-axis precisely where redundancy is *low*, smearing the slice-specific signals (a tumor that disappears across slices vs a vessel that continues across slices) that radiologists actually key on.

The paper's framing is workflow-aligned: read each slice independently with a 2D ViT, then learn inter-slice structure with a sparse-attention encoder over [CLS] tokens. The medical angle is direct — the rectal MRI case study (multi-phase, multi-plane T1/T2/DWI) is exactly the kind of multi-acquisition setting that fixed-grid 3D encoders structurally cannot handle.

Note that the "read like a radiologist" tag is a *design analogy*, not a measured property. The paper presents no eye-tracking, no attention-map analysis, and no expert reader study to test whether Z-former attention actually resembles radiologist gaze patterns or reading order — see Claim C5 below.

## Core Innovation

Two pieces do the work.

**Slice-as-token construction.** A 2D ViT-B/16 pre-trained with DINO on ImageNet-1K and then fine-tuned 50 epochs in-domain (global crops 0.8–1.0, local 0.1–0.3, rotations + auto-contrast + equalization) maps each axial slice $i \in \{1,\dots,L\}$ of shape $3\times480\times480$ to a [CLS] embedding $z^{[CLS]}_i$. The volume becomes a sequence $Z_{vol}=[z^{[CLS]}_1,\dots,z^{[CLS]}_L]$ rather than a 3D patch grid.

**Z-former with Masked Embedding Modeling.** A 12-layer, $d=768$ Big Bird sparse-attention transformer (sliding window 16 + 3 random blocks, **no global blocks**) is trained to reconstruct masked slice tokens:

$$\mathcal{L}_{MEM} = \frac{1}{|\mathcal{M}|}\,\bigl\lVert \hat Z_{masked} - Z_{vol}\bigr\rVert_1$$

Mask probability 0.3, 20 epochs, Adam lr $=10^{-4}$ with 50-iter warmup from $10^{-5}$. This is the inter-slice context module — and because it is sparse-attention over a token sequence rather than a fixed grid, the downstream pipeline naturally takes arbitrary $L$.

The Z-former output is then compressed by a **Perceiver Resampler** (32 learnable queries, dim 4096) and an MLP into Vicuna-7B-v1.5's embedding space, wrapped in `<Img/>...</Img>` tokens, and fine-tuned with LoRA on a joint 1:1 mix of report generation and VQA (RadGenome-ChestCT 381k filtered + 180k LLaMA-3-8B synthetic QAs).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | MS-VLM outperforms 3D-encoder baselines on report generation | Table 1: ROUGE-L 0.438, METEOR 0.396, F1 0.261 vs CT2Rep 0.432 / 0.328 / 0.175 | CT-RATE (3,039-vol test) | ⭐⭐ — single test set, no variance reported; CT2Rep wins BLEU-4 and Precision |
| C2 | MS-VLM beats baselines on LLM-judge metrics | Table 2: wins Presence / Location / Severity, **loses Hallucination** to CT2Rep | CT-RATE + GPT-4o-mini judge | ⭐⭐ — LLM judges are noisy, no judge-bias controls, no inter-rater agreement |
| C3 | MS-VLM is robust to variable z-length and beats fixed-z on long volumes | Table 3: MS-VLM($z=L$) F1 0.246 > MS-VLM($z=240$) F1 0.237 on $L>240$ subgroup | CT-RATE | ⭐⭐ — directionally supported but the gap is ~1 F1 point; no test of how large $L$ gets |
| C4 | Multi-view + multi-phase integration boosts rectal MRI accuracy | Table 4: all views/phases > T2-axial-only on 6 of 7 findings (T-stage 0.581 vs 0.279) | In-house rectal MRI (43-patient test) | ⭐⭐ — large effects but $n=43$, single hospital, no CIs, single-judge (o1-mini) |
| C5 | "Read like a radiologist" — slice-by-slice processing reflects radiologist workflow | Section 5 narrative; no attention-map analysis, no gaze comparison, no reader study | — | ⭐ — **purely rhetorical**. The architecture is radiologist-inspired; the model behavior is not shown to be radiologist-like. |
| C6 | Computational efficiency / faster convergence vs 3D-encoder pipelines | Figure 7: F1 trajectory over epochs; CT2Rep needs 40 epochs to reach what MS-VLM does in ~5 | CT-RATE | ⭐⭐ — convergence-epoch evidence is real, but **no FLOPs, no parameter counts, no per-epoch wall-clock, no inference latency, no head-to-head against Merlin or CT-CHAT** |
| C7 | Two-stage DINO + MEM pre-training is responsible for efficiency | Implicit from convergence figure; no MEM-vs-no-MEM or DINO-vs-ImageNet ablation | — | ⭐ — no isolation; the only ablation swaps in a 3D encoder wholesale |
| C8 | Joint VQA training improves report generation | "MS-VLM (w/o VQA)" row in Tables 1 & 2: F1 drops 0.261 → 0.232 | CT-RATE | ⭐⭐ — clean controlled ablation, single seed |
| C9 | Generalizes across imaging modalities (CT, MRI, PET-CT, 3D US per Discussion) | Only CT and rectal MRI shown | — | ⭐ — explicitly acknowledged as unvalidated; PET-CT / US claims are aspirational |

**Honest summary.** The Z-former + slice-token construction is a sensible architectural answer to a real critique of 3D-patch encoders, and MS-VLM does land on top of CT2Rep on most CT-RATE columns. Two of the paper's headline framings, however, deserve scrutiny:

1. **"Read like a radiologist" is rhetoric, not evidence.** No attention visualization, no gaze comparison, no expert reader study, no analysis of whether the Z-former actually attends slice-by-slice or just learns whatever helps the loss. The architecture is radiologist-inspired; the *model behavior* is not shown to be radiologist-like.
2. **"Efficient" is narrowly operationalized.** The efficiency argument is one convergence curve (Figure 7). The paper reports **no parameters, no FLOPs, no GPU-hours, no peak memory, no inference latency**, and the two most directly comparable contemporary 3D CT VLMs — Merlin (Blankemeier 2024) and CT-CHAT (from the CT-RATE team) — are cited but **never benchmarked head-to-head on report generation**. M3D-LaMed and RadFM are dismissed in one sentence.

Other gaps: no variance / no significance tests anywhere; rectal MRI test $n=43$ with no CIs; LLM-judge metrics are not calibrated against radiologists; per-abnormality F1 (Appendix E) shows real weakness on rarer findings (Bronchiectasis 0.111, Interlobular septal thickening 0.125) that the abstract does not surface.

## Method & Architecture

![MS-VLM architecture](/assets/images/paper/read-like-radiologist/fig_p007_01.png)
*Figure 1: MS-VLM architecture overview — per-slice 2D DINO ViT-B/16 [CLS] tokens are aggregated by the Z-former (Big Bird sparse attention), compressed by a 32-query Perceiver Resampler, and fed into Vicuna-7B-v1.5 (LoRA) wrapped in `<Img/>...</Img>` tokens.*

The training pipeline is four staged passes — DINO domain fine-tune, Z-former MEM pre-training, bridger alignment, then joint LLM instruction tuning on report-generation + VQA at 1:1.

![Four-stage training pipeline](/assets/images/paper/read-like-radiologist/fig_p008_01.png)
*Figure 2: Four-stage training. Stage 0 — domain DINO fine-tune of ViT-B/16 (50 epochs). Stage 1 — Z-former MEM pre-training (mask 0.3, L1, 20 epochs). Stage 2 — Perceiver bridger alignment (1 epoch, frozen DINO + Z-former). Stage 3 — Vicuna LoRA instruction tuning (5 epochs joint, 1:1 report:VQA).*

The rectal MRI motivation makes the variable-length advantage concrete — the test set demands seven structured findings keyed to multiple planes and phases.

![Rectal MRI report example](/assets/images/paper/read-like-radiologist/fig_p012_01.png)
*Figure 5: Rectal MRI report example, with seven key tumor-related findings (location, peritoneal involvement, T-staging, CRM, ASI, MLNI, EMVI) color-coded.*

To address VQA-pair scarcity, the authors generate ~180k synthetic QAs from CT-RATE reports with LLaMA-3-8B (filtering Yes/No-only answers).

![Synthetic VQA generation pipeline](/assets/images/paper/read-like-radiologist/fig_p010_02.png)
*Figure 4: Synthetic VQA pair generation from CT-RATE reports with LLaMA-3-8B. Same LLM family is later judged against by GPT-4o-mini / o1-mini — different family partly mitigates leakage but introduces other biases.*

Compute budget: CT-RATE training on 8× A100 40GB; rectal MRI on a single A100 40GB. Inputs $240\times3\times480\times480$ for CT, $120\times3\times480\times480$ for MRI (20 slices × 6 phases).

## Experimental Results

### Chest CT report generation on CT-RATE

| Method | BLEU-4 | ROUGE-L | METEOR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 3D-CT-GPT | 0.133 | 0.145 | 0.140 | — | — | — |
| CT2Rep (3D enc.) | **0.252** | 0.432 | 0.328 | **0.355** | 0.132 | 0.175 |
| MS-VLM (3D encoder) | 0.184 | 0.364 | 0.331 | 0.193 | 0.231 | 0.207 |
| MS-VLM (w/o VQA) | 0.189 | 0.374 | 0.313 | 0.192 | 0.298 | 0.232 |
| **MS-VLM (proposed)** | 0.232 | **0.438** | **0.396** | 0.222 | **0.329** | **0.261** |

CA averaged over 18 abnormalities. CT2Rep wins BLEU-4 and Precision; MS-VLM wins ROUGE-L, METEOR, Recall, and F1. **No variance, no significance tests reported.**

### GPT-4o-mini binary judge on CT-RATE

| Method | Presence | Location | Severity | Hallucination |
|---|---|---|---|---|
| CT2Rep (3D enc.) | 0.186 | 0.197 | 0.216 | **0.186** |
| MS-VLM (3D enc.) | 0.203 | 0.216 | 0.223 | 0.122 |
| MS-VLM (w/o VQA) | 0.212 | 0.207 | 0.209 | 0.124 |
| **MS-VLM (proposed)** | **0.217** | **0.225** | **0.234** | 0.127 |

3 of 4 categories to MS-VLM but **Hallucination loses to CT2Rep** — the authors attribute the gap to CT2Rep's CT-RATE-specific tokenizer. The win margins (~0.005–0.018) are small compared to LLM-judge noise that nobody calibrated against radiologists.

### Variable z-length robustness (CA metrics)

| Setting | Method | Precision | Recall | F1 |
|---|---|---|---|---|
| All volumes | CT2Rep ($z=240$) | 0.355 | 0.132 | 0.175 |
| | **MS-VLM ($z=240$)** | 0.222 | 0.329 | **0.261** |
| | MS-VLM ($z=L$) | 0.234 | 0.286 | 0.245 |
| Volumes with $L>240$ | CT2Rep ($z=240$) | 0.330 | 0.146 | 0.190 |
| | MS-VLM ($z=240$) | 0.229 | 0.258 | 0.237 |
| | **MS-VLM ($z=L$)** | 0.226 | 0.284 | **0.246** |

Variable-length helps on the long-volume subgroup, but the gap (0.246 vs 0.237) is roughly one F1 point — directional support for C3, not dramatic.

### Rectal MRI (43-patient test, o1-mini judge)

| Model | Loc. | Peri. | T-Stage | CRM | ASI | MLNI | EMVI |
|---|---|---|---|---|---|---|---|
| CT2Rep (T2 axial) | 0.209 | 0.488 | 0.163 | 0.143 | 0.133 | 0.108 | 0.000 |
| MS-VLM (3D enc., T2 ax.) | 0.116 | 0.372 | 0.116 | 0.171 | 0.121 | 0.426 | 0.111 |
| MS-VLM (T2 axial) | 0.209 | 0.395 | 0.279 | 0.308 | 0.114 | 0.298 | 0.111 |
| **MS-VLM (all views & phases)** | **0.233** | **0.512** | **0.581** | **0.421** | **0.222** | **0.667** | **0.143** |

All views × all phases dominates every column. Caveat: $n=43$ patients, single hospital (Chungnam University Hospital), no external site, no confidence intervals, no inter-rater agreement check on the o1-mini judge.

### Qualitative — CT report comparison

![Qualitative CT report comparison](/assets/images/paper/read-like-radiologist/fig_p018_01.png)
*Figure 6: CT-RATE qualitative example, color-coded TP/TN/FP/FN. MS-VLM recovers more true positives and fewer false negatives than CT2Rep / MS-VLM (3D encoder).*

### Efficiency claim — convergence epochs

![F1 convergence over epochs](/assets/images/paper/read-like-radiologist/fig_p023_01.png)
*Figure 7: 18-abnormality mean F1 over training epochs. MS-VLM passes CT2Rep's eventual F1=0.175 within ~5 epochs; CT2Rep needs ~40. This is the entire empirical basis for the "efficient" framing — no FLOPs, no params, no wall-clock, no Merlin / M3D-LaMed / CT-CHAT comparison is provided.*

### Rectal MRI qualitative

![Rectal MRI qualitative](/assets/images/paper/read-like-radiologist/fig_p033_02.png)
*Figure D.4: Rectal MRI qualitative — the multi-view / multi-phase MS-VLM more faithfully reproduces ground-truth findings than the T2-axial-only baselines.*

Per-abnormality CA (Appendix E) shows real heterogeneity: strong on Lung opacity (F1 0.505) and Lung nodule (0.499), weak on Bronchiectasis (0.111) and Interlobular septal thickening (0.125). VQA-mode F1 averages 0.297 vs report-mode 0.261.

## Limitations

**Acknowledged by the authors:**

- Hallucinations are not eliminated.
- Validation limited to chest CT and rectal MRI — no abdominal CT, no other body parts.
- No expert evaluation / reader study.
- Proof-of-concept scale; larger or higher-quality data could improve absolute numbers.

**Not addressed:**

- **No FLOP / parameter / latency / memory comparison** against Merlin, M3D-LaMed, CT-CHAT, or even CT2Rep. The "efficient" claim is *training-convergence-only*.
- **No head-to-head on report generation** against Merlin or CT-CHAT, the two most directly comparable contemporary 3D CT VLMs (both cited in related work).
- **No ablation isolating Z-former vs 2D ViT vs MEM vs Big Bird sparsity.** The only baseline ablation is a wholesale "swap in a 3D encoder" comparison.
- **No empirical validation** that the model behaves like a radiologist — no attention-map analysis, no gaze comparison, no expert reader study. The radiologist-mimicking framing is design-level, not behavior-level.
- **No external test site** for either CT-RATE (single-source) or rectal MRI (single-hospital, $n=43$).
- **No variance reporting, no significance testing, no multiple seeds** anywhere in the experimental section.
- Synthetic VQA generated by LLaMA-3 may inject linguistic patterns that match LLM-judge preferences.
- No discussion of failure modes on rare findings (Bronchiectasis F1=0.111).
- Vicuna-7B-v1.5 is a frozen 2023 artifact; no comparison to newer LLM backbones.
- The "insufficiently short reports" exclusion (CT) and "lacking sufficient detail on seven key findings" exclusion (MRI, 82 of 311 = 26%) are subjective filters that may up-bias reported scores.

## Why It Matters for Medical AI

The slice-as-token construction is a real architectural concession to how 3D imaging actually behaves: the z-axis is the axis with the *least* redundancy in clinical reads, so blurring it with cubic patches is the wrong inductive bias. Z-former + Perceiver gives the field a clean way to handle volumes of arbitrary $L$ and to fuse multi-plane × multi-phase MRI as a single token sequence — the rectal MRI experiment demonstrates the latter is not just a paper architecture, it can take a 6-phase × 5-plane volume and read it.

What MS-VLM does *not* yet establish: that this design is more *efficient* than Merlin / M3D-LaMed / CT-CHAT in any compute sense beyond convergence epochs against the older CT2Rep baseline; that the model attends slice-by-slice in a radiologist-like way; that it generalizes outside chest CT and a single hospital's rectal MRIs; that the LLM-judge wins survive a radiologist reader study. For practitioners interested in adopting the architecture, the takeaway is: the slice-token + sparse-attention recipe is reusable, but the headline framings ("read like a radiologist", "efficient") are not yet supported by the kind of evidence that would let you cite them as established facts.

## References

- arXiv: [2412.13558v1](https://arxiv.org/abs/2412.13558) — Lee, Park et al., "Read Like a Radiologist: Efficient Vision-Language Model for 3D Medical Imaging Interpretation", Dec 2024.
- CT-RATE: Hamamci et al., 2024 — public 25k chest CT / report dataset used as the primary benchmark.
- RadGenome-ChestCT: VQA dataset over CT-RATE, 1.3M QA pairs (filtered to ~381k for MS-VLM training).
- CT2Rep: 3D-encoder baseline; CT-RATE-specific tokenizer drives its Hallucination-judge advantage.
- 3D-CT-GPT, RadFM, M3D-LaMed: alternative 3D medical VLMs cited but not all benchmarked head-to-head.
- Merlin (Blankemeier et al., 2024) and CT-CHAT (Hamamci et al., 2024a): contemporaneous 3D CT VLMs cited in related work but **not benchmarked on report generation in this paper**.
- Big Bird sparse attention (Zaheer et al., 2020) — Z-former backbone (sliding window 16 + 3 random blocks, no global blocks).
- DINO (Caron et al., 2021) — slice-encoder pre-training objective; here used with ViT-B/16 fine-tuned 50 epochs in-domain.
- Perceiver Resampler / Flamingo (Alayrac et al., 2022) — bridge module pattern reused here with 32 learnable queries of dim 4096.
- Vicuna-7B-v1.5 + LoRA — frozen LLM backbone with low-rank instruction tuning.
