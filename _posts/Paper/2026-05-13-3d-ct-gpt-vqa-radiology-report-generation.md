---
title: "3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models"
excerpt: "A VQA-style 3D chest-CT report generator that glues a frozen CT-ViT to a LoRA-tuned Vicuna-7B via 3D average pooling + a single linear projection — fits on one 24 GB RTX 3090 and reaches BLEU 0.3836 on a private test set under the pre-train-then-fine-tune (T1) recipe."
categories:
  - Paper
  - CT-Report-Generation
  - LLM
permalink: /paper/3d-ct-gpt-vqa-radiology-report-generation/
tags:
  - 3D-CT-GPT
  - CT-ViT
  - Vicuna
  - LoRA
  - LLaVA-style-Projection
  - Radiology-Report-Generation
  - Chest-CT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- 3D-CT-GPT is a VQA-style 3D vision-language model for chest-CT report generation: a **frozen CT-ViT** encoder feeds a **3D average-pooled** token grid (kernel=2 → **512 tokens of dim 512**) through a **single linear projector** into a **LoRA-tuned Vicuna-7B**. The whole thing trains on a single 24 GB RTX 3090 (~14 GB stage 1, ~22 GB stage 2).
- The recipe that matters is **T1**: pre-train the projector for 5 epochs on a **CT-RATE subset of 8,070 cases**, then LoRA-fine-tune for 2 epochs on a **private in-house "Dataset-XY" of 1,887 cases** (split 1508 / 190 / 188). Two competing schedules — T2 (Dataset-XY only) and T3 (CT-RATE only) — are reported as the main comparison, not as ablations.
- **Headline result (Dataset-XY_val, T1):** **BLEU 0.3836 / ROUGE-1 0.4749 / ROUGE-L 0.3281 / METEOR 0.3565 / BERTScore-F1 0.8890**. In the only direct head-to-head, T3 beats M3D on the same private set (BLEU 0.2323 vs 0.0869). CT2Rep and RadFM are excluded; no clinical-correctness metric (RadGraph, abnormality F1, radiologist rating) is reported — every number is an n-gram-overlap or BERTScore proxy.

## Motivation

2D radiology-report generation is a mature subfield (X-rayGPT, LLaVA-Med, etc.), but **3D** chest-CT report generation is still scarce because (a) paired 3D-CT/report datasets are rare and (b) volumetric encoders blow up GPU memory. Existing 3D efforts (RadFM, M3D-LaMed, CT2Rep) either don't really do report generation at clinical quality (RadFM, M3D — better at anomaly detection / retrieval) or are too computationally heavy (CT2Rep). 3D-CT-GPT targets a "minimal but effective" recipe — a frozen 3D encoder, a LLaVA-style linear projection, and a LoRA-fine-tuned Vicuna-7B — that fits on one consumer GPU and still beats M3D on n-gram report metrics. The clinical pitch is straightforward automated chest-CT reporting for radiologist workload reduction.

![3D-CT-GPT positioning vs. RadFM / M3D-LaMed](/assets/images/paper/3d-ct-gpt/fig_p001_02.png)
*Figure 1: 3D-CT-GPT vs RadFM / M3D-LaMed — instead of the heavier spatial-pooling perceiver used by prior work, 3D-CT-GPT uses a CT-ViT + 3D average pooling + linear projector pipeline before the LLM, trading representational capacity for a footprint that fits on a single 24 GB GPU.*

## Core Innovation

- **3D average pooling instead of a spatial-pooling perceiver.** Where RadFM / M3D-LaMed use a perceiver-style learned pooler over the volumetric token grid, 3D-CT-GPT does a deterministic permute → 3D avg-pool (kernel=2) → reshape → permute, producing exactly **512 tokens of dim 512** per volume. The cost saving is the entire reason a 240×480×480 volume + Vicuna-7B fits on a single RTX 3090.
- **LLaVA-style single linear projection** $M_v = W \cdot P_v$ aligns CT tokens to Vicuna's word-embedding space; an ablation shows a 2-layer MLP is essentially a wash (BLEU 0.3418 vs 0.3476; linear actually wins on METEOR), so the simpler module is kept.
- **Two-stage training as a recipe.** Stage 1 freezes both CT-ViT and Vicuna and trains *only* the projector (LR 1e-3, ~14 GB). Stage 2 freezes CT-ViT but LoRA-tunes Vicuna along with the projector (LR 2e-4, ~22 GB). The interesting empirical finding is that the **schedule of which dataset feeds which stage** (T1 vs T2 vs T3) dominates the choice of projector or whether CT-ViT itself is fine-tuned.

## Claims & Evidence Analysis

| # | Claim | Evidence in paper | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | 3D-CT-GPT "significantly outperforms" existing methods in 3D chest-CT report generation. | Table 2 — direct comparison vs M3D on both Dataset-XY_val and CT-RATE_val; indirect comparison vs M3D literature numbers. | Dataset-XY_val (private, 188), CT-RATE_val (subset, 807). | ⭐⭐ |
| C2 | The CT-ViT + 3D average pooling + linear projector design is more efficient than RadFM / M3D-LaMed's spatial-pooling perceiver. | Conceptual diagram (Fig. 1); memory numbers (14 GB pre-train, 22 GB fine-tune) on a single RTX 3090. | – | ⭐⭐ — efficiency claim is supported by hardware footprint but no like-for-like FLOPs / latency table vs RadFM or M3D-LaMed. |
| C3 | Pre-training on public + fine-tuning on private (T1) is the best of three training strategies. | Table 2 "Training Strategies" rows: T1 > T2 > T3 across all metrics on Dataset-XY_val; Figure 3 qualitative example. | Dataset-XY_val only. | ⭐⭐ |
| C4 | The model "generalizes" across diverse datasets. | T2 model evaluated on CT-RATE_val and beats M3D there too. | CT-RATE_val (in-domain since CT-RATE is in T1's pre-train; T2 itself is not pre-trained on CT-RATE). | ⭐ — single cross-dataset test; no truly external test set; CT-RATE was used somewhere in the pipeline for most variants. |
| C5 | "Optimized temperature control mechanism" achieves a balanced trade-off between diversity and accuracy. | Figure 4 shows BLEU/ROUGE/METEOR vs T from 0.1–0.9. | Dataset-XY_val with T2 only. | ⭐ — "mechanism" is just standard sampling temperature; the optimum from the curves is T≈0.1, not the 0.7 used in the headline numbers. |
| C6 | CT2Rep's "overly complex architecture" makes it impractical and is therefore not compared. | No experimental evidence — CT2Rep is just excluded. | – | ⭐ — convenient exclusion; the strongest dedicated baseline is missing. |
| C7 | Computational efficiency / fits on a single 24 GB GPU. | Implementation Details: 14 GB pre-train, 22 GB fine-tune. | – | ⭐⭐⭐ — concrete and reproducible. |
| C8 | Fine-tuning the CT-ViT (vs frozen) measurably helps. | Ablation rows T2-Unfine vs T2 (BLEU 0.2950 → 0.3476). | Dataset-XY_val. | ⭐⭐ |

**Honest read.** The strongest claim — that 3D-CT-GPT outperforms 3D-CT report-generation baselines — is supported only against **one** baseline, M3D, partly via **literature-reported** numbers from a different evaluation pipeline. The dedicated 3D-CT report baseline in this niche, **CT2Rep**, is excluded with hand-waving justification, so "outperforms existing methods" should be read as "outperforms M3D in our evaluation setup." There is **no variance reporting** anywhere in Table 2 — important because the BLEU gaps between training variants (0.34 vs 0.38) sit well within the typical noise of single-run NLG metrics on a 188-sample test set. The "generalization" claim rests on a single cross-dataset evaluation where the **target dataset (CT-RATE_val) was the pre-training set for T1**, so it isn't out-of-distribution. The temperature "mechanism" is just standard sampling-temperature tuning, and the headline T=0.7 is *not* the optimum the curves reveal. Finally, **none of the experiments report clinical correctness** (no abnormality F1, no RadGraph entity/relation F1, no radiologist rating) — every number is an n-gram-overlap or BERTScore proxy, which is known to correlate poorly with clinical correctness in radiology reports.

## Method & Architecture

![3D-CT-GPT architecture: CT-ViT → 3D avg-pool → linear projector → Vicuna-7B](/assets/images/paper/3d-ct-gpt/fig_p003_02.png)
*Figure 2: Architecture of 3D-CT-GPT. (a) CT-ViT extracts a volumetric token grid $Z_x$ from the 240×480×480 CT input; (b) 3D average pooling (kernel=2) + reshape compress the grid into $P_v$ — 512 tokens of dim 512; (c) a single trainable linear projector $W$ maps $P_v$ into the LLM token space $M_v$; (d) image tokens are inserted between the system / instruction prompt embeddings and decoded by Vicuna-7B (LoRA-tuned in stage 2).*

**1) CT-ViT encoder (frozen).** Reused from CT-CLIP (Hamamci et al. 2024). The volumetric input $x \in \mathbb{R}^{240\times480\times480}$ is split into non-overlapping $15\times30\times30$ patches, each embedded into $D=512$. The output tensor $Z_x$ has shape $B \times T \times \frac{H}{2p_h} \times \frac{W}{2p_w} \times D$.

**2) Token-grid compression.** A permute → 3D average pool (kernel size 2) → reshape → permute pipeline (eqs. 1–5 in the paper) compresses $Z_x$ into $P_v$ of shape $B \times (T \cdot H' \cdot W') \times D$ — exactly **512 tokens of dim 512** per batch element. This deterministic pooler replaces the perceiver-style spatial pooler used by RadFM and M3D-LaMed.

**3) Linear projector.** A single trainable matrix $M_v = W \cdot P_v$ aligns the pooled CT tokens to Vicuna's word-embedding space — the LLaVA recipe. An ablation also tries a 2-layer MLP and finds it makes essentially no difference (see below).

**4) Prompt-level vision-language fusion.** The Vicuna tokenizer encodes the textual prompt; a special token id `-200` marks the image placeholder, which is replaced by the CT feature vectors. The full input becomes $M = \mathrm{concat}([M_{q_1}, M_v, M_{q_2}])$ and is fed to Vicuna-7B:

$$X_a = g\big(\mathrm{concat}([M_{q_1}, M_v, M_{q_2}])\big).$$

**5) Two-stage training on a custom VQA dataset.** Each entry is a (CT volume, question sampled from a fixed prompt pool such as *"What findings do you observe in this CT scan?"*, report-as-answer) triple with `<STOP>` separators in the prompt template.

- **Stage 1 — Pre-training:** freeze CT-ViT and Vicuna; train **only the projection layer**. LR = 1e-3, batch size 1, ~14 GB GPU memory, bfloat16.
- **Stage 2 — Instruction fine-tuning:** still freeze CT-ViT; train projector + Vicuna with **LoRA**. LR = 2e-4, batch size 1, ~22 GB GPU memory. Adam + cosine LR schedule. Single RTX 3090 (24 GB).

**6) Three training-strategy variants** (treated as a core experiment, not just an ablation):

- **T1**: pre-train 5 epochs on CT-RATE_train + LoRA-fine-tune 2 epochs on Dataset-XY_train (16 h).
- **T2**: simultaneously pre-train 5 epochs + fine-tune 2 epochs on **only** Dataset-XY_train (6 h).
- **T3**: pre-train + fine-tune **only** on CT-RATE_train (18 h).

**7) Decoding.** Sampling temperature is treated as a tunable knob. The reported headline numbers use $T=0.7$; the temperature sweep in §6 shows lower $T$ actually wins.

### Datasets

- **CT-RATE (public).** A subset of Hamamci et al. 2024's CT-RATE (25,692 non-contrast chest CT volumes, 21,304 unique patients, paired reports + multi-abnormality labels). The authors **subset 8,070 cases** as their working set, then split 0.8/0.1/0.1 → **6456 / 807 / 807**. Avg. report length ≈ 198 words. The selection criteria for the 8,070-of-25,692 subset are not disclosed.
- **Dataset-XY (private, in-house).** 2,000 chest-CT scans + reports from "a renowned international hospital." After de-identification, deduplication, manual review, and low-res filtering → **1,887 cases**, split 0.8/0.1/0.1 → **1508 / 190 / 188**. Patient ages 20–88 (mean 51.42), 44.7% F / 55.3% M. Avg. report length ≈ 88 words (less than half of CT-RATE).
- **Important: this paper does NOT use CTRG-Chest-548K.** Despite a common misreading, the working corpus is CT-RATE + Dataset-XY only.
- **Preprocessing.** HU windowing to $[-1000, +200]$; resampling to 0.75 mm × 0.75 mm × 1.5 mm; volumes cropped/padded to 240 × 480 × 480.

## Experimental Results

### Main quantitative comparison (Table 2)

| Setting | Model | Dataset | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore-F1 |
|---|---|---|---|---|---|---|---|---|
| Training strategies | **3D-CT-GPT (T1)** | **Dataset-XY_val** | **0.3836** | **0.4749** | **0.2191** | **0.3281** | **0.3565** | **0.8890** |
| Training strategies | 3D-CT-GPT (T2) | Dataset-XY_val | 0.3476 | 0.4446 | 0.1978 | 0.3092 | 0.3198 | 0.8862 |
| Training strategies | 3D-CT-GPT (T3) | Dataset-XY_val | 0.2323 | 0.3008 | 0.0706 | 0.1567 | 0.2509 | 0.8482 |
| Direct (private, unseen) | **3D-CT-GPT (T3)** | Dataset-XY_val | **0.2323** | **0.3008** | **0.0706** | **0.1567** | **0.2509** | **0.8482** |
| Direct (private, unseen) | M3D | Dataset-XY_val | 0.0869 | 0.1336 | 0.0227 | 0.1028 | 0.0710 | 0.8244 |
| Direct (public) | **3D-CT-GPT (T2)** | CT-RATE_val | **0.1327** | **0.2594** | **0.0586** | **0.1454** | **0.1403** | **0.8412** |
| Direct (public) | M3D | CT-RATE_val | 0.0299 | 0.1164 | 0.0223 | 0.0781 | 0.0549 | 0.8203 |
| Indirect (lit.) | M3D (Linear) | (lit.) | 0.1449 | 0.1925 | – | – | 0.1411 | 0.8832 |
| Indirect (lit.) | M3D (MLP) | (lit.) | 0.1515 | 0.1955 | – | – | 0.1438 | 0.8846 |
| Indirect (lit.) | **3D-CT-GPT (T1)** | Dataset-XY_val | **0.3836** | **0.4749** | **0.2191** | **0.3281** | **0.3565** | **0.8890** |
| Ablation (T2-based) | T2-Unfine (CT-ViT not finetuned) | Dataset-XY_val | 0.2950 | 0.4163 | 0.1830 | 0.2873 | 0.3037 | 0.8809 |
| Ablation (T2-based) | T2 (CT-ViT finetuned) | Dataset-XY_val | 0.3476 | 0.4446 | 0.1978 | 0.3092 | 0.3198 | 0.8862 |
| Ablation (T2-based) | T2-Linear (linear vs MLP projector) | Dataset-XY_val | 0.3418 | 0.4467 | 0.1992 | 0.3067 | 0.3338 | 0.8850 |

A few things worth flagging up front:

- The only **direct** head-to-head baseline is M3D. **CT2Rep is excluded** (no released pre-trained weights / inference code) and **RadFM is excluded** by deference to M3D's prior result that RadFM < M3D on 3D report metrics. CT2Rep — the strongest dedicated 3D-CT report-generation baseline — never sees an evaluation here.
- The "cross-dataset generalization" check uses **CT-RATE_val**, but T1 is **pre-trained on CT-RATE_train**. T2 (which is the variant evaluated on CT-RATE_val in the table) is not pre-trained on CT-RATE, but the broader generalization narrative is still propped up by a setting where the target distribution overlaps with the model's pre-training corpus.
- All metrics are NLG-overlap-style (BLEU/ROUGE/METEOR) plus BERTScore — there is **no clinical-correctness metric** (no abnormality-label F1, no RadGraph entity/relation F1, no radiologist evaluation), and these proxies are well-known to overstate report quality in radiology.

### Ablations and qualitative findings

![Sampling-temperature sweep on Dataset-XY_val (T2 model)](/assets/images/paper/3d-ct-gpt/fig_p007_01.png)
*Figure 3: Effect of decoding temperature (0.1–0.9) on BLEU / ROUGE / METEOR for the T2 model on Dataset-XY_val. BLEU and METEOR degrade roughly monotonically with temperature; ROUGE-L is essentially flat. The headline T=0.7 is not the optimum.*

- **Fine-tuning the CT-ViT** in stage 2 (LoRA + projector unfrozen, no LoRA adapters on CT-ViT) raises BLEU from **0.2950 → 0.3476** (+5.3 pp) over the no-fine-tune baseline.
- **Linear projector vs 2-layer MLP** is essentially a wash: linear yields BLEU 0.3418 vs MLP 0.3476, but linear actually wins on METEOR (0.3338 vs 0.3198) — supporting the choice of the simpler module.
- **Temperature sweep** (Figure 3, $T \in [0.1, 0.9]$): BLEU degrades roughly monotonically with temperature (≈0.38 → ≈0.34); METEOR similar; ROUGE-L roughly flat. The "optimal temperature control" claim is mild — small effect size (~3–4 BLEU pts), and the comparison default of 0.7 is *not* the best setting (T=0.1 wins on BLEU/METEOR).

![Representative chest-CT axial slice from the qualitative comparison](/assets/images/paper/3d-ct-gpt/fig_p007_02.png)
*Figure 4: A representative axial chest-CT slice from the qualitative comparison in Figure 3 of the paper (lower-thoracic level). T1's report most closely matches GT phrasing; T3 — trained only on CT-RATE — clearly drifts in style and adds CT-RATE-flavored content not in the private GT (e.g., "ground-glass densities", "compatible with the infectious process"), illustrating the cross-distribution style transfer effect.*

The T3 qualitative example is particularly worth singling out: the model is *plausibly* hallucinating findings that match the **stylistic distribution of its training reports** rather than the imaging evidence in the test scan. The paper does not analyze hallucination behavior at all — a notable gap given that the result is right there in Figure 3.

## Limitations

**Authors acknowledge (explicit):**

- Data scarcity for paired 3D-CT + report.
- Could not perform large-scale alignment training in stage 1.
- Computational constraints forced LoRA rather than full fine-tuning.
- Dataset-XY is small (1,887 cases).

**Not addressed (this audit's additions):**

- **No clinical-correctness evaluation.** No abnormality F1, no RadGraph entity/relation F1, no radiologist rating. NLG metrics for radiology reports are widely known to overstate quality.
- **CT2Rep is the obvious head-to-head baseline and is missing.** The justification ("partially open-source", "complex architecture") is weak.
- **No statistical-significance testing or multi-seed variance** — every Table 2 number is a single-run point estimate on a 188-sample test set.
- **CT-RATE subset selection (8,070 of 25,692) is unexplained.** Could bias both pre-training and the cross-dataset comparison.
- **Dataset-XY is single-institution, not externally validated.** Real generalization to other hospitals or scanners is untested.
- **The cross-dataset generalization test uses CT-RATE_val**, which T1 was pre-trained on — so the "generalizes across diverse datasets" claim sits on a setting where the model's pre-training corpus and its evaluation corpus overlap.
- **"Preserves spatial info" of pooled tokens is asserted, not measured.** No visualization or ablation comparing pooled vs unpooled CT-ViT tokens for downstream report quality.
- **Hallucination behavior** — the qualitative T3 example clearly hallucinates findings that aren't in the GT. Not analyzed.
- **No 2D-stacking baseline** beyond M3D itself.
- **Vicuna-7B is the only LLM evaluated.** No ablation across LLM sizes or families.

## Why It Matters for Medical AI

For practitioners, the takeaway is less "3D-CT-GPT is the new SOTA" — the evidence does not support that claim — and more that the **recipe** is plausible: a frozen CT-ViT + 3D average pooling + LLaVA-style linear projector + LoRA-tuned Vicuna-7B is a **single-RTX-3090 path** into 3D-CT report generation. That matters because the dominant alternative narrative in 3D medical VLMs has been "scale up the perceiver / scale up the encoder," which excludes most academic groups. The pre-train-on-public + fine-tune-on-private (T1) schedule is also the right default for any group that has a small in-house corpus and wants to use CT-RATE as a stepping stone.

The honest counterweight: the evaluation gap between this paper and a clinically credible result is large. Anyone deploying this style of system should plan to add (i) abnormality-F1 / RadGraph evaluation, (ii) a CT2Rep head-to-head, (iii) multi-seed variance, and (iv) at least one out-of-institution test set before drawing conclusions about clinical utility. Without those, the BLEU gaps in Table 2 are suggestive at best.

## References

- Paper: [3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models (arXiv:2409.19330)](https://arxiv.org/abs/2409.19330)
- CT-ViT / CT-CLIP backbone: Hamamci et al., *CT-CLIP: A foundation model for 3D chest CT volumes with contrastive language-image pre-training*, arXiv:2403.17834 (2024) — also the source of CT-RATE.
- M3D-LaMed (the only direct baseline): Bai et al., *M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models*, 2024.
- RadFM (excluded baseline): Wu et al., *Towards Generalist Foundation Model for Radiology*, 2023.
- CT2Rep (excluded baseline): Hamamci et al., *CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging*, 2024.
- LLaVA (projection-layer recipe): Liu et al., *Visual Instruction Tuning*, NeurIPS 2023.
- Vicuna-7B: LMSYS, *Vicuna: An Open-Source Chatbot Impressing GPT-4*, 2023.
