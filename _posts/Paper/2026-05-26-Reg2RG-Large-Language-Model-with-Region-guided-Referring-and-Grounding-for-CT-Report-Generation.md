---
title: "Reg2RG: Large Language Model with Region-guided Referring and Grounding for CT Report Generation"
excerpt: "Region-guided 3D CT report generation with SAT-Pro masks, decoupled texture/geometry streams, and a Region-Report Alignment training recipe — CE F1 0.253 on RadGenome-ChestCT vs M3D 0.148."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/reg2rg/
tags:
  - Reg2RG
  - 3D-CT-Report-Generation
  - Region-guided-MLLM
  - Local-Feature-Decoupling
  - Region-Report-Alignment
  - SAT-Pro
  - LLaMA2
  - LoRA
  - RadGenome-ChestCT
  - CTRG-Chest
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- Reg2RG feeds a LLaMA2-7B (LoRA) decoder a global 3D volume embedding plus a set of per-region embeddings produced by routing **SAT-Pro** segmentation masks through two decoupled streams — a cropped high-resolution **texture** stream and an uncropped **geometry** stream — so the model emits region-prefixed findings instead of one undifferentiated chest report.
- The training-time trick is **Region-Report Alignment (RRA)**: at every step the local-feature order is shuffled and the ground truth is reformatted as `"The region [i] is [area]. <region report>"`, turning anatomy recognition into an auxiliary auto-regressive target. The two interventions are bundled and never separately ablated.
- **Headline result on RadGenome-ChestCT (LLaMA2-7B + LoRA):** **BLEU-1 47.25, METEOR 44.07, ROUGE-L 36.65, CE F1 0.253** — a +0.041 absolute (≈ +19% relative) F1 jump over MedVInT (0.212) and +0.105 over M3D (0.148) on the same RadBERT/CT-CLIP label extractor. NLG metrics also sweep 6/6 on RadGenome and 5/6 on CTRG-Chest.

## Motivation

Existing 3D chest-CT report generators (CT2Rep, RadFM, M3D, Dia-LLaMA, HILT) collapse the volume into one global embedding and lose regional abnormality cues. The introduction's Fig. 1 shows the symptom plainly: the vanilla global-feature model misses an emphysematous nodule that the region-guided model catches in the same volume.

![Vanilla global-feature method misses regional abnormalities](/assets/images/paper/reg2rg/fig_p001_01.png)
*Figure 1a: Vanilla global-feature method misses regional abnormalities (gray = wrong diagnoses) — the motivating failure mode.*

![Reg2RG's region-guided generation captures the same volume's findings](/assets/images/paper/reg2rg/fig_p001_02.png)
*Figure 1b: Reg2RG's region-guided generation correctly identifies the regional abnormalities the global-only baseline missed.*

2D chest X-ray work (RGRG) already showed region-aware generation helps; extending it to 3D CT is harder because (i) naive cropped-region encoding throws away the geometric size/position cues a radiologist needs for "enlarged heart" or lesion localization, and (ii) general-domain region-aware MLLMs (Groma, GPT4RoI, RegionGPT, Ferret) were not designed around medical anatomy where regions are spatially fixed and interrelated. MedRegA's bbox-prompt approach is cited as the closest medical analogue but treated as less precise than feature-level region encoding. Reg2RG positions itself as the **first** "referring (describe a given region) + grounding (link descriptions back to regions)" framework for 3D CT report generation.

## Core Innovation

- **Local Feature Decoupling (LFD).** Each anatomical region is encoded by two parallel streams: a *texture* stream that masks then **crops** the volume around the region and runs a 3D ViT + Perceiver (inherited from RadFM) at higher effective resolution, and a *geometry* stream that runs a lightweight 3-layer 3D ViT on the **uncropped mask** itself to recover size/position information that cropping discarded. The two are concatenated per region.
- **Region-Report Alignment (RRA).** A training recipe that bundles two interventions: (a) every step the per-region embedding order is *shuffled* before being inserted into the prompt, removing positional shortcuts; (b) the ground truth is rewritten so each region's findings are preceded by an `"The region [i] is [area name]"` prefix that the LLM must auto-regressively predict before emitting that region's report. Prefixes are stripped at evaluation.
- **Global + local fusion.** Reg2RG keeps a single 32-token global embedding alongside the per-region tokens. Ablations show globals lift recall but hurt precision — RRA exists in part to compensate.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | First region-guided referring + grounding framework for 3D CT report generation | Novelty positioning; closest prior 2D work (RGRG) generates per-region independently, MedRegA uses bbox prompts not feature-level region tokens | n/a | ⭐⭐ |
| C2 | Outperforms SOTA on NLG metrics | Table I — sweeps 6/6 on RadGenome-ChestCT, 5/6 on CTRG-Chest (loses only ROUGE-L on CTRG) | 2 datasets | ⭐⭐⭐ |
| C3 | Outperforms SOTA on clinical efficacy (CE) | Table II — F1 0.253 vs MedVInT 0.212, RadFM 0.195, M3D 0.148; precision 0.423 vs 0.377 | RadGenome only | ⭐⭐ |
| C4 | LFD preserves texture detail at low compute | Table V CE F1 +0.053 with LFD; "minimal compute" claim **never quantified** (no FLOPs, latency, or memory table) | RadGenome | ⭐⭐ |
| C5 | Geometry features are essential | Table VI — texture-only 0.141 F1 < texture+geometry 0.210 (direct isolation) | RadGenome | ⭐⭐⭐ |
| C6 | Global-local collaboration captures inter-regional context | Table VI — adding global features 0.210 → 0.239 F1; recall 0.143 → 0.176 | RadGenome | ⭐⭐ |
| C7 | RRA improves diagnostic accuracy *and* interpretability | Table VII — F1 +0.014, BLEU-1 +1.04, METEOR +1.13 (ROUGE-L −0.36); "interpretability" claim has no human study and shuffle vs prefix are never ablated apart | RadGenome | ⭐⭐ |
| C8 | The model "grounds" reports to the correct regions | Table III — 8/10 regions ≥ 0.96 F1 on anatomy-prefix prediction; **no IoU, Dice, bbox, or pointing-game evaluation** anywhere | RadGenome | ⭐ |
| C9 | Large LLMs are essential for region-level referring | Table VIII — LLaMA2-7B beats GPT-2 on every metric (CE F1 +0.024, BLEU-4 +10.15) | RadGenome | ⭐⭐ |
| C10 | Improvements generalize to CTRG-Chest | Table I — 5/6 NLG metrics; but CTRG region-level GT was **LLM-synthesized by Qwen2.5-14B**, so evaluation is partially LLM-vs-LLM agreement | CTRG-Chest | ⭐⭐ |

**Honest read — the load-bearing weak spots.**

1. **No human GT masks anywhere in the pipeline.** Both train and test use SAT-Pro *predicted* masks. RadGenome-ChestCT's "ground-truth" masks were themselves SAT-generated, and CTRG masks are also produced by SAT (visibly coarser, per Fig. 6). There is no radiologist-annotated mask anywhere — the framework's claimed robustness to segmentation errors is deferred to Sec. V "future work".
2. **Region-recognition is anatomy-name prefix prediction, not spatial grounding.** Table III's per-region F1 ≥ 0.96 for 8/10 regions measures only whether the LLM can name back the anatomy of a region token it was *given*. Lung and pleura collapse to **0.443 / 0.442 F1** because SAT-Pro itself confuses them — the upstream silently caps everything downstream.
3. **RRA bundles two interventions and ablates them only jointly (Table VII, +0.014 F1, within the noise band for single-seed LLM fine-tuning).** The paper cannot distinguish whether the gain comes from shuffle preventing positional shortcuts, the auxiliary anatomy-classification target acting as a regularizer, or the prefix simply giving the LLM more context. The load-bearing mechanism is unidentified.
4. **"Grounding" reduces to categorical anatomy-label recognition on input region tokens.** Regions are fed in spatially (via SAT masks), but the output never re-emits spatial information beyond the categorical region label. No IoU, no Dice, no bbox, no pointing game on lesions — the biggest gap between title and experiments.
5. **No benchmark vs. MedRegion-CT (Kyung et al. 2025), the most directly comparable concurrent system.** MedRegion-CT also uses SAT pseudo-masks on RadGenome-ChestCT, but with a 2D-encoded R² token-pooling architecture, mask tokens, and a deterministic attribute extractor. Numbers (CA-F1 0.450 vs CE F1 0.253) are not directly comparable across papers' label-extractor pipelines, and neither paper benchmarks against the other — apples-to-apples remains an open question.
6. **No prompt-only baseline.** The instruction prompt is elaborate ("identify the corresponding anatomical areas for each region and then generate the respective report"). Because the encoder weights are inherited from RadFM (CE F1 0.195) and Reg2RG hits 0.253, plausibly ~30% of the relative gain is from the prompt + RRA recipe rather than the LFD/global-local architecture — but the paper never runs that ablation.
7. **Single-seed, single-checkpoint evaluation.** No variance bars, no significance tests, no multi-seed runs. The +0.014 RRA gain in Table VII is within the noise band typical of LLM fine-tuning.

## Method & Architecture

![Reg2RG full architecture page render](/assets/images/paper/reg2rg/page_004.png)
*Figure 2: Reg2RG architecture — SAT-Pro masks define n anatomical regions; each region is encoded by a texture stream (mask × volume, then crop, then RadFM 3D ViT + Perceiver → 32 tokens) and a geometry stream (uncropped mask through a lightweight 3-layer 3D ViT → 1 pooled token); the global volume goes through the same RadFM weights to 32 tokens; all are inserted into the LLaMA2-7B prompt with region order shuffled per step.*

The full generation is

$$R = \mathrm{LLM}(G, L_1, \dots, L_n)$$

where $G$ is the global embedding and $L_i$ are the per-region decoupled embeddings.

**1. Universal segmentation.** SAT-Pro (256 object queries, text-prompted) produces $\{M_{A_1}, \dots, M_{A_n}\}$ for the 10 chest anatomical regions on RadGenome-ChestCT (abdomen, bones, breasts, esophagus, heart, lungs, trachea & bronchi, mediastinum, pleura, thyroid). Same SAT module is used on CTRG-Chest. Intensities min-max normalized to [0,1]; both global and per-region volumes resized to 256 × 256 × 64.

**2. Texture stream.** For region $A_j$, element-wise multiply mask × volume, crop tight bbox around the region, encode with RadFM's 3D ViT + Perceiver adapter to 32 visual embeddings:

$$L_{A_j}^t = f_A(f_V(\mathrm{Crop}(M_{A_j} \odot V)))$$

Cropping preserves higher effective resolution without inflating input size.

**3. Geometry stream.** Encode the *uncropped* mask $M_{A_j}$ with a lightweight 3-layer 3D ViT, pool to one token, project:

$$L_{A_j}^g = f_P(f_M(M_{A_j}))$$

This recovers size/position that cropping discarded.

**4. Per-region concat:** $L_{A_j} = \mathrm{Concat}(L_{A_j}^t, L_{A_j}^g)$.

**5. Global features.** The full resized volume $V$ goes through the same RadFM ViT3D + Perceiver weights (shared with texture stream) to 32 global tokens: $G = f_A(f_V(V))$.

**6. Prompt assembly.** Special tokens `<image>` (replaced by $G$) and `<region 1>` … `<region n>` (replaced by the $L_i$). Instruction: *"The global visual information is as follows: <image>. The identified regions of interest are: region 1 <region 1>, …, region n <region n>. Please identify the corresponding anatomical areas for each region and then generate the respective report for each anatomical region."*

**7. Region-Report Alignment training (RRA).** Two bundled interventions:
   - (a) **Shuffle** the order of $L_{A_1}, \dots, L_{A_n}$ at every step before insertion (no fixed positional convention).
   - (b) **Prefixed ground truth**: each region's findings are preceded by `"The region [i] is [area name]"` and the LLM auto-regressively predicts that prefix before the report. Prefixes are stripped before scoring.

Loss is standard next-token cross-entropy.

**8. LLM backbone.** LLaMA2-7B fine-tuned with LoRA (rank 8, α 32, dropout 0.1). AdamW, lr 5e-5, constant schedule with warmup, ZeRO-2 + gradient checkpointing. Training: 6 epochs / 48h on 2× RTX 4090 (RadGenome-ChestCT), 10 epochs / 24h (CTRG-Chest). Effective batch 16. Last-epoch checkpoint used.

## Experimental Results

**Datasets.**
- **RadGenome-ChestCT** (on CT-RATE): 25,692 region-guided CT–report pairs, 21,304 patients, 10 chest regions, SAT-generated masks, GPT-4 + NER-distilled region-grounded reports. Official 24,128 / 1,564 train/test split. Voxel spacing 1×1×3 mm. Non-contrast only.
- **CTRG-Chest**: 1,804 CT–report pairs. No region-level GT — the authors use **Qwen2.5-14B to split full reports into region sections at training time**. Random 8:2 split (no patient-level grouping described).

**Main NLG comparison (Table I, verbatim):**

| Dataset | Method | BL-1 | BL-2 | BL-3 | BL-4 | MTR | RG-L |
|---|---|---|---|---|---|---|---|
| RadGenome-ChestCT | R2GenGPT (2023) | 43.28 | 34.11 | 28.16 | 24.16 | 39.85 | 32.26 |
| RadGenome-ChestCT | MedVInT (2023) | 44.28 | 34.91 | 28.75 | 24.60 | 40.39 | 32.58 |
| RadGenome-ChestCT | RadFM (2023) | 44.20 | 34.49 | 28.06 | 23.65 | 39.94 | 31.53 |
| RadGenome-ChestCT | CT2Rep (2024) | 44.42 | 34.43 | 27.94 | 23.56 | 40.16 | 30.99 |
| RadGenome-ChestCT | M3D (2024) | 43.57 | 34.48 | 28.54 | 24.49 | 39.95 | 32.61 |
| RadGenome-ChestCT | **Reg2RG (ours)** | **47.25** | **36.49** | **29.57** | **24.87** | **44.07** | **36.65** |
| CTRG-Chest | R2GenGPT | 41.82 | 36.37 | 32.70 | 30.10 | 47.05 | 50.93 |
| CTRG-Chest | MedVInT | 47.38 | 39.60 | 34.28 | 30.68 | 49.32 | 49.53 |
| CTRG-Chest | RadFM | 48.66 | 40.28 | 34.73 | 30.89 | 49.18 | 49.08 |
| CTRG-Chest | CT2Rep | 42.28 | 36.16 | 32.08 | 29.19 | 47.00 | 50.17 |
| CTRG-Chest | M3D | 46.27 | 39.02 | 34.23 | 30.86 | 49.26 | 50.24 |
| CTRG-Chest | **Reg2RG (ours)** | **49.63** | **41.43** | **35.91** | **32.04** | **49.71** | 47.76 |

Reg2RG sweeps every NLG metric on RadGenome-ChestCT and every metric *except ROUGE-L* on CTRG-Chest; the paper attributes the ROUGE-L gap to fragmented region-level outputs hurting longest-common-subsequence scoring.

**Clinical efficacy (Table II, RadGenome-ChestCT only — RadBERT label extractor on 18 abnormalities):**

| Method | Pre. | Rec. | F1 |
|---|---|---|---|
| R2GenGPT | 0.340 | 0.066 | 0.110 |
| MedVInT | 0.377 | 0.148 | 0.212 |
| RadFM | 0.382 | 0.131 | 0.195 |
| CT2Rep | 0.317 | 0.089 | 0.139 |
| M3D | 0.407 | 0.090 | 0.148 |
| **Reg2RG (ours)** | **0.423** | **0.181** | **0.253** |

**Region recognition (Table III).** Anatomy-prefix prediction F1 ≥ 0.96 for abdomen, bone, breast, esophagus, heart, mediastinum, thyroid, trachea & bronchi. **Lung F1 = 0.443, pleura F1 = 0.442** — SAT-Pro confuses the two; this is the visible failure mode of the entire upstream dependency.

**Per-region CE F1 (Table IV).** Highly uneven: abdomen 0.377, mediastinum 0.337, breast 0.190, lung 0.191, heart 0.190, pleura 0.115, esophagus 0.039, thyroid 0.061, **bone 0.000, trachea & bronchi 0.000**. The two zeros are because RadBERT (CT-RATE's 18-class chest-abnormality extractor) has no labels for bone or tracheal findings — the metric is silent there, not the model.

**Ablations.**

| Ablation | Variant | CE F1 | BLEU-4 | ROUGE-L | Comment |
|---|---|---|---|---|---|
| Table V (LFD) | w/o LFD | 0.200 | 25.69 | 37.64 | LFD removed |
| Table V (LFD) | **w/ LFD** | **0.253** | 24.87 | 36.65 | LFD adds +0.053 F1; small NLG regressions on BLEU-4/ROUGE-L |
| Table VI (feature mix) | texture only | 0.195 | — | — | RadFM-style baseline |
| Table VI (feature mix) | geometry only | 0.141 | — | — | Worse than texture alone |
| Table VI (feature mix) | texture + geometry | 0.210 | — | — | Geometry adds +0.015 over texture |
| Table VI (feature mix) | texture + geometry + global | 0.239 | — | — | Globals lift recall but **drop precision** 0.394 → 0.372 |
| Table VII (RRA on top of above) | w/o RRA | 0.239 | 46.21 | 37.01 | — |
| Table VII (RRA on top of above) | **w/ RRA** | **0.253** | **47.25** | 36.65 | +0.014 F1, +1.04 BLEU-1; reverses precision drop 0.372 → 0.423 |
| Table VIII (LLM) | GPT-2 | 0.229 | 14.72 | — | — |
| Table VIII (LLM) | **LLaMA2-7B** | **0.253** | **24.87** | — | Big BLEU-4 jump, smaller +0.024 CE F1 — complicates "large LLMs essential" framing |

**Qualitative.**

![Report-length distribution (KDE) — Reg2RG vs MedVInT vs GT](/assets/images/paper/reg2rg/fig_p009_01.png)
*Figure 3: Reg2RG's report-length distribution is closer to ground truth than MedVInT's (lower KL divergence).*

![Case-1 CT slice with regional color overlay](/assets/images/paper/reg2rg/fig_p010_01.png)
*Figure 4a: Case 1 — ground-truth slice with regional color overlay (input view).*

![Case-1 MedVInT report panel](/assets/images/paper/reg2rg/fig_p010_02.png)
*Figure 4b: Case 1 — MedVInT generated report (gray = misdiagnosis).*

![Case-1 Reg2RG report panel](/assets/images/paper/reg2rg/fig_p010_03.png)
*Figure 4c: Case 1 — Reg2RG generated report with region prefixes.*

![Case-2 CT slice with regional color overlay](/assets/images/paper/reg2rg/fig_p010_04.png)
*Figure 4d: Case 2 — ground-truth slice with regional overlay.*

![Case-2 MedVInT and Reg2RG comparison](/assets/images/paper/reg2rg/fig_p010_05.png)
*Figure 4e: Case 2 — MedVInT vs Reg2RG side-by-side, showing fewer regional misses for Reg2RG.*

![Case-2 Reg2RG full per-region report](/assets/images/paper/reg2rg/fig_p010_06.png)
*Figure 4f: Case 2 — Reg2RG full per-region report.*

![SAT-Pro region masks on RadGenome-ChestCT](/assets/images/paper/reg2rg/fig_p010_07.png)
*Figure 5a: Region masks generated by SAT-Pro on RadGenome-ChestCT — the upstream that the whole framework rides on.*

![Region-level reports with anatomy-prefix grounding (1/2)](/assets/images/paper/reg2rg/fig_p010_08.png)
*Figure 5b: Region-level reports with anatomy-prefix grounding — left half of the full 10-region output.*

![Region-level reports with anatomy-prefix grounding (2/2)](/assets/images/paper/reg2rg/fig_p010_09.png)
*Figure 5c: Region-level reports with anatomy-prefix grounding — right half of the full 10-region output.*

![SAT segmentation quality: RadGenome-ChestCT vs CTRG-Chest](/assets/images/paper/reg2rg/fig_p011_03.png)
*Figure 6: SAT segmentation quality — RadGenome-ChestCT masks are crisp; CTRG-Chest masks are visibly coarser, which is the paper's own explanation for smaller relative gains on CTRG.*

## Limitations

**Authors acknowledge.**
- Performance hinges on SAT segmentation quality; failures cascade silently (the lung/pleura collapse in Table III is the visible symptom).
- Local features are organ-level only — no lesion-level features; "future work" promises lesion segmentation/detection integration.
- Only chest CT, only non-contrast (inherits CT-RATE's restriction); no external evaluation. End-to-end segmentation training is "future work".

**Not addressed in the paper.**
- **No spatial grounding evaluation.** No IoU, Dice, bbox, or pointing-game numbers despite the "grounding" framing of the title. The output never re-emits spatial coordinates beyond a categorical anatomy label.
- **No GT mask anywhere.** Train/test both use SAT-Pro predictions; RadGenome's "GT" masks are themselves SAT-generated. There is no human-annotated mask in the whole pipeline.
- **RRA mechanism unidentified.** Shuffle and prefix are bundled and ablated only jointly (+0.014 F1, single-seed). The paper cannot tell which intervention does the work — or whether either does, beyond noise.
- **No prompt-only baseline.** Given the elaborate instruction, ~30% of the lift over the RadFM baseline (which shares encoder weights) plausibly comes from the prompt + RRA recipe rather than the LFD architecture, but this is never isolated.
- **No FLOPs/latency/memory table.** "Minimal computational overhead" of LFD is asserted, never measured.
- **No variance / multi-seed reporting.** Small ablation deltas (Table V LFD NLG, Table VII RRA) are not contextualized against run-to-run noise.
- **No comparison to MedRegion-CT, MAIRA-2, MedRegA, or MiniGPT-Med on a common split.** MedRegion-CT in particular is conceptually similar (SAT-mask-based region tokens on the same dataset) and architecturally different (per-slice 2D Rad-DINO + R² token pooling + deterministic attribute extractor) — neither paper benchmarks against the other.
- **CTRG-Chest evaluation is partially LLM-vs-LLM.** Region-level GT was synthesized by Qwen2.5-14B; the test set is built by random rather than patient-level split.
- **Single LLM backbone (LLaMA2-7B + LoRA)** apart from the GPT-2 ablation; no modern decoder (LLaMA-3.x, Qwen-2.5) tested.
- **Fixed 10-region taxonomy** defined by SAT; extending to finer-grained or lesion-level "regions" is untested.
- **Bone and trachea CE F1 = 0** because RadBERT has no labels there — the clinical-efficacy metric is structurally silent on 2/10 regions and the paper does not propose a workaround.

## Why It Matters for Medical AI

Reg2RG is one of the cleaner demonstrations that **3D radiology report quality is bottlenecked by region-conditioning, not by sheer encoder scale** — the same RadFM ViT3D + Perceiver weights jump from CE F1 0.195 (texture-only baseline) to 0.253 once region tokens and the RRA recipe are added. The texture/geometry decoupling is a reusable template for any task where cropping for resolution would otherwise discard scale and position. But the "grounding" claim in the title is the part the experiments do not actually deliver — what the paper proves is *region-conditioned referring with anatomy-name recognition*, not lesion- or coordinate-level grounding. Combined with the all-SAT-no-human-mask pipeline, this work should be read as a strong recipe for **region-aware report generation** while the title's grounding promise stays open for future, geometry-aware successors.

## References

- Paper: Chen, Bie, Jin, Chen. *Large Language Model with Region-guided Referring and Grounding for CT Report Generation*. arXiv:2411.15539v2 (IEEE TMI preprint formatting), 5 May 2025 (v1 Nov 2024).
- Code: [github.com/zhi-xuan-chen/Reg2RG](https://github.com/zhi-xuan-chen/Reg2RG)
- Segmentation backbone: SAT-Pro — Zhao et al., *Segment Anything in 3D Medical Images with Text Prompts*, 2023.
- Dataset (large): RadGenome-ChestCT — Zhang et al., 2024 (built on CT-RATE — Hamamci et al., 2024).
- Dataset (small): CTRG-Chest (region-level GT here synthesized via Qwen2.5-14B).
- Visual encoder pretraining: RadFM (3D ViT + Perceiver).
- LLM backbone: LLaMA2-7B fine-tuned with LoRA (rank 8, α 32).
- Most directly comparable concurrent work (cited in audit, never benchmarked against): MedRegion-CT (Kyung et al., 2025).
- Related region-aware report-generation lineage: RGRG (2D CXR), MedRegA, Groma, GPT4RoI, RegionGPT, Ferret, MAIRA-2, MiniGPT-Med.
