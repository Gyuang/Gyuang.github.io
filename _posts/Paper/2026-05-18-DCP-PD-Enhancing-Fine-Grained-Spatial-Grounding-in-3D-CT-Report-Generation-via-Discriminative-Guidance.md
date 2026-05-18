---
title: "DCP-PD: Enhancing Fine-Grained Spatial Grounding in 3D CT Report Generation via Discriminative Guidance"
excerpt: "Templating classifier outputs as a natural-language prefix and dropping them at random lifts CT-RATE macro-F1 from 0.440 to 0.603 — but a cue-only LLM with no image tokens already reaches 0.600, so most of the headline gain is classifier leakage rather than visual grounding."
categories: [Paper, CT-Report-Generation]
permalink: /paper/dcp-pd/
tags:
  - DCP-PD
  - 3D CT
  - Radiology Report Generation
  - Vision-Language Models
  - Prompt Dropout
  - Spatial Grounding
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- DCP-PD decouples a 3D-CT report generator from any specific auxiliary model by templating discriminator outputs (Discriminative Cue Prompts) into a natural-language prefix, then *intentionally drops cues at random during training* to stop the LLM from short-cutting through the prompt.
- The prompt-dropout mechanism is the load-bearing trick: with ground-truth cues but no dropout, removing the prompt at test time collapses F1 to **0.094**; with p=0.3 dropout, the same No-DCP test setting recovers to F1 0.366.
- Headline numbers on CT-RATE: macro-F1 **0.440 (Base VLM) → 0.530 (DCP-1) → 0.603 (DCP-2)**, ~37% relative over their own baseline. On out-of-distribution Rad-ChestCT, **F1 0.266 → 0.503** vs BTB3D-16 — but only +0.108 over their own Base VLM (0.395).

## Motivation

3D CT VLMs are typically trained with whole-volume / full-report contrastive or generative alignment, which is too coarse to pin *where* a lesion is, and they are evaluated with lexical overlap or holistic LLM-judge scores that hide spatial errors. Prior fixes either (a) tie the generator to a specific anatomical segmenter (Reg2RG, MedRegion-CT, fVLM) and require costly retraining whenever the auxiliary model changes, or (b) inject classifier predictions as text prompts (PromptMRG, Dia-LLaMA) but constrain the cue space to predefined presence/absence labels and over-fit to the classifier head.

DCP-PD keeps the prompt-injection interface, but tries to make it (i) **modular** — the discriminator is hot-swappable at inference — and (ii) **location-aware**, via a hierarchical question protocol (presence → laterality → lobe) that exposes whether the generated report actually grounds pathology to the correct sub-region.

## Core Innovation

Three pieces together:

1. **Cue templating as a natural-language prefix.** Classifier positives are converted to a sentence like `"The Image contains Lung Nodule, Lung Opacity, ... Please generate comprehensive CT report."` so the same templating interface accepts presence, laterality, or lobe-level cues. The VLM never sees a structured slot — only natural language.
2. **Prompt dropout (p=0.3).** During discriminative-guided SFT, each ground-truth cue entity is independently dropped before templating. There is *no auxiliary loss* — dropout is the entire regularizer that keeps the model from learning a "copy the prompt" shortcut.
3. **Hot-swappable discriminator at inference.** Because the VLM was trained on noisy / partially-dropped cues, swapping in a stronger external classifier at test time (DCP-2) yields strictly better generations without retraining the VLM.

## Claims & Evidence Analysis

| # | Claim | Evidence | Rating |
|---|---|---|---|
| C1 | DCP-PD improves macro F1 ~20% relative on CT-RATE | Table 4: 0.603 vs 0.501 (CT-AGRG w/ CT-Net) | ⭐⭐⭐ best in table, but the "20% relative" silently switches the baseline — the comparison is against a *different* system, not their own Base VLM (0.440 → 0.603 is the like-for-like number, ~37%). |
| C2 | DCP-PD lifts OOD F1 from 0.266 → 0.503 on Rad-ChestCT (~89% relative) | Table 5 vs BTB3D-16 | ⭐⭐ headline-optimized; their own structured Base VLM already hits 0.395, so the delta attributable to DCP-PD itself is only **+0.108 absolute**, not +0.237. Single OOD set, same modality / anatomy / language. |
| C3 | Prompt dropout prevents shortcut learning | Table 2: No-DCP F1 0.094 → 0.366; Table 3: attention reliance S_text 0.441 → 0.316; Table 7 dropout-rate sweep | ⭐⭐⭐ cleanly controlled, large effect, replicated across rates and corroborated by attention scores. The strongest experiment in the paper. |
| C4 | Plug-and-play discriminator — swap without retraining the VLM | DCP-1 (their probe) and DCP-2 (stronger external) both help (Table 1) | ⭐⭐ two discriminators is the minimum demo of swappability; never tested across architectures or differing label spaces. |
| C5 | OOD generalization | Rad-ChestCT (Table 5) only | ⭐⭐ one OOD dataset, same modality / anatomy / English reports; no significance reporting. |
| C6 | Scales to richer cue types (laterality, lobe) | Figs. 5, 6: DCP-PD beats Base VLM on 22/23 entries | ⭐⭐ consistent improvements, but absolute lobar-F1 still 0.1–0.5 and one regression (Cons-RML) — bounded by the underlying classifier. |
| C7 | Fine-grained spatial localization remains challenging even for SOTA | Figs. 5, 6 + Fig. 7 + Table 8 ablation | ⭐⭐⭐ refreshingly honest negative result — image-token ablation directly demonstrates current coarse benchmarks are nearly saturated by cue-only LLMs. |

**Honest read — four things to flag before quoting the numbers.**

1. **The F1 jump is almost entirely recall (0.423 → 0.700) and a chunk of it is classifier leakage.** The cue prompt literally hands the answer key to the LLM. Table 8 / Fig. 7 shows a cue-only LLM with **no image tokens at all** still reaches CT-RATE F1 = **0.600** vs the full model's 0.603. The image tokens only meaningfully help on lobar-level grounding (Fig. 7, +0.025 to +0.233 ΔF1 across 15 finding×lobe entries). On the headline CT-RATE F1, "DCP-PD" is mostly measuring "how good is the linear probe at predicting CT-RATE labels?", not "how well does the VLM ground vision."
2. **The OOD headline (0.266 → 0.503) is the gap vs BTB3D-16, not vs the authors' own structured Base VLM (0.395).** The like-for-like delta is +0.108, still real but far less dramatic than the "89% relative" framing. CT-CLIP, the newer Merlin release, and MedRegion-CT — all obvious contemporaneous OOD baselines for 3D CT representation — are not included.
3. **The evaluation pipeline is circular.** Qwen-30B parses the structured training reports into QAS1/2/3 labels *and* grades the generated test reports against those same parsed labels. The same LLM that produced the supervision signal grades the exam. Any systematic parser bias is invisible to the metric.
4. **DCP-2 is not fully characterized in the main text** — it is described as an "external classifier" with details deferred to Appendix B. Its training set, label overlap, and possible test-train leakage are not disclosed in the body, so the 0.603 number cannot be cleanly attributed to DCP-PD vs DCP-2's training data alone.

**Where the paper genuinely earns its claims.** The shortcut diagnostic in Table 2 (F1 collapses to 0.094 without dropout) and the image-token ablation in Table 8 / Fig. 7 are both well-controlled and the kind of self-critical experiment most submissions quietly drop. The lobar-level grounding figures (Figs. 5, 6) are an honest demonstration that DCP-PD makes grounding *better*, not *solved* — absolute F1 at lobar level still tops out around 0.4.

## Method & Architecture

![DCP-PD pipeline overview](/assets/images/paper/dcp-pd/page_004.png)
*Figure 1: DCP-PD pipeline — (a) report structuring via RadExtract + Gemini-2.5-Flash, (b) question-driven label extraction at three granularities (presence / laterality / lobe), (c) lightweight linear discriminator trained on multi-scale frozen CT embeddings, (d) two-stage VLM training with prompt dropout, (e) inference with hot-swappable discriminator.*

**Step-by-step.**

1. **Report structuring.** Raw CT-RATE reports are converted into sectioned anatomy-aware JSON via RadExtract driven by Gemini-2.5-Flash.
2. **Question-driven label extraction.** Three predefined yes/no question sets run on each structured report by Qwen-30B: QAS1 = 18-pathology presence; QAS2 = + laterality for Nodule/Cons/GGO/PE; QAS3 = + lobe (LUL, LLL, RUL, RML, RLL) for Nodule/Cons/GGO.
3. **CT pre-processing.** Resample, axial centered crop/pad, fixed in-plane crop, then stack **11 channels** corresponding to lung / mediastinum / bone / other HU-windows — an under-discussed but load-bearing design choice.
4. **Visual backbone — Atlas / Pillar-0.** 3D conv patchification stem feeds Multi-Scale Attention blocks. The authors use the **intermediate** representation $h^{(2)}_2$, not the final coarsest output, as a cost/detail trade-off.
5. **Multi-scale CT embedding for the classifier.** Pool token grids across all scales, concatenate to an $L \cdot C_\text{in}$ vector, train a lightweight multi-label linear probe with weighted BCE — one probe per question set (DCP-1/2/3).
6. **Cue-prompt templating.** Predicted positives are templated into natural language; QAS2/3 templates add laterality / lobe.
7. **Two-stage VLM training.**
   - **Stage 1 — CT-report alignment:** image encoder frozen, LLM (LLaMA-3 Instruct 8B) frozen; only the cross-attention visual projector + query tokens trained on structured reports.
   - **Stage 2 — Discriminative-guided SFT with prompt dropout:** image encoder still frozen; jointly train visual projector and LoRA adapters with causal-LM loss. Input is $[T_\text{img}; T_\text{disc}; T_\text{qry}]$. Each GT cue entity is independently dropped with probability $p=0.3$ before templating. No auxiliary loss.
8. **Inference.** The pretrained linear probe predicts positives, templates them, and passes them through the same prompt interface. The VLM is never retrained when swapping discriminators.

![Detailed inference architecture](/assets/images/paper/dcp-pd/page_007.png)
*Figure 2: Detailed inference architecture — 11-channel multi-window CT volume → Atlas multi-scale tokens → visual projector + linear classifier; the LLM is conditioned on the concatenation of image tokens, templated DCP tokens, and query tokens.*

## Experimental Results

### Main quantitative comparison on CT-RATE

| Method | Prec. | Recall | F1 | CRG | BLEU(mean) | METEOR |
|---|---|---|---|---|---|---|
| CT2Rep | 0.435 | 0.128 | 0.160 | 0.359 | 0.280 | 0.197 |
| Merlin | 0.295 | 0.112 | 0.160 | 0.352 | 0.260 | 0.148 |
| CT-CHAT | 0.450 | 0.158 | 0.184 | 0.368 | 0.272 | 0.215 |
| BTB3D-16 | 0.260 | 0.260 | 0.258 | 0.370 | 0.305 | 0.223 |
| LLM+S-LMR+CSE | 0.468 | 0.166 | 0.189 | – | 0.286 | 0.219 |
| CT-GRAPH | 0.396 | 0.248 | 0.296 | – | 0.353 | – |
| MS-VLM | 0.222 | 0.329 | 0.261 | – | – | – |
| CT-AGRG w/ CT-Net | 0.457 | 0.630 | 0.501 | – | – | 0.196 |
| Base VLM (Ours) | 0.520 | 0.423 | 0.440 | 0.457 | 0.293 | 0.214 |
| DCP-PD — DCP1 (Ours) | 0.467 | 0.681 | 0.530 | 0.504 | 0.305 | 0.222 |
| **DCP-PD — DCP2 (Ours)** | **0.540** | **0.700** | **0.603** | **0.556** | **0.306** | **0.226** |

![Per-pathology F1 and shortcut diagnostic](/assets/images/paper/dcp-pd/page_010.png)
*Figure 3: Per-pathology F1 — the linear discriminator beats the Base VLM on 17/18 findings; Tables 1–2 show DCP-PD lifts F1 0.440 → 0.603 and that without prompt dropout the No-DCP F1 collapses to 0.094 — the shortcut diagnostic that motivates the whole design.*

### OOD generalization on Rad-ChestCT

| Model | Prec. | Recall | F1 |
|---|---|---|---|
| CT2Rep | 0.299 | 0.139 | 0.133 |
| Merlin | 0.271 | 0.149 | 0.182 |
| CT-CHAT | 0.382 | 0.171 | 0.182 |
| BTB3D-16 | 0.272 | 0.329 | 0.266 |
| Base VLM (Ours) | 0.443 | 0.418 | 0.395 |
| DCP-PD — DCP1 (Ours) | 0.409 | 0.704 | 0.477 |
| **DCP-PD — DCP2 (Ours)** | **0.448** | **0.694** | **0.503** |

![OOD radar and Rad-ChestCT results](/assets/images/paper/dcp-pd/page_012.png)
*Figure 4: Per-pathology radar — DCP-PD attains the highest F1 on every one of the 18 CT-RATE findings; Table 5 shows OOD F1 rising from 0.266 (BTB3D-16) to 0.503 on Rad-ChestCT, but only +0.108 absolute over the authors' own Base VLM (0.395).*

### Critical ablations

- **Shortcut diagnostic (Table 2).** Training with GT cues but no dropout: removing cues at test time collapses F1 to **0.094** (recall 0.071). DCP-PD with p=0.3 recovers to F1 **0.366** / recall 0.373 in the same No-DCP test setting.
- **Attention reliance (Table 3).** S_text drops from 0.441 ± 0.054 (GT DCP) to **0.316 ± 0.036** (GT DCP-PD); S_image rises symmetrically. Modest but consistent evidence that dropout shifts attention back toward image tokens.
- **Dropout-rate sweep (Table 7).** p=0.1 already breaks the shortcut (No-DCP F1 0.310); p=0.3 best on most cue-guided settings; p=0.5/0.7 keeps improving No-DCP robustness while marginally degrading DCP-2 F1 (0.603 → 0.585 → 0.581).
- **Image-token ablation (Table 8 + Fig. 7).** A cue-only LLM with **no image tokens at all** still reaches CT-RATE F1 = **0.600** vs the full model's 0.603. Image tokens only buy +0.025 to +0.233 F1 across 15 lobar-level entries.
- **Predicted-cue vs GT-cue training (Table 9).** Training on the discriminator's noisy predictions barely improves over the base VLM (F1 0.438 vs 0.440); training with GT cues + serving with DCP-1 reaches 0.538 — explicitly motivating the prompt-dropout design.

### Qualitative example

![Qualitative comparison on axial CT slice](/assets/images/paper/dcp-pd/fig_p018_03.png)
*Figure 5: Qualitative case — Base VLM misses lung nodules while DCP-PD with object-level cues correctly reports them (full report comparison in Fig. 8 of the paper).*

## Limitations

**Acknowledged in the paper.**
- The discriminator bounds VLM performance (Fig. 6, Cons-RML regression).
- Current generation benchmarks are partly satisfiable by cue-only LLMs (Table 8 / Fig. 7).
- Fine-grained lobar grounding remains low in absolute terms (top-line lobar F1 ≈ 0.4).

**Not acknowledged but should be.**
- No seed variance / confidence intervals on Tables 1/4/5; no statistical significance testing.
- The report-structuring step depends on a closed-API LLM (Gemini-2.5-Flash) — reproducibility risk and a hidden source of label noise that propagates into every QAS.
- **DCP-2 is opaque in the main text.** Its training data and possible overlap with the CT-RATE test split are not disclosed in the body, so test-train leakage cannot be ruled out without checking the appendix.
- Only one OOD dataset (Rad-ChestCT, same modality / anatomy / scanner population / English reports).
- **No comparison to CT-CLIP, the newer Merlin release, or MedRegion-CT** — the obvious contemporaneous 3D-CT baselines.
- **Circular evaluation pipeline.** Qwen-30B parses train labels *and* grades the test reports against those same parsed labels; any systematic parser bias is invisible to the metric.
- No human-radiologist evaluation of generated reports.

## Why It Matters for Medical AI

The mechanism here — template a classifier as a natural-language prefix, drop it at random — is a clean way to make an LLM-based report generator robust to swapping its auxiliary head. That is a real engineering contribution for production medical-AI pipelines, where classifier and generator versioning are typically locked together.

But the paper's most useful contribution to the field is its self-critique. The Table 8 / Fig. 7 ablation, which the authors did not have to run, demonstrates that current CT report-generation benchmarks reward systems for *parroting the classifier* rather than for *reading the volume*. Anyone shipping or reviewing 3D-CT report generators should treat coarse F1 numbers with that asterisk: the question is no longer "does the model match the QA labels?" but "does adding image tokens add measurable lift over a cue-only LLM at the granularity that matters clinically?" — and at the lobar level, the honest answer in this paper is "yes, but small (+0.025 to +0.233 F1) and only sometimes."

## References

- Paper: arXiv:2604.10437v1 (12 Apr 2026) — *Enhancing Fine-Grained Spatial Grounding in 3D CT Report Generation via Discriminative Guidance*, Wang et al., Boston University / Siemens Healthineers.
- Code: not announced in the manuscript body.
- Related: CT-RATE [Hamamci et al.], Rad-ChestCT [Draelos et al.], Atlas / Pillar-0 backbone, RadExtract, BTB3D-16, CT-AGRG, CT-CHAT, Merlin, PromptMRG, Dia-LLaMA.
