---
title: "Ker-VLJEPA-3B: Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression"
excerpt: "A language-free LeJEPA ViT-L grafted into a frozen Llama 3.2 3B via a 4-phase curriculum sets a new (thin) SOTA on CT-RATE at macro F1 = 0.429, +3.6% over U-VLM."
categories:
  - Paper
  - CT-Report-Generation
tags:
  - Ker-VLJEPA
  - 3D-CT
  - Report-Generation
  - LeJEPA
  - Llama-3.2
  - Cross-Attention
  - Curriculum-Learning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
permalink: /paper/ker-vljepa/
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- Free-text radiology report generation from 3D thoracic CT using a **language-free** LeJEPA ViT-L grafted into a frozen Llama 3.2 3B decoder through a four-phase curriculum. The vision encoder never sees text supervision and stays frozen across all phases.
- Three load-bearing engineering choices: **(i) zone-constrained cross-attention** that compresses up to 600 axial slices into exactly 32 anatomically-ordered tokens (apex→base); **(ii) positive-findings-only training** that kills the 90%-normal-text gradient driving posterior collapse; **(iii) warm bridge initialization** that ports 416 random bridge/LoRA tensors from a prior converged Phase 3 run.
- Headline: **macro F1 = 0.429** on CT-RATE (2,984 validation volumes, official RadBERT extraction), **+3.6%** over U-VLM (0.414). With per-class threshold tuning on the eval set: 0.448 (+8.2%) — authors themselves flag this as a data-leakage upper bound.

## Motivation

3D CT report generation hits three walls at once. First, 300–600 axial slices blow through any practical LLM context window. Second, less than 1% of voxels carry pathology while ~90% of report tokens describe normal anatomy — a brutal imbalance that pushes LLMs into a generative "posterior collapse" where they emit fluent generic normal text regardless of the input volume. Third, every prior CT-RATE method (CT-CLIP, CT-CHAT, BTB3D, U-VLM) builds on a vision encoder that was itself trained with text or label supervision, so linguistic priors are baked into the visual backbone before generation training even begins.

The authors argue for total decoupling: pure self-supervised vision, language only at the bridge, and a curriculum that explicitly controls when and how the two modalities meet. The medical-AI framing is direct — CT-RATE is the established 3D thoracic-CT benchmark, and the bar to clear is U-VLM at macro F1 = 0.414.

## Core Innovation

1. **Frozen LeJEPA ViT-L vision backbone (Guided-Chest-CT-LeJEPA).** ViT-Large trained from scratch on CT-RATE via Latent-Euclidean JEPA with anatomy-guided semi-3D crops and an auxiliary 118-class organ-prediction objective. Per-slice embedding dim = 1024. **No text supervision of any kind**, frozen across all four phases.
2. **Zone-constrained cross-attention** compresses N≤600 slices into K=32 anatomically-grounded tokens. Each learnable region query attends only to slices inside its z-axis zone, so token 0 is the thoracic apex and token 31 is the base **by construction**.
3. **PCA whitening for Llama-3.2 anisotropy.** Llama 3.2 3B layer-14 text reps are catastrophically anisotropic (mean pairwise cosine = 0.949, d′ = 1.36). Whitening with the top 256 PCs of 22,773 training-report embeddings sends cosine to −0.001 and d′ to 16.03 — an 11.8× discriminability jump while retaining 97.3% of variance.
4. **Positive-findings-only training** in Phases 2 and 3 to prevent posterior collapse on normal-anatomy text.
5. **Warm bridge initialization** — the 27 bridge components and 392 LoRA tensors absent from the Phase 2 checkpoint are ported from a prior converged Phase 3 run rather than re-initialized randomly. The single largest engineering insight in the paper.
6. **Multi-layer Flamingo-style cross-attention** injected at LLM layers 7, 14, 21, with a **critical bug fix**: x-attn hooks must fire on every autoregressive decode step, not only at prefill. The fix alone yields a 2.5× generation F1 jump (0.122 → 0.304).

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | New SOTA on CT-RATE (macro F1 0.429, +3.6% over U-VLM) | Table IV, single run, no variance, no significance test | ⭐⭐ |
| C2 | Zone-constrained cross-attention compresses to 32 tokens while preserving spatial localization | Eqs. 1–4; no ablation vs uniform pooling / unconstrained perceiver; "by construction" argument only | ⭐ |
| C3 | 56.6% of generation quality comes from patient-specific visual content | Table IX, P3 F1 0.222→0.096 zeroed; shuffled (0.122) ≈ random (0.121) | ⭐⭐⭐ |
| C4 | Visual tokens contribute 2× more on pathology-specific words | Table X, ΔNLL pathology +0.020 vs generic +0.011 | ⭐⭐ |
| C5 | Positive-findings-only training eliminates posterior collapse (sustained >15 epochs) | Table II historical runs; Table XI +0.123 F1 | ⭐⭐ |
| C6 | Warm bridge gives immediate convergence (ep-1 F1 0.425 vs 0.360) and improves precision | Table VI head-to-head; isolates bridge-reset effect via "cold + better P2" control | ⭐⭐⭐ |
| C7 | PCA whitening yields 11.8× discriminability improvement | Table I, d′ 1.36 → 16.03 on Llama 3.2 3B layer-14 reps | ⭐⭐⭐ |
| C8 | Cross-attention fire-on-every-decode-step fix → 2.5× F1 (0.122 → 0.304) | Implementation note + Table XI baseline | ⭐⭐ |
| C9 | Language-free encoder matches or beats text-supervised encoders | Final F1 vs CT-CLIP/CT-CHAT/U-VLM; no controlled encoder swap | ⭐⭐ |
| C10 | Framework is modality-agnostic (any SSL encoder, imaging/genomic/sensor) | Asserted in abstract/intro/conclusion; zero non-imaging experiments | ⭐ |
| C11 | Selective xattn-freeze + EWC enables Phase 4 narrative adaptation without forgetting | Table III v1–v5; v5 = 0.418 vs v1 = 0.297 | ⭐⭐ |
| C12 | Per-class threshold optimization → 0.448 (+8.2% over U-VLM) | Table IV last row; authors flag as data-leakage upper bound | ⭐ |

**Honest read.** The strongest contributions in this paper are the **engineering audit findings**: the decode-step cross-attention bug fix (C8), the discovery that Llama embeddings are catastrophically anisotropic and PCA whitening fixes it (C7), the demonstration that bridge re-initialization — not Phase-2 representation quality — is the true Phase-3 bottleneck (C6), and the rigorous shuffled-vs-zeroed-vs-random ablation establishing genuine visual grounding (C3). These are reproducible insights the field can adopt directly.

The **SOTA headline (C1) is real but thin**: single run, no seeds, no confidence intervals, no significance test against U-VLM's 0.414 — a +0.015 gap that could easily sit inside run-to-run variance. Recall is dramatically higher (+22.1%) at substantial precision cost (−20.8%); the authors frame this as a clinical-safety design choice, but it could equally indicate over-generation of findings. The **language-free / modality-agnostic narrative (C9, C10)** is the paper's marketing hook and the **weakest-supported** claim: there is no controlled head-to-head where only the encoder differs (e.g., swap LeJEPA for a CT-CLIP encoder inside the same Ker-VLJEPA-3B curriculum), and the modality-agnostic claim is asserted without a single non-CT or non-imaging experiment. **Zone-constrained cross-attention (C2)** is plausible and elegant but is never ablated against the obvious baselines (uniform pooling to 32 tokens, unconstrained 32-token perceiver, K=16/64, adaptive K) — load-bearing yet empirically untested. Notable inconsistency: **λ_ewc is given as 100 in Eq. 13 text but as 500 in Fig. 2 caption.** And the threshold-optimized 0.448 result (C12), while correctly flagged as data-leakage by the authors, still appears in the headline number row of Table IV — borderline.

## Method & Architecture

![Ker-VLJEPA-3B system architecture](/assets/images/paper/ker-vljepa/page_004.png)
*Figure 1: System architecture (page 4). Frozen LeJEPA ViT-L per-slice embeddings → zone-constrained cross-attention compressing to 32 anatomically-ordered tokens → JEPA predictor + NormCalibrator → embedding grafting into Llama 3.2 3B with gated cross-attention adapters at layers 7 / 14 / 21.*

### Zone-constrained cross-attention

Add a physical z-positional encoding from DICOM z-spacing: $s'_i = s_i + \text{PE}(z_i)$. Partition the z-axis into $K=32$ contiguous zones $Z_k = \{i : (k-1)N/K \le i < kN/K\}$. Each of the $K$ learnable region queries $q_k$ (SVD-initialized from real slice embeddings) attends **only** to slices in its own zone via 16-head MHA:

$$v_k = \text{MHA}(q_k, \{s'_i\}_{i \in Z_k}, \{s'_i\}_{i \in Z_k})$$

producing $V \in \mathbb{R}^{32 \times 1024}$ (compression up to ~19:1), followed by a global TransformerEncoderLayer for inter-zone communication. The inductive bias is built-in: token 0 is the apex, token 31 is the base, with no other projector guaranteeing this ordering.

### JEPA predictor + norm calibration

Linear 1024 → 3072 (Llama hidden) with LayerNorm and dropout, initialized via SVD of text-embedding principal components. **NormCalibrator** rescales visual tokens to match Llama's measured mean text-norm (1.1484): $\alpha = \|e_{\text{text}}\| / \|\hat{V}\|$. Without it, the LLM treats visual tokens as anomalies.

### Embedding grafting + multi-layer injection

The chat template contains 32 `<|visual_region|>` placeholder tokens whose embeddings are replaced with $\tilde{V}$ via differentiable mask-scatter. Flamingo-style gated cross-attention adapters are injected at LLM layers 7, 14, 21 with per-layer Q/K/V/O (Xavier gain=0.3):

$$h'_l = h_l + \text{MHA}^{\text{xattn} }_l(h_l, \text{Linear}_l(\tilde{V}), \text{Linear}_l(\tilde{V}))$$

**Critical bug fix:** the x-attn hooks must fire on **every autoregressive decode step**, not only at prefill. A sequence-length guard was skipping x-attn during token-by-token decoding; removing it gave a 2.5× generation F1 jump (0.122 → 0.304).

### PCA whitening (the Llama 3.2 anisotropy fix)

![Llama 3.2 3B anisotropy and PCA whitening (Table I)](/assets/images/paper/ker-vljepa/page_005.png)
*Table I (page 5): Llama 3.2 3B text embeddings are catastrophically anisotropic (mean pairwise cosine = 0.949, d′ = 1.36). PCA whitening to 256-d sends cosine to −0.001 and d′ to 16.03 — an 11.8× discriminability gain — while retaining 97.3% of the variance.*

The JEPA embedding head $z_v = \text{Linear}_2(\text{GELU}(\text{LN}(\text{Linear}_1(\bar{V}))))$ projects pooled visual tokens into this 256-d isotropic space, which is what Phase 2 contrastive alignment then operates in.

### Four-phase curriculum

![Four-phase curriculum and warm bridge](/assets/images/paper/ker-vljepa/page_006.png)
*Figure 2 (page 6): Four-phase curriculum — visual alignment → contrastive bridge → generative fine-tuning → narrative adaptation. The warm bridge transfers 416 converged tensors across Phase 3 runs; the panel also contains Table II (posterior-collapse history).*

- **Phase 1 — Visual Alignment (Classification).** LLM and JEPA predictor frozen. Loss $L_1 = 1.5 L_{\text{BCE}} + 1.0 L_{\text{MIL}} + 1.0 L_{\text{orth}} + 0.5 L_{\text{MMD}}$. MMD uses IMQ kernel $k(x,y) = (1+\alpha\|x-y\|^2)^{-1/2}$ with $\alpha \approx 0.039$ matched per-sample against that sample's positive-condition text embeddings. Normal volumes (zero positives) get no MMD term. 20 epochs, batch 32, LR 5e-5.
- **Phase 2 — Contrastive Bridge.** InfoNCE with cross-GPU negatives (512 effective from 8 GPUs), learned τ init 0.10. **Positive-findings-only text targets** — aligning against raw reports collapses everything to a uniform "normal" embedding. 30 max epochs (early stop at 24), batch 64/GPU, LR 3e-5.
- **Phase 3 — Generative Fine-Tuning (positive-findings only).** Visual encoder frozen; train JEPA predictor + LoRA + xattn adapters + layer projectors. $L_3 = L_{\text{LM}} + \lambda_{\text{fcls}} L_{\text{focal}}(\gamma=2) + \lambda_{\text{jepa}} L_{\text{JEPA}} + 3.0 \cdot L_{\text{LLM-cls}}$. An LLM visual classifier on the last-layer hidden states forces visual information to be preserved through all layers. **LoRA frozen after epoch 6** to block language-prior shortcuts. ReduceLROnPlateau on gen_f1 (patience 3, factor 0.5). 50 max epochs (best at ep 9 with warm bridge), batch 8, LR 2e-5.
- **Phase 4 — Raw Narrative Adaptation.** Trains on verbatim CT-RATE Findings_EN. **Selective freezing:** cross-attention adapters + layer projectors frozen; only LoRA trainable, regularized by EWC: $L_4 = L_{\text{LM}} + \lambda_{\text{cls}} L_{\text{focal}} + \lambda_{\text{ewc}} \sum_i (\theta^{\text{LoRA}}_i - \theta^{P3}_i)^2$. **λ_ewc = 100 per Eq. 13 (Fig. 2 caption says 500 — inconsistency noted).** LR 5e-7 (40× lower than Phase 3). 8 epochs.

### Warm bridge initialization

After Phase 2 the 27 bridge components (3 layer projectors + 21 xattn adapter tensors) plus 392 LoRA tensors are absent from the Phase 2 checkpoint and would initialize randomly. The warm bridge ports these 416 tensors from a prior converged Phase 3 run into the new Phase 3. Effect: epoch-1 F1 0.425 vs 0.360 cold, epochs-to-F1>0.42 drops from 5 to 1, best F1 climbs from 0.427 (cold baseline) and 0.424 (cold + better P2) to 0.446.

## Experimental Results

### Main comparison on CT-RATE (2,984 val volumes, official RadBERT protocol)

![CT-RATE SOTA comparison (Table IV)](/assets/images/paper/ker-vljepa/page_007.png)
*Table IV (page 7): Ker-VLJEPA-3B P4 reaches macro F1 = 0.429, +3.6% over U-VLM (0.414); per-class threshold optimization yields 0.448 with the data-leakage caveat the authors acknowledge. The page also includes Table III (Phase 4 history), Table V (per-phase progression), and Table VI (warm bridge).*

| Method | Macro F1 | Macro Prec | Macro Rec |
|---|---|---|---|
| CT-CLIP | 0.194 | — | — |
| CT-CHAT | 0.287 | — | — |
| BTB3D | 0.354 | — | — |
| U-VLM | 0.414 | 0.491 | 0.429 |
| **Ker-VLJEPA-3B (P3)** | 0.422 | 0.380 | 0.517 |
| **Ker-VLJEPA-3B (P4)** | **0.429** | 0.389 | 0.524 |
| Ker-VLJEPA-3B (P4, per-class opt.) | 0.448 | 0.380 | 0.585 |

Ker-VLJEPA buys F1 with **recall** (+22.1% absolute over U-VLM) at substantial **precision cost** (−20.8% absolute). Whether this is a clinical-safety win or over-generation is a judgement call.

### Per-phase progression (Table V)

| Phase | Type | F1 | Prec | Rec | AUC |
|---|---|---|---|---|---|
| 1 | Cls | 0.460 | 0.463 | 0.551 | 0.811 |
| 2 | Cls | 0.465 | 0.490 | 0.532 | 0.816 |
| 3 | Gen | 0.422 | 0.380 | 0.517 | — |
| **4** | **Gen** | **0.429** | **0.389** | **0.524** | — |

Phase 4 over Phase 3 buys only +0.007 F1 while losing positive-findings-only's training stability — the marginal value of Phase 4 vs simply reporting Phase 3 is genuinely modest.

### Ablations worth highlighting

![Visual-token grounding (Table IX) and NLL semantic binding (Table X)](/assets/images/paper/ker-vljepa/page_008.png)
*Page 8 tables (VII per-class, VIII linear probe, IX visual-grounding ablation, X NLL semantic binding). Table IX is the strongest result in the paper: zeroing visual tokens collapses P3 F1 by 56.6%, and tokens shuffled from another patient perform no better than random — proof the model is reading patient-specific content, not statistical regularities.*

- **Cold vs warm bridge (Table VI).** Warm raises ep-1 F1 from 0.360 → **0.425** (+18%), epochs-to-F1>0.42 from 5 → 1, best F1 0.427 → **0.446**, precision 0.391 → 0.437. Critically, "cold bridge + better Phase 2" gives 0.424 — *worse* than the cold baseline — proving the 416 random bridge tensors are the true Phase-3 bottleneck, not Phase 2 representation quality.
- **Visual-token grounding (Table IX, 304 samples).** Zeroing visual tokens drops P3 F1 from 0.222 → 0.096 (**−56.6%**); shuffled tokens from another patient give 0.122 — indistinguishable from random noise. Precision collapses from 0.408 → 0.114 without visual tokens.
- **NLL semantic-binding (Table X, 200 samples).** ΔNLL on pathology words = +0.020 (zeroed) — **2× larger** than on generic text (+0.011). Shuffled tokens give +5.3% (vs +0.9–1.1% zeroed) because shuffled tokens *actively mislead*, while zeroed tokens just trigger cautious generation.
- **Linear probe (Table VIII).** Raw LeJEPA = 0.447 → Phase 1 norm-matched = 0.488 → Phase 2 = **0.495** → Phases 3 and 4 stay at 0.495 — frozen-encoder design preserves discriminability across all later phases.
- **Posterior collapse history (Table II).** Every prior fix collapsed within 1–4 epochs (baseline 0.198, +xattn-fix 0.304, +vis-dropout 0.262, +LLM-vis-cls 0.259). Only **+positive-findings (0.427, never collapses)** and then **+warm bridge (0.446)** sustained.
- **Phase 4 history (Table III).** Five configurations tried; only v5 (freeze cross-attn + EWC) reached 0.418. v1–v4 (no freeze / freeze LoRA / EWC+JEPA+MMD / EWC only) all regressed to 0.27–0.30.

![Component ablation (Table XI)](/assets/images/paper/ker-vljepa/page_009.png)
*Table XI (page 9): Component ablation. xattn-fix alone = 0.304; +positive-findings = 0.427 (+0.123, the single largest gain in the paper); +warm bridge = 0.446 (+0.019); +Phase 4 = 0.429 (task changes to raw narrative).*

## Limitations

**Authors acknowledge:**

- Single-benchmark evaluation (CT-RATE only); no multi-center / multi-vendor / multi-population validation.
- Pre-computed LeJEPA embeddings are frozen inputs; no end-to-end raw-voxel training.
- Per-class threshold optimization (0.448) carries data leakage — should be read as an upper bound only.
- 8× H200 GPU requirement limits accessibility.
- No formal observer study with clinical radiologists.

**Reviewer additions (the ones that hurt the headline):**

- **No variance reporting and no multiple seeds.** The +0.015 F1 gap over U-VLM may not be statistically significant; the paper offers no confidence intervals or significance test.
- **"Language-free is better" lacks a controlled encoder swap.** There is no experiment swapping LeJEPA for a text-supervised encoder (e.g., CT-CLIP) inside the same Ker-VLJEPA-3B curriculum. The C9 comparison is across very different methods, not a controlled ablation.
- **"Modality-agnostic" has zero non-imaging experiments.** The claim is asserted in abstract, intro, and conclusion without a single experiment on a non-CT, non-imaging modality.
- **K=32 zone tokens are never ablated.** No comparison vs uniform pooling to 32 tokens, vs unconstrained 32-token perceiver, vs K=16/64, vs adaptive K. The choice is presented as load-bearing but is empirically untested.
- **Per-class threshold tuning leakage.** The 0.448 number is correctly flagged as a data-leakage upper bound, yet it still appears in the headline row of Table IV.
- **λ_ewc inconsistency** between Eq. 13 text (100) and Fig. 2 caption (500).
- **Clinical narrative quality is never reported** (no BLEU/ROUGE/BERTScore/RadGraph/RaTE). The entire evaluation is downstream RadBERT label F1, which can reward findings-listing prose over coherent radiology reports.
- **Per-class results (Table VII)** show interlobular septal thickening at F1 = 0.203 with recall 0.142 — sparse-finding detection is still very weak.
- **Calibration / confidence not reported**, despite the "high-recall safety net" clinical framing.
- **LeJEPA pretraining and downstream evaluation share the same CT-RATE corpus** — while no text is used, this is in-distribution SSL and may not transfer to outside scanners.

## Why It Matters for Medical AI

If you read this paper for the SOTA number, you will be underwhelmed: +0.015 F1 over U-VLM with no variance reporting is not a knockout. If you read it for the **engineering postmortem**, it is one of the most useful 3D-CT VLM papers of the year. Four insights generalize beyond this exact architecture:

1. **Llama 3.2 3B text embeddings are catastrophically anisotropic at layer 14** (cosine 0.949) — any contrastive bridge into a small Llama needs PCA whitening or it is fighting representational geometry it cannot win.
2. **Cross-attention adapter hooks need to fire at every decode step, not only at prefill** — a sequence-length guard quietly silences x-attn during token-by-token generation and costs 2.5× F1.
3. **Positive-findings-only training is the actual cure for the 90%-normal-text posterior-collapse problem** in radiology VLMs — every other trick (visual dropout, LLM-visual classifier, x-attn fix) collapsed within 1–4 epochs.
4. **The "bridge" — random projector + LoRA tensors re-initialized between curriculum phases — is the dominant bottleneck**, not the representation quality fed into it. Warm-starting those 416 tensors from a prior converged run gives an instant 18% epoch-1 F1 gain.

These are reproducible, transferable engineering findings. The "language-free vision is better" thesis remains unproven without a controlled encoder swap, and the "modality-agnostic" framing should be read as future work, not a result. Use this paper as a curriculum-and-bridge engineering reference, not as a settled answer on whether SSL beats text-supervised pretraining for 3D CT.

## References

- Paper: arXiv:2603.23308v1 (cs.CV, 24 Mar 2026)
- Model weights: [Hugging Face — `IBI-CAAI/Ker-VLJEPA-3B`](https://huggingface.co/IBI-CAAI/Ker-VLJEPA-3B)
- Dataset: [CT-RATE (Hamamci et al., Nat. Biomed. Eng. 2025)](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- Label extractor: official CT-RATE RadBERT classifier
- Related baselines: CT-CLIP, CT-CHAT, BTB3D, U-VLM (CT-RATE benchmark family)
- Backbone components: LeJEPA (Latent-Euclidean JEPA), Llama 3.2 3B, Flamingo-style gated cross-attention, EWC (Kirkpatrick et al. 2017)
