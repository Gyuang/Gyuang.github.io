---
title: "Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation"
excerpt: "Diagnoses a dimensional-collapse bottleneck in 3D CT contrastive encoders (dim90 = 1-9 across four encoders) and bypasses it with a learned [RAG] token that retrieves training-corpus sentences during decoding, lifting CT-RATE Clinical F1 from 0.420 (CT-Agent) to 0.480."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/adarag-ct/
tags:
  - AdaRAG-CT
  - CT-Report-Generation
  - Retrieval-Augmented-Generation
  - Dimensional-Collapse
  - Self-RAG
  - CT-RATE
  - LLaMA-3
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- AdaRAG-CT reframes 3D CT report generation as a *representational bottleneck* problem: across four pre-trained 3D contrastive encoders (CT-CLIP zero-shot, CT-CLIP VocabFine, FVLM, ViSD-Boost) the effective dimensionality collapses to **dim90 = 1-9 out of 512** (vs. CLIP-ImageNet's 243), and all 18 CT-RATE findings show *tail-PC > top-PC* projection AUC — pathological signal lives in the directions cosine similarity throws away.
- The fix is a learned `[RAG]` token appended to the LLM vocabulary that adaptively triggers retrieval from a 572K organ-indexed sentence database built from training reports; on emission the system retrieves K=3 candidates, rolls back the in-progress sentence, and regenerates with the context prepended.
- Headline result: **Clinical F1 0.420 (CT-Agent) -> 0.480 (+6.0 pts)** at 8B on CT-RATE validation — but the matching ablation (Tab. 5) shows fixed-interval N=5 injection scores **0.494**, *beating* adaptive RAG, which directly undercuts the paper's framing of the adaptive mechanism as the primary driver.

## Motivation

3D CT report generation is dominated by VLMs that pair a contrastive 3D vision encoder (CT-CLIP, ViSD-Boost, FVLM) with a large LLM, yet published Clinical F1 on CT-RATE remains clinically inadequate (best reported ~0.42). A telling empirical hint: simply scaling the LLM (CT-CHAT 8B -> 70B) yields no improvement, so the bottleneck is *upstream* of the decoder. The authors diagnose the visual representation as the culprit and reframe retrieval not as world-knowledge grounding (the usual RAG framing) but as a **surrogate visual channel** that injects fine-grained pathology semantics the contrastive embedding cannot encode. The medical-AI angle is novel because it ties RAG design choices to the geometry of the visual encoder rather than the generator.

## Core Innovation

- **Bottleneck diagnosis with cross-domain control.** PCA on four 3D contrastive encoders shows `dim90 = 1-9` and participation ratio `PR = 1.2-6.7` in a 512-d space; CLIP ViT-B/32 ImageNet measured under the identical protocol hits `dim90 = 243`, `PR = 64.4`. A ~50x effective-dimension gap on matched dimensionality.
- **Variance-semantic projection test.** For each of the 18 CT-RATE findings, linear probes trained on tail PCs (1-2) outperform probes trained on top PCs (2) by mean Delta ~ +0.19 AUC. **All 18 findings show the pattern** — pathological signal is systematically misaligned with the high-variance subspace contrastive losses preserve.
- **Retrieval geometry analysis.** Organ-level Jaccard@10 of pathology label sets shows Txt2Txt > Img2Img > Img2Txt across lung/heart/esophagus, motivating a text-channel augmentation rather than denser visual features.
- **Adaptive `[RAG]` token + oracle-mixed training.** A new vocabulary token is supervised by per-sentence perplexity on ground-truth reports; with `p_oracle = 0.7` the model sees ground-truth sentences from later in the same report (wrapped in `<|ret_start|>...<|ret_end|>`) as oracle context during training, and real retrieved sentences otherwise. Context tokens are masked from the LM loss.
- **MMR-based candidate selection.** `MMR(d_i) = lambda * sim(d_i) - (1 - lambda) * max_{d_j in S} BLEU-2(d_i, d_j)` keeps retrieved sentences relevant but diverse.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | 3D CT contrastive embeddings exhibit severe dimensional collapse (2-9 effective dims of 512) | Tab. 1 PCA across 4 encoders; cross-domain control vs CLIP ImageNet (`dim90` 243 vs 1-9) | CT-RATE (encoders pretrained on CT-RATE / related) | ⭐⭐⭐ |
| C2 | The bottleneck is in the visual channel, not the LLM | Scale ablation: 8B base 0.455 vs 70B base 0.405 (Tab. 4); cites CT-CHAT 8B/70B near-identical | CT-RATE val | ⭐⭐ — 70B *underperforms* 8B, which could equally indicate undertrained projector or LoRA-rank-too-small at 70B rather than a clean capacity-vs-channel demonstration |
| C3 | Variance-semantic misalignment (tail-PC > top-PC) is systematic across all 18 findings | Tab. 6 cross-domain + Tab. 7 (all 18 findings, Delta +0.106 to +0.303) | CT-CLIP zero-shot on CT-RATE | ⭐⭐⭐ for the 18-of-18 result on this encoder; ⭐ for generality across encoders (per-finding test is only on CT-CLIP zero-shot) |
| C4 | Naive static retrieval can degrade performance | Tab. 5: fixed-interval N=3 = 0.453 vs RAG-trained no-context 0.462 (Delta -0.009) | CT-RATE val, 8B | ⭐ — only one fixed-interval setting underperforms and only by 0.009 (single run, no CI); claim is overstated since N=5 actually beats AdaRAG |
| C5 | AdaRAG-CT achieves SOTA Clinical F1 0.480 on CT-RATE, +6 pts over CT-Agent | Tab. 4 | CT-RATE val | ⭐⭐ — numerically correct vs CT-Agent's published 0.420, but no statistical test, no multi-seed variance, single split. Also loses to **CT-Agent on METEOR (0.246 vs 0.425) and ROUGE-L (0.354 vs 0.490)** and to CT-CHAT on BLEU-1; "SOTA" only on the clinical-efficacy axis |
| C6 | The adaptive `[RAG]` mechanism is the primary driver of gains | Fig. 2 (Two-Stage ~ Text2Text); Tab. 5 Adaptive 0.480 vs No-RAG 0.462 (+0.018) | CT-RATE val, 8B | ⭐ — **Tab. 5 directly contradicts this**: fixed-interval N=5 (0.494) beats Adaptive RAG (0.480) on Clin-F1. The defense is multi-metric (Adaptive wins on BLEU-4 / ROUGE-L) and no-hyperparameter, but the F1 framing is not supported |
| C7 | Both retrieval and generation components contribute | Tab. 5; Fig. 2 | CT-RATE val | ⭐⭐ — retrieval-vs-no-retrieval is clearly positive; adaptive-vs-fixed is not (see C6) |
| C8 | Oracle-mixed training acts as data augmentation and enables graceful fallback | "-context (OOD)" row drops to 0.402; mean 1.48 triggers/report | CT-RATE val | ⭐ — OOD drop is consistent with context reliance but no ablation on `p_oracle = 0.7` (only stated as best), no all-real vs all-oracle comparison |

**Honest read.** The diagnostic half of this paper (C1, C3) is genuinely strong for a single dataset: dimensional collapse on contrastive 3D medical encoders is shown across four encoders, with a clean cross-domain control on CLIP-ImageNet under identical dimensionality, and the per-finding projection test (18/18 with tail > top) rules out a few-outlier explanation. **The method half is weaker than the abstract implies.** Tab. 5 shows fixed-interval N=5 beating adaptive RAG on Clinical F1, which directly contradicts the framing of the adaptive mechanism as "the primary driver." The +6 pt headline over CT-Agent compares the authors' best model against a baseline number lifted from another publication; the authors' own reproduction in Tab. 10 reads CT-CHAT-8B Clinical F1 as 0.224 (vs. "not reported" originally) and CT-CHAT-70B as 0.161 (vs. 0.184 originally), which suggests baselines should be reproduced uniformly before declaring SOTA. There is no statistical significance test, no multi-seed variance, no external dataset, and **no head-to-head against BTB3D or MedRegion-CT under a unified protocol** — exactly the concurrent encoder-replacement methods that directly attack the bottleneck the authors diagnose. The 8B >= 70B scale argument is correlational; an underperforming 70B can reflect undertraining at fixed LoRA rank rather than a true information bottleneck.

## Method & Architecture

![AdaRAG-CT overview: organ-indexed sentence DB built from training reports, projected visual tokens, and adaptive [RAG] token triggering context injection during decoding](/assets/images/paper/adarag-ct/page_008.png)
*Figure 1: AdaRAG-CT overview. (a) Visual encoding — global CT-CLIP plus four ViSD-Boost organ embeddings (lung, heart, esophagus, aorta) each projected by a per-organ 2-layer MLP into LLM token space. (b) Sentence DB — training reports segmented into ~572K sentences via NLTK + regex, organ-tagged by a rule-based parser (Parser F1 > 0.95), embedded with BiomedVLP-CXR-BERT + a learned 256-d projector, indexed in per-organ FAISS (lung 398K / heart 82K / esophagus 50K / aorta 42K). (c) Adaptive inference — whenever the LLM emits `[RAG]`, the system generates the next sentence as the retrieval query, retrieves K=3 candidates via MMR, rolls back the in-progress sentence, and regenerates with the context prepended.*

### 1. Visual tokenization

Each CT volume yields K = 5 visual tokens: one global max-pooled CT-CLIP embedding in R^512 plus four organ-specific ViSD-Boost embeddings in R^256 (organs = {lung, heart, esophagus, aorta}), each projected by a per-organ 2-layer MLP:

$$
h_o = \mathrm{MLP}_o(v_o) \in \mathbb{R}^{d_{LLM}}
$$

### 2. Organ-indexed sentence database

Training reports are segmented into ~572K sentences (NLTK + regex), organ-tagged by a rule-based parser whose per-organ Parser F1 exceeds 0.95 (reassembly BLEU-1 vs. original report = 0.992). Sentences are embedded with BiomedVLP-CXR-BERT + a learned 256-d projector, fine-tuned contrastively against frozen organ image embeddings. Per-organ FAISS indices total 572,291 sentences.

### 3. `[RAG]` token + oracle-mixed training

A new `[RAG]` token is appended to the LLM vocabulary. Supervision proceeds by running the base model on ground-truth reports, computing per-sentence perplexity, and labeling sentences above a fixed percentile as retrieval targets. For each target, candidates are precomputed via image-based coarse filtering -> text-to-text re-ranking -> MMR selection:

$$
\mathrm{MMR}(d_i) = \lambda \cdot \mathrm{sim}(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} \mathrm{BLEU\text{-}2}(d_i, d_j)
$$

With probability `p_oracle = 0.7` the model sees ground-truth sentences from later in the same report (wrapped in `<|ret_start|>...<|ret_end|>`) as oracle context; otherwise it sees real retrieved sentences. Context tokens are masked from the LM loss. Training uses LoRA r=32 / alpha=64, lr 1e-5, batch 16, up to `K_rag = 4` triggers per sample.

### 4. Adaptive inference

During autoregressive decoding, on every `[RAG]` emission the system: (a) generates the next sentence as the retrieval query, (b) retrieves `K_fine = 3` candidates (Two-Stage: `K_coarse = 20` images -> organ-filtered pool -> MMR; or Text2Text: direct per-organ Txt2Txt + MMR), and (c) rolls back the in-progress sentence and regenerates with the retrieved context prepended. Mean 1.48 triggers/report (median 1); 22% zero, 61% 1-2, 17% >= 3.

### 5. Training recipe

Two-stage: (i) train projectors with LLM frozen, then (ii) LoRA-finetune LLM + projectors jointly. Identical recipe at 8B (LLaMA-3.1-8B-Instruct) and 70B (LLaMA-3.3-70B-Instruct) to isolate scale as the only variable.

## Experimental Results

### Bottleneck diagnosis (Table 1)

![Effective dimensionality across CT-CLIP, ViSD-Boost, FVLM vs CLIP-ImageNet baseline](/assets/images/paper/adarag-ct/page_005.png)
*Figure 2: Effective dimensionality across four 3D CT contrastive encoders vs CLIP ViT-B/32 ImageNet baseline on the same 512-d output. dim90 collapses to 1-9 on medical encoders vs 243 for CLIP-ImageNet; participation ratio drops from 64.4 to 1.2-6.7. Across-the-board ~50x gap.*

### Linear-probe AUC despite collapse (Table 2)

![Per-finding linear-probe AUC showing avg-pool encodes signal despite low effective dim](/assets/images/paper/adarag-ct/page_006.png)
*Figure 3: Frozen linear-probe AUC on 18 CT-RATE pathology labels. Probes are informative (AUC 0.59-0.97) despite dim90 = 1-9 — the signal is present, just compressed into low-variance directions that cosine similarity ignores.*

### Retrieval geometry (Table 3, Jaccard@10)

![Organ-level Jaccard@10 across Img2Img, Img2Txt, Txt2Txt and upper bound](/assets/images/paper/adarag-ct/page_007.png)

| Organ | Img2Img | Img2Txt | Txt2Txt | Upper Bound |
|---|---|---|---|---|
| Lung | 0.351 | 0.313 | **0.563** | 0.992 |
| Heart | 0.825 | 0.638 | **0.925** | 1.000 |
| Esophagus | 0.796 | 0.617 | **0.935** | 1.000 |
| Aorta | **0.741** | 0.617 | 0.666 | 1.000 |

*Figure 4: Txt2Txt > Img2Img > Img2Txt for lung/heart/esophagus motivates a text-channel augmentation; aorta is the lone exception. The Img2Txt floor across organs is consistent with the dimensional-collapse diagnosis — image queries cannot retrieve the right text neighborhoods.*

### Main comparison (CT-RATE validation, Table 4)

![CT-RATE validation comparison table with AdaRAG-CT reaching Clinical F1 0.480](/assets/images/paper/adarag-ct/page_010.png)

| Method | Params | Clin-F1 | Clin-P | Clin-R | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | LLaMA Score |
|---|---|---|---|---|---|---|---|---|---|
| CT2Rep | — | 0.160 | 0.435 | 0.128 | 0.372 | 0.213 | — | 0.197 | — |
| Merlin | 7B | 0.160 | 0.295 | 0.112 | 0.231 | 0.099 | — | 0.148 | — |
| CT-CHAT | 8B | — | — | — | 0.494 | — | 0.584 | 0.311 | 7.440 |
| CT-CHAT | 70B | 0.184 | 0.450 | 0.158 | 0.498 | — | 0.581 | 0.311 | 7.429 |
| BTB3D | 8B | 0.258 | 0.260 | 0.260 | 0.439 | 0.213 | — | 0.223 | — |
| SAMF | 3.8B | — | — | — | 0.440 | 0.261 | 0.417 | **0.417** | 7.165 |
| CT-Agent | — | 0.420 | 0.423 | 0.477 | 0.502 | 0.231 | 0.490 | 0.425 | — |
| Ours (base) | 8B | 0.455 | 0.474 | 0.469 | 0.463 | 0.205 | 0.315 | 0.206 | 7.30 |
| **AdaRAG-CT** | **8B** | **0.480** | **0.502** | **0.520** | 0.496 | 0.242 | 0.354 | 0.246 | **7.75** |
| Ours (base) | 70B | 0.405 | 0.468 | 0.373 | 0.449 | 0.213 | 0.334 | 0.208 | 7.10 |
| AdaRAG-CT | 70B | 0.426 | 0.483 | 0.413 | 0.497 | 0.250 | 0.361 | 0.232 | 7.53 |

Clinical-efficacy SOTA on Clin-F1 / P / R. But **CT-Agent still dominates on METEOR (0.425 vs 0.246) and ROUGE-L (0.490 vs 0.354)**, and CT-CHAT is on top of BLEU-1. The "SOTA" claim is true on one axis only. Note also that AdaRAG-CT-70B (0.426) actually scores *below* AdaRAG-CT-8B (0.480) — consistent with the paper's "bottleneck is visual" claim but, equivalently, with LoRA undertraining at 70B under a fixed rank.

### Context utilization ablation (Table 5, 8B) — the contradiction

| Strategy | Clin-F1 | Delta vs No-RAG | BLEU-4 | ROUGE-L |
|---|---|---|---|---|
| No RAG (RAG-trained, no context @ infer) | 0.462 | — | 0.228 | 0.349 |
| Fixed-interval N=3 | 0.453 | -0.009 | 0.192 | 0.307 |
| **Fixed-interval N=5** | **0.494** | **+0.032** | 0.209 | 0.329 |
| Fixed-interval N=7 | 0.482 | +0.020 | 0.228 | 0.346 |
| Adaptive RAG (ours) | 0.480 | +0.018 | **0.242** | **0.354** |
| -context (OOD) | 0.402 | -0.060 | 0.122 | 0.228 |

Fixed-interval N=5 (0.494) beats Adaptive RAG (0.480) on Clinical F1. The paper's defense is (a) "N=5 is ex-post tuned on the test distribution" — fair, but no held-out tuning experiment is presented to substantiate this — and (b) Adaptive wins on BLEU-4 and ROUGE-L. The "primary driver" framing is therefore supported by the multi-metric trade-off and the no-hyperparameter argument, **not by the headline metric**.

### Per-finding breakdown (Table 9)

13 of 18 findings improve. Biggest wins on findings that need descriptive detail: pulmonary fibrotic sequela **+0.200**, pleural effusion **+0.107**, hiatal hernia **+0.097**. Notable regressions: medical material **-0.174**, lymphadenopathy **-0.122**, peribronchial thickening **-0.085** — low prevalence + sparse / negative-template candidates per the authors, but a -17 pt regression on any clinical-efficacy axis deserves more analysis than the paper provides.

### Reproduced baselines under unified protocol (Table 10)

CT-CHAT-8B Clin-F1 reads 0.224 under the authors' eval pipeline (vs "—" / not reported originally); CT-CHAT-70B drops from 0.184 to 0.161. AdaRAG-CT-8B remains 0.480. Consistent in *direction* with the original-publication numbers but the magnitude mismatch is a caution against direct cross-publication comparison.

### Pipeline & trigger statistics

- Two-Stage vs. Text2Text retrieval are statistically indistinguishable on Clin-F1 with overlapping bootstrap 95% CIs across all 7K training steps (Fig. 2). Authors conclude the `[RAG]` mechanism, not the pipeline, is the active ingredient.
- "Nearly all triggers fire during the lung paragraph," the organ with the worst retrieval precision and most findings — raises whether the trigger has learned semantic uncertainty or merely the prior that lung paragraphs are long and varied.

## Limitations

**Authors acknowledge:**

- Single institution, single modality (CT-RATE non-contrast chest CT). Cross-encoder generality is argued but not externally validated.
- `p_oracle = 0.7` is dataset-dependent.
- Retrieval corpus introduces no external clinical knowledge — gains come from "opening a channel," not from new information.
- Template-heavy CT-RATE sentences ("No evidence of...") can trigger retrieval without adding evidence and push outputs toward corpus phrasing.
- Clinical F1 (binary 18-label presence) cannot measure size, laterality, severity, morphology, temporal change, or out-of-taxonomy findings.
- RAG is a workaround, not a fix: true resolution requires pathology-aware visual pre-training or dense per-token visual features.

**Additionally unaddressed (this review's flags):**

- **The headline ablation contradicts the headline claim.** Tab. 5 shows fixed-interval N=5 beats Adaptive RAG on Clinical F1; the "adaptive mechanism is the primary driver" framing is not supported by F1.
- **No head-to-head with BTB3D or MedRegion-CT under the unified protocol.** Only CT-CHAT is reproduced in Tab. 10; the two most relevant concurrent encoder-replacement baselines that directly attack the same bottleneck are not compared head-to-head.
- **"Naive RAG" baseline is weak.** Naive static retrieval is operationalized only as fixed-interval injection. No LaB-RAG, no direct image-to-report retrieval, no report-template baseline.
- **No ablation on `p_oracle`, `K_coarse`, `K_fine`, MMR lambda, or the perplexity-percentile training threshold.** `p_oracle = 0.7` is stated as best with no supporting sweep.
- **No leakage / near-duplicate analysis at inference.** `p_oracle = 0.7` explicitly injects same-report future sentences during training (by design, not test leakage), but whether retrieval at inference surfaces near-duplicates from same-patient or same-protocol studies in the training split is not tested or filtered.
- **No multi-seed variance for the main table.** Bootstrap CIs appear only in Fig. 2 for the pipeline comparison.
- **Loses CT-Agent on METEOR (0.246 vs 0.425) and ROUGE-L (0.354 vs 0.490)** and CT-CHAT on BLEU-1 — the "SOTA" claim is one-axis SOTA.
- **Single dataset, single modality.** No cross-institution or cross-modality validation of either the bottleneck diagnosis or the RAG gain.
- **Per-finding regressions** on medical material (-0.174) and lymphadenopathy (-0.122) are large in relative terms.
- **Trigger concentration on the lung paragraph** raises whether the `[RAG]` token has learned semantic uncertainty or a paragraph-length prior.
- **The 8B >= 70B scale argument is correlational.** An underperforming 70B is equally consistent with LoRA-rank-too-small or projector undertraining, not a clean "visual channel is the bottleneck" demonstration.

## Why It Matters for Medical AI

The *diagnostic* contribution is the durable one. Dimensional collapse on 3D contrastive medical encoders is shown across four encoders with a matched-dimensionality cross-domain control (CLIP-ImageNet on the same 512-d output), and the per-finding tail-PC > top-PC projection test is 18/18 — that is a clean empirical finding the field should internalize regardless of whether RAG is the right fix. If subsequent work confirms the collapse generalizes beyond CT-RATE and the four encoders tested, it reframes the design problem: contrastive losses optimize for *high-variance* directions that turn out to be variance-semantic *misaligned* for pathology, so the path forward is either (a) pathology-aware visual pre-training that puts the signal in the high-variance subspace (the BTB3D direction), or (b) richer per-token visual features that survive the projection.

The *method* contribution is more cautious. Adaptive RAG as published is a small but real improvement over an 8B no-RAG baseline (0.480 vs 0.455 / 0.462), but Tab. 5 shows fixed-interval N=5 beats it on the headline metric, and the paper has not yet been benchmarked against BTB3D / MedRegion-CT under a unified protocol. Practitioners building on this should: replicate the diagnostic experiments first (they generalize cleanly to any new encoder), then treat the `[RAG]` mechanism as a candidate baseline rather than a settled win until a multi-seed, head-to-head comparison appears.

## References

- **Paper**: Liang, R., Ma, Y., Xing, Y., Fan, Z., Pan, J., Sun, C., Li, L., Gong, K., Xu, J. *Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation.* arXiv preprint, March 2026. arXiv:2603.15822v1.
- **Code**: https://github.com/renjie-liang/Adaptive-RAG-for-3DCT-Report-Generation
- **Dataset (CT-RATE)**: Hamamci et al. 2025. 25,692 non-contrast chest CT studies from 21,304 patients with paired findings + impression and 18 binary pathology labels.
- **Related: encoder replacement** — BTB3D (Hamamci et al., NeurIPS 2025) — wavelet + causal-conv + LFQ tokenizer; MedRegion-CT — region-focused 3D CT report generation.
- **Related: contrastive 3D CT encoders** — CT-CLIP (Hamamci et al. 2024), ViSD-Boost, FVLM.
- **Related: RAG for medical generation** — Self-RAG (Asai et al. 2024) for the adaptive-retrieval-token recipe; LaB-RAG and report-template retrieval baselines that the paper does *not* compare against.
- **Related: VLM baselines** — CT2Rep, Merlin, CT-CHAT (8B/70B), SAMF, CT-Agent.
