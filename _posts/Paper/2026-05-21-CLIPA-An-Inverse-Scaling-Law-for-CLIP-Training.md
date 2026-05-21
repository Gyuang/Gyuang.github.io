---
title: "CLIPA: An Inverse Scaling Law for CLIP Training"
excerpt: "Larger CLIP encoders tolerate shorter image/text token sequences, yielding 67.8% zero-shot ImageNet-1k on 8xA100 in ~3 days and 83.0% with G/14 at ~33x less compute than OpenCLIP-G/14."
categories:
  - Paper
  - VLM-Alignment
  - Multimodal-Alignment
  - LLM
permalink: /paper/clips-inv-scale/
tags:
  - CLIPA
  - CLIP
  - Inverse-Scaling-Law
  - Token-Reduction
  - Syntax-Masking
  - Efficient-Pretraining
  - Vision-Language
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- CLIPA finds an **inverse scaling law**: the larger the CLIP encoder, the *shorter* the image/text token sequence it can train on without losing accuracy. The headline budget run hits **67.8% zero-shot ImageNet-1k on 8xA100 in ~3 days** (CLIPA-L/16, ~628 GPU-h) versus OpenCLIP-B/16's 67.1% at 10,700 GPU-h.
- Token-reduction strategy matters: for the image side, **resize beats every mask variant**; for the text side, **syntax masking** (priority-keep nouns then adjectives) beats truncation/random/block at extreme lengths. The authors attribute the gap to CLIP's discriminative objective preferring information-*preserving* reduction over the MIM-style information-*removing* reduction.
- Scaled up, CLIPA-G/14 on DataComp-1B reaches **83.0% IN-1k zero-shot at ~33x less compute than OpenCLIP-G/14** — though this number lumps multi-stage fine-tuning at multiple resolutions, so the apples-to-apples ratio is closer to ~15x.

## Motivation

CLIP-style contrastive pre-training scales beautifully with data and model size but the compute bill scales with it, locking academic labs out — OpenCLIP-B/16 needs ~176 A100 x 61h, OpenCLIP-L/14 ~50,800 GPU-h. Prior efficiency work attacked the data axis (curated subsets, dedup) or the image-token axis (FLIP's 50-75% random patch masking, RECLIP's resized inputs). CLIPA asks a sharper question: **how short can the input token sequence get before performance collapses, and does the breaking point depend on model size?** The answer turns out to be counter-intuitive — the breaking point moves *down* as the encoder grows. The medical-AI relevance is indirect but real: every downstream medical VLM that starts from a CLIP backbone (BiomedCLIP, PubMedCLIP, MedCLIP) inherits this compute floor, and a 15-30x cheaper recipe lowers the barrier to domain-specific CLIP pre-training on radiology / pathology corpora where compute, not data, is the binding constraint.

## Core Innovation

- **Inverse scaling law (image and text).** For a fixed performance-drop tolerance, the minimum token length *decreases* with encoder size. Demonstrated across S/16, B/16, L/16 for images and three sizes for text, across four reduction strategies per modality, on IN-1k and OOD/retrieval suites.
- **Information-preservation principle.** Image *resize* (anti-aliased bilinear down-sampling, e.g. 224 -> 112) Pareto-dominates random/grid/block masking; text *syntax masking* (POS-aware: nouns > adjectives > others) dominates truncation/random/block at extreme lengths. The authors argue this is because CLIP's contrastive objective rewards *preserving* signal, whereas MIM rewards *removing* signal for the predictor to recover.
- **CLIPA recipe.** Train large encoders on small token inputs (resize + truncate-or-syntax-mask), then briefly fine-tune at full resolution. Color-jitter (0.32 strength, 0.8 prob) + grayscale (0.2 prob); switch ViT CLS token -> global average pooling for stability at extreme token reduction.

## Claims & Evidence Analysis

| # | Claim | Evidence | Datasets | Strength |
|---|---|---|---|---|
| C1 | Larger CLIP encoders need fewer image tokens for equivalent performance drop (image-side inverse scaling) | Fig. 4: 4 strategies x 3 model sizes x ~5 token lengths; Figs. 5-6 corroborate on retrieval and OOD | LAION-400M -> IN-1k, COCO, IN-V2/R/A/Sketch | ⭐⭐⭐ |
| C2 | Same inverse scaling holds on the text side | Fig. 7 across S/B/L x 4 text strategies x 5 lengths | IN-1k | ⭐⭐ — clear at extreme lengths but one cell is non-monotone; retrieval saturates by 16 text tokens |
| C3 | Information-preserving reduction (image resize, text syntax mask) > aggressive masking | Sec. 3 head-to-head on L/16 (resize 68.9 vs random 67.6 vs grid 67.3 at 75% reduction) | LAION-400M -> IN-1k | ⭐⭐ — single-run, no seeds |
| C4 | CLIPA-L/16 = 67.8% IN-1k in ~3 days on 8xA100, ~17x fewer GPU-h than OpenCLIP-B/16 at matched accuracy | Tab. 1: 67.8 / 628 GPU-h vs OpenCLIP-B/16 67.1 / 10,700 GPU-h | LAION-400M -> IN-1k + 9 zero-shot evals | ⭐⭐⭐ — clear like-for-like; GPU-hours sourced from OpenCLIP's own report |
| C5 | CLIPA-G/14 = 83.0% IN-1k zero-shot, ~33x less compute than OpenCLIP-G/14 | Tab. 2 final row | DataComp-1B -> IN-1k + VTAB | ⭐⭐ — single best run; "33x" mixes pretrain (84²) + fine-tune (224², 336²) stages; no error bars |
| C6 | Inverse scaling generalizes beyond ViT to ConvNeXt | Sec. 4.3: ConvNeXt-T vs ConvNeXt-B under resize only | LAION-400M -> IN-1k | ⭐ — two model points, image side only, resize only |
| C7 | Shortening text to 16/8 with syntax mask *improves* B/16 and L/16 IN-1k | Fig. 7: syntax-mask curve dips below zero around length 16 for larger models | LAION-400M -> IN-1k | ⭐ — interesting but likely a LAION caption-noise denoising effect; not tested on cleaner data |
| C8 | Method preserves robustness/retrieval at scale | Tab. 1, Tab. 2 retrieval columns; VTAB Tab. 3 | COCO, Flickr30k, VTAB | ⭐⭐ — CLIPA-L/16 trails OpenCLIP-L/14 in absolute Recall@1; comparable only at matched compute |

**Honest read.** The image-side inverse scaling law (C1) is the genuinely well-supported empirical contribution — consistent direction across four strategies, three model sizes, four eval suites. The text-side version (C2) is real but noisier: retrieval saturates by ~16 text tokens, so the law is really about *classification*, not fine-grained alignment, and the paper does not always foreground that caveat. The headline efficiency numbers (C4, C5) are credible but the comparison is **accuracy-matched, not training-FLOP-matched** — CLIPA-L/16 still trails OpenCLIP-L/14 on COCO retrieval at higher GPU-h. The "33x cheaper than OpenCLIP-G/14" claim mixes pretrain compute at 84² with multi-stage fine-tuning at 224² and 336²; the apples-to-apples comparison is closer to ~15x once you align fine-tune budgets. **There is no variance reporting, no multiple seeds, and no significance testing anywhere** — single-run is standard for big-CLIP papers but worth flagging. The ConvNeXt generalization (C6) is two data points and should be read as suggestive, not established. C7 (shorter text helping) is the most intriguing hint but the most under-supported — LAION-400M is famously noisy, so this may simply be a denoising effect that would vanish on cleaner captions; the paper does not test this.

**Versus Long-CLIP.** Long-CLIP (Zhang et al., ECCV 2024) finds the *opposite directional* result: extending CLIP's 77-token text encoder to ~248 tokens via positional-embedding interpolation *improves* dense-caption retrieval and T2I conditioning. The two results do not conflict — they answer different questions. CLIPA studies **pre-training efficiency** with web-scale noisy short captions and measures *zero-shot classification* (where the signal concentrates in a handful of nouns — exactly what syntax masking preserves). Long-CLIP studies **inference / downstream alignment** with dense captions (DCI, ShareGPT4V) and measures *long-text retrieval* and *T2I conditioning*. Honest synthesis: short tokens are fine when (a) you are pre-training on noisy short captions and (b) you only care about coarse class-level zero-shot. Once you (a) have access to information-dense long captions and (b) need fine-grained or compositional alignment, more tokens help. This also reconciles with the ARO compositional under-performance the authors themselves flag in Sec. 7.

**Caption-quality confound.** CLIPA's text-reduction recipe is applied *to LAION captions*, which have a Zipfian quality distribution — the first 8 tokens of a LAION caption are usually the most semantically loaded (alt-text patterns: noun phrase, prepositional phrase, ...). Truncation and syntax masking implicitly exploit this caption-structure prior. The paper does not rerun the text ablation on better-curated data (DataComp-high-quality) or on machine-rewritten dense captions, so we cannot tell whether the inverse-scaling-on-text result is a property of large encoders or an artifact of LAION's caption structure. This is the single biggest open question.

**Practical implication for synthetic-caption pipelines.** If you generate long, dense synthetic captions (LLaVA-style, ShareGPT4V), CLIPA's specific text recipe (truncate to 8, syntax-mask) is the *wrong* choice — you are throwing away exactly the signal you paid the captioner to produce. The *image-resize* trick, however, is orthogonal and still applies. A sensible synthesis: keep CLIPA's image-resize + 2-stage fine-tune *with* full-length text on dense synthetic captions.

## Method & Architecture

![CLIPA inverse scaling teaser: three encoder sizes need progressively fewer image/text tokens](/assets/images/paper/clips-inv-scale/fig_p002_15.png)
*Figure 1: The headline inverse scaling — S/16 needs ~101 image / 16 text tokens; L/16 needs only ~50 / 6 to stay within a small accuracy tolerance. Larger encoders are progressively more token-efficient.*

The paper is really two artifacts: (i) an empirical scaling study across token-reduction strategies, (ii) a recipe (CLIPA) that exploits the finding.

1. **Backbone.** Standard CLIP: vanilla ViT image encoder (S/16, B/16, L/16, H/14, G/14) + non-autoregressive Transformer text encoder, contrastive InfoNCE on image-text pairs. Pre-trained on LAION-400M (Sec. 3-5) or LAION-2B / DataComp-1B (Sec. 6).
2. **Image-token reduction strategies (4).** Applied to a 224² ViT input (nominally 196 patches):
   - *Random masking* — drop a fraction of patches (FLIP strategy).
   - *Grid masking* — keep one patch per 2x2 window (fixed 75% drop).
   - *Block masking* — drop contiguous blocks.
   - *Resizing* — anti-aliased bilinear down-sampling, e.g. 224 -> 112 (~75% token reduction), preserves the entire image.
3. **Text-token reduction strategies (4).** Applied to the text encoder's max length (default 32):
   - *Truncation* — keep the first N tokens.
   - *Random masking*.
   - *Block masking* — keep one contiguous span.
   - *Syntax masking* — priority-keep nouns > adjectives > others.
4. **Sweep.** Three encoder sizes (S/B/L) x each strategy x image tokens in {196 -> 16} and text tokens in {32 -> 4}. Fixed recipe: 6.4 LAION-400M epochs, batch 32k, base lr 8e-6, gradient checkpointing, RandomResizedCrop 40-100%, AdamW (0.9, 0.95), wd 0.2, 1600 warmup steps, cosine decay. Followed by a 0.36-epoch full-resolution fine-tune at 224² with text length 32, base lr 8e-7.
5. **CLIPA at scale (Sec. 6).** Two-stage fine-tuning: 12.8B pre-train samples at (e.g.) 84² / 8 text tokens, then 512M samples at 224² with 30% random mask, then 128M samples at 336² with 40% mask. G/14 + DataComp-1B reaches 83.0% IN-1k.

![Image-token reduction strategies: original, random mask, grid mask, block mask, resize](/assets/images/paper/clips-inv-scale/page_003.png)
*Figure 2: The four image-token reduction strategies compared on a single example. Resize preserves the most information per surviving token; the three masking variants discard regions outright.*

![Text-token reduction strategies: truncation, random mask, block mask, syntax mask](/assets/images/paper/clips-inv-scale/page_004.png)
*Figure 3: The four text-token reduction strategies. Syntax masking is part-of-speech-aware (nouns > adjectives > others) and Pareto-dominates the others at extreme lengths.*

## Experimental Results

### Budget recipe vs OpenCLIP (Table 1, LAION-400M, 8xA100, all zero-shot)

| Model | Pre-train tokens | GPU-h | IN-1k | IN-V2 | IN-A | IN-R | ObjectNet | IN-Sk | COCO i->t | COCO t->i | Flickr i->t | Flickr t->i |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenCLIP-B/32 (re-eval) | full | 4,600 | 62.9 | 55.1 | 21.7 | 73.4 | 48.9 | 49.4 | 35.3 | 52.6 | 61.7 | 79.0 |
| OpenCLIP-B/16 (re-eval) | full | 10,700 | 67.1 | 59.6 | 33.2 | 77.9 | 51.5 | 52.4 | 38.3 | 55.4 | 65.5 | 83.3 |
| OpenCLIP-L/14 (re-eval) | full | 50,800 | 72.8 | 65.4 | 46.5 | 84.9 | 59.9 | 59.6 | 43.0 | 59.7 | 70.3 | 87.6 |
| **CLIPA-B/16 (I50, T16)** | reduced | **444** | 63.2 | 55.6 | 26.8 | 73.2 | 44.3 | 48.7 | 35.2 | 53.1 | 58.3 | 75.3 |
| **CLIPA-L/16 (I17, T16)** | reduced | **628** | 67.8 | 60.4 | 38.3 | 81.2 | 52.8 | 56.4 | 40.1 | 58.4 | 64.0 | 81.5 |
| **CLIPA-L/16 (I37, T8)** | reduced | **826** | 69.3 | 61.7 | 43.6 | 84.0 | 55.4 | 58.7 | 39.8 | 56.8 | 67.5 | 81.9 |

CLIPA-B/16 reaches +0.3% over OpenCLIP-B/32 at ~10x fewer GPU-h; CLIPA-L/16 reaches +0.7% over OpenCLIP-B/16 at ~17x fewer GPU-h.

![Inverse scaling law on image tokens across model sizes and reduction strategies](/assets/images/paper/clips-inv-scale/page_005.png)
*Figure 4: Inverse scaling law on the image side — larger models tolerate stronger token reduction across all four strategies. Resize is the flattest curve.*

![Retrieval (COCO) and OOD robustness corroborate the inverse scaling trend](/assets/images/paper/clips-inv-scale/page_006.png)
*Figure 5 (top) / Figure 6 (bottom): Retrieval and OOD robustness corroborate the image-side trend; a few cells on harder OOD sets are non-monotone.*

![Text-side inverse scaling and syntax-masking advantage](/assets/images/paper/clips-inv-scale/page_007.png)
*Figure 7: Inverse scaling holds for text tokens; syntax masking wins at extreme lengths. The curve dipping below zero for B/16 and L/16 around length 16 is the surprising "shorter text helps" finding.*

### Scaling (Table 2, LAION-2B and DataComp-1B; compute in PFLOPs x 1e12)

| Model | Data | Schedule | Compute | IN-1k |
|---|---|---|---|---|
| OpenCLIP-H/14 | LAION-2B | 32B@224² | 5.7 | 78.0 |
| OpenCLIP-G/14 | LAION-2B | 32B@224² + 6.7B | 29.8 | 80.1 |
| FLIP-H/14 (re-eval) | LAION-2B | 25.6B@224² + 128M | 2.4 | 78.4 |
| **CLIPA-H/14 (I36, T8)** | LAION-2B | 12.8B@84² + 128M@224² | **0.4** | 77.9 |
| **CLIPA-H/14 (I36, T8)** | LAION-2B | 12.8B@84² + 512M@224² + 128M@336² | **0.4** | **79.1** |
| **CLIPA-H/14 (I36, T8)** | DataComp-1B | 12.8B@84² + 128M@224² | 0.4 | 81.5 |
| **CLIPA-G/14 (I36, T8)** | DataComp-1B | 12.8B@84² + 512M@224² + 128M@336² | 0.9 | **83.0** |

OpenCLIP-G/14 -> CLIPA-G/14: 80.1 -> 83.0 at ~33x less compute headline (caveat: mixes resolution stages).

![Accuracy-compute Pareto and Table 1 efficiency headline](/assets/images/paper/clips-inv-scale/page_008.png)
*Figure 8 / Table 1: The accuracy-vs-compute Pareto frontier — CLIPA dominates at the academic budget regime.*

![Scaling: CLIPA at H/14 and G/14 on LAION-2B and DataComp-1B](/assets/images/paper/clips-inv-scale/page_009.png)
*Table 2: At scale, CLIPA-G/14 on DataComp-1B reaches 83.0% IN-1k with ~33x less compute than OpenCLIP-G/14.*

### Ablations / robustness highlights

- *Image-side ranking on L/16 at 75% reduction:* resize 68.9 > block 68.5 > random 67.6 > grid 67.3. **The MIM "random is best" intuition is inverted for CLIP.**
- *Text-side ranking on B/16 at length 4:* syntax mask (~3.0% drop) beats truncation (~4-5%), block (~4%), random (~4-5%).
- *Surprising finding:* for B/16 and L/16, shortening text from 32 -> 16 or 8 under syntax masking *improves* IN-1k accuracy — read by the authors as evidence that LAION captions are noisy and POS-priority filtering is effectively a denoiser.
- *Architecture transfer:* ConvNeXt-T -> ConvNeXt-B repeats the inverse scaling pattern under resize, suggesting it is not ViT-specific (but only two model points).
- *ARO (compositional, Tab. 14):* CLIPA-B/16 is slightly worse than OpenCLIP-B/16 on relation/attribute/order — an honest negative result flagged in Sec. 7.

## Limitations

Authors admit:

- ARO compositional benchmark: CLIPA-B/16 < OpenCLIP-B/16 on relation/attribute/order. Patchable with NegCLIP-style hard-negative fine-tuning but does not change pretrain conclusions.
- Web-data bias inheritance (Broader Impact).

Not addressed:

- **No variance / multiple-seed runs anywhere** — everything is single-run.
- No medical or scientific-domain evaluation. The claim that "large encoder needs few tokens" may not transfer to long, jargon-heavy radiology reports where load-bearing tokens are not nouns-first.
- The "shorter text helps" finding is confounded with caption noisiness — no controlled experiment on clean captions.
- Image-resize is presented as Pareto-dominant over masking, but the underlying compute equivalence (112² ≈ 75% mask) is FLOP-equivalent only for ViT, not for ConvNeXt and not for memory bandwidth at higher resolutions.
- The "33x speedup" mixes phases of different resolution; an apples-to-apples decomposition is missing — the honest figure is closer to ~15x.
- No ablation on batch size x token length interaction — the smaller-batch (32k vs OpenCLIP's 64k+) recipe may itself contribute to the GPU-hour savings independently of token length.
- Inverse scaling is *empirically* observed but not *theoretically* explained — no information-theoretic or representation-capacity argument is offered.

## Why It Matters for Medical AI

Every medical VLM that starts from a CLIP backbone — BiomedCLIP, PubMedCLIP, MedCLIP, BioViL — inherits CLIP's compute floor. CLIPA's image-resize trick (pretrain at 84² or 112², fine-tune at 224²/336²) is **orthogonally useful** for medical pretraining: radiology and pathology corpora are not the bottleneck on signal density per image (a chest X-ray retains diagnostic structure at 112²), and the resize recipe is data-agnostic. The text-side recipe is the **wrong default for medical pre-training** for two reasons: (i) medical captions (radiology reports, PMC figure captions) carry diagnostic signal in adjectives, modifiers, and prepositional phrases that syntax-masking-by-POS would discard, and (ii) machine-rewritten dense medical captions (LLaVA-Med, ShareGPT4V-style) explicitly pack the signal into longer-form text. A sensible recipe for a CLIPA-inspired biomedical CLIP run: keep the image-resize stage and the 2-stage fine-tune at 224²/336², but **do not aggressively truncate or syntax-mask the text** — pay the modest compute cost of full-length captions for the alignment quality.

## References

- **Paper:** Li, Wang, Xie. *An Inverse Scaling Law for CLIP Training.* NeurIPS 2023. arXiv:2305.07017.
- **Code:** https://github.com/UCSC-VLAA/CLIPA
- **Related:**
  - OpenCLIP — Ilharco et al., 2021. https://github.com/mlfoundations/open_clip
  - FLIP — Li et al., *Scaling Language-Image Pre-training via Masking*, CVPR 2023.
  - RECLIP — Li et al., *RECLIP: Resource-efficient CLIP by Training with Small Images*, TMLR 2023.
  - Long-CLIP — Zhang et al., ECCV 2024. The directionally-opposite "longer text helps" result on dense-caption alignment.
  - DataComp — Gadre et al., *DataComp: In Search of the Next Generation of Multimodal Datasets*, NeurIPS 2023.
