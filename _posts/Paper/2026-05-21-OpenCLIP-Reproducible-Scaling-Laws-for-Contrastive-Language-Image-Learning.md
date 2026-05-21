---
title: "Reproducible Scaling Laws for Contrastive Language-Image Learning"
excerpt: "An open three-axis CLIP sweep on LAION-2B fits power laws E = beta * C^alpha across classification, retrieval, linear probe, and fine-tune, with ViT-H/14 hitting 78.0% zero-shot ImageNet and a one-decade extrapolation projecting 81.9% for G/14 at 68B samples."
categories:
  - Paper
  - VLM-Alignment
  - Multimodal-Alignment
  - LLM
permalink: /paper/openclip-scale/
tags:
  - OpenCLIP
  - CLIP
  - Scaling-Laws
  - LAION
  - Contrastive-Learning
  - Vision-Language
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- A fully open three-axis sweep (model x data x samples-seen) on **LAION-400M / LAION-2B** with the OpenCLIP codebase fits power laws **E = beta * C^alpha** across zero-shot classification, retrieval, linear probing, and fine-tuning — using up to **34B samples seen and 1520 A100 GPUs**.
- The most useful practical finding is asymmetric: **samples-seen is the binding constraint** for unlocking the LAION-2B data-scale advantage — B/32 and B/16 show no LAION-400M -> LAION-2B gain at 3B/13B samples, but the gap opens at 34B.
- Task heterogeneity in the scaling exponents: **WIT-400M wins the slope on classification** (alpha = -0.16 to -0.24 vs LAION -0.11 to -0.13), but **LAION wins on retrieval** (Flickr30K alpha = -0.19 vs WIT -0.10; COCO -0.08 vs -0.05). Headline artifact is **ViT-H/14 on LAION-2B reaching 78.0% zero-shot ImageNet, 56.4% VTAB+, 73.4% MS-COCO R@5**.

## Motivation

Scaling laws for language (Kaplan, Chinchilla) and vision (Zhai et al.) have been mapped, but multi-modal contrastive training has lived in a private-data regime — ALIGN, BASIC, LiT, CoCa, LiMoE, Flamingo, and PaLI all train on proprietary corpora and frequently combine multi-stage encoders, which makes it impossible to separate "scale" from "data curation." The release of LAION-400M and LAION-2B opened the door to a fully reproducible CLIP sweep. The narrow scientific question is: do CLIP-family encoders obey clean power laws across model, data, and samples-seen, and does the pre-training distribution shift those exponents in task-dependent ways?

## Core Innovation

- **Three-axis grid.** Vary model (ViT-B/32, B/16, L/14, H/14, g/14), data (LAION-80M / 400M / 2B), and samples-seen (3B / 13B / 34B). Larger models are sparsely sampled because of compute: H/14 only at LAION-2B / 34B; g/14 only at LAION-2B / 13B.
- **Head-to-head against OpenAI WIT-400M.** Same architectures, same recipe, batch-size and LR schedule controls — what changes is the pre-training distribution. This is the lever for isolating data effects on the scaling exponent.
- **Pareto-frontier fits.** E = beta * C^alpha (C = GMAC-per-sample x samples-seen) is fit only on the Pareto frontier within log-spaced compute bins, so dominated points do not contaminate the slope.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Power-law scaling E = beta * C^alpha holds across model, data, samples-seen for contrastive VLMs | Linear log-log fits on Pareto frontier across 4 task families (Figs 1a, 1b, 2, 3, 4, 5) | ImageNet + 5 robustness sets, COCO, Flickr30K, CIFAR-100, VTAB, 8 FT tasks | ⭐⭐⭐ |
| C2 | The **pre-training distribution** drives task-specific scaling exponents — WIT-400M scales better on classification, LAION on retrieval | Matched ViT recipe: WIT alpha steeper on ImageNet (-0.16 vs -0.11) and robustness (-0.24 vs -0.13); LAION steeper on COCO (-0.08 vs -0.05) and Flickr30K (-0.19 vs -0.10); batch-size control rules out one alternative | ImageNet, COCO, Flickr30K | ⭐⭐ — direction consistent across two retrieval and two classification benchmarks, but no controlled re-filter of LAION is run, so causality remains hypothesis |
| C3 | Bottleneck effects exist — fixing one axis hides gains on another | B/32 and B/16 show no LAION-400M -> 2B gain at 13B samples but do at 34B; L/14 on LAION-400M does not improve 13B -> 34B but does on LAION-2B (Sec 4.1, App Tables 19, 21, 22) | ImageNet, MS-COCO | ⭐⭐⭐ |
| C4 | Zero-shot ImageNet gains come with matched robustness gains | alpha_OpenCLIP ImageNet = -0.11 vs robustness = -0.13 (close) | ImageNet + V2/R/Sketch/A + ObjectNet | ⭐⭐ — true for OpenCLIP, but for WIT the robustness slope is *steeper* than ImageNet (-0.24 vs -0.16), which the abstract glosses over |
| C5 | OpenCLIP H/14 on LAION-2B is the largest public CLIP and reaches 78.0% zero-shot ImageNet | Table 1 (one number, artifact shipped) | ImageNet, VTAB+, MS-COCO | ⭐⭐⭐ |
| C6 | Extrapolated ViT-G/14 at 68B samples will reach 81.92% zero-shot ImageNet top-1 | One-step linear extrapolation of the fitted power law (Fig 8, Table 4) | ImageNet (predicted) | ⭐ — no held-out validation, no CI, no saturation analysis |
| C7 | Batch-size differences (32K vs 88K) do not confound the WIT-vs-LAION comparison | Control sweep, 0.2-0.5% delta (Sec 3.2, App B.2.3) | ImageNet | ⭐⭐ — only one nuisance axis tested; LR, weight decay, prompt set differences remain |
| C8 | LAION-80M / LAION-400M are clean random subsets of LAION-2B | Random-400M-of-2B reference run matches LAION-400M B/32 within noise (App B.2.3) | LAION zero-shot ImageNet | ⭐⭐ |
| C9 | Test-set contamination via pHash is small enough not to confound results | pHash overlap <= 5.15% on ImageNet-Sketch, <= 1.5% elsewhere (Table 3, Fig 7); prior work argues duplicates do not move metrics much [55, 84] | ImageNet family, COCO, Flickr30K, CIFAR | ⭐⭐ — pHash is a weak detector (false negatives on heavy edits / translations); 5.15% on Sketch is non-trivial |
| C10 | Fine-tuned OpenCLIP H/14 reaches 88.5% ImageNet top-1, comparable to the best public-pretrained ImageNet models | Sec 4.4, App B.2.2 | ImageNet + 5 robustness sets | ⭐⭐⭐ |

**Honest read.** C1 and C3 are the load-bearing scientific contributions and are well-supported across multiple tasks and multiple model sizes. C5 and C10 are artifact claims and they work. C2 — the most-cited downstream takeaway, "dataset shapes the scaling exponent" — is suggestive but uncontrolled: the authors do not re-curate LAION with a different filter and re-fit alpha, which is the experiment that would actually pin causation. The batch-size control (C7) rules out only one alternative. C6, the 81.9% extrapolation that has been heavily quoted, is a one-decade linear extrapolation in log space with no error bars and no held-out check — treat it as a hypothesis, not a result. Critically, **no scaling fit is repeated across seeds anywhere in the paper, and no confidence interval is reported on any alpha**. Given 0.2-0.5% batch-size noise alone, several of the closer LAION-vs-WIT contrasts (VTAB linear probe -0.04 vs -0.06; CIFAR-100 full -0.19 vs -0.20) may not be statistically meaningful. The large contrasts (Flickr30K -0.19 vs -0.10; robustness -0.13 vs -0.24) are wide enough that variance is unlikely to flip the sign.

## Method & Architecture

![OpenCLIP zero-shot scaling — Figures 1a and 1b on page 3](/assets/images/paper/openclip-scale/page_003.png)
*Figure 1 (page 3 of the PDF): Figure 1a (top) plots zero-shot ImageNet and ImageNet-robustness error vs total compute — WIT scales faster (alpha = -0.16 / -0.24) than LAION (-0.11 / -0.13). Figure 1b (bottom) plots zero-shot retrieval on MS-COCO and Flickr30K — LAION scales faster (alpha = -0.08 / -0.19) than WIT (-0.05 / -0.10), reversing the classification ordering.*

The training recipe is standard CLIP InfoNCE with AdamW (beta_1 = 0.9, beta_2 = 0.98, weight decay 0.2) on cosine LR, re-tuned per samples-seen budget so short runs are not under-trained. Global batches are 86-88K on ~1000 A100s. Mixed precision is used throughout, but L/14, H/14, and g/14 require a **float16 -> bfloat16** switch to eliminate loss spikes — LR reduction, schedule changes, and grad clipping did not help. The training fleet peaked at **1520 A100s** on JUWELS Booster (Juelich) plus Stability AI's AWS cluster. The compute axis is reported as **C = GMAC-per-sample x samples-seen**, a FLOP proxy.

Four evaluation regimes feed the scaling fits:

1. Zero-shot classification on ImageNet + 5 distribution-shift sets (V2, R, Sketch, A, ObjectNet) + VTAB+ (35 datasets).
2. Zero-shot retrieval on MS-COCO and Flickr30K, R@5.
3. Linear probes (softmax regression with Adam) on ImageNet / CIFAR-100 in 10-shot, 25-shot, full, plus VTAB.
4. Fine-tuning on ImageNet (direct or via ImageNet-12k intermediate) and joint multi-task fine-tuning on Cars / DTD / EuroSAT / GTSRB / MNIST / RESISC45 / SUN397 / SVHN.

## Experimental Results

### Headline zero-shot at 224 px (Table 1)

| Model | Data | Arch | ImageNet | VTAB+ (avg 35) | MS-COCO R@5 (image) |
|---|---|---|---:|---:|---:|
| OpenAI CLIP [55] | WIT-400M | ViT-L/14 | 75.5 | 55.8 | 61.1 |
| OpenCLIP (ours) | LAION-2B | ViT-L/14 | 75.2 | 54.6 | 71.1 |
| **OpenCLIP (ours)** | **LAION-2B** | **ViT-H/14** | **78.0** | **56.4** | **73.4** |

OpenCLIP H/14 on LAION-2B matches WIT-400M L/14 on classification (78.0 vs 75.5 ImageNet, 56.4 vs 55.8 VTAB+) and decisively wins on retrieval (73.4 vs 61.1 MS-COCO R@5).

### Power-law fits E = beta * C^alpha

Lower |alpha| = flatter scaling = worse returns to compute. The point of the table is not who is higher today but who has steeper slope, i.e. who keeps gaining with more compute.

| Task | OpenCLIP (LAION) alpha | OpenAI CLIP (WIT) alpha | Winner on slope |
|---|---:|---:|---|
| ImageNet zero-shot error | -0.11 | **-0.16** | WIT |
| ImageNet robustness (avg of 5) | -0.13 | **-0.24** | WIT |
| MS-COCO retrieval (100 - R@5) | **-0.08** | -0.05 | LAION |
| Flickr30K retrieval (100 - R@5) | **-0.19** | -0.10 | LAION |
| Linear probe ImageNet (full) | -0.13 | **-0.20** | WIT |
| Linear probe CIFAR-100 (full) | -0.19 | **-0.20** | WIT (marginal) |
| VTAB linear probe (avg) | -0.04 | **-0.06** | WIT (marginal) |
| Fine-tune ImageNet | -0.09 | **-0.13** | WIT |
| Fine-tune ImageNet robustness | -0.17 | **-0.23** | WIT |
| Joint 8-task FT (zero-shot init) | -0.13 | **-0.17** | WIT |
| Joint 8-task FT (after FT) | -0.16 | **-0.23** | WIT |

The qualitative pattern is sharp: WIT wins the slope on every classification family (zero-shot, linear probe, fine-tune, robustness, joint 8-task), while LAION wins on both retrieval benchmarks. The paper's hypothesis is that the LAION filter — a CLIP-score threshold from OpenAI's own ViT-B/32 — is implicitly biased toward retrieval-style image-text alignment.

### Linear-probe scaling (page 8)

![Linear-probe and VTAB scaling — Figures 2 and 3 on page 8](/assets/images/paper/openclip-scale/page_008.png)
*Figures 2 and 3 (page 8 of the PDF): Figure 2 — linear-probe error on ImageNet (top row) and CIFAR-100 (bottom row) in 10-shot, 25-shot, and full-dataset regimes. Power-law slopes persist with limited labels. Figure 3 — VTAB linear-probe error scales at alpha = -0.04 (OpenCLIP) vs -0.06 (OpenAI CLIP), the gentlest slope in the paper and a useful upper bound on data-scaling returns for transfer.*

### Fine-tuning scaling (page 10)

![Fine-tune ImageNet and joint 8-task scaling — Figures 4 and 5 on page 10](/assets/images/paper/openclip-scale/page_010.png)
*Figures 4 and 5 (page 10 of the PDF): Figure 4 — fine-tuned ImageNet and robustness error vs fine-tune GMACs; WIT pre-training keeps its scaling-slope lead through full fine-tuning. Figure 5 — joint fine-tuning on eight diverse downstream tasks; error scales smoothly with pre-train compute under both zero-shot init (left) and fine-tuned init (right).*

### Extrapolations (Table 4)

Linear extrapolation of the fitted power law to **68B samples seen**:

| Model | ImageNet zero-shot top-1 (predicted) | MS-COCO R@5 (predicted) |
|---|---:|---:|
| ViT-H/14 @ 68B | 79.73 | 75.10 |
| ViT-g/14 @ 68B | 80.66 | 75.85 |
| **ViT-G/14 @ 68B** | **81.92** | **76.99** |

These are the heavily-cited "81.9% public CLIP" numbers. They are **not** trained models — they are one-step log-linear extrapolations off the fitted alpha, with no held-out validation point and no CI. Use them as a planning ceiling, not a measurement.

### Findings worth highlighting

- **Bottlenecks are asymmetric.** B/32 and B/16 show no LAION-400M -> 2B gain at 3B / 13B samples seen, but the gap opens at 34B. Symmetrically, L/14 on LAION-400M does not improve going 13B -> 34B (data bottleneck), but does on LAION-2B. The practical implication: samples-seen is the axis to push first if you want to cash in a data-scale upgrade.
- **Robustness scales with accuracy on LAION, faster than accuracy on WIT.** alpha_OpenCLIP (ImageNet) = -0.11 vs alpha_OpenCLIP (robustness) = -0.13 — near-proportional. alpha_WIT (ImageNet) = -0.16 vs alpha_WIT (robustness) = -0.24 — robustness scales *faster*. WIT's curation appears to have additional out-of-distribution payoff that LAION's CLIP-score filter does not reproduce.
- **Batch-size control.** 32K -> 88K shifts accuracy by 0.2-0.5%, so the slope gap is not a batch-size artifact.
- **Precision matters at scale.** Training fails (loss spikes) for L/14 and up under float16; bfloat16 fixes it. Practical reproducibility note, not a scientific claim.

### Duplicate audit

![Figure 7 pHash near-duplicate grid](/assets/images/paper/openclip-scale/fig_p022_01.png)
*Figure 7: pHash near-duplicate audit between downstream test sets and LAION-400M — top row downstream-set images, bottom row matched LAION-400M neighbours. Duplicates survive blur, recolour, text overlay, and crop; the last two columns are false positives on uniform-background ImageNet-Sketch images.*

Max overlap is 5.15% on ImageNet-Sketch and 3.80% on ImageNet-R; most other test sets are below 1.5% (Table 3). pHash is a weak detector — false negatives on heavy edits or translations are likely — but prior work [55, 84] argues that removing duplicates does not appreciably move the metrics.

## Limitations

**Authors admit:**

- Sparse sampling of the (model x data x samples-seen) grid; full HP tuning at large scale is unaffordable, mitigated by control runs.
- WIT-400M is private, so OpenAI CLIP has only three architecture points and cannot be re-fit at H/14 / g/14 scale.
- pHash duplicate check is approximate.
- Extrapolations may saturate; predictions are only "close to measured scales."

**Not addressed (auditor's flags):**

- **No variance or seed repeats** on any alpha fit; no CI on any reported coefficient. Several closer LAION-vs-WIT deltas (VTAB -0.04 vs -0.06; CIFAR-100 full -0.19 vs -0.20) are within plausible run-to-run noise.
- **No causal test of the dataset hypothesis (C2).** Re-filtering LAION with a stronger L/14 model and re-fitting — which the authors themselves propose in the Discussion — is the experiment that would pin causation.
- **Text-encoder scale is not separately ablated.** Text size scales with vision size via the architecture table, so "text encoder is a bottleneck" is not ruled out as an alternative explanation for plateaus.
- **Compute axis is GMAC, not actual FLOPs or wall-clock**, and mixes float16 vs bfloat16 runs — the precision switch quietly changes the regime for L/14+.
- **No language coverage beyond English** (LAION-2B is the English subset), so retrieval scaling advantages may not transfer to multilingual settings.
- **No medical / domain-shift evaluation** — directly relevant to downstream biomedical VLMs (BiomedCLIP, PMC-CLIP, MedCLIP) that bootstrap from OpenCLIP weights.
- **No optimal-allocation rule** is derived; the paper finds bottlenecks but does not produce a closed-form Chinchilla-style "for compute C, train an N-parameter model on D samples."
- **The 81.9% extrapolation is one decade in log space with no held-out validation point.** Saturation regimes are untested.

## Why It Matters for Medical AI

Most clinical VLMs in the public ecosystem — BiomedCLIP, PMC-CLIP, MedCLIP, CONCH, OmiCLIP — bootstrap from OpenCLIP weights or recipes. This paper is the empirical basis for two practical choices a medical-AI builder has to make: (i) which OpenCLIP checkpoint to fine-tune from for a given downstream task (the classification-vs-retrieval slope split suggests the right answer depends on whether your downstream is retrieval-shaped or classification-shaped), and (ii) how much marginal pre-training compute or data buys how much downstream gain (the bottleneck story says: do not scale data without scaling samples-seen). The paper also shows that LAION's CLIP-score filter is *not* neutral — it appears retrieval-biased — which matters because that same filter sits upstream of every biomedical CLIP that re-uses LAION-pretrained weights without re-curating.

## References

- Paper: Cherti, Beaumont, Wightman, Wortsman, Ilharco, Gordon, Schuhmann, Schmidt, Jitsev. "Reproducible scaling laws for contrastive language-image learning." CVPR 2023. arXiv:[2212.07143](https://arxiv.org/abs/2212.07143)
- Code & checkpoints: [github.com/LAION-AI/scaling-laws-openclip](https://github.com/LAION-AI/scaling-laws-openclip)
- OpenCLIP repo: [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- LAION-5B / LAION-2B: Schuhmann et al. NeurIPS 2022, arXiv:[2210.08402](https://arxiv.org/abs/2210.08402)
- Original CLIP: Radford et al. ICML 2021, arXiv:[2103.00020](https://arxiv.org/abs/2103.00020)
- Vision scaling reference: Zhai et al. "Scaling Vision Transformers." CVPR 2022, arXiv:[2106.04560](https://arxiv.org/abs/2106.04560)
- Compute-optimal training: Hoffmann et al. ("Chinchilla"), arXiv:[2203.15556](https://arxiv.org/abs/2203.15556)
