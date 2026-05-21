---
title: "Why are Visually-Grounded Language Models Bad at Image Classification?"
excerpt: "A diagnostic study showing VLMs (LLaVA, BLIP, GPT-4V) trail their own CLIP encoder by 20-30pts on ImageNet because of alignment-data scarcity, not architecture or objective: LLaVA1.5-7B 22.8 -> 84.4 on ImageNet and +11.8pt on ImageWikiQA after a projector-only fix."
categories:
  - Paper
  - VLM-Alignment
  - Multimodal-Alignment
  - LLM
tags:
  - VLM
  - LLaVA
  - CLIP
  - Image-Classification
  - Alignment
  - Instruction-Tuning
  - Projector
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
permalink: /paper/vlm-classify/
---

## TL;DR
- Visually-grounded LLMs (LLaVA, BLIP-2, InstructBLIP, GPT-4V, Gemini, Claude) underperform **their own CLIP/EVA vision encoder** by **20-30 points on ImageNet** and even more on Flowers102 / StanfordCars - the very models whose image tower is CLIP cannot match CLIP at zero-shot classification.
- The root cause is neither inference strategy, prompt format, lost visual information, nor the generative objective. It is **alignment-data coverage**: per-class accuracy is Spearman **rho = 0.76** correlated with class frequency in LLaVA's training corpus (classes with >10K mentions reach 82.7%, classes with <10 mentions drop to 3.0%).
- The fix is small and clean: adding classification-style data to instruction tuning lifts **LLaVA1.5-7B from 22.8% to 84.4% on ImageNet** and **+11.8pt absolute on the authors' new ImageWikiQA benchmark (38.0 -> 49.8)**. Projector-only fine-tuning matches a CLIP-L linear probe (85.7 vs 85.2). The paper is **NeurIPS 2024**, not ICLR 2025.

## Motivation
Image classification is the historical anchor of computer vision and a prerequisite for object-centric QA, knowledge grounding, and downstream reasoning - a clinical assistant cannot say "this lesion is malignant" if it cannot first identify the lesion type. Yet a VLM that literally wraps CLIP with a projector and an LLM collapses on the task CLIP solves best. Prior analyses of VLM weaknesses focused on architecture (Q-Former vs MLP), vision encoder choice (Tong et al. on CLIP failure pairs), or instruction recipe; none isolated whether the bottleneck is the **alignment data itself**. This paper runs that isolation experiment.

![The three-act argument of the paper: VLMs trail CLIP, six hypotheses are tested, only data survives, and a data fix closes the gap](/assets/images/paper/vlm-classify/page_002.png)
*Figure 1: The three-act argument. (left) VLMs trail CLIP across a wide model zoo; (middle) six hypotheses are tested and only data survives; (right) adding classification data to LLaVA fixes both classification and downstream QA. Source: paper Figure 1.*

## Core Innovation
The contribution is **diagnostic, not algorithmic**:

1. **A clean ablation tree over six hypotheses** for the VLM/CLIP gap - prompting, label-set size, generation-vs-scoring inference, hidden-state information loss, generative-objective expressivity, and finally training-data coverage - run on the same two open VLMs (LLaVA1.5-7B, BLIP-2) so that each hypothesis can be falsified with a matched control.
2. **A frequency-correlation analysis on LLaVA's full public training corpus** (the only fully open VLM data). Per-class ImageNet accuracy is regressed against per-class mention frequency, with CLIP as the null model (frequency-independent) and post-FT LLaVA as the recovery test (correlation should collapse).
3. **ImageWikiQA**, a 2,000-question 4-way MCQ benchmark built by parsing Wikipedia for each ImageNet class, GPT-4-generating five questions, replacing the class name with "this object," and filtering to questions GPT-4 can answer **with** the class name but **not** without it. Random = 25%, max-frequency baseline = 25.9%.
4. **A drop-in fix**: jointly fine-tune on ImageNet 1.28M classification triples + the original 665K LLaVA instruction-tuning corpus. **Projector-only** training is recommended over LoRA on the LLM because LoRA shows recurrent loss spikes (Appendix figures 8/9 across 4 reruns) while projector-only is monotone.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | Public + proprietary VLMs significantly underperform their own CLIP encoders at classification. | Table 1: 10 VLMs vs 2 CLIPs, open- and closed-world. Gap 20-30pt on ImageNet, larger on Flowers102 / StanfordCars. | ⭐⭐⭐ |
| C2 | The gap is not explained by prompt sensitivity, label-set size, or generation-vs-scoring inference. | Table 2 + Figure 2: prompt variants +/-3%, CoT marginal; probabilistic inference + CFG closes only ~25 of ~52pt; tested on LLaVA1.5-7B + BLIP2-2.7B only. | ⭐⭐ - no proprietary models in the inference ablation. |
| C3 | Classification-relevant information is preserved in VLM hidden states. | Table 3 left: last-layer probe **77.1% (LLaVA) / 81.4% (BLIP-2)** on ImageNet vs **85.2% CLIP** probe. | ⭐⭐⭐ for "mostly preserved"; ⭐⭐ for "as much as CLIP" - the 8-point probe gap is glossed over. |
| C4 | The generative objective is sufficient for classification given the right data. | Table 3 right: **LLaVA1.5-7B projector FT 85.7% matches CLIP-L linear 85.2%** on ImageNet; BLIP-2 projector FT 88.0% beats EVA-G linear 86.5%. | ⭐⭐⭐ |
| C5 | The primary cause is data: class frequency in VLM training strongly correlates with per-class accuracy. | Figure 3: **Spearman rho = 0.76** on ImageNet; CLIP-L per-class accuracy uncorrelated with LLaVA counts; correlation collapses after FT. | ⭐⭐⭐ on LLaVA; ⭐ on the implicit generalization to GPT-4V / Gemini / Claude (corpus inaccessible, diagnosis is by analogy). |
| C6 | Data **type** doesn't matter - captioning data containing the class works as well as classification data. | Table 4: caption-FT within 2-6pt of classification-FT on Flowers / Cars / Caltech. | ⭐⭐ - **ImageNet, the headline benchmark, is absent from Table 4**. |
| C7 | Improving classification improves general object-centric QA (+11.8pt on ImageWikiQA). | Table 5: LLaVA1.5-7B **38.0 -> 49.8** with joint FT. | ⭐⭐ - one VLM, no error bars, one benchmark the authors built. |
| C8 | Fine-tuning on classification alone causes catastrophic forgetting; combined data is required. | Table 5: ImageNet-only FT drops ImageWikiQA to **30.6 (-7.4pt vs zero-shot)** while combined FT reaches 49.8. | ⭐⭐ - single comparison, no VQAv2 / MM-Vet / MMMU follow-up. |
| C9 | Projector-only fine-tuning matches LoRA and is more stable. | Table 3 + Appendix loss curves (Figs p.18, p.21): 4 LoRA reruns show recurrent loss spikes; projector-only is monotone. | ⭐⭐ - stability claim is visual, not quantified. |

**Honest analysis.** The core thesis - that the bottleneck is alignment data, not architecture or objective - is the most robust contribution: it survives an inference ablation, a probe-and-fine-tune ablation, and a frequency-correlation analysis with a clean CLIP control. Three weak points to flag in the same breath as the headline:

1. **The proprietary-VLM diagnosis is indirect.** The mechanism (data correlation, projector probe) is only demonstrated for LLaVA and BLIP-2 because nothing else is open. The claim that GPT-4V / Gemini / Claude trail CLIP "for the same reason" is a plausibility argument; their gap could equally come from RLHF reward shaping or instruction-tuning bias.
2. **ImageWikiQA is GPT-4-built, GPT-4-filtered, and GPT-4 scores 100% with the class name** (circularity). The benchmark's difficulty distribution is shaped by GPT-4's own competence; the +11.8pt is encouraging but should not be read as "general capability improvement" - it is improvement on a benchmark whose ceiling is what GPT-4 considers verifiable.
3. **Catastrophic forgetting is real and the mitigation is empirical.** Fine-tuning on ImageNet alone makes ImageWikiQA *worse* (38.0 -> 30.6). The combined-data fix at the single tested mix ratio (1.28M classification : 665K instruction) is the only working point reported; no mix-ratio sweep, no other downstream benchmarks (VQAv2, MM-Vet, MMMU) to confirm the gain transfers.

Strongest claims: **C1, C3, C4, C5 (within LLaVA scope)**. Weakest: **C6** (ImageNet missing) and the proprietary-VLM extension of **C5**.

## Method & Architecture

The "method" here is the diagnostic protocol. Concretely:

1. **Benchmarking (Section 2).** 10 VLMs (3 proprietary, 7 public) vs CLIP-ViT-L/14-336px and EVA-ViT-G/14 on **ImageNet, Flowers102, StanfordCars, Caltech101**. Two protocols: *open-world* ("What is in the image?") and *closed-world* (prompt includes the full label list). VLM success = ground-truth class name appears in the free-form output. CLIP uses "a photo of a `<class>`" with cosine similarity, no prompt ensembling.

2. **Inference hypothesis sweep (Section 3.1).** Two prompt rewordings, fixed-vs-random label order, zero-shot CoT, label-set size $K \in \{2, 5, 20, 100\}$ (always including ground truth), and probabilistic inference scoring $p(\text{class} \mid \text{image}, \text{prompt})$ as either a token sum or a token average. Probabilistic ranking is further refined with **classifier-free guidance**:

   $$\text{score}(c) = t \cdot p(c \mid \text{image}, \text{prompt}) + (1 - t) \cdot p(c \mid \text{prompt}).$$

3. **Training hypothesis sweep (Section 3.2).** *Feature probing* on the VLM's last-layer hidden states with two poolings (last-token / mean-token), and *generative fine-tuning* with the template `"<image> What type of object is in this photo? <class name>"` under two configurations: projector-only (MLP for LLaVA, Q-Former for BLIP) and projector + LoRA on the LLM.

4. **Data hypothesis (Section 3.3).** Parse the public LLaVA1.5-7B two-stage corpus (~558K caption + 665K instruction). For each ImageNet class, count name + synonym occurrences (Appendix B.6) and compute **Spearman correlation** with per-class accuracy. Re-fine-tune with a per-class *caption* variant (GPT-4 generates captions that must contain the ground-truth label) to isolate format vs presence.

5. **Intervention + ImageWikiQA (Section 4).** Joint FT on **ImageNet 1.28M + LLaVA 665K**; evaluate on the 2,000-question, 4-way MCQ ImageWikiQA. The benchmark is filtered: a question is kept only if GPT-4 answers correctly with the class name AND incorrectly without it. Human validation: 96.5% accuracy with class name + Wikipedia. Random = 25.0%, max-frequency = 25.9%.

![Closed-world accuracy as a function of label-set size: LLaVA only closes the gap at K=2](/assets/images/paper/vlm-classify/fig_p005_05.png)
*Figure 2: Closed-world accuracy vs label-set size $K$. LLaVA1.5-7B closes about 40 points of the gap on ImageNet only when the candidate set shrinks to two - a non-solution at deployment scale where the label set is exactly what we want the model to span. Source: paper Figure 2a.*

## Experimental Results

### Headline classification gap (Table 1, closed-world top-1, %)

| Model | ImageNet | Flowers102 | StanfordCars | Caltech101 |
|-------|---------:|-----------:|-------------:|-----------:|
| BLIP2-2.7B | 14.2 | 2.7 | - | 22.3 |
| InstructBLIP-7B | 26.8 | - | - | 58.4 |
| InstructBLIP-13B | 20.0 | - | - | 59.5 |
| **LLaVA1.5-7B** | **10.2** | **0.0** | **-** | **62.1** |
| LLaVA-NeXT-V7B | 8.5 | 0.0 | - | 66.6 |
| LLaVA1.5-13B | 7.2 | 0.1 | - | 70.9 |
| LLaVA-NeXT-M7B | 16.1 | 3.6 | - | 77.3 |
| Claude-3 | 58.3 | 45.1 | 51.1 | 90.9 |
| Gemini-Pro | 62.0 | 66.6 | 56.0 | 91.6 |
| GPT-4 | 79.9 | 58.2 | 60.6 | 94.2 |
| CLIP-L | 76.0 | 77.5 | 74.8 | 95.8 |
| EVA-G | 81.0 | 90.2 | 79.2 | 97.9 |

Open-world numbers are even lower (LLaVA1.5-7B open-world Flowers = 5.9%, Cars = 0.0%). Proprietary VLMs are within 10-20pt of CLIP-L; open VLMs are catastrophically below it.

### Inference sweep (Table 2, LLaVA1.5-7B ImageNet open-world)

| Setting | Accuracy |
|---|---:|
| Base prompt | 22.8 |
| Prompt variant 1 | 19.7 |
| Prompt variant 2 | 21.6 |
| Prob. inference (token sum) | 34.8 |
| Prob. inference (token avg) | 35.3 |
| + Classifier-free guidance | 47.6 |
| CLIP-L (reference) | 74.8 |

Even the most aggressive scoring scheme leaves a ~27-point gap. Inference is **not** the bottleneck.

### Training sweep (Table 3, top-1 %)

| Method | ImageNet | Flowers | Cars | Caltech |
|--------|---------:|--------:|-----:|--------:|
| LLaVA1.5-7B last-token probe | 76.9 | 94.5 | 81.0 | 96.7 |
| LLaVA1.5-7B mean-token probe | 77.1 | 96.2 | 82.8 | 97.3 |
| BLIP2-2.7B mean-token probe | 81.4 | 98.9 | 92.6 | 98.0 |
| CLIP-L linear FT | 85.2 | 98.6 | 91.5 | 97.6 |
| **LLaVA1.5-7B projector FT** | **85.7** | **97.6** | **90.4** | **97.5** |
| EVA-G linear FT | 86.5 | 99.2 | 94.3 | 98.5 |
| BLIP2-2.7B projector FT | 88.0 | 99.0 | 93.9 | 98.8 |

Two clean findings: (a) probing the VLM's last layer recovers ~77% on ImageNet vs 85% for CLIP probe - **most information is preserved**; (b) projector-only FT **matches or beats** linear-probed CLIP, **eliminating the architectural gap entirely**.

### Data-frequency correlation (Figure 3)

![Per-class ImageNet accuracy binned by class frequency in LLaVA training data: LLaVA rises from 3% to 82.7%, CLIP and post-FT LLaVA are flat](/assets/images/paper/vlm-classify/fig_p007_01.png)
*Figure 3 (the money figure): per-class ImageNet accuracy binned by class frequency in LLaVA training data. LLaVA's curve climbs from 3.0% (<10 mentions) to 82.7% (>10K mentions); CLIP-L and fine-tuned LLaVA are flat. Spearman rho = 0.76. This is the paper's central evidence that the bottleneck is data coverage. Source: paper Figure 3.*

### Data-type ablation (Table 4)

| LLaVA1.5-7B variant | Flowers | Cars | Caltech |
|---|---:|---:|---:|
| Zero-shot | 5.9 | 0.0 | 47.1 |
| FT, classification template | 97.6 | 90.4 | 97.5 |
| FT, GPT-4 caption data | 92.0 | 85.4 | 95.7 |
| CLIP-L FT (reference) | 98.6 | 91.5 | 97.6 |

Caption data lands within ~5pt of classification-template data: **presence of the class name** drives the lift, not template formatting. ImageNet is notably absent from this table.

### ImageWikiQA (Table 5)

| Model | Accuracy (%) |
|---|---:|
| Random / Max-freq baseline | 25.0 / 25.9 |
| GPT-4 with ground-truth class | 100.0 |
| Human with GT class + Wikipedia | 96.5 |
| LLaVA1.5-7B with GT class | 55.9 |
| BLIP2-2.7B (image) | 21.7 |
| LLaVA1.5-7B (image) | 38.0 |
| LLaVA-NeXT-M7B | 41.9 |
| Gemini-Pro | 49.1 |
| Claude-3 | 54.3 |
| GPT-4 | 61.2 |
| LLaVA1.5-7B FT on ImageNet only | 30.6 |
| **LLaVA1.5-7B FT on ImageNet + LLaVA** | **49.8** |

Two non-negotiable observations: GPT-4 with the ground-truth class scores **100%** (circularity), and **ImageNet-only FT regresses to 30.6 (-7.4pt)** - catastrophic forgetting. The +11.8pt headline only materializes under joint training.

## Limitations

**Authors acknowledge.**
- Fine-tuning on classification alone causes catastrophic forgetting; the joint-data mitigation is empirical, not principled. No mix-ratio sweep.
- Probabilistic inference is expensive (one forward pass per class) and still leaves a 25-30pt gap.
- The frequency analysis is restricted to LLaVA because it is the only fully open VLM corpus.

**Not addressed.**
- **No medical, satellite, OCR, or low-resource-language classification.** The data-bottleneck thesis is established within Western natural-image distributions only.
- **No mechanistic explanation** for *why* the projector specifically needs more data than the LLM's text pretraining - the geometric reason features are "encoded but undecodable" is left as future work.
- **No comparison with concurrent alignment-data scaling work** (PaliGemma, Cambrian-1, MM1) at matched budgets.
- **No statistical variance.** No seeds, no confidence intervals on Tables 1-5; per-image bootstrap CIs would be cheap. A +11.8 absolute on 2,000 MCQs is comfortably above noise, but per-class bucket plots would benefit from confidence bands.
- **ImageWikiQA is self-built and GPT-4 dependent**; the difficulty distribution is filtered by GPT-4's own competence, so the +11.8pt should not be over-extrapolated.
- **Proprietary-VLM diagnosis is indirect**; GPT-4V / Gemini / Claude cannot be fine-tuned or audited, so the data thesis is inferred for them rather than demonstrated.

## Why It Matters for Medical AI
Clinical VLMs (LLaVA-Med, Med-Flamingo, PathChat) inherit the LLaVA recipe almost verbatim - CLIP-style encoder + projector + LLM, with instruction-tuning data dominated by reports and dialogues rather than classification. This paper's prediction is direct: a clinical VLM that cannot reliably name the lesion class (CXR finding, dermatology lesion, mushroom in a tox-screen image) will fail at downstream object-centric QA even when its image encoder *does* contain the discriminative signal. The practical recipe transfers cleanly:

- Audit class-frequency coverage in the medical instruction corpus before scaling compute.
- Mix classification-format data (image + label triplets, or label-bearing captions per Table 4) into the instruction-tuning mixture rather than tuning the LLM alone.
- Default to **projector-only** fine-tuning for stability when adding classification data; reserve LoRA-on-LLM for cases where text-side reasoning behavior must shift.
- Build downstream knowledge-grounded benchmarks (the clinical analog of ImageWikiQA) where the class name unlocks the question, so improvements in identification can be measured separately from improvements in language.

The big caveat for medical readers: the paper never evaluates on medical images, so the *magnitude* of the lift is unknown. Long-tail class distributions in medical data are far more severe than ImageNet's, and the joint-mixture ratio that works there may not transfer.

## References
- Paper: [arXiv:2405.18415](https://arxiv.org/abs/2405.18415) (NeurIPS 2024; v2, 3 Nov 2024)
- Project page: [yuhui-zh15.github.io/VLMClassifier-Website](https://yuhui-zh15.github.io/VLMClassifier-Website/)
- LLaVA1.5 training corpus (used for the frequency analysis): [llava-vl.github.io](https://llava-vl.github.io/)
- Related: Tong et al., *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*, CVPR 2024
- Related: Bujwid & Sullivan, *Large-Scale Zero-Shot Image Classification from Rich and Diverse Textual Descriptions*, 2021 (Wikipedia parse used by ImageWikiQA)
