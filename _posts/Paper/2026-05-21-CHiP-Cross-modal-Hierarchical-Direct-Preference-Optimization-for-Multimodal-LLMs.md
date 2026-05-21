---
title: "CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs"
excerpt: "Adding a cross-modal image-preference branch and three-granularity textual loss to DPO drops Object HalBench response-level hallucination from 11.0 to 4.9 on LLaVA-1.6-7B and 13.1 to 6.2 on Muffin-13B; the ablation shows the visual branch, not the hierarchy, is doing most of the lifting."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/chip/
tags:
  - CHiP
  - DPO
  - mDPO
  - MLLM
  - Hallucination
  - Alignment
  - Vision-Language
  - Cross-modal
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- Vanilla multimodal DPO over text-only preference pairs barely moves image-text alignment and still confuses hallucinated vs. ground-truth descriptions. CHiP bolts on (a) a **cross-modal image-preference branch** `L_DPOv` that uses a diffusion-perturbed "rejected image" with the chosen text held fixed, and (b) a **hierarchical textual loss** that combines response-level DPO with segment-level reweighting (`L_DPOs`) and a TDPO-style token-level KL term (`L_POk`).
- Headline number: **Object HalBench response-level hallucination drops 14.1 -> 4.9 on LLaVA-1.6-7B and 21.5 -> 6.2 on Muffin-13B**, framed as 55.5% / 52.7% relative-point improvements over flat DPO (which gives 11.0 and 13.1 respectively). All trained on the same RLHF-V 5K preference set, with the visual encoder frozen.
- The "Hierarchical" in the title is partly oversold: in Table 2 the visual branch alone removes 4.27 ObjHal R. when ablated, the segment term removes 3.63, and the token term removes only 1.16. HDPO without the cross-modal term beats DPO by just 1.84 points, while CMDPO (response + image) gains 1.27; the headline result is the *interaction* of the two ideas, not the hierarchy alone.

## Motivation

Hallucination is the single biggest blocker for clinical deployment of vision-language models - chest-X-ray report generation, pathology QA, radiology copilots all stall the same way. The dominant fix line (Silkie, HA-DPO, RLHF-V) extends RLHF/DPO with text-only preference pairs, but the authors show with a PCA visualization on LLaVA-1.6 (Fig. 1a-b) that flat multimodal DPO leaves a visible modality gap between image embeddings and ground-truth-text embeddings, while only weakly separating hallucinated from non-hallucinated text. Their argument is that fixing this needs two things at once: (i) an **explicit image-side preference signal** so the visual encoder / connector is actually optimized rather than going along for the ride, and (ii) **sub-response credit assignment** so the loss can pinpoint the tokens that actually caused the hallucination. CHiP's contribution is wiring both of those into a single DPO objective without changing the data (RLHF-V-5k is reused as-is) or unfreezing the encoder.

## Core Innovation

- **Cross-modal preference branch (`L_DPOv`).** Construct a rejected image `m_l` by running **forward diffusion with T=500 Gaussian-noise steps** on the original `m_w`. Then optimize $L_{DPOv} = -\log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid m_w, x)}{\pi_{ref}(y_w \mid m_w, x)} - \beta \log \frac{\pi_\theta(y_w \mid m_l, x)}{\pi_{ref}(y_w \mid m_l, x)} \right)$. The chosen text `y_w` is shared between both image conditions, so the only thing the gradient can attribute the preference to is the image. Out of {diffusion, blackness, crop, rotation, randomness} the diffusion construction wins by 3-6 points (Table 5).
- **Hierarchical textual loss (`L_HDPO`).** Three granularities composed: response-level mDPO (`L_DPOr`), a segment-level reweight (`L_DPOs`) that boosts log-probs on "changed segments" `y_c` identified as >=2-token spans in `y_l` absent from `y_w`, and a token-level term (`L_POk`) that is a TDPO-style sequential-KL penalty with stop-gradient on the chosen side - only the rejected sequence's KL is pushed down. The composition is `L_HDPO = L_DPOr + lambda * L_DPOs + gamma * L_POk` with lambda=1 (Muffin) or 3 (LLaVA), gamma=0.1.
- **Final CHiP objective (Eq. 12):** `L_CHiP = L_DPOv + L_DPOr + lambda * L_DPOs + gamma * L_POk`. Weights on `L_DPOv` and `L_DPOr` pinned to 1.
- **Recipe.** RLHF-V-5k (5,000 pairs, no new annotation), beta=0.5, lr=5e-7, batch 32, 3 epochs, **~3 h on 4xH100 for LLaVA-1.6-7B, ~5 h for Muffin-13B**. Visual encoder frozen by default - Table 3 shows unfreezing it actually *hurts* CHiP (39.6 -> 43.8 MMHal R.) even though it slightly helps vanilla DPO.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CHiP outperforms flat DPO on hallucination across four benchmarks | Table 1, both backbones, all 4 benchmarks except AMBER Cover and HallBench fA on LLaVA where CHiP < DPO | ObjHal, MMHal, HallusionBench, AMBER | ⭐⭐⭐ |
| C2 | CHiP beats DPO on ObjHal by 52.7% / 55.5% relative points | Table 1: (13.1->6.2)/13.1 = 52.7% (Muffin); (11.0->4.9)/11.0 = 55.5% (LLaVA) - arithmetic correct | ObjHal | ⭐⭐⭐ (correct, but single-run, no variance) |
| C3 | Both HDPO and visual preference optimization individually beat DPO | Table 2: HDPO alone (CHiP - L_DPOv) 9.19 < DPO 11.03; CMDPO alone (response + image) 9.76 < DPO 11.03 | ObjHal, MMHal | ⭐⭐ (single seed, one base model) |
| C4 | Combining hierarchy + visual is strictly best | Table 2: full CHiP dominates every ablation on ObjHal; on MMHal differences (39.63 vs 40.63/40.75/41.71/42.40) are within likely run-to-run noise | ObjHal, MMHal | ⭐⭐ |
| C5 | CHiP beats GPT-4V on ObjHal and AMBER | Table 1: CHiP-LLaVA ObjHal 4.9 vs GPT-4V 13.6; AMBER Hal 24.5 vs 30.7 | ObjHal, AMBER | ⭐⭐ (GPT-4V numbers are reported "as in prior work", not re-run) |
| C6 | CHiP brings image and ground-truth text representations closer; pushes hallucinated text farther | Fig. 1 + Fig. 7, PCA of last-token LLaMA embeddings on 150 COCO samples | COCO-150 | ⭐ (qualitative PCA, no quantitative similarity metric, hallucinated captions generated by Gemini 1.5 Flash with no manual verification) |
| C7 | CHiP doesn't make the model less talkative | Table 6: avg output length within ~10 tokens of DPO and base | AMBER, MMHal, ObjHal | ⭐⭐ |
| C8 | CHiP preserves general capability | Table 4: improves 5/6 of MMMU(val/test), MMB-CN, ScienceQA, LLaVA-Wild; only MMB-ENG drops 0.8 | MMMU, MMB, ScienceQA, LLaVA-Wild | ⭐⭐ |
| C9 | Diffusion-T=500 rejected image is the best construction | Table 5 / Fig. 6, single base model (LLaVA) | ObjHal, MMHal | ⭐⭐ |
| C10 | Beats mDPO / V-DPO / HA-DPO / POVID family | **Not directly evaluated.** Only compared to "DPO" (= the authors' own response-level mDPO) and to HA-LVA (not HA-DPO) in Table 1 | - | (no evidence) |

**Honest read.** Three things deserve to be flagged because the title and abstract foreground the *hierarchy* while the ablation tells a different story:

1. **Hierarchy is partially oversold; the cross-modal branch is the single biggest contributor.** Table 2 ablation on LLaVA: removing `L_DPOv` costs 4.27 ObjHal R. (full 4.92 -> 9.19), removing `L_DPOs` costs 3.63, removing `L_POk` costs only 1.16. HDPO-only (no cross-modal) gets 9.19 vs full CHiP at 4.92 - the headline gain is the *interaction* of cross-modal + hierarchy, not the hierarchy by itself. The token-level KL is the weakest individual piece; App. Table 7 confirms TDPO-alone improves ObjHal R. by only ~1.5 points over DPO.
2. **Headline 52.7% / 55.5% reductions are arithmetically correct but single-run.** No seeds, no variance, no significance tests on any number in the paper. On MMHal Ova the deltas between CHiP and ablated variants are all sub-1pt (2.89 vs 2.70-2.78) - treat these as ties. The Object HalBench drops (4.9, 6.2) are large enough to survive reasonable seed noise, but the rest of the table should be read with that caveat.
3. **Biggest concrete gap: no head-to-head with mDPO / V-DPO / HA-DPO / POVID** despite the paper positioning itself in exactly that design space. The cross-modal branch with a perturbed image and shared `y_w` is essentially mDPO's (Wang et al., NeurIPS 2024) "image hallucination" recipe; the paper does not cite or differentiate from that line. Without matched-data, matched-compute comparisons against those baselines, the claim that CHiP beats the cross-modal-DPO family is **unsupported**.

Smaller issues: the `gamma` symbol is overloaded between Eq. 7 (segment boost) and Eq. 10/12 (token-loss weight); the segment-difference heuristic mislabels synonym substitutions and reorderings as "hallucinated"; and the PCA figures (Fig. 1, 7) use Gemini-generated hallucinated captions without manual verification, so they are illustrative rather than evidential.

## Method & Architecture

![CHiP PCA alignment hero](/assets/images/paper/chip/fig_p009_06.png)
*Figure 7d: PCA of LLaMA last-token embeddings on 150 COCO images after CHiP training - image embeddings and ground-truth text embeddings cluster tightly together while hallucinated captions are pushed away. This is the paper's headline visual for the cross-modal alignment claim.*

CHiP is one DPO loss with four additive terms over a single `(x, m_w, y_w, y_l)` preference pair plus a constructed rejected image `m_l`:

- **`L_DPOr` (response, Eq. 5).** Vanilla mDPO over the whole sequence, conditioned on `m_w`.
- **`L_DPOs` (segment, Eq. 7 plugged into Eq. 5).** Identifies "changed segments" `y_c` between `y_w` and `y_l` and computes a reweighted log-likelihood $\log \pi_{seg}(y \mid x, m) = \frac{1}{C}\left[\sum_{y_i \in y} \log p(y_i \mid \cdot) + \gamma_{seg} \sum_{y_i \in y_c} \log p(y_i \mid \cdot)\right]$ with $C = |y| + \gamma_{seg} |y_c|$.
- **`L_POk` (token, Eq. 8-9).** TDPO-style sequential-KL: $L_{POk} = \text{sg}(\beta \cdot D_{SeqKL}(x, m, y_w; \pi_{ref} \Vert \pi_\theta)) - \beta \cdot D_{SeqKL}(x, m, y_l; \pi_{ref} \Vert \pi_\theta)$. Stop-gradient on the chosen side; only rejected-sequence KL is driven down.
- **`L_DPOv` (cross-modal, Eq. 11).** The novel piece. `y_w` shared between `m_w` and `m_l`; the diffusion-noised `m_l` is the "rejected image". Forces the visual pathway to contribute to the preference signal.

The PCA strip below shows the alignment effect across baseline -> DPO -> CMDPO (image branch alone) -> CHiP. The progression from (a) to (d) is the paper's main qualitative argument:

![PCA baseline LLaVA](/assets/images/paper/chip/fig_p009_01.png)
*Figure 7a: LLaVA baseline - large modality gap between image and ground-truth text; hallucinated and non-hallucinated text intermingle.*

![PCA LLaVA + DPO](/assets/images/paper/chip/fig_p009_02.png)
*Figure 7b: LLaVA + flat DPO separates hallucinated vs. ground-truth text but does not close the image-text modality gap.*

![PCA LLaVA + CMDPO](/assets/images/paper/chip/fig_p009_05.png)
*Figure 7c: LLaVA + CMDPO (cross-modal image branch added, no hierarchy) pulls image and ground-truth text closer - this single ablation is the strongest evidence that `L_DPOv` is the load-bearing component.*

## Experimental Results

Main hallucination table reproduced from Table 1 (lower is better for hallucination columns; up arrows mean higher is better):

| Model | ObjHal R.↓ | ObjHal M.↓ | MMHal Ova.↑ | MMHal R.↓ | HallBench qA↑ | HallBench fA↑ | HallBench aA↑ | AMBER CHAIR↓ | AMBER Cover↑ | AMBER Hal↓ | AMBER Cog↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4V (ref) | 13.6 | 7.3 | - | 31.3 | 28.8 | 39.9 | 65.3 | 4.6 | 67.1 | 30.7 | 2.6 |
| RLHF-V (ref) | 12.2 | 7.5 | 2.5 | 51.0 | - | - | - | 6.3 | 46.1 | 25.1 | 2.1 |
| Muffin-13B | 21.5 | 11.6 | 2.4 | 60.42 | 16.0 | 20.8 | 50.9 | 8.0 | 48.3 | 32.1 | 3.5 |
| Muffin + DPO | 13.1 | 6.6 | 2.5 | 52.1 | 17.4 | 23.4 | 52.5 | 6.2 | 46.9 | 26.5 | 2.5 |
| **Muffin + CHiP** | **6.2** | **3.9** | **2.6** | **49.0** | **19.1** | **24.9** | **54.0** | **4.4** | 45.3 | **17.6** | **1.5** |
| LLaVA-1.6-7B | 14.1 | 7.4 | 2.8 | 42.7 | 15.8 | 20.8 | 51.6 | 8.3 | 61.0 | 48.6 | 4.2 |
| LLaVA + DPO | 11.0 | 6.6 | 2.7 | 43.8 | 22.2 | **28.3** | 56.6 | 5.9 | 61.0 | 38.9 | 3.0 |
| **LLaVA + CHiP** | **4.9** | **3.2** | **2.9** | **39.6** | **23.5** | 26.0 | **58.5** | **3.7** | 57.8 | **24.5** | **1.6** |

Object HalBench is where the gap is clearest: CHiP-LLaVA beats GPT-4V on both R. (4.9 vs 13.6) and M. (3.2 vs 7.3). AMBER tells a more nuanced story - CHiP wins CHAIR, Hal, and Cog handily, but Cover *drops* (57.8 vs 61.0 LLaVA), which the authors read as the model "omitting ambiguous objects". HallusionBench fA also drops on LLaVA (26.0 vs 28.3 DPO), which the authors attribute to multi-objective dilution.

**Ablation (Table 2, LLaVA, ObjHal R. / M. / MMHal Ova / MMHal R.):**

| Variant | R.↓ | M.↓ | Ova.↑ | R.↓ |
|---|---|---|---|---|
| DPO | 11.03 | 6.61 | 2.73 | 43.75 |
| **CHiP (full)** | **4.92** | **3.21** | **2.89** | **39.63** |
| CHiP - L_DPOv (HDPO only) | 9.19 | 5.77 | 2.70 | 42.40 |
| CHiP - L_DPOs | 8.55 | 5.16 | 2.69 | 40.63 |
| CHiP - L_POk | 6.08 | 3.77 | 2.71 | 40.75 |
| CHiP - L_DPOs - L_POk (= CMDPO, response + image) | 9.76 | 5.47 | 2.78 | 41.71 |

Three things to take away from this ablation. First, the **visual branch `L_DPOv` is the single biggest contributor** - removing it costs 4.27 ObjHal R. points, more than removing the segment term (3.63) or the token term (1.16). Second, **hierarchy alone (HDPO, 9.19) is *worse* than CMDPO + segment alone** (8.55), and CMDPO alone beats vanilla DPO by only 1.27 R. points. The gains compound; neither piece on its own is enough. Third, **the token-level term `L_POk` is the weakest individual contributor**; App. Table 7 confirms TDPO-alone gives ObjHal R. = 9.56 vs. DPO 11.03, a real but small ~1.5 pt gain.

Other findings worth noting: Table 3 - **unfreezing the visual encoder helps DPO slightly but hurts CHiP** (39.6 -> 43.8 MMHal R.); the multi-objective loss seems to dilute alignment when the encoder is trainable. Table 5 - **diffusion T=500 is the best rejection-image construction** (ObjHal R. 4.9) vs randomness/blackness/rotation at 7.8-10.9, supporting the "rejected image must stay semantically close to chosen" intuition. Table 4 - CHiP is roughly neutral on MMMU / MMB / ScienceQA / LLaVA-Wild (+1.3 LLaVA-Wild, -0.8 MMB-ENG); **no obvious alignment tax**.

Human evaluation gives a more sober view of the gap:

![Human evaluation MMHal](/assets/images/paper/chip/fig_p007_01.png)
*Figure 3: Human evaluators on MMHal prefer CHiP over flat DPO in 24% of samples, prefer DPO in 12.5%, and call it a tie in 63.5%. The result is directionally positive but mostly ties - consistent with the small MMHal-Ova deltas in Table 1.*

Qualitative behavior on object-count and reflection scenes:

![Qualitative oysters](/assets/images/paper/chip/fig_p022_01.png)
*Figure 8 (top): CHiP corrects an object-count hallucination on a six-oyster scene where flat DPO still answers "six".*

![Qualitative sunglasses reflection](/assets/images/paper/chip/fig_p022_02.png)
*Figure 8 (bottom): CHiP gives a more cautious, image-grounded description of a sunglasses reflection while DPO over-asserts.*

## Limitations

- **No variance / no seeds / no significance tests** on any reported number. Differences <1 point on MMHal Ova. and similar columns should be treated as ties. The Object HalBench drops are large enough to survive reasonable seed noise, but the rest of the table is suggestive, not conclusive.
- **No head-to-head with mDPO (Wang et al., NeurIPS 2024), V-DPO (Xie et al.), HA-DPO, POVID, or RLHF-V's own DDPO under matched data/compute.** Only "DPO" (the authors' response-level mDPO implementation) is benchmarked. The cross-modal branch is methodologically very close to mDPO; without that comparison the claim of state-of-the-art among cross-modal DPO variants is unsupported.
- **Segment-difference heuristic is noisy.** Identifying "changed segments" as >=2-token spans in `y_l` absent from `y_w` mislabels synonym substitutions, reordering, and tone edits as hallucination - benign rewrites get up-weighted.
- **Overloaded notation.** `gamma` denotes both the segment boost (Eq. 7) and the token-loss weight (Eq. 10/12). The token-level gamma is set to 0.1; the segment-internal gamma is never reported explicitly.
- **PCA visualizations (Fig. 1, 7) are illustrative, not evidential.** Hallucinated captions used to label the clusters are generated by Gemini 1.5 Flash with no manual verification; there is no quantitative similarity metric or statistical test.
- **GPT-4V comparisons are taken from prior reports**, not re-run. The "CHiP beats GPT-4V on ObjHal/AMBER" claim is therefore conditional on those reported numbers being current.
- **Single training set (RLHF-V-5k).** No scaling-data or scaling-model experiments; no scaling curve to indicate whether the gains saturate.
- **No medical-domain evaluation** despite the obvious clinical-AI applicability. The motivation invokes medical use cases but the eval surface is COCO-derived (ObjHal/AMBER) plus GPT-4-judged general benchmarks.

## Why It Matters for Medical AI

The CHiP recipe is attractive to a medical-AI team for three reasons. (i) **Cost is low**: ~3-5 hours on 4xH100 for a 7-13B model, 5K preference pairs, and the visual encoder stays frozen. That fits the budget of a clinical lab that already has corrected radiology drafts on hand. (ii) **The cross-modal branch is data-free**: the rejected image is constructed by diffusion noise, not annotated, so the same trick is portable to chest-X-ray, pathology, fundus, or ultrasound without new labels - only the textual preference pairs need to come from clinicians. (iii) The dominant failure mode CHiP targets - the model "saying tumor because the scene-prior says tumor" - is exactly the over-confidence pattern that blocks clinical deployment of radiology copilots.

Three caveats before trusting the transfer story. First, the PCA-based alignment claim (Fig. 7) was validated on COCO-150 with Gemini-generated hallucinations; modality-conditioned priors in medical images (e.g. the strong texture prior of pathology) may not behave the same way, and "rejected image = diffusion T=500" might destroy clinically relevant signal more aggressively than it destroys COCO scene signal. Second, the segment-difference heuristic that drives `L_DPOs` will mislabel a lot of clinical paraphrase (e.g. "left lower lobe opacity" vs "LLL opacity") as hallucinated segments unless a clinician-tuned segment identifier is plugged in. Third, the lack of head-to-head with mDPO/V-DPO/HA-DPO means we cannot say CHiP is the right cross-modal-DPO recipe for a medical context - it is *a* recipe with a clean ablation and a real gain over flat DPO.

The two concrete experiments worth running before deploying CHiP in a clinical setting: replace diffusion-T=500 with modality-appropriate perturbations (e.g. mask the lesion ROI), and re-do the segment-difference step with a clinically aware diff that ignores paraphrase. Both are small changes to the existing released code.

## References

- Paper (arXiv): [arXiv:2501.16629](https://arxiv.org/abs/2501.16629) - Fu et al., *CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs*, ICLR 2025.
- Code: [https://github.com/LVUGAI/CHiP](https://github.com/LVUGAI/CHiP)
- Training data: Yu et al., *RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback*, CVPR 2024 - the 5K preference set used as-is.
- Direct Preference Optimization: Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- TDPO (token-level DPO inspiration): Zeng et al., *Token-level Direct Preference Optimization*, ICML 2024.
- mDPO (most natural baseline, not benchmarked in CHiP): Wang et al., *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*, NeurIPS 2024.
- V-DPO (related, not benchmarked): Xie et al., *V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization*, EMNLP 2024.
- HA-DPO (related, not benchmarked): Zhao et al., *Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization*, 2023.
- POVID (related, not benchmarked): Zhou et al., *Aligning Modalities in Vision Large Language Models via Preference Fine-tuning*, 2024.
- Object HalBench: Rohrbach et al., *Object Hallucination in Image Captioning*, EMNLP 2018.
- MMHal-Bench: Sun et al., *Aligning Large Multimodal Models with Factually Augmented RLHF*, 2023.
- HallusionBench: Guan et al., *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models*, CVPR 2024.
- AMBER: Wang et al., *AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation*, 2023.
