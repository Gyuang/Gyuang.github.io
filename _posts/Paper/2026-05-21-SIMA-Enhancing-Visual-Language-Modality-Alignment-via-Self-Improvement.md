---
title: "SIMA: Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement"
excerpt: "The same LVLM both generates two candidate responses (greedy + temperature) and ranks them with an in-context, image-grounded critic prompt that includes the GT answer and three visual metrics, then DPO-finetunes on the self-rewarded pair. Headline: +7.5% (7B) / +4.5% (13B) / +5.3% (VILA) average across 14 benchmarks, with the critic agreeing with human raters 89.8% of the time (vs 95.6% for GPT-4V using the same prompt)."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/sima/
tags:
  - SIMA
  - Self-Improvement
  - DPO
  - Self-Rewarding
  - LVLM
  - LLaVA
  - VILA
  - Hallucination
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- **SIMA** removes external teachers from LVLM preference tuning. The same model $\pi_\theta$ (a) generates two candidate responses per prompt — one greedy, one temperature-sampled — and (b) ranks them with an in-context critic prompt that contains the image, the question, the ground-truth answer, three visual critic metrics (object / relation / attribute), and two demonstration examples. DPO is then applied to the self-ranked $(y_w, y_l)$ pair.
- The headline result is **+7.5% average across 14 hallucination + comprehension benchmarks on LLaVA-1.5-7B**, with +4.5% on LLaVA-1.5-13B and +5.3% on VILA-7B, beating LLaVA-RLHF, HA-DPO, POVID, and GT-DPO without any GPT-4V calls, human labels, or perturbation-based negatives.
- The critical caveat: the critic sees the GT. The 89.8% critic-vs-human agreement (vs 95.6% for GPT-4V with the same prompt) is plausible, but the design is closer to **GT-anchored online distillation** than to a pure self-rewarding loop, and the paper never isolates self-generation from external-candidate generation while holding everything else constant.

## Motivation

Preference-tuning recipes for LVLMs (RLHF, DPO) historically rely on (i) human-rated pairs or (ii) third-party AI feedback (GPT-4, GPT-4V). Both inject a **distribution shift**: the hallucinations and weaknesses being labeled are not necessarily the ones $\pi_\theta$ actually produces, so DPO on those pairs trains the model against an off-target failure mode. Both are also expensive and, in regulated settings such as medical AI, sending PHI to a closed-source teacher is often a non-starter.

![SIMA improves LLaVA-1.5-7B by an average 7.5% across 14 hallucination and comprehension benchmarks](/assets/images/paper/sima/fig_p001_01.png)
*Figure 1: Per-benchmark deltas of LLaVA-1.5-7B + SIMA vs. LLaVA-1.5-7B across 14 hallucination and comprehension benchmarks; the radar covers a +7.5% average lift.*

![Consistent gains across three LVLM backbones](/assets/images/paper/sima/fig_p001_02.png)
*Figure 2: Normalized average performance before and after SIMA on LLaVA-1.5-7B (+7.5%), LLaVA-1.5-13B (+4.5%), and VILA-7B (+5.3%).*

The medical-AI subtext is implicit. The GE Healthcare authorship and the hallucination-reduction framing make the radiology/CAD use case clear: a self-contained alignment loop that never leaves the deployment environment is structurally attractive when external GPT-4V calls are not legally available.

## Core Innovation

- **Same model, two roles.** $\pi_\theta$ both generates candidates and judges them. The candidates come from (greedy decode, temperature-0.8 decode) of $\pi_\theta$ — there is no teacher model and no perturbation-of-ground-truth synthesis as in POVID.
- **In-context critic, no instruction-tuning of the judge.** The critic prompt supplies (a) image + question, (b) **the ground-truth answer**, (c) three visual critic metrics — object-description accuracy, relationship accuracy, attribute accuracy — and (d) two demonstration pairs. The LVLM then picks the better of the two candidates. The three metrics are the load-bearing prompt component: removing them drops human-agreement of the critic from 89.8% to 78.2%.
- **DPO on self-ranked pairs.** Standard DPO with $\pi_\text{ref}$ = SFT-initialized model. No new datasets are introduced — prompts come from the same SFT corpus the model was trained on (`complex_reasoning_77k` + `detail_23k` of LLaVA-Instruct-150K, or VILA's equivalent), so the only new supervision is the self-generated ranking signal.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | SIMA improves LVLM performance on average +7.5% (7B), +4.5% (13B), +5.3% (VILA) across 14 benchmarks. | Tables 1+2; Figures 1, 2. | 14 benches x 3 backbones. | ⭐⭐⭐ — broad coverage; the headline averages survive across backbones. |
| C2 | LVLMs can self-critic accurately **without** instruction-tuning the judge, given a well-designed prompt. | Table 4 — SIMA judge 89.8% human-agreement, close to GPT-4V (95.6%) using the same prompt. | 500 hand-rated pairs. | ⭐⭐ — one 500-pair eval, rated by the **authors themselves**, no inter-annotator $\kappa$, no blinding. The 5.8-pt gap to GPT-4V is non-trivial. |
| C3 | The three visual critic metrics (object/relation/attribute) materially improve critic quality. | Tables 3-4 — agreement drops from 89.8% to 78.2% without them; MM-Hal drops from 2.30 to 2.12. | Ablation on LLaVA-1.5-7B only. | ⭐⭐ — clean ablation but single backbone. |
| C4 | SIMA outperforms prior preference methods (LLaVA-RLHF, HA-DPO, POVID, GT-DPO). | Tables 1, 2. | 14 benches, but baselines are reported only on LLaVA-1.5-7B. | ⭐⭐ — wins on aggregate; several per-cell margins (e.g. SQA-I 69.1 vs POVID 69.2) are <1 point and within plausible seed variance. |
| C5 | Self-generation avoids the distribution-shift problem that external preference data creates. | Argued in §2.1. **No isolation experiment** swaps external for self-generated candidates while holding the rest constant. | — | ⭐ — argued, not measured. |
| C6 | The approach generalizes across LVLMs. | LLaVA-1.5-7B, LLaVA-1.5-13B, VILA-7B all improve. | 3 backbones. | ⭐⭐ — three is broad for this subfield but all are COCO/natural-image LVLMs. |
| C7 | Multi-iteration self-improvement keeps helping. | Table 8 — iter 2/3 mostly flat with some gains on Mementos. | 1 backbone (13B). | ⭐ — paper's own §3.3 admits "performance saturates in the third iteration." |
| C8 | Higher decoding temperature improves preference data quality. | Table 6 — monotonic trend up to $T=0.8$. | 1 backbone (7B). | ⭐⭐ — clean trend; interpretation (sampled candidate as a useful negative) is sensible. |

**Honest read.** C1 is genuinely well-supported — 14 benchmarks x 3 backbones is a wider sweep than most preference-tuning LVLM papers. Several of the surrounding claims oversell what was actually measured:

- **Single-seed everywhere.** Every number in the paper is one run. No std-dev, no significance testing. Many of the "wins" over POVID and GT-DPO are <1 point on benchmarks where seed variance is plausibly larger than the delta.
- **C2 leans on an author-graded human eval.** The 500 pairs were rated by the paper's own authors. No inter-annotator agreement, no blinded protocol. The 89.8% vs 95.6% gap to GPT-4V is meaningful — the self-critic is measurably worse than the closed-source teacher it claims to obviate.
- **C5 is rhetorical, not measured.** The clean comparison would be: same critic prompt, same DPO, but candidates drawn from an external LVLM vs self-generated. That isolation is absent. The "GT-DPO" baseline mixes self-generated negatives with the GT as positive — closer, but still not the right ablation.
- **No comparison to CSR.** Calibrated Self-Rewarding (Yu et al., 2024) is a concurrent and very similar method for self-rewarding LVLMs. The paper cites Yuan et al.'s self-rewarding LLM work but never engages CSR. Given the framing overlap, the omission is conspicuous.
- **C7 is over-claimed.** Table 8 shows iter-2/3 gains are within noise on most benches. Calling this "keeps improving" reads as marketing.
- **The critic sees the GT.** The authors defend this by saying the GT was already in SFT, so no new supervision is leaked. But at critic time the GT is being **re-used as a reward signal**, which is closer to a second pass of supervised distillation than to a clean self-rewarding loop. A fairer framing is "GT-anchored self-critic" or "online SFT-label distillation via DPO." The framing of "pure self-improvement" oversells what the procedure actually does.
- **Critic-as-same-model failure mode is acknowledged but not stress-tested.** If $\pi_\theta$ is systematically wrong about a class of images (e.g., reading text in TextVQA), both candidates and the judge inherit the same blind spot. The paper admits this for TextVQA but does not quantify how often the critic fails in correlated ways with the generator.

## Method & Architecture

SIMA is a three-stage loop (Algorithm 1 in the paper) over prompts drawn from the LVLM's own visual instruction-tuning corpus. The critic prompt is the load-bearing piece:

![Structure of the in-context self-critic prompt](/assets/images/paper/sima/fig_p004_01.png)
*Figure 3: The in-context self-critic prompt has four parts — image + question + ground-truth answer; three visual critic metrics (object / relationship / attribute accuracy); two demonstration rankings; and the two candidate responses to be ranked.*

### Stage 1 — Response self-generation

For each $(I, x)$ from the SFT data, the current model $\pi_\theta$ produces two candidates:

- $y^{(1)}$ via greedy decoding,
- $y^{(2)}$ via temperature sampling, $T=0.8$ in the main results.

Both come from $\pi_\theta$ — there is no teacher. The temperature ablation (Table 6) confirms that higher $T$ helps monotonically up to 0.8; the authors read this as needing the sampled candidate to be hallucination-prone enough to give DPO a usable negative.

### Stage 2 — In-context self-critic

$\pi_\theta$ is reused as the judge. The critic prompt contains:

1. **Image + question + ground-truth answer.** Unlike LLM self-rewarding (Yuan et al. 2024), where the judge can grade "format / helpfulness" alone, LVLM visual-accuracy judgments need a visual reference. The paper argues the GT is fair game because it was already part of SFT training.
2. **Three visual critic metrics:**
    - Accuracy in Object Description — penalize objects not in the GT and miscalled objects.
    - Accuracy in Depicting Relationships.
    - Accuracy in Describing Attributes.
3. **Two in-context demonstrations** of the ranking format.

$\pi_\theta$ outputs which of $\{y^{(1)}, y^{(2)}\}$ is better. The chosen response becomes $y_w$, the other $y_l$. Critic-quality evidence (Table 4) puts agreement-with-humans at 89.8%, vs 95.6% for GPT-4V using the same prompt; removing the three metrics drops the critic to 78.2%.

### Stage 3 — Preference tuning

Standard DPO with $\pi_\text{ref}$ = the SFT-initialized LVLM:

$$
\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x, I)}{\pi_\text{ref}(y_w \mid x, I)} - \beta \log \frac{\pi_\theta(y_l \mid x, I)}{\pi_\text{ref}(y_l \mid x, I)}\right)\right].
$$

### Training recipe

17k prompts sampled from `complex_reasoning_77k` + `detail_23k` of LLaVA-Instruct-150K (LLaVA backbones) or the equivalent VILA SFT data — no new data is introduced. LoRA finetuning. 3 epochs on LLaVA-1.5-7B (15 A100-hours), 1 epoch on LLaVA-1.5-13B (7 A100-h) and VILA-7B (6 A100-h) because the larger models overfit. A single A100-80GB suffices.

## Experimental Results

### Hallucination benchmarks (Table 1, LLaVA-1.5-7B base)

| Method | CHAIR_s ↓ | CHAIR_i ↓ | MM-Hal ↑ | Mementos-O ↑ | Mementos-B ↑ |
|---|---|---|---|---|---|
| LLaVA-1.5-7B | 50.8 | 11.7 | 2.04 | 39.29% | 23.02% |
| + LLaVA-RLHF | 45.3 | 11.1 | 2.11 | 40.53% | 22.71% |
| + GT-DPO | 47.3 | 11.2 | 2.00 | 43.67% | 24.35% |
| + HA-DPO | 46.5 | 10.7 | 1.97 | 41.07% | 23.58% |
| + POVID | 48.4 | 11.3 | 2.28 | 42.95% | 23.84% |
| **+ SIMA** | **40.9** | **10.4** | **2.30** | **46.08%** | **26.03%** |

SIMA wins every hallucination metric, often by a wide margin (CHAIR_s drops 50.8 -> 40.9, ~19% relative). On VILA-7B, which already has the lowest baseline CHAIR_s (34.7), SIMA still cuts it to 28.4.

### Comprehensive benchmarks (Table 2, LLaVA-1.5-7B base, partial)

| Method | LLaVAW | SQA-I | VQA-T | MME-P | MME-C | MMB | MM-Vet | SEED | VisWiz |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 63.4 | 66.8 | 58.2 | 1506.4 | 355.7 | 64.3 | 30.5 | 58.6 | 50.0 |
| + GT-DPO | 64.7 | 67.4 | 58.1 | 1510.8 | 365.0 | 64.6 | 31.2 | 60.4 | 53.8 |
| + POVID | 65.3 | 69.2 | 58.1 | 1493.5 | 363.5 | 64.1 | 31.3 | 60.3 | 54.0 |
| **+ SIMA** | **66.1** | 69.1 | **58.5** | 1507.7 | **379.3** | **64.9** | **31.6** | **60.6** | **54.4** |

Gains on comprehensive benchmarks are real but small — often <1 point per cell — and SIMA does not always strictly beat POVID/GT-DPO per cell (SQA-I 69.1 vs POVID 69.2). The story holds in aggregate but not in every column.

### Critic-metric ablation (Tables 3 / 4)

![Distribution of critic preferences with vs without the three visual metrics](/assets/images/paper/sima/fig_p007_01.png)
*Figure 4: Removing the three visual critic metrics from the prompt flips roughly a fifth of the preference labels and drops human-agreement from 89.8% to 78.2%.*

| Setting | Human agreement | CHAIR_s ↓ | MM-Hal ↑ |
|---|---|---|---|
| SIMA critic (full prompt) | **89.8%** | **40.9** | **2.30** |
| SIMA critic, no metrics | 78.2% | 41.5 | 2.12 |
| GPT-4V (same prompt) | 95.6% | — | — |

The three visual metrics carry the critic. Without them the model still ranks pairs, but agreement collapses by 11.6 points and the downstream DPO loses most of its Mementos gains.

### Iteration ablation (Table 8, LLaVA-1.5-13B)

![Multi-iteration average performance on LLaVA-1.5-13B](/assets/images/paper/sima/fig_p007_02.png)
*Figure 5: Iteration 1 captures essentially all of the gain; iterations 2 and 3 yield diminishing, often within-noise returns on average.*

Iter 1 -> 2 -> 3 average improvement plateaus. Some benches keep crawling up (Mementos-O 45.84 -> 46.02 -> 46.91), others drift down. No collapse, no second-round bonanza either. The paper's own §3.3 says "performance saturates in the third iteration," which is more accurate than the abstract's framing.

### Temperature ablation (Table 6, LLaVA-1.5-7B)

![Effect of decoding temperature on SIMA performance](/assets/images/paper/sima/fig_p007_03.png)
*Figure 6: Higher sampling temperature ($T$ up to 0.8) yields stronger SIMA — consistent with the sampled candidate needing to be hallucination-prone enough to act as a useful DPO negative.*

$T \in \{0.2, 0.4, 0.6, 0.8\}$; the trend is monotonically positive on most metrics. The authors' interpretation — the sampled candidate needs to be hallucination-prone to give DPO a useful negative — is consistent with the data.

### Qualitative examples (Figure 8)

![SIMA caption vs LLaVA-1.5-7B caption — cat on bench](/assets/images/paper/sima/fig_p014_01.png)
*Figure 7a: SIMA drops the hallucinated "white and black" color attribute and trims invented furniture from LLaVA's caption.*

![SIMA caption vs LLaVA-1.5-7B caption — cow and motorcycles](/assets/images/paper/sima/fig_p014_02.png)
*Figure 7b: SIMA preserves accurate spatial relations while removing speculative scene details. Failure analysis in the paper is anecdotal — one or two cases — with no quantitative breakdown of which error categories the critic flips vs misses.*

## Limitations

**Authors acknowledge.** SIMA's ceiling is bounded by $\pi_\theta$'s existing capabilities and by biases in the SFT corpus; TextVQA gains are negligible because both the generator and the critic share the same text-reading weakness. Self-critic data can reinforce distributional biases.

**Visible from the evidence but not addressed.**

- **No variance / multi-seed reporting on any number in the paper.** Several wins over POVID/GT-DPO are <1 point.
- **No external-judge sanity check beyond a one-shot 500-pair GPT-4V comparison.** No Likert agreement, no inter-annotator $\kappa$, no blinding. The 500 pairs were rated by the paper's own authors.
- **No comparison to concurrent self-rewarding LVLM methods.** CSR (Calibrated Self-Rewarding, Yu et al. 2024) is conspicuously absent from related work despite a near-identical framing.
- **No isolation experiment for "self-generated vs external candidates with everything else equal."** The claim that self-generation is the load-bearing piece (and not the critic prompt) is asserted, not measured.
- **GT-anchoring blurs "self-improvement" vs "online distillation".** The critic sees the GT at every step. The authors argue this is legitimate because the GT was in SFT, but using the GT as a reward signal at preference-data-construction time is closer to a second pass of supervised distillation than to a clean self-rewarding loop.
- **Critic-as-same-model failure mode is admitted but not quantified.** No breakdown of categories where the critic systematically agrees with the generator's hallucination.
- **No safety / jailbreak / robustness evaluation.** Only hallucination + comprehension.
- **Compute cost is undiscussed.** Generating 17k x 2 responses + 17k critic passes likely dominates the DPO step itself; no wall-clock comparison vs POVID or HA-DPO.
- **Distribution.** Every benchmark is COCO-distributed natural images. No medical, document, chart, or other domain shift — so the implicit medical-AI argument in the paper's framing is not backed by any medical eval.

## Why It Matters for Medical AI

The structural argument is independent of any medical experiment in the paper. In clinical and radiology deployments, sending images or text to GPT-4V is often legally blocked, and human preference data is expensive and slow to collect. A preference-tuning loop that is fully on-device and uses only the model's own outputs plus the SFT label set fits cleanly into a PHI-bounded environment. SIMA's value for medical AI is thus the **recipe**, not the results — none of the 14 benchmarks are medical, the SFT corpus is natural images, and the critic prompt is tuned for COCO-style scenes (objects / relations / attributes). Transferring SIMA to radiology would minimally require (i) replacing "object/relation/attribute" with a radiology-appropriate critic rubric (lesion presence, anatomical localization, finding-impression consistency), (ii) a medical SFT corpus from which to draw prompts and GTs, and (iii) an honest medical critic-quality study with clinician raters — none of which the paper provides. As a self-contained alignment recipe SIMA is attractive; as evidence that LVLM self-improvement transfers to clinical deployment it is silent.

## References

- Paper: Wang, Chen, Wang, Zhou, Zhou, Yao, Zhou, Goldstein, Bhatia, Kass-Hout, Huang, Xiao. *Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement.* ICLR 2025. [arXiv:2405.15973](https://arxiv.org/abs/2405.15973).
- Related: Rafailov et al., *Direct Preference Optimization (DPO)*, NeurIPS 2023.
- Related: Yuan et al., *Self-Rewarding Language Models*, ICML 2024 — the LLM-side precursor.
- Related: Yu et al., *Calibrated Self-Rewarding Vision Language Models (CSR)*, NeurIPS 2024 — concurrent and closely related; not cited.
- Related: Sun et al., *Aligning Large Multimodal Models with Factually Augmented RLHF (LLaVA-RLHF)*, ACL Findings 2024.
- Related: Zhao et al., *Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware DPO (HA-DPO)*, 2024.
- Related: Zhou et al., *Aligning Modalities in Vision Large Language Models via Preference Fine-tuning (POVID)*, ICML 2024.
- Related: Liu et al., *Visual Instruction Tuning (LLaVA)* and *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*, NeurIPS 2023 / CVPR 2024.
- Related: Lin et al., *VILA: On Pre-training for Visual Language Models*, CVPR 2024.
- Related: Rohrbach et al., *Object Hallucination in Image Captioning (CHAIR)*, EMNLP 2018.
- Related: Sun et al., *MM-Hal-Bench*, 2023; Wang et al., *Mementos*, 2024.
