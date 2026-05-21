---
title: "Calibrated Self-Rewarding Vision Language Models"
excerpt: "CSR mixes CLIPScore with the LVLM's own sentence-level likelihood inside iterative DPO and lifts LLaVA-1.5-7B by 7.62% averaged across 10 benchmarks (CHAIR_S 48.8 -> 21.0)."
categories: [Paper, VLM-Alignment, LLM]
permalink: /paper/csr/
tags:
  - CSR
  - VLM-Alignment
  - DPO
  - Self-Rewarding
  - CLIPScore
  - Hallucination
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- LVLMs hallucinate because their decoder collapses onto the language prior; existing fixes either need expensive GPT-4/human preference annotation (LLaVA-RLHF, Silkie, POVID, RLHF-V) or, in the naive self-rewarding case, reward the model's own text-only-plausible answers.
- CSR runs **sentence-level beam search**, scores each candidate with a calibrated reward `R(s) = lambda * RI(s) + (1 - lambda) * RT(s)` where `RT` is the LVLM's own cumulative token likelihood and `RI` is a CLIPScore image-text cosine similarity, then turns the best/worst cumulative-reward responses into a DPO pair and iterates for `T = 3` rounds.
- On LLaVA-1.5-7B, three iterations give a **+7.62% average gain across 10 benchmarks** (LLaVAW 63.4 -> 71.1, MM-Vet 30.5 -> 33.9, POPE 85.90 -> 87.01, CHAIR_S 48.8 -> 21.0, CHAIR_I 14.9 -> 6.0). The 13B version gains +5.25%, and CSR beats naive self-rewarding by roughly +2.43% on average.

## Motivation

LVLMs (LLaVA, InstructBLIP, mPLUG-Owl, Qwen-VL) hallucinate not because the LLM is unfactual or the vision backbone is weak, but because the multimodal stack treats the image as weak conditioning and falls back on the language prior. Existing preference-optimization fixes pay heavily for this:

- **LLaVA-RLHF** uses human preferences.
- **Silkie / VLFeedback** and **POVID** use GPT-4-curated preferences.
- **RLHF-V** uses fine-grained human edits.

All three sources produce preferences that lie *outside* the target model's own distribution, so the DPO chosen/rejected gap is trivial to fit and the alignment signal collapses. **Self-rewarding** (Yuan et al. 2024) removes the external annotator for LLMs, but transplanted naively to LVLMs it inherits exactly the modality misalignment that drives hallucination: the model rewards text-only-plausible answers and ignores the image. CSR's contribution is to inject a vision-grounded calibration term (CLIPScore) into the self-rewarding loop so the preferences stay on-policy yet are not blind to the image.

![Radar overview](/assets/images/paper/csr/fig_p002_01.png)
*Figure 1: CSR (green) dominates the LLaVA-1.5 baseline (orange) and naive self-rewarding (blue) on 11 of 12 benchmark axes for the 7B model.*

## Core Innovation

The single load-bearing idea is the **calibrated reward** applied at the sentence level during beam search:

$$R(s) = \lambda \cdot R_I(s) + (1 - \lambda) \cdot R_T(s)$$

- `RT(s) = prod_{o=1..No} P(r_o | x, r_1, ..., r_{o-1})` is the LVLM's own cumulative token probability for sentence `s` — the on-policy "self" signal.
- `RI(s) = max(100 * cos(F_I(x_v), F_T(s)), 0)` is the CLIPScore between the input image and the candidate sentence — the off-decoder vision grounding the naive self-rewarding setup is missing.

At each beam step the top-`k` and bottom-`k` candidates by `R(s)` survive; once a beam hits `<eos>` the response reward is `R(y) = sum_i R(s_i)`. The argmax response is `y_w`, the argmin is `y_l`. DPO is then run with the *previous iteration's* policy as the reference, and the whole loop is repeated for `T = 3` rounds — so the preferences stay on-policy across iterations.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | CSR improves LLaVA-1.5-7B by ~7.62% averaged over 10 benchmarks. | Table 1 + averaging procedure described in Sec 4.2 (MME_P/16, MME_C/4, CHAIR -> 100-CHAIR). | ⭐⭐⭐ — broad, internally consistent column gains; only weakness is single-seed numbers. |
| C2 | The CLIP calibration term — not vanilla self-rewarding — is the source of the gain. | Table 2 ablation: `Only RT` 68.46 vs full CSR **72.39** at 7B (a ~4-point gap). | ⭐⭐⭐ — the cleanest possible isolation of the calibration term. |
| C3 | CSR continuously improves over iterations. | Figures 3 and 4: monotonically rising 7B/13B/Vila-7B average score curves with diminishing returns. | ⭐⭐ — only `T = 3` tested; per-benchmark variance and any negative-transfer benchmarks are not shown in the main text. |
| C4 | CSR is compatible with different LVLM backbones. | Vila-7B experiment (Figure 4). | ⭐⭐ — tested on 2 backbones (LLaVA-1.5, Vila), **both use CLIP ViT-L/14**. SigLIP, EVA-CLIP, BLIP-2 Q-Former, and domain-shifted CLIPs (medical) are untested. |
| C5 | CSR redirects attention to visual tokens. | Figure 6 attention map case study (1 image). | ⭐ — qualitative single example, no aggregate metric, no causal intervention. |
| C6 | "Rigorous theoretical analysis verifies the effectiveness of introducing visual constraints" (Theorem 5.1). | Linear-Gaussian generative model + Gaussian policy. | ⭐⭐ — proves only *existence* of a beneficial `lambda < 1` under a strong text-bias assumption; no magnitude bound, no convergence rate, idealized setup. |
| C7 | CSR is annotation-free / captures the LVLM's "inherent preferences". | Preferences come from the model's own beams. | ⭐⭐ — **misleading**. CSR imports a frozen CLIP as the image-grounding grader; it is more accurately "self-rewarding + frozen CLIP reward model" than annotation-free. |
| C8 | "Substantial improvements over existing methods by 7.62%" (Abstract framing). | Table 1. | ⭐⭐ — the 7.62% is CSR's gain over the *base* LLaVA-1.5, not over the strongest competing preference method. Against the best per-column baseline (often RLHF-V or POVID) the lead is smaller, and on VisWiz-7B it is zero. |

**Honest synthesis.** The two strongest claims — that CSR beats broad baselines (C1) and that the *calibration* term, not vanilla self-rewarding, drives the gain (C2) — are well supported by Table 1 and the Table 2 ablation. The iteration-stability and backbone-compatibility claims are reasonable but narrow (3 iterations, 2 backbones, both CLIP-aligned). The framing of "self-rewarding" is **misleading**: by relying on CLIP for `RI`, CSR has imported an external reward model. POVID/Silkie/RLHF-V are honest about their external GPT-4 or human inputs; CSR's CLIP dependency deserves the same framing.

## Method & Architecture

![CSR method](/assets/images/paper/csr/fig_p003_01.png)
*Figure 2: CSR's iterative loop — sentence-level beam search produces candidates, each sentence is scored by the LVLM's own cumulative likelihood (RT) and CLIPScore (RI), the top/bottom cumulative-reward responses form a DPO preference pair, the LVLM is fine-tuned with LoRA, and the process repeats with the previous iteration's policy as the DPO reference.*

Step-by-step:

1. **Seed model.** `pi_ref` = LLaVA-1.5-7B/13B or Vila-7B; all updates are LoRA, not full FT.
2. **Prompt pool.** ~13k (image, prompt) pairs from LLaVA-150k's *detailed description* + *complex reasoning* subsets (COCO 2014 images). The same prompts are reused across iterations.
3. **Sentence-level beam search.** Responses are generated sentence-by-sentence using "." as the delimiter; each surviving beam is expanded into multiple candidate next-sentences.
4. **Score each candidate.** Compute `RT(s)` (cumulative token probability) and `RI(s)` (CLIPScore, with the CLIP vision encoder *aligned to the LVLM's vision encoder*); combine into `R(s) = lambda * RI + (1 - lambda) * RT`.
5. **Beam survival.** Keep top-`k` and bottom-`k` by `R(s)` for the next step.
6. **Cumulative response reward.** `R(y) = sum_i R(s_i)`; `y_w = argmax_y R(y)`, `y_l = argmin_y R(y)`.
7. **DPO update with rolling reference.** Minimize the standard DPO loss against `pi_{theta_{t-1}}` as the reference policy.
8. **Iterate** for `T = 3` rounds; this keeps the preference distribution on the *current* policy.
9. **Theory (Theorem 5.1, informal).** Under a CLIP-style linear generative model and a Gaussian LVLM policy, when the LVLM over-weights text (`||beta*^T V_1^T beta*|| << ||beta*^T V_2^T beta*||`), there exists `lambda < 1` such that the calibrated step beats pure self-rewarding (`lambda = 1`). The theorem only shows **existence**, not magnitude or convergence rate, and the text-bias condition is asserted rather than verified empirically.
10. **Cost.** Single A100 80GB; ~3.5h for 7B, ~5h for 13B over 3 iterations.

## Experimental Results

### Main table (Table 1, exact numbers; bold = best per column)

| Method | MME_P | MME_C | SEED | LLaVAW | MMB | MM-Vet | SQA_I | VisWiz | GQA | POPE | CHAIR_S | CHAIR_I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B (base) | 1510.7 | 348.2 | 58.6 | 63.4 | 64.3 | 30.5 | 66.8 | 50.0 | 62.0 | 85.90 | 48.8 | 14.9 |
| + Vlfeedback (Silkie) | 1432.7 | 321.8 | 59.3 | 62.1 | 64.0 | 31.2 | 66.2 | 52.6 | 63.2 | 83.72 | 40.3 | 13.2 |
| + Human-Prefer (LLaVA-RLHF) | 1490.6 | 335.0 | 58.1 | 63.7 | 63.4 | 31.1 | 65.8 | 51.7 | 61.3 | 81.50 | 38.7 | 11.3 |
| + POVID | 1452.8 | 325.3 | 60.2 | 68.7 | 64.9 | 31.8 | 68.8 | 53.6 | 61.7 | 86.90 | 35.2 | 8.3 |
| + RLHF-V | 1489.2 | 349.4 | 60.1 | 65.4 | 63.6 | 30.9 | 67.1 | 54.2 | 62.1 | 86.20 | 29.7 | 7.5 |
| + Self-Rewarding (Yuan et al.) | 1505.6 | 362.5 | 60.0 | 61.2 | 64.5 | 31.4 | 69.6 | 53.9 | 61.7 | 86.88 | 24.0 | 6.7 |
| **+ CSR (Ours, 7B)** | **1524.2** | **367.9** | **60.3** | **71.1** | **65.4** | **33.9** | **70.7** | 54.1 | **62.3** | **87.01** | **21.0** | **6.0** |
| LLaVA-1.5-13B (base) | 1531.3 | 295.4 | 61.6 | 70.7 | 67.7 | 35.4 | 71.6 | 53.6 | 63.3 | 85.90 | 48.3 | 14.1 |
| + Self-Rewarding (13B) | 1529.0 | 300.1 | 62.8 | 65.6 | 64.5 | 35.3 | 74.3 | 56.1 | 63.2 | 86.58 | 37.0 | 8.8 |
| **+ CSR (Ours, 13B)** | 1530.6 | **303.9** | **62.9** | **74.7** | **68.8** | **37.8** | **75.1** | **56.8** | **63.7** | **87.30** | **28.0** | **7.3** |

CSR is best on 11 of 12 columns at 7B; the lone loss is VisWiz-7B (RLHF-V 54.2 vs CSR 54.1, a rounding-level gap).

### Ablation — the calibration term is doing real work (Table 2, avg across benchmarks)

| Variant | 7B avg | 13B avg |
|---|---|---|
| Base (no CSR) | 66.61 | 68.08 |
| CSR with only `RT` (no CLIP) | 68.46 | 68.12 |
| CSR with only `RI` (CLIP only) | 67.49 | 69.23 |
| **CSR (`RT` + `RI`)** | **72.39** | **71.95** |

The ~4-point gap between the best single-term variant (`Only RT` = 68.46) and the full method (72.39) at 7B is the **strongest single piece of evidence** that the calibration is necessary, not cosmetic — and the cleanest test of whether the gain is just self-rewarding plus DPO.

### Iteration dynamics

![Iteration dynamics](/assets/images/paper/csr/fig_p007_02.png)
*Figure 3: Per-benchmark improvement across iterations for LLaVA-1.5 7B and 13B; gains are largest in iter 1 and saturate by iter 3.*

7B improves +7.62% averaged over 3 iterations, 13B +5.25%, Vila-7B +3.37% (with +14.0% on MM-Vet and +8.48% on VisWiz). Gains diminish monotonically. Figure 5 (image-relevance KDE) shows both chosen and rejected response distributions shifting right (higher CLIP relevance) and **moving closer together** across iterations. The authors frame this positively as "harder, more discriminative pairs"; it is equally consistent with reward saturation / signal collapse, which would also explain the diminishing returns. Only `T = 3` is tested, so the question remains open.

![Backbone transfer](/assets/images/paper/csr/fig_p007_01.png)
*Figure 4: CSR ported to Vila-7B improves the average benchmark score across 3 iterations — evidence for backbone compatibility, but both Vila and LLaVA-1.5 still use CLIP ViT-L/14 vision towers.*

## Limitations

**Author-acknowledged.**
- Gains diminish across iterations (honest, framed as convergence).
- Theorem 5.1 requires the model to be text-biased before CSR runs.

**Not foregrounded by the paper.**
- **"Self-rewarding" framing is misleading.** CSR imports CLIP as an external reward model. The claim that CSR "captures the LVLM's inherent preferences" without external annotation is technically true for the preference *pairs*, but the *ranking signal* is a frozen CLIP. CSR is closer to "self-rewarding + frozen CLIP grader" than to a pure annotation-free recipe.
- **Reward miscalibration risk.** CLIPScore is known to be coarse, to have text-length / vocabulary biases, and to be wrong on counting, spatial relations, attributes, and out-of-distribution domains (medical, satellite, document, fine-grained scientific imagery). If CLIP is wrong, CSR rewards hallucinations that happen to trigger high CLIP cosine similarity — the exact modality-bias failure mode the paper claims to fix.
- **Backbone-agnosticism untested.** The paper relies on "the CLIP vision encoder aligns with the LVLM vision encoder." This is true for LLaVA-1.5 and Vila (both CLIP ViT-L/14) but breaks for SigLIP, EVA-CLIP, BLIP-2 Q-Former, and domain-shifted CLIPs (BiomedCLIP, PubMedCLIP). C4 in the table above is rated ⭐⭐ for exactly this reason.
- **Iteration stability past 3 rounds is untested.** Figure 5's collapsing chosen/rejected distributions are as consistent with reward saturation as with "harder pairs". There is no run beyond `T = 3`.
- **Single-run, no variance.** Every Table 1 number is single-seed. Several head-to-head gaps are 0.1–2 points — multi-seed variance would change the winner on several benchmarks.
- **Unmatched compute / data budgets vs LLaVA-RLHF and POVID.** No matched-preference-data-scale, matched-fine-tuning-compute control is reported.
- **No medical-AI evaluation.** Porting CSR to LLaVA-Med or Med-Flamingo would require a domain-specific CLIP (BiomedCLIP, PubMedCLIP, ChexZero) — none are tested.
- **No failure-mode analysis.** The case study (Figure 7) shows successful denoising, but no example of CSR adding *new* hallucinations that happen to be CLIP-favored (e.g., generic-but-irrelevant captions). That is the predicted failure mode of CLIP-rewarded captioning and would be the most informative negative case.

## Why It Matters for Medical AI

Hallucination is *the* deployment blocker for medical VLMs (LLaVA-Med, Med-Flamingo, RadFM): a fluent but image-unfaithful report is worse than no report at all. CSR's preference loop is, in principle, attractive for medical settings because clinician preference annotation is even more expensive than GPT-4 labels and the on-policy property avoids the chosen-rejected distribution shift that has hurt POVID-style methods.

But the paper's own assumption — "the CLIP vision encoder aligns with the LVLM's vision encoder" — is the wall any medical port runs into. Generic CLIP is well-known to perform poorly on radiology, pathology, and dermatology imagery, so `RI` would reward hallucinations that look generic-medical to a web-trained CLIP. A medical CSR would need a domain-specific calibration model (BiomedCLIP, PubMedCLIP, ChexZero) and, ideally, a *fine-grained* grader (BLIP-2 ITM, a fine-tuned VQA verifier, or report-level NLI) rather than a cosine score. Until that is shown to work, CSR's medical applicability is **promising-but-unproven**, and the burden of proof is on follow-up work to demonstrate that swapping CLIP for a medical encoder does not just collapse `RI` into noise.

## References

- Paper (arXiv 2405.14622, NeurIPS 2024): <https://arxiv.org/abs/2405.14622>
- Code: <https://github.com/YiyangZhou/CSR>
- Self-Rewarding LLMs (Yuan et al., 2024): <https://arxiv.org/abs/2401.10020>
- DPO (Rafailov et al., 2023): <https://arxiv.org/abs/2305.18290>
- POVID (Zhou et al., 2024): <https://arxiv.org/abs/2402.11411>
- RLHF-V (Yu et al., 2024): <https://arxiv.org/abs/2312.00849>
- Silkie / VLFeedback (Li et al., 2023): <https://arxiv.org/abs/2312.10665>
- LLaVA-RLHF (Sun et al., 2023): <https://arxiv.org/abs/2309.14525>
- CLIPScore (Hessel et al., 2021): <https://arxiv.org/abs/2104.08718>
- BiomedCLIP (Zhang et al., 2023): <https://arxiv.org/abs/2303.00915>
