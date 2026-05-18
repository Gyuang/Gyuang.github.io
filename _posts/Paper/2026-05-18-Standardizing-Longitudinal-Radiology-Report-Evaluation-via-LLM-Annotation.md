---
title: "Standardizing Longitudinal Radiology Report Evaluation via Large Language Model Annotation"
excerpt: "A two-step open-weight-LLM annotator (Qwen2.5-32B) yields +11.3% F1 on longitudinal-sentence detection over the ImaGenome silver rule baseline and seeds L-MIMIC, a 95k-report CXR benchmark on which the best generator (Maira2) caps at 41.4% progression micro-F1."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/longitudinal-eval/
tags:
  - L-MIMIC
  - Longitudinal-Report-Evaluation
  - LLM-as-Annotator
  - Qwen2.5
  - MIMIC-CXR
  - ImaGenome
  - Radiology-Report-Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR

- A two-step LLM pipeline (Stanza sentence split -> longitudinal-vs-cross-sectional classification -> 50-disease keyword + {improved / no change / worsened} progression labeling) replaces the rule-based ImaGenome silver labeler. On 1,975 hand-labeled ImaGenome sentences it beats silver by **+11.3% F1** on longitudinal-sentence ID (LLaMA3.3-70B 94.0 vs 82.7) and **+5.3% F1** on progression (Qwen2.5-72B 83.8 vs 78.5).
- Qwen2.5-32B is selected for the 95,169-report MIMIC-CXR pass because it is **15.7x faster** than MedGemma-27B at comparable accuracy (2.07 s vs 31.62 s per query). The result is **L-MIMIC** -- a 73,093 / 588 / 1,863 train/val/test split with longitudinal flag + disease keyword + progression label per sentence.
- Seven SOTA report generators are then scored on L-MIMIC: the best (**Maira2**) reaches only **41.4% progression micro-F1** vs 58.2% diagnosis F1, and all models collapse on the minority "improved" / "worsened" classes (<19 F1). The longitudinal vs cross-sectional gap CheXbert-only metrics hide is now visible.

## Motivation

Every recent CXR report-generation paper that markets itself as "longitudinal" (Libra, Maira2, HC-LLM, MLRG, L_R2Gen, ...) is scored with CheXbert-style **cross-sectional** clinical-efficacy metrics or with TEM, which only matches surface temporal phrases. There is no open labeler that (a) identifies which sentences actually compare to prior imaging, (b) summarizes the per-disease *progression direction*, and (c) is open enough to run on both ground-truth and generated reports. Without one, the community cannot answer whether prior-image fusion modules genuinely improve longitudinal reasoning. The paper's bet is that LLMs -- whose strength is paraphrase-robust semantic extraction -- can act as the annotator, then be turned around to re-evaluate the field.

![Longitudinal vs cross-sectional sentences in a follow-up report; case for LLM annotation over rule-based methods](/assets/images/paper/longitudinal-eval/page_003.png)
*Figure 1: Longitudinal vs cross-sectional sentences in a follow-up report (top), and the case for LLM-based annotation over rule-based methods (bottom). Source: paper, p.3.*

## Core Innovation

- **Two-step prompt pipeline instead of a fine-tuned classifier.** Task 1 returns `<1>` if the sentence compares to prior imaging (with cue words *remain, compare, similar, stable, increased, still, new, again*) else `<0>`. Task 2 picks one of 50 curated diseases (deduplicated from ImaGenome gold; "normal"/"abnormal" removed, "support devices" added). Task 3 returns one of `<improved> / <no change> / <worsened> / <unmentioned>` for each (sentence, disease). All deterministic greedy decoding -- no fine-tuning.
- **Disease-specific worked examples in the prompt** combat polarity ambiguity: "New small bilateral pleural effusions" => worsened; for "low lung volumes" "lower" => worsened, "increased" => improved. MedGemma additionally requires an explicit "Please output <0> or <1>." suffix to enforce binary output -- a small but telling brittleness note.
- **Annotator selected on cost, not peak F1.** Qwen2.5-32B is *not* the top-F1 model on either subtask, but it is the only one whose throughput makes a 95k-report annotation pass feasible (15.7x MedGemma, ~210x single-GPU LLaMA3.3-70B). This trade-off is the engine that makes L-MIMIC possible -- and the source of its largest open risk.
- **L-MIMIC benchmark + evaluation framework.** A 73,093 / 588 / 1,863 train/val/test split over MIMIC-CXR follow-up reports, scored under (i) language metrics restricted to the longitudinal subset (BLEU-1..4, ROUGE-L, METEOR) and (ii) progression Acc/P/R/F1 -- micro-averaged and per class. The framework's stated capability is to detect three failure modes: *missing comparison*, *redundant (hallucinated) comparison*, *wrong trend*.

## Claims & Evidence Analysis

| # | Claim (as stated) | Evidence | Dataset | Strength |
|---|---|---|---|---|
| C1 | LLM annotation beats existing solutions by **+11.3 F1** on longitudinal detection and **+5.3 F1** on disease tracking. | Figure 3 numbers replicated below; hand-labeled sentence-level test. | ImaGenome 500-report subset (1,975 sentences) | ⭐⭐⭐ -- numerically clean on hand labels. |
| C2 | Qwen2.5-32B is the best **trade-off** between accuracy and efficiency for large-scale annotation. | Table A1 speed (2.07 s vs 31.62 s) + within-2-F1 of best on detection, top progression micro-Acc. | ImaGenome subset + 100-sentence speed test | ⭐⭐ -- efficiency is decisive; "best overall accuracy" is *not* the claim and would be false (LLaMA3.3-70B leads detection F1, Qwen2.5-72B leads progression F1). |
| C3 | Larger LLMs do not necessarily outperform smaller ones. | Qwen2.5-32B ~ Qwen2.5-72B; LLaMA3.3-70B trails on progression. | Same 1,975-sentence subset | ⭐⭐ -- supported in-family but confounded across architectures/training corpora. |
| C4 | Medically fine-tuned LLMs (MedResearcher-R1-32B, MedGemma-27B) do not consistently outperform general LLMs. | MedResearcher-R1-32B < Qwen2.5-32B base; MedGemma-27B mixed. | Same subset | ⭐⭐ -- holds for this annotation task, not necessarily for upstream report-generation tasks the medical LLMs were tuned for. |
| C5 | L-MIMIC provides a **standardized** benchmark for evaluating longitudinal report generation. | Construction of 73,093/588/1,863 split via Qwen2.5-32B auto-labels; usage to score 7 models. | MIMIC-CXR | ⭐⭐ -- internally consistent, but with **no human IAA on the L-MIMIC test split** (only on the upstream ImaGenome subset) "standardized" is a stretch. The Qwen2.5-32B worsened-class under-recall propagates directly into the benchmark. |
| C6 | Models with longitudinal priors outperform those without on longitudinal generation. | R2Gen/MedVersa progression-F1 ~33.6/33.9 vs prior-aware models 35.7-41.4. | L-MIMIC test (n=1,863) | ⭐⭐ -- direction is consistent; magnitude modest (~8 F1 best), confounded by model scale. No ablation isolates "prior" from "scale". |
| C7 | All current report-generation models exhibit considerable scope for improvement on longitudinal information. | Best progression F1 = 41.4%; minority-class F1 ceilings at 17.7 / 18.3. | L-MIMIC test | ⭐⭐⭐ -- gap is large enough that single-run noise does not threaten it. |
| C8 | The framework detects missing comparison, redundant comparison, and wrong-trend errors. | Argued in Section 4.3; not validated with an error-type-labeled dataset. | none directly | ⭐ -- capability claim with no controlled experiment (e.g., synthetically injected errors) that the framework's metrics correlate with these specific error modes. |

**Honest read.** The strongest claim is C1 (annotation accuracy on a held-out, hand-labeled set). The weakest is C8 (the error-taxonomy mapping is asserted, not measured). The headline "standardized benchmark" (C5) does real work in the abstract but is, on inspection, a single-LLM auto-label pass with **no human cross-check on the actual MIMIC test split** -- exactly the kind of circularity the paper criticizes in ImaGenome silver. The Qwen2.5-32B failure mode the authors themselves disclose (missing 46/190 worsened sentences in the ImaGenome eval) propagates directly into L-MIMIC labels: per-model progression scores in Table 2 below are **upper-bounded by Qwen2.5-32B's own recall on the minority worsened class** -- any generator that correctly catches a worsened case Qwen missed will be punished as "redundant." The paper does not quantify this confounder, does not report variance / seeds, and does not compare L-MIMIC scores against GREEN, RadGraph-F1, RaTE-Score, or LLM-as-judge metrics -- the only baseline annotator pitted against is ImaGenome silver. And while the project is filed in the CT report-generation bucket, **the paper itself only validates on chest X-ray**; CT, MRI, and other anatomies are out of scope.

## Method & Architecture

![Two-stage pipeline: per-sentence LLM annotation and the evaluation framework for generated reports](/assets/images/paper/longitudinal-eval/page_007.png)
*Figure 2: Two-stage pipeline. (a) Per-sentence longitudinal classification + disease-keyword extraction + progression labeling. (b) Evaluation framework that scores generated reports against LLM-annotated ground truth on language metrics (restricted to longitudinal sentences) and progression Acc/P/R/F1. Source: paper, p.7.*

1. **Sentence segmentation.** Each report is split into sentences with Stanza.
2. **Task 1 -- Longitudinal-sentence classification.** Three in-context examples + cue-word list; output `<1>` or `<0>`.
3. **Task 2 -- Disease-keyword extraction.** Per longitudinal sentence, the LLM picks the most related entry from a closed list of **50 diseases** (Appendix A.1). Support-device mentions are kept but excluded from progression scoring because they describe positional rather than pathological change.
4. **Task 3 -- Progression labeling.** Per (sentence, disease) pair, output one of `<improved> / <no change> / <worsened> / <unmentioned>` (the last to absorb hallucinated entities from generated reports). Disease-specific worked examples in the prompt handle polarity.
5. **Annotator candidates.** Five open-weight LLMs run with deterministic greedy decoding on A100s: MedGemma-27B, MedResearcher-R1-32B, Qwen2.5-32B, LLaMA3.3-70B, Qwen2.5-72B. Zero fine-tuning.
6. **Annotator selection for scale-out.** Qwen2.5-32B is chosen for the MIMIC-CXR pass: at equivalent F1 it is **15.7x faster** than MedGemma-27B (2.07 s vs 31.62 s per query, 38 vs 359 generated tokens) and ~210x faster than single-GPU LLaMA3.3-70B (658.88 s).
7. **L-MIMIC construction.** All MIMIC-CXR second-or-later visits with a "Findings" section are sentence-annotated by Qwen2.5-32B. Filtering to reports with at least one longitudinal sentence yields **73,093 train / 588 val / 1,863 test** (from 92,374 / 737 / 2,058 raw), total 95,169 reports annotated.
8. **Generation-model evaluation (Fig. 2b).** For each (reference, generated) pair: extract longitudinal sentences from each; compute BLEU-1..4 / ROUGE-L / METEOR on the longitudinal subset (in addition to whole-report); for every (disease, reference progression) pair, infer progression from the generated report and score micro- and per-class Acc/P/R/F1.

## Experimental Results

### LLM annotator vs ImaGenome silver

![Five LLMs vs ImaGenome silver on longitudinal-sentence ID and disease-progression labeling](/assets/images/paper/longitudinal-eval/page_009.png)
*Figure 3: Five LLMs vs ImaGenome silver on longitudinal-sentence ID (left) and disease-progression labeling (right). Headline gains come almost entirely from recall -- the rule baseline is precise (98.5) but misses ~29% of longitudinal sentences. Source: paper, p.9.*

| Annotator | Longit.-sentence F1 | Recall | Precision | Progression micro-F1 | Progression micro-Acc |
|---|---|---|---|---|---|
| ImaGenome silver (rule) | 82.7 | 71.2 | 98.5 | 78.5 | 88.7 |
| MedGemma-27B | 93.2 (+10.5) | 93.7 | 92.7 | 83.7 (+5.2) | 91.4 |
| MedResearcher-R1-32B | 90.5 (+7.8) | 86.0 | 95.4 | 79.4 (+0.9) | 90.5 |
| **Qwen2.5-32B** (scale-out pick) | **91.8 (+9.1)** | **88.8** | **95.1** | **83.2 (+4.7)** | **91.9** |
| LLaMA3.3-70B | **94.0** (+11.3) | 96.8 | 91.4 | 79.8 (+1.3) | 90.1 |
| Qwen2.5-72B | 89.5 (+6.8) | 85.1 | 94.4 | **83.8** (+5.3) | 91.4 |

Per-class progression F1 is metric-dependent and not dominated by any single annotator -- MedGemma-27B wins on "no change" (87.9) and on the rare "worsened" class, while Qwen2.5-32B wins on "improved" (87.1, +10.5 over silver). The paper acknowledges this only briefly.

### Inference cost (Table A1, 100 sentences)

| Model | Per-query time | Tokens generated | Notes |
|---|---|---|---|
| **Qwen2.5-32B** | **2.07 s** | **38** | scale-out pick |
| MedGemma-27B | 31.62 s | 359 | "thinks aloud" |
| LLaMA3.3-70B (1 GPU) | 658.88 s | -- | effectively unusable single-GPU |

Speed -- not peak F1 -- is the binding constraint at 95k-report scale.

### Report generators benchmarked on L-MIMIC (Table 2 + Tables B4-B6)

| Model | Prior img | Prior txt | Whole BLEU-4 | Whole METEOR | Longit. BLEU-4 | Longit. METEOR | Diagnosis F1 | Progr. micro-F1 | No-change F1 | Improved F1 | Worsened F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2Gen | -- | -- | 9.3 | 13.3 | 2.4 | 5.9 | 32.5 | 33.6 | 44.6 | 0.2 | 2.7 |
| MedVersa | -- | -- | 15.0 | 16.5 | 2.9 | 7.3 | 52.8 | 33.9 | 41.4 | 9.4 | 14.0 |
| L_R2Gen | yes | yes | 9.0 | 12.9 | 3.4 | 6.5 | 42.1 | 38.8 | 51.5 | 2.3 | 2.1 |
| MLRG | yes | yes | 15.0 | 16.8 | 6.9 | 9.7 | 52.8 | 35.7 | 44.4 | 10.5 | 12.5 |
| HC-LLM | yes | yes | 11.6 | 15.6 | 5.4 | 9.0 | 43.4 | 38.0 | 48.6 | 11.2 | 5.3 |
| **Maira2** | **yes** | **yes** | 18.9 | 19.5 | 9.6 | 12.5 | **58.2** | **41.4** | 49.7 | **17.7** | **18.3** |
| Libra | yes | no | **23.6** | **22.6** | **12.7** | **13.9** | 57.6 | 41.0 | **50.5** | 13.9 | 14.9 |

Take-aways:

- **The whole-report vs longitudinal gap is the headline of the table.** Libra's BLEU-4 nearly halves (23.6 -> 12.7) when restricted to longitudinal sentences; for R2Gen and L_R2Gen it collapses to single digits. The framework surfaces exactly the failure mode CheXbert-only metrics hide.
- **Prior-input models do beat priorless ones, but not by much on progression.** R2Gen/MedVersa sit at ~33% micro-F1; the five prior-aware models cluster in 35.7-41.4%. Maira2's +7.8 over R2Gen is well below what the model cards advertise.
- **Class imbalance dominates the per-class story.** Every model is best at "no change" (41-52 F1) and worst at "improved" (0.2-17.7) and "worsened" (2.1-18.3). R2Gen and L_R2Gen essentially never predict the minority classes -- they have collapsed to the majority label.
- **No variance reporting.** Single-run numbers throughout. Differences <2 F1 (e.g., Maira2 41.4 vs Libra 41.0) should not be over-interpreted.

## Limitations

**Authors acknowledge.**

- LLMs under-detect "worsened" progression (Qwen2.5-32B missed 46/190 worsened cases on the ImaGenome subset); newly emerged diseases are sometimes not labeled as worsened.
- Validation limited to chest X-ray; no other anatomy/modality -- **despite this paper sitting in our CT-Report-Generation category bucket, the paper itself has nothing on CT**.
- Generic LLM hallucination/bias on rare diseases; internet-augmented LLMs are floated as future work.

**Not acknowledged but worth flagging.**

- **No human inter-annotator agreement on the L-MIMIC test split itself** -- only on the upstream ImaGenome 1,975-sentence subset. Every per-model number in Table 2 inherits Qwen2.5-32B's blind spots.
- **No comparison against GREEN, RadGraph-F1, RaTE-Score, or any LLM-as-judge metric** -- the only baseline annotator pitted against is the rule-based ImaGenome silver. No correlation analysis between L-MIMIC progression-F1 and expert radiologist preference.
- **No inter-LLM agreement statistics** on L-MIMIC (e.g., Qwen2.5-32B vs MedGemma-27B disagreement rate would bound the noise floor).
- **No prompt-design ablation** (one- vs few-shot, with/without the "New = worsened" worked-example cues).
- **No seed variance / bootstrap CIs.** Single-run reports throughout.
- **The error taxonomy (missing / redundant / wrong-trend) is asserted, not validated.** A controlled experiment with synthetically injected errors would be needed to show the framework's metrics actually pick up each class.

## Why It Matters for Medical AI

Longitudinal radiology report generation has been the field's nominal direction for two years, but every paper has been forced to score itself on cross-sectional CheXbert F1 or rule-based TEM phrase matching -- neither of which actually measures whether the model tracked the change from the prior study. This paper provides the first open, executable evaluation that does. Even if L-MIMIC's labels carry Qwen2.5-32B's biases, the practical impact is real: the 41.4% progression-F1 ceiling now lives in the literature as a number generation-side researchers have to beat, and the whole-report-vs-longitudinal BLEU-4 collapse (23.6 -> 12.7 on Libra) tells reviewers exactly which papers' "longitudinal" framing is structural and which is decoration. The natural next steps -- a human IAA pass on the L-MIMIC test split, cross-comparison against GREEN / RadGraph-F1, and a CT-modality extension -- are well-defined enough that any follow-up paper can pick them up.

## References

- Wang, X., Figueredo, G., Li, R., Chen, X. *Standardizing Longitudinal Radiology Report Evaluation via Large Language Model Annotation.* arXiv:2601.16753v1 [cs.CL], 23 Jan 2026. https://arxiv.org/abs/2601.16753
- Code & L-MIMIC dataset: promised on acceptance (not yet public at submission).
- Related: Johnson et al., *MIMIC-CXR* (2019); Wu et al., *Chest ImaGenome* (2021); TEM (Liu et al., 2024); MIMIC-Diff-VQA (Hu et al., 2023); MS-CXR-T (Bannur et al., 2023); CheXbert (Smit et al., 2020); GREEN (Ostmeier et al., 2024); RadGraph-F1 (Jain et al., 2021); RaTE-Score (Zhao et al., 2024).
- Generators benchmarked: R2Gen, MedVersa, L_R2Gen, MLRG, HC-LLM, Maira2, Libra.
