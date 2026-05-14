---
title: "Merlin: A Computed Tomography Vision-Language Foundation Model and Dataset"
excerpt: "A 3D ResNet-152 + clinical Longformer pretrained on 25K Stanford abdominal CTs with paired radiology reports and ICD-derived PheWAS phenotypes, hitting zero-shot findings F1 0.741 internal / 0.647 external on a single 48 GB A6000."
categories:
  - Paper
  - CT-Report-Generation
  - Pathology
permalink: /paper/merlin-3d-ct-vision-language-foundation-model/
tags:
  - Merlin
  - 3D-CT
  - Vision-Language-Model
  - Foundation-Model
  - Contrastive-Pretraining
  - PheWAS
  - Report-Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-13
last_modified_at: 2026-05-13
---

## TL;DR

- **Merlin is a 3D vision-language foundation model for abdominal CT** — an I3D-inflated ResNet-152 image encoder paired with a clinical Longformer text encoder (4,096-token context), jointly supervised by ICD-derived **PheWAS phenotypes (BCE)** and full-volume **radiology reports (InfoNCE)** on 25,494 Stanford CTs, trained on a **single 48 GB A6000 in ~160 hours**.
- The key training trick is **anatomy-aware "report splitting"**: every other step alternates between the full findings section and a single-organ section, matching the distribution of downstream zero-shot prompts. Removing it drops zero-shot F1 from **0.741 to 0.656** — almost the entire internal-vs-external gap.
- **Headline result: zero-shot findings F1 = 0.741 (internal, 30 classes)** and **0.647 (external clinical)**, ~2.6x the OpenCLIP / finetuned BioMedCLIP baselines (~0.28). Linear-probe foundation-model comparisons look dominant; under full fine-tuning the gap collapses from ~55% to ~8%, so the linear-probe story oversells the absolute embedding advantage.

## Motivation

The US runs ~85M CT scans a year, ~1/4 of them abdominal, each with 300+ slices and ~20 minutes of radiologist time, against a workforce projected to be short by ~19,000 readers by 2036. Most existing radiology VLMs (MedCLIP, BiomedCLIP, LLaVA-Med, RadFM, Med-PaLM M, ...) are **2D and short-text-bound** even though most diagnostic imaging is volumetric and 21% of findings sections exceed 512 tokens. Compute is also a constraint: hospital systems that hold the data rarely have multi-node GPU clusters.

Merlin's pitch is a **single-GPU, fully 3D, EHR + report-supervised** foundation model that can serve as a generic backbone across diagnostic, prognostic, retrieval, segmentation, and report-drafting tasks on abdominal CT — and that can be retrained inside a hospital firewall.

## Core Innovation

- **Full 3D inputs, not 2D slices or 2.5D stacks.** Volumes are RAS-reoriented, resampled to 1.5x1.5x3 mm, HU-clipped to [-1000, 1000], and pad/center-cropped to **224x224x160**, fed end-to-end to an I3D-inflated ResNet-152 (2D ImageNet weights inflated along depth; out-of-plane stride 1 / kernel 3 in the stem — smaller receptive field beat larger in their phenotype ablation).
- **Joint EHR + report supervision.** 16,553 unique ICD-9/10 codes are mapped to **1,692 PheWAS phenotypes** (positives propagated up the hierarchy) and supervised by BCE on a 1,692-dim head; reports are aligned via InfoNCE between the image embedding and a **clinical Longformer (4,096 tokens)** text embedding.
- **Report splitting matches train and test prompt distributions.** Every other training step replaces the full findings section with a single-anatomy section (lower thorax, liver/biliary, gallbladder, spleen, pancreas, adrenals, kidneys/ureters, GI, peritoneum, pelvic organs, vasculature/lymph nodes, MSK), so contrastive prompts look like downstream zero-shot prompts.
- **Single-GPU pretraining.** AdamW lr 1e-5, cosine to 0 over 300 epochs, FP16 + gradient checkpointing on both encoders, batch size 18 on a 48 GB A6000.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Merlin is a 3D VLM that processes the entire CT volume at once | Architecture in Section 4.2; 224x224x160 input; I3D ResNet-152 | — | ⭐⭐⭐ |
| C2 | Trained on a single GPU in ~160 h | Reported in Section 2; A6000 48 GB; batch 18, FP16, gradient checkpointing | — | ⭐⭐ (single number, no log) |
| C3 | Zero-shot findings F1 0.741 internal / 0.647 external significantly beats OpenCLIP and finetuned BioMedCLIP | Table 4 + Figure 2b, p<0.001, 30 findings, bootstrap CIs | Internal 5,137 CT; external clinical | ⭐⭐⭐ |
| C4 | Generalises to fracture detection (zero-shot) | Table 4: VerSe F1 0.767 [0.630, 0.867] | VerSe (public) | ⭐ — single dataset, CI width 0.24, and the "zero-shot" pipeline crops sub-volumes using the **fracture annotations themselves** |
| C5 | Phenotype macro AUROC 0.812 over 692 phenotypes | Table 6, Figure 3a-d, ablations across ResNet/ConvNeXt/Swin | Internal 5,137 CT | ⭐⭐⭐ |
| C6 | 5-yr chronic-disease prediction AUROC 0.757, +7% over ImageNet I3D at 100% labels and +4.4% at 10% | Figure 5b/c | Internal | ⭐⭐ — only 9-20 positives per disease in the 10% regime, no external 5-yr cohort |
| C7 | Beats RadFM on report generation across all sections and metrics | Table 11; Figure 6b: full findings RadGraph-F1 **0.293 vs 0.008**, BLEU 0.102 vs 0.000, BERTScore 0.588 vs 0.224 | Stanford internal test | ⭐ — single weak baseline; absolute scores are still demo-range; qualitative example **misses the cholelithiasis finding** present in both human report and image |
| C8 | Beats nnUNet on 12/20 organs at 10% labels | Figure 7b-c; +4.7% avg Dice at 10%; -0.006 Dice at 100% | TotalSegmentator filtered 401 scans | ⭐⭐ — clear win in label-scarce regime; full-data result is essentially a tie that goes to nnUNet |
| C9 | Beats 2D VLMs, 2D-to-3D lifted VLMs, 3D vision-only SSL, MedImageInsight, Google CT FM via linear probe | Figure 8d-e + Supplemental Figure 14 (full FT) + Figure 15 | Internal | ⭐⭐⭐ on linear probe; ⭐⭐ on full FT — **gap shrinks from 54.7% to 8.1%** under full fine-tuning |
| C10 | Generalises to 44,098 external CTs across 3 sites | Figure 9, Table 12 | 3 external US sites | ⭐⭐ — Merlin retains the lead but **zero-shot F1 drops 0.741 to 0.647 and retrieval Recall@1 drops 0.696 to 0.328** |
| C11 | Outperforms chest-CT FMs (CT-CLIP, M3FM, CT-FM) on chest CT despite training only on abdomen | Figure 9c-d, AUC over 24 conditions | Site #3 chest-CT, 6,243 test scans | ⭐ — labels were generated by **Qwen3-30B without human review**; circular failure mode if Merlin and Qwen exploit the same report-language patterns |
| C12 | Multi-task EHR + reports beats either alone | Tables 5, 7, 9 | Internal | ⭐⭐ — consistent direction but small effect (report-only often within 1 pt) |
| C13 | I3D init beats random init | Tables 5, 7 (zero-shot F1 0.698 random vs 0.741 I3D) | Internal | ⭐⭐⭐ |

**Honest read.** The strongest claims are zero-shot findings classification (C3), phenotype classification (C5), and the linear-probe foundation-model comparisons (C9): they have multi-finding evidence, bootstrap CIs, sensible ablations, and head-to-head comparisons with current public 3D CT FMs. The headline narrative — *"3D VL pretraining on full volumes beats 2D and 2D-to-3D lifted baselines"* — is well-supported, with the important caveat that under **full fine-tuning the gap collapses from ~55% to ~8%** (Supplemental Section C.1, Figure 14), so the linear-probe numbers in Figure 8 oversell the absolute supremacy of the embeddings.

The **report-generation claim (C7) deserves the most scepticism.** Numerically Merlin crushes RadFM (RadGraph-F1 **0.293 vs 0.008**), but: (i) RadFM is a single, weak baseline whose abdominal-CT exposure is incidental — there is no comparison to MAIRA, CheXagent, Med-PaLM M, or even an LLM-only "report-the-template" baseline; (ii) the absolute scores (BLEU 0.10, ROUGE-2 0.26, RadGraph-F1 0.29 on the full report) are low-to-moderate by report-gen standards and the authors themselves call this an "early demonstration"; (iii) the qualitative example in Figure 6c shows Merlin **missing the cholelithiasis finding** that exists in both the human report and the image — a clinically material false negative — and supplemental Figure 13 shows additional issues including invented series/slice locations, contradictory sentences ("postcholecystectomy" vs "gallbladder appears normal"), and reports nearly 2x the length of the human reference.

## Method & Architecture

![Merlin pretraining and downstream evaluation overview](/assets/images/paper/merlin/page_004.png)
*Figure 1: Merlin pretraining strategy with paired CT volumes, EHR phenotype codes (BCE) and radiology reports (InfoNCE), plus the six downstream evaluation families (Figure 1).*

**A. Dataset assembly.** Stanford academic medical center, IRB-approved, retrospective abdominal CTs Dec 2012-Oct 2018; 18,321 patients (37% inpatient, 35% ED, 16% outpatient, 8% observation; 56% female; 47% non-Hispanic white, 28% unknown race). For each patient, take the abdominal series with the most axial slices (97% portal-venous phase per a published contrast-phase classifier), pair with the report's findings section and with all ICD-9/ICD-10 codes from the encounter that produced the scan.

**B. EHR preprocessing.** 16,553 unique ICD codes -> 1,692 PheWAS phenotypes via the Phecode mapping; positives propagated up the phenotype hierarchy (a positive child also flips parent labels). The final supervision per CT is a 1,692-dim binary vector — explicitly described as **weak (encounter-level, not pixel-level)**.

**C. Report preprocessing.** Regex extracts the findings section. Because long reports risk overfitting to short salient spans and create train/inference distribution shift vs zero-shot prompts, every other training step alternates between the full findings section and a single-anatomy section.

**D. CT preprocessing.** RAS reorientation, resample to 1.5x1.5x3 mm, HU clip [-1000, 1000] -> [0, 1], pad/center-crop to 224x224x160.

**E. Architecture.** I3D ResNet-152 image encoder (2D ImageNet weights inflated along the depth axis), out-of-plane stride 1 / kernel 3 in the stem; embedding dim 512. Text encoder is **clinical Longformer** with 4,096 context (chosen because 21% of findings exceed 512 tokens — Figure 4b). Architecture ablations also test ConvNeXt-{T,S,B,B*} and Swin Transformer; ResNet-152 wins on phenotype AUROC.

**F. Training objective.** BCE on the 1,692-dim phenotype head + InfoNCE on (image embedding, report embedding) pairs. Multi-task learning (MTL) outperforms a staged variant where stage 1 = phenotype only and stage 2 = phenotype loss (down-weighted) + InfoNCE.

**G. Optimizer / schedule.** AdamW, lr 1e-5, betas (0.9, 0.999), cosine decay to 0 over 300 epochs, FP16 mixed precision, gradient checkpointing on both encoders. Maximum batch size 18 on a single 48 GB A6000.

**H. Adaptation heads.**
- **5-year disease prediction**: one-head-per-disease multi-task fine-tune; Class 0 = no disease in 5-yr window (requires >=5 yr EHR follow-up), Class 1 = newly diagnosed in (t_s, t_a=5 yr], Class 2 = already diagnosed pre-CT (excluded), Class 3 = censored (excluded).
- **Report generation**: last hidden block (7x7x10x2048) -> linear adapter -> 4,096-dim, prepended to a per-section prompt `"<visual tokens>Generate a radiology report for <organ system>### <report section>###</s>"`, decoded by **RadLlama-7B** (Llama2-7B fine-tuned on MIMIC reports) with the adapter and 5% of LLM weights via LoRA. Reports are generated section-by-section, then concatenated for "full report" scoring.
- **Segmentation**: UNet decoder with skip connections from the Merlin encoder, full 224x224x160 volumes inside the nnUNet framework (default augmentation, 1,000-epoch schedule, polynomial LR).

![Zero-shot findings ablation — report splitting is the load-bearing trick](/assets/images/paper/merlin/fig_p018_03.png)
*Figure 2: Zero-shot findings ablation — I3D + MTL + report splitting is the only configuration that hits F1 0.741; removing report splitting costs 7.9 F1 points (Figure 2e).*

## Experimental Results

### Headline quantitative comparison

| Task | Metric | **Merlin** | Best baseline | Notes |
|---|---|---|---|---|
| Zero-shot findings (30 cls, internal) | F1 | **0.741 [0.727, 0.755]** | OpenCLIP K=1 0.276 / finetuned BioMedCLIP avg 0.285 | Table 4; p<0.001 |
| Zero-shot findings (external clinical) | F1 | **0.647 [0.607, 0.678]** | — | Table 4 |
| Zero-shot findings (VerSe vertebral fx) | F1 | **0.767 [0.630, 0.867]** | — | Table 4; CI width 0.24; pipeline crops via fracture annotations |
| Phenotype classification (692 codes) | macro AUROC | **0.812 [0.808, 0.816]** | ResNet-152 EHR-only 0.798 | Table 6 |
| Image->Findings retrieval (N=64, internal) | Recall@1 | **0.696** | OpenCLIP 0.016 / BiomedCLIP 0.021 | Figure 4c |
| Image->Findings retrieval (external) | Recall@1 | **0.328** | (~1/64 baseline) | Figure 4c — drops sharply OOD |
| 5-yr disease prediction (avg of 6, 100% labels) | AUROC | **0.757 [0.743, 0.772]** | ImageNet I3D 0.708 | Figure 5; with 10% labels Merlin 0.708 |
| Report gen, full findings | RadGraph-F1 | **0.293** | RadFM 0.008 | Table 11 — single weak baseline |
| Report gen, full findings | BERTScore | **0.588** | RadFM 0.224 | Table 11 |
| Report gen, full findings | ROUGE-2 | **0.262** | RadFM 0.011 | Table 11 |
| Report gen, full findings | BLEU | **0.102** | RadFM 0.000 | Table 11 |
| Segmentation (TotalSegmentator, 20 organs, 100% labels) | avg Dice | **0.91** | nnUNet 3D-fullres 0.92 | Figure 7b — nnUNet wins by 0.006 |
| Segmentation (10% labels) | avg Dice | **0.92** | nnUNet 0.88, ImageNet I3D 0.91 | Figure 7b — clearest seg win is in label-scarce regime |
| Chest-CT linear probe (24 dz, Site #3) | avg AUC | **best** | CT-CLIP -24.7%, M3FM -14.0%, CT-FM equivalent | Figure 9c-d; **labels generated by Qwen3-30B** |

![Zero-shot findings F1 across internal, external, and VerSe](/assets/images/paper/merlin/fig_p018_02.png)
*Figure 3: Zero-shot findings classification F1 — Merlin internal 0.74 vs OpenCLIP 0.36 / BioMedCLIP 0.43; external 0.65 and VerSe 0.77 (Figure 2b, Table 4).*

![Per-finding F1 radar over 30 abdominal findings](/assets/images/paper/merlin/fig_p018_01.png)
*Figure 4: Per-finding F1 radar — Merlin holds up on coarse findings (pleural effusion, splenomegaly, AAA) and degrades on fine ones (appendicitis, lymphadenopathy, free air) (Figure 2c).*

![Zero-shot data-scaling power law](/assets/images/paper/merlin/fig_p018_08.png)
*Figure 5: Zero-shot data-scaling curve, F1 = 0.458 * D^0.0524, indicating diminishing but non-zero returns at this data scale (Figure 2d).*

![Image->Findings retrieval Recall@1](/assets/images/paper/merlin/fig_p020_01.png)
*Figure 6: Recall@1 in pools of 64 — Merlin Image<->Findings 0.696/0.687 internal vs ~0.02 for OpenCLIP / BioMedCLIP; drops to ~0.33 externally (Figure 4c).*

![5-year chronic-disease prediction AUROC](/assets/images/paper/merlin/fig_p021_07.png)
*Figure 7: 5-year chronic-disease prediction — Merlin AUROC 0.757 vs ImageNet I3D 0.708 (100% labels); positives as low as 9-20 per disease in the 10% regime (Figure 5b).*

![Qualitative report generation example](/assets/images/paper/merlin/fig_p022_01.png)
*Figure 8: Qualitative report-gen example — Merlin correctly flags a thickened endometrial stripe but **misses the cholelithiasis** present in both the human report and the image (Figure 6c). Supplemental Figure 13 shows additional invented series/slice locations and contradictory sentences (e.g., "postcholecystectomy" vs "gallbladder appears normal").*

![20-organ Dice at 100% vs 10% labels](/assets/images/paper/merlin/fig_p023_01.png)
*Figure 9: 20-organ Dice at 100% vs 10% labels — Merlin (0.92) ties nnUNet (0.92) at 100% and beats it (0.92 vs 0.88) at 10% (Figure 7b).*

![Linear-probe foundation-model comparison](/assets/images/paper/merlin/fig_p024_01.png)
*Figure 10: Foundation-model linear-probe comparison vs 2D VLMs, 2D-to-3D lifted VLMs, 3D vision-only SSL, MedImageInsight, and Google CT-FM. Note: under full fine-tuning the gap collapses from ~55% to ~8% (Supplemental Figure 14), so the linear-probe lead overstates absolute embedding superiority.*

### Ablations and qualitative notes worth flagging

The most consequential ablation (Figure 2e / Table 5) is **report splitting**: removing it drops zero-shot F1 from 0.741 to 0.656 (-7.9 F1 pts, p << 0.01) — almost the entire gap between "Merlin internal" and "Merlin external" is reproducible by simply turning off report splitting. Conversely, on retrieval (Figure 4d, Table 9) splitting *hurts* slightly, because retrieval prompts are full sections; the authors are honest about the prompt-distribution-matching mechanism.

The EHR signal is real but small: report-only contrastive training already reaches zero-shot F1 0.730 (Table 5), so the EHR head adds ~1 F1 pt — the paper itself concedes this in Section 3.

Power-law fits: zero-shot F1 = 0.458 * D^0.0524, phenotype AUROC = 0.479 * D^0.0568 — diminishing but non-zero returns at this scale.

Counterfactual analysis (Figure 11) on pleural effusion and splenomegaly shows latent-shift edits to the relevant anatomy actually move predictions, providing weak but non-trivial evidence the model uses clinically-relevant features rather than shortcuts.

## Limitations

**Authors admit**: single-GPU training caps performance scaling; few public abdominal-CT VLM baselines exist; report-gen adapter and LLM choice are under-explored; segmentation decoder design is preliminary.

**Not addressed (or under-addressed)**:

- **No human reader study** comparing Merlin draft reports to dictated reports for accuracy or time savings — the central clinical value proposition is unverified.
- **Report generation is evaluated only on Stanford internal test data**; no external report-gen evaluation despite 44K external CTs being available.
- The **VerSe fracture-detection "zero shot" uses fracture annotations to crop sub-volumes**, which is at least partial supervision — labelling it pure zero-shot is a stretch.
- **Chest-CT external labels are LLM-generated by Qwen3-30B** with no human-annotated subset to calibrate how much the comparison rewards Merlin specifically.
- **No fairness analysis** along race / sex / age subgroups, despite 28% unknown race in the training cohort and known disparities in CT utilisation.
- **No calibration analysis** (only AUROC/F1) — particularly important for the 5-year prediction claim, which is positioned as a screening tool.
- The text encoder is clinical Longformer pretrained on different clinical text; **no analysis of whether report-gen errors correlate with text-encoder hallucinations vs image-encoder limitations**.
- **All "external" sites are still US health systems** (one out-of-state); no international, low-resource, or different-vendor stress test reported.
- The headline "6M+ images / 15K CTs / 1.8M EHR codes / 6M report tokens" numbers correspond to the **train split only** — a useful spec to be aware of when comparing cohort sizes.

## Why It Matters for Medical AI

Merlin is the first credible attempt at a **fully 3D, jointly EHR + report-supervised, single-GPU-pretrainable** foundation model for abdominal CT, and its zero-shot findings, phenotype, and linear-probe numbers are strong enough that it is a reasonable default backbone for downstream abdominal-CT work where labels are scarce. The training recipe — anatomy-aware report splitting, PheWAS phenotype heads, I3D inflation, clinical Longformer for long findings — is general enough that other groups should be able to replicate it on their own institutional CT corpora. Code, weights, and a manually de-identified 25,494-CT dataset are released at the StanfordMIMI/Merlin repository.

The cautionary half of the story is just as important. Under **full fine-tuning**, Merlin's lead over alternative pretraining recipes shrinks from ~55% to ~8% — the embedding superiority shown in linear probes does not fully survive end-to-end training. **Report generation** beats a single weak baseline (RadFM) on automatic metrics but still produces clinically material errors (the cholelithiasis miss; invented slice locations; contradictory sentences) and has not been evaluated by any human reader. The **chest-CT generalisation claim** rests on labels generated by Qwen3-30B, and the **VerSe "zero-shot" fracture detection** uses fracture annotations to crop the input volumes. In short: the diagnostic-classification and retrieval claims are solid foundation-model wins; the generative and out-of-domain claims should be read as promising early demonstrations, not deployment-ready capabilities.

## References

- **Paper (arXiv)**: [Merlin: A Computed Tomography Vision-Language Foundation Model and Dataset (arXiv:2406.06512)](https://arxiv.org/abs/2406.06512)
- **Code & weights**: [github.com/StanfordMIMI/Merlin](https://github.com/StanfordMIMI/Merlin)
- **PheWAS / Phecode mapping**: [Denny et al., PheWAS — discovering disease-disease associations from EHR data](https://phewascatalog.org/)
- **Clinical Longformer**: Li et al., *Clinical-Longformer and Clinical-BigBird: Transformers for long clinical sequences*
- **I3D**: Carreira & Zisserman, *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*, CVPR 2017
- **RadFM** (report-gen baseline): Wu et al., *Towards Generalist Foundation Model for Radiology*
- **TotalSegmentator** (segmentation benchmark): Wasserthal et al., *TotalSegmentator: Robust segmentation of 104 anatomical structures in CT images*
- **VerSe** (vertebral-fracture benchmark): Sekuboyina et al., *VerSe: A Vertebrae Labelling and Segmentation Benchmark*
