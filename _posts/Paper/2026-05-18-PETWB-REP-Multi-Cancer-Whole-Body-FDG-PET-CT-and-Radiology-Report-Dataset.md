---
title: "PETWB-REP: A Multi-Cancer Whole-Body FDG PET/CT and Radiology Report Dataset for Medical Imaging Research"
excerpt: "490 paired whole-body FDG PET/CT studies with bilingual radiology reports — released without a single baseline benchmark."
categories: [Paper, CT-Report-Generation, Dataset]
permalink: /paper/petwb-rep/
tags:
  - PETWB-REP
  - PET/CT
  - Radiology Report
  - Dataset
  - Multi-Cancer
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-18
last_modified_at: 2026-05-18
---

## TL;DR
- **N = 490 paired whole-body 18F-FDG PET/CT studies** with bilingual (Chinese + post-edited English) free-text radiology reports and structured metadata, distributed via Zenodo.
- Single-center, single-scanner (Shanghai Universal, Siemens Biograph 64) retrospective cohort — "multi-cancer" in coverage but heavily skewed: lung cancer alone is **34.29%** of cases.
- **Zero baseline experiments are reported** — no VLM, report-generation, or segmentation numbers. Downstream utility is asserted, not measured.

## Motivation
Public PET/CT corpora are either single-cancer (Lung-PET-CT-Dx, Head-Neck-PET-CT) or imaging-only with segmentation labels (FDG-PET-CT-Lesions / AutoPET). Almost none ship the *free-text radiology report* that clinicians actually produce. Multi-modal medical AI — report generation, PET/CT-grounded multimodal LLMs, radiomic+linguistic biomarker work — needs image+report pairs across multiple tumor types. PETWB-REP positions itself at that gap: a curated, de-identified, whole-body, multi-cancer image+report resource that downstream image-to-text work can pick up without a new IRB.

## Core Innovation
PETWB-REP is not a model — it is a *pairing*. The contribution is releasing whole-body functional+anatomical imaging alongside the matching narrative radiology report, in both the original Chinese and a physician-validated English translation, with a fully specified preprocessing pipeline (SUV conversion, z-score CT, PET→CT B-spline registration, NIfTI export) and a flat directory structure on Zenodo. Both raw and resampled volumes are released so downstream users can re-run their own preprocessing.

![PETWB-REP cancer-type distribution](/assets/images/paper/petwb-rep/fig_p003_01.png)
*Figure 1: Cancer-type distribution across the 490 PETWB-REP patients; lung cancer dominates at 34.29%, with the remaining cases spread thinly across ~10 tumor types.*

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Public PET/CT datasets pairing functional+anatomical imaging with detailed reports across multiple cancers are scarce. | Narrative comparison to Lung-PET-CT-Dx, Head-Neck-PET-CT, AutoPET / FDG-PET-CT-Lesions. | ⭐⭐ |
| C2 | Release of 490 paired whole-body PET/CT studies with matching reports. | Dataset itself; Data Records section; Figure 4 directory tree. | ⭐⭐⭐ |
| C3 | Multi-cancer coverage. | Cancer-type distribution (lung 34.29%, liver 10.00%, cervical 7.76%, pancreatic 5.51%, lymphoma 5.31%, renal 5.31%). | ⭐⭐⭐ for diversity *as listed*; ⭐ for *balance* — lung dominates ~1/3 of the corpus. |
| C4 | Rigorous de-identification across DICOM headers and reports. | Two-pass manual review + independent post-hoc audit (S. Wang). | ⭐⭐ — qualitative only; no quantitative PHI-leak metric or automated-scrubber comparison. |
| C5 | Reports are clinically reliable. | Authored by >10 y nuclear medicine physician, reviewed by >20 y senior; bilingual validation by S. Peng. | ⭐⭐ — no inter-rater κ reported. |
| C6 | Dataset is suitable for tumor segmentation, automated report generation, NLP extraction, and multi-modal fusion. | Stated in Usage Notes; **no baseline experiments**. | ⭐ — aspirational. |
| C7 | Image normalization, registration, and SUV conversion enable downstream use. | Explicit pipeline, SUV formula, dcm2niix conversion, B-spline PET→CT registration; both raw and resampled volumes released. | ⭐⭐⭐ for pipeline traceability. |
| C8 | Translated English reports preserve clinical meaning. | Google Translate + bilingual physician post-edit. | ⭐⭐ — no BLEU / RadGraph / entity-preservation score against a reference human translation. |

**Honest read.** The well-supported claims are the data-intrinsic ones — the corpus exists, the preprocessing pipeline is fully specified, and the cancer-type distribution is what is stated. The weak claims sit at the edges:

1. **"Multi-cancer" is true in coverage, misleading on balance.** Lung cancer is over a third of the cases; several listed types each have only a few percent. Anything trained on PETWB-REP without rebalancing will be a lung-biased model.
2. **De-identification quality is qualitative.** There is no quantitative residual-PHI audit (e.g., NER scrub recall, comparison to a reference scrubber). "Two researchers re-reviewed all files" is process, not measurement.
3. **English reports are downstream artifacts.** Google-translated Chinese text post-edited by one bilingual physician — no translation-fidelity score is reported.
4. **Downstream-application claims with zero baselines.** The paper asserts utility for segmentation, report generation, and multi-modal fusion but runs *no* baseline VLM, report-gen, or Dice number. Anyone citing PETWB-REP for "useful for report generation" is taking the authors at their word.
5. **Single-center, single-scanner.** Acknowledged but not quantified; no external validation cohort.

## Method & Architecture
This is a data descriptor — the "architecture" is the data construction pipeline.

![PETWB-REP construction pipeline](/assets/images/paper/petwb-rep/fig_p006_01.png)
*Figure 3: PETWB-REP construction pipeline — collection, de-identification, NIfTI conversion, SUV/z-score normalization, PET-to-CT registration, QC, and Zenodo release.*

**Cohort.** Retrospective, Shanghai Universal Medical Imaging Diagnostic Center, 2021–2024. Inclusion: confirmed malignancy + whole-body FDG PET/CT + matching report. IRB waived patient consent (retrospective + full de-identification). Final N = 490 (219 F / 271 M, mean age 60.98 ± 12.77).

**Acquisition.** 6 h fast, blood glucose < 11.1 mmol/L, IV 18F-FDG at 3.70–5.55 MBq/kg, 60-min uptake, skull base to mid-thigh on a Siemens Biograph 64. Low-dose CT for attenuation correction (120 kV, 170 mA, 3.0 mm slice). PET in 3D, 5–6 bed positions, 2.5 min/bed.

**Reconstruction.** OSEM, 2 iterations × 21 subsets, 5-mm Gaussian post-smoothing.

**Report authoring.** Board-certified nuclear medicine physician (>10 y), reviewed by senior specialist (>20 y). Findings organized by anatomic region (head/neck, chest, abdomen/pelvis, musculoskeletal); lesion size as single transaxial diameter; metabolism as SUVmax; plus Impression.

**Bilingual processing.** Originally Chinese; English produced via Google Translate, then validated/edited by a bilingual senior nuclear medicine physician. Both `*_zh.csv` and `*_en.csv` are released.

**De-identification.** All PHI removed from DICOM headers and report text (names, IDs, accession numbers, DOB, dates) and replaced with generic codes; two researchers manually re-reviewed; an independent reviewer audited NIfTI headers and reports for residual PHI.

**Normalization.** CT z-score standardized. PET converted to body-weight SUV:

$$\mathrm{SUV}_{bw} = \frac{RC\,(\mathrm{kBq/mL})}{ID\,(\mathrm{MBq})/BW\,(\mathrm{kg})}.$$

**Resampling + registration.** CT and SUV-PET resampled to 0.98 × 0.98 mm in-plane, 3.00 mm slice; PET registered to CT space via B-spline interpolation. Both raw and resampled volumes released.

**Distribution.** Released via Zenodo, split into `Imaging_data/` (per-subject `CT_RAW`, `PET_RAW`, `CT_NORM`, `PET_SUV`) and `Non-Imaging_data/` (`meta_data.csv`, `report_en.csv`, `report_zh.csv`).

![Representative PETWB-REP case](/assets/images/paper/petwb-rep/fig_p004_01.png)
*Figure 2: A representative whole-body FDG PET/CT scan paired with its de-identified radiology report — the unit of release in PETWB-REP.*

## Experimental Results
**There are no model-training experiments and no baseline benchmarks in this paper.** The "results" are a Technical Validation section.

| Validation axis | Procedure | Outcome |
|---|---|---|
| Image quality control | Visual review of all 490 cases by an experienced nuclear medicine physician for motion / truncation / metal artifacts and PET–CT registration | Pass (failing cases excluded upstream per inclusion criteria) |
| De-identification audit | Independent reviewer inspected NIfTI headers and reports for residual PHI | Confirmed absence of remaining PHI (qualitative) |
| Data integrity | Verified each subject folder has both PET + CT series and a matching report file consistent with `metadata.csv` | Verified |
| Report–metadata consistency | Cross-checked primary diagnosis in `metadata.csv` against report content | Confirmed |

No quantitative AI benchmark (segmentation Dice, report-gen BLEU/ROUGE/RadGraph, retrieval R@k), no inter-rater κ on the source reports, no test–retest variability, no automated PHI-scrub recall, and no translation-fidelity score on the English reports are reported.

![PETWB-REP directory layout](/assets/images/paper/petwb-rep/fig_p008_01.png)
*Figure 4: PETWB-REP directory layout — `Imaging_data/` (raw + normalized CT and PET per subject) and `Non-Imaging_data/` (`meta_data.csv`, `report_en.csv`, `report_zh.csv`).*

## Limitations

**Authors admit:**
- Single-institution, single-scanner — generalization is unproven.
- Retrospective collection causes missing / inconsistent clinical variables.
- Non-uniform cancer distribution mirrors referral patterns.

**Analyst additions (not surfaced in the paper):**
- **Zero baseline benchmarks.** A dataset release without numbers — no VLM, report-generation, or segmentation Dice — makes downstream "this dataset is useful for X" claims unverifiable from this paper alone.
- **No lesion-level annotations.** No bounding boxes, masks, or RECIST. Only narrative report + SUVmax.
- **2D lesion measurement.** Single transaxial diameter limits 3D radiomics, MTV/TLG computation, and volumetric modeling.
- **No inter-rater agreement** on source reports.
- **No quantitative de-identification audit** (e.g., NER scrub recall on a held-out PHI-inserted subset).
- **No translation-fidelity score** for English reports (BLEU / clinically-aware entity preservation against a reference human translation).
- **Zenodo license string is not quoted** in the extracted manuscript body — users must consult the Zenodo record before redistribution or commercial use.
- **No predefined train/val/test split** — comparability across future papers using PETWB-REP is at risk.
- **N = 490 is small** for deep learning on whole-body 3D PET/CT, especially split across ≥10 cancer types. For scale comparison, a PETARSeg-11K-style corpus (5,126 exams) is roughly 10× larger on imaging count — PETWB-REP's niche is the *paired report*, not the count.

## Why It Matters for Medical AI
The pairing of whole-body FDG PET/CT with narrative radiology reports is genuinely rare in the public domain — AutoPET / FDG-PET-CT-Lesions has more PET/CT cases (>1000) but no free-text reports, while CT-RATE / RadGenome / MIMIC-CXR-style report corpora are larger on the text side but not whole-body PET/CT. For groups training PET/CT report-generation models, multimodal LLM grounding on functional+anatomical imaging, or radiomic+linguistic biomarker pipelines, PETWB-REP is one of the few public starting points. The caveats above (lung dominance, single scanner, no baselines, no lesion masks, translation artifacts) mean any paper built on PETWB-REP should report (i) a cancer-type-stratified evaluation rather than a single aggregate metric, (ii) a documented preprocessing trail, and (iii) an external-cohort generalization check before drawing population-level conclusions.

## References
- Paper: PETWB-REP: A Multi-Cancer Whole-Body FDG PET/CT and Radiology Report Dataset for Medical Imaging Research — arXiv:2511.03194 (November 2025).
- Dataset: Zenodo record (consult the live record for the exact license string and DOI).
- Related: Lung-PET-CT-Dx; Head-Neck-PET-CT; FDG-PET-CT-Lesions / AutoPET; CT-RATE; RadGenome; MIMIC-CXR.
