---
title: "Mamba-SSM with LLM Reasoning for Feature Selection: Faithfulness-Aware Biomarker Discovery"
excerpt: "An LLM-filtered 17-gene panel reaches AUC 0.927 on TCGA-BRCA, beating a 5,000-gene variance baseline (0.903) with 294x fewer features — while recalling only 37.5% of validated BRCA biomarkers."
categories:
  - Paper
  - Pathology
  - LLM
permalink: /paper/mamba-ssm-llm-biomarker-discovery-tcga-brca/
tags:
  - Mamba-SSM
  - Feature Selection
  - LLM Reasoning
  - Chain-of-Thought Faithfulness
  - TCGA-BRCA
  - Neuro-Symbolic
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- A Mamba SSM trained on TCGA-BRCA RNA-seq produces gradient-saliency scores; DeepSeek-R1 (7B) then applies structured chain-of-thought with explicit reject/keep criteria to compress the top-50 candidate genes into a 17-gene panel — the LLM is the symbolic layer in a neuro-symbolic feature-selection pipeline.
- Headline result: the 17-gene panel reaches **AUC 0.927**, vs. **0.903** for a 5,000-gene variance baseline and **0.832** for the unfiltered 50-gene saliency set — **+0.024 AUC over baseline with 294x fewer features**.
- The paper's strongest contribution is a faithfulness audit against COSMIC/OncoKB/PAM50 ground truth: recall on validated BRCA genes is only **0.375**, FOXA1 (a canonical luminal-lineage pioneer TF) is confidently rejected, and yet the downstream classifier still wins on every metric — a phenomenon the authors call **selective faithfulness**.

## Motivation
RNA-seq feature selection for disease biomarkers has to fight confounders that mimic biological signal: tumour purity, immune infiltration, tissue composition, and TCGA batch effects. A Mamba SSM trained end-to-end on TCGA-BRCA dutifully learns to classify tumour vs. normal, but its top-50 gradient-saliency genes include muscle markers (MB, UTRN), generic immune adhesion genes (HLA-DRB1, ITGAL), and antisense RNAs with no documented BRCA role. That is fine for accuracy on the in-distribution split but disastrous for a clinically meaningful gene panel.

The authors ask a sharper version of the usual interpretability question: can an LLM's biomedical priors act as a *symbolic* filter that separates drivers from confounders — and, more importantly, **is the LLM's stated reasoning actually faithful to biology, or just a convenient narrative attached to a metric improvement?**

## Core Innovation
Two pieces, in this order:

1. **Structured CoT as a symbolic filter.** DeepSeek-R1 is prompted with the 50 saliency-ranked genes and forced to evaluate each one against five rejection criteria (R1–R5) and three keep criteria (K1–K3), with explicit instructions that "high saliency does not imply BRCA specificity" and a ban on rank-order selection. Post-hoc checks detect rank-copy outputs and hallucinated gene names.
2. **Faithfulness audit against curated biology.** The 50-gene input and the 17-gene output are both cross-referenced against a 101-gene ground truth (COSMIC CGC Tier-1 + OncoKB BRCA + PAM50 + canonical pathway genes) plus a "known non-BRCA" negative set. This yields selection-level precision/recall and surfaces named failure cases — the paper's most defensible result.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset | Strength |
|---|---|---|---|
| C1: Raw Mamba gradient saliency surfaces confounders that hurt downstream generalisation. | B2 (50-gene saliency) AUC 0.832 vs. B1 (5,000-gene variance) AUC 0.903, i.e. -0.071. | TCGA-BRCA, single split | ⭐⭐ |
| C2: LLM structured-CoT filtering converts the saliency set into a higher-performing 17-gene panel. | B3 AUC 0.927 vs. B1 0.903 (+0.024); 294x fewer features. | TCGA-BRCA, single split | ⭐⭐ |
| C3: Improvement persists across Accuracy, F1, and AUC, not only AUC. | Acc 0.8907 > 0.8785; F1 0.9033 > 0.8941. | TCGA-BRCA, single split | ⭐⭐ |
| **C4: "Selective faithfulness" — downstream performance can improve while recall on validated biology is poor.** | Recall = 0.375 on known BRCA genes yet B3 wins on AUC. | TCGA-BRCA + COSMIC/OncoKB/PAM50 | **⭐⭐⭐** |
| **C5: LLM biomedical reasoning can be confidently wrong about canonical biology (FOXA1 rejection).** | FOXA1 at rank 49 in G50, absent from B3 despite being in PAM50 and COSMIC. | TCGA-BRCA | **⭐⭐⭐** |
| C6: Precision-oriented confounder removal matters more than exhaustive recall for downstream performance. | Inference from C2 + C4. | TCGA-BRCA | ⭐ |
| C7: Structured CoT reduces but does not eliminate implausible reasoning. | 17.6% false-positive rate, NF-kB hallucination, FOXA1 omission. | TCGA-BRCA | ⭐⭐ |
| C8: Mamba's linear-time scaling makes 20k-gene vectors tractable where attention would be quadratic. | Cited motivation only — no Transformer comparison in this paper. | — | ⭐ |

**Honest read.** C2 and C3 sit on a single 80/20 split with a single random seed, no LASSO/ElasticNet baseline, no multi-seed variance estimate, and no external cohort (METABRIC). A +0.024 AUC margin on 985 training samples at batch size 8 trained for 15 epochs is well within plausible seed variance, so the metric-level claim should be treated as **one observation, not an established effect**. The paper's genuine three-star result is the *decoupling* of performance from biological faithfulness (C4–C5), which is methodologically novel for real biology — most CoT-faithfulness work is on synthetic tasks. The "selective faithfulness" *naming* (C6), however, over-generalises: the same numbers are also consistent with "the 8 unverifiable genes happen to carry the predictive signal and the LLM filtered nothing important," and the paper does not run the per-gene ablations needed to distinguish those.

## Method & Architecture

![Neuro-symbolic Mamba + LLM feature-selection pipeline](/assets/images/paper/mamba-ssm-biomarker/page_002.png)
*Figure 1: End-to-end neuro-symbolic pipeline. A Mamba SSM is trained on TCGA-BRCA RNA-seq; gradient saliency produces a 50-gene candidate pool, which DeepSeek-R1 structured CoT filters into the final 17-gene panel.*

**Phase 1 — Mamba classifier.** `OfficialMambaClassifier`: `Linear(1, 128)` per-gene embedding (d_model=128), one Mamba block with `d_state=16, d_conv=4, expand=2`, `AdaptiveAvgPool1d(1)` over the gene-sequence dimension, then `Linear(128, 1) -> Sigmoid`. Training uses AdamW at `lr=1e-4` for 15 epochs, batch size 8, class-weighted BCE with $w_{\text{normal}} = N_{\text{tumour}}/N_{\text{normal}}$ to handle the 8.8:1 imbalance.

**Phase 2 — Gradient saliency.** For each tumour sample $i$, enable input gradients and compute

$$s_j = \frac{1}{|T|} \sum_{i \in T} \left| \frac{\partial L}{\partial x_{ij} } \right|.$$

The top 50 genes by $s_j$ form $G_{50}$ (baseline B2).

**Phase 3 — Structured CoT with DeepSeek-R1 (7B).** Local Ollama, temperature 0.3. The prompt provides the saliency scores, explicitly states "high saliency does not imply BRCA specificity," requires every candidate to be evaluated against five rejection criteria (R1–R5) and three keep criteria (K1–K3), and forbids rank-order selection. Output is the 17-gene panel (B3).

![Saliency heatmap motivating the symbolic filter](/assets/images/paper/mamba-ssm-biomarker/page_008.png)
*Figure 2: Raw gradient-saliency over the top-50 genes across TCGA-BRCA samples. The high cross-sample variance among non-oncogenic clusters is what motivates the symbolic filtering layer.*

![Agentic chain-of-thought trace](/assets/images/paper/mamba-ssm-biomarker/page_009.png)
*Figure 3: Per-gene `<think>` blocks map Mamba saliency scores to biological rationale and reject/keep decisions, enabling the post-hoc faithfulness audit.*

**Phase 4 — Faithfulness audit.** Each of the 50 input genes is labelled {validated / known non-BRCA / unknown} x {selected / not selected}, yielding selection-level precision, recall, and a missed-gene list.

**Downstream benchmark.** The *same* Mamba classifier (same architecture, same single random seed, same 80/20 stratified split — 1,231 samples) is retrained on each of B1 (5,000 variance genes), B2 (50 saliency genes), and B3 (17 LLM-filtered genes).

## Experimental Results

### Main downstream benchmark (Table 1)

| Method | Genes | Accuracy | F1 | AUC |
|---|---|---|---|---|
| B1 Variance baseline | 5,000 | 0.8785 | 0.8941 | 0.903 |
| B2 Mamba saliency only (no LLM) | 50 | 0.7247 | 0.7813 | 0.832 |
| **B3 Mamba + LLM structured CoT** | **17** | **0.8907** | **0.9033** | **0.927** |

![Per-metric comparison across B1/B2/B3](/assets/images/paper/mamba-ssm-biomarker/page_007.png)
*Figure 4: Accuracy / F1 / AUC across B1 (5,000-gene variance), B2 (50-gene saliency), and B3 (17-gene LLM-filtered). B3 is the only condition that exceeds the variance baseline on every metric.*

The relationship is non-monotonic in gene count: shrinking 5,000 -> 50 *hurts* AUC by 0.071, but shrinking further to a *reasoned* 17 *gains* 0.024 over B1. The improvement therefore cannot be attributed to dimensionality reduction per se — it depends on which 17 genes are kept.

### Faithfulness audit (Table 2)

| Metric | Value |
|---|---|
| Selected genes with validated BRCA evidence | 6 / 17 (35.3%) |
| Known non-BRCA genes incorrectly kept | 3 / 17 (17.6%) |
| Genes with no ground-truth label (unverifiable) | 8 / 17 (47.1%) |
| Known BRCA genes in top-50 input | 16 |
| Correctly kept by LLM | 6 |
| Missed by LLM (false negatives) | 10 |
| **Recall on validated input genes** | **0.375** |

**Correctly kept (6):** MLPH (luminal A), ZEB1 (EMT/TNBC), XBP1 (ER-stress / luminal), INPP4B (PI3K/AKT tumour suppressor), RHOB (PAM50 Rho-GTPase), THY1 (BRCA stem-cell marker).
**Incorrectly kept (3):** ITGAL (immune adhesion), LMX1B (kidney/neural TF), PRKAG2-AS1 (antisense RNA, no BRCA role).
**Critical false negative:** **FOXA1** — pioneer TF defining luminal lineage, present in both PAM50 and COSMIC — appeared at rank 49 of the top-50 but was rejected by the LLM. This is the paper's cleanest example of *confidently-wrong* biomedical reasoning.

The LLM also occasionally justified retentions with generic / hallucinated pathway language (e.g., invoking "NF-kB signaling" without breast-specific evidence). Structured CoT reduces but does not eliminate such failures.

## Limitations
Author-acknowledged:

- Single controlled setting (TCGA-BRCA only).
- Single stratified split with a fixed seed; no multi-seed variability.
- No comparison to LASSO / ElasticNet / mutual-information baselines.
- "Causal necessity" is used operationally, not as formal causal identification.
- A parsing bug in `llm_gene_reasoning.json` prevented decision-level (per-rationale) faithfulness analysis; results are limited to selection-level outcomes.

Not addressed in the paper:

- No external validation cohort (METABRIC, GTEx-vs-TCGA).
- No comparison against simpler attribution methods (Integrated Gradients, SHAP, DeepLIFT) for the saliency step.
- DeepSeek-R1 7B is a single LLM — no sensitivity to model choice, temperature, or prompt perturbation.
- No ablation isolating *which* of the 17 genes drives the AUC lift; the 8 unverifiable genes may be doing most of the work.
- A 1,231 vs. 1,095+113 = 1,208 sample-count discrepancy between sections is unexplained.
- "294x fewer features" in the text vs. "250x" in Fig. 2 caption — minor but unreconciled.
- No discussion of LLM contamination: DeepSeek-R1 has almost certainly seen TCGA papers and PAM50 in pre-training, so the "reasoning" partly reflects memorised gene lists, not independent inference.

## Why It Matters for Medical AI
For clinical biomarker discovery the bar is not "does my classifier hit a high AUC on the same cohort I trained on" — it is "does my gene panel generalise, and can I defend each gene to a pathologist." This paper is one of the first to **directly measure the gap** between those two questions with a real curated biomedical knowledge base, and to show that the gap is non-trivial: a 17-gene LLM-filtered panel can win on AUC while missing 10 of 16 validated genes available in its input pool and confidently rejecting a textbook luminal driver (FOXA1). For medical-AI practitioners this is a useful caution against treating CoT-style "reasoning traces" as evidence of biological understanding — the trace can be locally plausible, globally wrong, and still produce a metric improvement. The audit framework (selection-level precision/recall against COSMIC + OncoKB + PAM50) is portable to any downstream LLM-in-the-loop feature-selection pipeline and is the most reusable contribution of the paper.

## References
- Paper (arXiv): https://arxiv.org/abs/2604.14334 (v2, 17 Apr 2026, q-bio.QM)
- Venue: ICLR 2026 Workshop on Logical Reasoning of Large Language Models
- Code: https://github.com/pushpakumarbalan/feature-selection
- Background — Mamba SSM: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2024.
- Ground-truth sources: COSMIC Cancer Gene Census (Tier-1), OncoKB, PAM50 (Parker et al., 2009).
- Related — CoT faithfulness: Turpin et al., "Language Models Don't Always Say What They Think," NeurIPS 2023.
