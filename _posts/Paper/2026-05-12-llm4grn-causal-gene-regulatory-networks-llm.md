---
title: "LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs – Evaluation Through Synthetic Data Generation"
excerpt: "Prompting GPT-4 / Llama-3.1-70B for TF candidates and feeding them into GRNBoost2 + GRouNdGAN yields the lowest Cosine/Euclidean/MMD on PBMC-ALL — but the 'causal' framing is unsupported and the Llama win is an artifact of TF count."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/llm4grn-causal-gene-regulatory-networks-llm/
tags:
  - LLM4GRN
  - Gene Regulatory Networks
  - Single-cell RNA-seq
  - GRNBoost2
  - GRouNdGAN
  - GPT-4
  - Llama-3.1
  - Causal Discovery
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- LLM4GRN prompts **GPT-4 / Llama-3.1-70B** to propose transcription-factor (TF) candidates and/or full bipartite TF→target graphs, then evaluates the resulting GRN by feeding it to GRouNdGAN and comparing the synthesized scRNA-seq data to held-out real cells.
- On PBMC-ALL, the hybrid **KB_Llama + GRNBoost2** pipeline reaches **Cosine 0.00022 / Euclidean 83 / MMD 0.0067 / RF-AUROC 0.58**, approaching the real-vs-real Control (0.00029 / 100 / 0.0051 / 0.49) and beating the human-KB GRNBoost2 baseline.
- "Causal" in the title is **aspirational** — there are no interventions, no identifiability argument, no held-out causal benchmark; what is measured is a distributional match under a constrained generator. The Llama-beats-GPT-4 headline is driven by TF count (266 vs 95), and every method — LLM, GRNBoost2, hybrid, or random — collapses CD4+ Naïve T cells from **65.1% to ~4.5%**.

## Motivation

Statistical GRN inferers (GRNBoost2, GENIE3, NOTEARS-family) struggle with the noise and dimensionality of scRNA-seq and depend on curated TF databases (TRANSFAC, RegNetwork, ENCODE, BioGRID, AnimalTFDB) that are not cell-type or condition specific. Because real biological data has no ground-truth causal graph, the field cannot directly ask "did we recover the right edges?". The authors' workaround is a **downstream proxy task**: a GRN is "good" if scRNA-seq data simulated under it matches the marginal/centroid statistics of held-out real cells. The medical-AI pitch is twofold — better GRNs for disease-mechanism and drug-target discovery, plus a privacy framing where LLM inference does not need to touch patient observational data.

## Core Innovation

The pipeline factors GRN construction into two LLM-substitutable subproblems — **TF list curation** and **edge inference** — and crosses them in four Settings (Fig. 1):

- **Setting 1.A** — Human KB TFs (TRANSFAC/AnimalTFDB) + GPT-4 edges
- **Setting 1.B** — Human KB TFs + GRNBoost2 edges (baseline)
- **Setting 2.A** — LLM TFs + LLM edges (fully LLM)
- **Setting 2.B** — LLM TFs + GRNBoost2 edges (hybrid)

The bipartite graph $G = (T, R, E)$ — TFs $T$, target genes $R$, directed TF→target edges — is hard-wired into GRouNdGAN, a WGAN-GP-based causal GAN whose per-target generators each consume pre-trained TF expression plus noise. Crucially, GRouNdGAN forces **each gene to be regulated by exactly 10 TFs**, so the only thing that really varies across Settings is *which* 10 TFs are wired in.

The novel contribution is the **evaluation framework itself** — synthetic-data distributional matching as a proxy for unobservable ground-truth GRN quality, scored with Cosine / Euclidean / MMD / RF-AUROC, all lower-is-better.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | LLMs can discover **causal** gene regulatory networks. | Indirect: synthetic data under LLM graphs matches real-data statistics better than random graphs (Table 1); t-SNE clusters look sensible (Fig. 5). **No intervention, no identifiability, no causal benchmark.** | ⭐ |
| C2 | Out-of-the-box LLMs capture complex biological interactions. | Self-overlap 0.77 across seeds (vs random 0.12); post-hoc plausibility list of "extra" TFs (TNFRSF18, CDK9, BRD4). | ⭐⭐ |
| C3 | Synthetic-data downstream evaluation is a valid substitute for ground-truth GRN evaluation. | Framework is novel and reasonable but its validity is **assumed**, not tested. No correlation shown between better synthetic-data metrics and recovering a known causal graph. | ⭐ |
| C4 | Hybrid `KB_LLM + GRNBoost2` outperforms baselines. | Wins on PBMC-ALL and PBMC-CTL across Cosine/Euclidean/MMD (Tables 1–2). **Loses on BoneMarrow** (Setting 1 KBH-GRNBoost2 ties or beats KBGPT4-GRNBoost2 within noise). | ⭐⭐ |
| C5 | Llama-3.1-70B beats GPT-4 as a knowledge base. | KBLlama-GRNBoost2 (0.00022 / 83 / 0.0067 / 0.58) vs KBGPT4-GRNBoost2 (0.00023 / 83 / 0.0069 / 0.59) on PBMC-ALL only. **Table 5 ablation shows the win is a function of TF count (266 vs 95)**, not knowledge quality. | ⭐ |
| C6 | LLM-derived GRN is "more robust" than GRNBoost2. | Self-overlap 0.77 (LLM) vs 0.75 (GRNBoost2) — a 2-point gap with no significance test. | ⭐ |
| C7 | Method preserves privacy (no observational data). | True for Setting 2.A only — but every **winning** configuration uses GRNBoost2, which is observational. | ⭐⭐ |
| C8 | LLM produces biologically plausible synthetic data. | Authors themselves admit every generated dataset distorts cell-type proportions massively (CD4+ Naïve T: **65.1% → ~4.5%**; CD8+ Naïve Cytotoxic: 0.1% → ~30%). Plausibility is asserted via dot plots, not quantified. | ⭐ |

**Honest read.** The headline framing — "Discovering **Causal** Gene Regulatory Networks" — is not supported. There is no intervention, no controlled perturbation, no comparison against an experimentally validated causal benchmark, and no identifiability argument. What is measured is whether GRouNdGAN, seeded with an LLM-proposed bipartite graph, produces synthetic scRNA-seq whose marginal/centroid statistics match real data better than a random graph. That is a **correlational distributional match under a constrained generator**, not a causal claim. The strongest defensible result is C4 (hybrid wins on 2/3 datasets) — but BoneMarrow contradicts the trend, no metric has a significance test, and several "best" cells (e.g. PBMC-ALL Cosine 0.00022±0.00006 vs 0.00023±0.00008 vs 0.00024±0.00004) are statistically indistinguishable.

## Method & Architecture

![LLM4GRN four-setting pipeline overview](/assets/images/paper/llm4grn/page_004.png)
*Figure 1: Four LLM4GRN pipelines — Setting 1 uses a human-curated TF list (1.A LLM edges, 1.B GRNBoost2 edges); Setting 2 uses an LLM-derived TF list (2.A fully LLM, 2.B LLM TFs + GRNBoost2 edges). The resulting bipartite GRN is fed into GRouNdGAN, and synthetic scRNA-seq is scored against held-out real cells.*

**LLM prompting (Appendix A.2).** GPT-4 / Llama-3.1-70B are queried zero-shot with chain-of-thought and excerpts from the original Paul 2015 / Zheng 2017 dataset papers — a setup whose "zero-shot" framing is questionable since those papers are almost certainly in the training corpora.

1. **TF extraction.** Iterate over the 1,000-gene panel in windows of 20 genes (50% overlap), prompt: "which of these are TFs?"
2. **Edge inference.** For each target gene, prompt: "which 10 TFs out of LIST regulate this gene?" — output forced to a fixed `<Answer>[...]</Answer>` of exactly 10 TFs.

**Causal synthetic data — GRouNdGAN.** A two-stage WGAN-GP causal GAN: (i) a causal-controller WGAN on TF expression is pre-trained; (ii) per-target generators consume the pre-trained TF expressions + noise to produce target gene expression, with the bipartite GRN baked into generator-input topology.

**Evaluation.** Four lower-is-better statistical metrics (Cosine / Euclidean / MMD / RF-AUROC) plus Scanpy biological analyses (UMAP, marker dot plots, cell-type proportion bars). Numbers reported as mean ± std over 4 synthetic datasets × 2 CV seeds.

## Experimental Results

**Main quantitative comparison (Tables 1–2, lower is better).**

| Dataset | Setting | Method | Cosine ↓ | Euclidean ↓ | MMD ↓ | RF-AUROC ↓ |
|---|---|---|---|---|---|---|
| PBMC-ALL | Control (real vs real) | — | 0.00029±0.00008 | 100±16 | 0.0051±0.001 | 0.49±0.017 |
| PBMC-ALL | Stage 1 (non-causal) | WGAN-GP | 0.00036±0.00009 | 107±15 | 0.0057±0.005 | 0.55±0.021 |
| PBMC-ALL | Setting 1 KBH | GPT-4 (LLM edges) | 0.00024±0.00004 | 89±7 | 0.0072±0.001 | 0.63±0.043 |
| PBMC-ALL | Setting 1 KBH | GRNBoost2 | 0.00047±0.00021 | 121±25 | 0.0139±0.006 | 0.73±0.050 |
| PBMC-ALL | Setting 1 KBH | Random graph | 0.00045±0.00013 | 121±22 | 0.0166±0.003 | 0.85±0.019 |
| PBMC-ALL | Setting 2 KBGPT4 | GPT-4 (LLM edges) | 0.00026±0.00009 | 90±13 | 0.0206±0.001 | 0.86±0.018 |
| PBMC-ALL | Setting 2 KBGPT4 | GRNBoost2 | 0.00023±0.00008 | 83±17 | 0.0069±0.001 | 0.59±0.028 |
| PBMC-ALL | Setting 2 KBLlama | Llama-3.1 (LLM edges) | 0.00029±0.00005 | 97±10 | 0.0100±0.0007 | 0.77±0.030 |
| **PBMC-ALL** | **Setting 2 KBLlama** | **GRNBoost2 (hybrid)** | **0.00022±0.00006** | **83±11** | **0.0067±0.0011** | **0.58±0.023** |
| PBMC-CTL | Setting 1 KBH | GRNBoost2 | 0.00025±0.00004 | 65±6 | 0.0053±0.000 | 0.59±0.028 |
| **PBMC-CTL** | **Setting 2 KBGPT4** | **GRNBoost2 (hybrid)** | **0.00020±0.00004** | **57±6** | **0.0049±0.000** | 0.59±0.019 |
| **BoneMarrow** | **Setting 1 KBH** | **GRNBoost2 (baseline)** | **0.00190±0.00023** | **77±5** | 0.0118±0.001 | **0.64±0.023** |
| BoneMarrow | Setting 2 KBGPT4 | GRNBoost2 | 0.00193±0.00023 | 78±4 | 0.0119±0.001 | 0.64±0.033 |

The hybrid **wins on 2/3 datasets**. On BoneMarrow the human-KB baseline ties or beats the LLM-augmented variant within noise — quietly contradicting the headline framing.

![Marker dot plots: best-performing hybrid (KBLlama + GRNBoost2)](/assets/images/paper/llm4grn/page_010.png)
*Figure 2a: Dot plot of top marker-gene expression per cell type for KBLlama + GRNBoost2 — the best-performing hybrid. Red = overexpression, dot size = fraction of cells expressing.*

![Marker dot plots: KBGPT4 + GRNBoost2](/assets/images/paper/llm4grn/page_010.png)
*Figure 2b: Same plot for KBGPT4 + GRNBoost2 — visibly noisier than the Llama hybrid, with multiple markers expressed across non-specific cell types.*

![Marker dot plots: human-KB baseline](/assets/images/paper/llm4grn/page_010.png)
*Figure 2c: KBH + GRNBoost2 (human-KB baseline) — noise concentrated in Naïve T and cytotoxic T compartments.*

**Ablations that undercut the headline.**

- **TF-count ablation (Table 5).** When KBLlama is sub-sampled from 266 TFs down to 95 / 75, performance degrades monotonically (Euclidean 83 → 103 → 102; RF-AUROC 0.58 → 0.65 → 0.67). The "Llama wins" effect is therefore largely a function of how many TF candidates the model emits, not better edge-level biological reasoning.
- **GRN overlaps (Fig. 2 in paper).** GPT-4 self-overlap = 0.77; GRNBoost2 self-overlap = 0.75; GPT-4 vs. GRNBoost2 = **only 0.21**; GPT-4 vs. random = 0.13. The two "expert" methods disagree more with each other than either disagrees with itself.
- **t-SNE (Fig. 5).** Random graphs produce "hallucinated" extra clusters absent from real data; LLM and GRNBoost2 graphs do not. This is the **only** place a causal-vs-correlational signal is even argued for.

![Cell-type proportion distortion: KBGPT4 hybrid](/assets/images/paper/llm4grn/page_011.png)
*Figure 3a: Cell-type proportions in KBGPT4 + GRNBoost2 synthetic PBMC. Real CD4+ Naïve T = 65.1%, here ~4.5% — every generated dataset distorts the real composition.*

![Cell-type proportion distortion: KBLlama hybrid](/assets/images/paper/llm4grn/page_011.png)
*Figure 3b: Cell-type proportions for KBLlama + GRNBoost2 — same distortion pattern (CD8+ Naïve Cytotoxic dominates at 34.1%). The failure is method-agnostic.*

![t-SNE: random vs LLM vs GRNBoost2 synthetic data](/assets/images/paper/llm4grn/page_022.png)
*Figure 4: t-SNE projections (real = red, synthetic = blue). Random-graph synthetic data produces "hallucinated" clusters (red circles); LLM- and GRNBoost2-graphs stay closer to the real distribution.*

## Limitations

**Acknowledged by authors.**

- GRouNdGAN's constraints (bipartite-only, identical fan-in of 10 TFs/gene, no cofactors) **flatten differences between GRN methods** — by their own admission, this "may result in fairly similar performance metrics."
- Multi-layer GRNs (TFs + cofactors + targets) are more biologically realistic.
- LLMs hallucinate confidently and inherit population biases — mouse data, non-European ancestry, female and pediatric data are under-represented.
- Llama underperforms on edge inference (Setting 1) despite winning on KB curation.
- Cell-type-specific GRNs are needed; the current method ignores cell-type heterogeneity.

**Not addressed.**

- The gap between distributional match and causality — no controlled-intervention experiment.
- **Training-data leakage.** The PBMC / BoneMarrow papers and their TF lists are publicly indexed and almost certainly in GPT-4's pretraining corpus; "zero-shot" is misleading.
- No comparison with newer LLM-causal-discovery baselines (Vashishtha 2023, Sheth 2024, Kıcıman 2023) on the same datasets.
- No experiment with a synthetic GRN benchmark that has ground-truth edges (e.g. BEELINE / BoolODE).
- Llama-vs-GPT-4 head-to-head is run on a single dataset.
- The cell-type-proportion catastrophe (65.1% → 4.5%) is described but never scored in any headline metric.
- No edge-level precision/recall against any reference GRN database (RegNetwork, TRRUST, ENCODE ChIP-seq) — only TF-list overlap.
- Bias and reproducibility implications of relying on a commercial closed model (GPT-4) for biological discovery.

## Why It Matters for Medical AI

GRNs are the substrate for disease-mechanism reasoning and drug-target discovery in single-cell genomics, and clinical scRNA-seq cohorts will only grow. The useful piece of this paper is the demonstration that **LLM-proposed TF lists are a workable prior** for a statistical GRN inferer — i.e. the hybrid KB_LLM → GRNBoost2 → GRouNdGAN loop produces better synthetic data than human-KB baselines on PBMC. That has real value as a *data augmentation* recipe for downstream classifiers when patient-level scRNA-seq is scarce.

What practitioners should **not** take away: that LLMs have demonstrated causal-discovery competence on biology, or that LLM-derived GRNs can replace experimentally validated networks in clinical decision support. The cell-type-proportion collapse alone — universal across methods, never quantified in the headline — should disqualify these synthetic datasets from any downstream task that depends on cell-type frequency (and most clinical scRNA-seq tasks do). For medical AI, the framing matters: this is a useful prior, not a causal discovery engine.

## References

- **Paper:** Afonja, Sheth, Binkyte et al., "LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs – Evaluation Through Synthetic Data Generation," arXiv:2410.15828v1, Oct 2024.
- **GRouNdGAN:** Zinati et al., "GRouNdGAN: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks," Nature Communications, 2024.
- **GRNBoost2:** Moerman et al., "GRNBoost2 and Arboreto: efficient and scalable inference of gene regulatory networks," Bioinformatics, 2019.
- **Datasets:** Zheng et al. (10x PBMC, 2017); Paul et al. (mouse bone marrow, 2015).
- **TF knowledge bases:** TRANSFAC, AnimalTFDB, RegNetwork, ENCODE, BioGRID.
