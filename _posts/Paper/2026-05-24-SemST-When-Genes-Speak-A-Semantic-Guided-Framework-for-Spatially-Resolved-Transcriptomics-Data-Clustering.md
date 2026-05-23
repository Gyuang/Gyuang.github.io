---
title: "When Genes Speak: A Semantic-Guided Framework for Spatially Resolved Transcriptomics Data Clustering"
excerpt: "SemST prompts a frozen Qwen3-4B with each spot's top-k highly expressed gene symbols, then uses a FiLM-style affine modulation (FSM) of a multi-view GCN to top ARI on 8/8 ST benchmarks (+10.80 ARI on DLPFC 151508 vs stDCL)."
categories:
  - Paper
  - Spatial-Transcriptomics
  - LLM
permalink: /paper/semst/
tags:
  - SemST
  - Spatial-Transcriptomics
  - LLM
  - Qwen3
  - FiLM
  - GCN
  - ZINB
  - Clustering
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-24
last_modified_at: 2026-05-24
---

## TL;DR

- ST clustering models (STAGATE, GraphST, Spatial-MGCN, stDCL, ...) treat the per-spot expression vector as a bag of anonymous numbers and discard the biological meaning carried by gene **symbols** themselves. SemST formats each spot's top-`k_g` highly expressed gene symbols into a fixed prompt, feeds it to a **frozen Qwen3-4B**, and caches the final-layer hidden state H_llm as a "biological" per-spot embedding.
- A trainable MLP projects H_llm into a `2d` vector that is split into a scale `α` and a shift `β`; the multi-view (spatial + KNN-expression) GCN feature Z_gcn is modulated as `Z_final = ((1 + α) ⊙ Z_gcn + β) W`. This **Fine-grained Semantic Modulation (FSM)** is the only architectural novelty - it is FiLM (Perez et al., 2018, cited) with a shared MLP head and a `(1 + α)` parameterization.
- Trained with **ZINB reconstruction + cross-view correlation reduction + spatial contrastive loss**, SemST tops ARI on **8 of 8 ST datasets**: DLPFC 151508 **67.93** vs stDCL 57.13 (**+10.80**); MBA **53.92** vs Spatial-MGCN 48.34 (**+5.58**); HBC **68.64** vs Spatial-MGCN 65.68; ME **50.50** vs Spatial-MGCN 44.54; MVC **64.27** vs GAAEST 59.41.

## Motivation

Spatial transcriptomics platforms (10x Visium ~55 µm, Stereo-seq ~0.5 µm, STARmap ~2 µm) couple gene expression with spatial coordinates, and the canonical task of *spatial domain identification* is dominated by GNN-based clusterers - STAGATE, SpaGCN, GraphST, SEDR, Spatial-MGCN, STAIG, DUSTED, MAFN, stDCL. All of them ingest the expression matrix X ∈ R^{N×M} as a bag of unlabeled numerical features and throw away the semantics of the gene **symbols** themselves - that *Postn* implies fibroblast ECM remodeling, that *Ttn* + *Actc1* implies cardiac muscle, that *MOBP* + *MBP* implies cortical white matter.

Concurrent single-cell work (GenePT, scELMo, SGN, OmiCLIP) has shown that LLMs can convert gene symbols into useful biological embeddings, but those works target single-cell annotation or histology-omics alignment, not ST clustering, and they leave the fusion question open. SemST closes that gap with FiLM-style modulation and motivates the choice empirically: in its own ablation (Table 2), naive concat/add fusion of LLM and GCN features sometimes **hurts** clustering relative to no LLM at all.

Medical-AI relevance is direct: better spatial-domain identification underwrites differential expression, pathway enrichment, and tumor-microenvironment analysis on human breast cancer (HBC), DLPFC cortical layering, and mouse-embryo organogenesis - all of which are benchmarks here.

## Core Innovation

**Symbol-prompted LLM embeddings as a modulation signal, not a fusion partner.** Two design choices matter:

1. **Per-spot natural-language prompt over gene symbols.** For each spot i, take the top `k_g` highly expressed gene symbols (`k_g = 20` for DLPFC/HBC/MBA, `k_g = 30` for ME/MVC) and slot them into a fixed bilingual prompt:
   - *System:* "You are an expert in bioinformatics. Represent the biological state of a cell characterized by the following highly expressed genes. Focus on capturing the functional essence relevant for spatial domain identification."
   - *User:* "Highly expressed genes: {gene symbols set}."

   The prompt is fed to a **frozen Qwen3-4B** once, the final-layer hidden state is cached to disk, and training never updates the LLM. There is no LLM training cost beyond storage.

2. **FiLM-style affine modulation of GCN features by the LLM embedding.** A single MLP head produces both the scale `α` and shift `β` from the same H_llm vector. The paper argues that emitting α and β from a shared head keeps them anchored in the same bio-semantic space, then lets backprop decouple them. This is FiLM (Perez et al., 2018) with a `(1 + α)` reparameterization and a shared head - a real but small twist.

The pipeline composition - **per-spot symbol prompt + frozen Qwen3 + multi-view ZINB-GCN + FiLM fusion** - is where the contribution lives. None of the individual primitives is new.

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | SemST achieves SOTA spatial-domain clustering on ST benchmarks. | Table 1 - wins ARI on 8/8 datasets by 3-10 points; NMI 7/8, ACC 8/8, F1 7/8. | DLPFC (5 slices, 2 donors), HBC, MBA, ME, MVC | ⭐⭐ |
| C2 | LLM-derived semantic embeddings - not just the GCN/ZINB stack - drive the gain. | Table 2: removing LLM drops ARI by 6.5 (151509), 3.2 (151510), 18.3 (151671), 9.1 (MBA). | 4 datasets, single seed | ⭐⭐ |
| C3 | The semantic *content* matters - random / unrelated-prompt / BERT embeddings cannot substitute. | Random, Unrelated-prompt, and BERT all underperform full SemST and sometimes underperform "w/o LLM". | Same 4 datasets, single seed | ⭐⭐ |
| C4 | FSM is the right fusion mechanism (vs cross-attention / concat / add). | Table 2: SemST > Add ≥ Concat > Cross-Attn on most cells, but Add hits ARI 74.14 on 151671 vs SemST 78.70. | 4 datasets, single seed | ⭐ |
| C5 | FSM is plug-and-play - bolted onto 9 baselines it consistently improves them. | Table 3: 9 baselines × 2 datasets × 4 metrics, mostly positive deltas. | HBC, MBA only | ⭐⭐ |
| C6 | The shared-MLP-for-α-and-β design is principled. | Verbal argument only - no ablation against a two-MLP variant. | — | ⭐ |
| C7 | "Genes can speak" - the LLM injects high-order biological knowledge. | UMAP (Fig. 3): Qwen3 separates cortical layers, BERT collapses. Modulation distributions (Fig. 6) are non-trivial. | DLPFC qualitative + numbers | ⭐⭐ |
| C8 | The method generalizes across platforms (Visium, Stereo-seq, STARmap). | Table 1, single section per non-DLPFC platform. | 1 slice × 3 non-Visium platforms | ⭐ |

**Honest read.** The ARI gains in C1 are real-looking and the qualitative DLPFC clustering in Figure 2 visibly recovers cortical layers that GraphST/STAIG smear. But several caveats hold every claim under 3 stars:

- **Single seed everywhere.** Every cell of Tables 1, 2, 3, 5 is a single run at seed = 100. GraphST-style clusterers typically swing 2-4 ARI run-to-run, which is the same magnitude as several "wins" in Table 3. No error bars anywhere.
- **C4 is overstated.** On DLPFC 151671, simple Add fusion already reaches ARI 74.14 (SemST 78.70). On other slices Concat is within 5 ARI of full SemST. The "you need FSM" claim is weaker than the prose suggests, and the suspiciously low Cross-Attention number (62.40 on 151509) is never analyzed.
- **C3 has confounds.** "Random Emb." samples N(0, 0.85²), which does not match the actual statistics of Qwen3-4B final-layer hidden states; "Unrelated Emb." still feeds the same LLM, so it produces structured (but content-irrelevant) embeddings, not "no signal". The clean ablation - shuffling the spot→embedding assignment to break the gene-content link while preserving the embedding manifold - is missing.
- **C7 (biological knowledge) is the most fragile claim, and prior-leakage risk is substantial.** DLPFC layer markers (*MOBP*, *PCP4*, *KRT17*, *AQP4*, *NEFH*, *MBP*, *SNAP25*, ...), HBC subtype markers, and MBA region markers are heavily documented in PubMed, Gene Ontology, and the Allen Brain Atlas - all of which Qwen3-4B has memorized during pretraining. The LLM is plausibly **retrieving curated textbook knowledge about marker genes** rather than "inferring biology from co-occurrence patterns". No held-out evaluation on novel / underannotated transcripts (e.g., recently discovered lncRNAs without curated summaries) is attempted, which would be the proper test of biological reasoning vs lookup.
- **FSM is essentially FiLM (cited).** A shared MLP head and `(1 + α)` parameterization are the only architectural twists. That is a small contribution dressed up as a new module name.
- **Supplementary "Gene Summaries via Qwen3-Embedding-4B average pooling" matches the symbol-prompt pipeline.** If averaging per-gene NCBI-summary embeddings works as well as the per-spot prompt, the central narrative ("emergent gene-set reasoning at the LLM") is undercut - what FSM is really exploiting may be a per-spot *bag-of-gene-knowledge* vector.
- **Missing baselines from the ST clustering literature.** **SpaGCN**, **STAligner**, and **conST** are absent from Table 1; SpaGCN is mentioned only in related work. These are not exotic baselines.
- **Plug-and-play evaluation is narrow.** Only HBC and MBA receive the Table 3 FSM-as-plug-in evaluation. The other six datasets are never tested in this regime.

In short: the numbers are real but the framing oversells. The contribution is best described as "a careful pipeline showing that frozen-LLM gene-symbol embeddings, fed through FiLM-style modulation, beat current ST clusterers on standard benchmarks." The "genes speak" mechanistic story is weakly supported.

## Method & Architecture

![SemST framework overview](/assets/images/paper/semst/page_003.png)
*Figure 1 — SemST framework. Two graphs (spatial neighbors + KNN cosine-similarity over expression) feed a multi-view GCN whose output Z_gcn is element-wise modulated by α and β derived from a frozen LLM's embedding of the top-k_g highly expressed gene symbols at each spot. Training uses ZINB reconstruction + cross-view correlation reduction + spatial contrastive loss.*

The pipeline is best read as seven steps:

1. **Inputs.** Per slice: expression matrix X ∈ R^{N×M}, spatial coordinates S ∈ R^{N×2}, gene-symbol list G = {g_1, ..., g_M}. Preprocessing keeps the top g most-variable genes (`g = 128` for MVC, `g = 3000` elsewhere) and total-count-normalizes to 10⁴.
2. **Two graphs over spots.**
   - **Spatial graph A_spa:** edge (i, j) if ‖S_i − S_j‖₂ ≤ r (r = 560 for DLPFC, 15 for HBC/MBA/ME/MVC).
   - **Expression graph A_fea:** KNN with cosine similarity on X; `k_n = 14` for DLPFC/HBC/MBA, `15` for ME/MVC.
3. **Multi-view GCN backbone.** Standard symmetric-normalized GCN propagation $H^{(l+1)} = \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2} H^{(l)} W^{(l)})$ runs independently on A_spa and A_fea (separate parameters), both consuming X as node features. The two views concatenate to `Z_gcn = Concat(H_spa, H_fea) ∈ R^{N×d}`.
4. **Symbol → LLM embedding.** For each spot, the top `k_g` (20 or 30, ablated 5-50) highly expressed gene symbols are slotted into the fixed prompt above and passed through frozen **Qwen3-4B**; the **final-layer hidden state** is cached as H_llm ∈ R^{N×d′}. The paper claims the LLM is "flexible" but only Qwen3-4B is tested.
5. **Fine-grained Semantic Modulation (FSM).** A single trainable MLP `f_mlp` projects H_llm to `Z_mod ∈ R^{N × 2d}`; split along the feature dim into `[α | β]` with α, β ∈ R^{N×d}. The final representation is

   $$Z_{\text{final}} = \big((1 + \alpha) \odot Z_{\text{gcn}} + \beta\big)\, W$$

   where ⊙ is Hadamard product and W is a trainable mixing matrix. The same MLP head emitting both α and β is the only architectural twist over vanilla FiLM (Perez et al. 2018, cited).
6. **Self-supervised objective.** Three losses combine:
   - **ZINB reconstruction** on raw counts (decoder maps Z_final → (μ, θ, π) of a zero-inflated negative binomial). Same loss as SEDR/stDCL - not novel.
   - **Cross-view correlation reduction** `L_cr = (1/p²) Σ (C_ij − I_ij)²` on cosine similarities between H_spa and H_fea (p = d/2), borrowed from MAFN (Zhu et al. 2024).
   - **Spatial regularization** `L_s`, a BPR-style contrastive loss pulling spatial neighbors together in Z_final.

   Total: `L = L_zinb + γ L_cr + λ L_s`; (γ, λ) = (0.1, 0.1) for DLPFC/HBC/MBA, (1, 1) for ME/MVC.
7. **Inference.** K-means on Z_final to recover spatial domains, with cluster count taken from dataset annotation (5-7 DLPFC, 20 HBC, 52 MBA, 12 ME, 7 MVC). Adam, lr = 1e-3, weight decay 5e-4, **single seed = 100**, single RTX 3090.

## Experimental Results

### Datasets

| Platform | Tissue | Slice | Spots | Genes (raw) | Clusters |
|---|---|---|---|---|---|
| 10x Visium | DLPFC | 151508 | 4383 | 33538 | 7 |
| 10x Visium | DLPFC | 151509 | 4789 | 33538 | 7 |
| 10x Visium | DLPFC | 151510 | 4634 | 33538 | 7 |
| 10x Visium | DLPFC | 151671 | 4110 | 33538 | 5 |
| 10x Visium | DLPFC | 151672 | 3888 | 33538 | 5 |
| 10x Visium | Human breast cancer (HBC) | Section 1 | 3798 | 36601 | 20 |
| 10x Visium | Mouse brain anterior (MBA) | Section 1 | 2695 | 32285 | 52 |
| Stereo-seq | Mouse embryo (ME) | E9.5-E1S1 | 5913 | 25568 | 12 |
| STARmap | Mouse visual cortex (MVC) | X | 1207 | 1020 | 7 |

Non-DLPFC datasets are evaluated on a **single slice each**; the five DLPFC slices come from two donors. STARmap MVC only has 1020 genes total, of which the top-30 are prompted to the LLM - a fundamentally different signal regime from Visium.

### Main quantitative comparison (Table 1, ARI × 100)

![Table 1 — quantitative comparison on 8 ST datasets](/assets/images/paper/semst/page_006.png)
*Table 1 — ARI / NMI / ACC / F1 across DLPFC, HBC, MBA, ME, MVC. SemST wins ARI on all 8 datasets.*

| Method | DLPFC 151508 | DLPFC 151509 | DLPFC 151510 | DLPFC 151671 | ME | MVC | HBC | MBA |
|---|---|---|---|---|---|---|---|---|
| STAGATE (Nat Comm '22) | 53.84 | 49.54 | 46.07 | 59.54 | 32.34 | 52.38 | 44.66 | 35.81 |
| GraphST (Nat Comm '23) | 48.61 | 51.93 | 51.29 | 61.02 | 29.75 | 36.16 | 52.63 | 41.32 |
| Spatial-MGCN (BIB '23) | 46.22 | 54.22 | 51.61 | 60.19 | 44.54 | 53.32 | 65.68 | 48.34 |
| GAAEST (Comm Biol '24) | 31.24 | 43.84 | 39.79 | 64.80 | 26.27 | 59.41 | 52.02 | 43.35 |
| SEDR (Genome Med '24) | 47.47 | 49.69 | 50.52 | 59.97 | 27.70 | 52.71 | 43.16 | 40.36 |
| MAFN (TKDE '24) | 54.54 | 70.48 | 69.91 | 72.41 | 38.48 | 59.12 | 59.20 | 44.15 |
| stDCL (Adv Sci '25) | 57.13 | 41.63 | 57.91 | 66.74 | 34.32 | 49.54 | 55.73 | 42.05 |
| DUSTED (AAAI '25) | 46.53 | 51.10 | 43.08 | 60.31 | 26.23 | 58.06 | 47.81 | 35.86 |
| STAIG (Nat Comm '25) | 50.36 | 59.45 | 52.88 | 45.17 | 27.20 | 58.26 | 57.86 | 33.35 |
| **SemST** | **67.93** | **73.90** | **73.92** | **78.70** | **50.50** | **64.27** | **68.64** | **53.92** |
| Δ vs best baseline | **+10.80** | **+3.42** | **+4.01** | **+6.29** | **+5.96** | **+4.86** | **+2.96** | **+5.58** |

NMI matches the same pattern on 7/8 datasets (loses MBA 70.12 vs STAGATE 72.49). ACC: 8/8. F1: 7/8 (Spatial-MGCN ties on MVC F1 within 0.49). Notably, **SpaGCN, STAligner, and conST are missing from the comparison.**

### Qualitative spatial-domain recovery

![Qualitative DLPFC, MBA, HBC clustering](/assets/images/paper/semst/page_005.png)
*Figure 2 — DLPFC slice 151672, MBA, and HBC clustering. SemST (rightmost) recovers cortical layers and HBC tumor regions much closer to manual annotation (leftmost) than GraphST / Spatial-MGCN / STAIG.*

### Ablation (Table 2, ARI × 100)

| Variant | 151509 | 151510 | 151671 | MBA |
|---|---|---|---|---|
| w/o LLM | 67.37 | 70.74 | 60.43 | 44.83 |
| Random Emb. (zero-mean, var ≈ 0.85) | 59.95 | 63.16 | 50.56 | 41.72 |
| Unrelated Emb. (LLM on non-bio prompt) | 65.24 | 65.96 | 60.11 | 43.89 |
| BERT (generic encoder) | 63.27 | 69.25 | 60.05 | 42.96 |
| Cross-Attention fusion | 62.40 | 66.07 | 63.14 | 47.69 |
| Concat fusion | 69.25 | 67.43 | 73.21 | 46.64 |
| Add fusion | 69.19 | 71.96 | 74.14 | 47.35 |
| **SemST (LLM + FSM)** | **73.90** | **73.92** | **78.70** | **53.92** |

Three findings worth flagging:

- **Random / Unrelated / BERT embeddings sometimes underperform `w/o LLM`.** Injecting *bad* embeddings is worse than no embeddings at all. The paper frames this as "semantics matter"; it can also be read as calibration brittleness in the FSM head.
- **Concat and Add are close.** On DLPFC 151671, Add reaches ARI 74.14 (SemST 78.70 - +4.56); Concat 73.21 (+5.49). The "you need FSM" claim is much weaker than the prose suggests on those slices.
- **The shared-MLP-for-α-and-β design (C6) is asserted, never ablated.** No two-MLP variant, no `α` (without `1 +`) variant.

### Plug-and-play FSM (Table 3) + LLM-vs-BERT UMAP + k_g sweep

![Plug-and-play FSM, Qwen3-vs-BERT UMAP, k_g sensitivity](/assets/images/paper/semst/page_007.png)
*Figure 3 + Table 3 + Figure 4 — Top: Table 3 bolts FSM onto 9 baselines on HBC and MBA; large gains on weak baselines (STAGATE on HBC: 44.66 → 58.16, +13.50 ARI), small or negative deltas on strong ones (Spatial-MGCN HBC NMI: 70.83 → 69.44). Bottom-left: Qwen3 embeddings on DLPFC 151508 form smooth cortical-layer gradients; BERT embeddings of the same prompts collapse to one blob. Bottom-right: k_g sweep peaks at 20 (MBA) or 30 (ME, MVC) and declines on both sides.*

The plug-and-play story (C5) is the **only one tested on more than four datasets** - and it's tested on just two (HBC, MBA). Gains are concentrated on weak baselines lifted closer to the LLM-informed manifold; on stronger baselines like Spatial-MGCN the deltas are small and occasionally negative. This is consistent with FSM acting as a regularizer toward the LLM prior, rather than as a universal upgrade.

### Supplementary qualitative + FSM behavior

![Supplementary qualitative ME/MVC, modulation distributions, before/after FSM](/assets/images/paper/semst/page_011.png)
*Figure 5 + Figure 6 + Figure 7 — Top: ME and MVC qualitative clustering. Middle: distributions of α and β across DLPFC / MBA / ME / MVC (non-zero means with substantial spread - FSM is not learning identity). Bottom: before-vs-after FSM modulation maps showing how the per-spot biological signal reshapes the GCN feature space.*

## Limitations

**Authors acknowledge (mostly indirectly):**
- LLM choice is "flexible" but only Qwen3-4B is reported; no GPT-4 / Llama / Med-PaLM / biomedical-LLM (BioMedLM, Meditron) comparison.
- `k_g` must be tuned per dataset (sweep 5-50, pick the sweet spot).
- Naive concat/add of LLM embeddings into baselines can hurt (Table 5, supplement) - they use this as motivation for FSM.

**Authors do not address:**
- **Prior leakage / contamination.** No test on genes / markers Qwen3 cannot have seen. The clean counterfactual - rename gene symbols to anonymized tokens with the same NCBI summaries - would isolate "symbol-as-key" lookup from "biology" semantics. Not run.
- **No variance.** Single seed = 100 throughout; no confidence intervals; differences of 1-3 ARI are reported as wins.
- **No comparison to GenePT / scELMo embeddings.** Both are cited but never used as a drop-in replacement for H_llm - that ablation would isolate whether *Qwen3 hidden states* or *the per-spot prompt design* is doing the work.
- **FSM design ablation incomplete.** Shared MLP vs two MLPs is argued for, not measured. `(1 + α)` vs `α` is not ablated. No ablation of FSM stacked across deeper layers.
- **Missing baselines.** **SpaGCN, STAligner, conST** are absent from Table 1; SpaGCN appears in related work only.
- **Plug-and-play evaluation is narrow.** Only HBC and MBA in Table 3; six other datasets are never tested in this regime.
- **Limited cancer evaluation.** One HBC slice; no Visium-HD, no 10x Xenium, no NSCLC / colorectal benchmarks where most clinical interest sits.
- **No downstream biology.** Clustering against manual labels is the only evaluation. No differential expression analysis, no pathway enrichment showing that semantic-guided clusters recover biologically distinct programs.
- **Compute / latency.** Qwen3-4B inference cost per slice is not reported; embeddings are precomputed once but for new tissues this is the practical bottleneck.
- **Supplementary "Gene Summaries via Qwen3-Embedding-4B average pooling"** matches the symbol-prompt pipeline closely - undercutting the "emergent gene-set reasoning" narrative.

## Why It Matters for Medical AI

Spatial domain identification underwrites downstream clinical analyses - differential expression, pathway enrichment, tumor-microenvironment characterization. A reliable +5-10 ARI on DLPFC cortical layering and HBC tumor regions, if it holds up under multi-seed evaluation and on novel markers, would be a meaningful operational upgrade for ST analysis pipelines. The bigger conceptual point - that ST methods should stop discarding gene **symbols** - is the right one, even if SemST under-tests its specific instantiation. The remaining open question for clinical deployment is whether the LLM's gain survives once the gene panel moves away from textbook markers that the LLM memorized during pretraining.

## References

- Paper: *When Genes Speak: A Semantic-Guided Framework for Spatially Resolved Transcriptomics Data Clustering*, Long et al., **AAAI 2026** (arXiv 2511.11380v1, Nov 2025).
- Code: [https://github.com/longjiangk/SemST](https://github.com/longjiangk/SemST)
- FiLM (the FSM antecedent): Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- Qwen3-4B: Yang et al., 2025a.
- Related ST clusterers: STAGATE (Nat Comm 2022), GraphST (Nat Comm 2023), Spatial-MGCN (BIB 2023), SEDR (Genome Med 2024), MAFN (TKDE 2024), stDCL (Adv Sci 2025), DUSTED (AAAI 2025), STAIG (Nat Comm 2025), GAAEST (Comm Biol 2024).
- Related LLM-for-genes work: GenePT, scELMo, SGN, OmiCLIP, Cell2Sentence.
- ZINB reconstruction: Yu et al., AAAI 2022.
