---
title: "Large-scale Foundation Model on Single-Cell Transcriptomics (scFoundation / xTrimoscFoundationα)"
excerpt: "A 100M-parameter asymmetric encoder-decoder pretrained on >50M human cells with Read-Depth Aware modeling; halves read-depth recovery error and lifts drug-blind IC50 PCC from 0.07 to 0.73 on PHA-793887."
categories: [Paper, BioInformatics, LLM]
permalink: /paper/scfoundation/
tags:
  - scFoundation
  - xTrimoGene
  - Single-Cell Transcriptomics
  - Foundation Model
  - Read-Depth Aware
  - Masked Autoencoder
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR
- **100M-parameter xTrimoGene** transformer pretrained on **>50M human single cells** spanning ~19,264 protein-coding genes — an asymmetric encoder-decoder where the encoder sees only the ~10% non-zero genes, sidestepping the full-transcriptome attention budget without resorting to kernel approximation.
- **Read-Depth Aware (RDA) objective**: two scalar tokens `T` (target counts) and `S` (source counts) condition a Bayesian-downsampled masked-prediction task. At inference, setting `T > S` performs **read-depth enhancement with no fine-tuning** — roughly **halving MAE/MRE** at ≤10% downsampling.
- Plugged frozen into existing pipelines, it lifts **drug-blind PCC on PHA-793887 from 0.07 → 0.73** (DeepCDR), raises **single-cell drug-sensitivity AUC by +0.22 / +0.28** on NVP-TAE684 / sorafenib (SCAD), and lowers top-20 DE MSE on the GEARS perturbation benchmark across 0/2, 1/2, 2/2 unseen splits.

## Motivation
Existing single-cell transformers (scBERT, scGPT, Geneformer) hit three structural walls that prevent scaling:

1. **No comprehensive pretraining corpus.** Prior work uses subsets of public scRNA-seq, often a few million cells from narrow tissue scopes.
2. **~20k-gene "sentences" overflow vanilla attention.** scBERT relies on Performer; scGPT subsamples genes. Both compromise either expressiveness or full-transcriptome coverage.
3. **Read depth varies by orders of magnitude across protocols.** Unlike random technical noise, this variation is systematic — naively scaling masked LM does not yield depth-invariant representations.

scFoundation targets a model that ingests **all** ~19k genes, treats read depth as a first-class signal during pretraining, and serves downstream tasks (drug response, perturbation, cell-type annotation) mostly without per-task fine-tuning. The strongest downstream demos are squarely medical-AI: tumor IC50 prediction on CCLE/GDSC and single-cell drug-resistance inference in patient-derived models.

## Core Innovation
- **Read-Depth Aware (RDA) pretraining.** Total counts `T` (raw) and `S` (downsampled) become scalar tokens prepended to the gene sequence. The model reconstructs masked positions of the downsampled input back to the raw expression vector — learning depth harmonization as part of the pretraining objective rather than as a downstream post-hoc fix.
- **Asymmetric encoder-decoder (xTrimoGene), MAE-style.** The encoder uses full vanilla self-attention but only over non-zero, non-masked gene positions (~10% of the full vector). The decoder concatenates encoder outputs with learned zero/mask embeddings to restore full length 19,266, and uses Performer kernel attention since full attention at that length is infeasible. This is the architectural lever that lets the model handle full-transcriptome inputs at 100M parameters.
- **Continuous (not bucketed) value embeddings.** Each scalar expression is mapped to a learned weighted summary over base embeddings — no binning. Ablation (Supp. Fig. 14) reports this beats discretization variants.
- **Frozen-embedding downstream wiring.** Drop-in replacement for the transcriptome subnetwork of DeepCDR (bulk drug response), SCAD (single-cell drug classification), and per-cell node features for GEARS (perturbation). Only cell-type annotation actually fine-tunes (one encoder layer + a 2-layer MLP head).

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|-------|----------|----------|
| C1 | 100M-param transformer on 50M cells follows an LLM-style scaling law on scRNA-seq | Fig. 2a power-law fit `y = -0.014·log10(FLOPs) + 0.611` across 3M/10M/100M | ⭐⭐ — three points, validation MSE only; "vs scBERT/scGPT/Geneformer" markers are **xTrimoGene runs at equivalent parameter counts, not those models retrained** |
| C2 | Asymmetric MAE-style architecture is more efficient than symmetric alternatives | Supp. Table 6 / Supp. Note 1 ablations | ⭐⭐ — relegated to supplement |
| C3 | RDA + continuous-value embedding beats discretization + standard MLM | Supp. Fig. 14, Supp. Fig. 15, Supp. Table 8 | ⭐⭐ |
| C4 | Read-depth enhancement without fine-tuning halves MAE/MRE at ≤10% sampling | Fig. 2b, Supp. Fig. 1 | ⭐⭐ — single internal split, no variance bars, downsampling is binomial (not real low-depth sequencing) |
| C5 | Beats MAGIC/SAVER/scImpute/scVI for clustering after enhancement | Fig. 2c–d | ⭐ — **one dataset (Baron)**; scFoundation **loses to SAVER at fold=1**; win requires T/S > ~2 |
| C6 | Zero-shot scFoundation matches scVI on Zheng68K and improves SIL | Fig. 2e–f | ⭐⭐ — single dataset; NMI/ARI gains modest |
| C7 | Improves cancer drug IC50 prediction with DeepCDR | Fig. 3b–d, leave-one-drug-out blind test on 223 drugs | ⭐⭐⭐ — large drug catalog, drug-blind protocol, consistent gain across cancer types |
| C8 | Improvements concentrate on chemotherapy vs targeted therapy | Fig. 3d grouped means | ⭐⭐ — biologically interpretable, no statistical test reported |
| C9 | Single-cell drug-resistance inference via SCAD, +0.2 AUC on hard drugs | Fig. 4b ROC curves | ⭐⭐ — large lift, but some SCAD baselines are below 0.5; single run, no CIs |
| C10 | Embeddings carry drug-sensitivity signal beyond raw expression (EpiSen) | Fig. 4c Spearman | ⭐⭐ |
| C11 | scFoundation-conditioned GEARS beats GEARS and CPA, especially for unseen-gene combos | Fig. 5b–d on Dixit, Adamson, Norman across 0/2, 1/2, 2/2 splits | ⭐⭐⭐ — three datasets, three split regimes, two baselines |
| C12 | Predicted magnitude score classifies 2-gene synergy/suppressor more accurately | Fig. 5e PCC 0.18 vs 0.01 | ⭐ — **0.18 is the winning PCC** — "better than near-zero" is a weak absolute claim |
| C13 | Cell-type annotation outperforms 6 baselines (incl. scBERT, CellTypist) | Supp. Table 4 macro-F1 | ⭐⭐ — only Zheng68K + Segerstolpe; gains concentrated in rare classes |
| C14 | Gene embeddings recover known cell-type TFs (KLF6, SPIB, MXD4) | Supp. Figs. 9–13 via SCENIC | ⭐ — qualitative hand-picked examples; no quantitative GRN benchmark |
| C15 | Foundation-model claim: serves diverse downstream tasks without fine-tuning | §Results 3–7 across tasks | ⭐⭐ — overstated: **cell-type annotation does fine-tune** the last encoder layer + MLP head |
| C16 | Outperforms scBERT / scGPT / Geneformer at comparable scales | Fig. 2a only | ⭐ — **no head-to-head on shared downstream benchmarks in main text**; competitor "comparison" is by parameter-equivalent xTrimoGene runs on pretraining loss |

**Honest read.** Two structurally strong claims (C7, C11): multi-dataset, externally defined baselines, large gains. The scaling-law plot (C1) is methodologically novel for the field but a 3-point fit, and the "vs scBERT/scGPT/Geneformer" dots are not those models — they are xTrimoGene at matched parameter counts. The "no fine-tuning needed" framing (C15) is partly accurate (read-depth enhancement, DeepCDR, SCAD, GEARS are frozen) but the annotation experiment does train an MLP head plus the last encoder layer. The cross-foundation-model comparison readers want (vs scGPT / Geneformer / UCE on shared downstream tasks) is not present in the main text — Supp. Table 1 compares features (parameter count, gene coverage), not benchmarks. Across the paper, almost no experiment reports error bars, run-to-run variance, or statistical tests; the cell-type annotation "average of top-3 replicates" is a biased estimator. The 2-gene synergy PCC of 0.18 (C12) wins over 0.01 but is itself a poor absolute predictor.

## Method & Architecture

![scFoundation pretraining overview](/assets/images/paper/scfoundation/page_003.png)
*Figure 1: scFoundation pretraining overview — 50M cells across 100+ tissues feed an RDA masked-prediction task where T and S indicator tokens encode raw and downsampled total counts; the asymmetric encoder consumes only nonzero genes, while the Performer-based decoder reconstructs all 19,264 genes.*

### Step-by-step
1. **Data unification.** Aggregate scRNA-seq from GEO, Single Cell Portal, HCA, hECA, DISCO, EMBL-EBI Expression Atlas. Reverse-engineer raw counts from normalized matrices by treating the smallest nonzero value as count=1; TPM/FPKM matrices that cannot be inverted are kept as-is. Align gene symbols to HUGO, zero-pad missing genes for fixed length n = 19,264. QC: keep cells with >200 expressed genes. Final corpus: >50M cells, 100+ tissues, normal + tumor; 100k random held-out cells for validation.
2. **Embedding module.** For each gene i, the scalar expression $x_i \ge 0$ is mapped via a **learned weighted summary** over base embeddings (no discretization). The value embedding is summed with a learnable gene-name lookup embedding $T_G^i$.
3. **Encoder.** Sees only nonzero, non-masked positions plus the two count indicator tokens — typically ~10% of full gene length. Vanilla full self-attention `Att(Q,K,V) = softmax(QK^T/√d) V`. Outputs intermediate embeddings $X_\text{inter} \in \mathbb{R}^{K \times d}$ that double as cell embeddings after pooling.
4. **Decoder.** Concatenates $X_\text{inter}$ with learned **zero embeddings** and **mask embeddings** to restore full length 19,266 (genes + T + S). Uses **Performer** kernel attention; output projected by a shared MLP to scalar predictions for all 19,264 genes.
5. **RDA pretraining loop** per cell:
   - Total raw count → `T`.
   - Hierarchical Bayesian downsampling → variant with total count `S` (sometimes T=S so the model also learns within-cell gene relations).
   - Log-normalize both vectors.
   - Mask 30% of input positions (zero and nonzero alike).
   - Forward masked input + [T, S] tokens through encoder → decoder.
   - Regression loss between predicted and raw expression at masked positions only.
6. **Cell embedding for downstream.** Concatenate **max-pool + mean-pool of all-gene encoder outputs + S-token embedding + T-token embedding** → 3,072-dim (4 × 768).
7. **Downstream wiring.** DeepCDR: replace its transcriptome MLP with the frozen embedding (bulk: set S = T = sum of expression). SCAD: unify bulk + single-cell embeddings, set T=10,000 for single cells. GEARS: per-cell gene context embeddings as co-expression graph node features (frozen scFoundation, only GEARS trained). Annotation: fine-tune one encoder layer + a 2-layer MLP head with weighted cross-entropy.

## Experimental Results

![Scaling and read-depth recovery](/assets/images/paper/scfoundation/page_004.png)
*Figure 2: Scaling behavior and read-depth recovery — validation MSE follows a clean power law in FLOPs (2a); MRE roughly halves vs the downsampled baseline at ≤10% sampling (2b); on Baron islets, NMI/ARI/SIL surpass all imputation baselines once T/S exceeds ~2 (2c–d); on Zheng68K, zero-shot embeddings match scVI on NMI/ARI and exceed it on SIL (2e–f).*

### Main comparison table

| Task | Dataset | Metric | **scFoundation** | Best baseline | Notes |
|------|---------|--------|------------------|---------------|-------|
| Pretraining scaling | 100k-cell valid | MSE | follows `-0.014·log10(FLOPs)+0.611` | scVI 0.98 (not transformer) | "vs scBERT/scGPT/Geneformer" markers = xTrimoGene at matched params |
| Read-depth enhancement | 10k held-out, 1–20% downsample | nonzero MRE / PCC | ≈ halves MRE vs downsample baseline at ≤10% | downsampled raw | "Notable reduction of half the MAE and MRE" (verbatim) |
| Imputation clustering | Baron islets | NMI/ARI/SIL | best at fold ≥ ~2 | SAVER best at fold=1 only | scFoundation **loses to SAVER at T=S** |
| PBMC clustering (zero-shot) | Zheng68K | NMI/ARI/SIL | SIL highest; NMI/ARI ≈ scVI | scVI trained on-data | scFoundation frozen |
| Bulk drug IC50 (drug-blind) | GDSC PHA-793887 | PCC | **0.73** | DeepCDR gene-expr: 0.07 | +0.66 absolute |
| Bulk drug IC50 (drug-blind) | GDSC zibotentan | PCC | **0.64** | baseline 0.49 | +0.15 |
| Bulk drug IC50 (best case) | LGG × WZ-1-84 | PCC / Spearman | **0.94 / 0.95** | — | N=8, tiny |
| Single-cell drug AUC | SCAD NVP-TAE684 | AUC | **0.84** | SCAD 0.62 | +0.22 |
| Single-cell drug AUC | SCAD Sorafenib | AUC | **0.84** | SCAD 0.56 | +0.28 |
| Single-cell drug AUC | SCAD PLX4720 | AUC | **0.66** | SCAD 0.38 | baseline worse than random |
| Single-cell drug AUC | SCAD Etoposide | AUC | **0.68** | SCAD 0.66 | marginal |
| EpiSen ↔ sensitivity | NVP-TAE684 | Spearman | **0.56** | 0.24 |  |
| EpiSen ↔ sensitivity | Sorafenib | Spearman | **−0.55** | −0.06 |  |
| Perturbation top-20 DE MSE | Dixit / Norman / Adamson | MSE | **lower than GEARS across 1-gene + 0/2, 1/2, 2/2 unseen** | GEARS, CPA | numeric values only in Fig. 5b |
| 2-gene mag-score | Norman | PCC | **0.18** | 0.01 | both low in absolute terms |
| Cell-type annotation | Zheng68K + Segerstolpe | macro-F1 | **highest of 7 methods** | CellTypist 2nd | gains concentrated in rare types per Supp. Table 5 |

### Medical-AI showcase — drug response

![Drug response](/assets/images/paper/scfoundation/page_005.png)
*Figure 3: Drug response — replacing DeepCDR's transcriptome MLP with scFoundation cell-line embeddings raises PCC across drugs and cancer types (3b); leave-one-drug-out blind test gains of up to +0.66 (PHA-793887) (3d); GSEA on low-IC50 predictions recovers known sphingolipid-doxorubicin and mTOR-vorinostat links (3e).*

![Single-cell drug sensitivity](/assets/images/paper/scfoundation/page_006.png)
*Figure 4: Single-cell drug-sensitivity transfer — scFoundation embeddings yield +0.2 AUC over SCAD on NVP-TAE684 and sorafenib (4b), with Spearman correlations to EpiSen scores of 0.56 / −0.55 vs 0.24 / −0.06 baseline (4c).*

### Mechanistic showcase — perturbation

![Perturbation prediction](/assets/images/paper/scfoundation/page_007.png)
*Figure 5: Perturbation prediction — using per-cell scFoundation gene embeddings as GEARS co-expression nodes reduces top-20 DE MSE across 1-gene and 0/2, 1/2, 2/2 unseen 2-gene splits (5b), and lifts magnitude-score PCC from 0.01 to 0.18 (5e).*

### Ablations worth flagging
- **Scaling law (Fig. 2a)** is the cleanest single-cell scaling-law plot to date — but it is a 3-point fit, and the markers labeled scBERT/scGPT/Geneformer-equivalent are xTrimoGene runs at those parameter counts, **not independent reproductions** of those models. Easy to misread.
- **T/S sensitivity.** Imputation plateaus once T > 3.5·S (Fig. 2c); below that, scFoundation can be **worse than SAVER at T=S**. Honestly reported.
- **Continuous vs discretized embeddings** (Supp. Fig. 14) supports the continuous scheme.
- **GSEA sanity check** on low-IC50 predictions surfaces sphingolipid signaling for doxorubicin and mTOR for vorinostat — biologically plausible.
- **GRN inference** (Supp. Figs. 9–13) identifies KLF6 (monocyte), SPIB (B cell), MXD4 (CD8 T cell) via SCENIC. Authors themselves call this "simplistic."

## Limitations

**Acknowledged by authors**
- Pretraining corpus, despite being "virtually all publicly available human scRNA-seq," is still incomplete relative to full human developmental and disease space.
- Heavy compute requirement; further optimization needed.
- Transcriptome-only; no multi-omics (genomic, epigenomic).
- Unsupervised pretraining ignores rich metadata (donor, tissue, condition).
- Pretraining loss does not converge to zero — authors recommend using embeddings rather than predicted expression values for downstream use.
- Their SCENIC-style GRN inference is "a simplistic approach."

**Not addressed in the paper (audit)**
- **No head-to-head with scGPT, Geneformer, or UCE on shared downstream tasks.** Supp. Table 1 compares features (parameter count, gene coverage, data size), not benchmarks. For four models all claiming foundation-model status, this is the comparison readers most want and the paper avoids it.
- **The "vs scBERT/scGPT/Geneformer" scaling-law markers (Fig. 2a) are not those models — they are xTrimoGene runs at equivalent parameter counts.** The plot does not establish architectural superiority.
- **"No fine-tuning needed" framing is overstated.** Read-depth enhancement, DeepCDR, SCAD, GEARS use frozen embeddings, but cell-type annotation trains an MLP head **plus the last encoder layer**.
- **Most tables lack variance / confidence intervals.** PCC, MSE, and AUC are reported as single numbers; cell-type annotation reports the biased "average of top-3 replicates" estimator.
- **The 2-gene magnitude PCC of 0.18 (Fig. 5e) wins but is absolutely poor** — it improves over near-zero baseline but should not be cited as accurate synergy prediction.
- No demographic / ancestry breakdown of the 50M cells.
- Downsampling for the read-depth experiment is binomial — does not simulate actual low-depth sequencing artifacts (amplification bias, chemistry-specific dropout).
- Annotation benchmark uses only Zheng68K + Segerstolpe (both old, both also used by scBERT); no atlas-scale evaluation (HLCA, Tabula Sapiens).
- The dramatic PHA-793887 lift (0.07 → 0.73) is not explained — is the baseline degenerate for that drug?
- Performer in the decoder is a kernel approximation; not ablated against full-attention decoders at smaller scales.
- Multimodal extension (predicting expression from ATAC-seq context) teased in Supp. Note 7 but no result shown.

## Why It Matters for Medical AI
The two strongest downstream demonstrations are squarely medical: **bulk tumor drug-response prediction** on CCLE × GDSC (223 drugs, drug-blind protocol) and **single-cell drug-resistance inference** on patient-derived models via SCAD. Drop-in replacement of an existing pipeline's transcriptome subnetwork lifts drug-blind IC50 PCC from 0.07 to 0.73 on PHA-793887 (the headline number) and single-cell sensitivity AUC by 0.22–0.28 on NVP-TAE684 and sorafenib — gains large enough to matter clinically if they hold up out-of-sample. The architectural lesson — that read-depth, the dominant systematic confound in scRNA-seq across protocols and labs, can be absorbed into the pretraining objective rather than handled downstream — generalizes to any clinical setting where samples are sequenced at heterogeneous depths (longitudinal cohorts, multi-site studies). The caveats are equally clinical: no variance reporting on key benchmarks, no demographic accounting on the 50M-cell corpus, and no head-to-head against peer foundation models on shared tasks.

## References
- **Paper**: Hao, M., Gong, J., Zeng, X., et al. *Large-scale foundation model on single-cell transcriptomics*. Nature Methods 21, 1481–1491 (August 2024). DOI: [10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7)
- **Code**: [github.com/biomap-research/scFoundation](https://github.com/biomap-research/scFoundation)
- **Weights / data**: Zenodo 8330924; figshare 24049200
- **xTrimoGene base architecture**: Gong, J., et al. *xTrimoGene: An efficient and scalable representation learner for single-cell RNA-seq data.* bioRxiv (2023).
- **Related single-cell transformers**: scBERT (Yang et al., Nat. Mach. Intell. 2022); scGPT (Cui et al., Nat. Methods 2024); Geneformer (Theodoris et al., Nature 2023); UCE (Rosen et al., bioRxiv 2023).
- **Downstream pipelines integrated**: DeepCDR (Liu et al., Bioinformatics 2020); SCAD (Zheng et al., Adv. Sci. 2023); GEARS (Roohani et al., Nat. Biotechnol. 2024); CPA (Lotfollahi et al., Mol. Syst. Biol. 2023).
