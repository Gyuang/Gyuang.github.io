---
title: "scGPT: Toward Building a Foundation Model for Single-Cell Multi-Omics Using Generative AI"
excerpt: "A 12-layer / 8-head / 512-dim transformer pretrained on 33M CELLxGENE normal human cells with a generative masked-attention objective; wins on cell-type annotation and integration, but the headline perturbation claims rest on a curated subset and n=7 test cases, and Geneformer is never benchmarked head-to-head."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/scgpt/
tags:
  - scGPT
  - Foundation-Model
  - Single-Cell
  - scRNA-seq
  - CELLxGENE
  - Transformer
  - Generative-Pretraining
  - Perturbation-Prediction
  - Multi-Omics
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- A 12-layer, 8-head, 512-dim transformer pretrained on **33M normal human scRNA-seq cells from CELLxGENE (15 May 2023 release)** with a custom non-autoregressive "generative masked attention" objective, then fine-tuned with a single backbone for cell-type annotation, perturbation prediction, multi-batch integration, multi-omic integration, and GRN inference.
- Headline numbers: **AvgBIO 0.821 on PBMC 10k integration** (vs. Harmony 0.784 / scVI 0.724 / Seurat v3 0.753); **~0.85 OOD accuracy on MS cell-type annotation** vs. scBERT/TOSICA; "5-20% margin" in Pearson_delta on Adamson/Norman/Replogle perturbation; **reverse-perturbation top-1 relevant 91.4% (6.4/7)** on a curated Norman 20-gene subset.
- The methodological core is an **attention-mask substitute for causal generation** (eq. 11) on inherently unordered gene tokens, plus **per-cell value binning** so bin index k always means "k-th expression quantile within this cell". The downstream wins on annotation and integration are well-supported; the perturbation pitch is weakest — curated Replogle subset, n=7 reverse-perturbation, and **Geneformer is never benchmarked head-to-head despite being the explicit comparator**.

## Motivation

Single-cell atlases (Human Cell Atlas, CELLxGENE) now contain tens of millions of cells across organs and modalities, but mainstream tooling (scVI, Seurat, Harmony, scGLUE, GEARS, TOSICA, scBERT) is task-specific and trained on small slices. By analogy with NLP foundation models, the authors argue one transformer pretrained on the full atlas can be fine-tuned for every downstream task with better performance than bespoke models. The medical-AI angle is that the same backbone is repurposed for disease-cell annotation (multiple sclerosis, tumor-infiltrating myeloid across nine cancer types) and for in-silico CRISPR perturbation design in K562 leukemia cells — i.e., scGPT is pitched as an off-the-shelf substrate for both clinical cell-atlas analysis and target discovery.

Geneformer (Theodoris et al., Nature 2023) is the prior the paper explicitly wants to leapfrog: Geneformer ranks genes by expression and uses BERT-style MLM, while scGPT keeps binned expression values and switches to a generative formulation. The catch — flagged below — is that Geneformer is invoked as the comparator but never benchmarked numerically.

## Core Innovation

- **Per-cell value binning.** Bin edges are recomputed *per cell* so that each non-zero bin contains 1/B of that cell's expressed genes. Bin index k therefore means "k-th expression quantile within this cell" regardless of sequencing depth — a normalization trick that defeats batch scale differences before tokens ever reach the transformer.
- **Generative masked attention on unordered tokens.** Because gene tokens have no canonical order, standard causal masking is meaningless. The authors define a mask (eq. 11) where every query attends to all "known" gene keys plus the query position itself, but no token attends to any other "unknown" token. Iterative inference (K=3) promotes the top 1/K most-confidently predicted unknown genes to "known" at each step — a substitute for GPT-style autoregression on a set, not a sequence.
- **Two prompt modes per cell.** *Gene-prompt* predicts expression at unknown positions from known gene-expression pairs; *cell-prompt* swaps the `<cls>` embedding into the gene-prompt step and predicts genome-wide expression from cell identity alone. Losses are summed before each optimizer step.
- **Same backbone, many fine-tune heads.** Cell-type annotation, perturbation, integration (single- and multi-omic), and GRN inference all reuse the pretrained 12×8×512 transformer with task-specific heads and loss combinations (GEP, GEPC, perturb-GEP, ECS, DAR, DSBN, cross-entropy).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Foundation model pretrained on >33M cells | Fig. 1d cell counts (Brain 13.2M + Blood 10.3M + Lung 2.1M + Heart 1.8M + Kidney 814k + Pancreas 210k + Intestine 94.5k); CELLxGENE 15 May 2023 release pinned | CELLxGENE | ⭐⭐⭐ |
| C2 | Specialized attention masking enables generative pretraining on non-sequential data | Eq. 11; Supp Fig 1 (not in on-disk PDF) | — | ⭐⭐ — no ablation vs. single-shot BERT-style MLM or K=1 |
| C3 | scGPT consistently outperforms TOSICA and scBERT on cell-type annotation | Fig. 2j (n=5 train/val splits, mean ± s.e.m.) | hPancreas, MS (OOD disease), Myeloid | ⭐⭐⭐ — multiple datasets, variance reported, OOD honest; but Seurat label-transfer / scANVI absent |
| C4 | "scGPT excels in perturbation prediction, outperforming others by 5-20% margins" | Fig. 3a; Supp Table 6 | Adamson, Norman, Replogle | ⭐⭐ — no error bars on bars; Replogle is the curated "strong phenotype" subset with missing-perturbed-gene rows dropped |
| C5 | scGPT predicts *unseen* genetic perturbations | Fig. 3a unseen-gene splits (Roohani 2023 protocol); 25% perturbations held out | All 3 perturb datasets | ⭐⭐ — "unseen" means unseen *perturbation token*; cell line (K562) and gene vocabulary are still in-distribution |
| **C6** | **Reverse perturbation: 91.4% relevant top-1, 65.7% correct top-8** | **Fig. 3g** | **Norman 20-gene subset (39 train / 3 val / 7 test of 210 combos)** | **⭐ — n=7 test cases; 20 genes were selected to "maximize the proportion of ground-truth data"; no bootstrap** |
| C7 | AvgBIO 0.821 on PBMC 10k integration, 5-10% above baselines | Fig. 4a | PBMC 10k (2 batches) | ⭐⭐ — single dataset; 0.821 vs. 0.784 is 0.037 absolute, generously rounded |
| C8 | 9% AvgBIO over Seurat v4 on BMMC paired RNA+protein | Fig. 4c | BMMC (90k cells, 12 donors, 48 cell types) | ⭐⭐ — single dataset; multi-omic baselines beyond Seurat v4 not on this dataset |
| C9 | Only method that resolves CD8+ naive as a distinct cluster on 10x Multiome | Fig. 4b | 10x Multiome PBMC | ⭐ — qualitative; no metric for the specific subtype |
| C10 | Gene embeddings recover HLA-I vs. HLA-II in zero-shot | Fig. 5a | pretrained whole-human model | ⭐⭐ — qualitative; cosine threshold hand-picked (>0.5); no head-to-head zero-shot vs. Geneformer embeddings |
| C11 | scGPT extracts 22 more enriched pathways than coexpression at Leiden res 40 | Fig. 5d-e | Immune Human dataset | ⭐⭐ — sensitivity across resolutions shown, but program-size confound not normalized |
| C12 | Attention top-20 genes for DDIT3 KO are 20/20 ChIP-Atlas-validated targets | Fig. 6b | Adamson CRISPRi (blood model fine-tune) | ⭐⭐ — striking, but only 2 TFs in main text; ±10kbp ChIP-Atlas windows are permissive |
| C13 | Scaling: more pretraining data → better downstream | Supp Fig 13a | varied pretrain 30k → 33M | ⭐⭐ — real trend; main text qualitative only, plateau unclear |
| C14 | Pretraining helps over training from scratch | Supp Tables 2-4 | downstream tasks | ⭐⭐ — magnitude not in main text |
| C15 | Aligning organ-specific pretraining to downstream tissue improves results | Supp Fig 8 (organ-specific × COVID-19) | COVID-19 × 7 organ models + whole-human + pan-cancer | ⭐⭐⭐ — clean controlled comparison (brain 13M trails blood 10M by 8% on COVID-19) |

**Audit-worthy point.** Cell-type annotation, integration, and organ-specific pretraining (C3, C7, C15) are well-supported with multiple datasets and at least some variance reporting. Scaling and pretrain-vs.-scratch (C13, C14) live mostly in the supplement. The perturbation claims (C4-C6) — the abstract's loudest pitch — are the weakest: C6 is **n=7 test cases**, the Replogle subset is curated to high-effect perturbations *and* drops perturbations whose perturbed gene is missing from expression (which mechanically inflates Pearson_delta on the perturbed gene's neighborhood), no CI/bootstrap on Fig. 3a, and no comparison to scFoundation / Geneformer / scBERT-perturbation extensions.

## Method & Architecture

![scGPT workflow: generative pretraining on 33M CELLxGENE cells and task-specific fine-tuning across annotation, perturbation, integration, and GRN inference](/assets/images/paper/scgpt/page_003.png)
*Figure 1: scGPT workflow — generative pretraining on 33M CELLxGENE normal-human cells, then task-specific fine-tuning. Inputs fuse gene identity, per-cell-binned expression, and condition tokens; the transformer uses a specialized attention mask for generative gene-expression prediction.*

### 1. Inputs

For each cell *i*, three parallel token streams of length up to **M = 1,200**:

1. **Gene tokens** $t_g^{(i)} = [\mathrm{id}(g_1), \dots, \mathrm{id}(g_M)]$. Vocabulary = union of human gene symbols + special tokens `<cls>`, `<pad>`. Order carries no meaning.
2. **Expression values** $x^{(i)} \in \mathbb{N}^M$ via per-cell value binning:

$$x_j^{(i)} = \begin{cases} k, & \text{if } X_{i,j} > 0 \text{ and } X_{i,j} \in [b_k, b_{k+1}] \\ 0, & \text{if } X_{i,j} = 0 \end{cases}$$

Bin edges $\{b_k\}$ are recomputed *per cell* so each non-zero bin contains 1/B of that cell's expressed genes — so $x_j^{(i)} = B$ always means "highest-expressed in this cell" regardless of sequencing depth. The bin embedding `emb_x` is a **fully-connected layer over the integer bin index**, not a lookup table, justified as preserving ordinal structure.

3. **Condition tokens** $t_c^{(i)}$ — per-position meta-info (perturbation status, etc.).

Element-wise sum is the input embedding:

$$h^{(i)} = \mathrm{emb}_g(t_g^{(i)}) + \mathrm{emb}_x(x^{(i)}) + \mathrm{emb}_c(t_c^{(i)}) \in \mathbb{R}^{M \times D}, \quad D = 512.$$

Batch and modality embeddings are deliberately **concatenated to the transformer output** (not added to the input), so self-attention does not amplify within-modality attention vs. cross-modality:

$$h'_n = \mathrm{concat}(h_n,\ \mathrm{emb}_b(t_b) + \mathrm{emb}_m(t_m)).$$

### 2. Backbone

- **12 transformer blocks × 8 attention heads × 512 embedding dim**, FFN hidden 512.
- **FlashAttention** to scale to M = 1,200 token inputs.
- A reserved `<cls>` token at position 0 produces the cell embedding $h_c^{(i)} \in \mathbb{R}^D$ for cell-level objectives.

### 3. Generative pretraining with specialized attention mask

Self-attention is modified by an additive mask $A_{\text{mask}} \in \{0, -\infty\}^{M \times M}$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d} } + A_{\text{mask} }\right) V$$

Each input token is either (a) `<cls>`, (b) "known" (token + expression bin embedded), or (c) "unknown" (token only). The mask rule (eq. 11):

$$a_{i,j} = \begin{cases} 0, & j \notin \text{unknown} \\ 0, & i=j \text{ and } j \in \text{unknown} \\ -\infty, & i \neq j \text{ and } j \in \text{unknown} \end{cases}$$

So every query can attend to all known gene keys plus the query position itself, but **no token can attend to any other unknown token**. Iterative inference at **K = 3** steps: at each step the top 1/K most-confidently predicted unknown genes are promoted to known for the next step. This is the authors' substitute for GPT-style causal masking on a non-sequential token set.

**Pretraining objective (eq. 12)** is MSE on the unknown positions:

$$\mathcal{L} = \frac{1}{|\mathcal{U}_{\text{unk} }|} \sum_{j \in \mathcal{U}_{\text{unk} }} \left(\mathrm{MLP}(h_n^{(i)}) - x_j^{(i)}\right)^2$$

The target $x_j^{(i)}$ is the *bin index* (an integer in $\{0,\dots,B\}$), but the loss is squared-error not cross-entropy — so the model is regressing on a discretized scalar, blurring "categorical token prediction" and "regression". Two prompt modes are mixed per cell (gene-prompt and cell-prompt) and their losses summed before each optimizer step.

### 4. Pretraining setup

- **33M cells, 99.7% train / 0.3% val** for the whole-human model (97/3 for organ-specific and pan-cancer).
- Only **non-zero genes** are input; cells with >1,200 non-zero genes are subsampled to 1,200 each step.
- **Mask ratio uniformly sampled from {0.25, 0.50, 0.75}** per batch.
- Adam, batch size 32, lr 1e-4, "weight decay 0.9 after each epoch" (the paper's ambiguous phrasing — possibly an lr-decay typo), **6 epochs**.

### 5. Fine-tune recipes per task

| Task | Inputs | Key losses | Notable |
|------|--------|------------|---------|
| Cell-type annotation | normalized + log + binned; common gene vocab | cross-entropy on classifier head | reuses pretrained weights everywhere except head |
| Perturbation | HVGs; **log1p values not bins**; binary perturb token; control-perturbed cell pairs | perturb-GEP | the only task that abandons binning |
| scRNA integration | 1,200 HVGs + log + bins | GEP + GEPC + ECS (weight 10) + DAR + DSBN | mask ratio 0.4; β=0.6 |
| Multi-omic (paired/mosaic) | 1,200 HVGs + 4,000 highly-variable peaks + all proteins | GEP + GEPC (+DAR if multi-batch) | inherits only gene embeddings; transformer reinitialized as 4-block |
| GRN — zero-shot | pretrained gene embeddings | none | kNN graph → Leiden; HLA / CD sanity check |
| GRN — attention | blood model fine-tuned on Adamson | perturb-GEP | aggregate attention from all 8 heads of last layer; per-row then per-column rank-normalized; "difference" = perturbed − control |

## Experimental Results

### Cell-type annotation (Fig. 2)

scGPT beats scBERT and TOSICA on accuracy / precision / recall / macro-F1 on hPancreas, MS (OOD disease), and pan-cancer Myeloid. Values are read off Fig. 2j axes; the paper does not provide an in-text table (exact numbers live in Supp Table 2):

| Metric | Dataset | TOSICA | scBERT | **scGPT** |
|--------|---------|--------|--------|-----------|
| Accuracy | hPancreas | ~0.50 | ~0.70 | **~0.85** |
| Accuracy | MS (OOD) | ~0.40 | ~0.60 | **~0.85** |
| Accuracy | Myeloid | ~0.60 | ~0.55 | **~0.70** |
| Macro-F1 | hPancreas | ~0.45 | ~0.50 | **~0.65** |
| Macro-F1 | MS (OOD) | ~0.30 | ~0.40 | **~0.60** |
| Macro-F1 | Myeloid | ~0.40 | ~0.40 | **~0.55** |

Reported as mean ± s.e.m. across **n=5 train/val splits** — the only place in the main figures where variance is shown. The hPancreas confusion matrix achieves precision >0.8 for most cell types but fails on rare types (mast, MHC-II, <50 reference cells out of 10,600) — honestly reported.

![hPancreas confusion matrix on cell-type annotation showing high precision for common types and failure on rare types like mast and MHC-II](/assets/images/paper/scgpt/fig_p004_03.png)
*Figure 2: hPancreas confusion matrix — precision >0.8 for most cell types except rare classes (mast, MHC-II, <50 cells in reference).*

### Perturbation prediction (Fig. 3)

scGPT vs. GEARS vs. linear regression. Pearson_delta (changes vs. unperturbed mean) and Pearson_delta on top-20 DE genes:

| Dataset | Method | Pearson_delta | Pearson_delta (DE-20) |
|---------|--------|---------------|----------------------|
| Adamson | **scGPT** | **~0.63** | **~0.75** |
| Adamson | GEARS | ~0.55 | ~0.65 |
| Adamson | Linear | ~0.45 | ~0.55 |
| Norman | **scGPT** | **~0.65** | **~0.70** |
| Norman | GEARS | ~0.60 | ~0.65 |
| Norman | Linear | ~0.45 | ~0.55 |
| Replogle | **scGPT** | **~0.50** | **~0.60** |
| Replogle | GEARS | ~0.45 | ~0.55 |
| Replogle | Linear | ~0.35 | ~0.45 |

Values read from Fig. 3a; exact numbers in Supp Table 6. The text claims "5-20% margins" — consistent with the bars but reported without error bars.

**Reverse perturbation (Fig. 3g, Norman 20-gene subset, 7 test cases):** scGPT top-1 *relevant* hit rate **6.4/7 = 91.4%**; top-8 *correct* (exact two-gene combo) hit rate **4.6/7 = 65.7%**. GEARS and a differential-gene baseline are shown for top-1 only, both clearly below scGPT bars but exact numbers not in main text.

### Multi-batch and multi-omic integration (Fig. 4)

![PBMC 10k integration UMAPs with scGPT achieving AvgBIO 0.821 vs Harmony 0.784, scVI 0.724, Seurat v3 0.753](/assets/images/paper/scgpt/fig_p007_01.png)
*Figure 3: PBMC 10k integration — scGPT AvgBIO 0.821 vs. Harmony 0.784 / scVI 0.724 / Seurat v3 0.753.*

| Dataset | Method | AvgBIO | AvgBATCH |
|---------|--------|--------|----------|
| PBMC 10k | **scGPT (fine-tuned)** | **0.821** | — |
| PBMC 10k | Harmony | 0.784 | — |
| PBMC 10k | scVI | 0.724 | — |
| PBMC 10k | Seurat v3 | 0.753 | — |
| BMMC (paired RNA+protein) | **scGPT** | **0.697** | — |
| BMMC | Seurat v4 | ~0.600 | — |
| ASAP PBMC (mosaic) | **scGPT** | 0.587 | **0.951** |
| ASAP PBMC | scMoMaT | 0.546 | 0.916 |
| 10x Multiome PBMC | **scGPT** | **0.758** | — |
| 10x Multiome PBMC | scGLUE | 0.747 | — |
| 10x Multiome PBMC | Seurat v4 | 0.722 | — |

![BMMC paired RNA+protein integration with scGPT separating CD4 naive, CD4 activated, and integrin-β7+ CD4 T cells](/assets/images/paper/scgpt/fig_p007_02.png)
*Figure 4: BMMC paired RNA+protein — scGPT separates CD4 naive vs. CD4 activated and integrin-β7+ CD4 T cells (AvgBIO 0.697 vs. Seurat v4 ~0.600).*

### COVID-19 organ-aware result (Fig. 4 / Supp Fig 8)

A clean controlled comparison: organ-specific submodels are evaluated on a COVID-19 dataset. The **brain submodel (13.2M pretraining cells) trails the blood submodel (10.3M cells) by ~8%** on COVID-19, because COVID-19 is blood/lung tissue. This is one of the more convincing pieces of evidence in the paper — pretraining-tissue alignment matters more than raw scale (C15).

### Gene-program / GRN analysis (Figs. 5-6)

- **Zero-shot HLA / CD networks** — pretrained gene embeddings cluster HLA-I vs. HLA-II correctly with hand-picked cosine threshold >0.5.
- **Pathway enrichment** — across Leiden resolutions 1-60, scGPT-derived gene programs yield 2-4× more Reactome enriched pathways than coexpression baseline. At resolution 40 scGPT and coexpression share 15 pathways; scGPT uniquely identifies 22 more, 14 of which are immune-related. Caveat: the average program size differs between methods, mechanically affecting Bonferroni-corrected pathway counts — not normalized.
- **Attention-based GRN on Adamson** — for DDIT3 KO, all top-20 most-influenced genes are ChIP-Atlas-validated DDIT3 targets; for BHLHE40, 19/20. Striking but only two transcription factors in main text and the ±10kbp ChIP-Atlas window is permissive.

## Honest Read

Five flags the paper's framing glosses over:

1. **CELLxGENE pretrain-downstream leakage is unaudited.** Pretraining filters out *disease* samples but never checks whether the *same source studies* underlying MS-healthy reference, Immune Human, or PBMC 10k appear in the CELLxGENE pretraining release. CELLxGENE aggregates HCA and many PBMC/immune atlases. For a paper whose central claim is foundation-model transfer, leaving leakage unbenchmarked is a real gap.
2. **Geneformer is never benchmarked head-to-head.** Geneformer (Theodoris 2023, Nature) is invoked as the closest prior throughout the introduction and discussion, but no figure or table compares scGPT to Geneformer on any task — annotation, integration, or zero-shot gene embeddings. The "advances beyond Geneformer" framing is asserted rather than measured.
3. **The Replogle perturbation subset is curated.** The original Replogle 2022 release is genome-wide CRISPRi; scGPT evaluates on a 1,823-perturbation subset selected for "strong transcriptional phenotype" — i.e., the easy half — *and* drops 150 perturbations whose perturbed gene shows no expression. Both filters inflate Pearson_delta on the gene's immediate transcriptional neighborhood for all methods, but especially for models that learn high-signal patterns.
4. **No comparisons to scFoundation, UCE, or GeneCompass.** All three are contemporary single-cell foundation models; none appear in any benchmark. For a Nature Methods foundation-model paper this is the comparison readers most want.
5. **Reverse-perturbation 91.4% / 65.7% is n=7.** The 20-gene subset of Norman two-gene combos was selected to "maximize the proportion of ground-truth data," yielding 39 train / 3 val / **7 test** combinations. Top-1 of 6.4/7 and top-8 of 4.6/7 are not statistically stable — no bootstrap, no CI. This is the headline of the perturbation pitch and it is fragile.

## Limitations

**Author-acknowledged (Discussion, p. 10):**

- Pretraining does not inherently mitigate batch effects → zero-shot performance is constrained on datasets with substantial technical variation.
- Biological ground truth is often absent; data quality varies (deferred to Supp Note 10).
- Future work: larger and more diverse pretraining (multi-omic, spatial, disease, perturbation, temporal), in-context instruction learning.

**Not addressed:**

- **MSE on a bin index.** Eq. 12 targets integers in {0,…,B} with squared-error loss, so off-by-3 errors weight differently than a categorical setup. The bin count *B* is never stated in the on-disk text.
- **K iterative steps** and **mask ratio** {0.25, 0.5, 0.75} are not ablated for downstream tasks; only K=3 is shown as a Supp Fig 1b illustration.
- **`emb_x` fully-connected** is justified as preserving ordinality but its capacity (linear? nonlinear MLP?) and impact vs. a lookup table is not ablated.
- **"Weight decay 0.9 after each epoch"** is ambiguous phrasing — likely lr decay, but the paper does not clarify.
- **No compute / wall-clock / energy** disclosure for 33M-cell pretraining over 6 epochs with FlashAttention.
- **No significance tests** on any of the headline benchmark deltas. Cell-type annotation has n=5 splits with s.e.m.; perturbation and integration are single-run bar charts.
- **Class-imbalance handling** — rare-class failure (mast, MHC-II) acknowledged but no oversampling or focal-loss treatment.
- **OOD beyond same-tissue/different-disease** — no mouse-to-human transfer or non-human samples despite the "foundation model" framing; pretraining is human-normal only.

## Why It Matters for Medical AI

scGPT is the most thoroughly downstream-tested single-cell foundation model to date, and the **organ-aware result (C15)** is genuinely useful: it tells practitioners that pretraining-tissue alignment dominates raw scale for clinical applications (e.g., a blood-only 10M-cell submodel beats a whole-human 33M model and a brain 13M model on COVID-19 PBMCs). For cell-atlas analysis, the cell-type annotation and integration recipes are immediately reusable; for target discovery, the attention-based GRN extraction on Adamson CRISPRi gives a reasonable hypothesis-generation pipeline. The caution is that the **perturbation-prediction headlines (5-20% margins, 91.4% reverse-perturbation)** are the part of the pitch most likely to be cited as evidence that foundation models are ready for in-silico CRISPR design, but they sit on a curated Replogle subset and seven test cases — readers should not yet conclude that scGPT outperforms task-specific perturbation models in the wild.

## References

- Paper: Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, **21**(8), 1470-1480. [doi:10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)
- arXiv preprint: [2305.06351](https://arxiv.org/abs/2305.06351)
- Code: [github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) (MIT)
- Model checkpoints: Zenodo [10.5281/zenodo.10466117](https://doi.org/10.5281/zenodo.10466117)
- Processed datasets: figshare [10.6084/m9.figshare.24954519.v1](https://doi.org/10.6084/m9.figshare.24954519.v1)
- Pretraining corpus: [CELLxGENE Census, 15 May 2023 release](https://cellxgene.cziscience.com/)
- Related: Theodoris et al. (2023). Transfer learning enables predictions in network biology (Geneformer). *Nature* **618**, 616-624.
- Related: Roohani et al. (2023). Predicting transcriptional outcomes of novel multigene perturbations (GEARS). *Nature Biotechnology*.
- Related: Dao et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness.
