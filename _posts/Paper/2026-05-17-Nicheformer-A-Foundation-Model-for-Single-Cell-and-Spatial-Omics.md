---
title: "Nicheformer: a foundation model for single-cell and spatial omics"
excerpt: "A 49.3M-parameter BERT-style transformer pretrained on 110M single-cell + image-based spatial transcriptomics cells (SpatialCorpus-110M) — the only model with positive R² on Xenium lung/colon density prediction where scVI/PCA produce negative R²."
categories:
  - Paper
  - BioInformatics
  - LLM
  - Spatial-Transcriptomics
permalink: /paper/nicheformer/
tags:
  - Nicheformer
  - Foundation-Model
  - Spatial-Transcriptomics
  - Single-Cell
  - Masked-Language-Modeling
  - Transformer
  - SpatialCorpus-110M
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- Nicheformer is a **49.3M-parameter BERT-style transformer** pretrained with masked-language modeling on **SpatialCorpus-110M** — 57.06M dissociated scRNA-seq cells plus 53.83M image-based spatial cells (MERFISH, Xenium, CosMx, ISS) across 73 tissues from human and mouse — aiming to learn a joint cellular representation that transfers spatial context onto dissociated data.
- The conceptual move is multimodal pretraining mixing dissociated and image-based spatial transcriptomics on equal footing, with technology-specific non-zero-mean rank tokenization plus contextual tokens for `<ASSAY>`, `<MODALITY>`, `<ORGANISM>`. Crucially — and this is the most surprising design choice — spatial information is encoded **only via the `<MODALITY>` token and technology-specific normalization**. There are no positional/coordinate embeddings, no neighborhood-graph attention, and no neighbor-aware tokenization. The model sees a spatial cell as a longer padded gene-rank sequence with an "I am spatial" flag.
- Headline result: on Xenium lung and colon density prediction, **fine-tuned Nicheformer is the only model with a positive R², while scVI and PCA baselines fall below random (negative R²)** (Fig. 5C/H). It also beats scVI/PCA on MERFISH brain niche/region F1 and on CosMx neighborhood-composition MAE across organs and radii. The most conspicuous omission — there is **no head-to-head comparison with scGPT, UCE, or CellPLM**, even though CellPLM is explicitly named as the one prior spatially-aware single-cell foundation model.

## Motivation

Single-cell RNA-seq destroys tissue context by dissociation, while image-based spatial transcriptomics (MERFISH, Xenium, CosMx) preserves location but only reads out hundreds to a few thousand targeted genes. Prior single-cell foundation models — Geneformer, scGPT, scFoundation, UCE — ignore spatial structure entirely. The one spatially-aware predecessor, **CellPLM**, used just 9M dissociated + 2M spatial cells and was not fine-tuned beyond gene imputation.

Nicheformer's pitch is that image-based spatial omics is now plentiful enough to support a spatially-informed foundation model that (i) generalizes across organs and technologies and (ii) transfers learned spatial structure onto vanilla scRNA-seq atlases. The medical-AI angle is that predicting niche composition and cellular density on tumor-bearing lung/colon Xenium sections offers a route to prognostic immune-infiltration features without actually running spatial assays at inference.

## Core Innovation

**Multimodal pretraining of dissociated and image-based spatial cells in a single masked-language-modeling corpus.** The key engineering choices that follow from this commitment are:

1. **Technology-specific non-zero-mean rank tokenization.** Each cell is normalized to 10,000 counts and divided by a non-zero-mean expression vector computed *separately per technology* (dissociated, MERFISH, Xenium, CosMx, ISS), then genes are rank-ordered descending with zeros dropped. The per-technology normalization is explicitly motivated by the much higher per-gene counts of image-based assays — without it, image-based cells would dominate the high-rank slots for every gene panel they share.
2. **Contextual tokens.** Each sequence is prefixed by `<ASSAY>`, `<MODALITY>` (dissociated vs spatial), and `<ORGANISM>` (human vs mouse) tokens. Sequence length is truncated/padded to N = 1,500; `<PAD>` is masked from attention.
3. **No explicit spatial inductive bias.** The "spatial" in *Nicheformer* lives entirely in the `<MODALITY>` token plus the per-technology normalization. There is no coordinate embedding, no neighbor-graph attention, no neighbor-token construction. This makes Nicheformer a much weaker spatial model than its name implies, and makes the missing CellPLM comparison (CellPLM does use a neighbor graph) especially conspicuous.
4. **Downstream evaluation that is genuinely new.** The paper proposes four downstream tasks defined on the spatial graph — spatial cell-type / niche / region label classification, neighborhood-composition regression at four radii, and neighborhood-density regression — and these task definitions are arguably the paper's most durable contribution beyond the corpus itself.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | Largest harmonized single-cell + spatial transcriptomics pretraining corpus to date (110M cells, 53.83M spatial). | Fig. 2A,B; Suppl. Tables 3-4 enumerate sources. | SpatialCorpus-110M | ⭐⭐⭐ |
| C2 | Nicheformer learns a representation useful for spatial downstream tasks, beating scVI/PCA. | Fig. 3B, Fig. 4C, Fig. 5C/H. | MERFISH brain, CosMx liver/lung, Xenium lung/colon | ⭐⭐ |
| C3 | Mixed dissociated + spatial pretraining is necessary; dissociated-only is worse. | Suppl. Fig. 2 (brain F1 drops). | MERFISH brain, one task | ⭐ |
| C4 | Nicheformer transfers spatial niche/region labels onto dissociated scRNA-seq data. | Fig. 3 C-K. | mouse motor cortex scRNA-seq → MERFISH labels | ⭐ |
| C5 | Captures neighborhood composition accurately across organs / technologies / resolutions. | Fig. 4C across brain, liver, lung × four radii. | MERFISH brain, CosMx liver, CosMx lung | ⭐⭐ |
| C6 | Predicts cellular density purely from transcriptome, including tumor-associated densification. | Fig. 5C, H (positive R² vs negative for baselines); Fig. 5D, E qualitative tumor zoom. | Xenium lung (2 sections / 2 donors) + Xenium colon (2 sections / 2 donors) | ⭐⭐ |
| C7 | Linear probing is "zero-shot-like" evidence of representation quality. | Repeated framing in Results. | All tasks | ⭐ |
| C8 | Best at 49.3M parameters (architecture choice). | Suppl. Fig. 1, Suppl. Table 1. | Hyperparameter sweep | ⭐ |

**Honest read.** The corpus (C1) and the *direction of effect* on the four headline benchmarks (C2, C5, C6) are well demonstrated against scVI/PCA, and the Xenium density result — where baselines collapse below random — is genuinely striking and the most defensible quantitative claim in the paper. But the audit weakens sharply on the obvious comparison questions:

- **No head-to-head with other single-cell foundation models.** Geneformer, scGPT, scFoundation, UCE, and CellPLM are all cited as the natural baselines, yet none appear in any results table. CellPLM in particular — explicitly named in the Introduction as the only prior spatially-aware model — is conspicuously absent.
- **Spatial encoding via `<MODALITY>` token only.** For a paper titled *Nicheformer*, the lack of any positional, coordinate, or neighbor-aware spatial inductive bias is a surprisingly weak design. The model's "spatial awareness" is, mechanically, a single flag token plus per-technology normalization.
- **Data mixing looks balanced but is not.** Despite a 53.83M / 57.06M near-balance in raw cell counts, sampling is uniform across the corpus with no curriculum, no per-modality weighting, and no class-balancing. Worse, **60.46% of the spatial half (≈32.15M cells) comes from one mouse-brain study**, and 55.23% of spatial cells are unannotated. Effective spatial coverage is far less diverse than the headline numbers suggest.
- **Donor counts are tiny on the density experiments.** Xenium lung and colon use n = 1 healthy + 1 diseased donor per organ; no leave-one-donor-out, no statistical test, no seed-spread reporting anywhere.
- **The "zero-shot label transfer" framing (C4) is supported only by a single mouse motor-cortex dataset** and even there shows large failure modes — 84.7% of the 133 non-neuronal cells are misclassified outside isocortex — that the figure caption admits but the Discussion downplays.
- **The ablation defending mixed-modality pretraining (C3)** is one F1 score on one task. There is no spatial-only counterfactual.

Treat this paper as a strong **resource + framework + benchmark-design** contribution rather than a settled empirical demonstration of foundation-model superiority for spatial omics.

## Method & Architecture

![Nicheformer pipeline: SpatialCorpus-110M, downstream tasks, gene-orthology harmonization, rank tokenizer, and pretrained embedding](/assets/images/paper/nicheformer/fig_p004_01.png)
*Figure 1 — Nicheformer pipeline: a BERT-style transformer trained with masked-language modeling on a 110M-cell pretraining corpus mixing dissociated scRNA-seq and image-based spatial transcriptomics, with rank-tokenized inputs prefixed by `<ASSAY>`/`<MODALITY>`/`<ORGANISM>` contextual tokens.*

The pipeline is best read as eight steps:

1. **Corpus assembly (SpatialCorpus-110M).** Concatenate dissociated scRNA-seq (CellXGene CENSUS + sfaira + GEO + HCA, 57.06M cells) with image-based spatial data (Vizgen MERFISH, 10x Xenium, Nanostring CosMx, ISS — 53.83M cells across ~10,600 tissue sections from 158 donors). Metadata is harmonized via NCBITaxon / Uberon / PATO / EFO; gene IDs map to ENSEMBL, and BioMart provides orthologs. Final vocabulary: **20,310 gene tokens** (16,981 orthologous + 3,178 human-specific + 151 mouse-specific).
2. **Gene-rank tokenization.** Normalize each cell to 10,000 counts, divide by a **technology-specific non-zero-mean expression vector** (separately computed for dissociated, MERFISH, Xenium, CosMx, ISS), rank genes descending, drop zeros. The per-technology normalization is what allows cells with vastly different effective vocabulary sizes (MERFISH brain measures 500 genes; dissociated atlases measure 20k) to live in the same sequence space.
3. **Contextual tokens.** Prepend `<ASSAY>`, `<MODALITY>`, `<ORGANISM>` tokens. Sequence length truncated/padded to N = 1,500; `<PAD>` masked from attention.
4. **Architecture.** 12 encoder blocks, 16 heads, hidden dim D = 512, FFN dim 1,024, learnable positional embeddings, ~49.3M parameters. **Pure self-attention over the gene-rank sequence — no graph attention, no spatial-coordinate embedding.**
5. **Pretraining objective.** BERT-style masked language modeling on 15% of tokens (80% `<MASK>` / 10% random / 10% unchanged), cross-entropy over gene + contextual vocabulary. Trained ~10 days on 12 × A100 40GB, bfloat16, AdamW, batch 9 × 10 grad-accum, LR 1e-5 → 1e-3 with 100k linear warmup then cosine decay, gradient clipping 1.0 → 0.5.
6. **Cell embedding.** Mean-pool over gene-token positions (excluding `<PAD>`) to obtain a 512-dim per-cell vector — the "Nicheformer embedding".
7. **Downstream adaptation.** Two regimes per task: (a) **linear probing** = frozen embedding + single linear layer (framed as zero-shot-like), (b) **fine-tuning** = end-to-end gradients through the transformer + linear head. All downstream training is a single epoch, LR 1e-3, batch 256.
8. **Novel downstream tasks on the spatial graph.** (i) Spatial cell-type / niche / region label classification (cross-entropy); (ii) **neighborhood composition** regression — for each cell, predict the soft cell-type proportion vector $N_r = \mathrm{softmax}(A X_l)$ where $A$ is a binary radius-$r$ adjacency matrix and $X_l$ is a one-hot cell-type matrix, evaluated at radii giving mean neighbor counts of 10/20/50/100, MSE loss; (iii) **neighborhood density** regression — predict $D_r = \sum_j A_{ij}$ (scalar per cell), MSE loss.

![SpatialCorpus-110M composition across dissociated and spatial halves, plus the metadata-harmonization schema](/assets/images/paper/nicheformer/fig_p007_01.png)
*Figure 2 — SpatialCorpus-110M composition: 57.06M dissociated cells across 17 organs (top) and 53.83M image-based spatial cells dominated by brain (60.46%) and lung (9.95%); 55% of spatial cells are unannotated.*

## Experimental Results

![Brain F1 bar chart, label-transfer heatmaps, and uncertainty maps for MERFISH mouse brain](/assets/images/paper/nicheformer/fig_p009_01.png)
*Figure 3 — On MERFISH mouse brain, fine-tuned Nicheformer beats scVI/PCA on niche and region F1 (B); transferring those labels to dissociated motor-cortex scRNA-seq is largely correct at the cell-type level (E) but mislabels 84.7% of non-neuronal cells for region (G, red box).*

![Neighborhood-composition workflow and MAE results across brain/liver/lung at four radii](/assets/images/paper/nicheformer/fig_p012_01.png)
*Figure 4 — Fine-tuned Nicheformer predicts neighborhood cell-type composition with lower MAE than scVI/PCA across all four radii and three organs (C), though per-cell-type accuracy tracks abundance in the pretraining corpus (D).*

![Xenium lung and colon density histograms, embedding UMAPs, and MAE / R² bars](/assets/images/paper/nicheformer/fig_p014_01.png)
*Figure 5 — Nicheformer linear probing recovers cellular neighborhood density in Xenium lung and colon — including tumor-region densification — while scVI/PCA baselines fall below random (negative R²).*

### Headline quantitative comparisons

Numbers below are read off Fig. 3B, Fig. 4C, and Fig. 5C/H (bar-chart heights, not tables); treat as approximate where not explicitly stated in text.

| Task | Dataset | Metric | **Nicheformer fine-tuned** | Nicheformer linear probing | scVI linear probe | PCA linear probe |
|---|---|---|---|---|---|---|
| Niche label | MERFISH mouse brain | F1 macro | **best** | second | lower | lowest |
| Region label | MERFISH mouse brain | F1 macro | **best** | second | lower | lowest |
| Niche label | CosMx liver (zonation) | F1 macro | **best** | worse than scVI/PCA (improved with extra liver-only pretraining, Suppl. Fig. 5F) | competitive | competitive |
| Neighborhood composition | MERFISH brain / CosMx liver / CosMx lung × radii 10/20/50/100 | MAE | **best at every radius and organ** | second | higher | highest |
| Neighborhood density (regression) | Xenium lung | MAE / R² | — (only linear probing reported) | **best, positive R²** | negative R² | negative R² |
| Neighborhood density (regression) | Xenium colon | MAE / R² | — | **best, positive R²** | negative R² | negative R² |

### Qualitative findings and ablations

- **Single-modality ablation (Suppl. Fig. 2).** Pretraining on dissociated-only data degrades F1 on brain spatial cell-type annotation, used to justify mixed-modality pretraining. One-sided — no spatial-only counterfactual.
- **Capacity vs. data scale (Suppl. Fig. 1, Suppl. Table 1).** Authors claim 49.3M parameters "resulted in the best performance compared to models with fewer parameters" — they did **not** sweep larger models.
- **Label transfer to scRNA-seq motor cortex (Fig. 3 D-K).** Of 33 spatial cell types the model correctly narrows to 9 motor-cortex-relevant ones. Glut neurons are correctly labeled at the cell-type level but deep-cortical subtypes (L6b, L6 CT, L5/6 NP) get misregionalized as midbrain Glut; **84.7% of the 133 non-neuronal cells are misclassified outside isocortex** for region prediction (Fig. 3G, called out in the figure caption). Framed as expected given transcriptional-diversity asymmetries (MB Glut has 657 subtypes vs 83 for NP-CT-L6b Glut), but the misclassifications are still substantial.
- **Per-cell-type composition errors (Fig. 4D).** Performance ranking correlates with cell-type abundance in the pretraining corpus; rare midbrain/hypothalamus cell types (MB GABA, MB Dopa, HY Glut) have visibly higher absolute error.
- **Tumor density (Fig. 5).** Xenium lung mean density 12.1 (cancer) vs 10.7 (healthy); colon 12.3 vs 10.7. Nicheformer linear-probe predictions reproduce both the higher cancer-region density and the spatial layout in the zoomed-in Fig. 5E panel; baselines fail outright.

## Limitations

**Authors admit:**

- Performance depends on cell-type / tissue abundance in the pretraining corpus (liver linear-probing failure is the worked example).
- Spatial coordinates are **not** used during pretraining — deliberate, motivated by wanting expression-only supervision. Future work suggested: graph-transformer encoding of spatial neighbor graphs.
- Interpretability not explored.
- Parameter / data / training-time scaling laws unexplored.
- Field lacks standardized spatial-foundation-model benchmarks (the paper effectively proposes its own).

**Not addressed (auditor's observations):**

- **Spatial encoding is mechanically very thin.** Only the `<MODALITY>` token and per-technology normalization carry "spatial-ness". No positional encoding of physical location, no spatial neighborhood attention, no neighbor-aware tokenization. This is a much weaker spatial inductive bias than the model's name implies, and is the single most surprising design choice in the paper.
- **Data mixing is uniform across a heavily skewed corpus.** No curriculum, no per-modality weighting, no class-balancing. With 60% of spatial cells from one mouse-brain study, effective spatial coverage is far less diverse than the headline 53.83M figure suggests.
- **No head-to-head with scGPT / UCE / CellPLM.** The most conspicuous omission. CellPLM is explicitly named as the only prior spatially-aware single-cell foundation model in the Introduction, yet never appears in any results table.
- **Benchmark design quality.** The new tasks (niche label, region label, neighborhood composition, neighborhood density) are biologically reasonable but evaluated only against non-pretrained baselines on a handful of datasets per task, often with n = 1-5 donors. Donor-level splits, statistical tests across runs, and external-cohort generalization are missing.
- **Inference compute / latency** for the 49.3M-parameter model over millions of cells is not reported.
- **The "zero-shot-like" framing of linear probing is a terminology stretch** — linear probing still requires labeled training data on the target dataset; it is not zero-shot in the standard CLIP / GPT sense.

## Why It Matters for Medical AI

The tumor-bearing Xenium lung and colon density result (Fig. 5) is the most clinically suggestive finding: Nicheformer recovers higher cellular density in cancer regions purely from transcriptomes, while scVI and PCA baselines collapse below random. If this generalizes beyond n = 2 donors per organ, the practical implication is that a transcriptome-only inference path could surface prognostic immune-infiltration features that today require an actual spatial assay — at a fraction of the per-sample cost.

The corpus contribution is also durable on its own. **SpatialCorpus-110M** harmonizes 57.06M dissociated and 53.83M image-based spatial cells across 73 tissues, with metadata mapped to NCBITaxon / Uberon / PATO / EFO and gene IDs to ENSEMBL — it is the cleanest substrate to date for anyone training a spatially-aware single-cell foundation model, even one with a stronger spatial inductive bias than Nicheformer's.

The honest framing: this is a research-grade resource + benchmark contribution with one striking quantitative result (Xenium density) and a list of obvious next experiments (head-to-head with scGPT/UCE/CellPLM, neighbor-graph attention, donor-level external-cohort validation) that the community should run before treating Nicheformer-style models as clinically actionable.

## References

- **Paper:** Schaar, A. C.*, Tejada-Lapuerta, A.*, Palla, G., Gutgesell, R., Halle, L., Minaeva, M., Vornholz, L., Dony, L., Drummer, F., Bahrami, M., Theis, F. J. *Nicheformer: a foundation model for single-cell and spatial omics.* **bioRxiv** preprint, April 17, 2024. DOI: [10.1101/2024.04.15.589472](https://doi.org/10.1101/2024.04.15.589472).
- **Affiliations:** Helmholtz Munich / TU Munich.
- **Related (single-cell foundation models):** Geneformer (Theodoris et al., 2023, *Nature*); scGPT (Cui et al., 2024, *Nat. Methods*); scFoundation (Hao et al., 2024, *Nat. Methods*); UCE (Rosen et al., 2023).
- **Related (spatially-aware single-cell FM, the missing comparison):** CellPLM (Wen et al., 2023).
- **Related (data sources):** CellXGene CENSUS v2023-07-15; sfaira; Human Cell Atlas; Vizgen MERFISH; 10x Xenium; Nanostring CosMx; ISS.
- **Related (downstream task baselines):** scVI (Lopez et al., 2018, *Nat. Methods*); PCA linear probing.
- **Related (label-transfer evaluation):** Yao et al., 2023 (MERFISH mouse brain); Yao et al., 2021 (primary motor cortex 10x v3 scRNA-seq).
