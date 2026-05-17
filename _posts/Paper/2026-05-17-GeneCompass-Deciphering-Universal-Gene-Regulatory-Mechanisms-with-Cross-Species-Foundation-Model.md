---
title: "GeneCompass: Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model"
excerpt: "A 100M-parameter BERT pre-trained on 126M human+mouse single-cell transcriptomes with four biological priors reaches macro-F1 0.87 on hMS cell-type annotation and AUC 0.95 on dosage sensitivity, but the title's 'universal' / 'cross-species' claim wins only 4 of 7 paired datasets."
categories:
  - Paper
  - BioInformatics
  - LLM
permalink: /paper/genecompass/
tags:
  - GeneCompass
  - Foundation-Model
  - Single-Cell
  - Cross-Species
  - Knowledge-Prior
  - BERT
  - Perturbation-Prediction
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- GeneCompass is a 12-layer, ~100M-parameter BERT trained on **scCompass-126M** (>=50M human + >=50M mouse single-cell transcriptomes) with a unified token dictionary of **17,465 homologous genes** and four biological priors (promoter via DNABERT, gene family / co-expression / TF-target GRN via gene2vec) injected as concatenated input embeddings — not as a separate regularizer.
- Dual self-supervised heads recover masked **gene IDs (cross-entropy)** and **absolute expression values (MSE)**, giving stronger supervision than rank-only objectives (Geneformer) for downstream quantitative perturbation tasks.
- Headline numbers: macro-F1 **0.87 / 0.84 / 0.71** on hMS / hLung / hLiver cell-type annotation (vs Geneformer 0.75 / 0.78 / 0.70), dosage-sensitivity **AUC 0.95**, GEARS perturbation top-20 DE **MSE -15.4%**. The advertised cross-species story is weaker: GeneCompass-CAME wins only **4 of 7** paired mouse->human datasets, with most deltas <=0.02.

## Motivation

Prior single-cell foundation models (Geneformer, scGPT, scFoundation) are all trained on a single species (human) and rely on either ranked gene order or binned expression alone. This leaves two structural gaps:

1. **No cross-species transfer.** Evolutionary conservation of gene regulation between vertebrates is ignored — every species ends up needing its own model and training corpus.
2. **No biological priors.** Decades of curated structural-biology knowledge (promoter sequences, gene families, TF->target GRNs, co-expression networks) sit unused outside the model.

GeneCompass argues that fusing a cross-species pre-training corpus with knowledge-as-tokens yields a more universal representation of gene regulation — which matters clinically because the downstream tasks the authors target (drug response, dosage sensitivity for CNV interpretation, in silico reprogramming, perturbation prediction) are directly translational.

## Core Innovation

- **Unified cross-species token dictionary.** 17,465 homologous gene IDs shared between human and mouse, plus a prepended species token. A single model now embeds both species in one space.
- **Knowledge-as-tokens.** Each gene token is the concatenation of (gene ID, *absolute* normalized expression, promoter embedding, GRN embedding, gene-family embedding, co-expression embedding). Priors are part of the input — they ride along through every attention layer rather than acting as auxiliary losses.
- **Dual MLM objective.** Masking 15% of gene tokens, two heads jointly predict the gene ID (CE) and the absolute expression value (MSE). The absolute-value head is what enables quantitative downstream perturbation (rather than just rank-shifts).

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | First knowledge-informed cross-species scRNA foundation model with 4 priors | Architecture (Fig. 1a, b); ablation on each prior (Ext. Fig. 1a) | scCompass-126M; hMS | ⭐⭐ |
| C2 | 126M cross-species pre-training improves single-species cell-type annotation over Geneformer | Fig. 3b: hMS / hLung / hLiver macro-F1 0.87 / 0.84 / 0.71 vs Geneformer 0.75 / 0.78 / 0.70 | hMS, hLung, hLiver | ⭐⭐⭐ |
| C3 | Cross-species pre-training enables mouse->human transfer better than single-species CAME | Fig. 3d: GeneCompass-CAME vs CAME on 7 paired datasets | brain, pancreas, retina, testis | ⭐ |
| C4 | Absolute expression head gives a "fine-grained" advantage over rank-only (Geneformer) | Implicit via Fig. 3 + qualitative reprogramming in Fig. 6 | hMS etc.; OSKM, ESC->Leydig | ⭐ |
| C5 | The 4 priors jointly boost performance | Ext. Fig. 1a leave-one-in ablation on hMS only | hMS | ⭐ |
| C6 | Pre-trained embeddings help GRN inference, drug-response, drug-induced expression, dosage sensitivity | Fig. 4b-e: monotonic improvement with corpus size; beats scGPT, Geneformer, DeepSEM/DeepCE | hESC; sci-Plex A549; L1000; dosage-sensitive TF set | ⭐⭐ |
| C7 | Replacing GEARS' gene embedding with GeneCompass improves perturbation prediction | Fig. 5b-g: MSE -15.4%, Pearson +2.2 pp, wrong-direction -19.4%, two-gene gain +12.5% | Norman Perturb-seq | ⭐⭐ |
| C8 | GeneCompass enables quantitative in silico reprogramming (OSKM->iPSC) and KO (Zbtb11/Zfp131->endoderm) | Fig. 6b-d embedding-shift density plots | qualitative cell pools | ⭐ |
| C9 | Leydig-cell screen recovers known regulators WT1, NR5A1, GATA4 (+ NR2F1, TCF21) | Fig. 6e bar-ranks | human ESC in silico | ⭐⭐ |
| C10 | AUC up to 0.95 on dosage sensitivity, beating Geneformer at every data scale | Fig. 4e | dosage-sensitive TF benchmark | ⭐⭐ |

**Overall reading.** GeneCompass' strongest claims — C2 (single-species annotation), C10 (dosage sensitivity), C7 (perturbation) — are credibly evidenced by direct head-to-head comparisons with Geneformer/GEARS using the published protocols, and the data-scaling story is internally consistent. The advertised *cross-species* story (C3), foregrounded in the title as "universal" and "cross-species," is the weakest in evidence: GeneCompass-CAME **loses on 3 of 7** paired datasets (Baron_m-h, Chen-Lake, testis_m-h), wins by <=0.02 on most others, and the abstract's "+7.5% retina" disagrees with the on-figure label (the FACS-panc8 entry reads 0.62->0.63). The 4-prior story (C1, C5) is similarly under-supported: the ablation is single-task (hMS only) and **promoter prior alone hurts** cell annotation — only the full +All combination is best. There is **no benchmark vs scFoundation or UCE** (the two most direct cross-species baselines), **no variance bars or significance tests** on any quantitative comparison, and **no held-out species** to validate the "universal" claim in the title.

## Method & Architecture

![GeneCompass overview: 126M cross-species corpus, multi-task MLM, knowledge priors](/assets/images/paper/genecompass/fig_p030_01.png)
*Figure 1: GeneCompass overview — a 12-layer transformer pre-trained on 126M human+mouse single-cell transcriptomes with four knowledge priors (promoter via DNABERT, gene family / co-expression / GRN via gene2vec), masking 15% of gene tokens and recovering both gene IDs (CE) and absolute expression values (MSE).*

### 1. Corpus — scCompass-126M

Over 126M single-cell transcriptomes were aggregated from GEO, SRA, ArrayExpress, GSA, CELLxGENE, HCA, 3CA, Cell BLAST, and TEDD (>=50M human + >=50M mouse). QC removes cells with <7 detected genes; counts are normalized + log1p transformed. Human and mouse gene lists are aligned by homology, producing a shared dictionary of **17,465 homologous genes** out of 36,092 total.

Organ composition is heavily skewed in the mouse half (Nervous system 32.5%, Lung 17.9%, Brain 17.9%, Pancreas 8.9%, Liver 6.5%, Airway 5.7% — three organs cover ~68%), a sampling bias the paper does not flag.

### 2. Input tokenization

For each cell: prepend a species token + `<CLS>`, then represent each gene as a token bundle of (gene ID, *absolute* normalized expression value, promoter embedding, GRN embedding, gene-family embedding, co-expression embedding). Genes are ranked within the cell by normalized expression. Sequences are **2,048 tokens x 768 dims** with gene padding.

### 3. Knowledge embeddings (all 768-dim)

- **Promoter:** 2,500 bp per gene (500 bp upstream + 2,000 bp downstream of TSS) -> DNABERT fine-tuned 40 epochs.
- **Co-expression:** Pearson correlation per gene pair on the corpus; pairs with PCC > 0.8 fed into **gene2vec**.
- **Gene family:** 1,645 human + 1,539 mouse families; all in-family gene pairs become gene2vec training samples.
- **GRN:** TF->target gene pairs from TRRUST, RegNetwork, STRING/ChIP-seq -> gene2vec on edge frequencies.

### 4. Backbone and objective

12-layer bidirectional transformer (BERT-style), 12 attention heads per layer, hidden 768, GELU, dropout 0.02, layer-norm eps = 1e-12, **~100M parameters total**.

Mask 15% of genes per cell. Two decoding heads simultaneously predict:

$$\mathcal{L}_{\text{id}} = \sum_{x \in n_{\text{unk}}} p(x)\log q(x)$$

$$\mathcal{L}_{\text{exp}} = \frac{1}{|n_{\text{unk}}|}\sum_{i \in n_{\text{unk}}}(\hat{x}_i^{(m)} - x_i^{(i)})^2$$

The absolute-value head — not just rank, as in Geneformer — is what enables quantitative perturbation downstream.

### 5. Training compute

AdamW, linear-decay LR with 10,000 warm-up steps, max LR 1e-3, batch size 10 (with gene padding), DeepSpeed for mixed precision and parallelism. Pre-training takes **9 days on 4 nodes x 8 NVIDIA A800 GPUs** (~6,900 GPU-hours).

### 6. Fine-tuning

The 12-layer encoder is kept; task-specific decoders include a single FC head for classification and integrations with GEARS / CPA / DeepSEM / DeepCE / CAME for perturbation, dose response, GRN inference, drug-induced expression, and cross-species annotation respectively. Cell embedding = `<CLS>` token from last layer (768-d); gene embeddings = per-gene last-layer outputs.

For **cross-species annotation (GeneCompass-CAME)** the model builds a heterogeneous graph (cell<->gene, cell<->cell, gene<->gene, self-loops), uses GeneCompass cell embeddings as initial cell-node features, and aggregates pre-trained embeddings of neighbor cells as gene-node initialization, followed by multi-head graph attention with label-smoothed cross-entropy.

For **in silico perturbation**, target gene expression is set to median (low) / max (high) for over-expression or fractions of original (1/2, 1/4, 0) for KO; tokens are re-ranked, a forward pass yields cosine-similarity shifts between perturbed and target cell-state embeddings.

## Experimental Results

### Single-species cell-type annotation (Fig. 3b, c)

| Dataset | from_scratch | Geneformer | **GeneCompass** |
|---|---|---|---|
| hMS (Macro-F1) | 0.68 | 0.75 | **0.87** |
| hMS (Accuracy) | 0.81 | 0.87 | **0.91** |
| hLung (Macro-F1) | 0.75 | 0.78 | **0.84** |
| hLung (Accuracy) | 0.87 | 0.90 | **0.91** |
| hLiver (Macro-F1) | 0.61 | 0.70 | **0.71** |
| hLiver (Accuracy) | 0.73 | 0.79 | **0.82** |
| mBrain (Macro-F1) | 0.72 | — | **0.98** |
| mLung (Macro-F1) | 0.47 | — | **0.63** |
| mPancreas (Macro-F1) | 0.34 | — | **0.70** |

Mouse columns omit Geneformer (it is a single-species human model), so the headline +26 / +16 / +36 macro-F1 deltas on mouse datasets are **improvements over no pre-training, not over a SOTA mouse baseline**.

![Cell-type annotation and cross-species transfer results](/assets/images/paper/genecompass/fig_p034_01.png)
*Figure 3: Cell-type annotation scaling with pre-training corpus (a), single-species human + mouse bars vs from-scratch / Geneformer (b, c), and cross-species mouse->human transfer via GeneCompass-CAME on 7 paired datasets (d).*

### Cross-species annotation (mouse -> human, Fig. 3d)

| Dataset | CAME | **GeneCompass-CAME** |
|---|---|---|
| NMDA-Mnseq | 0.84 | **0.91** |
| Campbell-Lake | 0.77 | **0.79** |
| FACS-panc8 | 0.62 | **0.63** |
| LDP60-Mnseq | 0.91 | **0.92** |
| Baron_m-h | **0.96** | 0.95 |
| Chen-Lake | **0.88** | 0.87 |
| testis_m-h | **0.75** | 0.72 |

GeneCompass-CAME wins 4 / 7 datasets, with the only large gain on NMDA-Mnseq (+7). On 3 of 7 paired datasets it is **worse** than vanilla CAME — the paper describes the overall result as "subtle yet noteworthy." There is also a **labeling inconsistency**: the abstract advertises "+7.5% retina improvement," but the figure shows the only retina-adjacent entry, FACS-panc8, going 0.62 -> 0.63.

### Other downstream tasks (Fig. 4)

![Pre-trained embeddings on GRN inference, dose response, drug-induced expression, dosage sensitivity](/assets/images/paper/genecompass/fig_p035_01.png)
*Figure 4: Pre-trained gene embeddings improve GRN inference (AUPRC), drug dose response (R^2), drug-induced expression (RMSE), and dosage sensitivity prediction (AUC up to 0.95), all monotonic in pre-training corpus size.*

- **GRN inference (AUPRC, hESC):** GeneCompass at 55M-cell pre-training reaches ~2.2 average precision, above DeepSEM / scGPT / Geneformer (~1.85-2.0). Monotonic in pre-training cell count.
- **Dose response (R^2):** at 5x10^7 cells, average R^2 ~0.92 with low variance vs Geneformer ~0.88 and scGPT ~0.85.
- **Drug-induced expression (RMSE):** GeneCompass drops to ~2.022 at 5x10^7 cells vs DeepCE / scGPT ~2.030 — a ~0.4% gap, no error bars.
- **Dosage sensitivity (AUC):** up to **0.95** with full corpus; Geneformer is bounded around 0.88. GeneCompass beats Geneformer at every training-data size.

### Perturbation prediction (Fig. 5, Norman Perturb-seq)

![GeneCompass replaces GEARS gene embedding for perturbation prediction](/assets/images/paper/genecompass/fig_p036_01.png)
*Figure 5: Replacing GEARS' co-expression+GO gene embedding with GeneCompass on the Norman Perturb-seq dataset — top-20 DE MSE -15.4%, Pearson 0.798 -> 0.820, wrong-direction predictions -19.4%, two-gene perturbation gain +12.5%.*

- **Top-20 DE MSE:** drops 15.4% (~0.245 -> ~0.207).
- **Pearson correlation:** 0.798 -> 0.820 (+2.2 pp).
- **Wrong-direction predictions among top-20 DE genes:** 356 -> 287 (-19.4%).
- **Top-20 DE deviation:** -5.9% (one-gene) and -12.5% (two-gene) vs GEARS.
- **Case example (TGFBR2+PRTG):** 17 of 20 top-DE genes are closer to ground truth than GEARS.

Single dataset, no other Perturb-seq corpora (e.g., Replogle), no statistical test.

### In silico reprogramming and differentiation (Fig. 6)

![In silico reprogramming and Leydig-cell regulator screen](/assets/images/paper/genecompass/fig_p038_01.png)
*Figure 6: Quantitative in silico perturbation — OSKM-driven iPSC reprogramming in human and mouse, Zbtb11 / Zfp131 dose-dependent KO toward endoderm, and a Leydig-cell regulator screen recovering WT1 / NR5A1 / GATA4 in the top ~30%.*

- Over-expressing OSKM in fibroblasts shifts cell embeddings toward iPSC in both species; high-level over-expression shifts more than low (qualitative density-plot result, no quantitative comparison metric).
- Decreasing Zbtb11 / Zfp131 to 1/2, 1/4, 0 in mouse ESCs gradually shifts embeddings toward endoderm.
- Genome-wide single-gene over-expression screen on human ESCs ranks WT1, NR2F1, TCF21, NR5A1, GATA4 in the top ~30% for Leydig-shift; 3 / 5 hits match prior wet-lab evidence.

### Ablations

![Knowledge-prior ablation on hMS](/assets/images/paper/genecompass/fig_p040_01.png)
*Extended Fig. 1: Ablation of the four priors on hMS (promoter prior alone hurts; +All is best), mPancreas confusion matrix, and recall delta vs TOSICA.*

- **Knowledge ablation (Ext. Fig. 1a, hMS only):** From a "no prior" baseline (~0.84 macro-F1 in the line plot), adding each prior alone changes accuracy as: **promoter alone drops below baseline**, gene family and co-expression each give modest gains, GRN gives the largest single-prior gain, and **+All** (all four combined) gives the highest score (~0.86). The promoter-prior regression is reported in one sentence and never investigated.
- **Pre-training data scale (Fig. 3a):** all curves (HM 12-layer, H 12-layer, M 12-layer, 6-layer variants) rise monotonically with cell count; the human+mouse 12-layer curve is uniformly above human-only and mouse-only — supporting the "adding cross-species corpus helps" claim *for human cell annotation*.

## Limitations

**Acknowledged by the authors.**
- Only two species (human + mouse); adding more species without solving species-specific expression patterns might offset gains.
- Priors are limited to four; enhancer, protein sequence, epigenomic / proteomic / metabolomic modalities are missing.
- Cross-species improvement is "subtle."

**Not addressed.**
- **No comparison vs scFoundation (50M cells) or UCE (multi-species)** — the two most direct cross-species baselines.
- **No variance bars, confidence intervals, or significance tests** on any reported number.
- Cross-species annotation **loses on 3 of 7** paired datasets; addressed only as "subtle."
- Mouse corpus organ composition is severely skewed (3 organs ~68%); no analysis of how this biases embeddings or affects mouse downstream evaluation.
- The "promoter prior hurts cell annotation alone" finding is mentioned once and never investigated.
- Pre-training compute (~6,900 A800-hours) is comparable to scGPT / Geneformer, but the marginal contribution of cross-species data vs raw scale is not characterized.
- The "universal gene regulatory mechanism" claim in the title is tested on **zero held-out species** — "universal" is overstated for an HM-only model.
- The abstract's "+7.5% retina improvement" disagrees with the on-figure FACS-panc8 0.62 -> 0.63 entry; this labeling inconsistency is unresolved.
- Data not yet released at preprint time; reproducibility hinges on the promised GitHub release.

## Why It Matters for Medical AI

The downstream tasks GeneCompass targets are directly clinical: **dosage-sensitivity AUC 0.95** is meaningful for interpreting copy-number variants and TF haploinsufficiency in disease; perturbation prediction (Norman Perturb-seq, MSE -15.4%) is the substrate for genetic screen design and target prioritization; in silico reprogramming with quantitative dose response (OSKM, Zbtb11 KO, Leydig screen) is the kind of preclinical "what if I knock this down" probe that wet-lab pipelines cannot run at scale. The cross-species transfer would also be clinically valuable — most disease models begin in mouse, and a unified embedding that meaningfully transfers to human would shorten translation — but the published evidence (4/7 wins, deltas <=0.02, no scFoundation/UCE comparison) does not yet justify the "universal" framing. As a foundation-model backbone for human-only quantitative tasks it is credible today; as the cross-species, knowledge-informed framework the title advertises, it needs (a) held-out species, (b) head-to-head vs scFoundation / UCE, (c) leave-one-prior-out ablations beyond hMS, and (d) statistical significance reporting before the claim is on solid ground.

## References

- **Paper:** Yang et al. "GeneCompass: Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model." bioRxiv 2023.09.26.559542 (posted September 28, 2023). <https://doi.org/10.1101/2023.09.26.559542>
- **Code:** <https://github.com/xCompass-AI/GeneCompass>
- **Related work:**
  - Geneformer (Theodoris et al., Nature 2023) — single-species human foundation model, ranked gene order.
  - scGPT (Cui et al., Nature Methods 2024) — single-species human, binned expression.
  - scFoundation (Hao et al., 2023) — 50M human cells, no priors, no cross-species.
  - UCE (Rosen et al., 2023) — multi-species universal cell embedding, explicit cross-species evaluation.
  - GEARS (Roohani et al., Nature Biotechnology 2024) — perturbation prediction baseline GeneCompass plugs into.
  - CAME — cross-species cell-type annotation baseline GeneCompass-CAME extends.
  - DNABERT, gene2vec — backbones for the four knowledge priors.
