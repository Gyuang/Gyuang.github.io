---
title: "A visual-omics foundation model to bridge histopathology with spatial transcriptomics"
excerpt: "OmiCLIP trains a CLIP-style dual encoder on 2.2M Visium image-transcriptome pairs by treating ranked gene symbols as a 'gene sentence', and powers the Loki platform across alignment, annotation, decomposition, retrieval, and ST expression prediction."
categories:
  - Paper
tags:
  - OmiCLIP
  - Loki
  - Spatial-Transcriptomics
  - Vision-Language
  - Foundation-Model
  - Computational-Pathology
  - Contrastive-Learning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- Pathology vision-language models (PLIP, CONCH) bridge H&E to natural-language captions; single-cell foundation models (scGPT, Geneformer) live only in the transcriptomic modality. OmiCLIP closes the gap by training a CLIP/CoCa-style dual encoder on **2.2M Visium image-transcriptome tiles across 1,007 samples and 32 organs** — but the load-bearing trick is that the "text" side is **not natural language**, it is a space-separated sentence of the top-50 expressed gene symbols (ranked by expression).
- The Loki platform packages five downstream modules over the joint embedding space: **Align, Annotate, Decompose, Retrieve, PredEx**. Annotate and Retrieve are zero-shot; the rest take a short finetune on the target slide.
- Headline numbers from the paper: median PCC **0.88 (ST-to-ST)** and **0.86 (image-to-ST)** alignment on ovarian carcinosarcoma vs. 0.71 (CAST) / 0.43 (GPSA) / 0.26 (PASTE); zero-shot marker-gene annotation F1 **0.59-0.96** vs. OpenAI CLIP 0.03-0.34; image to transcriptome retrieval Recall@10% **2.2-3.6x** better than PLIP and OpenAI CLIP; cell-type decomposition impact-score rank **#1 and #2 of 12 methods** on the in-house TNBC sample.

## Motivation

Pathology foundation models (UNI, GigaPath, PLIP, CONCH) bridge H&E either to other images or to natural-language captions. None carry molecular signal. On the other side, single-cell foundation models (scGPT, Geneformer, scFoundation) live entirely in the transcriptome and cannot reason about morphology. 10x Visium finally provides paired image-transcriptome data at scale, and the authors argue the missing primitive is a true image to transcriptome aligned space: a joint embedding where bulk RNA-seq, scRNA-seq references, or marker-gene lists can be used as "text queries" against an H&E slide, and where morphology can be used to retrieve, decompose, or predict spatial gene expression.

The clinical motivation is straightforward — Visium experiments cost upward of $1k per slide and are slow, whereas H&E is universal. If an H&E-only pipeline can decompose cell types, predict gene expression, or align to a 3D stack, it reduces sequencing cost and unlocks 3D pathology workflows.

## Core Innovation

**Gene symbols as CLIP "text", not English.** This is the conceptual move that separates OmiCLIP from prior work and is the single most important thing to internalize about the paper. A standard CLIP / PLIP / CONCH text encoder consumes English captions (e.g., *"H&E image of invasive ductal carcinoma"*). OmiCLIP keeps the same architecture — a causal-masking transformer text encoder initialized from LAION-5B — but feeds it a non-linguistic token sequence: for each Visium spot, the top-50 expressed genes ranked by expression level are concatenated with spaces to produce a sentence like

```
SNAP25 ENO2 CKB GRIN2C CAMK4 ... MTOR VPS13D
```

The text encoder treats this as if it were English. Rank-ordering (à la Geneformer / scFoundation) erases batch effects without depending on raw counts, and the resulting embedding becomes the contrastive partner for the per-spot H&E patch. The paper credits Cell2Sentence and GenePT for the gene-sentence idea but is the first to use it as the text branch of a vision-language model.

Two consequences fall out of this choice:

1. **Zero-shot use is unusually flexible.** Any source that can be reduced to a ranked gene list — bulk RNA-seq, scRNA-seq cells, hand-curated marker-gene panels — becomes a valid "text query" against an H&E patch via cosine similarity.
2. **It is not natural language.** The user cannot ask "where are the T cells?" in English; the model only knows gene symbols. The Loki platform partly papers over this by averaging Loki similarities with PLIP similarities at inference (Loki+PLIP fusion), but a true triple-modal model — gene-sentence + image + natural language — is left as future work.

Loki, the platform built on this backbone, exposes five tasks: **Align** (2D and 3D ST registration via a modified Coherent Point Drift on OmiCLIP embeddings), **Annotate** (zero-shot tissue / cell-type classification from bulk RNA-seq or marker-gene panels), **Decompose** (cell-type decomposition via Tangram applied on OmiCLIP embeddings rather than raw counts), **Retrieve** (image to transcriptome retrieval across an ST-bank), and **PredEx** (similarity-weighted gene-expression prediction from H&E only).

## Claims & Evidence Analysis

| # | Claim | Experimental evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CLIP-style CL on image + ranked-gene-symbol "sentences" yields a useful joint visual-omics embedding. | Calinski-Harabasz improves significantly after CL on both encoders; OmiCLIP image embeddings beat UNI and Prov-GigaPath on per-organ clustering. | ST-bank Ext. Figs. 1-2 | ⭐⭐⭐ |
| C2 | Loki Align beats PASTE / GPSA / CAST on ST-to-ST and image-to-ST alignment. | Sim P=2.9e-34 to 1.1e-8; small intestine n=7; ovarian carcinosarcoma PCC 0.88/0.86 vs. 0.71/0.43/0.26; Visium↔Xenium 0.08 mm; CPD-on-PCA ablation. | 200 sims + 3 real datasets | ⭐⭐⭐ |
| C3 | Loki Annotate beats OpenAI CLIP and matches/improves PLIP for zero-shot histopathology classification. | F1 0.59-0.96 (Loki) vs. 0.03-0.34 (OpenAI CLIP) vs. 0.50-0.93 (PLIP); Loki+PLIP fusion always best. | CRC7K, WSSS4LUAD, LC25000, PatchCamelyon | ⭐⭐⭐ |
| C4 | Loki Decompose is the best of 12 cell-type decomposition methods. | Impact 1.32 (Loki-ST), 1.11 (Loki-image) vs. 0.87 (RCTD next-best); swap-OmiCLIP-with-scGPT/Geneformer/scFoundation drops to rank 6/8/9. | In-house TNBC (Xenium GT), CRC Visium-HD, mouse brain | ⭐⭐ |
| C5 | Loki Retrieve significantly outperforms PLIP and OpenAI CLIP at image→transcriptome retrieval. | Recall@10% 2.2-3.6x baselines on 4 val sets; 2.8-3.4x on 4 external test studies; random baseline reported. | 4 val + 4 test ST + 8 in-house | ⭐⭐⭐ |
| C6 | Loki PredEx beats Hist2ST, HisToGene, BLEEP, mclSTExp at H&E→ST gene expression prediction. | Best MSE on 28/39 heart samples; best PCC on 16/39 samples. | 39 normal heart samples (Kanemaru 2023) | ⭐⭐ |
| C7 | Model is robust to image quality and sequencing-depth variation. | Gaussian-noise (n=10) and downsampling (n=500) tests. | ST-bank simulations | ⭐⭐ |
| C8 | Marker-gene + image classification gives a "triple-modal" boost. | Loki+PLIP fusion gains 1-13 F1 points on 4 datasets. | CRC7K, WSSS4LUAD, LC25000, PatchCamelyon | ⭐⭐ |
| C9 | OmiCLIP generalizes to bulk RNA-seq and scRNA-seq despite Visium-only pretraining. | Supplementary Notes 1-2: scRNA-seq cell annotation, TCGA tumor classification. | TCGA, GTEx, scRNA-seq references | ⭐⭐ |
| C10 | Loki "demonstrated consistent accuracy and robustness compared with 22 SOTA models on 5 simulations, 19 public + 4 in-house datasets" (abstract). | Sum of all the above. | All evaluation datasets | ⭐⭐ |

**Honest read.** The strongest evidence sits with **C1 / C2 / C3 / C5** — multi-dataset, statistically tested, and properly ablated. C1 in particular gives the cleanest argument for the gene-sentence idea: after CL, OmiCLIP image embeddings beat UNI and Prov-GigaPath at per-organ clustering, which is a fair pathology-FM comparison, and replacing OmiCLIP with scGPT / Geneformer / scFoundation in the *same* Tangram pipeline drops decomposition rank from #1 to 6/8/9 — i.e., it is the visual-omics joint training, not just "use a transcriptome FM", that is doing the work.

The ⭐⭐ rows deserve to stay at ⭐⭐. **C4**'s headline #1 ranking is real on the in-house TNBC sample (n=1 slide, with 12 methods compared), but on the second decomposition dataset (CRC Visium-HD) the bar plot is labeled "NS" against Tangram — i.e., not significantly better. **C6** wins MSE on 28/39 samples but PCC on only 16/39 (41%) — barely above chance among five methods (20%) and clearly not dominant. **C10** is a marketing aggregation: "22 SOTA / 5 sims / 19 + 4 datasets" counts each baseline separately across tasks; on any single task the comparison is against 3-11 baselines on 1-4 datasets. Two structural gaps the paper does not address: (a) no confidence intervals or seeds on the headline numbers (box plots show spot-wise, not run-to-run variance), and (b) all pretraining and most evaluation is on 10x Visium, so the "foundation model for spatial omics" framing is really a Visium-foundation model with Xenium / Visium-HD added via pseudo-Visium re-binning at inference.

## Method & Architecture

![OmiCLIP pretraining and the Loki platform overview](/assets/images/paper/omiclip_loki/page_003.png)
*Figure 1 — OmiCLIP contrastive pretraining on 2.2M Visium image-gene-sentence pairs across 32 organs, with the five downstream modules of the Loki platform: Align, Annotate, Decompose, Retrieve, PredEx.*

![Cross-modal similarity heatmap across 32 organs and disease conditions](/assets/images/paper/omiclip_loki/page_003.png)
*Figure 2 — OmiCLIP image↔transcriptome similarity heatmap. A strong diagonal within organ / disease confirms the joint embedding captures tissue identity across modalities.*

The architecture is CoCa-style with a vanilla ViT image encoder and a 76-token causal-masking text encoder. The pipeline is best read as ten steps:

1. **Gene-sentence construction.** For each Visium spot, take the top 50 expressed genes, rank by expression, concatenate gene symbols with spaces to form a 50-token sentence. Preprocessing: Ensembl→symbol conversion, housekeeping-gene removal, Seurat/Scanpy normalization, then rank-ordering.
2. **Image side.** H&E WSIs cropped to per-spot tiles matching the Visium spot footprint; slides under 2000x2000 px are dropped; tiles resized to 224x224.
3. **Text side.** Causal transformer initialized from LAION-5B. The authors note that LAION-5B initialization (which contains biological literature) already helps cluster tissue-similar patches before any finetuning — though they do not run a randomly-initialized ablation, leaving this contribution unquantified.
4. **Contrastive objective.** Symmetric InfoNCE over the 768-d image and text embeddings:

$$\mathcal{L}_{\text{Con}} = -\frac{1}{N}\Big[\sum_i \log\frac{\exp(x_i^\top y_i/\sigma)}{\sum_j \exp(x_i^\top y_j/\sigma)} + \sum_i \log\frac{\exp(y_i^\top x_i/\sigma)}{\sum_j \exp(y_i^\top x_j/\sigma)}\Big]$$

with $N=64$ (local batch), $\sigma$ a learned temperature. Trained 20 epochs on a single A100 80GB.

5. **Optional task finetuning.** Align (10 epochs), Decompose (5), PredEx (10 with 10-fold CV) finetune on the target slide's paired Visium data. Annotate and Retrieve are zero-shot.
6. **Loki Align.** Encode each spot's transcriptome (text encoder) and each H&E patch (image encoder) to 768-d; concatenate the first two PCA components with (x, y); run a modified Coherent Point Drift (CPD) where the M-step updates coordinates only, with a homography post-fit to avoid runaway warping.
7. **Loki Annotate.** Zero-shot. Encode an H&E patch and either a bulk RNA-seq sample (gene-sentence) or a marker-gene list (e.g., `"TP53 EPCAM KRAS ... DSP"` for tumor). Argmax cosine similarity gives the prediction. For multimodal annotation, normalized Loki + PLIP similarities are averaged.
8. **Loki Decompose.** Encode scRNA-seq cells via the text encoder, ST spots / H&E patches via the appropriate encoder, then run **Tangram's non-convex optimization on OmiCLIP embeddings** (not raw expression) to learn a cell→spot mapping. Cosine-distance loss; a non-maximum-suppression step keeps the argmax cell type per spot.
9. **Loki Retrieve.** Argmax of image-to-transcriptome cosine similarity over an ST-bank reference.
10. **Loki PredEx.** 10-fold CV. For a test spot $i$, predicted expression is the **similarity-weighted average of training-set expression**: $X_i = \sum_{j \in T} w_{ij} X_j / \sum_j w_{ij}$. Note the paper itself flags this is retrieval-based, not generative — OmiCLIP cannot synthesize a transcriptome, only weighted-average over similar ones.

## Experimental Results

![Loki Align — workflow and quantitative results across simulations, small intestine, and ovarian carcinosarcoma](/assets/images/paper/omiclip_loki/page_004.png)
*Figure 3 — Loki Align workflow and quantitative comparison against PASTE / GPSA / CAST on simulation, small-intestine (n=8 adjacent sections), and ovarian-carcinosarcoma datasets.*

![Loki Align — 3D stacking and qualitative side views](/assets/images/paper/omiclip_loki/page_005.png)
*Figure 4 — Side-view 3D stacking of 8 adjacent small-intestine sections. Loki ST-to-ST and image-to-ST preserve morphology, while PASTE rotates 3/7 source sections and GPSA introduces visible distortions.*

![Loki Annotate with bulk RNA-seq as a "text query"](/assets/images/paper/omiclip_loki/page_006.png)
*Figure 5 — Loki Annotate similarity heatmaps for breast cancer (tumor), heart failure (fibroblast), and normal breast (adipose) using bulk RNA-seq as a gene-sentence query, matching pathologist regions and CLAM attention.*

![Loki Decompose — cell-type decomposition vs. 11 baselines on TNBC and CRC](/assets/images/paper/omiclip_loki/page_008.png)
*Figure 6 — Loki Decompose ranks #1 (ST-mode, impact 1.32) and #2 (image-mode, 1.11) of 12 methods on the in-house TNBC sample; CRC WSI decomposition matches pathologist annotation and CLAM attention.*

### Headline quantitative comparisons

| Task | Dataset | Metric | OmiCLIP / Loki | Best non-Loki | Other baselines |
|---|---|---|---|---|---|
| ST-to-ST alignment | Ovarian carcinosarcoma (2 sections) | median PCC | **0.88** | 0.71 (CAST) | 0.43 (GPSA), 0.26 (PASTE) |
| Image-to-ST alignment | Ovarian carcinosarcoma | median PCC | **0.86** | n/a (only Loki) | — |
| ST-to-ST alignment | Ovarian carcinosarcoma | median Kendall's τ | **0.21** | 0.09 (CAST) | 0.04 (GPSA), 0.03 (PASTE) |
| ST-to-ST alignment | Small intestine (8 adj.) | median PCC range | **0.62-0.83** | CPD-PCA lower (P<0.001) | PASTE -0.25 to 0.39, GPSA 0.27-0.56 |
| Image-to-ST alignment | Small intestine | median PCC range | **0.67-0.80** | — | — |
| Visium↔Xenium alignment | Breast cancer | mean distance | **0.08 mm** | — | — |
| Zero-shot annotation | CRC7K | weighted F1 | **0.59** (Loki), **0.72** (Loki+PLIP) | 0.50 (PLIP) | 0.03-0.34 (OpenAI CLIP) |
| Zero-shot annotation | WSSS4LUAD | weighted F1 | **0.79** (Loki), **0.83** (Loki+PLIP) | 0.78 (PLIP) | OpenAI CLIP much lower |
| Zero-shot annotation | PatchCamelyon | weighted F1 | **0.60** (Loki), **0.62** (Loki+PLIP) | 0.58 (PLIP) | — |
| Zero-shot annotation | LC25000 | weighted F1 | **0.96** (Loki), **0.97** (Loki+PLIP) | 0.93 (PLIP) | — |
| Cell-type decomposition | In-house TNBC (Xenium GT) | impact score | **1.32 / 1.11** (Loki-ST / image) | 0.87 (RCTD) | 0.28 (CARD), 0.21 (Tangram), 0.18 (scGPT), 0.00 (Spatial Seurat), -0.20 (scFoundation), -0.26 (Geneformer), -0.61 (CytoSPACE), -1.06 (Cell2location), -1.82 (spatialDWLS) |
| Image→transcriptome retrieval | Brain (val) | Recall@10% | **0.227** | 0.103 (OpenAI CLIP) | 0.095 (PLIP), 0.101 (random) |
| Image→transcriptome retrieval | Heart (val) | Recall@10% | **0.291** | 0.104 | 0.103, 0.098 |
| Image→transcriptome retrieval | Kidney (val) | Recall@10% | **0.297** | 0.100 | 0.097, 0.101 |
| Image→transcriptome retrieval | Breast (val) | Recall@10% | **0.240** | 0.100 | 0.096, 0.094 |
| Image→transcriptome retrieval | Test (4 external) | Recall@10% | **0.208** | 0.075 | 0.067, 0.067 |
| ST gene expression prediction | 39 heart samples | best MSE in N samples | **28 / 39** | shared among Hist2ST / HisToGene / BLEEP / mclSTExp | — |
| ST gene expression prediction | 39 heart samples | best PCC in N samples | **16 / 39** | shared among 4 baselines | — |

### Ablations and robustness

- **Contrastive learning lifts encoder quality.** Calinski-Harabasz cluster scores improve significantly (P < 0.05-0.001) on **both** image and transcriptome embeddings after CL, and OmiCLIP image embeddings outperform UNI and Prov-GigaPath on per-organ clustering. This is the strongest single argument that CL improves over the dominant single-modal pathology FMs.
- **Training strategy.** Pretraining + finetuning > pretraining-only > train-from-scratch on alignment (median PCC 0.86 vs. 0.85 vs. 0.53) and decomposition (SSIM 0.30 vs. 0.13 vs. 0.0007). Pretraining matters; finetuning matters more for decomposition than for alignment.
- **CPD vs. PCA-CPD.** Running the same CPD registration on PCA of raw ST instead of OmiCLIP embeddings drops performance significantly (P < 0.001, Wilcoxon) — i.e., the OmiCLIP representation, not the CPD trick, carries the alignment result.
- **Image-noise robustness.** Gaussian noise added to H&E; Loki cosine similarity drops far less than PLIP / OpenAI CLIP. n=10 per condition (small).
- **Sequencing-depth robustness.** Downsampling ST from high→medium→low (UMI groups 11,792 / 4,512 / 615). Loki similarity is stable across all transitions (n=500).
- **Modality fusion.** Averaging image + transcriptome embeddings for alignment **does not** beat single modality — a negative result the authors honestly report.
- **Foundation-model swap.** Replacing OmiCLIP with scGPT / scFoundation / Geneformer inside the same Tangram-on-embeddings decomposition pipeline drops impact rank from #1 to 6 / 8 / 9 of 12. This isolates the contribution of visual-omics joint training over pure transcriptomic FMs.

### Qualitative findings

- On the 8-section small-intestine stack, Loki ST-to-ST and image-to-ST stack into a coherent 3D body, while PASTE rotates 3/7 sections and GPSA introduces visible distortions.
- On colorectal-cancer WSI (20 mm x 13 mm), Loki Decompose image-mode produces tumor / fibroblast / smooth-muscle / intestinal-epithelial density maps matching pathologist annotations and CLAM attention.
- On mouse-brain cortex, Loki recovers the L1 (VLMCs / astrocytes) → L2/3 → L4/5 → L6 → WM (oligodendrocytes) laminar pattern from H&E + scRNA-seq alone.

## Limitations

**Authors admit (Discussion):**

- Pretraining scale (2.2M pairs) is small versus billion-scale general VLM corpora.
- OmiCLIP is contrastive, **not generative** — it can only retrieve / weighted-average, not synthesize a transcriptomic profile. WSI-scale ST reconstruction would need a generative head on top.
- ST-bank covers 32 organs but rare conditions are underrepresented; finetuning is recommended for out-of-distribution use.
- Dual-modality fusion may suppress gains via modality dominance — confirmed by their own averaging experiment.
- The transcriptomic encoder is trained on Visium pseudo-bulk and only post-hoc evaluated on bulk RNA-seq / scRNA-seq.

**Not addressed (auditor's observations):**

- **No confidence intervals on most headline metrics.** Box plots show spot-wise, not run-to-run variance. Single-seed training throughout.
- **No clinical / prospective evaluation.** All datasets are public or in-house lab samples; no FDA-style validation, no inter-pathologist agreement, no out-of-site test.
- **Pretraining is 10x Visium only.** Slide-seq, Stereo-seq, Visium HD, and Xenium are handled at inference via pseudo-Visium re-binning, not a native multi-platform foundation model.
- **The "gene-sentence" is just a top-50 list with whitespace separation.** No ablation on (a) list length (why 50, not 20 or 200?), (b) ordering (rank vs. random), (c) separator choice, (d) housekeeping-gene removal. Obvious knobs, not turned.
- **Marker-gene annotation depends on hand-picked marker lists** (Supplementary Table 3). The strong F1 numbers partly reflect a well-chosen marker set; no sensitivity analysis to noisy markers.
- **Retrieval evaluation uses a proxy.** Ground-truth transcriptomes do not exist for the query images, so "image-to-transcriptome retrieval" is scored via image-image similarity in the retrieved tile, not direct transcriptome match.
- **No natural-language bridge.** Loki+PLIP fusion is the closest gesture toward English. A real triple-modal (image + gene-sentence + natural language) model is future work — and the precise gap a SpotWhisperer-style pipeline targets.
- **Text encoder is LAION-5B-initialized.** No ablation against random init, so it is unclear how much the natural-language prior helps vs. hurts on gene-symbol input.
- **One in-house TNBC sample (n=1) carries most of the decomposition headline.**

## Why It Matters for Medical AI

OmiCLIP is the cleanest demonstration to date that **the "text" side of a CLIP-style model does not need to be natural language**. Ranked gene-symbol strings, fed through a standard text encoder, are sufficient to produce a joint embedding that (a) beats pure pathology FMs (UNI, Prov-GigaPath) on per-organ clustering and (b) beats pure transcriptomic FMs (scGPT, Geneformer, scFoundation) when dropped into the same Tangram decomposition pipeline. That is a substantive conceptual win for any team building image + omics models.

For clinical workflows, the practical promise is an H&E-only inference path that can decompose cell types, predict gene expression, or align to a 3D stack — reducing the need for $1k-per-slide Visium runs at deployment. The caveats are equally important: pretraining is Visium-only, the headline cell-type decomposition number is from a single TNBC sample, gene-expression prediction is retrieval-based and wins PCC on only 41% of samples, and no prospective or multi-site clinical validation exists. The right framing is that OmiCLIP / Loki is a research-grade visual-omics foundation with a clear path to medical impact, not a clinical tool today.

It is also the natural baseline for a follow-up that swaps the gene-symbol "sentence" for true natural language via an LLM bridge — an architectural direction the authors themselves point to in the Discussion.

## References

- **Paper:** Chen, W.*, Zhang, P.*, Tran, T. N., Xiao, Y., Li, S., Shah, V. V., et al. *A visual-omics foundation model to bridge histopathology with spatial transcriptomics.* **Nature Methods** 22, 1568-1582 (July 2025). DOI: [10.1038/s41592-025-02707-1](https://doi.org/10.1038/s41592-025-02707-1).
- **Code:** [github.com/GuangyuWangLab2021/Loki](https://github.com/GuangyuWangLab2021/Loki/) — Loki platform implementation and ST-bank database.
- **Weights:** Hugging Face — `WangGuangyuLab/Loki`.
- **License:** CC-BY 4.0 (Open Access).
- **Related (gene-sentence lineage):** Cell2Sentence (Levine et al., 2023); GenePT (Chen & Zou, 2024).
- **Related (single-cell foundation models):** scGPT (Cui et al., 2024, *Nat. Methods*); Geneformer (Theodoris et al., 2023, *Nature*); scFoundation (Hao et al., 2024, *Nat. Methods*).
- **Related (pathology foundation models):** PLIP (Huang et al., 2023, *Nat. Med.*); CONCH (Lu et al., 2024, *Nat. Med.*); UNI (Chen et al., 2024, *Nat. Med.*); Prov-GigaPath (Xu et al., 2024, *Nature*).
- **Related (ST gene-expression prediction):** Hist2ST, HisToGene, BLEEP, mclSTExp.
- **Related (decomposition baselines):** Tangram, RCTD, CARD, Cell2location, CytoSPACE, spatialDWLS, Spatial Seurat.
