---
title: "CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation"
excerpt: "Hand-designed 3-level anatomical graph (34 fine -> 8 coarse -> 1 global) on frozen 3D CT features feeds a LoRA-tuned LLaMA2-7B and lifts CT-RATE report F1 by +7.9 absolute over Reg2RG (0.296 vs 0.217)."
categories:
  - Paper
  - CT-Report-Generation
permalink: /paper/ct-graph/
tags:
  - CT-GRAPH
  - CT-Report-Generation
  - Graph-Attention-Network
  - TotalSegmentator
  - LLaMA2
  - LoRA
  - CT-RATE
  - 3D-CT
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-26
last_modified_at: 2026-05-26
---

## TL;DR

- CT-GRAPH freezes a 3D CT encoder, pools features per TotalSegmentator mask, and arranges them into a **hand-designed 3-level anatomical graph** (34 fine organs -> 8 coarse systems {bones, lungs, abdomen, mediastinum, heart, esophagus, trachea, thyroid} -> 1 global node) processed by a hierarchical GAT before a LoRA-tuned LLaMA2-7B writes the report.
- On CT-RATE (n=1,505 test), it posts **macro-F1 = 0.296 vs Reg2RG 0.217 (+7.9 absolute)** — the headline claim. NLG gains are marginal (BLEU-1 actually loses to CT2Rep by 0.002), so the win is concentrated in clinical-efficacy metrics.
- The cleanest evidence is the **graph-construction ablation**: random graph 0.216 -> single-level (fine + global) 0.253 -> full hierarchy 0.296, isolating both anatomical edges and the coarse layer with backbone/LLM/schedule held fixed.

## Motivation

3D CT report generation is bottlenecked by two structural problems. First, memory: end-to-end 3D pipelines either downsample aggressively (M3D, RadFM) or fall back to 2D-slice processing (Read-Like-a-Radiologist). Second, granularity: methods that survive end-to-end (CT2Rep, Dia-LLaMA, Argus) rely on **global** volumetric tokens that wash out organ-level detail. Region-based predecessors (Reg2RG, VividMed) crop or pool whole organs but never (i) go below organ granularity (e.g., individual lobes), nor (ii) explicitly model anatomical part-whole relationships. CT-GRAPH targets both gaps by combining a frozen multi-scale encoder, fine-grained mask pooling, and a hierarchy-encoded GAT — keeping the LLM tunable only through LoRA and the visual encoder fully frozen.

## Core Innovation

- **Anatomy is the graph, and the graph is fixed.** Topology is not learned. TotalSegmentator's 104 anatomy classes are reduced to 34 fine nodes -> 8 coarse system nodes -> 1 global node, with edges only fine->coarse->global (a tree / DAG, no fine<->fine or fine<->global skip).
- **Multi-layer mask pooling.** Per-organ features are average-pooled across all encoder depths (nearest-neighbor mask resize per layer, channels concatenated), so a fine node carries both shallow texture and deep semantic features for the same voxel set.
- **Skip-connected hierarchical GAT.** Fine-to-coarse and coarse-to-global message passing with multi-head attention; a skip connection preserves the original global feature so the LLM still sees an unaggregated volumetric summary alongside the structured node embeddings.
- **Two-stage, mostly frozen.** Encoder fully frozen, features pre-extracted; LLaMA2-7B trained only via LoRA (rank 32, alpha 32). Six epochs on a single A100, ~10 h wall time.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset | Strength |
|---|---|---|---|---|
| C1 | CT-GRAPH yields +7.9 absolute F1 over prior SOTA on CT report generation | Table 6: F1 0.296 vs Reg2RG 0.217 | CT-RATE test (n=1,505) | ⭐⭐ — true on the chosen baseline set, but **M3D / RadFM / Argus / Dia-LLaMA / VividMed / CT-CHAT / Merlin / MedRegion-CT are named in related work and absent from the table**; single run, no variance |
| C2 | Hierarchical anatomical graph beats unstructured / random baselines | Table 4: Random 0.216, Single-Level 0.253, Full 0.296 | CT-RATE | ⭐⭐⭐ — same encoder, same LLM, same schedule; conclusion well-isolated |
| C3 | Fine-grained mask pooling captures localized pathologies better than global | Table 3 (linear probing) + Table 7 (per-pathology) | CT-RATE | ⭐⭐⭐ — global pooling collapses to near-zero recall/F1 across all 6 encoders; fine -> 0.34-0.43 |
| C4 | Pretraining scale alone does not dictate downstream performance | Table 5 / Fig. 2: TransVW (623 scans) ~ VoCo-160k (160k scans) | CT-RATE | ⭐⭐ — direction is consistent but n=6 encoders confounded by architecture, depth, channel dim, and objective |
| C5 | Frozen-feature two-stage pipeline is efficient | "10 h on a single A100 for 6 epochs, batch 8, pre-extracted features" | — | ⭐ — qualitative; no head-to-head training-time / memory comparison vs CT2Rep / Reg2RG |
| C6 | Method generalizes across encoders | Fig. 2 / Table 5: CT-GRAPH >= Global+Local for all 6 backbones | CT-RATE | ⭐⭐ — same-dataset only; no cross-dataset transfer |
| C7 | Coarse-level hierarchy is meaningful (not just fine + global) | Table 4: 0.253 -> 0.296 | CT-RATE | ⭐⭐ — single-run +0.043 F1 delta; consistent across CE metrics |
| C8 | Region-wise modeling helps localized findings (consolidation, pericardial effusion) | Table 7 per-pathology | CT-RATE | ⭐⭐ — direction right; pericardial-effusion F1 absolute is still only 0.136 |
| C9 | Hierarchical graph confers interpretability | Implied throughout | — | ⭐ — **no interpretability analysis presented**: no attention-weight visualizations, no node-importance studies, no failure-mode mapping. The claim is **structural, not empirical** |

**Honest read.** The graph-construction ablation (C2) and the global-vs-fine pooling story (C3) are the genuinely well-controlled results — same backbone, same LLM, same schedule, only the structure changes. The headline SOTA claim (C1) is real on the three baselines shown but the comparison set is conspicuously narrow given how many 3D CT-RG models the related-work section names; without M3D / RadFM / Argus / Dia-LLaMA / VividMed / CT-CHAT / Merlin / MedRegion-CT in Table 6, "state of the art" should be read as "best of CT2Rep and Reg2RG re-implemented under matched preprocessing." Crucially, **every reported number is a single run** — no seeds, no confidence intervals, no significance test on a 1,505-sample test set, which makes the +0.043 single-level-vs-full delta the kind of thing variance reporting could erode. The interpretability story (C9) is asserted by architecture, never demonstrated.

## Method & Architecture

![CT-GRAPH pipeline overview](/assets/images/paper/ct-graph/fig_p003_01.png)
*Figure 1: Pipeline overview — frozen 3D encoder + TotalSegmentator masks -> multi-scale mask pooling -> hierarchical GAT -> LoRA LLaMA2-7B decoder.*

![Fixed anatomical hierarchy](/assets/images/paper/ct-graph/fig_p003_25.png)
*Figure 2: Fixed anatomical hierarchy — fine organs (lung lobes, spleen, liver, ...) -> coarse systems (lungs, abdomen, mediastinum, bones, heart, esophagus, trachea, thyroid) -> global node.*

**Step by step.**

1. **Input.** CT volume $X \in \mathbb{R}^{H\times W\times D}$ resized to $512\times512\times256$; TotalSegmentator produces a multi-label mask $M \in \{0,\ldots,K\}^{H\times W\times D}$ with $K{=}34$ fine-grained anatomical labels.
2. **Frozen 3D encoder.** One of {SwinUNETR, VoCo-10k, VoCo-160k, Vox2Vec, TransVW, CT-FM}. Sliding-window inference (overlap 0.25) yields hierarchical feature maps $F^l$ at $L$ depths. The final CT-GRAPH uses **VoCo-160k**.
3. **Multi-layer mask pooling.** For each fine organ $k$ and each layer $l$, nearest-neighbor resize $M$ to $(H^l,W^l,D^l)$ and average-pool features over the voxel set $\mathcal{V}^l_k$:

$$f^l_k = \frac{1}{|\mathcal{V}^l_k|}\sum_{(i,j,t)\in\mathcal{V}^l_k} F^l[i,j,t,:].$$

Concatenate across layers: $f_k = [f^1_k;\ldots;f^L_k]$ ("Layer Fusion").

4. **Coarse-node features.** For each coarse system $c$, take the **union** of its fine-level masks and mask-pool the same way -> $f_c$. Orphan structures (e.g., esophagus) act as coarse nodes themselves and link directly to the global node.
5. **Global-node feature.** Adaptive-average-pool the deepest layer feature map to $(4,4,2)$, flatten through an MLP -> $h_g$.
6. **Node embeddings.** Each $f_k$ / $f_c$ / $f_g$ passes through a per-level 2-layer MLP with LayerNorm before attention.
7. **Hierarchical edges (fixed, non-learned).** $E_{fc} = \{(v_f,v_c) : v_c\in C, v_f\in F_{v_c}\}$, $E_{cg} = \{(v_c,v_g) : v_c\in C\}$. No fine<->fine or fine<->global edges; pure tree topology.
8. **GAT message passing (multi-head).** Fine-to-coarse update with self-loops:

$$h'_c = \sum_{v_f\in F_{v_c}\cup\{v_c\}} \alpha_{v_f v_c}\, W h_f,\quad \alpha_{v_f v_c} = \frac{\exp(\mathrm{LReLU}(a^\top[Wh_f\|Wh_c]))}{\sum_{v\in F_{v_c}\cup\{v_c\}} \exp(\mathrm{LReLU}(a^\top[Wh_v\|Wh_c]))}.$$

Coarse-to-global is analogous, with a **skip connection** preserving the original global feature. Multi-head outputs concatenated.

9. **LLM decoding.** Project each updated node embedding (global, coarse, fine) into LLaMA2-7B's latent space, concatenate with the embedded prompt *"Generate a medical report based on the visual information of the given CT image."*, and train with next-token cross-entropy.
10. **Training.** Frozen visual encoder + frozen LLM backbone, **LoRA** rank 32 / alpha 32 / dropout 0.1 on LLaMA2-7B (embedding + output head also trainable). AdamW lr $5\times10^{-5}$, batch 8, **6 epochs**, single A100, ~10 h wall time using pre-extracted features.

## Experimental Results

### Main comparison (Table 6, CT-RATE test, n=1,505)

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| CT2Rep | 0.4872 | 0.3712 | 0.2959 | 0.2390 | 0.2992 | 0.4132 | 0.230 | 0.140 | 0.141 |
| CT2Rep (w/ LLaMA) | 0.4875 | 0.3752 | 0.2989 | 0.2426 | 0.3027 | 0.4212 | 0.317 | 0.172 | 0.214 |
| Reg2RG | 0.4728 | 0.3681 | 0.2975 | 0.2437 | 0.2725 | 0.4005 | **0.457** | 0.156 | 0.217 |
| **CT-GRAPH (ours)** | **0.4850** | **0.3765** | **0.3024** | **0.2467** | **0.3126** | **0.4205** | 0.396 | **0.248** | **0.296** |

Headline +7.9 F1 = 0.296 - 0.217 (vs Reg2RG). NLG gains are marginal — BLEU-1 actually loses to CT2Rep by 0.002 — so the win is **CE-metric concentrated**. Note the recall jump (0.156 -> 0.248) at the cost of precision (0.457 -> 0.396): the model finds more findings but with more false positives than Reg2RG.

### Graph-construction ablation (Table 4, all VoCo-160k)

| Variant | BLEU-4 | ROUGE-L | METEOR | P | R | F1 |
|---|---|---|---|---|---|---|
| Global (compressed) | 0.2370 | 0.3073 | 0.4088 | 0.282 | 0.133 | 0.171 |
| Global (multiple tokens) | 0.2333 | 0.3044 | 0.4014 | 0.227 | 0.081 | 0.111 |
| Local only | 0.2365 | 0.3053 | 0.4084 | 0.338 | 0.182 | 0.230 |
| Global + Local | 0.2446 | 0.3089 | 0.4159 | 0.343 | 0.201 | 0.245 |
| Random Graph | 0.2384 | 0.3103 | 0.4092 | 0.328 | 0.172 | 0.216 |
| Single-Level Graph (no coarse) | 0.2360 | 0.3060 | 0.4097 | 0.385 | 0.202 | 0.253 |
| **CT-GRAPH (full)** | **0.2467** | **0.3126** | **0.4205** | **0.396** | **0.248** | **0.296** |

This is the paper's strongest evidence. Two clean answers, same encoder / LLM / schedule throughout: (a) anatomical edges matter (Random 0.216 vs Full 0.296), and (b) the coarse hierarchy contributes a smaller but consistent +0.043 F1 over a flat fine->global graph (0.253 -> 0.296).

### Encoder ablation (Table 5, full CT-GRAPH pipeline, varying frozen backbone)

![Encoder ablation across feature configurations](/assets/images/paper/ct-graph/fig_p006_01.png)
*Figure 3: F1 across six frozen 3D encoders for three feature configurations (Global, Global+Local, CT-GRAPH); CT-GRAPH matches or beats Global+Local on every backbone.*

VoCo-160k tops F1 (0.296), TransVW close second (0.292) despite 623-scan pretraining vs 160k, then CT-FM (0.271), VoCo-10k (0.264), SwinUNETR (0.249), Vox2Vec (0.238). NLG differences across encoders are within ~0.01 BLEU/METEOR; CE metrics swing by ~0.06 F1. Conclusion (correctly stated by the authors): encoder choice impacts factual content more than fluency, and pretraining scale alone does not predict downstream quality.

### Per-pathology (Table 7)

CT-GRAPH wins **15/18 pathologies**. Strongest absolute gains over Reg2RG: **medical material** (+0.271, 0.013 -> 0.284), **arterial wall calcification** (+0.237), **consolidation** (+0.129), **pericardial effusion** (+0.118, but absolute is only 0.136). Reg2RG retains the lead on **cardiomegaly** (0.390 vs 0.208 — a notable loss the paper does not really address), **lung opacity** (0.534 vs 0.449), and **pleural effusion** (0.574 vs 0.546).

### Pooling-level study (Table 3)

On linear-probing pathology classification, **global pooling collapses to near-zero recall/F1** across all six encoders; fine-level pooling wins F1 (e.g., 0.434 with Vox2Vec, 0.431 with CT-FM); coarse-level pooling tends to win precision. This is the empirical scaffolding the whole hierarchical-graph story rests on.

## Limitations

**Authors acknowledge** (mostly implicitly):
- Pleural and breast regions are absent / merged with lungs in the Reg2RG re-implementation, which may disadvantage that baseline.
- Encoder choice swings CE metrics by ~0.06 F1 (the method is encoder-sensitive).
- Per-pathology gains are uneven; wins are highlighted, losses (cardiomegaly) not really discussed.

**Not addressed:**
- **No variance / multi-seed.** All Tables 2-7 appear to be single training runs. On an n=1,505 test set the +0.043 F1 single-level-vs-full delta is exactly the magnitude that confidence intervals could compromise.
- **No comparison to M3D, RadFM, Argus, Dia-LLaMA, VividMed, CT-CHAT, Merlin, or MedRegion-CT** despite all being named in related work; the "+7.9 over SOTA" tagline rests on a 3-model comparison set (CT2Rep, CT2Rep+LLaMA, Reg2RG).
- **No external validation** (RadGraph entity F1, GREEN / RadCliQ, or any non-CT-RATE cohort). The single CE metric is itself extracted by another learned model (CT-CLIP), with its own error budget that compounds into the headline number.
- **No human / radiologist evaluation.**
- **Hierarchy is hand-designed.** No experiment on learned hierarchies, deeper decompositions (4+ levels), or robustness when TotalSegmentator mis-segments.
- **No robustness analysis** to mask noise (TotalSegmentator failures, partial-volume CTs, motion artifacts).
- **No attention-weight or message-passing visualization** to substantiate the implicit interpretability claim — the interpretability story is asserted by architecture, not by evidence.
- **Computational reporting is uneven** — wall time and "frozen features" are mentioned, but no fair training-time / inference-time / memory comparison to CT2Rep or Reg2RG.
- **Generalization beyond chest CT** (abdominal, head/neck) is entirely open; the current hierarchy is chest-anatomy-specific.

## Why It Matters for Medical AI

CT-GRAPH is a clean instance of a recurring lesson in medical-imaging report generation: when the target metric is **clinical efficacy** (entity-level F1 against extracted findings), structural priors that align tokens with anatomy beat both global pooling and learned-from-scratch attention. The hierarchical-graph architecture is portable in principle — anywhere a high-quality multi-label segmentation atlas exists (TotalSegmentator for body CT, brain atlases, dental CBCT), the same recipe (frozen encoder + mask-pool + fixed anatomical GAT + LoRA LLM) should transfer. Two caveats temper that enthusiasm: (i) the +7.9 F1 result is **comparison-set narrow** — without M3D / RadFM / Argus / Dia-LLaMA / VividMed / CT-CHAT in Table 6, "SOTA" is a weaker claim than the abstract makes it sound; (ii) the interpretability story that justifies anatomy-aligned graphs to clinicians is **structurally asserted but empirically absent** — no attention visualization, no node-importance study. For the field, the actionable takeaway is the **ablation cascade** (random graph -> single-level -> full hierarchy) — that is the experimental design future hierarchical-medical-VLM papers should copy and that future evaluations of CT-GRAPH itself should extend with multi-seed variance and a broader baseline set.

## References

- **Paper**: Kalisch, Hörst, Kleesiek, Herrmann, Seibold. *CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation*. arXiv:2508.05375, August 2025. [arXiv](https://arxiv.org/abs/2508.05375)
- **Code**: [github.com/hakal104/CT-GRAPH](https://github.com/hakal104/CT-GRAPH)
- **Dataset**: CT-RATE (Hamamci et al., 2024); RadGenome-Chest CT (Zhang et al., 2024) for region-wise reports.
- **Segmentation**: TotalSegmentator — 104 anatomy classes from CT.
- **Baselines compared**: CT2Rep (Hamamci et al., 2024), Reg2RG (Bai et al.).
- **Baselines named in related work but not compared**: M3D, RadFM, Argus, Dia-LLaMA, VividMed, CT-CHAT, Merlin, MedRegion-CT.
- **Frozen encoders evaluated**: SwinUNETR, VoCo-10k, VoCo-160k, Vox2Vec, TransVW, CT-FM.
- **LLM backbone**: LLaMA2-7B with LoRA (rank 32, alpha 32, dropout 0.1).
