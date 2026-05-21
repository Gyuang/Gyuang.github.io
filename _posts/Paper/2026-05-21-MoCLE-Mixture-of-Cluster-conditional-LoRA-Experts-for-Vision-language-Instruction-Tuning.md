---
title: "MoCLE: Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning"
excerpt: "Cluster instructions with k-means on MiniLM, route to per-cluster LoRA experts plus an always-on universal expert — 61.8 avg held-out vs 60.8 dataset routing / 58.1 token-MoLE on InstructBLIP-7B, but the medical cross-domain claim is overextrapolated."
categories:
  - Paper
  - VLM-Alignment
  - LLM
permalink: /paper/mocle/
tags:
  - MoCLE
  - Mixture of Experts
  - LoRA
  - Cluster-Conditional Routing
  - Vision-Language Instruction Tuning
  - InstructBLIP
  - LLaVA-1.5
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-21
last_modified_at: 2026-05-21
---

## TL;DR

- Vision-language instruction tuning has a real **negative-transfer** problem (Figure 1: task-only experts beat the full-mixture model on 5/7 held-out tasks). MoCLE attacks it with **k-means clustering of templated text instructions** using a frozen `all-MiniLM-L6-v2` encoder, then routes each input to a **per-cluster LoRA expert** via top-1 gating on the cluster id, while a separate **universal LoRA expert** runs always-on with weight $1-G_{\max}$.
- The headline architectural win: **cluster routing 61.8 avg > dataset routing 60.8 > sentence-MoLE 59.7 > token-MoLE (LLaVA-MoLE) 58.1** on InstructBLIP-7B held-out (Table 7), and **Cluster-MoE rank 8 (16.78M params) beats flat LoRA rank 64 (33.55M)** — so the gain is genuinely from routing structure, not capacity.
- The honest read: held-out gains of +2.9 to +3.9 on VSR/IconQA/TextVQA come **mostly from the universal expert** (Table 5 c vs d), "unseen" tasks share instruction style with training, **medical results regress catastrophically** (SLAKE-Closed 85.58 → 35.21, PathVQA Open column blank), and the paper ships **no variance, no seeds, no significance tests, no compute/latency numbers, and no explanation for K=64 (InstructBLIP) vs K=4 (LLaVA)**.

## Motivation

Instruction tuning is the default recipe for turning LVLMs (InstructBLIP, LLaVA, MiniGPT-4) into generalist assistants, but Figure 1 of this paper is a damning preliminary result: on 5 of 7 held-out tasks, an InstructBLIP fine-tuned on **only** the "vqa" subset or **only** the "cap" subset outperforms the model trained on the full mixture. The naive remedy — hand-define one expert per task family — fails because (a) the taxonomy ("vqa"/"cap") does not scale to hundreds of instruction sources, and (b) hard task routing kills zero-shot generalization since you cannot pick an expert for a novel task and some novel tasks (VSR, TextVQA) benefit from combining skills.

MoCLE positions itself as the automatic, data-driven middle path: cluster instructions, attach a small LoRA expert to each cluster, and keep an always-on universal LoRA so the model can still generalize. The medical-AI angle lands in §4.2, where the LLaVA-1.5 experiments mix VQA-RAD, SLAKE, PathVQA with geometric (Geo170K) and natural-image data — a setup with much harsher cross-domain conflict than the standard InstructBLIP held-in/held-out split.

![MoCLE motivation — negative transfer](/assets/images/paper/mocle/page_001.png)
*Figure 1: Page 1 of the paper. The motivating bar chart shows that task-specific InstructBLIP variants (vqa-only, cap-only) outperform the all-data model on 5 of 7 held-out tasks — concrete evidence of negative transfer in VL instruction tuning.*

## Core Innovation

MoCLE introduces a routing key that is neither token-level (LLaVA-MoLE) nor sentence-level (Octavius) but **cluster-level**: every training instruction is k-means-clustered by its MiniLM embedding, and the resulting cluster id becomes the input to a small per-layer linear gate. Three design choices distinguish the method:

1. **Visual tokens are excluded from clustering.** Only the templated text instruction is encoded, so the cluster signal is purely textual.
2. **Cluster assignments are frozen after k-means.** The cluster *embedding* $C[\cdot]$ is trainable (initialized at the centroid), but the hard cluster id is fixed for the entire training run.
3. **A separate universal LoRA expert is always active**, weighted by $1-G_{\max}$. As the gate sharpens ($\tau\to 0$), the universal expert's contribution shrinks toward 0; as it flattens, the universal expert dominates. So $\tau$ is mechanically a specialization-vs-generalization knob.

The contribution that does the most architectural work is #3: ablations (Table 5 c vs d) show that the held-out gains on VSR / IconQA / TextVQA collapse without the universal expert. The cluster routing per se gives a smaller, more uniform improvement.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Negative transfer exists in vision-language instruction tuning. | Figure 1: task-only experts (vqa-only, cap-only) beat the full-data model on 5/7 held-out InstructBLIP tasks. | ⭐⭐⭐ Clean, reproducible, well-designed preliminary study. |
| C2 | Cluster-conditioned routing beats token / sentence / dataset routing. | Table 7: Cluster 61.8 avg > Dataset 60.8 > Sentence-MoLE 59.7 > Token-MoLE 58.1. Figure 6 routing heatmap shows MoCLE's clean task-aligned routing vs. Sentence-MoLE's uniform routing. | ⭐⭐ Consistent gains across 6 held-out tasks, but **single backbone, single seed, no variance**. Largest differentials on VSR (58.9 vs 49.0 for token) and TextVQA (54.9 vs 46.6). |
| C3 | The universal expert restores zero-shot generalization. | Table 5 row c (no universal) vs row d (full): VSR 64.7 → 58.9, TextVQA 57.1 → 54.9. Table 6: top-1+universal 63.3 beats top-2 60.4. | ⭐⭐⭐ Two independent ablations converge; the mechanism is also clearly tied to the $1-G_{\max}$ math. **The held-out gains are mostly the universal expert, not the cluster routing.** |
| C4 | Cluster MoE > naively scaling LoRA rank. | Table 5: Cluster-MoE rank 8 (16.78M params) avg 61.8 > flat LoRA rank 64 (33.55M params) avg 60.7. | ⭐⭐ Controlled and convincing on InstructBLIP; **not replicated on LLaVA-1.5**. |
| C5 | MoCLE improves InstructBLIP across all 17 evaluated datasets. | Tables 2–3: every $\Delta$ positive; max gains VSR +3.9, POPE adv-F1 +4.5, A-OKVQA MC +4.4, TextVQA +3.2, IconQA +2.9. | ⭐⭐ Direction is uniformly right but **no variance, no seeds, no significance tests**. Deltas under ~1.5 absolute points (GQA +0.7, HM +0.3, VisualDialog +0.6) should be read as suggestive only. |
| C6 | Method generalizes to unseen tasks via the universal expert. | Held-out tasks (VSR, IconQA, ScienceQA, ...) in Tables 2–3. | ⭐⭐ "Unseen" here means held-out datasets with **similar instruction styles**; the paper does not test paraphrased instructions, novel templates, or out-of-task-family instructions. |
| C7 | MoCLE alleviates multi-domain task conflict including medical. | Table 4 LLaVA-1.5: GeoQA 46.89 → 53.59, VQA-RAD Open 77.94 → 81.98. | ⭐ **Overextrapolated.** **SLAKE-Closed regresses 85.58 (med-only) → 35.21 (MoCLE all)** and stays there — a 50-point gap MoCLE does not fix. **PathVQA Open is blank** for the two key rows with no explanation. The medical claim rests on VQA-RAD-Open improvements only. |
| C8 | K=64, 4 task experts, τ=0.05 are the right hyperparameters. | Figure 3 sweep (InstructBLIP). | ⭐ Sweet spot is shown for InstructBLIP only — but **LLaVA-1.5 silently switches to K=4** with no analogous sweep. The K=64 vs K=4 gap is the most underexplained design choice in the paper. |

**Honest summary.** The most credible claims are C1 (negative transfer is real, Figure 1 is excellent) and C3 (the universal-expert ablation is well-controlled and the mechanism is exact). The cluster-routing-vs-sentence-routing comparison (C2, Table 7 + Figure 6) is the most architecturally meaningful contribution — the routing heatmap on Figure 6 is the visualization where MoCLE most clearly wins on substance. Everything in Tables 2–3 is **single-seed, single-backbone, no significance** — so the small deltas are suggestive only. The medical claim (C7) is the clearest overreach: SLAKE-Closed collapses from 85.58 (medical-only LoRA) to 35.21 (MoCLE on the full mix), MoCLE does not recover it, and the paper does not call this out. The PathVQA Open blank cells in Table 4 are also unexplained. Finally, **compute/latency overhead is never measured** and there is no scaling study beyond 7B.

## Method & Architecture

![MoCLE architecture](/assets/images/paper/mocle/page_003.png)
*Figure 2: Page 3 of the paper. The MoCLE pipeline — instructions are k-means-clustered using `all-MiniLM-L6-v2` text embeddings (visual tokens excluded), a top-1 gate dispatches each input to one task LoRA expert based on the cluster embedding, and a separate universal LoRA expert is always active with weight $1-G_{\max}$.*

**Step by step.**

1. **Encode every training instruction.** Use `all-MiniLM-L6-v2` Sentence-Transformer on the *templated text instruction only*. Visual tokens are excluded — the clustering signal is purely textual.
2. **k-means cluster into K groups** by minimizing $\sum_{j=1}^{K}\sum_{e_i\in S_j}\lVert e_i - c_j\rVert^2$. Default K = 64 for InstructBLIP, K = 4 for LLaVA-1.5. Cluster index per training sample is precomputed and frozen.
3. **Insert MoCLE blocks** into selected linear projections of the frozen LLM: `q_proj, v_proj` for InstructBLIP, `up_proj, down_proj` for LLaVA-1.5. Each block has E task LoRA experts (default 4) plus 1 universal LoRA expert, rank-8 for InstructBLIP and rank-128 for LLaVA-1.5.
4. **Gate.** For input $x_i$ with cluster $C[x_i]$ (shared across all layers, initialized to centroid $c_j$, then trainable):

   $$G = \text{top-}k\!\left(\text{softmax}\!\left(\tfrac{1}{\tau}(W_{\text{gate}} C[x_i] + \epsilon)\right)\right),\quad \epsilon\sim\mathcal{N}(0, 1/E),\ \tau=0.05$$

   Top-k is set to k=1 (a single task expert is activated). $W_{\text{gate}}$ is per-layer; the cluster embedding $C[\cdot]$ is shared across layers.

5. **Universal expert combination.** The output mixes the chosen task expert, the always-on universal expert, and the frozen LLM linear:

   $$y_i = \Big(\sum_{e=1}^{E} G_e W_e + (1-G_{\max}) W_u\Big) x_i + W_0 x_i$$

   As $\tau\to 0$, $G_{\max}\to 1$, and the universal expert's weight shrinks — so $\tau$ is effectively a specialization-vs-generalization knob.

6. **Training.** Standard next-token cross-entropy on the instruction-tuning corpus. **No load-balancing loss** ("we found it might distort task specialization," footnote 1). For InstructBLIP, trainable params are the Q-Former + all LoRAs. For LLaVA-1.5, the MLP connector + all LoRAs.
7. **Inference.** Cluster the new instruction by nearest-centroid in MiniLM-embedding space, look up $C[\cdot]$, run the routed expert + universal expert. No retraining for unseen tasks.

![MoCLE gating math](/assets/images/paper/mocle/page_004.png)
*Figure 3: Page 4 of the paper. The gating softmax with temperature τ=0.05 and Gaussian noise $\epsilon\sim\mathcal{N}(0,1/E)$, plus the universal-expert combination $y_i = (\sum_e G_e W_e + (1-G_{\max}) W_u) x_i + W_0 x_i$.*

## Experimental Results

### InstructBLIP-7B held-in / held-out (Tables 2–3)

| Dataset | InstructBLIP | **+ MoCLE** | Δ |
|---|---|---|---|
| GQA | 48.6 | **49.3** | +0.7 |
| VSR | 60.8 | **64.7** | **+3.9** |
| IconQA | 43.4 | **46.3** | **+2.9** |
| VisualDialog | 46.3 | **46.9** | +0.6 |
| MME (perception) | 1202.9 | **1222.6** | +19.7 |
| POPE (adv. F1) | 77.6 | **82.1** | **+4.5** |
| A-OKVQA Direct | 58.8 | **61.5** | +2.7 |
| A-OKVQA MC | 73.8 | **78.2** | **+4.4** |
| OKVQA | 57.0 | **59.8** | +2.8 |
| VQAv2 test-dev | 77.4 | **78.9** | +1.5 |
| Flickr30K (CIDEr) | 81.3 | **81.9** | +0.6 |
| TextVQA | 53.9 | **57.1** | **+3.2** |
| HatefulMemes (AUC) | 65.3 | **65.6** | +0.3 |
| ScienceQA | 62.0 | **63.9** | +1.9 |
| MSVD-QA | 41.4 | **42.6** | +1.2 |
| MSRVTT-QA | 23.0 | **24.4** | +1.4 |
| iVQA | 51.3 | **53.2** | +1.9 |

Every $\Delta$ is positive, but no variance or significance is reported — so the cluster of sub-1.5-point gains (GQA, HM, VisualDialog, Flickr30K, video-QA) is suggestive rather than conclusive.

![InstructBLIP results and LLaVA cross-domain](/assets/images/paper/mocle/page_005.png)
*Figure 4: Page 5 of the paper. Tables 1–4 — architecture details, InstructBLIP held-in/held-out zero-shot results, and the LLaVA-1.5 cross-domain (natural + geometric + medical) comparison.*

### LLaVA-1.5-7B cross-domain (Table 4)

| Method | Train data | MME | MMB | SQA | GeoQA | VQA-RAD O / C | SLAKE O / C | PathVQA O / C |
|---|---|---|---|---|---|---|---|---|
| Single LoRA | LLaVA-665K | 1804 | 65.89 | 67.67 | – | – | – | – |
| Single LoRA | Geo170K | – | – | – | 57.82 | – | – | – |
| Single LoRA | Medical Mix | – | – | – | – | 53.90 / 84.19 | 86.05 / **85.58** | 38.07 / 91.77 |
| Single LoRA | All | 1794 | 64.69 | 66.78 | 46.89 | 77.94 / 84.61 | 82.45 / **35.56** | – / 90.71 |
| **MoCLE** | All | **1838** | **66.07** | 67.38 | **53.59** | **81.98** / 83.29 | **85.10** / **35.21** | – / 91.65 |

Two things stand out. First, the Single-LoRA-on-All vs. Single-LoRA-per-domain comparison cleanly reproduces the negative-transfer claim — training on everything *hurts* GeoQA (57.82 → 46.89) and several medical metrics. Second, **MoCLE recovers most of the geometric and natural-image gap (GeoQA 53.59, MME 1838 > 1804) but does not recover SLAKE-Closed at all (85.58 medical-only → 35.21 MoCLE), and the PathVQA-Open column is blank for two rows with no explanation in the paper.** The medical-conflict claim rests on VQA-RAD-Open improvements only.

### Ablations (Tables 5–7, Figure 3)

![MoCLE ablations](/assets/images/paper/mocle/page_006.png)
*Figure 5: Page 6 of the paper. Tables 5–7 — component ablation (Cluster MoE rank 8 vs flat LoRA rank 64, universal expert on/off), top-1+universal vs top-2 routing, and gating strategy comparison (cluster vs token / sentence / dataset routing).*

- **Universal expert is the big lever.** Removing it (Table 5 row c) drops VSR 64.7 → 58.9 and TextVQA 57.1 → 54.9. The held-out generalization gains are largely the universal expert, not the cluster routing.
- **Cluster MoE rank 8 (16.78M) > flat LoRA rank 64 (33.55M).** Avg 61.8 vs 60.7 with half the params — so the gain is not pure capacity.
- **Top-1 + universal beats top-2.** Top-2 collapses to avg 60.4 vs 63.3 for top-1+universal. The interpretation: top-2 reintroduces inter-expert conflict; the universal expert is a *better* second slot because it is always-on and carries shared skill.
- **Cluster routing > token / sentence / dataset routing (Table 7).** Cluster 61.8 > Dataset 60.8 > Sentence-MoLE 59.7 > Token-MoLE (LLaVA-MoLE) 58.1 avg. Largest differential on VSR (58.9 vs 49.0 for token) and TextVQA (54.9 vs 46.6).

![Hyperparameter sweep](/assets/images/paper/mocle/page_007.png)
*Figure 6: Page 7 of the paper. Figure 3 sweep — average held-out performance vs. number of clusters K, gate temperature τ, and number of task experts. K=64, τ=0.05, 4 experts are the defaults for InstructBLIP. LLaVA-1.5 silently switches to K=4 with no analogous sweep.*

![Clustering and routing analysis](/assets/images/paper/mocle/page_008.png)
*Figure 7: Page 8 of the paper. Figure 4 (t-SNE of instruction embeddings showing task-level grouping: VQA, VQG, captioning markers separate spatially), Figure 5 (K=64 cluster-assignment heatmap by dataset), and Figure 6 (routing decisions of MoCLE vs Sentence-MoLE — MoCLE shows clean, task-aligned diagonal routing; Sentence-MoLE's routing is nearly uniform across experts).*

## Limitations

**The authors acknowledge** that the scope is limited to text-conversation-style tasks; visual perception tasks (detection, segmentation), where conflicts are more severe, are not tested.

**Additional issues this review flags:**

- **Clustering uses only the text instruction** — image content does not influence routing. A medical-image VQA and a natural-image VQA that share an instruction template will land in the same cluster. The K=4 choice for LLaVA-1.5 sidesteps but does not resolve this.
- **Cluster assignments are frozen after k-means.** The cluster embedding is trainable; the hard cluster id is not. No experiment shows brittleness to re-clustering or to instruction paraphrase.
- **K=64 (InstructBLIP) vs K=4 (LLaVA-1.5) is unexplained.** Figure 3 establishes K=64 as the sweet spot for InstructBLIP only; LLaVA-1.5 quietly uses K=4 with no analogous sweep.
- **No load-balancing loss**, justified as preserving specialization — but then expert collapse prevention rests entirely on the noise term $\epsilon$, and no expert-utilization measurements are reported for LLaVA-1.5.
- **No compute or latency benchmark** of MoCLE vs single-LoRA baseline. Each layer has 4 task LoRAs + 1 universal + a per-layer gate; the overhead is never measured.
- **No variance, no seed averaging, no significance tests** anywhere in Tables 2–7.
- **"Unseen" means held-out datasets with similar instruction styles**, not paraphrased instructions or out-of-task-family templates. The strong generalization claim is supported only within that narrower definition.
- **Medical claim is overextrapolated.** SLAKE-Closed 85.58 → 35.21 in the multi-task regime is a catastrophic collapse MoCLE does not fix; PathVQA Open is blank in the headline table. The paper does not address either.
- **All experiments are at 7B scale** — no scaling study.

## Why It Matters for Medical AI

The cross-domain conflict regime in §4.2 (LLaVA-1.5 trained on natural + geometric + medical mix) is exactly the situation a generalist medical assistant ends up in: there is rarely a single clean training source, and mixing in non-medical data is the only way to keep the language model competent. MoCLE's contribution to that setting is genuine on a few metrics — VQA-RAD-Open goes from 77.94 to 81.98, and GeoQA recovers from 46.89 (collapsed under multi-task) to 53.59. The routing visualization in Figure 6 also gives an architecturally clean recipe for *how* cluster-conditioning preserves task specialization without manual taxonomy.

But the medical reading needs to be honest: **MoCLE does not actually fix the multi-task medical regression in this paper**. SLAKE-Closed sits at 35.21 vs 85.58 from the medical-only expert, and PathVQA-Open is missing entirely from Table 4. So the right takeaway for a clinical pipeline is that MoCLE is a useful template (cluster routing + always-on universal LoRA) for combating negative transfer, but it has not yet been demonstrated to recover medical closed-set VQA performance under cross-domain training. Anyone deploying this should plan to verify per-domain numbers, run multiple seeds, and probably re-tune K against an actual medical-VQA evaluation rather than inheriting K=4 from the paper.

## References

- Paper: Gou, Liu, Chen, et al. *Mixture of Cluster-Conditional LoRA Experts for Vision-Language Instruction Tuning.* CVPR 2024 (arXiv 2312.12379v5, Jul 2024). [arXiv:2312.12379](https://arxiv.org/abs/2312.12379)
- Sentence encoder: Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019; `all-MiniLM-L6-v2` checkpoint via `sentence-transformers`.
- Backbones: Dai et al., *InstructBLIP*, NeurIPS 2023; Liu et al., *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*, CVPR 2024.
- Related MoLE / routing baselines: LLaVA-MoLE (token routing), Octavius (sentence routing), Jang et al. dataset routing.
- Medical VQA datasets: VQA-RAD (Lau et al. 2018), SLAKE (Liu et al. 2021), PathVQA (He et al. 2020); geometric reasoning: Geo170K (Gao et al. 2023).
