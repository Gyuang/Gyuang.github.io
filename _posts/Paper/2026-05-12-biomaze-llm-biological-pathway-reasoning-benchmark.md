---
title: "BioMaze: Benchmarking and Enhancing Large Language Models for Biological Pathway Reasoning"
excerpt: "5.1K KEGG-pathway questions across 12 reasoning settings; PathSeeker agent wins on perturbed/intervened cells where prior graph-augmented LLMs collapse."
categories:
  - Paper
tags:
  - BioMaze
  - PathSeeker
  - LLM-Reasoning
  - Knowledge-Graph
  - KEGG
  - Pathway-Reasoning
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR
- BioMaze is a **5,100-question** expert-curated benchmark for biological *pathway* reasoning, built from ~6,000 PubMed intervention papers and tagged across **12 reasoning settings** (Inquiry × Condition × Target).
- PathSeeker is an LLM agent that interleaves **global `Search_Subgraph`** and **local `Neighbor_Subgraph`** calls on a KEGG-derived graph (8,939 nodes / 15,131 edges), with DFS-ordered deduplicated text encoding and a separate FinalReasoner step.
- On True/False GPT-3.5, PathSeeker hits **63.93 on the Perturbed subset** (vs. CoT 0-shot 61.48, G-Retriever 59.32) and **+14.35 pts** over CoT on Intervened open-ended — but absolute accuracies hover at 55–65 % on a binary task, so the benchmark is far from solved.

## Motivation
Pathway reasoning — predicting how a perturbation (mutation, drug, infection) propagates through a signaling or metabolic network to a downstream phenotype — is central to cell biology, pharmacology, and toxicology. Prior bio-LLM benchmarks cover protein design (e.g., ProteinGym), drug-property QA, or factual biomedical QA (PubMedQA, BioASQ, MedQA), but none of them probe multi-step *causal* inference over a known pathway graph. KEGG and Reactome encode the relevant topology, yet LLMs ingest the corresponding text during pre-training as unstructured prose, so they cannot reliably plan reasoning along graph topology. BioMaze positions itself as the missing piece for using LLMs in hypothesis generation and toxicity prediction.

## Core Innovation
PathSeeker mimics how a biologist browses pathway maps: it issues `Search_Subgraph(query, N)` for global hops and a new `Neighbor_Subgraph(line_id, query, N)` for local hops around a previously seen triple. Both calls are formulated as Prize-Collecting Steiner Tree (PCST) optimizations solved with Hegde et al.'s near-linear-time approximation, then binary-searched over the edge-cost penalty to hit a target subgraph size N. Returned subgraphs go through three post-processing steps before the LLM sees them: `RemoveSeen` (drop triples returned in earlier turns), `DFSOrder` (re-order so the layout follows a plausible biological flow rather than retrieval score), and `TripleToOrderedText` with *globally consecutive* line IDs so the LLM can reference any past triple in a subsequent `Neighbor_Subgraph` call. A separate **FinalReasoner** call consumes the full transcript and produces the answer, decoupling exploration from final decoding — the ablation shows this decoupling is the single most load-bearing component.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | LLMs struggle on pathway reasoning, especially in *perturbed* systems. | Tables 2–3: 5–12 pt Normal-vs-Perturbed gaps across LLaMA-8B → GPT-4; Figure 4 confirms across 7 backbones. | ⭐⭐⭐ |
| C2 | PathSeeker outperforms CoT and prior graph-augmented LLMs (CoK, ToG, G-Retriever). | Tables 2 and 3 — top or runner-up in nearly every cell for both backbones. | ⭐⭐ (single-run; no variance reporting) |
| C3 | Pathway augmentation especially helps on intervention cases. | Largest gain on Intervened OE GPT-3.5: **57.59 vs. CoT 43.24 (+14.35)**. | ⭐⭐ (one striking cell; other intervention cells +3–5 pts) |
| C4 | PathSeeker's accuracy stays roughly flat with reasoning step count. | Figure 5: PathSeeker is flat over 1–9 steps while CoT drops 15–25 pts. | ⭐⭐ (step counts are themselves LLM-labeled) |
| C5 | Pathway grounding reduces faulty reasoning. | Figure 6: 200-failure audit, hand-labeled by a biology PhD; FR errors shrink. | ⭐⭐ (small N, single annotator) |
| C6 | BioMaze covers diverse domains and reasoning types. | Figure 2 + Table 1: 6 KEGG super-categories × 12 fine-grained settings. | ⭐⭐⭐ |
| C7 | Local subgraph navigation is the key new ingredient. | Table 6 ablation: removing local search drops TF 61.87 → 57.78 (−4.09). | ⭐⭐ (ablation only on LLaMA-8B) |
| C8 | Scaling the backbone alone does not close the causal-reasoning gap. | Figure 4: Normal-vs-Perturbed gap visible across all 7 backbones, including GPT-4. | ⭐⭐⭐ |

The benchmark itself (C1, C6, C8) is the most solidly evidenced contribution — gaps replicate across seven backbones and two task formats. The method claims (C2, C3) are credible but thinner: every number is a single run with no seeds, no significance tests, and several "best" cells (e.g., GPT-3.5 TF Normal 63.55 vs. G-Retriever 64.14) are within plausible noise.

![BioMaze task vs. PathSeeker overview](/assets/images/paper/biomaze/page_002.png)
*Figure 1: A muscarinic-M3 → taste-receptor cascade illustrates the contrast between CoT relying on internal knowledge and PathSeeker grounding each step against the KEGG pathway graph.*

## Method & Architecture

![PathSeeker agent loop](/assets/images/paper/biomaze/page_005.png)
*Figure 2: PathSeeker alternates global `Search_Subgraph(query, N)` and local `Neighbor_Subgraph(line_id, query, N)` calls; returned subgraphs are deduplicated, DFS-ordered, and serialized with globally consecutive line IDs before a separate FinalReasoner step produces the answer.*

The pathway graph is built by merging all KEGG maps into one heterogeneous graph: **8,939 entries (nodes), 15,131 edges, 2,265 biological processes**, with each edge stored as `[Head IDs, Tail IDs, (Relation Type, Biological Process IDs)]` plus a description/function corpus that acts as the retrieval index.

At step $t$, given history $h_t = [o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t]$, the LLM selects an action $a_t = G(E, h_t)$ — either a global or a local subgraph call. The returned subgraph $S_t$ goes through:

$$\hat{S}_t = \mathrm{DFSOrder}(\mathrm{RemoveSeen}(S_t, [S_1, \ldots, S_{t-1}]))$$

and is then linearized as `Line_ID) Head | Tail | Relation and Biological Process`, with the line counter persisting across turns so the LLM can address any prior triple. After ≤10 steps (50 % of TF tasks finish in 4–6), FinalReasoner consumes the transcript:

$$a_r = G(E_r, [o_1, \ldots, o_T])$$

![BioMaze coverage and taxonomy](/assets/images/paper/biomaze/page_004.png)
*Figure 3: BioMaze covers six KEGG super-categories (metabolism, genetic information, environmental information, cellular processes, organismal systems, human diseases) and is taxonomized along three axes (Inquiry Type × Extra Condition × Investigation Target), yielding 12 fine-grained settings.*

## Experimental Results

Accuracy (%) on BioMaze True/False with a **GPT-3.5 backbone** (Table 2 in the paper); 50 % = random for binary tasks.

| Method | Graph | Normal | Perturbed | Natural | Intervened | Single | Interaction | Function |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Vanilla (0-shot) | ✗ | 57.92 | 54.60 | 56.99 | 54.88 | 59.91 | 55.63 | 56.68 |
| Vanilla (2-shot) | ✗ | 60.73 | 55.59 | 57.40 | 59.39 | 60.30 | 46.43 | 58.26 |
| CoT (0-shot) | ✗ | 59.92 | 61.48 | 62.74 | 51.00 | 57.69 | 56.75 | 66.25 |
| CoT (2-shot) | ✗ | 64.92 | 56.39 | 61.46 | 57.12 | 60.86 | 61.01 | 59.92 |
| ToG | ✓ | 59.60 | 50.83 | 53.92 | 62.50 | 53.40 | 60.00 | 55.21 |
| CoK | ✓ | 60.70 | 54.07 | 57.29 | 56.49 | 60.19 | 50.00 | 58.04 |
| G-Retriever | ✓ | 64.14 | 59.32 | 61.55 | 61.88 | 61.53 | 59.00 | 62.60 |
| **PathSeeker** | ✓ | **63.55** | **63.93** | **57.48** | **62.74** | **62.85** | **64.73** | **68.13** |

Additional findings:

- **Open-ended GPT-3.5 (Table 3)**: PathSeeker leads Perturbed (64.33 vs. CoT 0-shot 61.49) and especially Intervened (**57.59 vs. 43.24, +14.35**) — the headline gain.
- **Backbone scaling (Figure 4)**: Larger models (LLaMA3.1-70B, Qwen2.5-72B, GPT-4) score higher overall, but Normal-vs-Perturbed and Natural-vs-Intervened gaps persist at every scale.
- **Step-count robustness (Figure 5)**: CoT degrades sharply from 1 → 7+ reasoning steps; PathSeeker stays roughly flat — the strongest mechanistic evidence that graph navigation helps on long chains.
- **Error decomposition (Figure 6)**: PathSeeker substantially reduces Faulty Reasoning errors versus CoT; residual errors are mostly Omission in Reasoning, often traceable to missing KEGG coverage.
- **API usage (Tables 4–5)**: ~1.5 global + ~3.5 local searches per task on average, justifying the new `Neighbor_Subgraph` API.
- **Ablation (Table 6, LLaMA3-8B)**: Full PathSeeker 61.87 TF / 61.21 OE. Removing FinalReasoner → 56.97 / 58.25 (largest drop); local search → 57.78 / 57.46; TripleToText → 58.32 / 57.06; DFSOrder → 58.60 / 55.82; RemoveSeen → 57.48 / 58.96. Every component contributes 2–5 points.

![Backbone scaling, step-count robustness, and error decomposition](/assets/images/paper/biomaze/page_008.png)
*Figure 4: Across 7 backbones the Normal-vs-Perturbed gap persists (Fig. 4); PathSeeker's accuracy is roughly flat over reasoning-step count while CoT decays (Fig. 5); on a 200-failure audit PathSeeker shrinks Faulty Reasoning errors but leaves Omission errors that often trace back to KEGG coverage gaps (Fig. 6).*

![Main accuracy tables](/assets/images/paper/biomaze/page_007.png)
*Figure 5: Tables 2 and 3 — TF and open-ended accuracy on BioMaze with GPT-3.5 and LLaMA3.1-8B; PathSeeker leads on perturbed and intervened settings where prior graph-augmented methods drop.*

## Limitations

Acknowledged by the authors:

- KEGG coverage is incomplete; some BioMaze questions reference components absent from the graph, producing residual Omission errors even with PathSeeker.
- Ground truth reflects single source-paper conclusions, without cross-paper validation — a known reproducibility risk in biology.
- LLM-as-judge on open-ended tasks (LLaMA3.1-405B) is imperfect; Appendix A.9 quantifies but does not eliminate this.

Noticed but not addressed:

- **No variance / multi-seed reporting** on any table — every method comparison is a single run, and several "best" cells are within plausible noise of competitors.
- The LLM judge (LLaMA3.1-405B) is also one of the data generators — a non-trivial circularity risk.
- The 40 % expert-pass-rate filter is reported without inter-annotator agreement statistics.
- **No plain-RAG baseline** over the KEGG description corpus — only graph-structured methods are compared, so the gain attributable specifically to *graph* structure (vs. just having KEGG text in context) is unclear.
- Compute / cost of PathSeeker is not reported (~5 API calls per question × multi-turn agent ≈ 10–20× the cost of CoT).
- GPT-4 appears in the scaling plot but not the main method-comparison tables, weakening the "scales to frontier models" framing.
- No exploration of generalization to non-KEGG graphs (Reactome, WikiPathways, STRING).

## Why It Matters for Medical AI
Pathway-level causal reasoning is the bottleneck for many clinical LLM use cases: predicting how a candidate drug perturbs a signaling cascade, anticipating off-target toxicity, or interpreting a multi-omics readout in terms of upstream regulators. BioMaze provides the first benchmark that *isolates* this skill from generic biomedical QA, and PathSeeker offers a concrete recipe — interleave global and local KG navigation, keep a deduplicated DFS-ordered transcript, and decode the final answer with a separate reasoner — that translates directly to clinical-decision-support pipelines built on top of curated knowledge graphs (Reactome, DrugBank, Open Targets). The honest takeaway, though, is that absolute accuracies still sit in the 55–65 % band on a binary task, so deployment in safety-critical workflows remains premature; the contribution is a measurable target to optimize against, not a solved capability.

## References
- Paper (arXiv 2502.16660, v5): <https://arxiv.org/abs/2502.16660>
- Code & data: <https://github.com/zhao-ht/BioMaze>
- KEGG database (Kanehisa & Goto, 2000): <https://www.kegg.jp/>
- G-Retriever (He et al., 2024): retrieval-augmented LLM over KGs via PCST.
- ToG (Sun et al., 2024) — Think-on-Graph; CoK (Li et al., 2023) — Chain-of-Knowledge.
- Hegde, Indyk, Schmidt (2015): near-linear-time PCST approximation, used as PathSeeker's inner-loop solver.
