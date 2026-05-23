---
title: "ChatCAD+: Toward a Universal and Reliable Interactive CAD using LLMs"
excerpt: "A training-free orchestration of specialist CAD networks, BiomedCLIP domain routing, and an LLM-driven DFS over Merck Manuals lifts CheXbert F1 from 0.553 (ChatCAD) to 0.564 on a 1k MIMIC-CXR subset."
categories:
  - Paper
  - Pathology
permalink: /paper/chatcad-plus-universal-interactive-cad/
tags:
  - ChatCAD+
  - LLM-Agents
  - BiomedCLIP
  - Retrieval-Augmented-Generation
  - Medical-Report-Generation
  - Merck-Manuals
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- ChatCAD+ is a training-free pipeline that routes an input image to one of three specialist CAD networks (chest X-ray, dental panoramic X-ray, knee MRI) via a frozen BiomedCLIP domain identifier, then has an LLM rewrite a "prob2text" draft using top-k MIMIC-CXR exemplars retrieved by TF-IDF + KD-tree. On a 1,000-sample MIMIC-CXR subset (5 findings, 200 each) it scores **C-BLEU 4.409 / METEOR 24.337 / CheXbert F1 0.564**, edging the authors' prior ChatCAD (F1 0.553) and R2GenCMN (F1 0.458).
- The "reliable interaction" half drops dense-vector retrieval entirely and instead runs an **LLM-as-DFS** over a tree-structured Merck Manuals dictionary (topic -> section -> subsection) with backtracking on "no choice". On CMExam (2,190 Chinese disease Q-A pairs) this lifts ChatGLM3 from ACC 0.458 / ROUGE-L 14.915 to **ACC 0.472 / ROUGE-L 19.302** — small accuracy gain, but explanation ROUGE-L nearly doubles.
- The "universal" framing is the paper's weakest seam: only **3 of the 9 domains** in the BiomedCLIP router have end-to-end CAD models behind them, the retrieval pool (MIMIC-CXR train) is the same distribution as the test set, and the LLM-DFS-beats-LangChain claim is supported only by 4 hand-picked qualitative examples with no retrieval@k metric. The cross-LLM sweep (and the honest report that **PMC-LLaMA regresses from F1 0.532 -> 0.525** with the retrieval module on) is the strongest piece of evidence in the paper.

## Motivation

The authors' own ChatCAD (arXiv 2302.07257) was a one-modality demo: a chest-X-ray classifier piped numeric probabilities into ChatGPT, which narrated them. Two structural gaps followed. First, radiology workflows are multi-modal — dental panoramics, knee MRI, mammography, fundus, dermoscopy, endoscopy — and there is no single specialist that covers them. Second, when patients ask follow-up questions, the LLM answers from parametric memory rather than from any verifiable clinical reference, which is precisely the failure mode you cannot ship in a CAD product.

ChatCAD+ reframes the problem as two coupled subproblems. **Universal interpretation**: route the image to the right specialist via BiomedCLIP zero-shot, then translate that specialist's outputs into prose. **Reliable interaction**: force the LLM to answer from a curated knowledge base (Merck Manuals), and make the retrieval step itself driven by the LLM so it can backtrack when the candidate topic is wrong. The bet is that specialist-ensemble + retrieval beats training a single generalist medical VLM in the 2023-24 regime, particularly across the morphological gulf between modalities.

## Core Innovation

- **Hierarchical in-context report refinement.** A first-draft "preliminary report" is generated from prob2text descriptions of CAD outputs, then **k=3** nearest MIMIC-CXR reports are retrieved (TF-IDF over 17 thoracic-disease terms, L2 KD-tree on unit-hypersphere-normalized embeddings) and used as in-context exemplars for a rewrite pass. Nothing is fine-tuned.
- **LLM-as-DFS knowledge retrieval.** Instead of dense-vector similarity over chunked Merck text, the LLM walks the Merck tree top-down: pick one of 5 candidate topic titles -> read its abstract + section names -> descend or backtrack on "no choice". Backtracking is the part that distinguishes it from a vanilla agent loop, and is the answer to the failure mode where SBERT-style retrieval lands in the wrong topic entirely (e.g., palmoplantar keratodermas instead of periodontitis treatment).
- **BiomedCLIP > CLIP > PMC-CLIP** as the domain router. Domain identification is cosine similarity between an image embedding and per-domain text prompts. BiomedCLIP hits 98.9% on 9 cumulative domains; vanilla CLIP drops to 86.3%; PMC-CLIP collapses to ~50% even at 3 domains.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | ChatCAD+ is "universal" across imaging domains | Domain-ID accuracy across 9 modalities (Fig. 5); qualitative examples on 3 of those domains (Fig. 6) | Custom 9-domain composite | ⭐⭐ — only **3 of 9** domains have downstream CAD; the rest only test the CLIP router |
| C2 | Hierarchical in-context learning improves report quality over plain ChatCAD | Table III: F1 0.553 -> 0.564, METEOR 23.146 -> 24.337; ablation Fig. 7 saturates at k=3 | MIMIC-CXR 1k subset | ⭐⭐ — single dataset, single 1k subset, no variance/CI bars |
| C3 | ChatCAD+ generalizes across LLM backbones | Table IV: 8 of 9 LLMs improve; PMC-LLaMA regresses 0.532 -> 0.525 | MIMIC-CXR 1k | ⭐⭐⭐ — broad sweep + honest disclosure of the regression strengthens credibility |
| C4 | Specialist + LLM beats generalist medical VLMs | Table V vs. PMC-VQA, RadFM (zero-shot) | MIMIC-CXR 1k, zero-shot | ⭐ — output-length mismatch makes the comparison rhetorical; C-BLEU is "/", not a comparable number |
| C5 | LLM-DFS knowledge retrieval beats dense-vector LangChain retrieval | Tables VII-VIII color-coded qualitative examples (4 questions) | Merck Manuals + 4 manual queries | ⭐ — anecdotal; **no retrieval@k metric** over a labeled question -> section gold standard |
| C6 | Knowledge retrieval improves clinical QA | Table VI: ChatGLM3 ACC 0.458 -> 0.472, ROUGE-L 14.915 -> 19.302 | CMExam 2,190 subset | ⭐⭐ — real exam, but only two 6B Chinese LLMs (cost-driven), no significance tests, accuracy gain is 1-3 pts |
| C7 | BiomedCLIP is the best CLIP backbone for medical domain ID | Fig. 5 vs. CLIP and PMC-CLIP | 9-domain composite | ⭐⭐ — clear margin but only 3 CLIPs compared |
| C8 | System is "reliable" enough for patient consultation | Qualitative dialogues with attached Merck references | — | ⭐ — no hallucination rate, no clinician evaluation, no factuality benchmark |

**Honest read.** C3 is the strongest piece of evidence in the paper — running 9 LLM backbones and reporting that PMC-LLaMA regresses rather than hiding it builds credibility. C2 and C6 are quantitatively defensible but each rests on a single benchmark, no error bars, and in C2's case **the in-context exemplar pool (MIMIC-CXR train) is the same distribution as the 1k test subset**, which biases the retrieval ablation in ChatCAD+'s favor. C1 oversells "universal": the system actually does end-to-end CAD on **three** modalities, and the 9-domain accuracy figure is testing a CLIP zero-shot classifier, not the full pipeline. C5 and C8 — the two centerpiece "reliability" claims — rest almost entirely on 4 hand-picked qualitative examples with no retrieval@k, no clinician rating, and no hallucination audit.

## Method & Architecture

![ChatCAD+ system overview](/assets/images/paper/chatcad-plus/page_003.png)
*Figure 1: ChatCAD+ system overview (page render of Fig. 2). The image is routed by BiomedCLIP to one of three specialist CAD networks; CAD outputs are converted to prob2text descriptions; the LLM writes a preliminary report and then refines it with k=3 retrieved MIMIC-CXR exemplars.*

### 1. Domain identification

Given image $I$ and per-domain textual prompts $\{M_i\}$, BiomedCLIP encodes both sides and the predicted domain is

$$
D_{\text{pred} } = \arg\max_{i} \frac{I \cdot M_i}{\lVert I \rVert \cdot \lVert M_i \rVert}.
$$

CLIPs are frozen. Three domains are actually wired to specialist CAD models: chest X-ray (CheXpert-trained classifier + R2GenCMN report generator), dental panoramic X-ray (HC-Net periodontitis classifier trained on 300 private 2903x1536 panoramics), and knee MRI (graph-representation cartilage-defect model trained on 964 in-house Philips Achieva 3.0T MRIs).

### 2. prob2text

CAD probabilities are verbalized before being shown to the LLM. Three schemes:

- **P1** — direct numeric ("Cardiomegaly: 0.83").
- **P2** — binary thresholded at 0.5 ("Cardiomegaly: positive").
- **P3** — four-level illustrative phrasing on thresholds [0, 0.2) / [0.2, 0.5) / [0.5, 0.9) / [0.9, 1]: "No sign / Small possibility / Likely / Definitely". **P3 is the default** because it most closely matches human radiology phrasing.

### 3. Preliminary report

The LLM is prompted "Write a report based on results from Network(s)" with the prob2text description; output is a first-draft report.

### 4. Retrieval module

For every report $d$ in MIMIC-CXR train, compute a TF-IDF embedding over a 17-term thoracic-disease vocabulary:

$$
\text{TF-IDF}(t, d) = \frac{\#t \in d}{\lvert d \rvert} \cdot \log \frac{\lvert D \rvert}{\lvert \{d : t \in d\} \rvert}.
$$

L2-normalize onto the unit hypersphere so that cosine ranking can be served by an L2 KD-tree in $O(\log n)$:

$$
L_2(\hat{q}, \hat{v}) = 2r \cdot \sin(\theta/2).
$$

Retrieve **k=3** nearest reports (Fig. 7 ablation saturates here) and prompt the LLM to rewrite the preliminary report using those three as in-context exemplars. Output is the final enhanced report. The whole pipeline trains nothing — the underlying CAD models can be (re)trained independently.

![LLM-as-DFS pseudocode over the Merck Manuals tree](/assets/images/paper/chatcad-plus/fig_p005_04.png)
*Figure 2: Pseudocode for the LLM-driven depth-first search over the Merck Manuals dictionary. The LLM either returns content ("found"), descends into a subsection, or returns "no choice" — in which case the algorithm pops the stack and backtracks to the parent tier.*

### 5. Reliable interaction (Merck-tree DFS)

Merck Manuals is preprocessed into a tree: medical topic -> sections {abstract, symptoms and signs, diagnosis, treatment, prognosis} -> subsections. Given the user question, kNN over topic titles retrieves 5 candidates; the LLM picks one. The abstract + section names of the chosen topic are then shown; the LLM either returns the relevant content, descends, or backtracks. Iteration runs over the 3-4 tiers of Merck. Final prompt: *"Please refer to the following knowledge to answer [#Question] and provide an analysis: [#Knowledge]."* The retrieved snippet is shown to the user underlined for transparency.

### Defaults that matter

- k=3 retrieved reports (Fig. 7 saturation).
- P3 prob2text grading thresholds [0, 0.2) / [0.2, 0.5) / [0.5, 0.9) / [0.9, 1].
- Domain-ID encoder: BiomedCLIP (not vanilla CLIP, not PMC-CLIP — see Domain ID section below).
- Report-generation LLM: ChatGPT default; cross-LLM sweep covers ChatGLM2, LLaMA/LLaMA2 7B & 13B, Mistral 7B, PMC-LLaMA 7B, MedAlpaca 7B.
- Clinical-QA LLM: ChatGLM2/3 6B (the recursive DFS is too expensive on GPT-4 by the authors' admission).

## Experimental Results

### Main comparison on MIMIC-CXR (Table III, 1,000-sample subset)

| Method | Params | C-BLEU | ROUGE-L | METEOR | PRE | REC | F1 |
|---|---|---|---|---|---|---|---|
| R2GenCMN | 244M | 3.525 | **18.351** | 21.045 | 0.578 | 0.411 | 0.458 |
| VLCI | 357M | 3.834 | 16.217 | 20.806 | **0.617** | 0.299 | 0.377 |
| ChatCAD (P3) | 256M + 175B | 3.594 | 16.791 | 23.146 | 0.526 | 0.603 | 0.553 |
| ChatCAD+ (P1) | 256M + 175B | 4.073 | 16.820 | 22.700 | 0.548 | 0.484 | 0.502 |
| ChatCAD+ (P2) | 256M + 175B | 4.407 | 17.266 | 23.302 | 0.538 | 0.597 | 0.557 |
| **ChatCAD+ (P3)** | **256M + 175B** | **4.409** | 17.344 | **24.337** | 0.531 | **0.615** | **0.564** |

ChatCAD+ wins C-BLEU, METEOR, recall, and CheXbert F1, but loses ROUGE-L to R2GenCMN and precision to VLCI. The traditional encoder-decoder baselines remain better at lexical overlap and precision-heavy operating points; the LLM pipeline trades that for recall and clinical-finding F1. The test subset is 1k samples (200 per finding x 5 findings: cardiomegaly, edema, consolidation, atelectasis, pleural effusion), subsampled "due to OpenAI per-hour rate limits" — it is **not** the official MIMIC-CXR test split.

### Cross-LLM sweep (Table IV, brief)

ChatGPT > Mistral 7B ~ LLaMA2 13B > ChatGLM2 6B > LLaMA2 7B. **PMC-LLaMA actually degrades** with reliable-report-generation turned on (F1 0.532 -> 0.525), which the authors attribute to fully-finetuned medical LLMs losing instruction-following ability; MedAlpaca (PEFT) does not regress. Reporting this rather than burying it is the most credibility-building decision in the paper.

### Generalist-VLM comparison (Table V, zero-shot)

| Method | C-BLEU | ROUGE-L | METEOR | PRE | REC | F1 |
|---|---|---|---|---|---|---|
| PMC-VQA | / | 5.978 | 2.685 | 0.221 | 0.109 | 0.115 |
| RadFM | / | 7.366 | 4.756 | 0.487 | 0.095 | 0.156 |
| **ChatCAD+** | **4.409** | **17.344** | **24.337** | **0.531** | **0.615** | **0.564** |

The slash on C-BLEU indicates the generalist VLMs produce very short outputs and the metric collapses, so this is a rhetorical gap rather than a fair head-to-head.

### Domain identification (Fig. 5)

| Backbone | 7-domain accuracy | 9-domain accuracy |
|---|---|---|
| **BiomedCLIP** | **100%** | **98.9%** |
| Vanilla CLIP | — | 86.3% |
| PMC-CLIP | ~50% at 3 domains | — |

### Hierarchical in-context ablation (Fig. 7)

k=0 -> k=1 yields the biggest jump (any exemplar helps); k=3 ~= k=4 ~= k=5, justifying the default k=3.

### Clinical QA (Table VI)

| Backbone | Config | ACC | F1 | ROUGE-L |
|---|---|---|---|---|
| ChatGLM2 | base | 0.435 | — | 11.396 |
| ChatGLM2 | + Merck DFS | 0.465 | — | 18.957 |
| ChatGLM3 | base | 0.458 | 0.455 | 14.915 |
| **ChatGLM3** | **+ Merck DFS** | **0.472** | **0.505** | **19.302** |

Accuracy gains are 1.4-3 points; explanation ROUGE-L nearly doubles. The authors honestly flag that multi-choice answers are sometimes right "for the wrong reasons" and that retrieval mainly fixes the reasoning chain.

![Report length distribution by k vs. radiologist](/assets/images/paper/chatcad-plus/fig_p009_01.png)
*Figure 3: Report-length distribution as k varies vs. a human radiologist. ChatCAD+ matches the radiologist's length distribution; plain ChatCAD overshoots.*

### Qualitative case studies

![Dental panoramic X-ray case (periodontitis)](/assets/images/paper/chatcad-plus/fig_p007_01.png)
*Figure 4a: Panoramic dental X-ray used in the periodontitis case study from Fig. 6.*

![Chest X-ray case (pleural effusion / atelectasis)](/assets/images/paper/chatcad-plus/fig_p007_02.png)
*Figure 4b: Semi-erect portable chest X-ray used in the pleural-effusion / atelectasis case study from Fig. 6.*

![Knee MRI case (cartilage defect)](/assets/images/paper/chatcad-plus/fig_p007_03.png)
*Figure 4c: Knee MRI slice mosaic used in the cartilage-defect case study from Fig. 6.*

![Chest X-ray qualitative comparison (top row)](/assets/images/paper/chatcad-plus/fig_p010_01.png)
*Figure 5a: Qualitative report-generation comparison on a chest X-ray (Fig. 9, top row): ground-truth vs. R2GenCMN vs. ChatCAD vs. ChatCAD+.*

![Chest X-ray qualitative comparison (middle row)](/assets/images/paper/chatcad-plus/fig_p010_02.png)
*Figure 5b: Qualitative report-generation comparison (Fig. 9, middle row).*

![Chest X-ray qualitative comparison (bottom row)](/assets/images/paper/chatcad-plus/fig_p010_03.png)
*Figure 5c: Qualitative report-generation comparison (Fig. 9, bottom row).*

For the LLM-DFS-vs-LangChain comparison, the authors color-code retrieval relevance (green = related/important, purple = partial, red = unrelated) and show LangChain's paragraph-split SBERT pulling the wrong topic entirely (e.g., palmoplantar keratodermas instead of periodontitis treatment), while the DFS method lands in the correct "treatment" subsection. Four queries, no aggregate number.

## Limitations

**Authors acknowledge.**

- Report quality depends on having an in-distribution report database; MIMIC-CXR is the privileged source here.
- The recursive LLM-as-DFS is slow — bounded by OpenAI API latency, and recursion blows up token usage (which is why GPT-4 was not used for CMExam).
- Merck Manuals is one source; broader knowledge would help, and the closed license is a productionization risk.

**Not addressed.**

- No clinician evaluation of report quality or answer safety. "Reliability" is asserted rather than measured.
- No hallucination rate against retrieved evidence — i.e., how often the LLM contradicts the Merck snippet it was given.
- The retrieval ablation never tests the obvious baseline: dense sentence embeddings over the full Merck text (e.g., BGE, BiomedBERT-base) at the topic level rather than the chunked paragraph level used by LangChain.
- The CMExam setup is cross-lingual (Chinese questions, English Merck) and the paper never analyzes how often retrieval fails because of the language mismatch.
- The k=3 in-context retrieval is run over MIMIC-CXR train against a 1k MIMIC-CXR test subset, with no certification that retrieved exemplars are not near-duplicates of the test report — a leakage risk the paper does not address.
- Latency, throughput, and cost are mentioned but never quantified.
- No safety filter or refusal evaluation, despite the system being marketed for direct patient interaction.

## Why It Matters for Medical AI

ChatCAD+ is best read as a careful systems paper, not a modeling paper. Its lasting contributions are two patterns that have since spread across medical-LLM agents.

- **Specialist + LLM orchestration as an alternative to generalist medical VLMs.** In 2023-24 the field was choosing between (a) train one giant generalist (PMC-VQA, RadFM, BiomedGPT) and (b) ensemble specialists and use the LLM as a controller. ChatCAD+ argues for (b) on the practical grounds that domain coverage is easier to add by plugging in a new CAD model than by curating a new pretraining corpus. The asymmetry between Table V's headline numbers and the (b)-side's actual coverage (3 of 9 domains) shows the cost of that bet: orchestration scales cheaply along the LLM axis but linearly along the specialist axis.
- **LLM-as-DFS over a structured knowledge tree as an alternative to dense-vector RAG.** The Merck-tree retriever is the conceptual contribution that has aged best — modern medical-agent systems routinely walk taxonomies (ICD-10, MeSH, UMLS, Merck) rather than chunk-and-embed unstructured text. The honest weakness is that the paper validates it only on hand-picked qualitative cases; a held-out question -> section gold standard with retrieval@k would have made this the strongest claim in the paper instead of the weakest.

The PMC-LLaMA regression is also worth dwelling on: fully-finetuned medical LLMs lose instruction-following capacity, which means RAG-style retrieval pipelines actually need *general*-instruction-tuned backbones to work. That is a non-obvious empirical lesson that has held up across subsequent medical-agent work.

## References

- Paper: ChatCAD+: Toward a Universal and Reliable Interactive CAD using LLMs — Zhao et al., IEEE Transactions on Medical Imaging 2024. arXiv: [2305.15964](https://arxiv.org/abs/2305.15964)
- Prior work (same authors): ChatCAD — arXiv [2302.07257](https://arxiv.org/abs/2302.07257)
- Domain encoder: BiomedCLIP — Zhang et al., 2023
- Report-generation baseline: R2GenCMN — Chen et al., 2021
- Knowledge base: Merck Manuals Professional Edition
- Benchmarks: MIMIC-CXR (Johnson et al.), CMExam (Liu et al.)
