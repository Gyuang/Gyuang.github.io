---
title: "CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models"
excerpt: "A 3-page EICS Companion '25 demo that bolts a CheXagent + Mistral concept bottleneck onto a five-agent CrewAI/LlamaIndex RAG stack for chest X-ray reporting — no quantitative evaluation in the PDF."
categories:
  - Paper
tags:
  - CBM-RAG
  - Concept-Bottleneck-Model
  - Retrieval-Augmented-Generation
  - Multi-Agent
  - CheXagent
  - Radiology-Report-Generation
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- CBM-RAG is an **EICS Companion '25 3-page demo paper** that wires a Concept Bottleneck Model (CheXagent image embeddings + Mistral-embedded LLM-proposed concepts + a linear classifier) into a five-agent CrewAI/LlamaIndex RAG stack (per-disease ReAct agents + Radiologist Agent + Report Writer Agent + Chat Agent) for chest X-ray reporting on COVID-QU (Pneumonia / COVID-19 / Normal).
- The **interpretability hook is on the vision side**: per-concept contribution scores (weight x concept-vector cross product) and concept-level saliency heatmaps are surfaced as an editable, toggleable list in a Streamlit UI, with a chain-of-thought dropdown that exposes the multi-agent reasoning trace.
- **Headline result: there is none.** This PDF reports **no quantitative numbers** — no classification accuracy, no BLEU/ROUGE/CheXbert F1, no hallucination measurement, no usability study. The authors explicitly defer formal evaluation to future work and point to a companion ECIR 2025 paper for the underlying methodology.

## Motivation

LLM-driven chest X-ray report generators reduce radiologist workload but compound two failure modes: **hallucination** (the LLM invents findings unsupported by the image) and **opacity** (no traceable link from pixel evidence to textual claim). Retrieval-Augmented Generation patches the first by grounding text in external knowledge, but leaves the image encoder and the LLM as black boxes. The authors' argument is that clinical trust additionally requires transparency at the **image-understanding** stage — which clinical concept ("pulmonary consolidation," "nodule") drove a Pneumonia vs. COVID-19 vs. Normal prediction, and which retrieved guideline supports each sentence in the generated report. CBM-RAG's pitch is to combine concept bottlenecks on the vision side with multi-agent retrieval on the language side, then expose the whole chain in an auditable UI for a radiologist user.

## Core Innovation

- **LLM-proposed concept bottleneck on a CXR VLM.** Concepts come from an LLM prompt (no manual concept annotation), are embedded with **Mistral embed**, and matched by cosine similarity to **CheXagent** image embeddings; max-pooling collapses the similarity matrix to a normalized concept vector that drives a sparse linear three-class head.
- **Per-concept contributions and heatmaps as first-class UI objects.** The cross-product of the classifier weight matrix and the concept vector yields per-concept contribution scores; spatial similarity maps (pre-pooling) give a heatmap per concept. Both are exposed in Streamlit as an editable, sortable list — supporting CBM's classic "test-time intervention" property.
- **Five-agent RAG back-end.** One ReAct agent per disease class with its own vector store, a Radiologist Agent that interprets the CBM output and dispatches, a Report Writer Agent that synthesizes findings/diagnosis/guidelines, and a Chat Agent for follow-up — orchestrated by **CrewAI** over **LlamaIndex** indices, with user-uploaded PDFs/PPTs/text/audio/video (Whisper-transcribed) folded into the same retrieval pool.

## Claims & Evidence Analysis

| # | Claim | Evidence in this PDF | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | CBM + multi-agent RAG produces **transparent disease classification** by mapping CXR features to human-readable clinical concepts. | Architecture diagram (Fig. 1) and UI description; no concept-faithfulness / concept-accuracy / deletion-insertion metric. | COVID-QU | ⭐ |
| C2 | The system **mitigates hallucinations**. | Asserted in abstract and conclusion; mechanism plausible (RAG grounding + concept conditioning) but no hallucination metric and no non-RAG baseline. | none | ⭐ |
| C3 | The system **generates high-quality, tailored reports**. | No BLEU/ROUGE/CIDEr/CheXbert/RadGraph numbers; no human evaluation. | none | ⭐ |
| C4 | LLM-proposed concept sets are usable for CXR classification on COVID-QU. | Methodology reused from [1, 11, 16]; COVID-QU accuracy is not reported here. | COVID-QU | ⭐ (defers to companion ECIR paper) |
| C5 | Per-concept contribution scores and saliency heatmaps localize evidence. | Mathematically defined (weight x concept vector; spatial similarity map); no faithfulness audit (deletion test, pointing game vs. radiologist boxes). | COVID-QU | ⭐⭐ — well-defined, empirically unverified here |
| C6 | Multi-agent (ReAct + Radiologist + Report Writer + Chat) orchestration beats single-agent generation. | Not measured. No ablation, no single-LLM RAG baseline. | none | ⭐ |
| C7 | The interactive UI improves radiologist trust / diagnostic consistency. | Asserted; explicitly: "formal usability studies in real clinical settings are yet to be conducted." | none | ⭐ |
| C8 | Framework supports multimodal user inputs (PDF/PPT/text/audio/video) via Whisper + embedding. | Implementation described; public Streamlit demo and GitHub repo. No latency / retrieval-precision numbers. | none | ⭐⭐ — existence-of-feature claim, demo is public |
| C9 | The approach is end-to-end and clinically deployable. | Public demo + repo substantiate *existence*; clinical deployability not demonstrated. | none | ⭐ |

**Honest read.** This is a demo / companion paper, not an evaluation paper. Every empirical claim in the abstract — interpretability, hallucination mitigation, report quality, trust — appears without quantitative support. The architecture is clear enough to reproduce, the code and Streamlit demo back the *existence* claims, but the substantive evaluation lives in the companion ECIR 2025 paper by an overlapping author set [1]. The audit therefore returns almost uniformly 1-star ratings — not because the system is unsound, but because **this venue is a demo track and the paper does not attempt evaluation**.

## Method & Architecture

![CBM-RAG end-to-end workflow](/assets/images/paper/cbm-rag/fig_p002_01.png)
*Figure 1: CBM-RAG end-to-end workflow. Top: a CheXagent VLM + Mistral-embedded concept set produces a similarity vector that drives the three-class disease classifier and per-concept heatmaps. Bottom: per-disease ReAct agents + a Radiologist Agent + a Report Writer Agent + a Chat Agent ground the report in an NIH vector store plus user-uploaded media (PDF/PPT/text/audio/video, with audio/video transcribed by Whisper).*

**A. Concept Bottleneck for CXR classification**

1. **Concept set construction.** Following label-free CBM work, an LLM is prompted to propose clinically relevant concepts (e.g., "pulmonary consolidation," "nodule") — no human concept annotation.
2. **Image embedding.** The uploaded CXR is encoded by **CheXagent**, a VLM fine-tuned for chest X-ray interpretation.
3. **Concept text embedding.** Each concept string is embedded by **Mistral embed**.
4. **Similarity matrix.** Cosine similarity between the image embedding and every concept embedding.
5. **Concept vector.** Max-pooling collapses the similarity matrix to a single vector; values are normalized to $[0, 1]$.
6. **Linear classifier head.** A fully-connected layer over the concept vector predicts Pneumonia / COVID-19 / Normal, trained on COVID-QU.
7. **Per-concept contribution.** The cross-product of the trained weight matrix and the concept vector ranks concepts by their contribution to the decision.
8. **Saliency heatmaps.** Pre-pooling spatial similarity maps give a heatmap per concept on the original CXR.
9. **Editable concepts in the UI.** Concepts are sorted by $|\text{contribution}|$, each toggleable; users can manually adjust concept scores and re-run classification (CBM test-time intervention).

**B. Multi-agent RAG for report generation**

10. **Disease-specific ReAct agents.** Pneumonia Agent, COVID-19 Agent, Normal Agent — one ReAct agent per class, each with its own LlamaIndex vector store of clinical documentation.
11. **Radiologist Agent.** Reads the CBM output, dispatches to the appropriate disease agent, and queries a pre-configured NIH document database.
12. **Report Writer Agent.** Synthesizes the final structured radiology report — findings, diagnosis, guidelines.
13. **User-content ingestion.** PDFs, PPTs, text, MP3, MP4 are accepted at inference time; audio/video is transcribed by **OpenAI Whisper**, embedded, and added to a vector store alongside the NIH corpus.
14. **Chat Agent.** Real-time conversational interface over the CXR, the generated report, and the retrieval stores.
15. **Stack.** **CrewAI** for multi-agent orchestration, **LlamaIndex** for retrieval/indexing, **Streamlit** for the UI.
16. **Reasoning trace.** An optional chain-of-thought dropdown in the UI exposes the sequential reasoning across agents.

The paper does not report training losses, optimizer settings, learning rates, batch sizes, or epochs for the CBM head.

## Experimental Results

This is a 3-page companion/demo paper. **It reports no quantitative tables, no metric numbers, no ablations, no baseline comparisons.** The "results" are entirely the existence and behavior of the interactive system.

| Method | Metric | Dataset | Value |
|---|---|---|---|
| **CBM-RAG (this paper)** | classification accuracy / AUC / F1 | COVID-QU | **Not reported** |
| **CBM-RAG (this paper)** | report-generation metrics (BLEU/ROUGE/CIDEr/CheXbert F1) | — | **Not reported** |
| **CBM-RAG (this paper)** | hallucination rate | — | **Not reported** |
| **CBM-RAG (this paper)** | usability / radiologist trust scores | — | **Not reported (deferred to future work)** |

Qualitative behavior described in the paper:

- The identified concepts are shown as an editable list sorted by $|\text{contribution}|$; toggling a concept reveals its saliency heatmap on the CXR.
- Users can adjust concept scores and refine the disease prediction at test time.
- The Report Writer Agent produces a structured report containing findings, diagnosis, and guidelines, with an optional chain-of-thought dropdown that exposes the multi-agent reasoning sequence.
- The Chat Agent supports real-time follow-up queries grounded in the same retrieval stores.

The authors point to a longer companion work (ECIR 2025, Springer LNCS 15574, pp. 201-209) for the underlying methodology — quantitative evaluation, if it exists, lives there, not in this demo paper.

## Limitations

**Authors' admitted limitations**

- "Formal usability studies in real clinical settings are yet to be conducted." Future work mentions user evaluations, extension to other imaging modalities, and broader healthcare applications.

**Unaddressed by the authors**

- **No quantitative results of any kind in this PDF** — classification, generation, hallucination, latency, or cost.
- **Narrow disease space.** COVID-QU's three classes (Pneumonia / COVID-19 / Normal) do not exercise the breadth of CXR pathology (atelectasis, effusion, pneumothorax, mass, etc.). The one-ReAct-agent-per-class design does not obviously scale to the 14+ classes in ChestX-ray14 / MIMIC-CXR.
- **Unvalidated concept set.** LLM-proposed concepts are used without clinician review reported here; spurious or redundant concepts could inflate apparent interpretability.
- **Faithfulness of contribution scores.** The linear-readout x concept-vector contribution is mathematically defined but not audited against radiologist attention or against intervention outcomes.
- **VLM choice as a confound.** Using CheXagent as the image encoder bakes in whatever biases CheXagent has; the CBM does not eliminate that — it only re-expresses it in concept space.
- **Retrieval governance.** "A pre-configured NIH database" is under-specified — which documents, how indexed, how freshness is maintained, how conflicting guidelines are reconciled.
- **Hallucination claim is unfalsified.** RAG mitigates but does not eliminate hallucination; without measurement, the abstract's "mitigate hallucinations" is aspirational.
- **No latency / cost.** A CrewAI stack with Whisper + multiple ReAct loops + retrieval is non-trivial, and interactive radiology demands sub-minute responses.
- **Privacy.** Uploaded patient PDFs/audio embedded into a vector store has clear HIPAA/GDPR implications that are not discussed.
- **Reproducibility of evaluation.** Because no evaluation is run here, there is nothing empirical to reproduce — only the system to deploy.

## Why It Matters for Medical AI

The product idea — an editable, auditable concept bottleneck wired to a multi-agent RAG report writer, all surfaced in a Streamlit UI a radiologist could actually click through — is the right shape for clinical deployment of LLM-based reporting. Concept-level heatmaps plus a chain-of-thought trace give a radiologist somewhere to push back, which is exactly what a black-box generative system denies them. The caveat is that **this paper does not measure any of the properties it advertises**. A medical-AI reader should treat CBM-RAG as a UI/architecture proposal worth replicating against MIMIC-CXR / IU X-Ray with proper report-generation metrics (BLEU-4, ROUGE-L, CheXbert/RadGraph F1), an entity-grounding hallucination audit (e.g., RadGraph F1 or RaTEScore vs. a non-RAG baseline), a concept-faithfulness audit (deletion test, intervention sensitivity, pointing game vs. radiologist bounding boxes), and a real radiologist usability study with inter-rater reliability — the same gaps the authors themselves flag.

## References

- **Paper:** Alam, H. M. T., Srivastav, D., Selim, A. M., Kadir, M. A., Shuvo, M. M. H., Sonntag, D. *CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models.* EICS Companion '25, Trier, Germany. DOI: [10.1145/3731406.3731970](https://doi.org/10.1145/3731406.3731970). arXiv: [2504.20898v2](https://arxiv.org/abs/2504.20898).
- **Code:** [github.com/tifat58/enhanced-interpretable-report-generation-demo](https://github.com/tifat58/enhanced-interpretable-report-generation-demo)
- **Live demo:** [cxr-cbm-rag-dfki-iml-demo.streamlit.app](https://cxr-cbm-rag-dfki-iml-demo.streamlit.app/)
- **Companion methodology paper:** ECIR 2025, Springer LNCS 15574, pp. 201-209.
- **Building blocks:** CheXagent (image encoder); Mistral embed (text encoder); CrewAI (multi-agent orchestration); LlamaIndex (retrieval); OpenAI Whisper (audio/video transcription); ReAct (Yao et al.).
- **Dataset:** Chowdhury et al., *COVID-QU* (IEEE Access, 2020) — 33,920 chest X-ray images across Pneumonia / COVID-19 / Normal.
