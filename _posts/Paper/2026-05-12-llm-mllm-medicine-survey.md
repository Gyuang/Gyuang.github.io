---
title: "A Comprehensive Survey of Large Language Models and Multimodal Large Language Models in Medicine"
excerpt: "A 32-page taxonomy that decomposes medical MLLMs along vision encoder x LLM backbone x four-family modality alignment, catalogs ~30 medical LLMs and ~25 medical MLLMs with their datasets and fine-tuning recipes, but quotes headline numbers (Med-PaLM 2 86.5 MedQA, GPT-4V 90.7% USMLE-image) without methodological critique and misses the late-2024 reasoning/agentic wave."
categories:
  - Paper
tags:
  - Survey
  - Medical-LLM
  - Medical-MLLM
  - Modality-Alignment
  - Instruction-Tuning
  - RLHF
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- A 32-page taxonomy paper that frames the field as **five paradigm shifts** (feature -> structure -> objective -> prompt -> data engineering), then decomposes medical MLLMs along three orthogonal axes: **vision encoder x LLM backbone x modality-alignment family** (GATED XATTN-DENSE, Query-Based, Projection-Based, Prompt Augmentation). Tables 1-2 catalog ~30 medical LLMs and ~25 medical MLLMs; Table 3 catalogs medical datasets across six types.
- **What it adds over prior medical-LLM surveys** (He et al. 2023, Zhou et al. 2023, Thirunavukarasu et al. 2023): the technical scaffolding is the contribution. Earlier surveys lead with applications; this one leads with architecture and fine-tuning recipes (CPT / IFT / SFT / RLHF / RLAIF / DPO) and gives medical MLLMs first-class treatment.
- **Headline numbers quoted without scrutiny.** Med-PaLM 2 **86.5 on MedQA-USMLE**, GPT-4V **90.7% on USMLE image questions**, miniGPT-Med **+19% report-generation accuracy** are reported verbatim. Fig. 8's "medical LLM/MLLM beats traditional DL" panel is a stylized illustration, not a meta-analysis (no error bars, no significance tests, no normalization).

## Motivation

General LLMs and MLLMs (GPT-4, LLaMA, Flamingo, LLaVA) entered medicine in 2023-2024, but hospital adoption is blocked by (a) data scarcity and annotation cost, (b) compute cost for training and serving 7B-540B parameter models, (c) safety / ethics evaluation beyond benchmark accuracy, and (d) hallucination plus stale knowledge. Existing medical-LLM surveys either focus on text-only LLMs (He et al. 2023, Zhou et al. 2023) or stay at the application layer without technical depth. The authors position this survey to fill **both** gaps simultaneously: a technical recipe book *and* a clinical-application map, with medical MLLMs given equal treatment because medicine is inherently multimodal.

![Medical LLM/MLLM construction and evaluation pipeline](/assets/images/paper/llm-mllm-medicine-survey/page_002.png)
*Figure 1: End-to-end construction and evaluation pipeline for medical LLMs and MLLMs - data sources feed CPT and IFT/SFT/RLHF, which then route through automatic, human, and AI-as-judge evaluation. This is the survey's organizing diagram.*

## Core Innovation

The survey's novelty is **organizational**, not methodological. Three structural choices set it apart:

- **Five-stage paradigm framing** (Fig. 3). NLP is recast as a sequence of *feature engineering* (supervised), *structure engineering* (BERT/GPT-1 pretrain+FT), *objective engineering* (GPT-3 prompt), *prompt engineering* (Flamingo multimodal), and most recently *data engineering* (LIMA). The thesis is that data engineering is the next axis - a framing that anchors the dataset chapter.
- **Four-family modality-alignment taxonomy** for medical MLLMs (Section 3.2). This is the most useful contribution: rather than treating modality alignment as a single black box, the paper separates (i) **GATED XATTN-DENSE Layers** (Flamingo, Med-Flamingo), (ii) **Query-Based** (Q-Former in MedBLIP, XrayGLM, RadFM, CheXagent), (iii) **Projection-Based** (linear/MLP in LLaVA-Med, XrayGPT, Med-PaLM M, MAIRA-1/2, HuatuoGPT-Vision), and (iv) **Prompt Augmentation** (ChatCAD, ChatCAD+, OphGLM, Visual Med-Alpaca - expert-model outputs fed as text). This is the scaffolding readers will keep.
- **Strict IFT vs. SFT distinction.** Many medical-LLM papers conflate the two; this survey insists IFT is for instruction-following / zero-shot transfer, SFT is for task-specific datasets. The distinction matters for evaluating training-cost claims and for understanding why some models generalize while others memorize a benchmark.

![Five-stage paradigm shift](/assets/images/paper/llm-mllm-medicine-survey/page_004.png)
*Figure 2: Five-stage evolution of NLP from supervised learning through prompt-based unsupervised pretraining to multimodal and high-quality-data eras. GPT-3 is positioned as the start of "LLMs"; Flamingo is positioned as the start of "MLLMs"; LIMA anchors the data-engineering thesis.*

## Claims & Evidence Analysis

| # | Claim | Evidence cited | Strength | Notes |
|---|---|---|---|---|
| C1 | Decoder-only is the dominant LLM architecture | Wang et al. [89], Dai et al. [90]; Tables 1-2 show decoder-only is the most populated row | ⭐⭐⭐ | Well-established and reproducible from the cataloged literature. |
| C2 | Contrastive-pretrained ViTs (CLIP-ViT) outperform classification-pretrained ViTs as MLLM encoders | Chen et al. [95], cited once | ⭐⭐ | Single external citation; no head-to-head data in the survey itself. |
| C3 | Medical-pretrained vision encoders > natural-image-pretrained encoders for medical MLLMs | LLaVA-Med [18], MAIRA-1 [96], PathChat [97] | ⭐⭐ | Directional claim, no ablation table - effect size not measured here. |
| C4 | Q-Former is "an inefficient visual token compressor; adaptive average pooling outperforms it" | Yao et al. [126] | ⭐⭐ | Single recent paper; treated as settled but actually contested. |
| C5 | Med-PaLM 2 outperforms physicians on consumer medical Q's (86.5 MedQA) | [13] | ⭐⭐ | Repeated without the well-known caveats - rater pool, prompt sensitivity, leakage concerns. |
| C6 | miniGPT-Med achieves **+19% accuracy** on report generation | [118] | ⭐ | Single benchmark, single-run number, not externally validated. |
| C7 | LLM-based mental-health chatbots can substantially lower cost and improve accessibility | [15, 22, 23, 189-191] | ⭐⭐ | Multiple supporting refs, but mostly pilot studies - no clinical-trial-grade evidence. |
| C8 | AI-as-judge (GPT-4) matches human evaluation in most NLG tasks | Wang et al. [168] | ⭐⭐ | The survey itself flags GPT-4's positional, length, and self-preference biases - simultaneously endorses and undermines the claim. |
| C9 | RAG mitigates hallucination | [16, 110, 198] | ⭐⭐ | Conceptually sound, but residual hallucination rates after RAG in medical settings are never quantified. |
| C10 | Edge deployment via 6G MEC is a viable path | Lin et al. [226] | ⭐ | Speculative - 6G is not deployed; this is a future-direction paragraph, not evidence. |
| C11 | **No existing medical MLLM uses RLHF/RLAIF/DPO** | Self-derived from Table 2 | ⭐⭐⭐ | Verifiable from the catalog - only medical *LLMs* (HuatuoGPT, Zhongjing, Qilin-Med, ClinicalGPT) use preference alignment. Useful and honest observation as of the v3 cutoff. |
| C12 | "Data engineering" is the dominant emerging paradigm | LIMA [38], MedTrinity-25M [56] | ⭐⭐ | Plausible trend; two anchor citations is thin for a framing thesis. |

**Honest assessment.** The survey is genuinely useful as (a) a **catalog** - Tables 1-3 are reference material - and (b) a **structural decomposition** of medical MLLMs along the vision-encoder x backbone x alignment-family axes. Sections 3.2 (alignment families), 4.1 (CPT/IFT/SFT), and 4.2 (RLHF/RLAIF/DPO) are technically sound. The weak parts are concentrated in Section 5 and Fig. 8:

1. **Results are reported uncritically.** Med-PaLM 2 86.5 on MedQA-USMLE is quoted without flagging MedQA's well-documented pretraining-corpus contamination, the rater-pool composition of the consumer Q&A evaluation, or the prompt-sensitivity issue. miniGPT-Med +19% is a single number from a single paper.
2. **Fig. 8 is illustrative, not meta-analytic.** The "LLM/MLLM beats traditional DL" comparison has no error bars, no significance tests, no per-dataset normalization. It is a thesis statement rendered as a bar chart.
3. **The data-engineering thesis is rhetorical scaffolding.** LIMA + MedTrinity-25M is a thin evidentiary base for declaring a new paradigm.
4. **Major late-2024 omissions** (see Limitations) undercut the "comprehensive" framing.

## Method & Architecture

![MLLM core pipeline: vision encoder, alignment module, LLM backbone](/assets/images/paper/llm-mllm-medicine-survey/page_007.png)
*Figure 3: The MLLM architecture diagram that anchors Section 3.2 - image goes through a vision encoder V to image features Z_x, the alignment module converts Z_x into LLM-compatible tokens H_x, and the LLM backbone produces response R = L(H_x, T_x). The survey's four-family alignment taxonomy maps onto the middle box.*

### 1. Architectural taxonomy of medical LLMs (Section 3.1, Table 1)

The Table 1 catalog distinguishes three architectural families. Encoder-only models (BioBERT, PubMedBERT) are explicitly **excluded** as "pretrained language models, not LLMs." The Decoder-only column is by far the most populated (Med-PaLM, ChatDoctor, MEDITRON, HuatuoGPT, Zhongjing, Qilin-Med, ClinicalGPT, Apollo, PediatricsGPT, ...). The Encoder-Decoder column collects mainly Chinese clinical chatbots (DoctorGLM, BianQue, SoulChat) built on ChatGLM.

### 2. Architectural taxonomy of medical MLLMs (Section 3.2, Table 2)

Three axes:

- **Vision encoder.** The historical arc is ResNet -> ViT -> CLIP-ViT / EVA-CLIP-ViT. The survey's stated preference order is *contrastive > classification-pretrained* and *medical-pretrained > natural-image-pretrained*. Most current medical MLLMs use CLIP-ViT-L/14 or BiomedCLIP.
- **LLM backbone.** LLaMA / Vicuna / Mistral / Qwen / ChatGLM / Phi-2 / Yi. No surprises.
- **Modality alignment - the four families.** This is the load-bearing taxonomy:

| Family | Representative medical MLLMs | Mechanism |
|---|---|---|
| GATED XATTN-DENSE | Flamingo, **Med-Flamingo** | Gated cross-attention layers inserted between frozen LM blocks; image tokens attend in via gated residual. |
| Query-Based | MedBLIP, XrayGLM, RadFM, CheXagent | A learned Q-Former with ~32 queries compresses image features into a short token sequence. |
| Projection-Based | **LLaVA-Med**, XrayGPT, Med-PaLM M, MAIRA-1/2, HuatuoGPT-Vision | Linear or MLP projection from vision token dim to LM embedding dim - the cheap, currently-winning recipe. |
| Prompt Augmentation | ChatCAD, ChatCAD+, OphGLM, Visual Med-Alpaca | Image is processed by an expert model (segmenter, classifier, captioner); the expert's textual output is concatenated to the LM prompt. |

The implicit conclusion the survey nudges the reader toward: **Projection-Based is the present default** (simple, trains fast, ablates well) and **Query-Based is being challenged** (citing Yao et al. on adaptive average pooling).

### 3. Training recipe (Section 4)

The fine-tuning stack is laid out as a pipeline: **CPT** (continued pretraining injects medical knowledge) -> **IFT** (instruction-following, zero-shot transfer) -> **SFT** (task-specific specialization) -> **RLHF / RLAIF / DPO** (preference alignment). The crucial empirical observation from Table 2: **no medical MLLM** currently uses RLHF/RLAIF/DPO - preference alignment in medicine has so far been a text-only enterprise (HuatuoGPT, Zhongjing, Qilin-Med, ClinicalGPT).

### 4. Evaluation (Section 4.3)

Three families. **Automatic** - Accuracy, BLEU-1..4, ROUGE-N/L/W/S, GLEU, Distinct-n, CIDEr, BERTScore. **Human** - helpfulness, safety, ethics, comprehensiveness. **AI-as-judge** - GPT-4 evaluation, with the survey explicitly noting GPT-4's documented biases (prefers the first response, prefers longer responses, prefers self-generated responses).

![Three evaluation approaches and their trade-offs](/assets/images/paper/llm-mllm-medicine-survey/page_015.png)
*Figure 4: Trade-offs across automatic metrics, human evaluation, and AI-as-judge. Automatic metrics are cheap and reproducible but shallow; human evaluation is comprehensive but expensive; AI-as-judge is scalable but biased. The survey endorses AI-as-judge and immediately enumerates its biases - a tension worth flagging.*

## Experimental Results

The survey reports no experiments of its own. The quoted results, with sources, are:

| Model | Reported result | Benchmark | Source | Caveat |
|---|---|---|---|---|
| **Med-PaLM 2** | 86.5 | MedQA-USMLE | [13] | Pretraining-corpus contamination not addressed. |
| **GPT-4V** | 90.7% | USMLE image questions | [180] | Single evaluation, no clinical replication. |
| **miniGPT-Med** | +19% accuracy | Report generation | [118] | Single benchmark, single run. |
| **Med-PaLM M** | matches/exceeds SoTA on 14 tasks | 14 medical tasks | [21] | Mixed task definitions, not a unified benchmark. |
| **LIMA** | beats Alpaca / Bard with 1,000 curated prompts | General benchmarks | [38] | Anchor citation for the "data engineering" thesis. |
| **LLaVA-Med + MedTrinity-25M** | ~10% avg gain | 3 biomedical VQA datasets | [56] | Specific gain breakdown not shown in this survey. |

![Medical LLM/MLLM vs. traditional deep learning](/assets/images/paper/llm-mllm-medicine-survey/page_016.png)
*Figure 5 (the controversial one): the survey's Fig. 8 - claimed superiority of medical LLMs/MLLMs over traditional DL on QA and VQA. There are no error bars, no significance tests, no normalization across datasets. Read it as a thesis statement, not as evidence.*

### Dataset coverage (Table 3 - the actual contribution)

Six dataset types are cataloged. Selected entries:

| Type | Representative datasets | Scale |
|---|---|---|
| **EHR** | MIMIC-III, MIMIC-IV, CPRD | ~2M notes; 11.3M patients (CPRD) |
| **Literature** | PubMed (~4.5B words), PMC (~13.5B words), CORD-19 (140K papers) | corpus-scale |
| **QA** | MedQA-USMLE (61K), MedMCQA (194K), PubMedQA (612K unlabeled), MultiMedQA, Medical Meadow (160K), Huatuo-26M (26M), Psych8k, MedQuAD (47K) | 8K -> 26M pairs |
| **Dialogue** | HealthCareMagic-100K, MedDialog (3.4M Ch / 0.6M En), GenMedGPT-5K, CMtMedQA (70K) | up to 3.4M turns |
| **Image-text (MLLM)** | MIMIC-CXR (227K), CheXpert (224K), OpenI (7.4K), ROCO (81K), OpenPath (208K), PathCap (142K), MedMD (15.5M 2D + 180K 3D), PMC-OA (1.6M), PMC-15M (15M), MedTrinity-25M, LLaVA-Med-Alignment (600K), ChiMed-VL (580K), PubMedVision (647K) | up to 25M |
| **Instruction** | MedC-I (202M tokens), MedInstruct-52K, LLaVA-Med-Instruct (60K), ChiMed-VL-Instruction (469K), PathInstruct (180K), CheXinstruct (28 datasets) | 52K -> 469K pairs |

A large fraction of the instruction-following corpora are **GPT-4-synthesized**, which the survey flags but does not quantify. Privacy and licensing (MIMIC's PhysioNet credentialing, e.g.) are mentioned in Section 6.4 but not at the per-dataset level.

## Limitations

**Author-acknowledged.** Hallucination, training and deployment cost, knowledge staleness, privacy, bias and toxicity (Section 6 covers all five competently). Most medical MLLMs only support vision+text; time-series, audio, and 3D imaging are open (Section 7.3). No medical MLLM currently uses RLHF/RLAIF/DPO (Table 2).

**Author did not address (or addressed only weakly).** This is where the "comprehensive" framing strains the most against the December-2024 v3 cutoff:

- **Recency vs. comprehensiveness.** The v3 covers up to SigPhi-Med (Oct 2024) but omits or under-covers Med-Gemini, Gemini-Med, **GPT-4o medical evaluations**, MedSAM-2, BiomedParse, MedDr, AnatomySketch-Med, RadFM follow-ups, **Llama-3-based medical models**, Claude-3-Opus medical benchmarks, and Janus-Med. Open-source instruction datasets released late 2024 are only partially captured.
- **Reasoning-style models.** No discussion of medical chain-of-thought or **o1-style reasoning** despite their dominant late-2024 traction on MedQA / MedMCQA leaderboards. A significant omission for a Dec-2024 survey.
- **Agentic / tool-use systems.** "Medical agents" appear only as a future-direction bullet. **MedAgents**, **ClinicalAgent**, and **AgentClinic** frameworks - which were among the most-discussed medical-AI artifacts of 2024 - are not cataloged.
- **Retrieval-augmented systems.** RAG is mentioned for hallucination mitigation but never given its own taxonomy section despite being one of the dominant 2024 medical-AI design patterns (ChatCAD+ being the survey's own example).
- **Evaluation rigor.** No unified benchmark proposed. No discussion of MedQA / USMLE contamination of pretraining corpora. No mention of calibration / uncertainty quantification.
- **Real-world deployment evidence.** Section 5 lists applications but cites no FDA-cleared deployments and no prospective clinical trials of medical LLMs.
- **Language and equity.** The bias section does not address language equity or low-resource medical AI despite the survey itself cataloging many Chinese-focused models (Qilin-Med, HuatuoGPT, Zhongjing, TCM-GPT, PediatricsGPT, MedChatZH, Apollo).

![Open challenges and future directions](/assets/images/paper/llm-mllm-medicine-survey/page_022.png)
*Figure 6: The survey's Section 6/7 organizing diagram - hallucination, deployment cost, recency, privacy, and bias on the challenges side; edge deployment (6G MEC), medical agents, and "generalist medical assistant" on the future-directions side. The challenges half is well-supported; the future-directions half is speculative.*

## Why It Matters for Medical AI

Read this paper for two things:

1. **The catalogs.** Tables 1, 2, 3 are the most useful artifact - a quick reference for "what models exist, what data they trained on, what alignment module they use." Treat the catalog as the contribution and read past the synthesis prose.
2. **The four-family modality-alignment taxonomy.** GATED XATTN-DENSE / Query-Based / Projection-Based / Prompt Augmentation is a clean mental model for thinking about medical MLLM design choices. Combine it with the observation that **no medical MLLM yet uses RLHF/RLAIF/DPO** and you have a concrete agenda: preference-aligned, projection-based medical MLLMs with retrieval augmentation and an o1-style reasoning trace are the design space the next year of medical-MLLM papers will live in.

Do **not** read this paper for state-of-the-art performance numbers or for a meta-analysis of LLM-vs-traditional-DL on medical tasks - the survey is not doing that work, and Fig. 8 should not be cited as evidence that medical LLMs have surpassed traditional DL.

## References

- arXiv: [2405.08603 v3](https://arxiv.org/abs/2405.08603) (posted 30 Dec 2024)
- Authors: Hanguang Xiao, Feizhong Zhou, Xingyue Liu, Tianqi Liu, Zhipeng Li, Xin Liu, Xiaoxuan Huang (Chongqing University of Technology, School of Artificial Intelligence)
- Cataloged medical LLMs: Med-PaLM / Med-PaLM 2, ChatDoctor, MEDITRON, HuatuoGPT, Zhongjing, Qilin-Med, ClinicalGPT, Apollo, PediatricsGPT, MedChatZH, DoctorGLM, BianQue, SoulChat, ...
- Cataloged medical MLLMs: Med-Flamingo, LLaVA-Med, MedBLIP, XrayGLM, RadFM, CheXagent, XrayGPT, Med-PaLM M, MAIRA-1/2, HuatuoGPT-Vision, ChatCAD, ChatCAD+, OphGLM, Visual Med-Alpaca, miniGPT-Med, ...
- Anchor datasets: PMC-15M, MedTrinity-25M, MIMIC-CXR, MedQA-USMLE, MedMCQA, PubMedQA, LLaVA-Med-Instruct, ChiMed-VL-Instruction
- Related surveys (positioning): He et al. 2023, Zhou et al. 2023, Thirunavukarasu et al. 2023 (text-only medical LLM surveys)
