---
title: "From Perception to Cognition: A Survey of Vision-Language Interactive Reasoning in Multimodal Large Language Models"
excerpt: "A 31-page Renmin/Xiamen/HKUST survey that relabels MLLM progress as Perception (encoder + alignment) vs Cognition (decomposition + dynamic re-look), plots 80+ models on a 2021-2025 timeline, and ranks them on five leaderboards (MathVista/MathVerse/MMMU, VQA-RAD/SLAKE/PathVQA, ChartQA/DocVQA, MVBench/Video-MME, MELD) — Gemini 2.5 Pro tops scientific math, Med-PaLM M tops medical VQA but still trails humans on PathVQA (75.0 vs 85.2), Qwen2.5-VL-72B leads open-source charts/docs."
categories:
  - Paper
  - LLM-Agents
tags:
  - MLLM
  - Vision-Language
  - Reasoning
  - Survey
  - Perception-Cognition
  - Think-with-Image
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
permalink: /paper/perception-to-cognition-vl-reasoning-survey/
---

![2021-2025 MLLM evolution timeline (Figure 1)](/assets/images/paper/perception-to-cognition-survey/page_002.png)
*Figure 1 (Zhou et al., 2025): The 2021-2025 timeline that organizes the survey — a Perception line (CLIP, DINOv2, EVA-CLIP, Qwen2.5-VL, InternVL-2.5, ...) and a Cognition line (Multimodal-CoT, LLaVA-CoT, VLM-R1, DeepEyes, Pixel Reasoner, ...) running in parallel.*

## TL;DR

- A 31-page taxonomic survey (arXiv 2509.25373v4, Oct 2025) that reframes MLLM development as a **two-layer stack** — **Perception** (extracting and aligning fine-grained visual evidence) and **Cognition** (proactive, multi-step observe-think-verify reasoning) — and re-labels every recent MLLM technique as either "fixing perception" or "fixing cognition."
- The analytical contribution is the **Perception-Cognition framework** (Fig. 3) plus a 2021-2025 timeline (Fig. 1) that projects **80+ representative models** (CLIP, Flamingo, LLaVA, Qwen2.5-VL, InternVL-2.5, GPT-o1/o3, DeepEyes, VLM-R1, Chain-of-Spot, MINT-CoT, ...) onto two axes, then collapses cognitive methods into three buckets: problem-decomposition training (SFT / curriculum / DPO / GRPO), automated CoT data synthesis (T2V / V2T / interleaved + bootstrapped + preference), and inference-time tree search (ToT / MCTS).
- Headline leaderboard numbers quoted from primary sources (not produced by the survey): **Gemini 2.5 Pro 84.6 MathVista / 84.6 MathVerse / 73.3 MV-MATH / 82.0 MMMU**; **Med-PaLM M 90.0 VQA-RAD / 90.5 SLAKE / 75.0 PathVQA** (still below the human 85.2 on PathVQA — the only benchmark in any table where no model beats humans); **Qwen2.5-VL-72B 89.5 ChartQA / 96.4 DocVQA / 70.4 MVBench**; **STORM 70.6 MVBench** open-source video. Sentiment tables (MELD, HumanVBench) are sparse — proprietary developers rarely report.

## Motivation

The authors argue that the "how do we hook a vision encoder to an LLM?" question has been largely solved by the Flamingo -> LLaVA -> Qwen2.5-VL -> InternVL-2.5 -> GPT-4o lineage, yet MLLMs still hallucinate on tasks that require fine-grained perception or multi-step reasoning. They diagnose this as a layered failure:

- A **perceptual bottleneck** — CLIP-ViT's global-only contrastive pretraining cannot localize regions or pixels, and lacks structured / symbolic data such as charts and geometry.
- A **cognitive bottleneck** — single-pass, static visual encoding with no "rethink" mechanism, so the model decouples its final answer from the original visual evidence as reasoning chains grow longer.

Prior surveys are each only half of this picture: Yin et al. (2024) cover the standard MLLM stack; the reasoning surveys [29,30] cover CoT-style methods; the "thinking with images" surveys [31,32,33] cover the dynamic-perception subset. This paper claims to be the first to systematically map both layers and treat them as sequential, dependent stages. The medical angle is named explicitly in §4.2 (Medical Diagnosis) and is used as the motivating example for why the perception/cognition split matters — PathVQA still trails human experts (75.0 vs 85.2) because models lack the fine-grained perception (cellular morphology, staining) needed to ground specialized cognitive reasoning.

## Core Innovation

The contribution is structural rather than empirical. Three pieces:

1. **A bipartite Perception-Cognition lens** that splits the MLLM stack into (Perception: encoder + cross-modal alignment) and (Cognition: problem decomposition + dynamic forensics / "think with image") and labels every recent method by which layer it patches.
2. **A 2D timeline (Fig. 1)** that places 80+ models on these two axes between 2021 (CLIP) and 2025 (DeepEyes, VLM-R3, Argus, Pixel Reasoner).
3. **A Think-Act-Observe loop (Fig. 3 and Fig. 9)** that formalizes the cognitive controller: the LLM schedules perceptual operations (when to re-encode, what region to crop, which tool to call) while the vision encoder plus external tools act as the perceptual sub-system providing on-demand evidence.

No new model is proposed. No equations of substance.

## Claims & Evidence Analysis

| # | Claim | Evidence in this paper | Strength |
|---|-------|------------------------|----------|
| C1 | A "Perception -> Cognition" framework is a *novel* analytical lens not used by prior surveys ([28] Yin, [29,30] reasoning, [31,32,33] "thinking with images"). | §1.2 contribution list; structural argument only, no quantitative comparison to other surveys' taxonomies. | ⭐⭐ |
| C2 | Perception bottlenecks (weak fine-grained extraction; coarse alignment) cause hallucination. | §2.3.1 narrative + §3.1; relies on prior-paper claims (Eyes Wide Shut, Ferret-v2). No controlled perception-only ablation in this survey. | ⭐⭐ |
| C3 | Cognition bottlenecks (no rethink, no dynamic perception) are a *separate* root cause of hallucination. | §2.3.2 + §3.4; cites DeepEyes, Pixel Reasoner, Argus, CMMCoT. No head-to-head ablation isolating perception vs cognition. | ⭐⭐ |
| C4 | Process supervision + Best-of-N (Visual PRM in InternVL3) outperforms outcome supervision by 4-10 points on scientific benchmarks. | §4.1 narrative; numbers traceable to Table 6. Single model family, generalization untested. | ⭐⭐ |
| C5 | Med-PaLM M leads all medical benchmarks because of cross-task instruction fine-tuning. | Table 7. Numbers quoted from [272]; the survey itself runs no medical experiments. The causal "because" is interpretive. | ⭐ (causal "because") / **⭐⭐⭐** (leaderboard ordering) |
| C6 | Qwen2.5-VL's native dynamic-resolution ViT + absolute-time encoding explains its open-source dominance on ChartQA/DocVQA/MLVU. | §4.3-§4.4 interpretive paragraphs. Architectural attribution plausible but not causally tested (no resolution ablation). | ⭐ |
| C7 | Hallucination in long-chain reasoning is exacerbated by static, single-pass visual encoding. | §3.4 opening; supported indirectly by re-encoding methods that improve accuracy. Reported from primary sources, not re-validated. | ⭐⭐ |
| C8 | "Generate-and-verify" mechanisms guarantee logical rigor of MLLM answers. | §4.1 closing paragraph. Strong rhetorical claim; evidence is one model family (InternVL3 + VisualPRM). | ⭐ |
| C9 | The Perception->Cognition framework explains MLLM development from 2021 (CLIP) to 2025 (DeepEyes, VLM-R3, Argus). | Fig. 1 timeline. Bipartite labeling is useful organizationally but post-hoc — every model can be projected onto these two axes, so the framework cannot be falsified. | ⭐⭐ (organizationally) |

**Honest assessment.** This is a well-organized taxonomic survey, not an empirical contribution. The framework is genuinely useful as a *teaching* and *literature-mapping* device — splitting "thinking with images" into endogenous vs exogenous, then crossing that with single-pass vs multi-pass, is the clearest decomposition I have seen for this corner of the literature. But the central thesis ("perception bottlenecks and cognition bottlenecks are *separate* root causes of hallucination, in that order") is **asserted, not measured**. The five comparison tables re-quote single-run numbers from the original papers — there is **no meta-analysis, no variance reporting, no statistical significance, no contamination check, and no compute cost**. The "Med-PaLM M leads everywhere" finding (Table 7) is from a proprietary, non-reproducible model evaluation, and the survey does not discuss whether the medical benchmarks are contaminated in Med-PaLM M's training. The framework's predictive power is also untested — it post-hoc labels 80+ models on a 2D plane, but does not predict which methods will scale, which will fail, or which benchmarks will saturate. Editorial note: three sentences are duplicated verbatim in §3.3.1 ("However, the effectiveness of SFT is limited by its dependence on imitating a single 'correct' decomposition path...").

## Method & Architecture

![Survey structure and Perception-Cognition method tree (Figure 2)](/assets/images/paper/perception-to-cognition-survey/page_003.png)
*Figure 2 (Zhou et al., 2025): Survey structure. The method tree splits into Perception (encoder enhancement + cross-modal alignment) and Cognition (problem decomposition + dynamic forensics, a.k.a. "think with image").*

![Perception-Cognition loop (Figure 3)](/assets/images/paper/perception-to-cognition-survey/page_004.png)
*Figure 3 (Zhou et al., 2025): The Perception-Cognition loop — visual representation -> cross-modal alignment -> Plan / Observe / Reason / Verify cycle. This is the survey's central organizing diagram.*

The taxonomy has four axes:

### Perception axis 1 — Low-level visual perception (§3.1)

- **Single-encoder enhancement** (Table 1, group a-b): EVA-CLIP, SigLIP, MetaCLIP for fine-grained semantic alignment; DINOv2 series for self-supervised geometric/texture features; DIVA, VLV distill generative-model features back into CLIP.
- **Multi-encoder integration** (Table 1, group b-c): static concat / channel-fusion (Eyes Wide Shut, Prismatic VLMs, Ferret-v2, MouSi, BRAVE, SPHINX, ParGo, Layer Select Fuse); MoE routing over experts (MoME, MoVA, VisionWeaver, TOVE, R2-T2); distillation into a single student (Radio, UNIC, MoVE-KD, DUNE).
- The paper's diagnosis: static fusion creates feature conflicts; MoE fixes that at the cost of compute; distillation re-collapses compute. The open problem is a truly unified encoder.

### Perception axis 2 — Vision-language alignment (§3.2)

![Ovis visual embedding table (Figure 4)](/assets/images/paper/perception-to-cognition-survey/page_007.png)
*Figure 4 (Zhou et al., 2025): Ovis replaces the MLP projector with a learnable visual embedding *table* that parallels the text embedding table — one of the projection-layer variants surveyed.*

Three subroutes (Table 2):

- **(a) Improve the projection layer:** Honeybee (locality-preserving), ChartMoE / Uni-Med (MoE projector), LLaVA-Octopus (instruction-adaptive projector), Ovis2.5 (visual embedding *table* replacing the MLP — Fig. 4).
- **(b) Task-specific SFT:** MATCHA (math), LLaVA-Med (medical), Q-Instruct (low-level perception), ChartInstruct/ChartGemma (chart).
- **(c) Prompt tuning:** VPT, VPGTrans, TVP — PEFT-style visual prompts.

![Cross-modal fusion and output architecture (Figure 5)](/assets/images/paper/perception-to-cognition-survey/page_008.png)
*Figure 5 (Zhou et al., 2025): Instruction-encoding upgrades (Shikra coords-in-text -> Kosmos-2 grounding tokens -> GLaMM mask tokens -> ViP-LLaVA visual markers -> Draw-and-Understand external prompt encoder) and output-architecture upgrades (LISA `<SEG>` -> GSVA `<REJ>` -> VITRON / VisionLLM v2 unified decoders -> POPEN preference-based segmentation).*

**Dynamic perception (§3.2.3):** V* (LLM-guided hierarchical search), DyFo (MCTS-guided focus, training-free), FaST (fast/slow visual search adapter).

### Cognition axis 1 — Problem decomposition (§3.3)

![Automated CoT synthesis — T2V / V2T / Interleaved (Figure 6)](/assets/images/paper/perception-to-cognition-survey/page_011.png)
*Figure 6 (Zhou et al., 2025): Three paradigms for automated CoT data synthesis using an external teacher. T2V (Cogcom, MINT-CoT) inserts tool placeholders in text-first chains; V2T (MM-GCoT, SIFThinker, Pixel-Reasoner) goes vision-first to avoid hallucination but loses top-down structure; interleaved (LATTE, TACO) lets a teacher + tools "think-act-observe."*

![Bootstrapped CoT — positive sample refinement and preference data (Figures 7-8)](/assets/images/paper/perception-to-cognition-survey/page_012.png)
*Figures 7-8 (Zhou et al., 2025): Bootstrapped CoT. (Fig. 7) Positive-sample refinement — teacher/student MLLM + judger selects high/low-quality samples (STaR -> MC-CoT -> GCOT seed-and-filter bounding boxes). (Fig. 8) Preference-data generation — BPO injects image/chain perturbations as hard negatives.*

- **Training-based (§3.3.1, Table 3):**
  - *Imitation learning:* Multimodal-CoT (two-stage rationale -> answer to implicitly ground vision), Visual CoT / Visual Grounded Reasoning (cite bounding boxes), domain SFT (ChartGemma, Dolphins, Sce2DriveX).
  - *Curriculum learning:* LLaVA-CoT and LlamaV-o1 — three-stage "decompose -> ground -> integrate."
  - *Preference learning:* on-policy GRPO variants (VLM-R1, Visual-RFT, Reason-RFT, Seg-Zero, R1-OneVision, RAGEN); off-policy DPO variants (UV-CoT, V-DPO, VTS-DPO).
- **Automated CoT synthesis (§3.3.2):** External teacher (T2V / V2T / interleaved — Fig. 6) and bootstrapped (positive-sample refinement + preference data — Figs. 7-8).
- **Inference-time search (§3.3.3):** ToT-derived VisuoThink, MCTS-derived Socratic-MCTS / vrest / A*, using self-consistency w.r.t. visual evidence as reward signal.

### Cognition axis 2 — Dynamic forensics / "think with image" (§3.4)

Built around the Think-Act-Observe loop. Two categories:

- **Endogenous (§3.4.1, Table 4)** — no external tools, only internal attention.
  - *Single forward pass:* CVC (mask-then-predict re-weighting), ICoT (attention-map ROI re-encode), MINT-CoT (re-inject relevant tokens per step), Look-back (revisit attended cues on inconsistency).
  - *Multi forward pass:* CogCoM (chain-of-manipulations bounding boxes), DeepEyes (end-to-end RL deciding when to re-encode), Pixel Reasoner (SFT + RL), SIFThinker (adds depth), CMMCoT (cross-image memory bank).
- **Exogenous (§3.4.2, Table 5)** — MLLM-as-agent calling tools.
  - *ICL-based:* MM-ReAct, ViperGPT (program-as-agent), CLOVA, Visual Sketchpad.
  - *Fine-tuned:* LLaVA-Plus, TACO, LATTE, DWIM, Refocus, VPD (SFT on tool-use trajectories); VTool-R1, Visual-ARFT (RL on tool-call timing); OpenThinkimg (unified visual-tool API).

The **LLM-brain role decomposition** maps to: the LLM acts as the *cognitive controller* that schedules perceptual operations (when to re-encode, what region to crop, which tool to call), while the vision encoder + external tools act as the *perceptual sub-system* providing on-demand evidence.

## Experimental Results

The survey compiles five leaderboard tables (Table 6 scientific, Table 7 medical, Table 8 diagram, Table 9 video, Table 10 sentiment). Headline numbers, with the survey's own method-of-interest rows bolded where applicable:

| Benchmark | Best Proprietary | Best Open-Source | Human |
|-----------|------------------|------------------|-------|
| MathVista | **Gemini 2.5 Pro 84.6** | Ovis2-34B 76.1 / InternVL3-78B 75.1 | 60.3 |
| MathVerse | **Gemini 2.5 Pro 84.6** | Qwen2.5-VL-72B 57.6 | - |
| MATH-V | **Gemini 2.5 Pro 67.3** | Qwen2.5-VL-72B 38.1 | 68.82 |
| MV-MATH | **Gemini 2.5 Pro 73.3** | QVQ-72B-Preview 29.3 | 76.5 |
| MMMU | **Gemini 2.5 Pro 82.0** | InternVL3-78B 72.2 | 88.6 |
| VQA-RAD | **Med-PaLM M 90.0** | Med-Flamingo-80B 81.5 | 77.3 |
| SLAKE | **Med-PaLM M 90.5** | Med-Flamingo-80B 86.8 | 93.4 |
| **PathVQA** | Med-PaLM M 75.0 | Med-Flamingo-80B 70.3 | **85.2 (human still wins)** |
| ChartQA | GPT-4o 84.1 / Gemini 1.5 Pro 87.2 | **Qwen2.5-VL-72B 89.5** | - |
| DocVQA | Gemini 1.5 Pro 93.1 | **Qwen2.5-VL-72B 96.4** | - |
| TabMWP | **GPT-4o 97.4** | Qwen-VL-Chat-9B 92.1 | - |
| MLVU | - | **Qwen2.5-VL-72B 74.6** / STORM 72.9 | - |
| MVBench | GPT-4o 64.6 | **STORM 70.6** / Qwen2.5-VL-72B 70.4 | - |
| Video-MME (no subs) | **Gemini 1.5 Pro 75.0** | Qwen2.5-VL-72B 73.3 | - |
| MME-EMOTION | **Gemini 2.5 Pro 39.3** | LLaVA-OneVision-72B 37.9 | - |
| MELD (WF1) | - | **InternVL 2.5-20B 45.0** | - |

**Qualitative findings called out in-text:**

- **Process supervision works.** InternVL3 enhanced with VisualPRM Best-of-N improves +4-10 points over baseline by replacing outcome supervision with step-level process supervision — the survey uses this as evidence that "generate-and-verify" is currently the most effective cognition technique.
- **PathVQA is the lone unsolved benchmark.** In every comparison table, only PathVQA has *no* model (proprietary or open) that beats humans. The authors flag this as their go-to example of an unsolved cognition problem in a specialized perception domain.
- **Sentiment leaderboards are sparse.** Proprietary developers rarely report on MELD / HumanVBench, which the authors call a "global performance comparison deficit" (§4.5).

## Limitations

Authors acknowledge (§5 Future Directions):

- No truly unified visual encoder; image/video/3D integration incomplete (ATOKEN, TokLIP cited as partial attempts).
- Latent reasoning (Multimodal CoCoT, VTI, orthogonalization-based intervention) is nascent and not yet combined with hallucination suppression.
- Generative reasoning (Chameleon, Visual Planning, MVoT, Mind's Eye, ViLaSR) suffers from intermediate-image hallucination and training-data scarcity.
- Tool-augmented reasoning has an efficiency-accuracy tradeoff (GThinker verification slows inference); linear reasoning paths limit multi-step exploration — MCTS proposed as fix.
- Cross-image relation reasoning underexplored (only CmmCoT, Focus-Centric Visual Chain, Mantis cited).
- Real-world cognitive evaluation: benchmarks are too clean, too closed-ended.

What the authors do *not* address:

- **No meta-analysis.** The five tables are leaderboards, not normalized cross-benchmark analyses. Sample sizes, run counts, prompt variance are missing.
- **Benchmark contamination not discussed.** MathVista, ChartQA, ScienceQA have been on the web for years; recent proprietary models almost certainly trained on them.
- **No compute cost.** Tables compare 7B -> 78B -> 90B models on accuracy alone; no FLOPs, no per-token latency.
- **Multilingual MLLM coverage is thin.** Cited models are mostly US / UK / Chinese, with no critical discussion of language-specific failure modes.
- **Causal / architectural attributions are interpretive, not measured.** C5 (Med-PaLM M wins *because* of instruction-FT) and C6 (Qwen2.5-VL wins *because* of dynamic-resolution ViT) are plausible narratives without isolation experiments.
- **The framework's prescriptive value is unclear.** If perception and cognition are sequential, should new methods focus on one then the other? The survey does not commit.
- **Medical section (§4.2) is shallow** given the framework's importance: only Med-PaLM M, LLaVA-Med, Med-Flamingo, MedVInT, MiniGPT-v2 are covered. **No discussion of MedDr, BiomedGPT, RadFM, CheXagent** — which are central in the parallel medical-MLLM survey literature.
- **No safety / alignment** beyond hallucination-detection benchmarks (refusal calibration, bias, dual-use risk absent).
- **License / access of cited models not catalogued.**
- **Three sentences duplicated verbatim** in §3.3.1.

## Why It Matters for Medical AI

The medical track (§4.2 + Table 7) is the survey's most direct payload for clinical readers, and it is also the most concrete illustration of the framework: Med-PaLM M leads VQA-RAD (90.0) and SLAKE (90.5) but trails humans on PathVQA (75.0 vs 85.2) — the only "humans still win" entry in the entire survey. The authors interpret this gap as evidence that fine-grained *perception* of cellular morphology, staining, and tissue context is the missing piece, not cognition: PathVQA-style questions ("which cell type is shown in this H&E section?") demand the perceptual sub-system to localize and disambiguate sub-image features before any reasoning can begin. This is the framework's strongest concrete claim.

That said, the medical coverage is **shallow**. The survey lists Med-PaLM M, LLaVA-Med, Med-Flamingo, MedVInT, MiniGPT-v2 — but does not engage with the parallel medical-MLLM literature that has dominated the last two years: **MedDr, BiomedGPT, RadFM, CheXagent**, plus the medical-agent extensions (MedAgents, MDAgents, AgentClinic, MMedAgent, CT-Agent, AutoRG-Brain, VoxelPrompt) that already adopt this kind of perception/cognition vocabulary. Hallucination-detection benchmarks (HALT-MedVQA with FAKE-question probes, MedHallBench, Med-HallMark) are catalogued but not normalized against each other. PhysioNet credentialing, IRB constraints, and per-benchmark licensing are entirely absent — important for anyone planning a clinical evaluation.

Useful as a vocabulary source for medical-MLLM papers; not a substitute for a dedicated medical-MLLM survey.

## References

- Paper: Zhou, C. *et al.* (2025). *From Perception to Cognition: A Survey of Vision-Language Interactive Reasoning in Multimodal Large Language Models.* arXiv:2509.25373v4 (posted 16 Oct 2025). <https://arxiv.org/abs/2509.25373>
- Affiliations: Renmin University of China (corresponding: Yanbiao Ma), Xiamen University (corresponding: Yang Lu), HKUST (Yike Guo).
- Companion taxonomies critiqued: Yin et al. (2024) MLLM survey [28]; multimodal reasoning surveys [29, 30]; "thinking with images" surveys [31, 32, 33].
- Representative models indexed in Fig. 1 timeline: CLIP, EVA-CLIP, SigLIP, MetaCLIP, DINOv2, DIVA, VLV, Flamingo, BLIP-2, LLaVA, LLaVA-Med, LLaVA-CoT, LlamaV-o1, Qwen-VL, Qwen2.5-VL, InternVL series (InternVL3, InternVL-2.5), Ovis2.5, Honeybee, Ferret-v2, MoVA, Multimodal-CoT, Visual CoT, Visual-RFT, VLM-R1, Reason-RFT, Seg-Zero, RAGEN, UV-CoT, V-DPO, Cogcom, MINT-CoT, MM-GCoT, SIFThinker, Pixel Reasoner, LATTE, TACO, STaR, GCOT, BPO, VisuoThink, Socratic-MCTS, CVC, ICoT, Look-back, CogCoM, DeepEyes, CMMCoT, MM-ReAct, ViperGPT, CLOVA, Visual Sketchpad, LLaVA-Plus, DWIM, Refocus, VPD, VTool-R1, Visual-ARFT, OpenThinkimg, GPT-4o, GPT-o1, GPT-o3, Gemini 1.5 Pro, Gemini 2.5 Pro, Med-PaLM M, Med-Flamingo, MedVInT, STORM.
- Related "think with image" surveys (not cited by Zhou et al. but relevant): the "Brain-Perception-Action" LLM-agent survey (Xi et al. 2023) — see the companion post on this blog at `/paper/xi-llm-agents-brain-perception-action-survey/`.
