---
title: "GeneAgent: Self-verification Language Agent for Gene Set Knowledge Discovery using Domain Databases"
excerpt: "Wrapping GPT-4 in a generation->verify->modify->summarize cascade that grounds every claim in 18 curated databases lifts MsigDB Rouge-2 from 7.4% to 15.5% and enrichment-term exact-match from 56.0% to 80.7%."
categories:
  - Paper
  - BioInformatics
  - LLM
  - LLM-Agents
permalink: /paper/geneagent/
tags:
  - GeneAgent
  - GPT-4
  - LLM-Agents
  - Tool-Use
  - Self-Verification
  - Gene-Set-Enrichment
  - g:Profiler
  - Enrichr
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-17
last_modified_at: 2026-05-17
---

## TL;DR

- GeneAgent is a GPT-4-based language agent for gene set knowledge discovery (predicting the most relevant biological process name for an input gene set) that self-verifies every generated claim by autonomously calling **four Web APIs covering 18 expert-curated biological databases** (Enrichr, g:Profiler, NCBI E-utils, CustomAPI).
- The architectural bet is **selfVeri-Agent**: rather than letting the LLM second-guess itself (which the authors argue produces overconfident agreement), claims extracted from the LLM's draft are validated against external domain knowledge, and a *generation -> self-verification -> modification -> summarization* cascade rewrites the process name and the analytical narrative.
- On **1,106 gene sets** (GO 1,000 + NeST 50 + MsigDB 56), GeneAgent beats the Hu et al. 2023 GPT-4 baseline on every metric. Headline numbers: MsigDB **Rouge-L 23.9% -> 31.0%**, **Rouge-2 7.4% -> 15.5%**; enrichment-term exact-match accuracy **56.0% (no synopsis) / 68.8% (SPINDOCTOR ontological synopsis) -> 80.7%** when GeneAgent's verification report is fed back as the gene synopsis.

## Motivation

Gene set enrichment analysis (GSEA) traditionally relies on rank-based matching against curated databases like GO and MSigDB, but a growing share of genomics work involves gene sets that only marginally overlap with known enrichment functions -- exactly where conventional GSEA is weakest and where novel biology often lives. Prior LLM approaches (Hu et al. 2023's GPT-4 evaluation; SPINDOCTOR's gene-synopsis summarization) showed promise but inherit the standard LLM pathologies: non-deterministic outputs and hallucinated gene functions that make them unsafe for downstream biological interpretation.

GeneAgent's premise is that the right fix is not "make the LLM doubt itself" but "force the LLM to ground every claim in expert-curated databases through tool use." For medical/biomedical AI this matters because hallucinated gene-function calls can mislead translational pipelines -- the paper's case study on mouse B2905 melanoma sublines is exactly this kind of setting, where the goal is to interpret subclonal expression programs in immunotherapy-responsive vs resistant clades.

## Core Innovation

- **Cascaded generation -> self-verification -> modification -> summarization.** Four prompted modules over a single frozen GPT-4 backbone (version 20230613, Azure OpenAI, temperature 0) -- no fine-tuning anywhere. Claims are extracted from the draft, verified one-by-one against external databases, used to drive a revision of both the process name and the analytical narrative, then summarized.
- **selfVeri-Agent grounds in 18 databases through 4 APIs.** Enrichr (4 DBs: KEGG_2021_Human, Reactome_2022, BioPlanet_2019, MsigDB_Hallmark_2020), g:Profiler g:GOSt (8 DBs: GO, KEGG, Reactome, WikiPathways, Transfac, miRTarBase, CORUM, HPO), NCBI E-utils (Gene + PubMed), and a CustomAPI library (gene-disease, gene-domain, PPI, gene-complex). Each verification call returns a finding plus a four-way decision: Supports / Partially Supports / Refutes / Unknown.
- **Verify-twice property.** The same verification pipeline runs once on the process-name claim ($\mathcal{R}_P$) and again on the rewritten analytical narrative ($\mathcal{R}_A$), and the second pass re-checks the modified process name -- the paper highlights this re-verification as load-bearing for the final accuracy.
- **API masking for fair evaluation.** When evaluating GO gene sets, g:Profiler is masked (it would trivially recover GO ground truth); when evaluating MsigDB gene sets, the MsigDB_Hallmark_2020 database inside Enrichr is masked. This is the methodological control that makes the GPT-4 vs GeneAgent comparison non-circular -- though as discussed below, leakage between the remaining databases is not fully eliminated.

## Claims & Evidence Analysis

| # | Claim | Evidence | Dataset(s) | Strength |
|---|---|---|---|---|
| C1 | GeneAgent "consistently outperforms standard GPT-4 by a significant margin" on gene-set process-name generation | Fig. 2a Rouge across 3 datasets; Fig. 2b violin plots, one-tailed t-test p < 10^-3 on NeST/MsigDB and p < 0.05 on GO; Fig. 2c percentile distributions | GO (1,000) + NeST (50) + MsigDB (56) | ⭐⭐⭐ |
| C2 | Self-verification reduces hallucinations | Enrichment-term exact-match (Fig. 2d): 80.7% with verification report vs 56.0% no-synopsis vs 68.8% SPINDOCTOR ontological synopsis. "Unmatched terms" are equated with hallucinations following SPINDOCTOR's framing | MsigDB only (56 sets, 367 terms) | ⭐⭐ |
| C3 | selfVeri-Agent decisions are accurate | Human annotation: 92% (122/132) decisions judged correct | 10 NeST gene sets, single batch | ⭐ |
| C4 | GeneAgent integrates 18 databases via 4 APIs and uses them broadly | Fig. 3b/c utilization: NCBI Gene 7,304 calls; BioPlanet 3,116; Reactome 2,938; MsigDB Hallmark 2,048; CustomAPI components much less (PPI 280, Gene-Disease 115) | All 1,106 sets | ⭐⭐⭐ for volume, ⭐ for CustomAPI utility |
| C5 | GeneAgent provides "more informative gene functional synopsis than SPINDOCTOR" | Single bar comparison in Fig. 2d (80.7% vs 68.8%) | MsigDB (56 sets) | ⭐⭐ |
| C6 | GeneAgent generalizes across species (mouse) and to novel gene sets | Table 3 case study: 5/7 GeneAgent-preferred; 2/7 exact match with expert reference | 7 mouse B2905 melanoma sets, n=2 expert annotators | ⭐ |
| C7 | GeneAgent surpasses conventional GSEA (g:Profiler) | Extended Fig. 3a (similarity) and 3b (Rouge) on NeST + MsigDB only -- GO is omitted | NeST + MsigDB | ⭐⭐ (selection-of-datasets concern) |
| C8 | External-DB self-verification is better than LLM-only self-verification | No direct ablation against GPT-4 self-verification baselines (Self-Refine, RARR, Chain-of-Verification). Argument is conceptual / citational only | -- | ⭐ (uncontrolled) |
| C9 | 88.5% of non-supported claims get modified | Internal log statistic (p. 10) | -- | ⭐⭐ (process metric, not outcome) |

**Honest read.** The Rouge/semantic-similarity wins over GPT-4 (C1) are the most solid claim -- multi-dataset, statistically tested, with explicit database masking, and gains large enough (e.g., Rouge-2 doubling on MsigDB) to clear LLM stochasticity at temperature 0. Three structural problems sit underneath the rest:

1. **No ablation isolates the three architectural pieces.** GeneAgent stacks (a) the four-step cascade, (b) external-DB grounding, and (c) claim extraction before verification. The paper offers no run that turns off any one of these to see which is actually doing the work. C8 -- the conceptual differentiator from prior self-verification work -- is therefore not empirically supported: a GeneAgent variant with an LLM-only verifier (no DB) is the obvious missing baseline.
2. **The "hallucination" metric is built-in-advantage circular.** The 80.7% headline number for C2 / C5 follows SPINDOCTOR in operationally defining "hallucination" as "term not in g:Profiler's top-5 enrichment." But GeneAgent calls g:Profiler directly as part of its verification stack -- so the metric mechanically rewards systems that surface what g:Profiler already knows. A more convincing measure would require independent expert adjudication of final narratives on a held-out set.
3. **The human audit is small and unstandardized.** C3's 92% accuracy is based on **132 verification decisions from 10 NeST gene sets, one batch, no inter-annotator-agreement reported, annotator background unspecified**. Confidence interval at n=132 around 92% is roughly +-4.5 pp on its own, before you account for the absence of IAA on a four-way categorical decision.

Add to this no multi-seed variance reporting, no confidence intervals on Rouge / similarity, no cost / latency analysis for ~17 API calls per gene set, and the omission of GO from the vs-g:Profiler comparison (C7), and the picture is: a real improvement over GPT-4 that's been measured through metrics partly aligned with the verifier's own knowledge sources.

## Method & Architecture

![GeneAgent cascade: generation, self-verification, modification, summarization, with the RTK Signaling worked example](/assets/images/paper/geneagent/fig_p005_01.png)
*Figure 1: GeneAgent's four-step cascade (generation -> self-verification -> modification -> summarization) and selfVeri-Agent's claim-verification loop. In the worked example, an initial "MAPK Signaling Pathway" name is refuted by external databases and corrected to "RTK Signaling."*

Formally, GeneAgent processes a gene set $D = \{g_i\}_{i=1}^N$ through:

1. **Generation.** $\mathrm{GeneAgent}_g(D) = (P_{ini}, A_{ini})$ -- initial process name $P_{ini}$ and analytical narrative $A_{ini}$ from a single GPT-4 call.
2. **Claim extraction.** $P_{ini}$ is decomposed into atomic claims using phrasing like "be involved in" / "related to" (e.g., "ERBB2, ERBB4, FGFR2, FGFR4, HRAS, KRAS is involved in RTK Signaling").
3. **selfVeri-Agent on process name -> $\mathcal{R}_P$.** For each claim: extract gene names and process name, autonomously route to the appropriate Web API(s), and compile a verification report with finding + decision per claim.
4. **Modification.** $\mathrm{GeneAgent}_m(P_{ini}, A_{ini}, \mathcal{R}_P) = (P_{mod}, A_{mod})$ -- revise or retain $P_{ini}$ based on $\mathcal{R}_P$; if revised, rewrite $A_{ini}$ consistently.
5. **selfVeri-Agent on narrative -> $\mathcal{R}_A$.** The same pipeline re-runs on per-gene functional claims inside $A_{mod}$ and re-checks the new process name.
6. **Summarization.** $\mathrm{GeneAgent}_s(P_{mod}, A_{mod}, \mathcal{R}_A) = (P, A)$ merges all intermediate verification reports into the final output.

## Experimental Results

**Main benchmark (vs Hu et al. 2023 GPT-4).** Reading from Figure 2 and pp. 6-9:

| Metric | Dataset | GPT-4 (Hu et al.) | **GeneAgent** |
|---|---|---|---|
| Rouge-L | MsigDB | 0.239 | **0.310** |
| Rouge-1 | MsigDB | 0.239 | **0.310** |
| Rouge-2 | MsigDB | 0.074 | **0.155** |
| Mean MedCPT semantic similarity | GO | <0.705 | **0.705** (p < 0.05) |
| Mean MedCPT semantic similarity | NeST | <0.761 | **0.761** (p < 10^-3) |
| Mean MedCPT semantic similarity | MsigDB | <0.736 | **0.736** (p < 10^-3) |
| #sets similarity > 0.90 (all) | -- | 104 | **170** |
| #sets similarity > 0.70 (all) | -- | 545 | **614** |
| #sets exact (similarity = 1.00) | -- | 3 | **15** |
| #sets in top-90th percentile of background | all | 824 (74.5%) | **850 (76.9%)** |
| #sets in top-98th percentile | all | 598 | **675** |
| #sets at 100th percentile | all | 43 | **82** |
| Enrichment-term exact-match | MsigDB | 56.0% (no syn) / 68.8% (SPINDOCTOR) | **80.7%** (296/367) |

![Main results: Rouge across datasets, similarity violin plots, percentile distributions, enrichment-term match](/assets/images/paper/geneagent/fig_p006_01.png)
*Figure 2: GeneAgent vs GPT-4 (Hu et al. 2023) across Rouge (a), semantic similarity with one-tailed t-test (b, ** denotes p < 10^-3), background percentile distributions over 12,320 terms (c), and SPINDOCTOR-style enrichment-term exact-match (d). GeneAgent's verification report as gene synopsis reaches 80.7% exact-match accuracy on MsigDB vs 56.0% with no synopsis.*

**Self-verification audit (Fig. 3, p. 10).**

- Of **15,903 total claims** generated across 1,106 gene sets, 15,848 (99.6%) were successfully verified; 0.4% failed due to missing gene names for API queries.
- Decision breakdown: **84% Supports, 1% Partially Supports, 8% Refutes, 7% Unknown**.
- 16% non-supported claims span 794 gene sets; **703/794 (88.5%)** were actually modified in the modification step.
- API skew: Enrichr + g:Profiler dominate process-name verification; E-utils + CustomAPI dominate analytical-narrative verification. Total backend DB calls: 19,273 for 15,848 claims -> ~1.22 lookups / claim.
- **Human audit: 92% (122/132)** of selfVeri-Agent decisions judged correct on 10 NeST gene sets (Fig. 3d confusion matrix: 80 supports / 14 partial / 27 refute / 1 unknown). Inter-annotator agreement is not reported.

![Self-verification audit: 15,903-claim decision pie, API usage, DB call counts, human-verification matrix](/assets/images/paper/geneagent/fig_p011_01.png)
*Figure 3: Self-verification at scale -- 15,903 LLM-generated claims, 99.6% successfully verified (a); API selection differs by what's being verified (b); NCBI Gene (7,304 calls), BioPlanet, and Reactome dominate database usage (c); human spot-check of 132 NeST claims agrees with selfVeri-Agent 92% of the time (d).*

**Percentile-evaluation example.**

![Same gene set scored by GeneAgent's name vs GPT-4's name on a 12,320-term background](/assets/images/paper/geneagent/fig_p026_01.png)
*Extended Fig. 1: Same gene set, two systems -- GeneAgent's "Regulation of Cellular Response to Stress" (a) sits at the 98.9th percentile of the 12,320-term background; GPT-4's "Calcium Signaling Pathway Regulation" (b) sits at the 60.2nd percentile.*

**Mouse melanoma case study (Table 3).** Two-expert blind annotation across Relevance / Readability / Consistency / Comprehensiveness across 7 gene sets from B2905 sublines. GeneAgent wins the final decision on **5/7** sets, ties on 1 (both fail on mmu03010 HA-S where GeneAgent's "Cytosolic Ribosome" misses mitochondrial Mrpl10/Mrps21 and GPT-4 hallucinates "Synthesis"). Wins concentrate on Relevance and Comprehensiveness; Readability/Consistency are roughly tied. Two of 7 names exactly match expert reference (mmu04015 HA-S "Rap1 Signaling Pathway"; mmu05100 HA-S "Bacterial Invasion of Epithelial Cells").

![Mouse melanoma case study: GeneAgent vs GPT-4 gene-function clustering for mmu05022 LA-S](/assets/images/paper/geneagent/fig_p027_01.png)
*Extended Fig. 2: Mouse melanoma case study (mmu05022, LA-S clade) -- GeneAgent (a) places Ndufa10 into Complex I and labels Gpx7 as oxidative-stress protection; GPT-4 (b) omits Ndufa10 from oxidative phosphorylation and provides no biological role for Gpx7.*

**GeneAgent vs conventional GSEA.**

![GeneAgent vs g:Profiler GSEA on NeST and MsigDB](/assets/images/paper/geneagent/fig_p028_01.png)
*Extended Fig. 3: GeneAgent vs conventional g:Profiler GSEA on NeST and MsigDB -- GeneAgent's generated names are both semantically closer (a) and lexically closer (b) to expert reference terms than g:Profiler's top enrichment terms. (GO is conspicuously not shown.)*

## Limitations

**Author-admitted:**
- Only one backbone tested (GPT-4 2023-06-13). The design is not validated as a general LLM-agent recipe.
- Some reference-dissimilar names persist; suggested fix is adding more domain DBs.
- No gene-set preprocessing (e.g., removing genes incoherent with the rest of the set).
- Error analysis (Extended Tab. 1) identifies two failure modes: erroneously refuting a correct GPT-4 name ("EGFR Signaling Pathway Regulation," "Prostate Cancer Progression") and erroneously supporting an originally dissimilar name ("Catecholamine Biosynthesis" for response-to-pyrethroid). The verifier can hurt as well as help.

**Not addressed by the authors:**
- **No ablation isolating cascade vs DB-grounding vs claim-extraction.** The paper attributes wins to "external-DB self-verification," but with no LLM-only-verifier baseline, no no-cascade baseline, and no no-claim-extraction baseline, the contribution of each piece is unmeasured.
- **The hallucination metric has a built-in advantage.** SPINDOCTOR's "unmatched-to-g:Profiler-enrichment = hallucination" definition mechanically favors a system that calls g:Profiler. C2's 80.7 vs 68.8 / 56.0 numbers should be read accordingly.
- **The 92% human-audit number is fragile.** n=132 claims, 1 batch, 10 NeST gene sets, no inter-annotator agreement, no annotator background, no claim-sampling protocol disclosed. The metric audits *verification decisions*, not the *final generated narratives*.
- No comparison to prompt-only self-verification baselines (RARR, Self-Refine, Chain-of-Verification, GeneGPT -- the latter from the same group is the obvious comparator).
- No cost / latency analysis for ~17 API calls per gene set; reproducibility not pinned to DB snapshots.
- DB-overlap leakage between background terms and verifier-queried databases is not eliminated by the masking strategy -- only the most egregious cases (GO vs g:Profiler; MsigDB vs MsigDB_Hallmark_2020) are masked, while Reactome and KEGG term vocabularies overlap substantially with what remains.
- The case study (7 sets, n=2 annotators) reports no inter-annotator agreement.

## Why It Matters for Medical AI

For clinical and translational genomics pipelines, hallucinated gene-function calls are not stylistic mistakes -- they propagate into biomarker hypotheses, drug-target rationales, and case reports. GeneAgent's contribution is to demonstrate that wrapping a frozen biomedical LLM in a tool-use cascade that grounds every atomic claim in curated databases is operationally tractable (99.6% verification rate over 15,903 claims) and measurably improves over the LLM-alone baseline on standard gene-set benchmarks. The verify-then-modify pattern -- distinct from same-model self-consistency -- generalizes naturally to other biomedical LLM tasks where curated knowledge bases exist (drug-drug interactions via DrugBank, variant interpretation via ClinVar, radiology reporting via RadLex). The caveats are the same caveats that apply to any retrieval-grounded LLM in medicine: the system inherits both the strengths and the gaps of the databases it queries, and the "hallucination reduction" headline should be read as "alignment-to-curated-DB improvement" rather than "factuality improvement in an open-world sense." For users planning to deploy this on novel-biology questions where curated DBs are thin, the case-study failure modes (Mrpl10 / Mrps21 missed on mmu03010 HA-S) are the warning sign.

## References

- **Paper:** Wang, Jin, Wei, Tian, Lai, Zhu, Day, Ross, Lu. *GeneAgent: Self-verification Language Agent for Gene Set Knowledge Discovery using Domain Databases.* arXiv 2405.16205 (2024); published in Nature Methods (2025). [arXiv link](https://arxiv.org/abs/2405.16205)
- **Prior LLM-on-gene-sets baseline:** Hu et al., *Evaluation of large language models for discovery of gene set function.* arXiv 2309.04019 (2023).
- **Gene-synopsis baseline:** SPINDOCTOR (Talisman-paper benchmark, gene-set summarization via ontological synopsis).
- **Same-group prior work on LLM + biomedical tools:** GeneGPT (Jin et al., 2024).
- **Enrichment tools called by GeneAgent:** [g:Profiler g:GOSt](https://biit.cs.ut.ee/gprofiler/gost), [Enrichr](https://maayanlab.cloud/Enrichr/), [NCBI E-utils](https://www.ncbi.nlm.nih.gov/books/NBK25501/).
- **Case-study cohort:** mouse B2905 melanoma sublines (HGF-transgenic C57BL/6, UV-induced), with EvoGeneX-identified adaptively regulated gene sets.
