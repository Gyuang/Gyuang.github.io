---
title: "Bayesian Concept Bottleneck Models with LLM Priors"
excerpt: "Split-sample Metropolis-within-Gibbs with an LLM proposal kernel reaches accuracy 0.680 / AUC 0.874 on CUB family-level, narrowly above ResNet50 (0.664 / 0.853) but with overlapping CIs."
categories:
  - Paper
tags:
  - BC-LLM
  - Concept Bottleneck Models
  - Bayesian Inference
  - Metropolis-within-Gibbs
  - LLM Priors
  - Interpretability
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- Standard CBMs require a fixed candidate concept pool up front; **BC-LLM does Metropolis-within-Gibbs over a (potentially infinite) concept space**, with an LLM acting as both prior/proposal and concept extractor so concepts are discovered iteratively rather than prespecified.
- The technical move is a **split-sample Metropolis update**: the LLM proposes from a partial posterior fit on a subset $S$, and accept/reject uses held-out $S^c$ to override LLM mistakes — yielding provable convergence to the true concept set even when the LLM's prior is miscalibrated (Theorem 3.1).
- On CUB-Birds family-level, BC-LLM reaches **Acc 0.680 / AUC 0.874 / Brier 0.428**, edging LLM+CBM (0.640 / 0.810 / 0.452), Human+CBM (0.658 / 0.835 / 0.499), and fine-tuned ResNet50 (0.664 / 0.853 / 0.457) — but with overlapping 95% CIs in most cells.

## Motivation

CBMs promise auditable, interpretable predictions, but they typically need a hand-curated concept list. That bottlenecks coverage (some relevant concepts are always missed — medicine is full of "infinitely refinable" concepts like smoking status) and hands the accuracy crown to soft/embedding-based variants that leak label information. Recent LLM-as-concept-proposer methods inherit hallucination and self-inconsistency problems and still require a prespecified pool.

The medical-AI angle is explicit. The running examples are MIMIC-IV clinical notes and a real readmission-risk pipeline at Zuckerberg San Francisco General Hospital, where the authors argue that an LLM-driven Bayesian concept search can both improve accuracy and surface clinically actionable features that hand-engineered tabular features miss.

## Core Innovation

BC-LLM frames concept discovery as posterior sampling over concept sets $\vec{c} = (c_1, \dots, c_K)$ for an LR-based CBM

$$p(Y=1\mid x, \vec\theta, \vec c) = \sigma\!\left(\theta_0 + \sum_{k=1}^K \theta_k\,\phi_{c_k}(x)\right).$$

The chain proposes concept replacements with an LLM (the prior) but only accepts them when held-out data backs the proposal up. The accept step uses a multiple-try split-sample Metropolis-Hastings ratio whose held-out likelihood factor acts as a frequentist guardrail. As a result, the stationary distribution provably concentrates on the true concept maximizer $C^*$ even when the LLM is not a self-consistent Bayesian inference engine — the theoretical sell of the paper.

## Claims & Evidence Analysis

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | "Rigorous statistical inference and uncertainty quantification" despite LLM miscalibration. | Theorem 3.1: asymptotic stationary distribution concentrates on $C^*$. Theory only. | ⭐⭐ Clean asymptotic statement, but requires IID data, Gaussian prior on $\vec\theta$, and regularity; no finite-sample rate is shown empirically. |
| C2 | Outperforms interpretable baselines. | Table 1 (CUB family), Fig 3 (MIMIC), Fig 4 (ZSFG), Tables 2–3 (Claude, fMoW, Imagenette). | ⭐⭐⭐ Consistent direction across 5 settings; headline wins are real even when CIs overlap. |
| C3 | Beats black-box ResNet50 on CUB family-level. | Table 1: Acc 0.680 vs 0.664, AUC 0.874 vs 0.853, Brier 0.428 vs 0.457. | ⭐ 95% CIs overlap heavily. Authors themselves attribute it to ResNet50 overtraining on a tiny dataset; no general-purpose black-box (CLIP zero-shot, ViT, regularized fine-tune) is included. |
| C4 | Better calibration / uncertainty than baselines. | Brier in Table 1, Fig 3, Fig 4, Tables 2–3; OOD entropy in Table 1. | ⭐⭐ Brier is consistently best, **but no ECE, reliability diagrams, or temperature-scaling baseline is reported.** Worse: in the only OOD entropy column, **ResNet50 has higher entropy (0.914) than BC-LLM (0.865)** — the black-box is *more* appropriately uncertain on the headline OOD claim, contradicting "more robust to OOD samples." |
| C5 | Bayesian uncertainty correlates with prediction quality on held-out data. | One qualitative OOD example (Fig 2 right: dog in rainbow wings → 0.5) and the entropy column above. | ⭐ **No quantitative correlation analysis** between posterior variance / predictive entropy and held-out error. No selective prediction (accuracy-vs-coverage), no AURC, no abstention experiment. The "appropriately unsure" claim rests on one cherry-picked image and one entropy column where the black-box wins. |
| C6 | Converges to true concepts. | Theorem 3.1 + MIMIC concept Recall/Precision reaching Human(Oracle)+CBM by 800 obs. | ⭐⭐ Strong on the synthetic outcome where ground-truth concepts are baked in by construction; no analogous "true concept" experiment outside the simulation. |
| C7 | Robust to LLM choice / prompt phrasing. | Appendix D.1 (Command-R), Table 2 (Claude 3.5 Haiku), vague-prompt ablation. | ⭐⭐ Direction holds across LLMs; absolute numbers drop with Command-R. |
| C8 | Cost-effective: $O(nTK)$ vs $O(nW)$ queries. | Page 6 complexity argument; "less than a few minutes on a normal laptop" — v3 retains TODO "[check timings a bit more carefully]". | ⭐ No actual wall-clock or dollar-cost numbers vs LLM+CBM at $W=370$. |
| C9 | Helps a real hospital data-science team (ZSFG readmission). | AUC 0.60→0.64; clinician relevance 2.5 vs ≤2.1. | ⭐ **Single PHI run, n=1 split**, four raters, no inter-rater agreement statistic in main text, no blinded comparison protocol. Suggestive, not conclusive. |

**Honest summary.** The well-supported core is (a) the algorithmic contribution — split-sample MH with an LLM proposal kernel is a clean idea and Theorem 3.1 is the right theoretical statement — and (b) the broad direction that iterative concept search beats fixed-pool LLM+CBM and boosting-CBM across modalities. The over-generalizations are the abstract's "rigorous… uncertainty quantification" and "more robust to OOD" framings: no calibration metric beyond Brier is reported, no posterior-variance-vs-error correlation is shown, and the one OOD entropy column has the black-box winning. The v3 manuscript still ships visible TODOs ("[UPDATE FIGURE]", "[check timings a bit more carefully]") and literal "[???]" entries for the fMoW OOD-China AUC — manuscript-hygiene issues that should not have survived peer review.

## Method & Architecture

![BC-LLM pipeline](/assets/images/paper/bayesian-cbm-llm-priors/page_003.png)
*Figure 1: BC-LLM iteratively replaces one concept at a time — extract keyphrases, fit a residual keyphrase model, ask the LLM for candidates conditioned on a data subset $S$, annotate, and accept via a split-sample Metropolis step on $S^c$.*

The chain runs Metropolis-within-Gibbs over $\vec c$:

1. **Initialization.** Prompt the LLM to extract short keyphrases per observation (e.g. "diabetes", "blue head"). Fit a penalized LR of $Y$ on those keyphrases, then ask the LLM to convert the top-weighted ones into $K$ initial yes/no concept questions.
2. **Drop a concept, fit a keyphrase residual model.** For the index $k$ being updated, refit an LR for $Y$ given keyphrases plus the remaining $K{-}1$ concepts $\vec c_{-k}$, with a ridge penalty on the keyphrase coefficients (CV-chosen $\lambda$). The surviving high-coefficient keyphrases summarize what the dropped slot needs to recover.
3. **LLM proposes candidates from a partial posterior.** Split data into $S$ (size $\lfloor\omega n\rfloor$, $\omega\approx 0.5$) and $S^c$. Prompt the LLM with $\vec c_{-k}$ + summary from $S$ to propose $M$ candidate concepts $\check c_k^{(1)}, \dots, \check c_k^{(M)}$ and self-report a partial-posterior weight. This implicitly samples from $Q(C_k; \vec c_{-k}) = p(C_k \mid \vec c_{-k}, y_S, X)$.
4. **Annotate.** Use the LLM zero-shot to extract $\phi_{c}(x)$ for each candidate, batched across the $M$ candidates per observation. The LLM is permitted to output probabilities when uncertain — this is what produces the famous "0.5" outputs on OOD images.
5. **Multiple-try split-sample MH accept/reject (Algorithm 3).** Sample one candidate $\check c^{(\hat m)}$ with probability $\propto w_m = p(y_{S^c}\mid y_S, (\vec c_{-k}, \check c^{(m)}), X)\cdot Q(\check c^{(m)}; \vec c_{-k}, y_S, X)$ and accept with

   $$\alpha = \min\!\left\{ \frac{Q(c_k; \vec c_{-k}, y_S, X)\sum_{m=1}^M w_m}{Q(\check c^{(\hat m)}; \vec c_{-k}, y_S, X)\sum_{m \neq \hat m} w_m},\ 1\right\}.$$

   The held-out $S^c$ likelihood is the frequentist-style guardrail that overrides bad LLM proposals.
6. **Inner integral.** $p(y_{S^c}\mid y_S, \vec c, X)$ is approximated by a Laplace-like integral on the LR posterior under a Gaussian prior on $\vec\theta$ — asymptotically exact.

**Hyperparameters.** $K\in\{4,\dots,10\}$ (cap 20), outer loops $T=5$ (10 for better uncertainty), $\omega=0.5$, $M$ candidates per MH step batched. Cost: $O(nTK)$ LLM queries vs $O(nW)$ for fixed-pool CBMs with $W\sim 10^2$–$10^3$.

**Convergence (Theorem 3.1).** Under IID data, a Gaussian prior on $\vec\theta$, and regularity conditions (Assumption G.1), every stationary distribution $\pi$ of the chain concentrates on $C^* = \arg\max_{\vec c} L(\vec c)$ as $n\to\infty$, **even when the LLM is not a self-consistent Bayesian engine**.

## Experimental Results

### CUB-Birds, family-level (in-distribution + OOD)

| Method | Acc ↑ | AUC ↑ | Brier ↓ | Entropy (OOD) ↑ |
|---|---|---|---|---|
| **BC-LLM** | **0.680** (0.614, 0.747) | **0.874** (0.840, 0.907) | **0.428** (0.357, 0.500) | 0.865 (0.693, 1.036) |
| LLM+CBM | 0.640 (0.573, 0.707) | 0.810 (0.768, 0.853) | 0.452 (0.377, 0.528) | 0.663 (0.474, 0.852) |
| Boosting LLM+CBM | 0.538 (0.463, 0.614) | 0.722 (0.673, 0.772) | 0.577 (0.499, 0.654) | 0.842 (0.630, 1.054) |
| Human+CBM | 0.658 (0.591, 0.725) | 0.835 (0.791, 0.879) | 0.499 (0.414, 0.584) | 0.758 (0.558, 0.959) |
| LLM+CBM (no keyphrases) | 0.555 (0.488, 0.623) | 0.759 (0.713, 0.805) | 0.651 (0.548, 0.754) | 0.626 (0.495, 0.757) |
| ResNet50 (black-box) | 0.664 (0.613, 0.716) | 0.853 (0.821, 0.885) | 0.457 (0.398, 0.516) | **0.914** (0.748, 1.079) |

BC-LLM is best on Acc, AUC, and Brier — but **ResNet50 has the highest OOD entropy**, which directly undercuts the abstract's "robust to OOD" framing. All 95% CIs overlap broadly with the next-best baseline.

### MIMIC-IV simulated SDOH

![MIMIC scaling curves](/assets/images/paper/bayesian-cbm-llm-priors/page_008.png)
*Figure 2: MIMIC simulation — BC-LLM matches the oracle CBM (which uses the true concepts) in AUC and Brier by ~400 observations and dominates concept Recall and Precision at all sizes.*

![MIMIC dendrogram](/assets/images/paper/bayesian-cbm-llm-priors/page_008.png)
*Figure 3: At $n=100$ the posterior puts mass on generic concepts (e.g. "rehabilitation"); at $n=800$ it concentrates around the five true SDOH concepts (smoking, alcohol, drugs, retired, unemployed).*

Concept recall reaches ~0.75 at 800 obs versus ~0.5 for LLM+CBM and <0.2 for Boosting. Caveat: $Y$ is generated from a hand-specified LR over five SDOH concepts — a best-case setting for an LR-based CBM, not a real clinical label.

### ZSFG heart-failure readmission (real EHR)

![ZSFG concept dendrogram](/assets/images/paper/bayesian-cbm-llm-priors/page_010.png)
*Figure 4: ZSFG heart-failure readmission concepts learned by BC-LLM; highlighted concepts received clinician relevance ratings ≥2.5 and pointed to actionable interventions like outpatient follow-up scheduling.*

![Clinician relevance ratings](/assets/images/paper/bayesian-cbm-llm-priors/page_010.png)
*Figure 5: Four clinicians rated BC-LLM concepts at 2.5/3 for clinical relevance versus ≤2.1 for all comparator methods, on the heart-failure readmission task.*

| Method | AUC (95% CI) | Brier (95% CI) |
|---|---|---|
| **BC-LLM** | **0.64** (0.58, 0.70) | **0.14** (0.12, 0.62*) |
| LLM+CBM | 0.59 (0.52, 0.65) | 0.29 (0.25, 0.33) |
| Boosting LLM+CBM | 0.59 (0.52, 0.66) | 0.14 (0.12, 0.16) |
| Bag-of-words | 0.52 (0.46, 0.58) | 0.29 (0.25, 0.34) |

(*The BC-LLM Brier CI upper bound of 0.62 in the paper looks like a typo for 0.16, given the point estimate of 0.14 and the Boosting row.) Tabular-only AUC of 0.60 lifts to 0.64. Clinician relevance (1–3 scale): BC-LLM = **2.5**, all others ≤ 2.1. **This is n=1: a single PHI run with no error bars across seeds and no reproducibility path.**

### Posterior contraction and qualitative OOD

![Bunting posterior dendrograms](/assets/images/paper/bayesian-cbm-llm-priors/page_008.png)
*Figure 6: Concept posterior contraction on Bunting birds — with 1/3 of the training data BC-LLM still posts vague concepts like "colorful plumage"; at 3/3 it concentrates on distinguishing features like "a red belly (0.94)" and "a white belly (0.94)".*

![OOD concept extraction](/assets/images/paper/bayesian-cbm-llm-priors/page_008.png)
*Figure 7: Qualitative OOD demonstration — a dog in rainbow wings forces the LLM concept extractor to output 0.5 for "vibrant feathers", surfacing a human-intervention signal. This is the only direct evidence the paper offers for its OOD uncertainty claim.*

### Appendix robustness

With Claude 3.5 Haiku on CUB family-level, BC-LLM still leads (Acc 0.665 / AUC 0.849 / Brier 0.443 vs LLM+CBM 0.583 / 0.771 / 0.507). On Imagenette: Acc 0.987 / AUC 0.999 vs 0.902 / 0.988 for LLM+CBM. On fMoW USA: Acc 0.357 / AUC 0.904, dropping to 0.265 / **"[???]"** on OOD-China — the AUC is literally rendered as `[???]` in the v3 published table. The CUB-200 all-class number (Acc 0.562) is a single half-epoch with comparator rows missing.

## Limitations

**Acknowledged by the authors.**

- Method is intended for small $K$ (≤20); scaling to many concepts is left to future work via mini-batching.
- ZSFG experiment cannot be reproduced due to PHI.
- The theoretical guarantee is asymptotic; finite-sample LLM miscalibration can still bias the chain.

**Under-addressed or missing.**

- **No direct calibration metric.** No ECE, reliability diagrams, or temperature-scaling baseline. "Brier ↓" conflates calibration with sharpness.
- **No selective-prediction analysis.** No accuracy-vs-coverage curve, no AURC, no abstention test correlating posterior variance with held-out error — the central claim of the paper's uncertainty-quantification pitch.
- **No MCMC diagnostics.** $T=5$ is far below standard Gibbs convergence thresholds, and no $\hat R$, effective-sample-size, or chain-mixing analysis is shown. Theorem 3.1's guarantee is asymptotic in $n$, not in $T$.
- **No seed / chain-level variance.** All 95% CIs are test-set bootstrap, not re-runs of BC-LLM. With LLM stochasticity at every step this is a meaningful omission.
- **Cost numbers are hand-waved.** No table of actual GPT-4o-mini token counts or dollars-spent vs baselines, despite "cost-effective" being a headline claim.
- **OOD entropy comparison is incomplete.** Only one OOD setup, and ResNet50 wins on entropy. fMoW OOD-China AUC is "[???]" in the published table.
- **Concept-set identifiability.** Near-equivalent concepts (e.g. "smoking history" / "current smoker" / "former smoker") are described as "appropriately unsure", but no quantitative metric of posterior multimodality / mode collapse is given.
- **Manuscript hygiene.** The v3 PDF retains TODOs ("[check timings a bit more carefully]", "[UPDATE FIGURE]") and literal `[???]` entries. For a NeurIPS 2025 paper, the experimental section appears not to have been fully audited before posting.

## Why It Matters for Medical AI

For clinical deployment, "auditable" is not optional — and BC-LLM's pitch of iteratively *discovering* concepts from clinical notes rather than committing to a hand-curated list maps directly onto how clinical reasoning actually evolves over time. The ZSFG case study is the most interesting story in the paper: a real EHR readmission pipeline where the tabular-only model jumps from AUC 0.60 to 0.64 and clinicians rate the surfaced concepts at 2.5/3 versus ≤2.1 for the comparators, with discovered concepts like "frequent previous admissions" and "history of drug use" pointing to actionable follow-up scheduling.

That said, the medical-AI reader should treat the calibration claims with caution. The paper's "rigorous uncertainty quantification" framing rests on a theorem about asymptotic concept-set concentration, not on any direct calibration measurement on the medical task. There is no ECE, no reliability diagram, no abstention or accuracy-vs-coverage curve on the ZSFG cohort, and the only quantitative OOD comparison (CUB) has the black-box baseline being *more* appropriately uncertain than BC-LLM. The MIMIC simulation is a best-case for an LR-based CBM because $Y$ is generated from an LR over the true concepts by construction. Net: the algorithmic contribution is real and the clinical concept discovery is promising, but anyone planning to deploy this as a calibrated risk model should run their own calibration and selective-prediction battery before believing the abstract.

## References

- **Paper:** Feng, Kothari, Zier, Singh, Tan. *Bayesian Concept Bottleneck Models with LLM Priors.* NeurIPS 2025. arXiv:2410.15555 — [https://arxiv.org/abs/2410.15555](https://arxiv.org/abs/2410.15555)
- **Code:** [https://github.com/jjfeng/bc-llm](https://github.com/jjfeng/bc-llm)
- **Related work:** Koh et al., *Concept Bottleneck Models*, ICML 2020; Yuksekgonul et al., *Post-hoc Concept Bottleneck Models*, ICLR 2023; Yang et al., *Language in a Bottle (LaBo)*, CVPR 2023; Oikarinen et al., *Label-Free CBM*, ICLR 2023.
