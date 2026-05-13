---
title: "Towards Multi-Label Concept Bottleneck Models in Medical Imaging: An Exploratory Survey"
excerpt: "First systematic study of label-free CBMs on multi-label chest X-rays — RAD-DINO + BiomedCLIP reaches micro-AUC 0.8823 / macro-F1 0.4076, exposing AUC as misleading for rare pathologies."
categories:
  - Paper
  - Pathology
  - LLM
permalink: /paper/multi-label-cbm-medical-chest-xray/
tags:
  - CBM
  - Concept Bottleneck Models
  - Label-free CBM
  - RAD-DINO
  - BiomedCLIP
  - Chest X-ray
  - Multi-label Classification
  - Medical-AI
toc: true
toc_sticky: true
toc_label: "Contents"
date: 2026-05-12
last_modified_at: 2026-05-12
---

## TL;DR

- The authors run the **first systematic study of label-free Concept Bottleneck Models (CBMs)** on multi-label chest X-ray classification (NIH ChestX-ray14, 14 pathologies) — slotting six backbones (ResNet50, CLIP, BiomedCLIP, BioViL, BioViL-T, RAD-DINO) into a single GPT-4-driven concept pipeline.
- **RAD-DINO image encoder + BiomedCLIP text encoder is the best label-free multi-label CBM**, hitting **micro-AUC 0.8823 / macro-AUC 0.8220 / micro-F1 0.5054 / macro-F1 0.4076** on ChestX-ray14.
- The paper's strongest empirical contribution is negative: per-class AUC stays above ~0.87 on Hernia and Pneumonia while macro-F1 collapses to **0.04–0.07**, showing **AUC is misleading as a multi-label medical CBM metric** — precision/recall/F1 must be co-reported.

## Motivation

Black-box DNNs are not clinically trusted, and the original supervised CBMs (Koh et al., 2020) need expensive radiologist concept annotations. Label-free CBMs (Oikarinen 2023, Yang 2023) bypass that by letting an LLM (GPT-4) propose concepts and a VLM (CLIP / BiomedCLIP / BioViL / RAD-DINO) score image–concept similarity. The trained model is just a small MLP from concept-activation vector to labels — interpretable by construction.

But every prior label-free CBM benchmark is single-label (CUB, CIFAR, ImageNet) or single-label medical. Real chest X-rays are intrinsically **multi-label** — multiple pathologies co-occur, distributions are extremely long-tailed (Infiltration: 19,870 vs. Hernia: 227 — an 87× imbalance), and the concept activations themselves are a second multi-label problem on top of the pathology prediction. Whether label-free CBMs survive these conditions is unknown. This paper is the first to ask, and the authors deliberately frame it as a diagnostic survey rather than a new method.

## Core Innovation

Not a new architecture — a **structured benchmark** with three orthogonal axes:

1. **Six backbones** plugged into the same label-free CBM pipeline: ResNet50, CLIP, BiomedCLIP, BioViL, BioViL-T, RAD-DINO.
2. **Three embedding strategies**: global (CLS / pooled), patch, and combined.
3. **Four imbalance-aware losses**: BCE, weighted BCE, focal loss, two-way multi-label loss (Kobayashi 2023).

The point is to map *which axis matters* and *where label-free CBMs break* on real long-tailed clinical data — focusing on rare-class behavior rather than average AUC.

## Claims & Evidence Analysis

| Claim | Evidence | Dataset | Strength |
|---|---|---|---|
| C1: First systematic study of label-free CBMs in multi-label medical imaging. | Literature review §2; Table 4 categorizes all prior CBMs as multi-class. | ChestX-ray14 only | ⭐⭐ |
| C2: Medical-domain VLMs outperform generic ones (ResNet50, CLIP). | Table 1: BiomedCLIP / BioViL / BioViL-T / RAD-DINO all > ResNet50 on AUC and F1. | ChestX-ray14 | ⭐⭐⭐ |
| C3: RAD-DINO is the best backbone for label-free multi-label CBMs. | Table 1: RAD-DINO wins 7/8 metrics. | ChestX-ray14 | ⭐⭐ |
| **C4: AUC is a poor indicator of multi-label CBM prediction quality.** | Figures 3, 5: AUC ≈ 0.87 on Hernia vs. F1 ≈ 0.04. | ChestX-ray14 | ⭐⭐⭐ |
| C5: Class-balanced losses (WBCE / two-way) improve minority-class precision/recall. | Figure 4 per-class bar plots across 4 losses. | ChestX-ray14 | ⭐⭐ |
| C6: Global embeddings are the most reliable strategy. | Table 2 across BioViL / ResNet50 / BioViL-T. | ChestX-ray14 | ⭐⭐ |
| C7: Pseudo-concepts are reliable on common conditions but unstable on rare ones. | Section 4.4.4 + Figure 7: Cardiomegaly top-3 correct, Hernia none in overall top-10. | ChestX-ray14 | ⭐⭐ |
| C8: Pipeline supports test-time concept intervention / interpretability. | Implied by CBM design; not directly demonstrated. | — | ⭐ |

**Honest read.** C4 is the strongest, cleanest finding and genuinely changes how a reader should evaluate medical CBMs. C2 is solidly multi-VLM. Everything else is **single-dataset, single-run, no error bars** — no CheXpert / MIMIC-CXR cross-eval, no seed variance, no significance tests, no non-CBM baseline. C8 is inherited from CBM theory but never tested. Read the headline claims as hypothesis-generating, not confirmatory.

## Method & Architecture

![Label-free multi-label CBM pipeline](/assets/images/paper/multi-label-cbm-medical/page_005.png)
*Figure 1: Label-free multi-label CBM pipeline — GPT-4 generates 56 radiology concepts, a frozen VLM aligns image features to concept embeddings, and only a small MLP is trained on top of the concept bottleneck.*

The pipeline has four stages, all using **frozen** image and text encoders — only a small MLP is trained:

1. **Concept generation (LLM).** GPT-4 is prompted twice per pathology: "What are the useful radiology descriptors to distinguish {class}?" and "Which concepts distinguish {A} from {B}?" Raw outputs are filtered for length, redundancy, and overlap with the pathology name. **Final bank: 56 concepts for 14 pathologies** (e.g., for *Mass*: "pulmonary mass", "space occupying lesion", "well defined opacity"; for *Cardiomegaly*: "enlarged cardiac silhouette", "increased cardiothoracic ratio").
2. **Concept embedding (text encoder $E_T$).** Each concept $c_j$ is encoded with the frozen text encoder of the chosen VLM (PubMedBERT for BiomedCLIP, CXR-BERT for BioViL/BioViL-T, ViT-B/L text head for CLIP). RAD-DINO is image-only, so its concepts are encoded with BiomedCLIP's text encoder following Barsellotti 2025.
3. **Concept–image alignment.** Frozen image encoder $E_I$ produces an image embedding; cosine similarity to each concept embedding gives the **concept activation vector** $S_i \in \mathbb{R}^{|C|}$:

$$S_{i,j} = \frac{E_I(x_i) \cdot E_T(c_j)}{\|E_I(x_i)\|\,\|E_T(c_j)\|}$$

Three strategies are compared: global (CLS / pooled), patch (patch-level features aligned to concepts), and combined (concat).

4. **Multi-label classifier.** A small MLP $f_\theta : \mathbb{R}^{|C|} \to [0,1]^{14}$ with sigmoid output predicts all 14 pathologies. Because the only trainable parameters are in the MLP, **every prediction is forced through the 56-concept bottleneck**.

Training: Adam, lr = 1e-3, batch size 256, 100 epochs, dropout + weight decay + early stopping on validation loss. Per-label optimal thresholds tuned at evaluation time.

### Dataset

![NIH ChestX-ray14 label distribution](/assets/images/paper/multi-label-cbm-medical/page_008.png)
*Figure 2: NIH ChestX-ray14 label distribution — Infiltration appears in 19,870 images vs. Hernia in just 227, an 87× imbalance that drives the rare-class failure modes throughout the paper.*

**NIH ChestX-ray14** (Wang et al., 2017): 112,120 frontal radiographs from 30,805 patients, 14 thoracic pathologies labelled via NLP from radiology reports (noisy weak supervision). Official patient-wise split: 70% train / 10% val / 20% test. Single-institution, frontal-view only, no demographic balancing — and no external dataset (CheXpert / MIMIC-CXR / PadChest) is used for OOD validation.

## Experimental Results

### Backbone comparison (global embeddings, plain BCE)

| Backbone | AUC-mi | AUC-ma | F1-mi | F1-ma | Pr-mi | Pr-ma | Re-mi | Re-ma |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | 0.7304 | 0.5067 | 0.3556 | 0.1105 | 0.2564 | 0.0869 | 0.5801 | 0.2543 |
| CLIP | 0.7836 | 0.6256 | 0.3125 | 0.2210 | 0.2005 | 0.1462 | 0.7074 | 0.5353 |
| BiomedCLIP | 0.8296 | 0.7375 | 0.4163 | 0.3014 | 0.3178 | 0.2437 | 0.6031 | 0.4483 |
| BioViL | 0.7883 | 0.6436 | 0.3430 | 0.2131 | 0.2345 | 0.1513 | 0.6384 | 0.4120 |
| BioViL-T | 0.7838 | 0.6297 | 0.3430 | 0.2041 | 0.2382 | 0.1651 | 0.6127 | 0.3762 |
| **RAD-DINO** | **0.8823** | **0.8220** | **0.5054** | **0.4076** | **0.4345** | **0.3643** | 0.6079 | **0.4908** |

RAD-DINO wins on 7/8 metrics; BiomedCLIP is consistently second-best. Even so, the *winner's* macro-F1 is 0.4076 — meaning more than half of per-class F1 mass is still "missing."

### The AUC-vs-F1 gap (the paper's key result)

![Per-class metrics for the best CBM](/assets/images/paper/multi-label-cbm-medical/page_011.png)
*Figure 3: Per-pathology AUC / Recall / F1 / Precision for the best CBM (RAD-DINO). AUC stays roughly flat (~0.7–0.93) across all 14 pathologies, but Precision/F1 collapse on Hernia and Pneumonia — Pneumonia macro-F1 ≈ 0.07, Hernia macro-F1 ≈ 0.04 with plain BCE.*

This is the paper's central finding. **High per-class AUC is compatible with near-zero per-class F1** under multi-label imbalance — a well-known scoring artifact, but one that is almost universally ignored in CBM literature where AUC is the headline metric.

### Confusion matrices — majority vs. minority

![Majority-class confusion matrices](/assets/images/paper/multi-label-cbm-medical/page_012.png)
*Figure 4: Majority-class confusion matrices — Effusion (TPR 0.73) and Infiltration (TPR 0.75) are predicted reliably.*

![Minority-class confusion matrices](/assets/images/paper/multi-label-cbm-medical/page_012.png)
*Figure 5: Minority-class confusion matrices — 69% of Hernia cases and 76% of Pneumonia cases are missed (false negatives), despite per-class AUC > 0.87.*

### Loss-function ablation

![Loss-function ablation](/assets/images/paper/multi-label-cbm-medical/page_011.png)
*Figure 6: Loss-function ablation across all 14 classes. AUC barely moves under WBCE / focal / two-way loss, but minority-class recall and F1 rise visibly — WBCE lifts Hernia recall from ~0.02 (BCE) to ~0.31, and two-way loss lifts Pneumonia recall from 0.10 to 0.24.*

Embedding-type ablation (Table 2, BioViL / ResNet50 / BioViL-T × global/patch/combined): **global wins** on micro/macro AUC and F1; patch trades precision for recall; combined offers no consistent gain. Best in that sub-table: BioViL global (mi/ma AUC = 0.7883 / 0.6436); BioViL-T patch has the best macro-recall (0.435).

### Interpretability — concept contributions

![Top concepts for Infiltration](/assets/images/paper/multi-label-cbm-medical/page_014.png)
*Figure 7: Top positive concepts for Infiltration ("diffuse infiltrates", "hazy opacities", "interstitial edema", "interstitial markings", "alveolar infiltrate") are radiologically appropriate — concept attribution works on common classes.*

![Top concepts for Hernia](/assets/images/paper/multi-label-cbm-medical/page_014.png)
*Figure 8: Top concepts for Hernia ("flattened diaphragms", "hyperinflated lungs", "bowel loops above diaphragm", "stomach bubble in chest") are clinically plausible, but the authors note that none of Hernia's GPT-generated concepts appear in the model's overall top-10 — concept reliability degrades with sample count.*

## Limitations

Authors acknowledge: concept fidelity degrades on rare classes; AUC alone is misleading; imbalance-aware losses help but do not fully close the rare-class gap; concept prediction is itself a noisy multi-label problem.

Not addressed:

- **No external validation.** Single dataset (NIH ChestX-ray14); no CheXpert / MIMIC-CXR / PadChest / ChestX-Det. NIH labels are NLP-extracted and noisy — some "rare-class precision failures" might be label noise.
- **No seed variance, no confidence intervals, no significance tests.** All numbers are single-run.
- **No end-to-end (non-CBM) baseline** with matched backbones and losses, so the **interpretability tax** is unquantified.
- **No human / radiologist evaluation** of whether the 56 GPT-4 concepts are clinically faithful or sufficient.
- **Test-time concept intervention** — the marquee CBM feature (Koh 2020) — is never demonstrated.
- **Concept set is small** (56 concepts / 14 classes ≈ 4 per class); set size is not treated as a variable.
- **Patch embeddings underperform**, but no analysis of *why* — patch-token / concept alignment is a known open problem.
- **No segmentation or localization**, despite RAD-DINO being a strong spatial encoder.

## Why It Matters for Medical AI

For anyone shipping interpretable medical classifiers, the actionable takeaways are concrete and somewhat uncomfortable:

1. **Stop reporting only AUC.** Multi-label medical CBMs can hit AUC ≈ 0.88 while missing 70–80% of rare-pathology cases. Macro-F1, per-class precision/recall, and confusion matrices must be co-reported.
2. **Pick a domain-specific backbone.** Generic CLIP / ResNet50 are clearly worse than BiomedCLIP / BioViL / RAD-DINO on chest X-rays — at no extra training cost since encoders are frozen.
3. **RAD-DINO is a strong default** for the image side; BiomedCLIP for the text side; global pooling beats patch alignment until that's solved.
4. **Imbalance-aware losses are necessary but not sufficient.** WBCE and two-way loss help minority recall but do not close the F1 gap. Long-tailed clinical labels likely need data-side or hierarchy-side fixes too.
5. **Treat label-free CBM concepts as plausible but unverified.** Without a radiologist eyeballing the GPT-4 concept set, the "interpretability" guarantee is loose — especially on rare pathologies where concept attribution becomes unstable.

## References

- Mpinda, B. N., Hosseinzadeh, M., Bundele, V., & Lensch, H. P. A. (2026). *Towards Multi-Label Concept Bottleneck Models in Medical Imaging: An Exploratory Survey.* MIDL 2026 (under review). OpenReview ID: MeOQtY5kVM.
- Koh, P. W., et al. (2020). *Concept Bottleneck Models.* ICML.
- Oikarinen, T., et al. (2023). *Label-Free Concept Bottleneck Models.* ICLR.
- Yang, Y., et al. (2023). *Language in a Bottle.* CVPR.
- Wang, X., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database.* CVPR.
- Pérez-García, F., et al. (2024). *RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision.*
- Zhang, S., et al. (2023). *BiomedCLIP: a multimodal biomedical foundation model.*
- Boecking, B., et al. (2022). *BioViL: Making the Most of Text Semantics in Biomedical Vision–Language Processing.*
- Kobayashi, T. (2023). *Two-Way Multi-Label Loss.* CVPR.
- Lin, T.-Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
