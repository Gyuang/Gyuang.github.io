---
categories:
- Medical AI
date: 2025-07-25
excerpt: "Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis\
  \ Framework with Prompt-Generated Rationales \uB17C\uBB38 \uC694\uC57D"
last_modified_at: 2025-07-25
published: true
tags:
- LLM
- Clinical Reasoning
- Medical AI
- Prompt Learning
- Diagnosis
title: 'Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework
  with Prompt-Generated Rationales'
toc: true
toc_sticky: true
---

## Introduction

Clinical diagnosis requires complex reasoning that involves analyzing symptoms, medical history, and clinical findings to reach accurate conclusions, yet most existing medical NLP systems focus only on classification without providing interpretable reasoning pathways. This paper introduces a reasoning-aware diagnosis framework that leverages large language models to generate clinical rationales through prompt-based learning, eliminating the need for expensive clinician annotations while providing transparent diagnostic reasoning that mirrors clinical decision-making processes.

## Related Work 

### Clinical Reasoning Systems

Previous clinical AI systems primarily focused on diagnostic classification without providing reasoning explanations, limiting their clinical interpretability and trustworthiness.

### Prompt-based Learning

Existing prompt-based approaches in medical NLP have shown promise for few-shot learning but have not been systematically applied to generate clinical reasoning rationales for diagnosis.

## Methods

The reasoning-aware diagnosis framework operates through a two-stage approach that first generates clinical rationales and then leverages these rationales for improved diagnostic prediction:

### Step-by-Step Process

1. **Clinical Case Encoding**: Transform raw clinical data including patient symptoms, medical history, and examination findings into structured text representations that capture relevant clinical context and temporal relationships

2. **Prompt-based Rationale Generation**: Employ carefully designed prompts to guide large language models in generating step-by-step clinical reasoning paths that mimic physician thought processes, including differential diagnosis considerations and evidence evaluation

3. **Multi-stage Prompting Strategy**: Implement a hierarchical prompting approach where initial prompts elicit broad clinical reasoning, followed by focused prompts that refine specific diagnostic hypotheses and supporting evidence

4. **Rationale-aware Diagnosis Module**: Integrate the generated clinical rationales with original clinical features through attention mechanisms to produce final diagnostic predictions that are both accurate and explainable

5. **Consistency Regularization**: Apply consistency constraints between rationale-based and feature-based predictions to ensure coherent reasoning pathways that align with diagnostic outcomes

6. **Self-training Framework**: Iteratively improve rationale quality through pseudo-labeling techniques where high-confidence rationale-diagnosis pairs are used to augment training data without requiring additional clinical annotations


### Key Technical Components

- **Prompt Engineering**: Design domain-specific prompts that capture clinical reasoning patterns and diagnostic workflows used by practicing physicians
- **Rationale Integration**: Develop attention-based fusion mechanisms to effectively combine generated rationales with structured clinical features for enhanced diagnostic accuracy
- **Quality Control**: Implement automated quality assessment metrics to filter low-quality generated rationales and maintain reasoning coherence





## Dataset

The framework was evaluated on multiple clinical datasets to demonstrate its effectiveness across different medical domains and diagnostic scenarios:

**MIMIC-III Clinical Dataset**: Large-scale electronic health records containing admission notes, discharge summaries, and diagnostic codes for over 40,000 patients, providing diverse clinical scenarios and diagnostic challenges for comprehensive evaluation.

**DDXPlus Synthetic Dataset**: Structured differential diagnosis dataset with 1.3 million synthetic patient cases covering 49 medical conditions, designed specifically for evaluating automated diagnostic reasoning systems with ground-truth diagnostic pathways.

**PubMedQA Biomedical Dataset**: Collection of biomedical research questions and evidence-based answers that test the model's ability to reason from medical literature and clinical evidence for diagnostic decision-making.

## Results

The reasoning-aware diagnosis framework demonstrated significant improvements in both diagnostic accuracy and reasoning quality compared to baseline approaches. On the MIMIC-III dataset, the framework achieved 89.2% diagnostic accuracy (vs. 83.1% for standard classification baselines) while generating clinically coherent rationales that aligned with physician reasoning patterns. The DDXPlus evaluation showed 92.7% accuracy with substantial improvements in differential diagnosis ranking, where the generated rationales helped identify relevant alternative diagnoses. Human evaluation by clinical experts rated 85% of generated rationales as clinically plausible and useful for diagnostic decision support, with particular strength in symptom analysis and evidence integration steps of the reasoning process.

## Conclusion

This work demonstrates that large language models can effectively serve as clinical reasoners by generating high-quality diagnostic rationales through prompt-based learning, eliminating the need for expensive clinician annotations while maintaining clinical accuracy. The reasoning-aware framework provides a practical solution for developing interpretable medical AI systems that can support clinical decision-making with transparent reasoning pathways that align with physician thought processes.

## Key Takeaways

- **Cost-effective Rationale Generation**: Prompt-based approaches can generate clinically plausible reasoning without requiring expensive expert annotations, making interpretable medical AI more accessible
- **Improved Diagnostic Performance**: Integrating generated rationales with clinical features enhances diagnostic accuracy compared to classification-only approaches
- **Clinical Interpretability**: The framework produces reasoning pathways that are comprehensible and useful to healthcare professionals for diagnostic decision support
- **Scalable Framework**: The approach can be applied across different medical domains and clinical datasets without domain-specific architectural modifications