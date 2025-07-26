---
published: true
title: "Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes"
excerpt: "Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes 논문 요약"

categories:
  - Paper
tags:
  - [Chain-of-Thought, RAG, Rare Disease, Clinical Notes, Medical AI]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

Rare disease diagnosis from clinical notes presents significant challenges due to the complexity of phenotype-driven gene prioritization and the unstructured nature of real-world clinical documentation. This paper addresses these challenges by integrating Chain-of-Thought (CoT) reasoning with Retrieval-Augmented Generation (RAG) to enhance large language models' ability to process unstructured clinical notes and improve rare disease diagnostic accuracy.

## Methods

The proposed approach integrates Chain-of-Thought reasoning with Retrieval-Augmented Generation through the following systematic methodology:

• **Clinical Note Processing**: Raw unstructured clinical notes are preprocessed to extract relevant phenotypic information and clinical observations without requiring conversion to standardized HPO terms

• **Retrieval-Augmented Generation Setup**: A knowledge base containing rare disease information, gene-phenotype associations, and clinical diagnostic criteria is constructed to support the retrieval component

• **Chain-of-Thought Prompting**: The LLM is instructed to follow a structured reasoning process that mimics clinical diagnostic workflows, breaking down the diagnosis into intermediate reasoning steps

• **Multi-step Reasoning Framework**: The system first identifies key phenotypic features from clinical notes, then retrieves relevant disease information, and finally applies systematic reasoning to prioritize candidate genes

• **Knowledge Retrieval Integration**: Retrieved information from the knowledge base is dynamically incorporated into the reasoning chain to provide domain-specific context and support evidence-based decision making

• **Gene Prioritization Process**: The final step involves ranking candidate genes based on the integrated evidence from both the reasoning chain and retrieved knowledge, producing a ranked list of potential genetic causes

• **Iterative Refinement**: The system can refine its reasoning by incorporating additional retrieved information or adjusting the reasoning chain based on intermediate results





## Dataset

The evaluation was conducted using clinical datasets containing unstructured clinical notes with documented rare disease cases. The datasets included real-world clinical narratives with varying levels of detail and complexity, representing diverse rare disease presentations. Ground truth labels were established through confirmed genetic diagnoses, providing a robust evaluation framework for assessing the accuracy of phenotype-driven gene prioritization from unstructured clinical text.

## Results

The integration of Chain-of-Thought reasoning with Retrieval-Augmented Generation demonstrated significant improvements in rare disease diagnosis accuracy compared to baseline LLM approaches. The method showed enhanced performance in gene prioritization tasks, with notable improvements in handling complex phenotypic presentations and ambiguous clinical descriptions. The systematic reasoning approach enabled by CoT, combined with the domain-specific knowledge provided by RAG, resulted in more accurate and clinically relevant diagnostic suggestions, particularly for cases where traditional HPO-based approaches struggled with the unstructured nature of clinical documentation.