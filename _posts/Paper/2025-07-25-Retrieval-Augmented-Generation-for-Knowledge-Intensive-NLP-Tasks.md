---
categories:
- RAG
date: 2025-07-25
excerpt: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 논문 요약
last_modified_at: 2025-07-25
published: true
tags:
- - RAG
  - Retrieval
  - NLP
  - Language Model
title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
toc: true
toc_sticky: true
---

## Introduction

Knowledge-intensive NLP tasks require systems to access and manipulate large amounts of world knowledge, but traditional pre-trained language models struggle with precise knowledge access and cannot easily update their parametric knowledge. This paper introduces Retrieval-Augmented Generation (RAG), which combines pre-trained parametric models with non-parametric retrieval mechanisms to enhance performance on knowledge-intensive tasks while providing interpretable, updatable knowledge access.

## Methods

RAG operates through a systematic approach that combines parametric and non-parametric knowledge:

1. **Document Encoding**: Pre-encode a knowledge corpus (e.g., Wikipedia) using DPR (Dense Passage Retrieval) to create dense vector representations of text passages

2. **Query Processing**: For each input query, use a bi-encoder retrieval system to find the top-k most relevant documents from the pre-encoded corpus

3. **Context Augmentation**: Concatenate retrieved passages with the original input to create an augmented context that contains both the query and relevant background knowledge

4. **Generation**: Feed the augmented input through a pre-trained sequence-to-sequence model (BART) that generates responses conditioned on both the original query and retrieved knowledge

5. **End-to-End Training**: Train the entire system jointly, allowing the retriever and generator to learn complementary representations for knowledge-intensive tasks

6. **Two RAG Variants**:
   - **RAG-Sequence**: Retrieves documents once for the entire sequence generation
   - **RAG-Token**: Retrieves different documents for each token generation step, allowing more dynamic knowledge access

## Dataset

The paper evaluates RAG on multiple knowledge-intensive NLP benchmarks:

- **Open-domain QA**: Natural Questions, TriviaQA, WebQuestions, and CuratedTREC datasets requiring factual knowledge retrieval
- **Knowledge Corpus**: Uses Wikipedia as the non-parametric knowledge source, with approximately 21 million passages
- **Fact Verification**: FEVER dataset for claim verification against evidence passages  
- **Jeopardy Question Generation**: Tasks requiring both knowledge retrieval and creative generation

All datasets test the system's ability to access external knowledge beyond what's stored in model parameters, with Wikipedia providing a comprehensive, updatable knowledge base for retrieval.





## Results

![Results Table 16 0](/assets/images/paper/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/results_table_16_0.png)
*Figure: Results Table 16 0*


RAG demonstrates significant improvements across knowledge-intensive tasks compared to state-of-the-art parametric models. On open-domain question answering, RAG achieves substantial gains over BART baseline: +6.7% on Natural Questions, +4.4% on WebQuestions, and +2.9% on CuratedTREC. The system particularly excels in scenarios requiring up-to-date factual knowledge, as the non-parametric retrieval component can access current information without retraining. Additionally, RAG provides interpretable results by surfacing the specific passages used for generation, addressing the "black box" limitation of purely parametric approaches while maintaining competitive generation quality.

## Conclusion

RAG represents a fundamental breakthrough in combining parametric and non-parametric knowledge for NLP tasks. By integrating dense retrieval with pre-trained generation models, RAG addresses key limitations of purely parametric approaches: knowledge staleness, lack of interpretability, and difficulty in knowledge updates. The architecture's ability to leverage external knowledge sources while maintaining end-to-end trainability establishes RAG as a foundational framework for knowledge-intensive applications.

## Key Takeaways

- **Hybrid Architecture**: Successfully combines parametric (model weights) and non-parametric (external corpus) knowledge sources
- **Interpretability**: Provides transparent access to source knowledge through retrieved passages
- **Updatable Knowledge**: Enables knowledge updates without retraining by modifying the retrieval corpus
- **Strong Performance**: Achieves state-of-the-art results on multiple knowledge-intensive benchmarks
- **Foundation for Future Work**: Establishes the RAG paradigm that influences subsequent retrieval-augmented systems