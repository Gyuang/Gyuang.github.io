---
published: true
title: "Enhancing radiomics features via a large language model for classifying benign and malignant breast tumors in mammography"
excerpt: "Enhancing radiomics features via a large language model for classifying benign and malignant breast tumors in mammography 논문 요약"

categories:
  - VLM
tags:
  - [VLM, Vision-Language, Vision-Language, Computer Methods and Programs in Biomedicine]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

Traditional radiomics analysis of mammography images relies solely on quantitative features extracted from medical images, limiting its ability to incorporate clinical context and domain knowledge. This study addresses these limitations by leveraging large language models (LLMs) to enhance radiomics features with clinical knowledge, thereby improving the classification of benign and malignant breast tumors in mammography through a novel fusion approach that combines image-derived features with text-based clinical insights.

## Methods

The proposed approach enhances traditional radiomics analysis through LLM-guided feature augmentation:

1. **Radiomics Feature Extraction**: Extract quantitative features from mammography images including shape, texture, and intensity-based descriptors from regions of interest (ROIs) containing breast lesions

2. **Clinical Knowledge Integration**: Utilize pre-trained large language models to encode domain-specific clinical knowledge about breast cancer characteristics, imaging patterns, and diagnostic criteria

3. **Text-Based Feature Generation**: Generate textual descriptions of radiological findings and convert them into numerical representations using LLM embeddings to capture semantic relationships

4. **Feature Fusion Strategy**: Combine traditional radiomics features with LLM-derived clinical knowledge features through concatenation and weighted fusion approaches

5. **Multi-Modal Classification**: Train classifiers on the enhanced feature set using machine learning algorithms (e.g., Random Forest, SVM) to distinguish between benign and malignant breast tumors

6. **Feature Selection and Optimization**: Apply feature selection techniques to identify the most discriminative combined features and optimize model hyperparameters for improved classification performance

7. **Cross-Validation and Evaluation**: Implement k-fold cross-validation to assess model robustness and compare performance against baseline radiomics-only approaches using metrics such as accuracy, sensitivity, specificity, and AUC

## Dataset

The study utilized mammography datasets containing both benign and malignant breast lesions with corresponding radiological annotations. The dataset includes digital mammography images with expert-annotated regions of interest (ROIs) marking suspicious lesions, along with clinical metadata such as patient demographics, imaging characteristics, and histopathological confirmation of diagnosis. Data preprocessing involved image normalization, ROI standardization, and quality control measures to ensure consistent radiomics feature extraction across all cases.





## Results

The LLM-enhanced radiomics approach demonstrated significant improvements in breast tumor classification performance compared to traditional radiomics-only methods. The fusion of image-derived radiomics features with LLM-generated clinical knowledge features achieved higher accuracy, sensitivity, and specificity in distinguishing benign from malignant breast lesions. The enhanced model showed improved AUC scores and reduced false positive rates, indicating better diagnostic reliability. Ablation studies confirmed that the integration of clinical knowledge through LLMs provided complementary information that traditional image features alone could not capture, resulting in more robust and clinically relevant classification outcomes.