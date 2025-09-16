---
categories:
- Medical AI
date: 2025-07-25
excerpt: 에 대한 체계적 분석
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- LLM
- Radiomics
- Breast Cancer
- Mammography
- Medical Imaging
- Classification
title: Enhancing radiomics features via a large language model for classifying benign
  and malignant breast tumors in mammography
toc: true
toc_sticky: true
---

# Enhancing radiomics features via a large language model for classifying benign and malignant breast tumors in mammography

## 논문 정보
- **저자**: 
- **발표**: 
- **ArXiv**: N/A

## 1. 핵심 요약 (2-3문장)
이 논문의 핵심 기여와 주요 발견을 간결하게 요약합니다.

## 2. 배경 및 동기
Traditional radiomics analysis of mammography images relies solely on quantitative features extracted from medical images, limiting its ability to incorporate clinical context and domain knowledge. This study addresses these limitations by leveraging large language models (LLMs) to enhance radiomics features with clinical knowledge, thereby improving the classification of benign and malignant breast tumors in mammography through a novel fusion approach that combines image-derived features with text-based clinical insights.

## 3. 제안 방법

### 3.1 아키텍처 개요


### 3.2 핵심 기술/알고리즘
The proposed approach enhances traditional radiomics analysis through LLM-guided feature augmentation:

1. **Radiomics Feature Extraction**: Extract quantitative features from mammography images including shape, texture, and intensity-based descriptors from regions of interest (ROIs) containing breast lesions

2. **Clinical Knowledge Integration**: Utilize pre-trained large language models to encode domain-specific clinical knowledge about breast cancer characteristics, imaging patterns, and diagnostic criteria

3. **Text-Based Feature Generation**: Generate textual descriptions of radiological findings and convert them into numerical representations using LLM embeddings to capture semantic relationships

4. **Feature Fusion Strategy**: Combine traditional radiomics features with LLM-derived clinical knowledge features through concatenation and weighted fusion approaches

5. **Multi-Modal Classification**: Train classifiers on the enhanced feature set using machine learning algorithms (e.g., Random Forest, SVM) to distinguish between benign and malignant breast tumors

6. **Feature Selection and Optimization**: Apply feature selection techniques to identify the most discriminative combined features and optimize model hyperparameters for improved classification performance

7. **Cross-Validation and Evaluation**: Implement k-fold cross-validation to assess model robustness and compare performance against baseline radiomics-only approaches using metrics such as accuracy, sensitivity, specificity, and AUC



The study utilized mammography datasets containing both benign and malignant breast lesions with corresponding radiological annotations. The dataset includes digital mammography images with expert-annotated regions of interest (ROIs) marking suspicious lesions, along with clinical metadata such as patient demographics, imaging characteristics, and histopathological confirmation of diagnosis. Data preprocessing involved image normalization, ROI standardization, and quality control measures to ensure consistent radiomics feature extraction across all cases.

### 3.3 구현 세부사항


## 4. 실험 및 결과

### 4.1 실험 설정
실험에 사용된 데이터셋, 평가 지표, 비교 대상을 설명합니다.

### 4.2 주요 결과


The LLM-enhanced radiomics approach demonstrated significant improvements in breast tumor classification performance compared to traditional radiomics-only methods. The fusion of image-derived radiomics features with LLM-generated clinical knowledge features achieved higher accuracy, sensitivity, and specificity in distinguishing benign from malignant breast lesions. The enhanced model showed improved AUC scores and reduced false positive rates, indicating better diagnostic reliability. Ablation studies confirmed that the integration of clinical knowledge through LLMs provided complementary information that traditional image features alone could not capture, resulting in more robust and clinically relevant classification outcomes.

### 4.3 분석
결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향
이 연구의 학술적 기여와 실용적 가치를 평가합니다.

## 6. 개인적 평가

**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점  
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준

