---
categories:
- VLM
date: 2025-07-25
excerpt: 에 대한 체계적 분석
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Vision-Language
- Test-Time Adaptation
- Zero-shot Learning
- Prompt Tuning
- CLIP
title: Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models
  (TPT)
toc: true
toc_sticky: true
---

# Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models (TPT)

## 논문 정보
- **저자**: 
- **발표**: 
- **ArXiv**: N/A

## 1. 핵심 요약 (2-3문장)
이 논문의 핵심 기여와 주요 발견을 간결하게 요약합니다.

## 2. 배경 및 동기
![Figure 1 3](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_1_3.png)
*Figure: Figure 1 3*

Test-Time Prompt Tuning (TPT) introduces a novel approach that learns adaptive prompts on-the-fly with single test samples, addressing the limitation that training on domain-specific data reduces generalization capability to unseen domains. Unlike traditional prompt tuning methods that require additional training data, TPT maintains the zero-shot setting while improving model performance through test-time adaptation.

**Key Innovation**: TPT adapts prompts individually for each test sample without requiring additional training data, enabling dynamic optimization that improves zero-shot performance while preserving generalization capabilities.

**Paper Details**:
- **Authors**: Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, Chaowei Xiao
- **Publication**: NeurIPS 2022
- **arXiv**: [https://arxiv.org/abs/2209.07511](https://arxiv.org/abs/2209.07511)

## 3. 제안 방법

### 3.1 아키텍처 개요

![Architecture Diagram 5 5](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_5.png)
*Figure: Architecture Diagram 5 5*


![Architecture Overview 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_overview_1.png)
*Figure: Architecture Overview 1*



### 3.2 핵심 기술/알고리즘
![Architecture Diagram 5 5](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_5.png)
*Figure: Architecture Diagram 5 5*



![Method Diagram 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/method_diagram_1.png)
*Figure: Method Diagram 1*

TPT는 테스트 시점에서 각 샘플별로 프롬프트를 최적화합니다:

```
[Test Image] → [Data Augmentation] → [Multiple Views]
       ↓                                    ↓
[Learnable Prompts] → [CLIP Text Encoder] → [Text Features]
       ↑ Updated per sample                  ↓
   [Entropy Loss] ← [Confidence Selection] ← [Predictions]
```

각 테스트 이미지에 대해 entropy minimization으로 프롬프트를 개별 최적화합니다.



**Single-Sample Adaptation Paradigm**
- Adapt prompts individually for each test sample without requiring additional training data
- Maintain zero-shot generalization capability by avoiding domain-specific training
- Process each test sample independently with its own prompt optimization

**Core Philosophy**
- Leverage test-time information without violating zero-shot constraints
- Optimize prompts based on consistency across augmented views of the same sample
- Enable dynamic adaptation to distribution shifts at inference time



**Marginal Entropy Minimization**
```
L_ent = -∑ p(y) log p(y)
```

Where:
- `p(y)` is the marginal prediction probability across augmented views
- Minimize uncertainty in model predictions
- Encourage consistent predictions for the same test image under different augmentations

**Mathematical Formulation**
- Generate multiple augmented views of input test image
- Compute prediction probabilities for each augmented view
- Minimize entropy across these predictions to improve consistency
- Update learnable prompt parameters to reduce prediction uncertainty



**Reliability-Based Filtering**
- Filter out noisy or unreliable augmentations during prompt optimization
- Select high-confidence predictions to improve adaptation stability
- Prevent degradation from low-quality augmented samples

**Selection Strategy**
- Identify augmentations with high prediction confidence
- Use confidence threshold to filter unreliable samples
- Weighted combination of reliable predictions for stable optimization



**Iterative Optimization**
1. Generate multiple augmented views of the input test image
2. Compute initial predictions using pre-trained prompts
3. Iteratively update prompt parameters to minimize entropy across views
4. Select confident predictions for stable optimization
5. Output final prediction with adapted prompts

**Augmentation Strategy**
- Apply diverse image transformations (rotation, cropping, color jittering)
- Maintain semantic content while introducing visual variations
- Create sufficient diversity for meaningful entropy minimization



![Architecture Overview 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_overview_1.png)
*Figure: Architecture Overview 1*

**Seamless Framework Integration**
- Work within existing vision-language model frameworks
- Tune only the prompt embeddings while keeping pre-trained model weights frozen
- Leverage CLIP's text-image alignment for effective prompt adaptation

**Parameter Efficiency**
- Minimal computational overhead compared to full model fine-tuning
- Only prompt parameters require optimization during test time
- Preserve pre-trained knowledge while enabling task-specific adaptation



The evaluation covers multiple datasets testing different aspects of generalization:

**Natural Distribution Shifts**
- **ImageNet-A**: Adversarial examples and challenging cases
- **ImageNet-V2**: Matched frequency distribution shift
- **ImageNet-R**: Artistic renditions and stylized images
- **ImageNet-Sketch**: Black and white sketch representations

**Cross-Dataset Evaluation**
- Testing generalization across different image classification benchmarks
- Evaluation on datasets with varying visual characteristics
- Assessment of domain transfer capabilities

**Specialized Benchmarks**
- **Bongard-HOI**: Context-dependent visual reasoning tasks
- **Standard benchmarks**: Various computer vision datasets for comprehensive evaluation



1. **Test-Time Adaptation**: Dynamic prompt optimization at inference time can significantly improve zero-shot performance without requiring additional training data

2. **Entropy-Based Consistency**: Minimizing prediction entropy across augmented views provides an effective self-supervision signal for prompt adaptation

3. **Confidence Selection**: Filtering unreliable augmentations is crucial for stable and effective test-time optimization

4. **Zero-Shot Preservation**: The method successfully improves performance while maintaining the zero-shot setting, making it practical for real-world deployment

The key innovation lies in the test-time adaptation mechanism that dynamically optimizes prompts for individual samples, enabling better generalization compared to static prompts learned during training time.



![Figure 1 9](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_1_9.png)
*Figure: Figure 1 9*

![Figure 1 11](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_1_11.png)
*Figure: Figure 1 11*

![Figure 1 12](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_1_12.png)
*Figure: Figure 1 12*

![Figure 1 13](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_1_13.png)
*Figure: Figure 1 13*

![Figure 5 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_5_0.png)
*Figure: Figure 5 0*

![Figure 5 2](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_5_2.png)
*Figure: Figure 5 2*

![Figure 5 4](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/figure_5_4.png)
*Figure: Figure 5 4*

![Architecture Diagram 5 6](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_6.png)
*Figure: Architecture Diagram 5 6*

![Architecture Diagram 5 7](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_7.png)
*Figure: Architecture Diagram 5 7*

![Architecture Diagram 5 8](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_8.png)
*Figure: Architecture Diagram 5 8*

![Architecture Diagram 5 9](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/architecture_diagram_5_9.png)
*Figure: Architecture Diagram 5 9*

![Results Table 7 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_7_1.png)
*Figure: Results Table 7 1*

![Results Table 18 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_18_0.png)
*Figure: Results Table 18 0*

![Results Table 19 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_19_0.png)
*Figure: Results Table 19 0*

![Results Table 19 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_19_1.png)
*Figure: Results Table 19 1*

![Results Table 19 2](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_19_2.png)
*Figure: Results Table 19 2*

### 3.3 구현 세부사항

![Method Diagram 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/method_diagram_1.png)
*Figure: Method Diagram 1*



## 4. 실험 및 결과

### 4.1 실험 설정
실험에 사용된 데이터셋, 평가 지표, 비교 대상을 설명합니다.

### 4.2 주요 결과

![Results Table 7 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_7_0.png)
*Figure: Results Table 7 0*


![Results Table 7 1](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_7_1.png)
*Figure: Results Table 7 1*


![Results Table 18 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_18_0.png)
*Figure: Results Table 18 0*



![Results Table 7 0](/assets/images/paper/tpt-test-time-prompt-tuning-for-zero-shot-generalization/results_table_7_0.png)
*Figure: Results Table 7 0*

TPT demonstrates significant improvements in zero-shot performance across multiple challenging scenarios:

**Overall Performance Gains**
- Improves CLIP's zero-shot top-1 accuracy by **3.6% on average** across natural distribution shifts
- Outperforms existing prompt tuning methods like CoOp and CoCoOp that require additional training data
- Maintains competitive performance on cross-dataset generalization tasks

**Distribution Shift Robustness**
- Shows consistent gains across different types of distribution shifts
- Particularly effective on challenging datasets like ImageNet-A and ImageNet-Sketch
- Demonstrates improved handling of domain gaps compared to static prompts

**Zero-Shot Preservation**
- Achieves these improvements while preserving the zero-shot setting
- Does not require any task-specific training data
- More practical for real-world deployment scenarios where training data may not be available

**Method Comparison**
- Outperforms manual prompt engineering approaches
- Shows superior performance compared to training-time prompt optimization methods
- Demonstrates the effectiveness of test-time adaptation over static approaches

**Key Performance Highlights**
- Significant accuracy improvements on challenging distribution shifts
- Robust performance across diverse visual domains
- Consistent gains without sacrificing generalization capability

### 4.3 분석
결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향
이 연구의 학술적 기여와 실용적 가치를 평가합니다.

## 6. 개인적 평가

**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점  
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준

