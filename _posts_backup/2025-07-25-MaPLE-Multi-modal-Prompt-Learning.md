---
categories:
- VLM
date: 2025-07-25
excerpt: "\uC2DC\uAC01-\uC5B8\uC5B4 \uD504\uB86C\uD504\uD2B8\uC758 \uB3D9\uC2DC \uD559\
  \uC2B5\uC744 \uD1B5\uD55C \uB2E4\uC911 \uBAA8\uB2EC \uC801\uC751 \uD504\uB808\uC784\
  \uC6CC\uD06C"
last_modified_at: 2025-07-25
published: true
tags:
- VLM
- Prompt Learning
- CLIP
- Multi-modal
- Few-shot Learning
title: 'MaPLe: Multi-modal Prompt Learning'
toc: true
toc_sticky: true
---

## Introduction

![Figure 1 0](/assets/images/paper/maple-multi-modal-prompt-learning/figure_1_0.png)
*Figure: Figure 1 0*

기존의 프롬프트 학습 방법들은 **CLIP의 시각 또는 언어 브랜치 중 하나만을 적응**시키는 한계가 있었습니다. 이로 인해 두 모달리티 간의 정렬이 충분하지 않아 새로운 클래스, 데이터셋, 도메인 변화에 대한 일반화 성능이 제한적이었습니다.

**MaPLe(Multi-modal Prompt Learning)**은 이러한 문제를 해결하기 위해 **CLIP의 시각과 언어 브랜치를 동시에 적응**시키는 혁신적인 다중 모달 프롬프트 학습 프레임워크를 제안합니다. 특히 시각-언어 프롬프트 간의 **강한 결합을 촉진**하여 두 표현 공간 간의 정렬을 개선합니다.

<p align="center">
  <img src="https://arxiv.org/abs/2210.03117" alt="MaPLe Paper" style="width: 100%;">
</p>

## Methods

![Architecture Diagram 3 0](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_0.png)
*Figure: Architecture Diagram 3 0*

### Architecture Overview

![Method Diagram 2](/assets/images/paper/maple-multi-modal-prompt-learning/method_diagram_2.png)
*Figure: Method Diagram 2*

MaPLE은 비전과 언어 모든 브랜치에서 프롬프트를 학습합니다:

```
[Image] → [Vision Prompts + ViT] → [Image Features]
                ↓ Coupling Function      ↓
[Text] → [Language Prompts + Transformer] → [Text Features]
                                            ↓
                                    [Classification]
```

비전과 언어 프롬프트가 결합 함수를 통해 상호 작용하며 시너지를 만듭니다.

### Multi-modal Prompt Architecture

![Architecture Overview 1](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_overview_1.png)
*Figure: Architecture Overview 1*

MaPLe의 핵심은 **시각과 언어 브랜치에 대한 별도의 컨텍스트 프롬프트 학습**입니다:

**Frozen CLIP Backbone**
- 사전 훈련된 CLIP 모델은 **완전히 고정**하여 유지
- **학습 가능한 프롬프트 토큰만을 훈련**하여 효율성 확보
- 원본 CLIP의 강력한 표현 능력 보존

**Branch-aware Hierarchical Prompts**
- 시각과 언어 표현 공간을 **동적으로 적응**
- 각 브랜치별 특성에 맞는 프롬프트 설계
- 계층적 구조를 통한 점진적 특징 모델링

### Vision-Language Coupling Function

**Cross-modal Dependency Mechanism**
- **언어 프롬프트에 기반하여 시각 프롬프트를 조건화**
- 결합 메커니즘을 통한 두 모달리티 간 상호 시너지 유도
- 독립적인 단일 모달 솔루션 학습 방지

**Mutual Synergy Design**
- 시각 프롬프트와 언어 프롬프트 간 상호 의존성 강화
- 양방향 정보 공유를 통한 정렬 품질 향상
- 모달리티 간 의미적 일관성 확보

### Deep Contextual Prompting

**Multi-layer Prompt Integration**
- **다중 트랜스포머 블록의 초기 단계**에서 별도 컨텍스트 프롬프트 학습
- 단계별 특징 관계의 점진적 모델링 가능
- 계층적 프롬프트 배치를 통한 풍부한 컨텍스트 학습

**Progressive Feature Modeling**
- 각 층에서 서로 다른 수준의 추상화 학습
- 초기 층: 저수준 특징, 후반 층: 고수준 의미
- 전체적인 표현 품질 향상

### Training Procedure

**Parameter-efficient Learning**
- **CLIP 백본은 고정**, 프롬프트 파라미터만 최적화
- 표준 교차 엔트로피 손실을 사용한 훈련
- 결합 함수를 통한 시각-언어 정렬 개선 목표

**Coupling-aware Optimization**
- 시각-언어 프롬프트 간 상호작용 최적화
- 독립적 학습 방지를 위한 정규화 메커니즘
- 안정적인 수렴을 위한 학습률 스케줄링

### Prompt Integration

**Multi-layer Insertion Strategy**
- 시각과 텍스트 인코더의 **다중 층에 학습 가능한 프롬프트 토큰 삽입**
- 결합 함수를 통한 시각-언어 프롬프트 간 정보 공유
- 원본 CLIP 아키텍처 일관성 유지하며 프롬프트 유연성 추가

**Architecture Preservation**
- CLIP의 기본 구조 변경 없이 프롬프트 통합
- 기존 사전 훈련된 가중치의 효과적 활용
- 최소한의 추가 파라미터로 최대 성능 향상

## Dataset

평가는 **11개의 다양한 이미지 인식 데이터셋**에서 수행되었으며, 세 가지 주요 일반화 시나리오에 중점을 두었습니다:

**Novel Class Recognition**
- 훈련 중 보지 못한 새로운 클래스에 대한 분류 성능
- Few-shot 학습 환경에서의 일반화 능력 평가

**Cross-dataset Transfer**
- 서로 다른 데이터셋 간 전이 학습 성능
- 도메인 불변 표현 학습 효과 검증

**Domain Shift Adaptation**
- 도메인 변화에 대한 견고성 평가
- 분포 변화 상황에서의 성능 유지 능력

## Results

![Results Table 7 0](/assets/images/paper/maple-multi-modal-prompt-learning/results_table_7_0.png)
*Figure: Results Table 7 0*

**State-of-the-art Performance**
- 기존 최고 성능 방법인 **Co-CoOp 대비 3.45% 절대 성능 향상** (novel class 일반화)
- **전체 조화 평균 성능에서 2.72% 개선**
- 11개 다양한 이미지 인식 데이터셋에서 일관된 성능 향상

**Base-Novel Class Balance**
- 기존 클래스 성능 유지하면서 새로운 카테고리 일반화 향상
- 프롬프트 학습 접근법의 **일반적인 base-novel 클래스 성능 트레이드오프 문제 해결**
- 균형잡힌 성능으로 실용적 적용 가능성 증대

**Consistent Improvements**
- 다양한 평가 시나리오에서 **지속적인 성능 개선** 확인
- 특히 도메인 변화와 cross-dataset 전이에서 뛰어난 견고성
- 다중 모달 프롬프트 학습의 효과성 입증

## Key Takeaways

1. **Multi-modal Coupling**: 시각-언어 프롬프트의 동시 학습이 단일 모달리티 적응보다 우수
2. **Strong Alignment**: 결합 함수를 통한 모달리티 간 강한 정렬이 일반화 성능 향상의 핵심
3. **Parameter Efficiency**: CLIP 백본 고정으로 효율적이면서도 효과적인 적응 달성
4. **Balanced Performance**: Base-novel 클래스 간 성능 트레이드오프 문제 해결

## Additional Figures

![Figure 1 2](/assets/images/paper/maple-multi-modal-prompt-learning/figure_1_2.png)
*Figure: Figure 1 2*

![Figure 1 3](/assets/images/paper/maple-multi-modal-prompt-learning/figure_1_3.png)
*Figure: Figure 1 3*

![Figure 1 5](/assets/images/paper/maple-multi-modal-prompt-learning/figure_1_5.png)
*Figure: Figure 1 5*

![Figure 2 0](/assets/images/paper/maple-multi-modal-prompt-learning/figure_2_0.png)
*Figure: Figure 2 0*

![Figure 2 1](/assets/images/paper/maple-multi-modal-prompt-learning/figure_2_1.png)
*Figure: Figure 2 1*

![Figure 2 3](/assets/images/paper/maple-multi-modal-prompt-learning/figure_2_3.png)
*Figure: Figure 2 3*

![Architecture Diagram 3 1](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_1.png)
*Figure: Architecture Diagram 3 1*

![Architecture Diagram 3 2](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_2.png)
*Figure: Architecture Diagram 3 2*

![Architecture Diagram 3 3](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_3.png)
*Figure: Architecture Diagram 3 3*

![Architecture Diagram 3 4](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_4.png)
*Figure: Architecture Diagram 3 4*

![Architecture Diagram 3 5](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_5.png)
*Figure: Architecture Diagram 3 5*

![Architecture Diagram 3 6](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_6.png)
*Figure: Architecture Diagram 3 6*

![Architecture Diagram 3 7](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_7.png)
*Figure: Architecture Diagram 3 7*

![Architecture Diagram 3 8](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_8.png)
*Figure: Architecture Diagram 3 8*

![Architecture Diagram 3 9](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_9.png)
*Figure: Architecture Diagram 3 9*

![Architecture Diagram 3 10](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_10.png)
*Figure: Architecture Diagram 3 10*

![Architecture Diagram 3 11](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_11.png)
*Figure: Architecture Diagram 3 11*

![Architecture Diagram 3 12](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_12.png)
*Figure: Architecture Diagram 3 12*

![Architecture Diagram 3 13](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_13.png)
*Figure: Architecture Diagram 3 13*

![Architecture Diagram 3 14](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_14.png)
*Figure: Architecture Diagram 3 14*

![Architecture Diagram 3 15](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_15.png)
*Figure: Architecture Diagram 3 15*

![Architecture Diagram 3 16](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_16.png)
*Figure: Architecture Diagram 3 16*

![Architecture Diagram 3 17](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_17.png)
*Figure: Architecture Diagram 3 17*

![Architecture Diagram 3 18](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_18.png)
*Figure: Architecture Diagram 3 18*

![Architecture Diagram 3 19](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_19.png)
*Figure: Architecture Diagram 3 19*

![Architecture Diagram 3 20](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_20.png)
*Figure: Architecture Diagram 3 20*

![Architecture Diagram 3 21](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_21.png)
*Figure: Architecture Diagram 3 21*

![Architecture Diagram 3 22](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_22.png)
*Figure: Architecture Diagram 3 22*

![Architecture Diagram 3 23](/assets/images/paper/maple-multi-modal-prompt-learning/architecture_diagram_3_23.png)
*Figure: Architecture Diagram 3 23*

![Results Table 7 1](/assets/images/paper/maple-multi-modal-prompt-learning/results_table_7_1.png)
*Figure: Results Table 7 1*