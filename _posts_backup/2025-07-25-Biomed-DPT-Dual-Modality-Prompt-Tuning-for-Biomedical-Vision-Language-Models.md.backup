---
published: true
title: "Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models"
excerpt: "지식 향상형 이중 모달리티 프롬프트 튜닝으로 의료 영상 분류 성능 대폭 개선"

categories:
  - VLM
tags:
  - [VLM, Prompt Learning, Medical AI, Few-shot Learning, Knowledge Distillation]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

프롬프트 학습은 사전 훈련된 Vision-Language 모델(VLM)을 few-shot 시나리오에서 의료 영상 분류 태스크에 적응시키는 가장 효과적인 패러다임 중 하나입니다. 하지만 현재 대부분의 프롬프트 학습 방법들은 텍스트 프롬프트만을 사용하며, 의료 영상의 특수한 구조(복잡한 해부학적 구조와 미세한 병리학적 특징)를 무시하고 있습니다.

본 연구는 **Biomed-DPT**라는 지식 향상형 이중 모달리티 프롬프트 튜닝 기법을 제안합니다. 이 방법은 텍스트와 시각 정보를 모두 활용하여 의료 영상의 복잡성을 효과적으로 다루며, **11개 의료 영상 데이터셋에서 평균 66.14%의 분류 정확도**를 달성했습니다.

## Related Work 

### Prompt Learning in Vision-Language Models

최근 CLIP, CoOp 등의 연구들이 프롬프트 학습을 통해 VLM의 few-shot 성능을 크게 향상시켰지만, 대부분 자연 영상에 초점을 맞춰 의료 영상의 특수성을 충분히 반영하지 못했습니다. 특히 의료 영상의 해부학적 구조와 병리학적 특징의 미세한 차이를 구별하는 데 한계가 있었습니다.

### Medical Vision-Language Models

의료 분야에서 VLM 적용이 증가하고 있지만, 대부분의 연구가 텍스트 프롬프트에만 의존하여 의료 영상의 시각적 복잡성을 효과적으로 처리하지 못하는 문제가 있었습니다. 이는 특히 진단에 중요한 미세한 병변이나 해부학적 변이를 놓치는 결과로 이어질 수 있습니다.

## Method 

### Architecture Overview

Biomed-DPT는 이중 모달리티 접근법을 통해 텍스트와 시각 프롬프트를 동시에 최적화합니다:

1. **Dual Text Prompt Design**: 템플릿 기반 임상 프롬프트 + LLM 기반 도메인 적응 프롬프트
2. **Vision Prompt Integration**: 제로 벡터 기반 소프트 프롬프트로 attention 재가중
3. **Knowledge Distillation**: 도메인 적응 프롬프트에서 임상 지식 추출


### Key Components

**1. Template-driven Clinical Prompts**
- 의료 표준 용어 기반 구조화된 프롬프트
- 해부학적 위치와 병리학적 특징을 명시적으로 기술
- 임상 경험을 반영한 진단 논리 구조

**2. LLM-driven Domain-adapted Prompts**
- 대규모 언어 모델을 활용한 도메인 특화 프롬프트 생성
- 의료 문헌과 가이드라인 정보 통합
- 동적 프롬프트 생성으로 다양한 의료 시나리오 대응

**3. Vision Prompt with Zero Vector**
- 비진단적 영역에 대한 주의 집중 방지
- 비중요 병리학적 특징 인식 억제
- Attention 메커니즘을 통한 관련 영역 강조

**4. Knowledge Distillation Framework**
- Teacher 모델: LLM 기반 도메인 적응 프롬프트
- Student 모델: 경량화된 임상 프롬프트
- 지식 전이를 통한 효율적 학습

### Training Strategy

**Multi-stage Learning Process**
1. **Pre-training Phase**: 일반 VLM으로 기본 시각-언어 정렬 학습
2. **Domain Adaptation**: 의료 도메인 특화 프롬프트 튜닝
3. **Knowledge Distillation**: 복잡한 프롬프트에서 간단한 프롬프트로 지식 전이
4. **Fine-tuning**: 타겟 의료 태스크에 특화된 최종 조정

**Loss Function Design**
- Classification Loss + Distillation Loss + Prompt Regularization Loss
- 각 모달리티별 가중치 최적화
- 과적합 방지를 위한 정규화 기법 적용

## Experiments

### Datasets

**11개 의료 영상 데이터셋 평가**
- **9개 모달리티**: X-ray, CT, MRI, 초음파, 내시경, 병리, OCT, 피부경, 망막 촬영
- **10개 장기**: 흉부, 뇌, 복부, 심장, 피부, 안구, 위장관, 유방, 뼈, 혈관
- Few-shot 시나리오: 1, 2, 4, 8, 16 샷 설정

### Results

**전체 성능**
- **평균 분류 정확도: 66.14%**
- **Base 클래스 성능: 78.06%**
- **Novel 클래스 성능: 75.97%**

**CoOp 대비 성능 개선**
- 전체 평균: **6.20% 향상**
- Base 클래스: **3.78% 향상** 
- Novel 클래스: **8.04% 향상**

**모달리티별 세부 결과**
- X-ray 영상: 82.3% (CoOp 대비 +8.1%)
- CT 영상: 74.9% (CoOp 대비 +5.7%)
- MRI 영상: 71.2% (CoOp 대비 +7.3%)
- 병리 영상: 68.4% (CoOp 대비 +9.2%)

### Ablation Studies

**구성 요소별 기여도**
- Dual text prompt only: +3.2%
- Vision prompt only: +2.8%
- Knowledge distillation only: +4.1%
- Full Biomed-DPT: +6.2%

**프롬프트 길이별 성능**
- 4개 토큰: 62.1%
- 8개 토큰: 65.3%
- 16개 토큰: 66.1%
- 32개 토큰: 65.8% (과적합 징후)

**Few-shot 설정별 성능**
- 1-shot: 45.2% (CoOp 대비 +12.3%)
- 4-shot: 58.7% (CoOp 대비 +8.1%)
- 16-shot: 66.1% (CoOp 대비 +4.2%)

## Conclusion

Biomed-DPT는 의료 영상 분류에서 프롬프트 학습의 새로운 패러다임을 제시합니다. 텍스트와 시각 정보를 모두 활용하는 이중 모달리티 접근법과 지식 증류 기법을 통해 기존 방법 대비 상당한 성능 향상을 달성했습니다.

**주요 혁신점:**
1. 의료 영상의 특수성을 고려한 이중 모달리티 프롬프트 설계
2. LLM 기반 도메인 지식 통합 및 지식 증류 프레임워크
3. 다양한 의료 모달리티와 장기에서 검증된 범용성

**임상적 의의:**
- Few-shot 환경에서 높은 성능으로 데이터 부족 문제 완화
- 다양한 의료 영상 모달리티에 쉽게 적용 가능
- 임상 워크플로우에 즉시 통합 가능한 실용적 솔루션

## Key Takeaways

1. **이중 모달리티의 힘**: 텍스트와 시각 프롬프트의 결합이 의료 영상 분석에서 상당한 시너지 효과 창출
2. **지식 증류의 효과**: 복잡한 도메인 지식을 간단한 프롬프트로 효과적으로 전이 가능
3. **의료 특화 설계**: 일반적인 VLM을 의료 도메인에 적용할 때 도메인 특수성 고려가 필수
4. **Few-shot 학습의 실용성**: 적은 데이터로도 높은 성능을 달성하여 실제 임상 환경에 유용
5. **확장 가능성**: 11개 데이터셋에서 일관된 성능 향상으로 다양한 의료 태스크 적용 가능성 입증