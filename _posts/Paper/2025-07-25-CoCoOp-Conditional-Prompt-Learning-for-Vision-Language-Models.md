---
published: true
title: "Conditional Prompt Learning for Vision-Language Models (CoCoOp)"
excerpt: "입력별 조건부 프롬프트 학습을 통한 비전-언어 모델의 일반화 성능 향상"

categories:
  - VLM
tags:
  - [VLM, Vision-Language, Prompt Learning, Conditional Learning, Generalization, CLIP]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

**Context Optimization (CoOp)**은 수동 프롬프트 엔지니어링의 한계를 해결했지만, **학습된 컨텍스트 벡터가 베이스 클래스에 과적합되어 새로운 클래스에 대한 일반화 성능이 저하**되는 문제가 있었습니다. 

**CoCoOp(Conditional Prompt Learning)**은 이러한 CoOp의 한계를 해결하기 위해 **입력 이미지에 조건부인 동적 프롬프트**를 생성하는 혁신적 접근법을 제안합니다. 각 이미지에 대해 **개별 인스턴스에 적응하는 조건부 토큰**을 생성하여 정적 프롬프트의 과적합 문제를 해결하고, 미지의 클래스에 대한 일반화 성능을 크게 향상시킵니다.

## Methods

### Architecture Overview

CoCoOp의 아키텍처는 CoOp을 확장하여 조건부 프롬프트를 생성합니다:

```
[Image] → [CLIP Image Encoder] → [Image Features]
    ↓                                    ↓
[Meta-Net] → [Conditional Token]         ↓
    ↓                                    ↓
[V1][V2]...[VM][Conditional][CLASS] → [CLIP Text Encoder] → [Text Features]
    ↑ Learnable + Conditional                     ↓
                                          [Classification]
```

Meta-Net이 각 이미지에 대해 조건부 토큰을 생성하여 동적 프롬프트를 만듭니다.

CoCoOp은 다음과 같은 핵심 기술 혁신을 통해 CoOp을 확장합니다:

### 1. Meta-Net Architecture

**경량 신경망 구조**
- **Two-layer bottleneck 설계**: Linear-ReLU-Linear 구조
- **차원 축소**: 은닉층에서 입력 차원을 16배 축소
- **효율적 계산**: 최소한의 추가 파라미터로 조건부 토큰 생성
- **End-to-end 학습**: 전체 시스템과 함께 최적화

**메타 네트워크 역할**
```
Meta-Net: Image Features → Conditional Token
f_meta(φ(x)) = v_cond
```

여기서:
- `φ(x)`: 입력 이미지의 시각적 특징
- `f_meta`: 메타 네트워크 함수
- `v_cond`: 생성된 조건부 토큰

### 2. Input-Conditional Token Generation

**인스턴스별 적응 토큰**
- 각 입력 이미지마다 **고유한 조건부 토큰 벡터** 생성
- 이미지의 시각적 내용에 따라 **동적으로 프롬프트 조정**
- 정적 컨텍스트 벡터의 한계 극복
- **클래스 분포 변화에 대한 강건성** 확보

**조건부 토큰의 특성**
- 입력 이미지의 시각적 특징을 반영
- 학습 가능한 파라미터를 통해 최적화
- 텍스트 인코더의 임베딩 공간과 호환

### 3. Dynamic Prompt Construction

**조건부 프롬프트 구성**
```
Dynamic Prompt = [v_cond] [V]₁ [V]₂ ... [V]ₘ [CLASS]
```

구성 요소:
- `[v_cond]`: Meta-Net이 생성한 조건부 토큰
- `[V]ᵢ`: 학습 가능한 정적 컨텍스트 벡터
- `[CLASS]`: 목표 클래스명
- **입력별 개인화된 프롬프트** 실현

**정적 vs 동적 프롬프트**
- **CoOp**: 모든 입력에 동일한 정적 컨텍스트 사용
- **CoCoOp**: 입력별로 조건부 토큰을 통해 동적 조정
- **인스턴스 적응성**: 개별 이미지 특성에 맞춘 프롬프트

### 4. Instance-Adaptive Mechanism

**적응적 프롬프트 생성**
- **이미지별 특화**: 각 입력의 고유한 시각적 특성 반영
- **클래스 분포 불변성**: 베이스 클래스 분포에 덜 민감
- **동적 조정**: 입력 내용에 따른 자동 프롬프트 최적화
- **일반화 향상**: 새로운 클래스에 대한 적응성 증가

### 5. End-to-End Training

**통합 최적화**
- Meta-Net과 컨텍스트 벡터의 **공동 학습**
- 베이스 클래스에서 훈련하되 **일반화 능력 유지**
- 표준 교차 엔트로피 손실로 최적화
- CLIP의 사전 훈련된 파라미터 동결

**학습 목표**
```
L = -log P(y|x) = -log exp(sim(I, T_y)/τ) / Σ_c exp(sim(I, T_c)/τ)
```

여기서:
- `I`: 이미지 특징
- `T_y`: 동적으로 생성된 정답 클래스 프롬프트 특징
- `T_c`: 모든 클래스에 대한 프롬프트 특징

### 6. Prompt Template Integration

**CLIP 호환성**
- 기존 CLIP 프롬프트 템플릿 구조 유지
- 조건부 토큰의 seamless 통합
- 다양한 프롬프트 템플릿에 적용 가능
- **"a photo of a {class}"** 등 표준 템플릿 지원

## Dataset

CoCoOp의 실험은 **비전-언어 모델 평가에 널리 사용되는 11개 데이터셋**에서 수행되었습니다:

**평가 설정**
- **Base-to-Novel 일반화**: 베이스 클래스로 훈련 후 새로운 클래스에서 테스트
- **동일 데이터셋 내 분할**: 각 데이터셋을 베이스/새로운 클래스로 분할
- **Few-shot 학습 시나리오**: 제한된 라벨 데이터로 훈련
- **일반화 성능 중점**: 미지의 클래스에 대한 적응성 평가

**데이터셋 다양성**
- 다양한 시각적 도메인과 개념 포함
- 일반 객체부터 전문 도메인까지 광범위한 커버리지
- Real-world 적용 시나리오 반영
- 클래스 분포 변화에 대한 강건성 검증

## Results

CoCoOp은 CoOp 대비 **새로운 클래스에 대한 일반화 성능에서 현저한 향상**을 달성했습니다:

**새로운 클래스 정확도 향상**
- **CoOp**: 63.22% → **CoCoOp**: 71.69% (미지 클래스)
- **8.47%p의 절대 성능 향상** 달성
- 일반화 격차를 현저히 감소

**광범위한 성능 개선**
- **11개 데이터셋 중 5개**에서 **10% 이상 정확도 증가**
- 일관된 성능 향상 패턴 확인
- 도메인별 특화 없이 범용적 개선

**베이스 클래스 성능 유지**
- 베이스 클래스에서 **경쟁력 있는 성능 유지**
- 새로운 클래스 일반화와 베이스 성능의 **균형** 달성
- 과적합 문제 해결로 전체적 성능 향상

**전이 학습 성능**
- **단일 데이터셋을 넘어선 전이**에서 유망한 결과
- 도메인 간 일반화 능력 입증
- **정적 프롬프트 학습 대비 강한 도메인 일반화**

**과적합 문제 해결**
- CoOp의 베이스 클래스 **과적합 경향 성공적 완화**
- 학습-테스트 분포 차이에 대한 강건성 향상
- **효율적 few-shot 학습의 장점 유지**

**핵심 발견**
- **조건부 프롬프팅**이 학습된 프롬프트와 수동 프롬프트 간 일반화 격차를 현저히 감소
- Few-shot 프롬프트 학습의 **효율성 장점을 유지**하면서 일반화 성능 크게 향상
- 입력별 적응이 **클래스 분포 변화에 대한 강건성** 제공

## Key Takeaways

1. **Conditional Prompting**: 입력별 조건부 토큰이 정적 프롬프트의 과적합 문제 해결
2. **Instance Adaptation**: 개별 이미지 특성에 맞춘 동적 프롬프트가 일반화 성능 향상
3. **Meta-Network Efficiency**: 경량 메타 네트워크로 효율적인 조건부 토큰 생성
4. **Generalization Bridge**: 학습된 프롬프트와 수동 프롬프트 간 일반화 격차 해소