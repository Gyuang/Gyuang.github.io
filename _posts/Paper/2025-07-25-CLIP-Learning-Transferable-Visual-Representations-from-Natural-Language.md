---
published: true
title: "CLIP: Learning Transferable Visual Representations from Natural Language"
excerpt: "자연어 감독을 통한 전이 가능한 시각 표현 학습의 혁신적 접근법"

categories:
  - VLM
tags:
  - [VLM, Vision-Language, Contrastive Learning, Zero-shot, Transfer Learning]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

기존 컴퓨터 비전 모델들은 **고정된 객체 카테고리에만 제한**되어 있었고, 새로운 태스크나 도메인에 적용하려면 막대한 라벨링 비용이 필요했습니다. **CLIP(Contrastive Language-Image Pre-training)**은 이러한 한계를 해결하기 위해 **자연어 설명으로부터 시각적 개념을 학습**하는 혁신적 접근법을 제안합니다.

CLIP의 핵심 아이디어는 **인터넷에서 수집한 4억 개의 이미지-텍스트 쌍**을 사용해 대조 학습을 통해 시각-언어 표현을 학습하는 것입니다.


## Methods

### Core Architecture

CLIP는 **dual-encoder 구조**를 기반으로 합니다:

### Architecture Overview

CLIP는 **dual-encoder 구조**를 기반으로 하며, 다음과 같은 핵심 구성 요소로 이루어져 있습니다:

```
[Images] → [Image Encoder] → [Image Features]
                                    ↓
                            [Contrastive Learning]
                                    ↑
[Text] → [Text Encoder] → [Text Features]
```

**Image Encoder**
- **ResNet-50/101** 또는 **Vision Transformer (ViT)** 사용
- 입력 이미지를 d차원 특징 벡터로 인코딩
- Global average pooling 후 linear projection layer 적용
- Layer normalization으로 특징 벡터 정규화

**Text Encoder**  
- **Transformer 기반 언어 모델** (GPT-2 아키텍처 활용)
- 최대 76개 토큰의 텍스트 시퀀스 처리
- [CLS] 토큰의 출력을 텍스트 표현으로 사용
- 이미지 인코더와 동일한 차원으로 projection

### Contrastive Learning Framework

CLIP의 핵심은 **InfoNCE 기반 대조 학습**입니다:

**Training Objective**
```
L = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
```

여기서:
- I_i, T_i: i번째 이미지-텍스트 쌍의 임베딩
- sim(): 코사인 유사도 함수  
- τ: 학습 가능한 temperature 파라미터
- 배치 내 모든 negative 쌍에 대해 대조

**Symmetric Loss Design**
- 이미지→텍스트 방향과 텍스트→이미지 방향의 **대칭적 손실** 사용
- 양방향 검색 성능 향상과 표현 학습 안정성 확보
- 최종 손실은 두 방향 손실의 평균

### Data Scaling and Curation

**Large-scale Dataset Construction**
- **WebImageText (WIT)**: 인터넷에서 수집한 4억 개 이미지-텍스트 쌍
- 기존 supervised 데이터셋보다 **400배 큰 규모**
- 다양한 도메인과 개념을 포괄하는 자연 발생 데이터

**Data Filtering Strategy**
- 중복 제거를 위한 near-duplicate detection
- 최소 텍스트 길이 및 품질 필터링  
- 언어별 분포 균형 조정
- 개인정보 보호를 위한 필터링

### Zero-shot Transfer Learning

**Prompt Engineering for Classification**
- 클래스명을 "a photo of a {class}" 형태로 변환
- 다양한 prompt template을 ensemble하여 성능 향상
- 예시: "a photo of a {class}", "a picture of a {class}", etc.

**Text-as-Supervision Paradigm**
- 전통적인 one-hot 라벨 대신 **자연어 설명을 감독 신호**로 사용
- 클래스 간 의미적 관계를 자동으로 학습
- 새로운 클래스에 대한 즉시 일반화 가능

### Training Infrastructure

**Distributed Training Setup**
- 592개 V100 GPU에서 약 12일간 훈련
- 배치 크기 32,768로 대규모 대조 학습 수행
- Mixed precision training으로 메모리 효율성 향상

**Optimization Strategy**
- AdamW optimizer (β1=0.9, β2=0.98)
- Cosine annealing learning rate schedule
- Weight decay 0.2, gradient clipping 적용

## Datasets

**Training Data**
- **WebImageText (WIT)**: 4억 개 이미지-텍스트 쌍
- 인터넷에서 자연 발생한 다양한 도메인 데이터

**Evaluation Benchmarks**
- **ImageNet**: 1,000개 클래스 이미지 분류
- **CIFAR-10/100**: 작은 규모 이미지 분류  
- **STL-10**: Self-supervised learning 벤치마크
- **30+ 추가 데이터셋**에서 zero-shot 성능 평가

## Results

**Zero-shot Performance**
- ImageNet에서 **ResNet-50 supervised 모델과 유사한 성능** 달성
- 별도 훈련 없이 30개 이상 데이터셋에서 경쟁력 있는 결과
- 특히 **out-of-distribution 데이터**에서 뛰어난 견고성

**Few-shot Learning**
- 적은 수의 예시만으로도 빠른 적응 가능
- Linear probing에서 기존 self-supervised 방법들 대비 우수한 성능

**Robustness and Generalization**
- Distribution shift에 대한 강한 견고성
- 다양한 도메인 간 transfer 성능 우수

## Key Takeaways

1. **Scale Matters**: 대규모 다양한 데이터가 일반화 성능의 핵심
2. **Natural Language Supervision**: 자연어가 제공하는 풍부한 감독 신호의 위력
3. **Zero-shot Paradigm**: 사전 훈련된 표현의 즉시 전이 가능성
4. **Contrastive Learning**: 대조 학습을 통한 효과적인 멀티모달 표현 학습