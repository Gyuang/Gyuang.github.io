---
categories:
- Paper
- VLM
header:
  teaser: /assets/images/paper/prefix-tuning-teaser.png
tags:
- Prefix Tuning
- Parameter Efficient
- Generation
- NLP
title: 'Prefix-Tuning: Optimizing Continuous Prompts for Generation'
toc: true
toc_label: Prefix-Tuning
toc_sticky: true
---

# Prefix-Tuning: 생성 태스크를 위한 Continuous Prompt 최적화

## 논문 정보
- **저자**: Xiang Lisa Li, Percy Liang
- **발표**: ACL 2021
- **ArXiv**: [2101.00190](https://arxiv.org/abs/2101.00190)

## 핵심 아이디어

Prefix-Tuning은 언어 모델의 파라미터를 고정한 채로, **시퀀스 시작 부분에 학습 가능한 continuous vector(prefix)**를 추가하여 다양한 생성 태스크에 적응하는 방법론입니다.

## 방법론

### 1. Prefix 설계
- **위치**: 입력 시퀀스 맨 앞에 prefix 토큰들을 배치
- **형태**: `[PREFIX] [INPUT] → [OUTPUT]`
- **학습**: Prefix 벡터만 학습, 나머지 모델 파라미터는 고정

### 2. 아키텍처별 적용

![Architecture Overview 0](/assets/images/paper/prefix-tuning/architecture_overview_0.png)
*Figure: Architecture Overview 0*


#### GPT-2/GPT-3 (Autoregressive)
```
P_idx : [PREFIX_1, PREFIX_2, ..., PREFIX_n, x_1, x_2, ..., x_m, y_1, y_2, ..., y_k]
```

#### BERT (Bidirectional)  
```
P_idx : [PREFIX_1, PREFIX_2, ..., PREFIX_n, x_1, x_2, ..., x_m]
```

### 3. 레이어별 Prefix 적용
- **Key-Value 벡터**: 각 어텐션 레이어의 Key, Value 벡터에 prefix 적용
- **모든 레이어**: 입력 임베딩뿐만 아니라 모든 transformer 레이어에 prefix 정보 전파

## 프롬프트 위치의 중요성

### 1. 시작 위치 배치의 이유
- **어텐션 메커니즘**: 모든 후속 토큰이 prefix에 attend 가능
- **생성 과정 제어**: 생성 프로세스 전체를 prefix가 가이드
- **컨텍스트 제공**: 태스크별 컨텍스트를 명확히 제공

### 2. 다른 위치와의 비교
- **중간 삽입**: 컨텍스트가 분리되어 성능 저하
- **끝 부분**: 생성 시작 전에 정보 제공 불가
- **분산 배치**: 일관성 있는 가이던스 어려움

## 실험 결과

![Results Table 3 0](/assets/images/paper/prefix-tuning/results_table_3_0.png)
*Figure: Results Table 3 0*


### Table-to-Text 생성 (WebNLG, DART)
- **Full Fine-tuning**: BLEU 45.6 (WebNLG)
- **Prefix-Tuning**: BLEU 44.0 (WebNLG) - 1.6점 차이로 근접
- **파라미터**: 전체의 0.1%만 사용

### 요약 태스크 (XSum)
- **Full Fine-tuning**: ROUGE-L 22.8
- **Prefix-Tuning**: ROUGE-L 21.9 - 0.9점 차이
- **효율성**: GPU 메모리 사용량 대폭 감소

### 모델 크기별 성능
- **작은 모델**: Fine-tuning 대비 성능 차이 존재
- **큰 모델**: 성능 격차 현저히 감소
- **GPT-3**: Fine-tuning과 거의 동등한 성능

## 위치별 세부 분석

### 1. Prefix 길이의 영향
- **짧은 prefix (< 10)**: 제한적 성능
- **적절한 길이 (10-20)**: 최적 성능
- **긴 prefix (> 30)**: 오히려 성능 저하 (과적합)

### 2. 레이어별 기여도
- **하위 레이어**: 기본적인 언어 패턴 학습
- **중간 레이어**: 태스크별 구조 정보 학습
- **상위 레이어**: 구체적인 생성 전략 학습

### 3. 어텐션 패턴 분석
- **Prefix → Input**: 태스크 이해를 위한 어텐션
- **Input → Prefix**: 컨텍스트 인코딩을 위한 어텐션
- **Generation → Prefix**: 생성 가이던스를 위한 어텐션

## 기술적 세부사항

### 초기화 전략
```python
# 랜덤 초기화
prefix_embeddings = torch.randn(prefix_length, hidden_size)

# 실제 토큰 기반 초기화 (더 안정적)
prefix_embeddings = model.embed_tokens(prefix_token_ids)
```

### 학습 과정
1. **Prefix 파라미터만 업데이트**
2. **언어 모델은 고정**
3. **태스크별 손실 함수 사용**

## 장점과 한계

### 장점
- **파라미터 효율성**: 0.1% 파라미터만 사용
- **빠른 적응**: 새로운 태스크에 빠른 전환
- **메모리 효율성**: 하나의 백본 모델로 여러 태스크 처리
- **안정성**: Fine-tuning보다 안정적인 학습

### 한계
- **복잡한 태스크**: 매우 복잡한 추론 태스크에서는 성능 제한
- **초기 설정**: Prefix 길이 설정이 성능에 중요한 영향
- **생성 태스크 특화**: 분류 태스크에서는 제한적 효과

## 소프트 프롬프트 위치 연구에 미친 영향

Prefix-Tuning은 **시퀀스 시작 부분에 프롬프트를 배치하는 것의 중요성**을 입증한 초기 연구 중 하나입니다. 

### 주요 통찰
1. **위치의 일관성**: 시작 부분 배치가 전체 생성 과정에 일관된 영향
2. **어텐션 효율성**: 모든 생성 토큰이 prefix에 쉽게 접근 가능
3. **레이어별 전파**: 각 레이어에서 prefix 정보가 적절히 활용됨

이 연구는 이후 P-Tuning v2, 그리고 다양한 prompt positioning 연구의 기반이 되었으며, **소프트 프롬프트의 위치가 성능에 미치는 근본적 영향**을 이해하는 출발점이 되었습니다.