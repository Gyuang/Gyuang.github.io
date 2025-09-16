---
categories:
- Paper
- VLM
date: '2025-09-16'
excerpt: 'Prefix-Tuning: 생성 태스크를 위한 Continuous Prompt 최적화에 대한 체계적 분석'
header:
  teaser: /assets/images/paper/prefix-tuning-teaser.png
last_modified_at: '2025-09-16'
published: true
tags:
- Prefix Tuning
- Parameter Efficient
- Generation
- NLP
title: 'Prefix-Tuning: Optimizing Continuous Prompts for Generation'
toc: true
toc_sticky: true
---

# Prefix-Tuning: Optimizing Continuous Prompts for Generation

## 논문 정보
- **저자**: **: Xiang Lisa Li, Percy Liang
- **발표**: **: ACL 2021
- **ArXiv**: N/A

## 1. 핵심 요약 (2-3문장)
이 논문의 핵심 기여와 주요 발견을 간결하게 요약합니다.

## 2. 배경 및 동기
기존 방법의 한계점과 연구의 필요성을 설명합니다.

## 3. 제안 방법

### 3.1 아키텍처 개요

![Architecture Overview 0](/assets/images/paper/prefix-tuning/architecture_overview_0.png)
*Figure: Architecture Overview 0*



### 3.2 핵심 기술/알고리즘
- **저자**: Xiang Lisa Li, Percy Liang
- **발표**: ACL 2021
- **ArXiv**: [2101.00190](https://arxiv.org/abs/2101.00190)



Prefix-Tuning은 언어 모델의 파라미터를 고정한 채로, **시퀀스 시작 부분에 학습 가능한 continuous vector(prefix)**를 추가하여 다양한 생성 태스크에 적응하는 방법론입니다.




- **위치**: 입력 시퀀스 맨 앞에 prefix 토큰들을 배치
- **형태**: `[PREFIX] [INPUT] → [OUTPUT]`
- **학습**: Prefix 벡터만 학습, 나머지 모델 파라미터는 고정



![Architecture Overview 0](/assets/images/paper/prefix-tuning/architecture_overview_0.png)
*Figure: Model architecture and component design*
*Figure: Architecture Overview 0*



```
P_idx : [PREFIX_1, PREFIX_2, ..., PREFIX_n, x_1, x_2, ..., x_m, y_1, y_2, ..., y_k]
```


```
P_idx : [PREFIX_1, PREFIX_2, ..., PREFIX_n, x_1, x_2, ..., x_m]
```


- **Key-Value 벡터**: 각 어텐션 레이어의 Key, Value 벡터에 prefix 적용
- **모든 레이어**: 입력 임베딩뿐만 아니라 모든 transformer 레이어에 prefix 정보 전파




- **어텐션 메커니즘**: 모든 후속 토큰이 prefix에 attend 가능
- **생성 과정 제어**: 생성 프로세스 전체를 prefix가 가이드
- **컨텍스트 제공**: 태스크별 컨텍스트를 명확히 제공


- **중간 삽입**: 컨텍스트가 분리되어 성능 저하
- **끝 부분**: 생성 시작 전에 정보 제공 불가
- **분산 배치**: 일관성 있는 가이던스 어려움


- **Full Fine-tuning**: BLEU 45.6 (WebNLG)
- **Prefix-Tuning**: BLEU 44.0 (WebNLG) - 1.6점 차이로 근접
- **파라미터**: 전체의 0.1%만 사용


- **Full Fine-tuning**: ROUGE-L 22.8
- **Prefix-Tuning**: ROUGE-L 21.9 - 0.9점 차이
- **효율성**: GPU 메모리 사용량 대폭 감소




- **하위 레이어**: 기본적인 언어 패턴 학습
- **중간 레이어**: 태스크별 구조 정보 학습
- **상위 레이어**: 구체적인 생성 전략 학습


- **Prefix → Input**: 태스크 이해를 위한 어텐션
- **Input → Prefix**: 컨텍스트 인코딩을 위한 어텐션
- **Generation → Prefix**: 생성 가이던스를 위한 어텐션




```python

prefix_embeddings = torch.randn(prefix_length, hidden_size)


prefix_embeddings = model.embed_tokens(prefix_token_ids)
```


1. **Prefix 파라미터만 업데이트**
2. **언어 모델은 고정**
3. **태스크별 손실 함수 사용**




- **파라미터 효율성**: 0.1% 파라미터만 사용
- **빠른 적응**: 새로운 태스크에 빠른 전환
- **메모리 효율성**: 하나의 백본 모델로 여러 태스크 처리
- **안정성**: Fine-tuning보다 안정적인 학습


- **복잡한 태스크**: 매우 복잡한 추론 태스크에서는 성능 제한
- **초기 설정**: Prefix 길이 설정이 성능에 중요한 영향
- **생성 태스크 특화**: 분류 태스크에서는 제한적 효과


1. **위치의 일관성**: 시작 부분 배치가 전체 생성 과정에 일관된 영향
2. **어텐션 효율성**: 모든 생성 토큰이 prefix에 쉽게 접근 가능
3. **레이어별 전파**: 각 레이어에서 prefix 정보가 적절히 활용됨

이 연구는 이후 P-Tuning v2, 그리고 다양한 prompt positioning 연구의 기반이 되었으며, **소프트 프롬프트의 위치가 성능에 미치는 근본적 영향**을 이해하는 출발점이 되었습니다.

### 3.3 구현 세부사항


## 4. 실험 및 결과

### 4.1 실험 설정
실험에 사용된 데이터셋, 평가 지표, 비교 대상을 설명합니다.

### 4.2 주요 결과

![Results Table 3 0](/assets/images/paper/prefix-tuning/results_table_3_0.png)
*Figure: Results Table 3 0*



![Results Table 3 0](/assets/images/paper/prefix-tuning/results_table_3_0.png)
*Figure: Experimental results and performance metrics*
*Figure: Results Table 3 0*



- **작은 모델**: Fine-tuning 대비 성능 차이 존재
- **큰 모델**: 성능 격차 현저히 감소
- **GPT-3**: Fine-tuning과 거의 동등한 성능

### 4.3 분석
결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향
- **짧은 prefix (< 10)**: 제한적 성능
- **적절한 길이 (10-20)**: 최적 성능
- **긴 prefix (> 30)**: 오히려 성능 저하 (과적합)



Prefix-Tuning은 **시퀀스 시작 부분에 프롬프트를 배치하는 것의 중요성**을 입증한 초기 연구 중 하나입니다.

## 6. 개인적 평가

**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점  
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준

