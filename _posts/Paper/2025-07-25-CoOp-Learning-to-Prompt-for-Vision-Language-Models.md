---
categories:
- VLM
date: 2025-07-25
excerpt: Learning to Prompt for Vision-Language Models (CoOp)에 대한 체계적 분석과 핵심 기여 요약
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Vision-Language
- Prompt Learning
- Few-shot Learning
- CLIP
title: Learning to Prompt for Vision-Language Models (CoOp)
toc: true
toc_sticky: true
---

# Learning to Prompt for Vision-Language Models (CoOp)

## 논문 정보
- **저자**: Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu
- **발표**: IJCV 2022
- **ArXiv**: [2109.01134](https://arxiv.org/abs/2109.01134)

## 1. 핵심 요약 (2-3문장)
CoOp은 수동적인 프롬프트 엔지니어링의 한계를 극복하기 위해 학습 가능한 연속 벡터로 프롬프트 컨텍스트를 자동 최적화하는 방법을 제안합니다.

## 2. 배경 및 동기
![Figure 3 0](/assets/images/paper/coop-learning-to-prompt-for-vision-language-models/figure_3_0.png)
![Figure 3 0](/assets/images/paper/coop-learning-to-prompt-for-vision-language-models/figure_3_0.png)
*Figure: Figure 3 0*
기존 비전-언어 모델들은 **수동적인 프롬프트 엔지니어링**에 의존해왔습니다. 특히 CLIP과 같은 모델에서는 올바른 프롬프트 설계가 성능에 결정적 영향을 미치지만, 이는 **도메인 전문 지식이 필요하고 극도로 시간 소모적**이며, 단어 하나의 변화만으로도 성능이 크게 달라지는 문제가 있었습니다.
**CoOp(Context Optimization)**은 이러한 한계를 해결하기 위해 **학습 가능한 연속 벡터로 프롬프트 컨텍스트를 모델링**하는 자동화된 접근법을 제안합니다. 사전 훈련된 모델 파라미터는 고정한 채, 프롬프트 부분만을 학습하여 downstream 이미지 인식 작업에 효율적으로 적응할 수 있습니다.

## 3. 제안 방법

### 3.1 아키텍처 개요
시스템의 전체 아키텍처와 주요 구성 요소들을 설명합니다.

### 3.2 핵심 기술/알고리즘
핵심 기술적 혁신과 알고리즘에 대해 설명합니다.

### 3.3 구현 세부사항
구현과 관련된 중요한 기술적 세부사항들을 다룹니다.

## 4. 실험 및 결과

### 4.1 실험 설정
![Architecture Overview 1](/assets/images/paper/coop-learning-to-prompt-for-vision-language-models/architecture_overview_1.png)
*Figure: Model architecture and component design*
*Figure: Architecture Overview 1*

### 4.2 주요 결과
**Peak Performance**
- 특정 데이터셋에서 **45% 이상의 최고 성능 향상** 달성
- 도메인에 따라 극적인 성능 개선 효과
- 시간 소모적인 수동 프롬프트 튜닝 프로세스 **완전 제거**
- 다양한 이미지 인식 작업과 도메인에서 **일관된 성능 향상**

### 4.3 분석
실험 결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향


## 6. 개인적 평가

**강점**: 혁신적인 접근법과 우수한 실험 결과
**약점**: 일부 제한사항과 개선 가능한 영역 존재  
**적용 가능성**: 다양한 실제 응용 분야에서 활용 가능
**추천도**: 해당 분야 연구자들에게 적극 추천