---
categories:
- VLM
date: 2025-07-25
excerpt: 'MaPLe: Multi-modal Prompt Learning에 대한 체계적 분석과 핵심 기여 요약'
header: {}
last_modified_at: '2025-09-16'
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

# MaPLe: Multi-modal Prompt Learning

## 논문 정보
- **저자**: Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, Fahad Shahbaz Khan
- **발표**: CVPR 2023
- **ArXiv**: [2210.03117](https://arxiv.org/abs/2210.03117)

## 1. 핵심 요약 (2-3문장)
MaPLE은 vision과 language 브랜치 모두에서 프롬프트를 학습하여 멀티모달 프롬프트 학습의 효과를 극대화하는 방법을 제안합니다.

## 2. 배경 및 동기
![Figure 1 0](/assets/images/paper/maple-multi-modal-prompt-learning/figure_1_0.png)
*Figure: Figure 1 0*
기존의 프롬프트 학습 방법들은 **CLIP의 시각 또는 언어 브랜치 중 하나만을 적응**시키는 한계가 있었습니다. 이로 인해 두 모달리티 간의 정렬이 충분하지 않아 새로운 클래스, 데이터셋, 도메인 변화에 대한 일반화 성능이 제한적이었습니다.
**MaPLe(Multi-modal Prompt Learning)**은 이러한 문제를 해결하기 위해 **CLIP의 시각과 언어 브랜치를 동시에 적응**시키는 혁신적인 다중 모달 프롬프트 학습 프레임워크를 제안합니다. 특히 시각-언어 프롬프트 간의 **강한 결합을 촉진**하여 두 표현 공간 간의 정렬을 개선합니다.
<p align="center">
<img src="https://arxiv.org/abs/2210.03117" alt="MaPLe Paper" style="width: 100%;">
</p>

## 3. 제안 방법

### 3.1 아키텍처 개요
**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성

### 3.2 핵심 기술/알고리즘
**약점**: 아쉬웠던 부분이나 의문점
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준

### 3.3 구현 세부사항
구현과 관련된 중요한 기술적 세부사항들을 다룹니다.

## 4. 실험 및 결과

### 4.1 실험 설정
![Results Table 7 0](/assets/images/paper/maple-multi-modal-prompt-learning/results_table_7_0.png)
*Figure: Results Table 7 0*
![Results Table 7 1](/assets/images/paper/maple-multi-modal-prompt-learning/results_table_7_1.png)

### 4.2 주요 결과
- 11개 다양한 이미지 인식 데이터셋에서 일관된 성능 향상
**Base-Novel Class Balance**
- 기존 클래스 성능 유지하면서 새로운 카테고리 일반화 향상
- 프롬프트 학습 접근법의 **일반적인 base-novel 클래스 성능 트레이드오프 문제 해결**
- 균형잡힌 성능으로 실용적 적용 가능성 증대

### 4.3 분석
실험 결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향


## 6. 개인적 평가

**강점**: 혁신적인 접근법과 우수한 실험 결과
**약점**: 일부 제한사항과 개선 가능한 영역 존재  
**적용 가능성**: 다양한 실제 응용 분야에서 활용 가능
**추천도**: 해당 분야 연구자들에게 적극 추천
