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
MaPLe은 **시각과 언어 브랜치를 동시에 적응**시키는 다중 모달 프롬프트 학습 방법으로, **기존 대비 3-5% 성능 향상**을 달성했습니다. 특히 **매핑 네트워크를 통한 시각-언어 프롬프트 간 강한 결합**을 촉진하여 두 표현 공간의 정렬을 개선했습니다.

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
MaPLe은 **다중 모달 프롬프트 학습 아키텍처**로 CLIP의 비전 및 언어 브랜치를 동시에 적응시킵니다. 기존 방법들이 한 모달리티에만 집중하는 대신, **두 모달리티의 신중한 균형과 강한 결합**을 추구합니다.

**주요 아키텍처 구성:**
- **Vision Branch Prompts**: ViT의 다중 레이어에 학습 가능한 프롬프트 삽입
- **Language Branch Prompts**: 텍스트 인코더에 컨텍스트 프롬프트 추가
- **Mapping Network**: 비전-언어 프롬프트 간 상호작용 네트워크
- **Shared Projection**: 두 모달리티의 통일된 표현 공간

### 3.2 핵심 기술/알고리즘
**다중 모달 프롬프트 학습 (Multi-modal Prompt Learning):**

1. **Vision Prompts**: 각 ViT 블록에 학습 가능한 토큰 삽입
```
Vision Input: [CLS] + Vision_Prompts + Image_Patches
```

2. **Language Prompts**: 텍스트 인코더에 컨텍스트 프롬프트 추가
```
Text Input: [V₁][V₂]...[Vₘ][CLASS]
```

3. **Coupling Function**: 비전과 언어 프롬프트 간 상호 의존성 모델링
```
f: V_prompts ↔ L_prompts (bidirectional coupling)
```

**협력적 최적화 전략:**
- 두 모달리티의 **공동 훈련**으로 상호 보완적 학습
- **매핑 네트워크**를 통한 cross-modal alignment 강화
- **점진적 훈련**으로 안정적 수렴

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
MaPLe은 **다중 모달 프롬프트 학습**의 새로운 표준을 제시하며, 비전-언어 모델의 효율적 적응에 혁명적 기여를 했습니다.

**이론적 기여:**
1. **Multi-modal Prompt Paradigm**: 단일에서 다중 모달리티 프롬프트로의 진화
2. **Cross-modal Alignment**: 비전-언어 표현 공간의 체계적 정렬 방법론
3. **Coupling Mechanism**: 모달리티 간 상호작용의 수학적 모델링
4. **Generalization Theory**: Base-novel 클래스 균형의 새로운 이론적 기초

**실용적 영향:**
- **산업 표준**: 다중 모달 모델 개발의 베스트 프랙티스 제시
- **비용 효율성**: 전체 모델 재훈련 없이 고성능 달성
- **확장성**: 다양한 비전-언어 모델에 쉽게 적용
- **안정성**: 대규모 데이터셋에서도 안정적 성능

**후속 연구 촉진:**
MaPLe의 다중 모달 접근법은 **VPT, AdaptFormer, BitFit** 등 parameter-efficient 방법들과 결합되어 더욱 발전된 형태로 진화하고 있으며, **멀티모달 AI**의 핵심 기술로 자리잡았습니다.

## 6. 개인적 평가

**강점**: 혁신적인 접근법과 우수한 실험 결과
**약점**: 일부 제한사항과 개선 가능한 영역 존재  
**적용 가능성**: 다양한 실제 응용 분야에서 활용 가능
**추천도**: 해당 분야 연구자들에게 적극 추천