---
categories:
- paper
- vlm
date: 2025-07-25
excerpt: 'CoCoOp는 입력 인스턴스 조건부 토큰으로 CLIP를 적응시켜 11개 데이터셋 평균에서 Novel 클래스 정확도를 63.22%→71.69%로 끌어올리고, SUN397·Flowers102 등에서도 조화 평균을 크게 개선합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- CoCoOp
- Prompt Learning
- CLIP Adaptation
- Generalization
title: 'CoCoOp: Conditional Prompt Learning for Vision-Language Models'
toc: true
toc_sticky: true
---
# CoCoOp: Conditional Prompt Learning for Vision-Language Models

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `vlm`으로 맞춰져 있나요?
- [x] `excerpt`에 주요 개선 수치를 넣었나요?
- [x] 모든 섹션을 실제 논문 내용으로 채웠나요?
- [x] 결과 표는 핵심 지표 위주로 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- CoCoOp은 CLIP 프롬프트의 각 컨텍스트 토큰을 이미지 의존적 벡터로 생성해, CoOp의 정적 프롬프트가 보이지 않는 클래스에 과적합되는 문제를 해소합니다.
- 11개 데이터셋 평균에서 Novel 클래스 정확도가 63.22%→71.69%로 8.47pt 상승했고, 조화 평균도 71.66%→75.83%로 개선돼 전반적 전이를 안정화했습니다.
- SUN397(76.86% H)·Flowers102(81.71% H)·EuroSAT(71.21% H) 등에서 CLIP 대비 조화 평균이 크게 늘어나 고정 프롬프트 대비 강건성을 입증했습니다.

## 2. 배경 & 동기
- CLIP는 텍스트 프롬프트 디자인에 민감해, Zero-shot 상황에서 수작업 템플릿이 성능을 좌우합니다.
- CoOp는 학습 가능한 컨텍스트 벡터를 도입했지만, 학습에 사용된 base 클래스에 과하게 맞춰 novel 클래스 성능이 떨어졌습니다.
- 저자들은 인스턴스별 조건부 토큰을 생성하는 경량 네트워크를 추가해, 적응성을 유지하면서도 프롬프트 과적합을 방지하고자 했습니다.

## 3. 방법론
### 3.1 전체 구조
- CLIP 텍스트 인코더 앞단에 이미지 특징을 입력으로 받는 MLP를 배치해, 각 샘플마다 $v(x)$ 토큰을 생성합니다.
- 학습 시 base 클래스 이미지를 사용해 조건부 토큰과 클래스 토큰을 연결하고, CLIP의 크로스 엔트로피 손실로 미세 조정합니다.
- 추론에서는 같은 MLP가 novel 클래스 이미지에 대해서도 토큰을 생성하므로, 프롬프트가 샘플별로 적응합니다.

### 3.2 핵심 기법
- **Dynamic Prompting**: 이미지 의존적 토큰을 추가해 프롬프트가 클래스 외형마다 달라지도록 설계했습니다.
- **Lightweight Adapter**: 2층 MLP(512→512→512) 구조만 추가해 파라미터 증가를 최소화했습니다.
- **공유 클라스 토큰**: 클래스 이름은 CLIP 토큰화를 그대로 활용해, 새로운 클래스에도 확장 가능하도록 했습니다.

### 3.3 학습 및 구현 세부
- 백본으로 ViT-B/16 CLIP을 사용하고, 학습률 0.002·배치 4·에폭 10 조건으로 base 클래스 데이터에만 미세 조정했습니다.
- 이미지 인코더는 얼리고 텍스트 인코더와 MLP만 업데이트했습니다.
- 데이터셋은 ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101입니다.

## 4. 실험 & 결과
### 4.1 설정
- Base→Novel 일반화: base 클래스에서 학습 후 novel 클래스 성능을 평가해 CoOp 대비 일반화 성능을 비교했습니다.
- 비교군: Zero-shot CLIP, CoOp(정적 프롬프트), CoCoOp(조건부 프롬프트).
- 추가로 cross-dataset 전이 및 domain shift 실험도 진행했으나, 핵심 지표는 base→novel 조화 평균입니다.

### 4.2 주요 결과표
| 데이터셋 | CLIP H | CoOp H | **CoCoOp H** |
| --- | --- | --- | --- |
| 평균(11개) | 71.70 | 71.66 | **75.83** |
| ImageNet | 70.22 | 71.92 | **73.10** |
| SUN397 | 72.23 | 72.51 | **78.27** |
| Flowers102 | 74.83 | 74.06 | **81.71** |
| EuroSAT | 60.03 | 68.69 | **71.21** |

### 4.3 추가 분석
- OxfordPets·StanfordCars 등 fine-grained 데이터에서도 CoOp 대비 novel 클래스 정확도가 2~13pt 상승해, 인스턴스 조건부 토큰이 세밀한 차이를 포착함을 보였습니다.
- FGVCAircraft처럼 노이즈가 많은 경우에는 H가 27.74%로 CoOp보다 소폭 낮지만, novel 성능은 여전히 CLIP를 상회합니다.
- cross-dataset 전이 실험에서도 하나의 데이터에서 학습한 토큰을 다른 데이터에 적용해 조화 평균을 개선했습니다.

## 5. 의의 & 한계
- 소량의 파라미터만 추가해 CLIP를 다양한 downstream 데이터에 적응시키는 실용적 메서드를 제시했습니다.
- novel 클래스에서 큰 폭의 향상을 이뤄 prompts overfitting 문제를 완화했습니다.
- 다만 FGVCAircraft 같은 극단적 도메인에서는 여전히 성능이 불안정하며, 조건부 토큰 생성 네트워크가 추가 학습 데이터를 요구할 수 있습니다.

## 6. 개인 평가
**강점**: 프롬프트 학습에 조건부 토큰을 도입해 novel 클래스 일반화를 실질적으로 향상시켰습니다.  
**약점**: 일부 노이즈 많은 데이터에서는 여전히 하락세가 있으며, 추가 MLP가 inference latency를 조금 늘립니다.  
**적용 가능성**: CLIP 기반 다운스트림 태스크에서 “few-shot→novel” 시나리오를 다뤄야 할 때 강력한 기본 베이스라인이 됩니다.  
**추천도**: ★★★★☆ (프롬프트 학습을 연구하거나 CLIP 전이를 다루는 팀에게 유용)

## 7. 참고 자료
- 원문: [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)
- 코드: [CoOp/CoCoOp GitHub](https://github.com/KaiyangZhou/CoOp)
