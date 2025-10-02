---
categories:
- paper
- vlm
date: 2025-07-25
excerpt: 'CoOp는 CLIP 프롬프트 컨텍스트를 학습 가능한 벡터로 바꿔 base 클래스 정확도를 평균 82.69%까지 올렸지만, novel 클래스에서는 63.22%로 떨어져 과적합 문제가 드러납니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- CoOp
- Prompt Learning
- CLIP Adaptation
- Few-Shot
title: 'CoOp: Learning to Prompt for Vision-Language Models'
toc: true
toc_sticky: true
---
# CoOp: Learning to Prompt for Vision-Language Models

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `vlm`으로 설정돼 있나요?
- [x] `excerpt`에 핵심 수치를 포함했나요?
- [x] 모든 섹션이 논문 내용을 반영하나요?
- [x] 결과 표는 필요 지표만 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- CoOp는 CLIP 텍스트 프롬프트의 컨텍스트 단어를 학습 가능한 벡터로 바꿔, 수작업 템플릿 없이도 few-shot 환경에서 프롬프트를 최적화합니다.
- base 클래스 평균 정확도는 69.34%→82.69%로 상승했지만 novel 클래스 평균은 63.22%로 낮아져 조화 평균 71.66%로 CLIP와 비슷한 수준에 머물렀습니다.
- Flowers102(97.60% base vs 59.67% novel), EuroSAT(92.19% vs 54.74%) 등에서 큰 격차가 나타나, 프롬프트가 학습 분포에 과적합됨을 보여줍니다.

## 2. 배경 & 동기
- CLIP는 class name 앞뒤에 어떤 단어를 넣느냐에 따라 zero-shot 성능이 크게 달라져 prompt engineering 비용이 큽니다.
- NLP에서 발전한 prompt tuning을 비전 도메인으로 가져와 자동으로 최적 컨텍스트를 찾고자 했습니다.
- CoOp는 backbone을 동결하고 프롬프트 벡터만 학습해 파라미터 효율성을 유지합니다.

## 3. 방법론
### 3.1 전체 구조
- CLIP 텍스트 인코더 입력인 “[V]1 [V]2 … [V]M [CLASS]” 중 [V] 토큰을 learnable embedding으로 바꾸고, 나머지 구성은 유지합니다.
- 학습 시 base 클래스 이미지와 레이블만 사용해 cross-entropy로 컨텍스트 벡터를 업데이트합니다.
- 추론에서는 동일한 컨텍스트를 모든 샘플에 적용하므로, 간결하지만 정적(default) 프롬프트가 됩니다.

### 3.2 핵심 기법
- **Context Optimization**: 컨텍스트 길이 M(보통 16)을 설정한 후, classifier를 학습하면서 동시에 컨텍스트를 최적화합니다.
- **Backbone Freezing**: 이미지/텍스트 인코더를 고정해 미세 조정 비용과 catastrophic forgetting을 줄였습니다.
- **Few-shot Friendly**: 클래스당 K(=16) 이미지로 충분한 성능 향상을 얻을 수 있도록 설계했습니다.

### 3.3 학습 및 구현 세부
- optimizer: SGD, 학습률 0.002, 배치 4, 에폭 10.
- 이미지 인코더는 ViT-B/16 CLIP를 사용했고, 컨텍스트 길이 16으로 고정했습니다.
- 데이터셋은 ImageNet, Caltech101, OxfordPets 등 11개로 구성됐습니다.

## 4. 실험 & 결과
### 4.1 설정
- few-shot 학습(각 클래스 16샘플)을 통해 base 클래스에서 컨텍스트를 학습하고, novel 클래스 성능을 별도로 측정했습니다.
- 비교: Zero-shot CLIP, 수작업 프롬프트, CoOp.
- 평가: Base/New/Harmonic(%) 지표 중심.

### 4.2 주요 결과표
| 데이터셋 | CLIP H | **CoOp Base** | CoOp New | CoOp H |
| --- | --- | --- | --- | --- |
| 평균(11개) | 71.70 | **82.69** | 63.22 | 71.66 |
| Flowers102 | 74.83 | **97.60** | 59.67 | 74.06 |
| EuroSAT | 60.03 | **92.19** | 54.74 | 68.69 |
| StanfordCars | 68.65 | **78.12** | 60.40 | 68.13 |

### 4.3 추가 분석
- base 분포에선 수작업 템플릿보다 큰 폭의 성능 향상을 얻었지만, novel 클래스로 전환하면 오히려 제로샷 성능보다 낮아졌습니다.
- base에서 얻은 고정 프롬프트가 novel 이미지에 맞춰지지 않아 FGVCAircraft, DTD 등 도메인에서 큰 하락(-18pt 이상)이 관측됐습니다.
- 이러한 한계는 이후 CoCoOp에서 조건부 토큰을 도입하는 동기가 되었습니다.

## 5. 의의 & 한계
- 프롬프트 학습을 vision-language 전이에 처음 확장해 자동화 가능성을 열었습니다.
- base 클래스에선 큰 성능 향상을 보이지만, novel 클래스 일반화가 제한돼 후속 연구 필요성을 남겼습니다.
- CLIP backbone을 고정하므로 학습은 안정적이지만, 다양한 도메인 전이에 적용하려면 추가 설계가 필요합니다.

## 6. 개인 평가
**강점**: 단순한 구조로 prompt engineering 비용을 크게 줄이고, few-shot 학습에서 즉각적인 성능 향상을 제공합니다.  
**약점**: 정적 프롬프트라 novel 분포에 취약하며, 실제 응용에서는 추가 일반화 전략이 필수입니다.  
**적용 가능성**: 동일 도메인 내 편향이 적은 상황이나, base 클래스만 관심 있는 경우에 유용한 시작점입니다.  
**추천도**: ★★★☆☆ (Prompt 학습 개념을 익히고 싶은 연구자에게 좋지만, 실전 전이는 CoCoOp 등 후속 기법이 더 적합)

## 7. 참고 자료
- 원문: [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
- 코드: [CoOp GitHub](https://github.com/KaiyangZhou/CoOp)
