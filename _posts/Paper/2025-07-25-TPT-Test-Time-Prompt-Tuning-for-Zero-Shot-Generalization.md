---
categories:
- paper
- vlm
date: 2025-07-25
excerpt: Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models
  (TPT)에 대한 체계적 분석과 핵심 기여 요약
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Vision-Language
- Test-Time Adaptation
- Zero-shot Learning
- Prompt Tuning
- CLIP
title: Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models
  (TPT)
toc: true
toc_sticky: true
---

# Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models (TPT)

## 0. 체크리스트
- [ ] `categories` 두 번째 값이 `medical-ai`, `vlm`, `rag`, `multimodal`, `transformer` 등 실제 분류인지 확인했나요?
- [ ] `excerpt`에 구체적인 결과/기여가 들어가나요?
- [ ] 모든 섹션에 실제 내용이 채워졌나요? (플레이스홀더 금지)
- [ ] 수치/결과 표는 3~5개 이하로 요약했나요?
- [ ] 참고 링크(코드/데이터)가 있으면 마지막에 정리했나요?

> **작성 팁**: 각 절은 3~6문장 사이로 명확히 작성하고, 표나 리스트는 실제 실험 수치를 기반으로 요약합니다. 불필요한 영어 번역 반복, “혁신적인 연구” 같은 템플릿 문구는 사용하지 않습니다.

## 1. 핵심 요약 (3문장 이하)
- 주요 문제 정의와 모델/기법 이름
- 뛰어난 성능 혹은 특징 (정량적 수치 포함)
- 연구가 주는 실제 영향 혹은 차별점

## 2. 배경 & 동기
- 기존 접근 방식의 한계 또는 해결하고자 하는 문제
- 제안 방법이 필요해진 산업/연구적 배경
- 핵심 아이디어 미리보기 (고유한 관점 강조)

## 3. 방법론
### 3.1 전체 구조
- 모델/파이프라인 구조 도식 요약 (필요 시 Diagram 설명)
- 입력, 주요 모듈, 출력 흐름 설명

### 3.2 핵심 기법
- 새로 도입된 알고리즘, Loss, 모듈 설명
- 기존 모델 대비 개선 포인트 2~3가지 정리

### 3.3 학습 및 구현 세부
- 데이터 전처리/증강, 하이퍼파라미터, 학습 전략
- 재현성에 필요한 공개 코드/설정 등

## 4. 실험 & 결과
### 4.1 설정
- 데이터셋, 평가 지표, 비교 대상 (Baseline)
- 하드웨어나 학습 비용 등 실무자가 궁금해할 정보

### 4.2 주요 결과표
| Metric | Our Model | Baseline A | Baseline B |
| ------ | --------- | ---------- | ---------- |
| 예: Dice | 0.89 | 0.82 | 0.81 |
| 예: BLEU-4 | 0.72 | 0.60 | 0.58 |

### 4.3 추가 분석
- 에이블레이션, 오류 사례 분석, 사용자 평가 등 핵심 통찰 2~3가지

## 5. 의의 & 한계
- 임상/산업/연구에서의 실제 임팩트
- 한계나 향후 연구 방향 (정직하게 작성)

## 6. 개인 평가
**강점**: 
**약점**: 
**적용 가능성**: 
**추천도**: (예: ★★★★☆)

## 7. 참고 자료
- 원문: [링크](https://arxiv.org/abs/XXXX)
- 코드: [GitHub](https://github.com/...)
- 데이터: [URL]


## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/TPT-Test-Time-Prompt-Tuning-for-Zero-Shot-Generalization/fig_01.png)
캡션: Figure 1: Test-time Prompt Tuning (TPT) for image classification. We tune adaptive prompts on the fly with a single test sample, without the need for additional training data or annotations. TPT optimizes the prompt to encourage consistent predictions across augmented views by minimizing the marginal entropy. We introduce confidence selection to filter out noisy augmentations.

### Main Results Table
![Results](/assets/images/paper/TPT-Test-Time-Prompt-Tuning-for-Zero-Shot-Generalization/table_242.png)
캡션: Results. In Table (<>)2, we compare TPT with few-shot prompt tuning methods on generalization from ImageNet to fine-grained datasets. Note that TPT works in a zero-shot manner; thus it is not trained on ImageNet. Nonetheless, we find TPT to achieve on-par generalization as ImageNet trained CoCoOp. In Figure (<>)3, we present the results of the more challenging setting of cross-dataset generalizati…

## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build

