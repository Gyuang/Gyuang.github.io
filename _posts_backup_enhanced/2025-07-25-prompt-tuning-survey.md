---
categories:
- VLM
date: 2025-07-25
excerpt: 에 대한 체계적 분석
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Prompt Tuning
- Survey
- Vision-Language Models
- CLIP
- Few-shot Learning
- Zero-shot Learning
- Parameter-Efficient Fine-tuning
title: 'Prompt Tuning for Vision-Language Models: A Comprehensive Survey'
toc: true
toc_sticky: true
---

# Prompt Tuning for Vision-Language Models: A Comprehensive Survey

## 논문 정보
- **저자**: | 년도 | arXiv |
- **발표**: 
- **ArXiv**: N/A

## 1. 핵심 요약 (2-3문장)
이 논문의 핵심 기여와 주요 발견을 간결하게 요약합니다.

## 2. 배경 및 동기
비전-언어 모델(Vision-Language Models, VLMs)의 등장과 함께 **프롬프트 튜닝(Prompt Tuning)**은 효율적인 모델 적응을 위한 핵심 패러다임으로 부상했습니다. 특히 CLIP과 같은 대규모 사전 훈련된 모델의 성공은 전체 모델을 재훈련하지 않고도 downstream 작업에 효과적으로 적응할 수 있는 방법의 필요성을 대두시켰습니다.

**프롬프트 튜닝의 중요성**
- **Parameter Efficiency**: 전체 모델 파라미터 대신 소수의 프롬프트 토큰만 학습
- **Few-shot Learning**: 제한된 라벨된 데이터로도 강력한 성능 달성
- **Domain Adaptation**: 다양한 도메인과 작업에 빠른 적응 가능
- **Zero-shot Generalization**: 새로운 클래스나 도메인에 대한 일반화 능력

이 조사 연구는 **비전-언어 모델을 위한 프롬프트 튜닝 방법론들의 전체적인 landscape**를 제공하며, 최신 연구 동향과 핵심 기법들을 체계적으로 분석합니다.

## 3. 제안 방법

### 3.1 아키텍처 개요

![Architecture Diagram 3 0](/assets/images/paper/prompt-tuning-survey/architecture_diagram_3_0.png)
*Figure: Architecture Diagram 3 0*



### 3.2 핵심 기술/알고리즘
최근 비전-언어 모델 분야의 급속한 발전과 함께 **32개의 주요 프롬프트 튜닝 논문**들이 다양한 접근법을 제시하며 이 분야를 형성해왔습니다. 이들 연구는 다음과 같은 주요 방향으로 분류됩니다:

**프롬프트 튜닝 접근법 분류**
1. **Text Prompt Tuning Methods** (17편) - 언어 모달리티 중심 접근
2. **Visual Prompt Tuning Methods** (2편) - 시각 모달리티 중심 접근  
3. **Multi-modal Prompt Tuning Methods** (4편) - 양 모달리티 동시 적응
4. **Context Optimization Methods** (2편) - 컨텍스트 최적화 기법
5. **Test-Time Methods** (2편) - 테스트 시점 적응 방법
6. **Domain-Specific Methods** (2편) - 특정 도메인 특화 접근
7. **Feature Adaptation Methods** (3편) - 특징 공간 적응 기법

각 카테고리는 서로 다른 기술적 관점과 적용 시나리오를 다루며, **효율적인 파라미터 튜닝부터 강건한 일반화**에 이르기까지 다양한 목표를 추구합니다.



현재까지 본 블로그에서 다룬 주요 프롬프트 튜닝 논문들을 소개합니다:



**CoOp (Learning to Prompt for Vision-Language Models)**
- [Blog Post: CoOp-Learning-to-Prompt-for-Vision-Language-Models](/paper/CoOp-Learning-to-Prompt-for-Vision-Language-Models/)
- **핵심 기여**: 수동 프롬프트 엔지니어링을 학습 가능한 연속 벡터로 대체한 선구적 연구
- **기술적 특징**: Unified Context (UC)와 Class-Specific Context (CSC) 두 가지 구현 방식 제공
- **성과**: Few-shot 설정에서 수동 프롬프트 대비 최대 45% 성능 향상

**CoCoOp (Conditional Prompt Learning for Vision-Language Models)**
- [Blog Post: CoCoOp-Conditional-Prompt-Learning-for-Vision-Language-Models](/paper/CoCoOp-Conditional-Prompt-Learning-for-Vision-Language-Models/)
- **핵심 기여**: 입력 이미지에 조건화된 동적 프롬프트 생성으로 일반화 성능 향상
- **기술적 특징**: Meta-Net을 통한 인스턴스별 컨텍스트 벡터 생성
- **성과**: Base-to-novel 일반화에서 CoOp 대비 상당한 성능 개선



**MaPLe (Multi-modal Prompt Learning)**
- [Blog Post: MaPLE-Multi-modal-Prompt-Learning](/paper/MaPLE-Multi-modal-Prompt-Learning/)
- **핵심 기여**: 시각과 언어 브랜치를 동시에 적응시키는 다중 모달 프롬프트 학습
- **기술적 특징**: 시각-언어 결합 함수를 통한 강한 모달리티 정렬
- **성과**: Co-CoOp 대비 3.45% 절대 성능 향상 (novel class 일반화)



**TPT (Test-Time Prompt Tuning for Zero-Shot Generalization)**
- [Blog Post: TPT-Test-Time-Prompt-Tuning-for-Zero-Shot-Generalization](/paper/TPT-Test-Time-Prompt-Tuning-for-Zero-Shot-Generalization/)
- **핵심 기여**: 테스트 시점에서 단일 이미지로 프롬프트를 동적 적응
- **기술적 특징**: Confidence selection과 augmentation ensemble 기법
- **성과**: Natural distribution shift에서 기존 방법들 대비 뛰어난 견고성



비전-언어 모델을 위한 프롬프트 튜닝 분야의 전체 논문 목록을 카테고리별로 정리합니다:



| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Learning to Prompt for Vision-Language Models | Zhou et al. | 2022 | [2109.01134](https://arxiv.org/abs/2109.01134) |
| Conditional Prompt Learning for Vision-Language Models | Zhou et al. | 2022 | [2203.05557](https://arxiv.org/abs/2203.05557) |
| DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting | Rao et al. | 2022 | [2112.01518](https://arxiv.org/abs/2112.01518) |
| Prompt Distribution Learning | Lu et al. | 2022 | [2205.03340](https://arxiv.org/abs/2205.03340) |
| Learning to Prompt for Continual Learning | Wang et al. | 2022 | [2112.08654](https://arxiv.org/abs/2112.08654) |
| ProGrad: Prompt Gradient Projection for Continual Learning | Zhu et al. | 2023 | [2205.14865](https://arxiv.org/abs/2205.14865) |
| KgCoOp: Visual Representation Learning with Knowledge-guided Context Optimization | Yao et al. | 2023 | [2303.13283](https://arxiv.org/abs/2303.13283) |
| Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models | Chen et al. | 2024 | [2405.07990](https://arxiv.org/abs/2405.07990) |
| Diversity-Aware Meta Visual Prompting | Huang et al. | 2023 | [2303.08138](https://arxiv.org/abs/2303.08138) |
| Learning Transferable Visual Models From Natural Language Supervision | Radford et al. | 2021 | [2103.00020](https://arxiv.org/abs/2103.00020) |
| ProText: Prompt-based Text Classification with Improved Rationalization | Cheng et al. | 2023 | [2305.10449](https://arxiv.org/abs/2305.10449) |
| TCP: Textual-based Class-aware Prompt tuning for Visual-Language Model | Huang et al. | 2023 | [2311.18231](https://arxiv.org/abs/2311.18231) |
| SuS-X: Training-Free Name-Only Transfer of Vision-Language Models | Udandarao et al. | 2023 | [2211.16198](https://arxiv.org/abs/2211.16198) |
| PLOT: Prompt Learning with Optimal Transport for Vision-Language Models | Chen et al. | 2023 | [2210.01253](https://arxiv.org/abs/2210.01253) |
| ProDA: Prompt-based Data Augmentation for Low-Resource NLU Tasks | Qin et al. | 2022 | [2202.07179](https://arxiv.org/abs/2202.07179) |
| Prompt-based Learning for Unpaired Image Captioning | Li et al. | 2022 | [2204.03906](https://arxiv.org/abs/2204.03906) |
| Reading Books is Great, But Not if You Are Driving! | Materzynska et al. | 2022 | [2203.09346](https://arxiv.org/abs/2203.09346) |



| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Visual Prompt Tuning | Jia et al. | 2022 | [2203.12119](https://arxiv.org/abs/2203.12119) |
| Exploring Visual Prompts for Adapting Large-Scale Models | Bahng et al. | 2022 | [2203.17274](https://arxiv.org/abs/2203.17274) |



| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| MaPLe: Multi-modal Prompt Learning | Khattak et al. | 2023 | [2210.03117](https://arxiv.org/abs/2210.03117) |
| PromptSRC: Prompt-based Semi-supervised Learning for Scene Recognition | Khattak et al. | 2022 | [2210.10505](https://arxiv.org/abs/2210.10505) |
| Multi-modal Prompting for Low-Shot Temporal Action Localization | Liu et al. | 2023 | [2303.11732](https://arxiv.org/abs/2303.11732) |
| VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval | Huang et al. | 2023 | [2211.12764](https://arxiv.org/abs/2211.12764) |



| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Context Optimization for CLIP | Zhou et al. | 2022 | [2109.01134](https://arxiv.org/abs/2109.01134) |
| CLIP-Adapter: Better Vision-Language Models with Feature Adapters | Gao et al. | 2021 | [2110.04544](https://arxiv.org/abs/2110.04544) |



![Figure 1 2](/assets/images/paper/prompt-tuning-survey/figure_1_2.png)
*Figure: Figure 1 2*


| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models | Shu et al. | 2022 | [2209.07511](https://arxiv.org/abs/2209.07511) |
| Test-Time Training with Self-Supervision for Generalization under Distribution Shifts | Sun et al. | 2020 | [1909.13231](https://arxiv.org/abs/1909.13231) |



![Figure 1 1](/assets/images/paper/prompt-tuning-survey/figure_1_1.png)
*Figure: Figure 1 1*


| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains | Ge et al. | 2022 | [2212.04196](https://arxiv.org/abs/2212.04196) |
| Learning Domain-Specific Prompts for Vision-Language Models | Ramos et al. | 2023 | [2308.04875](https://arxiv.org/abs/2308.04875) |



![Architecture Diagram 3 0](/assets/images/paper/prompt-tuning-survey/architecture_diagram_3_0.png)
*Figure: Architecture Diagram 3 0*


| 논문 제목 | 저자 | 년도 | arXiv |
|-----------|------|------|-------|
| Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification | Zhang et al. | 2021 | [2111.03930](https://arxiv.org/abs/2111.03930) |
| CLIP-Adapter: Better Vision-Language Models with Feature Adapters | Gao et al. | 2021 | [2110.04544](https://arxiv.org/abs/2110.04544) |
| AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition | Chen et al. | 2022 | [2205.13535](https://arxiv.org/abs/2205.13535) |



프롬프트 튜닝 분야의 주요 기술적 혁신들을 정리하면:


- **CoOp**: 수동 텍스트 토큰을 학습 가능한 연속 벡터로 대체
- **Parameter Efficiency**: 전체 모델 동결, 프롬프트 토큰만 학습


- **CoCoOp**: 입력별 동적 프롬프트 생성으로 일반화 성능 향상
- **Meta-learning**: Meta-Net을 통한 인스턴스 조건화


- **MaPLe**: 시각-언어 브랜치 동시 적응으로 강한 모달리티 정렬
- **Cross-modal Coupling**: 결합 함수를 통한 상호 시너지 유도


- **TPT**: 테스트 시점 단일 이미지 기반 동적 프롬프트 적응
- **Distribution Robustness**: Natural shift에 대한 견고성 향상



![Figure 1 0](/assets/images/paper/prompt-tuning-survey/figure_1_0.png)
*Figure: Figure 1 0*


프롬프트 튜닝 방법들의 성능 발전 추이:

**Few-shot Learning**
- CoOp: 수동 프롬프트 대비 최대 45% 향상
- CoCoOp: Base-to-novel 일반화에서 CoOp 개선
- MaPLe: 전체 조화 평균에서 2.72% 추가 향상

**Zero-shot Generalization**
- TPT: Natural distribution shift에서 기존 방법 대비 우수한 견고성
- 테스트 시점 적응의 효과성 입증

**Parameter Efficiency**
- 모든 방법들이 전체 모델 재훈련 없이 효과적 적응 달성
- 프롬프트 토큰 수는 일반적으로 16개 미만으로 제한





**1. Generalization vs. Specialization Trade-off**
- Base class 성능 유지와 novel class 적응 간 균형
- Domain shift에 대한 견고성 확보

**2. Prompt Engineering Complexity**
- 최적 프롬프트 길이와 위치 결정
- 다양한 작업에 대한 범용적 설계

**3. Computational Efficiency**
- 테스트 시점 적응의 계산 비용
- 대규모 모델에서의 확장성



**1. Automated Prompt Design**
- 신경 아키텍처 탐색(NAS) 기반 프롬프트 구조 최적화
- 강화학습을 통한 프롬프트 정책 학습

**2. Multi-task Prompt Learning**
- 여러 작업 간 프롬프트 지식 공유
- 메타학습 기반 빠른 작업 적응

**3. Continual Prompt Learning**
- 새로운 작업 학습 시 이전 지식 보존
- 치명적 망각(Catastrophic Forgetting) 문제 해결

**4. Cross-modal Prompt Transfer**
- 시각-언어 외 다른 모달리티 조합 탐색
- 오디오, 3D 등 확장된 모달리티 지원

### 3.3 구현 세부사항


## 4. 실험 및 결과

### 4.1 실험 설정
실험에 사용된 데이터셋, 평가 지표, 비교 대상을 설명합니다.

### 4.2 주요 결과


실험 결과와 성능 분석을 제시합니다.

### 4.3 분석
결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향
비전-언어 모델을 위한 프롬프트 튜닝은 **효율적인 모델 적응의 핵심 패러다임**으로 자리잡았습니다. CoOp의 선구적 연구부터 최신 다중 모달 접근법들까지, 이 분야는 다음과 같은 주요 발전을 이루어왔습니다:

**주요 성과**
1. **Parameter Efficiency**: 전체 모델 대신 소수 프롬프트 토큰만으로 효과적 적응
2. **Few-shot Learning**: 제한된 데이터로도 강력한 성능 달성
3. **Generalization**: 새로운 클래스, 도메인, 분포에 대한 견고한 일반화
4. **Technical Diversity**: 단일 모달리티부터 다중 모달, 테스트 시점 적응까지 다양한 접근법

**미래 전망**
프롬프트 튜닝 분야는 **자동화된 프롬프트 설계, 다중 작업 학습, 지속적 학습** 등의 방향으로 발전할 것으로 예상됩니다. 특히 의료, 자율주행, 로보틱스 등 **실제 응용 분야에서의 실용성 검증**이 중요한 과제로 남아있습니다.

이 조사 연구를 통해 연구자들이 프롬프트 튜닝 분야의 전체적인 landscape를 이해하고, 향후 연구 방향을 설정하는 데 도움이 되기를 바랍니다.

---

**관련 포스트:**
- [CoOp: Learning to Prompt for Vision-Language Models](/paper/CoOp-Learning-to-Prompt-for-Vision-Language-Models/)
- [CoCoOp: Conditional Prompt Learning for Vision-Language Models](/paper/CoCoOp-Conditional-Prompt-Learning-for-Vision-Language-Models/)
- [MaPLe: Multi-modal Prompt Learning](/paper/MaPLE-Multi-modal-Prompt-Learning/)
- [TPT: Test-Time Prompt Tuning for Zero-Shot Generalization](/paper/TPT-Test-Time-Prompt-Tuning-for-Zero-Shot-Generalization/)

## 6. 개인적 평가

**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점  
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준

