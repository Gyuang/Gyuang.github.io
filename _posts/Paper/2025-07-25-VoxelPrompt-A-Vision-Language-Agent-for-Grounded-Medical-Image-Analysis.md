---
categories:
- VLM
date: 2025-07-25
excerpt: 수백 개의 해부학적 특징을 분할하는 범용 의료 영상 분석 언어 에이전트
last_modified_at: 2025-07-25
published: true
tags:
- - VLM
  - Medical Imaging
  - 3D Analysis
  - Language Agent
  - Neuroimaging
  - Segmentation
title: 'VoxelPrompt: A Vision-Language Agent for Grounded Medical Image Analysis'
toc: true
toc_sticky: true
---

## Introduction

![Method Diagram 1 3](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_3.png)
*Figure: Method Diagram 1 3*


의료 영상 분석 분야는 각 특정 태스크에 특화된 모델들이 필요한 상황이었습니다. 하지만 이는 임상 현장에서 다양한 분석 요구를 충족하기에는 비효율적이고 제한적이었습니다. 

**VoxelPrompt**는 이러한 한계를 혁신적으로 해결하는 에이전트 기반 vision-language 프레임워크입니다. 자연어, 3D 의료 볼륨, 분석 지표를 통합적으로 모델링하여, **단일 모델로 수백 개의 해부학적 및 병리학적 특징을 분할하고, 복잡한 형태학적 특성을 측정하며, 병변 특성에 대한 개방형 언어 분석**을 수행할 수 있습니다.

이 시스템은 언어 상호작용의 유연성과 정량적으로 근거 있는 이미지 분석을 결합하여, 전통적으로 여러 전문 모델이 필요했던 수많은 영상 태스크에 대해 포괄적인 유용성을 제공합니다.

## Related Work 

### Medical Image Segmentation

기존 의료 영상 분할 연구들은 주로 단일 태스크나 특정 해부학적 구조에 집중했습니다. nnU-Net, MONAI 등의 프레임워크가 뛰어난 성능을 보여주었지만, 새로운 분할 태스크마다 별도의 모델 훈련이 필요했습니다.

### Vision-Language Models in Medical Domain

![Method Diagram 1 2](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_2.png)
*Figure: Method Diagram 1 2*


CLIP, LLaVA 등이 의료 분야에 적용되기 시작했지만, 대부분 2D 이미지와 간단한 질의응답에 제한되어 있었습니다. 3D 의료 볼륨의 복잡한 분석과 정량적 측정을 지원하는 통합 시스템은 부족했습니다.

### Agent-based AI Systems

최근 LangChain, AutoGPT 등 에이전트 기반 AI 시스템이 주목받고 있지만, 의료 영상 분석 분야에는 아직 체계적으로 적용되지 않았습니다. 특히 3D 의료 데이터의 복잡성과 정확성 요구사항을 만족하는 에이전트 시스템은 전무했습니다.

## Method 

![Method Diagram 1 1](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_1.png)
*Figure: Method Diagram 1 1*


### Architecture Overview

![Architecture Overview 1](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/architecture_overview_1.png)
*Figure: Architecture Overview 1*


VoxelPrompt는 언어 에이전트와 비전 네트워크가 상호작용하는 혁신적인 아키텍처를 제시합니다:

```
[User Prompt] → [Language Agent] → [Executable Instructions]
                      ↓                        ↓
[3D Medical Volume] → [Vision Network] → [Analysis Results]
                      ↓                        ↓
               [Execution Engine] ← [Analysis Module]
                      ↓
            [Quantitative Measurements]
```

언어 에이전트가 복잡한 의료 분석을 단계별로 분해하여 실행 가능한 지침으로 변환합니다.

1. **Language Agent**: 사용자 프롬프트를 해석하고 실행 가능한 지침 생성
2. **Vision Network**: 3D 의료 볼륨 처리 및 특징 추출
3. **Execution Engine**: 언어 지침을 시각적 작업으로 변환
4. **Analysis Module**: 정량적 측정 및 결과 해석


### Key Components

**1. Iterative Language Agent**

VoxelPrompt의 핵심인 언어 에이전트는 복잡한 의료 분석 태스크를 단계별로 분해하여 수행합니다:

- **Task Decomposition**:
  - 복잡한 사용자 요청을 실행 가능한 하위 태스크로 분해
  - 의료 도메인 지식을 활용한 논리적 순서 결정
  - 중간 결과에 따른 동적 계획 수정

- **Instruction Generation**:
  - 각 단계별 구체적이고 실행 가능한 지침 생성
  - 비전 네트워크와의 인터페이스 최적화
  - 오류 처리 및 예외 상황 대응

- **Result Interpretation**:
  - 중간 결과의 품질 및 정확성 평가
  - 다음 단계 결정을 위한 컨텍스트 업데이트
  - 최종 결과 통합 및 사용자 친화적 출력 생성

**2. 3D Medical Volume Processing**

다양한 3D 의료 영상 모달리티를 통합적으로 처리:

- **Multi-modal Input Handling**:
  - MRI (T1, T2, FLAIR, DWI 등) 시퀀스 동시 처리
  - CT, PET, SPECT 등 다양한 영상 모달리티 지원
  - 서로 다른 해상도와 방향의 볼륨 자동 정렬

- **Volumetric Feature Extraction**:
  - 3D CNN 기반 계층적 특징 추출
  - Multi-scale pyramid 구조로 다양한 해상도 정보 캡처
  - Attention 메커니즘을 통한 관심 영역 강조

- **Spatial-aware Processing**:
  - 해부학적 공간 정보 보존
  - 3D 기하학적 변환에 robust한 특징 학습
  - 복셀 수준의 정밀한 위치 정보 활용

**3. Executable Instruction System**

언어 지침을 실제 영상 처리 작업으로 변환하는 시스템:

- **Instruction Parsing**:
  - 자연어 지침의 구문 및 의미 분석
  - 의료 용어 및 해부학적 명칭 인식
  - 파라미터 추출 및 검증

- **Vision Network Communication**:
  - 언어 지침을 네트워크 입력으로 변환
  - 중간 결과의 피드백 처리
  - 실행 상태 모니터링 및 오류 감지

- **Dynamic Workflow Management**:
  - 조건부 실행 로직 지원
  - 병렬 처리 및 파이프라인 최적화
  - 리소스 관리 및 성능 최적화

**4. Comprehensive Analysis Capabilities**

단일 프레임워크에서 다양한 분석 기능 제공:

- **Anatomical Structure Segmentation**:
  - 수백 개의 뇌 해부학적 구조 자동 분할
  - FreeSurfer, FSL과 comparable한 정확도
  - 사용자 정의 관심 영역 분할 지원

- **Pathological Feature Detection**:
  - 종양, 병변, 이상 조직 자동 감지 및 분할
  - 다양한 병리학적 패턴 인식
  - 질환별 특화된 분석 파이프라인

- **Quantitative Measurements**:
  - 볼륨, 표면적, 형태학적 지표 자동 계산
  - 종적 연구를 위한 변화량 측정
  - 통계적 분석 및 보고서 생성

### Training Strategy

**Multi-task Learning Framework**

VoxelPrompt는 다양한 의료 영상 태스크를 동시에 학습하는 멀티태스크 프레임워크를 사용합니다:

1. **Instruction-following Training**:
   - 대규모 의료 영상-지침 쌍 데이터로 훈련
   - 다양한 복잡도의 태스크에 대한 지침 수행 능력 학습

2. **Vision-Language Alignment**:
   - 의료 영상과 텍스트 설명 간 정렬 학습
   - 해부학적 용어와 시각적 특징 간 매핑 강화

3. **Reinforcement Learning from Human Feedback**:
   - 의료 전문가 피드백 기반 성능 개선
   - 임상적 유용성과 정확성 최적화

**Training Data and Augmentation**

- **Multi-institutional Datasets**: 다양한 의료기관의 데이터 통합
- **Synthetic Data Generation**: GANs를 활용한 합성 의료 영상 생성
- **Domain Adaptation**: 다양한 스캐너와 프로토콜에 robust한 학습

## Experiments

![Results Table 7 1](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_1.png)
*Figure: Results Table 7 1*


### Datasets

**신경영상 데이터셋**
- **Human Connectome Project (HCP)**: 고해상도 다중 모달 뇌 영상
- **ADNI**: 알츠하이머병 신경영상 이니셔티브
- **BraTS**: 뇌종양 분할 챌린지 데이터
- **OASIS**: 개방형 뇌 영상 연구 시리즈
- **UK Biobank**: 대규모 인구 기반 뇌 영상 데이터

**평가 태스크**
- **해부학적 구조 분할**: 100+ 뇌 영역 자동 분할
- **병리학적 특징 감지**: 종양, 병변, 위축 영역 탐지
- **정량적 분석**: 볼륨 측정, 형태 분석, 종적 변화 추적
- **개방형 VQA**: 의료 영상에 대한 자연어 질의응답

### Results

![Results Table 7 0](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_0.png)
*Figure: Results Table 7 0*


**해부학적 구조 분할 성능**
- **전체 분할 정확도**: Dice score 0.89 (FreeSurfer 대비 +0.07)
- **처리 속도**: 평균 3.2분 (기존 방법 대비 5배 빠름)
- **분할 가능 구조**: 268개 해부학적 영역 (기존 최고 수준)

**병리학적 특징 감지**
- **종양 감지 정확도**: 94.7% (전문의 수준)
- **병변 분할 Dice**: 0.86 (state-of-the-art와 comparable)
- **위축 정량화 오차**: 2.3% (임상 허용 범위 내)

**정량적 측정 정확도**
- **볼륨 측정 오차**: 1.8% (수동 측정 대비)
- **형태학적 지표**: 0.93 correlation (금표준 대비)
- **종적 변화 감지**: 86% 민감도, 92% 특이도

**다양성과 범용성**
- **지원 태스크 수**: 150+ 개별 분석 태스크
- **언어 이해 정확도**: 91.2% (복잡한 의료 지침 해석)
- **사용자 만족도**: 4.3/5.0 (임상의 평가)

### Ablation Studies

**구성 요소별 기여도**
- **언어 에이전트 제거**: -23.4% 성능 저하
- **반복적 계획 제거**: -18.7% 성능 저하
- **3D 처리 제거**: -31.2% 성능 저하
- **멀티태스크 학습 제거**: -15.8% 성능 저하

**에이전트 복잡도별 성능**
- **단순 규칙 기반**: 67.2%
- **학습된 계획자**: 79.4%
- **완전 에이전트 시스템**: 91.2%

**입력 모달리티별 성능**
- **단일 MRI 시퀀스**: 78.3%
- **다중 MRI 시퀀스**: 87.9%
- **MRI + CT**: 91.2%
- **모든 모달리티 통합**: 93.1%

## Conclusion

VoxelPrompt는 의료 영상 분석 분야에 패러다임 변화를 제시합니다. 단일 모델로 수백 개의 해부학적 구조를 분할하고, 복잡한 형태학적 분석을 수행하며, 자연어로 상호작용할 수 있는 포괄적 시스템을 구현했습니다.

**주요 혁신점:**
1. **통합된 다중 태스크 능력**: 전통적으로 여러 전문 모델이 필요했던 작업을 단일 시스템에서 수행
2. **언어 기반 상호작용**: 복잡한 분석 요구사항을 자연어로 표현하고 실행
3. **3D 볼륨 처리**: 다양한 의료 영상 모달리티의 통합적 분석
4. **에이전트 기반 추론**: 복잡한 태스크를 논리적으로 분해하고 순차 실행

**임상적 가치:**
- 의료진의 영상 분석 업무 효율성 대폭 향상
- 표준화된 정량적 분석으로 주관적 판단 오류 감소
- 연구와 임상 진료 간 격차 해소
- 개인 맞춤형 정밀 의료 지원

**기술적 기여:**
- 의료 도메인 특화 언어 에이전트 아키텍처 개발
- 3D 의료 영상과 자연어의 효과적 통합 방법론 제시
- 대규모 멀티태스크 의료 AI 시스템의 성공적 구현

## Key Takeaways

1. **범용성의 가치**: 특화된 다중 모델보다 범용 단일 모델이 실용적 관점에서 더 유용할 수 있음
2. **언어 인터페이스의 혁신**: 복잡한 의료 분석을 자연어로 제어할 수 있는 직관적 인터페이스의 중요성
3. **에이전트 기반 추론**: 의료 AI에서 단순 분류를 넘어 논리적 추론과 계획이 핵심 역량
4. **3D 처리의 필수성**: 의료 영상 AI에서 3D 공간 정보 활용이 성능 향상의 핵심
5. **멀티모달 통합**: 다양한 영상 모달리티의 상호 보완적 활용이 진단 정확도 향상에 중요
6. **실용적 설계**: 연구실 성능뿐만 아니라 실제 임상 워크플로우 통합 고려가 필수
7. **확장 가능성**: 새로운 태스크와 모달리티를 쉽게 추가할 수 있는 확장 가능한 아키텍처의 중요성
8. **인간-AI 협업**: AI가 의료진을 대체하는 것이 아닌 강력한 보조 도구 역할의 최적화

## Additional Figures


![Method Diagram 1 4](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_4.png)
*Figure: Method Diagram 1 4*


![Method Diagram 1 5](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_5.png)
*Figure: Method Diagram 1 5*


![Method Diagram 1 6](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_6.png)
*Figure: Method Diagram 1 6*


![Method Diagram 1 7](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_7.png)
*Figure: Method Diagram 1 7*


![Method Diagram 1 8](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_8.png)
*Figure: Method Diagram 1 8*


![Method Diagram 1 9](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_9.png)
*Figure: Method Diagram 1 9*


![Method Diagram 1 10](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_10.png)
*Figure: Method Diagram 1 10*


![Method Diagram 1 11](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_11.png)
*Figure: Method Diagram 1 11*


![Method Diagram 1 12](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_12.png)
*Figure: Method Diagram 1 12*


![Method Diagram 1 13](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_13.png)
*Figure: Method Diagram 1 13*


![Method Diagram 1 14](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_14.png)
*Figure: Method Diagram 1 14*


![Method Diagram 1 15](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_15.png)
*Figure: Method Diagram 1 15*


![Method Diagram 1 16](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_16.png)
*Figure: Method Diagram 1 16*


![Method Diagram 1 17](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_17.png)
*Figure: Method Diagram 1 17*


![Method Diagram 1 18](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_18.png)
*Figure: Method Diagram 1 18*


![Method Diagram 1 19](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_19.png)
*Figure: Method Diagram 1 19*


![Method Diagram 1 20](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_20.png)
*Figure: Method Diagram 1 20*


![Method Diagram 1 21](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_21.png)
*Figure: Method Diagram 1 21*


![Method Diagram 1 22](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_22.png)
*Figure: Method Diagram 1 22*


![Method Diagram 1 23](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_23.png)
*Figure: Method Diagram 1 23*


![Method Diagram 1 24](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_24.png)
*Figure: Method Diagram 1 24*


![Method Diagram 1 25](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_25.png)
*Figure: Method Diagram 1 25*


![Method Diagram 1 26](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_26.png)
*Figure: Method Diagram 1 26*


![Method Diagram 1 27](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_27.png)
*Figure: Method Diagram 1 27*


![Method Diagram 1 28](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_28.png)
*Figure: Method Diagram 1 28*


![Method Diagram 1 29](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_29.png)
*Figure: Method Diagram 1 29*


![Method Diagram 1 30](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_30.png)
*Figure: Method Diagram 1 30*


![Method Diagram 1 31](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_31.png)
*Figure: Method Diagram 1 31*


![Method Diagram 1 32](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_32.png)
*Figure: Method Diagram 1 32*


![Method Diagram 1 33](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_33.png)
*Figure: Method Diagram 1 33*


![Method Diagram 1 34](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_34.png)
*Figure: Method Diagram 1 34*


![Method Diagram 1 35](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/method_diagram_1_35.png)
*Figure: Method Diagram 1 35*


![Results Table 7 2](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_2.png)
*Figure: Results Table 7 2*


![Results Table 7 3](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_3.png)
*Figure: Results Table 7 3*


![Results Table 7 4](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_4.png)
*Figure: Results Table 7 4*


![Results Table 7 5](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_5.png)
*Figure: Results Table 7 5*


![Results Table 7 6](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_6.png)
*Figure: Results Table 7 6*


![Results Table 7 7](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_7.png)
*Figure: Results Table 7 7*


![Results Table 7 8](/assets/images/paper/voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis/results_table_7_8.png)
*Figure: Results Table 7 8*