---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: 'Brain-Adapter는 4,661 AD / 5,025 CN / 7,111 MCI MRI와 임상 리포트를 결합해 ADNI 다중모달 분류에서 매크로 F1 0.90, 가중 F1 0.91을 달성하며 3D DenseNet·ResNet 대비 최대 0.29 F1 향상을 기록합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- Multimodal LLM
- Adapter Tuning
- Alzheimer’s Disease
- MRI
- Clinical Reports
title: 'Brain-Adapter: Enhancing Neurological Disorder Analysis with Adapter-Tuning Multimodal Large Language Models'
toc: true
toc_sticky: true
---
# Brain-Adapter: Enhancing Neurological Disorder Analysis with Adapter-Tuning Multimodal Large Language Models

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `medical-ai`로 맞춰져 있나요?
- [x] `excerpt`에 핵심 정량 결과를 포함했나요?
- [x] 모든 섹션을 실제 논문 내용으로 채웠나요?
- [x] 결과 표를 핵심 지표로 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- Brain-Adapter는 3D MRI와 임상 보고서를 동시에 다루는 M3D 기반 멀티모달 LLM에 경량 어댑터를 삽입해 추가 지식을 주입하는 방법입니다.
- 선형 투영층만 미세 조정했을 때도 AD/NC/MCI 분류에서 매크로 F1 0.75를 달성했고, 이미지와 텍스트를 함께 사용할 경우 매크로 F1 0.90, 가중 F1 0.91로 3D DenseNet 대비 0.18~0.19pt 향상했습니다.
- 1천만 개 미만의 파라미터만 업데이트해 9에폭(A6000 GPU) 미세 조정으로 기존 단일 모달 CNN보다 정확한 조기 진단을 제시합니다.

## 2. 배경 & 동기
- 알츠하이머 진단에는 MRI·PET과 임상 메모, 인지 검사가 통합적으로 사용되지만 기존 CAD는 2D 이미지 또는 단일 모달에 치중했습니다.
- 3D 모델을 처음부터 학습하려면 연산 비용과 데이터 확보 부담이 커 실제 병원 적용이 어렵습니다.
- 저자들은 사전학습된 멀티모달 LLM(M3D+LLaMA2)을 유지한 채 가벼운 어댑터만 학습해 도메인 지식을 주입, 데이터 효율과 일반성을 동시에 확보하고자 했습니다.

## 3. 방법론
### 3.1 전체 구조
- Brain-Adapter는 3D CNN 기반 병목(residual) 블록으로 256³ MRI를 3D ViT 입력 크기(4×16×16 패치)로 압축한 뒤, M3D 비전 인코더와 LLaMA2-7B 텍스트 인코더를 통과시킵니다.
- 어댑터 출력과 임상 리포트 임베딩은 공통 잠재 공간에서 대비 학습되며, 선형 투영층을 통해 분류 헤드에 결합됩니다.
- 병렬로 영상·텍스트를 처리해 생성된 임베딩은 단일 레이어 MLP 분류기로 AD/NC/MCI 확률을 산출합니다.

### 3.2 핵심 기법
- **Adapter Bottleneck**: 3D 합성곱과 잔차 연결만 학습해 전체 파라미터를 고정하면서 도메인 특화 특징을 주입합니다.
- **대비 손실**: 이미지-텍스트 간 양방향 InfoNCE 손실로 임상 리포트와 영상 슬라이스를 정렬합니다.
- **소량 학습**: 70/30으로 나눈 ADNI 데이터에 9에폭을 학습, 적은 업데이트로도 비선형 차이를 포착하도록 합니다.

### 3.3 학습 및 구현 세부
- ADNI에서 AD 4,661건, CN 5,025건, MCI 7,111건을 수집하고, N4 보정·MNI 정합·배경 크롭·1mm 이방 리샘플링 후 256×256×256으로 통일했습니다.
- 어댑터는 AdamW(1e-3), 투영층은 1e-4로 학습했으며 배치 8, 9에폭 구성입니다.
- 다섯 가지 전처리 버전(MPR, GradWarp 등)을 모두 활용해 도메인 편향을 줄였습니다.

## 4. 실험 & 결과
### 4.1 설정
- 베이스라인: 3D ResNet-50, 3D DenseNet-121(이미지 단독), M3D 고정(FPM)과 선형 투영만 학습(TLP) 두 가지 설정.
- 평가: 정밀도(PRE), 재현율(SEN), F1을 각 클래스별·평균(Macro/Weighted)으로 보고했습니다.
- 하드웨어: NVIDIA A6000 1장 학습, 테스트는 동일 환경에서 수행했습니다.

### 4.2 주요 결과표
| 모델 | 모달리티 | AD F1 | CN F1 | MCI F1 | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3D ResNet-50 | MRI | 0.66 | 0.64 | 0.61 | 0.64 | 0.63 |
| 3D DenseNet-121 | MRI | 0.73 | 0.71 | 0.72 | 0.72 | 0.72 |
| Brain-Adapter (FPM) | MRI | 0.43 | 0.38 | 0.56 | 0.46 | 0.47 |
| **Brain-Adapter (TLP)** | MRI+Report | **0.75** | **0.73** | **0.77** | **0.90** | **0.91** |

### 4.3 추가 분석
- 단일 MRI만 사용할 때도 선형 투영층만 미세 조정하면 DenseNet보다 0.02~0.04p 높은 F1을 기록했습니다.
- 멀티모달 학습 시 매크로 SEN 0.90, PRE 0.92로 하위 단계보다 각각 0.40, 0.38p 상승했습니다.
- t-SNE 시각화에서 초기에는 CN·MCI가 혼재했으나 9에폭 이후 세 클래스가 뚜렷하게 분리돼 멀티모달 정렬 효과를 확인했습니다.

## 5. 의의 & 한계
- 수백만 파라미터만 추가 학습해도 3D 영상+임상 리포트를 아우르는 고성능 분류를 실현했습니다.
- 경량 어댑터 덕분에 메모리와 연산 비용이 적어 임상 환경에서 빠르게 업데이트 가능합니다.
- 다만 외부 벤치마크(예: OASIS)나 세분화된 임상 하위군에 대한 검증은 향후 과제로 남아 있습니다.

## 6. 개인 평가
**강점**: 사전학습 MLLM을 거의 고정하면서 도메인 지식을 주입해 데이터·연산 효율이 탁월합니다.  
**약점**: ADNI 단일 코호트에 집중돼 일반화 검증이 부족하고, multipath 어댑터 구성이 추가로 필요할 수 있습니다.  
**적용 가능성**: 병원 PACS에 이미 존재하는 MRI·임상 기록을 동시에 활용하려는 조기 진단 보조 시스템에 즉시 적용할 수 있습니다.  
**추천도**: ★★★★☆ (멀티모달 LLM을 의료 영상에 도입하려는 연구자에게 강력 추천)

## 7. 참고 자료
- 원문: [Brain-Adapter: Enhancing Neurological Disorder Analysis with Adapter-Tuning Multimodal Large Language Models](https://arxiv.org/abs/2405.05189)
- 코드: (논문 부록 제공) https://github.com/UTA-ML/Brain-Adapter
- 데이터: [ADNI](https://adni.loni.usc.edu/)
