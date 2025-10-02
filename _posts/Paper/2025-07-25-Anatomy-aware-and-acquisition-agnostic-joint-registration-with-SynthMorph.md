---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: SynthMorph는 합성 레이블 데이터로 학습한 공동 affine·deformable 등록기이며, 다기관 MRI에서 기존 도구보다
  최대 4.5% 높은 Dice와 1분 미만의 추론 시간을 제공합니다.
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- Medical Imaging
- Image Registration
- Deep Learning
- Neuroscience
- MRI
title: Anatomy-aware and acquisition-agnostic joint registration with SynthMorph
toc: true
toc_sticky: true
---
# Anatomy-aware and acquisition-agnostic joint registration with SynthMorph

## 0. 체크리스트
- [x] `categories`가 `medical-ai`로 설정돼 있나요?
- [x] `excerpt`에 핵심 성능 차이를 명시했나요?
- [x] 각 섹션을 실제 연구 내용으로 채웠나요?
- [x] 결과 표를 필수 지표로만 구성했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- SynthMorph는 레이블 지도로 합성한 다중 대비 MRI를 이용해 학습한, brain anatomy-aware 공동 affine·deformable 등록 네트워크입니다.
- 다기관 테스트에서 대조군 NiftyReg 대비 최대 ΔDice +4.5pt, 임상 glioblastoma 세트에서도 모든 메서드를 앞서며, V100 GPU 기준 1분 미만(affine CPU 72 s)으로 추론합니다.
- label 기반 손실과 hypernetwork형 deformable 모듈로 대비·해상도 변화, skull 유무, slice 두께 변화에 강건함을 보입니다.

## 2. 배경 & 동기
- 전통적 deformable 등록은 쌍마다 최적화가 필요해 한 건당 수 분~수십 분이 걸리며, 대비·해상도가 달라지면 성능 저하가 큽니다.
- 기존 딥러닝 등록기는 특정 모달리티에 overfit되거나 skull 제거 등 복잡한 전처리가 필요합니다.
- SynthMorph는 anatomy-aware 라벨 손실과 대규모 합성 data augmentation으로 도메인 차이를 극복하고, 단일 모델로 다양한 MRI 프로토콜에 대응합니다.

## 3. 방법론
### 3.1 전체 구조
- 두 입력 볼륨을 concat하여 affine U-Net이 12-DOF 매트릭스를 추정하고, 이어 hypernetwork가 변형 필드를 생성해 diffeomorphic deformable 등록을 수행합니다.
- 각 모듈은 symmetry를 보장하도록 쌍방향 입력을 동시에 처리하며, 최종 변형은 사용자 설정 regularization λ에 따라 통제됩니다.
- 파이프라인은 skull stripping 없이도 동작하도록 brain mask를 예측하지 않고도 anatomy-aware 손실을 적용합니다.

### 3.2 핵심 기법
- **레이블 기반 합성 학습**: 다양한 공공 세트의 분할 라벨을 활용해 contrast, 노이즈, 해상도를 무작위로 바꾼 페어를 생성해 도메인 불변 특성을 학습합니다.
- **Hypernetwork deformable 모듈**: λ를 입력으로 받아 변형 필드의 매끄러움을 조절하며, 적용 시점에 원하는 regularity(λ=0.1~0.6)를 선택할 수 있습니다.
- **Anatomy-aware loss**: 21개 주요 뇌 구조 Dice와 10개 소구조 Dice를 직접 최적화해 registration이 필요한 부위를 강조하고, 이미지 기반 보조 손실(MIND-MSE)을 병합합니다.

### 3.3 학습 및 구현 세부
- SynthMorph는 17개 공개 세트(ADNI, HCP-D, IXI 등)에서 추출한 라벨을 기반으로 3D 합성 MRI를 생성하며, translation ±30 mm, rotation ±45°, scale 90–110% 등 광범위한 변환을 랜덤 적용합니다.
- Adam 옵티마이저와 cosine 스케줄을 사용하고, mixed precision을 통해 단일 GPU에서 하루 내 학습이 가능하도록 설계되었습니다.
- 모델과 추론 유틸리티는 FreeSurfer에 포함돼 있으며, https://w3id.org/synthmorph 에서 사전학습 가중치를 제공받을 수 있습니다.

## 4. 실험 & 결과
### 4.1 설정
- **Affine 평가**: ADNI, MASI, IXI, GSP, glioblastoma 등 9개 세트를 대상으로 skull-stripped 이미지에서 정확도(Dice, NCC)를 비교.
- **Deformable 평가**: 같은 세트에 대해 NiftyReg affine 초기화 후 deformable 단계의 Dice·MIND-MSE·NCC를 측정하고, 각 도구의 end-to-end 수행까지 분석.
- **Robustness 테스트**: slice 두께 Δz 1–10 mm, skull 제거 여부, λ 조절, inverse consistency 등을 조사했습니다.

### 4.2 주요 결과표
| 시나리오 | 지표 | 최고 기존 도구 | SynthMorph | 비고 |
| --- | --- | --- | --- | --- |
| Affine 전체(large-21) | Dice | NiftyReg +0.0 | **Δ +3.7 pt 이상** | Glioblastoma 세트에서 Δ +4.5pt (p<1e-8) |
| Deformable (λ=0.5) | Dice | ANTs/NiftyReg | **동률~우위** | MIND-MSE/NCC 모든 세트에서 최저/최고 |
| 런타임 (CPU 1스레드) | 초 | Deeds 142.8, ANTs 777.8 | **Affine 72.4 ±0.8**, Deformable 887.4 ±2.5 | V100 GPU 추론 < 1분 |

### 4.3 추가 분석
- skull 미제거 시 classical 도구는 Dice가 최대 8% 감소했지만 SynthMorph는 0.05% 감소에 그쳤습니다.
- slice 두께 10 mm에서도 Dice 99% 수준을 유지하며, Deeds는 95% 미만으로 급락해 두께 변화에 취약함이 드러났습니다.
- λ를 0.45 이상으로 설정하면 folding voxel 비율이 0%가 되며 log-Jacobian spread도 baselines 대비 최소였습니다.

## 5. 의의 & 한계
- 합성 데이터 기반 학습으로 contrast·해상도 변화에 강건한 첫 통합 affine+deformable DL 등록기로, 임상 적용 시 반복 최적화 시간을 획기적으로 절감합니다.
- anatomy-aware Dice 최적화로 skull stripping 없이도 안정적인 brain alignment가 가능하다는 점이 큰 실용적 장점입니다.
- 다만 훈련에 고품질 라벨이 필요하고, 현재는 뇌 구조에 특화돼 다른 장기나 병리(예: 대형 종양)에는 추가 합성이 요구됩니다.

## 6. 개인 평가
**강점**: 합성 데이터와 hypernetwork 조합으로 범용성·추론 속도·정확도를 동시에 달성했고, 공개된 툴체인이 실무 적용을 쉽게 합니다.  
**약점**: 라벨 기반 합성 파이프라인이 복잡해 다른 장기에 확장할 때 준비 비용이 큽니다.  
**적용 가능성**: 다기관 뇌 MRI 정렬, longitudinal 추적, 대규모 atlas 구축 등 고속 정합이 필요한 프로젝트에 적합합니다.  
**추천도**: ★★★★★ (임상/연구용 등록 파이프라인을 현대화하려는 팀에 최우선 추천)

## 7. 참고 자료
- 원문: [Anatomy-aware and acquisition-agnostic joint registration with SynthMorph](https://arxiv.org/abs/2301.11329)
- 코드 및 유틸: [SynthMorph GitHub](https://github.com/voxelmorph/voxelmorph/tree/master/synthmorph)
- 데이터: [OASIS](https://www.oasis-brains.org/), [ABCD](https://abcdstudy.org/), [HCP-D](https://www.humanconnectome.org/study/hcp-lifespan-development), [MASi](https://masi.vanderbilt.edu/)
