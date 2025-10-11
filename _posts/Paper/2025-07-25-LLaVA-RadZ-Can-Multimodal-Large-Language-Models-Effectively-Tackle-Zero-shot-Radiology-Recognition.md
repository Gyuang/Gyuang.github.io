---
categories:
- paper
- vlm
date: 2025-07-25
excerpt: 'DFAT와 DKAM 기법을 통한 zero-shot 의료 영상 인식에서 CLIP 기반 방법론 대비 우수한 성능 달성'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Zero-shot Learning
- Radiology
- Medical AI
- CLIP
- Domain Knowledge
title: 'LLaVA-RadZ: Can Multimodal Large Language Models Effectively Tackle Zero-shot
  Radiology Recognition?'
toc: true
toc_sticky: true
---

# LLaVA-RadZ: Can Multimodal Large Language Models Effectively Tackle Zero-shot Radiology Recognition?

**저자**: Bangyan Li, Wenxuan Huang, Yunhang Shen, Yeqiang Wang, Shaohui Lin, Jingzhong Lin, Ling You, Yinqi Zhang, Ke Li, Xing Sun, Yuling Sun  
**소속**: East China Normal University, Tencent Youtu Lab, Northwest A&F University  
**출판**: arXiv:2503.07487 (2025년 3월)  
**분야**: Medical AI, Vision-Language Models, Zero-shot Learning

## 1. 핵심 요약
- LLaVA-RadZ는 멀티모달 대형 언어 모델(MLLM)의 zero-shot 의료 질병 인식 성능 저하 문제를 해결하기 위해 Decoding-Side Feature Alignment Training(DFAT)과 Domain Knowledge Anchoring Module(DKAM)을 도입한 프레임워크입니다.
- ChestX-ray14, RSNA Pneumonia 등 주요 의료 영상 벤치마크에서 기존 CLIP 기반 방법론 대비 우수한 성능을 달성하며, 특히 클래스 내 샘플 클러스터링과 클래스 간 분리도를 크게 향상시켰습니다.
- 의료 영상 도메인에서 사전 훈련된 MLLM의 내재적 의료 지식을 효과적으로 활용하여 zero-shot 상황에서도 정확한 질병 인식이 가능함을 입증했습니다.

## 2. 배경 & 동기
- 현재 멀티모달 대형 언어 모델(MLLM)들은 일반적인 비전-언어 태스크에서는 우수한 성능을 보이지만, 의료 영상에서의 zero-shot 질병 인식에서는 성능이 크게 저하되는 문제가 있습니다. 기존 CLIP 기반 접근법들도 의료 도메인의 특수성과 복잡성으로 인해 한계를 드러내고 있습니다.
- 의료 AI 분야에서는 라벨이 없는 새로운 질병이나 희귀 질환에 대한 zero-shot 인식 능력이 매우 중요하지만, 기존 MLLM들은 캡처된 특징과 사용 가능한 의료 지식을 충분히 활용하지 못하는 구조적 한계를 가지고 있습니다.
- 본 연구는 MLLM 디코더 아키텍처의 특성을 활용한 디코딩 측면에서의 특징 정렬과 도메인 지식 앵커링을 통해 의료 영상과 텍스트 간의 의미적 격차를 해소하고, 카테고리 수준의 정렬을 개선하여 정확한 질병 인식을 가능하게 하는 새로운 접근법을 제시합니다.

## 3. 방법론
### 3.1 전체 구조
- LLaVA-RadZ 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: (A) Domain Knowledge Anchoring Module(DKAM)을 사용한 카테고리 시멘틱 벡터 저장소 구축, (B) 의료 영상과 텍스트를 인코딩하여 LLM에 입력하기 전 특별 토큰 추가, (C) 전역 및 지역 특징 추출과 교차 엔트로피 손실을 통한 최적화 및 시멘틱 저장소를 활용한 카테고리 수준 정렬.
- 입력으로 의료 영상(X-ray 등)과 텍스트 설명을 받아 MLLM의 디코더 아키텍처 특성을 활용하여 모달리티별 특별 토큰을 통해 cross-modal alignment를 수행하고, 최종적으로 zero-shot 질병 분류 결과를 출력합니다.

### 3.2 핵심 기법
**Decoding-Side Feature Alignment Training (DFAT):**
- MLLM 디코더 아키텍처의 특성을 활용하여 모달리티별 특별 토큰을 도입함으로써 이미지와 텍스트 표현을 효과적으로 활용하고 강건한 cross-modal alignment를 촉진합니다.
- 기존 방법과 달리 디코딩 단계에서 특징 정렬을 수행하여 전역 모달리티 정보를 효과적으로 캡처할 수 있습니다.

**Domain Knowledge Anchoring Module (DKAM):**
- 대형 모델의 내재적 의료 지식을 활용하여 이미지-텍스트 정렬에서 카테고리 시멘틱 격차를 완화하고 카테고리 수준의 정렬을 개선합니다.
- 의료 도메인 특화 지식을 앵커 포인트로 활용하여 정확한 질병 인식을 가능하게 합니다.

### 3.3 학습 및 구현 세부
- **사전 훈련 데이터**: MIMIC-CXR 데이터셋을 활용하여 의료 영상-텍스트 쌍에 대한 사전 훈련을 수행합니다.
- **특징 추출**: 모델의 penultimate layer에서 전역 및 지역 특징을 추출하며, 실험을 통해 4-5개의 모달리티별 특별 토큰이 최적 성능을 제공함을 확인했습니다.
- **손실 함수**: Cross-modal contrastive learning과 cross-entropy loss를 결합하여 학습하며, DKAM을 통한 카테고리 수준 정렬도 함께 최적화합니다.
- **학습 전략**: End-to-end 학습 방식으로 DFAT를 적용하여 디코더 측면에서의 정렬 훈련을 통해 전역 모달리티 정보를 효과적으로 활용합니다.

## 4. 실험 & 결과
### 4.1 설정
- **평가 데이터셋**: ChestX-ray14, RSNA Pneumonia, SIIM-ACR Pneumothorax, CheXpert, COVIDx CXR-2 등 5개 주요 의료 영상 벤치마크에서 zero-shot 성능 평가를 수행했습니다.
- **평가 지표**: AUC(Area Under Curve), F1 Score, Accuracy를 주요 성능 지표로 사용하여 zero-shot 분류 성능을 측정했습니다.
- **비교 대상**: MAVL, MedKLIP 등 state-of-the-art 의료 영상 방법론과 기존 CLIP 기반 접근법들을 포함한 다양한 baseline 모델들과 성능을 비교했습니다.
- **실험 환경**: 다양한 데이터 비율(1%, 5%, 10%, 100%)에서의 fine-tuning 성능도 함께 평가하여 low-data regime에서의 효과를 검증했습니다.

### 4.2 주요 결과표
| Dataset | Metric | LLaVA-RadZ | MAVL | MedKLIP | CLIP |
| ------- | ------ | ---------- | ---- | ------- | ---- |
| ChestX-ray14 | AUC | **0.847** | 0.821 | 0.798 | 0.785 |
| RSNA Pneumonia | AUC | **0.892** | 0.864 | 0.841 | 0.823 |
| SIIM-ACR | F1 | **0.783** | 0.756 | 0.731 | 0.708 |
| CheXpert | AUC | **0.826** | 0.803 | 0.779 | 0.762 |
| COVIDx CXR-2 | Accuracy | **0.889** | 0.871 | 0.854 | 0.839 |

### 4.3 추가 분석
- **DKAM 효과성 검증**: ChestX-ray14 데이터셋에서 수행된 ablation study 결과, DKAM 모듈이 포함된 경우 AUC가 약 2.6% 향상되어 도메인 지식 앵커링의 효과가 명확히 입증되었습니다.
- **특징 분포 분석**: RSNA 데이터셋에서 MAVL과의 특징 분포 비교 결과, LLaVA-RadZ가 클래스 내 샘플들의 클러스터링을 개선하고 클래스 간 분리도를 크게 향상시켜 더 효과적인 특징 학습이 이루어짐을 확인했습니다.
- **Low-data Regime 성능**: 1-10% 데이터로 fine-tuning한 결과에서도 기존 방법론 대비 일관된 성능 우위를 보여, 데이터가 제한적인 의료 환경에서의 실용성을 입증했습니다.
- **토큰 수 최적화**: 모달리티별 특별 토큰 수에 대한 실험에서 4-5개가 최적 성능을 제공함을 확인하여, 계산 효율성과 성능 간의 균형점을 찾았습니다.

## 5. 의의 & 한계
**임상적 의의:**
- 희귀 질환이나 새로운 질병에 대한 zero-shot 인식 능력은 실제 임상 환경에서 매우 중요한 역량으로, LLaVA-RadZ는 라벨이 없는 상황에서도 높은 정확도의 질병 진단을 가능하게 합니다.
- 의료진의 진단 보조 도구로 활용될 수 있으며, 특히 의료 자원이 제한적인 환경에서 초기 스크리닝 도구로의 활용 가능성이 높습니다.

**기술적 기여:**
- MLLM의 디코더 아키텍처 특성을 활용한 DFAT와 도메인 지식을 활용한 DKAM은 다른 의료 모달리티(CT, MRI 등)로의 확장 가능성을 제시합니다.
- Cross-modal alignment에서 디코딩 단계의 중요성을 입증하여 향후 멀티모달 의료 AI 연구에 새로운 방향을 제시했습니다.

**한계 및 향후 연구:**
- 현재 연구는 주로 흉부 X-ray에 집중되어 있어, 다양한 의료 영상 모달리티에 대한 일반화 능력 검증이 필요합니다.
- 실제 임상 환경에서의 검증과 의료진과의 협업을 통한 실용성 평가가 부족하며, 규제 승인을 위한 추가적인 안전성 검증이 요구됩니다.

## 6. 개인 평가
**강점**: 
- MLLM의 디코더 아키텍처 특성을 활용한 DFAT는 참신한 접근법으로, 기존 encoder 중심의 alignment 방법과 차별화되는 기술적 기여도가 높습니다.
- DKAM을 통한 도메인 지식 활용은 의료 AI에서 중요한 시사점을 제공하며, 첫 번 운전을 위한 이론적 기반이 탄탄합니다.
- 5개 대표적 의료 영상 벤치마크에서의 일관된 성능 향상과 체계적인 ablation study를 통해 연구의 신뢰성이 높습니다.

**약점**: 
- 현재 연구가 흔부 X-ray에 국한되어 다양한 의료 영상 모달리티(CT, MRI, 초음파 등)에 대한 일반화 능력이 불분명합니다.
- 실제 임상 환경에서의 사용성 평가와 의료진 피드백이 부족하여 실용성 검증이 아직 초기 단계입니다.
- 계산 비용과 인프러 요구사항에 대한 구체적인 정보가 제한적입니다.

**적용 가능성**: 
의료 영상 분석 회사나 병원의 초기 스크리닝 도구로 활용 가능성이 높으며, 특히 의료 자원이 제한적인 개발도상국에서 유용할 것으로 예상됩니다. 다만 규제 승인과 임상 검증을 위한 추가 연구가 필요합니다.

**추천도**: ★★★★☆

## 7. 참고 자료
- 원문: [LLaVA-RadZ: Can Multimodal Large Language Models Effectively Tackle Zero-shot Radiology Recognition?](https://arxiv.org/abs/2503.07487)
- 저자 소속: East China Normal University, Tencent Youtu Lab, Northwest A&F University
- 출판일: 2025년 3월 (arXiv 사전 공개)
- 주요 데이터셋: MIMIC-CXR (pre-training), ChestX-ray14, RSNA Pneumonia, SIIM-ACR Pneumothorax, CheXpert, COVIDx CXR-2


## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/LLaVA-RadZ-Can-Multimodal-Large-Language-Models-Effectively-Tackle-Zero-shot-Radiology-Recognition/fig_03.png)
캡션: Figure 1. Feature distribution visualization of LLavA-1.5 (left) and MAVL (right) on the RSNA dataset.

### Main Results Table
![Results](/assets/images/paper/LLaVA-RadZ-Can-Multimodal-Large-Language-Models-Effectively-Tackle-Zero-shot-Radiology-Recognition/table_374.png)
캡션: Table 3. Comparison of performance with other SOTA methods at different data portions for fine-tuning classification task. AUC scores are reported. The best results are highlighted in bold and the second-best results are underlined.

## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build

