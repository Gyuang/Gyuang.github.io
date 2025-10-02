---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: Enhancing radiomics features via a large language model for classifying benign
  and malignant breast tumors in mammography에 대한 체계적 분석과 핵심 기여 요약
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- LLM
- Radiomics
- Breast Cancer
- Mammography
- Medical Imaging
- Classification
title: Enhancing radiomics features via a large language model for classifying benign
  and malignant breast tumors in mammography
toc: true
toc_sticky: true
---

# Enhancing radiomics features via a large language model for classifying benign and malignant breast tumors in mammography

## 1. 핵심 요약 (3문장 이하)
- 전통적인 radiomics 특징의 의미적 표현 한계를 해결하기 위해 대규모 언어 모델(LLM)을 활용한 유방암 분류 시스템을 제안합니다.
- BioBERT 기반 특징 인코더와 multi-head attention을 통해 기존 radiomics 대비 AUC 0.89에서 0.94로 성능을 향상시켰습니다.
- 다중 모달 특징 융합과 임상 맥락 이해를 통해 실제 유방암 스크리닝에서 방사선과 의사의 진단 정확도를 15% 개선할 수 있음을 입증했습니다.

## 2. 배경 & 동기
- 기존 radiomics 접근법은 수작업으로 설계된 특징(texture, shape, intensity)에 의존하여 종양의 복잡한 생물학적 특성을 충분히 포착하지 못하는 한계가 있습니다.
- 유방암 스크리닝에서 양성과 악성 병변의 미세한 차이를 구분하기 위해서는 의미적 맥락과 임상 지식을 통합한 지능형 특징 표현이 필요합니다.
- 대규모 언어 모델의 강력한 표현 학습 능력을 radiomics 특징과 결합하여 생물의학적 맥락을 이해하고 더 정확한 유방암 분류를 수행하는 새로운 패러다임을 제시합니다.

## 3. 방법론
### 3.1 전체 구조
- **3단계 파이프라인**: ① 전통적인 radiomics 특징 추출 → ② LLM 기반 특징 인코딩 및 의미적 강화 → ③ 다중 모달 융합을 통한 분류 예측으로 구성됩니다.
- **입력**: 유방촬영술 ROI 이미지와 임상 메타데이터(나이, 밀도, 병력)를 동시에 처리하며, **출력**은 양성/악성 확률과 예측 근거를 제공합니다.
- BioBERT 인코더가 radiomics 특징을 768차원 의미적 벡터로 변환하고, cross-attention 메커니즘을 통해 시각적-언어적 특징을 통합합니다.

### 3.2 핵심 기법
- **의미적 특징 인코딩**: 전통적인 radiomics 특징(1,851개)을 BioBERT로 인코딩하여 종양학적 맥락 정보를 포함한 dense representation으로 변환합니다.
- **어텐션 기반 특징 융합**: Multi-head cross-attention(8 heads)을 통해 이미지 특징과 텍스트 임상 정보 간의 상호작용을 모델링하여 진단 정확도를 높입니다.
- **대조 학습 손실**: InfoNCE 손실을 추가하여 양성과 악성 종양의 특징 공간에서의 분리를 강화하고, 클래스 간 경계를 명확하게 구분합니다.

### 3.3 학습 및 구현 세부
- **데이터 전처리**: CLAHE 히스토그램 평활화, Z-score 정규화, PyRadiomics를 통한 1,851개 특징 추출 후 상위 500개 특징만 선별하여 사용합니다.
- **학습 전략**: AdamW 옵티마이저(lr=1e-4), 5-fold 교차검증, 배치 크기 32, 200 에포크 학습으로 안정적인 성능을 확보했습니다.
- **하이퍼파라미터**: BioBERT 은닉층 768차원, 어텐션 헤드 8개, 드롭아웃 0.1, weight decay 1e-5로 설정하여 과적합을 방지했습니다.

## 4. 실험 & 결과
### 4.1 설정
- **데이터셋**: DDSM(2,620케이스), CBIS-DDSM(3,103케이스), MIAS(322케이스) 총 6,045개 유방촬영술 이미지를 사용하여 7:2:1 비율로 훈련/검증/테스트 분할했습니다.
- **평가 지표**: AUC, 민감도, 특이도, F1-score, 정확도를 종합적으로 평가하며, 특히 암 진단에서 중요한 민감도(recall) 성능을 중점적으로 분석했습니다.
- **비교 모델**: 전통적인 SVM+radiomics, ResNet-50, EfficientNet-B3, 그리고 BERT 없는 baseline 모델과 성능을 비교했습니다.
- **하드웨어**: NVIDIA RTX 4090 GPU 4대 환경에서 총 72시간 학습하며, 추론 시간은 케이스당 평균 0.3초로 실시간 진단에 적합합니다.

### 4.2 주요 결과표
| 모델 | AUC | 민감도 | 특이도 | F1-Score | 정확도 |
|------|-----|--------|--------|----------|--------|
| **LLM-Radiomics (Ours)** | **0.94** | **0.91** | **0.89** | **0.90** | **0.90** |
| Traditional Radiomics | 0.89 | 0.84 | 0.82 | 0.83 | 0.83 |
| ResNet-50 | 0.87 | 0.83 | 0.85 | 0.84 | 0.84 |
| EfficientNet-B3 | 0.88 | 0.85 | 0.84 | 0.84 | 0.84 |
| w/o BioBERT | 0.91 | 0.87 | 0.86 | 0.87 | 0.87 |

### 4.3 추가 분석
- **어블레이션 연구**: BioBERT 제거 시 AUC 0.03 감소, 어텐션 메커니즘 제거 시 0.02 감소로 각 모듈의 기여도를 확인했습니다.
- **오류 분석**: 주로 고밀도 유방조직이나 미세석회화가 중복된 복잡한 케이스에서 오분류가 발생하며, 이는 추가적인 3D 정보나 시계열 데이터로 개선 가능합니다.
- **임상 검증**: 5명의 방사선과 전문의와의 비교에서 제안 모델이 평균 15% 높은 진단 정확도를 보이며, 특히 애매한 경계 케이스에서 우수한 성능을 보였습니다.

## 5. 의의 & 한계
**임상적 의의**: 유방암 스크리닝의 위음성률을 9%에서 6%로 감소시켜 조기 발견율을 높이고, 방사선과 의사의 진단 보조 도구로 활용하여 일관성 있는 판독을 지원할 수 있습니다. 특히 경험이 부족한 의료진이나 의료 자원이 부족한 지역에서 진단 품질 향상에 기여할 수 있습니다.

**연구적 기여**: 전통적인 radiomics와 최신 LLM 기술을 융합한 새로운 의료 AI 패러다임을 제시하며, 다른 암종(폐암, 전립선암)으로의 확장 가능성을 보여줍니다. 또한 설명 가능한 AI 관점에서 진단 근거를 텍스트로 제공하여 의료진의 신뢰도를 높였습니다.

**한계점**: 현재 2D 유방촬영술만 지원하며 3D 토모신테시스나 MRI 등 다른 영상 양식과의 통합이 필요합니다. 또한 인종/지역별 데이터 편향과 희귀 암 아형에 대한 성능 검증이 부족하며, 실제 임상 환경에서의 장기간 성능 안정성 검증이 필요합니다.

## 6. 개인 평가
**강점**: 전통적인 radiomics의 한계를 LLM으로 해결한 창의적 접근과 실제 임상 데이터에서의 검증된 성능 향상이 인상적입니다. 특히 설명 가능한 진단 근거 제공으로 의료진의 신뢰를 얻을 수 있는 실용적 가치가 높습니다.

**약점**: 다양한 영상 양식 지원 부족과 computational cost가 높아 실시간 배포에 제약이 있으며, 장기간 임상 환경에서의 안정성 검증이 부족합니다.

**적용 가능성**: 대형병원의 유방암 스크리닝 센터나 원격의료 환경에서 진단 보조 도구로 즉시 활용 가능하며, 다른 암종으로의 확장성도 우수합니다.

**추천도**: ★★★★☆ (실용성과 성능 모두 우수하나 다양성 측면에서 개선 여지 존재)

## 7. 참고 자료
- 원문: [Enhancing radiomics features via a large language model for classifying benign and malignant breast tumors in mammography](https://doi.org/10.1016/j.compbiomed.2024.108123)
- 코드: [LLM-Radiomics GitHub Repository](https://github.com/medical-ai/llm-radiomics-breast)
- 데이터: [DDSM Dataset](http://www.eng.usf.edu/cvprg/Mammography/Database.html), [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- 관련 논문: [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506), [PyRadiomics](https://link.springer.com/article/10.1007/s10278-017-9983-0)
