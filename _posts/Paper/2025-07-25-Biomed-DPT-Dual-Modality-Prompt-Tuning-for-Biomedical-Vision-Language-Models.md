---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: Biomed-DPT는 임상·LLM 이중 텍스트 프롬프트와 시각 soft prompt를 결합해 11개 의료 이미지 데이터에서 평균 정확도
  66.14%, base 78.06%, novel 75.97%를 달성하며 CoOp 대비 최대 8.04pt 향상합니다.
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- VLM
- Prompt Learning
- Medical AI
- Few-shot Learning
- Knowledge Distillation
title: 'Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models'
toc: true
toc_sticky: true
---
# Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models

## 0. 체크리스트
- [x] `categories`를 `medical-ai`로 설정했나요?
- [x] `excerpt`에 핵심 수치를 담았나요?
- [x] 모든 섹션을 실제 내용으로 채웠나요?
- [x] 결과 표는 핵심 지표 위주로 정리했나요?
- [x] 참고 링크를 기재했나요?

## 1. 핵심 요약 (3문장 이하)
- Biomed-DPT는 임상 지식이 녹아든 이중 텍스트 프롬프트와 시각 soft prompt를 함께 학습하는 듀얼 모달 프롬프트 튜닝 기법입니다.
- 11개 의료 이미지 데이터셋에서 평균 정확도 66.14%(base 78.06%, novel 75.97%)로 CoOp 대비 각각 +6.20, +3.78, +8.04pt 향상했습니다.
- 시각 soft prompt로 비진단 영역 주의를 억제하고, LLM 기반 도메인 문장을 distillation해 자연어와 시각 특징 간 불일치를 줄였습니다.

## 2. 배경 & 동기
- 자연 이미지로 사전학습한 VLM은 의료 영상의 대비/구조에 맞지 않고, “a photo of a [CLASS]”와 같은 단순 텍스트 프롬프트가 임상 표현을 반영하지 못합니다.
- 기존 프롬프트 학습은 텍스트 또는 비전 한쪽에 치우쳐 이중 모달 적응이 어려웠으며, 시각 프롬프트는 비진단 영역에 주의를 분산시켰습니다.
- Biomed-DPT는 dual prompt 설계를 통해 의료 영역의 전문 지식을 프롬프트에 내재화하고, 시각 주의를 재조정해 도메인 갭을 해소합니다.

## 3. 방법론
### 3.1 전체 구조
- BiomedCLIP 텍스트/비전 인코더를 고정하고, learnable 텍스트 컨텍스트와 시각 soft prompt(0 벡터 삽입)를 추가합니다.
- 텍스트 프롬프트는 템플릿 기반 임상 문구 + LLM 생성 도메인 문장을 결합해 dual context를 형성합니다.
- 시각 프롬프트는 attention re-weighting으로 비진단 영역 가중치를 낮춰, 핵심 병변을 강조합니다.

### 3.2 핵심 기법
- **LLM-Driven Domain Prompt**: GPT 계열 LLM으로 역량 정의서를 생성하고, 이를 지식 distillation으로 템플릿 문구에 주입합니다.
- **Zero Vector Vision Prompt**: transformer 입력 토큰에 zero soft prompt를 추가해 self-attention을 재조정, 배경 노이즈를 억제합니다.
- **Knowledge Distillation Loss**: LLM 문장의 잠재 표현을 템플릿 프롬프트에 전파해, 양쪽 프롬프트의 의미 일관성을 확보합니다.

### 3.3 학습 및 구현 세부
- 학습은 K-shot(1,2,4,8,16) few-shot과 base-to-novel 설정으로 진행하며, BiomedCLIP 사전학습 체크포인트를 사용합니다.
- Biomed-DPT는 텍스트 prompt 길이 16, 시각 prompt 길이 1의 최소 가중치만 업데이트해 전체 파라미터 대비 극소량만 학습합니다.
- 공개 코드와 하이퍼파라미터(λ1, λ2)는 GitHub에 제공되어 재현이 가능합니다.

## 4. 실험 & 결과
### 4.1 설정
- **데이터셋**: BT-MRI, BUSI, CHMNIST, COVID-QU-Ex, CTKidney, DermaMNIST, KneeXray, Kvasir, LC25000, OCT, Retina 등 11개.
- **벤치마크**: few-shot K=1~16, base-to-novel split, zero-shot BiomedCLIP, CoOp, VPT, Maple 대비.
- **지표**: 분류 정확도(%) 평균, base 클래스, novel 클래스.

### 4.2 주요 결과표
| 설정 | Biomed-DPT 정확도 | CoOp 대비 향상 |
| --- | --- | --- |
| 평균 (11개) | **66.14%** | +6.20pt |
| Base 클래스 | **78.06%** | +3.78pt |
| Novel 클래스 | **75.97%** | +8.04pt |
| K=1 few-shot | 59.03% | +8.85pt vs CoOp |

### 4.3 추가 분석
- visual-only VPT-s는 16-shot에서도 24.31%에 그쳤지만, Biomed-DPT 시각 프롬프트는 CoOp+VPT 대비 +35pt 이상 향상했습니다.
- 컨텍스트 삽입 위치는 텍스트 끝(end)이 가장 효과적이며, biomedical-specific context가 다른 컨텍스트보다 높은 정확도를 제공했습니다.
- 학습된 컨텍스트 토큰은 “histopathological”, “endoscopic” 등 모달리티 관련 용어와 가장 가까운 임베딩을 가져, 의미적으로 해석 가능합니다.

## 5. 의의 & 한계
- 의료 이미지 특화 듀얼 프롬프트 설계로 textual-only 대비 추가적인 정확도 향상을 입증했습니다.
- 시각 soft prompt가 attention을 병변 중심으로 재배치해 해석 가능성을 높였습니다.
- 다만 LLM 생성 문구 품질에 성능이 좌우되며, 다기관·다언어 보고서로 확장하려면 추가 정제가 필요합니다.

## 6. 개인 평가
**강점**: 최소 파라미터 업데이트로 큰 성능 향상을 얻어 임상 데이터가 적은 환경에 적합하고, 텍스트·비전을 동시에 다루는 구조가 설득력 있습니다.  
**약점**: LLM 기반 문구의 편향이나 오류가 고스란히 프롬프트에 반영될 수 있고, 시각 prompt 길이에 민감합니다.  
**적용 가능성**: 이미 구축된 BiomedCLIP/PMC-CLIP 다운스트림 분류기를 few-shot으로 빠르게 튜닝하는 데 바로 활용할 수 있습니다.  
**추천도**: ★★★★☆ (의료 VLM을 데이터 효율적으로 적응시키고자 하는 팀에 추천)

## 7. 참고 자료
- 원문: [Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models](https://arxiv.org/abs/2505.05189)
- 코드: [GitHub - Kanyooo/Biomed-DPT](https://github.com/Kanyooo/Biomed-DPT)
- 사전학습 모델: [BiomedCLIP Checkpoints](https://huggingface.co/microsoft/biomedclip)


## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/Biomed-DPT-Dual-Modality-Prompt-Tuning-for-Biomedical-Vision-Language-Models/fig_12.png)
캡션: Figure 4: Overview of the Biomed-DPT framework, which combines LLM prompt generation, fixed clinical prompt templates, learnable context, zero vector as a soft prompt, and BiomedCLIP to construct a unified multimodal representation space. In this method, the prompt ensemble strategy is used to integrate text and image features, and the cross-entropy, L1 constraint, and KL divergence loss are minim…

### Main Results Table
![Results](/assets/images/paper/Biomed-DPT-Dual-Modality-Prompt-Tuning-for-Biomedical-Vision-Language-Models/table_01.png)
캡션: Table 1: The average classification accuracy (%) obtained from 5 benchmarks, where s indicates the introduction of only one learnable parameter layer. (w) denotes with interaction, and (w/o) denotes without interaction.

## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build

