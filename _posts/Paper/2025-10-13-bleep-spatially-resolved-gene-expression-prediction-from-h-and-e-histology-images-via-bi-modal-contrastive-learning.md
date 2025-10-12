---
categories:
- paper
- medical-ai
date: 2025-10-13
excerpt: 'BLEEP은 H&E–전사체 쌍으로 학습한 비모달 대조학습 임베딩으로 공간 유전자 발현을 예측, 도메인 일반화를 강화합니다.'
header: {}
last_modified_at: '2025-10-13'
published: true
tags:
- Spatial Transcriptomics
- Contrastive Learning
- Histology
- Gene Expression Prediction
title: 'BLEEP: Spatially Resolved Gene Expression Prediction from H&E via Bi-modal Contrastive Learning'
toc: true
toc_sticky: true
---
# BLEEP: Spatially Resolved Gene Expression Prediction from H&E via Bi-modal Contrastive Learning

## 0. 핵심 요약 (3문장 이하)
- BLEEP은 H&E 패치와 대응 spot 발현을 쌍으로 하여, 양 모달리티를 공통 임베딩 공간에 정렬하는 비모달 대조학습을 수행합니다.
- 추론 시 쿼리 이미지 패치의 발현을 레퍼런스 임베딩에서 근접 이웃/회귀 방식으로 추정해, 적은 라벨·도메인 변화에 견고합니다.
- 10x Visium 등 데이터셋에서 기존 지도학습 대비 유의한 상관/MAE 개선을 보고합니다.

## 1. 배경 & 동기
- 지도학습 기반 회귀는 데이터/도메인 변화에 취약하고, 다기관·다조직 일반화가 어렵습니다.
- 대조학습으로 모달리티 간 구조를 학습하면, 레퍼런스-쿼리 전환이 자연스럽고 전이/활용성이 높아집니다.

## 2. 방법론(개요)
- 학습: (이미지, 발현) 페어 → 비모달 대조학습(InfoNCE 유사)으로 공통 공간 정렬
- 추론: 쿼리 이미지 임베딩 → 레퍼런스 발현 임베딩 근접 이웃 회귀(kNN/가중합)
- 선택적: 유전자별 가중, 경로/표현형 조건부 임베딩 확장

## 3. 실험 & 결과(개요)
- 데이터: 인체 장기(간 등) 10x Visium
- 정량: 유전자/spot 상관·MAE 개선, 레퍼런스 크기 변화에의 민감도 분석
- 정성: 조직 구조 경계에서 예측 발현의 공간 연속성 개선

## 4. 의의 & 한계
- 의의: 라벨 효율·일반화 강화, 레퍼런스 확장으로 점진적 성능 개선 가능
- 한계: 레퍼런스 의존, 드문 유전자/도메인 편차에 대한 성능 하락 가능

## 5. 개인 평가
- 적용성: 다기관 병리–ST 매핑에 적합, 확장성 높음
- 보완점: 레퍼런스 선택/가중 전략, 불확실성/신뢰도 점수화 필요

## 6. 참고 자료
- arXiv: https://arxiv.org/abs/2306.01859
- PDF: https://arxiv.org/pdf/2306.01859.pdf

## 작성 체크리스트
- [ ] 데이터/지표 수치 인용 보강
- [ ] 주요 도식/표 추출 및 삽입 확인(자동 주입기 사용 가능)
- [ ] 링크/인용 형식 점검, 로컬 빌드 확인


## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/BLEEP_Spatially_Resolved_Gene_Expression_Prediction_2306.01859/fig_04.png)
캡션: For BLEEP, we use the pretrained ResNet50[(<>)10] as the image encoder and a fully connected network (FCN) with an output dimension of 256 as the expression encoder, which doubles as a projection head. The image features from the image encoder are passed through a separate projection head to bring the two modalities to the same dimension before applying the contrastive loss similar to CLIP[(<>)15]…

### Main Results Table
![Results](/assets/images/paper/BLEEP_Spatially_Resolved_Gene_Expression_Prediction_2306.01859/table_01.png)
캡션: Table 2: Predicted gene expression values with top 5 correlations with original profile for each method from one representative replicate.

## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build

