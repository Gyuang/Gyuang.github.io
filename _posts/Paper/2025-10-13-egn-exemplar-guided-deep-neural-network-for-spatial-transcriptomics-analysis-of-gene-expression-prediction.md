---
categories:
- paper
- bioinformatics
date: 2025-10-13
excerpt: 'EGN은 exemplar-guided 전략으로 공간 전사체 유전자 발현 예측을 개선하며, 확장판에서는 그래프 신경망으로 공간 관계를 강화합니다.'
header: {}
last_modified_at: '2025-10-13'
published: true
tags:
- Spatial Transcriptomics
- Exemplar
- Graph Neural Network
- Gene Expression Prediction
title: 'EGN: Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction'
toc: true
toc_sticky: true
---
# EGN: Exemplar Guided Deep Neural Network for Spatial Transcriptomics Analysis of Gene Expression Prediction

## 0. 핵심 요약 (3문장 이하)
- EGN은 유사한 패턴을 가진 exemplar를 활용해 학습/추론을 안내함으로써, 공간 전사체 유전자 발현 예측의 정확도를 향상합니다.
- 확장판에서는 그래프 신경망(GNN)을 도입해 spot 간 공간적 이웃 관계를 반영, 지역적 연속성과 전역 맥락을 동시에 모델링합니다.
- WACV 2023 본문과 bioRxiv 확장(그래프 기반)을 통해 정량·정성 개선을 보고합니다.

## 1. 배경 & 동기
- 단순 회귀는 희소/잡음 환경에서 불안정하며, 공간적 상관·참조 예시 활용이 제한적입니다.
- exemplar-guided + GNN 구성으로 유사 패턴 전이를 활용하고, 공간 구조를 보존한 예측을 달성합니다.

## 2. 방법론(개요)
- EGN(기본): 이미지 패치 특징 + exemplar 유사도 기반 가중 회귀/보정
- 확장(GNN): spot 그래프(공간 이웃)로 메시지 패싱하여 지역 맥락 반영
- 학습: 지도 손실(유전자별 MSE 등) + exemplar 정합/정규화 항

## 3. 실험 & 결과(개요)
- 데이터: 공개 ST 데이터셋(여러 조직)
- 정량: 유전자/spot 단위 상관·MAE 개선, 희소 유전자에서 안정성 향상
- 정성: 공간 경계에서 예측 매끄러움 개선, 생물학적 경로 일관성 증가

## 4. 의의 & 한계
- 의의: 예시/공간 정보 결합으로 ST 예측의 견고성·해석 가능성 개선
- 한계: exemplar 선택 품질/그래프 구성(이웃 정의)에 민감, 연산비 증가

## 5. 개인 평가
- 적용성: 조직별 도메인 편차가 큰 상황에서 유리, GNN 확장으로 공간 일관성 강화
- 보완점: exemplar 검색 인덱싱/캐싱, 유전자별 동적 가중·불확실성 추정

## 6. 참고 자료
- WACV 2023 PDF: https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Exemplar_Guided_Deep_Neural_Network_for_Spatial_Transcriptomics_Analysis_of_WACV_2023_paper.pdf
- 확장(bioRxiv): https://www.biorxiv.org/content/10.1101/2023.03.30.534914v1.full.pdf
- GitHub: https://github.com/Yan98/EGN

## 작성 체크리스트
- [ ] 데이터/지표 수치 인용 보강
- [ ] 주요 도식/표 추출 및 삽입 확인(자동 주입기 사용 가능)
- [ ] 링크/인용 형식 점검, 로컬 빌드 확인

