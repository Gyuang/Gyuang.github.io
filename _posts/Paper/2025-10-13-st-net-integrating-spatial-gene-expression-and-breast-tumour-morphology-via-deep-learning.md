---
categories:
- paper
- bioinformatics
date: 2025-10-13
excerpt: 'ST-Net은 유방암 H&E 조직영상의 형태학을 학습해 공간 전사체(spot) 단위 유전자 발현을 예측, 형태-분자 정보 융합 가능성을 보여줍니다.'
header: {}
last_modified_at: '2025-10-13'
published: true
tags:
- Spatial Transcriptomics
- Histopathology
- Gene Expression Prediction
- Deep Learning
title: 'ST-Net: Integrating spatial gene expression and breast tumour morphology via deep learning'
toc: true
toc_sticky: true
---
# ST-Net: Integrating spatial gene expression and breast tumour morphology via deep learning

## 0. 핵심 요약 (3문장 이하)
- ST-Net은 유방암 H&E 병리 이미지와 공간 전사체(Visium 등) spot의 매핑을 통해, 형태학 기반 딥러닝으로 spot별 유전자 발현을 예측합니다.
- 패치 임베딩→유전자 표현 회귀(또는 분류)로 구성되어 이미지-분자 신호를 연결하며, 유전자/spot 단위 상관·MAE 기준으로 기존 기법을 상회합니다.
- 임상적으로 조직 형태와 분자 상태 간의 연관 구조를 학습해, 저비용 영상 기반 전사체 추정과 후보 바이오마커 발굴에 기여합니다.

## 1. 배경 & 동기
- 공간 전사체(ST)는 위치 정보를 보존한 유전자 발현을 제공하지만 비용이 높고 처리량이 제한적입니다.
- H&E 병리 이미지는 저비용·고해상도로 형태학 정보를 제공하나 분자 정보를 직접 반영하지 않습니다.
- 두 modality를 정렬해 형태에서 분자를 추정하면, 대규모 코호트에서 분자 지표를 보완할 수 있습니다.

## 2. 방법론(개요)
- 입력: H&E 타일(spot 대응 패치)
- 모델: CNN 기반 특징 추출 → 유전자별 예측 헤드(회귀/분류)
- 학습: spot 매칭된 이미지-발현 페어 supervision, 유전자별 손실(예: MSE/Huber)
- 평가: 유전자/spot 단위 상관계수, MAE, 유전자 셋별 성능 비교

## 3. 실험 & 결과(개요)
- 데이터: 유방암 공간 전사체(수만 spot, 다수 환자)
- 정량: 유전자별 상관/MAE 향상, high-variance gene에서 특히 향상
- 정성: 형태학적 구조(종양/간질/괴사 등)에 따른 발현 패턴을 시각화해 생물학적 합리성 확인

## 4. 의의 & 한계
- 의의: 병리 영상에서 분자 신호를 예측하는 프레임워크로, 저비용 대규모 분자 프로파일링 보완 가능
- 한계: 데이터 도메인 편향, spot 해상도·정합 오차, 낮은 발현 유전자 예측의 한계

## 5. 개인 평가
- 적용성: 대규모 임상 병리 이미지 코호트에 ST 보완으로 즉시 유용
- 보완점: 다중 시퀀스/염색, 멀티태스크(세포 조성/경로활성) 확장, 불확실성 추정 도입 권장

## 6. 참고 자료
- 논문: Nature Biomedical Engineering (s41551-020-0578-x)
- PDF: https://ai.stanford.edu/~bryanhe/publications/stnet.pdf
- 코드: https://github.com/bryanhe/ST-Net

## 작성 체크리스트
- [ ] 데이터/지표 수치 인용 보강
- [ ] 주요 도식/표 추출 및 삽입 확인(자동 주입기 사용 가능)
- [ ] 링크/인용 형식 점검, 로컬 빌드 확인

