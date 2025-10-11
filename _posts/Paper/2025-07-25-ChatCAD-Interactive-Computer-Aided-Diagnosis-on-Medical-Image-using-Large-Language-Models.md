---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: 'ChatCAD는 CAD 분류·세그멘테이션·보고 결과를 GPT-3/ChatGPT에 연결해 다섯 질환 평균 F1 0.591→0.605, 침윤 재현율 0.626, 농축 재현율 0.803 등 보고 정확도를 크게 개선하면서 환자 질의응답까지 지원합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- ChatCAD
- Medical Dialogue
- Report Generation
- Prompt Engineering
- Explainable AI
title: 'ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models'
toc: true
toc_sticky: true
---
# ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `medical-ai`로 맞춰져 있나요?
- [x] `excerpt`에 정량 요약을 담았나요?
- [x] 모든 섹션이 논문을 반영하나요?
- [x] 결과 표를 핵심 지표로 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- ChatCAD는 기존 CAD 네트워크(진단 분류기, 병변 세그멘터, 보고 생성기)를 하나의 텍스트 인터페이스로 묶어 LLM이 보고를 재구성하고 환자와 질의응답을 수행하게 합니다.
- GPT-3 기반 ChatCAD는 다섯 질환 평균 F1 0.591, ChatGPT 사용 시 0.605로 기존 R2GenCMN 대비 0.26p 개선했으며 Recall이 크게 증가했습니다.
- 환자 질문에 대해 LLM이 “무기폐란 무엇인가?”, “어떤 약을 복용해야 하나?” 등 추가 설명을 제공해 상호작용형 CAD를 구현했습니다.

## 2. 배경 & 동기
- CAD 출력은 수치·마스크 형태라 환자나 비전문가가 이해하기 어렵고, 텍스트 보고는 누락/중복이 빈번했습니다.
- LLM은 풍부한 의학 지식을 갖지만 영상 데이터를 직접 해석하지 못하므로, CAD 결과를 텍스트로 중개할 인터페이스가 필요했습니다.
- ChatCAD는 CAD가 생성한 구조화된 정보를 LLM 프롬프트로 전달해 보고 품질과 상호작용성을 동시에 확보했습니다.

## 3. 방법론
### 3.1 전체 구조
- 분류기(예: CvT2), 세그멘터, 보고 생성기의 출력을 각각 문장으로 기술하고 환자 증상과 함께 LLM에 전달합니다.
- “Network A 결과를 반영해 보고서를 수정하라”는 명령형 프롬프트로 LLM이 여러 출력을 통합해 일관된 보고를 생성합니다.
- 환자 후속 질문은 같은 대화 세션에서 처리해 CAD 결과를 바탕으로 치료·원인 등을 설명합니다.

### 3.2 핵심 기법
- **Text Wrapping**: 스칼라 점수·마스크 면적을 자연어 문장(“좌하엽 35% 침범”)으로 변환해 LLM이 이해할 수 있도록 합니다.
- **Prompt Sequencing**: CAD 결과→보고 초안→수정 요청으로 이어지는 프롬프트 시퀀스로 LLM이 오류를 교정합니다.
- **Interactive QA**: 보고 생성 후 환자 질문을 받아 멀티턴 대화를 유지합니다.

### 3.3 학습 및 구현 세부
- 경우 기반 실험으로 GPT-3(text-babbage/curie/davinci)와 ChatGPT를 호출했습니다.
- 데이터는 MIMIC-CXR 등 공공 Chest X-ray 세트에서 확보한 5개 질환 레이블과 보고서를 활용했습니다.
- 보고 품질은 Precision/Recall/F1(질환별), BLEU/ROUGE 등 추가 지표로 평가했습니다.

## 4. 실험 & 결과
### 4.1 설정
- 베이스라인: CvT2, DistilGPT2, R2GenCMN.
- LLM 비교: GPT-3 세 가지 크기, ChatGPT.
- 병변별 지표와 평균 지표, 모델 크기별 보고 길이/무의미 보고 비율을 분석했습니다.

### 4.2 주요 결과표
| 모델 | 평균 Precision | 평균 Recall | 평균 F1 | 무의미 보고 비율 |
| --- | --- | --- | --- | --- |
| R2GenCMN | 0.622 | 0.231 | 0.340 | - |
| ChatCAD (text-babbage) | 0.531 | 0.238 | 0.329 | 39.6% |
| ChatCAD (text-davinci) | 0.601 | 0.626 | 0.591 | 0% |
| **ChatCAD (ChatGPT)** | 0.598 | **0.626** | **0.605** | 0% |

### 4.3 추가 분석
- 침윤(Infiltration) Recall 0.626, Consolidation Recall 0.803으로 baseline 대비 각각 0.387p, 0.682p 향상했습니다.
- 소형 LLM(text-babbage, text-curie)은 보고 길이가 짧고 무의미 출력 비율이 높아 모델 크기와 성능의 상관관계를 확인했습니다.
- 사용자는 보고 후 “어떤 약을 복용해야 하나?”, “Airspace consolidation은 무엇인가?” 등 질문을 통해 개인화된 설명을 받을 수 있었습니다.

## 5. 의의 & 한계
- CAD와 LLM을 연결해 보고 품질과 사용자 친화성을 동시에 개선한 초기 연구로, 향후 의료 상담 챗봇으로 확장 가능성을 보여줍니다.
- 프롬프트 설계와 CAD 모델 선택에 따라 성능 변동이 크고, LLM의 사실 검증 및 최신 지식 업데이트가 필요합니다.
- 실제 임상 도입 전에는 인간 판독자의 검증과 법적 책임 범위 설정이 필수입니다.

## 6. 개인 평가
**강점**: 멀티모달 출력을 LLM이 자연스럽게 통합해 판독 정확도와 설명력을 동시에 향상시켰습니다.  
**약점**: 텍스트 변환과 프롬프트 유지가 수작업에 많이 의존하며, 지식 기반이 없으면 잘못된 답이 나올 위험이 있습니다.  
**적용 가능성**: 판독 보조, 환자 설명 챗봇, 비전문가용 2차 의견 등 다양한 워크플로에 적용 가능합니다.  
**추천도**: ★★★★☆ (CAD와 LLM 융합을 실험해 보고 싶은 팀에 추천)

## 7. 참고 자료
- 원문: [ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models](https://arxiv.org/abs/2302.07257)
- 코드: [ChatCAD GitHub](https://github.com/ShanghaiTechAI/ChatCAD)
- 데이터: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)



## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/2302.07257_ChatCAD Interactive Computer Aided Diagnosis on Medical Image using Large Language Models/fig_01.png)

### Main Results Table
![Results](/assets/images/paper/2302.07257_ChatCAD Interactive Computer Aided Diagnosis on Medical Image using Large Language Models/table_29.png)
캡션: Table 2. F1-score comparison of different-size LLMs

## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build


## 작성 체크리스트

- [ ] 이미지가 논문 메인 아키텍처/결과표와 일치하는지 확인
- [ ] 캡션 문구가 자연스러운지 확인 (필요 시 수정)
- [ ] 해상도/가독성 확인 (너비 조정 필요 시 이미지 교체)
- [ ] 링크/출처 표기 적절성 점검
- [ ] 로컬 빌드 확인: bundle exec jekyll build

