---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: 'CoT-RAG는 5단계 체인오브소드와 RAG를 결합해 CHOP 임상 노트와 Phenopacket, PubMed 서사에서 Top-10 유전자 정확도를 최대 30pt 이상 향상시키고, ACTA1·ZIC3 사례에서 1순위 재배치를 달성합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- Clinical Notes
- Rare Disease
- Chain-of-Thought
- Retrieval
title: 'CoT-RAG on Rare Disease Diagnosis: Clinical Notes Study'
toc: true
toc_sticky: true
---
# CoT-RAG on Rare Disease Diagnosis: Clinical Notes Study

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `medical-ai`인가요?
- [x] `excerpt`에 핵심 성과를 넣었나요?
- [x] 모든 섹션을 실제 논문 내용으로 채웠나요?
- [x] 결과 요약이 명확한가요?
- [x] 참고 링크를 추가했나요?

## 1. 핵심 요약 (3문장 이하)
- 본 연구는 희귀질환 임상 노트에서 유전자와 질환을 추천하기 위해 Chain-of-Thought(CoT) 다단계 추론과 Retrieval-Augmented Generation(RAG)을 통합한 하이브리드 전략을 제안합니다.
- DeepSeek-R1-Distill-Llama-70B는 Phenopacket 노트에서 Top-10 유전자 정확도가 11.72%→42.13%, Llama3.3-70B는 32.68%→37.86%로 향상됐습니다.
- CHOP 인하우스 노트처럼 노이즈가 많은 환경에서는 CoT-driven RAG가 Top-10 질환 정확도를 29.55%→35.00%까지 올려 실제 임상 적용 가능성을 보여줍니다.

## 2. 데이터 & 과제
- **Phenopacket-derived**: 5,980건 중 845건을 평가에 사용, 평균 298단어, 468개 질환/418개 유전자 커버.
- **PubMed Free-text**: 255건, 평균 198단어, 95개 유전자 커버.
- **CHOP In-house**: 220건, 평균 1,606단어, 114개 질환/146개 유전자.
- 과제는 유전자 Top-N 추천과 질환 진단 Top-N 추천 두 가지입니다.

## 3. 방법론 요약
- 5단계 CoT 프로토콜(용어 추출→인구학 영향→유전자 매핑→유전 양식→Top-10 리스트)을 LLM에 강제합니다.
- RAG는 HPO, OMIM, Orphanet 등 외부 지식에서 관련 엔티티를 검색해 reasoning 단계에 삽입합니다.
- 두 가지 결합 전략을 실험: **RAG-driven CoT**(먼저 검색 후 CoT)과 **CoT-driven RAG**(먼저 CoT로 뼈대 생성 후 필요 시 검색).

## 4. 주요 결과
| 데이터/모델 | 전략 | Top-10 Gene | Top-1 Gene | Top-10 Disease |
| --- | --- | --- | --- | --- |
| Phenopacket / DeepSeek | Baseline | 11.72% | 5.68% | 31.65% |
| Phenopacket / DeepSeek | RAG-driven CoT | **42.13%** | **23.78%** | **37.87%** |
| Phenopacket / Llama3.3 | Baseline | 35.03% | ~20% | 32.68% |
| Phenopacket / Llama3.3 | CoT-driven RAG | **37.86%** | 24.97% | **35%+** |
| CHOP / DeepSeek | Baseline | 8.18% | 3.64% | 29.55% |
| CHOP / DeepSeek | CoT-driven RAG | **25.91%** | 10%대 | **35.00%** |
- PubMed 서사에서는 CoT 적용 시 2~3pt 상승, RAG는 제한적인 이득을 보였습니다.

## 5. 사례 분석
- **Case 1 (ACTA1)**: Baseline은 6위였지만 RAG-driven CoT가 1순위로 올려 정확한 우선순위를 제시했습니다.
- **Case 2 (ZIC3)**: CoT가 임상 징후를 구조화해 희귀 심장 기형과 연관된 ZIC3를 최상위 후보로 재배치했습니다.

## 6. 의의 & 한계
- 비정형 임상 노트에서도 RAG+CoT가 희귀질환 후보 유전자/질환 추천을 대폭 향상시킬 수 있음을 입증했습니다.
- 데이터 품질에 따라 최적 전략이 달라지므로, 실제 시스템에서는 노트 길이·품질을 감지해 RAG 순서를 조정해야 합니다.
- 외부 지식베이스 구축과 PHI 비식별화가 필수이며, Top-N 기반 평가라 downstream 유전체 필터링이 필요합니다.

## 7. 참고 자료
- 원문: [arXiv:2505.21862](https://arxiv.org/abs/2505.21862)
- 코드/데이터: 저자 제공 GitHub (추후 공개 예정)
