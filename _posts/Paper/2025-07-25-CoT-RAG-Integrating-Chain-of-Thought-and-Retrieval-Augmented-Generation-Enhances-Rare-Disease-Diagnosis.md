---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: 'CoT-RAG는 Phenopacket·PubMed·CHOP 임상 노트에서 Llama3.3 기반 Top-10 유전자 식별 정확도를 35.03%→37.86%, DeepSeek 기반은 11.72%→42.13%까지 끌어올려 희귀질환 진단 보조 성능을 크게 향상합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- Rare Disease
- Gene Prioritization
- Clinical Notes
- Chain of Thought
- Retrieval-Augmented Generation
title: 'Integrating Chain-of-Thought and Retrieval-Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes'
toc: true
toc_sticky: true
---
# Integrating Chain-of-Thought and Retrieval-Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `medical-ai`로 설정됐나요?
- [x] `excerpt`에 핵심 수치를 포함했나요?
- [x] 모든 섹션이 논문 내용을 반영하나요?
- [x] 결과 요약이 명확한가요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- 논문은 희귀질환 진단을 위해 Chain-of-Thought(CoT)와 Retrieval-Augmented Generation(RAG)을 결합한 두 전략(RAG-driven CoT, CoT-driven RAG)을 제안합니다.
- Phenopacket-derived 노트에서 DeepSeek-R1-Distill-Llama-70B의 Top-10 유전자 정확도가 11.72%→42.13%, Top-1은 5.68%→23.78%로 향상됐으며, Llama3.3-70B도 32.68%→37.86%(Top-10)로 개선됐습니다.
- 노이즈가 많은 CHOP in-house 노트에서는 CoT-driven RAG가 Top-10을 29.55%→35.00%, Top-1을 14.77%→20%대까지 끌어올려 실제 임상 환경에서도 효과를 확인했습니다.

## 2. 배경 & 동기
- 기존 LLM 기반 희귀질환 진단 연구는 구조화된 HPO 용어를 전제로 하지만, 실제 임상 노트는 비정형 텍스트입니다.
- HPO·OMIM 같은 외부 지식이 진단 성능에 기여한다는 점은 알려져 있으나, retrieval과 reasoning을 어떻게 결합할지 명확한 전략이 부족했습니다.
- 저자들은 임상의의 사고 과정을 모사한 5단계 CoT(용어 추출→인구학적 영향→유전자 매핑→유전 양식 평가→최종 리스트)와 RAG를 상호 보완적으로 활용했습니다.

## 3. 방법론
### 3.1 전체 구조
- **CoT Prompt**: “추출·분류/인구학 영향/유전자-질병 매핑/유전 양식·변이 체크/Top-10 랭킹” 다섯 단계를 강제해 reasoning 로그를 생성합니다.
- **RAG**: HPO, OMIM 등에서 관련 정보를 retrieval해 reasoning 중간 단계에 삽입합니다.
- **두 전략**:
  - *RAG-driven CoT*: retrieval로 확장된 지식을 먼저 확보한 뒤 CoT를 수행.
  - *CoT-driven RAG*: 기본 CoT로 개요를 잡은 뒤, 필요한 시점에 목표 지식을 재조회.

### 3.2 핵심 기법
- **데이터 전처리**: 5,980 Phenopacket-derived 노트, 255 PubMed 서술형 사례, 220 CHOP 임상 노트를 JSON 형태로 정리했습니다.
- **모델**: Llama3.3-70B-Instruct, DeepSeek-R1-Distill-Llama-70B 등 최신 오픈소스 LLM을 비교.
- **평가 지표**: Top-1/Top-10 유전자 또는 질환 정확도.

### 3.3 학습 및 구현 세부
- 프롬프트는 유전자 10개를 정확히 나열하도록 강제된 템플릿을 사용해 모델 출력 포맷을 정규화했습니다.
- RAG 검색은 HPO/OMIM/Orphanet, PubMed 초록 등을 대상으로 하며, BM25+임베딩 기반 검색기를 병행했습니다.
- In-house 노트는 HIPAA 준수를 위해 비식별화 후 사용했습니다.

## 4. 실험 & 결과
### 4.1 설정
- 세 데이터 소스: Phenopacket-derived(5980 중 845개 평가), PubMed free-text(255개), CHOP in-house(220개).
- 비교: Baseline LLM, CoT-only, RAG-only, RAG-driven CoT, CoT-driven RAG.
- 평가: 유전자 우선 순위와 질환 이름 추천 정확도.

### 4.2 주요 결과표
| 모델 & 전략 | Phenopacket Gene Top-10 | Phenopacket Disease Top-10 | PubMed Gene Top-10 | CHOP Disease Top-10 |
| --- | --- | --- | --- | --- |
| Llama3.3-70B Baseline | 35.03% | 32.68% | 27.84% | ~30% |
| Llama3.3 + CoT-driven RAG | **37.86%** | **35%+** | 30%대 초반 | **34.09%** |
| DeepSeek Baseline | 11.72% | 31.65% | 21.57% | 29.55% |
| DeepSeek + RAG-driven CoT | **42.13%** | **37.87%** | 22~23% | 28.18% |
| DeepSeek + CoT-driven RAG | 41.54% | 35%대 | 25%+ | **35.00%** |

### 4.3 추가 분석
- 고품질 Phenopacket 노트에서는 early retrieval(RAG-driven CoT)이 가장 큰 상승(Top-10 +30pt)을 보였습니다.
- PubMed 짧은 노트는 retrieval 노이즈가 많아 CoT-only 대비 소폭 향상에 그쳤습니다.
- CHOP 노트처럼 길고 노이즈가 많은 경우 CoT-driven RAG가 더 안정적이었으며, baseline 대비 Top-10이 5~8pt 상승했습니다.

## 5. 의의 & 한계
- CoT와 RAG를 합친 전략이 비정형 임상 노트에서도 희귀질환 후보 유전자 추천을 크게 개선함을 보여줬습니다.
- 데이터 품질에 따라 최적 전략이 다르므로, 실제 시스템에서는 노트 특성에 따라 RAG 순서를 조절해야 합니다.
- 지식베이스 구축과 프롬프트 설계가 복잡하며, HIPAA 보호가 필요한 실제 노트에서는 추가 보안 설계가 필요합니다.

## 6. 개인 평가
**강점**: 실제 임상 환경(노이즈 많은 노트)에서도 LLM의 진단 능력을 유의미하게 향상시킨 점이 고무적입니다.  
**약점**: Top-10 정확도 기준이라 후속 필터링이 필요하고, retrieval 품질에 매우 민감합니다.  
**적용 가능성**: 유전자 패널 추천, 희귀질환 진단 컨설팅 등 의학 의사결정 지원에 즉시 참고할 수 있습니다.  
**추천도**: ★★★★☆ (의료 LLM 연구자 및 병원 AI팀에 권장)

## 7. 참고 자료
- 원문: [Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes](https://arxiv.org/abs/2505.21862)
