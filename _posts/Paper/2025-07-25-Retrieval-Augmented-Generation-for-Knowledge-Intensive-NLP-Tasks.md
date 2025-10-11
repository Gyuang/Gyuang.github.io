---
categories:
- paper
- rag
date: 2025-07-25
excerpt: Dense retriever와 BART 생성기를 결합한 RAG가 Natural Questions와 TriviaQA 등 지식집약 QA에서
  큰 폭의 성능 향상을 달성한 내용을 정리합니다.
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- RAG
- Retrieval
- NLP
- Language Model
title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
toc: true
toc_sticky: true
---
# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `rag`로 맞춰져 있나요?
- [x] `excerpt`에 정량적 성과를 언급했나요?
- [x] 모든 섹션을 실제 내용으로 채웠나요?
- [x] 결과 표를 핵심 지표로 압축했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- RAG는 DPR 기반 retriever와 BART 기반 generator를 결합해 외부 지식을 인용하면서 자연스럽게 답변을 생성하는 end-to-end 오픈 도메인 QA 모델입니다.
- Natural Questions-open에서 RAG-Token은 exact match 45.5%로 기존 BART fine-tuning(29.8%) 대비 약 15.7pt, DPR+Reader(41.5%) 대비 4pt 이상 향상했습니다.
- TriviaQA-open에서는 EM 56.8%로 DPR 기반 추출식 모델을 5pt 이상 앞서며, 추론과 생성 품질을 동시에 확보했습니다.

## 2. 배경 & 동기
- 오픈 도메인 QA는 지식 검색과 언어 모델화를 동시에 요구하지만, 기존 파이프라인은 retrieval과 reader가 분리돼 최적화가 어렵습니다.
- 거대 언어 모델은 지식 누락과 환각 문제로 사실적 응답을 보장하지 못하며, 문서를 모두 파라미터로 저장하려면 메모리 비용이 과도합니다.
- RAG는 외부 위키피디아 문서를 실시간으로 검색하여 generator 입력에 주입함으로써 최신 지식을 활용하면서도 end-to-end 학습을 가능하게 합니다.

## 3. 방법론
### 3.1 전체 구조
- 입력 질문은 Dense Passage Retriever(DPR)를 통해 top-k 문서(보통 5~10개)를 검색합니다.
- 검색된 문서는 질문과 함께 BART encoder-decoder에 concatenation돼 조건부 생성 확률 $p(y|x, z)$를 계산합니다.
- 최종 확률은 문서별 생성 확률을 marginalization해 $p(y|x) = \sum_{z\in Z} p(y|x,z) p(z|x)$ 형태로 계산합니다.

### 3.2 핵심 기법
- **RAG-Sequence**: 전체 답변 시퀀스에 대해 문서별 확률을 독립적으로 계산한 후 가장 높은 문서를 선택합니다.
- **RAG-Token**: 각 생성 토큰 단위로 문서를 marginalize하여 정보량이 높은 문서에 동적으로 가중치를 부여합니다.
- retriever와 generator 파라미터를 end-to-end joint training하여 검색과 생성이 같은 목적 함수(negative log-likelihood)를 공유하도록 했습니다.

### 3.3 학습 및 구현 세부
- 백본은 DPR retriever(BERT-base encoders)와 BART-large generator이며, 위키피디아 21M passages를 인덱싱했습니다.
- 학습은 mixed precision, 배치 128, learning rate 3e-5( retriever )와 1e-5(generator )로 2~3 epoch 수행했고, retriever에는 warmup과 temperature scaling을 사용했습니다.
- 모델과 FAISS 인덱스 스크립트는 Hugging Face Transformers 기반으로 공개돼 손쉽게 재현할 수 있습니다.

## 4. 실험 & 결과
### 4.1 설정
- **Datasets**: Natural Questions-open, TriviaQA-open, WebQuestions, Jeopardy.
- **Baselines**: DPR reader, BART fine-tuning, Fusion-in-Decoder 등.
- 평가는 exact match(EM)와 F1, Rouge-L로 진행했습니다.

### 4.2 주요 결과표
| Dataset | Metric | DPR Reader | BART Fine-tune | RAG-Token |
| ------- | ------ | ---------- | -------------- | --------- |
| NQ-open | EM | 41.5 | 29.8 | **45.5** |
| TriviaQA | EM | 51.8 | 40.9 | **56.8** |
| WebQuestions | EM | 45.2 | 26.4 | **55.2** |

### 4.3 추가 분석
- RAG-Token은 RAG-Sequence 대비 모든 벤치마크에서 1~2pt 높으며, 특히 장문 검색이 필요한 질문에서 효과가 컸습니다.
- top-k 문서 수를 5→10으로 늘리면 성능이 소폭 상승하나, 20 이상에서는 노이즈로 감소했습니다.
- Zero-shot 요약까지 확장 실험을 수행해 retrieval이 hallucination을 줄이는 데 도움을 준다는 것을 실증했습니다.

## 5. 의의 & 한계
- RAG는 retrieval과 generation을 단일 목적 함수로 묶어 오픈 도메인 QA에서 생성형 모델의 활용 가능성을 보여줬습니다.
- 대규모 인덱스 업데이트가 필요하고, retriever가 놓친 지식을 generator가 스스로 보완하기 어렵다는 한계가 있습니다.
- 문서 출처를 그대로 전달하지 않아 fact-checking이나 provenance 요구에는 추가 설계가 필요합니다.

## 6. 개인 평가
**강점**: 검색과 생성 모듈을 결합한 end-to-end 구조, 재현 가능한 코드·모델 공개, 넓은 태스크에서의 일관된 성능 향상.  
**약점**: retriever 튜닝 비용과 인덱스 유지비가 높고, 도메인 특화 질문에는 추가 파인튜닝이 필수입니다.  
**적용 가능성**: 사내용 지식베이스 QA, 최신 정보가 필요한 요약/검색 챗봇 등에 매우 유용합니다.  
**추천도**: ★★★★★ (지식집약 QA/검색-생성 결합을 고려 중이라면 필독)

## 7. 참고 자료
- 원문: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- 코드: [GitHub - facebookresearch/rag](https://github.com/facebookresearch/rag)
- 데이터: [Natural Questions-open](https://ai.google.com/research/NaturalQuestions), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/)



## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/2005.11401_Retrieval Augmented Generation for Knowledge Intensive NLP Tasks/fig_03.png)
캡션: Figure 1: Overview of our approach. We combine a pre-trained retriever (Query Encoder + Document Index) with a pre-trained seq2seq model (Generator) and fine-tune end-to-end. For query x, we use Maximum Inner Product Search (MIPS) to find the top-K documents zi. For final prediction y, we treat z as a latent variable and marginalize over seq2seq predictions given different documents.

### Main Results Table
![Results](/assets/images/paper/2005.11401_Retrieval Augmented Generation for Knowledge Intensive NLP Tasks/table_01.png)
캡션: to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%.

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

