---
categories:
- paper
- rag
date: 2025-07-25
excerpt: 'CoT-RAG는 지식 그래프 기반 CoT 생성·RAG·의사 코드 실행을 결합해 ERNIE-Speed-128K 기준 네 가지 벤치마크 평균 정확도를 80.0%→95.2%로 끌어올리고, GPT-4o mini에서도 9개 과제 평균 99.5%를 달성합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- CoT-RAG
- Knowledge Graph
- Retrieval-Augmented Generation
- Reasoning
- LLM
title: 'CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models'
toc: true
toc_sticky: true
---
# CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `rag`인가요?
- [x] `excerpt`에 대표 성능 향상을 포함했나요?
- [x] 모든 섹션에 실제 실험 정보를 요약했나요?
- [x] 결과 표는 핵심 지표 위주인가요?
- [x] 참고 자료를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- CoT-RAG는 지식 그래프 기반 CoT 생성, 학습 가능한 케이스 인지 RAG, 의사 프로그램 실행 프롬프트 세 가지를 결합해 LLM 추론의 신뢰성과 정확도를 동시에 끌어올립니다.
- ERNIE-Speed-128K에서 LaB/LeB/CFBenchmark/AGIEval 네 가지 그래프 추론 벤치마크 평균 정확도가 80.0%(GraphRAG)→95.2%로 15.2pt 상승했습니다.
- GPT-4o mini·ERNIE-3.5·GLM-4 등 다양한 LLM에 적용해 9개의 산술·상식·기호 추론 과제 평균 4.0~44.3pt 향상을 기록했고, 추론 시간과 토큰 소비도 최대 33.4% 감소했습니다.

## 2. 배경 & 동기
- 단순 CoT 프롬프트는 LLM 자체 reasoning에 의존하기 때문에 신뢰성이 낮고, 자연어 체인보다 코드 스타일 추론이 더 강력하다는 보고가 많습니다.
- Retrieval-Augmented Generation은 지식 공백을 채우지만, 무작위 retrieval은 노이즈가 많고 reasoning 과정과 느슨하게 연결됩니다.
- 저자들은 지식 그래프를 중심으로 CoT와 RAG, 의사 코드 실행을 일관된 파이프라인에 통합해 정확성과 효율을 모두 확보하려 했습니다.

## 3. 방법론
### 3.1 전체 구조
- **Stage 1**: 도메인 지식 그래프를 기반으로 reasoning 태스크를 분해하고, 각 노드에 필요한 sub-case·sub-description을 채웁니다.
- **Stage 2**: Learnable Knowledge Case-aware RAG가 노드별로 관련 텍스트를 retrieval하고 pseudo-knowledge graph(PKG)를 완성합니다.
- **Stage 3**: Pseudo-Program Prompting Execution이 PKG를 의사 코드로 실행하도록 LLM을 안내해, 각 노드 결과를 결합합니다.

### 3.2 핵심 기법
- **Knowledge Graph-driven CoT Generation**: 그래프를 통해 reasoning 경로를 제약해 LLM의 자유도를 줄이고, 오류를 줄입니다.
- **Case-aware RAG**: 다양한 벡터 인덱스(FlatL2/IP/IVF 등)를 실험하며 retrieval 성능을 학습 가능한 파라미터로 조정합니다.
- **Pseudo-program Prompting**: NL Chain 대신 프로그램 형태로 명시된 절차를 실행하게 해, 논리적 일관성을 높입니다.

### 3.3 학습 및 구현 세부
- 인덱스: Meta 구글 index 구현을 활용하되, 본 논문은 IndexFlatL2, IndexFlatIP, IndexPQ, IndexIVFPQ 등 변형을 실험했습니다.
- 전문가 지식은 Domain Tree(DT) 노드로 구성되어, 그래프 구축에 필요한 세부 reasoning 로직을 제공합니다.
- 구현 코드는 PyTorch + HuggingFace 기반이며, GitHub(https://github.com/hustlfy123/CoT-RAG)에서 제공됩니다.

## 4. 실험 & 결과
### 4.1 설정
- **General reasoning**: LaB/LeB/CFBenchmark/AGIEval 그래프 기반 QA 벤치마크.
- **Arithmetic/Commonsense/Symbolic**: AQuA, GSM8K, MultiArith, SingleEq, HotpotQA, CSQA, SIQA, Last Letter, Coin Flip.
- **Vertical domains**: 법률(LFQA-Legal), 금융(CFBenchmark), 의료(MedQA), 교육(AGIEval) 등 4개 데이터셋.

### 4.2 주요 결과표
- **Table 1 (ERNIE-Speed-128K)**: GraphRAG 80.0% vs CoT-RAG 95.2% 평균. 개별 항목도 LaB 94.8→99.3, LeB 97.5→98.6, CFB 73.1→94.7, AGI 54.6→88.3으로 대폭 향상.
- **Table 4 (GPT-4o mini)**: 9개 추론 과제에서 CoT-RAG 평균 99.5%로 Zero-shot-CoT(87.5%)·Auto-CoT(86.8%)·PS(88.3%)를 모두 앞섰습니다.
- **Table 2 (domain-specific)**: GPT-4o mini 환경에서 법률·금융·의료·교육 데이터셋에서 8.9~80.6%p 정확도 개선, 평균 토큰 소비 33.4% 감소, 실행 시간 29.2% 단축.

### 4.3 추가 분석
- Retrieval 인덱스별 성능을 비교해, IndexFlatL2/IVFPQ 등 변형보다 제안한 학습형 인덱스가 평균 1~4pt 높은 정확도를 달성했습니다.
- Ablation에서 Knowledge Graph, RAG, Pseudo-program 중 하나라도 제거하면 정확도가 크게 하락해 세 요소의 상호 보완성을 확인했습니다.
- C++/Java 등 다른 pseudo-language로 변형해도 성능이 근접해(예: GSM8K 99.2% vs 98.7%) 프롬프트 언어에 대한 강건성을 보였습니다.

## 5. 의의 & 한계
- CoT와 RAG, 그리고 프로그램 실행을 결합해 LLM 추론의 신뢰성과 효율을 모두 향상시키는 포괄적 프레임워크를 제시했습니다.
- 다양한 LLM과 도메인에 적용되어 재현성이 높으며, 추론 비용 감소까지 보여 실무 적용 가능성이 큽니다.
- 단, 지식 그래프/전문가 규칙을 구축해야 하므로 초기 구축 비용이 크고, 매우 데이터 빈약한 분야에는 확장성이 제한될 수 있습니다.

## 6. 개인 평가
**강점**: 지식 그래프 기반 제약과 pseudo-program 실행을 통해 LLM reasoning을 안정화한 점이 인상적입니다.  
**약점**: 그래프·전문가 규칙 설계가 복잡하고, Stage 1 구축이 자동화되지 않았습니다.  
**적용 가능성**: 복잡한 의사결정, 법률 QA, 의료 QA 등 규칙 기반 reasoning을 필요로 하는 엔터프라이즈 서비스에 즉시 활용할 수 있습니다.  
**추천도**: ★★★★★ (고신뢰 추론이 필요한 LLM 응용 연구자에게 강력 추천)

## 7. 참고 자료
- 원문: [CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models](https://arxiv.org/abs/2503.12286)
- 코드: [CoT-RAG GitHub](https://github.com/hustlfy123/CoT-RAG)
