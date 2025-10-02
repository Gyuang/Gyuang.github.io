---
categories:
- paper
- medical-ai
date: 2025-07-25
excerpt: 'ChatCAD+는 도메인 분류 정확도 100%의 다중 CAD 라우팅과 계층형 in-context 학습, 메디컬 지식 검색을 결합해 ChestX-ray·Dental·MRI 등 9개 영역 보고에서 C-BLEU 4.41, F1 0.553으로 기존 LLM 대비 대폭 향상합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- Medical LLM
- Report Generation
- Knowledge Retrieval
- In-Context Learning
- Computer-Aided Diagnosis
title: 'ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs'
toc: true
toc_sticky: true
---
# ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `medical-ai`로 맞춰져 있나요?
- [x] `excerpt`에 핵심 성능을 포함했나요?
- [x] 모든 섹션이 논문 내용을 반영하나요?
- [x] 결과 표를 핵심 지표로 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- ChatCAD+는 의료 이미지를 도메인별 CAD 모델로 라우팅한 뒤, LLM 기반 계층형 in-context 학습으로 보고서를 생성하고, 외부 의료 지식 베이스를 참조해 환자 질의에 답변하는 통합 시스템입니다.
- CLIP 기반 도메인 식별기는 2,560장의 테스트 이미지에서 7개 영역을 100% 정확도로 분류했고, 9개 영역을 모두 포함해도 98.9% 정확도를 유지했습니다.
- ChatGPT에 구조화된 예시 보고와 신뢰할 수 있는 지식(메르크 매뉴얼 등)을 주입하면 C-BLEU 4.409, BLEU-4 0.760, ROUGE-L 17.344, METEOR 24.337, F1 0.553으로 기존 LLM 대비 보고 품질이 크게 향상됩니다.

## 2. 배경 & 동기
- 기존 LLM 기반 CAD는 특정 영상 도메인(X-ray 등)에만 최적화돼 범용성이 떨어지고, 텍스트 생성 품질도 제한적이었습니다.
- 환자 상담 단계에서는 최신 의학 지식이 반영되지 않거나 출처가 불분명한 답변이 제공되는 경우가 많아 신뢰성이 문제였습니다.
- 저자들은 다중 도메인 이미지를 자동 분류 후, 보고 생성과 상담 단계에 계층형 Retrieval-Augmented In-Context Learning을 도입해 신뢰도와 확장성을 동시에 확보하고자 했습니다.

## 3. 방법론
### 3.1 전체 구조
- Step 1: CLIP 또는 BiomedCLIP 임베딩을 이용한 도메인 식별기가 입력 영상을 카테고리화하고, 해당 CAD 모델의 예측(예: 진단 점수, 분할 마스크)을 텍스트로 요약합니다.
- Step 2: 보고 생성 모듈이 LLM 초안(report draft)을 작성한 후, TF-IDF 기반으로 유사한 상위 k개의 과거 보고를 검색해 계층형 프롬프트에 삽입하고 다시 생성합니다.
- Step 3: 환자 문의가 들어오면 메르크 매뉴얼·Mayo Clinic 등 신뢰 가능한 온라인 지식을 트리 구조로 탐색해, 관련 항목만 추출한 뒤 LLM이 답변을 합성합니다.

### 3.2 핵심 기법
- **Hierarchical In-Context Learning**: 초안 → 유사 사례 삽입 → 재생성 단계를 거쳐 LLM이 의료 문체와 핵심 소견을 보완합니다.
- **Knowledge-Grounded Dialogue**: 단순 답변 대신 의학 지식을 Retrieval-Augmented Generation 구조로 주입해 신뢰성과 출처를 강화합니다.
- **동적 모델 라우팅**: CAD 모듈을 추가하면 도메인 식별기와 텍스트 인터페이스만 업데이트하면 돼 확장성이 높습니다.

### 3.3 학습 및 구현 세부
- 도메인 분류는 CLIP·BiomedCLIP 임베딩에 선형 분류기를 학습했고, 최고 정확도는 CLIP 100%(7개 도메인) / 98.9%(9개 도메인)입니다.
- 보고 데이터셋은 MIMIC-CXR, ChestX-ray, Knee MRI 등 9개 도메인의 판독 보고를 포함하며, TF-IDF로 단어 중요도를 계산해 Hierarchical prompt를 구성했습니다.
- 상호작용 단계는 Merck Manuals, Mayo Clinic, Cleveland Clinic API를 활용해 다단계 탐색 후 지식 스니펫을 LLM에 전달합니다.

## 4. 실험 & 결과
### 4.1 설정
- 보고 품질: BLEU, METEOR, ROUGE-L, Precision/Recall/F1, C-BLEU(custom)를 측정합니다.
- 비교 모델: ChatGLM2, PMC-LLaMA, MedAlpaca, Mistral, LLaMA, LLaMA2, ChatGPT 등 (지식 미주입 vs 지식 주입).
- 환자 상호작용: 질의응답 정확도(정성 평가)와 지식 출처 대응력을 분석했습니다.

### 4.2 주요 결과표
| 모델 | 설정 | C-BLEU | BLEU-4 | ROUGE-L | METEOR | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| ChatGLM2-6B | 기본 | 2.073 | 0.313 | 14.119 | 23.008 | 0.553 |
| **ChatGLM2-6B + ChatCAD+** | 지식/예시 포함 | **3.236** | **0.566** | **15.559** | **23.397** | **0.577** |
| ChatGPT | 기본 | 3.594 | 0.556 | 16.791 | 23.146 | 0.553 |
| **ChatGPT + ChatCAD+** | 지식/예시 포함 | **4.409** | **0.760** | **17.344** | **24.337** | **0.553** |

### 4.3 추가 분석
- BiomedCLIP은 9개 도메인에서 86.3% 정확도로 다중 도메인에는 CLIP이 더 견고함을 확인했습니다.
- 계층형 프롬프트를 5단계까지 확장하면 BLEU-1/2/3/4와 ROUGE-L이 각각 약 0.14, 0.07, 0.07, 0.08, 0.8pt씩 상승했습니다.
- 신뢰 상호작용 실험에서 의료 지식 없이 생성된 답변은 사실 오류가 빈번했지만, 지식 주입 후 전문가 평가에서 “안전/정확” 판정 비율이 크게 증가했습니다.

## 5. 의의 & 한계
- 다수의 CAD 모델을 한 인터페이스로 통합해 “범용” 진단 보조를 구현했고, 예시/지식 주입으로 LLM 보고 품질을 안정화했습니다.
- 신뢰 가능한 외부 지식원을 통해 환자 상담까지 확장해 임상 적용 가능성을 높였습니다.
- 다만, 지식베이스와 과거 보고의 품질/편향이 그대로 반영될 수 있으며, 실시간 업데이트가 필요한 응급 상황에는 추가 검증이 필요합니다.

## 6. 개인 평가
**강점**: 도메인 식별→CAD→LLM 보고→지식 기반 상담까지 전 주기를 통합한 점이 매우 실용적입니다.  
**약점**: 지식 검색 파이프라인이 복잡해 유지보수 비용이 높고, F1 개선 폭이 모델마다 상이합니다.  
**적용 가능성**: 병원 PACS/EMR에 연결하면 영상 판독과 환자 상담을 동시에 보조하는 올인원 챗봇으로 활용할 수 있습니다.  
**추천도**: ★★★★☆ (의료 LLM+CAD 통합을 고민하는 연구자에게 추천)

## 7. 참고 자료
- 원문: [ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs](https://arxiv.org/abs/2407.16684)
- 코드: [공식 GitHub](https://github.com/ShanghaiTechAI/ChatCAD-plus) (논문 링크)
- 지식 베이스: Merck Manuals, Mayo Clinic, Cleveland Clinic 공개 자료
