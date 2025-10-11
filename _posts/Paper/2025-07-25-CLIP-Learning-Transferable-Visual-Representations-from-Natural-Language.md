---
categories:
- paper
- vlm
date: 2025-07-25
excerpt: 'CLIP은 4억 쌍의 웹 이미지-텍스트로 대비 학습해 ImageNet 제로샷 Top-1 76.2%를 달성하고, 27개 벤치마크 평균에서 선형 프로브 수준 성능을 무학습으로 확보합니다.'
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- CLIP
- Contrastive Learning
- Zero-Shot
- Vision-Language Model
title: 'CLIP: Learning Transferable Visual Representations from Natural Language Supervision'
toc: true
toc_sticky: true
---
# CLIP: Learning Transferable Visual Representations from Natural Language Supervision

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `vlm`으로 맞춰져 있나요?
- [x] `excerpt`에 핵심 수치를 포함했나요?
- [x] 모든 섹션이 실제 논문 내용을 반영하나요?
- [x] 결과 표를 핵심 지표 단계로 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- CLIP은 4억 개의 (이미지, 텍스트) 인터넷 캡션을 사용해 이미지·텍스트 인코더를 동시에 대비 학습시키는 단순한 pre-training 전략입니다.
- 사전학습 후 텍스트 인코더만으로 제로샷 분류기를 합성하며, ViT-L/14-336 모델은 ImageNet에서 추가 학습 없이 Top-1 76.2%를 기록해 완전 지도 ResNet-50과 동등한 성능을 보였습니다.
- 27개 전이 벤치마크 평균에서 제로샷 CLIP은 linear probe ResNet을 대체할 만큼 경쟁력 있고, 일부 과제(예: STL10)에서는 few-shot 튜닝보다 데이터 효율이 높습니다.

## 2. 배경 & 동기
- 기존 대규모 비전 모델은 고정된 카테고리 라벨(예: ImageNet 1K)에 의존해 새로운 개념을 학습하려면 재라벨링이 필요했습니다.
- NLP에서는 텍스트 기반 자가 지도 학습이 범용 전이를 가능하게 했지만, 비전 영역에선 웹 텍스트를 직접 이용한 대규모 실험이 부족했습니다.
- 저자들은 캡션-이미지 페어만으로도 범용 비전 표현을 학습할 수 있는지 검증하고자 CLIP을 제안했습니다.

## 3. 방법론
### 3.1 전체 구조
- 이미지와 텍스트 인코더(ResNet 계열 또는 ViT) 출력은 512차 임베딩으로 정규화되고, same-batch 모든 페어에 대한 InfoNCE 대비 손실을 최소화합니다.
- 학습 후 제로샷 분류는 각 클래스 설명(예: “a photo of a golden retriever”)을 텍스트 인코더에 통과시킨 임베딩과 이미지 임베딩의 코사인 유사도를 계산해 수행합니다.
- 고정 길이 배치(32,768)와 cosine 유사도 기반 대조 학습으로 대규모 데이터에서도 안정적으로 학습합니다.

### 3.2 핵심 기법
- **텍스트 기반 제로샷 클래스 합성**: 클래스 이름을 문장 템플릿에 삽입해 즉시 선형 분류기를 생성, 새로운 카테고리를 라벨링 없이 다룹니다.
- **대규모 데이터 스케일링**: 400M 웹 데이터로 2자릿수 컴퓨팅 스케일을 실험해 전이 성능이 계산 비용에 따라 예측 가능하게 향상됨을 보였습니다.
- **프롬프트 엔지니어링 & 앙상블**: 여러 템플릿과 다중 모델 평균을 사용하면 제로샷 Top-1이 추가로 1.3%p 향상됐습니다.

### 3.3 학습 및 구현 세부
- 데이터는 공개 웹 캡션을 필터링 없이 수집하고, 224~336 픽셀 해상도로 랜덤 크롭·수준 조정합니다.
- 최적화는 AdamW, Warmup+Cosine 스케줄, 기본 학습률 5e-4(ResNet) / 5e-5(ViT)로 설정했습니다.
- 텍스트는 76토큰으로 잘라 BPE 인코딩하며, 텍스트/이미지 인코더 모두 24~48 GPU로 수 주 학습했습니다.

## 4. 실험 & 결과
### 4.1 설정
- 모델: ResNet-50~RN50x64, ViT-B/32~ViT-L/14-336px 등 8종.
- 벤치마크: ImageNet, ImageNetV2/Sketch/A/R, CIFAR, STL10, Flowers102, EuroSAT 등 30여 개.
- 비교: 지도 학습 ResNet, SimCLRv2, MoCo v2, BiT-M 등과 제로샷/선형 프로브 기준.

### 4.2 주요 결과표
| 모델 | ImageNet Top-1 (zero-shot) | ImageNetV2 | 평균 27개 벤치(Top-1) |
| --- | --- | --- | --- |
| ResNet-50 (지도) | 76.2 (fully supervised) | 69.9 | 64.0 |
| CLIP ViT-B/32 | 68.3 | 62.9 | 63.2 |
| CLIP ViT-L/14 | 75.5 | 69.9 | 75.3 |
| **CLIP ViT-L/14@336** | **76.2** | **70.1** | **76.2** |

### 4.3 추가 분석
- 제로샷 CLIP은 STL10에서 16-shot logistic regression과 동일한 정확도를 0개의 라벨로 달성하는 등 데이터 효율이 높습니다.
- 패치 렌더링 취약성(ObjectNet, ImageNet-A)에선 성능이 낮지만, 프롬프트 앙상블과 클래스 설명을 늘리면 견고성이 개선됩니다.
- ImageNet Sketch/Vid 등 분포 이동 테스트에서도 지도학습 ResNet보다 낮은 하락폭을 보여 제로샷 일반화가 강함을 확인했습니다.

## 5. 의의 & 한계
- 라벨 정의 없이 언어만으로 시각 개념을 주입해 범용 제로샷 분류가 가능함을 입증했습니다.
- 텍스트 설명을 수정하면 새로운 태스크로 즉시 전환 가능해 모듈식 전이에 유리합니다.
- 다만 웹 데이터의 편향이 모델에 그대로 반영되고, 소수 클래스나 조작된 입력에 취약합니다.

## 6. 개인 평가
**강점**: 간결한 대비 학습 파이프라인으로 대규모 웹 데이터를 활용, Zero-shot 전이를 실용 수준으로 끌어올렸습니다.  
**약점**: 캡션 품질과 프롬프트 선택에 따라 성능 변동이 크며, 세밀한 위치/수량 추론은 여전히 어려움이 있습니다.  
**적용 가능성**: 산업·연구에서 라벨이 부족한 도메인에 즉시 적용 가능한 백본으로, 프롬프트 엔지니어링만으로 다양한 태스크를 다룰 수 있습니다.  
**추천도**: ★★★★★ (비전-언어 모델링과 제로샷 전이 전략을 배우려는 연구자 필독)

## 7. 참고 자료
- 원문: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- 코드: [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- 체크포인트: [Open CLIP Weights](https://github.com/mlfoundations/open_clip)



## 주요 도식/표

### Main Architecture
![Architecture](/assets/images/paper/2103.00020_Learning Transferable Visual Models From Natural Language Supervision/fig_01.png)
캡션: Figure 1. Summary of our approach. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descript…

### Main Results Table
![Results](/assets/images/paper/2103.00020_Learning Transferable Visual Models From Natural Language Supervision/table_5523.png)
캡션: Table 14. OCR performance on 5 datasets. All metrics are accuracy on the test set except for Hateful Memes which reports ROC AUC on the dev set. Single model SOTA reported to best of knowledge. ES Best reports the best performance across the 56 non-CLIP models in our evaluation suite. a((<>)Assiri, (<>)2020) b((<>)Jaderberg et al., (<>)2015) c((<>)Wang et al., (<>)2020) d((<>)Lippe et al., (<>)202…

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

