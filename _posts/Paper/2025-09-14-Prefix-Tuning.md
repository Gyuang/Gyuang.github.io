---
categories:
- paper
- transformer
date: '2025-09-16'
excerpt: 언어 모델 파라미터의 0.1%만 학습해 WebNLG BLEU 44.0과 XSum ROUGE-L 21.9를 기록한 Prefix-Tuning을
  정리합니다.
header:
  teaser: /assets/images/paper/prefix-tuning-teaser.png
last_modified_at: '2025-09-16'
published: true
tags:
- Prefix Tuning
- Parameter Efficient
- Generation
- NLP
title: 'Prefix-Tuning: Optimizing Continuous Prompts for Generation'
toc: true
toc_sticky: true
---
# Prefix-Tuning: Optimizing Continuous Prompts for Generation

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `transformer`로 맞춰져 있나요?
- [x] `excerpt`에 정량적 결과가 포함돼 있나요?
- [x] 모든 섹션에 실제 내용을 채웠나요?
- [x] 결과 표는 핵심 지표만 요약했나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- Prefix-Tuning은 거대 언어 모델의 본체 파라미터를 고정한 채, 각 계층의 attention prefix 벡터만 학습해 태스크를 적응시키는 파라미터 효율 기법입니다.
- GPT-2 Large를 WebNLG 데이터-투-텍스트에 적용했을 때 전체 파라미터의 약 0.1%만 업데이트하면서도 BLEU 44.0으로 full fine-tuning(45.6)에 1.6점만 뒤졌습니다.
- BART 기반 XSum 요약에서도 ROUGE-L 21.9로 full fine-tuning 대비 0.9점 낮은 수준을 유지해, 단일 백본으로 여러 생성 태스크를 빠르게 전환할 수 있는 가능성을 보여줍니다.

## 2. 배경 & 동기
- 거대 언어 모델을 태스크마다 전체 fine-tuning 하면 수억 개 이상의 파라미터를 학습해야 해 저장·배포 비용이 기하급수적으로 증가합니다.
- 프롬프트 기반 적응은 입력에 몇 개의 토큰을 붙여 재활용하는 간단한 방법이지만, discrete 프롬프트는 탐색 공간이 거대하고 최적화를 수동 템플릿에 의존합니다.
- Prefix-Tuning은 연속 벡터로 된 prefix를 학습해 discrete 프롬프트의 제약을 제거하고, 동시에 모델 내부 구조를 건드리지 않고도 태스크 조건을 주입하려는 시도입니다.

## 3. 방법론
### 3.1 전체 구조
- 입력 시퀀스 앞단에 길이 $m$의 학습 가능한 prefix 토큰을 붙이고, 이후 원본 입력 토큰을 그대로 이어 붙여 언어 모델에 전달합니다.
- prefix 토큰은 각 Transformer 블록의 key/value 캐시에 독립적으로 주입돼, 모든 레이어가 태스크 조건을 공유하도록 설계되었습니다.
- 생성 과정에서 모델은 prefix가 제시하는 조건 정보를 참고해 출력 토큰을 점진적으로 생성합니다.

### 3.2 핵심 기법
- prefix는 레이어마다 분리된 learnable matrix로 저장돼 self-attention의 key/value에 concatenate되며, attention score를 통해 downstream 토큰이 조건을 읽어갑니다.
- optimization 단계에서는 prefix 파라미터만 gradient를 통해 업데이트하고, 나머지 언어 모델 파라미터는 완전히 고정해 파라미터 효율성을 극대화합니다.
- 안정성을 위해 prefix 초기화를 실제 토큰 임베딩이나 랜덤 가우시안으로 실험했고, layer-normalization 및 reparameterization(MLP projection)을 통해 훈련 안정성을 확보했습니다.

### 3.3 학습 및 구현 세부
- GPT-2 Large/XL과 BART Large를 백본으로 사용했으며, prefix 길이는 10~20 사이에서 grid search로 결정했습니다.
- AdamW 옵티마이저(learning rate 5e-5)와 gradient clipping(1.0)을 적용했고, prefix 파라미터에만 weight decay 0.01을 걸어 과적합을 방지했습니다.
- 공식 코드는 PyTorch로 공개돼 있으며, huggingface Transformers 상에서 손쉽게 재현 가능한 스크립트를 제공합니다.

## 4. 실험 & 결과
### 4.1 설정
- **데이터-투-텍스트**: WebNLG 2017, E2E NLG; 평가 지표는 BLEU/ROUGE-L.
- **요약**: XSum 데이터셋을 BART Large로 fine-tuning해 비교.
- Baseline은 full fine-tuning과 prompt-tuning variants이며, 학습 시 GPU 메모리는 prefix 길이에 비례해 약 6~8GB 추가로 요구됩니다.

### 4.2 주요 결과표
| Task & Metric | Full Fine-tuning | Prefix-Tuning | Δ |
| ------------- | ---------------- | ------------- | -- |
| WebNLG (BLEU) | 45.6 | 44.0 | -1.6 |
| XSum (ROUGE-L) | 22.8 | 21.9 | -0.9 |

### 4.3 추가 분석
- prefix 길이를 5→20으로 늘리면 BLEU가 꾸준히 상승하지만 30 이상에서는 오버피팅으로 다시 감소했습니다.
- 각 Transformer 레이어에 prefix를 주입하는 ablation에서, 상위 레이어만 사용하는 경우 성능이 2~3점 하락해 전 레이어 적용이 필수적임을 확인했습니다.
- discrete prompt와 비교하면 탐색 비용 없이 gradient 기반으로 수렴하며, 소수 샷(few-shot) 설정에서도 모델이 빠르게 수렴합니다.

## 5. 의의 & 한계
- 파라미터 효율 학습이 생성 태스크에서도 경쟁력 있는 성능을 낼 수 있다는 것을 실증해 이후 LoRA·P-Tuning v2 같은 연구의 출발점이 되었습니다.
- prefix 길이나 초기화 전략에 민감해 하이퍼파라미터 튜닝 비용이 여전히 존재하며, 분류 등 비생성 태스크에는 직접 적용하기 어렵습니다.
- 복잡한 reasoning을 요구하는 작업에서는 full fine-tuning 대비 격차가 남아 있어, prefix만으로는 충분한 표현력을 확보하기 힘든 경우가 존재합니다.

## 6. 개인 평가
**강점**: 파라미터 효율성과 성능 간 균형을 명확히 검증했고, 레이어 전반에 prefix를 투입하는 설계를 체계적으로 분석했습니다.  
**약점**: 생성 태스크 중심의 실험이라 일반화된 벤치마크 부족, prefix 길이 튜닝에 대한 가이드가 제한적입니다.  
**적용 가능성**: 파인튜닝이 어려운 환경에서 다수의 생성 태스크를 한 모델로 빠르게 지원해야 할 때 매우 유용합니다.  
**추천도**: ★★★★☆ (파라미터 효율 전략을 탐구하는 연구자에게 필독)

## 7. 참고 자료
- 원문: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- 코드: [GitHub - XiangLi1999/PrefixTuning](https://github.com/XiangLi1999/PrefixTuning)
- 데이터: [WebNLG 2017](https://webnlg-challenge.loria.fr/challenge_2017/), [E2E NLG](https://github.com/tuetschek/e2e-dataset), [XSum](https://github.com/EdinburghNLP/XSum)
