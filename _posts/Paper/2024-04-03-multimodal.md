---
title:  "Deep Multimodal Guidance for Medical Image Classification" 
excerpt: "논문요약"

categories:
  - Multimodal
tags:
  - [Multimodal, Brain, Classification, Miccai]

toc: true
toc_sticky: true
 
date: 2024-04-03
last_modified_at: 2024-04-03

---

## Related Work 

### Multimodal classification

다중 모달 예측을 위한 딥러닝(DL) 기반 연구들이 주로 분류 작업에 집중하고 있으며, 이는 테스트 시 여러 모달리티의 존재를 포함한다고 언급합니다. 연구의 주된 초점은 다양하고 중복되는 기능을 효율적으로 통합하는 최적의 통합 전략을 찾는 것입니다. 특히, 입력 수준에서 데이터 통합이 가능하긴 하지만, 대부분의 작업이 피처 수준에서 통합을 선호하며, 이는 입력의 차원 불일치와 통합의 유연성 때문입니다. 또한, 일부 작업은 앙상블 학습 전략을 활용하는 결정 수준 통합 프레임워크를 사용합니다.

### Image Translation

 이미지 대 이미지 변환(또는 스타일 전환) 모델 학습은 열등한 모달리티를 우수한 모달리티로 변환하는 방법을 고려할 수 있습니다. 그러나 이 분야에서 큰 성공을 거두었음에도 불구하고, 다중 모달 의료 이미징의 맥락에서는 차원의 차이(예: 2D에서 3D로)와 소스와 타깃 사이의 크기 차이(예: 백만에서 십억 개의 복셀)로 인해 복잡하거나 이상적이지 않습니다. 이미지 변환은 중간 번역 작업만 최적화하며, 제안된 방법처럼 최종 분류 작업도 해결하지 않습니다.

### Student-Teacher (S-T) learning
 
Knowledge Distillation(KD)으로도 알려진 S-T 학습은 한 모델에서 다른 모델로 지식을 전달하는 것을 목표로 하며, 주로 라벨이 적거나 없는 애플리케이션과 모델 압축에 적용됩니다. Cross Modal Distillation은 Teacher의 모달리티 특정 표현을 활용하여 학생 모델에 지식을 증류하는 것을 목표로 합니다. 이러한 애플리케이션 대부분은 시각적 및 오디오 데이터 간 KD에 초점을 맞추지만, 교차 모달 의료 이미지 분석을 위한 KD 방법은 주로 분할에 초점을 맞춥니다.

요약하자면, 이전 작업들은 추론 성능을 향상시키기 위해 추론 중에 다중 모달 의료 이미지를 입력으로 사용하는 반면, 본 연구의 기여는 훈련 중에 다중 모달 데이터를 활용하여 단일 모달 입력만으로 추론 성능을 향상시키는 것입니다.

## Method 


<p align="center">
  <img src="/assets/images/paper/multimodal/deep_multimodal_guidance.png" alt="deep multimodal guidance" style="width: 100%;">
</p>
 
# Deep Multimodal Guidance Model

위의 Figure에서, 아래첨자 I와 S는 각각 Inferior, Superior한 modal을 뜻합니다.

## 기본적인 개념

### (a) Inferior한 modal과 Superior한 modal 학습
Inferior한 modal과 Superior한 modal을 각각 따로 인코더와 디코더를 이용하여 classification task를 학습시킵니다.

### (b) Guidance 모델 학습
(a)에서 학습한 인코더에서 Inferior한 modal에서 나온 latent representation을 Superior한 modal의 latent representation으로 바꾸는 Guidance 모델을 학습시킵니다.

### (c) Fine-tuning Guidance 모델
(a)에서 학습한 $$E_I$$와 $$D_S$$를 frozen 시킨 채로 이용하여 다시 한번 $$G$$를 fine-tuning 시킵니다.

### (d) 결합 Decoder 학습
마지막 단계로 Inferior한 modal을 이용하여 $$E_I$$와 $$G$$를 frozen시킨 채로 두 개의 representation을 모두 사용하는 $$D_C$$를 새로 만들어 학습시킵니다.

## Conclusion

1. Multi-modal medical image를 입력으로 사용하지않고 latent space의 representation을 guidance를 이용하여 modal간의 정보 교환을 진행하여 Uni-modal을 인풋으로 Inference 진행할 수 있도록 만들었습니다.

2. 훈련시키는 단계가 너무 많은데 좀 줄일수는 없을까? 