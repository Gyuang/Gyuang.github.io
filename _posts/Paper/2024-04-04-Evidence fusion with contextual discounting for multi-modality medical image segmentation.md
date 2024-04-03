---
title:  "Evidence fusion with contextual discounting for multi-modality medical image segmentation" 
excerpt: "논문요약"

categories:
  - Multimodal
tags:
  - [Multimodal, Dempster-Shafer theory, Uncertainty quantification, Segmentation, Brain, Miccai]

toc: true
toc_sticky: true
 
date: 2024-04-04
last_modified_at: 2024-04-04

---

## Related Work 


## Method 


<p align="center">
  <img src="/assets/images/paper/multimodal/Evidence fusion with contextual discounting for multi-modality medical image segmentation.png" alt="deep multimodal guidance" style="width: 100%;">
</p>
 
본 논문에서 제안된 방법은 맥락적 할인과 증거 통합을 통합한 새로운 딥러닝 프레임워크를 도입하여 다양한 모달리티에서 의료 이미지의 segmentation을 향상시킵니다. 이 프레임워크는 Encoder-Decoder Feature Extraction Module, Evidential Segmentation Module, 그리고 Multi-modality Evidence Fusion Module의 세 가지 주요 구성 요소로 이루어져 있습니다.

### Encoder-Decoder Feature Extraction Module
인코더-디코더 구조는 다양한 모달리티에서 의료 이미지의 특징을 포착하고 재구성하기 위해 설계되었습니다. 인코더는 입력 이미지를 낮은 차원의 특징 공간으로 압축하여 segmentation에 필요한 핵심 정보를 포착합니다. 디코더는 그 특징 공간에서 segmentation된 이미지를 재구성합니다.

- 입력: 다중 모달리티 의료 이미지.
- 출력: 이미지를 대표하는 특징 맵.

### Evidential Segmentation Module
이 모듈은 Dempster-Shafer 이론(DST)을 적용하여 추출된 특징에 기반한 각 픽셀(또는 복셀)의 segmentation 클래스에 대한 불확실성을 정량화합니다. 각 픽셀에 대해 각 모달리티별로 신념 함수를 계산하여 각 가능한 segmentation 클래스에 대한 신뢰도를 나타냅니다.

### Basic Concepts of DST

Hypotheses Set (Ω): 모든 가능한 segmentation 클래스의 유한 집합. 
$$\Omega = \{\omega_1, \omega_2, ..., \omega_K\}$$

Mass Function (m): Ω의 각 부분집합에 할당된 신념의 양을 나타내며, $\sum_{A \subseteq \Omega} m(A) = 1$을 만족합니다.
$$ m: 2^\Omega \rightarrow [0, 1] $$
Belief and Plausibility Functions: 어떤 가설 $A \subseteq \Omega$에 대해 신념의 하한과 상한을 나타냅니다.

### Mass Function Computation
각 픽셀과 각 모달리티에 대해, 질량 함수는 특징 벡터와 각 클래스의 전형적인 특징을 나타내는 미리 정의된 프로토타입 중심 사이의 거리를 기반으로 계산됩니다.


- Input: Feature vectors from the encoder-decoder module.
- Output: Mass functions representing the evidence of segmentation classes.


### Multi-modality Evidence Fusion Module
이 통합 모듈은 맥락적 정보와 Dempster의 결합 규칙을 기반으로 하는 할인 메커니즘을 적용하여 각 픽셀에 대해 모든 모달리티의 증거를 결합합니다.

### Contextual Discounting
각 모달리티의 증거는 맥락을 고려하여 다른 클래스에 대한 신뢰성을 반영하는 할인율 벡터에 의해 할인됩니다.

### Dempster’s Rule of Combination
모든 모달리티에서 할인된 증거는 Dempster의 규칙을 사용하여 결합되어 각 픽셀에 대한 최종적인 집계된 belief function를 생성하며, 이는 그 픽셀의 segmentation 클래스에 대한 불확실성을 정량화합니다.


- Input: Discounted mass functions from all modalities.
- Output: Combined belief function for each pixel.

### Loss Function
discounted Dice 지수를 기반으로 한 새로운 손실 함수가 전체 프레임워크를 훈련시키기 위해 도입되었습니다. 이 손실 함수는 segmentation 결과와 그 결과에 대한 신뢰도를 모두 고려함으로써 segmentation 정확도와 신뢰성을 극대화하려는 목표를 가집니다.


- 이 손실 함수는 모델이 정확하게 segmentation할 뿐만 아니라 높은 확률 영역에서 자신감을 가지고 불확실한 영역에서는 신중하게 행동하도록 장려합니다.

증거 통합과 맥락적 할인을 통합함으로써, 제안된 방법은 여러 모달리티에서 보완적인 정보를 효과적으로 활용하여, 특히 모호하거나 충돌하는 증거가 존재할 때 segmentation 정확도와 견고성을 향상시킵니다.
