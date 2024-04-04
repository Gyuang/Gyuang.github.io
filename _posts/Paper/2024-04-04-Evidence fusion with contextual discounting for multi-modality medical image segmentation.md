---
published: true
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
 
본 논문에서 제안된 방법은 contextual discounting과 evidence fusion을 통합한 새로운 딥러닝 프레임워크를 도입하여 다양한 모달리티에서 의료 이미지의 segmentation 성능을 향상시킵니다. 이 프레임워크는 Encoder-Decoder Feature Extraction Module, Evidential Segmentation Module, 그리고 Multi-modality Evidence Fusion Module의 세 가지 주요 구성 요소로 이루어져 있습니다.

### Encoder-Decoder Feature Extraction Module
인코더-디코더 구조는 다양한 모달리티에서 의료 이미지의 특징을 포착하고 재구성하기 위해 설계되었습니다. 인코더는 입력 이미지를 낮은 차원의 특징 공간으로 압축하여 segmentation에 필요한 핵심 정보를 포착합니다. 디코더는 그 특징 공간에서 segmentation된 이미지를 재구성합니다.

- Input: Multi-modality medical images.
- Output: Feature maps representing the images.

### Evidential Segmentation Module
이 모듈은 Dempster-Shafer 이론(DST)을 적용하여 추출된 특징에 기반한 각 픽셀(또는 복셀)의 segmentation 클래스에 대한 불확실성을 정량화합니다. 각 픽셀에 대해 각 모달리티별로 신념 함수를 계산하여 각 가능한 segmentation 클래스에 대한 신뢰도를 나타냅니다.

Evidential Segmentation (ES) 모듈은 input layer, 두 개의 hidden layer, 그리고 output layer으로 구성되어 있습니다. 이 구조는 입력 벡터를 처리하고 각 복셀의 분할 클래스에 대한 불확실성을 정량화하는 신념 함수를 출력하는 과정을 담당합니다. 

1. **Input layer**: 이 층은 입력 공간에서 프로토타입 $\mathbf{p}_i$에 해당하는 $I$개의 유닛으로 구성됩니다. 입력 벡터 $$\mathbf{x}$$에 대한 유닛 $$i$$의 활성화 $$s_i$$는 다음과 같이 정의됩니다:

   $$s_i = \alpha_i \exp(-\gamma_i \|\mathbf{x} - \mathbf{p}_i\|^2)$$

   여기서 $$\gamma_i > 0$$ 및 $$\alpha_i \in [0, 1]$$은 활성화를 조절하는 매개변수입니다.

2. **Hidden layer**: 각 프로토타입 $\mathbf{p}_i$가 제공하는 evidence를 나타내는 질량 함수 $m_i$를 계산합니다. 각 클래스 $$\omega_k$ ($$k = 1, \ldots, K$$)에 대한 질량 함수와 클래스 전체 집합 $\Omega$에 대한 질량 함수는 다음과 같이 계산됩니다:

   - 클래스 $$\omega_k$$에 대한 질량 함수:
     $$m_i(\{\omega_k\}) = u_{ik}s_i$$

   - 클래스 전체 집합 $$\Omega$$에 대한 질량 함수:
     $$m_i(\Omega) = 1 - s_i$$

   여기서 $$u_{ik}$$는 프로토타입 $i$가 클래스 $$\omega_k$$에 속하는 멤버십 정도를 나타내며, $$\sum_{k=1}^{K} u_{ik} = 1$$을 만족합니다.

3. **Output layer**: 모든 프로토타입에서 $m_1, \ldots, m_I$에 이르는 질량 함수를 Dempster의 결합 규칙을 사용하여 집계합니다. 이 과정은 각 복셀의 클래스에 대한 불확실성을 정량화하는 최종 신념 함수를 생성합니다.

이 구조를 통해 ES 모듈은 입력 벡터를 처리하여 각 복셀에 대한 분할 클래스의 불확실성을 정량화하는 신념 함수를 효과적으로 출력할 수 있습니다. 여러 프로토타입에서의 evidence를 결합함으로써, 불확실성을 정량화하고 통합하여 분할 작업의 정확성과 견고성을 향상시키는 데 기여합니다.

### Basic Concepts of DST(Dempster-Shafer Theory)

> [DST 참고 자료](https://gyuang.github.io/math/Dempster-Shafer/)

Hypotheses Set (Ω): 모든 가능한 segmentation 클래스의 유한 집합. 
$$\Omega = \{\omega_1, \omega_2, ..., \omega_K\}$$

Mass Function (m): Ω의 각 부분집합에 할당된 신념의 양을 나타내며, $$\sum_{A \subseteq \Omega} m(A) = 1$$을 만족합니다.
$$ m: 2^\Omega \rightarrow [0, 1] $$
Belief and Plausibility Functions: 어떤 가설 $$A \subseteq \Omega$$에 대해 신념의 하한과 상한을 나타냅니다.

$$
Bel(A) = \sum_{\emptyset \neq B \subseteq A} m(B)
Pl(A) = \sum_{B \cap A \neq \emptyset} m(B)
$$

### Mass Function Computation
각 픽셀과 각 모달리티에 대해, Mass Function는 특징 벡터와 각 클래스의 전형적인 특징을 나타내는 미리 정의된 프로토타입 중심 사이의 거리를 기반으로 계산됩니다.

- Input: Feature vectors from the encoder-decoder module.
- Output: Mass functions representing the evidence of segmentation classes.

### Multi-modality Evidence Fusion Module
이 통합 모듈은 맥락적 정보와 Dempster의 결합 규칙을 기반으로 하는 discount 메커니즘을 적용하여 각 픽셀에 대해 모든 모달리티의 evidence를 결합합니다.

이 논문에서는 DST의 discounting operation을 통해 소스 신뢰도를 정량화하는 문제를 다룹니다. 질량 함수 $$m$$이 $$\Omega$$에 대해 정의되고 $$\beta$$가 $[0,1]$ 내의 계수일 때, discount rate $$1-\beta$$를 사용하는 discounting operation은 $$m$$을 더 약하고 정보가 적은 질량 함수 $\beta m$으로 변환합니다. 

$$\beta m = \beta m + (1-\beta)m_?$$

여기서, $$ {}^\beta m $$ 는 $$m_?(\Omega) = 1$$로 정의된 공허한 질량 함수이며, 계수 $$\beta$$는 소스 질량 함수 $$m$$이 신뢰할 수 있다는 믿음의 정도입니다. $$\beta = 1$$일 때는 소스에서 제공된 질량 함수 $m$을 우리 지식의 설명으로 받아들이고, $\beta = 0$일 때는 거부하고 공허한 질량 함수 $m?$를 가집니다. 이 논문에서는 $$\beta \in [0,1]$$인 상황에 초점을 맞추고, Dempster의 규칙을 사용하여 부분적으로 신뢰할 수 있는 불확실한 evidence를 결합합니다.

[17]에서 제안된 바와 같이, 위의 discounting operation은 맥락적 discount으로 확장될 수 있습니다. 이 연산은 다른 맥락에서 정보 소스의 신뢰도에 대한 더 풍부한 메타 지식을 나타낼 수 있습니다. 이는 $$\beta = (\beta_1, ..., \beta_K)$$ 벡터에 의해 매개변수화되며, 여기서 $\beta_k$는 참 클래스가 $$\omega_k$$일 때 소스가 신뢰할 수 있다는 믿음의 정도입니다. discount된 질량 함수의 완전한 표현식은 [17]에 제공되며, 여기서는 나중에 사용될 해당 윤곽 함수의 표현식만 제공합니다.

$$\beta pl(\{\omega_k\}) = 1-\beta_k + \beta_k pl(\{\omega_k\}),  k= 1,...,K$$

독립적인 evidence에 의해 제공되는 여러 소스가 있을 때, discount된 evidence는 Dempster의 규칙에 의해 결합될 수 있습니다. 정보의 두 소스가 있고, 각각 $$S_1$$과 $$S_2$$에 의해 제공된 discount된 윤곽 함수가 $$\beta_1 pl_{S_1}$$과 $$\beta_2 pl_{S_2}$$이며, discount rate 벡터가 $$1-\beta_1$$과 $$1-\beta_2$$일 경우, 결합된 윤곽 함수는 $$\beta_1 pl_{S_1} \beta_2 pl_{S_2}$$의 곱에 비례합니다.

### Contextual Discounting
각 모달리티의 evidence는 맥락을 고려하여 다른 클래스에 대한 신뢰성을 반영하는 discount rate 벡터에 의해 discount됩니다.

$$m_i'(A) = \alpha_i \cdot m_i(A) + (1 - \alpha_i) \cdot m(\Omega)$$


### Dempster’s Rule of Combination
모든 모달리티에서 discount된 evidence는 Dempster의 규칙을 사용하여 결합되어 각 픽셀에 대한 최종적인 집계된 belief function를 생성하며, 이는 그 픽셀의 segmentation 클래스에 대한 불확실성을 정량화합니다.


- Input: Discounted mass functions from all modalities.
- Output: Combined belief function for each pixel.

$$(m_1 \oplus m_2)(A) = \frac{1}{1 - \kappa} \sum_{B \cap C = A} m_1(B) \cdot m_2(C)$$

여기서 $$\kappa = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)$$는 두 질량 함수 간의 degree of conflict를 나타냅니다.

### Loss Function
discounted Dice 지수를 기반으로 한 새로운 손실 함수가 전체 프레임워크를 훈련시키기 위해 도입되었습니다. 이 손실 함수는 segmentation 결과와 그 결과에 대한 신뢰도를 모두 고려함으로써 segmentation 정확도와 신뢰성을 극대화하려는 목표를 가집니다.

$$
\text{loss}_D = 1 - \frac{2 \sum_{n=1}^{N} \beta_{S_n} G_n}{\sum_{n=1}^{N} \beta_{S_n} + \sum_{n=1}^{N} G_n}
$$

여기서, $$\beta_{S_n}$$은 discount된 소스 정보를 통합함으로써 정규화된 n번째 복셀에 대한 세분화 출력을 나타내고, $$G_n$$은 n번째 복셀에 대한 ground truth를 나타내며, N은 볼륨 내 복셀의 총 수를 나타냅니다.$$ \beta_{S_n}$$은 다음과 같이 계산됩니다:

$$\beta_{S_n} = \frac{\prod_{h=1}^{H} \beta_h pl_{S_h}(\{\omega_k\})}{\sum_{k=1}^{K} \prod_{h=1}^{H} \beta_h pl_{S_h}(\{\omega_k\})}$$



이 수식에서, H는 discount된 소스의 수, $$\beta_h$$는 h번째 소스에 대한 discount rate, $$pl_{S_h}(\{\omega_k\})$$ 는 h번째 소스에서 k번째 클래스에 대한 가능성 함수, K는 세분화 클래스의 총 수를 나타냅니다.

- 이 손실 함수는 모델이 정확하게 segmentation할 뿐만 아니라 높은 확률 영역에서 자신감을 가지고 불확실한 영역에서는 신중하게 행동하도록 장려합니다.

evidence 통합과 맥락적 discount을 통합함으로써, 제안된 방법은 여러 모달리티에서 보완적인 정보를 효과적으로 활용하여, 특히 모호하거나 충돌하는 evidence가 존재할 때 segmentation 정확도와 견고성을 향상시킵니다.
