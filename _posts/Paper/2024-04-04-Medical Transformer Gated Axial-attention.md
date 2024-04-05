---
published: true
title:  "Medical Transformer: Gated Axial-Attention for Medical Image Segmentation" 
excerpt: "논문요약"

categories:
  - Transformer
tags:
  - [Transformer, attention, Brain, Segmentation]

toc: true
toc_sticky: true
 
date: 2024-04-06
last_modified_at: 2024-04-06

---
## Introduction

1. Image Segmentation에서 정확하고 general한 방법을 개발하는것은 의료 영상 분석의 주요 과제중 하나이며, computer-aided diagnosis 와 image-guided surgery systems에 필수적인 의료 스캔에서 장기나 병변을 segmentation하는 것은 의사가 정확한 진단을 내리고, 수술 절차를 계획하며, 치료 전략을 제안하는 데 도움을 줍니다.
2. 컴퓨터 비전에서 심층 합성곱 신경망(ConvNets)의 인기에 힘입어, ConvNets는 의료 이미지 분할에 빠르게 채택되었습니다. U-Net, V-Net, 3D U-Net, Res-UNet, Dense-UNet, YNet, U-Net++, KiU-Net, U-Net3+ 등 다양한 의료 영상 모달리티에 대한 이미지 및 체적 분할을 수행하기 위해 특별히 제안된 네트워크들이 있습니다. 이러한 방법들은 많은 어려운 데이터셋에서 인상적인 성능을 달성함으로써, 의료 스캔에서 장기나 병변을 분할하기 위해 ConvNets가 차별화된 특징을 학습하는 데 효과적임을 입증했습니다.
3. 그러나 ConvNets는 이미지 내에 존재하는 장거리 의존성을 모델링하는 능력이 부족합니다. 구체적으로, ConvNets의 각 합성곱 커널은 전체 이미지 내의 local 픽셀 하위 집합에만 주의를 기울이게 하여, 네트워크가 global context보다 local 패턴에 집중하도록 합니다. 이미지 피라미드, atrous 컨볼루션, 주의 메커니즘을 사용하여 ConvNets에 대한 장거리 의존성 모델링에 초점을 맞춘 연구가 있었습니다. 그러나 대다수의 이전 방법들은 의료 이미지 분할 작업에 대해 이러한 측면에 집중하지 않아 장거리 의존성 모델링에 대한 개선의 여지가 여전히 존재합니다.
4. 이에 대한 대안으로, 트랜스포머 기반 모델, 특히 MedT는 픽셀 지역 간의 장거리 의존성을 학습함으로써 잘못된 분류를 줄이는 데 효과적인 것으로 나타났습니다. 자연어 처리(NLP)에서 장거리 의존성을 인코딩할 수 있는 능력으로 인기를 얻은 트랜스포머는 컴퓨터 비전 분야에 최근 도입되었으며, 특히 분할 작업에 사용될 때 효과적입니다.
5. 이 논문에서는 특히 의료 이미지 분할을 위해 설계된 Medical Transformer(MedT)를 제안합니다. MedT는 게이트 위치-민감 축 주의 메커니즘을 사용하며, Local-Global(LoGo) 학습 전략을 채택하여 세분화된 local패치와 집중적으로 작업합니다. 이는 전체 이미지를 처리하는 것뿐만 아니라 local 패치 내의 더 세밀한 디테일에 초점을 맞추어 분할 성능을 향상시킵니다.

요약하자면, 이 논문은 (1) 작은 데이터셋에서도 잘 작동하는 게이트 위치-민감 축 주의 메커니즘을 제안하고, (2) 트랜스포머를 위한 효과적인 Local-Global(LoGo) 학습 방법론을 소개하며, (3) 의료 이미지 분할을 위해 특별히 고안된 Medical Transformer(MedT)를 제안하고, (4) 다양한 데이터셋에서 컨볼루션 네트워크 및 Fully attention architecture 보다 의료 이미지 분할 작업의 성능을 개선함을 입증합니다.

## Related work

### Axial attention 

1. **Self-Attention Mechanism**
기존의 self-Attention Mechanism은 전체 feature map에서 관련 있는 context를 볼 수 있는 장점이 있지만, 모든 patch에 대해 attention을 계산해야 하기 때문에 계산 비용이 매우 높습니다. 또한, 위치 정보의 활용이 불충분하여 non-local context를 계산할 때 위치 정보를 사용하지 않습니다.

$$ y_o = \sum_{p \in N} \text{softmax}_p (q_o^T k_p) v_p $$

2. **Stand-Alone Self Attention**

이 접근 방식에서는 모든 feature map pixel을 key로 사용하는 대신, query 주변의 MxM개만을 key로 사용하여 계산 복잡성을 줄입니다. 또한, query에 대한 relative positional encoding을 추가하여 위치 정보를 포함시킵니다. 결과적으로 각 pixel(query)은 주변 MxM 공간의 확장된 정보를 가지며, softmax 이후에도 dynamic한 prior를 생성할 수 있습니다.

$$ y_o = \sum_{p \in N_{mxm(o)}} \text{softmax}_p (q_o^T k_p + q_o^T r) v_p $$


3. **Position-Sensitivity Self Attention**

이 방식은 query뿐만 아니라 key와 value에 대해서도 relative positional encoding을 추가하여, 한 query에 대한 주변 patch의 attention만 계산할 때 위치 정보가 필수적임을 강조합니다. 추가된 positional encoding은 여러 head에 걸쳐 파라미터를 공유하여 큰 비용 증가 없이 long-range interaction과 positional information을 포함하는 position-sensitive self-attention을 생성합니다.

$$ y_o = \sum_{p \in N_{mxm(o)}} \text{softmax}_p (q_o^T k_p + q_o^T r^q + k_p^T r^k) (v_p + r^v) $$


4. **Axial Attention**

Axial Attention은 receptive field가 local constraint로 작용할 수 있는 stand-alone 메커니즘의 한계를 극복하고, global connection을 사용하여 global 정보를 포착합니다. 각 query에 대해 전체 HW를 key로 사용하는 대신 width-axis와 height-axis 방향으로 2번 적용하여 효율적인 계산을 달성합니다.

$$ y_{ij} = \sum_{w=1}^W \text{softmax}(q_{ij}^T k_{iw} + q_{ij}^T r_{iw}^q + k_{iw}^T r_{iw}^k) (v_{iw} + r_{iw}^v) $$
$$ y_ij = \sum_{w=1}^W softmax(q^{T}_ij k_iw +q^{T}_ij r^{q}_iw + q^{T}_ij r^{K}_iw ) (v_iw + r^{v}_iw) $$


## Method 

<p align="center">
  <img src="/assets/images/paper/transformer/Medical Transformer Gated Axial-Attention for Medical Image Segmentation1.png" alt="deep multimodal guidance" style="width: 100%;">
</p>

1. ** Gated axial-attention **

Axial attention 메커니즘은 비주얼 인식에 있어 뛰어난 계산 효율성으로 non-local context를 계산할 수 있으며, 위치적 편향을 메커니즘에 인코딩하고 입력 feature map 내의 장거리 상호작용을 인코딩할 수 있는 능력을 제공합니다. 그러나, 해당 모델은 대규모 분할 데이터셋에서 평가되었으며, 이는 axial attention이 key, query, 및 value에서 위치적 편향을 학습하기 쉽게 만듭니다. 그러나 의료 영상 분할에서 흔히 발생하는 소규모 데이터셋의 실험에서는 위치적 편향을 학습하기 어렵고, 따라서 장거리 상호작용을 정확히 인코딩하지 못할 수 있습니다. 학습된 상대적 위치 인코딩이 충분히 정확하지 않은 경우, 해당 key, query 및 value 텐서에 추가하면 성능이 저하될 수 있습니다. 따라서, non-local context 인코딩에서 위치적 편향이 미치는 영향을 제어할 수 있는 수정된 axial-attention 블록을 제안합니다. 제안된 수정을 통해 width 축에 적용된 자기 주의 메커니즘은 다음과 같이 공식적으로 작성될 수 있습니다:

$$ y_{ij} = \sum_{w=1}^{W} \text{softmax} \left( q_{ij}^T k_{iw} + G_Q q_{ij}^T r_{iw}^q + G_K k_{iw}^T r_{iw}^k \right) \left( G_{V1} v_{iw} + G_{V2} r_{iw}^v \right) $$

여기서, 자기 주의 수식은 추가된 게이팅 메커니즘을 포함하여 Eq. 2를 밀접하게 따릅니다. 또한, $$G_Q , G_K , G_{V1},G_{V2}$$는 학습 가능한 파라미터이며, 이들은 함께 학습된 상대적 위치 인코딩이 비-local 컨텍스트 인코딩에 미치는 영향을 제어하는 게이팅 메커니즘을 생성합니다. 일반적으로, 상대적 위치 인코딩이 정확하게 학습된 경우, 게이팅 메커니즘은 정확하지 않게 학습된 것들에 비해 높은 가중치를 할당합니다. 이러한 수정된 axial-attention 블록은 소규모 데이터셋에서도 위치적 편향의 영향을 효과적으로 제어하며 장거리 상호작용의 인코딩을 개선할 수 있습니다.

2. ** Local-Global Training **

의료 영상 분할과 같은 작업에 있어서 패치 기반의 트랜스포머 학습은 빠르지만, 패치별 학습만으로는 충분하지 않습니다. 패치별 학습은 네트워크가 패치 간 픽셀의 정보나 의존성을 학습하는 데 제한을 둡니다. 이미지에 대한 전반적인 이해를 향상시키기 위해, 두 가지 branch(branch)를 네트워크에 사용할 것을 제안합니다: 하나는 이미지의 원본 해상도에서 작동하는 global branch와 다른 하나는 이미지의 패치에서 작동하는 local branch입니다. global branch에서는 장거리 의존성을 모델링하기에 충분하다고 관찰되는 트랜스포머 모델의 첫 몇 블록으로 gated axial transformer 레이어의 수를 줄입니다. local branch에서는 원본 이미지의 차원 I의 1/4 크기인 16개의 패치를 생성합니다. local branch의 각 패치는 네트워크를 통해 전달되며, 출력 feature map은 위치에 따라 다시 샘플링되어 최종 출력 feature map을 얻습니다. 두 branch의 출력 feature map은 더해진 후 1×1 합성곱 레이어를 통과하여 출력 분할 마스크를 생성합니다. 이 전략은 global branch가 고수준 정보에 집중하고 local branch가 더 세밀한 세부 사항에 집중할 수 있게 하여 성능을 향상시킵니다.

제안된 Medical Transformer (MedT)는 기본 블록으로 gated axial attention 레이어를 사용하며, LoGo 전략을 학습에 사용합니다. 