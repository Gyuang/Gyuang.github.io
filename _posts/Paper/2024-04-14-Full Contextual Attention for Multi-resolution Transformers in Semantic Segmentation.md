---
published: true
title:  "Full Contextual Attention for Multi-resolution Transformers in Semantic Segmentation" 
excerpt: "논문요약"

categories:
  - Transformer
tags:
  - [Transformer, Multi-resolution, ]

toc: true
toc_sticky: true
 
date: 2024-04-14
last_modified_at: 2024-04-14

---

## Introduction

트랜스포머의 효과성: 트랜스포머는 입력 요소 간의 장거리 상호작용을 모델링 할 수 있는 능력 덕분에 다양한 분야에서 매우 효과적이었습니다. 이는 의미론적 분할과 같은 작업에 필수적인 요소입니다.

고해상도 이미지의 도전: 트랜스포머를 고해상도 이미지에 적용할 때의 확장성은 주의 메커니즘의 제곱 복잡도로 인해 제한됩니다. 이미지 패치의 수가 증가함에 따라 계산 비용이 많이 듭니다.

다중 해상도 접근법: 확장성 문제를 해결하기 위해 다중 해상도 전략으로 전환되었습니다. 이 접근법은 고해상도 특징 맵의 하위 창에서 주의를 계산하여 계산 요구 사항을 줄입니다. 그러나 이 기술은 일반적으로 해당 하위 창 내에서의 상호작용을 제한하여 더 넓은 맥락 정보를 포착할 수 있는 능력을 제한합니다.

GLAM (다중 해상도 트랜스포머에서의 글로벌 주의): GLAM의 도입은 이러한 제한을 극복하고자 합니다. 모든 규모에서 글로벌 주의를 허용함으로써 모델이 정확한 의미론적 분할에 필수적인 세밀한 공간적 세부 사항과 더 넓은 맥락 정보를 통합할 수 있도록 합니다.

Swin 아키텍처 통합: GLAM을 Swin 트랜스포머 아키텍처에 통합함으로써 작은 지역 창을 넘어 확장된 범위의 상호작용이 가능해집니다. 이 통합은 보행자, 자동차, 건물과 같은 다양한 요소에 대한 주의를 제공함으로써 기본 Swin 모델에 비해 더 나은 분할 성능을 가능하게 합니다. 이러한 방법은 복잡한 환경에서의 의미론적 분할 작업에 필수적인 상세하고 대규모 이미지 데이터를 효과적으로 처리할 수 있는 모델의 능력을 향상시키는 중요한 발전을 나타냅니다.


## Related Work 

**Multi-resolution transformers**

  최근 인공지능 분야에서는 트랜스포머 기반의 아키텍처가 이미지 인식과 처리 분야에서 두각을 나타내고 있습니다. 이 중에서도 다중 해상도를 활용한 트랜스포머 아키텍처는 보다 효율적이고 정교한 방식으로 이미지를 분석할 수 있는 새로운 가능성을 제시하고 있습니다. 이러한 아키텍처들은 기존의 단일 해상도를 사용하는 Vision Transformer (ViT)의 한계를 극복하고자 개발되었습니다.

  T2T ViT: 토큰 집계를 통한 풍부한 의미적 특징 맵 구축
  T2T (Tokens-to-Token) ViT는 기존 ViT의 구조를 개선하여 각 토큰을 더 큰 맥락에서 집계하고, 이를 통해 더 풍부한 의미적 특징 맵을 생성합니다. 이 과정은 이미지의 각 부분이 전체와 어떻게 연관되는지 더욱 정밀하게 파악할 수 있도록 돕습니다.

  TnT와 CrossViT: 미세 조정과 거친 해상도의 이중 처리
  TnT (Transformer-in-Transformer)와 CrossViT는 각각 미세 조정과 거친 해상도를 위한 두 개의 트랜스포머를 사용합니다. 이는 이미지의 다양한 스케일을 동시에 처리하여, 세밀한 디테일과 넓은 범위의 맥락 정보를 모두 포착할 수 있는 구조를 제공합니다.

  PvT: 피라미드 구조를 통한 공간 복잡성 감소
  PvT (Pyramidal Vision Transformer)는 완전한 피라미드 구조를 가진 첫 번째 백본으로, 창 기반 트랜스포머를 기반으로 합니다. 이 아키텍처는 고해상도 이미지 처리를 용이하게 하면서도 공간적 복잡성을 줄이고, 풍부한 특징 맵을 구축할 수 있도록 설계되었습니다.

  Swin Transformer와 그 변형: 이동된 창을 사용한 정보 공유 개선
  Swin Transformer는 창을 이동시키면서 각 창 간의 정보를 교차하여 공유하는 기법을 사용합니다. 이는 로컬 영역과 글로벌 영역 간의 연결을 강화하며, 전체 이미지에 대한 깊이 있는 이해를 가능하게 합니다.

  Twins와 CvT: 새로운 접근 방식의 도입
  Twins는 미세 조정과 거친 해상도 트랜스포머를 교차 배치하는 방식으로, CvT (Convolutional vision Transformer)는 선형 임베딩 대신 컨볼루션을 사용하여 효율적인 처리를 도모합니다.

  이러한 다중 해상도 트랜스포머 아키텍처들은 이미지 처리 분야에서 높은 정밀도와 효율성을 제공하며, 특히 복잡한 장면에서의 의미론적 분할과 같은 고도의 작업에 탁월한 성능을 발휘합니다. 

**Efficient Self-Attention**

 변형기(Transformer) 모델에서 긴 시퀀스를 처리하는 것은 오랜 도전 과제였습니다. 이는 변형기의 기본 자기 주의 메커니즘이 시퀀스 길이에 따라 제곱으로 복잡도가 증가하기 때문입니다. 이러한 문제를 해결하기 위해 많은 연구가 효율적인 자기 주의 메커니즘을 설계하는 데 집중하고 있습니다. 이러한 접근법은 크게 네 가지 주요 카테고리로 분류할 수 있습니다.

  1. 희소 근사(Sparse Approximation)
  희소 근사 방법은 주의 매트릭스를 희소하게 만들어 계산을 간소화합니다. 이 중 창 기반 패치 추출 방식을 사용하는 비전 변형기는 간단하지만 효과적인 주의 계산 방법을 제공하여 최근 주목을 받고 있습니다.

  2. 저랭크 근사(Low-Rank Approximation)
  저랭크 근사 방식은 주의 매트릭스를 낮은 차원으로 근사화하여 처리합니다. 예를 들어, Linformer는 이 카테고리의 대표적인 예로, 더 적은 계산 자원으로도 효율적인 결과를 도출할 수 있습니다.

  3. 메모리 기반 변형기(Memory-Based Transformers)
  메모리 기반 변형기는 정적 메모리로 사용될 추가 토큰의 버퍼를 구성하여 사용합니다. 이는 과거 정보를 효율적으로 저장하고 접근할 수 있도록 해서 처리 과정을 최적화합니다.

  4. 커널 기반 방법(Kernel-Based Methods)
  커널 기반 방법은 소프트맥스 커널의 선형 근사를 제공하여 계산을 단순화합니다. 이 방법은 특히 계산 속도를 빠르게 하고자 할 때 유용합니다.

  통합적 접근
  일부 비전 변형기는 여러 효율적인 주의 메커니즘을 결합하여 사용합니다. 예를 들어, 최근 ViT에서 영감을 받은 PvT는 창 기반 자기 주의와 Linformer에 가까운 주의 근사를 기반으로 합니다. 또한, ViL은 글로벌 토큰의 축소된 집합을 사용하여 희소한 주의를 균형있게 처리하고, 입력 이미지의 글로벌 표현을 추출합니다.

  이러한 다양한 효율적인 자기 주의 기법은 변형기가 더 긴 시퀀스를 효과적으로 처리할 수 있도록 도와주며, NLP 작업 뿐만 아니라 이미지 처리와 같은 다른 분야에서도 그 효용성을 발휘하고 있습니다.

## Method 

### GLAM
<p align="center">
  <img src="/assets/images/paper/transformer/GLAM.png" alt="ADAPT Architecture" style="width: 100%;">
</p>

  1. Global Tokens

    각 트랜스포머 블록에 대한 입력으로서, $$ z_l $$은 각 창의 토큰과 해당 글로벌 토큰이 결합된 배치를 포함합니다. 입력 구조는 다음과 같이 표현됩니다:

    $$ z_l \in \mathbb{R}^{Nr \times (Ng + Np) \times C} $$

    - $$ N_r $$ is the number of windows in the feature map.
    - $$ N_p $$ is the number of patches per window.
    - $$ N_g $$ is the dimension of the global tokens.
    - $$ C $$ is the dimension of the tokens.
    - $$ w^{l}_{k} $$ is sequence of windows after being processed $$l^th$$ GLAM-transformer block. 
  
    $$
    \forall k \in [1..Nr], \quad z_k = 
    \begin{bmatrix}
    g_k' \\
    w_k'
    \end{bmatrix}
    \in \mathbb{R}^{(Ng + Np) \times C}.
    $$

  2. GLAM-Transformer
  <p align="center">
    <img src="/assets/images/paper/transformer/GLAM-Transformer.png" alt="ADAPT Architecture" style="width: 100%;">
  </p>
  3. Non-Local Upsampling





