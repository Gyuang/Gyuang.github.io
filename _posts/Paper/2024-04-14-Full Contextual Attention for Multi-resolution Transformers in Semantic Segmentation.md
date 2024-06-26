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

transformer의 효과성: transformer는 입력 요소 간의 장거리 상호작용을 모델링 할 수 있는 능력 덕분에 다양한 분야에서 매우 효과적이었습니다. 이는 의미론적 segmentation과 같은 작업에 필수적인 요소입니다.

고해상도 이미지의 도전: transformer를 고해상도 이미지에 적용할 때의 확장성은 attention 메커니즘의 제곱 복잡도로 인해 제한됩니다. 이미지 패치의 수가 증가함에 따라 계산 비용이 많이 듭니다.

다중 해상도 접근법: 확장성 문제를 해결하기 위해 다중 해상도 전략으로 전환되었습니다. 이 접근법은 고해상도 특징 맵의 하위 window에서 attention를 계산하여 계산 요구 사항을 줄입니다. 그러나 이 기술은 일반적으로 해당 하위 window 내에서의 상호작용을 제한하여 더 넓은 맥락 정보를 포착할 수 있는 능력을 제한합니다.

GLAM (다중 해상도 transformer에서의 글로벌 attention): GLAM의 도입은 이러한 제한을 극복하고자 합니다. 모든 규모에서 글로벌 attention를 허용함으로써 모델이 정확한 의미론적 segmentation에 필수적인 세밀한 공간적 세부 사항과 더 넓은 맥락 정보를 통합할 수 있도록 합니다.

Swin 아키텍처 통합: GLAM을 Swin transformer 아키텍처에 통합함으로써 작은 지역 window을 넘어 확장된 범위의 상호작용이 가능해집니다. 이 통합은 보행자, 자동차, 건물과 같은 다양한 요소에 대한 attention를 제공함으로써 기본 Swin 모델에 비해 더 나은 segmentation 성능을 가능하게 합니다. 이러한 방법은 복잡한 환경에서의 의미론적 segmentation 작업에 필수적인 상세하고 대규모 이미지 데이터를 효과적으로 처리할 수 있는 모델의 능력을 향상시키는 중요한 발전을 나타냅니다.


## Related Work 

**Multi-resolution transformers**

  최근 인공지능 분야에서는 transformer 기반의 아키텍처가 이미지 인식과 처리 분야에서 두각을 나타내고 있습니다. 이 중에서도 다중 해상도를 활용한 transformer 아키텍처는 보다 효율적이고 정교한 방식으로 이미지를 분석할 수 있는 새로운 가능성을 제시하고 있습니다. 이러한 아키텍처들은 기존의 단일 해상도를 사용하는 Vision Transformer (ViT)의 한계를 극복하고자 개발되었습니다.

  T2T ViT: 토큰 집계를 통한 풍부한 의미적 특징 맵 구축
  T2T (Tokens-to-Token) ViT는 기존 ViT의 구조를 개선하여 각 토큰을 더 큰 맥락에서 집계하고, 이를 통해 더 풍부한 의미적 특징 맵을 생성합니다. 이 과정은 이미지의 각 부분이 전체와 어떻게 연관되는지 더욱 정밀하게 파악할 수 있도록 돕습니다.

  TnT와 CrossViT: 미세 조정과 거친 해상도의 이중 처리
  TnT (Transformer-in-Transformer)와 CrossViT는 각각 미세 조정과 거친 해상도를 위한 두 개의 transformer를 사용합니다. 이는 이미지의 다양한 스케일을 동시에 처리하여, 세밀한 디테일과 넓은 범위의 맥락 정보를 모두 포착할 수 있는 구조를 제공합니다.

  PvT: 피라미드 구조를 통한 공간 복잡성 감소
  PvT (Pyramidal Vision Transformer)는 완전한 피라미드 구조를 가진 첫 번째 백본으로, window 기반 transformer를 기반으로 합니다. 이 아키텍처는 고해상도 이미지 처리를 용이하게 하면서도 공간적 복잡성을 줄이고, 풍부한 특징 맵을 구축할 수 있도록 설계되었습니다.

  Swin Transformer와 그 변형: 이동된 window을 사용한 정보 공유 개선
  Swin Transformer는 window을 이동시키면서 각 window간의 정보를 교차하여 공유하는 기법을 사용합니다. 이는 로컬 영역과 글로벌 영역 간의 연결을 강화하며, 전체 이미지에 대한 깊이 있는 이해를 가능하게 합니다.

  Twins와 CvT: 새로운 접근 방식의 도입
  Twins는 fine and coarse transformer를 교차 배치하는 방식으로, CvT (Convolutional vision Transformer)는 선형 임베딩 대신 컨볼루션을 사용하여 효율적인 처리를 도모합니다.

  이러한 다중 해상도 transformer 아키텍처들은 이미지 처리 분야에서 높은 정밀도와 효율성을 제공하며, 특히 복잡한 장면에서의 의미론적 segmentation과 같은 고도의 작업에 탁월한 성능을 발휘합니다. 

**Efficient Self-Attention**

Transformer 모델에서 긴 시퀀스를 처리하는 것은 오랜 도전 과제였습니다. 이는 transformer의 기본 self attention 메커니즘이 시퀀스 길이에 따라 제곱으로 복잡도가 증가하기 때문입니다. 이러한 문제를 해결하기 위해 많은 연구가 효율적인 self attention 메커니즘을 설계하는 데 집중하고 있습니다. 이러한 접근법은 크게 네 가지 주요 카테고리로 분류할 수 있습니다.

  1. 희소 근사(Sparse Approximation)
  희소 근사 방법은 attention 매트릭스를 희소하게 만들어 계산을 간소화합니다. 이 중 window 기반 패치 추출 방식을 사용하는 비전 transformer는 간단하지만 효과적인 attention 계산 방법을 제공하여 최근 주목을 받고 있습니다.

  2. 저랭크 근사(Low-Rank Approximation)
  저랭크 근사 방식은 attention 매트릭스를 낮은 차원으로 근사화하여 처리합니다. 예를 들어, Linformer는 이 카테고리의 대표적인 예로, 더 적은 계산 자원으로도 효율적인 결과를 도출할 수 있습니다.

  3. 메모리 기반 transformer(Memory-Based Transformers)
  메모리 기반 transformer는 정적 메모리로 사용될 추가 토큰의 버퍼를 구성하여 사용합니다. 이는 과거 정보를 효율적으로 저장하고 접근할 수 있도록 해서 처리 과정을 최적화합니다.

  4. 커널 기반 방법(Kernel-Based Methods)
  커널 기반 방법은 소프트맥스 커널의 선형 근사를 제공하여 계산을 단순화합니다. 이 방법은 특히 계산 속도를 빠르게 하고자 할 때 유용합니다.

  통합적 접근
  일부 비전 transformer는 여러 효율적인 attention 메커니즘을 결합하여 사용합니다. 예를 들어, 최근 ViT에서 영감을 받은 PvT는 window 기반 self attention와 Linformer에 가까운 attention 근사를 기반으로 합니다. 또한, ViL은 글로벌 토큰의 축소된 집합을 사용하여 희소한 attention를 균형있게 처리하고, 입력 이미지의 글로벌 표현을 추출합니다.

  이러한 다양한 효율적인 self attention 기법은 transformer가 더 긴 시퀀스를 효과적으로 처리할 수 있도록 도와주며, NLP 작업 뿐만 아니라 이미지 처리와 같은 다른 분야에서도 그 효용성을 발휘하고 있습니다.

## Method 

<p align="center">
  <img src="/assets/images/paper/transformer/GLAM.png" alt="GLAM Architecture" style="width: 100%;">
</p>

### Global Tokens      
  
각 transformer 블록에 대한 입력으로서, $$ z_l $$은 각 window의 토큰과 해당 글로벌 토큰이 결합된 배치를 포함합니다. 입력 구조는 다음과 같이 표현됩니다:      

$$ z_l \in \mathbb{R}^{Nr \times (Ng + Np) \times C} $$   

- $$ N_r $$ is the number of windows in the feature map.     
- $$ N_p $$ is the number of patches per window.     
- $$ N_g $$ is the dimension of the global tokens.     
- $$ C $$ is the dimension of the tokens.    
- $$ w^{l}_{k} $$ is sequence of windows after being processed $$l^{th}$$ GLAM-transformer block.         

$$     \forall k \in [1..Nr], \quad z_k =      \begin{bmatrix}     g_k' \\     w_k'     \end{bmatrix}     \in \mathbb{R}^{(Ng + Np) \times C}.     $$

### GLAM-Transformer
<p align="center">
  <img src="/assets/images/paper/transformer/GLAM-Transformer.png" alt="GLAM-Transformer Architecture" style="width: 100%;">
</p>

GLAM-Transformer는 계층적 구조에서 window(window)간 통신을 가능하게 하며, 각 window에서 시각적 토큰이 자신의 로컬 통계를 통해 정보를 잡아내고, 글로벌 토큰을 통해 다른 window과의 정보를 주고받습니다. GLAM-Transformer 블록은 W-MSA(window 기반 selfattention)와 G-MSA(글로벌 selfattention) 단계를 통해 입력을 처리하고, 결과적으로 전체 이미지 영역 간의 상호작용을 모든 해상도에서 나타냅니다. 글로벌 토큰은 모든 window에 걸쳐 연산되어 전체적인 맥락을 이해하는 데 중요한 역할을 합니다.

즉 첫번째 단계에서 Global Token을 window의 patch 처럼 취급해서 concat하여 붙혀주고, W_MSA를 진행한 뒤에 각 윈도우의 global token들 끼리 G_MSA를 진행하는 것이다.

$$\hat{z^l} = W-MSA(z^{l-1})$$

$$g^l = G-MSA(\hat{g}^l)$$ 

$$z^l = \begin{bmatrix} g^{l^T}_k & \hat{w}^{l^T}_k \end{bmatrix}^T$$

$$A^{l}_r$$ 은 transformer block l에서 window r에대한  attention matrix를 뜻합니다. 

$$ A^{l}_r = \begin{bmatrix} A_{r,gg}^l & A_{r,gw}^l \\ A_{r,wg}^l & A_{r,ww}^l \end{bmatrix}$$

정사각 행렬 $$ A_{l_r,gg} \in \mathbb{R}^{N_g \times N_g} $$과 $$ A_{l_r,ww} \in \mathbb{R}^{N_p \times N_p} $$은 각각 글로벌 토큰과 공간 토큰 자체에 대한 attention를 나타냅니다. 행렬 $$ A_{l_r,gw} \in \mathbb{R}^{N_g \times N_p} $$과 $$ A_{l_r,wg} \in \mathbb{R}^{N_p \times N_g} $$은 로컬과 글로벌 토큰 사이의 교차 attention 행렬입니다. $$ B_l \in \mathbb{R}^{(N_r \cdot N_g) \times (N_r \cdot N_g)} $$을 모든 글로벌 토큰 시퀀스에서 글로벌 attention를 나타내는 행렬로 정의하고, $$ B_{l_{ij}} \in \mathbb{R}^{N_g \times N_g} $$를 창 i와 j 사이의 글로벌 토큰 간 attention를 나타내는 부분 행렬로 정의합니다.


$$g_{r}^l = \sum_{n=1}^{N_r} B_{rn}^{l} \hat{g}_n^l \\
          = \sum_{n=1}^{N_r} B_{rn}^{l} (A_{r,gg}^{l} g_r^{l-1} + A_{r,gw}^{l} w_r^{l-1})$$


$$
g_{k,r}^{l} = \sum_{r'=1}^{N_r} \sum_{i=1}^{N_p} \left (\sum_{j=1}^{N_g} b_{k,r,j,r'}  a_{j,r',(i+N_g)} w_{i,r'}^{l-1}\right ) + \sum_{r'=1}^{N_r}\left (\sum_{j=1}^{N_g} b_{k,r,j,r'} \sum_{i=1}^{N_g} a_{j,r',i} g_{i,r'}^{l-1}\right )
$$

이는 글로벌 attention 행렬 $$ G_k \in \mathbb{R}^{(N_r \cdot N_p) \times (N_r \cdot N_p)} $$으로 이어지며, 이 행렬은 $$ k $$번째 글로벌 토큰에 연관된 것으로 다음과 같이 주어집니다:
$$ [G_k]_{r',i} = \sum_{j=1}^{N_g} b_{k,r,j,r'} a_{j,r',(i+N_g)} + \sum_{j=1}^{N_g} b_{k,r,j,r'} \sum_{i=1}^{N_g} a_{j,r',i} $$
위의 식은 $$ l $$번째 GLAM-transformer 블록에서 $$ k $$번째 글로벌 토큰 $$ g_{k,r}^l $$의 임베딩을 제공하며, 모든 특징 맵 창의 시각적 토큰 $$ w_{i,r'}^{l-1} $$ (첫 번째 행)과 모든 글로벌 토큰 $$ g_{i,r'}^{l-1} $$ (두 번째 행)에 대해 나타냅니다. 이 두 항은 은 글로벌 임베딩 $$ g_{k,r}^l $$이 해상도에 관계없이 모든 이미지 영역 간의 상호작용을 포착한다는 것을 보여줍니다. 분해된 다양한 항목은 각 이미지 영역과 관련된 attention 맵으로 해석됩니다. 


### Non-Local Upsampling
<p align="center">
  <img src="/assets/images/paper/transformer/GLAM-NLU.png" alt="Non-Local Upsampling Architecture" style="width: 100%;">
</p>

NLU 모듈은 skip connection을 쿼리 행렬 $$ Q $$로 변환하고, 저해상도 의미적 특징에서 키 $$ K $$와 값 $$ V $$을 계산하여 attention 행렬 $$ A $$를 생성합니다. transformer 블록 내의 residual connection을 유지하며, 다음과 같이 표현됩니다:

- 쿼리 행렬: $$ Q \in \mathbb{R}^{(4N_p) \times C} $$
- 키: $$ K \in \mathbb{R}^{N_p \times C} $$
- 값: $$ V \in \mathbb{R}^{N_p \times C} $$
- attention 행렬: $$ A \in \mathbb{R}^{(4N_p) \times N_p} $$

업샘플링된 저해상도 특징은 채널 수를 조정하는 linear projection을 거쳐 합산되고, Feed Forward (FF) 레이어가 적용됩니다. 최종적으로, skip connection과 업샘플링된 의미적 특징이 결합되어, 표준 U-Net 아키텍처와 유사한 방식으로 NLU를 마무리짓습니다.
