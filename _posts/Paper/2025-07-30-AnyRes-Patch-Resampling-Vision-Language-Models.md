---
categories:
- paper
- multimodal
date: 2025-07-30
excerpt: AnyRes-GAN은 연속 스케일 패치 학습으로 FFHQ 1024px FID 4.06, patch-FID 2.96을 달성하며 기존
  멀티스케일·초해상 기법보다 고해상도 디테일을 크게 개선합니다.
header: {}
last_modified_at: '2025-09-16'
published: true
tags:
- AnyRes
- Patch Resampling
- LLaVA-NeXT
- Vision-Language Models
- High Resolution
- Dynamic Resolution
title: 'AnyRes: 비전-언어 모델의 혁신적 패치 리샘플링 기술 완전 분석'
toc: true
toc_sticky: true
---
# AnyRes: 비전-언어 모델의 혁신적 패치 리샘플링 기술 완전 분석

## 0. 체크리스트
- [x] `categories` 두 번째 값이 `multimodal`로 맞춰져 있나요?
- [x] `excerpt`에 대표 성능 수치를 포함했나요?
- [x] 모든 섹션이 실제 내용으로 채워졌나요?
- [x] 결과 표가 핵심 지표 중심으로 압축돼 있나요?
- [x] 참고 링크를 정리했나요?

## 1. 핵심 요약 (3문장 이하)
- AnyRes-GAN은 StyleGAN3를 기반으로 연속 좌표와 목표 스케일을 입력받아 임의 해상도의 패치를 합성하는 방식으로 학습/추론하는 고해상 이미지 생성 기법입니다.
- FFHQ 6K 데이터에서 FID@1024 4.06, patch-FID 2.96으로 Anycost-GAN 대비 각각 2.4pt, 15.4pt 향상했고, Real-ESRGAN 초해상보다 5.7배 낮은 patch-FID를 기록했습니다.
- 단일 모델로 256→2048 이상의 해상도를 커버하면서도 글로벌 구조와 로컬 디테일을 동시에 유지해, 2K 이상 무제한 배율 생성이 가능해집니다.

## 2. 배경 & 동기
- 고해상 GAN은 고정 해상도 데이터셋에 의존해, 네이티브 고해상 이미지의 세부 정보가 다운샘플링으로 사라지거나 저해상 샘플이 폐기됩니다.
- 기존 멀티스케일 학습(MS-PE, Anycost-GAN)은 이산 스케일만 지원하거나, 해상도마다 별도 블록이 필요해 파라미터/메모리 비용이 큽니다.
- AnyRes는 모든 이미지를 원본 해상도로 유지하고, 연속 스케일 패치를 샘플링해 모든 픽셀의 감독 신호를 활용하려는 목적에서 출발했습니다.

## 3. 방법론
### 3.1 전체 구조
- StyleGAN3 generator에 좌표 격자와 목표 스케일 조건을 추가해, 하나의 latent로부터 다양한 위치·해상도의 패치를 추출합니다.
- patch 기반 학습 단계에서 discriminator는 low-res 전체 이미지와 고해상 패치를 번갈아 보며, teacher regularization으로 base 스케일 일관성을 유지합니다.
- inference 시 latent와 좌표·스케일을 조합해 임의 해상도의 전체 이미지를 stitching 없이 렌더링합니다.

### 3.2 핵심 기법
- **Continuous-scale conditioning**: MLP로 인코딩한 연속 스케일 벡터를 각 스타일 모듈에 주입해, 훈련 범위를 벗어나는 해상도까지 외삽합니다.
- **Patch-FID 훈련**: 랜덤 위치·배율 패치를 샘플링해 high-frequency 통계를 유지하도록 discriminator를 학습시킵니다.
- **Inverse teacher regularization**: base 해상도(256px)에서 사전학습한 모델 출력을 목표로 하되, patch 단계에서는 고해상 디테일을 자유롭게 생성하도록 역방향 L1 손실을 사용합니다.

### 3.3 학습 및 구현 세부
- 4~8장의 V100(16GB)에서 mixed precision으로 학습하며, patch 크기 p는 도메인별 256~1024로 고정돼 메모리/연산량이 일정합니다.
- FFHQ는 70k 저해상(256)과 6k 고해상(최대 6K) 이미지를 조합했고, 교회/새/산 도메인도 고해상 이미지를 2~8%만 추가했습니다.
- 공식 코드는 [프로젝트 페이지](https://chail.github.io/anyres-gan/)와 GitHub에서 공개되어 있어 재현이 가능합니다.

## 4. 실험 & 결과
### 4.1 설정
- **도메인**: FFHQ-6K, LSUN+Flickr 교회, Flickr 새/산 등 네 가지 데이터셋.
- **평가 지표**: 해상도별 FID(256/512/1024)와 random patch-FID, 추가로 Precision/Recall, detectability AP를 측정했습니다.
- **비교 기준**: MS-PE, Anycost-GAN, LIIF/Real-ESRGAN 초해상 등 멀티스케일 및 SR 계열 모델과 비교.

### 4.2 주요 결과표
| 모델 | FID@256 | FID@512 | FID@1024 | pFID (random) |
| --- | --- | --- | --- | --- |
| MS-PE | 6.75 | 30.41 | – | – |
| Anycost-GAN | 4.24 | 5.94 | 6.47 | 18.39 |
| **AnyRes (ours)** | **3.34** | **3.71** | **4.06** | **2.96** |

### 4.3 추가 분석
- FFHQ patch-FID 2.96은 Real-ESRGAN(16.92) 대비 83% 낮아, 초해상 대비 훨씬 사실적인 고주파 디테일을 생성합니다.
- 교회/새/산 도메인에서도 patch-FID 9.89/6.52/7.99로 LIIF·Real-ESRGAN보다 큰 폭으로 개선했습니다.
- 고해상 탐지기(AP)가 해상도와 함께 상승(>90%)해, 시각적으로 자연스러우면서도 기존 탐지기에선 여전히 식별 가능합니다.

## 5. 의의 & 한계
- 단일 GAN으로 다해상 생성이 가능해 데이터 효율과 모델 크기를 동시에 잡았으며, 소수의 고해상 이미지 추가만으로 2K 이상 합성이 가능합니다.
- patch 기반 학습 덕분에 메모리 소비가 고정돼 대규모 배율 학습에 실용적입니다.
- 다만 base 해상도에서의 teacher 품질에 성능이 의존하며, 좌표 기반 조건이 없는 아키텍처에는 즉시 적용하기 어렵습니다.

## 6. 개인 평가
**강점**: 연속 스케일 조건과 patch 학습으로 고해상 디테일과 글로벌 레이아웃을 모두 잡았고, 데이터·연산 효율이 뛰어납니다.  
**약점**: teacher 사전학습과 하이퍼파라미터(λteacher, patch 범위) 조정이 필요해 파이프라인이 다소 복잡합니다.  
**적용 가능성**: 고해상 합성 데이터 제작, diffusion/GAN 하이브리드 초기화 등에서 강력한 백본으로 활용될 수 있습니다.  
**추천도**: ★★★★☆ (고해상 합성 품질과 효율을 동시에 추구하는 연구자에게 추천)

## 7. 참고 자료
- 원문: [Any-resolution Training for High-resolution Image Synthesis](https://arxiv.org/abs/2204.07156)
- 코드 & 데모: [Project Page](https://chail.github.io/anyres-gan/)
- 데이터: [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Flickr Church/Birds/Mountains](https://github.com/chail/anyres-gan)
