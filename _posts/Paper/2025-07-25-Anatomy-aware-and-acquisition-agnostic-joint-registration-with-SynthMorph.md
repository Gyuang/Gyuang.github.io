---
published: true
title: "Anatomy-aware and acquisition-agnostic joint registration with SynthMorph"
excerpt: "Anatomy-aware and acquisition-agnostic joint registration with SynthMorph 논문 요약"

categories:
  - Paper
tags:
  - [Medical Imaging, Image Registration, Deep Learning, Neuroscience, MRI]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

Affine image registration is a cornerstone of medical image analysis that aligns brain images for comparative studies and preprocessing pipelines. While classical optimization-based algorithms achieve excellent accuracy, they require time-consuming optimization for every image pair and struggle with anatomy-specific alignment requirements. SynthMorph addresses these limitations by introducing a deep learning approach that is both anatomy-aware and acquisition-agnostic, enabling fast and robust registration across diverse neuroimaging protocols without preprocessing.

## Methods

The SynthMorph methodology introduces several key innovations for robust medical image registration:

1. **Synthetic Training Data Generation**: Train networks exclusively using synthetically generated images from 100 whole-head tissue segmentations sourced from UKBB, OASIS, ABCD, and infant scan datasets

2. **Three Core Affine Registration Architectures**: 
   - Parameter Encoder: Directly predicts affine transformation parameters
   - Warp Decomposer: Decomposes displacement fields into affine components
   - Feature Detector: Extracts anatomical features for registration guidance

3. **Anatomy-Aware Optimization**: Optimize spatial overlap of select anatomical labels rather than all image structures, enabling networks to distinguish anatomy of interest from irrelevant structures

4. **Hypernetwork Integration**: Combine affine model with deformable hypernetwork that allows users to dynamically choose optimal deformation-field regularity at registration time

5. **Acquisition-Agnostic Training**: Apply random spatial transformations and image corruption during training to ensure robustness across different MRI contrasts, resolutions, and acquisition protocols

6. **Joint Affine-Deformable Registration**: Provide end-to-end solution combining both affine and deformable registration in a single framework

7. **Loss Function Design**: Utilize mean squared error loss function to optimize label overlap while maintaining spatial consistency through Jacobian regularization

<p align="center">
  <img src="/assets/images/paper/vlm/anatomy-aware_and_acquisition-agnostic_joint_registration_with_synthmorph_architecture.png" alt="Anatomy-aware and acquisition-agnostic joint registration with SynthMorph Architecture" style="width: 100%;">
</p>

## Dataset

The evaluation encompasses an extremely diverse set of neuroimaging data to capture real-world performance. Training data includes 100 tissue segmentations from major neuroimaging datasets (UKBB, OASIS, ABCD, infant scans). Evaluation datasets span multiple MRI contrasts including T1-weighted, T2-weighted, and proton density-weighted images with varying resolutions (0.4-1.2 mm) and subject populations ranging from ages 0-75 years, covering both adult and pediatric brain imaging scenarios.

## Results

SynthMorph demonstrates superior performance across diverse neuroimaging scenarios with significant improvements over classical methods. Key findings include high Dice overlap scores between transformed and fixed label maps, improved normalized cross-correlation of neighborhood descriptors, and optimal log-Jacobian spread for deformation field regularity. The method achieves fast registration times while maintaining accuracy comparable to or exceeding classical optimization-based approaches. Importantly, SynthMorph generalizes effectively across different MRI acquisition protocols without requiring domain-specific retraining, addressing the critical domain shift problem that limits other deep learning registration methods.