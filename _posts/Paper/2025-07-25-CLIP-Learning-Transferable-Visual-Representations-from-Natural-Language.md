---
published: true
title: "CLIP: Learning Transferable Visual Representations from Natural Language"
excerpt: "CLIP 논문 요약"

categories:
  - VLM
tags:
  - [VLM, Vision-Language, Contrastive Learning, Zero-shot]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

CLIP introduces a method for learning visual concepts from natural language descriptions.

## Related Work 

### Vision-Language Models

Previous vision-language models were limited in their ability to generalize...

### Computer Vision

Traditional computer vision approaches relied on supervised learning...

## Method 

### Architecture Overview

CLIP consists of an image encoder and text encoder trained with contrastive loss.


<p align="center">
  <img src="/assets/images/paper/vlm/clip_architecture.png" alt="CLIP: Learning Transferable Visual Representations from Natural Language Architecture" style="width: 100%;">
</p>


### Key Components

Image encoder (ResNet or ViT), Text encoder (Transformer), Contrastive learning objective

### Training Strategy

Contrastive learning on 400M image-text pairs from the internet





## Experiments

### Datasets

400M image-text pairs, evaluated on ImageNet, CIFAR-100, etc.

### Results

Zero-shot performance competitive with supervised ResNet-50 on ImageNet

### Ablation Studies

Studies on different architectures, training data scale, and loss functions

## Conclusion

CLIP demonstrates the power of learning from natural language supervision.

## Key Takeaways

1. Natural language provides rich supervision signal
2. Zero-shot transfer capabilities
3. Scalable to large datasets