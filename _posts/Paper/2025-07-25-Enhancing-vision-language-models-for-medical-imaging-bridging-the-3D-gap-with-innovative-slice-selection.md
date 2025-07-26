---
published: true
title: "Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection"
excerpt: "Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection 논문 요약"

categories:
  - VLM
tags:
  - [VLM, Vision-Language, 3D Medical Imaging, Slice Selection, Medical AI]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

Vision-language models (VLMs) are primarily designed for 2D inputs, which creates a significant gap when applying them to 3D medical imaging such as MRI and CT scans. This paper introduces Vote-MI, an innovative one-pass, unsupervised representative slice selection method that bridges this 3D gap by intelligently selecting the most representative 2D slices from 3D medical images, enabling existing 2D VLMs to effectively process 3D medical data.


## Related Work 

### Vision-Language Models

Existing vision-language models like CLIP, BLIP, and Med-Flamingo have shown remarkable success in various tasks but are primarily designed for 2D inputs. This limitation becomes particularly challenging in medical imaging where 3D scans (MRI, CT) are the standard. Previous approaches either processed all slices individually or used simple averaging, both of which fail to capture the most representative information efficiently.

### Medical Imaging and AI

Traditional approaches to 3D medical image analysis include 3D CNNs and volumetric processing methods. However, these approaches require significant computational resources and cannot leverage the powerful pre-trained 2D vision-language models. The Vote-MI method bridges this gap by enabling existing 2D models to work effectively with 3D medical data.

## Method 

### Vote-MI Slice Selection Method

The core innovation of this paper is the Vote-MI (Voting-based Medical Imaging) method, which addresses the 3D gap through the following step-by-step approach:

1. **3D Medical Image Input**: Start with a 3D medical image (MRI or CT scan) containing multiple 2D slices
2. **Unsupervised Analysis**: Apply one-pass, unsupervised analysis to evaluate each slice within the 3D volume
3. **Representative Slice Identification**: Use voting mechanism to identify the most representative 2D slice that best captures the critical medical information
4. **Slice Selection**: Select the optimal slice(s) that maximize information content while maintaining compatibility with 2D VLMs
5. **VLM Processing**: Feed the selected representative 2D slice to existing vision-language models (specifically Med-Flamingo)
6. **Medical Report Generation**: Generate medical reports or perform downstream tasks using the processed 2D representation

<p align="center">
  <img src="/assets/images/paper/vlm/enhancing_vision-language_models_for_medical_imaging_bridging_the_3d_gap_with_innovative_slice_selection_architecture.png" alt="Enhancing vision-language models for medical imaging: bridging the 3D gap with innovative slice selection Architecture" style="width: 100%;">
</p>

### Key Technical Components

- **One-Pass Processing**: Efficient single-pass algorithm for slice evaluation
- **Unsupervised Learning**: No requirement for labeled training data for slice selection
- **Voting Mechanism**: Systematic approach to rank and select the most informative slices
- **Med-Flamingo Integration**: Seamless integration with existing 2D vision-language models





## Experiments

### Datasets

**BrainMD Dataset**: The authors introduce BrainMD, a comprehensive multimodal dataset specifically designed for this research. It comprises 2,453 annotated 3D MRI brain scans with corresponding textual radiology reports and electronic health records. This dataset serves as the foundation for developing and evaluating the Vote-MI method.

**Benchmark Tasks**: Based on BrainMD, two benchmarks were developed:
- **BrainMD-select**: Focuses on evaluating the most representative 2D slice selection from 3D images
- **BrainBench**: Includes various vision-language downstream tasks for comprehensive evaluation

### Results

The Vote-MI method demonstrates significant performance improvements over random slice selection baselines. The key findings include:

**Performance Gains**: Vote-MI achieved substantial improvements in both learning scenarios:
- **Zero-shot Learning**: 14.6% absolute performance gain compared to random slice selection
- **Few-shot Learning**: 16.6% absolute performance gain compared to random slice selection

**Clinical Impact**: These results indicate that intelligent slice selection can dramatically improve the effectiveness of 2D vision-language models when applied to 3D medical imaging tasks. The method successfully bridges the dimensionality gap while maintaining or enhancing the diagnostic capabilities of existing VLMs, representing a significant step toward integrating AI in medical imaging to enhance patient care and facilitate medical research.

### Ablation Studies

The paper includes comprehensive ablation studies examining different components of the Vote-MI method. Key findings from the ablation studies show that the voting mechanism is crucial for identifying the most representative slices, and the one-pass approach significantly improves computational efficiency compared to multi-pass alternatives. The studies also validate the effectiveness of the unsupervised approach across different types of medical imaging tasks.

## Conclusion

This paper successfully addresses a critical limitation in applying vision-language models to medical imaging through the innovative Vote-MI slice selection method. The key contributions include: (1) introducing a one-pass, unsupervised approach to bridge the 3D gap between medical images and 2D VLMs, (2) developing the comprehensive BrainMD dataset with 2,453 annotated 3D MRI scans, and (3) demonstrating significant performance improvements of 14.6% and 16.6% for zero-shot and few-shot learning respectively. This work represents a significant step toward integrating AI in medical imaging to enhance patient care and facilitate medical research.

## Key Takeaways

- **3D Gap Solution**: Vote-MI effectively solves the fundamental mismatch between 3D medical images and 2D vision-language models through intelligent slice selection
- **Unsupervised Approach**: The method requires no labeled training data for slice selection, making it practical for real-world medical applications
- **Substantial Performance Gains**: Demonstrates 14.6-16.6% absolute improvements over random selection in both zero-shot and few-shot scenarios
- **Clinical Relevance**: The BrainMD dataset and benchmarks provide valuable resources for future medical AI research
- **Practical Integration**: The approach can be seamlessly integrated with existing 2D VLMs like Med-Flamingo without architectural modifications