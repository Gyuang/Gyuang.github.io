---
published: true
title: "Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation | Nature Communications"
excerpt: "Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation | Nature Communications 논문 요약"

categories:
  - VLM
tags:
  - [VLM, Multimodal LLM, 3D Brain CT, Radiology Report, Medical AI]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

3D brain CT radiology report generation faces significant challenges due to the complexity of volumetric medical imaging data and the critical need for clinically accurate diagnostic interpretations. This Nature Communications paper presents BrainGPT, a comprehensive multimodal large language model framework that leverages clinical visual instruction tuning (CVIT) to automatically generate high-quality radiology reports from 3D brain CT scans, addressing the gap between advanced AI capabilities and practical clinical applications.


## Related Work 

### Vision-Language Models

Previous vision-language models have primarily focused on 2D medical imaging applications, with limited exploration of 3D volumetric data processing for radiology report generation. While models like CLIP and BLIP have shown success in general vision-language tasks, their application to complex medical imaging scenarios requires specialized adaptations for clinical accuracy and diagnostic relevance.

### Medical Imaging AI

Computer vision approaches in medical imaging have traditionally employed convolutional neural networks for image classification and segmentation tasks. However, the integration of large language models with 3D medical imaging represents a significant advancement, enabling automated report generation that bridges the gap between visual pattern recognition and clinical narrative generation.

## Methods

The holistic framework for multimodal LLM in 3D brain CT radiology report generation follows a systematic approach:

1. **Architecture Foundation**: Build upon the Otter framework, integrating a frozen CLIP ViT-L/14 vision encoder with the LlaMA-7B large language model through a trainable perceiver resampler module

2. **Cross-Attention Integration**: Insert cross-gated attention layers into the LlaMA-7B architecture to distribute focus evenly across volumetric CT scan slices and handle 3D spatial relationships

3. **Clinical Visual Instruction Tuning (CVIT)**: Implement four distinct fine-tuning approaches:
   - Plain instruction tuning for basic visual-text alignment
   - In-context example instruction providing clinical exemplars
   - Template instruction using structured medical report formats
   - Keyword instruction focusing on critical diagnostic terminology

4. **Multi-Image Processing**: Format training data into image-instruction-answer triplets, tokenizing instructions and enhancing images before model input to enable multi-slice CT interpretation

5. **Volumetric Data Handling**: Process 3D brain CT scans by treating each axial slice as input while maintaining spatial coherence through cross-attention mechanisms

6. **Training Optimization**: Execute training for 3 epochs over 12 hours using two NVIDIA A100 GPUs, with specialized data augmentation for medical imaging

7. **Clinical Adaptation**: Fine-tune the base model with medical domain knowledge to ensure generated reports follow radiology conventions and diagnostic accuracy standards






## Experiments

### Dataset

The study utilizes the 3D-BrainCT dataset, a comprehensive collection of 18,885 text-scan pairs specifically curated for 3D brain CT radiology report generation. This dataset represents volumetric brain CT imaging data paired with corresponding clinical radiology reports, providing the foundation for training and evaluating BrainGPT models. The dataset addresses the scarcity of large-scale 3D medical imaging datasets with paired clinical text, enabling robust multimodal learning for automated radiology report generation in brain CT interpretation.

### Results

BrainGPT demonstrates significant improvements in automated 3D brain CT radiology report generation across multiple evaluation metrics. The model achieves strong performance on traditional NLP metrics with BLEU-1 scores of 44.35, BLEU-4 of 20.38, METEOR of 30.13, ROUGE-L of 47.6, and CIDEr-R of 211.77. More importantly, using the novel FORTE evaluation framework, BrainGPT attains an average F1-score of 0.71 across clinical dimensions (degree: 0.661, landmark: 0.706, feature: 0.693, impression: 0.779). In human evaluation studies involving 11 physician evaluators, 74% of BrainGPT-generated reports were indistinguishable from human-written ground truth in Turing-like tests, demonstrating the model's clinical viability for real-world radiology applications.

### Ablation Studies

The research evaluates four distinct BrainGPT variants through comprehensive ablation studies: BrainGPT-plain, BrainGPT-example, BrainGPT-template, and BrainGPT-keyword. These variants demonstrate the effectiveness of different clinical visual instruction tuning approaches, with template and keyword-based fine-tuning showing superior performance in generating clinically structured reports that align with standard radiology conventions.

## Conclusion

This work establishes a comprehensive framework for applying multimodal large language models to 3D brain CT radiology report generation. The key contributions include the creation of the 3D-BrainCT dataset, development of BrainGPT with clinical visual instruction tuning, and introduction of the FORTE evaluation framework. The research demonstrates that automated radiology report generation for volumetric medical imaging is achievable with high clinical accuracy, paving the way for practical AI applications in diagnostic radiology.

## Key Takeaways

The FORTE evaluation framework provides clinically relevant assessment beyond traditional NLP metrics, addressing the critical need for medical AI evaluation that captures diagnostic accuracy. BrainGPT's 74% human-indistinguishable performance in physician evaluations demonstrates significant potential for clinical deployment. The clinical visual instruction tuning approach successfully adapts general-purpose vision-language models to specialized medical domains, establishing a methodology that could extend to other medical imaging modalities and anatomical regions.