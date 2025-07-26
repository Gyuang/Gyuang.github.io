---
published: true
title: "ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models"
excerpt: "ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models 논문 요약"

categories:
  - VLM
tags:
  - [VLM, LLM, CAD, Medical Imaging, Interactive Diagnosis]

toc: true
toc_sticky: true
 
date: 2025-07-25
last_modified_at: 2025-07-25

---

## Introduction

Traditional computer-aided diagnosis (CAD) networks excel at medical image analysis but lack the ability to provide intuitive explanations and interactive dialogue with clinicians. ChatCAD addresses this limitation by integrating Large Language Models (LLMs) with existing CAD networks to create an interactive medical diagnosis system that combines visual understanding with natural language reasoning capabilities.


## Methods

ChatCAD integrates LLMs with computer-aided diagnosis networks through a multi-step pipeline that transforms visual information into natural language-based medical reasoning:

1. **Medical Image Processing**: Input medical images (chest X-rays, CT scans, etc.) are processed through specialized neural networks including disease classifiers, lesion detectors, and report generation models

2. **Multi-Network Output Generation**: Multiple CAD networks generate diverse outputs:
   - Disease classification results with confidence scores
   - Lesion segmentation masks and location information
   - Initial automated report drafts from vision-language models

3. **Text Description Transformation**: All visual outputs are converted into structured text descriptions that serve as the bridge between visual and linguistic information

4. **LLM Integration**: Text descriptions are fed into large language models (ChatGPT/GPT-based models) which leverage their medical knowledge and reasoning capabilities

5. **Interactive Report Generation**: The LLM synthesizes information to produce comprehensive medical reports, answer clinical questions, and engage in diagnostic dialogue

6. **Knowledge Retrieval Enhancement**: The system incorporates external medical knowledge bases (e.g., Merck Manual) to supplement LLM reasoning with authoritative medical references

7. **Multi-Modal Dialogue Interface**: Users can interact with the system through natural language queries about diagnoses, ask for explanations, or request additional analysis

<p align="center">
  <img src="/assets/images/paper/vlm/chatcad_interactive_computer-aided_diagnosis_on_medical_image_using_large_language_models_architecture.png" alt="ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models Architecture" style="width: 100%;">
</p>





## Dataset

The ChatCAD system was evaluated using multiple medical imaging datasets and knowledge resources:

**MIMIC-CXR Dataset**: Large-scale chest X-ray dataset with corresponding radiology reports used for training vision-language components and validating report generation capabilities. **External Medical Knowledge**: Integration with authoritative medical references including the Merck Manual Professional to enhance LLM reasoning with evidence-based medical knowledge. **Multi-Modal Evaluation Data**: Various medical imaging modalities including chest X-rays and CT scans to test the system's generalization across different imaging types and clinical scenarios.

## Results

ChatCAD demonstrates significant improvements in interactive medical diagnosis compared to traditional CAD systems:

**Enhanced Diagnostic Interpretation**: The system successfully combines quantitative CAD network outputs with qualitative natural language explanations, making medical AI more accessible to clinicians and patients. **Interactive Dialogue Capabilities**: Users can engage in meaningful conversations about diagnoses, request clarifications, and explore alternative diagnostic possibilities through natural language interaction. **Multi-Modal Report Generation**: ChatCAD produces comprehensive medical reports that integrate findings from multiple CAD networks while maintaining clinical accuracy and readability. The work has been published in prestigious venues including Nature Communications Engineering and IEEE Transactions on Medical Imaging, validating its scientific contribution to the intersection of AI and medical imaging.

## Conclusion

ChatCAD represents a significant advancement in medical AI by bridging the gap between powerful computer vision models and clinical usability. By integrating LLMs with traditional CAD networks, the system transforms complex medical image analysis into interactive, interpretable diagnostic assistance. This approach not only enhances the accuracy of medical image interpretation but also makes AI-driven diagnosis more accessible and trustworthy for healthcare professionals and patients alike.

## Key Takeaways

- **Interactive Medical AI**: ChatCAD enables natural language interaction with medical image analysis, making AI diagnosis more user-friendly and interpretable
- **Multi-Modal Integration**: The system successfully combines visual processing through CAD networks with textual reasoning through LLMs
- **Clinical Accessibility**: By providing explanations and engaging in dialogue, ChatCAD makes complex medical AI more accessible to both clinicians and patients
- **Scalable Framework**: The approach can be extended to various medical imaging modalities and clinical applications beyond radiology