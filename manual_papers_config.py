#!/usr/bin/env python3
"""
Manual Paper Configuration for Missing ArXiv IDs

This script contains manual configurations for well-known papers
that have ArXiv IDs but weren't automatically detected.
"""

# Well-known papers and their ArXiv IDs
MANUAL_PAPER_CONFIGS = {
    "CLIP: Learning Transferable Visual Representations from Natural Language": {
        "arxiv_id": "2103.00020",
        "post_file": "2025-07-25-CLIP-Learning-Transferable-Visual-Representations-from-Natural-Language.md",
        "folder_name": "clip-learning-transferable-visual-representations-from-natural-language"
    },
    "Learning to Prompt for Vision-Language Models (CoOp)": {
        "arxiv_id": "2109.01134",
        "post_file": "2025-07-25-CoOp-Learning-to-Prompt-for-Vision-Language-Models.md",
        "folder_name": "coop-learning-to-prompt-for-vision-language-models"
    },
    "Conditional Prompt Learning for Vision-Language Models (CoCoOp)": {
        "arxiv_id": "2203.05557",
        "post_file": "2025-07-25-CoCoOp-Conditional-Prompt-Learning-for-Vision-Language-Models.md",
        "folder_name": "cocoop-conditional-prompt-learning-for-vision-language-models"
    },
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": {
        "arxiv_id": "2005.11401",
        "post_file": "2025-07-25-Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks.md",
        "folder_name": "retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks"
    },
    "Visual Prompt Tuning": {
        "arxiv_id": "2203.12119",
        "post_file": "2025-07-28-Visual-Prompt-Tuning-in-VLMs-for-Medical-Applications.md",
        "folder_name": "visual-prompt-tuning"
    },
    "AnyRes: Any-Resolution Vision Language Model": {
        "arxiv_id": "2403.04306",
        "post_file": "2025-07-30-AnyRes-Patch-Resampling-Vision-Language-Models.md",
        "folder_name": "anyres-patch-resampling-vision-language-models"
    },
    # Additional papers that might have ArXiv IDs
    "VoxelPrompt: A Vision-Language Agent for Grounded Medical Image Analysis": {
        "arxiv_id": "2410.08397",
        "post_file": "2025-07-25-VoxelPrompt-A-Vision-Language-Agent-for-Grounded-Medical-Image-Analysis.md",
        "folder_name": "voxelprompt-a-vision-language-agent-for-grounded-medical-image-analysis"
    },
    "Brain-Adapter: Enhancing Neurological Disorder Analysis": {
        "arxiv_id": "2312.15413",
        "post_file": "2025-07-25-Brain-Adapter-Enhancing-Neurological-Disorder-Analysis-with-Adapter-Tuning-Multimodal-Large-Language-Models.md",
        "folder_name": "brain-adapter-enhancing-neurological-disorder-analysis-with-adapter-tuning-multimodal-large-language-models"
    }
}

if __name__ == "__main__":
    import json
    print("Manual Paper Configurations:")
    for title, config in MANUAL_PAPER_CONFIGS.items():
        print(f"- {title}: {config['arxiv_id']}")
    
    # Save to JSON for easy loading
    with open("/Users/gyu/Desktop/Gyuang.github.io/manual_papers.json", "w") as f:
        json.dump(MANUAL_PAPER_CONFIGS, f, indent=2)
    
    print(f"\nSaved {len(MANUAL_PAPER_CONFIGS)} manual configurations to manual_papers.json")