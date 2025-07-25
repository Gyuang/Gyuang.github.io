#!/usr/bin/env python3
"""
Batch VLM Paper Post Generator
Generates multiple VLM paper posts from a JSON configuration file.
"""

import json
import sys
from pathlib import Path
from generate_vlm_post import create_post_content, save_post, generate_filename, create_image_directory

def load_papers_config(config_file):
    """Load papers configuration from JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)

def create_sample_config():
    """Create a sample configuration file."""
    sample_config = {
        "papers": [
            {
                "title": "CLIP: Learning Transferable Visual Representations from Natural Language",
                "excerpt": "CLIP 논문 요약",
                "additional_tags": "Contrastive Learning, Zero-shot",
                "introduction": "CLIP introduces a method for learning visual concepts from natural language descriptions.",
                "related_work_vlm": "Previous vision-language models were limited in their ability to generalize...",
                "specific_domain": "Computer Vision",
                "related_work_domain": "Traditional computer vision approaches relied on supervised learning...",
                "architecture_description": "CLIP consists of an image encoder and text encoder trained with contrastive loss.",
                "architecture_image": "clip_architecture.png",
                "key_components": "Image encoder (ResNet or ViT), Text encoder (Transformer), Contrastive learning objective",
                "training_strategy": "Contrastive learning on 400M image-text pairs from the internet",
                "datasets": "400M image-text pairs, evaluated on ImageNet, CIFAR-100, etc.",
                "results": "Zero-shot performance competitive with supervised ResNet-50 on ImageNet",
                "ablation_studies": "Studies on different architectures, training data scale, and loss functions",
                "conclusion": "CLIP demonstrates the power of learning from natural language supervision.",
                "key_takeaways": "1. Natural language provides rich supervision signal\n2. Zero-shot transfer capabilities\n3. Scalable to large datasets"
            }
        ]
    }
    
    config_path = Path(__file__).parent.parent / "vlm_papers_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    return config_path

def main():
    """Main function for batch generation."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate VLM paper posts from configuration')
    parser.add_argument('--config', help='Path to configuration JSON file')
    args = parser.parse_args()
    
    # Get current script directory
    script_dir = Path(__file__).parent
    blog_root = script_dir.parent
    
    # Paths
    template_path = script_dir / "vlm_paper_template.md"
    posts_dir = blog_root / "_posts"
    
    # Check if template exists
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        sys.exit(1)
    
    # Determine configuration file
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = blog_root / "vlm_papers_config.json"
    
    if not config_file.exists():
        print("📋 No configuration file found. Creating sample configuration...")
        config_file = create_sample_config()
        print(f"✅ Sample configuration created: {config_file}")
        print("📝 Edit this file with your paper details and run the script again.")
        return
    
    # Load configuration
    config = load_papers_config(config_file)
    papers = config.get('papers', [])
    
    if not papers:
        print("❌ No papers found in configuration file.")
        return
    
    # Create image directory
    image_dir = create_image_directory()
    
    # Generate posts
    print(f"🚀 Generating {len(papers)} VLM paper posts...\n")
    
    created_files = []
    image_files_needed = []
    
    for i, paper_info in enumerate(papers, 1):
        try:
            print(f"📄 Processing paper {i}/{len(papers)}: {paper_info['title']}")
            
            # Generate filename and content
            filename = generate_filename(paper_info['title'])
            content = create_post_content(paper_info, template_path)
            
            # Save post
            filepath = save_post(content, filename, posts_dir)
            created_files.append(filepath)
            
            # Track image files needed
            if paper_info.get('architecture_image'):
                image_files_needed.append(paper_info['architecture_image'])
            
            print(f"   ✅ Created: {filepath}")
            
        except Exception as e:
            print(f"   ❌ Error processing paper: {e}")
    
    # Summary
    print(f"\n🎉 Batch generation complete!")
    print(f"📄 Created {len(created_files)} posts:")
    for filepath in created_files:
        print(f"   - {filepath}")
    
    if image_files_needed:
        print(f"\n🖼️  Image files needed in {image_dir}:")
        for img in image_files_needed:
            print(f"   - {img}")
    
    print(f"\n🚀 To preview your posts, run: bundle exec jekyll serve")

if __name__ == "__main__":
    main()