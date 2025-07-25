#!/usr/bin/env python3
"""
VLM Paper Post Generator for Jekyll Blog
Automates the creation of VLM paper posts with proper formatting and structure.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def get_user_input():
    """Collect paper information from user input."""
    print("=== VLM Paper Post Generator ===\n")
    
    paper_info = {}
    
    # Basic information
    paper_info['title'] = input("Paper title: ").strip()
    paper_info['excerpt'] = input("Brief excerpt (논문요약): ").strip() or "논문요약"
    paper_info['additional_tags'] = input("Additional tags (comma-separated, e.g., 'Medical, Multimodal'): ").strip()
    
    # Content sections
    print("\n--- Content Sections ---")
    paper_info['introduction'] = input("Introduction (brief summary): ").strip()
    paper_info['related_work_vlm'] = input("Related work - VLM section: ").strip()
    paper_info['specific_domain'] = input("Specific domain (e.g., Medical Imaging, Robotics): ").strip()
    paper_info['related_work_domain'] = input("Related work - Domain specific: ").strip()
    
    # Method section
    print("\n--- Method Section ---")
    paper_info['architecture_description'] = input("Architecture description: ").strip()
    paper_info['architecture_image'] = input("Architecture image filename (optional, will be saved in /assets/images/paper/vlm/): ").strip()
    paper_info['key_components'] = input("Key components description: ").strip()
    paper_info['training_strategy'] = input("Training strategy: ").strip()
    
    # Experiments
    print("\n--- Experiments ---")
    paper_info['datasets'] = input("Datasets used: ").strip()
    paper_info['results'] = input("Main results: ").strip()
    paper_info['ablation_studies'] = input("Ablation studies: ").strip()
    
    # Conclusion
    print("\n--- Conclusion ---")
    paper_info['conclusion'] = input("Conclusion: ").strip()
    paper_info['key_takeaways'] = input("Key takeaways: ").strip()
    
    return paper_info

def generate_filename(title):
    """Generate Jekyll-compatible filename."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Clean title for filename
    clean_title = title.replace(" ", "-").replace(":", "").replace(",", "").replace("'", "").replace('"', '')
    return f"{date_str}-{clean_title}.md"

def create_post_content(paper_info, template_path):
    """Create post content by filling the template."""
    
    # Read template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Replace template variables
    replacements = {
        '{{TITLE}}': paper_info['title'],
        '{{EXCERPT}}': paper_info['excerpt'],
        '{{DATE}}': current_date,
        '{{ADDITIONAL_TAGS}}': paper_info['additional_tags'],
        '{{INTRODUCTION}}': paper_info['introduction'],
        '{{RELATED_WORK_VLM}}': paper_info['related_work_vlm'],
        '{{SPECIFIC_DOMAIN}}': paper_info['specific_domain'],
        '{{RELATED_WORK_DOMAIN}}': paper_info['related_work_domain'],
        '{{ARCHITECTURE_DESCRIPTION}}': paper_info['architecture_description'],
        '{{ARCHITECTURE_IMAGE}}': paper_info['architecture_image'],
        '{{KEY_COMPONENTS}}': paper_info['key_components'],
        '{{TRAINING_STRATEGY}}': paper_info['training_strategy'],
        '{{DATASETS}}': paper_info['datasets'],
        '{{RESULTS}}': paper_info['results'],
        '{{ABLATION_STUDIES}}': paper_info['ablation_studies'],
        '{{CONCLUSION}}': paper_info['conclusion'],
        '{{KEY_TAKEAWAYS}}': paper_info['key_takeaways']
    }
    
    content = template
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    
    # Handle conditional architecture image
    if paper_info['architecture_image']:
        content = content.replace('{{#if ARCHITECTURE_IMAGE}}', '').replace('{{/if}}', '')
    else:
        # Remove the entire image block
        import re
        content = re.sub(r'{{#if ARCHITECTURE_IMAGE}}.*?{{/if}}', '', content, flags=re.DOTALL)
    
    # Clean up any remaining template syntax
    content = content.replace('{{#if ADDITIONAL_IMAGES}}', '').replace('{{ADDITIONAL_IMAGES}}', '').replace('{{/if}}', '')
    
    return content

def save_post(content, filename, posts_dir):
    """Save the post to the appropriate directory."""
    paper_dir = posts_dir / "Paper"
    paper_dir.mkdir(exist_ok=True)
    
    filepath = paper_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath

def create_image_directory():
    """Create VLM image directory if it doesn't exist."""
    image_dir = Path("assets/images/paper/vlm")
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir

def main():
    """Main function to orchestrate the post generation."""
    
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
    
    # Collect paper information
    try:
        paper_info = get_user_input()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    
    # Generate filename
    filename = generate_filename(paper_info['title'])
    
    # Create post content
    content = create_post_content(paper_info, template_path)
    
    # Save post
    filepath = save_post(content, filename, posts_dir)
    
    # Create image directory
    image_dir = create_image_directory()
    
    print(f"\n✅ VLM paper post created successfully!")
    print(f"📄 File: {filepath}")
    print(f"🖼️  Image directory: {image_dir}")
    
    if paper_info['architecture_image']:
        print(f"📋 Don't forget to add your architecture image: {image_dir / paper_info['architecture_image']}")
    
    print(f"\n🚀 To preview your post, run: bundle exec jekyll serve")

if __name__ == "__main__":
    main()