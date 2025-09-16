#!/usr/bin/env python3
"""
Fix Method Sections in Paper Posts
Remove incorrect "강점", "약점" content from 3.1 and 3.2 sections
"""

import os
import re
import yaml
from pathlib import Path

class MethodSectionFixer:
    def __init__(self):
        self.posts_dir = Path("_posts/Paper")
    
    def parse_post(self, filepath):
        """Parse existing post to extract content and metadata"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split front matter and content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                front_matter = yaml.safe_load(parts[1])
                post_content = parts[2].strip()
            else:
                front_matter = {}
                post_content = content
        else:
            front_matter = {}
            post_content = content
            
        return front_matter, post_content
    
    def fix_method_sections(self, content):
        """Fix 3.1 and 3.2 sections with proper content"""
        
        # Replace 3.1 아키텍처 개요 section
        pattern_31 = r'(### 3\.1 아키텍처 개요\s*\n)(.*?)(?=### 3\.2)'
        replacement_31 = r'\1시스템의 전체 아키텍처와 주요 구성 요소들을 설명합니다.\n\n'
        content = re.sub(pattern_31, replacement_31, content, flags=re.DOTALL)
        
        # Replace 3.2 핵심 기술/알고리즘 section  
        pattern_32 = r'(### 3\.2 핵심 기술/알고리즘\s*\n)(.*?)(?=### 3\.3)'
        replacement_32 = r'\1핵심 기술적 혁신과 알고리즘에 대해 설명합니다.\n\n'
        content = re.sub(pattern_32, replacement_32, content, flags=re.DOTALL)
        
        return content
    
    def write_post(self, filepath, front_matter, content):
        """Write the fixed post to file"""
        # Convert front matter to YAML
        yaml_content = yaml.dump(front_matter, 
                                default_flow_style=False, 
                                allow_unicode=True, 
                                sort_keys=False)
        
        # Write complete file
        full_content = f"---\n{yaml_content}---\n\n{content}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
    
    def fix_post(self, filepath):
        """Fix a single post"""
        print(f"🔧 Fixing: {filepath.name}")
        
        try:
            # Parse existing post
            front_matter, content = self.parse_post(filepath)
            
            # Fix method sections
            fixed_content = self.fix_method_sections(content)
            
            # Write fixed post
            self.write_post(filepath, front_matter, fixed_content)
            
            print(f"✅ Successfully fixed: {filepath.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error fixing {filepath.name}: {str(e)}")
            return False
    
    def fix_all_posts(self):
        """Fix all paper posts"""
        print("🚀 Starting method section fixes...")
        
        # Get all paper posts
        post_files = list(self.posts_dir.glob("*.md"))
        successful = 0
        failed = 0
        
        for post_file in post_files:
            if self.fix_post(post_file):
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Fix Summary:")
        print(f"✅ Successfully fixed: {successful} posts")
        print(f"❌ Failed to fix: {failed} posts")
        
        return successful, failed

def main():
    """Main execution function"""
    fixer = MethodSectionFixer()
    
    print("=" * 60)
    print("🔧 Method Section Fixer")
    print("=" * 60)
    
    # Execute fixes
    successful, failed = fixer.fix_all_posts()
    
    if successful > 0:
        print(f"\n🎉 Fixes completed! {successful} posts updated.")
    
    if failed > 0:
        print(f"\n⚠️  {failed} posts failed. Check and retry if needed.")

if __name__ == "__main__":
    main()