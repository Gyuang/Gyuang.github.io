#!/usr/bin/env python3
"""
Paper Post Rewriter using New Template Structure
Rewrites all existing paper posts to follow the new template format
"""

import os
import re
import yaml
import shutil
from pathlib import Path
from datetime import datetime

class PaperPostRewriter:
    def __init__(self):
        self.posts_dir = Path("_posts/Paper")
        self.backup_dir = Path("_posts_backup")
        self.template_path = Path("paper_summary_template.md")
        
        # Template structure sections
        self.new_sections = [
            "핵심 요약",
            "배경 및 동기", 
            "제안 방법",
            "실험 및 결과",
            "의의 및 영향",
            "개인적 평가"
        ]
        
    def create_backup(self):
        """Create backup of all paper posts"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        shutil.copytree(self.posts_dir, self.backup_dir)
        print(f"✅ Created backup at {self.backup_dir}")
    
    def parse_existing_post(self, filepath):
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
    
    def extract_paper_info(self, content):
        """Extract paper information from existing content"""
        info = {
            'title': '',
            'authors': '',
            'venue': '',
            'arxiv_id': '',
            'problem': '',
            'solution': '',
            'results': ''
        }
        
        # Extract title from content
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1).strip()
        
        # Extract authors
        author_match = re.search(r'저자[:\s]*(.+)', content)
        if author_match:
            info['authors'] = author_match.group(1).strip()
        
        # Extract venue
        venue_match = re.search(r'(발표|학회|저널)[:\s]*(.+)', content)
        if venue_match:
            info['venue'] = venue_match.group(2).strip()
            
        # Extract ArXiv ID
        arxiv_match = re.search(r'arxiv[:\s]*([0-9]+\.[0-9]+)', content, re.IGNORECASE)
        if arxiv_match:
            info['arxiv_id'] = arxiv_match.group(1)
            
        return info
    
    def categorize_content(self, content):
        """Categorize existing content into new template sections"""
        sections = {
            '핵심 요약': '',
            '배경 및 동기': '',
            '제안 방법': '',
            '실험 및 결과': '',
            '의의 및 영향': '',
            '개인적 평가': ''
        }
        
        # Split content by headers
        section_pattern = r'^#+\s+(.+)$'
        current_section = '핵심 요약'
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            header_match = re.match(section_pattern, line)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] += '\n'.join(current_content) + '\n\n'
                
                # Determine new section
                header_text = header_match.group(1).lower()
                current_content = []
                
                if any(keyword in header_text for keyword in ['introduction', '소개', '배경', '동기', '문제']):
                    current_section = '배경 및 동기'
                elif any(keyword in header_text for keyword in ['method', '방법', '접근', '아키텍처', 'architecture']):
                    current_section = '제안 방법'
                elif any(keyword in header_text for keyword in ['experiment', '실험', 'result', '결과', '성능']):
                    current_section = '실험 및 결과'
                elif any(keyword in header_text for keyword in ['conclusion', '결론', '의의', '영향', 'impact']):
                    current_section = '의의 및 영향'
                else:
                    current_section = '제안 방법'  # Default
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] += '\n'.join(current_content)
            
        return sections
    
    def extract_images(self, content):
        """Extract and categorize images from content"""
        images = {
            'architecture': [],
            'method': [],
            'results': [],
            'other': []
        }
        
        # Find all image references
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(img_pattern, content)
        
        for alt_text, img_path in matches:
            alt_lower = alt_text.lower()
            if any(keyword in alt_lower for keyword in ['architecture', '아키텍처', 'overview']):
                images['architecture'].append((alt_text, img_path))
            elif any(keyword in alt_lower for keyword in ['method', '방법', 'diagram']):
                images['method'].append((alt_text, img_path))
            elif any(keyword in alt_lower for keyword in ['result', '결과', 'table', 'graph']):
                images['results'].append((alt_text, img_path))
            else:
                images['other'].append((alt_text, img_path))
                
        return images
    
    def create_new_post(self, filepath, front_matter, content):
        """Create new post following the template structure"""
        paper_info = self.extract_paper_info(content)
        sections = self.categorize_content(content)
        images = self.extract_images(content)
        
        # Update front matter to follow new standards
        new_front_matter = {
            'categories': front_matter.get('categories', ['VLM']),
            'date': front_matter.get('date', datetime.now().strftime('%Y-%m-%d')),
            'excerpt': f"{paper_info.get('title', 'AI 연구 논문')}에 대한 체계적 분석",
            'header': front_matter.get('header', {}),
            'last_modified_at': datetime.now().strftime('%Y-%m-%d'),
            'published': True,
            'tags': front_matter.get('tags', ['AI', 'Research']),
            'title': front_matter.get('title', paper_info.get('title', 'Paper Analysis')),
            'toc': True,
            'toc_sticky': True
        }
        
        # Build new content structure
        new_content = f"""# {new_front_matter['title']}

## 논문 정보
- **저자**: {paper_info.get('authors', 'N/A')}
- **발표**: {paper_info.get('venue', 'N/A')}
- **ArXiv**: {f"[{paper_info['arxiv_id']}](https://arxiv.org/abs/{paper_info['arxiv_id']})" if paper_info.get('arxiv_id') else 'N/A'}

## 1. 핵심 요약 (2-3문장)
{sections.get('핵심 요약', '').strip() or '이 논문의 핵심 기여와 주요 발견을 간결하게 요약합니다.'}

## 2. 배경 및 동기
{sections.get('배경 및 동기', '').strip() or '기존 방법의 한계점과 연구의 필요성을 설명합니다.'}

## 3. 제안 방법

### 3.1 아키텍처 개요
{self.insert_architecture_images(images['architecture'])}

### 3.2 핵심 기술/알고리즘
{sections.get('제안 방법', '').strip() or '제안된 방법의 핵심 기술과 알고리즘을 설명합니다.'}

### 3.3 구현 세부사항
{self.insert_method_images(images['method'])}

## 4. 실험 및 결과

### 4.1 실험 설정
실험에 사용된 데이터셋, 평가 지표, 비교 대상을 설명합니다.

### 4.2 주요 결과
{self.insert_results_images(images['results'])}

{sections.get('실험 및 결과', '').strip() or '실험 결과와 성능 분석을 제시합니다.'}

### 4.3 분석
결과에 대한 정성적 분석과 해석을 제공합니다.

## 5. 의의 및 영향
{sections.get('의의 및 영향', '').strip() or '이 연구의 학술적 기여와 실용적 가치를 평가합니다.'}

## 6. 개인적 평가
{sections.get('개인적 평가', '').strip() or '''
**강점**: 이 논문의 주요 강점과 인상 깊었던 부분
**약점**: 아쉬웠던 부분이나 의문점  
**적용 가능성**: 실제 연구나 응용에서의 활용 가능성
**추천도**: 다른 연구자들에게 추천할 만한 수준
'''}
"""
        
        return new_front_matter, new_content
    
    def insert_architecture_images(self, images):
        """Insert architecture images with proper formatting"""
        if not images:
            return ""
        
        result = ""
        for i, (alt_text, img_path) in enumerate(images[:2]):  # Limit to 2 images
            result += f"\n![{alt_text}]({img_path})\n*Figure: {alt_text}*\n\n"
        return result
    
    def insert_method_images(self, images):
        """Insert method images with proper formatting"""
        if not images:
            return ""
            
        result = ""
        for i, (alt_text, img_path) in enumerate(images[:3]):  # Limit to 3 images
            result += f"\n![{alt_text}]({img_path})\n*Figure: {alt_text}*\n\n"
        return result
    
    def insert_results_images(self, images):
        """Insert results images with proper formatting"""
        if not images:
            return ""
            
        result = ""
        for i, (alt_text, img_path) in enumerate(images[:3]):  # Limit to 3 images
            result += f"\n![{alt_text}]({img_path})\n*Figure: {alt_text}*\n\n"
        return result
    
    def write_new_post(self, filepath, front_matter, content):
        """Write the new post to file"""
        # Convert front matter to YAML
        yaml_content = yaml.dump(front_matter, 
                                default_flow_style=False, 
                                allow_unicode=True, 
                                sort_keys=False)
        
        # Write complete file
        full_content = f"---\n{yaml_content}---\n\n{content}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
    
    def rewrite_post(self, filepath):
        """Rewrite a single post"""
        print(f"📝 Rewriting: {filepath.name}")
        
        try:
            # Parse existing post
            front_matter, content = self.parse_existing_post(filepath)
            
            # Create new structured content
            new_front_matter, new_content = self.create_new_post(filepath, front_matter, content)
            
            # Write new post
            self.write_new_post(filepath, new_front_matter, new_content)
            
            print(f"✅ Successfully rewrote: {filepath.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error rewriting {filepath.name}: {str(e)}")
            return False
    
    def rewrite_all_posts(self):
        """Rewrite all paper posts"""
        print("🚀 Starting paper post rewriting process...")
        
        # Create backup
        self.create_backup()
        
        # Get all paper posts
        post_files = list(self.posts_dir.glob("*.md"))
        successful = 0
        failed = 0
        
        for post_file in post_files:
            if self.rewrite_post(post_file):
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Rewriting Summary:")
        print(f"✅ Successfully rewrote: {successful} posts")
        print(f"❌ Failed to rewrite: {failed} posts")
        print(f"📁 Backup saved at: {self.backup_dir}")
        
        return successful, failed

def main():
    """Main execution function"""
    rewriter = PaperPostRewriter()
    
    print("=" * 60)
    print("📚 Paper Post Rewriter - Template Upgrade")
    print("=" * 60)
    
    # Auto-proceed with rewriting
    print("🚀 Auto-proceeding with rewriting process...")
    
    # Execute rewriting
    successful, failed = rewriter.rewrite_all_posts()
    
    if successful > 0:
        print(f"\n🎉 Rewriting completed! {successful} posts updated.")
        print("📝 All posts now follow the new template structure.")
    
    if failed > 0:
        print(f"\n⚠️  {failed} posts failed. Check backup and retry if needed.")

if __name__ == "__main__":
    main()