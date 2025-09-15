#!/usr/bin/env python3
"""
ArXiv Paper Image Extractor and Jekyll Blog Integrator

This script automates the process of:
1. Downloading PDFs from ArXiv URLs
2. Extracting figures/diagrams from PDFs
3. Organizing images in Jekyll directory structure
4. Auto-inserting images into existing blog posts

Requirements:
    pip install pymupdf pillow requests beautifulsoup4 python-frontmatter

Usage:
    python paper_image_extractor.py
"""

import os
import re
import sys
import json
import logging
import requests
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin

try:
    import fitz  # PyMuPDF
    import frontmatter
    from PIL import Image
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install pymupdf pillow requests beautifulsoup4 python-frontmatter")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArXivImageExtractor:
    def __init__(self, base_dir: str = "/Users/gyu/Desktop/Gyuang.github.io"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "assets" / "images" / "paper"
        self.posts_dir = self.base_dir / "_posts" / "Paper"
        
        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper configurations
        self.papers = {
            "P-Tuning v2": {
                "arxiv_id": "2110.07602",
                "post_file": "2025-09-14-P-Tuning-v2.md",
                "folder_name": "p-tuning-v2"
            },
            "Prefix-Tuning": {
                "arxiv_id": "2101.00190", 
                "post_file": "2025-09-14-Prefix-Tuning.md",
                "folder_name": "prefix-tuning"
            },
            "P-Tuning": {
                "arxiv_id": "2103.10385",
                "post_file": "2025-09-14-P-Tuning.md", 
                "folder_name": "p-tuning"
            },
            "Power of Scale": {
                "arxiv_id": "2104.08691",
                "post_file": "2025-09-14-Power-of-Scale-Prompt-Tuning.md",
                "folder_name": "power-of-scale"
            }
        }

    def download_arxiv_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF from ArXiv"""
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        logger.info(f"Downloading PDF from {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF {arxiv_id}: {e}")
            return None

    def extract_images_from_pdf(self, pdf_content: bytes, paper_name: str) -> List[Tuple[str, bytes]]:
        """Extract images from PDF using PyMuPDF"""
        images = []
        
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(pdf_content)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                logger.info(f"Processing {len(doc)} pages for {paper_name}")
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Extract image
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Skip small images (likely icons or decorations)
                            if pix.width < 100 or pix.height < 100:
                                pix = None
                                continue
                            
                            # Convert CMYK to RGB if needed
                            if pix.n - pix.alpha < 4:
                                img_data = pix.pil_tobytes(format="PNG")
                            else:
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                img_data = pix1.pil_tobytes(format="PNG")
                                pix1 = None
                            
                            # Generate meaningful filename
                            img_name = self.generate_image_name(paper_name, page_num, img_index, img_data)
                            images.append((img_name, img_data))
                            
                            pix = None
                            
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                            continue
                
                doc.close()
                
        except Exception as e:
            logger.error(f"Failed to process PDF for {paper_name}: {e}")
        
        logger.info(f"Extracted {len(images)} images from {paper_name}")
        return images

    def generate_image_name(self, paper_name: str, page_num: int, img_index: int, img_data: bytes) -> str:
        """Generate meaningful image names based on content analysis"""
        
        # Try to determine image type by analyzing content
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(img_data)
                temp_file.flush()
                
                img = Image.open(temp_file.name)
                width, height = img.size
                
                # Classify based on dimensions and position
                if page_num <= 2:
                    if width > height and width > 600:
                        return f"architecture_overview_{page_num}.png"
                    elif height > width:
                        return f"method_diagram_{page_num}.png"
                elif "result" in paper_name.lower() or page_num > len(self.papers) // 2:
                    return f"results_table_{page_num}_{img_index}.png"
                elif width > height:
                    return f"architecture_diagram_{page_num}_{img_index}.png"
                else:
                    return f"figure_{page_num}_{img_index}.png"
        except:
            pass
        
        # Fallback naming
        return f"figure_{page_num}_{img_index}.png"

    def save_images(self, paper_folder: str, images: List[Tuple[str, bytes]]) -> List[str]:
        """Save images to the paper directory"""
        paper_dir = self.images_dir / paper_folder
        paper_dir.mkdir(exist_ok=True)
        
        saved_images = []
        for img_name, img_data in images:
            img_path = paper_dir / img_name
            
            try:
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                saved_images.append(str(img_path.relative_to(self.base_dir)))
                logger.info(f"Saved image: {img_path}")
            except Exception as e:
                logger.error(f"Failed to save image {img_name}: {e}")
        
        return saved_images

    def optimize_images(self, paper_folder: str) -> None:
        """Optimize images for web use"""
        paper_dir = self.images_dir / paper_folder
        
        for img_path in paper_dir.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA'):
                        # Create white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                    
                    # Resize if too large
                    if img.width > 1200:
                        ratio = 1200 / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((1200, new_height), Image.Resampling.LANCZOS)
                    
                    # Save optimized image
                    img.save(img_path, "PNG", optimize=True, quality=85)
                    
            except Exception as e:
                logger.warning(f"Failed to optimize image {img_path}: {e}")

    def insert_images_into_post(self, post_file: str, paper_folder: str, images: List[str]) -> None:
        """Insert image references into Jekyll post"""
        post_path = self.posts_dir / post_file
        
        if not post_path.exists():
            logger.error(f"Post file not found: {post_path}")
            return
        
        try:
            # Read the post
            with open(post_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            content = post.content
            
            # Find good insertion points
            insertion_points = self.find_insertion_points(content)
            
            # Insert images at appropriate locations
            modified_content = self.insert_images_at_points(content, images, insertion_points, paper_folder)
            
            # Update the post
            post.content = modified_content
            
            # Write back to file
            with open(post_path, 'w', encoding='utf-8') as f:
                f.write(frontmatter.dumps(post))
            
            logger.info(f"Updated post: {post_file}")
            
        except Exception as e:
            logger.error(f"Failed to update post {post_file}: {e}")

    def find_insertion_points(self, content: str) -> List[Dict]:
        """Find good locations to insert images in the content"""
        points = []
        
        # Look for headers
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                level = len(match.group(1))
                title = match.group(2).lower()
                
                # Determine image type based on section title
                if any(keyword in title for keyword in ['architecture', 'method', '방법론', '아키텍처']):
                    points.append({
                        'line': i + 1,
                        'type': 'architecture',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['result', 'experiment', '결과', '실험']):
                    points.append({
                        'line': i + 1, 
                        'type': 'results',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['overview', 'introduction', '개요', '핵심']):
                    points.append({
                        'line': i + 1,
                        'type': 'overview', 
                        'title': match.group(2)
                    })
        
        return points

    def insert_images_at_points(self, content: str, images: List[str], insertion_points: List[Dict], paper_folder: str) -> str:
        """Insert images at the identified points"""
        lines = content.split('\n')
        
        # Sort insertion points by line number (reverse order for easier insertion)
        insertion_points.sort(key=lambda x: x['line'], reverse=True)
        
        # Match images to insertion points
        used_images = set()
        
        for point in insertion_points:
            # Find the best image for this section
            best_image = self.match_image_to_section(images, point, used_images)
            
            if best_image:
                used_images.add(best_image)
                
                # Create image markdown
                img_name = Path(best_image).stem.replace('_', ' ').title()
                img_markdown = f"\n![{img_name}](/{best_image})\n*Figure: {img_name}*\n"
                
                # Insert after the header
                if point['line'] < len(lines):
                    lines.insert(point['line'], img_markdown)
        
        # Add any remaining images at the end
        remaining_images = [img for img in images if img not in used_images]
        if remaining_images:
            lines.append("\n## Additional Figures\n")
            for img in remaining_images:
                img_name = Path(img).stem.replace('_', ' ').title()
                img_markdown = f"\n![{img_name}](/{img})\n*Figure: {img_name}*\n"
                lines.append(img_markdown)
        
        return '\n'.join(lines)

    def match_image_to_section(self, images: List[str], point: Dict, used_images: set) -> Optional[str]:
        """Match the best image to a section based on naming and type"""
        available_images = [img for img in images if img not in used_images]
        
        if not available_images:
            return None
        
        section_type = point['type']
        
        # Priority matching based on section type
        for img in available_images:
            img_name = Path(img).stem.lower()
            
            if section_type == 'architecture' and any(keyword in img_name for keyword in ['architecture', 'method', 'diagram']):
                return img
            elif section_type == 'results' and any(keyword in img_name for keyword in ['result', 'table', 'performance']):
                return img
            elif section_type == 'overview' and any(keyword in img_name for keyword in ['overview', 'figure_0', 'figure_1']):
                return img
        
        # Fallback: return first available image
        return available_images[0] if available_images else None

    def process_paper(self, paper_name: str, config: Dict) -> bool:
        """Process a single paper: download, extract, organize, and integrate"""
        logger.info(f"Processing paper: {paper_name}")
        
        # Download PDF
        pdf_content = self.download_arxiv_pdf(config['arxiv_id'])
        if not pdf_content:
            return False
        
        # Extract images
        images = self.extract_images_from_pdf(pdf_content, paper_name)
        if not images:
            logger.warning(f"No images extracted from {paper_name}")
            return False
        
        # Save images
        saved_images = self.save_images(config['folder_name'], images)
        if not saved_images:
            return False
        
        # Optimize images
        self.optimize_images(config['folder_name'])
        
        # Insert into post
        self.insert_images_into_post(config['post_file'], config['folder_name'], saved_images)
        
        logger.info(f"Successfully processed {paper_name}")
        return True

    def process_all_papers(self) -> Dict[str, bool]:
        """Process all configured papers"""
        results = {}
        
        for paper_name, config in self.papers.items():
            try:
                results[paper_name] = self.process_paper(paper_name, config)
            except Exception as e:
                logger.error(f"Failed to process {paper_name}: {e}")
                results[paper_name] = False
        
        return results

    def create_extraction_report(self, results: Dict[str, bool]) -> str:
        """Create a summary report of the extraction process"""
        from datetime import datetime
        report = []
        report.append("# ArXiv Paper Image Extraction Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Processing Results\n")
        for paper_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            config = self.papers[paper_name]
            
            report.append(f"### {paper_name}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **ArXiv ID**: {config['arxiv_id']}")
            report.append(f"- **Post File**: {config['post_file']}")
            report.append(f"- **Image Folder**: {config['folder_name']}")
            
            # Check if images were created
            paper_dir = self.images_dir / config['folder_name']
            if paper_dir.exists():
                images = list(paper_dir.glob("*.png"))
                report.append(f"- **Images Extracted**: {len(images)}")
                for img in images:
                    report.append(f"  - {img.name}")
            report.append("")
        
        report.append("## Directory Structure Created\n")
        report.append("```")
        report.append("assets/images/paper/")
        for config in self.papers.values():
            paper_dir = self.images_dir / config['folder_name']
            if paper_dir.exists():
                report.append(f"├── {config['folder_name']}/")
                for img in paper_dir.glob("*.png"):
                    report.append(f"│   ├── {img.name}")
        report.append("```")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    logger.info("Starting ArXiv Paper Image Extraction")
    
    extractor = ArXivImageExtractor()
    
    # Process all papers
    results = extractor.process_all_papers()
    
    # Generate report
    report = extractor.create_extraction_report(results)
    
    # Save report
    report_path = extractor.base_dir / "paper_extraction_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Extraction complete. Report saved to: {report_path}")
    
    # Print summary
    successful = sum(results.values())
    total = len(results)
    print(f"\n{'='*50}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Successfully processed: {successful}/{total} papers")
    print(f"Report location: {report_path}")
    
    if successful > 0:
        print(f"\nImages have been extracted and integrated into your Jekyll posts!")
        print(f"Check the assets/images/paper/ directory for the extracted images.")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)