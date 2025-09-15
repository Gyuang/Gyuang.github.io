#!/usr/bin/env python3
"""
Extract Missing ArXiv IDs and Process Additional Papers

This script searches for ArXiv IDs that were missed in the first pass
and attempts to download and process those papers.
"""

import os
import re
import sys
import json
import logging
import requests
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time

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

class MissingArXivExtractor:
    def __init__(self, base_dir: str = "/Users/gyu/Desktop/Gyuang.github.io"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "assets" / "images" / "paper"
        self.posts_dir = self.base_dir / "_posts" / "Paper"
        
        # Load existing analysis
        self.analysis_file = self.base_dir / "paper_analysis.json"
        self.existing_papers = self.load_existing_analysis()
        
        # Keep track of found ArXiv IDs
        self.found_arxiv_ids: Set[str] = set()
        
        # Enhanced patterns for ArXiv extraction
        self.arxiv_patterns = [
            r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'https://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv:([0-9]{4}\.[0-9]{4,5})',
            r'\[([0-9]{4}\.[0-9]{4,5})\]',
            r'arXiv:([0-9]{4}\.[0-9]{4,5})',
            r'(\d{4}\.\d{4,5})',  # Just the ID itself
        ]

    def load_existing_analysis(self) -> Dict:
        """Load the existing paper analysis"""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Analysis file not found: {self.analysis_file}")
            return {}

    def extract_all_arxiv_ids_from_content(self, content: str) -> List[str]:
        """Extract all possible ArXiv IDs from content"""
        found_ids = []
        
        for pattern in self.arxiv_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Validate ArXiv ID format (YYMM.NNNNN)
                if re.match(r'^\d{4}\.\d{4,5}$', match):
                    found_ids.append(match)
        
        return list(set(found_ids))  # Remove duplicates

    def find_main_arxiv_id_for_paper(self, post_file: Path) -> Optional[str]:
        """Find the main ArXiv ID for a paper post"""
        try:
            with open(post_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            content = post.content
            title = post.metadata.get('title', '')
            
            # Extract all ArXiv IDs from content
            arxiv_ids = self.extract_all_arxiv_ids_from_content(content)
            
            if not arxiv_ids:
                return None
            
            # If we found exactly one, return it
            if len(arxiv_ids) == 1:
                return arxiv_ids[0]
            
            # If multiple, try to find the main one based on context
            # Look for ArXiv IDs in headers or prominent positions
            lines = content.split('\n')
            for i, line in enumerate(lines[:20]):  # Check first 20 lines
                for arxiv_id in arxiv_ids:
                    if arxiv_id in line and ('arxiv' in line.lower() or 'paper' in line.lower()):
                        return arxiv_id
            
            # Return the first one as fallback
            return arxiv_ids[0]
            
        except Exception as e:
            logger.error(f"Error analyzing {post_file.name}: {e}")
            return None

    def scan_for_missing_arxiv_ids(self) -> Dict[str, str]:
        """Scan all posts for missing ArXiv IDs"""
        missing_arxiv = {}
        
        for paper_name, paper_info in self.existing_papers.items():
            if paper_info['arxiv_id'] is not None:
                continue  # Already has ArXiv ID
            
            post_file = self.posts_dir / paper_info['post_file']
            if not post_file.exists():
                continue
            
            arxiv_id = self.find_main_arxiv_id_for_paper(post_file)
            if arxiv_id:
                missing_arxiv[paper_name] = arxiv_id
                logger.info(f"Found ArXiv ID {arxiv_id} for {paper_name}")
        
        return missing_arxiv

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
                            img_name = self.generate_image_name(paper_name, page_num, img_index)
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

    def generate_image_name(self, paper_name: str, page_num: int, img_index: int) -> str:
        """Generate meaningful image names"""
        if page_num <= 2:
            if img_index == 0:
                return f"architecture_overview_{page_num}.png"
            else:
                return f"method_diagram_{page_num}_{img_index}.png"
        elif page_num > 5:  # Later pages likely contain results
            return f"results_table_{page_num}_{img_index}.png"
        else:
            return f"figure_{page_num}_{img_index}.png"

    def save_images(self, folder_name: str, images: List[Tuple[str, bytes]]) -> List[str]:
        """Save images to the paper directory"""
        paper_dir = self.images_dir / folder_name
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

    def optimize_images(self, folder_name: str) -> None:
        """Optimize images for web use"""
        paper_dir = self.images_dir / folder_name
        
        for img_path in paper_dir.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA'):
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

    def insert_images_into_post(self, post_file: str, images: List[str]) -> None:
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
            
            # Check if images are already inserted
            if any(img in content for img in images):
                logger.info(f"Images already present in {post_file}, skipping...")
                return
            
            # Find good insertion points
            insertion_points = self.find_insertion_points(content)
            
            # Insert images at appropriate locations
            modified_content = self.insert_images_at_points(content, images, insertion_points)
            
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
                if any(keyword in title for keyword in ['architecture', 'method', '방법론', '아키텍처', 'model', 'approach']):
                    points.append({
                        'line': i + 1,
                        'type': 'architecture',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['result', 'experiment', '결과', '실험', 'performance', 'evaluation']):
                    points.append({
                        'line': i + 1, 
                        'type': 'results',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['introduction', '개요', '핵심', 'overview']):
                    points.append({
                        'line': i + 1,
                        'type': 'overview', 
                        'title': match.group(2)
                    })
        
        return points

    def insert_images_at_points(self, content: str, images: List[str], insertion_points: List[Dict]) -> str:
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
            
            if section_type == 'architecture' and any(keyword in img_name for keyword in ['architecture', 'method', 'diagram', 'overview']):
                return img
            elif section_type == 'results' and any(keyword in img_name for keyword in ['result', 'table', 'performance']):
                return img
            elif section_type == 'overview' and any(keyword in img_name for keyword in ['overview', 'figure_0', 'figure_1']):
                return img
        
        # Fallback: return first available image
        return available_images[0] if available_images else None

    def process_missing_papers(self) -> Dict[str, bool]:
        """Process papers with newly found ArXiv IDs"""
        missing_arxiv = self.scan_for_missing_arxiv_ids()
        results = {}
        
        logger.info(f"Found {len(missing_arxiv)} papers with missing ArXiv IDs")
        
        for paper_name, arxiv_id in missing_arxiv.items():
            try:
                logger.info(f"Processing {paper_name} with ArXiv ID {arxiv_id}")
                
                # Get paper info
                paper_info = self.existing_papers[paper_name]
                
                # Download PDF
                pdf_content = self.download_arxiv_pdf(arxiv_id)
                if not pdf_content:
                    results[paper_name] = False
                    continue
                
                # Extract images
                images = self.extract_images_from_pdf(pdf_content, paper_name)
                if not images:
                    logger.warning(f"No images extracted from {paper_name}")
                    results[paper_name] = False
                    continue
                
                # Save images
                saved_images = self.save_images(paper_info['folder_name'], images)
                if not saved_images:
                    results[paper_name] = False
                    continue
                
                # Optimize images
                self.optimize_images(paper_info['folder_name'])
                
                # Insert into post
                self.insert_images_into_post(paper_info['post_file'], saved_images)
                
                # Update the analysis with the found ArXiv ID
                self.existing_papers[paper_name]['arxiv_id'] = arxiv_id
                
                results[paper_name] = True
                logger.info(f"Successfully processed {paper_name}")
                
                # Add delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process {paper_name}: {e}")
                results[paper_name] = False
        
        return results

    def save_updated_analysis(self) -> None:
        """Save the updated analysis with new ArXiv IDs"""
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.existing_papers, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated analysis saved to: {self.analysis_file}")

    def create_summary_report(self, results: Dict[str, bool]) -> str:
        """Create a summary report of the missing ArXiv processing"""
        from datetime import datetime
        report = []
        report.append("# Missing ArXiv IDs Processing Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        successful = sum(results.values())
        total = len(results)
        
        report.append("## Summary\n")
        report.append(f"- **Papers Processed**: {total}")
        report.append(f"- **Successfully Processed**: {successful}")
        report.append(f"- **Success Rate**: {successful/total*100:.1f}%\n" if total > 0 else "- **Success Rate**: 0%\n")
        
        if results:
            report.append("## Processing Results\n")
            for paper_name, success in results.items():
                status = "✅ Success" if success else "❌ Failed"
                arxiv_id = "Unknown"
                
                # Try to find the ArXiv ID
                if paper_name in self.existing_papers:
                    arxiv_id = self.existing_papers[paper_name].get('arxiv_id', 'Unknown')
                
                report.append(f"### {paper_name}")
                report.append(f"- **Status**: {status}")
                report.append(f"- **ArXiv ID**: {arxiv_id}")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    logger.info("Starting Missing ArXiv IDs Extraction")
    
    extractor = MissingArXivExtractor()
    
    # Process papers with missing ArXiv IDs
    results = extractor.process_missing_papers()
    
    # Save updated analysis
    extractor.save_updated_analysis()
    
    # Generate report
    report = extractor.create_summary_report(results)
    
    # Save report
    report_path = extractor.base_dir / "missing_arxiv_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Missing ArXiv processing complete. Report saved to: {report_path}")
    
    # Print summary
    successful = sum(results.values())
    total = len(results)
    print(f"\n{'='*60}")
    print(f"MISSING ARXIV PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Papers with missing ArXiv IDs found: {total}")
    print(f"Successfully processed: {successful}/{total}")
    print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "Success rate: 0%")
    print(f"Report location: {report_path}")
    
    return successful > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)