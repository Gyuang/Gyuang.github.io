#!/usr/bin/env python3
"""
Enhanced ArXiv Paper Image Extractor and Jekyll Blog Integrator

This script automates the process of:
1. Analyzing ALL paper posts to extract identifiers
2. Downloading PDFs from ArXiv and other sources
3. Extracting figures/diagrams from PDFs  
4. Organizing images in Jekyll directory structure
5. Auto-inserting images into ALL existing blog posts

Requirements:
    pip install pymupdf pillow requests beautifulsoup4 python-frontmatter lxml
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
from urllib.parse import urljoin, urlparse
import time
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
    import frontmatter
    from PIL import Image
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install pymupdf pillow requests beautifulsoup4 python-frontmatter lxml")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PaperInfo:
    """Data class to store paper information"""
    title: str
    post_file: str
    folder_name: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    other_urls: List[str] = None
    paper_title_en: Optional[str] = None
    paper_title_kr: Optional[str] = None

class EnhancedPaperExtractor:
    def __init__(self, base_dir: str = "/Users/gyu/Desktop/Gyuang.github.io"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "assets" / "images" / "paper"
        self.posts_dir = self.base_dir / "_posts" / "Paper"
        
        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all discovered papers
        self.papers: Dict[str, PaperInfo] = {}
        
        # Common patterns for extraction
        self.arxiv_patterns = [
            r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv:([0-9]{4}\.[0-9]{4,5})',
            r'\[([0-9]{4}\.[0-9]{4,5})\]',
            r'arXiv:([0-9]{4}\.[0-9]{4,5})'
        ]
        
        self.doi_patterns = [
            r'doi\.org/(10\.[0-9]+/[^\s\)]+)',
            r'DOI:?\s*(10\.[0-9]+/[^\s\)]+)',
            r'https?://dx\.doi\.org/(10\.[0-9]+/[^\s\)]+)'
        ]

    def analyze_all_posts(self) -> Dict[str, PaperInfo]:
        """Analyze all paper posts to extract paper information"""
        logger.info("Analyzing all paper posts...")
        
        post_files = list(self.posts_dir.glob("*.md"))
        logger.info(f"Found {len(post_files)} paper posts")
        
        for post_file in post_files:
            try:
                paper_info = self.analyze_single_post(post_file)
                if paper_info:
                    self.papers[paper_info.title] = paper_info
                    logger.info(f"Analyzed: {paper_info.title}")
            except Exception as e:
                logger.error(f"Failed to analyze {post_file.name}: {e}")
        
        logger.info(f"Successfully analyzed {len(self.papers)} papers")
        return self.papers

    def analyze_single_post(self, post_file: Path) -> Optional[PaperInfo]:
        """Analyze a single post file to extract paper information"""
        try:
            with open(post_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            content = post.content
            metadata = post.metadata
            
            # Extract basic info
            title = metadata.get('title', post_file.stem)
            folder_name = self.generate_folder_name(title, post_file.name)
            
            # Extract ArXiv ID
            arxiv_id = self.extract_arxiv_id(content)
            
            # Extract DOI
            doi = self.extract_doi(content)
            
            # Extract other URLs
            pdf_url, other_urls = self.extract_urls(content)
            
            # Extract paper titles
            paper_title_en, paper_title_kr = self.extract_paper_titles(content, metadata)
            
            paper_info = PaperInfo(
                title=title,
                post_file=post_file.name,
                folder_name=folder_name,
                arxiv_id=arxiv_id,
                doi=doi,
                pdf_url=pdf_url,
                other_urls=other_urls or [],
                paper_title_en=paper_title_en,
                paper_title_kr=paper_title_kr
            )
            
            return paper_info
            
        except Exception as e:
            logger.error(f"Error analyzing {post_file.name}: {e}")
            return None

    def extract_arxiv_id(self, content: str) -> Optional[str]:
        """Extract ArXiv ID from content"""
        for pattern in self.arxiv_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0]
        return None

    def extract_doi(self, content: str) -> Optional[str]:
        """Extract DOI from content"""
        for pattern in self.doi_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0]
        return None

    def extract_urls(self, content: str) -> Tuple[Optional[str], List[str]]:
        """Extract PDF URLs and other relevant URLs"""
        pdf_url = None
        other_urls = []
        
        # Find all URLs
        url_pattern = r'https?://[^\s\)\]>]+'
        urls = re.findall(url_pattern, content)
        
        for url in urls:
            # Clean up URL
            url = url.rstrip('.,;:)')
            
            if 'pdf' in url.lower() or url.endswith('.pdf'):
                pdf_url = url
            elif any(domain in url for domain in ['nature.com', 'ieee.org', 'acm.org', 'springer.com', 'elsevier.com']):
                other_urls.append(url)
        
        return pdf_url, other_urls

    def extract_paper_titles(self, content: str, metadata: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract English and Korean paper titles"""
        # Try to get from metadata first
        title = metadata.get('title', '')
        
        # Look for patterns in content
        title_patterns = [
            r'^#\s+(.+)$',  # Main heading
            r'title:\s*"([^"]+)"',  # YAML title
            r'^\*\*([^*]+)\*\*',  # Bold text
        ]
        
        paper_title_en = None
        paper_title_kr = None
        
        # Determine if title contains Korean characters
        if re.search(r'[ㄱ-ㅎ가-힣]', title):
            paper_title_kr = title
        else:
            paper_title_en = title
        
        return paper_title_en, paper_title_kr

    def generate_folder_name(self, title: str, filename: str) -> str:
        """Generate a folder name for organizing images"""
        # Remove date prefix from filename
        clean_name = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', filename)
        clean_name = clean_name.replace('.md', '')
        
        # Convert to lowercase and replace spaces/special chars with hyphens
        folder_name = re.sub(r'[^\w\s-]', '', clean_name.lower())
        folder_name = re.sub(r'[-\s]+', '-', folder_name)
        folder_name = folder_name.strip('-')
        
        return folder_name

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

    def try_alternative_sources(self, paper_info: PaperInfo) -> Optional[bytes]:
        """Try to find PDFs from alternative sources"""
        
        # Try DOI-based search
        if paper_info.doi:
            pdf_content = self.search_by_doi(paper_info.doi)
            if pdf_content:
                return pdf_content
        
        # Try direct PDF URL
        if paper_info.pdf_url:
            pdf_content = self.download_direct_pdf(paper_info.pdf_url)
            if pdf_content:
                return pdf_content
        
        # Try searching by title
        if paper_info.paper_title_en:
            pdf_content = self.search_by_title(paper_info.paper_title_en)
            if pdf_content:
                return pdf_content
        
        return None

    def search_by_doi(self, doi: str) -> Optional[bytes]:
        """Search for PDF using DOI"""
        try:
            # Try Sci-Hub (use with caution and respect copyright)
            # Note: This is for educational purposes only
            logger.info(f"Searching by DOI: {doi}")
            return None  # Placeholder - implement if needed
        except Exception as e:
            logger.error(f"DOI search failed for {doi}: {e}")
            return None

    def download_direct_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF from direct URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            if response.headers.get('content-type', '').startswith('application/pdf'):
                return response.content
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
        return None

    def search_by_title(self, title: str) -> Optional[bytes]:
        """Search for PDF by paper title"""
        # This could be extended to search academic databases
        logger.info(f"Title-based search not implemented for: {title}")
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
                elif page_num > 5:  # Later pages likely contain results
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
            
            # Check if images are already inserted
            if any(img in content for img in images):
                logger.info(f"Images already present in {post_file}, skipping...")
                return
            
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
                if any(keyword in title for keyword in ['architecture', 'method', '방법론', '아키텍처', 'overview', 'model']):
                    points.append({
                        'line': i + 1,
                        'type': 'architecture',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['result', 'experiment', '결과', '실험', 'performance']):
                    points.append({
                        'line': i + 1, 
                        'type': 'results',
                        'title': match.group(2)
                    })
                elif any(keyword in title for keyword in ['introduction', '개요', '핵심', 'approach']):
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
            
            if section_type == 'architecture' and any(keyword in img_name for keyword in ['architecture', 'method', 'diagram', 'overview']):
                return img
            elif section_type == 'results' and any(keyword in img_name for keyword in ['result', 'table', 'performance']):
                return img
            elif section_type == 'overview' and any(keyword in img_name for keyword in ['overview', 'figure_0', 'figure_1']):
                return img
        
        # Fallback: return first available image
        return available_images[0] if available_images else None

    def process_paper(self, paper_name: str, paper_info: PaperInfo) -> bool:
        """Process a single paper: download, extract, organize, and integrate"""
        logger.info(f"Processing paper: {paper_name}")
        
        # Try ArXiv first
        pdf_content = None
        if paper_info.arxiv_id:
            pdf_content = self.download_arxiv_pdf(paper_info.arxiv_id)
        
        # Try alternative sources if ArXiv failed
        if not pdf_content:
            pdf_content = self.try_alternative_sources(paper_info)
        
        if not pdf_content:
            logger.warning(f"Could not find PDF for {paper_name}")
            return False
        
        # Extract images
        images = self.extract_images_from_pdf(pdf_content, paper_name)
        if not images:
            logger.warning(f"No images extracted from {paper_name}")
            return False
        
        # Save images
        saved_images = self.save_images(paper_info.folder_name, images)
        if not saved_images:
            return False
        
        # Optimize images
        self.optimize_images(paper_info.folder_name)
        
        # Insert into post
        self.insert_images_into_post(paper_info.post_file, paper_info.folder_name, saved_images)
        
        logger.info(f"Successfully processed {paper_name}")
        return True

    def process_all_papers(self) -> Dict[str, bool]:
        """Process all discovered papers"""
        results = {}
        
        for paper_name, paper_info in self.papers.items():
            try:
                # Add small delay between requests to be respectful
                time.sleep(1)
                results[paper_name] = self.process_paper(paper_name, paper_info)
            except Exception as e:
                logger.error(f"Failed to process {paper_name}: {e}")
                results[paper_name] = False
        
        return results

    def save_paper_analysis(self) -> str:
        """Save the paper analysis to a JSON file"""
        analysis_file = self.base_dir / "paper_analysis.json"
        
        # Convert PaperInfo objects to dictionaries
        papers_dict = {}
        for name, info in self.papers.items():
            papers_dict[name] = {
                'title': info.title,
                'post_file': info.post_file,
                'folder_name': info.folder_name,
                'arxiv_id': info.arxiv_id,
                'doi': info.doi,
                'pdf_url': info.pdf_url,
                'other_urls': info.other_urls,
                'paper_title_en': info.paper_title_en,
                'paper_title_kr': info.paper_title_kr
            }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(papers_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Paper analysis saved to: {analysis_file}")
        return str(analysis_file)

    def create_extraction_report(self, results: Dict[str, bool]) -> str:
        """Create a comprehensive summary report"""
        from datetime import datetime
        report = []
        report.append("# Enhanced Paper Image Extraction Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Summary Statistics\n")
        total_papers = len(self.papers)
        arxiv_papers = sum(1 for p in self.papers.values() if p.arxiv_id)
        doi_papers = sum(1 for p in self.papers.values() if p.doi)
        processed_papers = sum(results.values())
        
        report.append(f"- **Total Papers Found**: {total_papers}")
        report.append(f"- **Papers with ArXiv IDs**: {arxiv_papers}")
        report.append(f"- **Papers with DOIs**: {doi_papers}")
        report.append(f"- **Successfully Processed**: {processed_papers}")
        report.append(f"- **Success Rate**: {processed_papers/total_papers*100:.1f}%\n")
        
        report.append("## Processing Results\n")
        for paper_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            paper_info = self.papers[paper_name]
            
            report.append(f"### {paper_name}")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Post File**: {paper_info.post_file}")
            report.append(f"- **ArXiv ID**: {paper_info.arxiv_id or 'Not found'}")
            report.append(f"- **DOI**: {paper_info.doi or 'Not found'}")
            report.append(f"- **Image Folder**: {paper_info.folder_name}")
            
            # Check if images were created
            paper_dir = self.images_dir / paper_info.folder_name
            if paper_dir.exists():
                images = list(paper_dir.glob("*.png"))
                report.append(f"- **Images Extracted**: {len(images)}")
                for img in images:
                    report.append(f"  - {img.name}")
            report.append("")
        
        report.append("## Papers Without ArXiv IDs\n")
        no_arxiv = [name for name, info in self.papers.items() if not info.arxiv_id]
        if no_arxiv:
            report.append("These papers may require manual PDF sourcing:\n")
            for paper_name in no_arxiv:
                info = self.papers[paper_name]
                report.append(f"- **{paper_name}**")
                report.append(f"  - Post: {info.post_file}")
                report.append(f"  - DOI: {info.doi or 'Not found'}")
                report.append(f"  - PDF URL: {info.pdf_url or 'Not found'}")
                report.append("")
        else:
            report.append("All papers have ArXiv IDs! 🎉\n")
        
        report.append("## Directory Structure Created\n")
        report.append("```")
        report.append("assets/images/paper/")
        for paper_info in self.papers.values():
            paper_dir = self.images_dir / paper_info.folder_name
            if paper_dir.exists():
                report.append(f"├── {paper_info.folder_name}/")
                for img in paper_dir.glob("*.png"):
                    report.append(f"│   ├── {img.name}")
        report.append("```")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    logger.info("Starting Enhanced Paper Image Extraction")
    
    extractor = EnhancedPaperExtractor()
    
    # Step 1: Analyze all posts
    logger.info("Step 1: Analyzing all paper posts...")
    papers = extractor.analyze_all_posts()
    
    # Save analysis
    analysis_file = extractor.save_paper_analysis()
    
    # Step 2: Process all papers
    logger.info("Step 2: Processing all papers...")
    results = extractor.process_all_papers()
    
    # Step 3: Generate report
    logger.info("Step 3: Generating comprehensive report...")
    report = extractor.create_extraction_report(results)
    
    # Save report
    report_path = extractor.base_dir / "enhanced_extraction_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Extraction complete. Report saved to: {report_path}")
    
    # Print summary
    successful = sum(results.values())
    total = len(results)
    print(f"\n{'='*60}")
    print(f"ENHANCED EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Papers analyzed: {len(papers)}")
    print(f"Papers with ArXiv IDs: {sum(1 for p in papers.values() if p.arxiv_id)}")
    print(f"Papers with DOIs: {sum(1 for p in papers.values() if p.doi)}")
    print(f"Successfully processed: {successful}/{total} papers")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Analysis file: {analysis_file}")
    print(f"Report location: {report_path}")
    
    if successful > 0:
        print(f"\nImages have been extracted and integrated into your Jekyll posts!")
        print(f"Check the assets/images/paper/ directory for the extracted images.")
    
    return successful > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)