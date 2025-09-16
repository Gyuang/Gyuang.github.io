#!/usr/bin/env python3
"""
Paper Posts Cleanup Script

This script analyzes and cleans up automatically inserted images and other issues
in paper posts for the Jekyll blog.

Issues addressed:
1. Excessive duplicate/similar images (especially tables and diagrams)
2. Poor image captions like "Figure: Method Diagram 1 3"
3. Images breaking text flow
4. Front matter inconsistencies
5. Duplicate content sections
6. Missing or generic image alt text

Author: Claude Code Assistant
Created: 2025-09-16
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperPostCleaner:
    """Main class for cleaning paper posts"""
    
    def __init__(self, posts_dir: str, dry_run: bool = True):
        self.posts_dir = Path(posts_dir)
        self.dry_run = dry_run
        self.issues_found = defaultdict(list)
        self.cleanup_stats = defaultdict(int)
        
        # Patterns for different types of issues
        self.patterns = {
            'image_markdown': re.compile(r'!\[(.*?)\]\((.*?)\)'),
            'generic_caption': re.compile(r'\*Figure: (.+?)\*'),
            'duplicate_section': re.compile(r'^## (.+?)$', re.MULTILINE),
            'method_diagram': re.compile(r'method_diagram_\d+_\d+\.png'),
            'results_table': re.compile(r'results_table_\d+_\d+\.png'),
            'architecture_overview': re.compile(r'architecture_overview_\d+\.png'),
            'excessive_newlines': re.compile(r'\n{3,}'),
            'empty_sections': re.compile(r'^## .+\n\n(?=## |$)', re.MULTILINE)
        }
        
        # High-value papers to prioritize
        self.priority_papers = {
            'clip', 'p-tuning', 'visual-prompt-tuning', 'coop', 'cocoop', 
            'maple', 'biomed-dpt', 'voxelprompt', 'chatcad', 'llava'
        }
    
    def analyze_all_posts(self) -> Dict[str, List[Dict]]:
        """Analyze all paper posts and categorize issues"""
        logger.info(f"Analyzing posts in {self.posts_dir}")
        
        all_issues = {
            'critical': [],      # Breaks page rendering
            'high': [],         # Significantly impacts readability  
            'medium': [],       # Minor formatting issues
            'low': []          # Cosmetic improvements
        }
        
        for post_file in self.posts_dir.glob("*.md"):
            if post_file.name.startswith('.'):
                continue
                
            logger.info(f"Analyzing {post_file.name}")
            post_issues = self.analyze_single_post(post_file)
            
            # Categorize by severity
            for issue in post_issues:
                severity = self.determine_severity(issue, post_file)
                all_issues[severity].append({
                    'file': post_file,
                    'issue': issue,
                    'priority': self.is_priority_paper(post_file)
                })
        
        return all_issues
    
    def analyze_single_post(self, post_file: Path) -> List[Dict]:
        """Analyze a single post for issues"""
        issues = []
        
        try:
            with open(post_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {post_file}: {e}")
            return issues
        
        # Parse front matter and content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                front_matter = parts[1]
                post_content = parts[2]
            else:
                front_matter = ""
                post_content = content
        else:
            front_matter = ""
            post_content = content
        
        # Check for various issues
        issues.extend(self.check_image_issues(post_content, post_file))
        issues.extend(self.check_content_structure_issues(post_content))
        issues.extend(self.check_front_matter_issues(front_matter))
        issues.extend(self.check_duplicate_content(post_content))
        
        return issues
    
    def check_image_issues(self, content: str, post_file: Path) -> List[Dict]:
        """Check for image-related issues"""
        issues = []
        images = self.patterns['image_markdown'].findall(content)
        
        if not images:
            return issues
        
        # Count image types
        image_types = defaultdict(int)
        for alt_text, src in images:
            if 'method_diagram' in src:
                image_types['method_diagrams'] += 1
            elif 'results_table' in src:
                image_types['results_tables'] += 1
            elif 'architecture_overview' in src:
                image_types['architecture_overviews'] += 1
            elif 'figure' in src:
                image_types['figures'] += 1
        
        # Check for excessive similar images
        if image_types['results_tables'] > 10:
            issues.append({
                'type': 'excessive_results_tables',
                'count': image_types['results_tables'],
                'description': f"Too many results tables ({image_types['results_tables']}), should limit to 3-5 most important ones",
                'severity': 'high'
            })
        
        if image_types['method_diagrams'] > 8:
            issues.append({
                'type': 'excessive_method_diagrams', 
                'count': image_types['method_diagrams'],
                'description': f"Too many method diagrams ({image_types['method_diagrams']}), should consolidate",
                'severity': 'medium'
            })
        
        # Check for poor captions
        poor_captions = 0
        for alt_text, src in images:
            if re.match(r'^(Figure|Method Diagram|Results Table|Architecture Overview)\s*\d+\s*\d*$', alt_text.strip()):
                poor_captions += 1
        
        if poor_captions > len(images) * 0.8:  # More than 80% poor captions
            issues.append({
                'type': 'poor_image_captions',
                'count': poor_captions,
                'total_images': len(images),
                'description': f"Most images have generic captions ({poor_captions}/{len(images)})",
                'severity': 'high'
            })
        
        # Check for images breaking text flow
        lines = content.split('\n')
        image_disruptions = 0
        for i, line in enumerate(lines):
            if self.patterns['image_markdown'].match(line.strip()):
                # Check if image is between short paragraphs
                prev_content = lines[i-1].strip() if i > 0 else ""
                next_content = lines[i+2].strip() if i < len(lines)-2 else ""
                
                if (len(prev_content) < 100 and len(next_content) < 100 and 
                    prev_content and next_content):
                    image_disruptions += 1
        
        if image_disruptions > 3:
            issues.append({
                'type': 'images_breaking_text_flow',
                'count': image_disruptions,
                'description': f"Images disrupting text flow in {image_disruptions} places",
                'severity': 'medium'
            })
        
        return issues
    
    def check_content_structure_issues(self, content: str) -> List[Dict]:
        """Check for content structure issues"""
        issues = []
        
        # Check for duplicate sections
        sections = self.patterns['duplicate_section'].findall(content)
        section_counts = Counter(sections)
        duplicates = {section: count for section, count in section_counts.items() if count > 1}
        
        if duplicates:
            issues.append({
                'type': 'duplicate_sections',
                'duplicates': duplicates,
                'description': f"Duplicate sections found: {list(duplicates.keys())}",
                'severity': 'high'
            })
        
        # Check for excessive newlines
        excessive_newlines = len(self.patterns['excessive_newlines'].findall(content))
        if excessive_newlines > 5:
            issues.append({
                'type': 'excessive_whitespace',
                'count': excessive_newlines,
                'description': f"Excessive whitespace in {excessive_newlines} places",
                'severity': 'low'
            })
        
        # Check for empty sections
        empty_sections = len(self.patterns['empty_sections'].findall(content))
        if empty_sections > 0:
            issues.append({
                'type': 'empty_sections',
                'count': empty_sections,
                'description': f"Empty sections found: {empty_sections}",
                'severity': 'medium'
            })
        
        return issues
    
    def check_front_matter_issues(self, front_matter: str) -> List[Dict]:
        """Check for front matter issues"""
        issues = []
        
        if not front_matter.strip():
            issues.append({
                'type': 'missing_front_matter',
                'description': "No front matter found",
                'severity': 'critical'
            })
            return issues
        
        try:
            fm_data = yaml.safe_load(front_matter)
        except yaml.YAMLError:
            issues.append({
                'type': 'invalid_yaml_front_matter',
                'description': "Invalid YAML in front matter",
                'severity': 'critical'
            })
            return issues
        
        # Check required fields
        required_fields = ['title', 'categories', 'tags', 'published']
        missing_fields = [field for field in required_fields if field not in fm_data]
        
        if missing_fields:
            issues.append({
                'type': 'missing_front_matter_fields',
                'missing_fields': missing_fields,
                'description': f"Missing required fields: {missing_fields}",
                'severity': 'high'
            })
        
        # Check for inconsistent tag formatting
        if 'tags' in fm_data and isinstance(fm_data['tags'], list):
            if len(fm_data['tags']) > 0 and isinstance(fm_data['tags'][0], list):
                issues.append({
                    'type': 'nested_tags_format',
                    'description': "Tags are in nested list format (should be flat)",
                    'severity': 'medium'
                })
        
        return issues
    
    def check_duplicate_content(self, content: str) -> List[Dict]:
        """Check for duplicate content blocks"""
        issues = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Look for similar paragraphs (simplified check)
        paragraph_counts = Counter(paragraphs)
        duplicates = {para: count for para, count in paragraph_counts.items() 
                     if count > 1 and len(para) > 100}
        
        if duplicates:
            issues.append({
                'type': 'duplicate_content_blocks',
                'count': len(duplicates),
                'description': f"Duplicate content blocks found: {len(duplicates)}",
                'severity': 'high'
            })
        
        return issues
    
    def determine_severity(self, issue: Dict, post_file: Path) -> str:
        """Determine the severity of an issue"""
        if 'severity' in issue:
            return issue['severity']
        
        # Default severity determination logic
        issue_type = issue.get('type', '')
        
        if issue_type in ['missing_front_matter', 'invalid_yaml_front_matter']:
            return 'critical'
        elif issue_type in ['excessive_results_tables', 'poor_image_captions', 'duplicate_sections']:
            return 'high'
        elif issue_type in ['excessive_method_diagrams', 'images_breaking_text_flow', 'empty_sections']:
            return 'medium'
        else:
            return 'low'
    
    def is_priority_paper(self, post_file: Path) -> bool:
        """Check if this is a high-priority paper"""
        filename_lower = post_file.stem.lower()
        return any(priority in filename_lower for priority in self.priority_papers)
    
    def clean_post(self, post_file: Path, issues: List[Dict]) -> Dict:
        """Clean a single post based on identified issues"""
        logger.info(f"Cleaning {post_file.name}")
        
        try:
            with open(post_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            logger.error(f"Error reading {post_file}: {e}")
            return {'success': False, 'error': str(e)}
        
        content = original_content
        changes_made = []
        
        for issue in issues:
            if issue['type'] == 'excessive_results_tables':
                content, change_desc = self.limit_results_tables(content)
                if change_desc:
                    changes_made.append(change_desc)
            
            elif issue['type'] == 'poor_image_captions':
                content, change_desc = self.improve_image_captions(content)
                if change_desc:
                    changes_made.append(change_desc)
            
            elif issue['type'] == 'excessive_whitespace':
                content, change_desc = self.fix_excessive_whitespace(content)
                if change_desc:
                    changes_made.append(change_desc)
            
            elif issue['type'] == 'nested_tags_format':
                content, change_desc = self.fix_front_matter_tags(content)
                if change_desc:
                    changes_made.append(change_desc)
            
            elif issue['type'] == 'duplicate_sections':
                content, change_desc = self.remove_duplicate_sections(content)
                if change_desc:
                    changes_made.append(change_desc)
        
        # Write cleaned content
        if not self.dry_run and content != original_content:
            try:
                # Create backup
                backup_file = post_file.with_suffix('.md.backup')
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write cleaned content
                with open(post_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Cleaned {post_file.name} (backup saved)")
            except Exception as e:
                logger.error(f"Error writing cleaned content to {post_file}: {e}")
                return {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'changes_made': changes_made,
            'original_length': len(original_content),
            'new_length': len(content),
            'reduction': len(original_content) - len(content)
        }
    
    def limit_results_tables(self, content: str) -> Tuple[str, str]:
        """Limit results tables to most important ones"""
        images = self.patterns['image_markdown'].findall(content)
        results_tables = []
        
        for i, (alt_text, src) in enumerate(images):
            if 'results_table' in src:
                results_tables.append((i, alt_text, src))
        
        if len(results_tables) <= 5:
            return content, ""
        
        # Keep first 3 and last 2 results tables (most representative)
        to_keep = set(results_tables[:3] + results_tables[-2:])
        
        lines = content.split('\n')
        new_lines = []
        removed_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            img_match = self.patterns['image_markdown'].match(line.strip())
            
            if img_match and 'results_table' in img_match.group(2):
                # Check if this table should be kept
                should_keep = any(img_match.group(1) == alt and img_match.group(2) == src 
                                for _, alt, src in to_keep)
                
                if should_keep:
                    new_lines.append(line)
                else:
                    # Remove image and its caption
                    removed_count += 1
                    # Skip caption line too if it exists
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('*Figure:'):
                        i += 1
            else:
                new_lines.append(line)
            
            i += 1
        
        if removed_count > 0:
            return '\n'.join(new_lines), f"Removed {removed_count} excessive results tables"
        
        return content, ""
    
    def improve_image_captions(self, content: str) -> Tuple[str, str]:
        """Improve generic image captions"""
        lines = content.split('\n')
        new_lines = []
        improved_count = 0
        
        for i, line in enumerate(lines):
            # Check for image followed by generic caption
            if self.patterns['image_markdown'].match(line.strip()):
                new_lines.append(line)
                
                # Check next line for generic caption
                if (i + 1 < len(lines) and 
                    lines[i + 1].strip().startswith('*Figure:') and
                    re.match(r'\*Figure: (Method Diagram|Results Table|Architecture Overview) \d+', lines[i + 1])):
                    
                    # Improve the caption
                    original_caption = lines[i + 1].strip()
                    improved_caption = self.generate_better_caption(original_caption, line)
                    new_lines.append(improved_caption)
                    improved_count += 1
                    
                    # Skip the original caption line
                    continue
            
            new_lines.append(line)
        
        if improved_count > 0:
            return '\n'.join(new_lines), f"Improved {improved_count} image captions"
        
        return content, ""
    
    def generate_better_caption(self, original_caption: str, image_line: str) -> str:
        """Generate a better caption for an image"""
        # Extract image type from filename
        img_match = self.patterns['image_markdown'].match(image_line.strip())
        if not img_match:
            return original_caption
        
        src = img_match.group(2)
        
        if 'method_diagram' in src:
            return "*Figure: System architecture and methodology overview*"
        elif 'results_table' in src:
            return "*Figure: Experimental results and performance metrics*"
        elif 'architecture_overview' in src:
            return "*Figure: Model architecture and component design*"
        elif 'figure' in src:
            return "*Figure: Research findings and analysis*"
        else:
            return original_caption
    
    def fix_excessive_whitespace(self, content: str) -> Tuple[str, str]:
        """Fix excessive whitespace"""
        original_content = content
        # Replace 3+ consecutive newlines with just 2
        content = self.patterns['excessive_newlines'].sub('\n\n', content)
        
        if content != original_content:
            return content, "Fixed excessive whitespace"
        return content, ""
    
    def fix_front_matter_tags(self, content: str) -> Tuple[str, str]:
        """Fix nested tags format in front matter"""
        if not content.startswith('---'):
            return content, ""
        
        parts = content.split('---', 2)
        if len(parts) < 3:
            return content, ""
        
        front_matter = parts[1]
        post_content = parts[2]
        
        try:
            fm_data = yaml.safe_load(front_matter)
            
            # Fix nested tags
            if 'tags' in fm_data and isinstance(fm_data['tags'], list):
                if len(fm_data['tags']) > 0 and isinstance(fm_data['tags'][0], list):
                    # Flatten nested tags
                    flattened_tags = []
                    for tag_group in fm_data['tags']:
                        if isinstance(tag_group, list):
                            flattened_tags.extend(tag_group)
                        else:
                            flattened_tags.append(tag_group)
                    
                    fm_data['tags'] = flattened_tags
                    
                    # Reconstruct content
                    new_front_matter = yaml.dump(fm_data, default_flow_style=False)
                    new_content = f"---\n{new_front_matter}---{post_content}"
                    
                    return new_content, "Fixed nested tags format"
            
        except yaml.YAMLError:
            pass
        
        return content, ""
    
    def remove_duplicate_sections(self, content: str) -> Tuple[str, str]:
        """Remove duplicate sections"""
        sections = content.split('\n## ')
        if len(sections) <= 1:
            return content, ""
        
        seen_sections = set()
        unique_sections = [sections[0]]  # Keep the first part (before any ##)
        removed_count = 0
        
        for section in sections[1:]:
            section_title = section.split('\n')[0].strip()
            
            if section_title not in seen_sections:
                seen_sections.add(section_title)
                unique_sections.append(section)
            else:
                removed_count += 1
        
        if removed_count > 0:
            new_content = '\n## '.join(unique_sections)
            return new_content, f"Removed {removed_count} duplicate sections"
        
        return content, ""
    
    def generate_report(self, all_issues: Dict[str, List[Dict]]) -> str:
        """Generate comprehensive cleanup report"""
        report = ["# Paper Posts Cleanup Analysis Report", ""]
        report.append(f"Generated on: {os.popen('date').read().strip()}")
        report.append("")
        
        # Summary statistics
        total_issues = sum(len(issues) for issues in all_issues.values())
        total_files = len(set(item['file'] for issues in all_issues.values() for item in issues))
        
        report.extend([
            "## Executive Summary",
            "",
            f"- **Total files analyzed**: {len(list(self.posts_dir.glob('*.md')))}",
            f"- **Files with issues**: {total_files}",
            f"- **Total issues found**: {total_issues}",
            ""
        ])
        
        # Issue breakdown by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            count = len(all_issues[severity])
            if count > 0:
                report.append(f"- **{severity.title()} issues**: {count}")
        
        report.append("")
        
        # Detailed analysis by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            issues = all_issues[severity]
            if not issues:
                continue
            
            report.extend([
                f"## {severity.title()} Priority Issues",
                ""
            ])
            
            # Group by issue type
            issue_types = defaultdict(list)
            for item in issues:
                issue_types[item['issue']['type']].append(item)
            
            for issue_type, items in issue_types.items():
                report.append(f"### {issue_type.replace('_', ' ').title()}")
                report.append("")
                
                for item in items[:5]:  # Show first 5 examples
                    file_name = item['file'].name
                    priority_marker = " ⭐" if item['priority'] else ""
                    issue_desc = item['issue'].get('description', 'No description')
                    report.append(f"- **{file_name}**{priority_marker}: {issue_desc}")
                
                if len(items) > 5:
                    report.append(f"- ... and {len(items) - 5} more files")
                
                report.append("")
        
        # Priority papers section
        priority_issues = [item for issues in all_issues.values() for item in issues if item['priority']]
        if priority_issues:
            report.extend([
                "## High-Value Papers Requiring Attention",
                "",
                "These papers have high research value and should be cleaned first:",
                ""
            ])
            
            priority_files = defaultdict(list)
            for item in priority_issues:
                priority_files[item['file']].append(item['issue'])
            
            for file_path, file_issues in priority_files.items():
                report.append(f"### {file_path.name}")
                for issue in file_issues[:3]:  # Top 3 issues per file
                    report.append(f"- {issue.get('description', issue.get('type', 'Unknown issue'))}")
                if len(file_issues) > 3:
                    report.append(f"- ... and {len(file_issues) - 3} more issues")
                report.append("")
        
        # Recommendations
        report.extend([
            "## Cleanup Recommendations",
            "",
            "### Immediate Actions (Critical/High Priority)",
            "",
            "1. **Fix critical front matter issues** - These break page rendering",
            "2. **Reduce excessive results tables** - Limit to 3-5 most important per post",
            "3. **Improve image captions** - Replace generic captions with descriptive ones",
            "4. **Remove duplicate sections** - Clean up redundant content",
            "",
            "### Medium Priority Actions", 
            "",
            "1. **Consolidate method diagrams** - Keep only essential architectural diagrams",
            "2. **Fix text flow disruptions** - Reorganize images for better readability",
            "3. **Clean up front matter formatting** - Standardize tags and metadata",
            "",
            "### Low Priority Actions",
            "",
            "1. **Fix excessive whitespace** - Improve visual formatting",
            "2. **Add alt text to images** - Improve accessibility",
            ""
        ])
        
        return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='Clean up paper posts')
    parser.add_argument('posts_dir', help='Directory containing paper posts')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Perform analysis only, do not modify files')
    parser.add_argument('--clean', action='store_true', 
                       help='Actually perform cleanup (removes dry-run mode)')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only, skip analysis')
    parser.add_argument('--priority-only', action='store_true',
                       help='Only process high-priority papers')
    
    args = parser.parse_args()
    
    if args.clean:
        args.dry_run = False
    
    cleaner = PaperPostCleaner(args.posts_dir, args.dry_run)
    
    if not args.report_only:
        # Analyze all posts
        all_issues = cleaner.analyze_all_posts()
        
        # Generate and save report
        report = cleaner.generate_report(all_issues)
        report_file = Path(args.posts_dir).parent / 'paper_cleanup_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Analysis complete. Report saved to {report_file}")
        
        # Perform cleanup if requested
        if not args.dry_run:
            logger.info("Starting cleanup process...")
            
            # Process by priority
            for severity in ['critical', 'high', 'medium', 'low']:
                if args.priority_only and severity not in ['critical', 'high']:
                    continue
                    
                for item in all_issues[severity]:
                    if args.priority_only and not item['priority']:
                        continue
                        
                    result = cleaner.clean_post(item['file'], [item['issue']])
                    if result['success']:
                        logger.info(f"Cleaned {item['file'].name}: {result.get('changes_made', [])}")
                    else:
                        logger.error(f"Failed to clean {item['file'].name}: {result.get('error')}")
        
        # Print summary
        total_issues = sum(len(issues) for issues in all_issues.values())
        print(f"\n📊 Analysis Summary:")
        print(f"   Total issues found: {total_issues}")
        print(f"   Critical: {len(all_issues['critical'])}")
        print(f"   High: {len(all_issues['high'])}")  
        print(f"   Medium: {len(all_issues['medium'])}")
        print(f"   Low: {len(all_issues['low'])}")
        print(f"\n📄 Full report: {report_file}")
        
        if args.dry_run:
            print(f"\n🔍 This was a dry run. Use --clean to actually modify files.")

if __name__ == "__main__":
    main()