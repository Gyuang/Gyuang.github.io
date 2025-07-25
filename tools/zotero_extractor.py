#!/usr/bin/env python3
"""
Zotero Library Metadata Extractor
Extracts paper metadata from Zotero library, specifically from mrichat collection.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import platform

def find_zotero_database():
    """Find the Zotero SQLite database location based on the operating system."""
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        potential_paths = [
            Path.home() / "Zotero" / "zotero.sqlite",
            Path.home() / "Library" / "Application Support" / "Zotero" / "Profiles" / "**" / "zotero.sqlite",
            Path.home() / "Documents" / "Zotero" / "zotero.sqlite"
        ]
    elif system == "Windows":
        potential_paths = [
            Path.home() / "Zotero" / "zotero.sqlite",
            Path(os.environ.get('APPDATA', '')) / "Zotero" / "Zotero" / "Profiles" / "**" / "zotero.sqlite"
        ]
    elif system == "Linux":
        potential_paths = [
            Path.home() / "Zotero" / "zotero.sqlite",
            Path.home() / ".zotero" / "zotero" / "**" / "zotero.sqlite",
            Path.home() / "snap" / "zotero-snap" / "common" / "Zotero" / "**" / "zotero.sqlite"
        ]
    else:
        print(f"❌ Unsupported operating system: {system}")
        return None
    
    # Check each potential path
    for path_pattern in potential_paths:
        if "**" in str(path_pattern):
            # Handle glob patterns
            parent = path_pattern.parent.parent if "**" in str(path_pattern.parent) else path_pattern.parent
            if parent.exists():
                for db_file in parent.rglob("zotero.sqlite"):
                    if db_file.exists():
                        return db_file
        else:
            if path_pattern.exists():
                return path_pattern
    
    return None

def get_zotero_collections(db_path):
    """Get all collections from Zotero database."""
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Query to get all collections
        query = """
        SELECT collectionID, collectionName, parentCollectionID
        FROM collections
        ORDER BY collectionName
        """
        
        cursor.execute(query)
        collections = cursor.fetchall()
        
        conn.close()
        return collections
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return []

def find_mrichat_collection(db_path):
    """Find the mrichat collection ID."""
    
    collections = get_zotero_collections(db_path)
    
    # Look for mrichat collection (case insensitive)
    for collection_id, name, parent_id in collections:
        if 'mrichat' in name.lower():
            return collection_id, name
    
    # If not found, show available collections
    print("❌ 'mrichat' collection not found.")
    print("\n📚 Available collections:")
    for collection_id, name, parent_id in collections:
        indent = "  " if parent_id else ""
        print(f"{indent}- {name} (ID: {collection_id})")
    
    return None, None

def extract_mrichat_papers(db_path, collection_id):
    """Extract paper metadata from mrichat collection."""
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Query to get papers from the specific collection
        query = """
        SELECT DISTINCT
            i.itemID,
            i.itemTypeID,
            COALESCE(f1.value, '') as title,
            COALESCE(f2.value, '') as abstract,
            COALESCE(f3.value, '') as date,
            COALESCE(f4.value, '') as doi,
            COALESCE(f5.value, '') as url,
            COALESCE(f6.value, '') as journal,
            COALESCE(f7.value, '') as volume,
            COALESCE(f8.value, '') as issue,
            COALESCE(f9.value, '') as pages
        FROM items i
        JOIN collectionItems ci ON i.itemID = ci.itemID
        LEFT JOIN itemData id1 ON i.itemID = id1.itemID AND id1.fieldID = 1  -- title
        LEFT JOIN itemDataValues f1 ON id1.valueID = f1.valueID
        LEFT JOIN itemData id2 ON i.itemID = id2.itemID AND id2.fieldID = 2  -- abstractNote
        LEFT JOIN itemDataValues f2 ON id2.valueID = f2.valueID
        LEFT JOIN itemData id3 ON i.itemID = id3.itemID AND id3.fieldID = 6  -- date
        LEFT JOIN itemDataValues f3 ON id3.valueID = f3.valueID
        LEFT JOIN itemData id4 ON i.itemID = id4.itemID AND id4.fieldID = 59 -- DOI
        LEFT JOIN itemDataValues f4 ON id4.valueID = f4.valueID
        LEFT JOIN itemData id5 ON i.itemID = id5.itemID AND id5.fieldID = 13 -- URL
        LEFT JOIN itemDataValues f5 ON id5.valueID = f5.valueID
        LEFT JOIN itemData id6 ON i.itemID = id6.itemID AND id6.fieldID = 38 -- publicationTitle
        LEFT JOIN itemDataValues f6 ON id6.valueID = f6.valueID
        LEFT JOIN itemData id7 ON i.itemID = id7.itemID AND id7.fieldID = 19 -- volume
        LEFT JOIN itemDataValues f7 ON id7.valueID = f7.valueID
        LEFT JOIN itemData id8 ON i.itemID = id8.itemID AND id8.fieldID = 76 -- issue
        LEFT JOIN itemDataValues f8 ON id8.valueID = f8.valueID
        LEFT JOIN itemData id9 ON i.itemID = id9.itemID AND id9.fieldID = 32 -- pages
        LEFT JOIN itemDataValues f9 ON id9.valueID = f9.valueID
        WHERE ci.collectionID = ? 
        AND i.itemTypeID IN (22, 31, 11, 40, 14)  -- journalArticle, preprint, conferencePaper, webpage, document
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        ORDER BY f1.value
        """
        
        cursor.execute(query, (collection_id,))
        papers = cursor.fetchall()
        
        # Get authors for each paper
        papers_with_authors = []
        for paper in papers:
            item_id = paper[0]
            
            # Get authors
            author_query = """
            SELECT COALESCE(c.firstName, '') || ' ' || COALESCE(c.lastName, '') as author_name
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            WHERE ic.itemID = ?
            ORDER BY ic.orderIndex
            """
            
            cursor.execute(author_query, (item_id,))
            authors = [row[0].strip() for row in cursor.fetchall() if row[0].strip()]
            
            # Create paper dictionary
            paper_dict = {
                'item_id': paper[0],
                'title': paper[2],
                'abstract': paper[3],
                'date': paper[4],
                'doi': paper[5],
                'url': paper[6],
                'journal': paper[7],
                'volume': paper[8],
                'issue': paper[9],
                'pages': paper[10],
                'authors': authors,
                'author_string': ', '.join(authors) if authors else ''
            }
            
            papers_with_authors.append(paper_dict)
        
        conn.close()
        return papers_with_authors
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return []

def save_metadata_json(papers, output_file):
    """Save extracted metadata to JSON file."""
    
    metadata = {
        'extraction_date': datetime.now().isoformat(),
        'total_papers': len(papers),
        'papers': papers
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return output_file

def create_vlm_config_from_metadata(papers, output_file):
    """Create VLM papers config file from extracted metadata."""
    
    vlm_papers = []
    
    for paper in papers:
        # Check if this looks like a VLM paper based on title/abstract
        title_lower = paper['title'].lower()
        abstract_lower = paper['abstract'].lower()
        
        vlm_keywords = [
            'vision', 'language', 'multimodal', 'clip', 'blip', 'flamingo',
            'visual', 'text', 'image', 'caption', 'vlm', 'vqa', 'visual question'
        ]
        
        is_vlm = any(keyword in title_lower or keyword in abstract_lower for keyword in vlm_keywords)
        
        if is_vlm:
            vlm_paper = {
                'title': paper['title'],
                'excerpt': f"{paper['title']} 논문 요약",
                'additional_tags': f"Vision-Language, {paper['journal']}" if paper['journal'] else "Vision-Language",
                'introduction': paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract'],
                'related_work_vlm': "기존 Vision-Language 모델들과의 비교 연구가 필요합니다.",
                'specific_domain': "Computer Vision",
                'related_work_domain': "컴퓨터 비전 분야의 관련 연구들을 분석합니다.",
                'architecture_description': "논문에서 제안하는 아키텍처에 대한 설명이 필요합니다.",
                'architecture_image': f"{paper['title'].replace(' ', '_').replace(':', '').lower()}_architecture.png",
                'key_components': "주요 구성 요소들에 대한 설명이 필요합니다.",
                'training_strategy': "훈련 전략에 대한 설명이 필요합니다.",
                'datasets': "사용된 데이터셋에 대한 정보가 필요합니다.",
                'results': "실험 결과에 대한 설명이 필요합니다.",
                'ablation_studies': "Ablation study 결과에 대한 설명이 필요합니다.",
                'conclusion': "논문의 결론 및 기여도에 대한 설명이 필요합니다.",
                'key_takeaways': "주요 시사점들을 정리해주세요.",
                'metadata': {
                    'doi': paper['doi'],
                    'journal': paper['journal'],
                    'authors': paper['author_string'],
                    'date': paper['date'],
                    'url': paper['url']
                }
            }
            vlm_papers.append(vlm_paper)
    
    config = {
        'extraction_info': {
            'source': 'Zotero mrichat collection',
            'extraction_date': datetime.now().isoformat(),
            'total_papers_in_collection': len(papers),
            'vlm_papers_found': len(vlm_papers)
        },
        'papers': vlm_papers
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return output_file, len(vlm_papers)

def main():
    """Main function to extract Zotero metadata."""
    
    print("=== Zotero mrichat Collection Metadata Extractor ===\n")
    
    # Find Zotero database
    print("🔍 Searching for Zotero database...")
    db_path = find_zotero_database()
    
    if not db_path:
        print("❌ Zotero database not found!")
        print("\n💡 Make sure Zotero is installed and has been run at least once.")
        print("💡 If Zotero is in a custom location, please specify the path to zotero.sqlite")
        sys.exit(1)
    
    print(f"✅ Found Zotero database: {db_path}")
    
    # Find mrichat collection
    print("\n🔍 Looking for 'mrichat' collection...")
    collection_id, collection_name = find_mrichat_collection(db_path)
    
    if not collection_id:
        sys.exit(1)
    
    print(f"✅ Found collection: '{collection_name}' (ID: {collection_id})")
    
    # Extract papers
    print(f"\n📚 Extracting papers from '{collection_name}' collection...")
    papers = extract_mrichat_papers(db_path, collection_id)
    
    if not papers:
        print("❌ No papers found in the collection.")
        sys.exit(1)
    
    print(f"✅ Found {len(papers)} papers")
    
    # Create output directory
    output_dir = Path("tools/zotero_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save full metadata
    metadata_file = output_dir / "mrichat_metadata.json"
    save_metadata_json(papers, metadata_file)
    print(f"📄 Full metadata saved: {metadata_file}")
    
    # Create VLM config
    vlm_config_file = Path("vlm_papers_from_zotero.json")
    vlm_config_path, vlm_count = create_vlm_config_from_metadata(papers, vlm_config_file)
    print(f"🤖 VLM papers config created: {vlm_config_path}")
    print(f"   - Found {vlm_count} potential VLM papers")
    
    # Show sample papers
    print(f"\n📋 Sample papers from collection:")
    for i, paper in enumerate(papers[:5], 1):
        print(f"{i}. {paper['title']}")
        if paper['authors']:
            print(f"   Authors: {paper['author_string']}")
        if paper['journal']:
            print(f"   Journal: {paper['journal']}")
        print()
    
    if len(papers) > 5:
        print(f"... and {len(papers) - 5} more papers")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Review the VLM papers config: {vlm_config_path}")
    print(f"2. Edit the config file to add detailed content for each paper")
    print(f"3. Run: python3 tools/batch_vlm_generator.py --config {vlm_config_path}")

if __name__ == "__main__":
    main()