#!/usr/bin/env python3
"""
Check Zotero database schema to find correct field IDs
"""

import sqlite3
from pathlib import Path

def check_zotero_schema():
    """Check the Zotero database schema."""
    
    db_path = Path.home() / "Zotero" / "zotero.sqlite"
    
    if not db_path.exists():
        print("❌ Zotero database not found")
        return
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Get field names and IDs
        print("📊 Zotero Field Schema:")
        cursor.execute("SELECT fieldID, fieldName FROM fields ORDER BY fieldID")
        fields = cursor.fetchall()
        
        field_map = {}
        for field_id, field_name in fields:
            field_map[field_name] = field_id
            print(f"  {field_id}: {field_name}")
        
        print(f"\n🔍 Key fields we need:")
        key_fields = ['title', 'abstractNote', 'date', 'DOI', 'url', 'publicationTitle', 'volume', 'issue', 'pages']
        for field in key_fields:
            if field in field_map:
                print(f"  {field}: {field_map[field]}")
            else:
                print(f"  {field}: NOT FOUND")
        
        # Check item types
        print(f"\n📊 Item Types:")
        cursor.execute("SELECT itemTypeID, typeName FROM itemTypes ORDER BY itemTypeID")
        item_types = cursor.fetchall()
        
        for type_id, type_name in item_types:
            print(f"  {type_id}: {type_name}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_zotero_schema()