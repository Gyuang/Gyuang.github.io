#!/usr/bin/env python3
"""
Zotero Debug Tool - Check what's in the mrichat collection
"""

import sqlite3
from pathlib import Path

def debug_mrichat_collection():
    """Debug the mrichat collection to see what items it contains."""
    
    db_path = Path.home() / "Zotero" / "zotero.sqlite"
    
    if not db_path.exists():
        print("❌ Zotero database not found")
        return
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Find MRIChat collection
        cursor.execute("SELECT collectionID, collectionName FROM collections WHERE collectionName LIKE '%mrichat%' OR collectionName LIKE '%MRIChat%'")
        collections = cursor.fetchall()
        
        if not collections:
            print("❌ No MRIChat collection found")
            return
        
        collection_id, collection_name = collections[0]
        print(f"✅ Found collection: {collection_name} (ID: {collection_id})")
        
        # Check all items in the collection
        cursor.execute("""
            SELECT COUNT(*) FROM collectionItems WHERE collectionID = ?
        """, (collection_id,))
        
        total_items = cursor.fetchone()[0]
        print(f"📚 Total items in collection: {total_items}")
        
        if total_items == 0:
            print("❌ Collection is empty")
            return
        
        # Check item types
        cursor.execute("""
            SELECT it.typeName, COUNT(*) as count
            FROM collectionItems ci
            JOIN items i ON ci.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE ci.collectionID = ?
            GROUP BY it.typeName
            ORDER BY count DESC
        """, (collection_id,))
        
        item_types = cursor.fetchall()
        print(f"\n📊 Item types in collection:")
        for item_type, count in item_types:
            print(f"  - {item_type}: {count}")
        
        # Get some sample items with titles
        cursor.execute("""
            SELECT DISTINCT
                i.itemID,
                it.typeName,
                COALESCE(idv.value, 'No Title') as title
            FROM collectionItems ci
            JOIN items i ON ci.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemData id ON i.itemID = id.itemID AND id.fieldID = 1  -- title field
            LEFT JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE ci.collectionID = ?
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT 10
        """, (collection_id,))
        
        items = cursor.fetchall()
        print(f"\n📋 Sample items:")
        for item_id, item_type, title in items:
            print(f"  - [{item_type}] {title}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    debug_mrichat_collection()