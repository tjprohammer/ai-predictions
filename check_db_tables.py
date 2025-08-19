#!/usr/bin/env python3
"""
Database Table Checker
======================
Check what tables exist in the database
"""

import sqlite3
from pathlib import Path

def check_database_tables():
    db_path = "S:/Projects/AI_Predictions/mlb-overs/data/mlb_data.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("📊 Available tables in database:")
        for table in tables:
            table_name = table[0]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get some column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print(f"   📋 {table_name}: {count:,} rows, {len(columns)} columns")
            
            # Show first few column names
            col_names = [col[1] for col in columns[:5]]
            print(f"      Columns: {', '.join(col_names)}{'...' if len(columns) > 5 else ''}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database_tables()
