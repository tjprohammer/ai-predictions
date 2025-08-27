#!/usr/bin/env python3
"""
Quick schema check for umpire columns
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    db_url = os.getenv('DATABASE_URL')
    # Convert SQLAlchemy format to psycopg2 format
    if 'postgresql+psycopg2://' in db_url:
        db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    print('🔍 Checking enhanced_games table schema for umpire columns...')
    cur.execute("""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        AND column_name LIKE '%umpire%'
        ORDER BY column_name;
    """)

    columns = cur.fetchall()
    if columns:
        print('✅ Found umpire columns:')
        for col_name, data_type, is_nullable in columns:
            print(f'   {col_name}: {data_type} (nullable: {is_nullable})')
    else:
        print('⚠️ No umpire columns found in enhanced_games table')
        
        # Check if we need to add them
        print('\n📋 Sample of existing columns:')
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games' 
            ORDER BY ordinal_position;
        """)
        
        all_columns = cur.fetchall()
        for col_name, data_type in all_columns[:15]:  # First 15 columns
            print(f'   {col_name}: {data_type}')
        
        if len(all_columns) > 15:
            print(f'   ... and {len(all_columns) - 15} more columns')
    
    conn.close()
    
except Exception as e:
    print(f"❌ Database check failed: {e}")
