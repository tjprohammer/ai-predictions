#!/usr/bin/env python3
"""
Database Table Checker
======================
Check what tables exist in the PostgreSQL database
"""

import os
import psycopg2
from sqlalchemy import create_engine, text

def check_database_tables():
    db_url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    try:
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Get all tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = result.fetchall()
            
            print("📊 Available tables in PostgreSQL database:")
            print(f"🔗 Connected to: {db_url.split('@')[1] if '@' in db_url else 'database'}")
            print()
            
            for table in tables:
                table_name = table[0]
                
                # Get row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = count_result.fetchone()[0]
                
                # Get column info
                columns_result = conn.execute(text(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    ORDER BY ordinal_position
                """))
                columns = columns_result.fetchall()
                
                print(f"   📋 {table_name}: {count:,} rows, {len(columns)} columns")
                
                # Show first few column names with types
                col_info = [f"{col[0]}({col[1]})" for col in columns[:5]]
                print(f"      Columns: {', '.join(col_info)}{'...' if len(columns) > 5 else ''}")
                print()
        
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL database: {e}")
        print(f"   Make sure PostgreSQL is running and accessible at: {db_url}")

if __name__ == "__main__":
    check_database_tables()
