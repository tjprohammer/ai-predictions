#!/usr/bin/env python3
"""
Create pregame features materialized view
"""
import sys
from sqlalchemy import create_engine, text

def create_pregame_view():
    """Create the pregame features materialized view"""
    
    # Database connection
    db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
    engine = create_engine(db_url)
    
    # Read SQL file
    with open('create_pregame_features_view.sql', 'r') as f:
        sql_content = f.read()
    
    # Execute SQL
    try:
        with engine.begin() as conn:
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            for i, stmt in enumerate(statements):
                print(f"Executing statement {i+1}/{len(statements)}...")
                if stmt:
                    conn.execute(text(stmt))
                    
        print("✅ Pregame features view created successfully!")
        
        # Check row count
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM pregame_features_v1")).scalar()
            print(f"📊 View contains {result:,} games with pregame features")
            
    except Exception as e:
        print(f"❌ Error creating view: {e}")
        sys.exit(1)

if __name__ == '__main__':
    create_pregame_view()
