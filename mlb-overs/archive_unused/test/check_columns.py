from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

with engine.begin() as conn:
    # Get all columns
    columns = pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY ordinal_position", conn)
    
    print(f"Total columns available: {len(columns)}")
    print("\nAll columns:")
    for i, col in enumerate(columns['column_name'].tolist(), 1):
        print(f"{i:2d}. {col}")
    
    # Check sample data with more columns
    sample = pd.read_sql("SELECT * FROM enhanced_games WHERE date = '2025-08-13' LIMIT 1", conn)
    print(f"\nSample data has {len(sample.columns)} columns")
