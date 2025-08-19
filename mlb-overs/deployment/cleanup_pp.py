import os
from sqlalchemy import create_engine, text

# Connect to database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

print('Cleaning up misaligned probability_predictions data...')
with engine.connect() as conn:
    # Delete today's bad rows
    delete_result = conn.execute(text("DELETE FROM probability_predictions WHERE game_date = DATE '2025-08-16'"))
    print(f'Deleted {delete_result.rowcount} misaligned rows for 2025-08-16')
    
    conn.commit()

engine.dispose()
