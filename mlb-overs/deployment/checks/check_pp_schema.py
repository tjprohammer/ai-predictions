import os
from sqlalchemy import create_engine, text

# Connect to database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

print('Checking probability_predictions table schema...')
with engine.connect() as conn:
    # Check the schema
    schema_query = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'probability_predictions'")
    result = conn.execute(schema_query)
    columns = [row[0] for row in result.fetchall()]
    print(f'probability_predictions columns: {columns}')
    
    # Check a sample row
    if columns:
        sample_query = text("SELECT * FROM probability_predictions LIMIT 1")
        sample_result = conn.execute(sample_query)
        sample_row = sample_result.fetchone()
        if sample_row:
            print(f'Sample row: {dict(zip(columns, sample_row))}')

engine.dispose()
