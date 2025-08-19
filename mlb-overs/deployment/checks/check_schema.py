import os
from sqlalchemy import create_engine, text

# Connect to database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

print('Checking probability_predictions table schema...')
with engine.connect() as conn:
    # Check the schema with proper order
    schema_query = text("""
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_name = 'probability_predictions'
        ORDER BY ordinal_position
    """)
    result = conn.execute(schema_query)
    for row in result.fetchall():
        print(f'{row[2]:2d}. {row[0]:20s} {row[1]}')

engine.dispose()
