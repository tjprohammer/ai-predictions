import os
from sqlalchemy import create_engine, text

# Connect to database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

print('Checking corrected probability_predictions data...')
with engine.connect() as conn:
    # Check a few rows to verify correct alignment
    query = text("""
        SELECT game_id, game_date, p_over, p_under, model_version, created_at
        FROM probability_predictions
        WHERE game_date = DATE '2025-08-16'
        ORDER BY game_id
        LIMIT 5
    """)
    
    result = conn.execute(query)
    print('\nFirst 5 rows:')
    for row in result.fetchall():
        print(f"  {row[0]} | {row[1]} | p_over={row[2]:.3f} | p_under={row[3]:.3f} | {row[4]} | {row[5]}")

    # Check the API view with confidence
    api_query = text("""
        SELECT game_id, home_team, away_team, confidence, over_probability, under_probability
        FROM api_games_today
        ORDER BY game_id
        LIMIT 5
    """)
    
    result = conn.execute(api_query)
    print('\nAPI view with confidence:')
    for row in result.fetchall():
        print(f"  {row[1]} vs {row[2]} | confidence={row[3]}% | p_over={row[4]:.3f} | p_under={row[5]:.3f}")

engine.dispose()
