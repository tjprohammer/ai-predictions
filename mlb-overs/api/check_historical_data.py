import sys
sys.path.append('.')
from sqlalchemy import create_engine, text
import pandas as pd

# Check completed games from last 30 days (without dual predictions)
engine = create_engine('postgresql://mlbuser:mlbpass@localhost/mlb')

query = text('''
    SELECT 
        date,
        COUNT(*) as total_games,
        COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed_games,
        COUNT(CASE WHEN predicted_total IS NOT NULL THEN 1 END) as with_original_predictions
    FROM enhanced_games 
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    AND date < CURRENT_DATE
    GROUP BY date
    ORDER BY date DESC
    LIMIT 10
''')

with engine.connect() as conn:
    df = pd.read_sql(query, conn)

print('Historical games available for backtesting:')
print(df.to_string(index=False))

# Get total counts
total_query = text('''
    SELECT 
        COUNT(*) as total_historical_games,
        COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed_historical_games,
        COUNT(CASE WHEN predicted_total IS NOT NULL THEN 1 END) as with_predictions
    FROM enhanced_games 
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    AND date < CURRENT_DATE
''')

with engine.connect() as conn:
    totals = pd.read_sql(total_query, conn).iloc[0]

print('\nSummary:')
print('Total historical games:', totals["total_historical_games"])
print('Completed historical games:', totals["completed_historical_games"])
print('With original predictions:', totals["with_predictions"])
