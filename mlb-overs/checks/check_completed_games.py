import os
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check yesterday's completed games
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')

print(f"Checking completed games for {yesterday} and {today}")
print("=" * 60)

# Check for completed games (games with total_runs filled)
query = '''
SELECT 
    date,
    COUNT(*) as total_games,
    COUNT(total_runs) as completed_games,
    COUNT(CASE WHEN total_runs IS NULL THEN 1 END) as pending_games,
    MIN(total_runs) as min_runs,
    MAX(total_runs) as max_runs,
    AVG(total_runs) as avg_runs
FROM enhanced_games 
WHERE date IN (:yesterday, :today)
GROUP BY date
ORDER BY date
'''

with engine.connect() as conn:
    result = conn.execute(text(query), {'yesterday': yesterday, 'today': today})
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

print("Games Summary:")
print(df.to_string(index=False))

# Check specific games from yesterday with predictions vs actual
query2 = '''
SELECT 
    home_team,
    away_team,
    market_total,
    predicted_total,
    predicted_total_learning,
    total_runs,
    CASE 
        WHEN total_runs IS NOT NULL AND predicted_total IS NOT NULL 
        THEN ABS(predicted_total - total_runs)
        ELSE NULL 
    END as enhanced_error,
    CASE 
        WHEN total_runs IS NOT NULL AND predicted_total_learning IS NOT NULL 
        THEN ABS(predicted_total_learning - total_runs)
        ELSE NULL 
    END as learning_error
FROM enhanced_games 
WHERE date = :yesterday
ORDER BY game_id
'''

with engine.connect() as conn:
    result2 = conn.execute(text(query2), {'yesterday': yesterday})
    df2 = pd.DataFrame(result2.fetchall(), columns=result2.keys())

print(f"\n\nYesterday's Games ({yesterday}):")
print("=" * 60)
print(df2.to_string(index=False))

if len(df2) > 0:
    completed = df2['total_runs'].notna().sum()
    with_enhanced = df2['predicted_total'].notna().sum()
    with_learning = df2['predicted_total_learning'].notna().sum()
    
    print(f"\nYesterday's Summary:")
    print(f"  Total games: {len(df2)}")
    print(f"  Completed games: {completed}")
    print(f"  With enhanced predictions: {with_enhanced}")
    print(f"  With learning predictions: {with_learning}")
    
    if completed > 0:
        avg_actual = df2['total_runs'].mean()
        print(f"  Average actual runs: {avg_actual:.2f}")
        
        if with_enhanced > 0:
            enhanced_mae = df2['enhanced_error'].mean()
            print(f"  Enhanced MAE: {enhanced_mae:.2f}")
            
        if with_learning > 0:
            learning_mae = df2['learning_error'].mean()
            print(f"  Learning MAE: {learning_mae:.2f}")
else:
    print("No games found for yesterday - need to collect completed games!")
