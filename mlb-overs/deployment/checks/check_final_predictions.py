import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Get today's predictions
query = '''
SELECT 
    home_team || ' @ ' || away_team as matchup,
    market_total,
    predicted_total_learning as learning_pred,
    predicted_total as enhanced_pred,
    confidence,
    recommendation,
    edge
FROM enhanced_games 
WHERE date = '2025-08-23' 
AND predicted_total_learning IS NOT NULL
ORDER BY game_id
'''

df = pd.read_sql(query, engine)
print('Final Predictions Comparison:')
print('=' * 80)
for _, row in df.iterrows():
    matchup = row['matchup']
    market = row['market_total']
    learning = row['learning_pred']
    enhanced = row['enhanced_pred'] or 'N/A'
    rec = row['recommendation']
    conf = row['confidence']
    print(f'{matchup:<35} Market: {market:>4.1f} | Learning: {learning:>5.2f} | Enhanced: {enhanced:<5} | {rec:>4} ({conf:>2}%)')

print(f'\nSummary:')
print(f'Learning Model - Mean: {df["learning_pred"].mean():.2f}, Std: {df["learning_pred"].std():.2f}, Range: {df["learning_pred"].min():.2f}-{df["learning_pred"].max():.2f}')
if not df['enhanced_pred'].isna().all():
    enhanced_data = df['enhanced_pred'].dropna()
    print(f'Enhanced Model - Mean: {enhanced_data.mean():.2f}, Std: {enhanced_data.std():.2f}, Range: {enhanced_data.min():.2f}-{enhanced_data.max():.2f}')
else:
    print('Enhanced Model - No successful predictions')
