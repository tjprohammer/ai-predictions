from sqlalchemy import create_engine
import pandas as pd
from datetime import date, timedelta

engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

with engine.begin() as conn:
    # Check games around today's date
    print('GAMES BY DATE (around Aug 13, 2025):')
    for days_offset in range(-2, 3):
        check_date = date(2025, 8, 13) + timedelta(days=days_offset)
        count = pd.read_sql(f"SELECT COUNT(*) as total FROM enhanced_games WHERE date = '{check_date}'", conn)
        total = count.iloc[0]['total']
        if total > 0:
            print(f'  {check_date}: {total} games')
    
    print()
    print('ALL GAMES FOR 2025-08-13:')
    games = pd.read_sql("SELECT game_id, home_team, away_team, home_score, away_score, total_runs FROM enhanced_games WHERE date = '2025-08-13' ORDER BY game_id", conn)
    print(games.to_string())
    
    print()
    print('TOTAL GAMES IN DATABASE:')
    total = pd.read_sql("SELECT COUNT(*) as total FROM enhanced_games", conn)
    print(f"Total games in database: {total.iloc[0]['total']}")
