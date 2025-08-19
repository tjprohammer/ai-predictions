import pandas as pd
from sqlalchemy import create_engine, text
import os
import requests

eng = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Find a game on 2025-08-12 and temporarily assign pitcher 701542
games = pd.read_sql(text("""
    SELECT game_id, home_team, away_team, home_sp_id, away_sp_id 
    FROM games 
    WHERE date = '2025-08-12' 
    LIMIT 1
"""), eng)

if not games.empty:
    game_id = games.iloc[0]['game_id']
    original_home_sp = games.iloc[0]['home_sp_id']
    print(f'Temporarily updating game {game_id} to use pitcher 701542 as home starter...')
    
    with eng.begin() as cx:
        cx.execute(text('UPDATE games SET home_sp_id = 701542 WHERE game_id = :gid'), 
                   {'gid': game_id})
    
    print('Testing API now...')
    
    try:
        response = requests.get('http://127.0.0.1:8000/predict?date=2025-08-12')
        if response.status_code == 200:
            data = response.json()
            for game in data['predictions']:
                if game['game_id'] == game_id:
                    home_pitcher = game['pitchers']['home']
                    print('SUCCESS! Found ERA data:')
                    print('Home pitcher:', home_pitcher['name'])
                    print('ERA Season:', home_pitcher.get('era_season'))
                    print('ERA L3:', home_pitcher.get('era_l3'))
                    print('ERA L5:', home_pitcher.get('era_l5'))
                    print('ERA L10:', home_pitcher.get('era_l10'))
                    break
        else:
            print('API Error:', response.text)
    except Exception as e:
        print('Error testing API:', e)
    
    # Restore original pitcher
    with eng.begin() as cx:
        cx.execute(text('UPDATE games SET home_sp_id = :original WHERE game_id = :gid'), 
                   {'original': original_home_sp, 'gid': game_id})
    print('\nRestored original pitcher assignment.')
else:
    print('No games found for 2025-08-12')
