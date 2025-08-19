#!/usr/bin/env python3
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.begin() as conn:
    # Get all games for today
    games = pd.read_sql(text("""
        SELECT game_id, home_team, away_team, predicted_total, market_total, 
               over_odds, under_odds, total_runs
        FROM enhanced_games 
        WHERE "date" = '2025-08-17'
        ORDER BY game_id
    """), conn)
    
    print(f'📅 Games available for 2025-08-17: {len(games)}')
    print('\n🏟️  All Games Today:')
    print('   Game ID | Teams                    | Pred | Market | Odds        | Actual')
    print('   --------|--------------------------|------|--------|-------------|-------')
    
    for _, game in games.iterrows():
        away = game['away_team'][:3].upper() if pd.notna(game['away_team']) else '???'
        home = game['home_team'][:3].upper() if pd.notna(game['home_team']) else '???'
        pred = f'{game["predicted_total"]:.1f}' if pd.notna(game['predicted_total']) else 'N/A'
        market = f'{game["market_total"]:.1f}' if pd.notna(game['market_total']) else 'N/A'
        over_odds = f'{int(game["over_odds"]):+d}' if pd.notna(game['over_odds']) else 'N/A'
        under_odds = f'{int(game["under_odds"]):+d}' if pd.notna(game['under_odds']) else 'N/A'
        actual = f'{game["total_runs"]:.0f}' if pd.notna(game['total_runs']) else 'TBD'
        
        print(f'   {game["game_id"]:>7} | {away}@{home:<17} | {pred:>4} | {market:>6} | {over_odds:>4}/{under_odds:<4} | {actual:>5}')

    # Show games with predictions vs those without
    has_pred = games['predicted_total'].notna().sum()
    has_market = games['market_total'].notna().sum() 
    has_odds = games['over_odds'].notna().sum()
    completed = games['total_runs'].notna().sum()
    
    print(f'\n📊 Data Availability:')
    print(f'   Games with predictions: {has_pred}/{len(games)}')
    print(f'   Games with market totals: {has_market}/{len(games)}')
    print(f'   Games with odds: {has_odds}/{len(games)}')
    print(f'   Games completed: {completed}/{len(games)}')
