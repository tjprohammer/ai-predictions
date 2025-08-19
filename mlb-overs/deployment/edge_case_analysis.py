#!/usr/bin/env python3
"""
Edge Case Analysis - Deep dive into filtered games
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.begin() as conn:
    # Get all games with team names
    games = pd.read_sql(text("""
        SELECT game_id, home_team, away_team, predicted_total, market_total, 
               over_odds, under_odds, total_runs
        FROM enhanced_games 
        WHERE "date" = '2025-08-17'
        ORDER BY game_id
    """), conn)

print("🔍 EDGE CASE ANALYSIS\n")

print("📊 Market vs Prediction Spreads:")
games['edge'] = games['predicted_total'] - games['market_total']
games = games.sort_values('edge', ascending=False)

for _, game in games.iterrows():
    away = game['away_team'][:3].upper()
    home = game['home_team'][:3].upper()
    edge = game['edge']
    
    if abs(edge) >= 2.0:
        emoji = "🔥" if edge > 0 else "❄️"
        print(f"   {emoji} {away}@{home}: {edge:+.1f} runs ({game['predicted_total']:.1f} vs {game['market_total']:.1f})")
    elif abs(edge) >= 1.0:
        emoji = "📈" if edge > 0 else "📉"
        print(f"   {emoji} {away}@{home}: {edge:+.1f} runs ({game['predicted_total']:.1f} vs {game['market_total']:.1f})")

print(f"\n🎯 Interesting Observations:")
print(f"   • Largest OVER edge: {games.loc[games['edge'].idxmax(), 'away_team'][:3]}@{games.loc[games['edge'].idxmax(), 'home_team'][:3]} (+{games['edge'].max():.1f})")
print(f"   • Largest UNDER edge: {games.loc[games['edge'].idxmin(), 'away_team'][:3]}@{games.loc[games['edge'].idxmin(), 'home_team'][:3]} ({games['edge'].min():.1f})")

# Look for unusual market totals
unusual = games[(games['market_total'] >= 15) | (games['market_total'] <= 7)]
if len(unusual) > 0:
    print(f"\n⚠️  Unusual Market Totals:")
    for _, game in unusual.iterrows():
        away = game['away_team'][:3].upper()
        home = game['home_team'][:3].upper()
        print(f"   {away}@{home}: Market={game['market_total']:.1f} (Pred={game['predicted_total']:.1f})")

# Odds analysis
games['over_implied'] = 100 / (100 + games['over_odds'].where(games['over_odds'] > 0, -games['over_odds']/(games['over_odds']-100)))
games['under_implied'] = 100 / (100 + games['under_odds'].where(games['under_odds'] > 0, -games['under_odds']/(games['under_odds']-100)))
games['vig'] = (games['over_implied'] + games['under_implied'] - 100).round(1)

print(f"\n💰 Sportsbook Vig Analysis:")
print(f"   Average vig: {games['vig'].mean():.1f}%")
print(f"   Lowest vig: {games['vig'].min():.1f}% ({games.loc[games['vig'].idxmin(), 'away_team'][:3]}@{games.loc[games['vig'].idxmin(), 'home_team'][:3]})")
print(f"   Highest vig: {games['vig'].max():.1f}% ({games.loc[games['vig'].idxmax(), 'away_team'][:3]}@{games.loc[games['vig'].idxmax(), 'home_team'][:3]})")

print(f"\n🎲 Game Status:")
total_games = len(games)
completed = games['total_runs'].notna().sum()
print(f"   {completed}/{total_games} games completed")
print(f"   {total_games - completed} games still in progress or upcoming")
