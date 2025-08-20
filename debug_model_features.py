"""
Debug model feature engineering to see why predictions are so low
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_model_features():
    """Debug the feature engineering that's producing low predictions"""
    
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    
    # Get a sample game to debug
    query = """
        SELECT 
            home_team, away_team, predicted_total, market_total,
            home_team_stats, away_team_stats,
            home_pitcher_stats, away_pitcher_stats
        FROM enhanced_games 
        WHERE date = '2025-08-20'
        AND home_team = 'Chicago Cubs'
        LIMIT 1
    """
    
    df = pd.read_sql(query, conn)
    
    print("🔍 DEBUGGING MODEL FEATURES")
    print("=" * 60)
    print(f"Game: {df.iloc[0]['away_team']} @ {df.iloc[0]['home_team']}")
    print(f"Predicted: {df.iloc[0]['predicted_total']}")
    print(f"Market: {df.iloc[0]['market_total']}")
    print()
    
    # Parse the stats JSON
    import json
    home_stats = json.loads(df.iloc[0]['home_team_stats'])
    away_stats = json.loads(df.iloc[0]['away_team_stats'])
    home_pitcher = json.loads(df.iloc[0]['home_pitcher_stats'])
    away_pitcher = json.loads(df.iloc[0]['away_pitcher_stats'])
    
    print("📊 TEAM OFFENSIVE STATS:")
    print("-" * 40)
    print(f"Home ({df.iloc[0]['home_team']}):")
    print(f"  Runs per game: {home_stats.get('runs_per_game', 'N/A')}")
    print(f"  OPS: {home_stats.get('ops', 'N/A')}")
    print(f"  Hits per game: {home_stats.get('hits_per_game', 'N/A')}")
    print(f"  Home runs per game: {home_stats.get('hr_per_game', 'N/A')}")
    print()
    print(f"Away ({df.iloc[0]['away_team']}):")
    print(f"  Runs per game: {away_stats.get('runs_per_game', 'N/A')}")
    print(f"  OPS: {away_stats.get('ops', 'N/A')}")
    print(f"  Hits per game: {away_stats.get('hits_per_game', 'N/A')}")
    print(f"  Home runs per game: {away_stats.get('hr_per_game', 'N/A')}")
    
    print("\n🥎 PITCHER STATS:")
    print("-" * 40)
    print(f"Home Pitcher:")
    print(f"  ERA: {home_pitcher.get('era', 'N/A')}")
    print(f"  WHIP: {home_pitcher.get('whip', 'N/A')}")
    print(f"  K/9: {home_pitcher.get('k_per_9', 'N/A')}")
    print(f"  BB/9: {home_pitcher.get('bb_per_9', 'N/A')}")
    print()
    print(f"Away Pitcher:")
    print(f"  ERA: {away_pitcher.get('era', 'N/A')}")
    print(f"  WHIP: {away_pitcher.get('whip', 'N/A')}")
    print(f"  K/9: {away_pitcher.get('k_per_9', 'N/A')}")
    print(f"  BB/9: {away_pitcher.get('bb_per_9', 'N/A')}")
    
    # Calculate expected runs based on team stats
    print("\n🧮 EXPECTED CALCULATION:")
    print("-" * 40)
    
    home_rpg = float(home_stats.get('runs_per_game', 0))
    away_rpg = float(away_stats.get('runs_per_game', 0))
    
    print(f"Basic team average: {home_rpg:.1f} + {away_rpg:.1f} = {home_rpg + away_rpg:.1f}")
    
    # Check if our features are reasonable
    print(f"\n⚠️  FEATURE SANITY CHECK:")
    print("-" * 40)
    
    if home_rpg < 2.0 or away_rpg < 2.0:
        print(f"🚨 PROBLEM: Team runs per game unusually low!")
        print(f"   Home: {home_rpg:.2f}, Away: {away_rpg:.2f}")
        print(f"   Normal MLB teams score 4-6 runs per game")
    
    if home_rpg + away_rpg < 6.0:
        print(f"🚨 PROBLEM: Combined expected runs too low!")
        print(f"   Total: {home_rpg + away_rpg:.2f}")
        print(f"   Normal MLB games total 8-10 runs")
    
    # Let's also check some recent actual scores to see if our data is realistic
    print(f"\n📈 RECENT ACTUAL SCORES:")
    print("-" * 40)
    
    recent_query = """
        SELECT home_team, away_team, actual_total
        FROM enhanced_games 
        WHERE date >= '2025-01-01' 
        AND actual_total IS NOT NULL
        ORDER BY date DESC
        LIMIT 10
    """
    
    recent_df = pd.read_sql(recent_query, conn)
    
    if len(recent_df) > 0:
        print("Recent games with actual totals:")
        for _, row in recent_df.iterrows():
            print(f"  {row['away_team']} @ {row['home_team']}: {row['actual_total']} runs")
        
        avg_actual = recent_df['actual_total'].mean()
        print(f"\nAverage actual total: {avg_actual:.1f} runs")
        
        if avg_actual < 7.0:
            print(f"🚨 PROBLEM: Actual game totals are unusually low!")
            print(f"   This suggests our data collection might be wrong")
    else:
        print("No recent actual totals found in database")
    
    conn.close()

if __name__ == "__main__":
    debug_model_features()
