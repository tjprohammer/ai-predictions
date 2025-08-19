#!/usr/bin/env python3
"""
Add pitcher rolling stats join to enhanced_bullpen_predictor.py
This implements the merge_asof logic to join real pitcher form data
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

def add_pitcher_rolling_stats(games_df, engine):
    """
    Add real pitcher rolling stats using merge_asof join with pitcher_daily_rolling table
    
    Args:
        games_df: DataFrame with games data including home_sp_id, away_sp_id, date
        engine: SQLAlchemy engine for database connection
        
    Returns:
        Enhanced games_df with real pitcher rate stats (ERA, WHIP, K/9, BB/9)
    """
    if games_df.empty:
        return games_df
        
    logger.info("🔄 Adding real pitcher rolling stats via merge_asof...")
    
    # Get all pitcher rolling stats
    roll_query = """
    SELECT pitcher_id, stat_date, gs, ip, er, bb, k, h, hr,
           era, whip, k_per_9, bb_per_9, hr_per_9
    FROM pitcher_daily_rolling
    ORDER BY pitcher_id, stat_date
    """
    
    roll_df = pd.read_sql(roll_query, engine)
    
    if roll_df.empty:
        logger.error("❌ No pitcher rolling stats found - run the materialization query first")
        return games_df
        
    logger.info(f"📊 Loaded {len(roll_df)} pitcher rolling stat rows")
    
    # Ensure proper data types for merge_asof
    games_df = games_df.copy()
    games_df['date'] = pd.to_datetime(games_df['date'])
    roll_df['stat_date'] = pd.to_datetime(roll_df['stat_date'])
    
    logger.info(f"📅 Date ranges - Games: {games_df['date'].min()} to {games_df['date'].max()}")
    logger.info(f"📅 Date ranges - Rolling: {roll_df['stat_date'].min()} to {roll_df['stat_date'].max()}")
    
    # For merge_asof to work properly, we need to sort carefully
    # Method: Do separate joins for home and away, then merge results
    
    # === HOME PITCHER JOIN ===
    logger.info("🏠 Joining home pitcher rolling stats...")
    
    # Filter and prepare home pitcher data
    home_pitchers = games_df[['game_id', 'date', 'home_sp_id']].copy()
    home_pitchers = home_pitchers.sort_values(['date'])  # Sort by date only for merge_asof
    
    # Prepare rolling data for home join
    roll_home = roll_df[['pitcher_id', 'stat_date', 'era', 'whip', 'k_per_9', 'bb_per_9', 'gs']].copy()
    roll_home = roll_home.rename(columns={'pitcher_id': 'home_sp_id'})
    roll_home = roll_home.sort_values(['stat_date'])  # Sort by date only
    
    # Perform home join
    home_joined = pd.merge_asof(
        home_pitchers, roll_home,
        left_on='date', right_on='stat_date',
        by='home_sp_id',
        direction='backward',
        suffixes=('', '_home')
    )
    
    # === AWAY PITCHER JOIN ===
    logger.info("✈️  Joining away pitcher rolling stats...")
    
    # Filter and prepare away pitcher data  
    away_pitchers = games_df[['game_id', 'date', 'away_sp_id']].copy()
    away_pitchers = away_pitchers.sort_values(['date'])  # Sort by date only
    
    # Prepare rolling data for away join
    roll_away = roll_df[['pitcher_id', 'stat_date', 'era', 'whip', 'k_per_9', 'bb_per_9', 'gs']].copy()
    roll_away = roll_away.rename(columns={'pitcher_id': 'away_sp_id'})
    roll_away = roll_away.sort_values(['stat_date'])  # Sort by date only
    
    # Perform away join
    away_joined = pd.merge_asof(
        away_pitchers, roll_away,
        left_on='date', right_on='stat_date', 
        by='away_sp_id',
        direction='backward',
        suffixes=('', '_away')
    )
    
    # === MERGE RESULTS BACK INTO GAMES ===
    logger.info("🎯 Merging pitcher stats back into games...")
    
    # Add home pitcher stats
    home_stats = home_joined[['game_id', 'era', 'whip', 'k_per_9', 'bb_per_9', 'gs']].rename(columns={
        'era': 'home_sp_era',
        'whip': 'home_sp_whip', 
        'k_per_9': 'home_sp_k_per_9',
        'bb_per_9': 'home_sp_bb_per_9',
        'gs': 'home_sp_starts'
    })
    
    # Add away pitcher stats
    away_stats = away_joined[['game_id', 'era', 'whip', 'k_per_9', 'bb_per_9', 'gs']].rename(columns={
        'era': 'away_sp_era',
        'whip': 'away_sp_whip',
        'k_per_9': 'away_sp_k_per_9', 
        'bb_per_9': 'away_sp_bb_per_9',
        'gs': 'away_sp_starts'
    })
    
    # Merge back into original games dataframe
    games_df = games_df.merge(home_stats, on='game_id', how='left', suffixes=('', '_home_new'))
    games_df = games_df.merge(away_stats, on='game_id', how='left', suffixes=('', '_away_new'))
    
    # Also create season stats (rolling stats ARE season stats)
    games_df['home_sp_season_era'] = games_df['home_sp_era']
    games_df['away_sp_season_era'] = games_df['away_sp_era'] 
    games_df['home_sp_season_whip'] = games_df['home_sp_whip']
    games_df['away_sp_season_whip'] = games_df['away_sp_whip']
    
    # Log diagnostics
    home_era_count = games_df['home_sp_era'].notna().sum()
    away_era_count = games_df['away_sp_era'].notna().sum()
    total_games = len(games_df)
    
    logger.info(f"✅ Real pitcher data joined:")
    logger.info(f"   Home ERA coverage: {home_era_count}/{total_games} ({100*home_era_count/total_games:.1f}%)")
    logger.info(f"   Away ERA coverage: {away_era_count}/{total_games} ({100*away_era_count/total_games:.1f}%)")
    
    if home_era_count > 0:
        logger.info(f"   Home ERA range: {games_df['home_sp_era'].min():.2f} - {games_df['home_sp_era'].max():.2f}")
        logger.info(f"   Home ERA std: {games_df['home_sp_era'].std():.3f}")
    if away_era_count > 0:
        logger.info(f"   Away ERA range: {games_df['away_sp_era'].min():.2f} - {games_df['away_sp_era'].max():.2f}")
        logger.info(f"   Away ERA std: {games_df['away_sp_era'].std():.3f}")
    
    return games_df


if __name__ == "__main__":
    # Test the function
    import os
    from sqlalchemy import create_engine
    
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    engine = create_engine(db_url)
    
    # Test with a sample game
    test_games = pd.DataFrame({
        'game_id': [776711],
        'date': ['2025-08-16'],
        'home_sp_id': [434378],  # Justin Verlander
        'away_sp_id': [547179],  # Michael Lorenzen
        'home_team': ['HOU'],
        'away_team': ['TEX']
    })
    
    print("🧪 Testing pitcher rolling stats join...")
    enhanced = add_pitcher_rolling_stats(test_games, engine)
    
    print(f"\nResults:")
    print(f"Home ERA: {enhanced['home_sp_era'].iloc[0]:.2f}")
    print(f"Away ERA: {enhanced['away_sp_era'].iloc[0]:.2f}")
    print(f"Home WHIP: {enhanced['home_sp_whip'].iloc[0]:.2f}")
    print(f"Away WHIP: {enhanced['away_sp_whip'].iloc[0]:.2f}")
