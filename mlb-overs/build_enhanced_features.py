#!/usr/bin/env python3
"""
Enhanced Build Features - Uses our collected enhanced historical data
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime

def build_enhanced_features():
    """Build features using our enhanced historical data"""
    
    print("🎯 BUILDING ENHANCED FEATURES FROM COLLECTED DATA")
    print("=" * 60)
    
    # Load our enhanced historical data
    try:
        enhanced_data = pd.read_parquet('data/enhanced_historical_games_2025.parquet')
        print(f"✅ Loaded enhanced historical data: {len(enhanced_data)} games")
        print(f"   Date range: {enhanced_data['date'].min()} to {enhanced_data['date'].max()}")
    except Exception as e:
        print(f"❌ Error loading enhanced data: {e}")
        return False
    
    # Prepare features for ML training based on what we actually have
    feature_columns = [
        # Core identifiers
        'game_id', 'date', 'home_team', 'away_team',
        
        # Target variable
        'total_runs',
        
        # Basic game data
        'home_score', 'away_score',
        
        # Weather features (convert to numeric)
        'temperature', 'wind_speed', 'wind_direction',
        'weather_condition',
        
        # Venue features
        'venue_id', 'venue_name',
        
        # Pitcher basic stats
        'home_sp_id', 'home_sp_er', 'home_sp_ip', 'home_sp_k', 'home_sp_bb', 'home_sp_h',
        'away_sp_id', 'away_sp_er', 'away_sp_ip', 'away_sp_k', 'away_sp_bb', 'away_sp_h',
        
        # Team basic stats
        'home_team_hits', 'home_team_runs', 'home_team_rbi', 'home_team_lob',
        'away_team_hits', 'away_team_runs', 'away_team_rbi', 'away_team_lob',
        
        # Game context
        'game_type', 'day_night'
    ]
    
    # Use all available columns
    feature_data = enhanced_data.copy()
    
    print(f"✅ Using all {len(feature_data.columns)} available columns")
    
    # Clean and convert data types
    print("🔧 Converting data types...")
    
    # Convert temperature to numeric
    feature_data['temperature'] = pd.to_numeric(feature_data['temperature'], errors='coerce')
    
    # Convert IP (innings pitched) to numeric (handle fractions like "6.1")
    def convert_ip_to_numeric(ip_str):
        if pd.isna(ip_str):
            return 0.0
        try:
            if isinstance(ip_str, str):
                if '.' in ip_str:
                    whole, fraction = ip_str.split('.')
                    return float(whole) + float(fraction) / 3.0  # .1 = 1/3 inning, .2 = 2/3 inning
                else:
                    return float(ip_str)
            return float(ip_str)
        except:
            return 0.0
    
    feature_data['home_sp_ip_numeric'] = feature_data['home_sp_ip'].apply(convert_ip_to_numeric)
    feature_data['away_sp_ip_numeric'] = feature_data['away_sp_ip'].apply(convert_ip_to_numeric)
    
    # Calculate pitcher ERAs
    feature_data['home_pitcher_era'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_er'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_era'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_er'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate pitcher WHIPs
    feature_data['home_pitcher_whip'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_h'] + feature_data['home_sp_bb']) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_whip'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_h'] + feature_data['away_sp_bb']) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate K/9 rates
    feature_data['home_pitcher_k9'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_k'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_k9'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_k'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate BB/9 rates
    feature_data['home_pitcher_bb9'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_bb'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_bb9'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_bb'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Create additional derived features
    print("🔧 Creating derived features...")
    
    # ERA features
    feature_data['total_era'] = feature_data['home_pitcher_era'] + feature_data['away_pitcher_era']
    feature_data['era_differential'] = feature_data['home_pitcher_era'] - feature_data['away_pitcher_era']
    
    # WHIP features
    feature_data['total_whip'] = feature_data['home_pitcher_whip'] + feature_data['away_pitcher_whip']
    feature_data['whip_differential'] = feature_data['home_pitcher_whip'] - feature_data['away_pitcher_whip']
    
    # Strikeout features
    feature_data['total_k9'] = feature_data['home_pitcher_k9'] + feature_data['away_pitcher_k9']
    feature_data['k9_differential'] = feature_data['home_pitcher_k9'] - feature_data['away_pitcher_k9']
    
    # Team features
    feature_data['total_team_hits'] = feature_data['home_team_hits'] + feature_data['away_team_hits']
    feature_data['hits_differential'] = feature_data['home_team_hits'] - feature_data['away_team_hits']
    
    # Weather factor (now with proper numeric temperature)
    feature_data['weather_factor'] = (
        (feature_data['temperature'] - 72) * 0.01 +  # Temperature effect
        feature_data['wind_speed'] * 0.02  # Wind effect
    ).fillna(0)
    
    # Home field advantage
    feature_data['score_differential'] = feature_data['home_score'] - feature_data['away_score']
    
    # Day/night game indicator
    feature_data['is_night_game'] = (feature_data['day_night'] == 'N').astype(int)
    
    # Venue factors (basic park effects)
    venue_factors = {
        'Coors Field': 1.15,  # High altitude, hitter friendly
        'Fenway Park': 1.05,  # Green Monster
        'Yankee Stadium': 1.08,  # Short right field
        'Minute Maid Park': 1.03,  # Crawford boxes
        'Great American Ball Park': 1.06,  # Hitter friendly
        'Citizens Bank Park': 1.04,  # Slight hitter park
        'Globe Life Field': 1.02,  # Slight hitter park
        # Pitcher friendly
        'Petco Park': 0.92,  # Large foul territory
        'Marlins Park': 0.94,  # Pitcher friendly
        'T-Mobile Park': 0.96,  # Marine layer
        'Comerica Park': 0.97,  # Large dimensions
        'Kauffman Stadium': 0.98,  # Large foul territory
    }
    
    feature_data['park_factor'] = feature_data['venue_name'].map(venue_factors).fillna(1.0)
    
    # Pitcher quality score
    feature_data['home_pitcher_quality'] = (
        (9.0 - feature_data['home_pitcher_era'].clip(0, 9)) / 9.0 * 0.4 +
        (feature_data['home_pitcher_k9'].clip(0, 15) / 15.0) * 0.3 +
        (1.0 - feature_data['home_pitcher_whip'].clip(0, 2) / 2.0) * 0.3
    ).clip(0, 1)
    
    feature_data['away_pitcher_quality'] = (
        (9.0 - feature_data['away_pitcher_era'].clip(0, 9)) / 9.0 * 0.4 +
        (feature_data['away_pitcher_k9'].clip(0, 15) / 15.0) * 0.3 +
        (1.0 - feature_data['away_pitcher_whip'].clip(0, 2) / 2.0) * 0.3
    ).clip(0, 1)
    
    feature_data['pitcher_quality_differential'] = feature_data['home_pitcher_quality'] - feature_data['away_pitcher_quality']
    
    # Remove rows with missing target variable
    if 'total_runs' in feature_data.columns:
        before_count = len(feature_data)
        feature_data = feature_data.dropna(subset=['total_runs'])
        after_count = len(feature_data)
        print(f"📊 Removed {before_count - after_count} games missing total_runs")
    
    # Save the enhanced training data
    output_path = 'features/train_enhanced.parquet'
    feature_data.to_parquet(output_path, index=False)
    print(f"✅ Saved enhanced training data: {output_path}")
    print(f"   Final feature count: {len(feature_data)} games")
    print(f"   Feature columns: {len(feature_data.columns)}")
    
    # Also save a backup of the original
    try:
        original_train = pd.read_parquet('features/train.parquet')
        original_train.to_parquet('features/train_backup.parquet', index=False)
        print(f"✅ Backed up original train.parquet")
    except:
        pass
    
    # Replace the original with our enhanced version
    feature_data.to_parquet('features/train.parquet', index=False)
    print(f"✅ Updated features/train.parquet with enhanced data")
    
    # Print feature summary
    print(f"\n📊 ENHANCED FEATURE SUMMARY:")
    print(f"   Training games: {len(feature_data)}")
    print(f"   Date range: {feature_data['date'].min()} to {feature_data['date'].max()}")
    print(f"   Feature columns: {len(feature_data.columns)}")
    
    if 'total_runs' in feature_data.columns:
        print(f"   Avg total runs: {feature_data['total_runs'].mean():.2f}")
        print(f"   Min/Max runs: {feature_data['total_runs'].min():.0f} / {feature_data['total_runs'].max():.0f}")
    
    return True

def main():
    """Main execution"""
    
    # Change to mlb-overs directory
    import os
    os.chdir('S:/Projects/AI_Predictions/mlb-overs')
    
    success = build_enhanced_features()
    
    if success:
        print(f"\n✅ ENHANCED FEATURES BUILD COMPLETE!")
        print(f"   The gameday.ps1 script will now use enhanced historical data")
        print(f"   ML model training will use real collected data")
        
        # Also create today's game features for inference
        create_todays_game_features()
    else:
        print(f"\n❌ Enhanced features build failed")
    
    return success

def create_todays_game_features():
    """Create features for today's games that match our enhanced historical data structure"""
    print(f"\n🎯 CREATING TODAY'S GAME FEATURES (ENHANCED FORMAT)")
    print(f"============================================")
    
    try:
        import os
        from sqlalchemy import create_engine, text
        
        # Database connection
        db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        engine = create_engine(db_url)
        
        # Get today's games with betting lines
        today = datetime.now().strftime('%Y-%m-%d')
        games_query = f"""
        SELECT g.game_id, g.date, g.home_team, g.away_team, g.home_sp_id, g.away_sp_id, 
               COALESCE(mt.k_total, g.close_total) as k_close, g.park_id
        FROM games g
        LEFT JOIN (
            SELECT game_id, k_total,
                   ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY updated_at DESC) as rn
            FROM markets_totals 
            WHERE date = '{today}' AND k_total IS NOT NULL
        ) mt ON g.game_id = mt.game_id AND mt.rn = 1
        WHERE g.date = '{today}'
        ORDER BY g.game_id
        """
        
        todays_games = pd.read_sql(games_query, engine)
        
        if todays_games.empty:
            print(f"📊 No games found for {today}")
            return
        
        print(f"📊 Found {len(todays_games)} games for {today}")
        
        # Create features that match our enhanced historical data structure
        # Using the same 35 columns as enhanced_historical_games_2025.parquet
        
        todays_features = pd.DataFrame()
        
        # Core identifiers (from database)
        todays_features['game_id'] = todays_games['game_id']
        todays_features['date'] = todays_games['date'] 
        todays_features['home_team'] = todays_games['home_team']
        todays_features['away_team'] = todays_games['away_team']
        
        # Scores (unknown for future games - use typical values)
        todays_features['home_score'] = 5.0  # Average MLB score
        todays_features['away_score'] = 4.0  # Average MLB score  
        todays_features['total_runs'] = 9.0  # Average total runs
        
        # Weather features (defaults)
        todays_features['weather_condition'] = 'Clear'
        todays_features['temperature'] = 75.0
        todays_features['wind_speed'] = 8.0
        todays_features['wind_direction'] = 'Out to CF'
        
        # Venue features
        todays_features['venue_id'] = todays_games['park_id']
        todays_features['venue_name'] = 'Unknown Venue'  # Would need venue lookup
        
        # Starting pitcher IDs (from database)
        todays_features['home_sp_id'] = todays_games['home_sp_id']
        todays_features['away_sp_id'] = todays_games['away_sp_id']
        
        # Starting pitcher stats (realistic defaults based on league averages)
        todays_features['home_sp_er'] = 3.0  # 3 earned runs typical
        todays_features['home_sp_ip'] = 6.0  # 6 innings typical
        todays_features['home_sp_k'] = 6.0   # 6 strikeouts typical
        todays_features['home_sp_bb'] = 2.0  # 2 walks typical  
        todays_features['home_sp_h'] = 6.0   # 6 hits typical
        
        todays_features['away_sp_er'] = 3.0
        todays_features['away_sp_ip'] = 6.0  
        todays_features['away_sp_k'] = 6.0
        todays_features['away_sp_bb'] = 2.0
        todays_features['away_sp_h'] = 6.0
        
        # Team offensive stats (typical values)
        todays_features['home_team_hits'] = 8.0   # 8 hits typical
        todays_features['home_team_runs'] = 5.0   # 5 runs typical
        todays_features['home_team_rbi'] = 5.0    # 5 RBI typical
        todays_features['home_team_lob'] = 6.0    # 6 LOB typical
        
        todays_features['away_team_hits'] = 7.0
        todays_features['away_team_runs'] = 4.0
        todays_features['away_team_rbi'] = 4.0  
        todays_features['away_team_lob'] = 6.0
        
        # Game context
        todays_features['game_type'] = 'Regular'
        todays_features['day_night'] = 'D'  # Day game
        
        # Add betting lines
        todays_features['k_close'] = todays_games['k_close']
        
        # Filter out games without betting lines for inference
        valid_games = todays_features[todays_features['k_close'].notna()].copy()
        
        if valid_games.empty:
            print(f"❌ No games with betting lines found")
            return
            
        print(f"📊 {len(valid_games)} games have betting lines")
        
        # Save for inference
        valid_games.to_parquet('data/game_totals_today.parquet', index=False)
        print(f"✅ Saved today's enhanced game features: data/game_totals_today.parquet")
        print(f"   Games with betting lines: {len(valid_games)}")
        print(f"   Feature columns: {len(valid_games.columns)}")
        
        # Apply the same feature engineering as we do for historical data
        print(f"🔧 Applying feature engineering to today's games...")
        enhanced_todays = apply_feature_engineering(valid_games)
        
        # Save the fully engineered features for ML inference
        enhanced_todays.to_parquet('data/game_totals_today.parquet', index=False)
        print(f"✅ Updated with engineered features: {len(enhanced_todays.columns)} total columns")
        
        # Show sample betting lines
        print(f"\n📊 Betting Lines Sample:")
        for _, game in enhanced_todays.head(3).iterrows():
            print(f"   {game['away_team']} @ {game['home_team']}: {game['k_close']}")
        
    except Exception as e:
        print(f"⚠️  Error creating today's features: {e}")
        import traceback
        traceback.print_exc()

def apply_feature_engineering(feature_data):
    """Apply the same feature engineering we use for historical data"""
    
    # Convert IP (innings pitched) to numeric (handle fractions like "6.1")
    def convert_ip_to_numeric(ip_val):
        if pd.isna(ip_val):
            return 0.0
        try:
            if isinstance(ip_val, str) and '.' in ip_val:
                whole, fraction = ip_val.split('.')
                return float(whole) + float(fraction) / 3.0  # .1 = 1/3 inning, .2 = 2/3 inning
            return float(ip_val)
        except:
            return 0.0
    
    feature_data = feature_data.copy()
    
    # Convert IP to numeric
    feature_data['home_sp_ip_numeric'] = feature_data['home_sp_ip'].apply(convert_ip_to_numeric)
    feature_data['away_sp_ip_numeric'] = feature_data['away_sp_ip'].apply(convert_ip_to_numeric)
    
    # Calculate pitcher ERAs
    feature_data['home_pitcher_era'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_er'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_era'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_er'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate pitcher WHIPs
    feature_data['home_pitcher_whip'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_h'] + feature_data['home_sp_bb']) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_whip'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_h'] + feature_data['away_sp_bb']) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate K/9 rates
    feature_data['home_pitcher_k9'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_k'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_k9'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_k'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Calculate BB/9 rates
    feature_data['home_pitcher_bb9'] = np.where(
        feature_data['home_sp_ip_numeric'] > 0,
        (feature_data['home_sp_bb'] * 9.0) / feature_data['home_sp_ip_numeric'],
        0.0
    )
    feature_data['away_pitcher_bb9'] = np.where(
        feature_data['away_sp_ip_numeric'] > 0,
        (feature_data['away_sp_bb'] * 9.0) / feature_data['away_sp_ip_numeric'],
        0.0
    )
    
    # Create additional derived features
    # ERA features
    feature_data['total_era'] = feature_data['home_pitcher_era'] + feature_data['away_pitcher_era']
    feature_data['era_differential'] = feature_data['home_pitcher_era'] - feature_data['away_pitcher_era']
    
    # WHIP features
    feature_data['total_whip'] = feature_data['home_pitcher_whip'] + feature_data['away_pitcher_whip']
    feature_data['whip_differential'] = feature_data['home_pitcher_whip'] - feature_data['away_pitcher_whip']
    
    # Strikeout features
    feature_data['total_k9'] = feature_data['home_pitcher_k9'] + feature_data['away_pitcher_k9']
    feature_data['k9_differential'] = feature_data['home_pitcher_k9'] - feature_data['away_pitcher_k9']
    
    # Team features
    feature_data['total_team_hits'] = feature_data['home_team_hits'] + feature_data['away_team_hits']
    feature_data['hits_differential'] = feature_data['home_team_hits'] - feature_data['away_team_hits']
    
    # Weather factor (now with proper numeric temperature)
    feature_data['weather_factor'] = (
        (feature_data['temperature'] - 72) * 0.01 +  # Temperature effect
        feature_data['wind_speed'] * 0.02  # Wind effect
    ).fillna(0)
    
    # Home field advantage
    feature_data['score_differential'] = feature_data['home_score'] - feature_data['away_score']
    
    # Day/night game indicator
    feature_data['is_night_game'] = (feature_data['day_night'] == 'N').astype(int)
    
    # Venue factors (basic park effects)
    venue_factors = {
        'Coors Field': 1.15,  # High altitude, hitter friendly
        'Fenway Park': 1.05,  # Green Monster
        'Yankee Stadium': 1.08,  # Short right field
        'Minute Maid Park': 1.03,  # Crawford boxes
        'Great American Ball Park': 1.06,  # Hitter friendly
        'Citizens Bank Park': 1.04,  # Slight hitter park
        'Globe Life Field': 1.02,  # Slight hitter park
        # Pitcher friendly
        'Petco Park': 0.92,  # Large foul territory
        'Marlins Park': 0.94,  # Pitcher friendly
        'T-Mobile Park': 0.96,  # Marine layer
        'Comerica Park': 0.97,  # Large dimensions
        'Kauffman Stadium': 0.98,  # Large foul territory
    }
    
    feature_data['park_factor'] = feature_data['venue_name'].map(venue_factors).fillna(1.0)
    
    # Pitcher quality score
    feature_data['home_pitcher_quality'] = (
        (9.0 - feature_data['home_pitcher_era'].clip(0, 9)) / 9.0 * 0.4 +
        (feature_data['home_pitcher_k9'].clip(0, 15) / 15.0) * 0.3 +
        (1.0 - feature_data['home_pitcher_whip'].clip(0, 2) / 2.0) * 0.3
    ).clip(0, 1)
    
    feature_data['away_pitcher_quality'] = (
        (9.0 - feature_data['away_pitcher_era'].clip(0, 9)) / 9.0 * 0.4 +
        (feature_data['away_pitcher_k9'].clip(0, 15) / 15.0) * 0.3 +
        (1.0 - feature_data['away_pitcher_whip'].clip(0, 2) / 2.0) * 0.3
    ).clip(0, 1)
    
    feature_data['pitcher_quality_differential'] = feature_data['home_pitcher_quality'] - feature_data['away_pitcher_quality']
    
    return feature_data

if __name__ == "__main__":
    main()
