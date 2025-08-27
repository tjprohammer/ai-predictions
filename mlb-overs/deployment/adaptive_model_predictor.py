#!/usr/bin/env python3
"""
Extract features only from enhanced predictor, then predict with adaptive model
"""

import sys
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
import logging
from pathlib import Path

sys.path.append('s:/Projects/AI_Predictions/mlb-overs/deployment')

# Import specific functions from enhanced predictor without instantiating the class
def extract_features_only():
    """Extract just the feature engineering part without model prediction"""
    # Import needed functions
    from enhanced_bullpen_predictor import (
        add_pitcher_rolling_stats, add_pitcher_advanced_stats, 
        add_team_advanced_stats, _inject_basic_ballpark_factors
    )
    
    # Setup database connection
    db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
    engine = create_engine(db_url)
    date_str = '2025-08-23'
    
    print(f"🔍 EXTRACTING FEATURES FOR {date_str}")
    print("=" * 60)
    
    # Get today's games data directly from database
    query = text("""
    SELECT
      lgf.*,
      eg.home_sp_id,
      eg.away_sp_id,
      eg.venue_name,
      COALESCE(NULLIF(lgf.temperature, 0), eg.temperature) AS temperature,
      COALESCE(NULLIF(lgf.wind_speed, 0), eg.wind_speed) AS wind_speed,
      COALESCE(eg.home_team_avg, 0.250) AS home_team_avg,
      COALESCE(eg.away_team_avg, 0.250) AS away_team_avg,
      COALESCE(eg.humidity, 50) AS humidity
      -- Add adaptive model required features
    FROM legitimate_game_features lgf
    JOIN enhanced_games eg ON lgf.game_id = eg.game_id
    WHERE lgf.date = :target_date
    ORDER BY lgf.game_id
    """)
    
    games_df = pd.read_sql(query, engine, params={'target_date': date_str})
    
    if games_df.empty:
        print("❌ No games found for today")
        return None
        
    print(f"✅ Found {len(games_df)} games for {date_str}")
    
    # Add pitcher rolling stats
    try:
        games_df = add_pitcher_rolling_stats(games_df, engine)
        print("✅ Added pitcher rolling stats")
    except Exception as e:
        print(f"⚠️ Pitcher rolling stats failed: {e}")
    
    # Add advanced pitcher stats
    try:
        games_df = add_pitcher_advanced_stats(games_df, engine)
        print("✅ Added advanced pitcher stats")
    except Exception as e:
        print(f"⚠️ Advanced pitcher stats failed: {e}")
    
    # Add advanced team stats
    try:
        games_df = add_team_advanced_stats(games_df, engine)
        print("✅ Added advanced team stats")
    except Exception as e:
        print(f"⚠️ Advanced team stats failed: {e}")
    
    # Basic feature engineering (simplified version)
    featured_df = games_df.copy()
    
    # Add missing team features with defaults
    team_defaults = {
        'home_team_iso': 0.150, 'away_team_iso': 0.150,
        'home_team_woba': 0.315, 'away_team_woba': 0.315,
        'home_team_xwoba': 0.315, 'away_team_xwoba': 0.315,
        'home_team_babip': 0.300, 'away_team_babip': 0.300,
        'home_team_wrcplus': 100, 'away_team_wrcplus': 100,
        'home_team_bb_pct': 0.085, 'away_team_bb_pct': 0.085,
        'home_team_k_pct': 0.220, 'away_team_k_pct': 0.220,
        'home_team_rpg_l30': 4.5, 'away_team_rpg_l30': 4.5,
        'home_team_games_l30': 25, 'away_team_games_l30': 25,
        'home_sp_era_std': 1.5, 'away_sp_era_std': 1.5,
        'home_pitcher_experience': 20, 'away_pitcher_experience': 20
    }
    
    for col, default_val in team_defaults.items():
        if col not in featured_df.columns:
            featured_df[col] = default_val
    
    # Add basic engineered features needed by adaptive model
    # Combined ERA features
    featured_df['combined_sp_era'] = (featured_df['home_sp_era'] + featured_df['away_sp_era']) / 2
    featured_df['sp_era_differential'] = featured_df['home_sp_era'] - featured_df['away_sp_era']
    
    # Combined WHIP
    featured_df['combined_whip'] = (featured_df['home_sp_whip'] + featured_df['away_sp_whip']) / 2
    
    # Combined K rate
    featured_df['combined_k_rate'] = (featured_df['home_sp_k_per_9'] + featured_df['away_sp_k_per_9']) / 2
    
    # Combined BB rate
    featured_df['combined_bb_rate'] = (featured_df['home_sp_bb_per_9'] + featured_df['away_sp_bb_per_9']) / 2
    
    # Combined HR rate
    featured_df['combined_hr_rate'] = (featured_df['home_sp_hr_per_9'] + featured_df['away_sp_hr_per_9']) / 2
    
    # Pitcher experience (average of home and away)
    featured_df['pitcher_experience'] = (featured_df['home_pitcher_experience'] + featured_df['away_pitcher_experience']) / 2
    
    # ERA consistency
    featured_df['era_consistency'] = (featured_df['home_sp_era_std'] + featured_df['away_sp_era_std']) / 2
    
    # Combined team offense
    featured_df['combined_offense_rpg'] = (featured_df['home_team_rpg_l30'] + featured_df['away_team_rpg_l30']) / 2
    
    # Combined WOBA
    featured_df['combined_woba'] = (featured_df['home_team_woba'] + featured_df['away_team_woba']) / 2
    
    # Combined wRCplus
    featured_df['combined_wrcplus'] = (featured_df['home_team_wrcplus'] + featured_df['away_team_wrcplus']) / 2
    
    # Combined power
    featured_df['combined_power'] = (featured_df['home_team_iso'] + featured_df['away_team_iso']) / 2
    
    # Offense imbalance
    featured_df['offense_imbalance'] = abs(featured_df['home_team_rpg_l30'] - featured_df['away_team_rpg_l30'])
    
    # Power imbalance
    featured_df['power_imbalance'] = abs(featured_df['home_team_iso'] - featured_df['away_team_iso'])
    
    # Discipline gap
    featured_df['discipline_gap'] = abs(featured_df['home_team_bb_pct'] - featured_df['away_team_bb_pct'])
    
    # Ballpark factors (simplified)
    featured_df['ballpark_run_factor'] = featured_df.get('ballpark_run_factor', 1.0).fillna(1.0)
    featured_df['ballpark_hr_factor'] = featured_df.get('ballpark_hr_factor', 1.0).fillna(1.0)
    featured_df['ballpark_offensive_factor'] = (featured_df['ballpark_run_factor'] + featured_df['ballpark_hr_factor']) / 2
    
    # Weather factors
    featured_df['temp_factor'] = (featured_df['temperature'] - 70) / 100  # Normalize around 70F
    featured_df['wind_factor'] = featured_df['wind_speed'] / 15  # Normalize around 15mph
    featured_df['humidity_factor'] = 1.0 + (featured_df['humidity'] - 50) * 0.001
    
    # Total team games
    featured_df['total_team_games'] = featured_df['home_team_games_l30'] + featured_df['away_team_games_l30']
    
    # Pitching vs offense
    featured_df['pitching_vs_offense'] = featured_df['combined_k_rate'] - featured_df['combined_offense_rpg']
    
    # Expected total (market total or estimate)
    featured_df['expected_total'] = featured_df.get('market_total', 8.5).fillna(8.5)
    
    print(f"✅ Feature engineering completed - {len(featured_df.columns)} features")
    
    return featured_df


def predict_with_adaptive_model():
    print("🎯 ADAPTIVE MODEL PREDICTION WITH EXTRACTED FEATURES")
    print("=" * 80)
    
    # Load adaptive model
    adaptive_path = "s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib"
    adaptive_model = joblib.load(adaptive_path)
    
    expected_features = list(adaptive_model['model'].feature_names_in_)
    print(f"✅ Adaptive model expects {len(expected_features)} features")
    
    # Extract features
    featured_df = extract_features_only()
    
    if featured_df is None:
        print("❌ Failed to extract features")
        return None
        
    actual_features = list(featured_df.columns)
    print(f"✅ Extracted {len(actual_features)} features")
    
    # Check which expected features are missing
    missing = []
    present = []
    for feat in expected_features:
        if feat in actual_features:
            present.append(feat)
        else:
            missing.append(feat)
    
    print(f"\n✅ PRESENT features: {len(present)}/{len(expected_features)}")
    if missing:
        print(f"❌ MISSING features ({len(missing)}):")
        for feat in missing:
            print(f"  - {feat}")
            
        # Fill missing features with defaults
        print(f"\n🔧 Filling missing features with defaults...")
        for feat in missing:
            if 'era' in feat.lower():
                featured_df[feat] = 4.20
            elif 'whip' in feat.lower():
                featured_df[feat] = 1.25
            elif 'bb_pct' in feat.lower():
                featured_df[feat] = 0.085
            elif 'k_pct' in feat.lower():
                featured_df[feat] = 0.220
            elif 'woba' in feat.lower():
                featured_df[feat] = 0.315
            elif 'wrcplus' in feat.lower():
                featured_df[feat] = 100
            elif 'iso' in feat.lower():
                featured_df[feat] = 0.150
            elif 'babip' in feat.lower():
                featured_df[feat] = 0.300
            else:
                featured_df[feat] = 0.0
                
    # Extract features in correct order
    X_adaptive = featured_df[expected_features]
    
    # Ensure no NaN values
    if X_adaptive.isnull().sum().sum() > 0:
        print("⚠️ Found NaN values, filling with zeros...")
        X_adaptive = X_adaptive.fillna(0)
    
    print(f"\n🎯 Making predictions for {len(X_adaptive)} games...")
    predictions = adaptive_model['model'].predict(X_adaptive)
    
    print(f"\n🏆 ADAPTIVE MODEL PREDICTIONS:")
    print("=" * 60)
    for i, pred in enumerate(predictions):
        home_team = featured_df.iloc[i].get('home_team', f'Home{i+1}')
        away_team = featured_df.iloc[i].get('away_team', f'Away{i+1}')
        print(f"  Game {i+1:2d}: {away_team:18s} @ {home_team:18s} = {pred:5.2f}")
    
    print(f"\n📊 Prediction Statistics:")
    print(f"  Average: {predictions.mean():.2f}")
    print(f"  Range:   {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  StdDev:  {predictions.std():.2f}")
    
    return predictions, featured_df

if __name__ == "__main__":
    predictions, featured_df = predict_with_adaptive_model()
