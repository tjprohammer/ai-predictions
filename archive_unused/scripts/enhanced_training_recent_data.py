#!/usr/bin/env python3
"""
Enhanced Training Pipeline
=========================

Train the model on more recent data with better features
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def enhanced_training_pipeline():
    """Train model on last 30-60 days of data with all features"""
    print("🚀 Enhanced Training Pipeline - Training on Recent Data")
    print("=" * 60)
    
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb', echo=False)
    
    # Get comprehensive recent data
    with engine.begin() as conn:
        query = """
        SELECT 
            date, game_id, home_team, away_team, home_score, away_score, total_runs,
            venue_name, temperature, wind_speed, wind_direction, weather_condition,
            home_sp_id, away_sp_id, home_sp_er, away_sp_er, home_sp_ip, away_sp_ip,
            home_sp_k, away_sp_k, home_sp_bb, away_sp_bb, home_sp_h, away_sp_h,
            home_team_hits, away_team_hits, home_team_runs, away_team_runs,
            home_team_rbi, away_team_rbi, home_team_lob, away_team_lob,
            venue_id, day_night, game_type
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL 
            AND home_score IS NOT NULL 
            AND away_score IS NOT NULL
            AND date >= CURRENT_DATE - INTERVAL '60 days'
        ORDER BY date DESC
        """
        
        training_data = pd.read_sql(text(query), conn)
        
        if training_data.empty:
            print("❌ No training data found")
            return
        
        print(f"✅ Loaded {len(training_data)} games from last 60 days")
        
        # Get team stats for feature engineering
        team_stats_query = """
        SELECT team, date, runs_pg, ba, woba, bb_pct, k_pct, babip, iso
        FROM teams_offense_daily 
        WHERE date >= CURRENT_DATE - INTERVAL '60 days'
            AND runs_pg IS NOT NULL
        ORDER BY team, date DESC
        """
        
        team_stats = pd.read_sql(text(team_stats_query), conn)
        
        # Get latest stats per team
        latest_team_stats = team_stats.groupby('team').first().reset_index()
        print(f"✅ Loaded team stats for {len(latest_team_stats)} teams")
    
    # Feature engineering
    print("🔧 Engineering features...")
    
    # Map team names to abbreviations for joining with team stats
    team_mapping = {
        'Arizona Diamondbacks': 'AZ', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
        'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
        'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
        'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
        'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
        'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
        'New York Yankees': 'NYY', 'Oakland Athletics': 'ATH', 'Philadelphia Phillies': 'PHI',
        'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
        'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
        'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH'
    }
    
    # Add team abbreviations
    training_data['home_team_abbr'] = training_data['home_team'].map(team_mapping)
    training_data['away_team_abbr'] = training_data['away_team'].map(team_mapping)
    
    # Merge with team stats
    home_team_stats = latest_team_stats.copy()
    home_team_stats.columns = [f'home_{col}' if col != 'team' else 'home_team_abbr' for col in home_team_stats.columns]
    
    away_team_stats = latest_team_stats.copy()
    away_team_stats.columns = [f'away_{col}' if col != 'team' else 'away_team_abbr' for col in away_team_stats.columns]
    
    training_data = training_data.merge(home_team_stats, on='home_team_abbr', how='left')
    training_data = training_data.merge(away_team_stats, on='away_team_abbr', how='left')
    
    # Fill missing values with defaults
    training_data = training_data.fillna({
        'temperature': 75, 'wind_speed': 5,
        'home_sp_er': 0, 'away_sp_er': 0, 'home_sp_ip': 1, 'away_sp_ip': 1,
        'home_sp_k': 0, 'away_sp_k': 0, 'home_sp_bb': 0, 'away_sp_bb': 0,
        'home_sp_h': 0, 'away_sp_h': 0,
        'home_runs_pg': 4.5, 'away_runs_pg': 4.5,
        'home_ba': 0.250, 'away_ba': 0.250,
        'home_woba': 0.310, 'away_woba': 0.310,
        'home_bb_pct': 8.0, 'away_bb_pct': 8.0,
        'home_k_pct': 22.0, 'away_k_pct': 22.0,
        'home_babip': 0.300, 'away_babip': 0.300,
        'home_iso': 0.150, 'away_iso': 0.150
    })
    
    # Create feature matrix
    feature_df = pd.DataFrame()
    
    # Team offensive features
    feature_df['home_runs_pg'] = training_data['home_runs_pg']
    feature_df['away_runs_pg'] = training_data['away_runs_pg']
    feature_df['home_ba'] = training_data['home_ba']
    feature_df['away_ba'] = training_data['away_ba']
    feature_df['home_woba'] = training_data['home_woba']
    feature_df['away_woba'] = training_data['away_woba']
    feature_df['home_bb_pct'] = training_data['home_bb_pct']
    feature_df['away_bb_pct'] = training_data['away_bb_pct']
    feature_df['home_k_pct'] = training_data['home_k_pct']
    feature_df['away_k_pct'] = training_data['away_k_pct']
    feature_df['home_babip'] = training_data['home_babip']
    feature_df['away_babip'] = training_data['away_babip']
    feature_df['home_iso'] = training_data['home_iso']
    feature_df['away_iso'] = training_data['away_iso']
    
    # Weather features
    feature_df['temperature'] = pd.to_numeric(training_data['temperature'], errors='coerce').fillna(75)
    feature_df['wind_speed'] = pd.to_numeric(training_data['wind_speed'], errors='coerce').fillna(5)
    
    # Wind direction encoding
    wind_direction_mapping = {
        'Out To CF': 3, 'Out To LF': 2, 'Out To RF': 2,
        'In From CF': -2, 'In From LF': -1, 'In From RF': -1,
        'L To R': 0, 'R To L': 0, 'Varies': 0
    }
    feature_df['wind_direction_effect'] = training_data['wind_direction'].map(wind_direction_mapping).fillna(0)
    
    # Pitcher features
    feature_df['home_sp_era'] = np.where(
        training_data['home_sp_ip'] > 0,
        (training_data['home_sp_er'] * 9) / training_data['home_sp_ip'],
        4.50
    )
    feature_df['away_sp_era'] = np.where(
        training_data['away_sp_ip'] > 0,
        (training_data['away_sp_er'] * 9) / training_data['away_sp_ip'],
        4.50
    )
    
    feature_df['home_sp_whip'] = np.where(
        training_data['home_sp_ip'] > 0,
        (training_data['home_sp_h'] + training_data['home_sp_bb']) / training_data['home_sp_ip'],
        1.30
    )
    feature_df['away_sp_whip'] = np.where(
        training_data['away_sp_ip'] > 0,
        (training_data['away_sp_h'] + training_data['away_sp_bb']) / training_data['away_sp_ip'],
        1.30
    )
    
    feature_df['home_sp_k_per_9'] = np.where(
        training_data['home_sp_ip'] > 0,
        (training_data['home_sp_k'] * 9) / training_data['home_sp_ip'],
        8.0
    )
    feature_df['away_sp_k_per_9'] = np.where(
        training_data['away_sp_ip'] > 0,
        (training_data['away_sp_k'] * 9) / training_data['away_sp_ip'],
        8.0
    )
    
    # Venue effects
    venue_effects = {
        'Coors Field': 1.2, 'Fenway Park': 1.1, 'Yankee Stadium': 1.1,
        'Minute Maid Park': 1.05, 'Citizens Bank Park': 1.05,
        'Petco Park': 0.9, 'Marlins Park': 0.95, 'Oakland Coliseum': 0.95
    }
    feature_df['venue_effect'] = training_data['venue_name'].map(venue_effects).fillna(1.0)
    
    # Day/night effect
    feature_df['is_night_game'] = (training_data['day_night'] == 'N').astype(int)
    
    # Combined features
    feature_df['total_offensive_power'] = (
        feature_df['home_runs_pg'] + feature_df['away_runs_pg'] +
        (feature_df['home_woba'] + feature_df['away_woba']) * 10
    ) / 3
    
    feature_df['pitcher_quality'] = (feature_df['home_sp_era'] + feature_df['away_sp_era']) / 2
    
    # Weather impact
    feature_df['weather_boost'] = (
        (feature_df['temperature'] - 70) * 0.02 +  # Warmer weather boost
        feature_df['wind_direction_effect'] * 0.1    # Wind effect
    )
    
    target = training_data['total_runs']
    
    # Remove any rows with missing target
    valid_mask = ~target.isna()
    X = feature_df[valid_mask]
    y = target[valid_mask]
    
    print(f"✅ Feature matrix: {X.shape[0]} games, {X.shape[1]} features")
    print(f"📊 Feature columns: {list(X.columns)}")
    
    # Train-test split (chronological to avoid data leakage)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"📈 Training set: {len(X_train)} games")
    print(f"🧪 Test set: {len(X_test)} games")
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("🤖 Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print("\\n📊 MODEL PERFORMANCE")
    print("=" * 40)
    print(f"🎯 Training MAE: {train_mae:.2f} runs")
    print(f"🧪 Test MAE: {test_mae:.2f} runs")
    print(f"📈 Training R²: {train_r2:.3f}")
    print(f"🧪 Test R²: {test_r2:.3f}")
    print(f"🔄 Cross-validation MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\n🔍 TOP 10 FEATURE IMPORTANCE")
    print("-" * 40)
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model and feature columns
    model_data = {
        'model': model,
        'feature_columns': list(X.columns),
        'training_stats': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'training_games': len(X_train),
            'test_games': len(X_test),
            'feature_count': len(X.columns)
        }
    }
    
    joblib.dump(model_data, 'S:/Projects/AI_Predictions/enhanced_model_recent_data.joblib')
    print(f"\\n💾 Enhanced model saved to 'enhanced_model_recent_data.joblib'")
    
    return model_data

def main():
    enhanced_training_pipeline()

if __name__ == "__main__":
    main()
