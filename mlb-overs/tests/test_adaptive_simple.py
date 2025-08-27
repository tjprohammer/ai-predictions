#!/usr/bin/env python3
"""
Test adaptive model with simple feature set
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text

def test_adaptive_simple():
    print("🔍 TESTING ADAPTIVE MODEL WITH SIMPLE FEATURES")
    print("=" * 60)
    
    # Connect to database
    engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))
    
    # Get today's games with just the basic features the adaptive model claims to need
    query = text("""
    SELECT 
        game_id,
        home_sp_whip,
        away_sp_whip,
        home_team_avg,
        away_team_avg,
        home_team_woba,
        away_team_woba,
        ballpark_run_factor,
        ballpark_hr_factor,
        temperature,
        wind_speed,
        humidity
    FROM enhanced_games 
    WHERE date = '2025-08-23'
    ORDER BY game_id
    """)
    
    simple_df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(simple_df)} games with simple features")
    print("Columns:", simple_df.columns.tolist())
    
    # Check for nulls
    print("\nNull counts:")
    for col in simple_df.columns:
        null_count = simple_df[col].isnull().sum()
        if null_count > 0:
            print(f"  {col}: {null_count} nulls")
    
    # Fill any nulls with reasonable defaults
    simple_df = simple_df.fillna({
        'home_sp_whip': 1.25,
        'away_sp_whip': 1.25,
        'home_team_avg': 0.250,
        'away_team_avg': 0.250,
        'home_team_woba': 0.320,
        'away_team_woba': 0.320,
        'ballpark_run_factor': 1.0,
        'ballpark_hr_factor': 1.0,
        'temperature': 75,
        'wind_speed': 10,
        'humidity': 50
    })
    
    print(f"\n✅ After filling nulls:")
    for col in simple_df.select_dtypes(include=[np.number]).columns:
        if col != 'game_id':
            print(f"  {col}: {simple_df[col].mean():.3f} ± {simple_df[col].std():.3f}")
    
    # Load adaptive model
    adaptive_path = "s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib"
    adaptive_model = joblib.load(adaptive_path)
    model = adaptive_model['model']
    feature_columns = adaptive_model['feature_columns']  # Should be the 11 basic features
    
    print(f"\n✅ Adaptive model loaded")
    print(f"Feature columns: {feature_columns}")
    
    # Extract only the required features
    X = simple_df[feature_columns]
    print(f"\n✅ Feature matrix shape: {X.shape}")
    
    # Try prediction
    try:
        predictions = model.predict(X)
        print(f"\n🎉 SUCCESS! Generated {len(predictions)} predictions")
        print(f"Predictions: {predictions}")
        print(f"Range: {predictions.min():.2f} - {predictions.max():.2f}")
        print(f"Mean: {predictions.mean():.2f}")
        
        # Create results dataframe
        results_df = simple_df[['game_id']].copy()
        results_df['predicted_total'] = predictions
        print(f"\nResults:")
        print(results_df)
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adaptive_simple()
