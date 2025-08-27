#!/usr/bin/env python3
"""
Direct adaptive model prediction using feature engineering from enhanced predictor
"""

import sys
import joblib
import pandas as pd
import numpy as np
sys.path.append('s:/Projects/AI_Predictions/mlb-overs/deployment')

from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def predict_with_adaptive_model():
    print("🔍 DIRECT ADAPTIVE MODEL PREDICTION")
    print("=" * 60)
    
    # Load adaptive model
    adaptive_path = "s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib"
    adaptive_model = joblib.load(adaptive_path)
    
    expected_features = list(adaptive_model['model'].feature_names_in_)
    print(f"✅ Adaptive model expects {len(expected_features)} features")
    
    # Get features from enhanced predictor (feature engineering only)
    try:
        predictor = EnhancedBullpenPredictor()
        pred_df, featured_df, X = predictor.predict_today_games('2025-08-23')
        
        if featured_df is not None:
            actual_features = list(featured_df.columns)
            print(f"✅ Enhanced predictor created {len(actual_features)} features")
            
            # Check which expected features are missing
            missing = []
            present = []
            for feat in expected_features:
                if feat in actual_features:
                    present.append(feat)
                else:
                    missing.append(feat)
            
            print(f"\n✅ PRESENT features: {len(present)}/{len(expected_features)}")
            print(f"❌ MISSING features: {missing}")
                
            if len(missing) == 0:
                print(f"\n🎉 ALL FEATURES PRESENT! Making adaptive predictions...")
                # Extract only the features the adaptive model needs in correct order
                X_adaptive = featured_df[expected_features]
                
                # Ensure no NaN values
                if X_adaptive.isnull().sum().sum() > 0:
                    print("⚠️ Found NaN values, filling with defaults...")
                    X_adaptive = X_adaptive.fillna(0)
                
                predictions = adaptive_model['model'].predict(X_adaptive)
                print(f"🎯 Adaptive model predictions:")
                for i, pred in enumerate(predictions):
                    game_id = featured_df.iloc[i]['game_id'] if 'game_id' in featured_df.columns else f"Game {i+1}"
                    home_team = featured_df.iloc[i]['home_team'] if 'home_team' in featured_df.columns else 'Home'
                    away_team = featured_df.iloc[i]['away_team'] if 'away_team' in featured_df.columns else 'Away'
                    print(f"  {game_id}: {away_team} @ {home_team} = {pred:.2f}")
                
                return predictions, featured_df
            else:
                print(f"\n❌ Cannot predict - {len(missing)} features missing")
                return None, featured_df
                
        else:
            print("❌ Enhanced predictor returned None for featured_df")
            return None, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    predictions, featured_df = predict_with_adaptive_model()
