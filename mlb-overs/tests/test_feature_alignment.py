#!/usr/bin/env python3
"""
Simple feature alignment test to fix the adaptive model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the deployment directory to Python path
sys.path.append('s:/Projects/AI_Predictions/mlb-overs/deployment')

from enhanced_bullpen_predictor import EnhancedBullpenPredictor
import joblib

def test_feature_alignment():
    print("🔍 TESTING FEATURE ALIGNMENT")
    print("=" * 60)
    
    # Load adaptive model to see what it expects
    model_path = Path("s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib")
    if not model_path.exists():
        print("❌ Adaptive model not found")
        return
        
    adaptive_model = joblib.load(model_path)
    expected_features = list(adaptive_model['model'].feature_names_in_)
    print(f"✅ Adaptive model expects {len(expected_features)} features")
    
    # Try to generate features with enhanced predictor
    try:
        predictor = EnhancedBullpenPredictor()
        print("✅ Enhanced predictor loaded")
        
        # Try to get today's games and features
        target_date = "2025-08-23"
        pred_df, featured_df, X = predictor.predict_today_games(target_date)
        
        if featured_df is not None:
            created_features = list(featured_df.columns)
            print(f"✅ Enhanced predictor created {len(created_features)} features")
            
            # Find missing features
            missing = set(expected_features) - set(created_features)
            extra = set(created_features) - set(expected_features)
            
            print(f"\n📊 FEATURE COMPARISON:")
            print(f"   Expected: {len(expected_features)}")
            print(f"   Created:  {len(created_features)}")
            print(f"   Missing:  {len(missing)}")
            print(f"   Extra:    {len(extra)}")
            
            if missing:
                print(f"\n❌ MISSING FEATURES ({len(missing)}):")
                for i, feat in enumerate(sorted(missing)):
                    print(f"   {i+1:2d}. {feat}")
                    if i >= 19:  # Show first 20
                        print(f"   ... and {len(missing)-20} more")
                        break
            
            # Show some key features that should exist
            key_features = ['home_sp_era', 'away_sp_era', 'home_sp_whip', 'away_sp_whip', 
                          'home_sp_k_per_9', 'away_sp_k_per_9', 'combined_k_rate']
            print(f"\n🔑 KEY FEATURE STATUS:")
            for feat in key_features:
                status = "✅" if feat in created_features else "❌"
                print(f"   {status} {feat}")
                
        else:
            print("❌ Enhanced predictor returned None for featured_df")
            
    except Exception as e:
        print(f"❌ Error with enhanced predictor: {e}")

if __name__ == "__main__":
    test_feature_alignment()
