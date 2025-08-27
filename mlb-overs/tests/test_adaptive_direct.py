#!/usr/bin/env python3
"""
Test adaptive model directly with today's features
"""

import sys
import joblib
import pandas as pd
import numpy as np
sys.path.append('s:/Projects/AI_Predictions/mlb-overs/deployment')

from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def test_adaptive_model():
    print("🔍 TESTING ADAPTIVE MODEL DIRECTLY")
    print("=" * 60)
    
    # Load adaptive model
    adaptive_path = "s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib"
    adaptive_model = joblib.load(adaptive_path)
    
    expected_features = list(adaptive_model['model'].feature_names_in_)
    print(f"✅ Adaptive model expects {len(expected_features)} features:")
    for i, feat in enumerate(expected_features):
        print(f"  {i+1:2d}. {feat}")
    
    print("\n" + "=" * 60)
    
    # Get today's features
    try:
        predictor = EnhancedBullpenPredictor()
        pred_df, featured_df, X = predictor.predict_today_games('2025-08-23')
        
        if featured_df is not None:
            actual_features = list(featured_df.columns)
            print(f"✅ Enhanced predictor creates {len(actual_features)} features")
            
            # Check which expected features are missing
            missing = []
            present = []
            for feat in expected_features:
                if feat in actual_features:
                    present.append(feat)
                else:
                    missing.append(feat)
            
            print(f"\n✅ PRESENT features ({len(present)}):")
            for feat in present:
                print(f"  ✓ {feat}")
                
            print(f"\n❌ MISSING features ({len(missing)}):")
            for feat in missing:
                print(f"  - {feat}")
                
            if len(missing) == 0:
                print(f"\n🎉 ALL FEATURES PRESENT! Testing prediction...")
                # Extract only the features the adaptive model needs
                X_adaptive = featured_df[expected_features]
                pred = adaptive_model['model'].predict(X_adaptive)
                print(f"🎯 Adaptive model predictions: {pred}")
            else:
                print(f"\n❌ Cannot test prediction - {len(missing)} features missing")
                
        else:
            print("❌ Enhanced predictor returned None for featured_df")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adaptive_model()
