#!/usr/bin/env python3
"""
Check exactly what adaptive model needs vs what we're creating
"""

import sys
import joblib
sys.path.append('s:/Projects/AI_Predictions/mlb-overs/deployment')
sys.path.append('s:/Projects/AI_Predictions/mlb-overs/models')

from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def compare_features():
    print("🔍 FEATURE COMPARISON")
    print("=" * 60)
    
    # Load adaptive model to see what it actually expects
    adaptive_path = "s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib"
    adaptive_model = joblib.load(adaptive_path)
    
    # Get the actual feature names the model was trained on
    expected_features = list(adaptive_model['model'].feature_names_in_)
    print(f"✅ Adaptive model trained on {len(expected_features)} features:")
    for feat in expected_features[:20]:  # Show first 20
        print(f"  - {feat}")
    if len(expected_features) > 20:
        print(f"  ... and {len(expected_features) - 20} more")
    
    print("\n" + "=" * 60)
    
    # Try to get features from enhanced predictor
    try:
        predictor = EnhancedBullpenPredictor()
        pred_df, featured_df, X = predictor.predict_today_games('2025-08-23')
        
        if featured_df is not None:
            actual_features = list(featured_df.columns)
            print(f"✅ Enhanced predictor creates {len(actual_features)} features")
            
            # Find missing and extra features
            missing = set(expected_features) - set(actual_features)
            extra = set(actual_features) - set(expected_features)
            common = set(expected_features) & set(actual_features)
            
            print(f"\n✅ COMMON features ({len(common)}):")
            for feat in sorted(list(common)[:10]):  # Show first 10
                print(f"  ✓ {feat}")
            if len(common) > 10:
                print(f"  ... and {len(common) - 10} more")
                
            print(f"\n❌ MISSING features ({len(missing)}):")
            for feat in sorted(list(missing)[:20]):  # Show first 20
                print(f"  - {feat}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
                
        else:
            print("❌ Enhanced predictor returned None for featured_df")
            
    except Exception as e:
        print(f"❌ Error testing enhanced predictor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_features()
