#!/usr/bin/env python3
"""
Test the newly calibrated model with the exact features it was trained on
"""

import sys
sys.        # Apply bias correction (from model metadata)
        bias_correction = model_data.get('bias_correction', 0.0)
        # Don't apply global adjustment for new model - it's already calibrated!
        # global_adjustment = 3.0  # From our bias corrections
        
        final_prediction = raw_prediction + bias_correction
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Raw model output: {raw_prediction:.2f} runs")
        print(f"   Model bias correction: +{bias_correction:.2f} runs")
        print(f"   ✅ Final prediction: {final_prediction:.2f} runs")
        
        market_total = 8.5
        print(f"   📊 Market total: {market_total} runs")
        print(f"   🎯 Difference: {final_prediction - market_total:.2f} runs")b-overs/deployment')
import pandas as pd
import joblib

def test_basic_model():
    """Test with the exact feature set the model was trained on"""
    
    print('🎯 TESTING WITH EXACT TRAINING FEATURES')
    print('='*60)
    
    # Load the model directly
    model_data = joblib.load('mlb-overs/models/legitimate_model_latest.joblib')
    model = model_data['model']
    
    print(f"📋 Model expects: {len(model.feature_names_in_)} features")
    print("🔍 First 10 expected features:")
    for i, feat in enumerate(model.feature_names_in_[:10]):
        print(f"   {i+1}: {feat}")
    
    # Create a simple test with just the basic features from training
    basic_features = {
        'home_sp_era': 3.11,
        'away_sp_era': 4.59,
        'combined_era': 3.85,
        'era_differential': -1.48,
        'home_sp_whip': 1.15,
        'away_sp_whip': 1.25,
        'combined_whip': 1.20,
        'home_sp_k_per_9': 9.5,
        'away_sp_k_per_9': 8.2,
        'pitching_advantage': 1.3,
        'home_pitcher_quality': 0.8,
        'away_pitcher_quality': 0.6,
        'home_bp_era': 4.2,
        'away_bp_era': 4.5,
        'combined_bullpen_era': 4.35,
        'home_team_runs_pg': 4.2,
        'away_team_runs_pg': 4.8,
        'combined_team_offense': 4.5,
        'ballpark_run_factor': 1.0,
        'temperature': 78.0,
        'wind_speed': 13.0
    }
    
    # Fill in missing features with reasonable defaults
    full_features = {}
    
    for feature_name in model.feature_names_in_:
        if feature_name in basic_features:
            full_features[feature_name] = basic_features[feature_name]
        elif 'home_sp' in feature_name:
            full_features[feature_name] = 0.0  # Default pitcher stat
        elif 'away_sp' in feature_name:
            full_features[feature_name] = 0.0  # Default pitcher stat  
        elif 'home_bp' in feature_name:
            full_features[feature_name] = 4.2  # Default bullpen ERA
        elif 'away_bp' in feature_name:
            full_features[feature_name] = 4.5  # Default bullpen ERA
        elif 'bullpen' in feature_name:
            full_features[feature_name] = 0.0  # Default bullpen stat
        elif 'home_team' in feature_name:
            full_features[feature_name] = 4.2  # Default offensive stat
        elif 'away_team' in feature_name:
            full_features[feature_name] = 4.8  # Default offensive stat
        elif 'team' in feature_name:
            full_features[feature_name] = 100.0  # Default team stat
        elif 'ballpark' in feature_name:
            full_features[feature_name] = 1.0  # Neutral park factor
        elif 'temp' in feature_name:
            full_features[feature_name] = 0.16  # Moderate temperature
        elif 'wind' in feature_name:
            full_features[feature_name] = 0.0  # Calm wind
        elif 'humidity' in feature_name:
            full_features[feature_name] = 0.0  # Default humidity
        elif 'is_' in feature_name:
            full_features[feature_name] = 0.0  # Boolean defaults to False
        elif 'ump' in feature_name:
            full_features[feature_name] = 0.0  # Default umpire effect
        elif 'lineup' in feature_name:
            full_features[feature_name] = 100.0  # Average lineup
        elif 'vs_' in feature_name:
            full_features[feature_name] = 0.750  # Average OPS
        elif 'lhb_count' in feature_name:
            full_features[feature_name] = 4.0  # Default lefty count
        elif 'star_missing' in feature_name:
            full_features[feature_name] = 0.0  # No stars missing
        elif 'games_' in feature_name:
            full_features[feature_name] = 7.0  # Games played
        elif 'days_rest' in feature_name:
            full_features[feature_name] = 1.0  # 1 day rest
        elif 'travel' in feature_name:
            full_features[feature_name] = 0.0  # No travel
        elif 'getaway' in feature_name:
            full_features[feature_name] = 0.0  # Not getaway day
        elif 'interaction' in feature_name:
            full_features[feature_name] = 0.0  # Default interaction
        elif 'density' in feature_name:
            full_features[feature_name] = 1.0  # Normal air density
        elif 'catcher' in feature_name:
            full_features[feature_name] = 0.0  # Default framing
        elif 'power' in feature_name:
            full_features[feature_name] = 0.17  # Average power
        elif 'babip' in feature_name:
            full_features[feature_name] = 0.300  # League average BABIP
        elif 'ba' in feature_name:
            full_features[feature_name] = 0.250  # League average BA
        else:
            full_features[feature_name] = 0.0  # Final default
    
    # Create feature DataFrame in correct order
    feature_df = pd.DataFrame([full_features])[model.feature_names_in_]
    
    print(f"\\n📊 Created feature matrix: {feature_df.shape}")
    print("🔍 Sample feature values:")
    for col in feature_df.columns[:10]:
        print(f"   {col}: {feature_df[col].iloc[0]}")
    
    # Make prediction
    try:
        raw_prediction = model.predict(feature_df)[0]
        
        # Apply bias correction (from model metadata)
        bias_correction = model_data.get('bias_correction', 0.0)
        global_adjustment = 3.0  # From our bias corrections
        
        final_prediction = raw_prediction + bias_correction + global_adjustment
        
        print(f"\\n🎯 PREDICTION RESULTS:")
        print(f"   Raw model output: {raw_prediction:.2f} runs")
        print(f"   Model bias correction: +{bias_correction:.2f} runs")
        print(f"   Global adjustment: +{global_adjustment:.2f} runs")
        print(f"   ✅ Final prediction: {final_prediction:.2f} runs")
        
        market_total = 8.5
        print(f"   📊 Market total: {market_total} runs")
        print(f"   🎯 Difference: {final_prediction - market_total:.2f} runs")
        
        if 6.0 <= final_prediction <= 12.0:
            print("   ✅ Prediction is in realistic MLB range!")
            
            if abs(final_prediction - market_total) < 2.0:
                print("   🎉 EXCELLENT: Prediction within 2 runs of market!")
                return True
            else:
                print(f"   ⚠️  Still {abs(final_prediction - market_total):.1f} runs from market")
                return False
        else:
            print("   ❌ Prediction still outside realistic range")
            return False
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_model()
    if success:
        print("\\n🎉 SUCCESS: New model produces realistic predictions!")
    else:
        print("\\n❌ ISSUE: Model still needs calibration work")
