"""
Inspect the actual model to see what's wrong with it
"""

import joblib
import numpy as np
from pathlib import Path

def inspect_model():
    """Inspect the actual model that's producing low predictions"""
    
    model_path = Path("s:/Projects/AI_Predictions/mlb-overs/models/legitimate_model_latest.joblib")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    print("🔍 INSPECTING MODEL")
    print("=" * 50)
    
    # Load the model bundle
    bundle = joblib.load(model_path)
    
    print(f"Model bundle type: {type(bundle)}")
    print(f"Bundle keys: {list(bundle.keys()) if isinstance(bundle, dict) else 'Not a dict'}")
    
    if isinstance(bundle, dict):
        if 'model' in bundle:
            model = bundle['model']
            print(f"Model type: {type(model)}")
            
            # Check if it's a sklearn model
            if hasattr(model, 'get_params'):
                print(f"Model parameters: {model.get_params()}")
            
            if hasattr(model, 'feature_importances_'):
                print(f"Number of features: {len(model.feature_importances_)}")
                print(f"Top 5 feature importances: {sorted(model.feature_importances_, reverse=True)[:5]}")
        
        if 'features' in bundle:
            features = bundle['features']
            print(f"Features: {features[:10]}...")  # First 10 features
            print(f"Total features: {len(features)}")
        
        if 'scaler' in bundle:
            scaler = bundle['scaler']
            print(f"Scaler type: {type(scaler)}")
            if hasattr(scaler, 'mean_'):
                print(f"Scaler mean (first 5): {scaler.mean_[:5]}")
                print(f"Scaler scale (first 5): {scaler.scale_[:5]}")
        
        if 'target_scaler' in bundle:
            target_scaler = bundle['target_scaler']
            print(f"Target scaler type: {type(target_scaler)}")
            if hasattr(target_scaler, 'mean_'):
                print(f"Target scaler mean: {target_scaler.mean_}")
                print(f"Target scaler scale: {target_scaler.scale_}")
                
                # THIS IS THE KEY CHECK!
                print(f"\n🎯 TARGET SCALING ANALYSIS:")
                print("-" * 30)
                print(f"Target mean: {target_scaler.mean_}")
                print(f"Target scale: {target_scaler.scale_}")
                
                # If the target was scaled incorrectly, this would cause low predictions
                if target_scaler.mean_ < 5.0:
                    print(f"🚨 PROBLEM: Target mean is only {target_scaler.mean_:.2f}")
                    print(f"   This suggests the model was trained on targets that were too low!")
                    print(f"   Normal MLB game totals should average around 8-9 runs")
                
                if target_scaler.scale_ < 1.0:
                    print(f"🚨 PROBLEM: Target scale is only {target_scaler.scale_:.2f}")
                    print(f"   This suggests very low variance in training targets")
    
    # Test a simple prediction
    print(f"\n🧪 TEST PREDICTION:")
    print("-" * 30)
    
    # Create dummy features (zeros)
    if isinstance(bundle, dict) and 'features' in bundle:
        n_features = len(bundle['features'])
        dummy_features = np.zeros((1, n_features))
        
        try:
            if 'scaler' in bundle and 'target_scaler' in bundle:
                # Scale features
                scaled_features = bundle['scaler'].transform(dummy_features)
                # Get prediction
                scaled_pred = bundle['model'].predict(scaled_features)[0]
                # Unscale prediction
                final_pred = bundle['target_scaler'].inverse_transform([[scaled_pred]])[0][0]
                
                print(f"Dummy prediction (all zeros): {final_pred:.2f}")
                
                if final_pred < 4.0:
                    print(f"🚨 PROBLEM: Even with dummy features, prediction is only {final_pred:.2f}")
                    print(f"   This confirms the model is fundamentally broken")
            
        except Exception as e:
            print(f"Error testing prediction: {e}")

if __name__ == "__main__":
    inspect_model()
