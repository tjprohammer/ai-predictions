import joblib
import os
import numpy as np

# Check what models we have available
models_dirs = ['models', 'mlb-overs/models']
for models_dir in models_dirs:
    if os.path.exists(models_dir):
        print(f"Available model files in {models_dir}:")
        for f in os.listdir(models_dir):
            if f.endswith('.joblib'):
                path = os.path.join(models_dir, f)
                size = os.path.getsize(path) / (1024*1024)  # MB
                print(f"  {f} ({size:.1f} MB)")
        print()

# Load and inspect the learning model
learning_model_path = 'mlb-overs/models/legitimate_model_latest.joblib'
if os.path.exists(learning_model_path):
    print(f"\n=== Loading {learning_model_path} ===")
    try:
        model_data = joblib.load(learning_model_path)
        print(f"Model data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("Model components:")
            for key, value in model_data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'n_features_in_'):
                    print(f"    Features: {value.n_features_in_}")
                elif hasattr(value, 'feature_names_in_'):
                    print(f"    Feature names: {len(value.feature_names_in_)}")
        else:
            print(f"Model object: {type(model_data)}")
            if hasattr(model_data, 'n_features_in_'):
                print(f"Features: {model_data.n_features_in_}")
                
    except Exception as e:
        print(f"Error loading model: {e}")

# Also check the adaptive model
adaptive_model_path = 'mlb-overs/models/adaptive_learning_model.joblib'
if os.path.exists(adaptive_model_path):
    print(f"\n=== Loading {adaptive_model_path} ===")
    try:
        model_data = joblib.load(adaptive_model_path)
        print(f"Model data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("Model components:")
            for key, value in model_data.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'n_features_in_'):
                    print(f"    Features: {value.n_features_in_}")
        else:
            print(f"Model object: {type(model_data)}")
            if hasattr(model_data, 'n_features_in_'):
                print(f"Features: {model_data.n_features_in_}")
                
    except Exception as e:
        print(f"Error loading adaptive model: {e}")
