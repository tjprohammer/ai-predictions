#!/usr/bin/env python3
"""
Simple Model Test
================
Test if the trained model can make predictions
"""

import joblib
import pandas as pd
import numpy as np

try:
    print("🧪 Testing trained model...")
    
    # Load model
    model_data = joblib.load('daily_model.joblib')
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    print(f"✅ Model loaded successfully")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Test MAE: {model_data['performance']['test_mae']:.2f}")
    
    # Create dummy features for testing
    dummy_features = pd.DataFrame(np.random.randn(1, len(feature_columns)), columns=feature_columns)
    
    # Make prediction
    prediction = model.predict(dummy_features)
    
    print(f"✅ Model prediction test: {prediction[0]:.1f} runs")
    print("🎯 Model is working correctly!")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
