#!/usr/bin/env python3
"""
Examine the learning model to see what's wrong with it
"""
import joblib
import os
import sys

def examine_learning_model():
    model_path = 'mlb-overs/models/legitimate_model_latest.joblib'
    
    try:
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print("\n📋 MODEL CONTENTS:")
        
        if isinstance(model, dict):
            for key, value in model.items():
                if key == 'model':
                    print(f"  {key}: {type(value)}")
                    if hasattr(value, 'feature_names_in_'):
                        print(f"    Features expected: {len(value.feature_names_in_)}")
                        print(f"    Sample features: {list(value.feature_names_in_)[:10]}")
                elif key == 'feature_columns':
                    print(f"  {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} features")
                    if hasattr(value, '__len__') and len(value) > 0:
                        print(f"    Sample: {list(value)[:10]}")
                elif key == 'feature_fill_values':
                    print(f"  {key}: {len(value) if hasattr(value, '__len__') else 'N/A'} defaults")
                    if hasattr(value, '__len__') and len(value) > 0:
                        sample_defaults = list(value.items())[:5]
                        print(f"    Sample defaults: {sample_defaults}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Model type: {type(model)}")
            
        # Check model performance if available
        if isinstance(model, dict) and 'performance' in model:
            perf = model['performance']
            print(f"\n📊 MODEL PERFORMANCE:")
            for key, value in perf.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    examine_learning_model()
