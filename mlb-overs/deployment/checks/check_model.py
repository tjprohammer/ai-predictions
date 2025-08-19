#!/usr/bin/env python3
import joblib

# Check enhanced model details
model_data = joblib.load('models/enhanced_mlb_predictor.joblib')

print("Enhanced Model Details:")
print(f"Model type: {model_data.get('model_type', 'Unknown')}")
print(f"Training RMSE: {model_data.get('train_rmse', 'N/A')}")
print(f"Test RMSE: {model_data.get('test_rmse', 'N/A')}")
print(f"Test R²: {model_data.get('test_r2', 'N/A')}")

feature_names = model_data.get('feature_names', [])
print(f"\nModel expects {len(feature_names)} features:")
for i, feature in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feature}")

print(f"\nTraining info:")
print(f"Training date: {model_data.get('training_date', 'N/A')}")
print(f"Training games: {model_data.get('training_games', 'N/A')}")
print(f"Test games: {model_data.get('test_games', 'N/A')}")
