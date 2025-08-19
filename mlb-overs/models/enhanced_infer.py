#!/usr/bin/env python3
"""
Enhanced MLB Predictions Inference
Uses enhanced historical data and enhanced ML model for accurate predictions
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from datetime import datetime
from sqlalchemy import create_engine, text

def sigmoid(x, alpha=1.0):
    """Sigmoid function for probability calibration"""
    return 1.0 / (1.0 + np.exp(-alpha * x))

def load_enhanced_model(model_path):
    """Load enhanced model with proper error handling"""
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict):
        # Enhanced model format
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        print(f"✅ Loaded enhanced ML model: {model_path}")
        print(f"   Model type: {model_data.get('model_type', 'Unknown')}")
        print(f"   Training RMSE: {model_data.get('train_rmse', 'N/A')}")
        print(f"   Test RMSE: {model_data.get('test_rmse', 'N/A')}")
        print(f"   Test R²: {model_data.get('test_r2', 'N/A')}")
        print(f"   Expected features: {len(feature_names)}")
        return model, scaler, feature_names
    else:
        # Legacy model format
        print(f"✅ Loaded legacy ML model: {model_path}")
        return model_data, None, []

def main():
    parser = argparse.ArgumentParser(description="Enhanced MLB Predictions Inference")
    parser.add_argument("data", help="Input parquet file with game features")
    parser.add_argument("out", help="Output parquet file for predictions")
    parser.add_argument("--model-path", default="models/enhanced_mlb_predictor.joblib", help="Path to model file")
    parser.add_argument("--database-url", help="Database connection URL")
    args = parser.parse_args()

    # Load enhanced model
    try:
        model, scaler, feature_names = load_enhanced_model(args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load input data
    try:
        df = pd.read_parquet(args.data)
        print(f"📊 Loaded {len(df)} games from {args.data}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    if df.empty:
        print("❌ No games found in input data")
        return

    # Filter games with betting lines
    original_count = len(df)
    df = df[df['k_close'].notna()]
    dropped_count = original_count - len(df)
    
    if dropped_count > 0:
        print(f"[infer] dropped {dropped_count} games without k_close (no sportsbook total)")
    
    if df.empty:
        print("❌ No games with betting lines found")
        # Create empty output
        empty_df = pd.DataFrame(columns=["game_id", "date", "home_team", "away_team", "k_close", "y_pred", "edge", "p_over", "p_under", "conf"])
        empty_df.to_parquet(args.out, index=False)
        return

    predictions = []
    
    for idx, row in df.iterrows():
        try:
            game_id = row.get('game_id', f'game_{idx}')
            home_team = row.get('home_team', 'HOME')
            away_team = row.get('away_team', 'AWAY')
            k_close = row.get('k_close', 8.5)
            game_date = row.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Prepare features for prediction
            if feature_names:
                # Use enhanced model with specified features
                feature_data = []
                for feature in feature_names:
                    if feature in row:
                        feature_data.append(row[feature])
                    else:
                        # Use default value for missing features
                        feature_data.append(0.0)
                
                feature_array = np.array(feature_data).reshape(1, -1)
            else:
                # Legacy model - use all numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in ['game_id', 'k_close']]
                feature_array = row[feature_cols].values.reshape(1, -1)
            
            # Apply scaling if available
            if scaler is not None:
                feature_array = scaler.transform(feature_array)
            
            # Make prediction
            y_pred = model.predict(feature_array)[0]
            
            # Calculate edge and probabilities
            edge = y_pred - k_close
            p_over = sigmoid(edge, alpha=1.1)
            p_under = 1.0 - p_over
            conf = max(p_over, p_under)
            
            prediction = {
                "game_id": game_id,
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "matchup": f"{away_team} @ {home_team}",
                "k_close": k_close,
                "y_pred": y_pred,
                "edge": edge,
                "p_over": p_over,
                "p_under": p_under,
                "conf": conf,
                "recommendation": "OVER" if edge > 0.5 else "UNDER" if edge < -0.5 else "NO_BET"
            }
            
            predictions.append(prediction)
            
        except Exception as e:
            print(f"⚠️  Error processing game {game_id}: {e}")
            continue
    
    if not predictions:
        print("❌ No successful predictions generated")
        return
    
    # Create output dataframe
    output_df = pd.DataFrame(predictions)
    
    # Save predictions
    output_df.to_parquet(args.out, index=False)
    print(f"✅ Saved {len(output_df)} predictions to {args.out}")
    
    # Print summary
    strong_bets = output_df[output_df['recommendation'] != 'NO_BET']
    if len(strong_bets) > 0:
        print(f"🎯 Strong recommendations: {len(strong_bets)}")
        for _, bet in strong_bets.iterrows():
            print(f"   {bet['matchup']}: {bet['recommendation']} (edge: {bet['edge']:.2f})")
    else:
        print("📊 No strong betting recommendations found")

if __name__ == "__main__":
    main()
