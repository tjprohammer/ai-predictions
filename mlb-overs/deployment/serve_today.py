#!/usr/bin/env python3
"""
serve_today.py
==============
Score today's games with the latest deployed model bundle to ensure feature alignment.
This ensures predictions come from the same schema as training and include bias correction.

Usage:
    python serve_today.py --date 2025-08-17
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def main():
    ap = argparse.ArgumentParser(description="Score today's games with latest model")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--model-path", default="../models/legitimate_model_latest.joblib", 
                    help="Path to model bundle")
    args = ap.parse_args()

    # Load the latest model bundle
    try:
        bundle = joblib.load(args.model_path)
        print(f"✅ Loaded model bundle: {bundle.get('version', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load model bundle: {e}")
        return

    # Extract model components
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    feature_fill_values = bundle["feature_fill_values"]
    bias_correction = float(bundle.get("bias_correction", 0.0))
    
    print(f"📊 Model expects {len(feature_columns)} features")
    print(f"🎯 Bias correction: {bias_correction:+.3f} runs")

    # Connect to database
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # Fetch today's game features
    with engine.begin() as conn:
        raw = pd.read_sql(text("""
            SELECT lgf.* FROM legitimate_game_features lgf
            JOIN enhanced_games eg USING (game_id, "date")
            WHERE lgf."date" = :d
        """), conn, params={"d": args.date})

    if raw.empty:
        print(f"❌ No games to score for {args.date}")
        return

    print(f"🎲 Found {len(raw)} games to score")

    # Engineer features using the same pipeline as training
    predictor = EnhancedBullpenPredictor()
    try:
        features = predictor.engineer_features(raw.copy())
        print(f"✅ Engineered {len(features.columns)} features")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return

    # Align to serving features (same schema as training)
    try:
        X = predictor.align_serving_features(features, strict=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_columns)
        else:
            X = X.reindex(columns=feature_columns)
        print(f"✅ Aligned to {len(X.columns)} serving features")
    except Exception as e:
        print(f"⚠️  Feature alignment failed: {e}")
        # Fallback: manual reindex
        X = features.reindex(columns=feature_columns)

    # Fill missing values with training medians
    X = X.fillna(pd.Series(feature_fill_values)).fillna(0.0)

    # Make predictions
    try:
        predictions = model.predict(X).astype(float)
        # Apply bias correction learned from holdout set
        predictions = predictions + bias_correction
        print(f"✅ Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    # Prepare output
    output = pd.DataFrame({
        "game_id": raw["game_id"], 
        "date": raw["date"], 
        "predicted_total": predictions
    })

    # Update enhanced_games table
    with engine.begin() as conn:
        updated_count = 0
        for _, row in output.iterrows():
            result = conn.execute(text("""
                UPDATE enhanced_games
                SET predicted_total = :pred
                WHERE game_id = :gid AND "date" = :d
            """), {
                "pred": float(row["predicted_total"]), 
                "gid": row["game_id"], 
                "d": row["date"]
            })
            if result.rowcount > 0:
                updated_count += 1

    print(f"✅ Updated {updated_count} games in enhanced_games table")
    
    # Show sample predictions
    print("\n📊 Sample Predictions:")
    print("   Game ID | Predicted Total")
    print("   --------|----------------")
    for _, row in output.head().iterrows():
        print(f"   {row['game_id']:>7} | {row['predicted_total']:>13.2f}")

    # Summary stats
    pred_mean = output["predicted_total"].mean()
    pred_std = output["predicted_total"].std()
    print(f"\n📈 Prediction Summary:")
    print(f"   Mean: {pred_mean:.2f} runs")
    print(f"   Std:  {pred_std:.2f} runs")
    print(f"   Range: {output['predicted_total'].min():.2f} - {output['predicted_total'].max():.2f}")

if __name__ == "__main__":
    main()
