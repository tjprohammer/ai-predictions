#!/usr/bin/env python3
"""
Fixed Model Training Script
===========================
Addresses the fundamental training-production mismatch by:
1. Using enhanced_games as training source (complete data)
2. Using same feature engineering pipeline as production  
3. Filtering out zero/constant features before training
4. Ensuring exact feature alignment between training and production

Key Fixes:
- No more 69% missing ERA data (legitimate_game_features)
- No more zero-value features corrupting training
- Same 125-feature set used in both training and production
- Consistent feature engineering pipeline
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Add the deployment directory to path for imports
sys.path.append('.')

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
MODELS_DIR = Path("../models")

def fetch_complete_training_data(engine, end_date, window_days=80, min_total=5, max_total=13):
    """
    Fetch training data from enhanced_games table instead of incomplete legitimate_game_features.
    This ensures we have complete ERA data and features matching production.
    """
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=window_days-1)).strftime("%Y-%m-%d")
    
    print(f"📊 Fetching complete training data from enhanced_games")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Target range: {min_total}-{max_total} runs")
    
    query = text("""
        SELECT eg.*
        FROM enhanced_games eg
        WHERE eg.date BETWEEN :start AND :end
        AND eg.total_runs IS NOT NULL
        AND eg.total_runs BETWEEN :min_total AND :max_total
        AND eg.market_total IS NOT NULL
        AND eg.market_total BETWEEN :min_total AND :max_total
        ORDER BY eg.date DESC
    """)
    
    df = pd.read_sql(query, engine, params={
        "start": start_date, 
        "end": end_date,
        "min_total": min_total,
        "max_total": max_total
    })
    
    print(f"✅ Loaded {len(df)} complete games with outcomes")
    
    return df

def engineer_production_features(df):
    """
    Use the exact same feature engineering as production to eliminate mismatch.
    """
    try:
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    except ImportError as e:
        print(f"❌ Cannot import EnhancedBullpenPredictor: {e}")
        raise
    
    print("🔧 Engineering features using production pipeline...")
    
    predictor = EnhancedBullpenPredictor()
    
    # Use the same engineer_features method as production
    featured = predictor.engineer_features(df)
    
    print(f"✅ Feature engineering complete: {featured.shape[1]} raw features")
    
    # Align to model features (same as production)
    X_aligned = predictor.align_serving_features(featured, strict=False)
    
    print(f"✅ Feature alignment complete: {X_aligned.shape[1]} aligned features")
    
    return featured, X_aligned

def remove_problematic_features(X, y, min_variance=0.001, max_zero_rate=0.5):
    """
    Remove features that would corrupt training:
    - Constant features (no variance)
    - Features with too many zeros
    - Features perfectly correlated with outcome
    """
    print(f"🔍 Checking for problematic features...")
    
    initial_features = len(X.columns)
    to_remove = []
    
    # 1. Remove constant features
    for col in X.columns:
        if X[col].var() < min_variance:
            to_remove.append(col)
            print(f"   ❌ Constant feature: {col} (var={X[col].var():.6f})")
    
    # 2. Remove features with too many zeros
    for col in X.columns:
        if col not in to_remove:
            zero_rate = (X[col] == 0).mean()
            if zero_rate > max_zero_rate:
                to_remove.append(col)
                print(f"   ❌ High zero rate: {col} ({zero_rate*100:.1f}% zeros)")
    
    # 3. Remove features too correlated with outcome (potential leakage)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in to_remove:
            try:
                corr = abs(X[col].corr(y))
                if corr > 0.95:
                    to_remove.append(col)
                    print(f"   ❌ High correlation with outcome: {col} (r={corr:.3f})")
            except:
                pass
    
    # Remove the problematic features
    if to_remove:
        X_clean = X.drop(columns=to_remove)
        print(f"🧹 Removed {len(to_remove)} problematic features")
        print(f"   Features: {initial_features} → {len(X_clean.columns)}")
    else:
        X_clean = X.copy()
        print(f"✅ No problematic features found")
    
    return X_clean

def train_fixed_model(X, y, test_size=0.2, random_state=42):
    """
    Train a clean model with the fixed feature set.
    """
    print(f"🤖 Training model with {X.shape[1]} clean features...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"📊 Model Performance:")
    print(f"   Training MAE: {train_mae:.3f}")
    print(f"   Test MAE: {test_mae:.3f}")
    print(f"   Training size: {len(X_train)} games")
    print(f"   Test size: {len(X_test)} games")
    
    # Feature importance
    feature_importance = pd.Series(
        model.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    print(f"\n🔍 Top 10 Feature Importances:")
    for i, (feature, importance) in enumerate(feature_importance.head(10).items()):
        print(f"   {i+1:2d}. {feature:<25} {importance:.3f}")
    
    return model, feature_importance, train_mae, test_mae

def main():
    parser = argparse.ArgumentParser(description="Fixed model training")
    parser.add_argument("--end", default="2025-08-19", help="End date for training")
    parser.add_argument("--window-days", type=int, default=80, help="Training window in days")
    parser.add_argument("--deploy", action="store_true", help="Deploy the trained model")
    
    args = parser.parse_args()
    
    print("🔧 FIXED MODEL TRAINING")
    print("=" * 60)
    print(f"End date: {args.end}")
    print(f"Window: {args.window_days} days")
    print()
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Connect to database
    engine = create_engine(DB_URL)
    
    # 1. Fetch complete training data from enhanced_games
    df = fetch_complete_training_data(engine, args.end, args.window_days)
    
    if len(df) < 50:
        print(f"❌ Insufficient training data: {len(df)} games")
        return 1
    
    # 2. Engineer features using production pipeline
    featured, X = engineer_production_features(df)
    
    # 3. Remove problematic features
    X_clean = remove_problematic_features(X, df['total_runs'])
    
    # 4. Train model
    model, feature_importance, train_mae, test_mae = train_fixed_model(X_clean, df['total_runs'])
    
    # 5. Save model bundle
    bundle = {
        'model': model,
        'feature_columns': list(X_clean.columns),
        'training_date': datetime.now().isoformat(),
        'training_window_days': args.window_days,
        'training_end_date': args.end,
        'training_size': len(df),
        'train_mae': train_mae,
        'test_mae': test_mae,
        'schema_version': 'fixed_enhanced_v1',
        'trainer_version': 'fixed_training_script_v1',
        'feature_engineering': 'enhanced_bullpen_predictor',
        'data_source': 'enhanced_games',
        'feature_count': len(X_clean.columns),
        'bias_correction': 0.0
    }
    
    model_path = MODELS_DIR / "fixed_model_latest.joblib"
    joblib.dump(bundle, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"   Features: {len(X_clean.columns)}")
    print(f"   Training MAE: {train_mae:.3f}")
    print(f"   Test MAE: {test_mae:.3f}")
    
    if args.deploy:
        # Copy to production model location
        production_path = MODELS_DIR / "legitimate_model_latest.joblib"
        joblib.dump(bundle, production_path)
        print(f"🚀 Deployed to production: {production_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
