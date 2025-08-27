#!/usr/bin/env python3
"""
Create Proper Pre-Game Adaptive Model
=====================================
Creates an 86-feature pre-game model using all legitimate features from the 118-feature 
legitimate model, excluding the 32 in-game data leak features.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_proper_pregame_model():
    """Create a proper pre-game adaptive model with ~86 features (no data leaks)"""
    
    # 1. Load the legitimate model to get feature lists
    logger.info("📊 Analyzing legitimate model features...")
    legitimate_path = Path('mlb-overs/models/legitimate_model_latest.joblib')
    legitimate_data = joblib.load(legitimate_path)
    all_features = legitimate_data['feature_columns']
    
    # 2. Identify in-game features (data leaks)
    in_game_keywords = ['score', '_runs', '_er', '_ip', 'actual_', 'final_']
    in_game_features = []
    pre_game_features = []
    
    for feature in all_features:
        if any(keyword in feature.lower() for keyword in in_game_keywords):
            in_game_features.append(feature)
        else:
            pre_game_features.append(feature)
    
    logger.info(f"✅ Feature analysis complete:")
    logger.info(f"   Total features: {len(all_features)}")
    logger.info(f"   In-game features (data leaks): {len(in_game_features)}")
    logger.info(f"   Pre-game features (legitimate): {len(pre_game_features)}")
    
    logger.info(f"\n🚫 In-game features being excluded:")
    for i, feature in enumerate(in_game_features[:10], 1):
        logger.info(f"   {i:2d}. {feature}")
    if len(in_game_features) > 10:
        logger.info(f"   ... and {len(in_game_features)-10} more")
    
    # 3. Load training data from enhanced_games
    logger.info(f"\n📈 Loading training data...")
    from sqlalchemy import create_engine
    
    db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
    engine = create_engine(db_url)
    
    # Get recent training data (last 30 days)
    query = """
    SELECT * FROM enhanced_games 
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
      AND total_runs IS NOT NULL
      AND total_runs > 0
    ORDER BY date DESC
    LIMIT 1000
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"✅ Loaded {len(df)} training games")
    
    # 4. Prepare features and target
    # Get available pre-game features
    available_features = [f for f in pre_game_features if f in df.columns]
    missing_features = [f for f in pre_game_features if f not in df.columns]
    
    logger.info(f"\n🎯 Feature availability:")
    logger.info(f"   Available pre-game features: {len(available_features)}")
    logger.info(f"   Missing features: {len(missing_features)}")
    
    if missing_features:
        logger.info(f"\n⚠️ Missing features (will be excluded):")
        for feature in missing_features[:10]:
            logger.info(f"   {feature}")
        if len(missing_features) > 10:
            logger.info(f"   ... and {len(missing_features)-10} more")
    
    # Use available features
    X = df[available_features].copy()
    y = df['total_runs']
    
    # Clean data
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    
    # Remove constant features
    feature_stds = X.std(numeric_only=True)
    non_constant_features = feature_stds[feature_stds > 0].index.tolist()
    X = X[non_constant_features]
    
    logger.info(f"✅ Final feature set: {len(X.columns)} features")
    logger.info(f"   Removed {len(available_features) - len(X.columns)} constant features")
    
    # 5. Train the model
    logger.info(f"\n🏗️ Training pre-game adaptive model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use similar parameters to the original model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluate the model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    logger.info(f"\n📊 Model Performance:")
    logger.info(f"   Training MAE: {train_mae:.3f}")
    logger.info(f"   Test MAE: {test_mae:.3f}")
    logger.info(f"   Training R²: {train_r2:.3f}")
    logger.info(f"   Test R²: {test_r2:.3f}")
    
    # 7. Show top features
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    all_features = feature_importance.sort_values(ascending=False)
    
    logger.info(f"\n🎯 All {len(all_features)} feature importances:")
    for i, (feature, importance) in enumerate(all_features.items(), 1):
        logger.info(f"   {i:2d}. {feature:<35} {importance:.6f} ({importance*100:.2f}%)")
    
    # 8. Create feature fill values
    fill_values = X.median(numeric_only=True).to_dict()
    
    # 9. Save the model
    logger.info(f"\n💾 Saving pre-game adaptive model...")
    
    model_data = {
        'model': model,
        'feature_columns': X.columns.tolist(),
        'feature_fill_values': fill_values,
        'model_type': 'pre_game_random_forest',
        'performance': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'training_info': {
            'n_samples': len(df),
            'n_features': len(X.columns),
            'excluded_in_game_features': len(in_game_features),
            'excluded_missing_features': len(missing_features)
        }
    }
    
    # Save to both locations
    save_paths = [
        Path('mlb-overs/models/adaptive_learning_model.joblib'),  # Replace the 11-feature version
        Path('mlb-overs/deployment/models/adaptive_learning_model.joblib')  # Also save to deployment
    ]
    
    for save_path in save_paths:
        save_path.parent.mkdir(exist_ok=True)
        joblib.dump(model_data, save_path)
        logger.info(f"✅ Saved to: {save_path}")
    
    logger.info(f"\n🎉 Pre-game adaptive model created successfully!")
    logger.info(f"   Features: {len(X.columns)} (up from 11)")
    logger.info(f"   Test R²: {test_r2:.3f}")
    logger.info(f"   No data leaks: ✅")
    
    return model_data

if __name__ == "__main__":
    create_proper_pregame_model()
