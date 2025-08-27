#!/usr/bin/env python3
"""
Generate Learning Predictions Using Existing Model
=================================================
Uses the existing legitimate_model_latest.joblib to generate learning predictions for today.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_learning_model():
    """Load the existing learning model"""
    model_path = 'mlb-overs/models/legitimate_model_latest.joblib'
    
    if not os.path.exists(model_path):
        logger.error(f"Learning model not found: {model_path}")
        return None, None
        
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        logger.info(f"✅ Loaded learning model with {len(feature_columns)} features")
        logger.info(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        logger.info(f"   Test MAE: {model_data.get('test_mae', 'Unknown')}")
        
        return model, feature_columns, model_data
        
    except Exception as e:
        logger.error(f"Failed to load learning model: {e}")
        return None, None, None

def get_todays_games(engine, target_date=None):
    """Get today's games that need learning predictions"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    query = '''
    SELECT * FROM enhanced_games 
    WHERE date = :date 
    AND total_runs IS NULL  -- Only upcoming games
    AND predicted_total IS NOT NULL  -- Has enhanced predictions
    ORDER BY game_id
    '''
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {'date': target_date})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    logger.info(f"📊 Found {len(df)} games needing learning predictions for {target_date}")
    return df

def engineer_features_for_learning(df, feature_columns):
    """Engineer features for the learning model"""
    featured_df = df.copy()
    
    # Create missing features that the learning model expects
    missing_features = []
    for feature in feature_columns:
        if feature not in featured_df.columns:
            missing_features.append(feature)
            
            # Create reasonable estimates for missing in-game features
            if 'inning' in feature.lower():
                featured_df[feature] = 5.0  # Average game midpoint
            elif 'runs_' in feature and 'current' in feature:
                # Estimate current runs based on team averages
                if 'home' in feature:
                    featured_df[feature] = featured_df.get('home_team_runs_per_game', 4.5) * 0.6
                else:
                    featured_df[feature] = featured_df.get('away_team_runs_per_game', 4.5) * 0.6
            elif 'pitch_count' in feature.lower():
                featured_df[feature] = 85.0  # Average starter pitch count
            elif 'bullpen_usage' in feature.lower():
                featured_df[feature] = 0.3  # Moderate bullpen usage estimate
            else:
                # For other missing features, use zero
                featured_df[feature] = 0.0
    
    if missing_features:
        logger.info(f"⚙️ Created estimates for {len(missing_features)} missing in-game features")
        
    # Ensure all required features exist
    for feature in feature_columns:
        if feature not in featured_df.columns:
            featured_df[feature] = 0.0
            
    # Select only the features the model expects, in the correct order
    X = featured_df[feature_columns].fillna(0.0)
    
    logger.info(f"✅ Engineered features: {X.shape}")
    return X

def main():
    target_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"🚀 Generating learning predictions for {target_date}")
    
    # Initialize database connection
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    # Load learning model
    model, feature_columns, model_data = load_learning_model()
    if model is None:
        logger.error("Failed to load learning model")
        return
    
    # Get today's games
    df = get_todays_games(engine, target_date)
    if df.empty:
        logger.info("No games need learning predictions")
        return
    
    # Engineer features
    X = engineer_features_for_learning(df, feature_columns)
    
    # Generate predictions
    try:
        raw_predictions = model.predict(X)
        
        # Apply bias correction if available
        bias_correction = model_data.get('bias_correction', 0.0)
        predictions = raw_predictions + bias_correction
        
        logger.info(f"🎯 Generated learning predictions:")
        logger.info(f"   Range: {predictions.min():.2f} - {predictions.max():.2f}")
        logger.info(f"   Mean: {predictions.mean():.2f}")
        logger.info(f"   Std: {predictions.std():.2f}")
        
        # Store predictions in database
        logger.info("💾 Storing learning predictions...")
        
        with engine.begin() as conn:
            for i, (_, row) in enumerate(df.iterrows()):
                update_sql = text("""
                    UPDATE enhanced_games 
                    SET predicted_total_learning = :prediction,
                        prediction_timestamp = NOW()
                    WHERE game_id = :game_id AND date = :date
                """)
                
                conn.execute(update_sql, {
                    'prediction': float(predictions[i]),
                    'game_id': row['game_id'],
                    'date': target_date
                })
        
        logger.info(f"✅ Successfully stored {len(predictions)} learning predictions!")
        
        # Show comparison with enhanced predictions
        logger.info("📊 Learning vs Enhanced Predictions:")
        for i, (_, row) in enumerate(df.iterrows()):
            enhanced_pred = row.get('predicted_total', 0)
            learning_pred = predictions[i]
            diff = learning_pred - enhanced_pred
            logger.info(f"   {row['home_team']} vs {row['away_team']}: Enhanced={enhanced_pred:.2f}, Learning={learning_pred:.2f}, Diff={diff:+.2f}")
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        return

if __name__ == "__main__":
    main()
