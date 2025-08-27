#!/usr/bin/env python3
"""
Generate Learning Predictions Using Dual Model System
====================================================
Uses the existing dual model predictor to generate learning predictions for today.
"""

import os
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# Add path for imports
sys.path.append('mlb-overs/models')
sys.path.append('mlb-overs/deployment')

try:
    from dual_model_predictor import DualModelPredictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run this script from the AI_Predictions root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    target_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"🚀 Generating learning predictions for {target_date}")
    
    # Initialize database connection
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    # Get today's games
    df = get_todays_games(engine, target_date)
    
    if df.empty:
        logger.info("No games need learning predictions")
        return
    
    # Initialize dual model predictor
    logger.info("🔄 Initializing dual model predictor...")
    dual_predictor = DualModelPredictor()
    
    # Generate learning predictions using the adaptive learning pipeline
    if dual_predictor.learning_model:
        logger.info("🎯 Generating learning predictions...")
        
        try:
            # Get enhanced features for prediction
            enhanced_df = dual_predictor.learning_model.get_enhanced_features(
                start_date=target_date, 
                end_date=target_date
            )
            
            if enhanced_df.empty:
                logger.warning("No enhanced features found for today's games")
                return
            
            # Filter to only upcoming games (no total_runs)
            upcoming_games = enhanced_df[enhanced_df['total_runs'].isna()].copy()
            
            if upcoming_games.empty:
                logger.warning("No upcoming games found in enhanced features")
                return
            
            logger.info(f"📊 Processing {len(upcoming_games)} upcoming games")
            
            # Train/load the adaptive model (this will use existing data)
            model, feature_columns = dual_predictor.learning_model.train_adaptive_model(enhanced_df)
            
            # Prepare features for prediction
            X = upcoming_games[feature_columns].fillna(0)
            
            # Generate predictions
            learning_predictions = model.predict(X)
            
            logger.info(f"🎯 Generated {len(learning_predictions)} learning predictions:")
            logger.info(f"   Range: {learning_predictions.min():.2f} - {learning_predictions.max():.2f}")
            logger.info(f"   Mean: {learning_predictions.mean():.2f}")
            logger.info(f"   Std: {learning_predictions.std():.2f}")
            
            # Store predictions in database
            logger.info("💾 Storing learning predictions...")
            
            with engine.begin() as conn:
                for i, (_, row) in enumerate(upcoming_games.iterrows()):
                    update_sql = text("""
                        UPDATE enhanced_games 
                        SET predicted_total_learning = :prediction,
                            prediction_timestamp = NOW()
                        WHERE game_id = :game_id AND date = :date
                    """)
                    
                    conn.execute(update_sql, {
                        'prediction': float(learning_predictions[i]),
                        'game_id': row['game_id'],
                        'date': target_date
                    })
            
            logger.info(f"✅ Successfully stored {len(learning_predictions)} learning predictions!")
            
        except Exception as e:
            logger.error(f"Failed to generate learning predictions: {e}")
            return
    
    else:
        logger.error("Learning model not available")
        return

if __name__ == "__main__":
    main()
