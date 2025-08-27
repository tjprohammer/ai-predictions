#!/usr/bin/env python3
"""
Generate Learning Predictions
============================
This script generates predictions using the learning model (legitimate_model_latest.joblib)
and stores them in the predicted_total_learning column.

The learning model normally requires 118 features including in-game data, but for 
pre-game predictions we'll use feature engineering to create reasonable estimates
for missing in-game features based on historical averages.
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

class LearningPredictor:
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
        self.model_data = None
        self.model = None
        self.feature_columns = None
        
    def load_learning_model(self):
        """Load the legitimate learning model"""
        model_path = 'mlb-overs/models/legitimate_model_latest.joblib'
        
        if not os.path.exists(model_path):
            logger.error(f"Learning model not found: {model_path}")
            return False
            
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.feature_columns = self.model_data['feature_columns']
            
            logger.info(f"✅ Loaded learning model with {len(self.feature_columns)} features")
            logger.info(f"   Training date: {self.model_data.get('training_date', 'Unknown')}")
            logger.info(f"   Test MAE: {self.model_data.get('test_mae', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load learning model: {e}")
            return False
    
    def load_today_games(self, target_date=None):
        """Load today's games that need learning predictions"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        query = '''
        SELECT * FROM enhanced_games 
        WHERE date = %s 
        AND total_runs IS NULL  -- Only upcoming games
        AND predicted_total IS NOT NULL  -- Only games with enhanced predictions
        AND (predicted_total_learning IS NULL OR predicted_total_learning = 0)  -- Missing learning predictions
        ORDER BY game_id
        '''
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), (target_date,))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        logger.info(f"📊 Loaded {len(df)} games needing learning predictions for {target_date}")
        return df
    
    def engineer_learning_features(self, df):
        """Engineer features for learning model using pre-game data + reasonable estimates"""
        featured_df = df.copy()
        
        # Create missing features that the learning model expects
        # These are estimates based on pre-game data and historical averages
        
        # If we don't have in-game features, create reasonable estimates
        missing_features = []
        for feature in self.feature_columns:
            if feature not in featured_df.columns:
                missing_features.append(feature)
                
                # Create reasonable estimates for missing in-game features
                if 'inning' in feature.lower():
                    featured_df[feature] = 5.0  # Average game goes ~9 innings, midpoint
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
                    # For other missing features, use zero (will be handled by model)
                    featured_df[feature] = 0.0
        
        if missing_features:
            logger.info(f"⚙️ Created estimates for {len(missing_features)} missing in-game features")
            
        # Ensure all required features exist
        for feature in self.feature_columns:
            if feature not in featured_df.columns:
                featured_df[feature] = 0.0
                
        # Select only the features the model expects, in the correct order
        X = featured_df[self.feature_columns].fillna(0.0)
        
        # Basic validation
        if X.isna().any().any():
            logger.warning("⚠️ Some features still contain NaN values, filling with 0")
            X = X.fillna(0.0)
            
        logger.info(f"✅ Engineered features: {X.shape}")
        
        return X
    
    def generate_learning_predictions(self, target_date=None):
        """Generate learning predictions for today's games"""
        
        # Load games
        df = self.load_today_games(target_date)
        if df.empty:
            logger.info("No games need learning predictions")
            return pd.DataFrame()
            
        # Engineer features
        X = self.engineer_learning_features(df)
        
        # Generate predictions
        try:
            raw_predictions = self.model.predict(X)
            
            # Apply bias correction if available
            bias_correction = self.model_data.get('bias_correction', 0.0)
            predictions = raw_predictions + bias_correction
            
            logger.info(f"🎯 Generated learning predictions:")
            logger.info(f"   Range: {predictions.min():.2f} - {predictions.max():.2f}")
            logger.info(f"   Mean: {predictions.mean():.2f}")
            logger.info(f"   Std: {predictions.std():.2f}")
            
            # Create results dataframe
            results_df = df[['game_id', 'date', 'home_team', 'away_team']].copy()
            results_df['predicted_total_learning'] = predictions
            
            return results_df
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            return pd.DataFrame()
    
    def store_learning_predictions(self, predictions_df):
        """Store learning predictions in database"""
        if predictions_df.empty:
            logger.info("No predictions to store")
            return False
            
        try:
            with self.engine.begin() as conn:
                for _, row in predictions_df.iterrows():
                    update_sql = text("""
                        UPDATE enhanced_games 
                        SET predicted_total_learning = :prediction,
                            prediction_timestamp = NOW()
                        WHERE game_id = :game_id AND date = :date
                    """)
                    
                    conn.execute(update_sql, {
                        'prediction': float(row['predicted_total_learning']),
                        'game_id': row['game_id'],
                        'date': row['date']
                    })
            
            logger.info(f"💾 Stored {len(predictions_df)} learning predictions in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            return False
    
    def run(self, target_date=None):
        """Main execution function"""
        logger.info("🚀 Starting Learning Predictions Generator")
        
        # Load model
        if not self.load_learning_model():
            return False
            
        # Generate predictions
        predictions_df = self.generate_learning_predictions(target_date)
        
        if predictions_df.empty:
            logger.info("✅ No learning predictions needed")
            return True
            
        # Store predictions
        success = self.store_learning_predictions(predictions_df)
        
        if success:
            logger.info("✅ Learning predictions generated and stored successfully!")
        else:
            logger.error("❌ Failed to store learning predictions")
            
        return success

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Learning Predictions')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    predictor = LearningPredictor()
    success = predictor.run(args.target_date)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
