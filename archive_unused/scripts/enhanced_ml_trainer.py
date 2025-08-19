#!/usr/bin/env python3
"""
Enhanced ML Training Model for MLB Predictions
==============================================

This replaces the simple historical averaging with a sophisticated ML model
that considers multiple factors:
- Pitcher matchups and performance
- Team offensive capabilities  
- Weather conditions
- Venue factors
- Recent form and trends

Uses Random Forest and Gradient Boosting for robust predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLBPredictor:
    def __init__(self, db_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb"):
        self.engine = create_engine(db_url)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_importance = None
        
    def collect_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Collect historical game data for training"""
        logger.info(f"Collecting training data from last {days_back} days")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            with self.engine.connect() as conn:
                # Simplified query to get basic enhanced_games data first
                query_sql = """
                SELECT 
                    date, home_team, away_team, total_runs, venue_name,
                    temperature, wind_speed, wind_direction, home_score, away_score,
                    home_sp_er, home_sp_ip, home_sp_k, home_sp_bb,
                    away_sp_er, away_sp_ip, away_sp_k, away_sp_bb,
                    day_night, weather_condition
                FROM enhanced_games
                WHERE date >= :start_date 
                AND date <= :end_date
                AND total_runs IS NOT NULL
                AND total_runs > 0
                ORDER BY date DESC
                LIMIT 2000
                """
                
                df = pd.read_sql(text(query_sql), conn, params={'start_date': start_date, 'end_date': end_date})
                
            logger.info(f"Collected {len(df)} historical games for training")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for ML model"""
        logger.info("Engineering features for ML model")
        
        featured_df = df.copy()
        
        # Handle missing values
        numeric_columns = ['home_runs_pg', 'away_runs_pg', 'home_ba', 'away_ba', 
                          'home_woba', 'away_woba', 'temperature', 'wind_speed']
        
        for col in numeric_columns:
            if col in featured_df.columns:
                featured_df[col] = pd.to_numeric(featured_df[col], errors='coerce')
                featured_df[col].fillna(featured_df[col].median(), inplace=True)
        
        # Create composite features
        if 'home_score' in featured_df.columns and 'away_score' in featured_df.columns:
            featured_df['combined_offense'] = featured_df['home_score'] + featured_df['away_score']
            featured_df['score_differential'] = featured_df['home_score'] - featured_df['away_score']
        
        # Pitcher performance features
        if 'home_sp_er' in featured_df.columns and 'home_sp_ip' in featured_df.columns:
            featured_df['home_era'] = np.where(featured_df['home_sp_ip'] > 0, 
                                             (featured_df['home_sp_er'] * 9) / featured_df['home_sp_ip'], 4.50)
            
        if 'away_sp_er' in featured_df.columns and 'away_sp_ip' in featured_df.columns:
            featured_df['away_era'] = np.where(featured_df['away_sp_ip'] > 0,
                                             (featured_df['away_sp_er'] * 9) / featured_df['away_sp_ip'], 4.50)
        
        if 'home_era' in featured_df.columns and 'away_era' in featured_df.columns:
            featured_df['combined_era'] = (featured_df['home_era'] + featured_df['away_era']) / 2
            featured_df['pitcher_quality'] = np.where(featured_df['combined_era'] < 3.5, 1.2,
                                                    np.where(featured_df['combined_era'] > 5.0, 0.8, 1.0))
        
        # Strikeout rates
        if 'home_sp_k' in featured_df.columns and 'home_sp_ip' in featured_df.columns:
            featured_df['home_k_rate'] = featured_df['home_sp_k'] / (featured_df['home_sp_ip'] * 3)  # Per batter faced
            
        if 'away_sp_k' in featured_df.columns and 'away_sp_ip' in featured_df.columns:
            featured_df['away_k_rate'] = featured_df['away_sp_k'] / (featured_df['away_sp_ip'] * 3)
        
        # Weather features
        if 'temperature' in featured_df.columns:
            featured_df['temperature'] = pd.to_numeric(featured_df['temperature'], errors='coerce')
            featured_df['temp_factor'] = np.where(featured_df['temperature'] > 80, 1.1, 1.0)
            featured_df['temp_factor'] = np.where(featured_df['temperature'] < 60, 0.9, featured_df['temp_factor'])
        
        if 'wind_speed' in featured_df.columns:
            featured_df['wind_speed'] = pd.to_numeric(featured_df['wind_speed'], errors='coerce')
            featured_df['wind_factor'] = np.where(featured_df['wind_speed'] > 15, 1.05, 1.0)
        
        # Day/night game feature
        if 'day_night' in featured_df.columns:
            featured_df['is_day_game'] = (featured_df['day_night'] == 'day').astype(int)
        
        # Venue encoding
        if 'venue_name' in featured_df.columns:
            if 'venue_name' not in self.label_encoders:
                self.label_encoders['venue_name'] = LabelEncoder()
                featured_df['venue_encoded'] = self.label_encoders['venue_name'].fit_transform(featured_df['venue_name'].fillna('Unknown'))
            else:
                featured_df['venue_encoded'] = self.label_encoders['venue_name'].transform(featured_df['venue_name'].fillna('Unknown'))
        
        # Day of week feature
        if 'date' in featured_df.columns:
            featured_df['date'] = pd.to_datetime(featured_df['date'])
            featured_df['day_of_week'] = featured_df['date'].dt.dayofweek
            featured_df['is_weekend'] = (featured_df['day_of_week'] >= 5).astype(int)
        
        # Month feature (seasonal effects)
        featured_df['month'] = featured_df['date'].dt.month
        
        return featured_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target variable"""
        
        # Define feature columns based on available data
        feature_cols = [
            'combined_offense', 'score_differential', 'home_era', 'away_era', 'combined_era', 'pitcher_quality',
            'home_k_rate', 'away_k_rate', 'temperature', 'wind_speed', 'temp_factor', 'wind_factor',
            'venue_encoded', 'day_of_week', 'is_weekend', 'month', 'is_day_game'
        ]
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_cols)} features: {available_cols}")
        
        X = df[available_cols].copy()
        
        # Handle any remaining NaN values
        X.fillna(X.median(), inplace=True)
        
        # Target variable
        y = df['total_runs'].values
        
        return X.values, y
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train the enhanced ML model"""
        logger.info("Training enhanced ML model")
        
        # Feature engineering
        featured_df = self.feature_engineering(df)
        
        # Prepare features
        X, y = self.prepare_features(featured_df)
        
        if len(X) == 0:
            logger.error("No features available for training")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model (Random Forest + Gradient Boosting)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Evaluate
        mae = mean_absolute_error(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"MAE: {mae:.3f}")
        logger.info(f"MSE: {mse:.3f}")
        logger.info(f"R²: {r2:.3f}")
        
        # Store the ensemble models
        self.model = {
            'rf': rf_model,
            'gb': gb_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        # Feature importance
        rf_importance = rf_model.feature_importances_
        feature_cols = [col for col in [
            'home_runs_pg', 'away_runs_pg', 'combined_runs_pg', 'offense_advantage',
            'home_ba', 'away_ba', 'combined_ba',
            'home_woba', 'away_woba', 'combined_woba',
            'temperature', 'wind_speed', 'temp_factor', 'wind_factor',
            'venue_encoded', 'day_of_week', 'is_weekend', 'month'
        ] if col in featured_df.columns]
        
        self.feature_importance = dict(zip(feature_cols, rf_importance))
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'feature_importance': self.feature_importance,
            'n_samples': len(X)
        }
    
    def predict_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for new games using trained model"""
        logger.info(f"Making predictions for {len(games_df)} games")
        
        if self.model is None:
            logger.error("Model not trained yet")
            return games_df
        
        # Feature engineering
        featured_df = self.feature_engineering(games_df)
        
        # Prepare features
        X, _ = self.prepare_features(featured_df)
        
        if len(X) == 0:
            logger.warning("No features available for prediction")
            games_df['predicted_total'] = 8.5  # Default fallback
            games_df['confidence'] = 0.1
            return games_df
        
        # Scale features
        X_scaled = self.model['scaler'].transform(X)
        
        # Make ensemble predictions
        rf_pred = self.model['rf'].predict(X_scaled)
        gb_pred = self.model['gb'].predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Calculate confidence based on model agreement
        prediction_variance = np.abs(rf_pred - gb_pred)
        confidence = np.clip(1.0 - (prediction_variance / 5.0), 0.1, 0.9)
        
        # Add predictions to dataframe
        games_df = games_df.copy()
        games_df['predicted_total'] = ensemble_pred
        games_df['confidence'] = confidence
        games_df['rf_prediction'] = rf_pred
        games_df['gb_prediction'] = gb_pred
        
        return games_df
    
    def save_model(self, filepath: str = 'models/enhanced_mlb_model.joblib'):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/enhanced_mlb_model.joblib'):
        """Load a trained model"""
        try:
            self.model = joblib.load(filepath)
            self.scaler = self.model['scaler']
            self.label_encoders = self.model['label_encoders']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    """Train and test the enhanced model"""
    predictor = EnhancedMLBPredictor()
    
    # Collect training data
    training_data = predictor.collect_training_data(days_back=365)
    
    if not training_data.empty:
        # Train model
        results = predictor.train_model(training_data)
        
        # Save model
        predictor.save_model()
        
        print("\n" + "="*50)
        print("ENHANCED MLB MODEL TRAINING COMPLETE")
        print("="*50)
        print(f"Training Samples: {results.get('n_samples', 0)}")
        print(f"Mean Absolute Error: {results.get('mae', 0):.3f} runs")
        print(f"R² Score: {results.get('r2', 0):.3f}")
        print("\nTop Feature Importance:")
        
        if results.get('feature_importance'):
            sorted_features = sorted(results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"  {feature}: {importance:.3f}")
        
        print("\nModel ready for predictions!")
        
    else:
        print("No training data available")

if __name__ == "__main__":
    main()
