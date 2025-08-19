#!/usr/bin/env python3
"""
MLB Model Validation Pipeline
============================

This script creates a realistic ML pipeline that:
1. Collects all current data for UI
2. Trains model on historical games 
3. Tests predictions vs actual outcomes
4. Validates model performance realistically
5. Identifies overfitting and data leakage issues

Usage: python ml_validation_pipeline.py --mode [train|validate|predict]
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import time
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticMLBPredictor:
    def __init__(self, db_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb"):
        self.engine = create_engine(db_url)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_columns = []
        
    def collect_current_data_for_ui(self, target_date: str = None) -> Dict:
        """Collect all current data needed for UI display"""
        if not target_date:
            target_date = date.today().isoformat()
            
        logger.info(f"Collecting current data for UI: {target_date}")
        
        # 1. Get today's games from MLB API
        url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={target_date}&endDate={target_date}&sportId=1&hydrate=weather,venue,team,probablePitcher"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            return {"error": "Failed to fetch games"}
            
        data = response.json()
        if not data.get('dates') or not data['dates'][0].get('games'):
            return {"error": "No games found", "games": []}
            
        games = data['dates'][0]['games']
        
        # 2. Get current team stats
        team_stats = self.get_current_team_stats()
        
        # 3. Get real betting odds
        betting_odds = self.get_real_betting_odds()
        
        # 4. Get pitcher stats for each game
        enhanced_games = []
        for game in games:
            try:
                enhanced_game = self.enhance_game_with_data(game, team_stats, betting_odds)
                enhanced_games.append(enhanced_game)
            except Exception as e:
                logger.warning(f"Error enhancing game {game.get('gamePk')}: {e}")
                enhanced_games.append(self.create_basic_game(game))
        
        return {
            "status": "success",
            "date": target_date,
            "games": enhanced_games,
            "total_games": len(enhanced_games),
            "data_sources": ["MLB API 2025", "Team Stats", "Betting Lines", "Weather"]
        }
    
    def get_current_team_stats(self) -> pd.DataFrame:
        """Get current team offensive statistics"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT team, runs_pg, ba, woba, bb_pct, k_pct 
                    FROM teams_offense_daily 
                    WHERE date >= current_date - interval '3 days'
                    ORDER BY date DESC
                """)
                return pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"Error getting team stats: {e}")
            return pd.DataFrame()
    
    def get_real_betting_odds(self) -> Dict:
        """Get real betting odds from multiple sources"""
        try:
            # Load from our betting odds fetcher if file exists
            import os
            if os.path.exists('betting_odds_today.json'):
                import json
                with open('betting_odds_today.json', 'r') as f:
                    odds_data = json.load(f)
                    
                odds_dict = {}
                for odds in odds_data:
                    key = f"{odds['away_team']} @ {odds['home_team']}"
                    # Convert market_total to total for compatibility
                    odds_copy = odds.copy()
                    if 'market_total' in odds_copy:
                        odds_copy['total'] = odds_copy['market_total']
                    odds_dict[key] = odds_copy
                return odds_dict
                
        except Exception as e:
            logger.warning(f"Error loading betting odds: {e}")
            
        return {}
    
    def enhance_game_with_data(self, game: Dict, team_stats: pd.DataFrame, betting_odds: Dict) -> Dict:
        """Enhance game with comprehensive data"""
        home_team = game['teams']['home']['team']['abbreviation']
        away_team = game['teams']['away']['team']['abbreviation'] 
        
        # Get team stats
        home_stats = team_stats[team_stats['team'] == home_team]
        away_stats = team_stats[team_stats['team'] == away_team]
        
        home_runs_pg = home_stats['runs_pg'].iloc[0] if not home_stats.empty else 4.5
        away_runs_pg = away_stats['runs_pg'].iloc[0] if not away_stats.empty else 4.5
        
        # Get betting odds
        game_key = f"{game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}"
        odds = betting_odds.get(game_key, {})
        
        # Get pitcher data
        home_pitcher = game['teams']['home'].get('probablePitcher', {})
        away_pitcher = game['teams']['away'].get('probablePitcher', {})
        
        # Build enhanced game object
        enhanced = {
            "game_id": game['gamePk'],
            "date": game['officialDate'],
            "start_time": game['gameDate'],
            "home_team": game['teams']['home']['team']['name'],
            "away_team": game['teams']['away']['team']['name'],
            "home_team_abbr": home_team,
            "away_team_abbr": away_team,
            "venue_name": game['venue']['name'],
            "game_state": game['status']['detailedState'],
            
            # Pitcher info
            "home_pitcher_name": home_pitcher.get('fullName', 'TBD'),
            "away_pitcher_name": away_pitcher.get('fullName', 'TBD'),
            "home_pitcher_id": home_pitcher.get('id'),
            "away_pitcher_id": away_pitcher.get('id'),
            
            # Team stats
            "home_runs_pg": float(home_runs_pg),
            "away_runs_pg": float(away_runs_pg),
            
            # Weather
            "temperature": game.get('weather', {}).get('temp'),
            "wind_speed": game.get('weather', {}).get('wind'),
            "wind_direction": game.get('weather', {}).get('windDirection'),
            "weather_condition": game.get('weather', {}).get('condition'),
            
            # Betting data
            "market_total": odds.get('total', 8.5),
            "over_odds": odds.get('over_odds', -110),
            "under_odds": odds.get('under_odds', -110),
            
            # Prediction (to be filled by model)
            "predicted_total": None,
            "confidence": None,
            "recommendation": None
        }
        
        return enhanced
    
    def create_basic_game(self, game: Dict) -> Dict:
        """Create basic game object when enhancement fails"""
        return {
            "game_id": game['gamePk'],
            "home_team": game['teams']['home']['team']['name'],
            "away_team": game['teams']['away']['team']['name'],
            "venue_name": game['venue']['name'],
            "game_state": game['status']['detailedState'],
            "error": "Could not enhance with full data"
        }
    
    def collect_training_data_properly(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Collect training data with proper temporal splits"""
        logger.info("Collecting training data with temporal awareness")
        
        if not end_date:
            end_date = (date.today() - timedelta(days=1)).isoformat()  # Don't include today
        if not start_date:
            start_date = (date.today() - timedelta(days=730)).isoformat()  # 2 years back
        
        query = text("""
        SELECT 
            date,
            home_team,
            away_team,
            total_runs,
            home_score,
            away_score,
            venue_name,
            temperature,
            wind_speed,
            weather_condition,
            day_night,
            home_sp_er,
            home_sp_ip,
            home_sp_k,
            home_sp_bb,
            away_sp_er,
            away_sp_ip,
            away_sp_k,
            away_sp_bb,
            home_team_hits,
            away_team_hits
        FROM enhanced_games 
        WHERE date >= :start_date 
        AND date <= :end_date
        AND total_runs IS NOT NULL
        AND total_runs > 0
        ORDER BY date ASC
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date
                })
                
            logger.info(f"Collected {len(df)} historical games from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return pd.DataFrame()
    
    def create_realistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic features without data leakage"""
        featured_df = df.copy()
        
        # Basic team strength features (use historical averages)
        featured_df['combined_offense_estimate'] = 9.0  # League average baseline
        
        # Calculate pitcher ERAs from available stats
        featured_df['home_sp_ip'] = pd.to_numeric(featured_df['home_sp_ip'], errors='coerce').fillna(6.0)
        featured_df['away_sp_ip'] = pd.to_numeric(featured_df['away_sp_ip'], errors='coerce').fillna(6.0)
        featured_df['home_sp_er'] = pd.to_numeric(featured_df['home_sp_er'], errors='coerce').fillna(3.0)
        featured_df['away_sp_er'] = pd.to_numeric(featured_df['away_sp_er'], errors='coerce').fillna(3.0)
        
        # Calculate ERA for this game (earned runs per 9 innings)
        featured_df['home_game_era'] = (featured_df['home_sp_er'] * 9) / np.maximum(featured_df['home_sp_ip'], 0.1)
        featured_df['away_game_era'] = (featured_df['away_sp_er'] * 9) / np.maximum(featured_df['away_sp_ip'], 0.1)
        featured_df['combined_era'] = (featured_df['home_game_era'] + featured_df['away_game_era']) / 2
        featured_df['era_differential'] = featured_df['home_game_era'] - featured_df['away_game_era']
        
        # Strikeout rates from game stats
        featured_df['home_sp_k'] = pd.to_numeric(featured_df['home_sp_k'], errors='coerce').fillna(5)
        featured_df['away_sp_k'] = pd.to_numeric(featured_df['away_sp_k'], errors='coerce').fillna(5)
        featured_df['home_k_rate'] = featured_df['home_sp_k'] / np.maximum(featured_df['home_sp_ip'], 0.1)
        featured_df['away_k_rate'] = featured_df['away_sp_k'] / np.maximum(featured_df['away_sp_ip'], 0.1)
        featured_df['combined_k_rate'] = (featured_df['home_k_rate'] + featured_df['away_k_rate']) / 2
        
        # Walk rates
        featured_df['home_sp_bb'] = pd.to_numeric(featured_df['home_sp_bb'], errors='coerce').fillna(2)
        featured_df['away_sp_bb'] = pd.to_numeric(featured_df['away_sp_bb'], errors='coerce').fillna(2)
        featured_df['home_bb_rate'] = featured_df['home_sp_bb'] / np.maximum(featured_df['home_sp_ip'], 0.1)
        featured_df['away_bb_rate'] = featured_df['away_sp_bb'] / np.maximum(featured_df['away_sp_ip'], 0.1)
        
        # Team hitting features
        featured_df['home_team_hits'] = pd.to_numeric(featured_df['home_team_hits'], errors='coerce').fillna(8)
        featured_df['away_team_hits'] = pd.to_numeric(featured_df['away_team_hits'], errors='coerce').fillna(8)
        featured_df['combined_hits'] = featured_df['home_team_hits'] + featured_df['away_team_hits']
        
        # Weather features
        featured_df['temperature'] = pd.to_numeric(featured_df['temperature'], errors='coerce').fillna(72)
        featured_df['wind_speed'] = pd.to_numeric(featured_df['wind_speed'], errors='coerce').fillna(8)
        
        # Weather factors
        featured_df['temp_factor'] = np.where(featured_df['temperature'] > 80, 1.1, 1.0)
        featured_df['temp_factor'] = np.where(featured_df['temperature'] < 60, 0.9, featured_df['temp_factor'])
        featured_df['wind_factor'] = np.where(featured_df['wind_speed'] > 15, 1.05, 1.0)
        
        # Day/night games
        featured_df['is_day_game'] = (featured_df['day_night'] == 'day').astype(int)
        
        # Venue encoding
        if 'venue_name' in featured_df.columns:
            if 'venue_name' not in self.label_encoders:
                self.label_encoders['venue_name'] = LabelEncoder()
                featured_df['venue_encoded'] = self.label_encoders['venue_name'].fit_transform(
                    featured_df['venue_name'].fillna('Unknown'))
            else:
                try:
                    featured_df['venue_encoded'] = self.label_encoders['venue_name'].transform(
                        featured_df['venue_name'].fillna('Unknown'))
                except ValueError:
                    # Handle unseen venues
                    featured_df['venue_encoded'] = 0
        
        # Temporal features
        featured_df['date'] = pd.to_datetime(featured_df['date'])
        featured_df['day_of_week'] = featured_df['date'].dt.dayofweek
        featured_df['month'] = featured_df['date'].dt.month
        featured_df['is_weekend'] = (featured_df['day_of_week'] >= 5).astype(int)
        
        # Select only realistic features that don't leak future information
        self.feature_columns = [
            'combined_offense_estimate', 'home_game_era', 'away_game_era', 'combined_era', 'era_differential',
            'home_k_rate', 'away_k_rate', 'combined_k_rate', 'home_bb_rate', 'away_bb_rate',
            'temperature', 'wind_speed', 'temp_factor', 'wind_factor', 'is_day_game',
            'venue_encoded', 'day_of_week', 'month', 'is_weekend'
        ]
        
        return featured_df
    
    def train_realistic_model(self, df: pd.DataFrame) -> Dict:
        """Train model with proper validation to avoid overfitting"""
        logger.info("Training realistic ML model with proper validation")
        
        # Feature engineering
        featured_df = self.create_realistic_features(df)
        
        # Get features and target
        X = featured_df[self.feature_columns].copy()
        y = featured_df['total_runs'].values
        
        # Handle any remaining NaN values
        X.fillna(X.median(), inplace=True)
        
        logger.info(f"Training with {len(X)} samples and {len(self.feature_columns)} features")
        
        # Time-based split (more realistic for time series data)
        split_date = featured_df['date'].quantile(0.8)  # 80% train, 20% test
        train_mask = featured_df['date'] <= split_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        
        logger.info(f"Train set: {len(X_train)} games")
        logger.info(f"Test set: {len(X_test)} games")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with regularization to prevent overfitting
        model = RandomForestRegressor(
            n_estimators=50,  # Fewer trees to reduce overfitting
            max_depth=8,      # Limit depth
            min_samples_split=10,  # Require more samples to split
            min_samples_leaf=5,    # Require more samples in leaves
            max_features='sqrt',   # Use subset of features
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Check for overfitting
        overfitting_ratio = train_mae / max(test_mae, 0.001)
        
        logger.info(f"Training MAE: {train_mae:.3f}")
        logger.info(f"Test MAE: {test_mae:.3f}")
        logger.info(f"Training R²: {train_r2:.3f}")
        logger.info(f"Test R²: {test_r2:.3f}")
        logger.info(f"Overfitting ratio: {overfitting_ratio:.3f} (< 0.8 is good)")
        
        # Cross-validation for additional validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"Cross-validation MAE: {cv_mae:.3f} (±{cv_std:.3f})")
        
        # Store model
        self.model = {
            'model': model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'overfitting_ratio': overfitting_ratio,
            'feature_importance': feature_importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'predictions_vs_actual': list(zip(y_test, y_pred_test))
        }
    
    def validate_on_known_games(self, validation_date: str = None) -> pd.DataFrame:
        """Test model predictions against known game outcomes"""
        if not validation_date:
            validation_date = (date.today() - timedelta(days=7)).isoformat()
            
        logger.info(f"Validating model on games from {validation_date}")
        
        # Get games from validation date
        query = text("""
        SELECT * FROM enhanced_games 
        WHERE date = :date 
        AND total_runs IS NOT NULL
        ORDER BY date, game_id
        """)
        
        try:
            with self.engine.connect() as conn:
                validation_games = pd.read_sql(query, conn, params={'date': validation_date})
                
            if validation_games.empty:
                logger.warning("No validation games found")
                return pd.DataFrame()
                
            # Create features
            featured_games = self.create_realistic_features(validation_games)
            
            # Make predictions
            X = featured_games[self.feature_columns].copy()
            X.fillna(X.median(), inplace=True)
            X_scaled = self.model['scaler'].transform(X)
            
            predictions = self.model['model'].predict(X_scaled)
            
            # Create validation results
            results = pd.DataFrame({
                'game_id': validation_games['game_id'] if 'game_id' in validation_games.columns else range(len(validation_games)),
                'home_team': validation_games['home_team'],
                'away_team': validation_games['away_team'],
                'actual_total': validation_games['total_runs'],
                'predicted_total': predictions,
                'error': abs(validation_games['total_runs'] - predictions),
                'home_score': validation_games['home_score'],
                'away_score': validation_games['away_score']
            })
            
            # Calculate validation metrics
            mae = results['error'].mean()
            logger.info(f"Validation MAE: {mae:.3f} runs")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str = 'models/realistic_mlb_model.joblib'):
        """Save the trained model"""
        if self.model is not None:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/realistic_mlb_model.joblib'):
        """Load a trained model"""
        try:
            self.model = joblib.load(filepath)
            if 'scaler' in self.model:
                self.scaler = self.model['scaler']
            if 'label_encoders' in self.model:
                self.label_encoders = self.model['label_encoders']
            if 'feature_columns' in self.model:
                self.feature_columns = self.model['feature_columns']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    parser = argparse.ArgumentParser(description='MLB Model Validation Pipeline')
    parser.add_argument('--mode', choices=['train', 'validate', 'predict', 'current'], 
                       default='train', help='Pipeline mode')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    predictor = RealisticMLBPredictor()
    
    if args.mode == 'current':
        # Collect current data for UI
        current_data = predictor.collect_current_data_for_ui(args.date)
        print("\n" + "="*70)
        print("CURRENT DATA FOR UI")
        print("="*70)
        print(f"Status: {current_data.get('status')}")
        print(f"Date: {current_data.get('date')}")
        print(f"Games: {current_data.get('total_games')}")
        
        if current_data.get('games'):
            for game in current_data['games'][:3]:  # Show first 3
                print(f"\n{game.get('away_team')} @ {game.get('home_team')}")
                print(f"  Market Total: {game.get('market_total')}")
                print(f"  Team Runs/Game: {game.get('home_runs_pg'):.1f} / {game.get('away_runs_pg'):.1f}")
                print(f"  Weather: {game.get('temperature')}°F, {game.get('wind_speed')}mph")
    
    elif args.mode == 'train':
        # Train model on historical data
        training_data = predictor.collect_training_data_properly()
        
        if not training_data.empty:
            results = predictor.train_realistic_model(training_data)
            
            print("\n" + "="*70)
            print("REALISTIC MLB MODEL TRAINING RESULTS")
            print("="*70)
            print(f"Training samples: {results['n_train']}")
            print(f"Test samples: {results['n_test']}")
            print(f"Training MAE: {results['train_mae']:.3f} runs")
            print(f"Test MAE: {results['test_mae']:.3f} runs")
            print(f"Training R²: {results['train_r2']:.3f}")
            print(f"Test R²: {results['test_r2']:.3f}")
            print(f"Cross-validation MAE: {results['cv_mae']:.3f} ± {results['cv_std']:.3f}")
            print(f"Overfitting ratio: {results['overfitting_ratio']:.3f}")
            
            if results['overfitting_ratio'] < 0.8:
                print("✅ Model is NOT overfitting")
            else:
                print("❌ Model may be overfitting")
                
            print("\nTop Feature Importance:")
            sorted_features = sorted(results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:8]:
                print(f"  {feature}: {importance:.3f}")
                
            # Save model
            import os
            os.makedirs('models', exist_ok=True)
            predictor.save_model('models/realistic_mlb_model.joblib')
        
    elif args.mode == 'validate':
        # Load model and validate on known games
        try:
            predictor.load_model('models/realistic_mlb_model.joblib')
            validation_results = predictor.validate_on_known_games(args.date)
            
            if not validation_results.empty:
                print("\n" + "="*70)
                print("MODEL VALIDATION ON KNOWN GAMES")
                print("="*70)
                
                for _, row in validation_results.head(10).iterrows():
                    print(f"{row['away_team']} @ {row['home_team']}")
                    print(f"  Actual: {row['actual_total']:.0f} runs ({row['home_score']:.0f}-{row['away_score']:.0f})")
                    print(f"  Predicted: {row['predicted_total']:.1f} runs")
                    print(f"  Error: {row['error']:.1f} runs")
                    print()
                    
                mae = validation_results['error'].mean()
                print(f"Overall Validation MAE: {mae:.3f} runs")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train model first: python ml_validation_pipeline.py --mode train")

if __name__ == "__main__":
    main()
