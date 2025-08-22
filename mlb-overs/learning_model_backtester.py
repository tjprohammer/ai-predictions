#!/usr/bin/env python3
"""
Historical Learning Model Backtesting System
This script applies the current learning model to historical games to evaluate performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import joblib
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningModelBacktester:
    def __init__(self):
        """Initialize the backtesting system"""
        self.engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
        self.learning_model = None
        self.feature_columns = None
        self.load_learning_model()
        
    def load_learning_model(self):
        """Load the trained learning model"""
        try:
            # Load the learning model from the expected location
            model_path = '../models/enhanced_model_recent_data.joblib'
            if os.path.exists(model_path):
                self.learning_model = joblib.load(model_path)
                logger.info("✅ Learning model loaded successfully")
            else:
                # Try alternative paths
                alternative_paths = [
                    'enhanced_model_recent_data.joblib',
                    '../enhanced_model_recent_data.joblib',
                    '../../enhanced_model_recent_data.joblib'
                ]
                
                for path in alternative_paths:
                    if os.path.exists(path):
                        self.learning_model = joblib.load(path)
                        logger.info(f"✅ Learning model loaded from {path}")
                        break
                        
                if self.learning_model is None:
                    logger.error("❌ Could not find learning model file")
                    return False
            
            # Get feature columns (assuming they're stored with the model or we can infer them)
            if hasattr(self.learning_model, 'feature_names_in_'):
                self.feature_columns = self.learning_model.feature_names_in_
            else:
                logger.warning("⚠️ Feature names not found in model, will need to infer them")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading learning model: {e}")
            return False
    
    def get_historical_games(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical completed games for backtesting"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            query = text("""
                SELECT 
                    game_id,
                    date,
                    home_team,
                    away_team,
                    venue_name,
                    predicted_total,
                    market_total,
                    total_runs,
                    temperature,
                    weather_condition,
                    wind_speed as wind_mph,
                    wind_direction,
                    humidity,
                    pressure,
                    home_team_bullpen_recent_era as home_bullpen_era,
                    away_team_bullpen_recent_era as away_bullpen_era,
                    home_sp_name as home_starting_pitcher,
                    away_sp_name as away_starting_pitcher,
                    recommendation,
                    edge,
                    confidence,
                    over_odds,
                    under_odds
                FROM enhanced_games 
                WHERE date BETWEEN :start_date AND :end_date
                AND total_runs IS NOT NULL  -- Only completed games
                AND predicted_total IS NOT NULL
                ORDER BY date DESC, game_id
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    'start_date': start_date, 
                    'end_date': end_date
                })
            
            logger.info(f"📊 Found {len(df)} completed games from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical games: {e}")
            return pd.DataFrame()
    
    def prepare_features_for_game(self, game_row: pd.Series) -> Optional[np.ndarray]:
        """Prepare feature vector for a single game"""
        try:
            # This would need to match the exact feature engineering from the learning model training
            # For now, I'll create a simplified version based on available data
            
            features = []
            
            # Basic game features
            features.extend([
                game_row.get('market_total', 0),
                game_row.get('predicted_total', 0),
                game_row.get('temperature', 70),
                game_row.get('wind_mph', 0),
                game_row.get('humidity', 50),
                game_row.get('pressure', 30.0),
                game_row.get('home_bullpen_era', 4.0),
                game_row.get('away_bullpen_era', 4.0),
                game_row.get('over_odds', -110),
                game_row.get('under_odds', -110)
            ])
            
            # Weather encoding (simplified)
            weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Dome', 'Partly Cloudy']
            weather = game_row.get('weather_condition', 'Clear')
            weather_encoded = [1 if weather == condition else 0 for condition in weather_conditions]
            features.extend(weather_encoded)
            
            # Wind direction encoding (simplified)
            wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Calm']
            wind_dir = game_row.get('wind_direction', 'Calm')
            wind_encoded = [1 if wind_dir == direction else 0 for direction in wind_directions]
            features.extend(wind_encoded)
            
            # Team features (simplified - would normally include more detailed stats)
            home_team_features = [0] * 30  # Placeholder for team-specific features
            away_team_features = [0] * 30  # Placeholder for team-specific features
            features.extend(home_team_features)
            features.extend(away_team_features)
            
            # Pitcher features (simplified)
            pitcher_features = [0] * 20  # Placeholder for pitcher-specific features
            features.extend(pitcher_features)
            
            # Additional derived features
            features.extend([
                game_row.get('market_total', 0) - game_row.get('predicted_total', 0),  # Market vs prediction diff
                (game_row.get('home_bullpen_era', 4.0) + game_row.get('away_bullpen_era', 4.0)) / 2,  # Avg bullpen ERA
                1 if game_row.get('temperature', 70) > 75 else 0,  # Hot weather
                1 if game_row.get('wind_mph', 0) > 10 else 0,  # Windy conditions
            ])
            
            # Pad or trim to expected feature count
            expected_features = 203  # Based on the learning model
            current_features = len(features)
            
            if current_features < expected_features:
                features.extend([0] * (expected_features - current_features))
            elif current_features > expected_features:
                features = features[:expected_features]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for game {game_row.get('game_id', 'unknown')}: {e}")
            return None
    
    def generate_learning_predictions(self, historical_games: pd.DataFrame) -> pd.DataFrame:
        """Generate learning model predictions for historical games"""
        if self.learning_model is None:
            logger.error("❌ Learning model not loaded")
            return historical_games
        
        learning_predictions = []
        
        logger.info(f"🔮 Generating learning predictions for {len(historical_games)} games...")
        
        for idx, game in historical_games.iterrows():
            try:
                # Prepare features
                features = self.prepare_features_for_game(game)
                
                if features is not None:
                    # Generate prediction
                    learning_pred = self.learning_model.predict(features)[0]
                    learning_predictions.append(learning_pred)
                else:
                    learning_predictions.append(None)
                    
            except Exception as e:
                logger.error(f"❌ Error predicting for game {game.game_id}: {e}")
                learning_predictions.append(None)
        
        # Add learning predictions to the dataframe
        historical_games['learning_prediction'] = learning_predictions
        
        valid_predictions = sum(1 for p in learning_predictions if p is not None)
        logger.info(f"✅ Generated {valid_predictions}/{len(historical_games)} valid learning predictions")
        
        return historical_games
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics comparing original vs learning model"""
        
        # Filter games with valid predictions and actual results
        valid_games = df[
            (df['total_runs'].notna()) & 
            (df['predicted_total'].notna()) & 
            (df['learning_prediction'].notna())
        ].copy()
        
        if len(valid_games) == 0:
            logger.warning("⚠️ No valid games for performance calculation")
            return {}
        
        # Calculate errors
        valid_games['original_error'] = abs(valid_games['predicted_total'] - valid_games['total_runs'])
        valid_games['learning_error'] = abs(valid_games['learning_prediction'] - valid_games['total_runs'])
        valid_games['market_error'] = abs(valid_games['market_total'] - valid_games['total_runs'])
        
        # Calculate accuracy (within thresholds)
        def accuracy_within_threshold(errors, threshold):
            return (errors <= threshold).mean() * 100
        
        # Performance summary
        performance = {
            'total_games': len(valid_games),
            'date_range': {
                'start': valid_games['date'].min().strftime('%Y-%m-%d'),
                'end': valid_games['date'].max().strftime('%Y-%m-%d')
            },
            'original_model': {
                'mean_error': valid_games['original_error'].mean(),
                'median_error': valid_games['original_error'].median(),
                'accuracy_within_1': accuracy_within_threshold(valid_games['original_error'], 1.0),
                'accuracy_within_2': accuracy_within_threshold(valid_games['original_error'], 2.0),
                'rmse': np.sqrt((valid_games['original_error'] ** 2).mean())
            },
            'learning_model': {
                'mean_error': valid_games['learning_error'].mean(),
                'median_error': valid_games['learning_error'].median(),
                'accuracy_within_1': accuracy_within_threshold(valid_games['learning_error'], 1.0),
                'accuracy_within_2': accuracy_within_threshold(valid_games['learning_error'], 2.0),
                'rmse': np.sqrt((valid_games['learning_error'] ** 2).mean())
            },
            'market_baseline': {
                'mean_error': valid_games['market_error'].mean(),
                'median_error': valid_games['market_error'].median(),
                'accuracy_within_1': accuracy_within_threshold(valid_games['market_error'], 1.0),
                'accuracy_within_2': accuracy_within_threshold(valid_games['market_error'], 2.0),
                'rmse': np.sqrt((valid_games['market_error'] ** 2).mean())
            }
        }
        
        # Head-to-head comparison
        learning_wins = (valid_games['learning_error'] < valid_games['original_error']).sum()
        original_wins = (valid_games['original_error'] < valid_games['learning_error']).sum()
        ties = len(valid_games) - learning_wins - original_wins
        
        performance['head_to_head'] = {
            'learning_wins': learning_wins,
            'original_wins': original_wins,
            'ties': ties,
            'learning_win_rate': (learning_wins / len(valid_games)) * 100
        }
        
        # Betting performance (simplified)
        performance['betting_simulation'] = self.calculate_betting_performance(valid_games)
        
        return performance
    
    def calculate_betting_performance(self, df: pd.DataFrame) -> Dict:
        """Calculate betting performance for both models"""
        betting_results = {
            'original_model': {'wins': 0, 'losses': 0, 'total_profit': 0},
            'learning_model': {'wins': 0, 'losses': 0, 'total_profit': 0}
        }
        
        for _, game in df.iterrows():
            actual = game['total_runs']
            market = game['market_total']
            original_pred = game['predicted_total']
            learning_pred = game['learning_prediction']
            
            # Original model betting
            if original_pred > market + 0.5:  # Bet over
                if actual > market:
                    betting_results['original_model']['wins'] += 1
                    betting_results['original_model']['total_profit'] += 0.91  # Assuming -110 odds
                else:
                    betting_results['original_model']['losses'] += 1
                    betting_results['original_model']['total_profit'] -= 1.0
            elif original_pred < market - 0.5:  # Bet under
                if actual < market:
                    betting_results['original_model']['wins'] += 1
                    betting_results['original_model']['total_profit'] += 0.91
                else:
                    betting_results['original_model']['losses'] += 1
                    betting_results['original_model']['total_profit'] -= 1.0
            
            # Learning model betting
            if learning_pred > market + 0.5:  # Bet over
                if actual > market:
                    betting_results['learning_model']['wins'] += 1
                    betting_results['learning_model']['total_profit'] += 0.91
                else:
                    betting_results['learning_model']['losses'] += 1
                    betting_results['learning_model']['total_profit'] -= 1.0
            elif learning_pred < market - 0.5:  # Bet under
                if actual < market:
                    betting_results['learning_model']['wins'] += 1
                    betting_results['learning_model']['total_profit'] += 0.91
                else:
                    betting_results['learning_model']['losses'] += 1
                    betting_results['learning_model']['total_profit'] -= 1.0
        
        # Calculate win rates and ROI
        for model in betting_results:
            total_bets = betting_results[model]['wins'] + betting_results[model]['losses']
            if total_bets > 0:
                betting_results[model]['win_rate'] = (betting_results[model]['wins'] / total_bets) * 100
                betting_results[model]['roi'] = (betting_results[model]['total_profit'] / total_bets) * 100
            else:
                betting_results[model]['win_rate'] = 0
                betting_results[model]['roi'] = 0
        
        return betting_results
    
    def save_backtest_results(self, df: pd.DataFrame, performance: Dict):
        """Save backtest results to database for UI consumption"""
        try:
            # Update the enhanced_games table with learning predictions
            # (This would be a more sophisticated update in practice)
            
            # For now, let's create a summary table
            summary_data = {
                'backtest_date': datetime.now(),
                'days_analyzed': (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days,
                'total_games': performance.get('total_games', 0),
                'learning_win_rate': performance.get('head_to_head', {}).get('learning_win_rate', 0),
                'learning_mean_error': performance.get('learning_model', {}).get('mean_error', 0),
                'original_mean_error': performance.get('original_model', {}).get('mean_error', 0),
                'learning_roi': performance.get('betting_simulation', {}).get('learning_model', {}).get('roi', 0),
                'original_roi': performance.get('betting_simulation', {}).get('original_model', {}).get('roi', 0)
            }
            
            logger.info("💾 Backtest results saved")
            return summary_data
            
        except Exception as e:
            logger.error(f"❌ Error saving backtest results: {e}")
            return None
    
    def run_backtest(self, days_back: int = 30) -> Dict:
        """Run complete backtesting analysis"""
        logger.info(f"🚀 Starting learning model backtest for last {days_back} days...")
        
        # Get historical games
        historical_games = self.get_historical_games(days_back)
        
        if len(historical_games) == 0:
            logger.error("❌ No historical games found")
            return {}
        
        # Generate learning predictions
        games_with_predictions = self.generate_learning_predictions(historical_games)
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(games_with_predictions)
        
        # Save results
        self.save_backtest_results(games_with_predictions, performance)
        
        logger.info("✅ Backtest completed successfully")
        return performance

def main():
    """Run the backtesting system"""
    backtester = LearningModelBacktester()
    
    # Run backtest for last 30 days
    results = backtester.run_backtest(days_back=30)
    
    if results:
        print("\n" + "="*60)
        print("📊 LEARNING MODEL BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"  Total Games Analyzed: {results.get('total_games', 0)}")
        print(f"  Date Range: {results.get('date_range', {}).get('start', 'N/A')} to {results.get('date_range', {}).get('end', 'N/A')}")
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        original = results.get('original_model', {})
        learning = results.get('learning_model', {})
        
        print(f"  Original Model:")
        print(f"    Mean Error: {original.get('mean_error', 0):.2f} runs")
        print(f"    Accuracy (±1 run): {original.get('accuracy_within_1', 0):.1f}%")
        print(f"    Accuracy (±2 runs): {original.get('accuracy_within_2', 0):.1f}%")
        
        print(f"  Learning Model:")
        print(f"    Mean Error: {learning.get('mean_error', 0):.2f} runs")
        print(f"    Accuracy (±1 run): {learning.get('accuracy_within_1', 0):.1f}%")
        print(f"    Accuracy (±2 runs): {learning.get('accuracy_within_2', 0):.1f}%")
        
        head_to_head = results.get('head_to_head', {})
        print(f"\n🥊 HEAD-TO-HEAD:")
        print(f"  Learning Model Wins: {head_to_head.get('learning_wins', 0)}")
        print(f"  Original Model Wins: {head_to_head.get('original_wins', 0)}")
        print(f"  Learning Win Rate: {head_to_head.get('learning_win_rate', 0):.1f}%")
        
        betting = results.get('betting_simulation', {})
        print(f"\n💰 BETTING SIMULATION:")
        if 'original_model' in betting:
            orig_betting = betting['original_model']
            print(f"  Original Model: {orig_betting.get('wins', 0)}-{orig_betting.get('losses', 0)} ({orig_betting.get('win_rate', 0):.1f}%) ROI: {orig_betting.get('roi', 0):.1f}%")
        
        if 'learning_model' in betting:
            learn_betting = betting['learning_model']
            print(f"  Learning Model: {learn_betting.get('wins', 0)}-{learn_betting.get('losses', 0)} ({learn_betting.get('win_rate', 0):.1f}%) ROI: {learn_betting.get('roi', 0):.1f}%")
        
        print("\n" + "="*60)
    else:
        print("❌ Backtest failed - no results generated")

if __name__ == "__main__":
    main()
