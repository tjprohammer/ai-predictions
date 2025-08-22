#!/usr/bin/env python3
"""
Comprehensive Learning Model Backtester
=======================================
A thorough backtesting system that applies the learning model to historical games
to evaluate its performance against the original model.

This uses the EXACT same feature engineering pipeline as the production system
to ensure accurate backtesting results.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "deployment"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveLearningBacktester:
    """
    Comprehensive backtesting system for the learning model
    """
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or 'postgresql://mlbuser:mlbpass@localhost/mlb'
        self.engine = create_engine(self.db_url)
        self.learning_model = None
        self.feature_names = None
        
        # Load the learning model and metadata
        self.load_learning_model()
        
    def load_learning_model(self) -> bool:
        """Load the learning model and its metadata"""
        try:
            # Try to load the adaptive learning model
            model_path = Path(__file__).parent / "models" / "adaptive_learning_model.joblib"
            
            if model_path.exists():
                model_data = joblib.load(model_path)
                
                # Check if it's a dictionary with nested model
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        self.learning_model = model_data['model']
                        logger.info(f"✅ Loaded learning model from {model_path} (extracted from dict)")
                        
                        # Load feature names from the model data
                        if 'feature_columns' in model_data:
                            self.feature_names = model_data['feature_columns']
                            logger.info(f"✅ Loaded {len(self.feature_names)} feature names from model data")
                        else:
                            logger.warning("⚠️ Feature columns not found in model data")
                    else:
                        logger.error("❌ Model dictionary doesn't contain 'model' key")
                        return False
                else:
                    # Direct model object
                    self.learning_model = model_data
                    logger.info(f"✅ Loaded learning model from {model_path} (direct model)")
                
                # Try to load feature names from separate file if not already loaded
                if self.feature_names is None:
                    features_path = Path(__file__).parent / "models" / "comprehensive_features.json"
                    if features_path.exists():
                        with open(features_path, 'r') as f:
                            self.feature_names = json.load(f)
                        logger.info(f"✅ Loaded {len(self.feature_names)} feature names from JSON file")
                    else:
                        logger.warning("⚠️ Feature names file not found")
                        
                return True
            else:
                logger.error(f"❌ Learning model not found at {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading learning model: {e}")
            return False
    
    def get_historical_completed_games(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical completed games with their original predictions"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Query to get completed games with original predictions
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
                    wind_speed,
                    wind_direction,
                    humidity,
                    pressure,
                    home_team_bullpen_recent_era,
                    away_team_bullpen_recent_era,
                    home_sp_name,
                    away_sp_name,
                    recommendation,
                    edge,
                    confidence,
                    over_odds,
                    under_odds,
                    
                    -- Advanced team stats for feature engineering
                    home_team_ops,
                    away_team_ops,
                    home_team_runs_l7,
                    away_team_runs_l7,
                    home_sp_season_era,
                    away_sp_season_era,
                    plate_umpire
                    
                FROM enhanced_games
                WHERE date BETWEEN :start_date AND :end_date
                AND total_runs IS NOT NULL  -- Only completed games
                AND predicted_total IS NOT NULL  -- Only games with original predictions
                ORDER BY date DESC, game_id
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date
                })
            
            logger.info(f"📊 Found {len(df)} completed games with original predictions")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical games: {e}")
            return pd.DataFrame()
    
    def get_full_game_features(self, game_id: str) -> pd.DataFrame:
        """Get the complete feature set for a specific game"""
        try:
            query = text("SELECT * FROM enhanced_games WHERE game_id = :game_id")
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'game_id': game_id})
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching full features for game {game_id}: {e}")
            return pd.DataFrame()
    
    def prepare_learning_features(self, game_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare features for the learning model using the EXACT same pipeline
        as the production system
        """
        try:
            # Import the actual feature engineering pipeline
            from adaptive_learning_pipeline import AdaptiveLearningPipeline
            
            # Create pipeline instance
            pipeline = AdaptiveLearningPipeline()
            
            # Use the comprehensive preprocessing
            X_processed = pipeline.comprehensive_feature_preprocessing(game_df)
            
            # Ensure we have all expected features
            if self.feature_names:
                # Create a feature vector with all expected features
                feature_vector = []
                for feature_name in self.feature_names:
                    if feature_name in X_processed.columns:
                        value = X_processed[feature_name].iloc[0]
                        # Handle NaN values
                        if pd.isna(value):
                            value = 0
                        feature_vector.append(value)
                    else:
                        # Missing feature, use default value
                        feature_vector.append(0)
                
                return np.array(feature_vector).reshape(1, -1)
            else:
                # Fallback: use numeric columns
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                X_numeric = X_processed[numeric_cols].fillna(0)
                return X_numeric.values
                
        except Exception as e:
            logger.error(f"❌ Error preparing learning features: {e}")
            return None
    
    def generate_learning_prediction(self, game_row: pd.Series) -> Optional[float]:
        """Generate a learning model prediction for a single game"""
        try:
            # Get full feature set for this game
            full_game_df = self.get_full_game_features(game_row['game_id'])
            
            if full_game_df.empty:
                logger.warning(f"⚠️ No full features found for game {game_row['game_id']}")
                return None
            
            # Prepare features
            features = self.prepare_learning_features(full_game_df)
            
            if features is None:
                return None
                
            # Generate prediction
            prediction = self.learning_model.predict(features)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Error generating learning prediction for game {game_row['game_id']}: {e}")
            return None
    
    def run_comprehensive_backtest(self, days_back: int = 30) -> Dict:
        """
        Run a comprehensive backtest comparing original vs learning model predictions
        """
        logger.info(f"🚀 Starting comprehensive backtest for last {days_back} days...")
        
        if self.learning_model is None:
            logger.error("❌ Learning model not loaded")
            return {}
        
        # Get historical completed games
        historical_games = self.get_historical_completed_games(days_back)
        
        if historical_games.empty:
            logger.error("❌ No historical games found")
            return {}
        
        logger.info(f"🔮 Generating learning predictions for {len(historical_games)} games...")
        
        # Generate learning predictions for all games
        results = []
        
        for idx, game in historical_games.iterrows():
            try:
                # Generate learning prediction
                learning_pred = self.generate_learning_prediction(game)
                
                if learning_pred is None:
                    logger.warning(f"⚠️ Skipping game {game['game_id']} - could not generate learning prediction")
                    continue
                
                # Calculate performance metrics
                original_pred = game['predicted_total']
                actual_total = game['total_runs']
                market_total = game['market_total']
                
                # Prediction errors
                original_error = abs(original_pred - actual_total)
                learning_error = abs(learning_pred - actual_total)
                
                # Market comparison
                original_vs_market = original_pred - market_total
                learning_vs_market = learning_pred - market_total
                
                # Betting outcomes (simplified - over/under on market total)
                actual_over = actual_total > market_total
                original_pick_over = original_pred > market_total
                learning_pick_over = learning_pred > market_total
                
                original_correct = (original_pick_over == actual_over)
                learning_correct = (learning_pick_over == actual_over)
                
                game_result = {
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'actual_total': actual_total,
                    'market_total': market_total,
                    'original_prediction': original_pred,
                    'learning_prediction': learning_pred,
                    'original_error': original_error,
                    'learning_error': learning_error,
                    'original_vs_market': original_vs_market,
                    'learning_vs_market': learning_vs_market,
                    'original_correct_pick': original_correct,
                    'learning_correct_pick': learning_correct,
                    'learning_better_error': learning_error < original_error,
                    'prediction_difference': learning_pred - original_pred
                }
                
                results.append(game_result)
                
                if idx % 20 == 0:
                    logger.info(f"📈 Processed {idx + 1}/{len(historical_games)} games...")
                    
            except Exception as e:
                logger.error(f"❌ Error processing game {game['game_id']}: {e}")
                continue
        
        if not results:
            logger.error("❌ No valid results generated")
            return {}
        
        # Calculate comprehensive statistics
        results_df = pd.DataFrame(results)
        
        stats = self.calculate_comprehensive_stats(results_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results_{timestamp}.json"
        
        output = {
            'backtest_summary': stats,
            'individual_games': results,
            'metadata': {
                'games_analyzed': len(results),
                'date_range': f"{historical_games['date'].min()} to {historical_games['date'].max()}",
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        return output
    
    def calculate_comprehensive_stats(self, results_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance statistics"""
        
        # Basic accuracy metrics
        original_mae = results_df['original_error'].mean()
        learning_mae = results_df['learning_error'].mean()
        
        original_rmse = np.sqrt((results_df['original_error'] ** 2).mean())
        learning_rmse = np.sqrt((results_df['learning_error'] ** 2).mean())
        
        # Picking accuracy (over/under)
        original_pick_accuracy = results_df['original_correct_pick'].mean() * 100
        learning_pick_accuracy = results_df['learning_correct_pick'].mean() * 100
        
        # Head-to-head comparison
        learning_wins = results_df['learning_better_error'].sum()
        total_games = len(results_df)
        learning_win_rate = (learning_wins / total_games) * 100
        
        # Market comparison
        original_vs_market_mae = abs(results_df['original_vs_market']).mean()
        learning_vs_market_mae = abs(results_df['learning_vs_market']).mean()
        
        # Model agreement
        avg_prediction_diff = results_df['prediction_difference'].mean()
        prediction_diff_std = results_df['prediction_difference'].std()
        close_agreement = (abs(results_df['prediction_difference']) < 0.5).sum()
        close_agreement_rate = (close_agreement / total_games) * 100
        
        stats = {
            'prediction_accuracy': {
                'original_mae': round(original_mae, 3),
                'learning_mae': round(learning_mae, 3),
                'mae_improvement': round(original_mae - learning_mae, 3),
                'original_rmse': round(original_rmse, 3),
                'learning_rmse': round(learning_rmse, 3),
                'rmse_improvement': round(original_rmse - learning_rmse, 3)
            },
            'picking_accuracy': {
                'original_pick_accuracy': round(original_pick_accuracy, 1),
                'learning_pick_accuracy': round(learning_pick_accuracy, 1),
                'pick_accuracy_improvement': round(learning_pick_accuracy - original_pick_accuracy, 1)
            },
            'head_to_head': {
                'learning_wins': learning_wins,
                'original_wins': total_games - learning_wins,
                'learning_win_rate': round(learning_win_rate, 1),
                'total_games': total_games
            },
            'market_comparison': {
                'original_vs_market_mae': round(original_vs_market_mae, 3),
                'learning_vs_market_mae': round(learning_vs_market_mae, 3),
                'market_improvement': round(original_vs_market_mae - learning_vs_market_mae, 3)
            },
            'model_agreement': {
                'avg_prediction_difference': round(avg_prediction_diff, 3),
                'prediction_std': round(prediction_diff_std, 3),
                'close_agreement_count': close_agreement,
                'close_agreement_rate': round(close_agreement_rate, 1)
            }
        }
        
        return stats
    
    def print_results_summary(self, results: Dict):
        """Print a formatted summary of the backtest results"""
        if not results:
            print("❌ No results to display")
            return
            
        summary = results['backtest_summary']
        metadata = results['metadata']
        
        print(f"\n{'='*60}")
        print(f"🎯 COMPREHENSIVE LEARNING MODEL BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"📅 Period: {metadata['date_range']}")
        print(f"🎮 Games Analyzed: {metadata['games_analyzed']}")
        print(f"⏰ Generated: {metadata['generated_at']}")
        
        print(f"\n📊 PREDICTION ACCURACY")
        print(f"   Original MAE:  {summary['prediction_accuracy']['original_mae']}")
        print(f"   Learning MAE:  {summary['prediction_accuracy']['learning_mae']}")
        print(f"   Improvement:   {summary['prediction_accuracy']['mae_improvement']:+.3f}")
        
        print(f"\n🎯 PICKING ACCURACY (Over/Under)")
        print(f"   Original:      {summary['picking_accuracy']['original_pick_accuracy']:.1f}%")
        print(f"   Learning:      {summary['picking_accuracy']['learning_pick_accuracy']:.1f}%")
        print(f"   Improvement:   {summary['picking_accuracy']['pick_accuracy_improvement']:+.1f}%")
        
        print(f"\n⚔️  HEAD-TO-HEAD COMPARISON")
        print(f"   Learning Wins: {summary['head_to_head']['learning_wins']}")
        print(f"   Original Wins: {summary['head_to_head']['original_wins']}")
        print(f"   Learning Win Rate: {summary['head_to_head']['learning_win_rate']:.1f}%")
        
        print(f"\n📈 MARKET COMPARISON")
        print(f"   Original vs Market MAE: {summary['market_comparison']['original_vs_market_mae']}")
        print(f"   Learning vs Market MAE: {summary['market_comparison']['learning_vs_market_mae']}")
        print(f"   Market Improvement:     {summary['market_comparison']['market_improvement']:+.3f}")
        
        print(f"\n🤝 MODEL AGREEMENT")
        print(f"   Avg Difference:    {summary['model_agreement']['avg_prediction_difference']:+.3f}")
        print(f"   Close Agreement:   {summary['model_agreement']['close_agreement_rate']:.1f}%")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT")
        mae_better = summary['prediction_accuracy']['mae_improvement'] > 0
        pick_better = summary['picking_accuracy']['pick_accuracy_improvement'] > 0
        win_rate = summary['head_to_head']['learning_win_rate']
        
        if mae_better and pick_better and win_rate > 60:
            assessment = "🟢 LEARNING MODEL OUTPERFORMS ORIGINAL"
        elif mae_better or pick_better or win_rate > 55:
            assessment = "🟡 LEARNING MODEL SHOWS PROMISE"
        else:
            assessment = "🔴 ORIGINAL MODEL STILL SUPERIOR"
        
        print(f"   {assessment}")
        print(f"{'='*60}\n")

def main():
    """Run the comprehensive backtest"""
    try:
        backtester = ComprehensiveLearningBacktester()
        
        # Run backtest for last 30 days
        results = backtester.run_comprehensive_backtest(days_back=30)
        
        if results:
            backtester.print_results_summary(results)
            print("✅ Comprehensive backtest completed successfully!")
        else:
            print("❌ Backtest failed - no results generated")
            
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")

if __name__ == "__main__":
    main()
