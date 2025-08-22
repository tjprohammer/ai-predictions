#!/usr/bin/env python3
"""
Test Learning Model Performance Against Historical Games

This script tests whether our retrained learning model actually performs better
than the original predictions on historical games. It's the key validation
to see if the learning model is actually improving our prediction accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import sys
import os
from pathlib import Path
import logging
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from deployment.enhanced_bullpen_predictor import EnhancedBullpenPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LearningModelPerformanceTester:
    """Test learning model performance against historical data"""
    
    def __init__(self):
        """Initialize the performance tester"""
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Load the latest retrained learning model
        self.model = self._load_latest_learning_model()
        self.feature_engineer = EnhancedBullpenPredictor()
        
        logger.info("🔬 Learning Model Performance Tester initialized")
        if self.model:
            logger.info("✅ Learning model loaded successfully")
        else:
            logger.warning("⚠️ No learning model found")
    
    def _load_latest_learning_model(self):
        """Load the most recent retrained learning model"""
        # Check both model_analysis directory and models directory
        model_dirs = [
            Path(__file__).parent,  # model_analysis/
            Path(__file__).parent.parent / "models"  # models/
        ]
        
        pattern = "learning_model_retrained_*.joblib"
        model_files = []
        
        for model_dir in model_dirs:
            model_files.extend(list(model_dir.glob(pattern)))
        
        if not model_files:
            logger.warning("No retrained learning models found")
            return None
        
        # Get the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading learning model: {latest_model}")
        
        try:
            return joblib.load(latest_model)
        except Exception as e:
            logger.error(f"Failed to load learning model: {e}")
            return None
    
    def get_historical_games_with_predictions(self, start_date, end_date):
        """Get historical games with both original predictions and actual results"""
        logger.info(f"Fetching historical games from {start_date} to {end_date}")
        
        with psycopg2.connect(**self.db_config) as conn:
            query = """
            SELECT 
                game_id,
                date,
                home_team,
                away_team,
                predicted_total,
                market_total,
                total_runs,
                edge,
                confidence,
                recommendation,
                -- Get all the raw data needed for feature engineering
                home_score,
                away_score,
                temperature,
                wind_speed,
                wind_direction,
                humidity,
                air_pressure,
                weather_condition as conditions,
                venue_name,
                home_sp_id as home_starter_id,
                away_sp_id as away_starter_id
            FROM enhanced_games
            WHERE date >= %s 
              AND date <= %s
              AND total_runs IS NOT NULL
              AND predicted_total IS NOT NULL
            ORDER BY date, game_id
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            logger.info(f"Found {len(df)} historical games with complete data")
            return df
    
    def prepare_features_for_games(self, games_df):
        """Prepare features for learning model prediction"""
        logger.info("Preparing features for learning model...")
        
        all_features = []
        
        for _, game in games_df.iterrows():
            try:
                # Create a minimal game record for feature engineering
                game_record = {
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'temperature': game['temperature'],
                    'wind_speed': game['wind_speed'],
                    'wind_direction': game['wind_direction'],
                    'humidity': game['humidity'],
                    'pressure': game['air_pressure'],
                    'conditions': game['conditions'],
                    'venue_name': game['venue_name'],
                    'home_starter_id': game['home_starter_id'],
                    'away_starter_id': game['away_starter_id']
                }
                
                # Generate features using the same pipeline as production
                features = self.feature_engineer.create_features([game_record])
                
                if features is not None and len(features) > 0:
                    # Add game identifier
                    features_dict = features.iloc[0].to_dict()
                    features_dict['game_id'] = game['game_id']
                    all_features.append(features_dict)
                
            except Exception as e:
                logger.warning(f"Failed to generate features for game {game['game_id']}: {e}")
                continue
        
        if not all_features:
            logger.error("No features generated for any games")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(all_features)
        logger.info(f"Generated features for {len(features_df)} games")
        return features_df
    
    def compare_predictions(self, games_df, days_back=30):
        """Compare learning model predictions vs original predictions"""
        logger.info("🔬 Starting prediction comparison analysis...")
        
        if self.model is None:
            logger.error("No learning model available for comparison")
            return None
        
        # Prepare features for learning model
        features_df = self.prepare_features_for_games(games_df)
        
        if len(features_df) == 0:
            logger.error("No features generated - cannot perform comparison")
            return None
        
        # Merge features with game data
        comparison_df = games_df.merge(
            features_df[['game_id']], 
            on='game_id', 
            how='inner'
        )
        
        if len(comparison_df) == 0:
            logger.error("No games matched after feature merge")
            return None
        
        # Get features for prediction (excluding game_id)
        feature_cols = [col for col in features_df.columns if col != 'game_id']
        X_features = features_df[feature_cols]
        
        logger.info(f"Making learning model predictions with {len(feature_cols)} features...")
        
        try:
            # Make learning model predictions
            learning_predictions = self.model.predict(X_features)
            
            # Add learning predictions to comparison dataframe
            comparison_df = comparison_df.copy()
            comparison_df['learning_model_prediction'] = learning_predictions[:len(comparison_df)]
            
            # Calculate metrics for both models
            actual = comparison_df['total_runs']
            original = comparison_df['predicted_total']
            learning = comparison_df['learning_model_prediction']
            
            # Original model metrics
            original_mae = mean_absolute_error(actual, original)
            original_rmse = np.sqrt(mean_squared_error(actual, original))
            original_r2 = r2_score(actual, original)
            
            # Learning model metrics
            learning_mae = mean_absolute_error(actual, learning)
            learning_rmse = np.sqrt(mean_squared_error(actual, learning))
            learning_r2 = r2_score(actual, learning)
            
            # Calculate improvement
            mae_improvement = ((original_mae - learning_mae) / original_mae) * 100
            rmse_improvement = ((original_rmse - learning_rmse) / original_rmse) * 100
            r2_improvement = ((learning_r2 - original_r2) / abs(original_r2)) * 100
            
            # Print comparison results
            logger.info("=" * 80)
            logger.info("🎯 PREDICTION COMPARISON RESULTS")
            logger.info("=" * 80)
            logger.info(f"Test Period: {comparison_df['date'].min()} to {comparison_df['date'].max()}")
            logger.info(f"Games Analyzed: {len(comparison_df)}")
            logger.info("")
            logger.info("📊 ORIGINAL MODEL PERFORMANCE:")
            logger.info(f"   MAE:  {original_mae:.3f} runs")
            logger.info(f"   RMSE: {original_rmse:.3f} runs") 
            logger.info(f"   R²:   {original_r2:.3f}")
            logger.info("")
            logger.info("🤖 LEARNING MODEL PERFORMANCE:")
            logger.info(f"   MAE:  {learning_mae:.3f} runs")
            logger.info(f"   RMSE: {learning_rmse:.3f} runs")
            logger.info(f"   R²:   {learning_r2:.3f}")
            logger.info("")
            logger.info("📈 IMPROVEMENT ANALYSIS:")
            logger.info(f"   MAE Improvement:  {mae_improvement:+.1f}%")
            logger.info(f"   RMSE Improvement: {rmse_improvement:+.1f}%")
            logger.info(f"   R² Improvement:   {r2_improvement:+.1f}%")
            logger.info("")
            
            if mae_improvement > 0:
                logger.info("✅ LEARNING MODEL IS PERFORMING BETTER!")
            else:
                logger.info("❌ Learning model needs improvement")
            
            logger.info("=" * 80)
            
            return {
                'comparison_df': comparison_df,
                'original_metrics': {
                    'mae': original_mae,
                    'rmse': original_rmse,
                    'r2': original_r2
                },
                'learning_metrics': {
                    'mae': learning_mae,
                    'rmse': learning_rmse,
                    'r2': learning_r2
                },
                'improvements': {
                    'mae': mae_improvement,
                    'rmse': rmse_improvement,
                    'r2': r2_improvement
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction comparison failed: {e}")
            return None
    
    def create_comparison_visualizations(self, results):
        """Create visualizations comparing the two models"""
        if not results:
            return
        
        comparison_df = results['comparison_df']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Model vs Original Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(comparison_df['total_runs'], comparison_df['predicted_total'], 
                   alpha=0.6, label='Original Model', color='blue')
        ax1.scatter(comparison_df['total_runs'], comparison_df['learning_model_prediction'], 
                   alpha=0.6, label='Learning Model', color='red')
        ax1.plot([0, 20], [0, 20], 'k--', alpha=0.5)
        ax1.set_xlabel('Actual Total Runs')
        ax1.set_ylabel('Predicted Total Runs')
        ax1.set_title('Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error distribution
        ax2 = axes[0, 1]
        original_errors = comparison_df['total_runs'] - comparison_df['predicted_total']
        learning_errors = comparison_df['total_runs'] - comparison_df['learning_model_prediction']
        
        ax2.hist(original_errors, bins=20, alpha=0.6, label='Original Model', color='blue')
        ax2.hist(learning_errors, bins=20, alpha=0.6, label='Learning Model', color='red')
        ax2.set_xlabel('Prediction Error (Actual - Predicted)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance over time
        ax3 = axes[1, 0]
        comparison_df['date'] = pd.to_datetime(comparison_df['date'])
        daily_orig_mae = comparison_df.groupby('date').apply(
            lambda x: mean_absolute_error(x['total_runs'], x['predicted_total'])
        )
        daily_learn_mae = comparison_df.groupby('date').apply(
            lambda x: mean_absolute_error(x['total_runs'], x['learning_model_prediction'])
        )
        
        ax3.plot(daily_orig_mae.index, daily_orig_mae.values, 
                label='Original Model', color='blue', linewidth=2)
        ax3.plot(daily_learn_mae.index, daily_learn_mae.values, 
                label='Learning Model', color='red', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Daily MAE')
        ax3.set_title('Performance Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Metrics comparison bar chart
        ax4 = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'R²']
        original_values = [
            results['original_metrics']['mae'],
            results['original_metrics']['rmse'],
            results['original_metrics']['r2']
        ]
        learning_values = [
            results['learning_metrics']['mae'],
            results['learning_metrics']['rmse'],
            results['learning_metrics']['r2']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, original_values, width, label='Original Model', color='blue', alpha=0.7)
        ax4.bar(x + width/2, learning_values, width, label='Learning Model', color='red', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"learning_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comparison plot saved as: {plot_filename}")
        
        plt.show()

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Learning Model Performance')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days back to test (default: 30)')
    parser.add_argument('--start-date', type=str,
                       help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for testing (YYYY-MM-DD)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create comparison visualizations')
    
    args = parser.parse_args()
    
    # Calculate date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    logger.info(f"🔬 Testing learning model performance from {start_date} to {end_date}")
    
    # Initialize tester
    tester = LearningModelPerformanceTester()
    
    # Get historical games
    games_df = tester.get_historical_games_with_predictions(start_date, end_date)
    
    if len(games_df) == 0:
        logger.error("No historical games found for testing")
        return
    
    # Compare predictions
    results = tester.compare_predictions(games_df, args.days)
    
    if results and args.visualize:
        tester.create_comparison_visualizations(results)

if __name__ == "__main__":
    main()
