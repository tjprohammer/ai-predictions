#!/usr/bin/env python3
"""
Simple Learning Model Performance Test

Test the retrained learning model against historical games
using the existing working learning model analyzer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import sys
import os
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model_analysis.learning_model_analyzer import LearningModelAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLearningModelTester:
    """Simple test of learning model performance"""
    
    def __init__(self):
        """Initialize the tester"""
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Use the existing learning model analyzer
        self.analyzer = LearningModelAnalyzer()
        
        logger.info("🔬 Simple Learning Model Tester initialized")
    
    def get_historical_predictions_comparison(self, start_date, end_date):
        """Get historical games and compare original vs learning model predictions"""
        logger.info(f"Testing period: {start_date} to {end_date}")
        
        # Get historical games with actual results
        with psycopg2.connect(**self.db_config) as conn:
            query = """
            SELECT 
                game_id,
                date,
                home_team,
                away_team,
                predicted_total as original_prediction,
                total_runs as actual_total,
                market_total,
                edge,
                confidence,
                recommendation
            FROM enhanced_games
            WHERE date >= %s 
              AND date <= %s
              AND total_runs IS NOT NULL
              AND predicted_total IS NOT NULL
            ORDER BY date, game_id
            """
            
            historical_df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        logger.info(f"Found {len(historical_df)} historical games")
        
        if len(historical_df) == 0:
            logger.error("No historical games found")
            return None
        
        # Get learning model predictions for the same period
        learning_results = self.analyzer.apply_learning_model_to_historical_games(start_date, end_date)
        
        if not learning_results or 'analysis' not in learning_results:
            logger.error("Failed to get learning model predictions")
            return None
        
        # Extract predictions from learning results
        if 'comparison_data' in learning_results:
            # Use comparison data if available
            learning_df = learning_results['comparison_data']
            learning_predictions_df = learning_df[['game_id', 'learning_model_prediction']].copy()
        else:
            logger.error("No comparison data in learning results")
            return None
        
        # Merge the data
        comparison_df = historical_df.merge(
            learning_predictions_df[['game_id', 'learning_model_prediction']], 
            on='game_id', 
            how='inner'
        )
        
        logger.info(f"Successfully matched {len(comparison_df)} games for comparison")
        
        if len(comparison_df) == 0:
            logger.error("No games matched between historical and learning predictions")
            return None
        
        return comparison_df
    
    def calculate_metrics(self, actual, predicted, model_name):
        """Calculate performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        return {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'count': len(actual)
        }
    
    def compare_models(self, comparison_df):
        """Compare original model vs learning model performance"""
        logger.info("🎯 COMPARING MODEL PERFORMANCE")
        logger.info("=" * 60)
        
        actual = comparison_df['actual_total']
        original = comparison_df['original_prediction']
        learning = comparison_df['learning_model_prediction']
        
        # Calculate metrics for both models
        original_metrics = self.calculate_metrics(actual, original, "Original Model")
        learning_metrics = self.calculate_metrics(actual, learning, "Learning Model")
        
        # Calculate improvements
        mae_improvement = ((original_metrics['mae'] - learning_metrics['mae']) / original_metrics['mae']) * 100
        rmse_improvement = ((original_metrics['rmse'] - learning_metrics['rmse']) / original_metrics['rmse']) * 100
        r2_improvement = ((learning_metrics['r2'] - original_metrics['r2']) / abs(original_metrics['r2'])) * 100
        
        # Print results
        logger.info(f"📊 Test Period: {comparison_df['date'].min()} to {comparison_df['date'].max()}")
        logger.info(f"🎮 Games Analyzed: {len(comparison_df)}")
        logger.info("")
        logger.info("📈 ORIGINAL MODEL:")
        logger.info(f"   MAE:  {original_metrics['mae']:.3f} runs")
        logger.info(f"   RMSE: {original_metrics['rmse']:.3f} runs")
        logger.info(f"   R²:   {original_metrics['r2']:.3f}")
        logger.info("")
        logger.info("🤖 LEARNING MODEL:")
        logger.info(f"   MAE:  {learning_metrics['mae']:.3f} runs")
        logger.info(f"   RMSE: {learning_metrics['rmse']:.3f} runs")
        logger.info(f"   R²:   {learning_metrics['r2']:.3f}")
        logger.info("")
        logger.info("📊 IMPROVEMENT:")
        logger.info(f"   MAE:  {mae_improvement:+.2f}%")
        logger.info(f"   RMSE: {rmse_improvement:+.2f}%")
        logger.info(f"   R²:   {r2_improvement:+.2f}%")
        logger.info("")
        
        if mae_improvement > 0:
            logger.info("✅ LEARNING MODEL IS BETTER!")
            logger.info(f"   Average improvement: {mae_improvement:.1f}% reduction in error")
        elif mae_improvement > -5:
            logger.info("⚖️ Models perform similarly")
        else:
            logger.info("❌ Original model is still better")
            logger.info("   Learning model needs more training or different features")
        
        logger.info("=" * 60)
        
        # Sample comparison
        logger.info("🔍 SAMPLE GAME COMPARISONS:")
        sample = comparison_df.head(5)
        for _, game in sample.iterrows():
            orig_error = abs(game['actual_total'] - game['original_prediction'])
            learn_error = abs(game['actual_total'] - game['learning_model_prediction'])
            better = "✅ Learning" if learn_error < orig_error else "❌ Original"
            
            logger.info(f"   {game['away_team']} @ {game['home_team']} ({game['date']})")
            logger.info(f"     Actual: {game['actual_total']} | Original: {game['original_prediction']:.1f} (err: {orig_error:.1f}) | Learning: {game['learning_model_prediction']:.1f} (err: {learn_error:.1f}) | {better}")
        
        return {
            'original_metrics': original_metrics,
            'learning_metrics': learning_metrics,
            'improvements': {
                'mae': mae_improvement,
                'rmse': rmse_improvement,
                'r2': r2_improvement
            },
            'comparison_df': comparison_df
        }
    
    def test_performance(self, days_back=15):
        """Test learning model performance over recent days"""
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"🔬 Testing learning model performance ({days_back} days)")
        
        # Get comparison data
        comparison_df = self.get_historical_predictions_comparison(start_date, end_date)
        
        if comparison_df is None:
            logger.error("Failed to get comparison data")
            return None
        
        # Compare models
        results = self.compare_models(comparison_df)
        
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Learning Model Performance Test')
    parser.add_argument('--days', type=int, default=15, 
                       help='Number of days back to test (default: 15)')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting simple learning model performance test ({args.days} days)")
    
    # Initialize tester
    tester = SimpleLearningModelTester()
    
    # Test performance
    results = tester.test_performance(args.days)
    
    if results:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("❌ Test failed")

if __name__ == "__main__":
    main()
