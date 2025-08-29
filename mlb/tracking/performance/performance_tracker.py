#!/usr/bin/env python3
"""
Daily Performance Tracker
=========================
Tracks the legitimate model's performance over time to validate the 84.5% accuracy claim.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track legitimate model performance over time"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or 'postgresql://mlbuser:mlbpass@localhost/mlb'
        self.engine = create_engine(self.db_url)
        self.predictions_dir = Path(__file__).parent
    
    def check_recent_performance(self, days_back: int = 7) -> dict:
        """Check performance of recent predictions"""
        logger.info(f"📊 Checking performance for last {days_back} days...")
        
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            # Get actual results
            query = text("""
                SELECT game_id, date, away_team, home_team, 
                       market_total, total_runs,
                       CASE 
                           WHEN total_runs > market_total THEN 'OVER'
                           ELSE 'UNDER'
                       END as actual_result
                FROM enhanced_games 
                WHERE date >= :start_date 
                AND date < :end_date
                AND total_runs IS NOT NULL
                AND market_total IS NOT NULL
                ORDER BY date DESC
            """)
            
            with self.engine.connect() as conn:
                actual_df = pd.read_sql(query, conn, params={
                    'start_date': start_date, 
                    'end_date': end_date
                })
            
            if actual_df.empty:
                return {'error': 'No completed games found in date range'}
            
            logger.info(f"📈 Found {len(actual_df)} completed games")
            
            # Load prediction files for the date range
            predictions_data = []
            
            for days_ago in range(days_back):
                pred_date = end_date - timedelta(days=days_ago)
                pred_file = self.predictions_dir / f"daily_predictions_{pred_date}.json"
                
                if pred_file.exists():
                    try:
                        with open(pred_file, 'r') as f:
                            pred_data = json.load(f)
                        
                        for pred in pred_data.get('predictions', []):
                            predictions_data.append({
                                'prediction_date': pred_date,
                                'game_id': pred.get('game_id'),
                                'away_team': pred.get('away_team'),
                                'home_team': pred.get('home_team'),
                                'predicted_total': pred.get('predicted_total'),
                                'market_total': pred.get('market_total'),
                                'over_under_pick': pred.get('over_under_pick'),
                                'confidence': pred.get('confidence')
                            })
                    except Exception as e:
                        logger.warning(f"Error loading {pred_file}: {e}")
            
            if not predictions_data:
                return {'error': 'No prediction files found'}
            
            pred_df = pd.DataFrame(predictions_data)
            logger.info(f"📋 Found {len(pred_df)} predictions")
            
            # Match predictions with actual results
            # Match by team names and date
            matched_results = []
            
            for _, pred_row in pred_df.iterrows():
                # Find matching actual game
                actual_match = actual_df[
                    (actual_df['away_team'].str.contains(pred_row['away_team'][:3], case=False, na=False)) &
                    (actual_df['home_team'].str.contains(pred_row['home_team'][:3], case=False, na=False)) &
                    (actual_df['date'] == pred_row['prediction_date'])
                ]
                
                if not actual_match.empty:
                    actual_game = actual_match.iloc[0]
                    
                    correct = pred_row['over_under_pick'] == actual_game['actual_result']
                    
                    matched_results.append({
                        'prediction_date': pred_row['prediction_date'],
                        'teams': f"{pred_row['away_team']} @ {pred_row['home_team']}",
                        'predicted_pick': pred_row['over_under_pick'],
                        'actual_result': actual_game['actual_result'],
                        'correct': correct,
                        'confidence': pred_row['confidence'],
                        'market_total': pred_row['market_total'],
                        'predicted_total': pred_row['predicted_total'],
                        'actual_total': actual_game['total_runs']
                    })
            
            if not matched_results:
                return {'error': 'No matching predictions and results found'}
            
            # Calculate performance metrics
            results_df = pd.DataFrame(matched_results)
            
            total_picks = len(results_df)
            correct_picks = results_df['correct'].sum()
            picking_accuracy = (correct_picks / total_picks) * 100
            
            # Performance by confidence
            conf_stats = results_df.groupby('confidence')['correct'].agg(['count', 'sum', 'mean']).reset_index()
            conf_stats['accuracy'] = conf_stats['mean'] * 100
            
            performance_summary = {
                'date_range': f"{start_date} to {end_date}",
                'total_predictions': total_picks,
                'correct_predictions': correct_picks,
                'picking_accuracy': round(picking_accuracy, 1),
                'by_confidence': conf_stats.to_dict('records'),
                'recent_games': results_df.tail(10).to_dict('records')
            }
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return {'error': str(e)}
    
    def display_performance(self, performance: dict):
        """Display performance results"""
        if 'error' in performance:
            print(f"❌ Error: {performance['error']}")
            return
        
        print(f"\n📊 Model Performance Report")
        print("=" * 50)
        print(f"Date Range: {performance['date_range']}")
        print(f"Total Predictions: {performance['total_predictions']}")
        print(f"Correct Predictions: {performance['correct_predictions']}")
        print(f"Picking Accuracy: {performance['picking_accuracy']}%")
        
        if performance['picking_accuracy'] >= 80:
            print("🏆 EXCELLENT performance!")
        elif performance['picking_accuracy'] >= 70:
            print("✅ GOOD performance!")
        elif performance['picking_accuracy'] >= 60:
            print("👍 DECENT performance!")
        else:
            print("⚠️ Below expectations")
        
        print("\n📈 Performance by Confidence:")
        for conf in performance['by_confidence']:
            print(f"   {conf['confidence']}: {conf['sum']}/{conf['count']} = {conf['accuracy']:.1f}%")
        
        print(f"\n🎯 Recent Games (last 10):")
        for game in performance['recent_games'][-5:]:  # Show last 5
            result_icon = "✅" if game['correct'] else "❌"
            print(f"   {result_icon} {game['teams']}: {game['predicted_pick']} -> {game['actual_result']}")

def main():
    """Main performance check"""
    print("📊 Legitimate Model Performance Tracker")
    print("=" * 50)
    
    tracker = PerformanceTracker()
    
    # Check last 7 days
    performance = tracker.check_recent_performance(days_back=7)
    tracker.display_performance(performance)
    
    # Also check last 14 days for more data
    print("\n" + "="*50)
    print("📊 Extended Performance (14 days)")
    performance_14 = tracker.check_recent_performance(days_back=14)
    tracker.display_performance(performance_14)

if __name__ == "__main__":
    main()
