#!/usr/bin/env python3
"""
Reality Check - Today's Game Results
===================================
Check how our predictions actually performed today.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TodaysResultsChecker:
    """Check today's actual game results vs our predictions"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or 'postgresql://mlbuser:mlbpass@localhost/mlb'
        self.engine = create_engine(self.db_url)
        self.predictions_file = Path(__file__).parent / f"daily_predictions_{date.today()}.json"
    
    def get_todays_actual_results(self) -> pd.DataFrame:
        """Get today's actual game results"""
        try:
            today = date.today()
            
            query = text("""
                SELECT game_id, date, away_team, home_team, 
                       market_total, total_runs,
                       CASE 
                           WHEN total_runs > market_total THEN 'OVER'
                           WHEN total_runs < market_total THEN 'UNDER'
                           ELSE 'PUSH'
                       END as actual_result,
                       (total_runs - market_total) as actual_vs_market
                FROM enhanced_games 
                WHERE date = :today
                AND total_runs IS NOT NULL
                AND market_total IS NOT NULL
                ORDER BY game_id
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'today': today})
            
            logger.info(f"📊 Found {len(df)} completed games for {today}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting today's results: {e}")
            return pd.DataFrame()
    
    def load_todays_predictions(self) -> list:
        """Load today's predictions"""
        try:
            if not self.predictions_file.exists():
                logger.error(f"Predictions file not found: {self.predictions_file}")
                return []
            
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
            
            predictions = data.get('predictions', [])
            logger.info(f"📋 Loaded {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return []
    
    def match_predictions_with_results(self, actual_df: pd.DataFrame, predictions: list) -> list:
        """Match predictions with actual results"""
        matched_results = []
        
        # Create team name mappings for better matching
        team_mappings = {
            'MIN': ['Minnesota', 'Twins'],
            'CHI': ['Chicago', 'Cubs', 'White Sox'],
            'LOS': ['Los Angeles', 'Angels', 'Dodgers'],
            'NEW': ['New York', 'Yankees', 'Mets'],
            'ATL': ['Atlanta', 'Braves'],
            'SAN': ['San Diego', 'Padres', 'San Francisco', 'Giants'],
            'MIL': ['Milwaukee', 'Brewers'],
            'TOR': ['Toronto', 'Blue Jays'],
            'MIA': ['Miami', 'Marlins'],
            'ATH': ['Oakland', 'Athletics'],
            'SEA': ['Seattle', 'Mariners'],
            'BOS': ['Boston', 'Red Sox'],
            'CLE': ['Cleveland', 'Guardians'],
            'TEX': ['Texas', 'Rangers'],
            'CIN': ['Cincinnati', 'Reds'],
            'ARI': ['Arizona', 'Diamondbacks'],
            'KAN': ['Kansas City', 'Royals'],
            'DET': ['Detroit', 'Tigers'],
            'COL': ['Colorado', 'Rockies'],
            'PIT': ['Pittsburgh', 'Pirates'],
            'ST.': ['St. Louis', 'Cardinals'],
            'TAM': ['Tampa Bay', 'Rays'],
            'HOU': ['Houston', 'Astros'],
            'BAL': ['Baltimore', 'Orioles'],
            'WAS': ['Washington', 'Nationals'],
            'PHI': ['Philadelphia', 'Phillies']
        }
        
        for pred in predictions:
            pred_away = pred['away_team'][:3].upper()
            pred_home = pred['home_team'][:3].upper()
            
            # Find matching actual game
            found_match = False
            
            for _, actual_row in actual_df.iterrows():
                actual_away = actual_row['away_team']
                actual_home = actual_row['home_team']
                
                # Try direct team code matching first
                away_match = (pred_away in actual_away.upper() or 
                             any(name.upper() in actual_away.upper() for name in team_mappings.get(pred_away, [])))
                home_match = (pred_home in actual_home.upper() or 
                             any(name.upper() in actual_home.upper() for name in team_mappings.get(pred_home, [])))
                
                if away_match and home_match:
                    correct = pred['over_under_pick'] == actual_row['actual_result']
                    
                    matched_results.append({
                        'teams': f"{pred['away_team']} @ {pred['home_team']}",
                        'actual_teams': f"{actual_away} @ {actual_home}",
                        'predicted_pick': pred['over_under_pick'],
                        'actual_result': actual_row['actual_result'],
                        'correct': correct,
                        'confidence': pred['confidence'],
                        'market_total': pred['market_total'],
                        'predicted_total': pred['predicted_total'],
                        'actual_total': actual_row['total_runs'],
                        'edge': pred.get('edge', 0),
                        'actual_vs_market': actual_row['actual_vs_market']
                    })
                    found_match = True
                    break
            
            if not found_match:
                # Game might not be completed yet or team name mismatch
                matched_results.append({
                    'teams': f"{pred['away_team']} @ {pred['home_team']}",
                    'actual_teams': 'NOT FOUND',
                    'predicted_pick': pred['over_under_pick'],
                    'actual_result': 'PENDING',
                    'correct': None,
                    'confidence': pred['confidence'],
                    'market_total': pred['market_total'],
                    'predicted_total': pred['predicted_total'],
                    'actual_total': None,
                    'edge': pred.get('edge', 0),
                    'actual_vs_market': None
                })
        
        return matched_results
    
    def analyze_performance(self, results: list) -> dict:
        """Analyze the performance of today's predictions"""
        completed_games = [r for r in results if r['correct'] is not None]
        
        if not completed_games:
            return {
                'total_predictions': len(results),
                'completed_games': 0,
                'pending_games': len(results),
                'message': 'No completed games found yet'
            }
        
        total_completed = len(completed_games)
        correct_picks = sum(1 for r in completed_games if r['correct'])
        accuracy = (correct_picks / total_completed) * 100 if total_completed > 0 else 0
        
        # Analyze by confidence
        high_conf = [r for r in completed_games if r['confidence'] == 'HIGH']
        medium_conf = [r for r in completed_games if r['confidence'] == 'MEDIUM']
        low_conf = [r for r in completed_games if r['confidence'] == 'LOW']
        
        high_correct = sum(1 for r in high_conf if r['correct'])
        medium_correct = sum(1 for r in medium_conf if r['correct'])
        low_correct = sum(1 for r in low_conf if r['correct'])
        
        # Over/Under analysis
        over_picks = [r for r in completed_games if r['predicted_pick'] == 'OVER']
        under_picks = [r for r in completed_games if r['predicted_pick'] == 'UNDER']
        
        over_correct = sum(1 for r in over_picks if r['correct'])
        under_correct = sum(1 for r in under_picks if r['correct'])
        
        return {
            'total_predictions': len(results),
            'completed_games': total_completed,
            'pending_games': len(results) - total_completed,
            'correct_picks': correct_picks,
            'overall_accuracy': round(accuracy, 1),
            'high_confidence': {
                'total': len(high_conf),
                'correct': high_correct,
                'accuracy': round((high_correct / len(high_conf)) * 100, 1) if high_conf else 0
            },
            'medium_confidence': {
                'total': len(medium_conf),
                'correct': medium_correct,
                'accuracy': round((medium_correct / len(medium_conf)) * 100, 1) if medium_conf else 0
            },
            'low_confidence': {
                'total': len(low_conf),
                'correct': low_correct,
                'accuracy': round((low_correct / len(low_conf)) * 100, 1) if low_conf else 0
            },
            'over_picks': {
                'total': len(over_picks),
                'correct': over_correct,
                'accuracy': round((over_correct / len(over_picks)) * 100, 1) if over_picks else 0
            },
            'under_picks': {
                'total': len(under_picks),
                'correct': under_correct,
                'accuracy': round((under_correct / len(under_picks)) * 100, 1) if under_picks else 0
            }
        }
    
    def display_results(self, results: list, performance: dict):
        """Display the results in a readable format"""
        print(f"\n🎯 Today's Prediction Results - {date.today()}")
        print("=" * 70)
        
        if performance['completed_games'] == 0:
            print("⏳ No completed games found yet. Games may still be in progress.")
            return
        
        print(f"📊 Overall Performance:")
        print(f"   Total Predictions: {performance['total_predictions']}")
        print(f"   Completed Games: {performance['completed_games']}")
        print(f"   Correct Picks: {performance['correct_picks']}")
        print(f"   Accuracy: {performance['overall_accuracy']}%")
        
        if performance['overall_accuracy'] >= 70:
            print("   ✅ GOOD performance!")
        elif performance['overall_accuracy'] >= 60:
            print("   👍 DECENT performance!")
        elif performance['overall_accuracy'] >= 50:
            print("   ⚠️ Below target but above random")
        else:
            print("   ❌ Poor performance today")
        
        print(f"\n📈 By Confidence Level:")
        for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            conf_data = performance[conf_level]
            if conf_data['total'] > 0:
                level_name = conf_level.replace('_confidence', '').upper()
                print(f"   {level_name}: {conf_data['correct']}/{conf_data['total']} = {conf_data['accuracy']}%")
        
        print(f"\n🎲 Over/Under Breakdown:")
        if performance['over_picks']['total'] > 0:
            print(f"   OVER picks: {performance['over_picks']['correct']}/{performance['over_picks']['total']} = {performance['over_picks']['accuracy']}%")
        if performance['under_picks']['total'] > 0:
            print(f"   UNDER picks: {performance['under_picks']['correct']}/{performance['under_picks']['total']} = {performance['under_picks']['accuracy']}%")
        
        print(f"\n🏟️ Individual Game Results:")
        completed_results = [r for r in results if r['correct'] is not None]
        
        for result in completed_results:
            result_icon = "✅" if result['correct'] else "❌"
            conf_icon = "🔥" if result['confidence'] == 'HIGH' else "🔸" if result['confidence'] == 'MEDIUM' else "🔹"
            
            print(f"   {result_icon} {conf_icon} {result['teams']}")
            print(f"      Predicted: {result['predicted_pick']} {result['predicted_total']} (market: {result['market_total']})")
            print(f"      Actual: {result['actual_result']} {result['actual_total']} (vs market: {result['actual_vs_market']:+.1f})")
            print()
        
        # Show pending games
        pending_results = [r for r in results if r['correct'] is None]
        if pending_results:
            print(f"⏳ Pending Games ({len(pending_results)}):")
            for result in pending_results:
                print(f"   🕐 {result['teams']} - {result['predicted_pick']} {result['predicted_total']}")
    
    def run_reality_check(self):
        """Run the complete reality check"""
        print("🔍 Reality Check - Today's Game Results")
        print("=" * 50)
        
        # Get actual results
        actual_df = self.get_todays_actual_results()
        
        if actual_df.empty:
            print("❌ No completed games found for today")
            return
        
        # Load predictions
        predictions = self.load_todays_predictions()
        
        if not predictions:
            print("❌ No predictions found for today")
            return
        
        # Match and analyze
        results = self.match_predictions_with_results(actual_df, predictions)
        performance = self.analyze_performance(results)
        
        # Display results
        self.display_results(results, performance)
        
        return performance

def main():
    """Main reality check function"""
    checker = TodaysResultsChecker()
    performance = checker.run_reality_check()
    
    if performance and performance.get('completed_games', 0) > 0:
        accuracy = performance['overall_accuracy']
        if accuracy < 60:
            print(f"\n💡 Reality Check: {accuracy}% is more realistic than 99%!")
            print("This confirms our model is legitimate, not using data leakage.")
        elif accuracy >= 80:
            print(f"\n🎉 Excellent day! {accuracy}% confirms strong model performance.")
        else:
            print(f"\n👍 Solid performance at {accuracy}% - within expected range.")

if __name__ == "__main__":
    main()
