#!/usr/bin/env python3
"""
Live Prediction System with Continuous Learning
Integrates the continuously learning models with live game predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from daily_learning_pipeline import DailyLearningPipeline
import psycopg2
import pandas as pd
from datetime import datetime
import json

class LiveLearningPredictor:
    def __init__(self):
        self.pipeline = DailyLearningPipeline()
        
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser', 
            password='mlbpass'
        )
    
    def get_todays_games(self):
        """Get today's scheduled games"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        conn = self.get_db_connection()
        
        # Get games scheduled for today that haven't finished yet
        query = """
        SELECT 
            game_id, home_team, away_team, venue_name,
            temperature, humidity, wind_speed, air_pressure,
            home_sp_season_era, away_sp_season_era,
            home_sp_whip, away_sp_whip,
            home_sp_days_rest, away_sp_days_rest,
            home_team_avg, away_team_avg,
            ballpark_run_factor, ballpark_hr_factor,
            umpire_ou_tendency, market_total,
            day_night, getaway_day, doubleheader,
            home_sp_hand, away_sp_hand,
            predicted_total, confidence, recommendation, edge,
            home_score, away_score, total_runs
        FROM enhanced_games 
        WHERE date = %s
        ORDER BY home_team
        """
        
        df = pd.read_sql_query(query, conn, params=[today])
        conn.close()
        
        return df
    
    def prepare_game_features(self, game_row):
        """Prepare features for a single game"""
        
        features = [
            game_row['temperature'] or 70,
            game_row['humidity'] or 50,
            game_row['wind_speed'] or 5,
            game_row['air_pressure'] or 30.0,
            game_row['home_sp_season_era'] or 4.5,
            game_row['away_sp_season_era'] or 4.5,
            game_row['home_sp_whip'] or 1.3,
            game_row['away_sp_whip'] or 1.3,
            game_row['home_sp_days_rest'] or 4,
            game_row['away_sp_days_rest'] or 4,
            game_row['home_team_avg'] or 0.250,
            game_row['away_team_avg'] or 0.250,
            game_row['ballpark_run_factor'] or 1.0,
            game_row['ballpark_hr_factor'] or 1.0,
            game_row['umpire_ou_tendency'] or 0.0,
            1 if game_row['day_night'] == 'N' else 0,
            1 if game_row['getaway_day'] else 0,
            1 if game_row['doubleheader'] else 0,
            1 if game_row['home_sp_hand'] == 'L' and game_row['away_sp_hand'] == 'L' else 0,
            1 if game_row['home_sp_hand'] == 'R' and game_row['away_sp_hand'] == 'R' else 0,
            game_row['market_total'] or 8.5
        ]
        
        return features
    
    def get_learning_predictions_for_today(self):
        """Get learning model predictions for today's games"""
        
        print("🎯 CONTINUOUS LEARNING PREDICTIONS FOR TODAY")
        print("=" * 60)
        
        # Check if production model is available
        status = self.pipeline.get_model_status()
        if status.get('status') != 'Active':
            print("❌ No active learning model available")
            return
        
        print(f"🤖 Using model: {status['model_type']} (version: {status['version']})")
        print(f"📊 Model MAE: {status['performance'].get('mae', 'N/A'):.2f}")
        print()
        
        # Get today's games
        games_df = self.get_todays_games()
        
        if len(games_df) == 0:
            print("No games scheduled for today")
            return
        
        print(f"Found {len(games_df)} games scheduled for today\\n")
        
        learning_predictions = []
        
        for _, game in games_df.iterrows():
            # Prepare features
            features = self.prepare_game_features(game)
            
            # Get learning model prediction
            learning_result = self.pipeline.predict_with_production_model(features)
            
            if learning_result:
                learning_pred = learning_result['predicted_total']
                
                # Compare with current system prediction and market
                current_pred = game['predicted_total'] if pd.notna(game['predicted_total']) else None
                market_total = game['market_total'] if pd.notna(game['market_total']) else None
                
                # Calculate differences
                vs_current = None
                vs_market = None
                
                if current_pred:
                    vs_current = learning_pred - current_pred
                
                if market_total:
                    vs_market = learning_pred - market_total
                
                # Determine learning model recommendation
                learning_rec = "HOLD"
                learning_edge = 0
                
                if market_total:
                    edge = abs(learning_pred - market_total)
                    if edge >= 0.5:  # Minimum edge threshold
                        if learning_pred > market_total:
                            learning_rec = "OVER"
                            learning_edge = edge
                        else:
                            learning_rec = "UNDER"
                            learning_edge = edge
                
                prediction_data = {
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'venue': game['venue_name'],
                    'learning_prediction': learning_pred,
                    'current_prediction': current_pred,
                    'market_total': market_total,
                    'vs_current': vs_current,
                    'vs_market': vs_market,
                    'learning_recommendation': learning_rec,
                    'learning_edge': learning_edge,
                    'current_recommendation': game['recommendation'],
                    'model_version': learning_result['model_version'],
                    'is_completed': pd.notna(game['total_runs']),
                    'actual_total': game['total_runs'] if pd.notna(game['total_runs']) else None
                }
                
                learning_predictions.append(prediction_data)
        
        # Display results
        print("🎯 LEARNING MODEL vs CURRENT SYSTEM COMPARISON:")
        print("-" * 80)
        print(f"{'Game':<35} {'Learning':<8} {'Current':<7} {'Market':<7} {'Δ Curr':<7} {'Δ Mkt':<7} {'L.Rec':<5} {'Edge':<5}")
        print("-" * 80)
        
        learning_bets = []
        current_bets = []
        
        for pred in learning_predictions:
            game_str = pred['game'][:33]
            learning_pred = f"{pred['learning_prediction']:.1f}"
            current_pred = f"{pred['current_prediction']:.1f}" if pred['current_prediction'] else "N/A"
            market_pred = f"{pred['market_total']:.1f}" if pred['market_total'] else "N/A"
            vs_current = f"{pred['vs_current']:+.1f}" if pred['vs_current'] is not None else "N/A"
            vs_market = f"{pred['vs_market']:+.1f}" if pred['vs_market'] is not None else "N/A"
            learning_rec = pred['learning_recommendation']
            edge = f"{pred['learning_edge']:.1f}" if pred['learning_edge'] > 0 else ""
            
            # Color coding for recommendations
            if learning_rec in ['OVER', 'UNDER']:
                learning_bets.append(pred)
            
            current_rec = pred['current_recommendation']
            if current_rec in ['OVER', 'UNDER']:
                current_bets.append(pred)
            
            print(f"{game_str:<35} {learning_pred:<8} {current_pred:<7} {market_pred:<7} {vs_current:<7} {vs_market:<7} {learning_rec:<5} {edge:<5}")
        
        # Summary comparison
        print("\\n📊 BETTING RECOMMENDATIONS COMPARISON:")
        print(f"🤖 Learning Model Bets: {len(learning_bets)}")
        print(f"🔧 Current System Bets: {len(current_bets)}")
        
        if learning_bets:
            print("\\n🎯 LEARNING MODEL BETTING PICKS:")
            for bet in learning_bets:
                confidence_indicator = "🔥" if bet['learning_edge'] >= 1.0 else "⚡"
                print(f"  {confidence_indicator} {bet['game']}: {bet['learning_recommendation']} {bet['learning_prediction']:.1f} (vs market {bet['market_total']:.1f}, edge: {bet['learning_edge']:.1f})")
        
        # Performance tracking for completed games
        completed_games = [p for p in learning_predictions if p['is_completed']]
        if completed_games:
            print(f"\\n✅ COMPLETED GAMES PERFORMANCE ({len(completed_games)} games):")
            
            learning_errors = []
            current_errors = []
            
            for game in completed_games:
                actual = game['actual_total']
                learning_error = abs(game['learning_prediction'] - actual)
                learning_errors.append(learning_error)
                
                if game['current_prediction']:
                    current_error = abs(game['current_prediction'] - actual)
                    current_errors.append(current_error)
                
                vs_current_str = f"vs {game['current_prediction']:.1f}" if game['current_prediction'] else ""
                print(f"  {game['game']}: Learning {game['learning_prediction']:.1f} {vs_current_str} → Actual {actual} (error: {learning_error:.1f})")
            
            if learning_errors:
                learning_mae = sum(learning_errors) / len(learning_errors)
                print(f"\\n📈 Learning Model MAE: {learning_mae:.2f}")
                
                if current_errors:
                    current_mae = sum(current_errors) / len(current_errors)
                    improvement = current_mae - learning_mae
                    print(f"📈 Current System MAE: {current_mae:.2f}")
                    print(f"🎯 Learning Improvement: {improvement:+.2f} runs")
        
        return learning_predictions

def main():
    """Main execution"""
    
    predictor = LiveLearningPredictor()
    
    print("🤖 LIVE LEARNING PREDICTION SYSTEM")
    print("=" * 50)
    print("Comparing continuously learning models with current system")
    print()
    
    # Get learning predictions for today
    predictions = predictor.get_learning_predictions_for_today()
    
    if predictions:
        # Save predictions to file
        today = datetime.now().strftime('%Y-%m-%d')
        output_file = f"learning_predictions_{today}.json"
        
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f"\\n💾 Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()
