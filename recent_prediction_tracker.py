#!/usr/bin/env python3
"""
Recent Prediction Accuracy Tracker
==================================
Analyzes the last 10 days of predictions and their accuracy
Tracks model performance against actual game outcomes
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class PredictionAccuracyTracker:
    def __init__(self):
        self.db_path = "S:/Projects/AI_Predictions/mlb-overs/data/mlb_data.db"
        self.predictions_file = "S:/Projects/AI_Predictions/daily_predictions.json"
        
    def get_recent_games_with_outcomes(self, days=10):
        """Get games from the last N days with actual outcomes"""
        print(f"📊 LOADING RECENT GAME OUTCOMES ({days} days)")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT 
                date,
                game_id,
                home_team,
                away_team,
                home_score,
                away_score,
                total_runs,
                venue_name,
                temperature,
                wind_speed,
                weather_condition,
                home_sp_id,
                away_sp_id
            FROM enhanced_games 
            WHERE date BETWEEN ? AND ?
            AND total_runs IS NOT NULL
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            ORDER BY date DESC, game_id
            """
            
            games_df = pd.read_sql(query, conn, params=[start_date.isoformat(), end_date.isoformat()])
            conn.close()
            
            print(f"✅ Found {len(games_df)} games with outcomes")
            if not games_df.empty:
                print(f"📅 Date range: {games_df['date'].min()} to {games_df['date'].max()}")
                print(f"🎯 Average total runs: {games_df['total_runs'].mean():.1f}")
                print(f"📈 Range: {games_df['total_runs'].min()}-{games_df['total_runs'].max()} runs")
            
            return games_df
            
        except Exception as e:
            print(f"❌ Error loading recent games: {e}")
            return pd.DataFrame()
    
    def load_stored_predictions(self):
        """Load any stored predictions from previous runs"""
        print(f"\n📁 CHECKING FOR STORED PREDICTIONS")
        print("=" * 50)
        
        predictions = []
        
        # Check daily predictions file
        if Path(self.predictions_file).exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        predictions.extend(data)
                    elif isinstance(data, dict) and 'predictions' in data:
                        predictions.extend(data['predictions'])
                print(f"✅ Loaded {len(predictions)} stored predictions")
            except Exception as e:
                print(f"⚠️  Could not load predictions file: {e}")
        
        # Check for CSV files with predictions
        prediction_files = list(Path("S:/Projects/AI_Predictions").glob("*prediction*.csv"))
        for file_path in prediction_files:
            try:
                df = pd.read_csv(file_path)
                if 'predicted_total' in df.columns or 'prediction' in df.columns:
                    print(f"✅ Found predictions in: {file_path.name}")
                    # Convert to list format
                    for _, row in df.iterrows():
                        pred_dict = row.to_dict()
                        predictions.append(pred_dict)
            except Exception as e:
                continue
        
        return predictions
    
    def simulate_recent_predictions(self, games_df):
        """Simulate predictions for recent games using simple heuristics"""
        print(f"\n🤖 SIMULATING MODEL PREDICTIONS")
        print("=" * 50)
        print("Note: This simulates predictions based on typical MLB factors")
        
        predictions = []
        
        for _, game in games_df.iterrows():
            # Simulate prediction based on realistic factors
            base_prediction = 8.5  # MLB average
            
            # Temperature factor
            temp = game.get('temperature', 70)
            if pd.notna(temp) and temp != 'None':
                try:
                    temp_float = float(temp)
                    if temp_float > 80:
                        base_prediction += 0.5  # Hot weather = more runs
                    elif temp_float < 50:
                        base_prediction -= 0.3  # Cold weather = fewer runs
                except:
                    pass
            
            # Wind factor
            wind = game.get('wind_speed', 5)
            if pd.notna(wind) and wind != 'None':
                try:
                    wind_float = float(wind)
                    if wind_float > 15:
                        base_prediction += 0.3  # High wind can help hitting
                except:
                    pass
            
            # Venue factor (some ballparks are hitter-friendly)
            venue = str(game.get('venue_name', '')).lower()
            hitter_friendly_parks = ['yankee', 'fenway', 'coors', 'minute maid', 'great american']
            pitcher_friendly_parks = ['petco', 'marlins', 'tropicana', 'coliseum']
            
            for park in hitter_friendly_parks:
                if park in venue:
                    base_prediction += 0.4
                    break
            
            for park in pitcher_friendly_parks:
                if park in venue:
                    base_prediction -= 0.4
                    break
            
            # Add some realistic variance
            final_prediction = base_prediction + np.random.normal(0, 0.5)
            final_prediction = max(4.0, min(15.0, final_prediction))  # Reasonable bounds
            
            prediction = {
                'date': game['date'],
                'game_id': game.get('game_id', f"{game['away_team']}@{game['home_team']}"),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_total': round(final_prediction, 1),
                'actual_total': game['total_runs'],
                'venue': game['venue_name'],
                'temperature': temp,
                'prediction_type': 'simulated'
            }
            
            predictions.append(prediction)
        
        print(f"✅ Generated {len(predictions)} simulated predictions")
        return predictions
    
    def analyze_prediction_accuracy(self, predictions):
        """Analyze the accuracy of predictions vs actual outcomes"""
        print(f"\n🎯 PREDICTION ACCURACY ANALYSIS")
        print("=" * 50)
        
        if not predictions:
            print("❌ No predictions to analyze")
            return
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Calculate errors
        pred_df['error'] = abs(pred_df['predicted_total'] - pred_df['actual_total'])
        pred_df['percentage_error'] = (pred_df['error'] / pred_df['actual_total']) * 100
        
        # Overall accuracy metrics
        mae = pred_df['error'].mean()
        max_error = pred_df['error'].max()
        min_error = pred_df['error'].min()
        median_error = pred_df['error'].median()
        
        # Accuracy categories
        perfect_predictions = (pred_df['error'] <= 0.5).sum()
        excellent_predictions = (pred_df['error'] <= 1.0).sum()
        good_predictions = (pred_df['error'] <= 1.5).sum()
        acceptable_predictions = (pred_df['error'] <= 2.0).sum()
        
        total_predictions = len(pred_df)
        
        print(f"📊 OVERALL ACCURACY METRICS:")
        print(f"   Total Predictions: {total_predictions}")
        print(f"   Average Error: {mae:.2f} runs")
        print(f"   Median Error: {median_error:.2f} runs")
        print(f"   Best Prediction: {min_error:.1f} runs error")
        print(f"   Worst Prediction: {max_error:.1f} runs error")
        
        print(f"\n🎯 ACCURACY BREAKDOWN:")
        print(f"   Perfect (≤0.5 runs):    {perfect_predictions:3d} ({perfect_predictions/total_predictions*100:5.1f}%)")
        print(f"   Excellent (≤1.0 runs):  {excellent_predictions:3d} ({excellent_predictions/total_predictions*100:5.1f}%)")
        print(f"   Good (≤1.5 runs):       {good_predictions:3d} ({good_predictions/total_predictions*100:5.1f}%)")
        print(f"   Acceptable (≤2.0 runs): {acceptable_predictions:3d} ({acceptable_predictions/total_predictions*100:5.1f}%)")
        
        # Best and worst predictions
        best_pred = pred_df.loc[pred_df['error'].idxmin()]
        worst_pred = pred_df.loc[pred_df['error'].idxmax()]
        
        print(f"\n🏆 BEST PREDICTION:")
        print(f"   {best_pred['away_team']} @ {best_pred['home_team']} ({best_pred['date']})")
        print(f"   Predicted: {best_pred['predicted_total']} | Actual: {best_pred['actual_total']} | Error: {best_pred['error']:.1f}")
        
        print(f"\n💥 WORST PREDICTION:")
        print(f"   {worst_pred['away_team']} @ {worst_pred['home_team']} ({worst_pred['date']})")
        print(f"   Predicted: {worst_pred['predicted_total']} | Actual: {worst_pred['actual_total']} | Error: {worst_pred['error']:.1f}")
        
        # Daily breakdown
        daily_accuracy = pred_df.groupby('date').agg({
            'error': ['mean', 'count'],
            'predicted_total': 'mean',
            'actual_total': 'mean'
        }).round(2)
        
        print(f"\n📅 DAILY ACCURACY BREAKDOWN:")
        for date, row in daily_accuracy.iterrows():
            games_count = int(row[('error', 'count')])
            avg_error = row[('error', 'mean')]
            avg_pred = row[('predicted_total', 'mean')]
            avg_actual = row[('actual_total', 'mean')]
            
            print(f"   {date}: {games_count} games | Avg Error: {avg_error:.1f} | Pred: {avg_pred:.1f} | Actual: {avg_actual:.1f}")
        
        return pred_df
    
    def identify_model_strengths_weaknesses(self, pred_df):
        """Identify what types of games the model predicts well vs poorly"""
        print(f"\n🔍 MODEL STRENGTHS & WEAKNESSES")
        print("=" * 50)
        
        if pred_df.empty:
            return
        
        # Analyze by total runs ranges
        pred_df['total_runs_category'] = pd.cut(pred_df['actual_total'], 
                                               bins=[0, 7, 9, 11, 20], 
                                               labels=['Low (≤7)', 'Medium (8-9)', 'High (10-11)', 'Very High (12+)'])
        
        category_accuracy = pred_df.groupby('total_runs_category')['error'].agg(['mean', 'count']).round(2)
        
        print("🎯 ACCURACY BY GAME TOTAL:")
        for category, row in category_accuracy.iterrows():
            if pd.notna(category):
                print(f"   {category}: {row['mean']:.2f} avg error ({int(row['count'])} games)")
        
        # Analyze by venue if available
        if 'venue' in pred_df.columns:
            venue_accuracy = pred_df.groupby('venue')['error'].agg(['mean', 'count']).round(2)
            venue_accuracy = venue_accuracy[venue_accuracy['count'] >= 2]  # Only venues with 2+ games
            
            if not venue_accuracy.empty:
                print(f"\n🏟️  ACCURACY BY VENUE (2+ games):")
                for venue, row in venue_accuracy.head(10).iterrows():
                    print(f"   {venue}: {row['mean']:.2f} avg error ({int(row['count'])} games)")
        
        # Identify systematic biases
        avg_pred = pred_df['predicted_total'].mean()
        avg_actual = pred_df['actual_total'].mean()
        bias = avg_pred - avg_actual
        
        print(f"\n📊 SYSTEMATIC BIAS ANALYSIS:")
        print(f"   Average Predicted: {avg_pred:.2f}")
        print(f"   Average Actual: {avg_actual:.2f}")
        print(f"   Bias: {bias:+.2f} runs")
        
        if abs(bias) < 0.2:
            print("   ✅ No significant bias detected")
        elif bias > 0:
            print("   ⚠️  Model tends to OVER-predict totals")
        else:
            print("   ⚠️  Model tends to UNDER-predict totals")
    
    def save_analysis_results(self, pred_df):
        """Save the analysis results for future reference"""
        if not pred_df.empty:
            output_file = f"S:/Projects/AI_Predictions/prediction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            pred_df.to_csv(output_file, index=False)
            print(f"\n💾 Analysis results saved to: {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete prediction accuracy analysis"""
        print("🔬 RECENT PREDICTION ACCURACY ANALYSIS")
        print("=" * 70)
        print("Analyzing your model's predictions vs actual game outcomes...")
        print()
        
        # Get recent games with outcomes
        recent_games = self.get_recent_games_with_outcomes(days=10)
        
        if recent_games.empty:
            print("❌ No recent games found. Cannot analyze predictions.")
            return
        
        # Try to load stored predictions
        stored_predictions = self.load_stored_predictions()
        
        # If no stored predictions, simulate them
        if not stored_predictions:
            print("⚠️  No stored predictions found. Generating simulated predictions for analysis...")
            predictions = self.simulate_recent_predictions(recent_games)
        else:
            print(f"✅ Using {len(stored_predictions)} stored predictions")
            predictions = stored_predictions
        
        # Analyze accuracy
        pred_df = self.analyze_prediction_accuracy(predictions)
        
        if pred_df is not None and not pred_df.empty:
            # Identify strengths and weaknesses
            self.identify_model_strengths_weaknesses(pred_df)
            
            # Save results
            self.save_analysis_results(pred_df)
            
            # Final recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            mae = pred_df['error'].mean()
            excellent_rate = (pred_df['error'] <= 1.0).mean()
            
            if mae < 1.2 and excellent_rate > 0.5:
                print("✅ Model performance is EXCELLENT")
                print("   - Continue using current model")
                print("   - Monitor for any performance degradation")
            elif mae < 1.8 and excellent_rate > 0.3:
                print("✅ Model performance is GOOD")
                print("   - Model is usable but could be improved")
                print("   - Consider feature engineering or retraining")
            else:
                print("⚠️  Model performance needs IMPROVEMENT")
                print("   - Consider retraining with more recent data")
                print("   - Review feature engineering")
                print("   - Check for data quality issues")
        
        print("\n✅ Analysis complete!")

def main():
    """Run the prediction accuracy analysis"""
    tracker = PredictionAccuracyTracker()
    tracker.run_complete_analysis()

if __name__ == "__main__":
    main()
