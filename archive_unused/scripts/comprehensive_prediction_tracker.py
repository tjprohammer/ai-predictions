#!/usr/bin/env python3
"""
Comprehensive Prediction Tracking System
========================================

This system tracks and displays:
- Market totals from betting lines
- Actual game scores and outcomes
- Model predictions vs reality
- Pitcher matchup analysis
- Historical accuracy metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import json
from sqlalchemy import create_engine, text
import requests
from typing import Dict, List, Optional

class ComprehensivePredictionTracker:
    def __init__(self):
        self.engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb', echo=False)
        self.tracking_data = []
        
    def get_market_totals_and_predictions(self):
        """Get current market totals and model predictions"""
        print("📊 Fetching market totals and model predictions...")
        
        # Load today's predictions with market totals
        predictions_file = 'S:/Projects/AI_Predictions/daily_predictions.json'
        betting_odds_file = 'S:/Projects/AI_Predictions/betting_odds_today.json'
        
        try:
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            with open(betting_odds_file, 'r') as f:
                betting_odds = json.load(f)
            
            print(f"✅ Loaded {len(predictions)} predictions and {len(betting_odds)} betting lines")
            
            # Combine predictions with betting odds
            combined_data = []
            for pred in predictions:
                game_id = pred.get('game_id')
                
                # Find matching betting odds
                matching_odds = None
                for odds in betting_odds:
                    if (odds.get('home_team') == pred.get('home_team') and 
                        odds.get('away_team') == pred.get('away_team')):
                        matching_odds = odds
                        break
                
                combined_data.append({
                    'game_id': game_id,
                    'date': pred.get('date'),
                    'home_team': pred.get('home_team'),
                    'away_team': pred.get('away_team'),
                    'home_pitcher': pred.get('home_pitcher', 'Unknown'),
                    'away_pitcher': pred.get('away_pitcher', 'Unknown'),
                    'model_prediction': pred.get('predicted_total'),
                    'model_confidence': pred.get('confidence'),
                    'model_recommendation': pred.get('recommendation'),
                    'market_total': matching_odds.get('total') if matching_odds else 8.5,
                    'over_odds': matching_odds.get('over_odds') if matching_odds else -110,
                    'under_odds': matching_odds.get('under_odds') if matching_odds else -110,
                    'venue': pred.get('venue'),
                    'weather': pred.get('weather_condition'),
                    'temperature': pred.get('temperature')
                })
            
            return combined_data
            
        except Exception as e:
            print(f"❌ Error loading prediction files: {e}")
            return []
    
    def get_completed_game_results(self, days_back=7):
        """Get results for completed games in the last N days"""
        print(f"🏟️ Fetching completed game results from last {days_back} days...")
        
        with self.engine.begin() as conn:
            query = """
            SELECT 
                date, game_id, home_team, away_team, home_score, away_score, 
                total_runs, venue_name, temperature, weather_condition,
                home_sp_id, away_sp_id, home_sp_er, away_sp_er, 
                home_sp_ip, away_sp_ip, home_sp_k, away_sp_k
            FROM enhanced_games 
            WHERE total_runs IS NOT NULL 
                AND home_score IS NOT NULL 
                AND away_score IS NOT NULL
                AND date >= CURRENT_DATE - INTERVAL ':days_back days'
            ORDER BY date DESC
            """
            
            completed_games = pd.read_sql(text(query), conn, params={'days_back': days_back})
            
            if completed_games.empty:
                print("❌ No completed games found")
                return []
            
            print(f"✅ Found {len(completed_games)} completed games")
            
            # Convert to list of dictionaries
            results = []
            for _, game in completed_games.iterrows():
                results.append({
                    'date': game['date'].strftime('%Y-%m-%d'),
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_pitcher_id': game['home_sp_id'],
                    'away_pitcher_id': game['away_sp_id'],
                    'actual_home_score': int(game['home_score']) if pd.notna(game['home_score']) else 0,
                    'actual_away_score': int(game['away_score']) if pd.notna(game['away_score']) else 0,
                    'actual_total': int(game['total_runs']) if pd.notna(game['total_runs']) else 0,
                    'venue': game['venue_name'],
                    'weather': game['weather_condition'],
                    'temperature': game['temperature'],
                    'home_pitcher_era': round((game['home_sp_er'] * 9) / game['home_sp_ip'], 2) if pd.notna(game['home_sp_ip']) and game['home_sp_ip'] > 0 else 0,
                    'away_pitcher_era': round((game['away_sp_er'] * 9) / game['away_sp_ip'], 2) if pd.notna(game['away_sp_ip']) and game['away_sp_ip'] > 0 else 0,
                    'home_pitcher_k': int(game['home_sp_k']) if pd.notna(game['home_sp_k']) else 0,
                    'away_pitcher_k': int(game['away_sp_k']) if pd.notna(game['away_sp_k']) else 0
                })
            
            return results
    
    def load_historical_predictions(self, days_back=7):
        """Load historical predictions for comparison"""
        print(f"📈 Loading historical predictions from last {days_back} days...")
        
        # This would load from a historical predictions database or files
        # For now, we'll simulate some historical predictions
        historical_predictions = []
        
        # Try to load from a historical predictions file
        try:
            with open('S:/Projects/AI_Predictions/historical_predictions.json', 'r') as f:
                historical_data = json.load(f)
                
            # Filter to last N days
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            for pred in historical_data:
                if pred.get('date', '') >= cutoff_date:
                    historical_predictions.append(pred)
                    
            print(f"✅ Loaded {len(historical_predictions)} historical predictions")
            
        except FileNotFoundError:
            print("📝 No historical predictions file found, will create one")
            
        return historical_predictions
    
    def match_predictions_with_results(self, predictions, results):
        """Match historical predictions with actual game results"""
        print("🔗 Matching predictions with actual results...")
        
        matched_data = []
        
        for result in results:
            # Find matching prediction
            matching_pred = None
            for pred in predictions:
                if (pred.get('home_team') == result['home_team'] and 
                    pred.get('away_team') == result['away_team'] and
                    pred.get('date') == result['date']):
                    matching_pred = pred
                    break
            
            if matching_pred:
                # Calculate prediction accuracy
                predicted_total = matching_pred.get('model_prediction', 0)
                actual_total = result['actual_total']
                prediction_error = abs(predicted_total - actual_total)
                percentage_error = (prediction_error / actual_total * 100) if actual_total > 0 else 0
                
                # Determine if market total was correct
                market_total = matching_pred.get('market_total', 8.5)
                market_correct = "OVER" if actual_total > market_total else "UNDER"
                model_correct = matching_pred.get('model_recommendation', 'UNKNOWN')
                recommendation_accurate = (market_correct == model_correct)
                
                matched_data.append({
                    'date': result['date'],
                    'game': f"{result['away_team']} @ {result['home_team']}",
                    'home_team': result['home_team'],
                    'away_team': result['away_team'],
                    'venue': result['venue'],
                    'weather': result['weather'],
                    'temperature': result['temperature'],
                    'home_pitcher_id': result['home_pitcher_id'],
                    'away_pitcher_id': result['away_pitcher_id'],
                    'home_pitcher_era': result['home_pitcher_era'],
                    'away_pitcher_era': result['away_pitcher_era'],
                    'actual_home_score': result['actual_home_score'],
                    'actual_away_score': result['actual_away_score'],
                    'actual_total': actual_total,
                    'market_total': market_total,
                    'model_prediction': predicted_total,
                    'model_confidence': matching_pred.get('model_confidence', 0),
                    'model_recommendation': model_correct,
                    'actual_outcome': market_correct,
                    'prediction_error': round(prediction_error, 1),
                    'percentage_error': round(percentage_error, 1),
                    'recommendation_accurate': recommendation_accurate,
                    'market_vs_model_diff': round(predicted_total - market_total, 1)
                })
        
        print(f"✅ Matched {len(matched_data)} predictions with results")
        return matched_data
    
    def create_comprehensive_tracking_display(self, matched_data):
        """Create comprehensive visual display of tracking data"""
        print("📊 Creating comprehensive tracking display...")
        
        if not matched_data:
            print("❌ No matched data to display")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle('Comprehensive MLB Prediction Tracking Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2], width_ratios=[1, 1, 1])
        
        # 1. Model Accuracy vs Market Totals
        ax1 = fig.add_subplot(gs[0, 0])
        model_predictions = [d['model_prediction'] for d in matched_data]
        market_totals = [d['market_total'] for d in matched_data]
        actual_totals = [d['actual_total'] for d in matched_data]
        
        ax1.scatter(market_totals, model_predictions, alpha=0.7, s=60, label='Model vs Market')
        ax1.plot([min(market_totals), max(market_totals)], 
                [min(market_totals), max(market_totals)], 'r--', lw=2, label='Perfect Agreement')
        ax1.set_xlabel('Market Total')
        ax1.set_ylabel('Model Prediction')
        ax1.set_title('Model Predictions vs Market Totals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction Accuracy Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        errors = [d['prediction_error'] for d in matched_data]
        ax2.hist(errors, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean Error: {np.mean(errors):.1f}')
        ax2.set_xlabel('Prediction Error (runs)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Model Prediction Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Recommendation Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        recommendation_accuracy = [d['recommendation_accurate'] for d in matched_data]
        accurate_count = sum(recommendation_accuracy)
        total_count = len(recommendation_accuracy)
        accuracy_pct = (accurate_count / total_count * 100) if total_count > 0 else 0
        
        labels = ['Correct', 'Incorrect']
        sizes = [accurate_count, total_count - accurate_count]
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Recommendation Accuracy\\n({accurate_count}/{total_count} = {accuracy_pct:.1f}%)')
        
        # 4. Actual vs Predicted Scatter
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(actual_totals, model_predictions, alpha=0.7, s=60, c=errors, cmap='RdYlBu_r')
        ax4.plot([min(actual_totals), max(actual_totals)], 
                [min(actual_totals), max(actual_totals)], 'r--', lw=2)
        ax4.set_xlabel('Actual Total Runs')
        ax4.set_ylabel('Predicted Total Runs')
        ax4.set_title('Predicted vs Actual (colored by error)')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Prediction Error')
        
        # 5. Performance by Venue
        ax5 = fig.add_subplot(gs[1, 1])
        venue_performance = {}
        for d in matched_data:
            venue = d['venue']
            if venue not in venue_performance:
                venue_performance[venue] = {'errors': [], 'accuracy': []}
            venue_performance[venue]['errors'].append(d['prediction_error'])
            venue_performance[venue]['accuracy'].append(d['recommendation_accurate'])
        
        # Get top 5 venues by game count
        top_venues = sorted(venue_performance.items(), 
                           key=lambda x: len(x[1]['errors']), reverse=True)[:5]
        
        venue_names = [v[0][:15] for v in top_venues]  # Truncate names
        venue_avg_errors = [np.mean(v[1]['errors']) for v in top_venues]
        
        bars = ax5.bar(venue_names, venue_avg_errors, alpha=0.7, color='skyblue')
        ax5.set_xlabel('Venue')
        ax5.set_ylabel('Average Prediction Error')
        ax5.set_title('Average Prediction Error by Venue')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Over Time
        ax6 = fig.add_subplot(gs[1, 2])
        # Sort data by date
        sorted_data = sorted(matched_data, key=lambda x: x['date'])
        dates = [d['date'] for d in sorted_data]
        sorted_errors = [d['prediction_error'] for d in sorted_data]
        
        ax6.plot(range(len(sorted_errors)), sorted_errors, 'o-', alpha=0.7)
        ax6.set_xlabel('Game Sequence')
        ax6.set_ylabel('Prediction Error')
        ax6.set_title('Prediction Error Over Time')
        ax6.grid(True, alpha=0.3)
        
        # 7. Market vs Model Difference Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        market_model_diffs = [d['market_vs_model_diff'] for d in matched_data]
        ax7.hist(market_model_diffs, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', label='Perfect Agreement')
        ax7.axvline(np.mean(market_model_diffs), color='blue', linestyle='--', 
                   label=f'Mean Diff: {np.mean(market_model_diffs):.1f}')
        ax7.set_xlabel('Model - Market Total')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Model vs Market Total Differences')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Pitcher ERA Impact
        ax8 = fig.add_subplot(gs[2, 1])
        home_eras = [d['home_pitcher_era'] for d in matched_data if d['home_pitcher_era'] > 0]
        away_eras = [d['away_pitcher_era'] for d in matched_data if d['away_pitcher_era'] > 0]
        combined_eras = [(d['home_pitcher_era'] + d['away_pitcher_era'])/2 
                        for d in matched_data if d['home_pitcher_era'] > 0 and d['away_pitcher_era'] > 0]
        game_totals = [d['actual_total'] for d in matched_data 
                      if d['home_pitcher_era'] > 0 and d['away_pitcher_era'] > 0]
        
        if combined_eras and game_totals:
            ax8.scatter(combined_eras, game_totals, alpha=0.7, s=60)
            ax8.set_xlabel('Average Pitcher ERA')
            ax8.set_ylabel('Actual Total Runs')
            ax8.set_title('Pitcher ERA vs Game Total Runs')
            ax8.grid(True, alpha=0.3)
        
        # 9. Model Confidence vs Accuracy
        ax9 = fig.add_subplot(gs[2, 2])
        confidences = [d['model_confidence'] for d in matched_data]
        if confidences:
            ax9.scatter(confidences, errors, alpha=0.7, s=60)
            ax9.set_xlabel('Model Confidence (%)')
            ax9.set_ylabel('Prediction Error')
            ax9.set_title('Model Confidence vs Prediction Error')
            ax9.grid(True, alpha=0.3)
        
        # 10. Detailed Results Table
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('tight')
        ax10.axis('off')
        
        # Create table data
        table_data = []
        for d in sorted(matched_data, key=lambda x: x['date'], reverse=True)[:10]:  # Last 10 games
            table_data.append([
                d['date'],
                f"{d['away_team']} @ {d['home_team']}",
                f"{d['actual_away_score']}-{d['actual_home_score']} ({d['actual_total']})",
                f"{d['market_total']}",
                f"{d['model_prediction']}",
                d['model_recommendation'],
                d['actual_outcome'],
                '✓' if d['recommendation_accurate'] else '✗',
                f"{d['prediction_error']}"
            ])
        
        headers = ['Date', 'Game', 'Score (Total)', 'Market', 'Model', 'Rec', 'Actual', 'Correct', 'Error']
        
        table = ax10.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code the table
        for i in range(len(table_data)):
            if table_data[i][7] == '✓':  # Correct recommendation
                table[(i+1, 7)].set_facecolor('lightgreen')
            else:
                table[(i+1, 7)].set_facecolor('lightcoral')
        
        ax10.set_title('Recent Game Results and Predictions', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('comprehensive_prediction_tracking.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive tracking display saved as 'comprehensive_prediction_tracking.png'")
    
    def save_tracking_data(self, matched_data):
        """Save comprehensive tracking data to files"""
        print("💾 Saving tracking data...")
        
        if matched_data:
            # Save as CSV
            df = pd.DataFrame(matched_data)
            df.to_csv('comprehensive_prediction_tracking.csv', index=False)
            print("✅ Tracking data saved to 'comprehensive_prediction_tracking.csv'")
            
            # Calculate and save summary statistics
            errors = [d['prediction_error'] for d in matched_data]
            accurate_recommendations = sum([d['recommendation_accurate'] for d in matched_data])
            
            summary = {
                'tracking_date': datetime.now().isoformat(),
                'total_games_tracked': len(matched_data),
                'average_prediction_error': float(np.mean(errors)),
                'median_prediction_error': float(np.median(errors)),
                'max_prediction_error': float(np.max(errors)),
                'min_prediction_error': float(np.min(errors)),
                'recommendation_accuracy': float(accurate_recommendations / len(matched_data) * 100),
                'total_correct_recommendations': accurate_recommendations,
                'model_vs_market_average_diff': float(np.mean([d['market_vs_model_diff'] for d in matched_data]))
            }
            
            with open('comprehensive_tracking_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print("✅ Summary statistics saved to 'comprehensive_tracking_summary.json'")
    
    def run_comprehensive_tracking(self, days_back=7):
        """Run the complete comprehensive tracking system"""
        print("🚀 Starting Comprehensive Prediction Tracking")
        print("=" * 60)
        
        # Get current predictions and market data
        current_predictions = self.get_market_totals_and_predictions()
        
        # Get completed game results
        completed_results = self.get_completed_game_results(days_back)
        
        # Load historical predictions
        historical_predictions = self.load_historical_predictions(days_back)
        
        # Match predictions with results
        matched_data = self.match_predictions_with_results(historical_predictions, completed_results)
        
        if matched_data:
            # Create comprehensive display
            self.create_comprehensive_tracking_display(matched_data)
            
            # Save tracking data
            self.save_tracking_data(matched_data)
            
            # Print summary
            errors = [d['prediction_error'] for d in matched_data]
            accurate_count = sum([d['recommendation_accurate'] for d in matched_data])
            
            print("\\n📊 COMPREHENSIVE TRACKING SUMMARY")
            print("=" * 40)
            print(f"🎯 Games Tracked: {len(matched_data)}")
            print(f"📈 Average Prediction Error: {np.mean(errors):.1f} runs")
            print(f"🎲 Recommendation Accuracy: {accurate_count}/{len(matched_data)} ({accurate_count/len(matched_data)*100:.1f}%)")
            print(f"📋 Market vs Model Avg Diff: {np.mean([d['market_vs_model_diff'] for d in matched_data]):.1f}")
            
            # Show best and worst predictions
            best = min(matched_data, key=lambda x: x['prediction_error'])
            worst = max(matched_data, key=lambda x: x['prediction_error'])
            
            print(f"\\n🟢 Best Prediction:")
            print(f"   {best['game']} - Predicted: {best['model_prediction']}, Actual: {best['actual_total']}, Error: {best['prediction_error']}")
            
            print(f"\\n🔴 Worst Prediction:")
            print(f"   {worst['game']} - Predicted: {worst['model_prediction']}, Actual: {worst['actual_total']}, Error: {worst['prediction_error']}")
            
        else:
            print("❌ No matching data found for tracking")
        
        # Display current predictions for today
        if current_predictions:
            print(f"\\n📅 TODAY'S PREDICTIONS ({len(current_predictions)} games)")
            print("=" * 60)
            for pred in current_predictions[:5]:  # Show first 5
                print(f"🏟️  {pred['away_team']} @ {pred['home_team']}")
                print(f"   Market Total: {pred['market_total']} | Model: {pred['model_prediction']} | Rec: {pred['model_recommendation']}")
                print(f"   Pitchers: {pred['home_pitcher']} vs {pred['away_pitcher']}")
                print(f"   Venue: {pred['venue']} | Weather: {pred['weather']}")
                print()

def main():
    """Run the comprehensive prediction tracking system"""
    tracker = ComprehensivePredictionTracker()
    tracker.run_comprehensive_tracking(days_back=7)

if __name__ == "__main__":
    main()
