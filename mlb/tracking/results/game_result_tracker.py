#!/usr/bin/env python3
"""
Real-Time Game Result Tracker and Model Validator
================================================

This system will:
1. Monitor games in progress and collect final scores
2. Validate our predictions against actual outcomes
3. Track model accuracy metrics over time
4. Identify prediction bias patterns
5. Generate performance reports
6. Update model corrections based on results
"""

import psycopg2
import requests
import json
from datetime import datetime, timedelta
import time
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("Note: schedule module not available - scheduler features disabled")
from typing import Dict, List, Tuple, Optional

def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

class GameResultTracker:
    def __init__(self):
        self.db_conn = connect_db()
        self.api_base = "http://localhost:8000"
        
    def get_games_in_progress(self, target_date: str = None) -> List[Dict]:
        """Get games that are in progress or recently finished"""
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        conn = self.db_conn
        cursor = conn.cursor()
        
        # Get games that don't have final results yet
        cursor.execute('''
        SELECT game_id, home_team, away_team, predicted_total, market_total, 
               recommendation, edge, confidence, total_runs, home_score, away_score,
               game_state, start_time
        FROM enhanced_games 
        WHERE date = %s 
        AND (total_runs IS NULL OR game_state != 'Final')
        ORDER BY game_id
        ''', (target_date,))
        
        games = []
        for row in cursor.fetchall():
            games.append({
                'game_id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'predicted_total': float(row[3]) if row[3] else None,
                'market_total': float(row[4]) if row[4] else None,
                'recommendation': row[5],
                'edge': float(row[6]) if row[6] else None,
                'confidence': float(row[7]) if row[7] else None,
                'total_runs': row[8],
                'home_score': row[9],
                'away_score': row[10],
                'game_state': row[11],
                'start_time': row[12]
            })
            
        return games
    
    def fetch_live_scores(self, game_id: str) -> Optional[Dict]:
        """Fetch live scores from MLB API or other source"""
        # This would integrate with MLB's live scoring API
        # For now, return mock data structure
        try:
            # In real implementation, this would call MLB Stats API
            # https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2025-08-20
            
            # Mock implementation - replace with actual API call
            return {
                'game_id': game_id,
                'status': 'Final',  # or 'In Progress', 'Scheduled'
                'home_score': None,  # Will be filled from real API
                'away_score': None,
                'total_runs': None,
                'inning': None,
                'inning_state': None  # 'Top', 'Middle', 'Bottom', 'End'
            }
        except Exception as e:
            print(f"⚠️ Error fetching live scores for game {game_id}: {e}")
            return None
    
    def update_game_results(self, game_id: str, home_score: int, away_score: int, 
                          status: str = 'Final') -> bool:
        """Update game results in database"""
        try:
            total_runs = home_score + away_score
            cursor = self.db_conn.cursor()
            
            cursor.execute('''
            UPDATE enhanced_games 
            SET home_score = %s, away_score = %s, total_runs = %s, 
                game_state = %s, result_updated_at = NOW()
            WHERE game_id = %s
            ''', (home_score, away_score, total_runs, status, game_id))
            
            self.db_conn.commit()
            print(f"✅ Updated results for game {game_id}: {home_score}-{away_score} = {total_runs} runs")
            return True
            
        except Exception as e:
            print(f"❌ Error updating game {game_id}: {e}")
            self.db_conn.rollback()
            return False
    
    def validate_predictions(self, target_date: str = None) -> Dict:
        """Validate predictions against actual results"""
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        cursor = self.db_conn.cursor()
        
        # Get completed games with predictions
        cursor.execute('''
        SELECT game_id, home_team, away_team, predicted_total, market_total,
               recommendation, edge, confidence, total_runs,
               ABS(predicted_total - total_runs) as prediction_error,
               ABS(market_total - total_runs) as market_error,
               CASE 
                 WHEN recommendation = 'OVER' AND total_runs > market_total THEN 'WIN'
                 WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 'WIN'  
                 WHEN recommendation = 'HOLD' THEN 'HOLD'
                 ELSE 'LOSS'
               END as bet_result
        FROM enhanced_games 
        WHERE date = %s 
        AND total_runs IS NOT NULL 
        AND predicted_total IS NOT NULL
        AND game_state = 'Final'
        ORDER BY confidence DESC
        ''', (target_date,))
        
        results = cursor.fetchall()
        
        if not results:
            return {'status': 'No completed games with predictions found'}
        
        # Calculate metrics
        total_games = len(results)
        our_total_error = sum(row[9] for row in results)  # prediction_error
        market_total_error = sum(row[10] for row in results)  # market_error
        
        # Betting performance
        wins = sum(1 for row in results if row[11] == 'WIN')
        losses = sum(1 for row in results if row[11] == 'LOSS')
        holds = sum(1 for row in results if row[11] == 'HOLD')
        
        actionable_games = wins + losses
        win_rate = (wins / actionable_games * 100) if actionable_games > 0 else 0
        
        # Confidence analysis
        high_conf_games = [r for r in results if r[7] and r[7] >= 70]  # confidence >= 70%
        high_conf_wins = sum(1 for r in high_conf_games if r[11] == 'WIN')
        high_conf_rate = (high_conf_wins / len(high_conf_games) * 100) if high_conf_games else 0
        
        return {
            'date': target_date,
            'total_games': total_games,
            'actionable_games': actionable_games,
            'holds': holds,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'our_avg_error': round(our_total_error / total_games, 2),
            'market_avg_error': round(market_total_error / total_games, 2),
            'accuracy_vs_market': 'BETTER' if our_total_error < market_total_error else 'WORSE',
            'high_confidence_games': len(high_conf_games),
            'high_confidence_win_rate': round(high_conf_rate, 1),
            'detailed_results': [{
                'game': f"{row[2]} @ {row[1]}",
                'predicted': row[3],
                'market': row[4], 
                'actual': row[8],
                'recommendation': row[5],
                'result': row[11],
                'confidence': row[7]
            } for row in results]
        }
    
    def generate_daily_report(self, target_date: str = None) -> str:
        """Generate comprehensive daily performance report"""
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        validation = self.validate_predictions(target_date)
        
        if 'status' in validation:
            return f"📊 DAILY REPORT - {target_date}\n{validation['status']}"
        
        report = f"""
📊 DAILY PREDICTION PERFORMANCE REPORT - {target_date}
{'='*60}

🎯 OVERALL PERFORMANCE:
   Total Games: {validation['total_games']}
   Actionable Picks: {validation['actionable_games']}
   Hold Recommendations: {validation['holds']}
   
🏆 BETTING RESULTS:
   Wins: {validation['wins']} 
   Losses: {validation['losses']}
   Win Rate: {validation['win_rate']}%
   
📈 PREDICTION ACCURACY:
   Our Average Error: {validation['our_avg_error']} runs
   Market Average Error: {validation['market_avg_error']} runs
   Performance vs Market: {validation['accuracy_vs_market']}
   
⭐ HIGH CONFIDENCE ANALYSIS:
   High Confidence Games (70%+): {validation['high_confidence_games']}
   High Confidence Win Rate: {validation['high_confidence_win_rate']}%

📋 DETAILED RESULTS:
"""
        
        for game in validation['detailed_results']:
            status_emoji = "✅" if game['result'] == 'WIN' else "❌" if game['result'] == 'LOSS' else "⏸️"
            report += f"   {status_emoji} {game['game']}: {game['recommendation']} "
            report += f"(Pred: {game['predicted']}, Market: {game['market']}, Actual: {game['actual']}) "
            report += f"- {game['result']}\n"
        
        return report
    
    def track_model_bias(self, days_back: int = 7) -> Dict:
        """Analyze model bias patterns over recent games"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
        SELECT 
            COUNT(*) as total_games,
            AVG(predicted_total - total_runs) as avg_bias,
            STDDEV(predicted_total - total_runs) as bias_stddev,
            COUNT(CASE WHEN predicted_total > total_runs THEN 1 END) as over_predictions,
            COUNT(CASE WHEN predicted_total < total_runs THEN 1 END) as under_predictions,
            AVG(CASE WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1.0 ELSE 0.0 END) as over_success_rate,
            AVG(CASE WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1.0 ELSE 0.0 END) as under_success_rate
        FROM enhanced_games 
        WHERE date BETWEEN %s AND %s
        AND total_runs IS NOT NULL 
        AND predicted_total IS NOT NULL
        AND game_state = 'Final'
        ''', (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        bias_data = cursor.fetchone()
        
        if not bias_data or not bias_data[0]:
            return {'status': 'Insufficient data for bias analysis'}
        
        return {
            'analysis_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'total_games': bias_data[0],
            'average_bias': round(bias_data[1], 3) if bias_data[1] else 0,
            'bias_consistency': round(bias_data[2], 3) if bias_data[2] else 0,
            'over_predictions': bias_data[3],
            'under_predictions': bias_data[4],
            'over_bet_success': round(bias_data[5] * 100, 1) if bias_data[5] else 0,
            'under_bet_success': round(bias_data[6] * 100, 1) if bias_data[6] else 0,
            'bias_direction': 'OVER' if bias_data[1] and bias_data[1] > 0.1 else 'UNDER' if bias_data[1] and bias_data[1] < -0.1 else 'NEUTRAL'
        }

def monitor_games_continuous():
    """Continuously monitor games and update results"""
    tracker = GameResultTracker()
    
    print("🎯 Starting continuous game monitoring...")
    
    while True:
        try:
            current_time = datetime.now()
            target_date = current_time.strftime('%Y-%m-%d')
            
            # Get games in progress
            games = tracker.get_games_in_progress(target_date)
            
            if games:
                print(f"📊 Monitoring {len(games)} games for {target_date}")
                
                for game in games:
                    # In real implementation, fetch live scores here
                    # live_data = tracker.fetch_live_scores(game['game_id'])
                    # if live_data and live_data['status'] == 'Final':
                    #     tracker.update_game_results(...)
                    pass
            
            # Generate report if it's late enough (games typically end by midnight)
            if current_time.hour >= 23:  # 11 PM or later
                validation = tracker.validate_predictions(target_date)
                if validation.get('total_games', 0) > 0:
                    report = tracker.generate_daily_report(target_date)
                    print(report)
                    
                    # Save report to file
                    with open(f'daily_reports/report_{target_date}.txt', 'w') as f:
                        f.write(report)
            
            # Wait 15 minutes before next check
            time.sleep(900)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error in monitoring loop: {e}")
            time.sleep(60)  # Wait 1 minute on error

def main():
    """Main function for manual testing and reports"""
    tracker = GameResultTracker()
    
    print("🎯 MLB PREDICTION VALIDATION SYSTEM")
    print("=" * 50)
    
    # Test with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check games in progress
    games = tracker.get_games_in_progress(today)
    print(f"📊 Found {len(games)} games in progress for {today}")
    
    # Show validation if any games are complete
    validation = tracker.validate_predictions(today)
    if 'total_games' in validation and validation['total_games'] > 0:
        report = tracker.generate_daily_report(today)
        print(report)
    else:
        print("⏳ No completed games to validate yet")
    
    # Show model bias analysis
    bias = tracker.track_model_bias(7)
    if 'total_games' in bias:
        print(f"\n🔍 MODEL BIAS ANALYSIS (Last 7 days):")
        print(f"   Average Bias: {bias['average_bias']} runs ({bias['bias_direction']})")
        print(f"   Over Bet Success: {bias['over_bet_success']}%")
        print(f"   Under Bet Success: {bias['under_bet_success']}%")
    
    # Manual result entry for testing
    print(f"\n📝 MANUAL RESULT ENTRY:")
    print(f"   Example: tracker.update_game_results('776652', 5, 4, 'Final')")

if __name__ == "__main__":
    main()
