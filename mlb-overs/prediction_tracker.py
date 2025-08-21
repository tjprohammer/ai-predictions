#!/usr/bin/env python3
"""
Prediction Performance Tracking System
Tracks all model predictions against actual game results to validate accuracy
"""

import psycopg2
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import requests

class PredictionTracker:
    def __init__(self):
        self.conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        self.create_tracking_tables()
    
    def create_tracking_tables(self):
        """Create tables to track prediction performance"""
        cursor = self.conn.cursor()
        
        # Main prediction tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_tracking (
                id SERIAL PRIMARY KEY,
                game_id VARCHAR(255),
                date DATE,
                home_team VARCHAR(100),
                away_team VARCHAR(100),
                venue VARCHAR(255),
                
                -- Current Model Predictions
                current_predicted_total DECIMAL(4,1),
                current_recommendation VARCHAR(10),
                current_edge DECIMAL(4,1),
                current_confidence DECIMAL(5,2),
                
                -- Learning Model Predictions  
                learning_predicted_total DECIMAL(4,1),
                learning_recommendation VARCHAR(10),
                learning_edge DECIMAL(4,1),
                learning_model_version VARCHAR(100),
                
                -- Market Data
                market_total DECIMAL(4,1),
                over_odds INTEGER,
                under_odds INTEGER,
                
                -- Actual Results
                actual_total DECIMAL(4,1),
                game_completed BOOLEAN DEFAULT FALSE,
                actual_result VARCHAR(10), -- OVER/UNDER/PUSH
                
                -- Performance Metrics
                current_error DECIMAL(4,1), -- |predicted - actual|
                learning_error DECIMAL(4,1), -- |predicted - actual|
                current_correct BOOLEAN,     -- recommendation correct
                learning_correct BOOLEAN,    -- recommendation correct
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(game_id, date)
            )
        """)
        
        # Performance summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_summary (
                id SERIAL PRIMARY KEY,
                date DATE,
                model_type VARCHAR(20), -- 'current' or 'learning'
                
                -- Accuracy Metrics
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy_rate DECIMAL(5,2),
                
                -- Error Metrics
                mean_absolute_error DECIMAL(4,2),
                median_absolute_error DECIMAL(4,2),
                rmse DECIMAL(4,2),
                
                -- Betting Performance
                total_bets INTEGER,
                winning_bets INTEGER,
                win_rate DECIMAL(5,2),
                total_edge DECIMAL(6,2),
                
                -- Daily Stats
                best_prediction_error DECIMAL(4,1),
                worst_prediction_error DECIMAL(4,1),
                avg_confidence DECIMAL(5,2),
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(date, model_type)
            )
        """)
        
        self.conn.commit()
        cursor.close()
    
    def record_predictions(self, date: str, games_data: List[Dict]):
        """Record predictions for tracking"""
        cursor = self.conn.cursor()
        
        for game in games_data:
            try:
                cursor.execute("""
                    INSERT INTO prediction_tracking (
                        game_id, date, home_team, away_team, venue,
                        current_predicted_total, current_recommendation, current_edge, current_confidence,
                        learning_predicted_total, learning_recommendation, learning_edge, learning_model_version,
                        market_total, over_odds, under_odds
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id, date) 
                    DO UPDATE SET
                        current_predicted_total = EXCLUDED.current_predicted_total,
                        current_recommendation = EXCLUDED.current_recommendation,
                        current_edge = EXCLUDED.current_edge,
                        current_confidence = EXCLUDED.current_confidence,
                        learning_predicted_total = EXCLUDED.learning_predicted_total,
                        learning_recommendation = EXCLUDED.learning_recommendation,
                        learning_edge = EXCLUDED.learning_edge,
                        learning_model_version = EXCLUDED.learning_model_version,
                        market_total = EXCLUDED.market_total,
                        over_odds = EXCLUDED.over_odds,
                        under_odds = EXCLUDED.under_odds,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    game.get('game_id', f"{game['away_team']}@{game['home_team']}_{date}"),
                    date,
                    game['home_team'],
                    game['away_team'],
                    game.get('venue', ''),
                    game.get('predicted_total'),
                    game.get('recommendation'),
                    game.get('edge'),
                    game.get('confidence'),
                    game.get('learning_prediction'),
                    game.get('learning_recommendation'),
                    game.get('learning_edge'),
                    game.get('model_version'),
                    game.get('market_total'),
                    game.get('over_odds'),
                    game.get('under_odds')
                ))
            except Exception as e:
                print(f"Error recording prediction for {game.get('home_team', 'Unknown')}: {e}")
        
        self.conn.commit()
        cursor.close()
    
    def update_actual_results(self, date: str):
        """Fetch and update actual game results"""
        cursor = self.conn.cursor()
        
        # Get games that need result updates
        cursor.execute("""
            SELECT game_id, home_team, away_team, market_total,
                   current_predicted_total, current_recommendation,
                   learning_predicted_total, learning_recommendation
            FROM prediction_tracking 
            WHERE date = %s AND game_completed = FALSE
        """, (date,))
        
        games_to_update = cursor.fetchall()
        
        for game_id, home_team, away_team, market_total, current_pred, current_rec, learning_pred, learning_rec in games_to_update:
            try:
                # Get actual result from enhanced_games table (which has final scores)
                cursor.execute("""
                    SELECT home_score, away_score, game_state
                    FROM enhanced_games 
                    WHERE home_team = %s AND away_team = %s AND date = %s
                """, (home_team, away_team, date))
                
                result = cursor.fetchone()
                if result and result[2] == 'Final':  # Game completed
                    home_score, away_score, game_state = result
                    actual_total = home_score + away_score
                    
                    # Determine actual result vs market
                    if market_total:
                        if actual_total > market_total:
                            actual_result = 'OVER'
                        elif actual_total < market_total:
                            actual_result = 'UNDER'
                        else:
                            actual_result = 'PUSH'
                    else:
                        actual_result = None
                    
                    # Calculate errors
                    current_error = abs(current_pred - actual_total) if current_pred else None
                    learning_error = abs(learning_pred - actual_total) if learning_pred else None
                    
                    # Check if recommendations were correct
                    current_correct = None
                    learning_correct = None
                    
                    if current_rec and actual_result and actual_result != 'PUSH':
                        current_correct = (current_rec == actual_result)
                    
                    if learning_rec and actual_result and actual_result != 'PUSH':
                        learning_correct = (learning_rec == actual_result)
                    
                    # Update tracking record
                    cursor.execute("""
                        UPDATE prediction_tracking SET
                            actual_total = %s,
                            game_completed = TRUE,
                            actual_result = %s,
                            current_error = %s,
                            learning_error = %s,
                            current_correct = %s,
                            learning_correct = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE game_id = %s AND date = %s
                    """, (
                        actual_total, actual_result, current_error, learning_error,
                        current_correct, learning_correct, game_id, date
                    ))
                    
                    print(f"✅ Updated {away_team} @ {home_team}: {actual_total} runs ({actual_result})")
                    
            except Exception as e:
                print(f"Error updating results for {away_team} @ {home_team}: {e}")
        
        self.conn.commit()
        cursor.close()
    
    def calculate_performance_metrics(self, date: str):
        """Calculate and store performance metrics for both models"""
        cursor = self.conn.cursor()
        
        # Calculate metrics for current model
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN current_correct = TRUE THEN 1 END) as correct_predictions,
                AVG(current_error) as mean_absolute_error,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY current_error) as median_absolute_error,
                SQRT(AVG(current_error * current_error)) as rmse,
                COUNT(CASE WHEN current_recommendation != 'HOLD' THEN 1 END) as total_bets,
                COUNT(CASE WHEN current_recommendation != 'HOLD' AND current_correct = TRUE THEN 1 END) as winning_bets,
                SUM(current_edge) as total_edge,
                MIN(current_error) as best_prediction_error,
                MAX(current_error) as worst_prediction_error,
                AVG(current_confidence) as avg_confidence
            FROM prediction_tracking 
            WHERE date = %s AND game_completed = TRUE AND current_predicted_total IS NOT NULL
        """, (date,))
        
        current_metrics = cursor.fetchone()
        
        if current_metrics and current_metrics[0] > 0:  # Has data
            total_pred, correct_pred, mae, median_ae, rmse, total_bets, winning_bets, total_edge, best_error, worst_error, avg_conf = current_metrics
            
            accuracy_rate = (correct_pred / total_pred * 100) if total_pred > 0 else 0
            win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
            
            cursor.execute("""
                INSERT INTO model_performance_summary (
                    date, model_type, total_predictions, correct_predictions, accuracy_rate,
                    mean_absolute_error, median_absolute_error, rmse, total_bets, winning_bets,
                    win_rate, total_edge, best_prediction_error, worst_prediction_error, avg_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, model_type) 
                DO UPDATE SET
                    total_predictions = EXCLUDED.total_predictions,
                    correct_predictions = EXCLUDED.correct_predictions,
                    accuracy_rate = EXCLUDED.accuracy_rate,
                    mean_absolute_error = EXCLUDED.mean_absolute_error,
                    median_absolute_error = EXCLUDED.median_absolute_error,
                    rmse = EXCLUDED.rmse,
                    total_bets = EXCLUDED.total_bets,
                    winning_bets = EXCLUDED.winning_bets,
                    win_rate = EXCLUDED.win_rate,
                    total_edge = EXCLUDED.total_edge,
                    best_prediction_error = EXCLUDED.best_prediction_error,
                    worst_prediction_error = EXCLUDED.worst_prediction_error,
                    avg_confidence = EXCLUDED.avg_confidence
            """, (
                date, 'current', total_pred, correct_pred, accuracy_rate,
                mae, median_ae, rmse, total_bets, winning_bets,
                win_rate, total_edge, best_error, worst_error, avg_conf
            ))
        
        # Calculate metrics for learning model
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN learning_correct = TRUE THEN 1 END) as correct_predictions,
                AVG(learning_error) as mean_absolute_error,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY learning_error) as median_absolute_error,
                SQRT(AVG(learning_error * learning_error)) as rmse,
                COUNT(CASE WHEN learning_recommendation != 'HOLD' THEN 1 END) as total_bets,
                COUNT(CASE WHEN learning_recommendation != 'HOLD' AND learning_correct = TRUE THEN 1 END) as winning_bets,
                SUM(learning_edge) as total_edge,
                MIN(learning_error) as best_prediction_error,
                MAX(learning_error) as worst_prediction_error
            FROM prediction_tracking 
            WHERE date = %s AND game_completed = TRUE AND learning_predicted_total IS NOT NULL
        """, (date,))
        
        learning_metrics = cursor.fetchone()
        
        if learning_metrics and learning_metrics[0] > 0:  # Has data
            total_pred, correct_pred, mae, median_ae, rmse, total_bets, winning_bets, total_edge, best_error, worst_error = learning_metrics
            
            accuracy_rate = (correct_pred / total_pred * 100) if total_pred > 0 else 0
            win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
            
            cursor.execute("""
                INSERT INTO model_performance_summary (
                    date, model_type, total_predictions, correct_predictions, accuracy_rate,
                    mean_absolute_error, median_absolute_error, rmse, total_bets, winning_bets,
                    win_rate, total_edge, best_prediction_error, worst_prediction_error, avg_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, model_type) 
                DO UPDATE SET
                    total_predictions = EXCLUDED.total_predictions,
                    correct_predictions = EXCLUDED.correct_predictions,
                    accuracy_rate = EXCLUDED.accuracy_rate,
                    mean_absolute_error = EXCLUDED.mean_absolute_error,
                    median_absolute_error = EXCLUDED.median_absolute_error,
                    rmse = EXCLUDED.rmse,
                    total_bets = EXCLUDED.total_bets,
                    winning_bets = EXCLUDED.winning_bets,
                    win_rate = EXCLUDED.win_rate,
                    total_edge = EXCLUDED.total_edge,
                    best_prediction_error = EXCLUDED.best_prediction_error,
                    worst_prediction_error = EXCLUDED.worst_prediction_error,
                    avg_confidence = EXCLUDED.avg_confidence
            """, (
                date, 'learning', total_pred, correct_pred, accuracy_rate,
                mae, median_ae, rmse, total_bets, winning_bets,
                win_rate, total_edge, best_error, worst_error, None  # learning model doesn't have confidence yet
            ))
        
        self.conn.commit()
        cursor.close()
    
    def get_performance_comparison(self, start_date: str, end_date: str = None) -> Dict:
        """Get performance comparison between models over date range"""
        if not end_date:
            end_date = start_date
            
        cursor = self.conn.cursor()
        
        # Get aggregate performance metrics
        cursor.execute("""
            SELECT 
                model_type,
                SUM(total_predictions) as total_predictions,
                SUM(correct_predictions) as correct_predictions,
                AVG(mean_absolute_error) as avg_mae,
                AVG(median_absolute_error) as avg_median_ae,
                AVG(rmse) as avg_rmse,
                SUM(total_bets) as total_bets,
                SUM(winning_bets) as winning_bets,
                AVG(win_rate) as avg_win_rate,
                SUM(total_edge) as total_edge,
                AVG(best_prediction_error) as avg_best_error,
                AVG(worst_prediction_error) as avg_worst_error
            FROM model_performance_summary 
            WHERE date BETWEEN %s AND %s
            GROUP BY model_type
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        
        performance_data = {}
        for row in results:
            model_type = row[0]
            total_pred, correct_pred, avg_mae, avg_median_ae, avg_rmse, total_bets, winning_bets, avg_win_rate, total_edge, avg_best_error, avg_worst_error = row[1:]
            
            accuracy_rate = (correct_pred / total_pred * 100) if total_pred > 0 else 0
            actual_win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
            
            performance_data[model_type] = {
                'total_predictions': total_pred,
                'correct_predictions': correct_pred,
                'accuracy_rate': round(accuracy_rate, 1),
                'mean_absolute_error': round(avg_mae, 2) if avg_mae else None,
                'median_absolute_error': round(avg_median_ae, 2) if avg_median_ae else None,
                'rmse': round(avg_rmse, 2) if avg_rmse else None,
                'total_bets': total_bets,
                'winning_bets': winning_bets,
                'win_rate': round(actual_win_rate, 1),
                'total_edge': round(total_edge, 2) if total_edge else 0,
                'avg_best_error': round(avg_best_error, 2) if avg_best_error else None,
                'avg_worst_error': round(avg_worst_error, 2) if avg_worst_error else None
            }
        
        cursor.close()
        return performance_data
    
    def get_recent_predictions_with_results(self, days: int = 7) -> List[Dict]:
        """Get recent predictions with actual results for verification"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                date, home_team, away_team, venue,
                current_predicted_total, learning_predicted_total, market_total, actual_total,
                current_recommendation, learning_recommendation, actual_result,
                current_error, learning_error, current_correct, learning_correct,
                game_completed
            FROM prediction_tracking 
            WHERE date >= %s
            ORDER BY date DESC, home_team
        """, (datetime.now() - timedelta(days=days),))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0].strftime('%Y-%m-%d'),
                'game': f"{row[2]} @ {row[1]}",  # away @ home
                'venue': row[3],
                'current_predicted': float(row[4]) if row[4] else None,
                'learning_predicted': float(row[5]) if row[5] else None,
                'market_total': float(row[6]) if row[6] else None,
                'actual_total': float(row[7]) if row[7] else None,
                'current_rec': row[8],
                'learning_rec': row[9],
                'actual_result': row[10],
                'current_error': float(row[11]) if row[11] else None,
                'learning_error': float(row[12]) if row[12] else None,
                'current_correct': row[13],
                'learning_correct': row[14],
                'completed': row[15]
            })
        
        cursor.close()
        return results

if __name__ == "__main__":
    import sys
    tracker = PredictionTracker()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "update":
            date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
            print(f"Updating results for {date}...")
            tracker.update_actual_results(date)
            tracker.calculate_performance_metrics(date)
            print("✅ Results updated and metrics calculated")
        
        elif sys.argv[1] == "performance":
            start_date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
            end_date = sys.argv[3] if len(sys.argv) > 3 else start_date
            
            performance = tracker.get_performance_comparison(start_date, end_date)
            print(f"\n📊 PERFORMANCE COMPARISON ({start_date} to {end_date})")
            print("=" * 60)
            
            for model_type, metrics in performance.items():
                print(f"\n{model_type.upper()} MODEL:")
                print(f"  Predictions: {metrics['total_predictions']}")
                print(f"  Accuracy: {metrics['accuracy_rate']}%")
                print(f"  MAE: {metrics['mean_absolute_error']}")
                print(f"  Betting Win Rate: {metrics['win_rate']}%")
                print(f"  Total Edge: {metrics['total_edge']}")
    else:
        print("Usage:")
        print("  python prediction_tracker.py update [date]")
        print("  python prediction_tracker.py performance [start_date] [end_date]")
