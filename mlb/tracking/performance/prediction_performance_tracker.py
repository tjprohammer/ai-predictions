#!/usr/bin/env python3
"""
Prediction Performance Tracker

This system tracks predictions vs actual results to prove model accuracy.
Includes both learning model and current model predictions for comparison.
"""

import psycopg2
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionTracker:
    def __init__(self):
        self.conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        self.cursor = self.conn.cursor()
        self._ensure_tracking_table()

    def _ensure_tracking_table(self):
        """Create prediction tracking table if it doesn't exist"""
        
        # Create game_results table first
        create_results_table = """
        CREATE TABLE IF NOT EXISTS game_results (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            home_team VARCHAR(50) NOT NULL,
            away_team VARCHAR(50) NOT NULL,
            home_score INTEGER DEFAULT 0,
            away_score INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'Scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, date)
        );
        """
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS prediction_tracking (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            home_team VARCHAR(50) NOT NULL,
            away_team VARCHAR(50) NOT NULL,
            venue VARCHAR(100),
            
            -- Current Model Predictions
            current_prediction DECIMAL(4,2),
            current_recommendation VARCHAR(10),
            current_confidence DECIMAL(5,2),
            current_edge DECIMAL(4,2),
            
            -- Learning Model Predictions  
            learning_prediction DECIMAL(4,2),
            learning_recommendation VARCHAR(10),
            learning_edge DECIMAL(4,2),
            model_version VARCHAR(50),
            
            -- Market Data
            market_total DECIMAL(4,2),
            over_odds INTEGER,
            under_odds INTEGER,
            
            -- Actual Results
            actual_total DECIMAL(4,2),
            game_completed BOOLEAN DEFAULT FALSE,
            result_updated_at TIMESTAMP,
            
            -- Performance Tracking
            current_correct BOOLEAN,
            learning_correct BOOLEAN,
            current_edge_realized DECIMAL(4,2),
            learning_edge_realized DECIMAL(4,2),
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(game_id, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_prediction_tracking_date ON prediction_tracking(date);
        CREATE INDEX IF NOT EXISTS idx_prediction_tracking_completed ON prediction_tracking(game_completed);
        """
        
        try:
            self.cursor.execute(create_results_table)
            self.cursor.execute(create_table_sql)
            self.conn.commit()
            logger.info("Prediction tracking table ready")
        except Exception as e:
            logger.error(f"Error creating tracking table: {e}")
            self.conn.rollback()

    def record_predictions(self, date: str):
        """Record predictions for a given date from both models"""
        try:
            # Get comprehensive predictions
            self.cursor.execute("""
                SELECT game_id, home_team, away_team, venue, predicted_total, 
                       recommendation, confidence, edge, market_total, over_odds, under_odds
                FROM enhanced_games 
                WHERE date = %s AND predicted_total IS NOT NULL
            """, (date,))
            
            comprehensive_games = self.cursor.fetchall()
            
            # Get learning predictions
            learning_file = f"enhanced_predictions_{date}.json"
            try:
                with open(learning_file, 'r') as f:
                    learning_data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(learning_data, list):
                    learning_games = learning_data
                elif isinstance(learning_data, dict) and "games" in learning_data:
                    learning_games = learning_data["games"]
                else:
                    learning_games = []
                    
            except FileNotFoundError:
                logger.warning(f"Learning predictions file not found: {learning_file}")
                learning_games = []
            
            # Process each game
            for game in comprehensive_games:
                game_id, home_team, away_team, venue, current_pred, current_rec, \
                confidence, current_edge, market_total, over_odds, under_odds = game
                
                # Find matching learning prediction
                game_key = f"{away_team} @ {home_team}"
                learning_game = None
                for lg in learning_games:
                    if isinstance(lg, dict) and lg.get("game") == game_key:
                        learning_game = lg
                        break
                
                # Insert or update prediction record
                insert_sql = """
                INSERT INTO prediction_tracking (
                    game_id, date, home_team, away_team, venue,
                    current_prediction, current_recommendation, current_confidence, current_edge,
                    learning_prediction, learning_recommendation, learning_edge, model_version,
                    market_total, over_odds, under_odds
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id, date) 
                DO UPDATE SET
                    current_prediction = EXCLUDED.current_prediction,
                    current_recommendation = EXCLUDED.current_recommendation,
                    current_confidence = EXCLUDED.current_confidence,
                    current_edge = EXCLUDED.current_edge,
                    learning_prediction = EXCLUDED.learning_prediction,
                    learning_recommendation = EXCLUDED.learning_recommendation,
                    learning_edge = EXCLUDED.learning_edge,
                    model_version = EXCLUDED.model_version,
                    market_total = EXCLUDED.market_total,
                    over_odds = EXCLUDED.over_odds,
                    under_odds = EXCLUDED.under_odds,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                values = (
                    game_id, date, home_team, away_team, venue,
                    current_pred, current_rec, confidence, current_edge,
                    learning_game.get("learning_prediction") if learning_game else None,
                    learning_game.get("learning_recommendation") if learning_game else None,
                    learning_game.get("learning_edge") if learning_game else None,
                    learning_game.get("model_version") if learning_game else None,
                    market_total, over_odds, under_odds
                )
                
                self.cursor.execute(insert_sql, values)
            
            self.conn.commit()
            logger.info(f"Recorded predictions for {len(comprehensive_games)} games on {date}")
            
        except Exception as e:
            logger.error(f"Error recording predictions for {date}: {e}")
            self.conn.rollback()

    def update_actual_results(self, date: str):
        """Update actual results for completed games"""
        try:
            # Get actual results from game_results table
            self.cursor.execute("""
                SELECT gr.game_id, gr.home_score + gr.away_score as actual_total
                FROM game_results gr
                JOIN prediction_tracking pt ON gr.game_id = pt.game_id AND gr.date = pt.date
                WHERE pt.date = %s AND pt.game_completed = FALSE AND gr.status = 'Final'
            """, (date,))
            
            results = self.cursor.fetchall()
            
            for game_id, actual_total in results:
                # Update the tracking record with actual results
                self.cursor.execute("""
                    UPDATE prediction_tracking 
                    SET 
                        actual_total = %s,
                        game_completed = TRUE,
                        result_updated_at = CURRENT_TIMESTAMP,
                        current_correct = CASE 
                            WHEN current_recommendation = 'OVER' AND %s > market_total THEN TRUE
                            WHEN current_recommendation = 'UNDER' AND %s < market_total THEN TRUE
                            WHEN current_recommendation = 'HOLD' THEN NULL
                            ELSE FALSE
                        END,
                        learning_correct = CASE 
                            WHEN learning_recommendation = 'OVER' AND %s > market_total THEN TRUE
                            WHEN learning_recommendation = 'UNDER' AND %s < market_total THEN TRUE
                            WHEN learning_recommendation = 'HOLD' THEN NULL
                            ELSE FALSE
                        END,
                        current_edge_realized = CASE 
                            WHEN current_recommendation = 'OVER' AND %s > market_total THEN current_edge
                            WHEN current_recommendation = 'UNDER' AND %s < market_total THEN current_edge
                            ELSE current_edge * -1
                        END,
                        learning_edge_realized = CASE 
                            WHEN learning_recommendation = 'OVER' AND %s > market_total THEN learning_edge
                            WHEN learning_recommendation = 'UNDER' AND %s < market_total THEN learning_edge
                            ELSE learning_edge * -1
                        END
                    WHERE game_id = %s AND date = %s
                """, (actual_total, actual_total, actual_total, actual_total, actual_total, 
                     actual_total, actual_total, actual_total, actual_total, game_id, date))
            
            self.conn.commit()
            logger.info(f"Updated results for {len(results)} games on {date}")
            
        except Exception as e:
            logger.error(f"Error updating results for {date}: {e}")
            self.conn.rollback()

    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for both models"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Overall stats
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN game_completed THEN 1 END) as completed_games,
                    
                    -- Current Model Performance
                    COUNT(CASE WHEN current_correct = TRUE THEN 1 END) as current_correct,
                    COUNT(CASE WHEN current_correct = FALSE THEN 1 END) as current_incorrect,
                    AVG(CASE WHEN current_correct = TRUE THEN current_edge_realized END) as current_avg_edge_won,
                    AVG(CASE WHEN current_correct = FALSE THEN current_edge_realized END) as current_avg_edge_lost,
                    
                    -- Learning Model Performance  
                    COUNT(CASE WHEN learning_correct = TRUE THEN 1 END) as learning_correct,
                    COUNT(CASE WHEN learning_correct = FALSE THEN 1 END) as learning_incorrect,
                    AVG(CASE WHEN learning_correct = TRUE THEN learning_edge_realized END) as learning_avg_edge_won,
                    AVG(CASE WHEN learning_correct = FALSE THEN learning_edge_realized END) as learning_avg_edge_lost
                    
                FROM prediction_tracking 
                WHERE date >= %s AND date <= %s AND game_completed = TRUE
            """, (start_date, end_date))
            
            summary = self.cursor.fetchone()
            
            if summary[0] == 0:  # No data
                return {"error": "No prediction data found for the specified period"}
            
            total_predictions, completed_games, current_correct, current_incorrect, \
            current_avg_edge_won, current_avg_edge_lost, learning_correct, learning_incorrect, \
            learning_avg_edge_won, learning_avg_edge_lost = summary
            
            # Calculate percentages
            current_total_calls = current_correct + current_incorrect
            learning_total_calls = learning_correct + learning_incorrect
            
            current_accuracy = (current_correct / current_total_calls * 100) if current_total_calls > 0 else 0
            learning_accuracy = (learning_correct / learning_total_calls * 100) if learning_total_calls > 0 else 0
            
            return {
                "period": f"{start_date} to {end_date}",
                "total_predictions": total_predictions,
                "completed_games": completed_games,
                "completion_rate": f"{completed_games/total_predictions*100:.1f}%" if total_predictions > 0 else "0%",
                
                "current_model": {
                    "total_calls": current_total_calls,
                    "correct": current_correct,
                    "incorrect": current_incorrect,
                    "accuracy": f"{current_accuracy:.1f}%",
                    "avg_edge_won": round(current_avg_edge_won or 0, 2),
                    "avg_edge_lost": round(current_avg_edge_lost or 0, 2)
                },
                
                "learning_model": {
                    "total_calls": learning_total_calls,
                    "correct": learning_correct,
                    "incorrect": learning_incorrect,
                    "accuracy": f"{learning_accuracy:.1f}%", 
                    "avg_edge_won": round(learning_avg_edge_won or 0, 2),
                    "avg_edge_lost": round(learning_avg_edge_lost or 0, 2)
                },
                
                "comparison": {
                    "learning_vs_current_accuracy": f"{learning_accuracy - current_accuracy:+.1f}%",
                    "learning_better": learning_accuracy > current_accuracy
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

    def get_recent_predictions(self, days: int = 7) -> List[Dict]:
        """Get recent predictions with results"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            self.cursor.execute("""
                SELECT 
                    date, home_team, away_team, venue,
                    current_prediction, current_recommendation, current_edge,
                    learning_prediction, learning_recommendation, learning_edge,
                    market_total, actual_total, game_completed,
                    current_correct, learning_correct,
                    current_edge_realized, learning_edge_realized
                FROM prediction_tracking 
                WHERE date >= %s AND date <= %s
                ORDER BY date DESC, home_team
            """, (start_date, end_date))
            
            predictions = []
            for row in self.cursor.fetchall():
                predictions.append({
                    "date": row[0].strftime("%Y-%m-%d"),
                    "game": f"{row[2]} @ {row[1]}",
                    "venue": row[3],
                    "current_prediction": float(row[4]) if row[4] else None,
                    "current_recommendation": row[5],
                    "current_edge": float(row[6]) if row[6] else None,
                    "learning_prediction": float(row[7]) if row[7] else None,
                    "learning_recommendation": row[8],
                    "learning_edge": float(row[9]) if row[9] else None,
                    "market_total": float(row[10]) if row[10] else None,
                    "actual_total": float(row[11]) if row[11] else None,
                    "game_completed": row[12],
                    "current_correct": row[13],
                    "learning_correct": row[14],
                    "current_edge_realized": float(row[15]) if row[15] else None,
                    "learning_edge_realized": float(row[16]) if row[16] else None
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []

def main():
    """Main function to run tracking operations"""
    tracker = PredictionTracker()
    
    # Record today's predictions
    today = datetime.now().date().strftime("%Y-%m-%d")
    print(f"Recording predictions for {today}...")
    tracker.record_predictions(today)
    
    # Update results for recent dates
    for i in range(7):  # Check last 7 days for completed games
        check_date = (datetime.now().date() - timedelta(days=i)).strftime("%Y-%m-%d")
        tracker.update_actual_results(check_date)
    
    # Show performance summary
    print("\n=== PERFORMANCE SUMMARY (Last 30 Days) ===")
    summary = tracker.get_performance_summary(30)
    if "error" in summary:
        print(f"Error: {summary['error']}")
    else:
        print(f"Period: {summary['period']}")
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Completed Games: {summary['completed_games']} ({summary['completion_rate']})")
        print()
        print("CURRENT MODEL:")
        print(f"  Accuracy: {summary['current_model']['accuracy']} ({summary['current_model']['correct']}/{summary['current_model']['total_calls']})")
        print(f"  Avg Edge Won: {summary['current_model']['avg_edge_won']}")
        print(f"  Avg Edge Lost: {summary['current_model']['avg_edge_lost']}")
        print()
        print("LEARNING MODEL:")
        print(f"  Accuracy: {summary['learning_model']['accuracy']} ({summary['learning_model']['correct']}/{summary['learning_model']['total_calls']})")
        print(f"  Avg Edge Won: {summary['learning_model']['avg_edge_won']}")
        print(f"  Avg Edge Lost: {summary['learning_model']['avg_edge_lost']}")
        print()
        print("COMPARISON:")
        print(f"  Learning vs Current: {summary['comparison']['learning_vs_current_accuracy']}")
        print(f"  Learning Better: {summary['comparison']['learning_better']}")

if __name__ == "__main__":
    main()
