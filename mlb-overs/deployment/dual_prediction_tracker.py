#!/usr/bin/env python3
"""
Dual Prediction Tracker for UI
==============================
Generates JSON data for UI to display both original and learning model predictions.
This integrates with your existing UI that tracks predictions.

Output format matches your current prediction tracking system but adds learning model data.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd

# Set up logging  
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment"""
    return os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

def get_dual_predictions(target_date: str = None) -> dict:
    """
    Get dual predictions for UI display
    
    Returns JSON structure compatible with existing UI:
    {
        "date": "2025-08-22",
        "games": [
            {
                "game_id": "...",
                "home_team": "...",
                "away_team": "...",
                "predictions": {
                    "original": 8.5,
                    "learning": 9.2,
                    "primary": 9.2,  # Currently used prediction
                    "market": 8.0
                },
                "comparison": {
                    "difference": 0.7,
                    "learning_higher": true,
                    "confidence": "medium"
                },
                "actual": null,  # Will be filled after game completes
                "status": "upcoming"
            }
        ],
        "summary": {
            "total_games": 15,
            "avg_difference": 0.45,
            "learning_higher_count": 8,
            "original_higher_count": 7
        }
    }
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    engine = create_engine(get_database_url())
    
    query = text("""
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            predicted_total,
            predicted_total_original,
            predicted_total_learning,
            market_total,
            total_runs,
            prediction_timestamp,
            
            -- Calculate status
            CASE 
                WHEN total_runs IS NOT NULL THEN 'completed'
                WHEN date < CURRENT_DATE THEN 'in_progress'
                ELSE 'upcoming'
            END as status
            
        FROM enhanced_games
        WHERE date = :target_date
        AND (predicted_total_original IS NOT NULL OR predicted_total_learning IS NOT NULL)
        ORDER BY prediction_timestamp DESC;
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'target_date': target_date})
        
        if df.empty:
            log.warning(f"No dual predictions found for {target_date}")
            return {
                "date": target_date,
                "games": [],
                "summary": {
                    "total_games": 0,
                    "avg_difference": 0,
                    "learning_higher_count": 0,
                    "original_higher_count": 0
                }
            }
        
        # Process games
        games = []
        differences = []
        learning_higher = 0
        original_higher = 0
        
        for _, row in df.iterrows():
            game_data = {
                "game_id": row['game_id'],
                "home_team": row['home_team'],
                "away_team": row['away_team'],
                "predictions": {
                    "original": float(row['predicted_total_original']) if pd.notna(row['predicted_total_original']) else None,
                    "learning": float(row['predicted_total_learning']) if pd.notna(row['predicted_total_learning']) else None,
                    "primary": float(row['predicted_total']) if pd.notna(row['predicted_total']) else None,
                    "market": float(row['market_total']) if pd.notna(row['market_total']) else None
                },
                "actual": float(row['total_runs']) if pd.notna(row['total_runs']) else None,
                "status": row['status'],
                "timestamp": row['prediction_timestamp'].isoformat() if pd.notna(row['prediction_timestamp']) else None
            }
            
            # Calculate comparison metrics
            orig_pred = game_data["predictions"]["original"]
            learn_pred = game_data["predictions"]["learning"]
            
            if orig_pred is not None and learn_pred is not None:
                difference = learn_pred - orig_pred
                differences.append(difference)
                
                game_data["comparison"] = {
                    "difference": round(difference, 2),
                    "learning_higher": difference > 0,
                    "confidence": "high" if abs(difference) > 1.0 else "medium" if abs(difference) > 0.5 else "low"
                }
                
                if difference > 0:
                    learning_higher += 1
                elif difference < 0:
                    original_higher += 1
            else:
                game_data["comparison"] = {
                    "difference": None,
                    "learning_higher": None,
                    "confidence": "unknown"
                }
            
            games.append(game_data)
        
        # Calculate summary
        summary = {
            "total_games": len(games),
            "avg_difference": round(sum(differences) / len(differences), 3) if differences else 0,
            "learning_higher_count": learning_higher,
            "original_higher_count": original_higher,
            "both_models_count": len([g for g in games if g["predictions"]["original"] is not None and g["predictions"]["learning"] is not None])
        }
        
        result = {
            "date": target_date,
            "games": games,
            "summary": summary
        }
        
        log.info(f"✅ Retrieved dual predictions for {target_date}: {len(games)} games")
        return result
        
    except Exception as e:
        log.error(f"Failed to get dual predictions: {e}")
        raise
    finally:
        engine.dispose()

def save_dual_predictions_json(target_date: str = None, output_dir: str = None):
    """Save dual predictions to JSON file for UI consumption"""
    
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "data")
    
    # Get dual predictions
    data = get_dual_predictions(target_date)
    
    # Save to file
    output_file = Path(output_dir) / f"dual_predictions_{target_date}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    log.info(f"💾 Saved dual predictions to {output_file}")
    
    # Also save as "latest" for UI
    latest_file = Path(output_dir) / "dual_predictions_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    log.info(f"💾 Saved latest dual predictions to {latest_file}")
    
    return output_file

def get_prediction_performance_summary(days_back: int = 7) -> dict:
    """
    Get performance summary comparing both models over the last N days
    """
    engine = create_engine(get_database_url())
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    query = text("""
        SELECT 
            date,
            COUNT(*) as total_games,
            COUNT(CASE WHEN predicted_total_original IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as original_with_results,
            COUNT(CASE WHEN predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as learning_with_results,
            
            -- Original model performance
            AVG(CASE WHEN predicted_total_original IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(predicted_total_original - total_runs) END) as original_mae,
            
            -- Learning model performance  
            AVG(CASE WHEN predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(predicted_total_learning - total_runs) END) as learning_mae,
            
            -- Market performance
            AVG(CASE WHEN market_total IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(market_total - total_runs) END) as market_mae,
                
            -- Count where learning model was better
            COUNT(CASE WHEN predicted_total_original IS NOT NULL AND predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL
                AND ABS(predicted_total_learning - total_runs) < ABS(predicted_total_original - total_runs) 
                THEN 1 END) as learning_wins,
                
            -- Count where original model was better
            COUNT(CASE WHEN predicted_total_original IS NOT NULL AND predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL
                AND ABS(predicted_total_original - total_runs) < ABS(predicted_total_learning - total_runs) 
                THEN 1 END) as original_wins
            
        FROM enhanced_games
        WHERE date BETWEEN :start_date AND :end_date
        AND total_runs IS NOT NULL
        GROUP BY date
        ORDER BY date DESC;
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        
        if df.empty:
            return {"error": "No completed games with predictions found"}
        
        # Calculate overall performance
        total_original_mae = df['original_mae'].mean()
        total_learning_mae = df['learning_mae'].mean()
        total_market_mae = df['market_mae'].mean()
        
        total_learning_wins = df['learning_wins'].sum()
        total_original_wins = df['original_wins'].sum()
        total_comparisons = total_learning_wins + total_original_wins
        
        summary = {
            "period": f"{start_date} to {end_date}",
            "days_analyzed": len(df),
            "performance": {
                "original_model_mae": round(float(total_original_mae), 3) if pd.notna(total_original_mae) else None,
                "learning_model_mae": round(float(total_learning_mae), 3) if pd.notna(total_learning_mae) else None,
                "market_mae": round(float(total_market_mae), 3) if pd.notna(total_market_mae) else None
            },
            "head_to_head": {
                "learning_wins": int(total_learning_wins),
                "original_wins": int(total_original_wins),
                "total_comparisons": int(total_comparisons),
                "learning_win_rate": round(total_learning_wins / total_comparisons, 3) if total_comparisons > 0 else 0
            },
            "daily_breakdown": df.to_dict('records')
        }
        
        log.info(f"📊 Performance summary for {days_back} days:")
        log.info(f"   Learning model MAE: {summary['performance']['learning_model_mae']}")
        log.info(f"   Original model MAE: {summary['performance']['original_model_mae']}")
        log.info(f"   Learning win rate: {summary['head_to_head']['learning_win_rate']*100:.1f}%")
        
        return summary
        
    except Exception as e:
        log.error(f"Failed to get performance summary: {e}")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dual prediction tracking data for UI')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)', default=None)
    parser.add_argument('--output-dir', type=str, help='Output directory for JSON files', default=None)
    parser.add_argument('--performance', action='store_true', help='Show performance summary')
    parser.add_argument('--days-back', type=int, help='Days back for performance summary', default=7)
    
    args = parser.parse_args()
    
    try:
        if args.performance:
            summary = get_prediction_performance_summary(args.days_back)
            print(json.dumps(summary, indent=2, default=str))
        else:
            output_file = save_dual_predictions_json(args.date, args.output_dir)
            print(f"✅ Dual predictions saved to: {output_file}")
            
    except Exception as e:
        log.error(f"Failed: {e}")
        sys.exit(1)
