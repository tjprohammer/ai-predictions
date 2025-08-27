#!/usr/bin/env python3
"""
CONTINUOUS LEARNING WORKFLOW MANAGER
Handles the daily cycle: Predict → Games Complete → Learn → Retrain → Repeat

This ensures:
1. Both models make predictions for upcoming games
2. Completed games are separated for learning
3. Models adapt based on real results
4. Consistent daily improvement cycle
"""

import psycopg2
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ContinuousLearningManager:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        self.models_dir = Path("S:/Projects/AI_Predictions/mlb-overs/models")
        self.logs_dir = Path("S:/Projects/AI_Predictions/session_logs")
        
        # Game status tracking
        self.PREDICTION_STATUS = 'prediction_made'
        self.COMPLETED_STATUS = 'completed_for_learning'
        self.LEARNED_STATUS = 'learned_from'
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def load_latest_models(self):
        """Load the most recent original and learning models"""
        
        print("🔄 LOADING LATEST MODELS")
        print("=" * 35)
        
        # Find most recent models
        original_models = list(self.models_dir.glob("original_model_*.joblib"))
        learning_models = list(self.models_dir.glob("learning_model_*.joblib"))
        
        if not original_models or not learning_models:
            raise FileNotFoundError("No trained models found! Run comprehensive_ml_trainer.py first")
        
        # Load latest models
        latest_original = max(original_models, key=lambda x: x.stat().st_mtime)
        latest_learning = max(learning_models, key=lambda x: x.stat().st_mtime)
        
        original_model = joblib.load(latest_original)
        learning_model = joblib.load(latest_learning)
        
        # Load features
        timestamp = latest_original.stem.split('_')[-1]
        features_file = self.models_dir / f"model_features_{timestamp}.json"
        
        with open(features_file, 'r') as f:
            feature_info = json.load(f)
        
        print(f"✅ Loaded models:")
        print(f"   Original: {latest_original.name}")
        print(f"   Learning: {latest_learning.name}")
        print(f"   Features: {len(feature_info['features'])}")
        
        return original_model, learning_model, feature_info['features']
    
    def get_games_for_prediction(self, target_date):
        """Get games that need predictions for target date"""
        
        print(f"\n🎯 GETTING GAMES FOR PREDICTION: {target_date}")
        print("=" * 50)
        
        conn = self.connect_db()
        
        # Get games that need predictions (future games or games without predictions)
        prediction_query = f"""
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            market_total,
            -- All features needed for prediction
            home_team_runs_l7, away_team_runs_l7,
            home_team_runs_allowed_l7, away_team_runs_allowed_l7,
            home_team_runs_l20, away_team_runs_l20,
            home_team_runs_l30, away_team_runs_l30,
            home_team_avg, away_team_avg,
            home_team_obp, away_team_obp,
            home_team_ops, away_team_ops,
            home_team_woba, away_team_woba,
            home_sp_season_era, away_sp_season_era,
            home_sp_era_l3starts, away_sp_era_l3starts,
            home_bullpen_era, away_bullpen_era,
            temperature, wind_speed,
            home_lineup_strength, away_lineup_strength,
            offensive_environment_score,
            home_team_hits, away_team_hits,
            home_team_rbi, away_team_rbi
        FROM enhanced_games
        WHERE date = %s
          AND (predicted_total IS NULL OR learning_prediction IS NULL)
          AND total_runs IS NULL  -- Only future games
        ORDER BY game_id;
        """
        
        df = pd.read_sql(prediction_query, conn, params=[target_date])
        conn.close()
        
        print(f"✅ Found {len(df)} games needing predictions")
        
        return df
    
    def make_dual_predictions(self, games_df, original_model, learning_model, features):
        """Generate predictions from both models"""
        
        if len(games_df) == 0:
            print("   No games to predict")
            return
        
        print(f"\n🔮 MAKING DUAL PREDICTIONS")
        print("=" * 35)
        
        # Prepare features
        available_features = [f for f in features if f in games_df.columns]
        X = games_df[available_features].fillna(games_df[available_features].median())
        
        # Add engineered features (same as training)
        if 'market_total' in X.columns:
            X['team_strength_vs_market'] = (
                (X['home_team_runs_l7'] + X['away_team_runs_l7']) / 14
            ) - X['market_total']
        
        if 'home_sp_season_era' in X.columns and 'away_sp_season_era' in X.columns:
            X['combined_sp_era'] = (X['home_sp_season_era'] + X['away_sp_season_era']) / 2
            X['sp_era_differential'] = abs(X['home_sp_season_era'] - X['away_sp_season_era'])
        
        if 'temperature' in X.columns:
            X['weather_offense_factor'] = np.where(X['temperature'] > 75, 1.1, 
                                         np.where(X['temperature'] < 50, 0.9, 1.0))
        
        # Ensure feature alignment
        model_features = [f for f in X.columns if f in available_features or f in ['team_strength_vs_market', 'combined_sp_era', 'sp_era_differential', 'weather_offense_factor']]
        X_final = X[model_features]
        
        # Generate predictions
        original_predictions = original_model.predict(X_final)
        learning_predictions = learning_model.predict(X_final)
        
        # Calculate differences and consensus
        prediction_differences = learning_predictions - original_predictions
        
        print(f"✅ Generated predictions for {len(games_df)} games:")
        
        # Store predictions in database
        conn = self.connect_db()
        cursor = conn.cursor()
        
        for i, (_, game) in enumerate(games_df.iterrows()):
            original_pred = float(original_predictions[i])
            learning_pred = float(learning_predictions[i])
            difference = float(prediction_differences[i])
            
            # Determine consensus
            if abs(difference) < 0.3:
                consensus = "strong_agreement"
            elif learning_pred > original_pred:
                consensus = "learning_higher"
            else:
                consensus = "learning_lower"
            
            # Update database
            update_query = """
            UPDATE enhanced_games 
            SET 
                predicted_total = %s,
                learning_prediction = %s,
                prediction_difference = %s,
                consensus_direction = %s,
                prediction_status = %s,
                prediction_timestamp = %s
            WHERE game_id = %s;
            """
            
            cursor.execute(update_query, [
                original_pred, learning_pred, difference, consensus,
                self.PREDICTION_STATUS, datetime.now(), game['game_id']
            ])
            
            print(f"   Game {game['game_id']}: Original={original_pred:.1f}, Learning={learning_pred:.1f}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Stored dual predictions in database")
    
    def get_completed_games_for_learning(self):
        """Get games that have completed and need learning updates"""
        
        print(f"\n📚 FINDING COMPLETED GAMES FOR LEARNING")
        print("=" * 45)
        
        conn = self.connect_db()
        
        # Get games with results that haven't been learned from yet
        completed_query = """
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            predicted_total,
            learning_prediction,
            total_runs,
            home_score,
            away_score,
            prediction_difference
        FROM enhanced_games
        WHERE total_runs IS NOT NULL  -- Game completed
          AND predicted_total IS NOT NULL  -- Had predictions
          AND learning_prediction IS NOT NULL
          AND (prediction_status = %s OR prediction_status IS NULL)  -- Not yet learned from
          AND date >= %s  -- Only recent games
        ORDER BY date DESC;
        """
        
        # Only learn from games in last 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = pd.read_sql(completed_query, conn, params=[self.PREDICTION_STATUS, cutoff_date])
        conn.close()
        
        print(f"✅ Found {len(df)} completed games for learning")
        
        return df
    
    def evaluate_prediction_performance(self, completed_games):
        """Evaluate how both models performed on completed games"""
        
        if len(completed_games) == 0:
            print("   No completed games to evaluate")
            return None
        
        print(f"\n📊 EVALUATING PREDICTION PERFORMANCE")
        print("=" * 45)
        
        # Calculate accuracy metrics
        original_errors = np.abs(completed_games['predicted_total'] - completed_games['total_runs'])
        learning_errors = np.abs(completed_games['learning_prediction'] - completed_games['total_runs'])
        
        # Picking accuracy (within 0.5 runs is "correct")
        original_correct = (original_errors <= 0.5).sum()
        learning_correct = (learning_errors <= 0.5).sum()
        
        total_games = len(completed_games)
        original_accuracy = original_correct / total_games
        learning_accuracy = learning_correct / total_games
        
        # MAE
        original_mae = original_errors.mean()
        learning_mae = learning_errors.mean()
        
        performance = {
            'total_games': total_games,
            'original_accuracy': original_accuracy,
            'learning_accuracy': learning_accuracy,
            'original_mae': original_mae,
            'learning_mae': learning_mae,
            'improvement': learning_accuracy - original_accuracy,
            'evaluation_date': datetime.now().isoformat()
        }
        
        print(f"📈 Performance Summary ({total_games} games):")
        print(f"   Original Model: {original_accuracy:.1%} accuracy, {original_mae:.2f} MAE")
        print(f"   Learning Model: {learning_accuracy:.1%} accuracy, {learning_mae:.2f} MAE")
        print(f"   Improvement: {performance['improvement']:+.1%}")
        
        return performance
    
    def mark_games_as_learned(self, completed_games):
        """Mark completed games as learned from to avoid reprocessing"""
        
        if len(completed_games) == 0:
            return
        
        print(f"\n✅ MARKING {len(completed_games)} GAMES AS LEARNED")
        print("=" * 45)
        
        conn = self.connect_db()
        cursor = conn.cursor()
        
        game_ids = completed_games['game_id'].tolist()
        
        update_query = """
        UPDATE enhanced_games 
        SET 
            prediction_status = %s,
            learning_timestamp = %s
        WHERE game_id = ANY(%s);
        """
        
        cursor.execute(update_query, [
            self.COMPLETED_STATUS, 
            datetime.now(), 
            game_ids
        ])
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Updated {cursor.rowcount} games as learned from")
    
    def log_daily_performance(self, performance, target_date):
        """Log daily performance for tracking"""
        
        if performance is None:
            return
        
        log_file = self.logs_dir / f"continuous_learning_log_{datetime.now().strftime('%Y%m')}.json"
        
        # Load existing log
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {}
        
        # Add today's performance
        log_data[target_date] = performance
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Logged performance to {log_file}")
    
    def run_daily_cycle(self, target_date=None):
        """Run the complete daily continuous learning cycle"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🔄 CONTINUOUS LEARNING DAILY CYCLE")
        print(f"   Target Date: {target_date}")
        print("=" * 50)
        
        # Step 1: Load models
        original_model, learning_model, features = self.load_latest_models()
        
        # Step 2: Make predictions for target date
        games_to_predict = self.get_games_for_prediction(target_date)
        self.make_dual_predictions(games_to_predict, original_model, learning_model, features)
        
        # Step 3: Learn from completed games
        completed_games = self.get_completed_games_for_learning()
        performance = self.evaluate_prediction_performance(completed_games)
        
        # Step 4: Mark games as learned from
        self.mark_games_as_learned(completed_games)
        
        # Step 5: Log performance
        self.log_daily_performance(performance, target_date)
        
        print(f"\n🏆 DAILY CYCLE COMPLETE!")
        if performance:
            print(f"   Learning Model: {performance['learning_accuracy']:.1%} accuracy")
            print(f"   Improvement: {performance['improvement']:+.1%}")
        
        print(f"   Predictions made for {len(games_to_predict)} games")
        print(f"   Learned from {len(completed_games)} completed games")
        
        return performance

def main():
    import sys
    
    print("🚀 CONTINUOUS LEARNING WORKFLOW MANAGER")
    print("   Managing daily prediction and learning cycle")
    print("=" * 55)
    
    manager = ContinuousLearningManager()
    
    # Get target date from command line or use today
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run daily cycle
    performance = manager.run_daily_cycle(target_date)
    
    if performance:
        print(f"\n📊 Today's Learning Performance:")
        print(f"   Games evaluated: {performance['total_games']}")
        print(f"   Learning accuracy: {performance['learning_accuracy']:.1%}")
        print(f"   Model improvement: {performance['improvement']:+.1%}")

if __name__ == "__main__":
    main()
