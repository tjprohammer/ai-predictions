#!/usr/bin/env python3
"""
Learning Impact Tracker
========================
Compare predictions before and after incorporating 20-session learning insights
to measure the improvement in prediction accuracy.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

class LearningImpactTracker:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
        self.engine = create_engine(self.db_url)
        self.results_dir = Path(".")  # Use current directory for results
        self.results_dir.mkdir(exist_ok=True)
        
    def get_data_for_comparison(self, days_back=60):
        """Get data for comparing model performance"""
        start_date = (datetime.now() - timedelta(days=days_back)).date()
        end_date = datetime.now().date()
        
        query = """
        SELECT * FROM enhanced_games 
        WHERE date >= %(start_date)s AND date <= %(end_date)s
        AND total_runs IS NOT NULL
        ORDER BY date DESC
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
            
        print(f"📊 Retrieved {len(df)} games for comparison testing")
        return df
    
    def train_baseline_model(self, df):
        """Train a basic model (old approach) - numeric features only"""
        print("🏗️ Training baseline model (old approach)...")
        
        # Old approach - only numeric features, basic preprocessing
        X = df.select_dtypes(include=[np.number]).drop(columns=['total_runs', 'id'], errors='ignore')
        y = df['total_runs']
        
        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Basic preprocessing
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Basic model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        baseline_metrics = {
            'features_used': len(X.columns),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        print(f"   Features: {baseline_metrics['features_used']}")
        print(f"   Test R²: {baseline_metrics['test_r2']:.3f}")
        print(f"   Test MAE: {baseline_metrics['test_mae']:.3f}")
        
        return model, baseline_metrics, X_test, y_test
    
    def train_enhanced_model(self, df):
        """Train enhanced model with 20-session learning insights"""
        print("🚀 Training enhanced model (with learning insights)...")
        
        # Import from current directory
        import sys
        sys.path.append('.')
        sys.path.append('..')
        sys.path.append('../..')
        sys.path.append('../../core')  # Add MLB core directory
        
        try:
            from adaptive_learning_pipeline import AdaptiveLearningPipeline
        except ImportError:
            # Try alternative import paths
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("adaptive_learning_pipeline", 
                                                           "adaptive_learning_pipeline.py")
                if spec is None:
                    spec = importlib.util.spec_from_file_location("adaptive_learning_pipeline", 
                                                               "../adaptive_learning_pipeline.py")
                if spec is None:
                    spec = importlib.util.spec_from_file_location("adaptive_learning_pipeline", 
                                                               "../../core/adaptive_learning_pipeline.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                AdaptiveLearningPipeline = module.AdaptiveLearningPipeline
            except:
                # Fallback - create a simplified version
                print("   ⚠️ Using simplified preprocessing (can't import full pipeline)")
                return self._train_enhanced_model_simplified(df)
        
        # Use the adaptive learning pipeline
        pipeline = AdaptiveLearningPipeline()
        
        # Prepare data
        exclude_cols = ['total_runs', 'id', 'game_id', 'date', 'actual_total', 'home_score', 'away_score']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df['total_runs']
        
        # Remove missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Comprehensive preprocessing
        X_processed = pipeline.comprehensive_feature_preprocessing(X)
        X_weighted = pipeline.apply_learning_weights(X_processed)
        
        # Split data (same random state for fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.3, random_state=42)
        
        # Enhanced model with optimal parameters from 20-session learning
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        enhanced_metrics = {
            'features_used': len(X_weighted.columns),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        print(f"   Features: {enhanced_metrics['features_used']}")
        print(f"   Test R²: {enhanced_metrics['test_r2']:.3f}")
        print(f"   Test MAE: {enhanced_metrics['test_mae']:.3f}")
        
        return model, enhanced_metrics, X_test, y_test
    
    def compare_models(self, baseline_metrics, enhanced_metrics):
        """Compare baseline vs enhanced model performance"""
        print("\n📊 MODEL COMPARISON RESULTS")
        print("="*50)
        
        # Calculate improvements
        improvements = {}
        for metric in ['test_r2', 'test_mae', 'test_rmse']:
            baseline_val = baseline_metrics[metric]
            enhanced_val = enhanced_metrics[metric]
            
            if metric == 'test_r2':
                # Higher is better for R²
                improvement = (enhanced_val - baseline_val) / baseline_val * 100
            else:
                # Lower is better for MAE/RMSE
                improvement = (baseline_val - enhanced_val) / baseline_val * 100
            
            improvements[metric] = improvement
        
        # Display comparison
        print(f"🔧 Baseline Model (Old Approach):")
        print(f"   Features Used: {baseline_metrics['features_used']}")
        print(f"   Test R²: {baseline_metrics['test_r2']:.3f}")
        print(f"   Test MAE: {baseline_metrics['test_mae']:.3f}")
        print(f"   Test RMSE: {baseline_metrics['test_rmse']:.3f}")
        
        print(f"\n🚀 Enhanced Model (With Learning):")
        print(f"   Features Used: {enhanced_metrics['features_used']}")
        print(f"   Test R²: {enhanced_metrics['test_r2']:.3f}")
        print(f"   Test MAE: {enhanced_metrics['test_mae']:.3f}")
        print(f"   Test RMSE: {enhanced_metrics['test_rmse']:.3f}")
        
        print(f"\n📈 IMPROVEMENTS:")
        print(f"   Feature Count: +{enhanced_metrics['features_used'] - baseline_metrics['features_used']} features")
        print(f"   R² Score: {improvements['test_r2']:+.1f}%")
        print(f"   MAE (Error): {improvements['test_mae']:+.1f}%")
        print(f"   RMSE (Error): {improvements['test_rmse']:+.1f}%")
        
        # Overall assessment
        avg_improvement = np.mean([improvements['test_r2'], improvements['test_mae'], improvements['test_rmse']])
        print(f"\n🎯 OVERALL IMPROVEMENT: {avg_improvement:+.1f}%")
        
        if avg_improvement > 5:
            print("✅ SIGNIFICANT IMPROVEMENT - Learning insights are working!")
        elif avg_improvement > 0:
            print("🔄 MODERATE IMPROVEMENT - Learning insights show promise")
        else:
            print("⚠️ MINIMAL IMPROVEMENT - May need to refine learning approach")
        
        # Save comparison results
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': baseline_metrics,
            'enhanced_metrics': enhanced_metrics,
            'improvements': improvements,
            'overall_improvement': avg_improvement
        }
        
        results_file = self.results_dir / f"learning_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return comparison_results
    
    def track_prediction_accuracy(self, days_to_track=7):
        """Track prediction accuracy over time"""
        print(f"\n🎯 Tracking prediction accuracy over last {days_to_track} days...")
        
        # Get recent predictions and actual results
        query = """
        SELECT p.game_id, p.predicted_total, p.game_date, g.total_runs as actual_total
        FROM probability_predictions p
        JOIN enhanced_games g ON p.game_id = g.game_id
        WHERE p.game_date >= %(start_date)s
        AND g.total_runs IS NOT NULL
        ORDER BY p.game_date DESC
        """
        
        start_date = (datetime.now() - timedelta(days=days_to_track)).date()
        
        with self.engine.connect() as conn:
            results_df = pd.read_sql(query, conn, params={'start_date': start_date})
        
        if len(results_df) == 0:
            print("⚠️ No recent predictions found to track")
            return None
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(results_df['actual_total'], results_df['predicted_total'])
        rmse = np.sqrt(mean_squared_error(results_df['actual_total'], results_df['predicted_total']))
        r2 = r2_score(results_df['actual_total'], results_df['predicted_total'])
        
        # Compare to session 11 target (0.898 MAE)
        target_mae = 0.898
        improvement_from_target = (target_mae - mae) / target_mae * 100
        
        print(f"📊 Recent Prediction Accuracy ({len(results_df)} games):")
        print(f"   MAE: {mae:.3f} (Target: {target_mae:.3f})")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   R²: {r2:.3f}")
        print(f"   vs Target: {improvement_from_target:+.1f}%")
        
        if mae <= target_mae:
            print("✅ MEETING TARGET - Learning insights are effective!")
        else:
            print("🔄 ROOM FOR IMPROVEMENT - Continue refining approach")
        
        return {
            'games_tracked': len(results_df),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'target_mae': target_mae,
            'improvement_from_target': improvement_from_target
        }

def main():
    """Main execution for learning impact tracking"""
    tracker = LearningImpactTracker()
    
    # Get comparison data
    print("🔄 Loading data for model comparison...")
    df = tracker.get_data_for_comparison(days_back=60)
    
    if len(df) < 50:
        print("❌ Insufficient data for reliable comparison")
        return
    
    # Train both models
    baseline_model, baseline_metrics, _, _ = tracker.train_baseline_model(df)
    enhanced_model, enhanced_metrics, _, _ = tracker.train_enhanced_model(df)
    
    # Compare performance
    comparison_results = tracker.compare_models(baseline_metrics, enhanced_metrics)
    
    # Track recent accuracy
    accuracy_tracking = tracker.track_prediction_accuracy(days_to_track=14)
    
    print(f"\n🏁 Learning Impact Analysis Complete!")

if __name__ == "__main__":
    main()
