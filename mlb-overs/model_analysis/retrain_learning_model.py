"""
Retrain Learning Model with Current Feature Engineering Pipeline
This ensures the learning model uses the same features as the production system
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add deployment path for predictor access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deployment'))
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

class LearningModelRetrainer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Initialize the predictor to use its feature engineering
        self.predictor = EnhancedBullpenPredictor()
        
    def get_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical games for training with ALL raw data"""
        
        conn = psycopg2.connect(**self.db_config)
        
        query = """
        SELECT *
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND total_runs IS NOT NULL
        AND total_runs > 0
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        print(f"✅ Retrieved {len(df)} training games from {start_date} to {end_date}")
        print(f"   Raw columns: {len(df.columns)}")
        
        return df
        
    def engineer_features_for_training(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Apply the same feature engineering as production system"""
        
        print(f"🔧 Applying production feature engineering to {len(df)} games...")
        
        # Use the predictor's feature engineering pipeline
        featured_df = self.predictor.engineer_features(df)
        
        print(f"✅ Feature engineering complete: {len(featured_df.columns)} features generated")
        
        # Extract target variable
        target = df['total_runs'].copy()
        
        # Remove non-feature columns
        feature_cols_to_remove = [
            'id', 'game_id', 'date', 'home_team', 'away_team', 'total_runs',
            'home_score', 'away_score', 'created_at', 'market_total', 
            'predicted_total', 'confidence', 'recommendation', 'edge',
            'home_team_id', 'away_team_id', 'venue_id', 'home_sp_id', 'away_sp_id',
            'home_sp_name', 'away_sp_name', 'plate_umpire', 'venue', 'ballpark',
            'game_time_utc', 'game_timezone', 'home_catcher', 'away_catcher'
        ]
        
        # Remove columns that exist
        for col in feature_cols_to_remove:
            if col in featured_df.columns:
                featured_df = featured_df.drop(columns=[col])
                
        print(f"✅ Cleaned features: {len(featured_df.columns)} training features")
        
        # Handle any remaining non-numeric columns
        numeric_df = featured_df.select_dtypes(include=[np.number])
        print(f"✅ Numeric features: {len(numeric_df.columns)}")
        
        # Fill any NaN values
        numeric_df = numeric_df.fillna(0)
        
        # Remove any problematic or inconsistent features that aren't available in production
        problematic_features = [
            'home_sp_ip_l3', 'away_sp_ip_l3',
            'home_sp_pitch_ct_trend', 'away_sp_pitch_ct_trend', 
            'home_sp_tto_penalty', 'away_sp_tto_penalty',
            'home_sp_form_l5', 'away_sp_form_l5',
            'home_sp_consistency', 'away_sp_consistency'
        ]
        
        # Remove features that aren't consistently available
        features_removed = []
        for feature in problematic_features:
            if feature in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[feature])
                features_removed.append(feature)
                
        if features_removed:
            print(f"🗑️  Removed inconsistent features: {features_removed}")
        
        print(f"📊 Training data shape: {numeric_df.shape}")
        print(f"📊 Target shape: {target.shape}")
        
        return numeric_df, target
        
    def train_learning_model(self, X: pd.DataFrame, y: pd.Series) -> object:
        """Train a new learning model with current features"""
        
        print(f"🎯 Training learning model...")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"   Training set: {X_train.shape[0]} games")
        print(f"   Test set: {X_test.shape[0]} games")
        
        # Train Random Forest model (same as production system)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📈 MODEL PERFORMANCE:")
        print(f"   Training MAE: {train_mae:.3f}")
        print(f"   Test MAE: {test_mae:.3f}")
        print(f"   Training RMSE: {train_rmse:.3f}")
        print(f"   Test RMSE: {test_rmse:.3f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {i+1:2d}. {row['feature']:<25} ({row['importance']:.3f})")
            
        return model, {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse, 
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'feature_importance': feature_importance,
            'feature_names': list(X.columns)
        }
        
    def save_model(self, model: object, metrics: dict, model_name: str = None):
        """Save the retrained learning model"""
        
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"learning_model_retrained_{timestamp}.joblib"
            
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model with metadata
        model_package = {
            'model': model,
            'feature_names': metrics['feature_names'],
            'training_metrics': metrics,
            'created_date': datetime.now().isoformat(),
            'feature_engineering_version': 'production_current'
        }
        
        joblib.dump(model_package, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"   Features: {len(metrics['feature_names'])}")
        print(f"   Test MAE: {metrics['test_mae']:.3f}")
        
        return model_path
        
    def retrain_full_pipeline(self, days_back: int = 90) -> str:
        """Complete retraining pipeline"""
        
        print("🚀 LEARNING MODEL RETRAINING PIPELINE")
        print("=" * 60)
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"📅 Training period: {start_date} to {end_date} ({days_back} days)")
        
        try:
            # Step 1: Get training data
            training_data = self.get_training_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(training_data) < 50:
                raise ValueError(f"Insufficient training data: {len(training_data)} games")
                
            # Step 2: Engineer features
            X, y = self.engineer_features_for_training(training_data)
            
            if X.shape[1] == 0:
                raise ValueError("No features generated")
                
            # Step 3: Train model
            model, metrics = self.train_learning_model(X, y)
            
            # Step 4: Save model
            model_path = self.save_model(model, metrics)
            
            print("\n✅ RETRAINING COMPLETE!")
            print(f"   Model path: {model_path}")
            print(f"   Training games: {len(training_data)}")
            print(f"   Features: {X.shape[1]}")
            print(f"   Test MAE: {metrics['test_mae']:.3f}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def retrain_learning_model(days_back: int = 90) -> str:
    """Convenience function to retrain the learning model"""
    retrainer = LearningModelRetrainer()
    return retrainer.retrain_full_pipeline(days_back)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain learning model with current feature engineering')
    parser.add_argument('--days', type=int, default=90, 
                       help='Number of days of historical data to use (default: 90)')
    parser.add_argument('--test', action='store_true',
                       help='Run a quick test with recent data')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running test retraining with 30 days of data...")
        model_path = retrain_learning_model(30)
    else:
        print(f"🚀 Retraining with {args.days} days of historical data...")
        model_path = retrain_learning_model(args.days)
    
    if model_path:
        print(f"\n🎉 SUCCESS! Retrained model available at:")
        print(f"   {model_path}")
        print("\nTo use this model, update the learning_model_analyzer.py to load this new model.")
    else:
        print("\n❌ Retraining failed. Check the logs above for details.")
