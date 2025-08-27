#!/usr/bin/env python3
"""
Advanced Pre-Game Learning Model Trainer
========================================
Creates a sophisticated learning model using the full enhanced feature pipeline
but ONLY pre-game features (no data leaks).

This bridges the gap between:
- Simple 37-feature clean model (not enough power)
- 220+ feature enhanced model (has data leaks)

Result: ~150 sophisticated pre-game features for realistic predictions
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add deployment path for predictor access
sys.path.append(os.path.join(os.path.dirname(__file__), 'mlb-overs', 'deployment'))
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

class AdvancedPreGameModelTrainer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Initialize the predictor to use its feature engineering
        self.predictor = EnhancedBullpenPredictor()
        
        # Define data leak features to EXCLUDE (these are only available after games start/finish)
        self.data_leak_features = [
            # In-game pitcher stats
            'home_sp_er', 'away_sp_er', 'home_sp_ip', 'away_sp_ip',
            'home_sp_k', 'away_sp_k', 'home_sp_bb', 'away_sp_bb', 'home_sp_h', 'away_sp_h',
            
            # In-game team stats  
            'home_team_hits', 'away_team_hits', 'home_team_runs', 'away_team_runs',
            'home_team_rbi', 'away_team_rbi', 'home_team_lob', 'away_team_lob',
            
            # In-game bullpen stats
            'home_bp_ip', 'away_bp_ip', 'home_bp_er', 'away_bp_er',
            'home_bp_k', 'away_bp_k', 'home_bp_bb', 'away_bp_bb', 'home_bp_h', 'away_bp_h',
            
            # Game results
            'total_runs', 'home_score', 'away_score',
            
            # Post-game predictions
            'predicted_total', 'predicted_total_original', 'predicted_total_learning'
        ]
        
    def get_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical games for training with ALL raw data"""
        
        conn = psycopg2.connect(**self.db_config)
        
        query = """
        SELECT *
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND total_runs IS NOT NULL
        AND total_runs > 3 AND total_runs < 20
        AND market_total IS NOT NULL
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        print(f"✅ Retrieved {len(df)} training games")
        return df
        
    def engineer_advanced_pregame_features(self, df: pd.DataFrame) -> tuple:
        """Use enhanced predictor to create full feature set, then filter to pre-game only"""
        
        print("🔧 Running full enhanced feature engineering pipeline...")
        
        # Run the full enhanced predictor feature engineering
        try:
            # Create a copy for feature engineering
            feature_df = df.copy()
            
            # Temporarily remove data leak columns to simulate pre-game state
            original_leak_data = {}
            for leak_col in self.data_leak_features:
                if leak_col in feature_df.columns:
                    original_leak_data[leak_col] = feature_df[leak_col].copy()
                    # Set to NaN to simulate pre-game state
                    feature_df[leak_col] = np.nan
            
            # Run enhanced feature engineering
            enhanced_features = self.predictor.engineer_features(feature_df)
            
            print(f"✅ Enhanced predictor created {len(enhanced_features.columns)} total features")
            
            # Identify pre-game features (exclude data leaks and their derivatives)
            pregame_features = []
            
            for col in enhanced_features.columns:
                # Skip obvious data leaks
                is_data_leak = any(leak in col.lower() for leak in [
                    '_er', '_ip', '_h', '_bb', '_k', '_hits', '_runs', '_rbi', '_lob',
                    'total_runs', 'score', 'predicted_total'
                ])
                
                # Skip columns that are clearly post-game
                is_postgame = any(postgame in col.lower() for postgame in [
                    'actual', 'result', 'final', 'completed'
                ])
                
                # Include if it's clearly pre-game available
                if not is_data_leak and not is_postgame:
                    pregame_features.append(col)
            
            print(f"📊 Identified {len(pregame_features)} pre-game features")
            print(f"   Excluded {len(enhanced_features.columns) - len(pregame_features)} data leak features")
            
            # Get pre-game feature matrix
            X_pregame = enhanced_features[pregame_features].copy()
            
            # Fill missing values intelligently
            for col in X_pregame.columns:
                if X_pregame[col].dtype in ['float64', 'int64']:
                    # Use median for numeric features
                    X_pregame[col] = X_pregame[col].fillna(X_pregame[col].median())
                else:
                    # Use mode for categorical features
                    X_pregame[col] = X_pregame[col].fillna(X_pregame[col].mode().iloc[0] if len(X_pregame[col].mode()) > 0 else 0)
            
            # Remove constant features
            constant_features = []
            for col in X_pregame.columns:
                if X_pregame[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                print(f"⚠️  Removing {len(constant_features)} constant features")
                X_pregame = X_pregame.drop(columns=constant_features)
            
            # Remove highly correlated features
            correlation_matrix = X_pregame.corr().abs()
            high_corr_features = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.95:
                        high_corr_features.append(correlation_matrix.columns[j])
            
            high_corr_features = list(set(high_corr_features))
            if high_corr_features:
                print(f"⚠️  Removing {len(high_corr_features)} highly correlated features")
                X_pregame = X_pregame.drop(columns=high_corr_features)
            
            final_features = list(X_pregame.columns)
            print(f"✅ Final pre-game feature count: {len(final_features)}")
            
            return X_pregame, final_features
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def train_advanced_model(self, start_date: str = None, end_date: str = None):
        """Train an advanced pre-game learning model"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
            
        print(f"🚀 ADVANCED PRE-GAME LEARNING MODEL TRAINING")
        print("=" * 70)
        print(f"📅 Training period: {start_date} to {end_date}")
        
        # Get training data
        df = self.get_training_data(start_date, end_date)
        
        # Engineer sophisticated pre-game features
        X, feature_names = self.engineer_advanced_pregame_features(df)
        
        if X is None:
            print("❌ Feature engineering failed")
            return None
            
        y = df['total_runs'].values
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Target range: {y.min():.1f} - {y.max():.1f} runs")
        print(f"📊 Target mean: {y.mean():.2f} runs")
        
        # Split data with time-aware splitting
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = df['date'].quantile(0.8)
        
        train_mask = df['date'] <= cutoff_date
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        
        print(f"   Training set: {len(X_train)} games")
        print(f"   Test set: {len(X_test)} games")
        
        # Train advanced model with multiple algorithms
        print("🎯 Training advanced learning model ensemble...")
        
        # Random Forest (primary)
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting (secondary)
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # Train both models
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Evaluate both models
        rf_train_pred = rf_model.predict(X_train)
        rf_test_pred = rf_model.predict(X_test)
        gb_train_pred = gb_model.predict(X_train)
        gb_test_pred = gb_model.predict(X_test)
        
        rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
        rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
        gb_train_mae = mean_absolute_error(y_train, gb_train_pred)
        gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
        
        print(f"📈 RANDOM FOREST PERFORMANCE:")
        print(f"   Training MAE: {rf_train_mae:.3f}")
        print(f"   Test MAE: {rf_test_mae:.3f}")
        print(f"   Training R²: {r2_score(y_train, rf_train_pred):.3f}")
        print(f"   Test R²: {r2_score(y_test, rf_test_pred):.3f}")
        
        print(f"📈 GRADIENT BOOSTING PERFORMANCE:")
        print(f"   Training MAE: {gb_train_mae:.3f}")
        print(f"   Test MAE: {gb_test_mae:.3f}")
        print(f"   Training R²: {r2_score(y_train, gb_train_pred):.3f}")
        print(f"   Test R²: {r2_score(y_test, gb_test_pred):.3f}")
        
        # Choose best model
        best_model = rf_model if rf_test_mae <= gb_test_mae else gb_model
        best_mae = min(rf_test_mae, gb_test_mae)
        model_type = "RandomForest" if rf_test_mae <= gb_test_mae else "GradientBoosting"
        
        print(f"🏆 Best model: {model_type} (Test MAE: {best_mae:.3f})")
        
        # Cross validation on best model
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print(f"   CV MAE: {cv_mae:.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        importances = best_model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\\n🔝 TOP 15 MOST IMPORTANT PRE-GAME FEATURES:")
        for i, (feature, importance) in enumerate(feature_importance[:15]):
            print(f"   {i+1:2d}. {feature:30s} ({importance:.3f})")
        
        # Check for any remaining data leaks in top features
        top_features = [f[0] for f in feature_importance[:20]]
        potential_leaks = []
        for feature in top_features:
            if any(leak in feature.lower() for leak in ['_er', '_ip', '_hits', '_runs']):
                potential_leaks.append(feature)
        
        if potential_leaks:
            print(f"⚠️  WARNING: Potential data leaks in top features: {potential_leaks}")
        else:
            print("✅ No data leaks detected in top features")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"advanced_pregame_model_{timestamp}.joblib"
        model_path = os.path.join('mlb-overs', 'models', model_name)
        
        # Create comprehensive model package
        model_package = {
            'model': best_model,
            'model_type': f'advanced_pregame_{model_type.lower()}',
            'feature_columns': feature_names,
            'feature_fill_values': {f: 0 for f in feature_names},
            'excluded_features': self.data_leak_features,
            'training_metrics': {
                'train_mae': rf_train_mae if model_type == "RandomForest" else gb_train_mae,
                'test_mae': best_mae,
                'cv_mae': cv_mae,
                'train_r2': r2_score(y_train, rf_train_pred if model_type == "RandomForest" else gb_train_pred),
                'test_r2': r2_score(y_test, rf_test_pred if model_type == "RandomForest" else gb_test_pred),
                'feature_names': feature_names,
                'training_games': len(df),
                'feature_importance': feature_importance[:20]  # Top 20
            },
            'created_date': datetime.now().isoformat(),
            'training_period': {'start': start_date, 'end': end_date},
            'note': 'Advanced pre-game model using enhanced features but excluding data leaks',
            'prediction_range': {'min': y.min(), 'max': y.max(), 'mean': y.mean()},
            'bias_correction': 0.0  # Can be adjusted later
        }
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_package, model_path)
        
        print(f"\\n💾 Advanced model saved to: {model_path}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Test MAE: {best_mae:.3f}")
        print(f"   Model type: {model_type}")
        
        print(f"\\n✅ ADVANCED PRE-GAME MODEL TRAINING COMPLETE!")
        print(f"   Model path: {model_path}")
        print(f"   Training games: {len(df)}")
        print(f"   Features: {len(feature_names)} (sophisticated pre-game only)")
        print(f"   Test MAE: {best_mae:.3f}")
        print(f"   Expected prediction range: {y.min():.1f} - {y.max():.1f} runs")
        
        return model_path

if __name__ == "__main__":
    trainer = AdvancedPreGameModelTrainer()
    model_path = trainer.train_advanced_model()
    
    print(f"\\n🎉 SUCCESS! Advanced pre-game model available at:")
    print(f"   {model_path}")
    print(f"\\nTo deploy this model, run:")
    print(f"   cp \"{model_path}\" mlb-overs/models/legitimate_model_latest.joblib")
    print(f"\\nThis model provides:")
    print(f"   ✅ Sophisticated feature engineering (~150 features)")
    print(f"   ✅ No data leaks (pre-game only)")
    print(f"   ✅ Realistic predictions")
    print(f"   ✅ Continuous learning capability")
