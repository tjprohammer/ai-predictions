#!/usr/bin/env python3
"""
🎯 Enhanced Learning Integration System
Incorporates 20-session learnings into production pipeline and tracks improvement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import json
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedLearningIntegrator:
    """Integrates advanced 20-session learnings into production pipeline"""
    
    def __init__(self, db_path='S:/Projects/AI_Predictions/mlb-overs/data/mlb_data.db'):
        self.db_path = db_path
        self.feature_importance_weights = self._load_learned_weights()
        self.best_session_config = self._load_best_session_config()
        
    def _load_learned_weights(self):
        """Load feature importance weights from 20-session learning"""
        try:
            with open('fixed_session_logs/fixed_twenty_session_results.json', 'r') as f:
                data = json.load(f)
            
            # Get feature importance from best session (#11)
            best_session = None
            for session in data['sessions']:
                if session['session'] == 11:  # Best performing session
                    best_session = session
                    break
            
            if best_session:
                return {feature[0]: feature[1] for feature in best_session['top_features']}
            else:
                return {}
        except:
            print("⚠️  Could not load session results, using default weights")
            return {}
    
    def _load_best_session_config(self):
        """Load configuration from best performing session"""
        return {
            'target_mae': 0.898,  # Best session MAE
            'target_r2': 0.911,   # Best session R²
            'top_feature': 'away_team_rbi',
            'feature_dominance': {
                'core_baseball': 0.60,
                'score_based': 0.40
            }
        }
    
    def load_enhanced_data(self, start_date='2025-04-01', end_date=None):
        """Load data with enhanced feature engineering from learnings"""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🔄 Loading enhanced data from {start_date} to {end_date}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Enhanced query based on 20-session learnings
            query = """
            SELECT * FROM enhanced_game_data 
            WHERE game_date >= ? AND game_date <= ?
            AND total_runs IS NOT NULL
            ORDER BY game_date ASC
            """
            
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            conn.close()
            
            print(f"✅ Loaded {len(df)} games")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def engineer_enhanced_features(self, df):
        """Apply enhanced feature engineering based on learnings"""
        print("🔧 Applying enhanced feature engineering...")
        
        # Core baseball features (most important category - 45.4%)
        core_features = []
        
        # Top performing features from learning
        priority_features = [
            'away_team_rbi', 'away_team_runs', 'home_team_runs', 
            'home_team_rbi', 'away_team_hits', 'home_team_hits'
        ]
        
        for feature in priority_features:
            if feature in df.columns:
                core_features.append(feature)
                
                # Create rolling averages (learned from best sessions)
                if 'team' in feature:
                    team_side = feature.split('_')[0]  # 'home' or 'away'
                    stat = feature.split('_')[-1]      # 'rbi', 'runs', 'hits'
                    
                    # 5-game rolling average
                    df[f'{team_side}_team_{stat}_5game_avg'] = (
                        df.groupby(f'{team_side}_team')[feature]
                        .rolling(window=5, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
                    # Season momentum (last 10 games vs season average)
                    df[f'{team_side}_team_{stat}_momentum'] = (
                        df.groupby(f'{team_side}_team')[feature]
                        .rolling(window=10, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                        / 
                        df.groupby(f'{team_side}_team')[feature]
                        .expanding()
                        .mean()
                        .reset_index(0, drop=True)
                    ).fillna(1.0)
        
        # Pitching features (second most important - 27.2%)
        pitching_features = []
        for col in df.columns:
            if any(x in col.lower() for x in ['sp_', 'bp_', '_er', '_era', '_whip', '_ip']):
                pitching_features.append(col)
        
        # Enhanced pitching matchup features
        if 'home_sp_era' in df.columns and 'away_sp_era' in df.columns:
            df['pitching_advantage'] = df['away_sp_era'] - df['home_sp_era']
            df['pitching_quality_sum'] = df['home_sp_era'] + df['away_sp_era']
        
        # Team efficiency metrics (learned patterns)
        if all(col in df.columns for col in ['home_team_runs', 'home_team_hits']):
            df['home_team_efficiency'] = (
                df['home_team_runs'] / (df['home_team_hits'] + 1)
            ).fillna(0)
            
        if all(col in df.columns for col in ['away_team_runs', 'away_team_hits']):
            df['away_team_efficiency'] = (
                df['away_team_runs'] / (df['away_team_hits'] + 1)
            ).fillna(0)
        
        # Score prediction features (40% session dominance)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['score_differential'] = df['home_score'] - df['away_score']
            df['total_score'] = df['home_score'] + df['away_score']
        
        print(f"✅ Enhanced features engineered")
        return df
    
    def select_optimal_features(self, df):
        """Select features based on 20-session learning optimal distribution"""
        if 'total_runs' not in df.columns:
            print("❌ Target variable 'total_runs' not found")
            return df
        
        # Feature categories from learning (targeting 182 features like best sessions)
        feature_categories = {
            'core_baseball': [],     # Target: ~39 features (45.4% importance)
            'pitching': [],          # Target: ~32 features (27.2% importance)
            'environmental': [],     # Target: ~23 features (0.7% importance)
            'umpire': [],           # Target: ~12 features (0.8% importance)
            'market': [],           # Target: ~7 features (2.0% importance)
            'sophisticated': [],    # Target: ~15 features (2.0% importance)
            'other': []             # Target: ~49 features (21.9% importance)
        }
        
        # Categorize features based on learning patterns
        for col in df.columns:
            if col == 'total_runs':
                continue
                
            col_lower = col.lower()
            
            # Core baseball (highest priority)
            if any(x in col_lower for x in ['team_runs', 'team_rbi', 'team_hits', 'team_avg', 'team_obp']):
                feature_categories['core_baseball'].append(col)
            
            # Pitching (second priority)
            elif any(x in col_lower for x in ['sp_', 'bp_', '_era', '_whip', '_er', '_ip', '_k', '_bb']):
                feature_categories['pitching'].append(col)
            
            # Environmental
            elif any(x in col_lower for x in ['temp', 'wind', 'weather', 'ballpark', 'park_']):
                feature_categories['environmental'].append(col)
            
            # Umpire
            elif any(x in col_lower for x in ['umpire', 'plate_umpire']):
                feature_categories['umpire'].append(col)
            
            # Market
            elif any(x in col_lower for x in ['predicted', 'odds', 'spread', 'total']):
                feature_categories['market'].append(col)
            
            # Sophisticated
            elif any(x in col_lower for x in ['weighted', 'iso', 'momentum', 'efficiency', 'advantage']):
                feature_categories['sophisticated'].append(col)
            
            # Other
            else:
                feature_categories['other'].append(col)
        
        # Select optimal number of features per category
        selected_features = []
        
        # Core baseball: Take all (most important)
        selected_features.extend(feature_categories['core_baseball'])
        
        # Pitching: Take top features by correlation
        if feature_categories['pitching']:
            correlations = df[feature_categories['pitching'] + ['total_runs']].corr()['total_runs'].abs()
            top_pitching = correlations.sort_values(ascending=False).head(32).index.tolist()
            selected_features.extend([f for f in top_pitching if f != 'total_runs'])
        
        # Other categories: sample based on learned importance
        for category, max_features in [
            ('environmental', 23), ('umpire', 12), ('market', 7), 
            ('sophisticated', 15), ('other', 49)
        ]:
            features = feature_categories[category]
            if features:
                if len(features) <= max_features:
                    selected_features.extend(features)
                else:
                    # Take top features by variance
                    variances = df[features].var().sort_values(ascending=False)
                    selected_features.extend(variances.head(max_features).index.tolist())
        
        # Remove duplicates and ensure we have target
        final_features = list(set(selected_features))
        if 'total_runs' not in final_features:
            final_features.append('total_runs')
        
        print(f"📊 Selected {len(final_features)-1} optimal features for training")
        return df[final_features]
    
    def train_enhanced_model(self, df):
        """Train model using enhanced configuration from 20-session learning"""
        if len(df) < 100:
            print("❌ Insufficient data for training")
            return None, None
        
        # Prepare features
        X = df.drop(['total_runs'], axis=1, errors='ignore')
        y = df['total_runs']
        
        # Handle missing values (learned strategy)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Time series split (maintaining temporal order like learning sessions)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train enhanced model (Random Forest with learned parameters)
        model = RandomForestRegressor(
            n_estimators=200,      # More trees for stability
            max_depth=15,          # Balanced depth
            min_samples_split=5,   # Prevent overfitting
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Validate performance across time splits
        mae_scores = []
        r2_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            print(f"  Fold {fold+1}: MAE={mae:.3f}, R²={r2:.3f}")
        
        # Final model on all data
        model.fit(X, y)
        
        avg_mae = np.mean(mae_scores)
        avg_r2 = np.mean(r2_scores)
        
        print(f"📊 Enhanced Model Performance:")
        print(f"   Average MAE: {avg_mae:.3f} runs")
        print(f"   Average R²: {avg_r2:.3f}")
        print(f"   Target MAE: {self.best_session_config['target_mae']:.3f} runs")
        print(f"   Target R²: {self.best_session_config['target_r2']:.3f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Enhanced Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return model, {
            'mae_scores': mae_scores,
            'r2_scores': r2_scores,
            'avg_mae': avg_mae,
            'avg_r2': avg_r2,
            'feature_importance': feature_importance,
            'label_encoders': label_encoders
        }
    
    def evaluate_improvement(self, current_mae, current_r2):
        """Evaluate if the enhanced learning is improving predictions"""
        target_mae = self.best_session_config['target_mae']
        target_r2 = self.best_session_config['target_r2']
        
        mae_improvement = target_mae - current_mae
        r2_improvement = current_r2 - target_r2
        
        print(f"\n🎯 IMPROVEMENT ANALYSIS:")
        print(f"   Current MAE: {current_mae:.3f} vs Target: {target_mae:.3f}")
        print(f"   MAE Change: {mae_improvement:+.3f} runs")
        
        print(f"   Current R²: {current_r2:.3f} vs Target: {target_r2:.3f}")
        print(f"   R² Change: {r2_improvement:+.3f}")
        
        # Overall improvement score
        mae_pct = (mae_improvement / target_mae) * 100
        r2_pct = (r2_improvement / target_r2) * 100
        
        overall_improvement = (mae_pct + r2_pct) / 2
        
        print(f"\n📈 OVERALL IMPROVEMENT: {overall_improvement:+.1f}%")
        
        if overall_improvement > 0:
            print("✅ ENHANCED LEARNING IS IMPROVING PREDICTIONS!")
        elif overall_improvement > -5:
            print("⚠️  Performance is similar to baseline")
        else:
            print("❌ Performance below baseline - need refinement")
        
        return {
            'mae_improvement': mae_improvement,
            'r2_improvement': r2_improvement,
            'overall_improvement': overall_improvement,
            'is_improving': overall_improvement > 0
        }
    
    def save_enhanced_model(self, model, metadata, filename=None):
        """Save the enhanced model with metadata"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_model_{timestamp}.joblib"
        
        model_data = {
            'model': model,
            'metadata': metadata,
            'learning_config': self.best_session_config,
            'timestamp': datetime.now().isoformat(),
            'feature_weights': self.feature_importance_weights
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Enhanced model saved: {filename}")
        return filename

def main():
    """Demonstrate enhanced learning integration"""
    print("🎯 ENHANCED LEARNING INTEGRATION SYSTEM")
    print("="*60)
    
    # Initialize integrator
    integrator = EnhancedLearningIntegrator()
    
    # Load and process data
    df = integrator.load_enhanced_data()
    if df.empty:
        print("❌ No data loaded")
        return
    
    # Apply enhanced feature engineering
    df = integrator.engineer_enhanced_features(df)
    
    # Select optimal features
    df = integrator.select_optimal_features(df)
    
    # Train enhanced model
    model, metadata = integrator.train_enhanced_model(df)
    
    if model is None:
        print("❌ Training failed")
        return
    
    # Evaluate improvement
    improvement = integrator.evaluate_improvement(
        metadata['avg_mae'], 
        metadata['avg_r2']
    )
    
    # Save if improved
    if improvement['is_improving']:
        integrator.save_enhanced_model(model, metadata)
        print("\n🎉 Enhanced model saved for production use!")
    else:
        print("\n🔄 Model needs further refinement before production")

if __name__ == "__main__":
    main()
