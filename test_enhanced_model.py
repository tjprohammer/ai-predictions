#!/usr/bin/env python3
"""
Quick test to retrain a model with enhanced features and measure improvement
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from mlb.features.enhanced_feature_engine import EnhancedFeatureEngine
import numpy as np

def main():
    print("🚀 Testing Enhanced Features Model Performance")
    print("=" * 60)
    
    # Connect to database
    engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    
    with engine.connect() as conn:
        # Get recent games with complete data for training
        query = text('''
            SELECT * FROM enhanced_games 
            WHERE date >= '2025-08-01' 
            AND predicted_total IS NOT NULL 
            AND total_runs IS NOT NULL
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            LIMIT 200
        ''')
        df = pd.read_sql(query, conn)
        
        print(f"📊 Loaded {len(df)} games for testing")
        
        # Add enhanced features
        engine_obj = EnhancedFeatureEngine()
        enhanced_df = engine_obj.process_enhanced_features(df.copy())
        
        print(f"✨ Enhanced from {len(df.columns)} to {len(enhanced_df.columns)} features")
        
        # Prepare features for modeling
        target = 'total_runs'
        
        # Select numeric features only
        feature_cols = []
        for col in enhanced_df.columns:
            if col != target and enhanced_df[col].dtype in ['int64', 'float64']:
                # Skip columns that are perfect predictors (home_score, away_score)
                if col not in ['home_score', 'away_score', 'home_team_runs', 'away_team_runs', 'id']:
                    feature_cols.append(col)
        
        # Remove any features with too many missing values
        valid_features = []
        for col in feature_cols:
            missing_pct = enhanced_df[col].isnull().mean()
            if missing_pct < 0.5:  # Less than 50% missing
                valid_features.append(col)
        
        print(f"🎯 Using {len(valid_features)} features for modeling")
        
        # Prepare data
        X = enhanced_df[valid_features].fillna(0)
        y = enhanced_df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"📈 Training set: {len(X_train)} games")
        print(f"📉 Test set: {len(X_test)} games")
        
        # Train baseline model (original features only)
        original_features = [col for col in valid_features if col in df.columns]
        print(f"🔄 Baseline model with {len(original_features)} original features")
        
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        baseline_model.fit(X_train[original_features], y_train)
        baseline_pred = baseline_model.predict(X_test[original_features])
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_r2 = r2_score(y_test, baseline_pred)
        
        # Train enhanced model (all features)
        print(f"🚀 Enhanced model with {len(valid_features)} features")
        enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
        enhanced_model.fit(X_train, y_train)
        enhanced_pred = enhanced_model.predict(X_test)
        enhanced_mae = mean_absolute_error(y_test, enhanced_pred)
        enhanced_r2 = r2_score(y_test, enhanced_pred)
        
        # Show results
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"📉 Baseline Model MAE:  {baseline_mae:.3f} runs")
        print(f"📈 Enhanced Model MAE:  {enhanced_mae:.3f} runs")
        print(f"🎯 Improvement:        {((baseline_mae - enhanced_mae) / baseline_mae * 100):+.1f}%")
        print()
        print(f"📉 Baseline Model R²:   {baseline_r2:.3f}")
        print(f"📈 Enhanced Model R²:   {enhanced_r2:.3f}")
        print(f"🎯 R² Improvement:     {((enhanced_r2 - baseline_r2) / abs(baseline_r2) * 100):+.1f}%")
        
        # Show feature importance for enhanced features
        feature_importance = enhanced_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🔥 TOP 15 FEATURE IMPORTANCE (Enhanced Model):")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            enhanced_marker = "🆕" if row['feature'] not in df.columns else "📊"
            print(f"{i+1:2d}. {enhanced_marker} {row['feature']}: {row['importance']:.4f}")
        
        # Count enhanced features in top performers
        top_features = importance_df.head(20)
        enhanced_in_top = sum(1 for f in top_features['feature'] if f not in df.columns)
        print(f"\n🚀 Enhanced features in top 20: {enhanced_in_top}/20 ({enhanced_in_top/20*100:.1f}%)")

if __name__ == "__main__":
    main()
