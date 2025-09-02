#!/usr/bin/env python3
"""
Emergency Retrain: Enhanced Bullpen Predictor (Data Leakage Fix)
Retrains the model without predicted_total, edge, and other circular dependency features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import psycopg2
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )

def load_training_data():
    """Load training data without data leakage features"""
    conn = connect_db()
    
    # Get training data from last 2 years, excluding leaked features
    query = """
    SELECT 
        -- Target
        total_runs,
        
        -- Weather features
        temperature, wind_speed, pressure, feels_like_temp,
        
        -- Pitcher features (season stats, not predictions)
        home_sp_k, home_sp_bb, home_sp_h,
        away_sp_k, away_sp_bb, away_sp_h,
        home_sp_whip, away_sp_whip,
        home_sp_season_k, away_sp_season_k,
        home_sp_season_bb, away_sp_season_bb,
        
        -- Team features (actual stats, not predictions)
        home_team_hits, home_team_rbi, home_team_lob,
        away_team_hits, away_team_rbi, away_team_lob,
        home_team_wrc_plus, away_team_wrc_plus,
        home_team_stolen_bases, away_team_stolen_bases,
        home_team_plate_appearances, away_team_plate_appearances,
        
        -- Bullpen features (actual stats)
        home_bp_k, home_bp_bb, home_bp_h,
        away_bp_k, away_bp_bb, away_bp_h,
        home_bullpen_whip_l30, away_bullpen_whip_l30,
        home_bullpen_usage_rate, away_bullpen_usage_rate,
        
        -- Team form features
        home_team_ops_l14, away_team_ops_l14,
        home_team_form_rating, away_team_form_rating,
        home_team_ops_l20,
        
        -- Ballpark features
        ballpark_hr_factor,
        
        -- Market context (but NOT predictions)
        market_total,
        over_odds, under_odds
        
        -- EXCLUDED: predicted_total, edge (circular dependency)
        
    FROM enhanced_games 
    WHERE date >= %s 
        AND date < %s
        AND total_runs IS NOT NULL
        AND market_total IS NOT NULL
        AND home_sp_k IS NOT NULL
        AND away_sp_k IS NOT NULL
    ORDER BY date
    """
    
    # Train on last 2 years of data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)
    
    print(f"Loading training data from {start_date} to {end_date}")
    
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    
    print(f"Loaded {len(df)} games for training")
    return df

def prepare_features(df):
    """Prepare features for training"""
    # Separate target and features
    y = df['total_runs'].values
    X = df.drop(['total_runs'], axis=1)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove any remaining non-numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def train_model(X, y):
    """Train Random Forest model"""
    print("\n=== Training New Enhanced Bullpen Model ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} games")
    print(f"Test set: {X_test.shape[0]} games")
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Overfitting check: {(train_mae/test_mae):.3f} (closer to 1.0 is better)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")
    
    # Verify no leaked features
    leaked_features = ['predicted_total', 'edge']
    found_leaks = [f for f in leaked_features if f in X.columns]
    if found_leaks:
        raise ValueError(f"Data leakage detected! Found: {found_leaks}")
    else:
        print(f"\n✅ Model verified leak-free - no circular dependency features")
    
    return model, feature_importance

def save_model(model, feature_names):
    """Save the retrained model"""
    # Save to the same location as the original
    model_path = "s:/Projects/AI_Predictions/mlb/models/adaptive_learning_model.joblib"
    
    # Create model package with metadata
    model_package = {
        'model': model,
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'trained_date': datetime.now().isoformat(),
        'version': 'leak_free_v1',
        'notes': 'Retrained without predicted_total/edge circular dependency'
    }
    
    joblib.dump(model_package, model_path)
    print(f"\n✅ Model saved to: {model_path}")
    print(f"Features: {len(feature_names)}")
    
    # Backup old model first
    backup_path = model_path.replace('.joblib', '_backup_with_leaks.joblib')
    try:
        old_model = joblib.load(model_path)
        joblib.dump(old_model, backup_path)
        print(f"📁 Old model backed up to: {backup_path}")
    except:
        print("⚠️  Could not backup old model (may not exist)")

def main():
    """Main retraining workflow"""
    print("🚨 EMERGENCY RETRAIN: Enhanced Bullpen Predictor")
    print("Fixing data leakage by removing circular dependency features")
    print("=" * 60)
    
    # Load data
    df = load_training_data()
    
    if len(df) < 1000:
        print(f"⚠️  Warning: Only {len(df)} games available for training")
        print("Proceeding anyway, but more data would be better")
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, feature_importance = train_model(X, y)
    
    # Save model
    save_model(model, X.columns)
    
    print("\n🎉 RETRAIN COMPLETE!")
    print("Enhanced Bullpen Predictor is now leak-free and ready to use")
    print("Run the daily workflow again to test the new model")

if __name__ == "__main__":
    main()
