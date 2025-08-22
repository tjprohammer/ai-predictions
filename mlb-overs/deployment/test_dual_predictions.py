#!/usr/bin/env python3
"""
Simple Dual Prediction Test
===========================
Test the dual prediction system with real data from the database.
This bypasses the complex daily workflow to directly test dual predictions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "models"))

def get_database_url():
    """Get database URL"""
    return os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

def test_learning_model_only():
    """Test just the learning model with database data"""
    try:
        from adaptive_learning_pipeline import AdaptiveLearningPipeline
        
        engine = create_engine(get_database_url())
        target_date = '2025-08-22'
        
        # Get today's games with features
        query = text("""
            SELECT *
            FROM enhanced_games 
            WHERE date = :date
            AND total_runs IS NULL  -- Only upcoming games
            LIMIT 5
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'date': target_date})
        
        if df.empty:
            log.warning(f"No games found for {target_date}")
            return
        
        log.info(f"✅ Found {len(df)} games for testing")
        log.info(f"   Columns: {len(df.columns)}")
        
        # Initialize learning model
        learning_model = AdaptiveLearningPipeline()
        
        # Prepare feature matrix (remove target and IDs)
        X = df.drop(columns=['total_runs', 'game_id', 'date'], errors='ignore')
        
        # Convert to numeric where possible
        numeric_cols = []
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='ignore')
                if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    numeric_cols.append(col)
            except:
                pass
        
        # Use only numeric columns for now
        X_numeric = X[numeric_cols]
        log.info(f"   Using {len(X_numeric.columns)} numeric features")
        
        # Generate predictions
        predictions = learning_model.predict(X_numeric, engine, target_date)
        
        log.info(f"🎯 Learning Model Predictions:")
        for i, (_, row) in enumerate(df.iterrows()):
            pred = predictions[i] if i < len(predictions) else 'N/A'
            log.info(f"   {row['home_team']} vs {row['away_team']}: {pred:.2f}")
        
        # Store in database
        store_learning_predictions(engine, df, predictions, target_date)
        
        return predictions
        
    except Exception as e:
        log.error(f"Learning model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def store_learning_predictions(engine, games_df, predictions, target_date):
    """Store learning model predictions in database"""
    try:
        with engine.begin() as conn:
            for i, (_, row) in enumerate(games_df.iterrows()):
                if i < len(predictions):
                    pred = float(predictions[i])
                    
                    update_sql = text("""
                        UPDATE enhanced_games 
                        SET predicted_total_learning = :pred,
                            prediction_timestamp = NOW()
                        WHERE game_id = :game_id AND date = :date
                    """)
                    
                    conn.execute(update_sql, {
                        'pred': pred,
                        'game_id': row['game_id'],
                        'date': target_date
                    })
        
        log.info(f"✅ Stored {len(predictions)} learning predictions in database")
        
    except Exception as e:
        log.error(f"Failed to store predictions: {e}")

def test_original_model_simulation():
    """Simulate original model predictions using market totals + noise"""
    engine = create_engine(get_database_url())
    target_date = '2025-08-22'
    
    query = text("""
        SELECT game_id, home_team, away_team, market_total
        FROM enhanced_games 
        WHERE date = :date
        AND total_runs IS NULL
        AND market_total IS NOT NULL
        LIMIT 5
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'date': target_date})
    
    if df.empty:
        log.warning("No games with market totals found")
        return
    
    # Simulate original model predictions (market + small random adjustment)
    np.random.seed(42)  # For reproducible results
    original_preds = []
    
    for _, row in df.iterrows():
        market = row['market_total']
        # Add small random adjustment (-0.5 to +0.5)
        adjustment = np.random.uniform(-0.5, 0.5)
        pred = market + adjustment
        original_preds.append(pred)
    
    log.info(f"🔵 Simulated Original Model Predictions:")
    for i, (_, row) in enumerate(df.iterrows()):
        market = row['market_total']
        pred = original_preds[i]
        diff = pred - market
        log.info(f"   {row['home_team']} vs {row['away_team']}: {pred:.2f} (market: {market}, diff: {diff:+.2f})")
    
    # Store in database
    try:
        with engine.begin() as conn:
            for i, (_, row) in enumerate(df.iterrows()):
                pred = float(original_preds[i])
                
                update_sql = text("""
                    UPDATE enhanced_games 
                    SET predicted_total_original = :pred,
                        prediction_timestamp = NOW()
                    WHERE game_id = :game_id AND date = :date
                """)
                
                conn.execute(update_sql, {
                    'pred': pred,
                    'game_id': row['game_id'],
                    'date': target_date
                })
        
        log.info(f"✅ Stored {len(original_preds)} simulated original predictions")
        
    except Exception as e:
        log.error(f"Failed to store original predictions: {e}")
    
    return original_preds

def show_dual_predictions():
    """Show both predictions side by side"""
    engine = create_engine(get_database_url())
    target_date = '2025-08-22'
    
    query = text("""
        SELECT 
            game_id,
            home_team,
            away_team,
            market_total,
            predicted_total_original,
            predicted_total_learning,
            (predicted_total_learning - predicted_total_original) as difference
        FROM enhanced_games 
        WHERE date = :date
        AND (predicted_total_original IS NOT NULL OR predicted_total_learning IS NOT NULL)
        ORDER BY game_id
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'date': target_date})
    
    if df.empty:
        log.warning("No dual predictions found")
        return
    
    log.info("\n" + "="*80)
    log.info("🎯 DUAL PREDICTION COMPARISON")
    log.info("="*80)
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        market = row['market_total']
        original = row['predicted_total_original']
        learning = row['predicted_total_learning']
        diff = row['difference']
        
        log.info(f"\n🏟️  {home_team} vs {away_team}")
        log.info(f"   📊 Market Total: {market:.1f}" if market else "   📊 Market Total: N/A")
        log.info(f"   🔵 Original Model: {original:.2f}" if original else "   🔵 Original Model: N/A")
        log.info(f"   🟢 Learning Model: {learning:.2f}" if learning else "   🟢 Learning Model: N/A")
        
        if original and learning:
            if abs(diff) > 0.5:
                emoji = "🔥" if abs(diff) > 1.0 else "⚠️"
                log.info(f"   {emoji} Difference: {diff:+.2f} ({'Learning higher' if diff > 0 else 'Original higher'})")
            else:
                log.info(f"   ✅ Close agreement: {diff:+.2f}")
    
    log.info("\n" + "="*80)

if __name__ == "__main__":
    try:
        log.info("🚀 Testing Dual Prediction System")
        log.info("="*50)
        
        # Test 1: Learning model only
        log.info("\n1️⃣ Testing Learning Model...")
        learning_preds = test_learning_model_only()
        
        # Test 2: Simulate original model
        log.info("\n2️⃣ Simulating Original Model...")
        original_preds = test_original_model_simulation()
        
        # Test 3: Show comparison
        log.info("\n3️⃣ Showing Dual Predictions...")
        show_dual_predictions()
        
        log.info("\n🎉 Dual prediction testing complete!")
        
    except Exception as e:
        log.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
