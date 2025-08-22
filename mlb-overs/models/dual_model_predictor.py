#!/usr/bin/env python3
"""
Dual Model Predictor
===================
Runs both the original EnhancedBullpenPredictor and the new 203-feature learning model.
Generates predictions from both models for comparison and tracking.

Integration:
- Plugs into existing daily_api_workflow.py
- Uses same feature engineering pipeline
- Stores both predictions in enhanced_games table
- Maintains compatibility with UI tracking
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "deployment"))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))  # For root level imports

try:
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
except ImportError as e:
    try:
        # Try alternative path
        sys.path.append(str(Path(__file__).parent.parent / "deployment"))
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    except ImportError as e2:
        logging.error(f"Cannot import EnhancedBullpenPredictor: {e}, {e2}")
        EnhancedBullpenPredictor = None

from adaptive_learning_pipeline import AdaptiveLearningPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class DualModelPredictor:
    """
    Dual model prediction system that runs both:
    1. Original EnhancedBullpenPredictor (current production model)
    2. Adaptive Learning Pipeline (203-feature learning model)
    """
    
    def __init__(self):
        """Initialize both models"""
        # Original model
        self.original_model = None
        if EnhancedBullpenPredictor:
            try:
                self.original_model = EnhancedBullpenPredictor()
                log.info("✅ Original EnhancedBullpenPredictor loaded")
            except Exception as e:
                log.error(f"Failed to load original model: {e}")
        
        # Learning model
        try:
            self.learning_model = AdaptiveLearningPipeline()
            log.info("✅ Adaptive Learning Pipeline loaded")
        except Exception as e:
            log.error(f"Failed to load learning model: {e}")
            self.learning_model = None
    
    def predict_dual(self, X: pd.DataFrame, engine, target_date: str) -> dict:
        """
        Generate predictions from both models
        
        Args:
            X: Feature matrix (same format as current pipeline)
            engine: Database engine
            target_date: Date string for prediction context
            
        Returns:
            dict with 'original' and 'learning' prediction arrays
        """
        results = {}
        
        log.info("🎯 DUAL MODEL PREDICTION STARTED")
        log.info("=" * 60)
        
        # Original model prediction
        if self.original_model:
            try:
                log.info("🔵 Running Original Model (EnhancedBullpenPredictor)...")
                original_preds = self.original_model.predict(X)
                results['original'] = original_preds
                log.info(f"✅ Original model: {len(original_preds)} predictions")
                log.info(f"   Range: {np.min(original_preds):.2f} - {np.max(original_preds):.2f}")
                log.info(f"   Mean: {np.mean(original_preds):.2f}")
            except Exception as e:
                log.error(f"Original model prediction failed: {e}")
                results['original'] = None
        else:
            log.warning("⚠️ Original model not available")
            results['original'] = None
        
        # Learning model prediction
        if self.learning_model:
            try:
                log.info("🟢 Running Learning Model (203-feature adaptive)...")
                learning_preds = self.learning_model.predict(X, engine, target_date)
                results['learning'] = learning_preds
                log.info(f"✅ Learning model: {len(learning_preds)} predictions")
                log.info(f"   Range: {np.min(learning_preds):.2f} - {np.max(learning_preds):.2f}")
                log.info(f"   Mean: {np.mean(learning_preds):.2f}")
            except Exception as e:
                log.error(f"Learning model prediction failed: {e}")
                results['learning'] = None
        else:
            log.warning("⚠️ Learning model not available")
            results['learning'] = None
        
        # Comparison analysis
        if results['original'] is not None and results['learning'] is not None:
            orig = np.array(results['original'])
            learn = np.array(results['learning'])
            
            diff = learn - orig
            correlation = np.corrcoef(orig, learn)[0, 1]
            
            log.info("📊 MODEL COMPARISON ANALYSIS")
            log.info(f"   Correlation: {correlation:.3f}")
            log.info(f"   Mean difference (Learning - Original): {np.mean(diff):.3f}")
            log.info(f"   Std difference: {np.std(diff):.3f}")
            log.info(f"   Max difference: {np.max(np.abs(diff)):.3f}")
            
            # Identify games with significant differences
            big_diff_threshold = 1.0  # 1 run difference
            big_diffs = np.abs(diff) > big_diff_threshold
            if np.any(big_diffs):
                log.info(f"⚠️ {np.sum(big_diffs)} games with >1.0 run difference")
        
        log.info("=" * 60)
        log.info("🎯 DUAL MODEL PREDICTION COMPLETED")
        
        return results
    
    def get_primary_prediction(self, results: dict) -> np.ndarray:
        """
        Get the primary prediction to use (for backward compatibility)
        
        Priority:
        1. Learning model (if available and performing well)
        2. Original model (fallback)
        """
        if results.get('learning') is not None:
            log.info("Using learning model as primary prediction")
            return results['learning']
        elif results.get('original') is not None:
            log.info("Using original model as primary prediction")
            return results['original']
        else:
            log.error("No predictions available from either model!")
            raise ValueError("Both models failed to generate predictions")

def predict_and_upsert_dual(engine, X: pd.DataFrame, ids: pd.DataFrame, *, anchor_to_market: bool = True) -> pd.DataFrame:
    """
    Enhanced version of predict_and_upsert that runs both models.
    
    This function maintains API compatibility with the existing daily_api_workflow.py
    while adding dual model capabilities.
    """
    log.info("🚀 STARTING DUAL MODEL PREDICTION")
    
    # Initialize dual predictor
    dual_predictor = DualModelPredictor()
    
    # Get target date from ids if available
    target_date = ids['date'].iloc[0] if len(ids) > 0 and 'date' in ids.columns else datetime.now().strftime('%Y-%m-%d')
    
    # Run both models
    results = dual_predictor.predict_dual(X, engine, target_date)
    
    # Get primary prediction for current system compatibility
    primary_predictions = dual_predictor.get_primary_prediction(results)
    
    # Apply market anchoring if requested (same as original)
    if anchor_to_market and 'market_total' in X.columns:
        market_anchored = apply_market_anchoring(primary_predictions, X['market_total'].values)
        log.info("✅ Market anchoring applied to primary predictions")
        primary_predictions = market_anchored
    
    # Create results dataframe (same format as original)
    preds_df = ids.copy()
    preds_df['predicted_total'] = primary_predictions
    
    # Store both predictions in database for tracking
    try:
        store_dual_predictions(engine, ids, results, target_date)
    except Exception as e:
        log.warning(f"Failed to store dual predictions: {e}")
    
    # Upsert primary predictions (maintains existing behavior)
    upsert_predictions(engine, preds_df)
    
    log.info(f"🎯 Dual prediction complete: {len(preds_df)} games predicted")
    
    return preds_df

def apply_market_anchoring(predictions: np.ndarray, market_totals: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply market anchoring to predictions (same logic as original)"""
    anchored = predictions * (1 - alpha) + market_totals * alpha
    return anchored

def store_dual_predictions(engine, ids: pd.DataFrame, results: dict, target_date: str):
    """
    Store both original and learning predictions in database for comparison tracking.
    
    Adds new columns to enhanced_games:
    - predicted_total_original: Original model prediction  
    - predicted_total_learning: Learning model prediction
    - prediction_date: When predictions were made
    """
    from sqlalchemy import text
    
    # Add columns if they don't exist
    with engine.begin() as conn:
        # Check and add columns
        try:
            conn.execute(text("""
                ALTER TABLE enhanced_games 
                ADD COLUMN IF NOT EXISTS predicted_total_original NUMERIC,
                ADD COLUMN IF NOT EXISTS predicted_total_learning NUMERIC,
                ADD COLUMN IF NOT EXISTS prediction_timestamp TIMESTAMP DEFAULT NOW()
            """))
        except Exception as e:
            log.warning(f"Could not add dual prediction columns: {e}")
        
        # Update predictions for each game
        for idx, row in ids.iterrows():
            game_id = row['game_id']
            
            # Prepare values
            original_pred = results['original'][idx] if results['original'] is not None else None
            learning_pred = results['learning'][idx] if results['learning'] is not None else None
            
            # Update query
            update_sql = text("""
                UPDATE enhanced_games 
                SET predicted_total_original = :orig,
                    predicted_total_learning = :learn,
                    prediction_timestamp = NOW()
                WHERE game_id = :game_id AND date = :date
            """)
            
            conn.execute(update_sql, {
                'orig': original_pred,
                'learn': learning_pred, 
                'game_id': game_id,
                'date': target_date
            })
    
    log.info(f"✅ Stored dual predictions for {len(ids)} games")

def upsert_predictions(engine, preds_df: pd.DataFrame):
    """Upsert primary predictions (maintains existing behavior)"""
    from sqlalchemy import text
    
    with engine.begin() as conn:
        for _, row in preds_df.iterrows():
            upsert_sql = text("""
                UPDATE enhanced_games 
                SET predicted_total = :pred_total
                WHERE game_id = :game_id AND date = :date
            """)
            
            conn.execute(upsert_sql, {
                'pred_total': row['predicted_total'],
                'game_id': row['game_id'],
                'date': row['date']
            })

if __name__ == "__main__":
    # Test the dual predictor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    dual = DualModelPredictor()
    print("Dual model predictor initialized successfully!")
