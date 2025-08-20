#!/usr/bin/env python3
"""
Training-Production Feature Consistency Validator
================================================
Validates that the training script and daily production workflow use identical features.

This script:
1. Simulates the training feature engineering process
2. Simulates the production feature engineering process  
3. Compares feature sets, data quality, and processing differences
4. Identifies any mismatches that could cause training/serving skew

Usage:
    python validate_training_production_consistency.py --date 2025-08-20
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Set up paths
sys.path.append('..')  # Add parent directory for imports

# Database configuration
DB_URL = "postgresql://mlbuser:mlbpass@localhost:5432/mlb"

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def simulate_training_features(engine, target_date, window_days=60):
    """
    Simulate the exact feature engineering process used in training.
    """
    log = logging.getLogger(__name__)
    log.info("🔧 SIMULATING TRAINING FEATURE ENGINEERING")
    
    # 1. Fetch training data (same as train_model.py)
    end_date = target_date
    start_date = (pd.to_datetime(end_date) - timedelta(days=window_days)).strftime('%Y-%m-%d')
    
    query = f"""
    SELECT * FROM enhanced_games 
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    AND total_runs BETWEEN 5 AND 13
    AND total_runs IS NOT NULL
    ORDER BY date, game_id
    """
    
    df = pd.read_sql(query, engine)
    log.info(f"📊 Training data: {len(df)} games from {start_date} to {end_date}")
    
    if len(df) == 0:
        return None, None, None
    
    # 2. Use training feature engineering (from train_model.py)
    # Import here to avoid circular dependencies
    original_cwd = os.getcwd()
    try:
        # Add parent directory to Python path
        parent_dir = os.path.abspath('..')
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
        
        # Create database engine for enhanced pipeline
        predictor = EnhancedBullpenPredictor()
        
        # Set the current target date for enhanced pipeline
        if 'date' in df.columns:
            predictor._current_target_date = df['date'].iloc[0]
        
        # Store engine reference for enhanced pipeline to use
        if hasattr(predictor, 'enhanced_pipeline') and predictor.enhanced_pipeline:
            predictor.enhanced_pipeline.engine = engine
        
        # Use the same engineer_features method as training
        featured = predictor.engineer_features(df)
        
        # For training, we use raw features instead of aligning to existing model
        numeric_features = featured.select_dtypes(include=[np.number])
        
        # Remove ID columns and target-like columns
        id_columns = [col for col in numeric_features.columns if 'id' in col.lower() or col in ['game_id', 'total_runs']]
        X_training = numeric_features.drop(columns=id_columns, errors='ignore')
        
        log.info(f"✅ Training features: {featured.shape[1]} raw → {X_training.shape[1]} training features")
        
        return df, featured, X_training
        
    except Exception as e:
        log.error(f"Training simulation failed: {e}")
        return None, None, None
    finally:
        # Restore sys.path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

def simulate_production_features(engine, target_date):
    """
    Simulate the exact feature engineering process used in production.
    """
    log = logging.getLogger(__name__)
    log.info("🔧 SIMULATING PRODUCTION FEATURE ENGINEERING")
    
    # Import here to avoid circular dependencies
    original_cwd = os.getcwd()
    try:
        # Add parent directory to Python path
        parent_dir = os.path.abspath('..')
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from daily_api_workflow import engineer_and_align, load_today_games
        
        # 1. Load today's games (same as production)
        df = load_today_games(engine, target_date)
        
        if len(df) == 0:
            log.warning(f"No games found for {target_date}")
            return None, None, None
        
        log.info(f"📊 Production data: {len(df)} games for {target_date}")
        
        # 2. Use production feature engineering
        featured, X_aligned, predictions = engineer_and_align(df, target_date)
        
        log.info(f"✅ Production features: {featured.shape[1]} raw → {X_aligned.shape[1]} aligned features")
        
        return df, featured, X_aligned
        
    except Exception as e:
        log.error(f"Production simulation failed: {e}")
        return None, None, None
    finally:
        # Restore sys.path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

def compare_feature_sets(training_featured, training_X, production_featured, production_X):
    """
    Compare training and production feature sets.
    """
    log = logging.getLogger(__name__)
    log.info("🔍 COMPARING TRAINING VS PRODUCTION FEATURES")
    
    print("\n" + "="*80)
    print("FEATURE SET COMPARISON")
    print("="*80)
    
    # 1. Raw feature comparison
    training_raw_cols = set(training_featured.columns)
    production_raw_cols = set(production_featured.columns)
    
    print(f"\n📊 RAW FEATURES:")
    print(f"  Training:   {len(training_raw_cols)} features")
    print(f"  Production: {len(production_raw_cols)} features")
    
    common_raw = training_raw_cols & production_raw_cols
    training_only = training_raw_cols - production_raw_cols
    production_only = production_raw_cols - training_raw_cols
    
    print(f"  Common:     {len(common_raw)} features")
    print(f"  Training only: {len(training_only)} features")
    print(f"  Production only: {len(production_only)} features")
    
    if training_only:
        print(f"  Training-only features: {list(training_only)[:10]}{'...' if len(training_only) > 10 else ''}")
    if production_only:
        print(f"  Production-only features: {list(production_only)[:10]}{'...' if len(production_only) > 10 else ''}")
    
    # 2. Final feature comparison
    training_final_cols = set(training_X.columns)
    production_final_cols = set(production_X.columns)
    
    print(f"\\n🎯 FINAL FEATURES:")
    print(f"  Training:   {len(training_final_cols)} features")
    print(f"  Production: {len(production_final_cols)} features")
    
    common_final = training_final_cols & production_final_cols
    training_final_only = training_final_cols - production_final_cols
    production_final_only = production_final_cols - training_final_cols
    
    print(f"  Common:     {len(common_final)} features")
    print(f"  Training only: {len(training_final_only)} features")
    print(f"  Production only: {len(production_final_only)} features")
    
    if training_final_only:
        print(f"  Training-only features: {list(training_final_only)[:10]}{'...' if len(training_final_only) > 10 else ''}")
    if production_final_only:
        print(f"  Production-only features: {list(production_final_only)[:10]}{'...' if len(production_final_only) > 10 else ''}")
    
    # 3. Feature overlap analysis
    overlap_score = len(common_final) / max(len(training_final_cols), len(production_final_cols))
    print(f"\\n📈 FEATURE OVERLAP SCORE: {overlap_score:.3f} ({overlap_score*100:.1f}%)")
    
    if overlap_score < 0.9:
        print("⚠️  WARNING: Low feature overlap - potential training/serving skew!")
    elif overlap_score < 0.95:
        print("⚠️  CAUTION: Moderate feature mismatch detected")
    else:
        print("✅ GOOD: High feature overlap")
    
    return {
        'raw_overlap': len(common_raw) / max(len(training_raw_cols), len(production_raw_cols)),
        'final_overlap': overlap_score,
        'training_raw_count': len(training_raw_cols),
        'production_raw_count': len(production_raw_cols),
        'training_final_count': len(training_final_cols),
        'production_final_count': len(production_final_cols),
        'common_final': common_final,
        'training_only_final': training_final_only,
        'production_only_final': production_final_only
    }

def compare_data_quality(training_featured, production_featured):
    """
    Compare data quality between training and production.
    """
    log = logging.getLogger(__name__)
    log.info("🔍 COMPARING DATA QUALITY")
    
    print(f"\\n" + "="*80)
    print("DATA QUALITY COMPARISON")
    print("="*80)
    
    # Find common columns for comparison
    common_cols = set(training_featured.columns) & set(production_featured.columns)
    
    print(f"\\n📊 Analyzing {len(common_cols)} common features...")
    
    quality_issues = []
    
    for col in sorted(common_cols):
        # Skip ID columns
        if 'id' in col.lower() or col in ['game_id']:
            continue
        
        train_vals = pd.to_numeric(training_featured[col], errors='coerce')
        prod_vals = pd.to_numeric(production_featured[col], errors='coerce')
        
        # Check null rates
        train_null_rate = train_vals.isna().mean()
        prod_null_rate = prod_vals.isna().mean()
        
        # Check variance
        train_std = train_vals.std(skipna=True)
        prod_std = prod_vals.std(skipna=True)
        
        # Check for significant differences
        if abs(train_null_rate - prod_null_rate) > 0.1:
            quality_issues.append(f"{col}: null rate diff {train_null_rate:.3f} vs {prod_null_rate:.3f}")
        
        if train_std > 0 and prod_std > 0:
            std_ratio = max(train_std, prod_std) / min(train_std, prod_std)
            if std_ratio > 2.0:
                quality_issues.append(f"{col}: std diff {train_std:.3f} vs {prod_std:.3f}")
        elif train_std == 0 and prod_std > 0:
            quality_issues.append(f"{col}: constant in training but varies in production")
        elif prod_std == 0 and train_std > 0:
            quality_issues.append(f"{col}: varies in training but constant in production")
    
    print(f"\\n🔍 Quality Issues Found: {len(quality_issues)}")
    for issue in quality_issues[:20]:  # Show first 20
        print(f"  ⚠️  {issue}")
    if len(quality_issues) > 20:
        print(f"  ... and {len(quality_issues) - 20} more issues")
    
    if len(quality_issues) == 0:
        print("✅ No significant data quality issues detected")
    
    return quality_issues

def main():
    parser = argparse.ArgumentParser(description="Validate training-production feature consistency")
    parser.add_argument("--date", default="2025-08-20", help="Date to analyze")
    parser.add_argument("--window-days", type=int, default=60, help="Training window in days")
    
    args = parser.parse_args()
    
    log = setup_logging()
    
    print("🔧 TRAINING-PRODUCTION CONSISTENCY VALIDATOR")
    print("=" * 80)
    print(f"Target date: {args.date}")
    print(f"Training window: {args.window_days} days")
    print()
    
    # Connect to database
    engine = create_engine(DB_URL)
    
    try:
        # 1. Simulate training features
        train_df, train_featured, train_X = simulate_training_features(engine, args.date, args.window_days)
        
        if train_df is None:
            log.error("Failed to simulate training features")
            return 1
        
        # 2. Simulate production features
        prod_df, prod_featured, prod_X = simulate_production_features(engine, args.date)
        
        if prod_df is None:
            log.error("Failed to simulate production features")
            return 1
        
        # 3. Compare feature sets
        feature_comparison = compare_feature_sets(train_featured, train_X, prod_featured, prod_X)
        
        # 4. Compare data quality
        quality_issues = compare_data_quality(train_featured, prod_featured)
        
        # 5. Summary
        print(f"\\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✅ Training simulation: {len(train_df)} games → {train_X.shape[1]} features")
        print(f"✅ Production simulation: {len(prod_df)} games → {prod_X.shape[1]} features")
        print(f"📊 Feature overlap: {feature_comparison['final_overlap']:.3f} ({feature_comparison['final_overlap']*100:.1f}%)")
        print(f"⚠️  Data quality issues: {len(quality_issues)}")
        
        if feature_comparison['final_overlap'] >= 0.95 and len(quality_issues) < 10:
            print("\\n🎉 RESULT: Training and production pipelines are CONSISTENT!")
            return 0
        elif feature_comparison['final_overlap'] >= 0.9:
            print("\\n⚠️  RESULT: Minor inconsistencies detected - review recommended")
            return 0
        else:
            print("\\n❌ RESULT: Significant inconsistencies detected - action required!")
            return 1
            
    except Exception as e:
        log.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
