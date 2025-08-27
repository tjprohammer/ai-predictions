#!/usr/bin/env python3
"""
Feature Alignment Diagnostic and Fix
====================================
Diagnose and fix the feature mismatch between current pipeline and models.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "models"))

def analyze_feature_mismatch():
    """Analyze what features each model expects vs what pipeline creates"""
    
    # Load the legitimate model features
    model_path = Path(__file__).parent.parent / "models" / "legitimate_model_latest.joblib"
    model_data = joblib.load(model_path)
    legitimate_features = set(model_data['feature_columns'])
    
    # Load adaptive learning model features  
    adaptive_path = Path(__file__).parent.parent / "models" / "adaptive_learning_model.joblib"
    adaptive_data = joblib.load(adaptive_path)
    adaptive_features = set(adaptive_data['feature_columns'])
    
    print("🔍 FEATURE MISMATCH ANALYSIS")
    print("=" * 60)
    print(f"Legitimate model expects: {len(legitimate_features)} features")
    print(f"Adaptive model expects: {len(adaptive_features)} features")
    
    # Load a sample of current features from enhanced_games
    from sqlalchemy import create_engine, text
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    # Get current feature columns from enhanced_games table
    query = text("""
        SELECT * FROM enhanced_games 
        WHERE date = '2025-08-23' 
        LIMIT 1
    """)
    sample_df = pd.read_sql(query, engine)
    current_features = set(sample_df.columns)
    
    print(f"Current enhanced_games columns: {len(current_features)}")
    
    # Analysis
    print("\n📊 FEATURE OVERLAP ANALYSIS")
    print("-" * 40)
    
    # Legitimate model overlap
    leg_missing = legitimate_features - current_features
    leg_extra = current_features - legitimate_features
    leg_overlap = legitimate_features & current_features
    
    print(f"\nLegitimate Model Analysis:")
    print(f"  Features in common: {len(leg_overlap)}/{len(legitimate_features)} ({100*len(leg_overlap)/len(legitimate_features):.1f}%)")
    print(f"  Missing from enhanced_games: {len(leg_missing)}")
    print(f"  Extra in enhanced_games: {len(leg_extra)}")
    
    if leg_missing:
        print(f"\n❌ MISSING FROM ENHANCED_GAMES (first 20):")
        for i, feat in enumerate(sorted(leg_missing)[:20]):
            print(f"    {i+1:2d}. {feat}")
        if len(leg_missing) > 20:
            print(f"    ... and {len(leg_missing)-20} more")
    
    # Adaptive model overlap
    adapt_missing = adaptive_features - current_features
    adapt_extra = current_features - adaptive_features  
    adapt_overlap = adaptive_features & current_features
    
    print(f"\nAdaptive Model Analysis:")
    print(f"  Features in common: {len(adapt_overlap)}/{len(adaptive_features)} ({100*len(adapt_overlap)/len(adaptive_features):.1f}%)")
    print(f"  Missing from enhanced_games: {len(adapt_missing)}")
    print(f"  Extra in enhanced_games: {len(adapt_extra)}")
    
    if adapt_missing:
        print(f"\n❌ MISSING FROM ENHANCED_GAMES (adaptive):")
        for i, feat in enumerate(sorted(adapt_missing)):
            print(f"    {i+1:2d}. {feat}")
    
    return {
        'legitimate_features': legitimate_features,
        'adaptive_features': adaptive_features,
        'current_features': current_features,
        'legitimate_missing': leg_missing,
        'adaptive_missing': adapt_missing
    }

def test_feature_engineering():
    """Test what features the current pipeline actually creates"""
    print("\n🧪 TESTING CURRENT FEATURE ENGINEERING")
    print("=" * 60)
    
    # Import the predictor and test feature engineering
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    
    predictor = EnhancedBullpenPredictor()
    
    # Create sample data similar to what comes from enhanced_games
    sample_data = pd.DataFrame({
        'game_id': ['2025082301'],
        'date': ['2025-08-23'],
        'home_team': ['New York Yankees'],
        'away_team': ['Boston Red Sox'],
        'home_sp_id': [641482],
        'away_sp_id': [607192],
        'market_total': [8.5],
        'temperature': [75],
        'wind_speed': [10],
        'venue_name': ['Yankee Stadium'],
        'home_sp_era': [3.45],
        'away_sp_era': [4.21],
        'home_sp_whip': [1.12],
        'away_sp_whip': [1.28],
        'home_sp_k_per_9': [9.2],
        'away_sp_k_per_9': [8.8],
        'home_sp_bb_per_9': [2.8],
        'away_sp_bb_per_9': [3.1],
        'home_sp_starts': [25],
        'away_sp_starts': [23]
    })
    
    try:
        # Test feature engineering
        featured_df = predictor.engineer_features(sample_data)
        engineered_features = set(featured_df.columns)
        
        print(f"✅ Feature engineering successful!")
        print(f"   Input features: {len(sample_data.columns)}")
        print(f"   Output features: {len(engineered_features)}")
        
        # Check against model expectations
        model_features = set(predictor.feature_columns)
        
        overlap = engineered_features & model_features
        missing = model_features - engineered_features
        extra = engineered_features - model_features
        
        print(f"\n📊 ENGINEERED FEATURES vs MODEL EXPECTATIONS:")
        print(f"   Overlap: {len(overlap)}/{len(model_features)} ({100*len(overlap)/len(model_features):.1f}%)")
        print(f"   Missing: {len(missing)}")
        print(f"   Extra: {len(extra)}")
        
        if missing:
            print(f"\n❌ FEATURES MISSING FROM ENGINEERING (first 15):")
            for i, feat in enumerate(sorted(missing)[:15]):
                print(f"    {i+1:2d}. {feat}")
            if len(missing) > 15:
                print(f"    ... and {len(missing)-15} more")
        
        # Test the problematic feature mentioned in error
        if 'combined_k_rate' in missing:
            print(f"\n🔍 PROBLEMATIC FEATURE ANALYSIS:")
            print(f"   'combined_k_rate' is MISSING from feature engineering")
            print(f"   But model expects it (it's in the error message)")
            
            # Check if we have the inputs to create it
            k9_inputs = ['home_sp_k_per_9', 'away_sp_k_per_9']
            k9_available = [col for col in k9_inputs if col in engineered_features]
            print(f"   K/9 inputs available: {k9_available}")
        
        return {
            'success': True,
            'engineered_features': engineered_features,
            'missing_from_engineering': missing,
            'sample_data': featured_df
        }
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def create_feature_alignment_fix():
    """Create a fix for the feature alignment issue"""
    print("\n🔧 CREATING FEATURE ALIGNMENT FIX")
    print("=" * 60)
    
    # Get the analysis results
    analysis = analyze_feature_mismatch()
    engineering_test = test_feature_engineering()
    
    if not engineering_test['success']:
        print("❌ Cannot create fix - feature engineering test failed")
        return
    
    # Key missing features that need to be added to feature engineering
    legitimate_missing = analysis['legitimate_missing']
    critical_missing = []
    
    # Identify critical missing features
    for feat in legitimate_missing:
        if any(keyword in feat for keyword in ['combined_k_rate', 'expected_total', 'offense_imbalance']):
            critical_missing.append(feat)
    
    print(f"🎯 CRITICAL MISSING FEATURES TO FIX:")
    for feat in critical_missing:
        print(f"   - {feat}")
    
    # Create fix recommendations
    fixes = []
    
    if 'combined_k_rate' in critical_missing:
        fixes.append({
            'feature': 'combined_k_rate',
            'fix': "featured_df['combined_k_rate'] = (featured_df['home_sp_k_per_9'] + featured_df['away_sp_k_per_9']) / 2",
            'location': '_original_engineer_features method'
        })
    
    if 'expected_total' in critical_missing:
        fixes.append({
            'feature': 'expected_total', 
            'fix': "featured_df['expected_total'] = featured_df['market_total']",
            'location': '_original_engineer_features method'
        })
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    for i, fix in enumerate(fixes, 1):
        print(f"{i}. {fix['feature']}:")
        print(f"   Add to {fix['location']}:")
        print(f"   {fix['fix']}")
    
    return fixes

if __name__ == "__main__":
    analysis = analyze_feature_mismatch()
    engineering_test = test_feature_engineering()
    fixes = create_feature_alignment_fix()
    
    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"   Run this script to see detailed feature mismatch analysis")
