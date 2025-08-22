#!/usr/bin/env python3

import pandas as pd
import psycopg2
import sys
sys.path.append('mlb-overs/deployment')
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

def compare_features():
    """Compare features we generate vs what model expects"""
    
    print("🔍 FEATURE COMPARISON ANALYSIS")
    print("="*60)
    
    # Load expected features
    with open('expected_model_features.txt', 'r') as f:
        expected_features = [line.strip() for line in f.readlines()]
    
    print(f"📋 Model expects: {len(expected_features)} features")
    
    # Initialize predictor and get a sample game
    predictor = EnhancedBullpenPredictor()
    
    # Get sample game data
    conn = psycopg2.connect(
        host="localhost",
        database="mlb_predictions",
        user="postgres",
        password="password123"
    )
    
    query = """
    SELECT * FROM enhanced_games 
    WHERE game_date >= CURRENT_DATE - INTERVAL '1 day'
    ORDER BY game_date DESC, game_time DESC
    LIMIT 1
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ No recent games found")
        return
    
    game = df.iloc[0]
    print(f"📊 Analyzing features for: {game['away_team']} @ {game['home_team']}")
    
    # Generate features
    features = predictor.engineer_features(game)
    actual_features = list(features.columns)
    
    print(f"🔧 We generate: {len(actual_features)} features")
    
    # Compare features
    expected_set = set(expected_features)
    actual_set = set(actual_features)
    
    missing_features = expected_set - actual_set
    extra_features = actual_set - expected_set
    matching_features = expected_set & actual_set
    
    print(f"\n✅ Matching features: {len(matching_features)}")
    print(f"❌ Missing features: {len(missing_features)}")
    print(f"⚠️  Extra features: {len(extra_features)}")
    
    if missing_features:
        print(f"\n🚨 MISSING FEATURES ({len(missing_features)}):")
        for i, feat in enumerate(sorted(missing_features)):
            print(f"   {i+1}: {feat}")
    
    if extra_features:
        print(f"\n⚠️  EXTRA FEATURES ({len(extra_features)}):")
        for i, feat in enumerate(sorted(extra_features)):
            print(f"   {i+1}: {feat}")
    
    # Show some feature values for matching features
    print(f"\n📈 SAMPLE FEATURE VALUES (first 10 matching):")
    for feat in sorted(matching_features)[:10]:
        value = features[feat].iloc[0]
        print(f"   {feat}: {value}")
    
    print(f"\n🎯 FEATURE ALIGNMENT SUMMARY:")
    print(f"   Model expects: {len(expected_features)} features")
    print(f"   We generate: {len(actual_features)} features") 
    print(f"   Match rate: {len(matching_features)/len(expected_features)*100:.1f}%")
    
    if len(missing_features) == 0:
        print("✅ All required features available!")
    else:
        print(f"❌ {len(missing_features)} critical features missing")

if __name__ == "__main__":
    compare_features()
