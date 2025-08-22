#!/usr/bin/env python3
"""
Check Dual Predictions
======================
Shows where to find both original and learning model predictions
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os

def check_dual_predictions():
    """Check current dual predictions in database"""
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    # Check today's predictions
    query = text("""
    SELECT 
        game_id,
        home_team,
        away_team,
        market_total,
        predicted_total as current_prediction,
        predicted_total_original as original_model,
        predicted_total_learning as learning_model,
        prediction_timestamp,
        total_runs as actual_result
    FROM enhanced_games 
    WHERE date = '2025-08-22'
    AND (predicted_total_original IS NOT NULL OR predicted_total_learning IS NOT NULL)
    ORDER BY prediction_timestamp DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print('🎯 DUAL PREDICTIONS FOR TODAY (2025-08-22)')
    print('='*80)
    
    if df.empty:
        print("❌ No dual predictions found! Let's check what's in the database...")
        
        # Check if we have any enhanced_games data
        check_query = text("SELECT COUNT(*) as total, COUNT(predicted_total) as with_pred FROM enhanced_games WHERE date = '2025-08-22'")
        with engine.connect() as conn:
            check_df = pd.read_sql(check_query, conn)
        
        print(f"📊 Enhanced games for today: {check_df['total'].iloc[0]} total, {check_df['with_pred'].iloc[0]} with predictions")
        
        # Check column existence
        cols_query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        AND column_name IN ('predicted_total_original', 'predicted_total_learning')
        """)
        with engine.connect() as conn:
            cols_df = pd.read_sql(cols_query, conn)
        
        print(f"📋 Dual prediction columns: {cols_df['column_name'].tolist()}")
        return
    
    for _, row in df.iterrows():
        print(f"\n🏟️  {row['home_team']} vs {row['away_team']}")
        print(f"   Game ID: {row['game_id']}")
        
        if pd.notna(row['market_total']):
            print(f"   📊 Market Total: {row['market_total']:.1f}")
        else:
            print("   📊 Market Total: N/A")
            
        if pd.notna(row['original_model']):
            print(f"   🔵 Original Model: {row['original_model']:.2f}")
        else:
            print("   🔵 Original Model: N/A")
            
        if pd.notna(row['learning_model']):
            print(f"   🟢 Learning Model: {row['learning_model']:.2f}")
        else:
            print("   🟢 Learning Model: N/A")
            
        if pd.notna(row['current_prediction']):
            print(f"   📈 Current Used: {row['current_prediction']:.2f}")
        else:
            print("   📈 Current Used: N/A")
            
        # Calculate difference
        if pd.notna(row['original_model']) and pd.notna(row['learning_model']):
            diff = row['learning_model'] - row['original_model']
            if abs(diff) > 1.0:
                emoji = "🔥"
            elif abs(diff) > 0.5:
                emoji = "⚠️"
            else:
                emoji = "✅"
            print(f"   {emoji} Difference: {diff:+.2f} ({'Learning higher' if diff > 0 else 'Original higher'})")
        
        if pd.notna(row['actual_result']):
            print(f"   ✅ Actual Result: {row['actual_result']:.0f}")
            
            # Calculate accuracy if we have both predictions and actual
            if pd.notna(row['original_model']):
                orig_error = abs(row['original_model'] - row['actual_result'])
                print(f"   📊 Original Error: {orig_error:.2f}")
                
            if pd.notna(row['learning_model']):
                learn_error = abs(row['learning_model'] - row['actual_result'])
                print(f"   📊 Learning Error: {learn_error:.2f}")
        else:
            print("   ⏳ Game Pending")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Games: {len(df)}")
    print(f"   Original Predictions: {df['original_model'].notna().sum()}")
    print(f"   Learning Predictions: {df['learning_model'].notna().sum()}")
    print(f"   Both Models: {(df['original_model'].notna() & df['learning_model'].notna()).sum()}")
    
    if (df['original_model'].notna() & df['learning_model'].notna()).sum() > 0:
        both_df = df[(df['original_model'].notna() & df['learning_model'].notna())]
        avg_diff = (both_df['learning_model'] - both_df['original_model']).mean()
        print(f"   Average Difference: {avg_diff:+.2f} runs")

def check_feature_coverage():
    """Check how many features are being used"""
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    print('\n🔍 FEATURE COVERAGE ANALYSIS')
    print('='*50)
    
    # Get column count from enhanced_games
    query = text("""
    SELECT COUNT(*) as column_count
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games'
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        total_columns = result[0]
    
    print(f"📊 Total columns in enhanced_games: {total_columns}")
    
    # Check for key feature categories
    feature_categories = {
        'Pitcher Features': ['era', 'whip', 'pitcher', 'sp_', 'bullpen'],
        'Team Features': ['team_', 'home_', 'away_', 'runs', 'hits'],
        'Ballpark Features': ['ballpark', 'venue', 'park'],
        'Weather Features': ['temp', 'wind', 'weather', 'humidity'],
        'Umpire Features': ['umpire', 'official'],
        'Market Features': ['market', 'odds', 'line'],
        'Advanced Features': ['woba', 'wrc', 'iso', 'babip', 'fip']
    }
    
    for category, keywords in feature_categories.items():
        query = text(f"""
        SELECT COUNT(*) as count
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games'
        AND (
            {' OR '.join([f"LOWER(column_name) LIKE '%{keyword}%'" for keyword in keywords])}
        )
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            count = result[0]
        
        print(f"   {category}: {count} columns")
    
    print(f"\n💡 The learning model should be using close to all {total_columns} features")
    print("   This gives it more information than the original model for predictions")

if __name__ == "__main__":
    check_dual_predictions()
    check_feature_coverage()
