#!/usr/bin/env python3
"""
Quick check of tomorrow's predictions with Ultra-80 prioritization
"""
import psycopg2
import pandas as pd

def check_tomorrow_predictions():
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    
    query = """
    SELECT 
        game_id,
        home_team,
        away_team,
        market_total,
        predicted_total_learning as ultra80_pred,
        predicted_total as learning_pred,
        COALESCE(predicted_total_learning, predicted_total) as published_pred,
        ROUND((COALESCE(predicted_total_learning, predicted_total) - market_total)::numeric, 2) as edge,
        recommendation,
        confidence
    FROM enhanced_games 
    WHERE date = '2025-08-29'
    AND market_total IS NOT NULL
    ORDER BY ABS(COALESCE(predicted_total_learning, predicted_total) - market_total) DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print("🎯 TOMORROW'S PREDICTIONS (2025-08-29) - ULTRA-80 PRIORITIZED")
    print("=" * 80)
    print(f"Found {len(df)} games with predictions")
    print()
    
    actionable_count = 0
    for i, row in df.iterrows():
        ultra_pred = row['ultra80_pred'] if pd.notna(row['ultra80_pred']) else None
        learning_pred = row['learning_pred'] if pd.notna(row['learning_pred']) else None
        published = row['published_pred']
        market = row['market_total']
        edge = row['edge']
        rec = row['recommendation']
        conf = row['confidence'] if pd.notna(row['confidence']) else 0
        
        # Count actionable picks
        if rec in ['OVER', 'UNDER']:
            actionable_count += 1
        
        print(f"{i+1}. {row['away_team']} @ {row['home_team']}")
        print(f"   Market: {market}")
        if ultra_pred:
            print(f"   Ultra-80: {ultra_pred:.2f}")
        if learning_pred:
            print(f"   Learning: {learning_pred:.2f}")
        print(f"   Published: {published:.2f} | Edge: {edge:+.2f}")
        print(f"   Recommendation: {rec} | Confidence: {conf:.0f}%")
        print()
    
    print(f"📊 SUMMARY: {actionable_count} actionable picks out of {len(df)} games")
    
    # Show top actionable picks
    actionable = df[df['recommendation'].isin(['OVER', 'UNDER'])].copy()
    if len(actionable) > 0:
        print(f"\n🎯 TOP ACTIONABLE PICKS:")
        for i, row in actionable.head(5).iterrows():
            edge = row['edge']
            conf = row['confidence'] if pd.notna(row['confidence']) else 0
            print(f"  {row['away_team']} @ {row['home_team']}: {row['recommendation']} {row['market_total']} (Edge: {edge:+.2f}, Conf: {conf:.0f}%)")

if __name__ == "__main__":
    check_tomorrow_predictions()
