#!/usr/bin/env python3
"""Compare retrained model predictions with market totals"""

import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def check_prediction_bias():
    print("🔍 Checking new prediction bias vs market totals...")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass',
            port=5432
        )
        
        # Get today's games with predictions and market totals
        query = """
        SELECT 
            home_team,
            away_team,
            market_total,
            predicted_total,
            (predicted_total - market_total) as difference,
            recommendation
        FROM enhanced_games 
        WHERE date = '2025-08-14'
        AND predicted_total IS NOT NULL
        AND market_total IS NOT NULL
        ORDER BY difference DESC
        """
        
        df = pd.read_sql(query, conn)
        
        if df.empty:
            print("❌ No games found with both predictions and market totals")
            return
        
        print(f"\n📊 Found {len(df)} games with predictions:")
        print(f"{'Game':<35} {'Market':<8} {'Predicted':<10} {'Diff':<8} {'Rec':<6}")
        print("-" * 75)
        
        total_diff = 0
        over_count = 0
        under_count = 0
        
        for _, row in df.iterrows():
            game = f"{row['away_team']} @ {row['home_team']}"
            if len(game) > 35:
                game = game[:32] + "..."
            
            diff = row['difference']
            total_diff += diff
            
            if row['recommendation'] == 'OVER':
                over_count += 1
            elif row['recommendation'] == 'UNDER':
                under_count += 1
            
            print(f"{game:<35} {row['market_total']:<8.1f} {row['predicted_total']:<10.1f} {diff:>+7.1f} {row['recommendation']:<6}")
        
        avg_diff = total_diff / len(df)
        
        print("-" * 75)
        print(f"📈 BIAS ANALYSIS:")
        print(f"   Average difference: {avg_diff:+.1f} runs")
        print(f"   Total recommendations: {over_count} OVER, {under_count} UNDER")
        
        if abs(avg_diff) < 0.5:
            print(f"   ✅ GOOD: Low bias ({avg_diff:+.1f})")
        elif avg_diff > 1.0:
            print(f"   ⚠️  HIGH OVER BIAS: {avg_diff:+.1f}")
        elif avg_diff < -1.0:
            print(f"   ⚠️  HIGH UNDER BIAS: {avg_diff:+.1f}")
        else:
            print(f"   ⚡ MODERATE BIAS: {avg_diff:+.1f}")
        
        # Compare to previous bias
        print(f"\n🆚 BEFORE RETRAINING: +1.8 runs bias (100% OVER)")
        print(f"🆕 AFTER RETRAINING:  {avg_diff:+.1f} runs bias ({over_count}/{len(df)} OVER)")
        
        if avg_diff < 1.8:
            print("🎉 BIAS SIGNIFICANTLY REDUCED!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_prediction_bias()
