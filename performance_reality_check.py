#!/usr/bin/env python3
"""
Performance Reality Check Script
Calculates actual system performance to replace misleading claims
"""

import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import os

def check_backtest_performance():
    """Check performance from backtest files"""
    print("=== BACKTEST PERFORMANCE ANALYSIS ===")
    
    backtest_file = "exports/v13_backtest_results.csv"
    if os.path.exists(backtest_file):
        df = pd.read_csv(backtest_file)
        
        # Calculate overall accuracy
        correct = (df['side'] == df['result']).sum()
        total = len(df)
        accuracy = correct / total * 100
        
        print(f"📊 V13 Backtest Results:")
        print(f"   Total predictions: {total}")
        print(f"   Correct predictions: {correct}")
        print(f"   ACTUAL ACCURACY: {accuracy:.1f}%")
        
        # Breakdown by prediction type
        over_games = df[df['side'] == 'OVER']
        under_games = df[df['side'] == 'UNDER']
        
        if len(over_games) > 0:
            over_correct = (over_games['side'] == over_games['result']).sum()
            over_rate = over_correct / len(over_games) * 100
            print(f"   OVER accuracy: {over_rate:.1f}% ({over_correct}/{len(over_games)})")
        
        if len(under_games) > 0:
            under_correct = (under_games['side'] == under_games['result']).sum()
            under_rate = under_correct / len(under_games) * 100
            print(f"   UNDER accuracy: {under_rate:.1f}% ({under_correct}/{len(under_games)})")
        
        # Recent performance
        recent = df.tail(50) if len(df) >= 50 else df
        recent_correct = (recent['side'] == recent['result']).sum()
        recent_accuracy = recent_correct / len(recent) * 100
        print(f"   Recent {len(recent)} games: {recent_accuracy:.1f}% accuracy")
        
        return accuracy
    else:
        print(f"❌ Backtest file not found: {backtest_file}")
        return None

def check_database_performance():
    """Check recent performance from database"""
    print("\n=== DATABASE PERFORMANCE CHECK ===")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        
        # Check recent games with predictions and results
        query = """
        SELECT 
            date,
            COUNT(*) as total_games,
            COUNT(CASE WHEN actual_total IS NOT NULL THEN 1 END) as completed_games
        FROM enhanced_games 
        WHERE date >= %s 
        GROUP BY date 
        ORDER BY date DESC 
        LIMIT 10;
        """
        
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = pd.read_sql(query, conn, params=(thirty_days_ago,))
        
        print("📅 Recent Games Status:")
        print(df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")

def main():
    print("🔍 MLB PREDICTION SYSTEM - REALITY CHECK")
    print("=" * 50)
    
    # Check backtest performance
    accuracy = check_backtest_performance()
    
    # Check database status
    check_database_performance()
    
    print("\n=== REALITY CHECK SUMMARY ===")
    if accuracy:
        if accuracy > 55:
            print(f"✅ System performing above random: {accuracy:.1f}%")
        elif accuracy > 45:
            print(f"⚠️  System performing around random: {accuracy:.1f}%")
        else:
            print(f"❌ System performing below random: {accuracy:.1f}%")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Focus on feature engineering")
    print("   2. Investigate model overfitting") 
    print("   3. Review data quality")
    print("   4. Test different algorithms")
    print("   5. Consider ensemble methods")
    
    print("\n🎯 REALISTIC TARGETS:")
    print("   - Short term: 52-55% accuracy")
    print("   - Medium term: 55-60% accuracy") 
    print("   - Long term: 60%+ accuracy")
    print("   - Professional level: 55%+ sustained")

if __name__ == "__main__":
    main()
