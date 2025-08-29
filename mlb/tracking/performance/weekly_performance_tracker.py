#!/usr/bin/env python3
"""
Weekly Performance Tracker
==========================

Track betting performance over the past week to identify trends and patterns.
"""

import psycopg2
from datetime import datetime, timedelta

def connect_db():
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def weekly_analysis():
    """Analyze betting performance over the past 7 days"""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get past 7 days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6)
    
    print(f"📊 WEEKLY PERFORMANCE ANALYSIS")
    print(f"📅 {start_date} to {end_date}")
    print("=" * 60)
    
    # Daily breakdown
    for i in range(7):
        current_date = start_date + timedelta(days=i)
        
        cursor.execute('''
        SELECT 
            COUNT(*) as total_picks,
            COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed,
            COUNT(CASE WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1 END) +
            COUNT(CASE WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1 END) as wins,
            AVG(confidence) as avg_confidence
        FROM enhanced_games 
        WHERE date = %s 
        AND recommendation IN ('OVER', 'UNDER')
        AND total_runs IS NOT NULL
        ''', (current_date,))
        
        result = cursor.fetchone()
        total_picks, completed, wins, avg_conf = result
        
        if completed > 0:
            win_rate = (wins / completed * 100)
            profit = (wins * 90.91) - ((completed - wins) * 100)
            profit_display = f"+${profit:.0f}" if profit > 0 else f"-${abs(profit):.0f}"
            
            status = "🔥" if win_rate >= 70 else "🎉" if win_rate >= 60 else "👍" if win_rate >= 50 else "📉"
            
            print(f"{current_date.strftime('%a %m/%d')}: {wins}/{completed} ({win_rate:.0f}%) {profit_display} {status}")
        else:
            print(f"{current_date.strftime('%a %m/%d')}: No games")
    
    # Weekly totals
    cursor.execute('''
    SELECT 
        COUNT(*) as total_picks,
        COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed,
        COUNT(CASE WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1 END) +
        COUNT(CASE WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1 END) as wins,
        AVG(confidence) as avg_confidence,
        AVG(ABS(predicted_total - total_runs)) as avg_error
    FROM enhanced_games 
    WHERE date BETWEEN %s AND %s
    AND recommendation IN ('OVER', 'UNDER')
    AND total_runs IS NOT NULL
    ''', (start_date, end_date))
    
    result = cursor.fetchone()
    total_picks, completed, wins, avg_conf, avg_error = result
    
    if completed > 0:
        win_rate = (wins / completed * 100)
        profit = (wins * 90.91) - ((completed - wins) * 100)
        profit_display = f"+${profit:.0f}" if profit > 0 else f"-${abs(profit):.0f}"
        
        print(f"\n📈 WEEKLY TOTALS:")
        print(f"   Record: {wins}/{completed} ({win_rate:.1f}%)")
        print(f"   Profit/Loss: {profit_display}")
        print(f"   Avg Confidence: {avg_conf:.1f}%")
        print(f"   Avg Error: {avg_error:.1f} runs")
        
        # Performance insights
        if win_rate >= 60 and profit > 0:
            print(f"   🎯 EXCELLENT WEEK! Profitable and high win rate")
        elif win_rate >= 55:
            print(f"   📈 SOLID WEEK! Above average performance")
        elif profit > 0:
            print(f"   💰 PROFITABLE WEEK! Even with lower win rate")
        else:
            print(f"   🔄 LEARNING WEEK! Analyze patterns for improvement")
    
    # Best performing bet types
    cursor.execute('''
    SELECT 
        recommendation,
        COUNT(*) as bets,
        COUNT(CASE WHEN 
            (recommendation = 'OVER' AND total_runs > market_total) OR
            (recommendation = 'UNDER' AND total_runs < market_total)
        THEN 1 END) as wins,
        AVG(confidence) as avg_conf
    FROM enhanced_games 
    WHERE date BETWEEN %s AND %s
    AND recommendation IN ('OVER', 'UNDER')
    AND total_runs IS NOT NULL
    GROUP BY recommendation
    ''', (start_date, end_date))
    
    bet_types = cursor.fetchall()
    if bet_types:
        print(f"\n🎲 BET TYPE PERFORMANCE:")
        for bet_type, bets, wins, avg_conf in bet_types:
            win_rate = (wins / bets * 100) if bets > 0 else 0
            print(f"   {bet_type}: {wins}/{bets} ({win_rate:.0f}%) - Avg Conf: {avg_conf:.0f}%")
    
    conn.close()

def confidence_analysis():
    """Analyze performance by confidence levels"""
    conn = connect_db()
    cursor = conn.cursor()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6)
    
    print(f"\n🎯 CONFIDENCE LEVEL ANALYSIS:")
    print("-" * 40)
    
    confidence_ranges = [
        (80, 100, "High Confidence (80%+)"),
        (65, 79, "Medium-High (65-79%)"),
        (50, 64, "Medium (50-64%)"),
        (0, 49, "Low Confidence (<50%)")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        cursor.execute('''
        SELECT 
            COUNT(*) as bets,
            COUNT(CASE WHEN 
                (recommendation = 'OVER' AND total_runs > market_total) OR
                (recommendation = 'UNDER' AND total_runs < market_total)
            THEN 1 END) as wins
        FROM enhanced_games 
        WHERE date BETWEEN %s AND %s
        AND recommendation IN ('OVER', 'UNDER')
        AND total_runs IS NOT NULL
        AND confidence BETWEEN %s AND %s
        ''', (start_date, end_date, min_conf, max_conf))
        
        result = cursor.fetchone()
        bets, wins = result
        
        if bets > 0:
            win_rate = (wins / bets * 100)
            profit = (wins * 90.91) - ((bets - wins) * 100)
            profit_display = f"+${profit:.0f}" if profit > 0 else f"-${abs(profit):.0f}"
            
            print(f"   {label}: {wins}/{bets} ({win_rate:.0f}%) {profit_display}")
    
    conn.close()

if __name__ == "__main__":
    weekly_analysis()
    confidence_analysis()
