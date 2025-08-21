#!/usr/bin/env python3
"""
Historical Performance Analysis
Analyze betting performance across all historical periods
"""

import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import json
from decimal import Decimal

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def analyze_historical_performance():
    """Analyze performance across all historical data"""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("📊 COMPREHENSIVE HISTORICAL PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Get overall data availability
    cursor.execute("""
        SELECT 
            COUNT(*) as total_games,
            COUNT(predicted_total) as with_predictions,
            COUNT(total_runs) as with_results,
            COUNT(CASE WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as complete_games,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM enhanced_games 
        WHERE date >= '2025-05-01'
    """)
    
    total, pred, results, complete, earliest, latest = cursor.fetchone()
    print(f"📈 DATA OVERVIEW:")
    print(f"   Total Games: {total:,}")
    print(f"   With Predictions: {pred:,}")
    print(f"   With Results: {results:,}")
    print(f"   Complete Games: {complete:,}")
    print(f"   Date Range: {earliest} to {latest}")
    print()
    
    # Monthly performance analysis
    print("🗓️ MONTHLY PERFORMANCE BREAKDOWN:")
    print("-" * 70)
    
    cursor.execute("""
        SELECT 
            EXTRACT(YEAR FROM date) as year,
            EXTRACT(MONTH FROM date) as month,
            COUNT(*) as total_games,
            COUNT(CASE WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as analyzable_games,
            
            -- Direction accuracy (prediction vs actual compared to market)
            COUNT(CASE 
                WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL AND market_total IS NOT NULL
                AND ((predicted_total > market_total AND total_runs > market_total) 
                     OR (predicted_total < market_total AND total_runs < market_total))
                THEN 1 
            END) as direction_correct,
            
            -- Betting performance (only games with recommendation)
            COUNT(CASE 
                WHEN recommendation IN ('OVER', 'UNDER') AND total_runs IS NOT NULL AND market_total IS NOT NULL
                THEN 1
            END) as betting_games,
            
            COUNT(CASE 
                WHEN recommendation = 'OVER' AND total_runs > market_total
                THEN 1
                WHEN recommendation = 'UNDER' AND total_runs < market_total  
                THEN 1
            END) as betting_wins,
            
            -- Average prediction error
            ROUND(AVG(CASE 
                WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(predicted_total - total_runs) 
            END), 2) as avg_prediction_error,
            
            -- Average market error  
            ROUND(AVG(CASE 
                WHEN market_total IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(market_total - total_runs) 
            END), 2) as avg_market_error
            
        FROM enhanced_games 
        WHERE date >= '2025-05-01'
        GROUP BY EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
        ORDER BY year DESC, month DESC
    """)
    
    monthly_results = cursor.fetchall()
    
    print("Month   | Games | Analyzable | Direction Acc | Betting | Betting Acc | Pred Error | Market Error | Profit*")
    print("-" * 100)
    
    total_profit = 0
    total_betting_games = 0
    total_betting_wins = 0
    
    for year, month, total, analyzable, direction_correct, betting_games, betting_wins, pred_error, market_error in monthly_results:
        
        if analyzable > 0:
            direction_accuracy = (direction_correct / analyzable) * 100
        else:
            direction_accuracy = 0
            
        if betting_games > 0:
            betting_accuracy = (betting_wins / betting_games) * 100
            # Calculate profit assuming -110 odds and $100 bets
            monthly_profit = (betting_wins * 90.91) - ((betting_games - betting_wins) * 100)
            total_profit += monthly_profit
            total_betting_games += betting_games
            total_betting_wins += betting_wins
        else:
            betting_accuracy = 0
            monthly_profit = 0
        
        month_name = f"{int(year)}-{int(month):02d}"
        
        print(f"{month_name:7s} | {total:5d} | {analyzable:10d} | {direction_accuracy:11.1f}% | {betting_games:7d} | {betting_accuracy:9.1f}% | {pred_error:10.2f} | {market_error:12.2f} | ${monthly_profit:+7.0f}")
    
    print("-" * 100)
    if total_betting_games > 0:
        overall_betting_accuracy = (total_betting_wins / total_betting_games) * 100
        print(f"TOTALS  | {'':<5} | {'':<10} | {'':<11} | {total_betting_games:7d} | {overall_betting_accuracy:9.1f}% | {'':<10} | {'':<12} | ${total_profit:+7.0f}")
    
    print("\n* Profit calculated at $100/bet with -110 odds")
    
    # Daily performance for recent periods
    print("\n📅 RECENT DAILY PERFORMANCE (Last 14 days with data):")
    print("-" * 80)
    
    cursor.execute("""
        SELECT 
            date,
            COUNT(*) as total_games,
            COUNT(CASE WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as analyzable_games,
            
            -- Direction accuracy
            COUNT(CASE 
                WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL AND market_total IS NOT NULL
                AND ((predicted_total > market_total AND total_runs > market_total) 
                     OR (predicted_total < market_total AND total_runs < market_total))
                THEN 1 
            END) as direction_correct,
            
            -- Betting performance
            COUNT(CASE 
                WHEN recommendation IN ('OVER', 'UNDER') AND total_runs IS NOT NULL AND market_total IS NOT NULL
                THEN 1
            END) as betting_games,
            
            COUNT(CASE 
                WHEN recommendation = 'OVER' AND total_runs > market_total
                THEN 1
                WHEN recommendation = 'UNDER' AND total_runs < market_total  
                THEN 1
            END) as betting_wins
            
        FROM enhanced_games 
        WHERE date >= CURRENT_DATE - INTERVAL '14 days'
        AND total_runs IS NOT NULL
        GROUP BY date
        ORDER BY date DESC
        LIMIT 14
    """)
    
    daily_results = cursor.fetchall()
    
    print("Date       | Games | Analyzable | Direction Acc | Betting | Betting Acc | Daily Profit")
    print("-" * 80)
    
    for date, total, analyzable, direction_correct, betting_games, betting_wins in daily_results:
        
        if analyzable > 0:
            direction_accuracy = (direction_correct / analyzable) * 100
        else:
            direction_accuracy = 0
            
        if betting_games > 0:
            betting_accuracy = (betting_wins / betting_games) * 100
            daily_profit = (betting_wins * 90.91) - ((betting_games - betting_wins) * 100)
        else:
            betting_accuracy = 0
            daily_profit = 0
        
        print(f"{date} | {total:5d} | {analyzable:10d} | {direction_accuracy:11.1f}% | {betting_games:7d} | {betting_accuracy:9.1f}% | ${daily_profit:+9.0f}")
    
    # Confidence level analysis
    print("\n🎯 CONFIDENCE LEVEL ANALYSIS:")
    print("-" * 60)
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN confidence >= 80 THEN '80%+'
                WHEN confidence >= 70 THEN '70-79%'
                WHEN confidence >= 60 THEN '60-69%'
                WHEN confidence >= 50 THEN '50-59%'
                ELSE 'Below 50%'
            END as confidence_range,
            COUNT(*) as total_games,
            
            -- Betting performance
            COUNT(CASE 
                WHEN recommendation IN ('OVER', 'UNDER') AND total_runs IS NOT NULL AND market_total IS NOT NULL
                THEN 1
            END) as betting_games,
            
            COUNT(CASE 
                WHEN recommendation = 'OVER' AND total_runs > market_total
                THEN 1
                WHEN recommendation = 'UNDER' AND total_runs < market_total  
                THEN 1
            END) as betting_wins
            
        FROM enhanced_games 
        WHERE confidence IS NOT NULL 
        AND total_runs IS NOT NULL
        AND date >= '2025-05-01'
        GROUP BY 
            CASE 
                WHEN confidence >= 80 THEN '80%+'
                WHEN confidence >= 70 THEN '70-79%'
                WHEN confidence >= 60 THEN '60-69%'
                WHEN confidence >= 50 THEN '50-59%'
                ELSE 'Below 50%'
            END
        ORDER BY MIN(confidence) DESC
    """)
    
    conf_results = cursor.fetchall()
    
    print("Confidence  | Total | Betting | Win Rate | Profit*")
    print("-" * 50)
    
    for conf_range, total, betting_games, betting_wins in conf_results:
        if betting_games > 0:
            win_rate = (betting_wins / betting_games) * 100
            profit = (betting_wins * 90.91) - ((betting_games - betting_wins) * 100)
        else:
            win_rate = 0
            profit = 0
        
        print(f"{conf_range:11s} | {total:5d} | {betting_games:7d} | {win_rate:7.1f}% | ${profit:+6.0f}")
    
    conn.close()

def create_performance_trends():
    """Create detailed performance trends over time"""
    
    conn = get_db_connection()
    
    # Get weekly performance trends
    query = """
        SELECT 
            DATE_TRUNC('week', date) as week_start,
            COUNT(*) as total_games,
            COUNT(CASE WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as analyzable_games,
            
            -- Direction accuracy
            COUNT(CASE 
                WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL AND market_total IS NOT NULL
                AND ((predicted_total > market_total AND total_runs > market_total) 
                     OR (predicted_total < market_total AND total_runs < market_total))
                THEN 1 
            END) as direction_correct,
            
            -- Betting performance
            COUNT(CASE 
                WHEN recommendation IN ('OVER', 'UNDER') AND total_runs IS NOT NULL AND market_total IS NOT NULL
                THEN 1
            END) as betting_games,
            
            COUNT(CASE 
                WHEN recommendation = 'OVER' AND total_runs > market_total
                THEN 1
                WHEN recommendation = 'UNDER' AND total_runs < market_total  
                THEN 1
            END) as betting_wins,
            
            AVG(CASE 
                WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL 
                THEN ABS(predicted_total - total_runs) 
            END) as avg_prediction_error
            
        FROM enhanced_games 
        WHERE date >= '2025-05-01'
        AND total_runs IS NOT NULL
        GROUP BY DATE_TRUNC('week', date)
        ORDER BY week_start DESC
        LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    
    print("\n📈 WEEKLY PERFORMANCE TRENDS:")
    print("-" * 80)
    print("Week Starting | Games | Direction Acc | Betting | Win Rate | Pred Error | Weekly Profit")
    print("-" * 80)
    
    for _, row in df.iterrows():
        week_start = row['week_start'].strftime('%Y-%m-%d')
        
        if row['analyzable_games'] > 0:
            direction_acc = (row['direction_correct'] / row['analyzable_games']) * 100
        else:
            direction_acc = 0
            
        if row['betting_games'] > 0:
            win_rate = (row['betting_wins'] / row['betting_games']) * 100
            weekly_profit = (row['betting_wins'] * 90.91) - ((row['betting_games'] - row['betting_wins']) * 100)
        else:
            win_rate = 0
            weekly_profit = 0
        
        pred_error = row['avg_prediction_error'] or 0
        
        print(f"{week_start} | {row['total_games']:5.0f} | {direction_acc:11.1f}% | {row['betting_games']:7.0f} | {win_rate:7.1f}% | {pred_error:9.2f} | ${weekly_profit:+10.0f}")
    
    conn.close()

def main():
    """Main execution"""
    print(f"🔍 Historical Performance Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        analyze_historical_performance()
        create_performance_trends()
        
        print("\n✅ ANALYSIS COMPLETE")
        print("\n🔍 KEY INSIGHTS:")
        print("   • Check monthly performance trends to see model evolution")
        print("   • Review confidence level performance to optimize betting strategy") 
        print("   • Use weekly trends to identify recent performance patterns")
        print("   • Compare prediction vs market errors to validate model edge")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
