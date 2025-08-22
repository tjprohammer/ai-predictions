#!/usr/bin/env python3
"""
Historical Training Data Builder
===============================

Systematically collect and organize historical prediction data for model training.
This includes:
1. Backfilling missing market data from historical odds
2. Re-running predictions on historical games with current model
3. Creating clean training datasets
4. Identifying performance patterns over time
"""

import psycopg2
import requests
import time
from datetime import datetime, timedelta
import json
import pandas as pd
from pathlib import Path

def connect_db():
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def analyze_historical_gaps():
    """Identify gaps in our historical data"""
    conn = connect_db()
    cursor = conn.cursor()
    
    print("🔍 ANALYZING HISTORICAL DATA GAPS")
    print("=" * 50)
    
    # Find dates with missing market data
    cursor.execute('''
    SELECT date, 
           COUNT(*) as total_games,
           COUNT(CASE WHEN market_total IS NULL THEN 1 END) as missing_market,
           COUNT(CASE WHEN total_runs IS NULL THEN 1 END) as missing_actual
    FROM enhanced_games
    WHERE date >= '2024-09-15'
    GROUP BY date
    HAVING COUNT(CASE WHEN market_total IS NULL THEN 1 END) > 0
    ORDER BY date DESC
    LIMIT 20
    ''')
    
    gaps = cursor.fetchall()
    
    print(f"📅 Dates Missing Market Data (top 20):")
    total_missing_market = 0
    for date, total, missing_market, missing_actual in gaps:
        print(f"  {date}: {missing_market}/{total} games missing market data")
        total_missing_market += missing_market
    
    print(f"\\n📊 Total games missing market data: {total_missing_market}")
    
    # Check seasonal patterns
    cursor.execute('''
    SELECT 
        EXTRACT(MONTH FROM date) as month,
        COUNT(*) as games,
        COUNT(CASE WHEN market_total IS NOT NULL AND total_runs IS NOT NULL THEN 1 END) as complete,
        AVG(CASE WHEN predicted_total IS NOT NULL AND total_runs IS NOT NULL 
            THEN ABS(predicted_total - total_runs) END) as avg_error
    FROM enhanced_games
    WHERE date >= '2024-09-15'
    GROUP BY EXTRACT(MONTH FROM date)
    ORDER BY month
    ''')
    
    seasonal = cursor.fetchall()
    
    print(f"\\n📈 SEASONAL DATA QUALITY:")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month, games, complete, avg_error in seasonal:
        if games > 0:
            completeness = (complete/games*100)
            month_name = month_names[int(month)]
            error_str = f"{avg_error:.2f}" if avg_error else "N/A"
            print(f"  {month_name}: {complete}/{games} ({completeness:.0f}%) - Avg Error: {error_str}")
    
    conn.close()
    return total_missing_market

def backfill_market_data(start_date, end_date):
    """Attempt to backfill missing market data from various sources"""
    print(f"\\n🔄 BACKFILLING MARKET DATA: {start_date} to {end_date}")
    print("-" * 50)
    
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get games missing market data
    cursor.execute('''
    SELECT game_id, date, home_team, away_team, predicted_total
    FROM enhanced_games
    WHERE date BETWEEN %s AND %s
    AND market_total IS NULL
    AND predicted_total IS NOT NULL
    ORDER BY date DESC
    ''', (start_date, end_date))
    
    missing_games = cursor.fetchall()
    
    print(f"Found {len(missing_games)} games missing market data")
    
    # For historical data, we can estimate market totals based on:
    # 1. Similar games from that time period
    # 2. Team averages
    # 3. Venue averages
    
    updated_count = 0
    
    for game_id, date, home_team, away_team, predicted_total in missing_games:
        # Find similar games around that date for market estimation
        cursor.execute('''
        SELECT AVG(market_total) as avg_market
        FROM enhanced_games
        WHERE date BETWEEN %s AND %s
        AND market_total IS NOT NULL
        AND (home_team = %s OR away_team = %s OR home_team = %s OR away_team = %s)
        ''', (date - timedelta(days=7), date + timedelta(days=7), 
              home_team, home_team, away_team, away_team))
        
        result = cursor.fetchone()
        estimated_market = result[0] if result[0] else None
        
        if estimated_market:
            # Update with estimated market total (mark as estimated)
            cursor.execute('''
            UPDATE enhanced_games 
            SET market_total = %s,
                notes = COALESCE(notes || '; ', '') || 'Market total estimated from similar games'
            WHERE game_id = %s
            ''', (round(estimated_market, 1), game_id))
            
            updated_count += 1
            
            if updated_count % 10 == 0:
                print(f"  Updated {updated_count} games...")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Updated {updated_count} games with estimated market data")
    return updated_count

def create_training_dataset(output_dir="training_data"):
    """Create clean training datasets from historical data"""
    print(f"\\n📊 CREATING TRAINING DATASETS")
    print("-" * 50)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    conn = connect_db()
    
    # Complete games with all required data
    query = '''
    SELECT 
        date,
        game_id,
        home_team,
        away_team,
        predicted_total,
        market_total,
        total_runs as actual_total,
        CASE 
            WHEN predicted_total > market_total THEN 'OVER_PREDICTION'
            WHEN predicted_total < market_total THEN 'UNDER_PREDICTION'
            ELSE 'NEUTRAL_PREDICTION'
        END as prediction_direction,
        CASE 
            WHEN total_runs > market_total THEN 'OVER_ACTUAL'
            WHEN total_runs < market_total THEN 'UNDER_ACTUAL'
            ELSE 'NEUTRAL_ACTUAL'
        END as actual_direction,
        ABS(predicted_total - total_runs) as prediction_error,
        ABS(market_total - total_runs) as market_error,
        predicted_total - market_total as edge,
        recommendation,
        confidence,
        venue_name,
        temperature,
        weather_condition,
        EXTRACT(MONTH FROM date) as month,
        EXTRACT(DOW FROM date) as day_of_week,
        -- Performance metrics
        CASE 
            WHEN (predicted_total > market_total AND total_runs > market_total) OR
                 (predicted_total < market_total AND total_runs < market_total)
            THEN 1 ELSE 0 
        END as direction_correct,
        CASE 
            WHEN recommendation = 'OVER' AND total_runs > market_total THEN 1
            WHEN recommendation = 'UNDER' AND total_runs < market_total THEN 1
            WHEN recommendation = 'HOLD' THEN NULL
            ELSE 0
        END as bet_won
    FROM enhanced_games
    WHERE predicted_total IS NOT NULL
    AND market_total IS NOT NULL
    AND total_runs IS NOT NULL
    AND date >= '2024-09-15'
    ORDER BY date DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    
    print(f"📈 Dataset Summary:")
    print(f"  Total complete games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Prediction accuracy: {df['direction_correct'].mean():.1%}")
    
    # Save main dataset
    df.to_csv(f"{output_dir}/complete_historical_data.csv", index=False)
    print(f"✅ Saved: {output_dir}/complete_historical_data.csv")
    
    # Create betting-focused dataset
    betting_df = df[df['recommendation'].isin(['OVER', 'UNDER'])].copy()
    betting_df['profit'] = betting_df['bet_won'].apply(
        lambda x: 90.91 if x == 1 else -100 if x == 0 else 0
    )
    
    betting_df.to_csv(f"{output_dir}/betting_performance_data.csv", index=False)
    print(f"✅ Saved: {output_dir}/betting_performance_data.csv")
    
    # Create monthly summaries
    monthly_summary = df.groupby(['month']).agg({
        'game_id': 'count',
        'direction_correct': 'mean',
        'prediction_error': 'mean',
        'market_error': 'mean',
        'bet_won': lambda x: x.dropna().mean()
    }).round(3)
    
    monthly_summary.to_csv(f"{output_dir}/monthly_performance_summary.csv")
    print(f"✅ Saved: {output_dir}/monthly_performance_summary.csv")
    
    # Create team-specific analysis
    team_summary = df.groupby(['home_team']).agg({
        'game_id': 'count',
        'direction_correct': 'mean',
        'prediction_error': 'mean',
        'actual_total': 'mean'
    }).round(2)
    
    team_summary.to_csv(f"{output_dir}/team_performance_analysis.csv")
    print(f"✅ Saved: {output_dir}/team_performance_analysis.csv")
    
    conn.close()
    
    return len(df)

def generate_model_insights():
    """Analyze historical data to identify model improvement opportunities"""
    print(f"\\n🧠 GENERATING MODEL INSIGHTS")
    print("-" * 50)
    
    conn = connect_db()
    cursor = conn.cursor()
    
    # Find patterns in prediction errors
    cursor.execute('''
    SELECT 
        CASE 
            WHEN ABS(predicted_total - total_runs) <= 1 THEN 'Excellent (≤1)'
            WHEN ABS(predicted_total - total_runs) <= 2 THEN 'Good (≤2)'
            WHEN ABS(predicted_total - total_runs) <= 3 THEN 'Fair (≤3)'
            ELSE 'Poor (>3)'
        END as accuracy_bucket,
        COUNT(*) as games,
        AVG(confidence) as avg_confidence,
        COUNT(CASE WHEN recommendation IN ('OVER', 'UNDER') THEN 1 END) as actionable,
        AVG(CASE 
            WHEN (recommendation = 'OVER' AND total_runs > market_total) OR
                 (recommendation = 'UNDER' AND total_runs < market_total)
            THEN 1 ELSE 0 END) as bet_win_rate
    FROM enhanced_games
    WHERE predicted_total IS NOT NULL
    AND market_total IS NOT NULL
    AND total_runs IS NOT NULL
    AND date >= '2024-09-15'
    GROUP BY 1
    ORDER BY 
        CASE 
            WHEN ABS(predicted_total - total_runs) <= 1 THEN 1
            WHEN ABS(predicted_total - total_runs) <= 2 THEN 2
            WHEN ABS(predicted_total - total_runs) <= 3 THEN 3
            ELSE 4
        END
    ''')
    
    accuracy_analysis = cursor.fetchall()
    
    print("🎯 PREDICTION ACCURACY ANALYSIS:")
    for bucket, games, avg_conf, actionable, bet_win_rate in accuracy_analysis:
        print(f"  {bucket}: {games} games, {avg_conf:.0f}% confidence, {bet_win_rate:.1%} bet win rate")
    
    # Weather impact analysis
    cursor.execute('''
    SELECT 
        CASE 
            WHEN weather_condition ILIKE '%rain%' OR weather_condition ILIKE '%storm%' THEN 'Wet'
            WHEN weather_condition ILIKE '%wind%' THEN 'Windy'
            WHEN weather_condition ILIKE '%clear%' OR weather_condition ILIKE '%sunny%' THEN 'Clear'
            ELSE 'Other'
        END as weather_type,
        COUNT(*) as games,
        AVG(total_runs) as avg_actual,
        AVG(predicted_total) as avg_predicted,
        AVG(ABS(predicted_total - total_runs)) as avg_error
    FROM enhanced_games
    WHERE predicted_total IS NOT NULL
    AND total_runs IS NOT NULL
    AND weather_condition IS NOT NULL
    AND date >= '2024-09-15'
    GROUP BY 1
    HAVING COUNT(*) >= 10
    ORDER BY avg_error
    ''')
    
    weather_analysis = cursor.fetchall()
    
    print(f"\\n🌤️ WEATHER IMPACT ANALYSIS:")
    for weather, games, avg_actual, avg_pred, avg_error in weather_analysis:
        print(f"  {weather}: {games} games, {avg_actual:.1f} actual, {avg_pred:.1f} predicted, {avg_error:.2f} error")
    
    conn.close()

def main():
    """Run complete historical data analysis and enhancement"""
    print("🏗️ HISTORICAL TRAINING DATA BUILDER")
    print("=" * 60)
    
    # 1. Analyze current data gaps
    missing_count = analyze_historical_gaps()
    
    # 2. Backfill recent missing data
    if missing_count > 0:
        start_date = datetime.now().date() - timedelta(days=30)
        end_date = datetime.now().date()
        backfill_market_data(start_date, end_date)
    
    # 3. Create training datasets
    total_games = create_training_dataset()
    
    # 4. Generate insights
    generate_model_insights()
    
    print(f"\\n✅ HISTORICAL DATA BUILDER COMPLETE")
    print(f"📊 Total training games available: {total_games}")
    print(f"📁 Training datasets saved to: ./training_data/")
    print(f"🚀 Ready for enhanced model training!")

if __name__ == "__main__":
    main()
