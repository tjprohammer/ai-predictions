#!/usr/bin/env python3
"""
14-Day Performance Analysis - Working with actual database schema
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# Use same database connection as daily workflow
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def get_engine(url: str = DATABASE_URL):
    return create_engine(url, pool_pre_ping=True)

def analyze_14_day_performance():
    """Analyze 14 days of prediction performance using actual available columns"""
    
    engine = get_engine()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    print(f"[ANALYSIS] 14-Day Performance Analysis ({start_date} to {end_date})")
    print("=" * 70)
    
    # Query with actual available columns
    query = text("""
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            predicted_total,
            total_runs,
            market_total,
            confidence,
            
            -- Error metrics
            ABS(predicted_total - total_runs) as abs_error,
            (predicted_total - total_runs) as error,
            
            -- Weather factors (available)
            temperature,
            humidity,
            wind_speed,
            wind_direction,
            weather_condition,
            
            -- Pitcher factors (available)
            home_sp_season_era,
            away_sp_season_era,
            home_sp_season_k,
            away_sp_season_k,
            home_sp_season_bb,
            away_sp_season_bb,
            home_sp_season_ip,
            away_sp_season_ip,
            home_sp_whip,
            away_sp_whip,
            
            -- Game context
            EXTRACT(dow FROM date::date) as day_of_week,
            day_night,
            venue_name,
            ballpark_run_factor,
            ballpark_hr_factor,
            
            -- Market factors
            CASE WHEN market_total IS NOT NULL 
                 THEN ABS(predicted_total - market_total) 
                 ELSE NULL END as market_deviation
            
        FROM enhanced_games 
        WHERE date >= :start_date 
          AND date <= :end_date
          AND total_runs IS NOT NULL
          AND predicted_total IS NOT NULL
        ORDER BY date DESC, game_id
    """)
    
    df = pd.read_sql(query, engine, params={
        "start_date": start_date,
        "end_date": end_date
    })
    
    print(f"[DATA] Loaded {len(df)} games with predictions and outcomes")
    
    if len(df) == 0:
        print("[ERROR] No data found for analysis period")
        return
    
    # Overall Performance
    print(f"\n[OVERALL PERFORMANCE]")
    print("-" * 30)
    mae = df['abs_error'].mean()
    bias = df['error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    accuracy_1 = (df['abs_error'] <= 1.0).mean() * 100
    accuracy_2 = (df['abs_error'] <= 2.0).mean() * 100
    
    print(f"MAE: {mae:.3f} runs")
    print(f"RMSE: {rmse:.3f} runs")
    print(f"Bias: {bias:.3f} runs")
    print(f"≤1 Run Accuracy: {accuracy_1:.1f}%")
    print(f"≤2 Run Accuracy: {accuracy_2:.1f}%")
    
    # Error by Scoring Range
    print(f"\n[ERROR BY SCORING RANGE]")
    print("-" * 30)
    df['scoring_range'] = pd.cut(df['total_runs'], 
                               bins=[0, 7, 9, 11, float('inf')], 
                               labels=['Low (≤7)', 'Medium (8-9)', 'High (10-11)', 'Very High (12+)'])
    
    range_stats = df.groupby('scoring_range').agg({
        'abs_error': ['mean', 'count'],
        'error': 'mean',
        'total_runs': 'mean'
    }).round(3)
    
    for range_name in range_stats.index:
        games = range_stats.loc[range_name, ('abs_error', 'count')]
        mae_range = range_stats.loc[range_name, ('abs_error', 'mean')]
        bias_range = range_stats.loc[range_name, ('error', 'mean')]
        avg_runs = range_stats.loc[range_name, ('total_runs', 'mean')]
        print(f"{range_name}: {mae_range:.3f} MAE, {bias_range:+.3f} bias ({games} games, {avg_runs:.1f} avg runs)")
    
    # Weather Impact Analysis
    print(f"\n[WEATHER IMPACT]")
    print("-" * 20)
    
    if df['temperature'].notna().sum() > 0:
        df['temp_range'] = pd.cut(df['temperature'], 
                                bins=[0, 65, 75, 85, float('inf')], 
                                labels=['Cold (<65°)', 'Cool (65-74°)', 'Warm (75-84°)', 'Hot (85°+)'])
        
        temp_stats = df.groupby('temp_range').agg({
            'abs_error': ['mean', 'count'],
            'error': 'mean',
            'total_runs': 'mean'
        }).round(3)
        
        for temp_range in temp_stats.index:
            if pd.notna(temp_range):
                games = temp_stats.loc[temp_range, ('abs_error', 'count')]
                mae_temp = temp_stats.loc[temp_range, ('abs_error', 'mean')]
                bias_temp = temp_stats.loc[temp_range, ('error', 'mean')]
                avg_runs = temp_stats.loc[temp_range, ('total_runs', 'mean')]
                print(f"{temp_range}: {mae_temp:.3f} MAE, {bias_temp:+.3f} bias ({games} games, {avg_runs:.1f} avg runs)")
    
    # Pitcher Quality Impact
    print(f"\n[PITCHER QUALITY IMPACT]")
    print("-" * 30)
    
    # Combined ERA analysis
    df['combined_era'] = (df['home_sp_season_era'] + df['away_sp_season_era']) / 2
    df['era_range'] = pd.cut(df['combined_era'], 
                           bins=[0, 3.5, 4.5, 5.5, float('inf')], 
                           labels=['Elite (<3.5)', 'Good (3.5-4.5)', 'Average (4.5-5.5)', 'Poor (5.5+)'])
    
    era_stats = df.groupby('era_range').agg({
        'abs_error': ['mean', 'count'],
        'error': 'mean',
        'total_runs': 'mean'
    }).round(3)
    
    for era_range in era_stats.index:
        if pd.notna(era_range):
            games = era_stats.loc[era_range, ('abs_error', 'count')]
            mae_era = era_stats.loc[era_range, ('abs_error', 'mean')]
            bias_era = era_stats.loc[era_range, ('error', 'mean')]
            avg_runs = era_stats.loc[era_range, ('total_runs', 'mean')]
            print(f"{era_range}: {mae_era:.3f} MAE, {bias_era:+.3f} bias ({games} games, {avg_runs:.1f} avg runs)")
    
    # Market Deviation Analysis
    print(f"\n[MARKET DEVIATION ANALYSIS]")
    print("-" * 35)
    
    market_games = df[df['market_deviation'].notna()]
    if len(market_games) > 0:
        print(f"Games with market data: {len(market_games)}")
        print(f"Average market deviation: {market_games['market_deviation'].mean():.3f} runs")
        
        market_games['deviation_range'] = pd.cut(market_games['market_deviation'], 
                                               bins=[0, 0.5, 1.0, 2.0, float('inf')], 
                                               labels=['Close (≤0.5)', 'Small (0.5-1.0)', 'Medium (1.0-2.0)', 'Large (2.0+)'])
        
        dev_stats = market_games.groupby('deviation_range').agg({
            'abs_error': ['mean', 'count'],
            'error': 'mean'
        }).round(3)
        
        for dev_range in dev_stats.index:
            if pd.notna(dev_range):
                games = dev_stats.loc[dev_range, ('abs_error', 'count')]
                mae_dev = dev_stats.loc[dev_range, ('abs_error', 'mean')]
                bias_dev = dev_stats.loc[dev_range, ('error', 'mean')]
                print(f"{dev_range}: {mae_dev:.3f} MAE, {bias_dev:+.3f} bias ({games} games)")
    else:
        print("No market data available")
    
    # Day of Week Analysis
    print(f"\n[DAY OF WEEK ANALYSIS]")
    print("-" * 25)
    
    dow_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
              5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
    df['day_name'] = df['day_of_week'].map(dow_map)
    
    dow_stats = df.groupby('day_name').agg({
        'abs_error': ['mean', 'count'],
        'error': 'mean',
        'total_runs': 'mean'
    }).round(3)
    
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        if day in dow_stats.index:
            games = dow_stats.loc[day, ('abs_error', 'count')]
            mae_day = dow_stats.loc[day, ('abs_error', 'mean')]
            bias_day = dow_stats.loc[day, ('error', 'mean')]
            avg_runs = dow_stats.loc[day, ('total_runs', 'mean')]
            print(f"{day}: {mae_day:.3f} MAE, {bias_day:+.3f} bias ({games} games, {avg_runs:.1f} avg runs)")
    
    # Improvement Opportunities
    print(f"\n[IMPROVEMENT OPPORTUNITIES]")
    print("-" * 35)
    
    opportunities = []
    
    # Hot weather bias
    if 'temp_range' in df.columns:
        hot_games = df[df['temperature'] > 85]
        if len(hot_games) > 3:
            hot_mae = hot_games['abs_error'].mean()
            hot_bias = hot_games['error'].mean()
            if hot_mae > mae * 1.2:
                opportunities.append(f"Hot weather (85°+): {hot_mae:.3f} MAE vs {mae:.3f} overall")
            if abs(hot_bias) > 0.5:
                opportunities.append(f"Hot weather bias: {hot_bias:+.3f} runs (systematic)")
    
    # Poor pitching bias
    poor_pitching = df[df['combined_era'] > 5.5]
    if len(poor_pitching) > 3:
        poor_mae = poor_pitching['abs_error'].mean()
        poor_bias = poor_pitching['error'].mean()
        if poor_mae > mae * 1.2:
            opportunities.append(f"Poor pitching (ERA >5.5): {poor_mae:.3f} MAE vs {mae:.3f} overall")
        if abs(poor_bias) > 0.5:
            opportunities.append(f"Poor pitching bias: {poor_bias:+.3f} runs (systematic)")
    
    # Large market deviation
    if len(market_games) > 0:
        large_dev = market_games[market_games['market_deviation'] > 2.0]
        if len(large_dev) > 3:
            dev_mae = large_dev['abs_error'].mean()
            dev_bias = large_dev['error'].mean()
            if dev_mae > mae * 1.2:
                opportunities.append(f"Large market deviation (>2.0): {dev_mae:.3f} MAE vs {mae:.3f} overall")
            if abs(dev_bias) > 0.5:
                opportunities.append(f"Large market deviation bias: {dev_bias:+.3f} runs")
    
    # High scoring games
    high_scoring = df[df['total_runs'] > 11]
    if len(high_scoring) > 3:
        high_mae = high_scoring['abs_error'].mean()
        high_bias = high_scoring['error'].mean()
        if high_mae > mae * 1.2:
            opportunities.append(f"High scoring games (12+ runs): {high_mae:.3f} MAE vs {mae:.3f} overall")
        if abs(high_bias) > 1.0:
            opportunities.append(f"High scoring bias: {high_bias:+.3f} runs (large systematic bias)")
    
    if opportunities:
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp}")
    else:
        print("No major systematic biases identified")
    
    # Summary and Recommendations
    print(f"\n[RECOMMENDATIONS FOR MAE IMPROVEMENT]")
    print("-" * 45)
    print(f"Current MAE: {mae:.3f} runs")
    print(f"Current Bias: {bias:.3f} runs")
    print()
    print("1. WEATHER ADJUSTMENTS: Add temperature-specific corrections to bias system")
    print("2. PITCHER QUALITY: Implement ERA-based adjustments for extreme matchups")
    print("3. MARKET DEVIATION: Use large deviations as a signal for model uncertainty")
    print("4. SCORING RANGE: Apply range-specific corrections (already partially implemented)")
    print("5. DAY PATTERNS: Consider day-of-week factors for systematic differences")
    print()
    print("NEXT STEPS:")
    print("- Add weather conditions to bias correction system")
    print("- Implement pitcher quality scoring for extreme matchups")
    print("- Use market deviation as a confidence modifier")
    print("- Consider ballpark factors for venue-specific adjustments")
    
    return df

if __name__ == "__main__":
    df = analyze_14_day_performance()
