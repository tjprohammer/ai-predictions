#!/usr/bin/env python3
"""
ANALYZE CURRENT DATA COVERAGE FOR FULL SEASON
Check what data we have and what we need to collect for comprehensive training
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date

class SeasonDataAnalyzer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def analyze_current_coverage(self):
        """Analyze what data we currently have"""
        
        print("📊 ANALYZING CURRENT SEASON DATA COVERAGE")
        print("=" * 50)
        
        conn = self.connect_db()
        
        # Get overall coverage by month
        query = """
        SELECT 
            DATE_TRUNC('month', date) as month,
            COUNT(*) as total_games,
            COUNT(total_runs) as games_with_scores,
            COUNT(home_sp_season_era) as games_with_era,
            COUNT(home_team_runs_l7) as games_with_rolling,
            COUNT(market_total) as games_with_market,
            
            -- Calculate percentages
            ROUND(COUNT(total_runs)::numeric / COUNT(*) * 100, 1) as score_pct,
            ROUND(COUNT(home_sp_season_era)::numeric / COUNT(*) * 100, 1) as era_pct,
            ROUND(COUNT(home_team_runs_l7)::numeric / COUNT(*) * 100, 1) as rolling_pct,
            ROUND(COUNT(market_total)::numeric / COUNT(*) * 100, 1) as market_pct
            
        FROM enhanced_games
        WHERE date >= '2025-03-01'  -- Start of 2025 season
        GROUP BY DATE_TRUNC('month', date)
        ORDER BY month;
        """
        
        monthly_df = pd.read_sql(query, conn)
        
        print("📅 MONTHLY DATA COVERAGE:")
        print()
        print("Month        | Games | Scores | ERAs  | Rolling | Market")
        print("-" * 55)
        
        total_games = 0
        total_with_scores = 0
        
        for _, row in monthly_df.iterrows():
            month_name = row['month'].strftime('%Y-%m')
            total_games += row['total_games']
            total_with_scores += row['games_with_scores']
            
            print(f"{month_name}     | {row['total_games']:5d} | {row['score_pct']:5.1f}% | {row['era_pct']:5.1f}% | {row['rolling_pct']:6.1f}% | {row['market_pct']:5.1f}%")
        
        print("-" * 55)
        print(f"TOTAL        | {total_games:5d} | {total_with_scores/total_games*100:5.1f}% overall")
        
        # Check date range
        date_query = """
        SELECT 
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as unique_days
        FROM enhanced_games;
        """
        
        date_df = pd.read_sql(date_query, conn)
        
        print(f"\n📅 DATE RANGE:")
        print(f"   Earliest game: {date_df['earliest_date'].iloc[0]}")
        print(f"   Latest game: {date_df['latest_date'].iloc[0]}")
        print(f"   Unique days: {date_df['unique_days'].iloc[0]}")
        
        conn.close()
        return monthly_df
    
    def identify_missing_periods(self):
        """Identify specific periods with missing data"""
        
        print(f"\n🔍 IDENTIFYING MISSING DATA PERIODS")
        print("-" * 40)
        
        conn = self.connect_db()
        
        # Check for gaps in the season
        query = """
        WITH date_series AS (
            SELECT generate_series(
                '2025-03-20'::date,  -- Spring training/season start
                '2025-08-24'::date,  -- Current date
                '1 day'::interval
            )::date as expected_date
        ),
        game_dates AS (
            SELECT DISTINCT date as actual_date
            FROM enhanced_games
        )
        SELECT 
            expected_date,
            CASE WHEN actual_date IS NULL THEN 'MISSING' ELSE 'HAS_DATA' END as status
        FROM date_series ds
        LEFT JOIN game_dates gd ON ds.expected_date = gd.actual_date
        WHERE expected_date NOT IN (
            SELECT generate_series(
                '2025-07-15'::date,  -- All-Star break typically
                '2025-07-18'::date,
                '1 day'::interval
            )::date
        )
        ORDER BY expected_date;
        """
        
        gaps_df = pd.read_sql(query, conn)
        
        # Find consecutive missing periods
        missing_dates = gaps_df[gaps_df['status'] == 'MISSING']['expected_date'].tolist()
        
        if missing_dates:
            print(f"📊 Found {len(missing_dates)} missing dates:")
            
            # Group consecutive dates
            consecutive_periods = []
            current_period = [missing_dates[0]]
            
            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i-1]).days == 1:
                    current_period.append(missing_dates[i])
                else:
                    consecutive_periods.append(current_period)
                    current_period = [missing_dates[i]]
            consecutive_periods.append(current_period)
            
            print("   Missing periods:")
            for period in consecutive_periods:
                if len(period) == 1:
                    print(f"     {period[0]} (1 day)")
                else:
                    print(f"     {period[0]} to {period[-1]} ({len(period)} days)")
        else:
            print("   ✅ No missing dates found!")
        
        conn.close()
        return missing_dates
    
    def assess_training_data_needs(self):
        """Determine what data we need for proper ML training"""
        
        print(f"\n🎯 TRAINING DATA REQUIREMENTS ASSESSMENT")
        print("-" * 45)
        
        conn = self.connect_db()
        
        # Get current high-quality data counts
        query = """
        SELECT 
            COUNT(*) as total_games,
            COUNT(CASE WHEN total_runs IS NOT NULL 
                      AND home_team_runs_l7 IS NOT NULL 
                      AND away_team_runs_l7 IS NOT NULL
                      AND home_sp_season_era IS NOT NULL 
                      AND away_sp_season_era IS NOT NULL
                      AND home_team_avg IS NOT NULL
                      AND away_team_avg IS NOT NULL
                      THEN 1 END) as complete_games,
            
            -- By time period
            COUNT(CASE WHEN date >= '2025-07-01' AND total_runs IS NOT NULL THEN 1 END) as recent_complete,
            COUNT(CASE WHEN date >= '2025-05-01' AND date < '2025-07-01' AND total_runs IS NOT NULL THEN 1 END) as mid_season_complete,
            COUNT(CASE WHEN date >= '2025-03-01' AND date < '2025-05-01' AND total_runs IS NOT NULL THEN 1 END) as early_season_complete
            
        FROM enhanced_games
        WHERE date >= '2025-03-01';
        """
        
        training_df = pd.read_sql(query, conn)
        
        total = training_df['total_games'].iloc[0]
        complete = training_df['complete_games'].iloc[0]
        recent = training_df['recent_complete'].iloc[0]
        mid_season = training_df['mid_season_complete'].iloc[0]
        early_season = training_df['early_season_complete'].iloc[0]
        
        print(f"📊 CURRENT TRAINING DATA STATUS:")
        print(f"   Total games in DB: {total:,}")
        print(f"   Complete games: {complete:,} ({complete/total*100:.1f}%)")
        print()
        print(f"   Recent (Jul-Aug): {recent:,} games")
        print(f"   Mid-season (May-Jun): {mid_season:,} games")
        print(f"   Early season (Mar-Apr): {early_season:,} games")
        
        # ML recommendations
        print(f"\n🤖 ML TRAINING RECOMMENDATIONS:")
        
        if complete >= 1000:
            print(f"   ✅ EXCELLENT: {complete:,} complete games - great for training!")
        elif complete >= 500:
            print(f"   ✅ GOOD: {complete:,} complete games - sufficient for training")
        elif complete >= 300:
            print(f"   ⚠️ MINIMAL: {complete:,} complete games - workable but need more")
        else:
            print(f"   ❌ INSUFFICIENT: {complete:,} complete games - need much more data")
        
        # Seasonal distribution check
        seasonal_balance = min(recent, mid_season, early_season) / max(recent, mid_season, early_season) if max(recent, mid_season, early_season) > 0 else 0
        
        if seasonal_balance > 0.7:
            print(f"   ✅ Good seasonal balance across the year")
        elif seasonal_balance > 0.4:
            print(f"   ⚠️ Moderate seasonal imbalance")
        else:
            print(f"   ❌ Poor seasonal balance - need more early/mid season data")
        
        conn.close()
        return complete, seasonal_balance
    
    def recommend_data_collection_strategy(self, complete_games, seasonal_balance, missing_dates):
        """Recommend comprehensive data collection approach"""
        
        print(f"\n🚀 COMPREHENSIVE DATA COLLECTION STRATEGY")
        print("=" * 50)
        
        priorities = []
        
        # Priority 1: Missing recent games (critical)
        if len([d for d in missing_dates if d >= date(2025, 8, 1)]) > 0:
            priorities.append({
                'priority': 1,
                'task': 'Collect missing August games',
                'reason': 'Recent games needed for current predictions',
                'effort': 'LOW',
                'impact': 'HIGH'
            })
        
        # Priority 2: Complete feature collection for existing games
        if complete_games < 1000:
            priorities.append({
                'priority': 2,
                'task': 'Complete features for existing games',
                'reason': 'Fill missing ERAs, rolling stats, market totals',
                'effort': 'MEDIUM',
                'impact': 'HIGH'
            })
        
        # Priority 3: Full season backfill
        if seasonal_balance < 0.7:
            priorities.append({
                'priority': 3,
                'task': 'Backfill early/mid season games',
                'reason': 'Need balanced training data across season',
                'effort': 'HIGH',
                'impact': 'MEDIUM'
            })
        
        # Priority 4: Historical data (if needed)
        if complete_games < 500:
            priorities.append({
                'priority': 4,
                'task': 'Collect 2024 season data',
                'reason': 'Insufficient 2025 data for robust training',
                'effort': 'VERY_HIGH',
                'impact': 'HIGH'
            })
        
        print("📋 RECOMMENDED COLLECTION PRIORITIES:")
        print()
        
        for i, task in enumerate(priorities, 1):
            print(f"   {task['priority']}. {task['task']}")
            print(f"      Reason: {task['reason']}")
            print(f"      Effort: {task['effort']} | Impact: {task['impact']}")
            print()
        
        # Immediate next steps
        print("⚡ IMMEDIATE NEXT STEPS:")
        
        if complete_games >= 500 and seasonal_balance > 0.4:
            print("   1. ✅ You can start training with current data")
            print("   2. 📊 Collect missing recent games (Aug 23-24)")
            print("   3. 🔧 Fill remaining feature gaps")
            print("   4. 📈 Gradually backfill more historical data")
        else:
            print("   1. 🔧 Complete features for existing games FIRST")
            print("   2. 📊 Backfill missing games from current season")
            print("   3. ⚖️ Balance seasonal data distribution")
            print("   4. 🎯 Then proceed with training")
        
        return priorities

def main():
    print("📈 COMPREHENSIVE SEASON DATA ANALYSIS")
    print("   Assessing current coverage and collection needs")
    print("=" * 60)
    
    analyzer = SeasonDataAnalyzer()
    
    # Analyze current coverage
    monthly_df = analyzer.analyze_current_coverage()
    
    # Identify missing periods
    missing_dates = analyzer.identify_missing_periods()
    
    # Assess training readiness
    complete_games, seasonal_balance = analyzer.assess_training_data_needs()
    
    # Get recommendations
    priorities = analyzer.recommend_data_collection_strategy(complete_games, seasonal_balance, missing_dates)
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Complete training games: {complete_games:,}")
    print(f"   Seasonal balance: {seasonal_balance:.2f}")
    print(f"   Missing dates: {len(missing_dates)}")
    print(f"   Collection priorities: {len(priorities)}")

if __name__ == "__main__":
    main()
