#!/usr/bin/env python3
"""
GAME SCORE VALIDATION
Double-check actual game scores vs what's in our database
"""

import psycopg2
import pandas as pd
from datetime import datetime

class GameScoreValidator:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def check_sample_games_scores(self):
        """Check the specific games from our preview for score accuracy"""
        
        print("🔍 VALIDATING GAME SCORES FROM PREVIEW")
        print("=" * 60)
        
        conn = self.connect_db()
        
        # Get the same sample games we showed in preview
        query = """
        WITH monthly_samples AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY DATE_TRUNC('month', date) 
                       ORDER BY RANDOM()
                   ) as rn
            FROM enhanced_games
            WHERE date >= '2025-03-01'
              AND total_runs IS NOT NULL
              AND home_team_runs_l7 IS NOT NULL
              AND away_team_runs_l7 IS NOT NULL
        )
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            
            -- Check if total_runs = home_score + away_score
            (home_score + away_score) as calculated_total,
            ABS(total_runs - (home_score + away_score)) as score_difference,
            
            -- ERA information
            home_sp_season_era,
            away_sp_season_era,
            home_bullpen_era,
            away_bullpen_era
            
        FROM monthly_samples
        WHERE rn <= 3  -- Just 3 per month for detailed checking
        ORDER BY date, home_team;
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
    
    def display_score_validation(self, df):
        """Display detailed score validation"""
        
        print(f"📊 CHECKING {len(df)} SAMPLE GAMES FOR SCORE ACCURACY")
        print("=" * 70)
        
        score_issues = 0
        era_issues = 0
        
        for _, game in df.iterrows():
            print(f"\n🏟️  {game['date']} | {game['home_team']} vs {game['away_team']}")
            print("-" * 60)
            
            # Score validation
            db_total = game['total_runs']
            calculated_total = game['calculated_total']
            difference = game['score_difference']
            
            print(f"📊 SCORES:")
            print(f"   Database: {game['home_team']} {game['home_score']:.0f} - {game['away_score']:.0f} {game['away_team']}")
            print(f"   Total Runs: DB={db_total:.0f} | Calculated={calculated_total:.0f} | Diff={difference:.1f}")
            
            if difference > 0.1:
                print(f"   ❌ SCORE MISMATCH! Database total doesn't match individual scores")
                score_issues += 1
            else:
                print(f"   ✅ Scores match correctly")
            
            # ERA validation
            print(f"\n⚾ ERA DATA:")
            home_sp_str = f"{game['home_sp_season_era']:.2f}" if pd.notna(game['home_sp_season_era']) else 'NULL'
            away_sp_str = f"{game['away_sp_season_era']:.2f}" if pd.notna(game['away_sp_season_era']) else 'NULL'
            home_bp_str = f"{game['home_bullpen_era']:.2f}" if pd.notna(game['home_bullpen_era']) else 'NULL'
            away_bp_str = f"{game['away_bullpen_era']:.2f}" if pd.notna(game['away_bullpen_era']) else 'NULL'
            
            print(f"   Home SP ERA: {home_sp_str}")
            print(f"   Away SP ERA: {away_sp_str}")
            print(f"   Home Bullpen: {home_bp_str}")
            print(f"   Away Bullpen: {away_bp_str}")
            
            # Check for 0.00 ERAs or missing data
            era_problems = []
            if pd.isna(game['home_sp_season_era']) or game['home_sp_season_era'] == 0.0:
                era_problems.append("Home SP ERA missing/zero")
            if pd.isna(game['away_sp_season_era']) or game['away_sp_season_era'] == 0.0:
                era_problems.append("Away SP ERA missing/zero")
            
            if era_problems:
                print(f"   ⚠️  ERA ISSUES: {'; '.join(era_problems)}")
                era_issues += 1
            else:
                print(f"   ✅ ERA data looks reasonable")
        
        # Summary
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"   Games checked: {len(df)}")
        print(f"   Score mismatches: {score_issues} ({score_issues/len(df)*100:.1f}%)")
        print(f"   ERA issues: {era_issues} ({era_issues/len(df)*100:.1f}%)")
        
        if score_issues > 0:
            print(f"\n❌ CRITICAL: {score_issues} games have incorrect total_runs values!")
            print(f"   This will severely impact training accuracy.")
        
        if era_issues > 0:
            print(f"\n⚠️  WARNING: {era_issues} games missing critical ERA data")
    
    def check_era_data_completeness(self):
        """Check overall ERA data completeness and values"""
        
        print(f"\n🔍 ERA DATA COMPLETENESS ANALYSIS")
        print("=" * 50)
        
        conn = self.connect_db()
        
        era_query = """
        SELECT 
            COUNT(*) as total_games,
            COUNT(home_sp_season_era) as home_sp_era_count,
            COUNT(away_sp_season_era) as away_sp_era_count,
            COUNT(home_bullpen_era) as home_bp_era_count,
            COUNT(away_bullpen_era) as away_bp_era_count,
            
            -- Count zero ERAs
            SUM(CASE WHEN home_sp_season_era = 0.0 THEN 1 ELSE 0 END) as home_sp_zero_count,
            SUM(CASE WHEN away_sp_season_era = 0.0 THEN 1 ELSE 0 END) as away_sp_zero_count,
            
            -- ERA ranges
            MIN(home_sp_season_era) as min_home_sp_era,
            MAX(home_sp_season_era) as max_home_sp_era,
            AVG(home_sp_season_era) as avg_home_sp_era,
            
            MIN(away_sp_season_era) as min_away_sp_era,
            MAX(away_sp_season_era) as max_away_sp_era,
            AVG(away_sp_season_era) as avg_away_sp_era
            
        FROM enhanced_games
        WHERE date >= '2025-03-01'
          AND total_runs IS NOT NULL;
        """
        
        era_df = pd.read_sql(era_query, conn)
        conn.close()
        
        # Display results
        total = era_df.iloc[0]['total_games']
        
        print(f"📊 ERA COMPLETENESS ({total} total games):")
        print(f"   Home SP ERA:    {era_df.iloc[0]['home_sp_era_count']:4} ({era_df.iloc[0]['home_sp_era_count']/total*100:.1f}%)")
        print(f"   Away SP ERA:    {era_df.iloc[0]['away_sp_era_count']:4} ({era_df.iloc[0]['away_sp_era_count']/total*100:.1f}%)")
        print(f"   Home Bullpen:   {era_df.iloc[0]['home_bp_era_count']:4} ({era_df.iloc[0]['home_bp_era_count']/total*100:.1f}%)")
        print(f"   Away Bullpen:   {era_df.iloc[0]['away_bp_era_count']:4} ({era_df.iloc[0]['away_bp_era_count']/total*100:.1f}%)")
        
        print(f"\n❌ ZERO ERA PROBLEMS:")
        print(f"   Home SP 0.00 ERAs: {era_df.iloc[0]['home_sp_zero_count']}")
        print(f"   Away SP 0.00 ERAs: {era_df.iloc[0]['away_sp_zero_count']}")
        
        print(f"\n📈 ERA VALUE RANGES:")
        print(f"   Home SP: {era_df.iloc[0]['min_home_sp_era']:.2f} - {era_df.iloc[0]['max_home_sp_era']:.2f} (avg: {era_df.iloc[0]['avg_home_sp_era']:.2f})")
        print(f"   Away SP: {era_df.iloc[0]['min_away_sp_era']:.2f} - {era_df.iloc[0]['max_away_sp_era']:.2f} (avg: {era_df.iloc[0]['avg_away_sp_era']:.2f})")
        
        return era_df.iloc[0]
    
    def find_specific_score_mismatches(self):
        """Find all games where total_runs != home_score + away_score"""
        
        print(f"\n🔍 FINDING ALL SCORE MISMATCHES")
        print("=" * 50)
        
        conn = self.connect_db()
        
        mismatch_query = """
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            (home_score + away_score) as calculated_total,
            ABS(total_runs - (home_score + away_score)) as difference
        FROM enhanced_games
        WHERE date >= '2025-03-01'
          AND total_runs IS NOT NULL
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND ABS(total_runs - (home_score + away_score)) > 0.1
        ORDER BY difference DESC, date;
        """
        
        mismatch_df = pd.read_sql(mismatch_query, conn)
        conn.close()
        
        if len(mismatch_df) == 0:
            print("✅ No score mismatches found!")
            return None
        
        print(f"❌ Found {len(mismatch_df)} games with score mismatches:")
        print()
        
        for _, game in mismatch_df.head(10).iterrows():  # Show worst 10
            print(f"   {game['date']} | {game['home_team']} {game['home_score']:.0f}-{game['away_score']:.0f} {game['away_team']}")
            print(f"      DB Total: {game['total_runs']:.0f} | Should be: {game['calculated_total']:.0f} | Diff: {game['difference']:.1f}")
            print()
        
        if len(mismatch_df) > 10:
            print(f"   ... and {len(mismatch_df) - 10} more")
        
        return mismatch_df

def main():
    print("🔍 COMPREHENSIVE GAME DATA VALIDATION")
    print("   Checking scores and ERA data accuracy")
    print("=" * 60)
    
    validator = GameScoreValidator()
    
    # 1. Check sample games in detail
    print("STEP 1: Detailed validation of sample games")
    sample_df = validator.check_sample_games_scores()
    validator.display_score_validation(sample_df)
    
    # 2. Check ERA completeness
    print("\nSTEP 2: ERA data completeness analysis")
    era_stats = validator.check_era_data_completeness()
    
    # 3. Find all score mismatches
    print("\nSTEP 3: Finding all score mismatches")
    mismatch_df = validator.find_specific_score_mismatches()
    
    # Final assessment
    print(f"\n🎯 VALIDATION CONCLUSION:")
    if mismatch_df is not None and len(mismatch_df) > 0:
        print(f"   ❌ CRITICAL: {len(mismatch_df)} games have incorrect scores")
        print(f"   📝 NEXT STEPS: Fix these score mismatches before training")
    
    zero_eras = era_stats['home_sp_zero_count'] + era_stats['away_sp_zero_count']
    if zero_eras > 0:
        print(f"   ⚠️  WARNING: {zero_eras} games have 0.00 ERA values")
        print(f"   📝 NEXT STEPS: Calculate proper team season ERAs")
    
    if (mismatch_df is None or len(mismatch_df) == 0) and zero_eras == 0:
        print(f"   ✅ Data quality is excellent - ready for training!")

if __name__ == "__main__":
    main()
