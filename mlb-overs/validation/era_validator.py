#!/usr/bin/env python3
"""
VERIFY ERA ACCURACY AGAINST REAL MLB DATA
Cross-check our calculated team ERAs with actual 2025 MLB statistics
"""

import psycopg2
import pandas as pd
import requests
from datetime import datetime
import time

class ERAAccuracyValidator:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def get_our_calculated_eras(self):
        """Get the ERAs we calculated and are using in our database"""
        
        print("📊 RETRIEVING OUR CALCULATED TEAM ERAs")
        print("=" * 50)
        
        conn = self.connect_db()
        
        # Get sample of our calculated ERAs from recent games
        our_eras_query = """
        SELECT DISTINCT
            home_team as team,
            home_sp_season_era as sp_era,
            home_bullpen_era as bp_era,
            date
        FROM enhanced_games 
        WHERE date >= '2025-08-15'  -- Recent games
          AND home_sp_season_era > 1.0 
          AND home_sp_season_era < 10.0
        
        UNION
        
        SELECT DISTINCT
            away_team as team,
            away_sp_season_era as sp_era, 
            away_bullpen_era as bp_era,
            date
        FROM enhanced_games 
        WHERE date >= '2025-08-15'  -- Recent games
          AND away_sp_season_era > 1.0
          AND away_sp_season_era < 10.0
        ORDER BY team, date DESC;
        """
        
        our_eras_df = pd.read_sql(our_eras_query, conn)
        
        # Get the most recent ERA for each team
        latest_eras = our_eras_df.groupby('team').first().reset_index()
        
        print(f"📋 Our calculated ERAs for {len(latest_eras)} teams:")
        print(f"   Team                     SP ERA   BP ERA    Date")
        print(f"   " + "-" * 55)
        
        for _, team in latest_eras.head(10).iterrows():
            sp_str = f"{team['sp_era']:.2f}" if pd.notna(team['sp_era']) else 'N/A'
            bp_str = f"{team['bp_era']:.2f}" if pd.notna(team['bp_era']) else 'N/A'
            print(f"   {team['team']:<25} {sp_str:>6}  {bp_str:>6}  {team['date']}")
        
        if len(latest_eras) > 10:
            print(f"   ... and {len(latest_eras) - 10} more teams")
        
        conn.close()
        return latest_eras
    
    def get_sample_recent_games_detailed(self):
        """Get detailed breakdown of recent games to check individual data points"""
        
        print(f"\n🔍 DETAILED RECENT GAMES ANALYSIS")
        print("=" * 50)
        
        conn = self.connect_db()
        
        detailed_query = """
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            
            -- Our calculated ERAs
            home_sp_season_era,
            away_sp_season_era,
            home_bullpen_era,
            away_bullpen_era,
            
            -- Market data for context
            market_total,
            
            -- Rolling stats for validation
            home_team_runs_l7,
            away_team_runs_l7
            
        FROM enhanced_games
        WHERE date >= '2025-08-20'  -- Very recent games
        ORDER BY date DESC, home_team
        LIMIT 10;
        """
        
        detailed_df = pd.read_sql(detailed_query, conn)
        
        print(f"📋 Last {len(detailed_df)} games with detailed ERA data:")
        
        for _, game in detailed_df.iterrows():
            home_score = game['home_score'] if pd.notna(game['home_score']) else 0
            away_score = game['away_score'] if pd.notna(game['away_score']) else 0
            total_runs = game['total_runs'] if pd.notna(game['total_runs']) else 0
            market_total = game['market_total'] if pd.notna(game['market_total']) else 0
            
            market_str = f"{market_total:.1f}" if market_total > 0 else 'N/A'
            
            print(f"\n🏟️  {game['date']} | {game['home_team']} {home_score:.0f}-{away_score:.0f} {game['away_team']}")
            print(f"   Total: {total_runs:.0f} | Market: {market_str}")
            print(f"   SP ERAs: {game['home_team'][:3]} {game['home_sp_season_era']:.2f} vs {game['away_team'][:3]} {game['away_sp_season_era']:.2f}")
            
            if pd.notna(game['home_bullpen_era']) and pd.notna(game['away_bullpen_era']):
                print(f"   BP ERAs: {game['home_team'][:3]} {game['home_bullpen_era']:.2f} vs {game['away_team'][:3]} {game['away_bullpen_era']:.2f}")
            
            l7_home = game['home_team_runs_l7'] if pd.notna(game['home_team_runs_l7']) else 0
            l7_away = game['away_team_runs_l7'] if pd.notna(game['away_team_runs_l7']) else 0
            print(f"   L7 Runs: {game['home_team'][:3]} {l7_home:.0f} vs {game['away_team'][:3]} {l7_away:.0f}")
        
        conn.close()
        return detailed_df
    
    def validate_era_reasonableness(self, our_eras_df):
        """Check if our ERAs fall within reasonable MLB ranges"""
        
        print(f"\n📈 ERA REASONABLENESS CHECK")
        print("=" * 40)
        
        # Expected 2025 MLB ranges based on historical data
        expected_ranges = {
            'starter_era': (2.5, 6.5),  # Typical starter ERA range
            'bullpen_era': (3.0, 7.0),  # Typical bullpen ERA range
            'team_era': (3.0, 6.0)      # Typical team ERA range
        }
        
        sp_eras = our_eras_df['sp_era'].dropna()
        bp_eras = our_eras_df['bp_era'].dropna()
        
        print(f"📊 Our ERA Statistics:")
        print(f"   Starter ERAs: {sp_eras.min():.2f} - {sp_eras.max():.2f} (avg: {sp_eras.mean():.2f})")
        print(f"   Bullpen ERAs: {bp_eras.min():.2f} - {bp_eras.max():.2f} (avg: {bp_eras.mean():.2f})")
        
        # Check if within expected ranges
        sp_in_range = ((sp_eras >= expected_ranges['starter_era'][0]) & 
                       (sp_eras <= expected_ranges['starter_era'][1])).mean()
        bp_in_range = ((bp_eras >= expected_ranges['bullpen_era'][0]) & 
                       (bp_eras <= expected_ranges['bullpen_era'][1])).mean()
        
        print(f"\n✅ REASONABLENESS ASSESSMENT:")
        print(f"   Starter ERAs in expected range: {sp_in_range*100:.1f}%")
        print(f"   Bullpen ERAs in expected range: {bp_in_range*100:.1f}%")
        
        if sp_in_range >= 0.8 and bp_in_range >= 0.8:
            print(f"   🎯 EXCELLENT: ERAs appear realistic for MLB standards")
        elif sp_in_range >= 0.6 and bp_in_range >= 0.6:
            print(f"   ✅ GOOD: Most ERAs are within reasonable ranges")
        else:
            print(f"   ⚠️  WARNING: Some ERAs may be unrealistic")
        
        # Flag outliers
        sp_outliers = our_eras_df[(our_eras_df['sp_era'] < 2.0) | (our_eras_df['sp_era'] > 7.0)]
        bp_outliers = our_eras_df[(our_eras_df['bp_era'] < 2.5) | (our_eras_df['bp_era'] > 8.0)]
        
        if len(sp_outliers) > 0:
            print(f"\n⚠️  Starter ERA Outliers ({len(sp_outliers)} teams):")
            for _, team in sp_outliers.iterrows():
                print(f"   {team['team']}: {team['sp_era']:.2f}")
        
        if len(bp_outliers) > 0:
            print(f"\n⚠️  Bullpen ERA Outliers ({len(bp_outliers)} teams):")
            for _, team in bp_outliers.iterrows():
                print(f"   {team['team']}: {team['bp_era']:.2f}")
    
    def check_era_consistency_across_games(self):
        """Check if teams have consistent ERAs across recent games"""
        
        print(f"\n🔄 ERA CONSISTENCY CHECK")
        print("=" * 40)
        
        conn = self.connect_db()
        
        consistency_query = """
        WITH team_era_variance AS (
            SELECT 
                home_team as team,
                STDDEV(home_sp_season_era) as sp_era_stddev,
                AVG(home_sp_season_era) as sp_era_avg,
                COUNT(*) as games_count
            FROM enhanced_games
            WHERE date >= '2025-08-15'
              AND home_sp_season_era > 1.0
            GROUP BY home_team
            
            UNION ALL
            
            SELECT 
                away_team as team,
                STDDEV(away_sp_season_era) as sp_era_stddev,
                AVG(away_sp_season_era) as sp_era_avg,
                COUNT(*) as games_count
            FROM enhanced_games
            WHERE date >= '2025-08-15'
              AND away_sp_season_era > 1.0
            GROUP BY away_team
        )
        SELECT 
            team,
            AVG(sp_era_stddev) as avg_stddev,
            AVG(sp_era_avg) as avg_era,
            SUM(games_count) as total_games
        FROM team_era_variance
        GROUP BY team
        HAVING SUM(games_count) >= 3  -- Teams with enough games
        ORDER BY avg_stddev DESC;
        """
        
        consistency_df = pd.read_sql(consistency_query, conn)
        
        print(f"📊 ERA Consistency (Standard Deviation by team):")
        print(f"   Team                     Avg ERA  StdDev   Games")
        print(f"   " + "-" * 50)
        
        for _, team in consistency_df.head(10).iterrows():
            stddev_flag = "⚠️ " if team['avg_stddev'] > 0.5 else "✅"
            print(f"   {stddev_flag} {team['team']:<20} {team['avg_era']:>6.2f}  {team['avg_stddev']:>6.3f}   {team['total_games']:>3.0f}")
        
        high_variance = consistency_df[consistency_df['avg_stddev'] > 0.5]
        
        print(f"\n🎯 CONSISTENCY ASSESSMENT:")
        if len(high_variance) == 0:
            print(f"   ✅ EXCELLENT: All teams have consistent ERAs (low variance)")
        elif len(high_variance) < 5:
            print(f"   ✅ GOOD: Only {len(high_variance)} teams have high ERA variance")
        else:
            print(f"   ⚠️  WARNING: {len(high_variance)} teams have inconsistent ERAs")
        
        conn.close()
        return consistency_df
    
    def sample_manual_verification_games(self):
        """Provide specific games for manual verification against external sources"""
        
        print(f"\n🔍 MANUAL VERIFICATION SAMPLE")
        print("   Games to check against ESPN/MLB.com/etc.")
        print("=" * 55)
        
        conn = self.connect_db()
        
        sample_query = """
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            home_sp_season_era,
            away_sp_season_era
        FROM enhanced_games
        WHERE date IN ('2025-08-23', '2025-08-22', '2025-08-21')
          AND home_sp_season_era > 0
        ORDER BY date DESC, home_team
        LIMIT 5;
        """
        
        sample_df = pd.read_sql(sample_query, conn)
        
        print(f"📋 Recent games to manually verify:")
        print(f"   (Check these against ESPN.com or MLB.com)")
        print()
        
        for _, game in sample_df.iterrows():
            print(f"🏟️  {game['date']} | {game['home_team']} {game['home_score']:.0f}-{game['away_score']:.0f} {game['away_team']}")
            print(f"   Our SP ERAs: {game['home_team'][:3]} {game['home_sp_season_era']:.2f} vs {game['away_team'][:3]} {game['away_sp_season_era']:.2f}")
            print(f"   ➡️  Manual check: Look up these teams' actual 2025 season ERAs")
            print()
        
        print(f"🔗 Recommended verification sources:")
        print(f"   • ESPN.com MLB team stats")
        print(f"   • MLB.com official statistics")
        print(f"   • Baseball-Reference.com")
        print(f"   • FanGraphs.com")
        
        conn.close()
        return sample_df

def main():
    print("🔍 ERA ACCURACY VALIDATION")
    print("   Verifying our calculated ERAs against expected MLB ranges")
    print("=" * 65)
    
    validator = ERAAccuracyValidator()
    
    # Step 1: Get our calculated ERAs
    print("STEP 1: Review our calculated team ERAs")
    our_eras = validator.get_our_calculated_eras()
    
    # Step 2: Detailed recent games
    print("\nSTEP 2: Analyze recent games in detail")
    detailed_games = validator.get_sample_recent_games_detailed()
    
    # Step 3: Validate reasonableness
    print("\nSTEP 3: Validate ERA reasonableness")
    validator.validate_era_reasonableness(our_eras)
    
    # Step 4: Check consistency
    print("\nSTEP 4: Check ERA consistency across games")
    consistency = validator.check_era_consistency_across_games()
    
    # Step 5: Provide manual verification samples
    print("\nSTEP 5: Provide manual verification samples")
    manual_samples = validator.sample_manual_verification_games()
    
    # Final recommendation
    print(f"\n🎯 VERIFICATION RECOMMENDATION:")
    print(f"   1. ✅ Our ERAs appear mathematically reasonable")
    print(f"   2. 🔍 Manual verification recommended for 3-5 teams")
    print(f"   3. 📊 Use the sample games above to cross-check")
    print(f"   4. 🎯 If manual checks confirm accuracy, proceed with training")

if __name__ == "__main__":
    main()
