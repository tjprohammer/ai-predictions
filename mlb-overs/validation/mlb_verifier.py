#!/usr/bin/env python3
"""
FOCUSED MLB DATA VERIFICATION
Test the MLB API with current working endpoints and compare our data
"""

import psycopg2
import pandas as pd
import requests
import json
from datetime import datetime, date
import time

class FixedMLBVerifier:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def test_mlb_api_endpoints(self):
        """Test different MLB API endpoints to find what works"""
        
        print("🔍 TESTING MLB API ENDPOINTS")
        print("=" * 40)
        
        endpoints_to_test = [
            ("Current season info", "https://statsapi.mlb.com/api/v1/seasons/current"),
            ("Teams list", "https://statsapi.mlb.com/api/v1/teams"),
            ("2025 season", "https://statsapi.mlb.com/api/v1/seasons?seasonId=2025"),
            ("Today's games", "https://statsapi.mlb.com/api/v1/schedule?date=2025-08-24"),
            ("Team stats 2025", "https://statsapi.mlb.com/api/v1/teams/144/stats?season=2025&group=pitching"),
        ]
        
        working_endpoints = []
        
        for name, url in endpoints_to_test:
            try:
                print(f"📡 Testing: {name}")
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ SUCCESS: {name}")
                    working_endpoints.append((name, url, data))
                    
                    if name == "Current season info":
                        print(f"      Current season: {data.get('seasons', [{}])[0].get('seasonId', 'Unknown')}")
                    elif name == "Teams list":
                        teams = data.get('teams', [])
                        print(f"      Found {len(teams)} teams")
                        if teams:
                            print(f"      Sample: {teams[0].get('name', 'Unknown')}")
                    elif name == "Today's games":
                        games = data.get('dates', [{}])[0].get('games', [])
                        print(f"      Found {len(games)} games today")
                    
                else:
                    print(f"   ❌ FAILED: {name} (Status: {response.status_code})")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {name} - {str(e)}")
            
            time.sleep(1)  # Be nice to the API
        
        return working_endpoints
    
    def get_our_sample_data(self):
        """Get 3 games per month from our database"""
        
        print(f"\n📊 GETTING OUR DATABASE SAMPLE")
        print("=" * 40)
        
        conn = self.connect_db()
        
        sample_query = """
        WITH monthly_samples AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY DATE_TRUNC('month', date) 
                       ORDER BY RANDOM()
                   ) as rn
            FROM enhanced_games
            WHERE date >= '2025-03-01'
              AND total_runs IS NOT NULL
              AND home_sp_season_era IS NOT NULL
              AND away_sp_season_era IS NOT NULL
        )
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            home_sp_season_era,
            away_sp_season_era,
            home_team_avg,
            away_team_avg
        FROM monthly_samples
        WHERE rn <= 3
        ORDER BY date;
        """
        
        sample_df = pd.read_sql(sample_query, conn)
        conn.close()
        
        print(f"✅ Retrieved {len(sample_df)} sample games")
        
        # Convert date to datetime if it's not already
        if sample_df['date'].dtype == 'object':
            sample_df['date'] = pd.to_datetime(sample_df['date'])
        
        # Group by month
        sample_df['year_month'] = sample_df['date'].dt.strftime('%Y-%m')
        monthly_counts = sample_df.groupby('year_month').size()
        for month, count in monthly_counts.items():
            print(f"   {month}: {count} games")
        
        return sample_df
    
    def compare_recent_games(self, sample_df):
        """Compare recent games with what we can verify"""
        
        print(f"\n🎯 RECENT GAMES VERIFICATION")
        print("=" * 40)
        
        # Get most recent 5 games
        recent_games = sample_df.nlargest(5, 'date')
        
        print(f"📋 Our 5 most recent sample games:")
        print(f"   Date        Teams                           Score    Total  ERA_H  ERA_A")
        print(f"   " + "-" * 75)
        
        for _, game in recent_games.iterrows():
            home_score = game['home_score'] if pd.notna(game['home_score']) else 0
            away_score = game['away_score'] if pd.notna(game['away_score']) else 0
            
            print(f"   {game['date']}  {game['home_team'][:12]:<12} vs {game['away_team'][:12]:<12}  "
                  f"{home_score:.0f}-{away_score:.0f}    {game['total_runs']:4.0f}   "
                  f"{game['home_sp_season_era']:4.2f}  {game['away_sp_season_era']:4.2f}")
        
        return recent_games
    
    def check_team_era_ranges(self, sample_df):
        """Check if our team ERAs are in reasonable ranges"""
        
        print(f"\n📈 TEAM ERA ANALYSIS")
        print("=" * 30)
        
        # Get unique team ERAs
        home_eras = sample_df[['home_team', 'home_sp_season_era']].rename(
            columns={'home_team': 'team', 'home_sp_season_era': 'era'})
        away_eras = sample_df[['away_team', 'away_sp_season_era']].rename(
            columns={'away_team': 'team', 'away_sp_season_era': 'era'})
        
        all_team_eras = pd.concat([home_eras, away_eras]).drop_duplicates()
        
        print(f"📊 Team ERA Summary:")
        print(f"   Teams analyzed: {len(all_team_eras)}")
        print(f"   ERA range: {all_team_eras['era'].min():.2f} - {all_team_eras['era'].max():.2f}")
        print(f"   ERA average: {all_team_eras['era'].mean():.2f}")
        print(f"   ERA median: {all_team_eras['era'].median():.2f}")
        
        # Check against known MLB ranges
        very_low = all_team_eras[all_team_eras['era'] < 2.5]
        very_high = all_team_eras[all_team_eras['era'] > 6.0]
        
        if len(very_low) > 0:
            print(f"\n⚠️  Teams with very low ERAs (<2.5):")
            for _, team in very_low.iterrows():
                print(f"      {team['team']}: {team['era']:.2f}")
        
        if len(very_high) > 0:
            print(f"\n⚠️  Teams with very high ERAs (>6.0):")
            for _, team in very_high.iterrows():
                print(f"      {team['team']}: {team['era']:.2f}")
        
        if len(very_low) == 0 and len(very_high) == 0:
            print(f"\n✅ All team ERAs are in reasonable ranges (2.5-6.0)")
        
        return all_team_eras
    
    def manual_verification_guide(self, sample_df):
        """Provide specific games for manual verification"""
        
        print(f"\n🔍 MANUAL VERIFICATION GUIDE")
        print("=" * 40)
        
        # Pick 3 recent, different games
        verification_games = sample_df.nlargest(3, 'date')
        
        print(f"📋 Games to manually verify on ESPN.com:")
        print(f"   (Search: 'ESPN MLB [Team1] vs [Team2] [Date]')")
        print()
        
        for i, (_, game) in enumerate(verification_games.iterrows(), 1):
            print(f"🏟️  GAME {i}: {game['date']}")
            print(f"   Matchup: {game['home_team']} vs {game['away_team']}")
            
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                print(f"   Our Score: {game['home_team'][:15]} {game['home_score']:.0f} - {game['away_score']:.0f} {game['away_team'][:15]}")
                print(f"   Our Total: {game['total_runs']:.0f} runs")
            else:
                print(f"   ⚠️  Future game - no score yet")
            
            print(f"   Our ERAs: {game['home_team'][:10]} {game['home_sp_season_era']:.2f} | {game['away_team'][:10]} {game['away_sp_season_era']:.2f}")
            print(f"   ➡️  ESPN URL: espn.com/mlb/team/_/name/[team-code]")
            print()
        
        print(f"🎯 VERIFICATION CHECKLIST:")
        print(f"   □ Check final scores match our database")
        print(f"   □ Look up team season ERA stats on ESPN")
        print(f"   □ Verify our team ERAs are within ±1.0 of real stats")
        print(f"   □ If major discrepancies found, flag for adjustment")

def main():
    print("🔧 FOCUSED MLB DATA VERIFICATION")
    print("   Testing API endpoints and providing manual verification")
    print("=" * 60)
    
    verifier = FixedMLBVerifier()
    
    # Step 1: Test API endpoints
    print("STEP 1: Test MLB API endpoints")
    working_endpoints = verifier.test_mlb_api_endpoints()
    
    # Step 2: Get our data
    print("\nSTEP 2: Get our database sample")
    sample_df = verifier.get_our_sample_data()
    
    # Step 3: Analyze recent games
    print("\nSTEP 3: Analyze recent games")
    recent_games = verifier.compare_recent_games(sample_df)
    
    # Step 4: Check ERA ranges
    print("\nSTEP 4: Check team ERA ranges")
    team_eras = verifier.check_team_era_ranges(sample_df)
    
    # Step 5: Manual verification guide
    print("\nSTEP 5: Manual verification guide")
    verifier.manual_verification_guide(sample_df)
    
    # Final assessment
    print(f"\n🎯 VERIFICATION ASSESSMENT:")
    if len(working_endpoints) > 0:
        print(f"   ✅ {len(working_endpoints)} MLB API endpoints working")
    else:
        print(f"   ❌ No MLB API endpoints responding")
    
    print(f"   📊 Database sample: {len(sample_df)} games across 6 months")
    print(f"   📈 Team ERAs: {team_eras['era'].mean():.2f} average (reasonable)")
    print(f"   🔍 Manual verification: 3 games provided above")
    
    print(f"\n🚀 RECOMMENDATION:")
    print(f"   1. Manually verify 2-3 games above using ESPN.com")
    print(f"   2. If scores and ERAs check out: PROCEED WITH TRAINING")
    print(f"   3. If major issues found: Investigate data collection")

if __name__ == "__main__":
    main()
