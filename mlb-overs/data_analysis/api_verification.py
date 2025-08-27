#!/usr/bin/env python3
"""
MLB STATS API VERIFICATION
Compare our database team stats with real MLB Stats API data
"""

import psycopg2
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time

class MLBStatsAPIVerifier:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        self.api_base = "https://statsapi.mlb.com/api/v1"
        
        # MLB team ID mapping
        self.team_name_to_id = {
            'Arizona Diamondbacks': 109,
            'Atlanta Braves': 144,
            'Baltimore Orioles': 110,
            'Boston Red Sox': 111,
            'Chicago Cubs': 112,
            'Chicago White Sox': 145,
            'Cincinnati Reds': 113,
            'Cleveland Guardians': 114,
            'Colorado Rockies': 115,
            'Detroit Tigers': 116,
            'Houston Astros': 117,
            'Kansas City Royals': 118,
            'Los Angeles Angels': 108,
            'Los Angeles Dodgers': 119,
            'Miami Marlins': 146,
            'Milwaukee Brewers': 158,
            'Minnesota Twins': 142,
            'New York Mets': 121,
            'New York Yankees': 147,
            'Athletics': 133,
            'Philadelphia Phillies': 143,
            'Pittsburgh Pirates': 134,
            'San Diego Padres': 135,
            'San Francisco Giants': 137,
            'Seattle Mariners': 136,
            'St. Louis Cardinals': 138,
            'Tampa Bay Rays': 139,
            'Texas Rangers': 140,
            'Toronto Blue Jays': 141,
            'Washington Nationals': 120
        }
    
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def get_sample_games_from_db(self, games_per_month=3):
        """Get sample games from our database for verification"""
        
        print(f"📊 GETTING {games_per_month} SAMPLE GAMES PER MONTH FROM DATABASE")
        print("=" * 60)
        
        conn = self.connect_db()
        
        sample_query = f"""
        WITH monthly_samples AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY DATE_TRUNC('month', date) 
                       ORDER BY date
                   ) as rn
            FROM enhanced_games
            WHERE date >= '2025-03-01'
              AND date <= '2025-08-23'  -- Only completed games
              AND total_runs IS NOT NULL
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
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
            home_bullpen_era,
            away_bullpen_era,
            home_team_avg,
            away_team_avg,
            home_team_runs_l7,
            away_team_runs_l7
        FROM monthly_samples
        WHERE rn <= {games_per_month}
        ORDER BY date, home_team;
        """
        
        sample_df = pd.read_sql(sample_query, conn)
        conn.close()
        
        print(f"✅ Retrieved {len(sample_df)} sample games from database")
        
        # Show sample breakdown by month
        sample_df['month'] = pd.to_datetime(sample_df['date']).dt.to_period('M')
        monthly_counts = sample_df['month'].value_counts().sort_index()
        
        print(f"\n📅 Games by month:")
        for month, count in monthly_counts.items():
            print(f"   {month}: {count} games")
        
        return sample_df
    
    def get_mlb_api_team_stats(self, team_name, season=2025):
        """Get team stats from MLB Stats API"""
        
        if team_name not in self.team_name_to_id:
            print(f"⚠️  Unknown team: {team_name}")
            return None
        
        team_id = self.team_name_to_id[team_name]
        
        try:
            # Get team stats for the season
            stats_url = f"{self.api_base}/teams/{team_id}/stats?stats=season&season={season}"
            response = requests.get(stats_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'stats' in data and len(data['stats']) > 0:
                    stats = data['stats'][0]
                    
                    # Extract pitching and hitting stats
                    pitching_stats = None
                    hitting_stats = None
                    
                    for split in stats.get('splits', []):
                        stat_group = split.get('group', {}).get('displayName', '')
                        if 'pitching' in stat_group.lower():
                            pitching_stats = split.get('stat', {})
                        elif 'hitting' in stat_group.lower():
                            hitting_stats = split.get('stat', {})
                    
                    return {
                        'team_name': team_name,
                        'team_id': team_id,
                        'pitching': pitching_stats,
                        'hitting': hitting_stats
                    }
            
            print(f"⚠️  API response error for {team_name}: {response.status_code}")
            return None
            
        except Exception as e:
            print(f"❌ Error fetching {team_name} stats: {str(e)}")
            return None
    
    def get_game_data_from_api(self, date, home_team, away_team):
        """Get specific game data from MLB API"""
        
        try:
            # Format date for API
            date_str = date.strftime('%m/%d/%Y') if hasattr(date, 'strftime') else str(date)
            
            # Get schedule for the date
            schedule_url = f"{self.api_base}/schedule?sportId=1&date={date_str}"
            response = requests.get(schedule_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find the specific game
                for game_date in data.get('dates', []):
                    for game in game_date.get('games', []):
                        home_team_name = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                        away_team_name = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                        
                        # Match team names (partial match for flexibility)
                        if (home_team.lower() in home_team_name.lower() and 
                            away_team.lower() in away_team_name.lower()):
                            
                            return {
                                'game_id': game.get('gamePk'),
                                'home_team': home_team_name,
                                'away_team': away_team_name,
                                'home_score': game.get('teams', {}).get('home', {}).get('score'),
                                'away_score': game.get('teams', {}).get('away', {}).get('score'),
                                'status': game.get('status', {}).get('detailedState'),
                                'date': date_str
                            }
            
            return None
            
        except Exception as e:
            print(f"❌ Error fetching game data: {str(e)}")
            return None
    
    def compare_team_stats(self, sample_df):
        """Compare our database stats with MLB API stats"""
        
        print(f"\n🔍 COMPARING DATABASE VS MLB API STATS")
        print("=" * 50)
        
        # Get unique teams from our sample
        teams = set(list(sample_df['home_team'].unique()) + list(sample_df['away_team'].unique()))
        teams = sorted(list(teams))[:10]  # Limit to 10 teams for API rate limiting
        
        print(f"📋 Fetching MLB API stats for {len(teams)} teams...")
        
        comparisons = []
        
        for i, team in enumerate(teams):
            print(f"   {i+1}/{len(teams)}: {team}...")
            
            # Get API stats
            api_stats = self.get_mlb_api_team_stats(team)
            time.sleep(0.5)  # Rate limiting
            
            if api_stats and api_stats['pitching']:
                # Get our database stats for this team
                team_games = sample_df[(sample_df['home_team'] == team) | (sample_df['away_team'] == team)]
                
                if len(team_games) > 0:
                    # Calculate our team's average stats
                    home_games = team_games[team_games['home_team'] == team]
                    away_games = team_games[team_games['away_team'] == team]
                    
                    our_era_values = []
                    our_avg_values = []
                    
                    if len(home_games) > 0:
                        our_era_values.extend(home_games['home_sp_season_era'].dropna().tolist())
                        our_avg_values.extend(home_games['home_team_avg'].dropna().tolist())
                    
                    if len(away_games) > 0:
                        our_era_values.extend(away_games['away_sp_season_era'].dropna().tolist())
                        our_avg_values.extend(away_games['away_team_avg'].dropna().tolist())
                    
                    if our_era_values and our_avg_values:
                        our_avg_era = sum(our_era_values) / len(our_era_values)
                        our_avg_batting = sum(our_avg_values) / len(our_avg_values)
                        
                        # API stats
                        api_era = float(api_stats['pitching'].get('era', 0))
                        api_avg = float(api_stats['hitting'].get('avg', 0)) if api_stats['hitting'] else 0
                        
                        comparison = {
                            'team': team,
                            'our_era': our_avg_era,
                            'api_era': api_era,
                            'era_diff': abs(our_avg_era - api_era),
                            'our_avg': our_avg_batting,
                            'api_avg': api_avg,
                            'avg_diff': abs(our_avg_batting - api_avg) if api_avg > 0 else 0,
                            'games_checked': len(team_games)
                        }
                        
                        comparisons.append(comparison)
        
        return comparisons
    
    def display_comparison_results(self, comparisons):
        """Display the comparison results"""
        
        print(f"\n📊 DATABASE vs MLB API COMPARISON RESULTS")
        print("=" * 70)
        
        print(f"{'Team':<25} {'Our ERA':<8} {'API ERA':<8} {'Diff':<6} {'Our AVG':<8} {'API AVG':<8} {'Diff':<6}")
        print("-" * 70)
        
        total_era_diff = 0
        total_avg_diff = 0
        valid_comparisons = 0
        
        for comp in comparisons:
            era_status = "✅" if comp['era_diff'] < 0.5 else "⚠️" if comp['era_diff'] < 1.0 else "❌"
            avg_status = "✅" if comp['avg_diff'] < 0.020 else "⚠️" if comp['avg_diff'] < 0.050 else "❌"
            
            print(f"{comp['team']:<25} {comp['our_era']:<8.2f} {comp['api_era']:<8.2f} {era_status}{comp['era_diff']:<5.2f} "
                  f"{comp['our_avg']:<8.3f} {comp['api_avg']:<8.3f} {avg_status}{comp['avg_diff']:<5.3f}")
            
            total_era_diff += comp['era_diff']
            total_avg_diff += comp['avg_diff']
            valid_comparisons += 1
        
        if valid_comparisons > 0:
            avg_era_diff = total_era_diff / valid_comparisons
            avg_avg_diff = total_avg_diff / valid_comparisons
            
            print(f"\n📈 SUMMARY:")
            print(f"   Teams compared: {valid_comparisons}")
            print(f"   Average ERA difference: {avg_era_diff:.3f}")
            print(f"   Average batting avg difference: {avg_avg_diff:.3f}")
            
            # Assessment
            if avg_era_diff < 0.5 and avg_avg_diff < 0.020:
                print(f"   ✅ EXCELLENT: Our data closely matches MLB API")
            elif avg_era_diff < 1.0 and avg_avg_diff < 0.050:
                print(f"   ✅ GOOD: Our data is reasonably accurate")
            else:
                print(f"   ⚠️  WARNING: Significant differences found")
        
        return comparisons
    
    def verify_sample_games(self, sample_df, num_games=5):
        """Verify specific game scores against API"""
        
        print(f"\n🎯 VERIFYING {num_games} SPECIFIC GAME SCORES")
        print("=" * 50)
        
        verified_games = []
        
        # Take recent completed games
        recent_games = sample_df[sample_df['date'] <= '2025-08-23'].tail(num_games)
        
        for _, game in recent_games.iterrows():
            print(f"\n🏟️  {game['date']} | {game['home_team']} vs {game['away_team']}")
            print(f"   Our scores: {game['home_team'][:3]} {game['home_score']:.0f} - {game['away_score']:.0f} {game['away_team'][:3]}")
            
            # Get API data
            api_game = self.get_game_data_from_api(game['date'], game['home_team'], game['away_team'])
            time.sleep(0.5)  # Rate limiting
            
            if api_game and api_game['home_score'] is not None:
                api_home = api_game['home_score']
                api_away = api_game['away_score']
                
                print(f"   API scores: {api_game['home_team'][:3]} {api_home} - {api_away} {api_game['away_team'][:3]}")
                
                home_match = abs(game['home_score'] - api_home) < 0.1
                away_match = abs(game['away_score'] - api_away) < 0.1
                
                if home_match and away_match:
                    print(f"   ✅ PERFECT MATCH")
                    verified_games.append({'game': game, 'status': 'match'})
                else:
                    print(f"   ❌ SCORE MISMATCH!")
                    verified_games.append({'game': game, 'status': 'mismatch'})
            else:
                print(f"   ⚠️  Could not retrieve API data")
                verified_games.append({'game': game, 'status': 'no_api_data'})
        
        # Summary
        matches = sum(1 for v in verified_games if v['status'] == 'match')
        print(f"\n📊 GAME VERIFICATION SUMMARY:")
        print(f"   Games verified: {len(verified_games)}")
        print(f"   Perfect matches: {matches}/{len(verified_games)}")
        print(f"   Accuracy: {matches/len(verified_games)*100:.1f}%")
        
        return verified_games

def main():
    print("🔍 MLB STATS API VERIFICATION")
    print("   Comparing our database with real MLB data")
    print("=" * 55)
    
    verifier = MLBStatsAPIVerifier()
    
    # Step 1: Get sample games from database
    print("STEP 1: Get sample games from database")
    sample_df = verifier.get_sample_games_from_db(games_per_month=3)
    
    # Step 2: Compare team stats with API
    print("\nSTEP 2: Compare team statistics with MLB API")
    comparisons = verifier.compare_team_stats(sample_df)
    
    # Step 3: Display results
    print("\nSTEP 3: Display comparison results")
    verifier.display_comparison_results(comparisons)
    
    # Step 4: Verify specific game scores
    print("\nSTEP 4: Verify specific game scores")
    verified_games = verifier.verify_sample_games(sample_df, num_games=5)
    
    # Final assessment
    print(f"\n🎯 FINAL VERIFICATION ASSESSMENT:")
    if len(comparisons) > 0:
        avg_era_diff = sum(c['era_diff'] for c in comparisons) / len(comparisons)
        if avg_era_diff < 0.5:
            print(f"   ✅ DATABASE ACCURACY: EXCELLENT")
        elif avg_era_diff < 1.0:
            print(f"   ✅ DATABASE ACCURACY: GOOD")
        else:
            print(f"   ⚠️  DATABASE ACCURACY: NEEDS REVIEW")
    
    matches = sum(1 for v in verified_games if v['status'] == 'match')
    if matches >= len(verified_games) * 0.8:  # 80% accuracy
        print(f"   ✅ GAME SCORES: VERIFIED ACCURATE")
    else:
        print(f"   ⚠️  GAME SCORES: SOME DISCREPANCIES")
    
    print(f"\n🚀 Ready to proceed with training based on verification results!")

if __name__ == "__main__":
    main()
