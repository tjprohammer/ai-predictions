#!/usr/bin/env python3
"""
WORKING Team Stats Ingestor
==========================

Collects recent team performance statistics for prediction purposes.
Gets team averages, recent form, and offensive/defensive metrics
that help predict future game outcomes.
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import argparse
from datetime import datetime, timedelta

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def get_team_season_stats(team_id):
    """Get team season statistics from MLB API"""
    try:
        # Get team stats for current season
        url = f'https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season=2025'
        response = requests.get(url)
        data = response.json()
        
        if 'stats' in data and len(data['stats']) > 0:
            stats = data['stats'][0]['splits'][0]['stat']
            return {
                'avg': float(stats.get('avg', 0.250)),
                'ops': float(stats.get('ops', 0.700)),
                'runs': int(stats.get('runs', 0)),
                'hits': int(stats.get('hits', 0)),
                'rbi': int(stats.get('rbi', 0)),
                'hr': int(stats.get('homeRuns', 0)),
                'games': int(stats.get('gamesPlayed', 1))
            }
    except Exception as e:
        print(f"⚠️ Error getting stats for team {team_id}: {e}")
    
    # Return reasonable defaults if API fails
    return {
        'avg': 0.250,
        'ops': 0.700, 
        'runs': 400,
        'hits': 1000,
        'rbi': 400,
        'hr': 150,
        'games': 100
    }

def get_team_id_from_name(team_name):
    """Convert team name to MLB team ID"""
    team_mapping = {
        'Arizona Diamondbacks': 109,
        'Atlanta Braves': 144,
        'Baltimore Orioles': 110,
        'Boston Red Sox': 111,
        'Chicago White Sox': 145,
        'Chicago Cubs': 112,
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
        'New York Yankees': 147,
        'New York Mets': 121,
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
    return team_mapping.get(team_name, 0)

def collect_team_performance_stats(target_date=None):
    """Collect team performance stats for target date's games"""
    print("🏏 Collecting Team Performance Statistics")
    print("=" * 40)
    
    # Use target date if provided, otherwise use current date
    if target_date:
        date_filter = f"date = '{target_date}'"
        print(f"📅 Collecting for target date: {target_date}")
    else:
        date_filter = "date = CURRENT_DATE"
        print("📅 Collecting for current date")
    
    engine = get_engine()
    updated_count = 0
    
    try:
        with engine.begin() as conn:
            # Get target date's games that need team stats
            todays_games = pd.read_sql(f"""
                SELECT game_id, home_team, away_team
                FROM enhanced_games 
                WHERE {date_filter}
                AND game_id IS NOT NULL
                ORDER BY game_id
            """, conn)
            
            print(f"🔍 Collecting team stats for {len(todays_games)} games...")
            
            for _, game in todays_games.iterrows():
                game_id = game['game_id']
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Get team IDs
                home_team_id = get_team_id_from_name(home_team)
                away_team_id = get_team_id_from_name(away_team)
                
                # Get team stats
                home_stats = get_team_season_stats(home_team_id)
                away_stats = get_team_season_stats(away_team_id)
                
                # Calculate averages per game
                home_rpg = home_stats['runs'] / max(home_stats['games'], 1)  # Runs per game
                away_rpg = away_stats['runs'] / max(away_stats['games'], 1)
                
                home_hpg = home_stats['hits'] / max(home_stats['games'], 1)  # Hits per game
                away_hpg = away_stats['hits'] / max(away_stats['games'], 1)
                
                home_rbi_pg = home_stats['rbi'] / max(home_stats['games'], 1)  # RBI per game
                away_rbi_pg = away_stats['rbi'] / max(away_stats['games'], 1)
                
                # Get batting averages
                home_avg = home_stats['avg']
                away_avg = away_stats['avg']
                
                # For today's games, use season averages as predictive stats
                # These represent the team's expected performance level
                # Build WHERE clause based on target_date parameter
                if target_date:
                    date_condition = "AND date = :target_date"
                else:
                    date_condition = "AND date = CURRENT_DATE"
                
                update_sql = text(f"""
                    UPDATE enhanced_games 
                    SET 
                        home_team_hits = :home_hits,
                        home_team_runs = :home_runs,
                        home_team_rbi = :home_rbi,
                        home_team_lob = 6,  -- Average LOB per game
                        home_team_avg = :home_avg,
                        away_team_hits = :away_hits,
                        away_team_runs = :away_runs,
                        away_team_rbi = :away_rbi,
                        away_team_lob = 6,   -- Average LOB per game
                        away_team_avg = :away_avg
                    WHERE game_id = :game_id
                      {date_condition}
                """)
                
                # Also update legitimate_game_features with runs per game data
                lgf_update_sql = text(f"""
                    UPDATE legitimate_game_features
                    SET 
                        home_team_runs_pg = :home_rpg,
                        away_team_runs_pg = :away_rpg,
                        combined_team_offense = :combined_offense,
                        home_team_woba = :home_woba,
                        away_team_woba = :away_woba,
                        combined_team_woba = :combined_woba,
                        home_team_power = :home_power,
                        away_team_power = :away_power,
                        combined_team_power = :combined_power,
                        home_team_wrcplus = :home_wrcplus,
                        away_team_wrcplus = :away_wrcplus,
                        combined_team_wrcplus = :combined_wrcplus
                    WHERE game_id = :game_id
                      {date_condition}
                """)
                
                # Prepare parameters for SQL execution
                sql_params = {
                    'game_id': game_id,
                    'home_hits': round(home_hpg),
                    'home_runs': round(home_rpg), 
                    'home_rbi': round(home_rbi_pg),
                    'home_avg': home_avg,
                    'away_hits': round(away_hpg),
                    'away_runs': round(away_rpg),
                    'away_rbi': round(away_rbi_pg),
                    'away_avg': away_avg,
                    'home_rpg': home_rpg,
                    'away_rpg': away_rpg,
                    'combined_offense': (home_rpg + away_rpg) / 2,
                    # 🔧 FIXED: Better OPS to wOBA conversion - OPS*0.4 gives realistic range 0.26-0.36
                    'home_woba': max(0.250, min(0.370, home_stats.get('ops', 0.700) * 0.4)),
                    'away_woba': max(0.250, min(0.370, away_stats.get('ops', 0.700) * 0.4)),
                    'combined_woba': max(0.250, min(0.370, (home_stats.get('ops', 0.700) + away_stats.get('ops', 0.700)) * 0.2)),
                    'home_power': home_stats.get('hr', 20) / home_stats.get('games', 162),  # HR per game
                    'away_power': away_stats.get('hr', 20) / away_stats.get('games', 162),
                    'combined_power': ((home_stats.get('hr', 20) + away_stats.get('hr', 20)) / 
                                     (home_stats.get('games', 162) + away_stats.get('games', 162))),
                    # wRC+ approximation: (OPS-0.65)*150 + 100 gives realistic 70-140 range
                    'home_wrcplus': max(70, min(140, (home_stats.get('ops', 0.700) - 0.65) * 150 + 100)),
                    'away_wrcplus': max(70, min(140, (away_stats.get('ops', 0.700) - 0.65) * 150 + 100)),
                    'combined_wrcplus': max(70, min(140, ((home_stats.get('ops', 0.700) + away_stats.get('ops', 0.700)) / 2 - 0.65) * 150 + 100))
                }
                
                # Add target_date to params if specified
                if target_date:
                    sql_params['target_date'] = target_date
                
                # Update both tables
                result = conn.execute(update_sql, sql_params)
                lgf_result = conn.execute(lgf_update_sql, sql_params)
                
                if result.rowcount > 0 or lgf_result.rowcount > 0:
                    updated_count += 1
                    print(f"📊 {away_team} @ {home_team}")
                    print(f"   Home: {home_rpg:.1f} R/G, {home_hpg:.1f} H/G, {home_stats['avg']:.3f} AVG")
                    print(f"   Away: {away_rpg:.1f} R/G, {away_hpg:.1f} H/G, {away_stats['avg']:.3f} AVG")
        
        return updated_count
        
    except Exception as e:
        print(f"❌ Error collecting team stats: {e}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect team performance statistics for MLB games')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("🏏 Team Performance Data Collection")
    print("=" * 40)
    print("📈 Collecting team season averages for prediction purposes")
    print()
    
    updated = collect_team_performance_stats(args.target_date)
    
    if updated > 0:
        print(f"\n✅ Successfully collected team stats for {updated} games")
        print("🎯 Team performance data ready for ML model")
    else:
        print("❌ No team stats collected")

if __name__ == "__main__":
    main()
