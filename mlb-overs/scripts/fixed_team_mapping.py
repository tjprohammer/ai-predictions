#!/usr/bin/env python3
"""
Fixed team name mapping for Oakland Athletics issue
"""

import sys
import pandas as pd
import psycopg2
from datetime import datetime

# Database connection function
def get_connection():
    """Get a database connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass',
        port=5432
    )

def normalize_team_name(team_name):
    """Normalize team name to match database format"""
    team_mapping = {
        'Athletics': 'Oakland Athletics',
        'A\'s': 'Oakland Athletics',
        'Oakland A\'s': 'Oakland Athletics',
        'Oakland Athletics': 'Oakland Athletics',
        'Tampa Bay Rays': 'Tampa Bay Rays',
        'Rays': 'Tampa Bay Rays',
        'St. Louis Cardinals': 'St. Louis Cardinals',
        'Cardinals': 'St. Louis Cardinals'
    }
    
    return team_mapping.get(team_name, team_name)

def test_team_mapping():
    """Test the team mapping with real data"""
    date_str = '2025-08-23'
    
    # Get today's teams
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT home_team, away_team 
                FROM legitimate_game_features 
                WHERE date = %s
            """, (date_str,))
            
            game_teams = set()
            for row in cursor.fetchall():
                game_teams.add(row[0])
                game_teams.add(row[1])
    
    print(f"🔍 TESTING TEAM NAME MAPPING FOR {date_str}")
    print("=" * 60)
    print(f"Teams from games: {sorted(game_teams)}")
    print()
    
    # Apply mapping
    mapped_teams = []
    for team in sorted(game_teams):
        mapped = normalize_team_name(team)
        if mapped != team:
            print(f"🔄 MAPPED: '{team}' → '{mapped}'")
        else:
            print(f"✓ OK: '{team}'")
        mapped_teams.append(mapped)
    
    print()
    print(f"Final mapped teams: {sorted(set(mapped_teams))}")
    
    # Test database query with mapped teams
    with get_connection() as conn:
        with conn.cursor() as cursor:
            team_placeholders = ','.join(['%s'] * len(mapped_teams))
            
            query = f"""
            SELECT DISTINCT team, COUNT(*) as record_count
            FROM teams_offense_daily 
            WHERE team IN ({team_placeholders})
            GROUP BY team
            ORDER BY team
            """
            
            cursor.execute(query, mapped_teams)
            results = cursor.fetchall()
            
            print(f"\n📊 DATABASE RESULTS:")
            print("=" * 40)
            for team, count in results:
                print(f"✅ {team}: {count} records")
            
            print(f"\nFound {len(results)} teams in database out of {len(mapped_teams)} requested")
            
            if len(results) == len(mapped_teams):
                print("🎉 ALL TEAMS FOUND! Team mapping is working correctly.")
                return True
            else:
                missing = set(mapped_teams) - {team for team, _ in results}
                print(f"❌ Missing teams: {missing}")
                return False

if __name__ == "__main__":
    success = test_team_mapping()
    if success:
        print("\n✅ Team mapping test PASSED - ready to update predictor")
    else:
        print("\n❌ Team mapping test FAILED - need to investigate further")
