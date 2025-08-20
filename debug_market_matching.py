#!/usr/bin/env python3
"""
Debug Market Matching Logic
===========================
Test the exact team matching logic to see why we're missing 4 games.
"""

import requests
import psycopg2
import os
from datetime import datetime

def debug_team_matching():
    """Debug why only 10/14 games are matching"""
    
    # Get API games
    api_key = "e5f3d288beef9aa8b7c1f5604de3e5fe"
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=totals&dateFormat=iso&oddsFormat=american'
    response = requests.get(url)
    api_games = response.json()

    # Get database games  
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )
    cursor = conn.cursor()
    cursor.execute('SELECT game_id, home_team, away_team FROM enhanced_games WHERE date = %s ORDER BY home_team', ('2025-08-20',))
    db_games = cursor.fetchall()
    conn.close()

    print("DETAILED TEAM MATCHING DEBUG")
    print("=" * 50)
    print()

    matches_found = 0
    
    for i, api_game in enumerate(api_games, 1):
        api_away = api_game['away_team']
        api_home = api_game['home_team']
        
        print(f"{i:2d}. API: {api_away} @ {api_home}")
        
        # Test matching logic (same as real_market_ingestor.py)
        team_matches = []
        for game_id, db_home, db_away in db_games:
            # This is the exact logic from real_market_ingestor.py line 146-148
            home_match = (db_home in api_home or api_home in db_home)
            away_match = (db_away in api_away or api_away in db_away)
            
            if home_match and away_match:
                team_matches.append((game_id, db_home, db_away))
                print(f"    ✅ MATCH: {db_away} @ {db_home} (game_id: {game_id})")
        
        if not team_matches:
            print(f"    ❌ NO MATCH FOUND!")
            # Show what we tried to match against
            print(f"       Looking for API teams in database:")
            for game_id, db_home, db_away in db_games:
                home_check = f"'{db_home}' in '{api_home}' = {db_home in api_home} | '{api_home}' in '{db_home}' = {api_home in db_home}"
                away_check = f"'{db_away}' in '{api_away}' = {db_away in api_away} | '{api_away}' in '{db_away}' = {api_away in db_away}"
                print(f"       DB: {db_away} @ {db_home}")
                print(f"         Home: {home_check}")
                print(f"         Away: {away_check}")
        else:
            matches_found += 1
        
        print()
    
    print(f"SUMMARY: {matches_found}/14 games matched")
    print(f"Missing: {14 - matches_found} games")

if __name__ == "__main__":
    debug_team_matching()
