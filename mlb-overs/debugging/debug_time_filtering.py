#!/usr/bin/env python3
"""
Debug Game Time Filtering
=========================
Check if games are being filtered out due to time/date issues.
"""

import requests
import os
from datetime import datetime, timezone

def debug_time_filtering():
    """Debug time-based filtering that might exclude games"""
    
    # Get API games
    api_key = "e5f3d288beef9aa8b7c1f5604de3e5fe"
    url = f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={api_key}&regions=us&markets=totals&dateFormat=iso&oddsFormat=american'
    response = requests.get(url)
    api_games = response.json()

    target_date = "2025-08-20"
    target_dt = datetime.fromisoformat(f"{target_date}T00:00:00+00:00").date()
    now_utc = datetime.now(timezone.utc)
    include_live = False  # Same as default in market ingestor
    
    print("GAME TIME FILTERING DEBUG")
    print("=" * 40)
    print(f"Target date: {target_date}")
    print(f"Current time (UTC): {now_utc}")
    print(f"Include live games: {include_live}")
    print()

    passed_filters = 0
    skipped_live = 0
    skipped_date = 0
    skipped_no_time = 0
    
    for i, api_game in enumerate(api_games, 1):
        api_away = api_game['away_team']
        api_home = api_game['home_team']
        commence_iso = api_game.get("commence_time")
        
        print(f"{i:2d}. {api_away} @ {api_home}")
        print(f"    Commence time: {commence_iso}")
        
        # Parse time (same logic as market ingestor)
        try:
            ct = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            print(f"    Parsed time (UTC): {ct}")
            print(f"    Game date: {ct.date()}")
        except Exception as e:
            ct = None
            print(f"    ❌ Time parsing failed: {e}")
            skipped_no_time += 1
            continue
        
        # Check date filter
        if not ct or ct.date() != target_dt:
            print(f"    ❌ FILTERED: Wrong date (expected {target_dt}, got {ct.date() if ct else 'None'})")
            skipped_date += 1
            continue
        
        # Check live filter
        is_live = ct and ct <= now_utc
        print(f"    Is live: {is_live} (game time <= current time)")
        
        if is_live and not include_live:
            print(f"    ❌ FILTERED: Live game (pregame only)")
            skipped_live += 1
            continue
        
        print(f"    ✅ PASSED all time filters")
        passed_filters += 1
        print()
    
    print(f"SUMMARY:")
    print(f"  Passed all filters: {passed_filters}")
    print(f"  Skipped (live): {skipped_live}")
    print(f"  Skipped (wrong date): {skipped_date}")
    print(f"  Skipped (no time): {skipped_no_time}")
    print(f"  Total games: {len(api_games)}")

if __name__ == "__main__":
    debug_time_filtering()
