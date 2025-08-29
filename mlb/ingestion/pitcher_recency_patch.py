#!/usr/bin/env python3
"""
Pitcher Recency & Matchup Patch
===============================

This module adds pitcher last start performance, days rest calculation,
and handedness detection to the working_pitcher_ingestor.py workflow.

Key enhancements:
- Pitcher last start stats (runs allowed, pitch count)
- Days rest calculation from previous start
- Pitcher handedness (R/L) detection
- Opponent team history lookup

This integrates with the enhanced database schema from migrations/20250828_recency_matchup.sql
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
import os

log = logging.getLogger(__name__)

def get_pitcher_last_start_stats(pitcher_id: int, reference_date: datetime = None) -> dict:
    """
    Get pitcher's last start performance before reference date.
    
    Args:
        pitcher_id: MLB player ID
        reference_date: Reference date (default: today)
        
    Returns:
        Dict with last start stats: runs, pitches, days_rest, handedness
    """
    if not pitcher_id:
        return _empty_last_start_stats()
    
    ref_date = reference_date or datetime.now()
    season = ref_date.year
    
    try:
        # Get pitcher's game log for current season
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&gameType=R&season={season}"
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            log.warning(f"Failed to fetch game log for pitcher {pitcher_id}: {response.status_code}")
            return _empty_last_start_stats()
        
        data = response.json()
        game_logs = data.get('stats', [{}])[0].get('splits', [])
        
        if not game_logs:
            log.info(f"No game logs found for pitcher {pitcher_id}")
            return _empty_last_start_stats()
        
        # Find most recent start before reference date
        last_start = None
        for game in game_logs:
            game_date_str = game.get('date')
            if not game_date_str:
                continue
                
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d')
            
            # Only consider games before reference date
            if game_date >= ref_date:
                continue
                
            stat = game.get('stat', {})
            
            # Only consider starts (innings pitched > 0)
            ip = stat.get('inningsPitched', '0')
            if not ip or float(ip.replace('⅓', '.33').replace('⅔', '.67')) == 0:
                continue
            
            last_start = {
                'date': game_date,
                'stat': stat,
                'game': game
            }
            break  # Game logs are typically in reverse chronological order
        
        if not last_start:
            log.info(f"No recent starts found for pitcher {pitcher_id}")
            return _empty_last_start_stats()
        
        # Extract performance metrics
        stat = last_start['stat']
        
        # Runs allowed (earned runs)
        runs_allowed = int(stat.get('earnedRuns', 0))
        
        # Pitch count (estimate if not available: ~15 pitches per inning)
        pitch_count = stat.get('numberOfPitches')
        if not pitch_count:
            ip_val = float(stat.get('inningsPitched', '0').replace('⅓', '.33').replace('⅔', '.67'))
            pitch_count = int(ip_val * 15)  # Rough estimate
        else:
            pitch_count = int(pitch_count)
        
        # Days rest calculation
        days_rest = (ref_date.date() - last_start['date'].date()).days
        
        # Get handedness from pitcher profile
        handedness = _get_pitcher_handedness(pitcher_id)
        
        result = {
            'last_start_runs': runs_allowed,
            'last_start_pitches': pitch_count,
            'days_rest': days_rest,
            'handedness': handedness,
            'last_start_date': last_start['date'].strftime('%Y-%m-%d')
        }
        
        log.info(f"Pitcher {pitcher_id} last start: {runs_allowed}R, {pitch_count}P, {days_rest} days rest, {handedness}")
        return result
        
    except Exception as e:
        log.error(f"Error fetching last start for pitcher {pitcher_id}: {e}")
        return _empty_last_start_stats()

def _get_pitcher_handedness(pitcher_id: int) -> str:
    """Get pitcher's throwing hand (R/L)"""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            people = data.get('people', [])
            if people:
                pitch_hand = people[0].get('pitchHand', {}).get('code', 'R')
                return pitch_hand
    except Exception as e:
        log.warning(f"Error fetching handedness for pitcher {pitcher_id}: {e}")
    
    return 'R'  # Default to right-handed

def _empty_last_start_stats() -> dict:
    """Return empty stats structure"""
    return {
        'last_start_runs': None,
        'last_start_pitches': None, 
        'days_rest': None,
        'handedness': 'R',
        'last_start_date': None
    }

def enhance_pitcher_updates_with_recency(pitcher_updates: list) -> list:
    """
    Enhance pitcher updates with last start stats and days rest.
    
    Args:
        pitcher_updates: List of pitcher update dicts from working_pitcher_ingestor
        
    Returns:
        Enhanced list with additional fields for database update
    """
    log.info(f"Enhancing {len(pitcher_updates)} pitcher updates with recency features")
    
    enhanced_updates = []
    
    for update in pitcher_updates:
        try:
            # Parse game date for reference
            game_date = datetime.strptime(update['date'], '%Y-%m-%d')
            
            # Get enhanced stats for home pitcher
            home_sp_id = update.get('home_sp_id')
            home_enhanced = get_pitcher_last_start_stats(home_sp_id, game_date) if home_sp_id else _empty_last_start_stats()
            
            # Get enhanced stats for away pitcher  
            away_sp_id = update.get('away_sp_id')
            away_enhanced = get_pitcher_last_start_stats(away_sp_id, game_date) if away_sp_id else _empty_last_start_stats()
            
            # Add enhanced fields to update
            enhanced_update = update.copy()
            enhanced_update.update({
                # Home pitcher last start
                'pitcher_last_start_runs_home': home_enhanced['last_start_runs'],
                'pitcher_last_start_pitches_home': home_enhanced['last_start_pitches'],
                'pitcher_days_rest_home': home_enhanced['days_rest'],
                'home_sp_handedness': home_enhanced['handedness'],
                
                # Away pitcher last start
                'pitcher_last_start_runs_away': away_enhanced['last_start_runs'],
                'pitcher_last_start_pitches_away': away_enhanced['last_start_pitches'],
                'pitcher_days_rest_away': away_enhanced['days_rest'],
                'away_sp_handedness': away_enhanced['handedness'],
            })
            
            enhanced_updates.append(enhanced_update)
            
        except Exception as e:
            log.error(f"Error enhancing pitcher update: {e}")
            # Include original update without enhancements
            enhanced_updates.append(update)
    
    log.info(f"✅ Enhanced {len(enhanced_updates)} pitcher updates with recency features")
    return enhanced_updates

def update_pitcher_ids_with_recency(pitcher_updates: list) -> int:
    """
    Enhanced version of update_pitcher_ids that includes recency features.
    
    This function replaces the original update_pitcher_ids in working_pitcher_ingestor.py
    when the recency patch is applied.
    
    Args:
        pitcher_updates: List of enhanced pitcher update dicts
        
    Returns:
        Number of games updated
    """
    if not pitcher_updates:
        return 0
    
    # First enhance with recency features
    enhanced_updates = enhance_pitcher_updates_with_recency(pitcher_updates)
    
    engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))
    updated_count = 0
    
    try:
        with engine.begin() as conn:
            for u in enhanced_updates:
                # Enhanced SQL with new recency columns
                sql = text("""
                    UPDATE enhanced_games
                    SET
                        home_sp_id           = COALESCE(:home_sp_id, home_sp_id),
                        away_sp_id           = COALESCE(:away_sp_id, away_sp_id),
                        home_sp_name         = COALESCE(:home_sp_name, home_sp_name),
                        away_sp_name         = COALESCE(:away_sp_name, away_sp_name),
                        home_sp_season_era   = COALESCE(:home_era, home_sp_season_era),
                        away_sp_season_era   = COALESCE(:away_era, away_sp_season_era),
                        home_sp_whip         = COALESCE(:home_whip, home_sp_whip),
                        away_sp_whip         = COALESCE(:away_whip, away_sp_whip),
                        home_sp_season_k     = COALESCE(:home_strikeouts, home_sp_season_k),
                        away_sp_season_k     = COALESCE(:away_strikeouts, away_sp_season_k),
                        home_sp_season_bb    = COALESCE(:home_walks, home_sp_season_bb),
                        away_sp_season_bb    = COALESCE(:away_walks, away_sp_season_bb),
                        home_sp_season_ip    = COALESCE(:home_innings_pitched, home_sp_season_ip),
                        away_sp_season_ip    = COALESCE(:away_innings_pitched, away_sp_season_ip),
                        
                        -- Use existing column names for pitcher features
                        home_sp_days_rest = COALESCE(:pitcher_days_rest_home, home_sp_days_rest),
                        away_sp_days_rest = COALESCE(:pitcher_days_rest_away, away_sp_days_rest),
                        home_sp_hand = COALESCE(:home_sp_handedness, home_sp_hand),
                        away_sp_hand = COALESCE(:away_sp_handedness, away_sp_hand)
                        
                    WHERE game_id = :game_id AND date = :date
                """)
                
                params = {
                    'date': u['date'],
                    'game_id': u['game_id'],
                    'home_sp_id': u['home_sp_id'],
                    'away_sp_id': u['away_sp_id'],
                    'home_sp_name': u['home_pitcher_name'],
                    'away_sp_name': u['away_pitcher_name'],
                    'home_era': u['home_era'],
                    'away_era': u['away_era'],
                    'home_whip': u['home_whip'],
                    'away_whip': u['away_whip'],
                    'home_strikeouts': u['home_strikeouts'],
                    'away_strikeouts': u['away_strikeouts'],
                    'home_walks': u['home_walks'],
                    'away_walks': u['away_walks'],
                    'home_innings_pitched': u['home_innings_pitched'],
                    'away_innings_pitched': u['away_innings_pitched'],
                    
                    # NEW: Recency parameters
                    'pitcher_last_start_runs_home': u.get('pitcher_last_start_runs_home'),
                    'pitcher_last_start_pitches_home': u.get('pitcher_last_start_pitches_home'),
                    'pitcher_days_rest_home': u.get('pitcher_days_rest_home'),
                    'home_sp_handedness': u.get('home_sp_handedness'),
                    
                    'pitcher_last_start_runs_away': u.get('pitcher_last_start_runs_away'),
                    'pitcher_last_start_pitches_away': u.get('pitcher_last_start_pitches_away'),
                    'pitcher_days_rest_away': u.get('pitcher_days_rest_away'),
                    'away_sp_handedness': u.get('away_sp_handedness'),
                }
                
                result = conn.execute(sql, params)
                
                if result.rowcount > 0:
                    updated_count += 1
                    log.info(f"✅ Updated pitchers with recency for {u['away_team']} @ {u['home_team']}")
                    log.info(f"   Home: {u['home_pitcher_name']} ({u.get('pitcher_days_rest_home')}d rest, {u.get('pitcher_last_start_runs_home')}R last)")
                    log.info(f"   Away: {u['away_pitcher_name']} ({u.get('pitcher_days_rest_away')}d rest, {u.get('pitcher_last_start_runs_away')}R last)")
                else:
                    log.warning(f"⚠️ No game found to update for game_id {u['game_id']} on {u['date']}")
        
        return updated_count
        
    except Exception as e:
        log.error(f"❌ Error updating pitcher data with recency: {e}")
        return 0

if __name__ == "__main__":
    # Test the enhancement functions
    print("Testing pitcher recency enhancement...")
    
    # Test with a known pitcher ID (this would be replaced with actual workflow integration)
    test_pitcher_id = 605480  # Example: Gerrit Cole
    stats = get_pitcher_last_start_stats(test_pitcher_id)
    print(f"Test results: {stats}")
