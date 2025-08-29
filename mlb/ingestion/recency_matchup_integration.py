#!/usr/bin/env python3
"""
Recency & Matchup Feature Integration
====================================

This script integrates the pitcher and team recency patches into the existing workflow.
It provides utilities to apply the enhanced features to games and validate the results.

Usage:
    python recency_matchup_integration.py --date 2025-08-28 --apply-patches
    python recency_matchup_integration.py --validate-features
    python recency_matchup_integration.py --backfill-historical --days 30

Key features:
- Applies pitcher last start stats and days rest
- Applies team handedness splits and lineup composition  
- Applies bullpen quality proxies
- Validates feature completeness and quality
- Supports backfilling historical data
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from pitcher_recency_patch import (
    get_pitcher_last_start_stats, 
    enhance_pitcher_updates_with_recency,
    update_pitcher_ids_with_recency
)

from team_handedness_patch import (
    get_team_vs_handedness_stats,
    enhance_team_updates_with_handedness,
    update_team_handedness_features,
    apply_empirical_bayes_blending
)

from sqlalchemy import create_engine, text
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def apply_recency_features_to_date(target_date: str) -> dict:
    """
    Apply all recency and matchup features for a specific date.
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        
    Returns:
        Dict with results summary
    """
    log.info(f"🔬 Applying recency+matchup features for {target_date}")
    
    engine = get_engine()
    date_dt = datetime.strptime(target_date, '%Y-%m-%d')
    results = {
        'date': target_date,
        'games_processed': 0,
        'pitcher_updates': 0,
        'team_updates': 0,
        'errors': []
    }
    
    try:
        # Step 1: Get today's games from enhanced_games
        games_query = text("""
            SELECT DISTINCT 
                game_id, 
                date,
                home_team_id as home_team,
                away_team_id as away_team,
                home_sp_id,
                away_sp_id,
                home_sp_name,
                away_sp_name,
                total_runs,
                market_total
            FROM enhanced_games 
            WHERE date = :target_date
            ORDER BY game_id
        """)
        
        games_df = pd.read_sql(games_query, engine, params={'target_date': target_date})
        results['games_processed'] = len(games_df)
        
        if games_df.empty:
            log.warning(f"No games found for {target_date}")
            return results
        
        log.info(f"Found {len(games_df)} games to enhance for {target_date}")
        completed_games = games_df['total_runs'].notna().sum()
        future_games = games_df['total_runs'].isna().sum()
        log.info(f"  - {completed_games} completed games, {future_games} future games")
        
        # Step 2: Apply pitcher recency features
        if not games_df.empty:
            pitcher_updates = []
            
            for _, game in games_df.iterrows():
                try:
                    # Build pitcher update structure
                    pitcher_update = {
                        'date': target_date,
                        'game_id': game['game_id'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'home_sp_id': game['home_sp_id'],
                        'away_sp_id': game['away_sp_id'],
                        'home_pitcher_name': game['home_sp_name'],
                        'away_pitcher_name': game['away_sp_name'],
                        # These would normally come from pitcher ingestor
                        'home_era': None,
                        'away_era': None,
                        'home_whip': None,
                        'away_whip': None,
                        'home_strikeouts': None,
                        'away_strikeouts': None,
                        'home_walks': None,
                        'away_walks': None,
                        'home_innings_pitched': None,
                        'away_innings_pitched': None,
                    }
                    
                    pitcher_updates.append(pitcher_update)
                    
                except Exception as e:
                    log.error(f"Error processing game {game['game_id']}: {e}")
                    results['errors'].append(f"Game {game['game_id']}: {e}")
            
            # Apply pitcher recency enhancements
            if pitcher_updates:
                try:
                    updated_count = update_pitcher_ids_with_recency(pitcher_updates)
                    results['pitcher_updates'] = updated_count
                    log.info(f"✅ Applied pitcher recency features to {updated_count} games")
                except Exception as e:
                    log.error(f"❌ Error applying pitcher features: {e}")
                    results['errors'].append(f"Pitcher features: {e}")
        
        # Step 3: Apply team handedness features
        if not games_df.empty:
            team_updates = []
            
            for _, game in games_df.iterrows():
                team_update = {
                    'date': target_date,
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                }
                team_updates.append(team_update)
            
            # Apply team handedness enhancements
            if team_updates:
                try:
                    updated_count = update_team_handedness_features(team_updates)
                    results['team_updates'] = updated_count
                    log.info(f"✅ Applied team handedness features to {updated_count} games")
                except Exception as e:
                    log.error(f"❌ Error applying team features: {e}")
                    results['errors'].append(f"Team features: {e}")
        
        log.info(f"🎯 Recency feature application complete for {target_date}")
        log.info(f"   Games: {results['games_processed']}, Pitcher updates: {results['pitcher_updates']}, Team updates: {results['team_updates']}")
        
        return results
        
    except Exception as e:
        log.error(f"❌ Error applying recency features: {e}")
        results['errors'].append(f"General error: {e}")
        return results

def validate_feature_completeness(target_date: str = None) -> dict:
    """
    Validate that recency and matchup features are properly populated.
    
    Args:
        target_date: Optional date to validate (default: today)
        
    Returns:
        Dict with validation results
    """
    if not target_date:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    log.info(f"🔍 Validating feature completeness for {target_date}")
    
    engine = get_engine()
    validation_results = {
        'date': target_date,
        'total_games': 0,
        'pitcher_features_complete': 0,
        'team_features_complete': 0,
        'missing_features': [],
        'quality_issues': []
    }
    
    try:
        # Check feature completeness
        validation_query = text("""
            SELECT 
                game_id,
                date,
                home_team_id,
                away_team_id,
                
                -- Pitcher features
                pitcher_last_start_runs_home,
                pitcher_last_start_pitches_home,
                pitcher_days_rest_home,
                home_sp_handedness,
                pitcher_last_start_runs_away,
                pitcher_last_start_pitches_away,
                pitcher_days_rest_away,
                away_sp_handedness,
                
                -- Team handedness features
                team_wrc_plus_vs_rhp_7d_home,
                team_wrc_plus_vs_lhp_7d_home,
                team_wrc_plus_vs_rhp_30d_home,
                team_wrc_plus_vs_lhp_30d_home,
                lineup_r_pct_home,
                lineup_l_pct_home,
                bullpen_era_7d_home,
                bullpen_era_30d_home,
                
                team_wrc_plus_vs_rhp_7d_away,
                team_wrc_plus_vs_lhp_7d_away,
                team_wrc_plus_vs_rhp_30d_away,
                team_wrc_plus_vs_lhp_30d_away,
                lineup_r_pct_away,
                lineup_l_pct_away,
                bullpen_era_7d_away,
                bullpen_era_30d_away
                
            FROM enhanced_games 
            WHERE date = :target_date
            AND total_runs IS NULL  -- Only future games
        """)
        
        df = pd.read_sql(validation_query, engine, params={'target_date': target_date})
        validation_results['total_games'] = len(df)
        
        if df.empty:
            log.info(f"No games found for validation on {target_date}")
            return validation_results
        
        # Check pitcher feature completeness
        pitcher_features = [
            'pitcher_last_start_runs_home', 'pitcher_last_start_pitches_home', 'pitcher_days_rest_home',
            'pitcher_last_start_runs_away', 'pitcher_last_start_pitches_away', 'pitcher_days_rest_away'
        ]
        
        pitcher_complete = 0
        for _, row in df.iterrows():
            if all(pd.notna(row[col]) for col in pitcher_features):
                pitcher_complete += 1
        
        validation_results['pitcher_features_complete'] = pitcher_complete
        
        # Check team feature completeness
        team_features = [
            'team_wrc_plus_vs_rhp_7d_home', 'team_wrc_plus_vs_lhp_7d_home',
            'lineup_r_pct_home', 'bullpen_era_7d_home',
            'team_wrc_plus_vs_rhp_7d_away', 'team_wrc_plus_vs_lhp_7d_away',
            'lineup_r_pct_away', 'bullpen_era_7d_away'
        ]
        
        team_complete = 0
        for _, row in df.iterrows():
            if all(pd.notna(row[col]) for col in team_features):
                team_complete += 1
        
        validation_results['team_features_complete'] = team_complete
        
        # Check for missing features
        for feature in pitcher_features + team_features:
            missing_count = df[feature].isna().sum()
            if missing_count > 0:
                validation_results['missing_features'].append({
                    'feature': feature,
                    'missing_count': missing_count,
                    'missing_pct': round(missing_count / len(df) * 100, 1)
                })
        
        # Quality checks
        # Check for unrealistic values
        if 'team_wrc_plus_vs_rhp_7d_home' in df.columns:
            wrc_stats = df['team_wrc_plus_vs_rhp_7d_home'].dropna()
            if len(wrc_stats) > 0:
                if wrc_stats.min() < 50 or wrc_stats.max() > 200:
                    validation_results['quality_issues'].append("wRC+ values outside realistic range (50-200)")
        
        if 'pitcher_days_rest_home' in df.columns:
            rest_stats = df['pitcher_days_rest_home'].dropna()
            if len(rest_stats) > 0:
                if rest_stats.min() < 0 or rest_stats.max() > 10:
                    validation_results['quality_issues'].append("Days rest outside realistic range (0-10)")
        
        # Log validation results
        log.info(f"📊 Validation Results for {target_date}:")
        log.info(f"   Total games: {validation_results['total_games']}")
        log.info(f"   Pitcher features complete: {pitcher_complete}/{len(df)} ({pitcher_complete/len(df)*100:.1f}%)")
        log.info(f"   Team features complete: {team_complete}/{len(df)} ({team_complete/len(df)*100:.1f}%)")
        
        if validation_results['missing_features']:
            log.warning(f"   Missing features detected: {len(validation_results['missing_features'])}")
            for missing in validation_results['missing_features'][:5]:  # Show first 5
                log.warning(f"     {missing['feature']}: {missing['missing_count']} missing ({missing['missing_pct']}%)")
        
        if validation_results['quality_issues']:
            log.warning(f"   Quality issues: {validation_results['quality_issues']}")
        
        return validation_results
        
    except Exception as e:
        log.error(f"❌ Error validating features: {e}")
        validation_results['errors'] = [str(e)]
        return validation_results

def backfill_historical_features(days: int = 7) -> dict:
    """
    Backfill recency features for historical dates.
    
    Args:
        days: Number of days back to backfill
        
    Returns:
        Dict with backfill results
    """
    log.info(f"🔄 Backfilling recency features for last {days} days")
    
    backfill_results = {
        'days_processed': 0,
        'total_games': 0,
        'total_updates': 0,
        'errors': []
    }
    
    end_date = datetime.now()
    
    for i in range(days):
        target_date = (end_date - timedelta(days=i+1)).strftime('%Y-%m-%d')
        
        try:
            log.info(f"Processing {target_date}...")
            results = apply_recency_features_to_date(target_date)
            
            backfill_results['days_processed'] += 1
            backfill_results['total_games'] += results['games_processed']
            backfill_results['total_updates'] += results['pitcher_updates'] + results['team_updates']
            
            if results['errors']:
                backfill_results['errors'].extend([f"{target_date}: {e}" for e in results['errors']])
                
        except Exception as e:
            log.error(f"Error processing {target_date}: {e}")
            backfill_results['errors'].append(f"{target_date}: {e}")
    
    log.info(f"✅ Backfill complete: {backfill_results['days_processed']} days, {backfill_results['total_games']} games, {backfill_results['total_updates']} updates")
    return backfill_results

def main():
    parser = argparse.ArgumentParser(description='Apply recency and matchup features')
    parser.add_argument('--date', help='Target date (YYYY-MM-DD)', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--apply-patches', action='store_true', help='Apply all recency patches for target date')
    parser.add_argument('--validate-features', action='store_true', help='Validate feature completeness')
    parser.add_argument('--backfill-historical', action='store_true', help='Backfill historical data')
    parser.add_argument('--days', type=int, default=7, help='Number of days for backfill (default: 7)')
    
    args = parser.parse_args()
    
    if args.apply_patches:
        results = apply_recency_features_to_date(args.date)
        print(f"\n📊 Results for {args.date}:")
        print(f"   Games processed: {results['games_processed']}")
        print(f"   Pitcher updates: {results['pitcher_updates']}")
        print(f"   Team updates: {results['team_updates']}")
        if results['errors']:
            print(f"   Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"     {error}")
    
    elif args.validate_features:
        results = validate_feature_completeness(args.date)
        print(f"\n🔍 Validation Results for {args.date}:")
        print(f"   Total games: {results['total_games']}")
        print(f"   Pitcher features complete: {results['pitcher_features_complete']}")
        print(f"   Team features complete: {results['team_features_complete']}")
        if results['missing_features']:
            print(f"   Missing features: {len(results['missing_features'])}")
        if results['quality_issues']:
            print(f"   Quality issues: {len(results['quality_issues'])}")
    
    elif args.backfill_historical:
        results = backfill_historical_features(args.days)
        print(f"\n🔄 Backfill Results:")
        print(f"   Days processed: {results['days_processed']}")
        print(f"   Total games: {results['total_games']}")
        print(f"   Total updates: {results['total_updates']}")
        if results['errors']:
            print(f"   Errors: {len(results['errors'])}")
    
    else:
        print("Please specify an action: --apply-patches, --validate-features, or --backfill-historical")
        print("Use --help for more options")

if __name__ == "__main__":
    main()
