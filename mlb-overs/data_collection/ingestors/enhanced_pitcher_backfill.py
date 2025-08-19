#!/usr/bin/env python3
"""
Enhanced Pitcher Backfill System
Fetches comprehensive historical pitcher data including:
- Season-to-date ERA, IP, ER
- Rolling 3/5/10 game ERA windows  
- vs Team performance history
- Complete pitcher performance profiles

This replaces the basic backfill with a comprehensive historical data injection system.
"""

import pandas as pd
import numpy as np
import statsapi
from sqlalchemy import create_engine, text
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database connection
DB_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

def get_connection():
    """Get database connection"""
    return create_engine(DB_URL)

def innings_to_float(ip) -> float | None:
    """Convert innings pitched string to float"""
    if ip is None: 
        return None
    s = str(ip).strip()
    if not s: 
        return None
    # handle formats like "5.1" (5 and 1/3 innings)
    if "." in s:
        try:
            whole, frac = s.split(".")
            return float(whole) + float(frac) / 3.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def team_key(team_id: int) -> str:
    """Get team key from team ID"""
    try:
        rec = (statsapi.lookup_team(team_id) or [])[0]
        return rec.get('teamName') or rec.get('name') or str(team_id)
    except Exception:
        return str(team_id)

def fetch_teams_any(game_pk: int) -> tuple[dict, dict | None]:
    """
    Return (teams_dict, live) where teams_dict looks like:
    {"home": {"team": {...}, "players": {...}}, "away": {...}}
    live is the liveFeed payload if we needed it (else None).
    """
    # try 1: boxscore_data
    try:
        box = statsapi.boxscore_data(game_pk) or {}
        teams = box.get("teams") or {}
        if teams:
            return teams, None
    except Exception:
        pass

    # try 2: raw boxscore
    try:
        raw = statsapi.get("game_boxscore", {"gamePk": game_pk}) or {}
        teams = raw.get("teams") or {}
        if teams:
            return teams, None
    except Exception:
        pass

    # try 3: liveFeed (most reliable)
    live = None
    try:
        live = statsapi.get("game", {"gamePk": game_pk}) or {}
        live_teams = (((live.get("liveData") or {}).get("boxscore") or {}).get("teams") or {})
        if live_teams:
            # synthesize the same structure: include players dicts
            # team ids/names live under gameData.teams
            gmeta = statsapi.get("game", {"gamePk": game_pk}) or {}
            game_teams = ((gmeta.get("gameData") or {}).get("teams") or {})
            return {
                "home": {
                    "team": game_teams.get("home", {}),
                    "players": (live_teams.get("home", {}) or {}).get("players", {}) or {}
                },
                "away": {
                    "team": game_teams.get("away", {}),
                    "players": (live_teams.get("away", {}) or {}).get("players", {}) or {}
                }
            }, live
    except Exception:
        pass

    return {}, None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_ip_er(pitching_dict):
    """Extract innings pitched and earned runs from statsapi pitching data"""
    if not pitching_dict:
        return None, None
    
    try:
        ip_str = pitching_dict.get("inningsPitched", "0.0")
        er_str = pitching_dict.get("earnedRuns", "0")
        
        # Handle fractional innings (e.g., "5.1" = 5 and 1/3 innings, "5.2" = 5 and 2/3 innings)
        if "." in str(ip_str):
            whole, frac = str(ip_str).split(".")
            frac_int = int(frac) if frac else 0
            # Convert baseball innings format: .1 = 1/3, .2 = 2/3
            ip_float = float(whole) + (frac_int / 3.0)
        else:
            ip_float = float(ip_str)
        
        er_int = int(er_str) if er_str else 0
        
        return ip_float, er_int
    except (ValueError, AttributeError):
        return None, None

def get_pitcher_season_stats(pitcher_id, season_year):
    """Fetch season-to-date stats for a pitcher"""
    try:
        # Get pitcher season stats
        stats = statsapi.player_stat_data(pitcher_id, group="[pitching]", type="season")
        
        season_stats = None
        for stat_group in stats.get("stats", []):
            if (stat_group.get("group") == "pitching" and 
                stat_group.get("season") == str(season_year) and
                stat_group.get("type") == "season"):
                season_stats = stat_group.get("stats", {})
                break
        
        if not season_stats:
            return None
        
        # Extract key metrics
        ip_str = season_stats.get("inningsPitched", "0.0")
        if "." in str(ip_str):
            whole, frac = str(ip_str).split(".")
            ip_season = float(whole) + float(frac) / 3.0
        else:
            ip_season = float(ip_str)
        
        er_season = int(season_stats.get("earnedRuns", 0))
        era_season = float(season_stats.get("era", 0.0)) if season_stats.get("era") else None
        
        return {
            "ip_season": ip_season,
            "er_season": er_season, 
            "era_season": era_season,
            "wins": int(season_stats.get("wins", 0)),
            "losses": int(season_stats.get("losses", 0)),
            "games_started": int(season_stats.get("gamesStarted", 0)),
            "strikeouts": int(season_stats.get("strikeOuts", 0)),
            "walks": int(season_stats.get("baseOnBalls", 0)),
            "hits_allowed": int(season_stats.get("hits", 0)),
            "whip": float(season_stats.get("whip", 0.0)) if season_stats.get("whip") else None
        }
    except Exception as e:
        logger.warning(f"Failed to get season stats for pitcher {pitcher_id}: {e}")
        return None

def calculate_rolling_eras(pitcher_games_df):
    """Calculate rolling ERA windows (L3, L5, L10) from pitcher game history"""
    if pitcher_games_df.empty:
        return {}
    
    # Sort by date descending (most recent first)  
    pitcher_games_df = pitcher_games_df.sort_values('date', ascending=False).reset_index(drop=True)
    
    rolling_eras = {}
    
    for window in [3, 5, 10]:
        window_games = pitcher_games_df.head(window)
        if len(window_games) >= min(window, 2):  # Need at least 2 games for meaningful ERA
            total_ip = window_games['ip'].sum()
            total_er = window_games['er'].sum()
            
            if total_ip > 0:
                rolling_era = (total_er * 9.0) / total_ip
                rolling_eras[f"era_l{window}"] = round(rolling_era, 2)
            else:
                rolling_eras[f"era_l{window}"] = None
        else:
            rolling_eras[f"era_l{window}"] = None
    
    return rolling_eras

def get_pitcher_vs_team_history(pitcher_id, vs_team_id, engine):
    """Get pitcher's historical performance vs specific team"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT ip, er, era_game, date
                FROM pitchers_starts 
                WHERE pitcher_id = :pid AND opp_team = :team
                AND ip IS NOT NULL AND er IS NOT NULL
                ORDER BY date DESC
                LIMIT 10
            """)
            
            df = pd.read_sql(query, conn, params={
                "pid": str(pitcher_id),
                "team": vs_team_id
            })
            
            if df.empty:
                return None
            
            total_ip = df['ip'].sum()
            total_er = df['er'].sum()
            
            if total_ip > 0:
                vs_team_era = (total_er * 9.0) / total_ip
                return {
                    "vs_team_era": round(vs_team_era, 2),
                    "vs_team_games": len(df),
                    "vs_team_ip": round(total_ip, 1),
                    "vs_team_er": int(total_er)
                }
            
    except Exception as e:
        logger.warning(f"Failed to get vs team history for pitcher {pitcher_id} vs {vs_team_id}: {e}")
    
    return None

def process_pitcher_comprehensive(pitcher_id, season_year, engine):
    """Get comprehensive pitcher data including season stats and rolling windows"""
    try:
        # Get season stats from MLB API
        season_stats = get_pitcher_season_stats(pitcher_id, season_year)
        
        # Get recent game history for rolling calculations
        with engine.connect() as conn:
            game_history_query = text("""
                SELECT ip, er, era_game, date, opp_team
                FROM pitchers_starts 
                WHERE pitcher_id = :pid
                AND ip IS NOT NULL AND er IS NOT NULL
                AND date >= :start_date
                ORDER BY date DESC
                LIMIT 20
            """)
            
            start_date = f"{season_year}-01-01"
            game_history = pd.read_sql(game_history_query, conn, params={
                "pid": str(pitcher_id),
                "start_date": start_date
            })
        
        # Calculate rolling ERAs
        rolling_stats = calculate_rolling_eras(game_history)
        
        # Combine all stats
        comprehensive_stats = {
            "pitcher_id": str(pitcher_id),
            "season_year": season_year,
            "last_updated": datetime.now().date()
        }
        
        if season_stats:
            comprehensive_stats.update(season_stats)
        
        comprehensive_stats.update(rolling_stats)
        
        return comprehensive_stats
    
    except Exception as e:
        logger.error(f"Failed to process comprehensive data for pitcher {pitcher_id}: {e}")
        return None

def create_pitcher_comprehensive_table(engine):
    """Create table for comprehensive pitcher stats if it doesn't exist"""
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS pitcher_comprehensive_stats (
            pitcher_id TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            ip_season REAL,
            er_season INTEGER,
            era_season REAL,
            era_l3 REAL,
            era_l5 REAL,
            era_l10 REAL,
            wins INTEGER,
            losses INTEGER,
            games_started INTEGER,
            strikeouts INTEGER,
            walks INTEGER,
            hits_allowed INTEGER,
            whip REAL,
            last_updated DATE,
            PRIMARY KEY (pitcher_id, season_year)
        )
    """)
    
    with engine.begin() as conn:
        conn.execute(create_table_sql)
    
    logger.info("Created/verified pitcher_comprehensive_stats table")

def backfill_comprehensive_pitcher_data(start_date, end_date, season_year=None, target_pitcher_ids=None):
    """Enhanced backfill with comprehensive pitcher statistics"""
    
    if target_pitcher_ids:
        logger.info(f"Starting targeted pitcher backfill for {len(target_pitcher_ids)} pitchers from {start_date} to {end_date}")
    else:
        logger.info(f"Starting comprehensive pitcher backfill from {start_date} to {end_date}")
    
    if season_year is None:
        season_year = datetime.strptime(start_date, "%Y-%m-%d").year
    
    eng = get_connection()
    
    # Create comprehensive stats table
    create_pitcher_comprehensive_table(eng)
    
    # Get games in date range
    with eng.connect() as cx:
        games = pd.read_sql(text("""
            SELECT DISTINCT game_id, date
            FROM games
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC
        """), cx, params={"s": start_date, "e": end_date})

    if games.empty:
        logger.warning("No games found in specified date range")
        return

    logger.info(f"Processing {len(games)} games for comprehensive pitcher data")
    
    # First, run basic game-level backfill (reuse existing logic)
    basic_rows = []
    pitcher_ids_found = set()
    
    pbar = tqdm(total=len(games), desc="Processing games for pitcher data", unit="game")
    
    for _, game_row in games.iterrows():
        gid = int(game_row["game_id"])
        game_date = game_row["date"]
        
        try:
            # Check if game is final
            gmeta = statsapi.get("game", {"gamePk": gid}) or {}
            status = ((gmeta.get("gameData") or {}).get("status") or {}).get("abstractGameState")
            if status not in ("Final", "Completed Early", "Game Over"):
                pbar.update(1)
                continue
        except Exception:
            pbar.update(1)
            continue
        
        teams, live = fetch_teams_any(gid)
        if not teams:
            pbar.update(1)
            continue
        
        # Process pitchers from both teams
        for side in ("home", "away"):
            t = teams.get(side) or {}
            players = t.get("players") or {}
            team_id = (t.get("team") or {}).get("id")
            opp_side = "home" if side == "away" else "away"
            opp_team_id = (teams.get(opp_side) or {}).get("team", {}).get("id")

            for pid_key, pdata in players.items():
                pos = (pdata.get("position") or {}).get("abbreviation")
                if pos != "P":
                    continue

                pitching = (pdata.get("stats") or {}).get("pitching") or {}
                ip_f, er_v = extract_ip_er(pitching)
                
                # Try fallback methods for IP/ER
                if ip_f is None and er_v is None:
                    try:
                        raw = statsapi.get("boxscore", {"gamePk": gid}) or {}
                        p2 = (raw.get("teams", {}).get(side, {}).get("players", {}) or {}).get(pid_key, {})
                        pitching2 = (p2.get("stats") or {}).get("pitching") or {}
                        ip_f, er_v = extract_ip_er(pitching2)
                    except Exception:
                        pass

                if ip_f is None and er_v is None:
                    continue

                era_game = float(9.0 * er_v / ip_f) if (ip_f and ip_f > 0 and er_v is not None) else None
                pid = (pdata.get("person") or {}).get("id")
                
                if pid:
                    # If target pitcher IDs specified, only process those pitchers
                    if target_pitcher_ids and pid not in target_pitcher_ids:
                        continue
                        
                    pitcher_ids_found.add(pid)
                    
                    basic_rows.append({
                        "start_id": f"{gid}_{pid}",
                        "game_id": gid,
                        "pitcher_id": pid,
                        "team": team_key(team_id) if team_id else None,
                        "opp_team": team_key(opp_team_id) if opp_team_id else None,
                        "is_home": (side == "home"),
                        "date": pd.to_datetime(game_date).date(),
                        "ip": ip_f,
                        "er": er_v,
                        "era_game": era_game,
                    })
        
        pbar.update(1)
    
    pbar.close()
    
    # Insert basic game data first
    if basic_rows:
        logger.info(f"Inserting {len(basic_rows)} basic pitcher game records")
        df_basic = pd.DataFrame(basic_rows).drop_duplicates(["start_id"])
        
        with eng.begin() as cx:
            df_basic.to_sql("tmp_ps_enhanced", cx, index=False, if_exists="replace")
            cx.execute(text("""
                INSERT INTO pitchers_starts (start_id, game_id, pitcher_id, team, opp_team, is_home, date, ip, er, era_game)
                SELECT start_id, game_id, CAST(pitcher_id AS TEXT), team, opp_team, is_home, date, ip, er, era_game
                FROM tmp_ps_enhanced
                ON CONFLICT (start_id) DO UPDATE SET
                  team = EXCLUDED.team,
                  opp_team = EXCLUDED.opp_team,
                  is_home = EXCLUDED.is_home,
                  ip = COALESCE(EXCLUDED.ip, pitchers_starts.ip),
                  er = COALESCE(EXCLUDED.er, pitchers_starts.er),
                  era_game = COALESCE(EXCLUDED.era_game, pitchers_starts.era_game)
            """))
            cx.execute(text("DROP TABLE tmp_ps_enhanced"))
        
        logger.info(f"Successfully inserted basic pitcher data for {len(df_basic)} records")
    
    # Now process comprehensive stats for each unique pitcher
    # Add target pitcher IDs even if not found in games (for targeted backfill)
    if target_pitcher_ids:
        pitcher_ids_found.update(target_pitcher_ids)
        logger.info(f"Added {len(target_pitcher_ids)} target pitcher IDs to processing queue")
    
    logger.info(f"Processing comprehensive stats for {len(pitcher_ids_found)} unique pitchers")
    
    comprehensive_rows = []
    comp_pbar = tqdm(total=len(pitcher_ids_found), desc="Fetching comprehensive pitcher stats", unit="pitcher")
    
    for pitcher_id in pitcher_ids_found:
        try:
            comp_stats = process_pitcher_comprehensive(pitcher_id, season_year, eng)
            if comp_stats:
                comprehensive_rows.append(comp_stats)
        except Exception as e:
            logger.warning(f"Failed to process comprehensive stats for pitcher {pitcher_id}: {e}")
        
        comp_pbar.update(1)
        time.sleep(0.1)  # Rate limiting for MLB API
    
    comp_pbar.close()
    
    # Insert comprehensive stats
    if comprehensive_rows:
        logger.info(f"Inserting comprehensive stats for {len(comprehensive_rows)} pitchers")
        df_comp = pd.DataFrame(comprehensive_rows)
        
        # Use direct pandas to_sql with replace_all for conflict resolution
        logger.info(f"Inserting comprehensive stats for {len(df_comp)} pitchers using direct insert")
        
        try:
            # First, remove any existing records for these pitchers and season
            pitcher_ids_str = ','.join([f"'{pid}'" for pid in df_comp['pitcher_id'].unique()])
            season_year = df_comp['season_year'].iloc[0] if len(df_comp) > 0 else 2024
            
            with eng.begin() as cx:
                cx.execute(text(f"""
                    DELETE FROM pitcher_comprehensive_stats 
                    WHERE pitcher_id IN ({pitcher_ids_str}) 
                    AND season_year = {season_year}
                """))
                
                # Insert new records
                df_comp.to_sql(
                    "pitcher_comprehensive_stats", 
                    cx, 
                    index=False, 
                    if_exists="append",
                    method='multi'
                )
            
            logger.info(f"Successfully inserted comprehensive stats for {len(df_comp)} pitchers")
            
        except Exception as e:
            logger.error(f"Failed to insert comprehensive stats: {e}")
            # Fallback: try individual inserts
            logger.info("Attempting individual record insertion as fallback...")
            
            success_count = 0
            for _, row in df_comp.iterrows():
                try:
                    with eng.begin() as cx:
                        cx.execute(text("""
                            INSERT INTO pitcher_comprehensive_stats (
                                pitcher_id, season_year, ip_season, er_season, era_season,
                                era_l3, era_l5, era_l10, wins, losses, games_started,
                                strikeouts, walks, hits_allowed, whip, last_updated
                            ) VALUES (
                                :pitcher_id, :season_year, :ip_season, :er_season, :era_season,
                                :era_l3, :era_l5, :era_l10, :wins, :losses, :games_started,
                                :strikeouts, :walks, :hits_allowed, :whip, :last_updated
                            )
                            ON CONFLICT (pitcher_id, season_year) DO UPDATE SET
                                ip_season = EXCLUDED.ip_season,
                                er_season = EXCLUDED.er_season,
                                era_season = EXCLUDED.era_season,
                                era_l3 = EXCLUDED.era_l3,
                                era_l5 = EXCLUDED.era_l5,
                                era_l10 = EXCLUDED.era_l10,
                                wins = EXCLUDED.wins,
                                losses = EXCLUDED.losses,
                                games_started = EXCLUDED.games_started,
                                strikeouts = EXCLUDED.strikeouts,
                                walks = EXCLUDED.walks,
                                hits_allowed = EXCLUDED.hits_allowed,
                                whip = EXCLUDED.whip,
                                last_updated = EXCLUDED.last_updated
                        """), row.to_dict())
                        success_count += 1
                except Exception as row_e:
                    logger.warning(f"Failed to insert pitcher {row['pitcher_id']}: {row_e}")
            
            logger.info(f"Fallback insertion completed: {success_count}/{len(df_comp)} records")
        
    else:
        logger.info("No comprehensive stats to insert")
    
    logger.info("Enhanced pitcher backfill completed successfully!")
    
    # Print summary statistics
    with eng.connect() as cx:
        game_count = pd.read_sql(text("SELECT COUNT(*) as cnt FROM pitchers_starts"), cx).iloc[0]['cnt']
        comp_count = pd.read_sql(text("SELECT COUNT(*) as cnt FROM pitcher_comprehensive_stats"), cx).iloc[0]['cnt']
        
        logger.info(f"Total pitcher game records: {game_count}")
        logger.info(f"Total comprehensive pitcher profiles: {comp_count}")

def update_pitcher_api_helpers(engine):
    """Update the API helper functions to use comprehensive stats table"""
    
    # Test query to validate comprehensive data availability
    with engine.connect() as cx:
        test_query = text("""
            SELECT pitcher_id, era_season, era_l3, era_l5, era_l10
            FROM pitcher_comprehensive_stats 
            WHERE era_season IS NOT NULL
            LIMIT 5
        """)
        
        test_results = pd.read_sql(test_query, cx)
        if not test_results.empty:
            logger.info("Comprehensive pitcher data successfully populated:")
            for _, row in test_results.iterrows():
                logger.info(f"  Pitcher {row['pitcher_id']}: Season ERA {row['era_season']}, L3: {row['era_l3']}, L5: {row['era_l5']}, L10: {row['era_l10']}")
        else:
            logger.warning("No comprehensive pitcher data found in database")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Pitcher Backfill with Comprehensive Stats")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, help="Season year (default: inferred from start date)")
    parser.add_argument("--pitcher-ids", help="Comma-separated list of pitcher IDs to target (optional)")
    
    args = parser.parse_args()
    
    # Parse pitcher IDs if provided
    target_pitcher_ids = None
    if args.pitcher_ids:
        try:
            target_pitcher_ids = set(int(pid.strip()) for pid in args.pitcher_ids.split(','))
            print(f"Targeting {len(target_pitcher_ids)} specific pitchers: {sorted(target_pitcher_ids)}")
        except ValueError as e:
            print(f"Error parsing pitcher IDs: {e}")
            exit(1)
    
    try:
        backfill_comprehensive_pitcher_data(args.start, args.end, args.season, target_pitcher_ids)
        
        # Test the comprehensive data
        eng = get_connection()
        update_pitcher_api_helpers(eng)
        
    except Exception as e:
        logger.error(f"Enhanced backfill failed: {e}")
        raise
