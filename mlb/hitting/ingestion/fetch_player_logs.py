"""
MLB Player Game Logs Fetcher

This module fetches daily player game logs from MLB API and populates
the player_game_logs table for hitting analysis.
"""

import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time

log = logging.getLogger(__name__)

class MLBPlayerLogsFetcher:
    """Fetches and stores player game logs from MLB API"""
    
    def __init__(self, engine):
        self.engine = engine
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def get_games_for_date(self, game_date: str) -> List[Dict]:
        """Get all games for a specific date"""
        url = f"{self.base_url}/schedule/games/?sportId=1&date={game_date}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for game in data.get('dates', [{}])[0].get('games', []):
                if game.get('status', {}).get('statusCode') in ['F', 'C', 'O']:  # Final, Completed, or Official
                    try:
                        # Handle different possible team data structures
                        home_team = game['teams']['home']['team']
                        away_team = game['teams']['away']['team']
                        
                        # Try different fields for team abbreviation
                        home_abbr = home_team.get('abbreviation') or home_team.get('teamCode') or home_team.get('fileCode', '')
                        away_abbr = away_team.get('abbreviation') or away_team.get('teamCode') or away_team.get('fileCode', '')
                        
                        games.append({
                            'game_id': game['gamePk'],
                            'home_team': home_abbr,
                            'away_team': away_abbr
                        })
                    except KeyError as e:
                        log.warning(f"Could not parse game {game.get('gamePk', 'unknown')}: {e}")
                        continue
            
            log.info(f"Found {len(games)} completed games for {game_date}")
            return games
            
        except requests.RequestException as e:
            log.error(f"Error fetching games for {game_date}: {e}")
            return []
    
    def get_boxscore_data(self, game_id: int) -> Dict:
        """Get detailed boxscore data for a game"""
        url = f"{self.base_url}/game/{game_id}/boxscore"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except requests.RequestException as e:
            log.error(f"Error fetching boxscore for game {game_id}: {e}")
            return {}
    
    def parse_player_stats(self, boxscore: Dict, game_info: Dict, game_date: str) -> List[Dict]:
        """Parse player hitting stats from boxscore"""
        
        if not boxscore:
            return []
        
        player_logs = []
        
        # Process both teams
        for side in ['home', 'away']:
            team_abbr = game_info[f'{side}_team']
            
            # Get opposing team info
            opposing_side = 'away' if side == 'home' else 'home'
            opposing_team = game_info[f'{opposing_side}_team']
            
            team_data = boxscore.get('teams', {}).get(side, {})
            batters = team_data.get('batters', [])
            
            # Get starting pitcher info
            opposing_pitchers = boxscore.get('teams', {}).get(opposing_side, {}).get('pitchers', [])
            starting_pitcher_id = None
            starting_pitcher_name = None
            starting_pitcher_hand = None
            
            if opposing_pitchers:
                # Starting pitcher is typically the first one
                sp_id = opposing_pitchers[0]
                sp_data = boxscore.get('teams', {}).get(opposing_side, {}).get('players', {}).get(f'ID{sp_id}', {})
                if sp_data:
                    starting_pitcher_id = sp_id
                    starting_pitcher_name = sp_data.get('person', {}).get('fullName')
                    starting_pitcher_hand = sp_data.get('person', {}).get('pitchHand', {}).get('code')
            
            # Process each batter
            for batter_id in batters:
                player_data = team_data.get('players', {}).get(f'ID{batter_id}', {})
                
                if not player_data:
                    continue
                    
                person = player_data.get('person', {})
                stats = player_data.get('stats', {}).get('batting', {})
                
                if not stats:
                    continue
                
                # Parse batting order position
                batting_order = player_data.get('battingOrder')
                lineup_spot = None
                if batting_order:
                    try:
                        # Batting order like "100" means 1st in lineup, "200" means 2nd, etc.
                        lineup_spot = int(str(batting_order)[0])
                    except (ValueError, IndexError):
                        pass
                
                # Calculate additional stats
                hits = stats.get('hits', 0)
                doubles = stats.get('doubles', 0)
                triples = stats.get('triples', 0)
                home_runs = stats.get('homeRuns', 0)
                at_bats = stats.get('atBats', 0)
                walks = stats.get('baseOnBalls', 0)
                hbp = stats.get('hitByPitch', 0)
                sac_flies = stats.get('sacFlies', 0)
                
                # Calculate singles
                singles = hits - doubles - triples - home_runs
                
                # Calculate total bases
                total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)
                
                # Calculate extra base hits
                extra_base_hits = doubles + triples + home_runs
                
                # Calculate rate stats
                batting_avg = hits / at_bats if at_bats > 0 else 0.000
                
                # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
                obp_denominator = at_bats + walks + hbp + sac_flies
                on_base_pct = (hits + walks + hbp) / obp_denominator if obp_denominator > 0 else 0.000
                
                # SLG = TB / AB
                slugging_pct = total_bases / at_bats if at_bats > 0 else 0.000
                
                # OPS = OBP + SLG
                ops = on_base_pct + slugging_pct
                
                player_log = {
                    'date': game_date,
                    'game_id': game_info['game_id'],
                    'player_id': batter_id,
                    'player_name': person.get('fullName', ''),
                    'team': team_abbr,
                    'opponent': opposing_team,
                    'home_away': 'H' if side == 'home' else 'A',
                    'lineup_spot': lineup_spot,
                    'starting_pitcher_id': starting_pitcher_id,
                    'starting_pitcher_name': starting_pitcher_name,
                    'starting_pitcher_hand': starting_pitcher_hand,
                    # Core hitting stats
                    'plate_appearances': stats.get('plateAppearances', 0),
                    'at_bats': at_bats,
                    'hits': hits,
                    'singles': singles,
                    'doubles': doubles,
                    'triples': triples,
                    'home_runs': home_runs,
                    'runs': stats.get('runs', 0),
                    'runs_batted_in': stats.get('rbi', 0),
                    # Plate discipline
                    'walks': walks,
                    'intentional_walks': stats.get('intentionalWalks', 0),
                    'strikeouts': stats.get('strikeOuts', 0),
                    # Base running
                    'sb': stats.get('stolenBases', 0),
                    'cs': stats.get('caughtStealing', 0),
                    # Advanced metrics
                    'total_bases': total_bases,
                    'extra_base_hits': extra_base_hits,
                    # Rate stats
                    'batting_avg': round(batting_avg, 3),
                    'on_base_pct': round(on_base_pct, 3),
                    'slugging_pct': round(slugging_pct, 3),
                    'ops': round(ops, 3),
                    # Contact quality (defaults for now)
                    'ground_balls': 0,
                    'fly_balls': 0,
                    'line_drives': 0,
                    'pop_ups': 0,
                    # Game situation
                    'left_on_base': stats.get('leftOnBase', 0),
                    'double_plays': stats.get('groundIntoDoublePlay', 0)
                }
                
                player_logs.append(player_log)
                
        return player_logs
    
    def save_player_logs(self, player_logs: List[Dict]) -> int:
        """Save player logs to database"""
        
        if not player_logs:
            return 0
        
        df = pd.DataFrame(player_logs)
        
        # Convert numeric columns
        numeric_cols = [
            'plate_appearances', 'at_bats', 'hits', 'singles', 'doubles', 'triples',
            'home_runs', 'runs', 'runs_batted_in', 'walks', 'intentional_walks', 
            'strikeouts', 'sb', 'cs', 'total_bases', 'extra_base_hits',
            'ground_balls', 'fly_balls', 'line_drives', 'pop_ups',
            'left_on_base', 'double_plays'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert rate stats
        rate_cols = ['batting_avg', 'on_base_pct', 'slugging_pct', 'ops']
        for col in rate_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.000)
        
        # Save to database using INSERT ON CONFLICT
        with self.engine.begin() as conn:
            # Create temp table
            df.to_sql('temp_player_logs', conn, if_exists='replace', index=False)
            
            # Upsert into main table with proper type casting
            upsert_query = text("""
                INSERT INTO player_game_logs (
                    date, game_id, player_id, player_name, team,
                    opponent, home_away, lineup_spot, starting_pitcher_id,
                    starting_pitcher_name, starting_pitcher_hand,
                    plate_appearances, at_bats, hits, singles, doubles, triples,
                    home_runs, runs, runs_batted_in, walks, intentional_walks,
                    strikeouts, sb, cs, total_bases, extra_base_hits,
                    batting_avg, on_base_pct, slugging_pct, ops,
                    ground_balls, fly_balls, line_drives, pop_ups,
                    left_on_base, double_plays
                )
                SELECT 
                    date::date, game_id, player_id::bigint, player_name, team,
                    opponent, home_away, lineup_spot::integer, starting_pitcher_id::bigint,
                    starting_pitcher_name, starting_pitcher_hand,
                    plate_appearances::integer, at_bats::integer, hits::integer, 
                    singles::integer, doubles::integer, triples::integer,
                    home_runs::integer, runs::integer, runs_batted_in::integer, 
                    walks::integer, intentional_walks::integer,
                    strikeouts::integer, sb::integer, cs::integer, 
                    total_bases::integer, extra_base_hits::integer,
                    batting_avg::numeric(4,3), on_base_pct::numeric(4,3), 
                    slugging_pct::numeric(4,3), ops::numeric(4,3),
                    ground_balls::integer, fly_balls::integer, line_drives::integer, 
                    pop_ups::integer, left_on_base::integer, double_plays::integer
                FROM temp_player_logs
                ON CONFLICT (player_id, game_id)
                DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    team = EXCLUDED.team,
                    opponent = EXCLUDED.opponent,
                    home_away = EXCLUDED.home_away,
                    lineup_spot = EXCLUDED.lineup_spot,
                    starting_pitcher_id = EXCLUDED.starting_pitcher_id,
                    starting_pitcher_name = EXCLUDED.starting_pitcher_name,
                    starting_pitcher_hand = EXCLUDED.starting_pitcher_hand,
                    plate_appearances = EXCLUDED.plate_appearances,
                    at_bats = EXCLUDED.at_bats,
                    hits = EXCLUDED.hits,
                    singles = EXCLUDED.singles,
                    doubles = EXCLUDED.doubles,
                    triples = EXCLUDED.triples,
                    home_runs = EXCLUDED.home_runs,
                    runs = EXCLUDED.runs,
                    runs_batted_in = EXCLUDED.runs_batted_in,
                    walks = EXCLUDED.walks,
                    intentional_walks = EXCLUDED.intentional_walks,
                    strikeouts = EXCLUDED.strikeouts,
                    sb = EXCLUDED.sb,
                    cs = EXCLUDED.cs,
                    total_bases = EXCLUDED.total_bases,
                    extra_base_hits = EXCLUDED.extra_base_hits,
                    batting_avg = EXCLUDED.batting_avg,
                    on_base_pct = EXCLUDED.on_base_pct,
                    slugging_pct = EXCLUDED.slugging_pct,
                    ops = EXCLUDED.ops,
                    ground_balls = EXCLUDED.ground_balls,
                    fly_balls = EXCLUDED.fly_balls,
                    line_drives = EXCLUDED.line_drives,
                    pop_ups = EXCLUDED.pop_ups,
                    left_on_base = EXCLUDED.left_on_base,
                    double_plays = EXCLUDED.double_plays,
                    updated_at = NOW()
            """)
            
            conn.execute(upsert_query)
            
            # Drop temp table
            conn.execute(text("DROP TABLE temp_player_logs"))
        
        log.info(f"Saved {len(df)} player game logs")
        return len(df)
    
    def fetch_date(self, target_date: str) -> int:
        """Fetch all player logs for a specific date"""
        
        log.info(f"Fetching player logs for {target_date}")
        
        # Get games for the date
        games = self.get_games_for_date(target_date)
        
        if not games:
            log.warning(f"No completed games found for {target_date}")
            return 0
        
        all_player_logs = []
        
        # Process each game
        for game in games:
            log.info(f"Processing game {game['game_id']}: {game['away_team']} @ {game['home_team']}")
            
            # Fetch boxscore
            boxscore = self.get_boxscore_data(game['game_id'])
            
            if not boxscore:
                log.warning(f"No boxscore data for game {game['game_id']}")
                continue
            
            # Parse player stats
            player_logs = self.parse_player_stats(boxscore, game, target_date)
            all_player_logs.extend(player_logs)
            
            # Be nice to the API
            time.sleep(0.5)
        
        # Save all logs
        total_saved = self.save_player_logs(all_player_logs)
        
        log.info(f"Completed fetching {target_date}: {len(games)} games, {total_saved} player logs")
        return total_saved
    
    def fetch_date_range(self, start_date: str, end_date: str) -> int:
        """Fetch player logs for a date range"""
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_logs = 0
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            logs_count = self.fetch_date(date_str)
            total_logs += logs_count
            
            current += timedelta(days=1)
            
            # Be nice to the API between dates
            time.sleep(1)
        
        log.info(f"Completed date range {start_date} to {end_date}: {total_logs} total logs")
        return total_logs
    
    def refresh_materialized_views(self):
        """Refresh materialized views after data updates"""
        
        log.info("Refreshing materialized views...")
        
        with self.engine.begin() as conn:
            conn.execute(text("SELECT refresh_hitting_views()"))
        
        log.info("Materialized views refreshed")

def main():
    """CLI interface for fetching player logs"""
    import argparse
    import sys
    from pathlib import Path
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description='Fetch MLB player game logs')
    parser.add_argument('--date', type=str, help='Single date to fetch (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range (YYYY-MM-DD)')
    parser.add_argument('--yesterday', action='store_true', help='Fetch yesterday\'s games')
    parser.add_argument('--refresh-views', action='store_true', help='Refresh materialized views after fetch')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load environment
    load_dotenv(Path(__file__).parent.parent / '.env')
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Initialize fetcher
    fetcher = MLBPlayerLogsFetcher(engine)
    
    # Determine what to fetch
    if args.yesterday:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Fetching yesterday's player logs: {target_date}")
        fetcher.fetch_date(target_date)
    elif args.date:
        print(f"Fetching player logs for: {args.date}")
        fetcher.fetch_date(args.date)
    elif args.start_date and args.end_date:
        print(f"Fetching player logs from {args.start_date} to {args.end_date}")
        fetcher.fetch_date_range(args.start_date, args.end_date)
    else:
        print("Please specify --date, --start-date/--end-date, or --yesterday")
        sys.exit(1)
    
    # Refresh views if requested
    if args.refresh_views:
        fetcher.refresh_materialized_views()

if __name__ == "__main__":
    main()
