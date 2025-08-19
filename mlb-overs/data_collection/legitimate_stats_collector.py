#!/usr/bin/env python3
"""
MLB Season Statistics Collector
===============================
Collects legitimate pre-game pitcher and team statistics for model training
"""

import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBStatsCollector:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.db_path = "S:/Projects/AI_Predictions/mlb-overs/data/legitimate_stats.db"
        self.session = requests.Session()
        
        # Create database if it doesn't exist
        self.init_database()
    
    def init_database(self):
        """Initialize the legitimate stats database"""
        logger.info("Initializing legitimate stats database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pitcher season stats table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_season_stats (
            pitcher_id INTEGER,
            season INTEGER,
            date TEXT,
            games_played INTEGER,
            games_started INTEGER,
            innings_pitched REAL,
            earned_runs INTEGER,
            hits_allowed INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            home_runs_allowed INTEGER,
            era REAL,
            whip REAL,
            k_per_9 REAL,
            bb_per_9 REAL,
            updated_at TEXT,
            PRIMARY KEY (pitcher_id, date)
        )
        """)
        
        # Team season stats table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_season_stats (
            team_id INTEGER,
            season INTEGER,
            date TEXT,
            games_played INTEGER,
            runs INTEGER,
            hits INTEGER,
            doubles INTEGER,
            triples INTEGER,
            home_runs INTEGER,
            rbi INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            batting_avg REAL,
            on_base_pct REAL,
            slugging_pct REAL,
            ops REAL,
            runs_per_game REAL,
            updated_at TEXT,
            PRIMARY KEY (team_id, date)
        )
        """)
        
        # Game matchups table (for prediction)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_matchups (
            game_id INTEGER PRIMARY KEY,
            date TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_pitcher_id INTEGER,
            away_pitcher_id INTEGER,
            venue_id INTEGER,
            game_time_et TEXT,
            created_at TEXT
        )
        """)
        
        # Ballpark factors table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ballpark_factors (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT,
            run_factor REAL,
            hr_factor REAL,
            elevation INTEGER,
            foul_territory TEXT,
            dimensions_lf INTEGER,
            dimensions_cf INTEGER,
            dimensions_rf INTEGER
        )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def get_current_season_stats(self, pitcher_id, as_of_date):
        """Get pitcher's season stats as of a specific date"""
        try:
            # Get pitcher stats for current season
            season = int(as_of_date[:4])
            url = f"{self.base_url}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'pitching'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to get stats for pitcher {pitcher_id}: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('stats') or not data['stats'][0].get('splits'):
                return None
            
            stats = data['stats'][0]['splits'][0]['stat']
            
            # Calculate derived stats
            innings_pitched = float(stats.get('inningsPitched', '0'))
            earned_runs = int(stats.get('earnedRuns', 0))
            hits_allowed = int(stats.get('hits', 0))
            walks = int(stats.get('baseOnBalls', 0))
            strikeouts = int(stats.get('strikeOuts', 0))
            
            era = float(stats.get('era', 0))
            whip = (hits_allowed + walks) / max(innings_pitched, 0.1)
            k_per_9 = (strikeouts * 9) / max(innings_pitched, 0.1)
            bb_per_9 = (walks * 9) / max(innings_pitched, 0.1)
            
            return {
                'pitcher_id': pitcher_id,
                'season': season,
                'date': as_of_date,
                'games_played': int(stats.get('gamesPlayed', 0)),
                'games_started': int(stats.get('gamesStarted', 0)),
                'innings_pitched': innings_pitched,
                'earned_runs': earned_runs,
                'hits_allowed': hits_allowed,
                'walks': walks,
                'strikeouts': strikeouts,
                'home_runs_allowed': int(stats.get('homeRuns', 0)),
                'era': era,
                'whip': round(whip, 3),
                'k_per_9': round(k_per_9, 2),
                'bb_per_9': round(bb_per_9, 2),
                'updated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pitcher stats for {pitcher_id}: {e}")
            return None
    
    def get_team_season_stats(self, team_id, as_of_date):
        """Get team's season offensive stats as of a specific date"""
        try:
            season = int(as_of_date[:4])
            url = f"{self.base_url}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'hitting'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to get team stats for {team_id}: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('stats') or not data['stats'][0].get('splits'):
                return None
            
            stats = data['stats'][0]['splits'][0]['stat']
            
            games_played = int(stats.get('gamesPlayed', 1))
            runs = int(stats.get('runs', 0))
            runs_per_game = runs / max(games_played, 1)
            
            return {
                'team_id': team_id,
                'season': season,
                'date': as_of_date,
                'games_played': games_played,
                'runs': runs,
                'hits': int(stats.get('hits', 0)),
                'doubles': int(stats.get('doubles', 0)),
                'triples': int(stats.get('triples', 0)),
                'home_runs': int(stats.get('homeRuns', 0)),
                'rbi': int(stats.get('rbi', 0)),
                'walks': int(stats.get('baseOnBalls', 0)),
                'strikeouts': int(stats.get('strikeOuts', 0)),
                'batting_avg': float(stats.get('avg', 0)),
                'on_base_pct': float(stats.get('obp', 0)),
                'slugging_pct': float(stats.get('slg', 0)),
                'ops': float(stats.get('ops', 0)),
                'runs_per_game': round(runs_per_game, 2),
                'updated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting team stats for {team_id}: {e}")
            return None
    
    def get_daily_schedule(self, date_str):
        """Get today's games and starting pitchers"""
        try:
            url = f"{self.base_url}/schedule"
            params = {
                'date': date_str,
                'sportId': 1,
                'hydrate': 'probablePitcher'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to get schedule: {response.status_code}")
                return []
            
            data = response.json()
            games = []
            
            for date_data in data.get('dates', []):
                for game in date_data.get('games', []):
                    # Only regular season games
                    if game.get('gameType') != 'R':
                        continue
                    
                    home_pitcher = game.get('teams', {}).get('home', {}).get('probablePitcher', {})
                    away_pitcher = game.get('teams', {}).get('away', {}).get('probablePitcher', {})
                    
                    if not home_pitcher.get('id') or not away_pitcher.get('id'):
                        logger.warning(f"Missing pitcher info for game {game.get('gamePk')}")
                        continue
                    
                    games.append({
                        'game_id': game.get('gamePk'),
                        'date': date_str,
                        'home_team_id': game.get('teams', {}).get('home', {}).get('team', {}).get('id'),
                        'away_team_id': game.get('teams', {}).get('away', {}).get('team', {}).get('id'),
                        'home_pitcher_id': home_pitcher.get('id'),
                        'away_pitcher_id': away_pitcher.get('id'),
                        'venue_id': game.get('venue', {}).get('id'),
                        'game_time_et': game.get('gameDate'),
                        'created_at': datetime.now().isoformat()
                    })
            
            logger.info(f"Found {len(games)} games for {date_str}")
            return games
            
        except Exception as e:
            logger.error(f"Error getting schedule for {date_str}: {e}")
            return []
    
    def save_pitcher_stats(self, pitcher_stats):
        """Save pitcher stats to database"""
        if not pitcher_stats:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO pitcher_season_stats 
            (pitcher_id, season, date, games_played, games_started, innings_pitched,
             earned_runs, hits_allowed, walks, strikeouts, home_runs_allowed,
             era, whip, k_per_9, bb_per_9, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pitcher_stats['pitcher_id'], pitcher_stats['season'], pitcher_stats['date'],
                pitcher_stats['games_played'], pitcher_stats['games_started'], 
                pitcher_stats['innings_pitched'], pitcher_stats['earned_runs'],
                pitcher_stats['hits_allowed'], pitcher_stats['walks'],
                pitcher_stats['strikeouts'], pitcher_stats['home_runs_allowed'],
                pitcher_stats['era'], pitcher_stats['whip'],
                pitcher_stats['k_per_9'], pitcher_stats['bb_per_9'],
                pitcher_stats['updated_at']
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving pitcher stats: {e}")
        finally:
            conn.close()
    
    def save_team_stats(self, team_stats):
        """Save team stats to database"""
        if not team_stats:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO team_season_stats
            (team_id, season, date, games_played, runs, hits, doubles, triples,
             home_runs, rbi, walks, strikeouts, batting_avg, on_base_pct,
             slugging_pct, ops, runs_per_game, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team_stats['team_id'], team_stats['season'], team_stats['date'],
                team_stats['games_played'], team_stats['runs'], team_stats['hits'],
                team_stats['doubles'], team_stats['triples'], team_stats['home_runs'],
                team_stats['rbi'], team_stats['walks'], team_stats['strikeouts'],
                team_stats['batting_avg'], team_stats['on_base_pct'],
                team_stats['slugging_pct'], team_stats['ops'],
                team_stats['runs_per_game'], team_stats['updated_at']
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving team stats: {e}")
        finally:
            conn.close()
    
    def save_game_matchups(self, games):
        """Save game matchups to database"""
        if not games:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for game in games:
                cursor.execute("""
                INSERT OR REPLACE INTO game_matchups
                (game_id, date, home_team_id, away_team_id, home_pitcher_id,
                 away_pitcher_id, venue_id, game_time_et, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game['game_id'], game['date'], game['home_team_id'],
                    game['away_team_id'], game['home_pitcher_id'],
                    game['away_pitcher_id'], game['venue_id'],
                    game['game_time_et'], game['created_at']
                ))
            
            conn.commit()
            logger.info(f"Saved {len(games)} game matchups")
            
        except Exception as e:
            logger.error(f"Error saving game matchups: {e}")
        finally:
            conn.close()
    
    def collect_daily_pregame_stats(self, date_str):
        """Collect all pre-game stats for a specific date"""
        logger.info(f"Collecting pre-game stats for {date_str}")
        
        # Get today's games
        games = self.get_daily_schedule(date_str)
        if not games:
            logger.warning(f"No games found for {date_str}")
            return
        
        # Save game matchups
        self.save_game_matchups(games)
        
        # Collect stats for all pitchers and teams
        unique_pitchers = set()
        unique_teams = set()
        
        for game in games:
            unique_pitchers.add(game['home_pitcher_id'])
            unique_pitchers.add(game['away_pitcher_id'])
            unique_teams.add(game['home_team_id'])
            unique_teams.add(game['away_team_id'])
        
        logger.info(f"Collecting stats for {len(unique_pitchers)} pitchers and {len(unique_teams)} teams")
        
        # Collect pitcher stats (as of yesterday)
        yesterday = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for pitcher_id in unique_pitchers:
            logger.info(f"Getting stats for pitcher {pitcher_id}")
            pitcher_stats = self.get_current_season_stats(pitcher_id, yesterday)
            if pitcher_stats:
                self.save_pitcher_stats(pitcher_stats)
            time.sleep(0.1)  # Rate limiting
        
        # Collect team stats (as of yesterday)
        for team_id in unique_teams:
            logger.info(f"Getting stats for team {team_id}")
            team_stats = self.get_team_season_stats(team_id, yesterday)
            if team_stats:
                self.save_team_stats(team_stats)
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Completed collecting pre-game stats for {date_str}")

def main():
    """Main function to collect today's pre-game stats"""
    collector = MLBStatsCollector()
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    print("🏈 MLB LEGITIMATE STATS COLLECTOR")
    print("=" * 50)
    print(f"Collecting pre-game stats for: {today}")
    print()
    
    try:
        collector.collect_daily_pregame_stats(today)
        print("✅ Successfully collected pre-game statistics!")
        print(f"📊 Data saved to: {collector.db_path}")
        
    except Exception as e:
        print(f"❌ Error collecting stats: {e}")
        logger.error(f"Main collection error: {e}")

if __name__ == "__main__":
    main()
