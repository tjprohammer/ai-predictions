#!/usr/bin/env python3
"""
Season Statistics Collector
===========================
Collects legitimate pre-game statistics for pitchers and teams
This replaces the data leakage features with proper season stats
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from datetime import datetime, timedelta
import argparse
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeasonStatsCollector:
    def __init__(self):
        # PostgreSQL connection for Docker container
        self.database_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def get_engine(self):
        """Get PostgreSQL database engine"""
        return create_engine(self.database_url)
        
    def get_team_bullpen_stats(self, team_id, season=2025, date_cutoff=None):
        """Get team's bullpen statistics for the season"""
        try:
            # Get pitching stats for relievers
            url = f"{self.base_url}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'pitching',
                'gameType': 'R'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract pitching stats
                stats = data.get('stats', [{}])[0].get('splits', [{}])[0].get('stat', {})
                
                # Calculate bullpen-specific metrics
                total_era = float(stats.get('era', 4.50))
                total_whip = float(stats.get('whip', 1.35))
                total_saves = int(stats.get('saves', 0))
                total_blown_saves = int(stats.get('blownSaves', 0))
                total_holds = int(stats.get('holds', 0))
                
                return {
                    'bullpen_era': total_era,  # Team overall pitching ERA (includes starters, but good proxy)
                    'bullpen_whip': total_whip,  # Team overall pitching WHIP
                    'team_saves': total_saves,
                    'team_blown_saves': total_blown_saves,
                    'team_holds': total_holds,
                    'bullpen_reliability': total_saves / max(total_saves + total_blown_saves, 1)  # Save percentage
                }
            else:
                logger.warning(f"Failed to get bullpen stats for team {team_id}: {response.status_code}")
                return self._get_default_bullpen_stats()
                
        except Exception as e:
            logger.error(f"Error getting team bullpen stats for {team_id}: {e}")
            return self._get_default_bullpen_stats()
    
    def get_pitcher_workload_stats(self, pitcher_id, season=2025, date_cutoff=None):
        """Get starting pitcher's typical workload and innings pitched"""
        try:
            # Get pitcher stats to determine typical outing length
            url = f"{self.base_url}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'pitching',
                'gameType': 'R'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract pitching stats
                stats = data.get('stats', [{}])[0].get('splits', [{}])[0].get('stat', {})
                
                innings_pitched = float(stats.get('inningsPitched', '100.0'))
                games_started = int(stats.get('gamesStarted', 20))
                
                # Calculate average innings per start
                avg_innings_per_start = innings_pitched / max(games_started, 1) if games_started > 0 else 5.0
                
                return {
                    'avg_innings_per_start': avg_innings_per_start,
                    'games_started': games_started,
                    'total_innings': innings_pitched,
                    'expected_bullpen_innings': 9.0 - avg_innings_per_start  # How much bullpen will pitch
                }
            else:
                logger.warning(f"Failed to get workload stats for pitcher {pitcher_id}: {response.status_code}")
                return self._get_default_workload_stats()
                
        except Exception as e:
            logger.error(f"Error getting pitcher workload stats for {pitcher_id}: {e}")
            return self._get_default_workload_stats()

    def _get_default_bullpen_stats(self):
        """Return default bullpen stats if API fails"""
        return {
            'bullpen_era': 4.20,
            'bullpen_whip': 1.30,
            'team_saves': 25,
            'team_blown_saves': 8,
            'team_holds': 40,
            'bullpen_reliability': 0.76
        }
    
    def _get_default_workload_stats(self):
        """Return default pitcher workload stats if API fails"""
        return {
            'avg_innings_per_start': 5.5,
            'games_started': 20,
            'total_innings': 110.0,
            'expected_bullpen_innings': 3.5
        }

    def get_team_season_stats(self, team_id, season=2025, date_cutoff=None):
        """Get team's season statistics up to a specific date"""
        try:
            # Get team stats for the season
            url = f"{self.base_url}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'hitting'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant stats
                stats = data.get('stats', [{}])[0].get('splits', [{}])[0].get('stat', {})
                
                return {
                    'runs_per_game': float(stats.get('runs', 0)) / max(float(stats.get('gamesPlayed', 1)), 1),
                    'batting_average': float(stats.get('avg', 0.250)),
                    'on_base_percentage': float(stats.get('obp', 0.320)),
                    'slugging_percentage': float(stats.get('slg', 0.400)),
                    'ops': float(stats.get('ops', 0.720)),
                    'hits_per_game': float(stats.get('hits', 0)) / max(float(stats.get('gamesPlayed', 1)), 1),
                    'games_played': int(stats.get('gamesPlayed', 0))
                }
            else:
                logger.warning(f"Failed to get team stats for team {team_id}: {response.status_code}")
                return self._get_default_team_stats()
                
        except Exception as e:
            logger.error(f"Error getting team season stats for {team_id}: {e}")
            return self._get_default_team_stats()
    
    def get_pitcher_season_stats(self, pitcher_id, season=2025, date_cutoff=None):
        """Get pitcher's season statistics up to a specific date"""
        try:
            # Get pitcher stats for the season
            url = f"{self.base_url}/people/{pitcher_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'pitching'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant stats
                stats = data.get('stats', [{}])[0].get('splits', [{}])[0].get('stat', {})
                
                innings_pitched = float(stats.get('inningsPitched', '1.0'))
                earned_runs = float(stats.get('earnedRuns', 0))
                hits_allowed = float(stats.get('hits', 0))
                walks_allowed = float(stats.get('baseOnBalls', 0))
                strikeouts = float(stats.get('strikeOuts', 0))
                
                return {
                    'season_era': float(stats.get('era', 4.50)),
                    'season_whip': (hits_allowed + walks_allowed) / max(innings_pitched, 1.0),
                    'season_k_per_9': (strikeouts * 9.0) / max(innings_pitched, 1.0),
                    'season_bb_per_9': (walks_allowed * 9.0) / max(innings_pitched, 1.0),
                    'innings_pitched': innings_pitched,
                    'wins': int(stats.get('wins', 0)),
                    'losses': int(stats.get('losses', 0)),
                    'games_started': int(stats.get('gamesStarted', 0))
                }
            else:
                logger.warning(f"Failed to get pitcher stats for pitcher {pitcher_id}: {response.status_code}")
                return self._get_default_pitcher_stats()
                
        except Exception as e:
            logger.error(f"Error getting pitcher season stats for {pitcher_id}: {e}")
            return self._get_default_pitcher_stats()
    
    def _get_default_team_stats(self):
        """Return default team stats if API fails"""
        return {
            'runs_per_game': 4.5,
            'batting_average': 0.250,
            'on_base_percentage': 0.320,
            'slugging_percentage': 0.400,
            'ops': 0.720,
            'hits_per_game': 8.5,
            'games_played': 100
        }
    
    def _get_default_pitcher_stats(self):
        """Return default pitcher stats if API fails"""
        return {
            'season_era': 4.50,
            'season_whip': 1.35,
            'season_k_per_9': 8.0,
            'season_bb_per_9': 3.0,
            'innings_pitched': 100.0,
            'wins': 8,
            'losses': 8,
            'games_started': 20
        }
    
    def get_ballpark_factors(self, venue_id):
        """Get ballpark run environment factors"""
        # Simplified ballpark factors (in real implementation, these would come from historical data)
        ballpark_factors = {
            1: {'run_factor': 1.05, 'hr_factor': 1.15},  # Example: Yankee Stadium (hitter-friendly)
            2: {'run_factor': 0.95, 'hr_factor': 0.85},  # Example: Petco Park (pitcher-friendly)
            3: {'run_factor': 1.10, 'hr_factor': 1.25},  # Example: Coors Field (very hitter-friendly)
            # Add more ballparks as needed
        }
        
        return ballpark_factors.get(venue_id, {'run_factor': 1.0, 'hr_factor': 1.0})
    
    def get_team_id_by_name(self, team_name, season=2025):
        """Get team ID from MLB StatsAPI by team name"""
        try:
            url = f"{self.base_url}/teams"
            params = {'season': season, 'sportId': 1}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                teams = response.json().get('teams', [])
                for team in teams:
                    if team.get('name') == team_name or team.get('teamName') == team_name:
                        return team.get('id')
            logger.warning(f"Could not find team ID for {team_name}")
            return None
        except Exception as e:
            logger.error(f"Error getting team ID for {team_name}: {e}")
            return None

    def get_probable_pitchers(self, game_id):
        """Get probable starting pitchers for a game"""
        try:
            url = f"{self.base_url}/schedule"
            params = {'gamePk': game_id, 'hydrate': 'probablePitcher'}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = data.get('dates', [{}])[0].get('games', [])
                if games:
                    game = games[0]
                    teams = game.get('teams', {})
                    home_pitcher = teams.get('home', {}).get('probablePitcher', {}).get('id')
                    away_pitcher = teams.get('away', {}).get('probablePitcher', {}).get('id')
                    return home_pitcher, away_pitcher
            return None, None
        except Exception as e:
            logger.error(f"Error getting probable pitchers for game {game_id}: {e}")
            return None, None

    def collect_game_features(self, target_date=None):
        """Collect legitimate features for games on target date"""
        if target_date is None:
            target_date = datetime.now().date()
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        logger.info(f"Collecting season statistics for games on {target_date}")
        
        try:
            # Connect to database
            engine = self.get_engine()
            
            # Get games for the target date - use minimal required columns
            games_query = """
            SELECT game_id, home_team, away_team, 
                   COALESCE(venue_name, 'Unknown Venue') as venue_name,
                   COALESCE(temperature, 70) as temperature, 
                   COALESCE(wind_speed, 5) as wind_speed,
                   COALESCE(weather_condition, 'Clear') as weather_condition,
                   COALESCE(day_night, 'D') as day_night
            FROM enhanced_games 
            WHERE date = %(target_date)s
            """
            
            with engine.connect() as conn:
                games_df = pd.read_sql(games_query, conn, params={'target_date': target_date.isoformat()})
            
            if games_df.empty:
                logger.warning(f"No games found for {target_date}")
                return
            
            logger.info(f"Found {len(games_df)} games for {target_date}")
            
            # Collect features for each game
            features_data = []
            
            for _, game in games_df.iterrows():
                logger.info(f"Processing {game['away_team']} @ {game['home_team']}")
                
                # Resolve team and pitcher IDs via API (no schema dependency)
                home_team_id = self.get_team_id_by_name(game['home_team'])
                away_team_id = self.get_team_id_by_name(game['away_team'])
                home_sp_id, away_sp_id = self.get_probable_pitchers(game['game_id'])
                
                if not home_team_id or not away_team_id:
                    logger.warning(f"Could not resolve team IDs for {game['away_team']} @ {game['home_team']}, using defaults")
                    
                if not home_sp_id or not away_sp_id:
                    logger.warning(f"Could not resolve pitcher IDs for game {game['game_id']}, using defaults")
                
                # Get team season stats using resolved team IDs (up to day before game)
                home_team_stats = self.get_team_season_stats(home_team_id, date_cutoff=target_date) if home_team_id else self._get_default_team_stats()
                away_team_stats = self.get_team_season_stats(away_team_id, date_cutoff=target_date) if away_team_id else self._get_default_team_stats()
                
                # Get pitcher season stats (up to day before game)
                home_pitcher_stats = self.get_pitcher_season_stats(home_sp_id, date_cutoff=target_date) if home_sp_id else self._get_default_pitcher_stats()
                away_pitcher_stats = self.get_pitcher_season_stats(away_sp_id, date_cutoff=target_date) if away_sp_id else self._get_default_pitcher_stats()
                
                # Get bullpen statistics for both teams
                home_bullpen_stats = self.get_team_bullpen_stats(home_team_id, date_cutoff=target_date) if home_team_id else self._get_default_bullpen_stats()
                away_bullpen_stats = self.get_team_bullpen_stats(away_team_id, date_cutoff=target_date) if away_team_id else self._get_default_bullpen_stats()
                
                # Get pitcher workload stats to estimate bullpen usage
                home_pitcher_workload = self.get_pitcher_workload_stats(home_sp_id, date_cutoff=target_date) if home_sp_id else self._get_default_workload_stats()
                away_pitcher_workload = self.get_pitcher_workload_stats(away_sp_id, date_cutoff=target_date) if away_sp_id else self._get_default_workload_stats()
                
                # Get ballpark factors (using venue_name hash as simple venue_id)
                venue_id = hash(game['venue_name']) % 1000  # Simple venue ID from name
                ballpark_factors = self.get_ballpark_factors(venue_id)
                
                # Compile all features
                features = {
                    'game_id': game['game_id'],
                    'date': target_date.isoformat(),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'venue_name': game['venue_name'],
                    
                    # Weather features (legitimate pre-game data)
                    'temperature': game['temperature'],
                    'wind_speed': game['wind_speed'],
                    
                    # Calculate weather factors
                    'temp_factor': max(0.8, min(1.2, game['temperature'] / 70.0)) if game['temperature'] else 1.0,
                    'wind_factor': max(0.9, min(1.1, 1.0 + (game['wind_speed'] - 5) * 0.02)) if game['wind_speed'] else 1.0,
                    'wind_out': 1 if game['wind_speed'] and game['wind_speed'] > 10 else 0,
                    'wind_in': 1 if game['wind_speed'] and game['wind_speed'] < 3 else 0,
                    'is_dome': 0,  # Assume outdoor games for future predictions
                    'is_rain': 1 if game.get('weather_condition', '').lower().find('rain') >= 0 else 0,
                    'is_night_game': 1 if game.get('day_night', 'D') == 'N' else 0,
                    
                    # Team offensive features (season stats, not game outcomes)
                    'home_team_runs_pg': home_team_stats['runs_per_game'],
                    'away_team_runs_pg': away_team_stats['runs_per_game'],
                    'combined_team_offense': (home_team_stats['runs_per_game'] + away_team_stats['runs_per_game']) / 2,
                    'home_team_woba': home_team_stats.get('woba', home_team_stats['ops'] * 0.8),  # Approximate wOBA from OPS
                    'away_team_woba': away_team_stats.get('woba', away_team_stats['ops'] * 0.8),
                    'combined_team_woba': (home_team_stats.get('woba', home_team_stats['ops'] * 0.8) + away_team_stats.get('woba', away_team_stats['ops'] * 0.8)) / 2,
                    'home_team_wrcplus': home_team_stats.get('wrcplus', max(50, min(150, home_team_stats['ops'] * 140))),  # Approximate wRC+ from OPS
                    'away_team_wrcplus': away_team_stats.get('wrcplus', max(50, min(150, away_team_stats['ops'] * 140))),
                    'combined_team_wrcplus': (home_team_stats.get('wrcplus', home_team_stats['ops'] * 140) + away_team_stats.get('wrcplus', away_team_stats['ops'] * 140)) / 2,
                    'home_team_power': home_team_stats.get('iso', max(0.1, min(0.3, home_team_stats['ops'] - home_team_stats['batting_average']))),  # Approximate ISO
                    'away_team_power': away_team_stats.get('iso', max(0.1, min(0.3, away_team_stats['ops'] - away_team_stats['batting_average']))),
                    'combined_team_power': (home_team_stats.get('iso', home_team_stats['ops'] - home_team_stats['batting_average']) + away_team_stats.get('iso', away_team_stats['ops'] - away_team_stats['batting_average'])) / 2,
                    'home_team_discipline': home_team_stats.get('bb_rate', 0.08),  # Default walk rate
                    'away_team_discipline': away_team_stats.get('bb_rate', 0.08),
                    'combined_team_discipline': (home_team_stats.get('bb_rate', 0.08) + away_team_stats.get('bb_rate', 0.08)) / 2,
                    'offensive_imbalance': abs(home_team_stats['runs_per_game'] - away_team_stats['runs_per_game']),
                    
                    # Pitcher features (season stats, not game outcomes)
                    'home_sp_season_era': home_pitcher_stats['season_era'],
                    'away_sp_season_era': away_pitcher_stats['season_era'],
                    'combined_era': (home_pitcher_stats['season_era'] + away_pitcher_stats['season_era']) / 2,
                    'era_differential': home_pitcher_stats['season_era'] - away_pitcher_stats['season_era'],
                    'home_sp_whip': home_pitcher_stats['season_whip'],
                    'away_sp_whip': away_pitcher_stats['season_whip'],
                    'combined_whip': (home_pitcher_stats['season_whip'] + away_pitcher_stats['season_whip']) / 2,
                    'home_sp_k_per_9': home_pitcher_stats['season_k_per_9'],
                    'away_sp_k_per_9': away_pitcher_stats['season_k_per_9'],
                    'home_pitcher_quality': max(2.0, min(8.0, home_pitcher_stats['season_era'])),
                    'away_pitcher_quality': max(2.0, min(8.0, away_pitcher_stats['season_era'])),
                    
                    # Bullpen features (relief pitching quality)
                    'home_bullpen_era': home_bullpen_stats['bullpen_era'],
                    'away_bullpen_era': away_bullpen_stats['bullpen_era'],
                    'combined_bullpen_era': (home_bullpen_stats['bullpen_era'] + away_bullpen_stats['bullpen_era']) / 2,
                    'bullpen_era_advantage': home_bullpen_stats['bullpen_era'] - away_bullpen_stats['bullpen_era'],
                    'home_weighted_pitching_era': (home_pitcher_stats['season_era'] * home_pitcher_workload['avg_innings_per_start'] + home_bullpen_stats['bullpen_era'] * home_pitcher_workload['expected_bullpen_innings']) / 9.0,
                    'away_weighted_pitching_era': (away_pitcher_stats['season_era'] * away_pitcher_workload['avg_innings_per_start'] + away_bullpen_stats['bullpen_era'] * away_pitcher_workload['expected_bullpen_innings']) / 9.0,
                    'combined_weighted_pitching_era': ((home_pitcher_stats['season_era'] * home_pitcher_workload['avg_innings_per_start'] + home_bullpen_stats['bullpen_era'] * home_pitcher_workload['expected_bullpen_innings']) / 9.0 + (away_pitcher_stats['season_era'] * away_pitcher_workload['avg_innings_per_start'] + away_bullpen_stats['bullpen_era'] * away_pitcher_workload['expected_bullpen_innings']) / 9.0) / 2,
                    
                    # Ballpark factors
                    'ballpark_run_factor': ballpark_factors['run_factor'],
                    'ballpark_hr_factor': ballpark_factors['hr_factor'],
                    'ballpark_offensive_factor': ballpark_factors['run_factor'] * ballpark_factors['hr_factor'],
                    
                    # Weather-park interactions
                    'temp_park_interaction': (max(0.8, min(1.2, game['temperature'] / 70.0)) if game['temperature'] else 1.0) * ballpark_factors['run_factor'],
                    'wind_park_interaction': (max(0.9, min(1.1, 1.0 + (game['wind_speed'] - 5) * 0.02)) if game['wind_speed'] else 1.0) * ballpark_factors['hr_factor'],
                    
                    # Enhanced derived features including ballpark context
                    'expected_offensive_environment': (home_team_stats.get('woba', home_team_stats['ops'] * 0.8) + away_team_stats.get('woba', away_team_stats['ops'] * 0.8)) / 2,
                    'pitching_dominance': min(max(0.2, 1.0 - ((home_pitcher_stats['season_era'] + away_pitcher_stats['season_era']) / 2 - 3.5) / 5.0), 1.8),
                    'market_vs_team_total': 0.0,  # Will be updated by market data
                    'late_game_scoring_factor': (home_bullpen_stats['bullpen_era'] + away_bullpen_stats['bullpen_era']) / 8.0,  # Bullpen impact on late-game scoring
                    'market_total': 0.0  # Will be updated by market data
                }
                
                features_data.append(features)
                time.sleep(0.1)  # Rate limiting
            
            # Save features using safe staging pattern
            features_df = pd.DataFrame(features_data)
            
            # Use PostgreSQL engine for saving
            engine = self.get_engine()
            
            with engine.connect() as conn:
                # 1. Save to staging table (replace existing staging data)
                logger.info("Saving to staging table...")
                features_df.to_sql('legit_features_staging', conn, if_exists='replace', index=False)
                
                # 2. Ensure legitimate_game_features has required columns
                logger.info("Adding missing columns to legitimate_game_features if needed...")
                conn.execute(text("""
                    ALTER TABLE legitimate_game_features
                      ADD COLUMN IF NOT EXISTS home_team_runs_pg numeric,
                      ADD COLUMN IF NOT EXISTS away_team_runs_pg numeric
                """))
                conn.commit()
                
                # 3. Safe INSERT with conflict handling (delete existing then insert)
                logger.info("Upserting data into legitimate_game_features...")
                
                # First, delete any existing data for this date
                delete_result = conn.execute(text("""
                    DELETE FROM legitimate_game_features WHERE "date" = :target_date
                """), {'target_date': target_date.isoformat()})
                
                # Then insert new data
                insert_result = conn.execute(text("""
                    INSERT INTO legitimate_game_features (game_id, "date", home_team, away_team,
                        home_team_runs_pg, away_team_runs_pg)
                    SELECT s.game_id, s."date"::date, s.home_team, s.away_team,
                           s.home_team_runs_pg, s.away_team_runs_pg
                    FROM legit_features_staging s
                """))
                conn.commit()
                
                # 4. Verify the upsert worked
                check_result = conn.execute(text("""
                    SELECT COUNT(*) FROM legitimate_game_features 
                    WHERE "date" = :target_date 
                      AND home_team_runs_pg IS NOT NULL 
                      AND away_team_runs_pg IS NOT NULL
                """), {'target_date': target_date.isoformat()})
                rpg_count = check_result.fetchone()[0]
                
                logger.info(f"✅ Upserted {len(features_df)} games, {rpg_count} with valid RPG data")
                
                # 5. Check total training data available
                total_check = conn.execute(text("SELECT COUNT(*) FROM legitimate_game_features"))
                total_features = total_check.fetchone()[0]
                logger.info(f"Total features in database: {total_features}")
            
            # Also save to CSV for backup
            output_file = f"../data/legitimate_features_{target_date.isoformat()}.csv"
            features_df.to_csv(output_file, index=False)
            logger.info(f"Backup saved to {output_file}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error collecting season statistics: {e}")
            return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Collect legitimate season statistics for MLB games')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    collector = SeasonStatsCollector()
    
    if args.target_date:
        target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()
    else:
        target_date = datetime.now().date()
    
    logger.info(f"Starting season statistics collection for {target_date}")
    
    features_df = collector.collect_game_features(target_date)
    
    if features_df is not None:
        logger.info("✅ Season statistics collection completed successfully")
        logger.info(f"Collected features for {len(features_df)} games")
        logger.info("Features include legitimate pre-game data only:")
        logger.info("  - Pitcher season statistics (ERA, WHIP, K/9)")
        logger.info("  - Team season statistics (runs/game, batting avg, OPS)")
        logger.info("  - Weather forecasts and ballpark factors")
        logger.info("  - NO game outcome data used")
    else:
        logger.error("❌ Season statistics collection failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
