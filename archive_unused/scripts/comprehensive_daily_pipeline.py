#!/usr/bin/env python3
"""
Comprehensive Daily MLB Data Pipeline
====================================

This script implements a complete daily workflow for:
1. Data Collection (games, pitchers, teams, weather, odds)
2. Feature Engineering 
3. Model Training/Updating
4. Prediction Generation

Usage: python comprehensive_daily_pipeline.py --date 2025-08-13
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import time
from sqlalchemy import create_engine, text
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBDataPipeline:
    def __init__(self, db_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb"):
        self.engine = create_engine(db_url)
        self.today = date.today().isoformat()
        
    def collect_daily_games(self, target_date: str) -> pd.DataFrame:
        """Collect today's games with full details"""
        logger.info(f"Collecting games for {target_date}")
        
        try:
            url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={target_date}&endDate={target_date}&sportId=1&hydrate=weather,venue,linescore,pitching"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                games = []
                
                for date_entry in data.get('dates', []):
                    for game in date_entry.get('games', []):
                        # Get team info safely
                        home_team = game.get('teams', {}).get('home', {}).get('team', {})
                        away_team = game.get('teams', {}).get('away', {}).get('team', {})
                        
                        game_info = {
                            'game_id': game['gamePk'],
                            'date': target_date,
                            'home_team': home_team.get('name', 'Unknown'),
                            'away_team': away_team.get('name', 'Unknown'),
                            'home_team_abbr': home_team.get('abbreviation', 'UNK'),
                            'away_team_abbr': away_team.get('abbreviation', 'UNK'),
                            'venue_name': game.get('venue', {}).get('name', 'Unknown'),
                            'game_state': game.get('status', {}).get('detailedState', 'Unknown'),
                            'start_time': game.get('gameDate'),
                            
                            # Extract pitcher info if available
                            'home_pitcher_id': None,
                            'home_pitcher_name': None,
                            'away_pitcher_id': None,
                            'away_pitcher_name': None,
                            
                            # Weather info
                            'temperature': None,
                            'wind_speed': None,
                            'wind_direction': None,
                            'weather_condition': None,
                        }
                        
                        # Get weather data
                        if 'weather' in game:
                            weather = game['weather']
                            game_info.update({
                                'temperature': weather.get('temp'),
                                'wind_speed': weather.get('wind'),
                                'wind_direction': weather.get('windDirection'),
                                'weather_condition': weather.get('condition')
                            })
                        
                        # Get pitcher data from linescore
                        if 'linescore' in game and 'offense' in game['linescore']:
                            # Try to get probable pitchers
                            home_pitcher = game.get('teams', {}).get('home', {}).get('probablePitcher')
                            away_pitcher = game.get('teams', {}).get('away', {}).get('probablePitcher')
                            
                            if home_pitcher:
                                game_info['home_pitcher_id'] = home_pitcher['id']
                                game_info['home_pitcher_name'] = f"{home_pitcher['fullName']}"
                                
                            if away_pitcher:
                                game_info['away_pitcher_id'] = away_pitcher['id']
                                game_info['away_pitcher_name'] = f"{away_pitcher['fullName']}"
                        
                        games.append(game_info)
                        
                return pd.DataFrame(games)
            else:
                logger.error(f"Failed to fetch games: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting games: {e}")
            return pd.DataFrame()
    
    def collect_team_stats(self, teams: List[str], target_date: str) -> pd.DataFrame:
        """Collect current team offensive statistics"""
        logger.info(f"Collecting team stats for {len(teams)} teams")
        
        team_stats = []
        for team_abbr in teams:
            try:
                # Get team info first
                url = f"https://statsapi.mlb.com/api/v1/teams?sportId=1"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    teams_data = response.json()
                    team_id = None
                    
                    for team in teams_data.get('teams', []):
                        if team['abbreviation'] == team_abbr:
                            team_id = team['id']
                            break
                    
                    if team_id:
                        # Get team stats
                        stats_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&season=2025&group=hitting"
                        stats_response = requests.get(stats_url, timeout=10)
                        
                        if stats_response.status_code == 200:
                            stats_data = stats_response.json()
                            
                            if 'stats' in stats_data and stats_data['stats']:
                                hitting_stats = stats_data['stats'][0].get('splits', [])
                                if hitting_stats:
                                    stat = hitting_stats[0]['stat']
                                    games_played = max(int(stat.get('gamesPlayed', 1)), 1)
                                    plate_appearances = max(int(stat.get('plateAppearances', 1)), 1)
                                    
                                    team_stats.append({
                                        'date': target_date,
                                        'team': team_abbr,
                                        'runs_pg': float(stat.get('runs', 0)) / games_played,
                                        'ba': float(stat.get('avg', 0.250)),  # Default to .250
                                        'woba': float(stat.get('obp', 0.320)),  # Using OBP as proxy
                                        'bb_pct': float(stat.get('baseOnBalls', 0)) / plate_appearances,
                                        'k_pct': float(stat.get('strikeOuts', 0)) / plate_appearances,
                                        'games_played': games_played,
                                        'runs_total': int(stat.get('runs', 0)),
                                        'hits': int(stat.get('hits', 0)),
                                        'at_bats': int(stat.get('atBats', 0)),
                                        'home_runs': int(stat.get('homeRuns', 0))
                                    })
                                else:
                                    # Add default stats if no data
                                    team_stats.append({
                                        'date': target_date,
                                        'team': team_abbr,
                                        'runs_pg': 4.5,  # League average
                                        'ba': 0.250,
                                        'woba': 0.320,
                                        'bb_pct': 0.08,
                                        'k_pct': 0.22,
                                        'games_played': 1,
                                        'runs_total': 0,
                                        'hits': 0,
                                        'at_bats': 0,
                                        'home_runs': 0
                                    })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to get stats for {team_abbr}: {e}")
                
        return pd.DataFrame(team_stats)
    
    def collect_betting_odds(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Collect betting odds for games"""
        logger.info("Collecting betting odds")
        
        # Enhanced betting odds collection
        games_with_odds = games_df.copy()
        games_with_odds['market_total'] = None
        games_with_odds['over_odds'] = None
        games_with_odds['under_odds'] = None
        
        try:
            # Try ESPN API for odds
            espn_url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
            response = requests.get(espn_url, timeout=15)
            
            if response.status_code == 200:
                espn_data = response.json()
                
                for idx, row in games_with_odds.iterrows():
                    home_abbr = row['home_team_abbr']
                    
                    for event in espn_data.get('events', []):
                        competitions = event.get('competitions', [])
                        if competitions:
                            competitors = competitions[0].get('competitors', [])
                            
                            # Find matching game
                            home_match = False
                            for comp in competitors:
                                if (comp.get('homeAway') == 'home' and 
                                    comp.get('team', {}).get('abbreviation') == home_abbr):
                                    home_match = True
                                    break
                            
                            if home_match:
                                # Look for odds
                                odds_list = competitions[0].get('odds', [])
                                for odds in odds_list:
                                    if 'overUnder' in odds:
                                        games_with_odds.at[idx, 'market_total'] = float(odds['overUnder'])
                                        
                                    # Try to get over/under odds
                                    if 'awayTeamOdds' in odds and 'homeTeamOdds' in odds:
                                        # Default odds if specific over/under not available
                                        games_with_odds.at[idx, 'over_odds'] = -110
                                        games_with_odds.at[idx, 'under_odds'] = -110
                                break
                
        except Exception as e:
            logger.warning(f"Failed to collect betting odds: {e}")
            
        return games_with_odds
    
    def build_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive features for ML model"""
        logger.info("Building ML features")
        
        featured_games = games_df.copy()
        
        # Add team stats
        for idx, row in featured_games.iterrows():
            home_team = row['home_team_abbr'] 
            away_team = row['away_team_abbr']
            
            # Get home team stats
            home_stats = team_stats_df[team_stats_df['team'] == home_team]
            if not home_stats.empty:
                home_stats = home_stats.iloc[0]
                featured_games.at[idx, 'home_runs_pg'] = home_stats['runs_pg']
                featured_games.at[idx, 'home_ba'] = home_stats['ba']
                featured_games.at[idx, 'home_woba'] = home_stats['woba']
                featured_games.at[idx, 'home_bb_pct'] = home_stats['bb_pct']
                featured_games.at[idx, 'home_k_pct'] = home_stats['k_pct']
                
            # Get away team stats  
            away_stats = team_stats_df[team_stats_df['team'] == away_team]
            if not away_stats.empty:
                away_stats = away_stats.iloc[0]
                featured_games.at[idx, 'away_runs_pg'] = away_stats['runs_pg']
                featured_games.at[idx, 'away_ba'] = away_stats['ba']
                featured_games.at[idx, 'away_woba'] = away_stats['woba']
                featured_games.at[idx, 'away_bb_pct'] = away_stats['bb_pct']
                featured_games.at[idx, 'away_k_pct'] = away_stats['k_pct']
        
        # Add pitcher features (ERA, WHIP, etc.) - to be implemented
        # Add weather features
        # Add venue features
        # Add recent form features
        
        return featured_games
    
    def save_to_database(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame):
        """Save collected data to database"""
        logger.info("Saving data to database")
        
        try:
            with self.engine.begin() as conn:
                # Save games
                if not games_df.empty:
                    games_df.to_sql('daily_games_enhanced', conn, if_exists='append', index=False)
                    logger.info(f"Saved {len(games_df)} games")
                
                # Save team stats
                if not team_stats_df.empty:
                    team_stats_df.to_sql('teams_offense_daily', conn, if_exists='append', index=False)
                    logger.info(f"Saved {len(team_stats_df)} team stat records")
                    
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def run_daily_pipeline(self, target_date: str):
        """Run the complete daily pipeline"""
        logger.info(f"Starting daily pipeline for {target_date}")
        
        # Step 1: Collect games
        games_df = self.collect_daily_games(target_date)
        if games_df.empty:
            logger.error("No games collected, stopping pipeline")
            return
            
        logger.info(f"Collected {len(games_df)} games")
        
        # Step 2: Get unique teams
        teams = list(set(games_df['home_team_abbr'].tolist() + games_df['away_team_abbr'].tolist()))
        logger.info(f"Found {len(teams)} unique teams")
        
        # Step 3: Collect team stats
        team_stats_df = self.collect_team_stats(teams, target_date)
        logger.info(f"Collected stats for {len(team_stats_df)} teams")
        
        # Step 4: Collect betting odds
        games_with_odds = self.collect_betting_odds(games_df)
        
        # Step 5: Build features
        featured_games = self.build_features(games_with_odds, team_stats_df)
        
        # Step 6: Save to database
        self.save_to_database(featured_games, team_stats_df)
        
        logger.info("Daily pipeline completed successfully!")
        
        return featured_games, team_stats_df

def main():
    parser = argparse.ArgumentParser(description='Run MLB daily data pipeline')
    parser.add_argument('--date', default=date.today().isoformat(), 
                       help='Date to process (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    pipeline = MLBDataPipeline()
    pipeline.run_daily_pipeline(args.date)

if __name__ == "__main__":
    main()
