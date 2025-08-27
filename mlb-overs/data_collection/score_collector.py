#!/usr/bin/env python3
"""
Collect Final Scores
===================
Collects final scores for completed games and updates the enhanced_games table.
This should be run each morning to capture yesterday's completed game results.
"""

import os
import sys
import json
import logging
import argparse
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalScoreCollector:
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
        
    def get_completed_games_from_api(self, target_date):
        """Get completed games from MLB API for a specific date"""
        try:
            # Format date for MLB API
            formatted_date = datetime.strptime(target_date, '%Y-%m-%d').strftime('%m/%d/%Y')
            
            # MLB Stats API endpoint for games on a specific date
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}"
            
            logger.info(f"Fetching completed games from MLB API for {target_date}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            completed_games = []
            
            if 'dates' in data and len(data['dates']) > 0:
                for date_info in data['dates']:
                    for game in date_info.get('games', []):
                        if game.get('status', {}).get('statusCode') == 'F':  # Final
                            home_score = game.get('teams', {}).get('home', {}).get('score', 0)
                            away_score = game.get('teams', {}).get('away', {}).get('score', 0)
                            total_runs = home_score + away_score
                            
                            home_team = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                            away_team = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                            game_id = str(game.get('gamePk', ''))
                            
                            completed_games.append({
                                'game_id': game_id,
                                'home_team': home_team,
                                'away_team': away_team,
                                'home_score': home_score,
                                'away_score': away_score,
                                'total_runs': total_runs,
                                'date': target_date
                            })
            
            logger.info(f"Found {len(completed_games)} completed games")
            return completed_games
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch games from MLB API: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing MLB API response: {e}")
            return []
    
    def update_database_with_scores(self, completed_games):
        """Update enhanced_games table with final scores"""
        if not completed_games:
            logger.info("No completed games to update")
            return 0
            
        updated_count = 0
        
        try:
            with self.engine.begin() as conn:
                for game in completed_games:
                    # Try to match by team names and date
                    update_sql = text("""
                        UPDATE enhanced_games 
                        SET total_runs = :total_runs,
                            home_score = :home_score,
                            away_score = :away_score
                        WHERE date = :date 
                        AND (
                            (home_team = :home_team AND away_team = :away_team)
                            OR game_id = :game_id
                        )
                        AND total_runs IS NULL
                    """)
                    
                    result = conn.execute(update_sql, {
                        'total_runs': game['total_runs'],
                        'home_score': game['home_score'],
                        'away_score': game['away_score'],
                        'date': game['date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'game_id': game['game_id']
                    })
                    
                    if result.rowcount > 0:
                        updated_count += 1
                        logger.info(f"✅ Updated: {game['away_team']} @ {game['home_team']} = {game['total_runs']} runs")
            
            logger.info(f"💾 Updated {updated_count} games with final scores")
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
            return 0
    
    def collect_scores_for_date(self, target_date):
        """Main function to collect and store final scores for a date"""
        logger.info(f"🏁 Collecting final scores for {target_date}")
        
        # Get completed games from API
        completed_games = self.get_completed_games_from_api(target_date)
        
        if not completed_games:
            logger.warning(f"No completed games found for {target_date}")
            return False
        
        # Update database
        updated_count = self.update_database_with_scores(completed_games)
        
        if updated_count > 0:
            logger.info(f"✅ Successfully updated {updated_count} completed games")
            return True
        else:
            logger.warning("No games were updated in the database")
            return False

def main():
    parser = argparse.ArgumentParser(description='Collect Final Scores for MLB Games')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to yesterday')
    args = parser.parse_args()
    
    # Default to yesterday if no date provided
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    collector = FinalScoreCollector()
    success = collector.collect_scores_for_date(target_date)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
