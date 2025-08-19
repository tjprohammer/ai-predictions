#!/usr/bin/env python3
"""
Enhanced API endpoint that combines historical predictions with comprehensive game data.
Pulls ERA, weather, betting lines, and all available game information.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGamesAPI:
    def __init__(self):
        self.db_params = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass',
            'port': '5432'
        }
        self.engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)
    
    def get_comprehensive_games_today(self) -> Dict[str, Any]:
        """
        Get comprehensive game data including:
        - Historical predictions
        - Team statistics
        - Pitcher ERA data
        - Weather conditions
        - Betting lines
        - Venue information
        """
        try:
            # Get historical predictions using the function
            historical_predictions = self.get_historical_predictions()
            
            # Get detailed game data
            games_with_details = []
            
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                for pred in historical_predictions.get('predictions', []):
                    game_id = pred['game_id']
                    
                    # Get comprehensive game data
                    game_details = self._get_game_details(cursor, game_id)
                    if game_details:
                        # Merge prediction with game details
                        enhanced_game = {
                            **game_details,
                            'historical_prediction': {
                                'predicted_total': pred['predicted_total'],
                                'confidence': pred['confidence'],
                                'similar_games_count': pred['similar_games_count'],
                                'historical_range': pred['historical_range'],
                                'method': pred['prediction_method']
                            }
                        }
                        games_with_details.append(enhanced_game)
            
            return {
                'games': games_with_details,
                'count': len(games_with_details),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'data_sources': {
                    'historical_predictions': True,
                    'pitcher_stats': True,
                    'team_stats': True,
                    'weather_data': True,
                    'betting_lines': True,
                    'venue_info': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive games: {e}")
            return {'error': str(e), 'games': [], 'count': 0}
    
    def get_historical_predictions(self) -> Dict[str, Any]:
        """Get historical predictions using the same logic"""
        try:
            with self.engine.begin() as conn:
                # Get today's games
                current_games = pd.read_sql("""
                    SELECT id, home_team, away_team 
                    FROM daily_games 
                    ORDER BY id
                """, conn)
                
                if current_games.empty:
                    return {'predictions': [], 'count': 0}
                
                # Get historical games for similarity matching
                historical_games = pd.read_sql("""
                    SELECT * FROM enhanced_games 
                    WHERE total_runs IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 2000
                """, conn)
                
                predictions = []
                
                for _, game in current_games.iterrows():
                    home_team = game['home_team']
                    away_team = game['away_team']
                    
                    # Find similar historical games
                    similar_games = historical_games[
                        ((historical_games['home_team'] == home_team) & 
                         (historical_games['away_team'] == away_team)) |
                        ((historical_games['home_team'] == away_team) & 
                         (historical_games['away_team'] == home_team))
                    ]
                    
                    # If not enough direct matchups, include team games
                    if len(similar_games) < 10:
                        team_games = historical_games[
                            (historical_games['home_team'].isin([home_team, away_team])) |
                            (historical_games['away_team'].isin([home_team, away_team]))
                        ]
                        similar_games = pd.concat([similar_games, team_games]).drop_duplicates()
                    
                    if len(similar_games) > 0:
                        # Calculate prediction from historical outcomes
                        total_runs = similar_games['total_runs'].values
                        predicted_total = float(np.mean(total_runs))
                        
                        # Calculate confidence based on consistency
                        std_dev = np.std(total_runs)
                        confidence = max(0.5, min(0.95, 1.0 - (std_dev / 15.0)))
                        
                        min_runs = int(np.min(total_runs))
                        max_runs = int(np.max(total_runs))
                        
                        predictions.append({
                            'game_id': game['id'],
                            'away_team': away_team,
                            'home_team': home_team,
                            'predicted_total': round(predicted_total, 1),
                            'confidence': round(confidence, 2),
                            'similar_games_count': len(similar_games),
                            'historical_range': f"{min_runs}-{max_runs} runs",
                            'prediction_method': 'historical_similarity'
                        })
                
                return {
                    'predictions': predictions,
                    'count': len(predictions),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'method': 'historical_similarity',
                    'total_historical_games': len(historical_games)
                }
                
        except Exception as e:
            logger.error(f"Error getting historical predictions: {e}")
            return {'error': str(e), 'predictions': []}
    
    def _get_game_details(self, cursor, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed game information from database"""
        try:
            # Get basic game info
            cursor.execute("""
                SELECT 
                    g.id,
                    g.start_time,
                    g.home_team,
                    g.away_team,
                    g.venue,
                    g.game_state,
                    g.weather_temp,
                    g.weather_wind_mph,
                    g.weather_wind_dir,
                    g.weather_conditions,
                    g.weather_humidity,
                    g.weather_pressure
                FROM daily_games g
                WHERE g.id = %s
            """, (game_id,))
            
            game_row = cursor.fetchone()
            if not game_row:
                return None
            
            game_data = dict(game_row)
            
            # Get pitcher information with ERA
            cursor.execute("""
                SELECT 
                    home_sp_id,
                    home_sp_name,
                    away_sp_id,
                    away_sp_name
                FROM daily_games 
                WHERE id = %s
            """, (game_id,))
            
            pitcher_row = cursor.fetchone()
            if pitcher_row:
                pitcher_data = dict(pitcher_row)
                
                # Get pitcher ERA stats from team_stats table
                home_era = self._get_pitcher_era(cursor, pitcher_data['home_sp_name'], game_data['home_team'])
                away_era = self._get_pitcher_era(cursor, pitcher_data['away_sp_name'], game_data['away_team'])
                
                game_data['pitchers'] = {
                    'home_name': pitcher_data['home_sp_name'],
                    'home_era': home_era,
                    'home_id': pitcher_data['home_sp_id'],
                    'away_name': pitcher_data['away_sp_name'],
                    'away_era': away_era,
                    'away_id': pitcher_data['away_sp_id']
                }
            
            # Get team statistics
            game_data['team_stats'] = self._get_team_stats(cursor, game_data['home_team'], game_data['away_team'])
            
            # Get betting information
            game_data['betting_info'] = self._get_betting_info(cursor, game_id)
            
            # Format weather data
            if any([game_data.get('weather_temp'), game_data.get('weather_wind_mph')]):
                game_data['weather'] = {
                    'temperature': game_data.get('weather_temp'),
                    'wind_speed': game_data.get('weather_wind_mph'),
                    'wind_direction': game_data.get('weather_wind_dir'),
                    'conditions': game_data.get('weather_conditions'),
                    'humidity': game_data.get('weather_humidity'),
                    'pressure': game_data.get('weather_pressure')
                }
            else:
                game_data['weather'] = None
            
            # Clean up individual weather fields
            for field in ['weather_temp', 'weather_wind_mph', 'weather_wind_dir', 
                         'weather_conditions', 'weather_humidity', 'weather_pressure']:
                game_data.pop(field, None)
            
            return game_data
            
        except Exception as e:
            logger.error(f"Error getting game details for {game_id}: {e}")
            return None
    
    def _get_pitcher_era(self, cursor, pitcher_name: str, team: str) -> Optional[float]:
        """Get pitcher ERA from team stats"""
        if not pitcher_name:
            return None
            
        try:
            # Try to find pitcher ERA in team_stats table
            cursor.execute("""
                SELECT era 
                FROM team_stats 
                WHERE team_name = %s 
                AND era IS NOT NULL
                ORDER BY date DESC 
                LIMIT 1
            """, (team,))
            
            result = cursor.fetchone()
            if result and result['era']:
                return float(result['era'])
            
            # If not found, try a more general approach with pitcher names
            cursor.execute("""
                SELECT era 
                FROM team_stats 
                WHERE era IS NOT NULL
                ORDER BY date DESC 
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            if results:
                # Return average of recent ERAs as fallback
                eras = [float(r['era']) for r in results if r['era']]
                if eras:
                    return sum(eras) / len(eras)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting pitcher ERA for {pitcher_name}: {e}")
            return None
    
    def _get_team_stats(self, cursor, home_team: str, away_team: str) -> Dict[str, Any]:
        """Get team statistics"""
        try:
            team_stats = {}
            
            for team_type, team_name in [('home', home_team), ('away', away_team)]:
                cursor.execute("""
                    SELECT 
                        runs_per_game,
                        era,
                        hits_per_game,
                        home_runs_per_game,
                        batting_avg
                    FROM team_stats 
                    WHERE team_name = %s 
                    ORDER BY date DESC 
                    LIMIT 1
                """, (team_name,))
                
                result = cursor.fetchone()
                if result:
                    team_stats[team_type] = {
                        'runs_per_game': result.get('runs_per_game'),
                        'era': result.get('era'),
                        'hits_per_game': result.get('hits_per_game'),
                        'home_runs_per_game': result.get('home_runs_per_game'),
                        'batting_avg': result.get('batting_avg')
                    }
                else:
                    team_stats[team_type] = {}
            
            return team_stats
            
        except Exception as e:
            logger.error(f"Error getting team stats: {e}")
            return {}
    
    def _get_betting_info(self, cursor, game_id: str) -> Optional[Dict[str, Any]]:
        """Get betting lines and odds"""
        try:
            # Check if we have betting data table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'betting_lines'
                )
            """)
            
            if not cursor.fetchone()[0]:
                return None
            
            cursor.execute("""
                SELECT 
                    total_line,
                    over_odds,
                    under_odds,
                    home_ml,
                    away_ml,
                    spread,
                    book_name
                FROM betting_lines 
                WHERE game_id = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (game_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'total_line': result.get('total_line'),
                    'over_odds': result.get('over_odds'),
                    'under_odds': result.get('under_odds'),
                    'home_moneyline': result.get('home_ml'),
                    'away_moneyline': result.get('away_ml'),
                    'spread': result.get('spread'),
                    'book': result.get('book_name')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting betting info for {game_id}: {e}")
            return None

def main():
    """Test the enhanced API"""
    api = EnhancedGamesAPI()
    result = api.get_comprehensive_games_today()
    
    print("Enhanced Games API Test:")
    print(f"Found {result.get('count', 0)} games")
    
    if result.get('games'):
        sample_game = result['games'][0]
        print(f"\nSample game: {sample_game.get('away_team')} @ {sample_game.get('home_team')}")
        print(f"Historical prediction: {sample_game.get('historical_prediction', {}).get('predicted_total')}")
        print(f"Weather: {sample_game.get('weather')}")
        print(f"Pitchers: {sample_game.get('pitchers')}")
    
    # Pretty print first game
    if result.get('games'):
        print("\nFirst game details:")
        print(json.dumps(result['games'][0], indent=2, default=str))

if __name__ == "__main__":
    main()
