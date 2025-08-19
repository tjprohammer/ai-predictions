#!/usr/bin/env python3
"""
Legitimate Feature Engineering
==============================
Creates proper pre-game features from season statistics
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class LegitimateFeatureEngineer:
    def __init__(self, db_path="S:/Projects/AI_Predictions/mlb-overs/data/legitimate_stats.db"):
        self.db_path = db_path
        
        # Ballpark run factors (historical data)
        self.ballpark_factors = {
            1: 1.045,   # Fenway Park (hitter-friendly)
            2: 0.985,   # Tropicana Field
            3: 1.035,   # Yankee Stadium (hitter-friendly)
            4: 0.975,   # Oriole Park at Camden Yards
            5: 0.955,   # Tropicana Field (pitcher-friendly)
            6: 1.025,   # Progressive Field
            7: 0.965,   # Kauffman Stadium (pitcher-friendly)
            8: 1.015,   # Angel Stadium
            9: 0.995,   # U.S. Cellular Field
            10: 1.025,  # Comerica Park
            11: 1.045,  # Minute Maid Park (hitter-friendly)
            12: 0.985,  # Marlins Park (pitcher-friendly)
            13: 1.185,  # Coors Field (very hitter-friendly)
            14: 0.975,  # Petco Park (pitcher-friendly)
            15: 1.055,  # Great American Ball Park
            # Add more as needed
        }
    
    def get_pitcher_features(self, pitcher_id, as_of_date):
        """Get pitcher features as of a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT * FROM pitcher_season_stats 
            WHERE pitcher_id = ? AND date = ?
            ORDER BY date DESC LIMIT 1
            """
            
            df = pd.read_sql(query, conn, params=[pitcher_id, as_of_date])
            conn.close()
            
            if df.empty:
                logger.warning(f"No pitcher stats found for {pitcher_id} on {as_of_date}")
                return self.get_default_pitcher_features()
            
            stats = df.iloc[0]
            
            # Create engineered features
            features = {
                'era': float(stats['era']),
                'whip': float(stats['whip']),
                'k_per_9': float(stats['k_per_9']),
                'bb_per_9': float(stats['bb_per_9']),
                'games_started': int(stats['games_started']),
                'innings_pitched': float(stats['innings_pitched']),
                
                # Derived features
                'era_quality': self.categorize_era(stats['era']),
                'control_rating': self.calculate_control_rating(stats['k_per_9'], stats['bb_per_9']),
                'experience_factor': min(stats['games_started'] / 20.0, 1.0),  # Normalize experience
                'durability_factor': min(stats['innings_pitched'] / 150.0, 1.0),  # Normalize workload
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting pitcher features for {pitcher_id}: {e}")
            return self.get_default_pitcher_features()
    
    def get_team_features(self, team_id, as_of_date):
        """Get team offensive features as of a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT * FROM team_season_stats 
            WHERE team_id = ? AND date = ?
            ORDER BY date DESC LIMIT 1
            """
            
            df = pd.read_sql(query, conn, params=[team_id, as_of_date])
            conn.close()
            
            if df.empty:
                logger.warning(f"No team stats found for {team_id} on {as_of_date}")
                return self.get_default_team_features()
            
            stats = df.iloc[0]
            
            features = {
                'runs_per_game': float(stats['runs_per_game']),
                'batting_avg': float(stats['batting_avg']),
                'on_base_pct': float(stats['on_base_pct']),
                'slugging_pct': float(stats['slugging_pct']),
                'ops': float(stats['ops']),
                'home_runs': int(stats['home_runs']),
                'walks': int(stats['walks']),
                'strikeouts': int(stats['strikeouts']),
                
                # Derived features
                'offensive_rating': self.calculate_offensive_rating(stats),
                'power_factor': self.calculate_power_factor(stats),
                'patience_factor': self.calculate_patience_factor(stats),
                'contact_factor': self.calculate_contact_factor(stats),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting team features for {team_id}: {e}")
            return self.get_default_team_features()
    
    def get_ballpark_features(self, venue_id):
        """Get ballpark features"""
        run_factor = self.ballpark_factors.get(venue_id, 1.0)
        
        return {
            'ballpark_run_factor': run_factor,
            'ballpark_hitter_friendly': 1 if run_factor > 1.02 else 0,
            'ballpark_pitcher_friendly': 1 if run_factor < 0.98 else 0,
            'ballpark_neutral': 1 if 0.98 <= run_factor <= 1.02 else 0,
        }
    
    def get_weather_features(self, temperature, wind_speed, weather_condition):
        """Get weather-based features"""
        try:
            temp = float(temperature) if temperature else 70.0
            wind = float(wind_speed) if wind_speed else 5.0
            
            features = {
                'temperature': temp,
                'wind_speed': wind,
                'temp_run_factor': self.calculate_temp_factor(temp),
                'wind_run_factor': self.calculate_wind_factor(wind),
                'weather_clear': 1 if weather_condition in ['Clear', 'Sunny'] else 0,
                'weather_overcast': 1 if weather_condition in ['Overcast', 'Cloudy'] else 0,
                'weather_indoor': 1 if weather_condition in ['Roof Closed', 'Dome'] else 0,
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing weather features: {e}")
            return {
                'temperature': 70.0,
                'wind_speed': 5.0,
                'temp_run_factor': 1.0,
                'wind_run_factor': 1.0,
                'weather_clear': 0,
                'weather_overcast': 0,
                'weather_indoor': 0,
            }
    
    def create_matchup_features(self, home_pitcher_features, away_pitcher_features, 
                               home_team_features, away_team_features):
        """Create matchup-specific features"""
        
        # Pitcher vs team matchups
        era_advantage = away_pitcher_features['era'] - home_pitcher_features['era']
        control_advantage = home_pitcher_features['control_rating'] - away_pitcher_features['control_rating']
        
        # Team offensive vs pitching
        home_offense_vs_away_pitching = home_team_features['offensive_rating'] / max(away_pitcher_features['era_quality'], 0.1)
        away_offense_vs_home_pitching = away_team_features['offensive_rating'] / max(home_pitcher_features['era_quality'], 0.1)
        
        matchup_features = {
            'era_difference': era_advantage,
            'control_difference': control_advantage,
            'home_offense_advantage': home_offense_vs_away_pitching,
            'away_offense_advantage': away_offense_vs_home_pitching,
            'total_offensive_power': home_team_features['offensive_rating'] + away_team_features['offensive_rating'],
            'pitching_quality_average': (home_pitcher_features['era_quality'] + away_pitcher_features['era_quality']) / 2,
            'run_environment': home_team_features['runs_per_game'] + away_team_features['runs_per_game'],
        }
        
        return matchup_features
    
    def build_game_features(self, game_data, weather_data=None):
        """Build complete feature set for a game"""
        
        # Calculate as_of_date (day before game)
        game_date = datetime.strptime(game_data['date'], '%Y-%m-%d')
        as_of_date = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get individual component features
        home_pitcher_features = self.get_pitcher_features(game_data['home_pitcher_id'], as_of_date)
        away_pitcher_features = self.get_pitcher_features(game_data['away_pitcher_id'], as_of_date)
        
        home_team_features = self.get_team_features(game_data['home_team_id'], as_of_date)
        away_team_features = self.get_team_features(game_data['away_team_id'], as_of_date)
        
        ballpark_features = self.get_ballpark_features(game_data.get('venue_id'))
        
        # Weather features
        if weather_data:
            weather_features = self.get_weather_features(
                weather_data.get('temperature'),
                weather_data.get('wind_speed'),
                weather_data.get('weather_condition')
            )
        else:
            weather_features = self.get_weather_features(70, 5, 'Clear')
        
        # Matchup features
        matchup_features = self.create_matchup_features(
            home_pitcher_features, away_pitcher_features,
            home_team_features, away_team_features
        )
        
        # Combine all features with prefixes
        final_features = {}
        
        # Add prefixed features
        for key, value in home_pitcher_features.items():
            final_features[f'home_pitcher_{key}'] = value
        
        for key, value in away_pitcher_features.items():
            final_features[f'away_pitcher_{key}'] = value
        
        for key, value in home_team_features.items():
            final_features[f'home_team_{key}'] = value
        
        for key, value in away_team_features.items():
            final_features[f'away_team_{key}'] = value
        
        # Add other features
        final_features.update(ballpark_features)
        final_features.update(weather_features)
        final_features.update(matchup_features)
        
        # Add game context
        final_features['is_weekend'] = 1 if game_date.weekday() >= 5 else 0
        final_features['month'] = game_date.month
        
        return final_features
    
    # Helper methods for feature calculations
    def categorize_era(self, era):
        """Convert ERA to quality rating"""
        if era < 3.0:
            return 5.0  # Ace
        elif era < 3.5:
            return 4.0  # Very good
        elif era < 4.0:
            return 3.0  # Good
        elif era < 4.5:
            return 2.0  # Average
        else:
            return 1.0  # Below average
    
    def calculate_control_rating(self, k_per_9, bb_per_9):
        """Calculate pitcher control rating"""
        k_bb_ratio = k_per_9 / max(bb_per_9, 1.0)
        return min(k_bb_ratio / 3.0, 2.0)  # Normalize to 0-2 scale
    
    def calculate_offensive_rating(self, stats):
        """Calculate team offensive rating"""
        # Weighted combination of key offensive stats
        ops_weight = 0.4
        rpg_weight = 0.6
        
        ops_normalized = min(float(stats['ops']) / 0.8, 1.5)  # Normalize around .800 OPS
        rpg_normalized = min(float(stats['runs_per_game']) / 5.0, 1.5)  # Normalize around 5 RPG
        
        return (ops_weight * ops_normalized) + (rpg_weight * rpg_normalized)
    
    def calculate_power_factor(self, stats):
        """Calculate team power factor"""
        return min(float(stats['slugging_pct']) / 0.45, 1.5)
    
    def calculate_patience_factor(self, stats):
        """Calculate team patience factor"""
        games = max(int(stats['games_played']), 1)
        walks_per_game = int(stats['walks']) / games
        return min(walks_per_game / 3.5, 1.5)  # Normalize around 3.5 walks per game
    
    def calculate_contact_factor(self, stats):
        """Calculate team contact factor"""
        games = max(int(stats['games_played']), 1)
        strikeouts_per_game = int(stats['strikeouts']) / games
        return max(1.5 - (strikeouts_per_game / 9.0), 0.5)  # Lower strikeouts = better contact
    
    def calculate_temp_factor(self, temperature):
        """Calculate temperature effect on run scoring"""
        # Warmer weather generally increases run scoring
        baseline_temp = 70.0
        temp_effect = (temperature - baseline_temp) * 0.005
        return 1.0 + max(min(temp_effect, 0.15), -0.15)  # Cap at ±15%
    
    def calculate_wind_factor(self, wind_speed):
        """Calculate wind effect on run scoring"""
        # Higher wind can help or hurt depending on direction
        # For simplicity, assume moderate wind helps hitting
        if wind_speed > 15:
            return 1.05  # Strong wind helps
        elif wind_speed < 5:
            return 0.98  # No wind hurts slightly
        else:
            return 1.0   # Normal wind
    
    def get_default_pitcher_features(self):
        """Default pitcher features when data is missing"""
        return {
            'era': 4.50,
            'whip': 1.35,
            'k_per_9': 8.0,
            'bb_per_9': 3.5,
            'games_started': 10,
            'innings_pitched': 60.0,
            'era_quality': 2.0,
            'control_rating': 1.0,
            'experience_factor': 0.5,
            'durability_factor': 0.4,
        }
    
    def get_default_team_features(self):
        """Default team features when data is missing"""
        return {
            'runs_per_game': 4.5,
            'batting_avg': 0.250,
            'on_base_pct': 0.320,
            'slugging_pct': 0.420,
            'ops': 0.740,
            'home_runs': 80,
            'walks': 350,
            'strikeouts': 800,
            'offensive_rating': 1.0,
            'power_factor': 1.0,
            'patience_factor': 1.0,
            'contact_factor': 1.0,
        }

def main():
    """Test the feature engineering"""
    engineer = LegitimateFeatureEngineer()
    
    # Example game data
    game_data = {
        'date': '2025-08-14',
        'home_team_id': 147,  # Yankees
        'away_team_id': 142,  # Twins  
        'home_pitcher_id': 607074,
        'away_pitcher_id': 641672,
        'venue_id': 3  # Yankee Stadium
    }
    
    weather_data = {
        'temperature': 78,
        'wind_speed': 8,
        'weather_condition': 'Clear'
    }
    
    print("🔧 TESTING LEGITIMATE FEATURE ENGINEERING")
    print("=" * 50)
    
    features = engineer.build_game_features(game_data, weather_data)
    
    print(f"Generated {len(features)} features:")
    for feature, value in sorted(features.items()):
        print(f"  {feature}: {value}")

if __name__ == "__main__":
    main()
