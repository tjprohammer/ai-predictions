#!/usr/bin/env python3
"""
Comprehensive Historical Training Data Builder
==============================================
Builds training dataset using all 1,877 historical games from enhanced_games
with proper season stats, ballpark factors, and weather data.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTrainingBuilder:
    def __init__(self):
        self.db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        self.engine = create_engine(self.db_url)
        
    def build_historical_features(self):
        """Build comprehensive training features from enhanced_games"""
        logger.info("Building comprehensive historical training features...")
        
        # Get all completed games with outcomes
        base_query = """
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            venue_name,
            venue_id,
            day_night,
            
            -- Outcomes (target variable)
            total_runs,
            home_score,
            away_score,
            
            -- Starting Pitcher Data (actual season stats at time of game)
            home_sp_name,
            home_sp_season_era,
            home_sp_ip,
            home_sp_k,
            home_sp_bb,
            home_sp_h,
            home_sp_er,
            
            away_sp_name, 
            away_sp_season_era,
            away_sp_ip,
            away_sp_k,
            away_sp_bb,
            away_sp_h,
            away_sp_er,
            
            -- Weather Data
            temperature,
            wind_speed,
            wind_direction,
            weather_condition,
            
            -- Betting Market Data
            market_total,
            over_odds,
            under_odds
            
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL
        AND date < CURRENT_DATE  -- Only historical games
        ORDER BY date
        """
        
        logger.info("Loading base game data...")
        df = pd.read_sql(base_query, self.engine)
        logger.info(f"Loaded {len(df):,} historical games from {df['date'].min()} to {df['date'].max()}")
        
        # Calculate derived pitcher stats
        df = self._add_pitcher_features(df)
        
        # Add ballpark factors
        df = self._add_ballpark_factors(df)
        
        # Add team season stats (rolling averages)
        df = self._add_team_season_stats(df)
        
        # Add weather factors
        df = self._add_weather_features(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"✅ Built comprehensive training dataset with {len(df):,} games and {len(df.columns)} features")
        return df
    
    def _add_pitcher_features(self, df):
        """Add calculated pitcher features"""
        logger.info("Adding pitcher features...")
        
        # Calculate WHIP from available data
        df['home_sp_whip'] = (df['home_sp_bb'] + df['home_sp_h']) / df['home_sp_ip'].replace(0, np.nan)
        df['away_sp_whip'] = (df['away_sp_bb'] + df['away_sp_h']) / df['away_sp_ip'].replace(0, np.nan)
        
        # Calculate K/9 rates
        df['home_sp_k_per_9'] = (df['home_sp_k'] * 9) / df['home_sp_ip'].replace(0, np.nan)
        df['away_sp_k_per_9'] = (df['away_sp_k'] * 9) / df['away_sp_ip'].replace(0, np.nan)
        
        # Calculate BB/9 rates
        df['home_sp_bb_per_9'] = (df['home_sp_bb'] * 9) / df['away_sp_ip'].replace(0, np.nan)
        df['away_sp_bb_per_9'] = (df['away_sp_bb'] * 9) / df['away_sp_ip'].replace(0, np.nan)
        
        # Pitching quality score (lower is better)
        df['home_pitcher_quality'] = df['home_sp_season_era'] * (1 + df['home_sp_whip'])
        df['away_pitcher_quality'] = df['away_sp_season_era'] * (1 + df['away_sp_whip'])
        
        # Combined pitcher metrics
        df['combined_era'] = (df['home_sp_season_era'] + df['away_sp_season_era']) / 2
        df['combined_whip'] = (df['home_sp_whip'] + df['away_sp_whip']) / 2
        df['era_differential'] = df['home_sp_season_era'] - df['away_sp_season_era']
        df['pitching_advantage'] = (df['home_sp_k_per_9'] + df['away_sp_k_per_9']) / 2
        
        return df
    
    def _add_ballpark_factors(self, df):
        """Add ballpark run/HR factors"""
        logger.info("Adding ballpark factors...")
        
        # Known ballpark factors (could be enhanced with historical data)
        ballpark_factors = {
            'Coors Field': {'run_factor': 1.15, 'hr_factor': 1.20},  # High altitude
            'Great American Ball Park': {'run_factor': 1.08, 'hr_factor': 1.12},
            'Progressive Field': {'run_factor': 0.95, 'hr_factor': 0.92},
            'Fenway Park': {'run_factor': 1.02, 'hr_factor': 1.05},  # Green Monster
            'Yankee Stadium': {'run_factor': 1.05, 'hr_factor': 1.10},  # Short right field
            'Petco Park': {'run_factor': 0.92, 'hr_factor': 0.88},  # Pitcher friendly
            'Tropicana Field': {'run_factor': 0.98, 'hr_factor': 0.95},  # Dome
            'Minute Maid Park': {'run_factor': 1.03, 'hr_factor': 1.08},
            'Marlins Park': {'run_factor': 0.96, 'hr_factor': 0.93},
            'Chase Field': {'run_factor': 1.02, 'hr_factor': 1.04},
        }
        
        # Default factors for unknown parks
        default_factors = {'run_factor': 1.00, 'hr_factor': 1.00}
        
        # Apply ballpark factors
        df['ballpark_run_factor'] = df['venue_name'].map(
            {park: factors['run_factor'] for park, factors in ballpark_factors.items()}
        ).fillna(default_factors['run_factor'])
        
        df['ballpark_hr_factor'] = df['venue_name'].map(
            {park: factors['hr_factor'] for park, factors in ballpark_factors.items()}
        ).fillna(default_factors['hr_factor'])
        
        # Combined ballpark offensive factor
        df['ballpark_offensive_factor'] = (df['ballpark_run_factor'] + df['ballpark_hr_factor']) / 2
        
        return df
    
    def _add_team_season_stats(self, df):
        """Add team offensive stats using rolling averages"""
        logger.info("Adding team season statistics...")
        
        # Sort by date to calculate rolling stats
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate rolling team averages (30-game window)
        team_stats = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing team stats for game {idx+1:,} of {len(df):,}")
                
            current_date = row['date']
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get recent games for each team (last 30 games before current date)
            home_recent = df[
                (df['date'] < current_date) & 
                ((df['home_team'] == home_team) | (df['away_team'] == home_team))
            ].tail(30)
            
            away_recent = df[
                (df['date'] < current_date) & 
                ((df['home_team'] == away_team) | (df['away_team'] == away_team))
            ].tail(30)
            
            # Calculate home team stats
            if len(home_recent) > 0:
                home_runs = []
                for _, game in home_recent.iterrows():
                    if game['home_team'] == home_team:
                        home_runs.append(game['home_score'])
                    else:
                        home_runs.append(game['away_score'])
                home_rpg = np.mean(home_runs) if home_runs else 4.5
            else:
                home_rpg = 4.5  # League average default
            
            # Calculate away team stats
            if len(away_recent) > 0:
                away_runs = []
                for _, game in away_recent.iterrows():
                    if game['home_team'] == away_team:
                        away_runs.append(game['home_score'])
                    else:
                        away_runs.append(game['away_score'])
                away_rpg = np.mean(away_runs) if away_runs else 4.5
            else:
                away_rpg = 4.5  # League average default
            
            team_stats.append({
                'home_team_runs_per_game': home_rpg,
                'away_team_runs_per_game': away_rpg
            })
        
        # Add team stats to dataframe
        team_stats_df = pd.DataFrame(team_stats)
        df = pd.concat([df, team_stats_df], axis=1)
        
        # Team offensive features
        df['combined_team_offense'] = (df['home_team_runs_per_game'] + df['away_team_runs_per_game']) / 2
        df['offensive_imbalance'] = abs(df['home_team_runs_per_game'] - df['away_team_runs_per_game'])
        
        return df
    
    def _add_weather_features(self, df):
        """Add weather-based features"""
        logger.info("Adding weather features...")
        
        # Temperature effects
        df['temp_factor'] = (df['temperature'] - 70) * 0.01  # Hotter = more offense
        
        # Wind effects  
        df['wind_factor'] = df['wind_speed'] * 0.05
        
        # Wind direction effects
        df['wind_out'] = df['wind_direction'].str.contains('Out|out', na=False).astype(int)
        df['wind_in'] = df['wind_direction'].str.contains('In|in', na=False).astype(int)
        
        # Weather condition effects
        df['is_dome'] = df['weather_condition'].str.contains('Dome|dome', na=False).astype(int)
        df['is_rain'] = df['weather_condition'].str.contains('Rain|rain', na=False).astype(int)
        
        # Day/night game
        df['is_night_game'] = (df['day_night'] == 'N').astype(int)
        
        return df
    
    def _add_derived_features(self, df):
        """Add derived interaction features"""
        logger.info("Adding derived features...")
        
        # Weather-park interactions
        df['temp_park_interaction'] = df['temp_factor'] * df['ballpark_run_factor']
        df['wind_park_interaction'] = df['wind_factor'] * df['ballpark_hr_factor'] * df['wind_out']
        
        # Pitching-offensive environment
        df['expected_offensive_environment'] = (
            df['combined_team_offense'] * df['ballpark_offensive_factor'] * 
            (1 + df['temp_factor']) * (2 - df['combined_era'] / 5.0)
        )
        
        # Pitching dominance
        df['pitching_dominance'] = 1 / (df['combined_era'] + df['combined_whip'])
        
        # Market efficiency check
        df['market_vs_team_total'] = df['market_total'] - df['combined_team_offense']
        
        return df
    
    def save_training_data(self, df):
        """Save comprehensive training data to database"""
        logger.info("Saving comprehensive training data...")
        
        # Select final feature set
        feature_columns = [
            # Game identifiers
            'game_id', 'date', 'home_team', 'away_team', 'venue_name',
            
            # Target variable
            'total_runs',
            
            # Pitcher features
            'home_sp_season_era', 'away_sp_season_era', 'combined_era', 'era_differential',
            'home_sp_whip', 'away_sp_whip', 'combined_whip',
            'home_sp_k_per_9', 'away_sp_k_per_9', 'pitching_advantage',
            'home_pitcher_quality', 'away_pitcher_quality', 'pitching_dominance',
            
            # Team features
            'home_team_runs_per_game', 'away_team_runs_per_game', 'combined_team_offense',
            'offensive_imbalance',
            
            # Ballpark features
            'ballpark_run_factor', 'ballpark_hr_factor', 'ballpark_offensive_factor',
            
            # Weather features
            'temperature', 'wind_speed', 'temp_factor', 'wind_factor',
            'wind_out', 'wind_in', 'is_dome', 'is_rain', 'is_night_game',
            
            # Derived features
            'temp_park_interaction', 'wind_park_interaction', 
            'expected_offensive_environment', 'market_vs_team_total',
            
            # Market data
            'market_total'
        ]
        
        # Filter to available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        training_df = df[available_columns].copy()
        
        # Clean data
        training_df = training_df.dropna(subset=['total_runs'])
        training_df = training_df.fillna(0)  # Fill remaining NAs with 0
        
        logger.info(f"Final training dataset: {len(training_df):,} games x {len(available_columns)} features")
        logger.info(f"Average runs per game: {training_df['total_runs'].mean():.2f}")
        logger.info(f"Date range: {training_df['date'].min()} to {training_df['date'].max()}")
        
        # Save to database (replace existing legitimate_game_features)
        with self.engine.begin() as conn:
            # Clear existing training data
            conn.execute("DELETE FROM legitimate_game_features")
            
            # Insert comprehensive historical data
            training_df.to_sql('legitimate_game_features', conn, if_exists='append', index=False)
            
        logger.info("✅ Comprehensive training data saved successfully!")
        return training_df

def main():
    """Build comprehensive historical training data"""
    builder = ComprehensiveTrainingBuilder()
    
    # Build features from all historical games
    df = builder.build_historical_features()
    
    # Save for model training
    training_df = builder.save_training_data(df)
    
    print("\n🎯 COMPREHENSIVE TRAINING DATA READY:")
    print(f"   📊 Games: {len(training_df):,}")
    print(f"   📅 Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    print(f"   🏃 Avg runs/game: {training_df['total_runs'].mean():.2f}")
    print(f"   🎲 Features: {len(training_df.columns)}")
    print()
    print("Ready for model retraining with comprehensive historical data!")

if __name__ == "__main__":
    main()
