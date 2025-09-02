#!/usr/bin/env python3
"""
Enhanced Feature Engineering Pipeline for MLB Predictions
Focus on high-impact features that correlate with actual game outcomes
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureEngine:
    """Create high-impact features for better MLB predictions"""
    
    def __init__(self):
        self.engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        
    def add_pitcher_recent_form(self, df):
        """Add pitcher recent performance metrics (last 5 starts)"""
        logger.info("🎯 Adding pitcher recent form features...")
        
        # Get recent pitcher performance for each game
        for _, game in df.iterrows():
            home_pitcher_id = game.get('home_sp_id')
            away_pitcher_id = game.get('away_sp_id')
            game_date = pd.to_datetime(game['date'])
            
            # Get last 5 starts for home pitcher
            if home_pitcher_id:
                home_recent = self._get_pitcher_recent_stats(home_pitcher_id, game_date, 5)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_l5_era'] = home_recent.get('era', 4.50)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_l5_whip'] = home_recent.get('whip', 1.30)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_l5_k9'] = home_recent.get('k9', 8.0)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_l5_runs_allowed'] = home_recent.get('runs_allowed', 3.0)
                
            # Get last 5 starts for away pitcher  
            if away_pitcher_id:
                away_recent = self._get_pitcher_recent_stats(away_pitcher_id, game_date, 5)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_l5_era'] = away_recent.get('era', 4.50)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_l5_whip'] = away_recent.get('whip', 1.30)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_l5_k9'] = away_recent.get('k9', 8.0)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_l5_runs_allowed'] = away_recent.get('runs_allowed', 3.0)
        
        # Create derived features
        df['sp_recent_era_diff'] = df['home_sp_l5_era'] - df['away_sp_l5_era']
        df['sp_recent_form_combined'] = (df['home_sp_l5_era'] + df['away_sp_l5_era']) / 2
        df['sp_recent_dominance'] = (df['home_sp_l5_k9'] + df['away_sp_l5_k9']) / 2
        
        logger.info(f"✅ Added {len([c for c in df.columns if 'l5_' in c])} pitcher recent form features")
        return df
        
    def _get_pitcher_recent_stats(self, pitcher_id, game_date, num_games=5):
        """Get pitcher's last N game statistics"""
        try:
            query = text("""
            SELECT 
                AVG(earned_runs::float / GREATEST(innings_pitched, 0.1) * 9) as era,
                AVG((hits_allowed + walks_allowed)::float / GREATEST(innings_pitched, 0.1)) as whip,
                AVG(strikeouts::float / GREATEST(innings_pitched, 0.1) * 9) as k9,
                AVG(earned_runs::float) as runs_allowed,
                COUNT(*) as games_found
            FROM pitcher_game_logs 
            WHERE pitcher_id = :pitcher_id 
                AND game_date < :game_date
                AND innings_pitched > 0
            ORDER BY game_date DESC 
            LIMIT :num_games
            """)
            
            result = self.engine.execute(query, {
                'pitcher_id': pitcher_id,
                'game_date': game_date,
                'num_games': num_games
            }).fetchone()
            
            if result and result.games_found >= 2:  # Need at least 2 recent games
                return {
                    'era': float(result.era) if result.era else 4.50,
                    'whip': float(result.whip) if result.whip else 1.30,
                    'k9': float(result.k9) if result.k9 else 8.0,
                    'runs_allowed': float(result.runs_allowed) if result.runs_allowed else 3.0
                }
            else:
                return {'era': 4.50, 'whip': 1.30, 'k9': 8.0, 'runs_allowed': 3.0}
                
        except Exception as e:
            logger.warning(f"Could not get recent stats for pitcher {pitcher_id}: {e}")
            return {'era': 4.50, 'whip': 1.30, 'k9': 8.0, 'runs_allowed': 3.0}
            
    def add_team_recent_form(self, df):
        """Add team recent offensive/defensive performance (last 10 games)"""
        logger.info("🎯 Adding team recent form features...")
        
        for _, game in df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = pd.to_datetime(game['date'])
            
            # Get recent team performance
            home_recent = self._get_team_recent_stats(home_team, game_date, 10)
            away_recent = self._get_team_recent_stats(away_team, game_date, 10)
            
            # Home team recent stats
            df.loc[df['game_id'] == game['game_id'], 'home_team_l10_runs_pg'] = home_recent.get('runs_per_game', 4.5)
            df.loc[df['game_id'] == game['game_id'], 'home_team_l10_runs_allowed_pg'] = home_recent.get('runs_allowed_per_game', 4.5)
            df.loc[df['game_id'] == game['game_id'], 'home_team_l10_ops'] = home_recent.get('ops', 0.750)
            
            # Away team recent stats
            df.loc[df['game_id'] == game['game_id'], 'away_team_l10_runs_pg'] = away_recent.get('runs_per_game', 4.5)
            df.loc[df['game_id'] == game['game_id'], 'away_team_l10_runs_allowed_pg'] = away_recent.get('runs_allowed_per_game', 4.5)
            df.loc[df['game_id'] == game['game_id'], 'away_team_l10_ops'] = away_recent.get('ops', 0.750)
            
        # Create derived features
        df['team_recent_offense_diff'] = df['home_team_l10_runs_pg'] - df['away_team_l10_runs_pg']
        df['team_recent_pitching_diff'] = df['away_team_l10_runs_allowed_pg'] - df['home_team_l10_runs_allowed_pg']  # Lower is better
        df['expected_total_from_recent'] = (df['home_team_l10_runs_pg'] + df['away_team_l10_runs_pg']) / 2
        
        logger.info(f"✅ Added {len([c for c in df.columns if 'l10_' in c])} team recent form features")
        return df
        
    def _get_team_recent_stats(self, team, game_date, num_games=10):
        """Get team's last N game statistics"""
        try:
            query = text("""
            SELECT 
                AVG(runs) as runs_per_game,
                AVG(runs_allowed) as runs_allowed_per_game,
                AVG(COALESCE(ops, 0.750)) as ops,
                COUNT(*) as games_found
            FROM team_game_logs 
            WHERE team = :team 
                AND game_date < :game_date
            ORDER BY game_date DESC 
            LIMIT :num_games
            """)
            
            result = self.engine.execute(query, {
                'team': team,
                'game_date': game_date,
                'num_games': num_games
            }).fetchone()
            
            if result and result.games_found >= 5:  # Need at least 5 recent games
                return {
                    'runs_per_game': float(result.runs_per_game) if result.runs_per_game else 4.5,
                    'runs_allowed_per_game': float(result.runs_allowed_per_game) if result.runs_allowed_per_game else 4.5,
                    'ops': float(result.ops) if result.ops else 0.750
                }
            else:
                return {'runs_per_game': 4.5, 'runs_allowed_per_game': 4.5, 'ops': 0.750}
                
        except Exception as e:
            logger.warning(f"Could not get recent stats for team {team}: {e}")
            return {'runs_per_game': 4.5, 'runs_allowed_per_game': 4.5, 'ops': 0.750}
            
    def add_matchup_history(self, df):
        """Add pitcher vs team historical matchup data"""
        logger.info("🎯 Adding pitcher vs team matchup features...")
        
        for _, game in df.iterrows():
            home_pitcher_id = game.get('home_sp_id')
            away_pitcher_id = game.get('away_sp_id')
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = pd.to_datetime(game['date'])
            
            # Home pitcher vs away team history
            if home_pitcher_id:
                home_vs_away = self._get_pitcher_vs_team_history(home_pitcher_id, away_team, game_date)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_vs_opp_era'] = home_vs_away.get('era', 4.50)
                df.loc[df['game_id'] == game['game_id'], 'home_sp_vs_opp_starts'] = home_vs_away.get('starts', 0)
                
            # Away pitcher vs home team history
            if away_pitcher_id:
                away_vs_home = self._get_pitcher_vs_team_history(away_pitcher_id, home_team, game_date)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_vs_opp_era'] = away_vs_home.get('era', 4.50)
                df.loc[df['game_id'] == game['game_id'], 'away_sp_vs_opp_starts'] = away_vs_home.get('starts', 0)
        
        # Derived features
        df['sp_matchup_era_diff'] = df['home_sp_vs_opp_era'] - df['away_sp_vs_opp_era']
        df['sp_matchup_experience'] = df['home_sp_vs_opp_starts'] + df['away_sp_vs_opp_starts']
        
        logger.info("✅ Added pitcher vs team matchup features")
        return df
        
    def _get_pitcher_vs_team_history(self, pitcher_id, opposing_team, game_date):
        """Get pitcher's historical performance against specific team"""
        try:
            query = text("""
            SELECT 
                AVG(earned_runs::float / GREATEST(innings_pitched, 0.1) * 9) as era,
                COUNT(*) as starts
            FROM pitcher_game_logs pgl
            JOIN games g ON pgl.game_id = g.game_id
            WHERE pgl.pitcher_id = :pitcher_id 
                AND (g.home_team = :opposing_team OR g.away_team = :opposing_team)
                AND g.game_date < :game_date
                AND pgl.innings_pitched > 0
            """)
            
            result = self.engine.execute(query, {
                'pitcher_id': pitcher_id,
                'opposing_team': opposing_team,
                'game_date': game_date
            }).fetchone()
            
            if result and result.starts > 0:
                return {
                    'era': float(result.era) if result.era else 4.50,
                    'starts': int(result.starts)
                }
            else:
                return {'era': 4.50, 'starts': 0}
                
        except Exception as e:
            logger.warning(f"Could not get matchup history: {e}")
            return {'era': 4.50, 'starts': 0}
            
    def add_bullpen_usage_features(self, df):
        """Add bullpen fatigue and usage patterns"""
        logger.info("🎯 Adding bullpen usage features...")
        
        for _, game in df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = pd.to_datetime(game['date'])
            
            # Get recent bullpen usage (last 3 games)
            home_bullpen = self._get_bullpen_usage(home_team, game_date, 3)
            away_bullpen = self._get_bullpen_usage(away_team, game_date, 3)
            
            df.loc[df['game_id'] == game['game_id'], 'home_bullpen_l3_innings'] = home_bullpen.get('innings', 3.0)
            df.loc[df['game_id'] == game['game_id'], 'home_bullpen_l3_era'] = home_bullpen.get('era', 4.00)
            df.loc[df['game_id'] == game['game_id'], 'away_bullpen_l3_innings'] = away_bullpen.get('innings', 3.0)
            df.loc[df['game_id'] == game['game_id'], 'away_bullpen_l3_era'] = away_bullpen.get('era', 4.00)
            
        # Bullpen fatigue indicator (more innings = more fatigue)
        df['bullpen_fatigue_diff'] = df['home_bullpen_l3_innings'] - df['away_bullpen_l3_innings']
        df['bullpen_quality_diff'] = df['away_bullpen_l3_era'] - df['home_bullpen_l3_era']  # Lower ERA is better
        
        logger.info("✅ Added bullpen usage and fatigue features")
        return df
        
    def _get_bullpen_usage(self, team, game_date, num_days=3):
        """Get team's recent bullpen usage"""
        try:
            start_date = game_date - timedelta(days=num_days)
            
            query = text("""
            SELECT 
                SUM(innings_pitched) as total_innings,
                AVG(earned_runs::float / GREATEST(innings_pitched, 0.1) * 9) as era
            FROM pitcher_game_logs pgl
            JOIN games g ON pgl.game_id = g.game_id
            WHERE (g.home_team = :team OR g.away_team = :team)
                AND g.game_date >= :start_date
                AND g.game_date < :game_date
                AND pgl.is_starter = false
                AND pgl.innings_pitched > 0
            """)
            
            result = self.engine.execute(query, {
                'team': team,
                'start_date': start_date,
                'game_date': game_date
            }).fetchone()
            
            if result and result.total_innings:
                return {
                    'innings': float(result.total_innings),
                    'era': float(result.era) if result.era else 4.00
                }
            else:
                return {'innings': 3.0, 'era': 4.00}
                
        except Exception as e:
            logger.warning(f"Could not get bullpen usage for {team}: {e}")
            return {'innings': 3.0, 'era': 4.00}
            
    def add_interaction_features(self, df):
        """Create interaction features between key variables"""
        logger.info("🎯 Adding interaction features...")
        
        # Temperature * Ballpark interactions
        df['temp_ballpark_runs'] = df.get('temperature', 70) * df.get('ballpark_run_factor', 1.0)
        df['temp_ballpark_hrs'] = df.get('temperature', 70) * df.get('ballpark_hr_factor', 1.0)
        
        # Wind * Ballpark interactions  
        df['wind_ballpark_effect'] = df.get('wind_speed', 5) * df.get('ballpark_hr_factor', 1.0)
        
        # Pitcher form * Ballpark
        df['sp_form_ballpark'] = df.get('sp_recent_form_combined', 4.5) * df.get('ballpark_run_factor', 1.0)
        
        # Team offense * Pitcher matchup
        df['team_offense_vs_pitching'] = df.get('expected_total_from_recent', 9.0) - df.get('sp_recent_form_combined', 4.5)
        
        # Day/Night * Temperature
        is_night = df.get('day_night', 'N') == 'N'
        df['night_temp_effect'] = is_night.astype(int) * df.get('temperature', 70)
        
        logger.info("✅ Added 6 interaction features")
        return df
        
    def add_umpire_tendencies(self, df):
        """Add umpire strike zone and run environment effects"""
        logger.info("🎯 Adding umpire tendency features...")
        
        for _, game in df.iterrows():
            umpire = game.get('plate_umpire')
            if umpire:
                umpire_stats = self._get_umpire_tendencies(umpire)
                df.loc[df['game_id'] == game['game_id'], 'umpire_run_factor'] = umpire_stats.get('run_factor', 1.0)
                df.loc[df['game_id'] == game['game_id'], 'umpire_strike_zone_size'] = umpire_stats.get('zone_size', 1.0)
            else:
                df.loc[df['game_id'] == game['game_id'], 'umpire_run_factor'] = 1.0
                df.loc[df['game_id'] == game['game_id'], 'umpire_strike_zone_size'] = 1.0
                
        logger.info("✅ Added umpire tendency features")
        return df
        
    def _get_umpire_tendencies(self, umpire_name):
        """Get umpire's historical run environment"""
        try:
            query = text("""
            SELECT 
                AVG(total_runs::float) / 9.0 as run_factor,
                AVG(COALESCE(plate_umpire_boost_factor, 1.0)) as zone_size
            FROM enhanced_games 
            WHERE plate_umpire = :umpire_name
                AND total_runs IS NOT NULL
            """)
            
            result = self.engine.execute(query, {'umpire_name': umpire_name}).fetchone()
            
            if result and result.run_factor:
                return {
                    'run_factor': float(result.run_factor),
                    'zone_size': float(result.zone_size) if result.zone_size else 1.0
                }
            else:
                return {'run_factor': 1.0, 'zone_size': 1.0}
                
        except Exception as e:
            logger.warning(f"Could not get umpire tendencies: {e}")
            return {'run_factor': 1.0, 'zone_size': 1.0}
            
    def process_enhanced_features(self, df):
        """Apply all enhanced feature engineering steps"""
        logger.info("🚀 Starting enhanced feature engineering pipeline...")
        
        original_features = len(df.columns)
        
        # Apply all feature engineering steps
        df = self.add_pitcher_recent_form(df)
        df = self.add_team_recent_form(df)
        df = self.add_matchup_history(df)
        df = self.add_bullpen_usage_features(df)
        df = self.add_interaction_features(df)
        df = self.add_umpire_tendencies(df)
        
        new_features = len(df.columns) - original_features
        logger.info(f"✅ Enhanced feature engineering complete!")
        logger.info(f"   Added {new_features} new features")
        logger.info(f"   Total features: {len(df.columns)}")
        
        return df

def main():
    """Test the enhanced feature engine"""
    engine = EnhancedFeatureEngine()
    
    # Test with recent games
    test_query = text("""
    SELECT * FROM enhanced_games 
    WHERE date >= '2025-08-25' 
    AND total_runs IS NOT NULL 
    LIMIT 5
    """)
    
    df = pd.read_sql(test_query, engine.engine)
    print(f"Testing with {len(df)} games...")
    
    enhanced_df = engine.process_enhanced_features(df)
    
    # Show new features
    new_features = [col for col in enhanced_df.columns if col not in df.columns]
    print(f"\nNew features added: {len(new_features)}")
    for feature in new_features:
        print(f"  - {feature}")

if __name__ == "__main__":
    main()
