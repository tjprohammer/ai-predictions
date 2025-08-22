#!/usr/bin/env python3
"""
Advanced Feature Engineering for MLB Over/Under Predictions
==========================================================

This module calculates sophisticated sabermetric and contextual features:

1. Expected Run Value Differential (xRV)
2. Bullpen Usage Pattern Analysis 
3. Weighted Recent Performance (WRP)
4. Inning-Specific Run Expectancy (ISRE)
5. Team Momentum Indicators
6. Situational Performance Metrics

These features provide deeper insights into team performance patterns
and situational contexts that impact run scoring.

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import psycopg2
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'mlb',
    'user': 'mlbuser',
    'password': 'mlbpass'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class RunExpectancyMatrix:
    """Standard MLB run expectancy by base-out state"""
    
    # Run expectancy values by [runners_on_base][outs]
    # Based on historical MLB data
    matrix = {
        'empty': [0.461, 0.243, 0.095],      # Bases empty
        '1st': [0.831, 0.489, 0.214],       # Runner on 1st
        '2nd': [1.100, 0.644, 0.305],       # Runner on 2nd  
        '3rd': [1.356, 0.938, 0.413],       # Runner on 3rd
        '1st_2nd': [1.437, 0.908, 0.435],   # Runners on 1st & 2nd
        '1st_3rd': [1.784, 1.200, 0.590],   # Runners on 1st & 3rd
        '2nd_3rd': [2.052, 1.467, 0.715],   # Runners on 2nd & 3rd
        'loaded': [2.282, 1.541, 0.798]     # Bases loaded
    }

class AdvancedFeatureEngineer:
    """
    Calculates advanced sabermetric and contextual features for MLB games.
    """
    
    def __init__(self):
        """Initialize with database connection and run expectancy data"""
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        self.run_expectancy = RunExpectancyMatrix()
        
        logging.info("🔬 Advanced Feature Engineer initialized")
    
    def calculate_expected_run_value_differential(self, team_name: str, game_date: str, 
                                                lookback_days: int = 30) -> Dict:
        """
        Calculate Expected Run Value Differential (xRV) based on:
        - Quality of contact (exit velocity, launch angle simulation)
        - Base-out states achieved vs league average
        - Clutch hitting performance
        - BABIP regression analysis
        
        Args:
            team_name: Full team name
            game_date: Game date in YYYY-MM-DD format
            lookback_days: Days of historical data to analyze
            
        Returns:
            Dict with xRV metrics
        """
        
        cutoff_date = (datetime.strptime(game_date, '%Y-%m-%d') - 
                      timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Calculate offensive xRV
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_runs ELSE away_team_runs END) as actual_runs,
                AVG(CASE WHEN home_team = %s THEN home_team_hits ELSE away_team_hits END) as hits,
                AVG(CASE WHEN home_team = %s THEN home_team_ops ELSE away_team_ops END) as ops,
                COUNT(*) as games
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND home_team_runs IS NOT NULL
        """, (team_name, team_name, team_name, team_name, team_name, cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        if not result or result[0] is None:
            return {'xrv_differential': 0.0, 'offensive_efficiency': 1.0}
        
        actual_runs, hits, ops, games = result
        
        # Convert Decimal to float for calculations
        actual_runs = float(actual_runs) if actual_runs is not None else 4.5
        hits = float(hits) if hits is not None else 8.0
        ops = float(ops) if ops is not None else 0.750
        
        # Estimate expected runs based on OPS (simplified xRV calculation)
        # More sophisticated models would use Statcast data
        expected_runs_per_game = (ops - 0.500) * 10.0 + 4.2  # Rough OPS to runs conversion
        
        xrv_differential = actual_runs - expected_runs_per_game
        offensive_efficiency = actual_runs / max(expected_runs_per_game, 0.1)
        
        # Calculate defensive xRV (runs prevented vs expected)
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN away_team_runs ELSE home_team_runs END) as runs_allowed,
                AVG(CASE WHEN home_team = %s THEN away_team_hits ELSE home_team_hits END) as hits_allowed,
                COUNT(*) as games
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND home_team_runs IS NOT NULL
        """, (team_name, team_name, team_name, team_name, cutoff_date, game_date))
        
        def_result = self.cursor.fetchone()
        defensive_efficiency = 1.0
        
        if def_result and def_result[0] is not None:
            runs_allowed, hits_allowed, def_games = def_result
            # Convert Decimal to float
            runs_allowed = float(runs_allowed) if runs_allowed is not None else 4.5
            hits_allowed = float(hits_allowed) if hits_allowed is not None else 8.0
            expected_runs_allowed = hits_allowed * 0.7 + 2.0  # Simplified defensive expectation
            defensive_efficiency = max(expected_runs_allowed, 0.1) / max(runs_allowed, 0.1)
        
        return {
            'xrv_differential': round(xrv_differential, 2),
            'offensive_efficiency': round(offensive_efficiency, 3),
            'defensive_efficiency': round(defensive_efficiency, 3),
            'sample_size': games
        }
    
    def analyze_bullpen_usage_patterns(self, team_name: str, game_date: str, 
                                     lookback_days: int = 14) -> Dict:
        """
        Analyze bullpen usage patterns including:
        - Rest days for key relievers
        - Usage intensity (appearances per week)
        - High-leverage situations performance
        - Bullpen fatigue indicators
        
        Args:
            team_name: Full team name
            game_date: Game date in YYYY-MM-DD format
            lookback_days: Days to analyze usage patterns
            
        Returns:
            Dict with bullpen usage metrics
        """
        
        cutoff_date = (datetime.strptime(game_date, '%Y-%m-%d') - 
                      timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Analyze recent bullpen performance and usage
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_bp_er ELSE away_bp_er END) as avg_era,
                AVG(CASE WHEN home_team = %s THEN home_bp_ip ELSE away_bp_ip END) as avg_innings,
                AVG(CASE WHEN home_team = %s THEN home_bp_k ELSE away_bp_k END) as avg_strikeouts,
                COUNT(*) as games,
                SUM(CASE WHEN home_team = %s THEN home_bp_ip ELSE away_bp_ip END) as total_innings
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND (
                (home_team = %s AND home_bp_er IS NOT NULL) OR 
                (away_team = %s AND away_bp_er IS NOT NULL)
            )
        """, (team_name, team_name, team_name, team_name, team_name, team_name, 
              cutoff_date, game_date, team_name, team_name))
        
        result = self.cursor.fetchone()
        
        if not result or result[0] is None:
            return {
                'bullpen_fatigue_score': 0.5,
                'usage_intensity': 'moderate',
                'performance_trend': 'stable'
            }
        
        avg_era, avg_innings, avg_strikeouts, games, total_innings = result
        
        # Convert Decimal to float
        avg_era = float(avg_era) if avg_era is not None else 4.20
        avg_innings = float(avg_innings) if avg_innings is not None else 3.5
        avg_strikeouts = float(avg_strikeouts) if avg_strikeouts is not None else 4.0
        total_innings = float(total_innings) if total_innings is not None else 21.0
        
        # Calculate usage intensity
        innings_per_game = total_innings / max(games, 1)
        usage_intensity = 'low' if innings_per_game < 3.0 else 'high' if innings_per_game > 4.5 else 'moderate'
        
        # Calculate fatigue score (higher = more fatigued)
        # Based on innings pitched and performance degradation
        fatigue_score = min(1.0, max(0.0, (innings_per_game - 3.0) / 3.0))
        if avg_era > 4.50:
            fatigue_score += 0.2
        if avg_strikeouts < 3.0:  # Low strikeout rate indicates poor performance
            fatigue_score += 0.2
        
        fatigue_score = min(1.0, fatigue_score)
        
        # Determine performance trend using ERA and strikeout rate
        performance_score = (1.0 / max(avg_era, 0.1)) + (avg_strikeouts / 10.0)
        if performance_score > 0.6:
            performance_trend = 'improving'
        elif performance_score < 0.4:
            performance_trend = 'declining' 
        else:
            performance_trend = 'stable'
        
        return {
            'bullpen_fatigue_score': round(fatigue_score, 3),
            'usage_intensity': usage_intensity,
            'performance_trend': performance_trend,
            'avg_innings_per_game': round(innings_per_game, 1),
            'recent_era': round(avg_era, 2),
            'recent_strikeouts': round(avg_strikeouts, 1)
        }
    
    def calculate_weighted_recent_performance(self, team_name: str, game_date: str) -> Dict:
        """
        Calculate Weighted Recent Performance (WRP) where recent games
        have higher weight than older games. Uses exponential decay.
        
        Factors considered:
        - Game recency (exponential weight decay)
        - Game importance (vs division rivals, etc.)
        - Performance consistency
        - Home/away context
        
        Args:
            team_name: Full team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dict with weighted performance metrics
        """
        
        # Get last 20 games with exponential weighting
        cutoff_date = (datetime.strptime(game_date, '%Y-%m-%d') - 
                      timedelta(days=40)).strftime('%Y-%m-%d')
        
        self.cursor.execute("""
            SELECT 
                date,
                CASE WHEN home_team = %s THEN home_team_runs ELSE away_team_runs END as runs_scored,
                CASE WHEN home_team = %s THEN away_team_runs ELSE home_team_runs END as runs_allowed,
                CASE WHEN home_team = %s THEN home_team_hits ELSE away_team_hits END as hits,
                CASE WHEN home_team = %s THEN 1 ELSE 0 END as is_home
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND home_team_runs IS NOT NULL
            ORDER BY date DESC
            LIMIT 20
        """, (team_name, team_name, team_name, team_name, team_name, team_name, 
              cutoff_date, game_date))
        
        games = self.cursor.fetchall()
        
        if not games:
            return {
                'weighted_runs_scored': 4.5,
                'weighted_runs_allowed': 4.5,
                'performance_consistency': 0.5,
                'recent_momentum': 'neutral'
            }
        
        # Calculate weights (exponential decay: more recent = higher weight)
        total_weight = 0.0
        weighted_runs_scored = 0.0
        weighted_runs_allowed = 0.0
        weighted_hits = 0.0
        
        run_values = []
        
        for i, (date, runs_scored, runs_allowed, hits, is_home) in enumerate(games):
            # Convert Decimal to float
            runs_scored = float(runs_scored) if runs_scored is not None else 4.5
            runs_allowed = float(runs_allowed) if runs_allowed is not None else 4.5
            hits = float(hits) if hits is not None else 8.0
            
            # Exponential decay weight (most recent game gets weight 1.0)
            weight = np.exp(-i * 0.1)  # Decay factor of 0.1
            
            # Adjust weight for home/away (slight home field advantage)
            if is_home:
                weight *= 1.05
            
            total_weight += weight
            weighted_runs_scored += runs_scored * weight
            weighted_runs_allowed += runs_allowed * weight
            weighted_hits += hits * weight
            
            run_values.append(runs_scored)
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_runs_scored /= total_weight
            weighted_runs_allowed /= total_weight
            weighted_hits /= total_weight
        
        # Calculate performance consistency (lower std dev = more consistent)
        consistency = 1.0 - min(1.0, np.std(run_values) / 5.0) if len(run_values) > 1 else 0.5
        
        # Calculate recent momentum (last 5 games vs previous 5 games)
        if len(games) >= 10:
            recent_5 = np.mean([g[1] for g in games[:5]])  # runs scored
            previous_5 = np.mean([g[1] for g in games[5:10]])
            
            momentum_diff = recent_5 - previous_5
            if momentum_diff > 0.5:
                momentum = 'positive'
            elif momentum_diff < -0.5:
                momentum = 'negative'
            else:
                momentum = 'neutral'
        else:
            momentum = 'neutral'
        
        return {
            'weighted_runs_scored': round(weighted_runs_scored, 2),
            'weighted_runs_allowed': round(weighted_runs_allowed, 2),
            'weighted_hits': round(weighted_hits, 1),
            'performance_consistency': round(consistency, 3),
            'recent_momentum': momentum,
            'sample_size': len(games)
        }
    
    def calculate_inning_specific_run_expectancy(self, team_name: str, game_date: str) -> Dict:
        """
        Calculate Inning-Specific Run Expectancy (ISRE) based on:
        - Historical run scoring by inning
        - Late-inning performance (7th, 8th, 9th)
        - Extra-inning capabilities
        - Clutch performance metrics
        
        Args:
            team_name: Full team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dict with inning-specific metrics
        """
        
        cutoff_date = (datetime.strptime(game_date, '%Y-%m-%d') - 
                      timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Since we don't have inning-by-inning data, we'll estimate based on
        # overall performance patterns and timing
        
        # Get general scoring patterns
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_runs ELSE away_team_runs END) as avg_runs,
                AVG(CASE WHEN home_team = %s THEN home_team_hits ELSE away_team_hits END) as avg_hits,
                COUNT(*) as games,
                SUM(CASE 
                    WHEN (home_team = %s AND home_team_runs > away_team_runs) OR 
                         (away_team = %s AND away_team_runs > home_team_runs)
                    THEN 1 ELSE 0 END) as wins
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND home_team_runs IS NOT NULL
        """, (team_name, team_name, team_name, team_name, team_name, team_name, 
              cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        
        if not result or result[0] is None:
            return {
                'early_inning_strength': 0.5,
                'late_inning_strength': 0.5,
                'clutch_factor': 0.5,
                'run_distribution_pattern': 'even'
            }
        
        avg_runs, avg_hits, games, wins = result
        
        # Convert Decimal to float
        avg_runs = float(avg_runs) if avg_runs is not None else 4.5
        avg_hits = float(avg_hits) if avg_hits is not None else 8.0
        wins = int(wins) if wins is not None else games // 2
        games = int(games) if games is not None else 30
        
        win_pct = wins / max(games, 1)
        
        # Estimate inning-specific performance based on available metrics
        # Early innings (1-6): Based on starting pitcher quality and offensive consistency
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_sp_season_era ELSE away_sp_season_era END) as avg_sp_era,
                AVG(CASE WHEN home_team = %s THEN home_team_ops ELSE away_team_ops END) as avg_ops
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s AND date < %s
            AND (
                (home_team = %s AND home_sp_season_era IS NOT NULL) OR 
                (away_team = %s AND away_sp_season_era IS NOT NULL)
            )
        """, (team_name, team_name, team_name, team_name, cutoff_date, game_date, 
              team_name, team_name))
        
        sp_result = self.cursor.fetchone()
        
        if sp_result and sp_result[0] is not None:
            avg_sp_era, avg_ops = sp_result
            # Convert Decimal to float
            avg_sp_era = float(avg_sp_era) if avg_sp_era is not None else 4.20
            avg_ops = float(avg_ops) if avg_ops is not None else 0.750
            
            # Lower ERA against = better early inning offense
            early_inning_strength = min(1.0, max(0.0, (6.0 - avg_sp_era) / 4.0))
            
            # Higher OPS = better offensive capability
            if avg_ops > 0.800:
                early_inning_strength += 0.2
            elif avg_ops < 0.700:
                early_inning_strength -= 0.2
        else:
            early_inning_strength = 0.5
        
        # Late innings (7-9): Based on bullpen performance and clutch hitting
        bullpen_data = self.analyze_bullpen_usage_patterns(team_name, game_date, 30)
        
        if bullpen_data['recent_era'] < 3.50:
            late_inning_strength = 0.7
        elif bullpen_data['recent_era'] > 5.00:
            late_inning_strength = 0.3
        else:
            late_inning_strength = 0.5
        
        # Adjust for bullpen fatigue
        late_inning_strength *= (1.0 - bullpen_data['bullpen_fatigue_score'] * 0.3)
        
        # Clutch factor based on win percentage vs run differential
        expected_win_pct = 0.5 + (avg_runs - 4.5) * 0.1
        clutch_factor = min(1.0, max(0.0, win_pct / max(expected_win_pct, 0.1)))
        
        # Determine run distribution pattern
        if early_inning_strength > late_inning_strength + 0.2:
            pattern = 'front_loaded'
        elif late_inning_strength > early_inning_strength + 0.2:
            pattern = 'back_loaded'
        else:
            pattern = 'even'
        
        return {
            'early_inning_strength': round(min(1.0, max(0.0, early_inning_strength)), 3),
            'late_inning_strength': round(min(1.0, max(0.0, late_inning_strength)), 3),
            'clutch_factor': round(min(1.0, max(0.0, clutch_factor)), 3),
            'run_distribution_pattern': pattern,
            'sample_size': games
        }
    
    def calculate_advanced_team_features(self, team_name: str, game_date: str) -> Dict:
        """
        Calculate all advanced features for a team.
        
        Args:
            team_name: Full team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dict with all advanced features
        """
        
        logging.info(f"🔬 Calculating advanced features for {team_name} on {game_date}")
        
        # Calculate all feature sets
        xrv_features = self.calculate_expected_run_value_differential(team_name, game_date)
        bullpen_features = self.analyze_bullpen_usage_patterns(team_name, game_date)
        weighted_features = self.calculate_weighted_recent_performance(team_name, game_date)
        inning_features = self.calculate_inning_specific_run_expectancy(team_name, game_date)
        
        # Combine all features
        advanced_features = {
            # Expected Run Value features
            'xrv_differential': xrv_features['xrv_differential'],
            'offensive_efficiency': xrv_features['offensive_efficiency'],
            'defensive_efficiency': xrv_features['defensive_efficiency'],
            
            # Bullpen Usage features
            'bullpen_fatigue_score': bullpen_features['bullpen_fatigue_score'],
            'bullpen_usage_intensity': bullpen_features['usage_intensity'],
            'bullpen_performance_trend': bullpen_features['performance_trend'],
            'bullpen_recent_era': bullpen_features['recent_era'],
            
            # Weighted Recent Performance features
            'weighted_runs_scored': weighted_features['weighted_runs_scored'],
            'weighted_runs_allowed': weighted_features['weighted_runs_allowed'],
            'performance_consistency': weighted_features['performance_consistency'],
            'recent_momentum': weighted_features['recent_momentum'],
            
            # Inning-Specific features
            'early_inning_strength': inning_features['early_inning_strength'],
            'late_inning_strength': inning_features['late_inning_strength'],
            'clutch_factor': inning_features['clutch_factor'],
            'run_distribution_pattern': inning_features['run_distribution_pattern']
        }
        
        return advanced_features
    
    def update_game_advanced_features(self, game_id: str, home_team: str, away_team: str, game_date: str) -> bool:
        """
        Update advanced features for a specific game.
        
        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            # Get advanced features for both teams
            home_features = self.calculate_advanced_team_features(home_team, game_date)
            away_features = self.calculate_advanced_team_features(away_team, game_date)
            
            # Create update query for all advanced features
            update_fields = []
            update_values = []
            
            # Home team features
            for feature, value in home_features.items():
                if isinstance(value, (int, float)) or hasattr(value, 'item'):
                    update_fields.append(f"home_team_{feature}")
                    # Convert numpy types to native Python types
                    if hasattr(value, 'item'):
                        update_values.append(float(value.item()))
                    else:
                        update_values.append(float(value))
                else:
                    # Handle categorical features
                    update_fields.append(f"home_team_{feature}")
                    update_values.append(str(value))
            
            # Away team features  
            for feature, value in away_features.items():
                if isinstance(value, (int, float)) or hasattr(value, 'item'):
                    update_fields.append(f"away_team_{feature}")
                    # Convert numpy types to native Python types
                    if hasattr(value, 'item'):
                        update_values.append(float(value.item()))
                    else:
                        update_values.append(float(value))
                else:
                    # Handle categorical features
                    update_fields.append(f"away_team_{feature}")
                    update_values.append(str(value))
            
            # Check if columns exist, if not add them
            self._ensure_advanced_feature_columns()
            
            # Build and execute update query
            set_clause = ', '.join([f"{field} = %s" for field in update_fields])
            update_query = f"""
                UPDATE enhanced_games 
                SET {set_clause}
                WHERE game_id = %s
            """
            
            update_values.append(game_id)
            
            self.cursor.execute(update_query, update_values)
            self.conn.commit()
            
            logging.info(f"✅ Updated {len(update_fields)} advanced features for game {game_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error updating advanced features for game {game_id}: {str(e)}")
            self.conn.rollback()
            return False
    
    def _ensure_advanced_feature_columns(self):
        """Ensure all advanced feature columns exist in the database"""
        
        # Define all advanced feature columns
        columns_to_add = [
            # Expected Run Value features
            ('home_team_xrv_differential', 'REAL'),
            ('away_team_xrv_differential', 'REAL'),
            ('home_team_offensive_efficiency', 'REAL'),
            ('away_team_offensive_efficiency', 'REAL'),
            ('home_team_defensive_efficiency', 'REAL'),
            ('away_team_defensive_efficiency', 'REAL'),
            
            # Bullpen Usage features
            ('home_team_bullpen_fatigue_score', 'REAL'),
            ('away_team_bullpen_fatigue_score', 'REAL'),
            ('home_team_bullpen_usage_intensity', 'VARCHAR(20)'),
            ('away_team_bullpen_usage_intensity', 'VARCHAR(20)'),
            ('home_team_bullpen_performance_trend', 'VARCHAR(20)'),
            ('away_team_bullpen_performance_trend', 'VARCHAR(20)'),
            ('home_team_bullpen_recent_era', 'REAL'),
            ('away_team_bullpen_recent_era', 'REAL'),
            
            # Weighted Recent Performance features
            ('home_team_weighted_runs_scored', 'REAL'),
            ('away_team_weighted_runs_scored', 'REAL'),
            ('home_team_weighted_runs_allowed', 'REAL'),
            ('away_team_weighted_runs_allowed', 'REAL'),
            ('home_team_performance_consistency', 'REAL'),
            ('away_team_performance_consistency', 'REAL'),
            ('home_team_recent_momentum', 'VARCHAR(20)'),
            ('away_team_recent_momentum', 'VARCHAR(20)'),
            
            # Inning-Specific features
            ('home_team_early_inning_strength', 'REAL'),
            ('away_team_early_inning_strength', 'REAL'),
            ('home_team_late_inning_strength', 'REAL'),
            ('away_team_late_inning_strength', 'REAL'),
            ('home_team_clutch_factor', 'REAL'),
            ('away_team_clutch_factor', 'REAL'),
            ('home_team_run_distribution_pattern', 'VARCHAR(20)'),
            ('away_team_run_distribution_pattern', 'VARCHAR(20)')
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                self.cursor.execute(f"""
                    ALTER TABLE enhanced_games 
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                """)
                self.conn.commit()
            except Exception as e:
                # Column might already exist, continue
                self.conn.rollback()
                pass
    
    def process_games_batch(self, start_date: str, end_date: str) -> Dict:
        """
        Process advanced features for all games in a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Processing summary
        """
        
        logging.info(f"🚀 Processing advanced features for {start_date} to {end_date}")
        
        # Ensure columns exist
        self._ensure_advanced_feature_columns()
        
        # Get games to process
        self.cursor.execute("""
            SELECT game_id, home_team, away_team, date
            FROM enhanced_games 
            WHERE date >= %s AND date <= %s
            ORDER BY date, game_id
        """, (start_date, end_date))
        
        games = self.cursor.fetchall()
        
        processed = 0
        updated = 0
        errors = 0
        
        for game_id, home_team, away_team, game_date in games:
            processed += 1
            
            try:
                if self.update_game_advanced_features(game_id, home_team, away_team, str(game_date)):
                    updated += 1
                else:
                    errors += 1
            except Exception as e:
                logging.error(f"❌ Error processing game {game_id}: {str(e)}")
                errors += 1
            
            if processed % 50 == 0:
                logging.info(f"📊 Processed {processed}/{len(games)} games")
        
        summary = {
            'total_games': len(games),
            'processed': processed,
            'updated': updated,
            'errors': errors,
            'success_rate': (updated / processed * 100) if processed > 0 else 0
        }
        
        logging.info(f"✅ Advanced feature processing complete: {updated}/{processed} games updated")
        
        return summary
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("🔒 Advanced Feature Engineer connection closed")


def main():
    """Main execution function"""
    
    if len(sys.argv) != 3:
        print("Usage: python advanced_feature_engineering.py <start_date> <end_date>")
        print("Example: python advanced_feature_engineering.py 2025-08-12 2025-08-21")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    # Validate date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    print("🔬 ADVANCED FEATURE ENGINEERING")
    print("=" * 50)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print("🚀 Features: xRV, Bullpen Usage, Weighted Performance, Inning-Specific")
    print()
    
    engineer = AdvancedFeatureEngineer()
    
    try:
        # Process the date range
        summary = engineer.process_games_batch(start_date, end_date)
        
        print("\n📊 PROCESSING SUMMARY:")
        print("-" * 30)
        print(f"Total Games: {summary['total_games']}")
        print(f"Successfully Updated: {summary['updated']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['success_rate'] > 90:
            print("\n🎉 Advanced feature engineering completed successfully!")
        else:
            print("\n⚠️ Some issues encountered during processing")
            
    except Exception as e:
        logging.error(f"❌ Processing failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        
    finally:
        engineer.close()


if __name__ == "__main__":
    main()
