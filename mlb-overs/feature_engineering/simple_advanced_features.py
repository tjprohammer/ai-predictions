#!/usr/bin/env python3
"""
🔬 SIMPLE ADVANCED FEATURE ENGINEERING
Just populate the columns that we've already added successfully
"""

import sys
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import logging

# Configure logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleAdvancedFeaturePopulator:
    def __init__(self):
        self.conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        logger.info("🔬 Simple Advanced Feature Populator initialized")

    def close(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("🔒 Simple Advanced Feature Populator connection closed")

    def populate_simple_features(self, start_date: str, end_date: str):
        """
        Populate simple advanced features using basic calculations
        """
        logger.info(f"🚀 Processing simple advanced features for {start_date} to {end_date}")
        
        # Get all games in the date range
        self.cursor.execute("""
            SELECT game_id, date, home_team, away_team,
                   home_team_runs, away_team_runs, total_runs,
                   home_team_ops, away_team_ops,
                   home_bp_er, away_bp_er, home_bp_ip, away_bp_ip
            FROM enhanced_games 
            WHERE date >= %s AND date <= %s
            ORDER BY date, game_id
        """, (start_date, end_date))
        
        games = self.cursor.fetchall()
        logger.info(f"Found {len(games)} games to process")
        
        updated_count = 0
        error_count = 0
        
        for i, game in enumerate(games):
            try:
                game_id = game['game_id']
                date = game['date']
                home_team = game['home_team']
                away_team = game['away_team']
                
                logger.info(f"🔬 Processing game {game_id}: {away_team} @ {home_team} on {date}")
                
                # Calculate simple features
                features = self.calculate_simple_features(game)
                
                # Update the database
                self.update_game_features(game_id, features)
                updated_count += 1
                
                if (i + 1) % 25 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(games)} games")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error processing game {game_id}: {e}")
        
        logger.info(f"✅ Simple advanced feature processing complete: {updated_count}/{len(games)} games updated")
        logger.info(f"📊 PROCESSING SUMMARY:")
        logger.info(f"------------------------------")
        logger.info(f"Total Games: {len(games)}")
        logger.info(f"Successfully Updated: {updated_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Success Rate: {(updated_count / len(games) * 100):.1f}%")

    def calculate_simple_features(self, game) -> dict:
        """
        Calculate simple advanced features using basic math
        """
        home_runs = float(game['home_team_runs']) if game['home_team_runs'] else 0.0
        away_runs = float(game['away_team_runs']) if game['away_team_runs'] else 0.0
        total_runs = float(game['total_runs']) if game['total_runs'] else home_runs + away_runs
        
        home_ops = float(game['home_team_ops']) if game['home_team_ops'] else 0.750
        away_ops = float(game['away_team_ops']) if game['away_team_ops'] else 0.750
        
        home_bp_er = float(game['home_bp_er']) if game['home_bp_er'] else 2.0
        away_bp_er = float(game['away_bp_er']) if game['away_bp_er'] else 2.0
        home_bp_ip = float(game['home_bp_ip']) if game['home_bp_ip'] else 3.0
        away_bp_ip = float(game['away_bp_ip']) if game['away_bp_ip'] else 3.0
        
        # Simple feature calculations
        features = {
            # Expected Run Value Differential (simplified)
            'home_team_xrv_differential': round(home_ops - 0.750, 3),  # Difference from league average
            'away_team_xrv_differential': round(away_ops - 0.750, 3),
            
            # Offensive/Defensive Efficiency (simplified)
            'home_team_offensive_efficiency': round(min(1.5, max(0.5, home_ops / 0.750)), 3),
            'away_team_offensive_efficiency': round(min(1.5, max(0.5, away_ops / 0.750)), 3),
            'home_team_defensive_efficiency': round(min(1.5, max(0.5, 1.0 - (home_bp_er / 4.5))), 3),
            'away_team_defensive_efficiency': round(min(1.5, max(0.5, 1.0 - (away_bp_er / 4.5))), 3),
            
            # Bullpen Fatigue (simplified)
            'home_team_bullpen_fatigue_score': round(min(1.0, max(0.0, home_bp_ip / 5.0)), 3),
            'away_team_bullpen_fatigue_score': round(min(1.0, max(0.0, away_bp_ip / 5.0)), 3),
            
            # Usage Intensity (simplified categories)
            'home_team_bullpen_usage_intensity': 'high' if home_bp_ip > 4.5 else 'low' if home_bp_ip < 2.5 else 'moderate',
            'away_team_bullpen_usage_intensity': 'high' if away_bp_ip > 4.5 else 'low' if away_bp_ip < 2.5 else 'moderate',
            
            # Performance Trend (simplified)
            'home_team_bullpen_performance_trend': 'improving' if home_bp_er < 3.0 else 'declining' if home_bp_er > 5.0 else 'stable',
            'away_team_bullpen_performance_trend': 'improving' if away_bp_er < 3.0 else 'declining' if away_bp_er > 5.0 else 'stable',
            
            # Recent ERA (use game data as proxy)
            'home_team_bullpen_recent_era': round(home_bp_er, 2),
            'away_team_bullpen_recent_era': round(away_bp_er, 2),
            
            # Weighted Performance (simplified - use OPS as proxy)
            'home_team_weighted_runs_scored': round(home_runs * (home_ops / 0.750), 1),
            'away_team_weighted_runs_scored': round(away_runs * (away_ops / 0.750), 1),
            'home_team_weighted_runs_allowed': round(away_runs * (1.0 - (home_ops - 0.750) / 2.0), 1),
            'away_team_weighted_runs_allowed': round(home_runs * (1.0 - (away_ops - 0.750) / 2.0), 1),
            
            # Performance Consistency (simplified)
            'home_team_performance_consistency': round(min(1.0, max(0.0, 1.0 - abs(home_ops - 0.750))), 3),
            'away_team_performance_consistency': round(min(1.0, max(0.0, 1.0 - abs(away_ops - 0.750))), 3),
            
            # Recent Momentum (simplified)
            'home_team_recent_momentum': 'positive' if home_runs > away_runs else 'negative' if home_runs < away_runs else 'neutral',
            'away_team_recent_momentum': 'positive' if away_runs > home_runs else 'negative' if away_runs < home_runs else 'neutral',
            
            # Inning Strength (simplified using OPS)
            'home_team_early_inning_strength': round(home_ops * 0.9, 3),  # Slight discount for early
            'away_team_early_inning_strength': round(away_ops * 0.9, 3),
            'home_team_late_inning_strength': round(home_ops * 1.1, 3),   # Slight boost for late
            'away_team_late_inning_strength': round(away_ops * 1.1, 3),
            
            # Clutch Factor (simplified)
            'home_team_clutch_factor': round(min(1.5, max(0.5, home_ops / 0.750 * (1.0 + (home_runs / 10.0)))), 3),
            'away_team_clutch_factor': round(min(1.5, max(0.5, away_ops / 0.750 * (1.0 + (away_runs / 10.0)))), 3),
            
            # Run Distribution Pattern (simplified)
            'home_team_run_distribution_pattern': 'explosive' if home_runs > 8 else 'steady' if home_runs > 4 else 'limited',
            'away_team_run_distribution_pattern': 'explosive' if away_runs > 8 else 'steady' if away_runs > 4 else 'limited'
        }
        
        return features

    def update_game_features(self, game_id: str, features: dict):
        """
        Update game with calculated features
        """
        update_query = """
            UPDATE enhanced_games SET
                home_team_xrv_differential = %s,
                away_team_xrv_differential = %s,
                home_team_offensive_efficiency = %s,
                away_team_offensive_efficiency = %s,
                home_team_defensive_efficiency = %s,
                away_team_defensive_efficiency = %s,
                home_team_bullpen_fatigue_score = %s,
                away_team_bullpen_fatigue_score = %s,
                home_team_bullpen_usage_intensity = %s,
                away_team_bullpen_usage_intensity = %s,
                home_team_bullpen_performance_trend = %s,
                away_team_bullpen_performance_trend = %s,
                home_team_bullpen_recent_era = %s,
                away_team_bullpen_recent_era = %s,
                home_team_weighted_runs_scored = %s,
                away_team_weighted_runs_scored = %s,
                home_team_weighted_runs_allowed = %s,
                away_team_weighted_runs_allowed = %s,
                home_team_performance_consistency = %s,
                away_team_performance_consistency = %s,
                home_team_recent_momentum = %s,
                away_team_recent_momentum = %s,
                home_team_early_inning_strength = %s,
                away_team_early_inning_strength = %s,
                home_team_late_inning_strength = %s,
                away_team_late_inning_strength = %s,
                home_team_clutch_factor = %s,
                away_team_clutch_factor = %s,
                home_team_run_distribution_pattern = %s,
                away_team_run_distribution_pattern = %s
            WHERE game_id = %s
        """
        
        self.cursor.execute(update_query, (
            features['home_team_xrv_differential'],
            features['away_team_xrv_differential'], 
            features['home_team_offensive_efficiency'],
            features['away_team_offensive_efficiency'],
            features['home_team_defensive_efficiency'],
            features['away_team_defensive_efficiency'],
            features['home_team_bullpen_fatigue_score'],
            features['away_team_bullpen_fatigue_score'],
            features['home_team_bullpen_usage_intensity'],
            features['away_team_bullpen_usage_intensity'],
            features['home_team_bullpen_performance_trend'],
            features['away_team_bullpen_performance_trend'],
            features['home_team_bullpen_recent_era'],
            features['away_team_bullpen_recent_era'],
            features['home_team_weighted_runs_scored'],
            features['away_team_weighted_runs_scored'],
            features['home_team_weighted_runs_allowed'],
            features['away_team_weighted_runs_allowed'],
            features['home_team_performance_consistency'],
            features['away_team_performance_consistency'],
            features['home_team_recent_momentum'],
            features['away_team_recent_momentum'],
            features['home_team_early_inning_strength'],
            features['away_team_early_inning_strength'],
            features['home_team_late_inning_strength'],
            features['away_team_late_inning_strength'],
            features['home_team_clutch_factor'],
            features['away_team_clutch_factor'],
            features['home_team_run_distribution_pattern'],
            features['away_team_run_distribution_pattern'],
            game_id
        ))
        
        self.conn.commit()

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_advanced_features.py <start_date> <end_date>")
        print("Example: python simple_advanced_features.py 2025-08-12 2025-08-21")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    print("🔬 SIMPLE ADVANCED FEATURE ENGINEERING")
    print("=" * 50)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print("🚀 Features: Simplified xRV, Bullpen, Performance, Momentum")
    print()
    
    try:
        populator = SimpleAdvancedFeaturePopulator()
        populator.populate_simple_features(start_date, end_date)
        populator.close()
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
