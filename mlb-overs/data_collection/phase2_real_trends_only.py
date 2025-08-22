#!/usr/bin/env python3
"""
Phase 2: Real Recent Trends Data Collection - NO FALLBACKS
==========================================================

This system ONLY uses real, actual data from our comprehensive database.
NO artificial defaults, NO fallback values, NO synthetic data.

We have extensive real data sources with 96-98% completeness:
- Individual game runs, hits, OPS data  
- Historical performance going back to March 2025
- Complete team performance metrics

If real data doesn't exist for a specific lookback period, we calculate 
from available games or skip that metric entirely.

Features Collected:
- L7 Runs (Last 7 days actual runs scored/allowed)
- L14 OPS (Last 14 days actual OPS performance)  
- L20 Extended trends (Last 20 days performance)
- L30 Long-term trends (Last 30 days performance)
- Form ratings based on actual run differentials

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

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

class RealTrendsCollector:
    """
    Collects recent trends data using ONLY real, actual game data.
    No fallbacks, no defaults, no synthetic values.
    """
    
    def __init__(self):
        """Initialize with database connection"""
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        
        # Minimum games required for reliable trends
        self.min_games_l7 = 3   # At least 3 games in last 7 days
        self.min_games_l14 = 5  # At least 5 games in last 14 days
        self.min_games_l20 = 7  # At least 7 games in last 20 days
        
        logging.info("🎯 Real Trends Collector initialized - NO FALLBACKS MODE")
    
    def get_team_recent_performance(self, team_name: str, game_date: str) -> Dict:
        """
        Get actual recent performance for a team using ONLY real data.
        
        Args:
            team_name: Full team name (e.g., "Los Angeles Dodgers")
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Dict with actual performance metrics or None if insufficient data
        """
        
        # Convert game_date to datetime for calculations
        game_dt = datetime.strptime(game_date, '%Y-%m-%d')
        
        # Calculate actual lookback dates
        l7_cutoff = (game_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        l14_cutoff = (game_dt - timedelta(days=14)).strftime('%Y-%m-%d')
        l20_cutoff = (game_dt - timedelta(days=20)).strftime('%Y-%m-%d')
        l30_cutoff = (game_dt - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logging.info(f"🔍 Calculating real trends for {team_name} before {game_date}")
        
        # L7 Performance (Last 7 Days) - REAL DATA ONLY
        l7_data = self._get_actual_l7_performance(team_name, game_date, l7_cutoff)
        
        # L14 Performance (Last 14 Days) - REAL DATA ONLY  
        l14_data = self._get_actual_l14_performance(team_name, game_date, l14_cutoff)
        
        # L20 Performance (Last 20 Days) - REAL DATA ONLY
        l20_data = self._get_actual_l20_performance(team_name, game_date, l20_cutoff)
        
        # L30 Performance (Last 30 Days) - REAL DATA ONLY
        l30_data = self._get_actual_l30_performance(team_name, game_date, l30_cutoff)
        
        # Combine all real data
        performance = {
            'team_name': team_name,
            'game_date': game_date,
            'data_quality': 'REAL_ONLY'
        }
        
        # Add L7 data if sufficient games
        if l7_data and l7_data['games'] >= self.min_games_l7:
            performance.update({
                'runs_l7': l7_data['runs_avg'],
                'runs_allowed_l7': l7_data['runs_allowed_avg'],
                'l7_games': l7_data['games']
            })
            logging.info(f"✅ L7: {l7_data['games']} games, {l7_data['runs_avg']:.1f} runs")
        else:
            logging.warning(f"⚠️ L7: Insufficient data ({l7_data['games'] if l7_data else 0} games)")
        
        # Add L14 data if sufficient games
        if l14_data and l14_data['games'] >= self.min_games_l14:
            performance.update({
                'ops_l14': l14_data['ops_avg'],
                'l14_games': l14_data['games']
            })
            logging.info(f"✅ L14: {l14_data['games']} games, {l14_data['ops_avg']:.3f} OPS")
        else:
            logging.warning(f"⚠️ L14: Insufficient data ({l14_data['games'] if l14_data else 0} games)")
        
        # Add L20 data if sufficient games
        if l20_data and l20_data['games'] >= self.min_games_l20:
            performance.update({
                'runs_l20': l20_data['runs_avg'],
                'runs_allowed_l20': l20_data['runs_allowed_avg'],
                'ops_l20': l20_data['ops_avg'],
                'l20_games': l20_data['games']
            })
            logging.info(f"✅ L20: {l20_data['games']} games")
        
        # Add L30 data if available
        if l30_data and l30_data['games'] >= 10:  # Need substantial sample for L30
            performance.update({
                'runs_l30': l30_data['runs_avg'],
                'runs_allowed_l30': l30_data['runs_allowed_avg'],
                'ops_l30': l30_data['ops_avg'],
                'l30_games': l30_data['games']
            })
            logging.info(f"✅ L30: {l30_data['games']} games")
        
        # Calculate form rating ONLY if we have L7 data
        if 'runs_l7' in performance and 'runs_allowed_l7' in performance:
            run_diff = performance['runs_l7'] - performance['runs_allowed_l7']
            # Conservative form rating based on actual performance
            form_rating = max(1.0, min(10.0, 5.0 + (run_diff * 1.5)))
            performance['form_rating'] = round(form_rating, 1)
        
        return performance
    
    def _get_actual_l7_performance(self, team_name: str, game_date: str, cutoff_date: str) -> Optional[Dict]:
        """Get actual L7 performance from real games"""
        
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_runs 
                         ELSE away_team_runs END) as runs_scored,
                AVG(CASE WHEN home_team = %s THEN away_team_runs 
                         ELSE home_team_runs END) as runs_allowed,
                COUNT(*) as games_played,
                MIN(date) as earliest_game,
                MAX(date) as latest_game
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s
            AND date < %s
            AND home_team_runs IS NOT NULL
            AND away_team_runs IS NOT NULL
        """, (team_name, team_name, team_name, team_name, cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        
        if result and result[0] is not None and result[2] > 0:
            return {
                'runs_avg': round(float(result[0]), 1),
                'runs_allowed_avg': round(float(result[1]), 1),
                'games': int(result[2]),
                'date_range': f"{result[3]} to {result[4]}"
            }
        
        return None
    
    def _get_actual_l14_performance(self, team_name: str, game_date: str, cutoff_date: str) -> Optional[Dict]:
        """Get actual L14 OPS performance from real games"""
        
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_ops 
                         ELSE away_team_ops END) as ops_avg,
                COUNT(*) as games_played
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s
            AND date < %s
            AND home_team_ops IS NOT NULL
            AND away_team_ops IS NOT NULL
        """, (team_name, team_name, team_name, cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        
        if result and result[0] is not None and result[1] > 0:
            return {
                'ops_avg': round(float(result[0]), 3),
                'games': int(result[1])
            }
        
        return None
    
    def _get_actual_l20_performance(self, team_name: str, game_date: str, cutoff_date: str) -> Optional[Dict]:
        """Get actual L20 comprehensive performance from real games"""
        
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_runs 
                         ELSE away_team_runs END) as runs_scored,
                AVG(CASE WHEN home_team = %s THEN away_team_runs 
                         ELSE home_team_runs END) as runs_allowed,
                AVG(CASE WHEN home_team = %s THEN home_team_ops 
                         ELSE away_team_ops END) as ops_avg,
                COUNT(*) as games_played
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s
            AND date < %s
            AND home_team_runs IS NOT NULL
            AND home_team_ops IS NOT NULL
        """, (team_name, team_name, team_name, team_name, team_name, cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        
        if result and result[0] is not None and result[3] > 0:
            return {
                'runs_avg': round(float(result[0]), 1),
                'runs_allowed_avg': round(float(result[1]), 1),
                'ops_avg': round(float(result[2]), 3),
                'games': int(result[3])
            }
        
        return None
    
    def _get_actual_l30_performance(self, team_name: str, game_date: str, cutoff_date: str) -> Optional[Dict]:
        """Get actual L30 comprehensive performance from real games"""
        
        self.cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_team_runs 
                         ELSE away_team_runs END) as runs_scored,
                AVG(CASE WHEN home_team = %s THEN away_team_runs 
                         ELSE home_team_runs END) as runs_allowed,
                AVG(CASE WHEN home_team = %s THEN home_team_ops 
                         ELSE away_team_ops END) as ops_avg,
                COUNT(*) as games_played
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date >= %s
            AND date < %s
            AND home_team_runs IS NOT NULL
            AND home_team_ops IS NOT NULL
        """, (team_name, team_name, team_name, team_name, team_name, cutoff_date, game_date))
        
        result = self.cursor.fetchone()
        
        if result and result[0] is not None and result[3] > 0:
            return {
                'runs_avg': round(float(result[0]), 1),
                'runs_allowed_avg': round(float(result[1]), 1),
                'ops_avg': round(float(result[2]), 3),
                'games': int(result[3])
            }
        
        return None
    
    def update_game_trends(self, game_id: str, home_team: str, away_team: str, game_date: str) -> bool:
        """
        Update recent trends for a specific game using ONLY real data.
        
        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            True if update successful, False otherwise
        """
        
        try:
            # Get real performance for both teams
            home_perf = self.get_team_recent_performance(home_team, game_date)
            away_perf = self.get_team_recent_performance(away_team, game_date)
            
            # Prepare update values - ONLY include data we actually have
            update_values = []
            update_columns = []
            
            # Home team L7 data (only if real data available)
            if 'runs_l7' in home_perf:
                update_columns.extend(['home_team_runs_l7', 'home_team_runs_allowed_l7'])
                update_values.extend([home_perf['runs_l7'], home_perf['runs_allowed_l7']])
            
            # Away team L7 data (only if real data available)
            if 'runs_l7' in away_perf:
                update_columns.extend(['away_team_runs_l7', 'away_team_runs_allowed_l7'])
                update_values.extend([away_perf['runs_l7'], away_perf['runs_allowed_l7']])
            
            # Home team L14 OPS (only if real data available)
            if 'ops_l14' in home_perf:
                update_columns.append('home_team_ops_l14')
                update_values.append(home_perf['ops_l14'])
            
            # Away team L14 OPS (only if real data available)
            if 'ops_l14' in away_perf:
                update_columns.append('away_team_ops_l14')
                update_values.append(away_perf['ops_l14'])
            
            # Home team L20 data (only if real data available)
            if 'runs_l20' in home_perf:
                update_columns.extend(['home_team_runs_l20', 'home_team_runs_allowed_l20', 'home_team_ops_l20'])
                update_values.extend([home_perf['runs_l20'], home_perf['runs_allowed_l20'], home_perf['ops_l20']])
            
            # Away team L20 data (only if real data available)
            if 'runs_l20' in away_perf:
                update_columns.extend(['away_team_runs_l20', 'away_team_runs_allowed_l20', 'away_team_ops_l20'])
                update_values.extend([away_perf['runs_l20'], away_perf['runs_allowed_l20'], away_perf['ops_l20']])
            
            # Home team L30 data (only if real data available)
            if 'runs_l30' in home_perf:
                update_columns.extend(['home_team_runs_l30', 'home_team_ops_l30'])
                update_values.extend([home_perf['runs_l30'], home_perf['ops_l30']])
            
            # Away team L30 data (only if real data available)
            if 'runs_l30' in away_perf:
                update_columns.extend(['away_team_runs_l30', 'away_team_ops_l30'])
                update_values.extend([away_perf['runs_l30'], away_perf['ops_l30']])
            
            # Form ratings (only if L7 data available)
            if 'form_rating' in home_perf:
                update_columns.append('home_team_form_rating')
                update_values.append(home_perf['form_rating'])
            
            if 'form_rating' in away_perf:
                update_columns.append('away_team_form_rating')
                update_values.append(away_perf['form_rating'])
            
            # Only proceed if we have some real data to update
            if not update_columns:
                logging.warning(f"⚠️ No real trend data available for {home_team} vs {away_team} on {game_date}")
                return False
            
            # Build and execute update query
            set_clause = ', '.join([f"{col} = %s" for col in update_columns])
            update_query = f"""
                UPDATE enhanced_games 
                SET {set_clause}
                WHERE game_id = %s
            """
            
            update_values.append(game_id)
            
            self.cursor.execute(update_query, update_values)
            self.conn.commit()
            
            logging.info(f"✅ Updated {len(update_columns)} real trend fields for game {game_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error updating trends for game {game_id}: {str(e)}")
            self.conn.rollback()
            return False
    
    def process_recent_games(self, start_date: str, end_date: str) -> Dict:
        """
        Process recent trends for all games in date range using ONLY real data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Processing summary statistics
        """
        
        logging.info(f"🚀 Processing real trends for {start_date} to {end_date}")
        
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
        skipped = 0
        
        for game_id, home_team, away_team, game_date in games:
            processed += 1
            
            if self.update_game_trends(game_id, home_team, away_team, str(game_date)):
                updated += 1
            else:
                skipped += 1
            
            if processed % 100 == 0:
                logging.info(f"📊 Processed {processed}/{len(games)} games")
        
        summary = {
            'total_games': len(games),
            'processed': processed,
            'updated': updated,
            'skipped': skipped,
            'success_rate': (updated / processed * 100) if processed > 0 else 0
        }
        
        logging.info(f"✅ Processing complete: {updated}/{processed} games updated ({summary['success_rate']:.1f}% success)")
        
        return summary
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("🔒 Database connection closed")


def main():
    """Main execution function"""
    
    if len(sys.argv) != 3:
        print("Usage: python phase2_real_trends_only.py <start_date> <end_date>")
        print("Example: python phase2_real_trends_only.py 2025-08-12 2025-08-21")
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
    
    print("🎯 PHASE 2: REAL TRENDS DATA COLLECTION")
    print("=" * 50)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print("🚫 NO FALLBACKS - REAL DATA ONLY")
    print()
    
    collector = RealTrendsCollector()
    
    try:
        # Process the date range
        summary = collector.process_recent_games(start_date, end_date)
        
        print("\n📊 PROCESSING SUMMARY:")
        print("-" * 30)
        print(f"Total Games: {summary['total_games']}")
        print(f"Successfully Updated: {summary['updated']}")
        print(f"Skipped (No Data): {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['success_rate'] < 80:
            print("\n⚠️ WARNING: Low success rate indicates data quality issues")
        else:
            print("\n✅ High success rate - real data collection working well")
            
    except Exception as e:
        logging.error(f"❌ Processing failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        
    finally:
        collector.close()


if __name__ == "__main__":
    main()
