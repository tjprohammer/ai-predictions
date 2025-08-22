#!/usr/bin/env python3
"""
Comprehensive Offensive Statistics Backfill
===========================================
Backfills missing offensive metrics (OBP, SLG, ISO, wOPS, etc.) from MLB Stats API
NO FAKE DATA - Only real MLB statistics from official sources
"""

import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OffensiveStatsBackfill:
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MLB-Predictions-Research/1.0'
        })
        
        # Database connection
        self.conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        
        # Team ID mapping from MLB API
        self.team_mappings = {}
        self._initialize_team_mappings()
    
    def _initialize_team_mappings(self):
        """Get team ID mappings from MLB API"""
        try:
            response = self.session.get(f"{self.base_url}/teams", timeout=10)
            if response.status_code == 200:
                teams = response.json()['teams']
                for team in teams:
                    self.team_mappings[team['abbreviation']] = team['id']
                    self.team_mappings[team['teamName']] = team['id']
                    self.team_mappings[team['name']] = team['id']
                logger.info(f"Loaded {len(teams)} team mappings")
            else:
                logger.error(f"Failed to load team mappings: {response.status_code}")
        except Exception as e:
            logger.error(f"Error loading team mappings: {e}")
    
    def get_team_id_from_name(self, team_name: str) -> Optional[int]:
        """Convert team name to MLB API team ID"""
        # Try exact match first
        if team_name in self.team_mappings:
            return self.team_mappings[team_name]
        
        # Try partial matches
        for name, team_id in self.team_mappings.items():
            if team_name.lower() in name.lower() or name.lower() in team_name.lower():
                return team_id
        
        logger.warning(f"Could not find team ID for: {team_name}")
        return None
    
    def get_team_season_stats(self, team_id: int, season: int, as_of_date: str = None) -> Optional[Dict]:
        """Get comprehensive team offensive stats from MLB API"""
        try:
            url = f"{self.base_url}/teams/{team_id}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'hitting'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to get stats for team {team_id}: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('stats') or not data['stats'][0].get('splits'):
                logger.warning(f"No stats found for team {team_id}")
                return None
            
            stats = data['stats'][0]['splits'][0]['stat']
            
            # Calculate all offensive metrics
            at_bats = int(stats.get('atBats', 0))
            hits = int(stats.get('hits', 0))
            doubles = int(stats.get('doubles', 0))
            triples = int(stats.get('triples', 0))
            home_runs = int(stats.get('homeRuns', 0))
            walks = int(stats.get('baseOnBalls', 0))
            hit_by_pitch = int(stats.get('hitByPitch', 0))
            sacrifice_flies = int(stats.get('sacFlies', 0))
            
            # Calculate advanced metrics
            avg = float(stats.get('avg', 0))
            obp = float(stats.get('obp', 0))
            slg = float(stats.get('slg', 0))
            ops = float(stats.get('ops', 0))
            
            # Calculate ISO (Isolated Power)
            iso = slg - avg if slg > 0 and avg > 0 else 0
            
            # Calculate wOBA (weighted On-Base Average) - approximation
            # wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR) / (AB + BB + HBP + SF)
            singles = hits - doubles - triples - home_runs
            plate_appearances = at_bats + walks + hit_by_pitch + sacrifice_flies
            
            if plate_appearances > 0:
                woba = (0.69*walks + 0.72*hit_by_pitch + 0.89*singles + 
                       1.27*doubles + 1.62*triples + 2.10*home_runs) / plate_appearances
            else:
                woba = 0
            
            # wRC+ approximation (league average wOBA ≈ 0.320)
            league_woba = 0.320
            wrc_plus = (woba / league_woba * 100) if league_woba > 0 else 100
            
            return {
                'batting_avg': avg,
                'on_base_pct': obp,
                'slugging_pct': slg,
                'ops': ops,
                'iso': iso,
                'woba': woba,
                'wrc_plus': wrc_plus,
                'runs': int(stats.get('runs', 0)),
                'rbi': int(stats.get('rbi', 0)),
                'hits': hits,
                'home_runs': home_runs,
                'stolen_bases': int(stats.get('stolenBases', 0)),
                'strikeouts': int(stats.get('strikeOuts', 0)),
                'walks': walks,
                'plate_appearances': plate_appearances,
                'at_bats': at_bats,
                'games_played': int(stats.get('gamesPlayed', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting team stats for {team_id}: {e}")
            return None
    
    def add_missing_columns(self):
        """Add missing offensive metric columns to enhanced_games table"""
        cursor = self.conn.cursor()
        
        columns_to_add = [
            ('home_team_obp', 'DECIMAL(4,3)'),
            ('away_team_obp', 'DECIMAL(4,3)'),
            ('home_team_slg', 'DECIMAL(4,3)'),
            ('away_team_slg', 'DECIMAL(4,3)'),
            ('home_team_ops', 'DECIMAL(4,3)'),
            ('away_team_ops', 'DECIMAL(4,3)'),
            ('home_team_iso', 'DECIMAL(4,3)'),
            ('away_team_iso', 'DECIMAL(4,3)'),
            ('home_team_woba', 'DECIMAL(4,3)'),
            ('away_team_woba', 'DECIMAL(4,3)'),
            ('home_team_wrc_plus', 'DECIMAL(5,1)'),
            ('away_team_wrc_plus', 'DECIMAL(5,1)'),
            ('home_team_stolen_bases', 'INTEGER'),
            ('away_team_stolen_bases', 'INTEGER'),
            ('home_team_plate_appearances', 'INTEGER'),
            ('away_team_plate_appearances', 'INTEGER'),
            ('combined_team_ops', 'DECIMAL(4,3)'),
            ('combined_team_woba', 'DECIMAL(4,3)'),
            ('offensive_environment_score', 'DECIMAL(4,3)')
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                cursor.execute(f"""
                    ALTER TABLE enhanced_games 
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                """)
                logger.info(f"Added column: {column_name}")
            except Exception as e:
                logger.warning(f"Column {column_name} might already exist: {e}")
        
        self.conn.commit()
    
    def get_games_needing_backfill(self) -> List[Tuple]:
        """Get games that need offensive stats backfill"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT game_id, date, home_team, away_team, home_team_id, away_team_id
            FROM enhanced_games 
            WHERE date >= '2025-03-20'
            AND date <= '2025-08-21'
            AND (home_team_obp IS NULL OR away_team_obp IS NULL)
            ORDER BY date ASC
        """)
        
        return cursor.fetchall()
    
    def backfill_game_offensive_stats(self, game_id: str, date, home_team: str, 
                                    away_team: str, home_team_id: int, away_team_id: int):
        """Backfill offensive stats for a single game"""
        # Convert date to string if it's a date object
        if hasattr(date, 'year'):
            season = date.year
            date_str = date.strftime('%Y-%m-%d')
        else:
            season = int(str(date)[:4])
            date_str = str(date)
        
        # Get team stats from MLB API
        home_stats = None
        away_stats = None
        
        if home_team_id:
            home_stats = self.get_team_season_stats(home_team_id, season, date_str)
        else:
            # Try to get team ID from team name
            team_id = self.get_team_id_from_name(home_team)
            if team_id:
                home_stats = self.get_team_season_stats(team_id, season, date_str)
        
        if away_team_id:
            away_stats = self.get_team_season_stats(away_team_id, season, date_str)
        else:
            # Try to get team ID from team name
            team_id = self.get_team_id_from_name(away_team)
            if team_id:
                away_stats = self.get_team_season_stats(team_id, season, date_str)
        
        if not home_stats or not away_stats:
            logger.warning(f"Could not get complete stats for game {game_id} ({home_team} vs {away_team})")
            return False
        
        # Calculate combined metrics
        combined_ops = (home_stats['ops'] + away_stats['ops']) / 2
        combined_woba = (home_stats['woba'] + away_stats['woba']) / 2
        offensive_environment = combined_woba  # Primary offensive environment indicator
        
        # Update database
        cursor = self.conn.cursor()
        
        update_query = """
            UPDATE enhanced_games 
            SET 
                home_team_obp = %s,
                away_team_obp = %s,
                home_team_slg = %s,
                away_team_slg = %s,
                home_team_ops = %s,
                away_team_ops = %s,
                home_team_iso = %s,
                away_team_iso = %s,
                home_team_woba = %s,
                away_team_woba = %s,
                home_team_wrc_plus = %s,
                away_team_wrc_plus = %s,
                home_team_stolen_bases = %s,
                away_team_stolen_bases = %s,
                home_team_plate_appearances = %s,
                away_team_plate_appearances = %s,
                combined_team_ops = %s,
                combined_team_woba = %s,
                offensive_environment_score = %s
            WHERE game_id = %s
        """
        
        values = (
            home_stats['on_base_pct'],
            away_stats['on_base_pct'],
            home_stats['slugging_pct'],
            away_stats['slugging_pct'],
            home_stats['ops'],
            away_stats['ops'],
            home_stats['iso'],
            away_stats['iso'],
            home_stats['woba'],
            away_stats['woba'],
            home_stats['wrc_plus'],
            away_stats['wrc_plus'],
            home_stats['stolen_bases'],
            away_stats['stolen_bases'],
            home_stats['plate_appearances'],
            away_stats['plate_appearances'],
            combined_ops,
            combined_woba,
            offensive_environment,
            game_id
        )
        
        cursor.execute(update_query, values)
        self.conn.commit()
        
        logger.info(f"✅ Updated game {game_id}: {home_team} (OPS: {home_stats['ops']:.3f}) vs {away_team} (OPS: {away_stats['ops']:.3f})")
        return True
    
    def run_comprehensive_backfill(self):
        """Run comprehensive offensive stats backfill"""
        logger.info("🔄 Starting comprehensive offensive statistics backfill...")
        
        # Add missing columns
        self.add_missing_columns()
        
        # Get games needing backfill
        games_to_process = self.get_games_needing_backfill()
        logger.info(f"Found {len(games_to_process)} games needing offensive stats backfill")
        
        successful_updates = 0
        failed_updates = 0
        
        for i, (game_id, date, home_team, away_team, home_team_id, away_team_id) in enumerate(games_to_process):
            try:
                if self.backfill_game_offensive_stats(game_id, date, home_team, away_team, home_team_id, away_team_id):
                    successful_updates += 1
                else:
                    failed_updates += 1
                
                # Progress update
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(games_to_process)} games processed")
                
                # Rate limiting - respect MLB API
                time.sleep(0.2)  # 200ms delay between requests
                
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                failed_updates += 1
        
        logger.info(f"🎯 Backfill complete!")
        logger.info(f"   ✅ Successful updates: {successful_updates}")
        logger.info(f"   ❌ Failed updates: {failed_updates}")
        logger.info(f"   📊 Success rate: {successful_updates/(successful_updates+failed_updates)*100:.1f}%")
        
        # Validate results
        self.validate_backfill_results()
    
    def validate_backfill_results(self):
        """Validate the backfill results"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_games,
                COUNT(home_team_obp) as with_obp,
                COUNT(home_team_slg) as with_slg,
                COUNT(home_team_ops) as with_ops,
                COUNT(home_team_iso) as with_iso,
                COUNT(home_team_woba) as with_woba,
                COUNT(combined_team_ops) as with_combined_ops
            FROM enhanced_games 
            WHERE date >= '2025-03-20' AND date <= '2025-08-21'
        """)
        
        total, obp, slg, ops, iso, woba, combined = cursor.fetchone()
        
        logger.info("📊 BACKFILL VALIDATION RESULTS:")
        logger.info(f"   Total games: {total}")
        logger.info(f"   OBP coverage: {obp} ({obp/total*100:.1f}%)")
        logger.info(f"   SLG coverage: {slg} ({slg/total*100:.1f}%)")
        logger.info(f"   OPS coverage: {ops} ({ops/total*100:.1f}%)")
        logger.info(f"   ISO coverage: {iso} ({iso/total*100:.1f}%)")
        logger.info(f"   wOBA coverage: {woba} ({woba/total*100:.1f}%)")
        logger.info(f"   Combined OPS coverage: {combined} ({combined/total*100:.1f}%)")
        
        # Sample validation
        cursor.execute("""
            SELECT home_team, away_team, home_team_ops, away_team_ops, combined_team_ops, home_team_woba, away_team_woba
            FROM enhanced_games 
            WHERE date = '2025-08-20'
            AND home_team_ops IS NOT NULL
            LIMIT 3
        """)
        
        logger.info("📋 SAMPLE BACKFILLED DATA:")
        for row in cursor.fetchall():
            home, away, h_ops, a_ops, combined, h_woba, a_woba = row
            logger.info(f"   {home} vs {away}: OPS {h_ops:.3f}-{a_ops:.3f} (avg: {combined:.3f}), wOBA {h_woba:.3f}-{a_woba:.3f}")

def main():
    """Main execution function"""
    backfiller = OffensiveStatsBackfill()
    backfiller.run_comprehensive_backfill()

if __name__ == "__main__":
    main()
