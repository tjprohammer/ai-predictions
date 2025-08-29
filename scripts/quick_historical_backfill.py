#!/usr/bin/env python3
"""
ENHANCED HISTORICAL DATA BACKFILL
================================

High-ROI data enhancements based on systematic analysis:

PRIORITY 1 (Immediate Impact):
1. Weather + park orientation (huge for totals) - wind_out_cf, air_density, roof_state
2. Bullpen fatigue & availability (very predictive after ASB)
3. Opener/bulk & starter workload signals

PRIORITY 2 (Strong Signals):
4. Lineup quality & uncertainty (don't overfit names)
5. Umpire run environment 
6. Travel, rest & scheduling context

PRIORITY 3 (Incremental):
7. Defense & catcher effects
8. Market microstructure (for EV gating)

Implements complete schema and feature transforms for production system.
"""

import psycopg2
import pandas as pd
import requests
import sys
import time
import os
import math
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedHistoricalBackfill:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        self.api_base = "https://statsapi.mlb.com/api/v1"
        self.weather_api_key = os.getenv('WEATHER_API_KEY', '')  # Need weather API key
        
        # Park azimuth data (home plate → center field degrees)
        self.park_azimuth = {
            'Fenway Park': 50,
            'Yankee Stadium': 60,
            'Camden Yards': 70,
            'Tropicana Field': 40,
            'Rogers Centre': 85,
            'Progressive Field': 75,
            'Comerica Park': 95,
            'Kauffman Stadium': 90,
            'Target Field': 105,
            'Guaranteed Rate Field': 30,
            'Angel Stadium': 290,
            'Minute Maid Park': 40,  # Fixed: 400 % 360 = 40
            'Globe Life Field': 25,
            'T-Mobile Park': 45,
            'Oakland Coliseum': 320,
            'Coors Field': 20,
            'Chase Field': 15,
            'Petco Park': 260,
            'Dodger Stadium': 25,
            'Oracle Park': 340,
            'Wrigley Field': 65,
            'Great American Ball Park': 55,
            'PNC Park': 40,
            'American Family Field': 95,
            'Busch Stadium': 85,
            'Truist Park': 75,
            'LoanDepot Park': 80,
            'Nationals Park': 90,
            'Citizens Bank Park': 85,
            'Citi Field': 45
        }
        
        # Roof/dome info
        self.roof_info = {
            'Tropicana Field': {'type': 'dome', 'retractable': False},
            'Rogers Centre': {'type': 'retractable', 'retractable': True},
            'Minute Maid Park': {'type': 'retractable', 'retractable': True},
            'Globe Life Field': {'type': 'retractable', 'retractable': True},
            'Chase Field': {'type': 'retractable', 'retractable': True},
            'T-Mobile Park': {'type': 'retractable', 'retractable': True},
            'American Family Field': {'type': 'retractable', 'retractable': True},
            'LoanDepot Park': {'type': 'retractable', 'retractable': True}
        }
    
    def safe_azimuth(self, deg: int) -> int:
        """Ensure azimuth is in [0,360) range"""
        return int(deg) % 360
    
    def _table_exists(self, table: str) -> bool:
        conn = self.get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = current_schema() AND table_name = %s
            """, (table,))
        ok = cur.fetchone() is not None
        cur.close(); conn.close()
        return ok

    def _first_present_column(self, table: str, candidates) -> Optional[str]:
        conn = self.get_db_connection()
        df = pd.read_sql("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            """, conn, params=(table,))
        conn.close()
        cols = set(df['column_name'].tolist())
        for c in candidates:
            if c in cols: return c
        return None
    
    def _ensure_columns(self, table: str, cols: dict):
        """
        Ensure each column in `cols` exists on `table`.
        cols = { "col_name": "SQL_TYPE [DEFAULT ...]" }
        """
        conn = self.get_db_connection()
        try:
            # Get existing columns first
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = %s
                """, (table,))
                existing = {r[0] for r in cur.fetchall()}
            
            # Add missing columns one by one in separate transactions
            for name, sqltype in cols.items():
                if name not in existing:
                    try:
                        # Use a fresh connection for each column addition
                        fresh_conn = self.get_db_connection()
                        with fresh_conn.cursor() as cur:
                            # Double-check column doesn't exist in case of race condition
                            cur.execute("""
                                SELECT 1 FROM information_schema.columns
                                WHERE table_schema = current_schema()
                                  AND table_name = %s AND column_name = %s
                            """, (table, name))
                            
                            if not cur.fetchone():
                                cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")
                                logger.debug(f"✅ Added column {name} to {table}")
                            else:
                                logger.debug(f"⚠️  Column {name} already exists in {table}")
                        
                        fresh_conn.commit()
                        fresh_conn.close()
                        
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.debug(f"⚠️  Column {name} already exists in {table}")
                        else:
                            logger.warning(f"⚠️  Could not add column {name} to {table}: {e}")
                        # Don't let column addition failures break the whole process
                        continue
                else:
                    logger.debug(f"✓ Column {name} already exists in {table}")
        finally:
            conn.close()
    
    def get_db_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def wind_out_component(self, wind_speed: float, wind_dir_deg: float, park_azimuth_deg: float) -> Optional[float]:
        """Calculate wind component blowing out to center field"""
        if None in (wind_speed, wind_dir_deg, park_azimuth_deg):
            return None
        theta = math.radians((wind_dir_deg - park_azimuth_deg) % 360)
        return wind_speed * math.cos(theta)
    
    def air_density_proxy(self, temp_f: float, pressure_hPa: float) -> Optional[float]:
        """Calculate air density proxy (higher = denser air = suppress HR)"""
        if None in (temp_f, pressure_hPa):
            return None
        temp_K = (temp_f - 32) * 5.0/9.0 + 273.15
        return pressure_hPa / temp_K
    
    def create_enhanced_schema(self):
        """Create enhanced schema tables for high-ROI data"""
        
        logger.info("🏗️  Creating enhanced schema tables...")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # 1. Weather + park orientation
        weather_schema = """
        CREATE TABLE IF NOT EXISTS weather_hourly (
            game_id INTEGER,
            obs_ts TIMESTAMP,
            temp_f FLOAT,
            wind_speed FLOAT,
            wind_dir_deg FLOAT,
            rel_humidity FLOAT,
            pressure_hPa FLOAT,
            precip_prob FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (game_id, obs_ts)
        );
        
        CREATE TABLE IF NOT EXISTS park_meta (
            venue_name VARCHAR(100) PRIMARY KEY,
            azimuth_deg_home_to_cf INTEGER,
            altitude_ft INTEGER,
            roof_type VARCHAR(20), -- 'dome', 'retractable', 'open'
            has_humidor BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # 2. Bullpen fatigue & availability
        bullpen_schema = """
        CREATE TABLE IF NOT EXISTS reliever_usage (
            date              DATE NOT NULL,
            pitcher_id        INTEGER NOT NULL,
            team              VARCHAR(50) NOT NULL,
            role              VARCHAR(16),              -- RP/CL/SU/OPENER/BULK
            innings           FLOAT,
            pitches           INTEGER,
            score_diff_entry  INTEGER,                  -- optional, keep null for now
            inning_entry      INTEGER,                  -- optional
            created_at        TIMESTAMP DEFAULT NOW(),
            updated_at        TIMESTAMP,
            PRIMARY KEY (date, pitcher_id)
        );

        CREATE INDEX IF NOT EXISTS idx_relief_by_team_date ON reliever_usage(team, date);
        
        CREATE TABLE IF NOT EXISTS bullpen_availability (
            team VARCHAR(50),
            date DATE,
            arms_avail INTEGER,
            pitches_last3 INTEGER,
            innings_last3 FLOAT,
            lhp_avail INTEGER,
            rhp_avail INTEGER,
            bp_fatigue_raw FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (team, date)
        );
        """
        
        # 3. Opener/bulk & starter workload
        starter_schema = """
        CREATE TABLE IF NOT EXISTS pitcher_start_meta (
            pitcher_id INTEGER,
            date DATE,
            is_opener BOOLEAN DEFAULT FALSE,
            bulk_pitcher_id INTEGER,
            rest_days INTEGER,
            last_pitches INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (pitcher_id, date)
        );
        
        CREATE TABLE IF NOT EXISTS pitcher_pitch_metrics_daily (
            pitcher_id INTEGER,
            date DATE,
            fb_velo FLOAT,
            spin_rate FLOAT,
            usage_fourseam FLOAT,
            usage_slider FLOAT,
            usage_changeup FLOAT,
            usage_curveball FLOAT,
            velo_delta_L3 FLOAT,
            spin_delta_L3 FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (pitcher_id, date)
        );
        """
        
        # 4. Lineup quality & uncertainty (NEW TABLE - leave legacy lineups alone)
        lineup_schema_enh = """
        -- Enhanced lineups (do NOT touch legacy `lineups`)
        CREATE TABLE IF NOT EXISTS lineups_enh (
            game_id     INTEGER NOT NULL,
            ts          TIMESTAMP NOT NULL,
            confirmed   BOOLEAN DEFAULT FALSE,
            spot        INTEGER NOT NULL,         -- 1..9 (+ DH if you use it)
            player_id   INTEGER NOT NULL,
            bats        CHAR(1),                  -- L/R/S
            created_at  TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (game_id, ts, spot)
        );

        -- Latest lineup per game helper view
        CREATE OR REPLACE VIEW lineups_enh_latest AS
        SELECT DISTINCT ON (game_id, spot)
            game_id, ts, confirmed, spot, player_id, bats, created_at
        FROM lineups_enh
        ORDER BY game_id, spot, confirmed DESC, ts DESC;

        CREATE INDEX IF NOT EXISTS idx_lineups_enh_game_ts ON lineups_enh(game_id, ts DESC);
        
        CREATE TABLE IF NOT EXISTS batter_daily_rolling (
            player_id INTEGER,
            date DATE,
            xwoba_v_rhp FLOAT,
            xwoba_v_lhp FLOAT,
            pa_last10 INTEGER,
            pa_last30 INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (player_id, date)
        );
        """
        
        # 5. Umpire run environment
        umpire_schema = """
        CREATE TABLE IF NOT EXISTS umpire_game_assignments (
            game_id INTEGER PRIMARY KEY,
            plate_ump VARCHAR(100),
            first_base_ump VARCHAR(100),
            second_base_ump VARCHAR(100),
            third_base_ump VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS umpire_stats (
            umpire VARCHAR(100),
            as_of_date DATE,
            runs_factor FLOAT,
            csw_rate FLOAT,
            bb_rate_adj FLOAT,
            sample_games INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (umpire, as_of_date)
        );
        """
        
        # 6. Travel, rest & scheduling
        travel_schema = """
        CREATE TABLE IF NOT EXISTS team_schedule (
            team VARCHAR(50),
            date DATE,
            city VARCHAR(100),
            tz VARCHAR(10),
            start_local TIME,
            is_night BOOLEAN,
            series_id VARCHAR(50),
            travel_miles_24h FLOAT,
            tz_shift INTEGER,
            day_after_night BOOLEAN,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (team, date)
        );
        """
        
        # 7. Defense & catcher effects
        defense_schema = """
        CREATE TABLE IF NOT EXISTS team_defense_daily (
            team VARCHAR(50),
            date DATE,
            drs_120 FLOAT,
            oaa_120 FLOAT,
            if_oaa FLOAT,
            of_oaa FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (team, date)
        );
        
        CREATE TABLE IF NOT EXISTS catcher_defense_daily (
            player_id INTEGER,
            date DATE,
            framing_runs_6000pitches FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (player_id, date)
        );
        """
        
        # 8. Game conditions summary (precomputed features)
        conditions_schema = """
        CREATE TABLE IF NOT EXISTS game_conditions (
            game_id INTEGER PRIMARY KEY,
            
            -- Weather/Park
            wind_out_cf FLOAT,
            air_density_proxy FLOAT,
            roof_open BOOLEAN,
            roof_state TEXT, -- 'open', 'closed', 'unknown' for retractables
            
            -- Bullpen
            home_bp_fatigue FLOAT,
            away_bp_fatigue FLOAT,
            home_arms_avail INTEGER,
            away_arms_avail INTEGER,
            
            -- Starters
            home_sp_is_opener BOOLEAN,
            away_sp_is_opener BOOLEAN,
            home_sp_rest_days INTEGER,
            away_sp_rest_days INTEGER,
            home_sp_velo_delta_L3 FLOAT,
            away_sp_velo_delta_L3 FLOAT,
            
            -- Lineups
            home_lineup_core_xwoba FLOAT,
            away_lineup_core_xwoba FLOAT,
            home_missing_regulars INTEGER,
            away_missing_regulars INTEGER,
            home_platoon_advantage INTEGER,
            away_platoon_advantage INTEGER,
            
            -- Umpire
            plate_ump_runs_factor FLOAT,
            
            -- Travel/Rest
            home_tz_shift INTEGER,
            away_tz_shift INTEGER,
            home_miles_24h FLOAT,
            away_miles_24h FLOAT,
            home_day_after_night BOOLEAN,
            away_day_after_night BOOLEAN,
            
            -- Defense
            home_defense_quality FLOAT,
            away_defense_quality FLOAT,
            home_catcher_frame_z FLOAT,
            away_catcher_frame_z FLOAT,
            
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Execute all schema creation
        all_schemas = [
            weather_schema,
            bullpen_schema, 
            starter_schema,
            lineup_schema_enh,
            umpire_schema,
            travel_schema,
            defense_schema,
            conditions_schema
        ]
        
        for schema in all_schemas:
            try:
                cursor.execute(schema)
                logger.info("✅ Schema section created successfully")
            except Exception as e:
                logger.error(f"❌ Error creating schema: {str(e)}")
                conn.rollback()
                return False
        
        # --- Dynamic index creation: respects whatever key player_bio actually has ---
        try:
            idx_sql = [
                "CREATE INDEX IF NOT EXISTS idx_enhanced_games_date ON enhanced_games(date)",
                "CREATE INDEX IF NOT EXISTS idx_enhanced_games_game_id ON enhanced_games(game_id)",
                "CREATE INDEX IF NOT EXISTS idx_pitcher_start_meta ON pitcher_start_meta(pitcher_id, date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_bullpen_availability ON bullpen_availability(team, date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_park_meta_venue ON park_meta(venue_name)",
                "CREATE INDEX IF NOT EXISTS idx_relief_by_team_date ON reliever_usage(team, date)",
                "CREATE INDEX IF NOT EXISTS idx_ump_stats_asof ON umpire_stats(umpire, as_of_date DESC)",
            ]

            bio_pk = self._first_present_column('player_bio', ['mlb_id','player_id','id'])
            if bio_pk:
                idx_sql.append(f"CREATE INDEX IF NOT EXISTS idx_player_bio_id ON player_bio({bio_pk})")
            else:
                logger.warning("⚠️  player_bio has no mlb_id/player_id/id column; skipping that index")

            with conn.cursor() as cur2:
                for s in idx_sql:
                    cur2.execute(s)
                cur2.execute("ALTER TABLE park_meta ALTER COLUMN has_humidor SET DEFAULT TRUE")
            logger.info("✅ Performance indexes created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating indexes dynamically: {str(e)}")
            conn.rollback()
            return False
        
        # --- Ensure game_conditions has all expected columns (migrate old tables forward)
        try:
            self._ensure_columns("game_conditions", {
                # Weather/Park
                "wind_out_cf": "FLOAT",
                "air_density_proxy": "FLOAT",
                "roof_open": "BOOLEAN",
                "roof_state": "TEXT",

                # Bullpen
                "home_bp_fatigue": "FLOAT",
                "away_bp_fatigue": "FLOAT",
                "home_arms_avail": "INTEGER",
                "away_arms_avail": "INTEGER",

                # Starters
                "home_sp_is_opener": "BOOLEAN",
                "away_sp_is_opener": "BOOLEAN",
                "home_sp_rest_days": "INTEGER",
                "away_sp_rest_days": "INTEGER",
                "home_sp_velo_delta_L3": "FLOAT",
                "away_sp_velo_delta_L3": "FLOAT",

                # Lineups
                "home_lineup_core_xwoba": "FLOAT",
                "away_lineup_core_xwoba": "FLOAT",
                "home_missing_regulars": "INTEGER",
                "away_missing_regulars": "INTEGER",
                "home_platoon_advantage": "INTEGER",
                "away_platoon_advantage": "INTEGER",

                # Umpire
                "plate_ump_runs_factor": "FLOAT",

                # Travel/Rest
                "home_tz_shift": "INTEGER",
                "away_tz_shift": "INTEGER",
                "home_miles_24h": "FLOAT",
                "away_miles_24h": "FLOAT",
                "home_day_after_night": "BOOLEAN",
                "away_day_after_night": "BOOLEAN",

                # Defense (even if you're not inserting them yet, keep schema future-proof)
                "home_defense_quality": "FLOAT",
                "away_defense_quality": "FLOAT",
                "home_catcher_frame_z": "FLOAT",
                "away_catcher_frame_z": "FLOAT",

                # IP caps (already handled elsewhere, but safe if missing)
                "home_sp_ip_cap": "FLOAT",
                "away_sp_ip_cap": "FLOAT",
            })
            logger.info("✅ game_conditions schema migrated/verified")
        except Exception as e:
            logger.error(f"❌ Error migrating game_conditions columns: {e}")
            conn.rollback()
            return False
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("🎯 Enhanced schema creation completed!")
        return True
    
    def populate_park_meta(self):
        """Populate park metadata with azimuth and roof info - FIXED"""
        
        logger.info("🏟️  Populating park metadata...")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Insert park metadata
        for venue, azimuth_deg in self.park_azimuth.items():
            roof_data = self.roof_info.get(venue, {'type': 'open', 'retractable': False})
            
            # Estimate altitude (simplified - would need more precise data)
            altitude_estimates = {
                'Coors Field': 5200,  # Denver
                'Chase Field': 1100,  # Phoenix
                'Globe Life Field': 550,  # Arlington
                'Kauffman Stadium': 750,  # Kansas City
                'Target Field': 815,  # Minneapolis
            }
            altitude = altitude_estimates.get(venue, 50)  # Sea level default
            
            # Safe azimuth and default humidor=True (MLB standard since 2022)
            safe_az = self.safe_azimuth(azimuth_deg)
            
            insert_query = """
            INSERT INTO park_meta (
                venue_name, azimuth_deg_home_to_cf, altitude_ft, 
                roof_type, has_humidor
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (venue_name) 
            DO UPDATE SET
                azimuth_deg_home_to_cf = EXCLUDED.azimuth_deg_home_to_cf,
                altitude_ft = EXCLUDED.altitude_ft,
                roof_type = EXCLUDED.roof_type,
                has_humidor = EXCLUDED.has_humidor
            """
            
            cursor.execute(insert_query, (
                venue, safe_az, altitude,
                roof_data['type'], True  # Default humidor=True for all parks
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Populated {len(self.park_azimuth)} park metadata records")
    
    def calculate_bullpen_fatigue(self, start_date: date, end_date: date):
        """Calculate and populate bullpen fatigue metrics"""
        
        if not self._table_exists('pitcher_game_logs'):
            logger.info("⏭️  Skipping bullpen fatigue: pitcher_game_logs not present")
            return
        
        logger.info(f"⚾ Calculating bullpen fatigue from {start_date} to {end_date}")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Detect the bio primary key column dynamically
        bio_pk = self._first_present_column('player_bio', ['mlb_id','player_id','id']) or 'pitcher_id'  # fallback no-op
        
        current_date = start_date
        processed_count = 0
        
        while current_date <= end_date:
            try:
                # Calculate 3-day rolling bullpen usage for each team
                fatigue_query = f"""
                WITH reliever_last3 AS (
                    SELECT 
                        ru.team,
                        COUNT(DISTINCT ru.pitcher_id) as arms_used,
                        SUM(ru.pitches) as total_pitches,
                        SUM(ru.innings) as total_innings,
                        COUNT(CASE WHEN pb.throws = 'L' THEN 1 END) as lhp_used,
                        COUNT(CASE WHEN pb.throws = 'R' THEN 1 END) as rhp_used
                    FROM reliever_usage ru
                    LEFT JOIN player_bio pb ON pb.{bio_pk} = ru.pitcher_id
                    WHERE ru.date BETWEEN %s - INTERVAL '3 days' AND %s - INTERVAL '1 day'
                      AND ru.team = %s
                    GROUP BY ru.team
                ),
                active_bullpen AS (
                    -- Use actual game logs to determine active bullpen (26-man roster)
                    SELECT 
                        %s as team,
                        COUNT(CASE WHEN pb.throws = 'L' AND pgl.role IN ('RP', 'CL', 'SU', 'OPENER', 'BULK') THEN 1 END) as total_lhp,
                        COUNT(CASE WHEN pb.throws = 'R' AND pgl.role IN ('RP', 'CL', 'SU', 'OPENER', 'BULK') THEN 1 END) as total_rhp,
                        COUNT(CASE WHEN pgl.role IN ('RP', 'CL', 'SU', 'OPENER', 'BULK') THEN 1 END) as total_relievers
                    FROM pitcher_game_logs pgl
                    LEFT JOIN player_bio pb ON pb.{bio_pk} = pgl.pitcher_id
                    WHERE pgl.game_date BETWEEN %s - INTERVAL '7 days' AND %s
                      AND pgl.team = %s
                      AND pgl.role IN ('RP', 'CL', 'SU', 'OPENER', 'BULK')  -- Active relievers only
                    GROUP BY pgl.team
                )
                SELECT 
                    COALESCE(rl3.arms_used, 0) as arms_used,
                    COALESCE(rl3.total_pitches, 0) as pitches_last3,
                    COALESCE(rl3.total_innings, 0) as innings_last3,
                    COALESCE(ab.total_lhp - COALESCE(rl3.lhp_used, 0), COALESCE(ab.total_lhp, 0)) as lhp_avail,
                    COALESCE(ab.total_rhp - COALESCE(rl3.rhp_used, 0), COALESCE(ab.total_rhp, 0)) as rhp_avail,
                    COALESCE(ab.total_relievers - COALESCE(rl3.arms_used, 0), COALESCE(ab.total_relievers, 0)) as arms_avail
                FROM active_bullpen ab
                LEFT JOIN reliever_last3 rl3 ON rl3.team = ab.team
                """
                
                # Get all teams for this date
                teams_query = """
                SELECT DISTINCT home_team as team FROM enhanced_games WHERE date = %s
                UNION
                SELECT DISTINCT away_team as team FROM enhanced_games WHERE date = %s
                """
                cursor.execute(teams_query, (current_date, current_date))
                teams = [row[0] for row in cursor.fetchall()]
                
                for team in teams:
                    cursor.execute(fatigue_query, (
                        current_date, current_date, team,  # reliever_last3 params
                        team, current_date, current_date, team  # active_bullpen params
                    ))
                    result = cursor.fetchone()
                    
                    if result:
                        arms_used, pitches_last3, innings_last3, lhp_avail, rhp_avail, arms_avail = result
                        
                        # Calculate fatigue index (normalized)
                        bp_fatigue_raw = innings_last3 + (0.5 * pitches_last3 / 100.0)  # Scale pitches
                        
                        # Insert bullpen availability
                        insert_query = """
                        INSERT INTO bullpen_availability (
                            team, date, arms_avail, pitches_last3, innings_last3,
                            lhp_avail, rhp_avail, bp_fatigue_raw
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (team, date)
                        DO UPDATE SET
                            arms_avail = EXCLUDED.arms_avail,
                            pitches_last3 = EXCLUDED.pitches_last3,
                            innings_last3 = EXCLUDED.innings_last3,
                            lhp_avail = EXCLUDED.lhp_avail,
                            rhp_avail = EXCLUDED.rhp_avail,
                            bp_fatigue_raw = EXCLUDED.bp_fatigue_raw
                        """
                        
                        cursor.execute(insert_query, (
                            team, current_date, arms_avail, pitches_last3, innings_last3,
                            lhp_avail, rhp_avail, bp_fatigue_raw
                        ))
                        
                        processed_count += 1
                
                if current_date.day % 7 == 0:  # Commit weekly
                    conn.commit()
                    logger.info(f"💾 Committed bullpen data through {current_date}")
                
            except Exception as e:
                logger.error(f"❌ Error processing bullpen data for {current_date}: {str(e)}")
                conn.rollback()
            
            current_date += timedelta(days=1)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Processed {processed_count} bullpen fatigue records")
    
    def calculate_starter_workload(self, start_date: date, end_date: date):
        """Calculate starter workload and opener detection"""
        
        if not self._table_exists('pitcher_game_logs'):
            logger.info("⏭️  Skipping starter workload: pitcher_game_logs not present")
            return
        
        logger.info(f"🎯 Calculating starter workload from {start_date} to {end_date}")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        current_date = start_date
        processed_count = 0
        
        while current_date <= end_date:
            try:
                # Get games for this date
                games_query = """
                SELECT game_id, home_sp_id, away_sp_id 
                FROM enhanced_games 
                WHERE date = %s AND total_runs IS NOT NULL
                """
                cursor.execute(games_query, (current_date,))
                games = cursor.fetchall()
                
                for game_id, home_sp_id, away_sp_id in games:
                    for sp_id in [home_sp_id, away_sp_id]:
                        if not sp_id:
                            continue
                        
                        # Calculate rest days and last start info
                        workload_query = """
                        WITH last_start AS (
                            SELECT 
                                pgl.game_date,
                                pgl.pitches_thrown,
                                pgl.innings_pitched
                            FROM pitcher_game_logs pgl
                            WHERE pgl.pitcher_id = %s 
                              AND pgl.game_date < %s
                            ORDER BY pgl.game_date DESC
                            LIMIT 1
                        )
                        SELECT 
                            COALESCE(EXTRACT(DAY FROM (%s - ls.game_date)), 999) as rest_days,
                            COALESCE(ls.pitches_thrown, 0) as last_pitches,
                            COALESCE(ls.innings_pitched, 0) as last_innings
                        FROM last_start ls
                        """
                        
                        cursor.execute(workload_query, (sp_id, current_date, current_date))
                        result = cursor.fetchone()
                        
                        if result:
                            rest_days, last_pitches, last_innings = result
                            
                            # Detect opener (simplified logic)
                            is_opener = last_innings < 2.0 and rest_days <= 3
                            
                            # Insert starter metadata
                            insert_query = """
                            INSERT INTO pitcher_start_meta (
                                pitcher_id, date, is_opener, rest_days, last_pitches
                            ) VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (pitcher_id, date)
                            DO UPDATE SET
                                is_opener = EXCLUDED.is_opener,
                                rest_days = EXCLUDED.rest_days,
                                last_pitches = EXCLUDED.last_pitches
                            """
                            
                            cursor.execute(insert_query, (
                                sp_id, current_date, is_opener, rest_days, last_pitches
                            ))
                            
                            processed_count += 1
                
                if current_date.day % 5 == 0:  # Commit every 5 days
                    conn.commit()
                    logger.info(f"💾 Committed starter workload through {current_date}")
                
            except Exception as e:
                logger.error(f"❌ Error processing starter workload for {current_date}: {str(e)}")
                conn.rollback()
            
            current_date += timedelta(days=1)
        
        conn.commit()
        cursor.close()
        conn.close()
        
    def fill_pitcher_rolling_gaps(self):
        """Fill gaps in pitcher rolling averages using existing daily data"""
        
        if not self._table_exists('pitcher_game_logs'):
            logger.info("⏭️  Skipping pitcher rolling gaps: pitcher_game_logs not present")
            return
        
        logger.info("🔄 Filling pitcher rolling average gaps...")
        
        conn = self.get_db_connection()
        
        # Find games missing pitcher rolling data but where daily data exists
        gap_query = """
        WITH missing_pitcher_data AS (
            SELECT DISTINCT
                eg.date,
                eg.home_sp_id as pitcher_id,
                eg.home_team as team
            FROM enhanced_games eg
            LEFT JOIN pitcher_daily_rolling pdr ON pdr.pitcher_id = eg.home_sp_id 
                AND pdr.stat_date = eg.date
            WHERE eg.date >= '2024-08-01'
              AND eg.home_sp_id IS NOT NULL
              AND pdr.pitcher_id IS NULL
              
            UNION
            
            SELECT DISTINCT
                eg.date,
                eg.away_sp_id as pitcher_id,
                eg.away_team as team
            FROM enhanced_games eg
            LEFT JOIN pitcher_daily_rolling pdr ON pdr.pitcher_id = eg.away_sp_id 
                AND pdr.stat_date = eg.date
            WHERE eg.date >= '2024-08-01'
              AND eg.away_sp_id IS NOT NULL
              AND pdr.pitcher_id IS NULL
        )
        SELECT 
            mpd.date,
            mpd.pitcher_id,
            mpd.team,
            COUNT(*) OVER() as total_gaps
        FROM missing_pitcher_data mpd
        ORDER BY mpd.date DESC, mpd.pitcher_id
        LIMIT 100
        """
        
        gaps_df = pd.read_sql(gap_query, conn)
        
        if gaps_df.empty:
            logger.info("✅ No pitcher rolling data gaps found")
            conn.close()
            return
        
        logger.info(f"🎯 Found {gaps_df['total_gaps'].iloc[0] if not gaps_df.empty else 0} pitcher data gaps")
        
        cursor = conn.cursor()
        filled_count = 0
        
        for _, row in gaps_df.iterrows():
            try:
                pitcher_id = row['pitcher_id']
                target_date = row['date']
                
                # Calculate rolling averages for this pitcher/date
                rolling_query = """
                WITH recent_games AS (
                    SELECT 
                        game_date,
                        earned_runs,
                        innings_pitched,
                        hits_allowed,
                        walks_allowed,
                        strikeouts,
                        home_runs_allowed
                    FROM pitcher_game_logs 
                    WHERE pitcher_id = %s
                      AND game_date < %s
                      AND game_date >= %s - INTERVAL '30 days'
                    ORDER BY game_date DESC
                    LIMIT 10
                )
                SELECT 
                    COUNT(*) as games_count,
                    AVG(CASE 
                        WHEN innings_pitched > 0 
                        THEN (earned_runs * 9.0) / innings_pitched 
                        ELSE NULL 
                    END) as era,
                    AVG(CASE 
                        WHEN innings_pitched > 0 
                        THEN (hits_allowed + walks_allowed) / innings_pitched 
                        ELSE NULL 
                    END) as whip,
                    SUM(innings_pitched) as total_innings,
                    SUM(strikeouts) as total_strikeouts,
                    SUM(walks_allowed) as total_walks
                FROM recent_games
                """
                
                cursor.execute(rolling_query, (pitcher_id, target_date, target_date))
                rolling_result = cursor.fetchone()
                
                if rolling_result and rolling_result[0] > 0:  # games_count > 0
                    games_count, era, whip, total_innings, total_k, total_bb = rolling_result
                    
                    # Insert calculated rolling data
                    insert_query = """
                    INSERT INTO pitcher_daily_rolling (
                        pitcher_id, stat_date, era, whip, 
                        games_in_period, innings_pitched,
                        strikeouts, walks_allowed, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (pitcher_id, stat_date) 
                    DO UPDATE SET
                        era = EXCLUDED.era,
                        whip = EXCLUDED.whip,
                        games_in_period = EXCLUDED.games_in_period,
                        innings_pitched = EXCLUDED.innings_pitched,
                        strikeouts = EXCLUDED.strikeouts,
                        walks_allowed = EXCLUDED.walks_allowed,
                        updated_at = NOW()
                    """
                    
                    cursor.execute(insert_query, (
                        pitcher_id, target_date, era, whip,
                        games_count, total_innings, total_k, total_bb
                    ))
                    
                    filled_count += 1
                    
                    if filled_count % 50 == 0:
                        conn.commit()
                        logger.info(f"💾 Committed {filled_count} pitcher rolling records")
            
            except Exception as e:
                logger.error(f"❌ Error filling pitcher {pitcher_id} for {target_date}: {str(e)}")
                conn.rollback()
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Filled {filled_count} pitcher rolling data gaps")
    
    def compute_game_conditions(self, start_date: date, end_date: date) -> None:
        logger.info(f"📊 Computing game conditions for {start_date} to {end_date}")
        conn = self.get_db_connection()
        cur = conn.cursor()

        def _try_int(x):
            try: return int(x)
            except: return None

        # Build helper maps
        velo_map   = self.build_pitcher_velo_map(start_date, end_date)
        umpire_map = self.build_umpire_map(start_date, end_date)
        travel_map = self.build_travel_map(start_date, end_date)
        lineup_map = self.build_missing_regulars_and_corexwoba(start_date, end_date)
        ip_cap_map = self.build_ip_cap_map(start_date, end_date)

        # Pull games + park meta (normalized if xref exists)
        has_vxref = self._table_exists('venue_xref') \
            and (self._first_present_column('venue_xref', ['raw_venue']) is not None) \
            and (self._first_present_column('venue_xref', ['normalized_venue']) is not None)

        if has_vxref:
            q = """
            SELECT
              eg.game_id, eg.date::date AS d, eg.venue, eg.home_team, eg.away_team,
              eg.home_sp_id, eg.away_sp_id,
              pm.azimuth_deg_home_to_cf AS azimuth,
              pm.roof_type
            FROM enhanced_games eg
            LEFT JOIN venue_xref vx ON vx.raw_venue = eg.venue
            LEFT JOIN park_meta pm   ON pm.venue_name = COALESCE(vx.normalized_venue, eg.venue)
            WHERE eg.date BETWEEN %s AND %s
            ORDER BY eg.date, eg.game_id
            """
        else:
            q = """
            SELECT
              eg.game_id, eg.date::date AS d, eg.venue, eg.home_team, eg.away_team,
              eg.home_sp_id, eg.away_sp_id,
              pm.azimuth_deg_home_to_cf AS azimuth,
              pm.roof_type
            FROM enhanced_games eg
            LEFT JOIN park_meta pm ON pm.venue_name = eg.venue
            WHERE eg.date BETWEEN %s AND %s
            ORDER BY eg.date, eg.game_id
            """
        games = pd.read_sql(q, conn, params=(start_date, end_date))

        # Delete existing rows in the range
        cur.execute("""
          DELETE FROM game_conditions gc
          USING enhanced_games eg
          WHERE gc.game_id::text = eg.game_id::text
            AND eg.date BETWEEN %s AND %s
        """, (start_date, end_date))

        rows = 0
        for _, r in games.iterrows():
            gid = _try_int(r.game_id)
            if gid is None:
                logger.debug(f"⏭️  Skipping non-integer game_id={r.game_id!r}")
                continue
            
            d         = pd.to_datetime(r.d).date()
            home_team = r.home_team
            away_team = r.away_team
            azimuth   = float(r.azimuth) if not pd.isna(r.azimuth) else None
            roof_type = (r.roof_type or 'open').lower()

            # Weather placeholders (until you wire an API)
            if roof_type == 'dome':
                wind_out_cf = 0.0
                air_density = self.air_density_proxy(72.0, 1013.25)
                roof_open   = False
                roof_state  = 'closed'
            elif roof_type == 'retractable':
                wind_out_cf = self.wind_out_component(5.0, 180.0, azimuth or 0.0)
                air_density = self.air_density_proxy(72.0, 1013.25)
                roof_open   = False
                roof_state  = 'unknown'
            else:
                wind_out_cf = None
                air_density = None
                roof_open   = None
                roof_state  = None

            # Bullpen
            cur.execute("""
                SELECT bp_fatigue_raw, arms_avail
                FROM bullpen_availability
                WHERE team = %s AND date <= %s
                ORDER BY date DESC LIMIT 1
            """, (home_team, d))
            hb = cur.fetchone() or (0.0, 7)
            home_bp_fatigue, home_arms_avail = float(hb[0]), int(hb[1])

            cur.execute("""
                SELECT bp_fatigue_raw, arms_avail
                FROM bullpen_availability
                WHERE team = %s AND date <= %s
                ORDER BY date DESC LIMIT 1
            """, (away_team, d))
            ab = cur.fetchone() or (0.0, 7)
            away_bp_fatigue, away_arms_avail = float(ab[0]), int(ab[1])

            # Starters
            def starter_meta(pid):
                if not pid or pd.isna(pid): return (False, 4)
                cur.execute("""
                    SELECT is_opener, rest_days
                    FROM pitcher_start_meta
                    WHERE pitcher_id = %s AND date <= %s
                    ORDER BY date DESC LIMIT 1
                """, (int(pid), d))
                row = cur.fetchone()
                return (bool(row[0]), int(row[1])) if row else (False, 4)

            home_is_opener, home_rest = starter_meta(r.home_sp_id)
            away_is_opener, away_rest = starter_meta(r.away_sp_id)

            # Velo delta (L3)
            home_velo = float(velo_map.get((int(r.home_sp_id) if pd.notna(r.home_sp_id) else 0, d), 0.0))
            away_velo = float(velo_map.get((int(r.away_sp_id) if pd.notna(r.away_sp_id) else 0, d), 0.0))

            # Umpire
            plate_ump_rf = float(umpire_map.get(gid, 1.0))

            # Travel / day-after-night
            home_tz_shift, home_dan = travel_map.get((gid, 'home'), (0, False))
            away_tz_shift, away_dan = travel_map.get((gid, 'away'), (0, False))

            # Lineup quality & missing regulars
            # returns (home_missing, away_missing, home_core_xwoba, away_core_xwoba, home_platoon_adv, away_platoon_adv)
            lm = lineup_map.get(gid, (0,0,0.300,0.300,0,0))
            home_missing, away_missing, home_xw, away_xw, home_padv, away_padv = lm

            # IP caps
            home_ip_cap = float(ip_cap_map.get((int(r.home_sp_id) if pd.notna(r.home_sp_id) else 0, d), 5.0))
            away_ip_cap = float(ip_cap_map.get((int(r.away_sp_id) if pd.notna(r.away_sp_id) else 0, d), 5.0))

            # Insert
            cur.execute("""
            INSERT INTO game_conditions (
            game_id,
            wind_out_cf, air_density_proxy, roof_open, roof_state,
            home_bp_fatigue, away_bp_fatigue, home_arms_avail, away_arms_avail,
            home_sp_is_opener, away_sp_is_opener, home_sp_rest_days, away_sp_rest_days,
            home_sp_velo_delta_L3, away_sp_velo_delta_L3,
            home_lineup_core_xwoba, away_lineup_core_xwoba,
            home_missing_regulars, away_missing_regulars,
            home_platoon_advantage, away_platoon_advantage,
            plate_ump_runs_factor,
            home_tz_shift, away_tz_shift,
            home_day_after_night, away_day_after_night,
            home_sp_ip_cap, away_sp_ip_cap
            ) VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,
            %s,%s,
            %s,%s,
            %s,%s,
            %s,
            %s,%s,
            %s,%s,
            %s,%s
            )
            ON CONFLICT (game_id) DO UPDATE SET
            wind_out_cf               = EXCLUDED.wind_out_cf,
            air_density_proxy         = EXCLUDED.air_density_proxy,
            roof_open                 = EXCLUDED.roof_open,
            roof_state                = EXCLUDED.roof_state,
            home_bp_fatigue           = EXCLUDED.home_bp_fatigue,
            away_bp_fatigue           = EXCLUDED.away_bp_fatigue,
            home_arms_avail           = EXCLUDED.home_arms_avail,
            away_arms_avail           = EXCLUDED.away_arms_avail,
            home_sp_is_opener         = EXCLUDED.home_sp_is_opener,
            away_sp_is_opener         = EXCLUDED.away_sp_is_opener,
            home_sp_rest_days         = EXCLUDED.home_sp_rest_days,
            away_sp_rest_days         = EXCLUDED.away_sp_rest_days,
            home_sp_velo_delta_L3     = EXCLUDED.home_sp_velo_delta_L3,
            away_sp_velo_delta_L3     = EXCLUDED.away_sp_velo_delta_L3,
            home_lineup_core_xwoba    = EXCLUDED.home_lineup_core_xwoba,
            away_lineup_core_xwoba    = EXCLUDED.away_lineup_core_xwoba,
            home_missing_regulars     = EXCLUDED.home_missing_regulars,
            away_missing_regulars     = EXCLUDED.away_missing_regulars,
            home_platoon_advantage    = EXCLUDED.home_platoon_advantage,
            away_platoon_advantage    = EXCLUDED.away_platoon_advantage,
            plate_ump_runs_factor     = EXCLUDED.plate_ump_runs_factor,
            home_tz_shift             = EXCLUDED.home_tz_shift,
            away_tz_shift             = EXCLUDED.away_tz_shift,
            home_day_after_night      = EXCLUDED.home_day_after_night,
            away_day_after_night      = EXCLUDED.away_day_after_night,
            home_sp_ip_cap            = EXCLUDED.home_sp_ip_cap,
            away_sp_ip_cap            = EXCLUDED.away_sp_ip_cap
            """, (
                gid,
                wind_out_cf, air_density, roof_open, roof_state,
                home_bp_fatigue, away_bp_fatigue, home_arms_avail, away_arms_avail,
                home_is_opener, away_is_opener, home_rest, away_rest,
                home_velo, away_velo,
                home_xw, away_xw,
                home_missing, away_missing,
                home_padv, away_padv,
                plate_ump_rf,
                home_tz_shift or 0, away_tz_shift or 0,
                home_dan, away_dan,
                home_ip_cap, away_ip_cap
            ))
            rows += 1

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"✅ Computed/updated {rows} game condition records")

    
    def run_enhanced_backfill(self, start_date: str = None, end_date: str = None):
        """Run the enhanced historical data backfill"""
        
        if start_date is None:
            start_dt = date.today() - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            
        if end_date is None:
            end_dt = date.today()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        logger.info("🚀 Starting enhanced historical data backfill...")
        logger.info(f"📅 Date range: {start_dt} to {end_dt}")
        
        # Step 1: Create enhanced schema
        if not self.create_enhanced_schema():
            logger.error("❌ Schema creation failed")
            return
        
        # Step 2: Populate park metadata
        self.populate_park_meta()
        
        # Step 3: Populate reliever usage (required for bullpen fatigue)
        self.populate_reliever_usage(start_dt, end_dt)
        
        # Step 4: Calculate bullpen fatigue
        self.calculate_bullpen_fatigue(start_dt, end_dt)
        
        # Step 5: Calculate starter workload
        self.calculate_starter_workload(start_dt, end_dt)
        
        # Step 6: Fill existing gaps
        self.fill_pitcher_rolling_gaps()
        self.collect_missing_handedness_data()
        
        # Step 7: Compute precomputed game conditions
        self.compute_game_conditions(start_dt, end_dt)
        
        logger.info("✅ Enhanced backfill completed!")
        
        # Step 7: Analyze improvements
        self.analyze_improvements()

    def collect_missing_handedness_data(self):
        """Collect missing player handedness data from MLB API"""
        logger.info("🔄 Collecting missing player handedness data...")
        conn = self.get_db_connection()

        # Only do this path if player_bio actually has mlb_id
        bio_has_mlb = self._first_present_column('player_bio', ['mlb_id'])
        if not bio_has_mlb:
            logger.info("ℹ️  player_bio has no mlb_id column; skipping MLB API enrichment.")
            conn.close()
            return

        # Find players missing handedness data
        missing_query = """
        SELECT DISTINCT
            eg.home_sp_id as pitcher_id
        FROM enhanced_games eg
        LEFT JOIN player_bio pb ON pb.mlb_id = eg.home_sp_id
        WHERE eg.date >= '2024-08-01'
          AND eg.home_sp_id IS NOT NULL
          AND pb.mlb_id IS NULL
        
        UNION
        
        SELECT DISTINCT
            eg.away_sp_id as pitcher_id
        FROM enhanced_games eg
        LEFT JOIN player_bio pb ON pb.mlb_id = eg.away_sp_id
        WHERE eg.date >= '2024-08-01'
          AND eg.away_sp_id IS NOT NULL
          AND pb.mlb_id IS NULL
        
        ORDER BY pitcher_id
        LIMIT 100
        """
        
        missing_df = pd.read_sql(missing_query, conn)
        
        if missing_df.empty:
            logger.info("✅ No missing handedness data found")
            conn.close()
            return
        
        logger.info(f"🎯 Found {len(missing_df)} players missing handedness data")
        
        cursor = conn.cursor()
        collected_count = 0
        
        for _, row in missing_df.iterrows():
            try:
                pitcher_id = row['pitcher_id']
                
                # Get player data from MLB API
                player_url = f"{self.api_base}/people/{pitcher_id}"
                response = requests.get(player_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'people' in data and len(data['people']) > 0:
                        player = data['people'][0]
                        
                        # Extract player info
                        full_name = player.get('fullName', '')
                        bats = player.get('batSide', {}).get('code', 'U')  # U = Unknown
                        throws = player.get('pitchHand', {}).get('code', 'U')
                        position = player.get('primaryPosition', {}).get('abbreviation', 'P')
                        
                        # Insert into player_bio
                        insert_query = """
                        INSERT INTO player_bio (
                            mlb_id, full_name, bats, throws, position, created_at
                        ) VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (mlb_id) 
                        DO UPDATE SET
                            full_name = EXCLUDED.full_name,
                            bats = EXCLUDED.bats,
                            throws = EXCLUDED.throws,
                            position = EXCLUDED.position,
                            updated_at = NOW()
                        """
                        
                        cursor.execute(insert_query, (
                            pitcher_id, full_name, bats, throws, position
                        ))
                        
                        collected_count += 1
                        
                        if collected_count % 20 == 0:
                            conn.commit()
                            logger.info(f"💾 Committed {collected_count} player records")
                
                # Rate limiting
                time.sleep(0.3)
            
            except Exception as e:
                logger.error(f"❌ Error collecting data for pitcher {pitcher_id}: {str(e)}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Collected {collected_count} player handedness records")
    
    def _safetz(self, t: Optional[str]) -> Optional[int]:
        """Parse timezone strings to UTC offset minutes"""
        if not t: 
            return None
        t = str(t)
        # Try fixed buckets
        tzmap = {"ET": -300, "EST": -300, "EDT": -240,
                 "CT": -360, "CST": -360, "CDT": -300,
                 "MT": -420, "MST": -420, "MDT": -360,
                 "PT": -480, "PST": -480, "PDT": -420}
        if t in tzmap: 
            return tzmap[t]
        # Try IANA tail (America/New_York → ET)
        if "New_York" in t: 
            return -300
        if "Chicago" in t: 
            return -360
        if "Denver" in t or "Phoenix" in t: 
            return -420
        if "Los_Angeles" in t: 
            return -480
        # Try +/-HH:MM
        if t.startswith(("+","-")) and ":" in t:
            sign = -1 if t.startswith("+") else 1  # DB stores "+03:00" = east of UTC; offsets minutes from UTC (negative west)
            hh, mm = t[1:].split(":")
            return sign * (int(hh)*60 + int(mm))
        return None

    def build_pitcher_velo_map(self, start_date: date, end_date: date) -> Dict[Tuple[int, date], float]:
        """
        Returns {(pitcher_id, game_date): velo_delta_L3} from pitcher_pitch_metrics_daily.
        If you only have per-game velo in pitcher_game_logs, adapt the query accordingly.
        """
        conn = self.get_db_connection()
        q = """
        WITH span AS (
          SELECT %s::date AS d0, %s::date AS d1
        ), base AS (
          SELECT ppm.pitcher_id, ppm.date, ppm.fb_velo
          FROM pitcher_pitch_metrics_daily ppm
          JOIN span s ON ppm.date BETWEEN s.d0 - INTERVAL '35 days' AND s.d1
          WHERE ppm.fb_velo IS NOT NULL
        ),
        l3 AS (
          SELECT pitcher_id, date,
                 AVG(fb_velo) OVER (PARTITION BY pitcher_id
                                    ORDER BY date
                                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS l3_avg
          FROM base
        ),
        l30 AS (
          SELECT pitcher_id, date,
                 AVG(fb_velo) OVER (PARTITION BY pitcher_id
                                    ORDER BY date
                                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW) AS l30_avg
          FROM base
        )
        SELECT b.pitcher_id, b.date::date,
               (COALESCE(l3.l3_avg, b.fb_velo) - COALESCE(l30.l30_avg, b.fb_velo)) AS velo_delta_l3
        FROM base b
        LEFT JOIN l3 ON (l3.pitcher_id=b.pitcher_id AND l3.date=b.date)
        LEFT JOIN l30 ON (l30.pitcher_id=b.pitcher_id AND l30.date=b.date)
        WHERE b.date BETWEEN %s AND %s
        """
        try:
            df = pd.read_sql(q, conn, params=(start_date, end_date, start_date, end_date))
            result = { (int(r.pitcher_id), pd.to_datetime(r.date).date()): float(r.velo_delta_l3 or 0.0) for _, r in df.iterrows() }
        except Exception as e:
            logger.warning(f"⚠️  Velo map query failed (table may not exist): {e}")
            result = {}
        finally:
            conn.close()
        return result

    def build_umpire_map(self, start_date: date, end_date: date) -> Dict[int, float]:
        """
        Returns {game_id: plate_ump_runs_factor_smooth}
        Needs your loader to have filled umpire_game_assignments and umpire_stats (UmpScorecards).
        Smoothing by sample size (weight up to 80 games).
        """
        if not self._table_exists('umpire_game_assignments') or not self._table_exists('umpire_stats'):
            logger.info("ℹ️  Skipping umpire map: assignments/stats tables not present")
            return {}

        conn = self.get_db_connection()
        q = """
        WITH g AS (
          SELECT eg.game_id::text AS game_id_txt,
                 eg.date::date AS game_date,
                 ug.plate_ump
          FROM enhanced_games eg
          JOIN umpire_game_assignments ug
            ON ug.game_id::text = eg.game_id::text
          WHERE eg.date BETWEEN %s AND %s
        ),
        u AS (
          SELECT g.game_id_txt,
                 us.runs_factor, us.sample_games
          FROM g
          JOIN LATERAL (
            SELECT *
            FROM umpire_stats us
            WHERE us.umpire = g.plate_ump
              AND us.as_of_date <= g.game_date
            ORDER BY us.as_of_date DESC
            LIMIT 1
          ) us ON TRUE
        )
        SELECT game_id_txt,
               CASE WHEN sample_games IS NULL THEN 1.0
                    ELSE (LEAST(sample_games,80)::float/80.0) * COALESCE(runs_factor,1.0)
                       + (1.0 - LEAST(sample_games,80)::float/80.0) * 1.0
               END AS rf
        FROM u
        """
        try:
            df = pd.read_sql(q, conn, params=(start_date, end_date))
            # Return keyed by INT when possible, else ignore (compute_game_conditions will also guard)
            out = {}
            for _, r in df.iterrows():
                try:
                    gid = int(r.game_id_txt)
                    out[gid] = float(r.rf)
                except:
                    continue
            return out
        except Exception as e:
            logger.warning(f"⚠️  Umpire map query failed: {e}")
            return {}
        finally:
            conn.close()

    def build_travel_map(self, start_date: date, end_date: date) -> Dict[Tuple[int,str], Tuple[Optional[int], bool]]:
        """
        Returns {(game_id,'home'|'away'): (tz_shift_minutes, day_after_night)} from team_schedule.
        Assumes team_schedule has tz (ET/CT/MT/PT), start_local, is_night.
        """
        conn = self.get_db_connection()
        # current day schedule for teams in range
        sched_q = """
        SELECT ts.team, ts.date::date AS d, ts.tz, ts.start_local, ts.is_night
        FROM team_schedule ts
        WHERE ts.date BETWEEN %s AND %s
        """
        try:
            sch = pd.read_sql(sched_q, conn, params=(start_date, end_date))
            if sch.empty:
                return {}
                
            sch['tz_off'] = sch['tz'].apply(self._safetz)
            key = sch[['team','d','tz_off','start_local','is_night']].copy()

            # prior-day schedule
            sch_prev = sch.copy()
            sch_prev['d'] = sch_prev['d'] + pd.to_timedelta(1, unit='D')  # shift forward for join on "current date"
            prev_key = sch_prev[['team','d','tz_off','start_local','is_night']].rename(columns={
                'tz_off':'prev_tz_off','start_local':'prev_start','is_night':'prev_is_night'
            })

            merged = key.merge(prev_key, on=['team','d'], how='left')
            merged['tz_shift'] = merged['tz_off'] - merged['prev_tz_off']
            merged['day_after_night'] = (merged['prev_is_night'].fillna(False)) & (pd.to_datetime(merged['start_local'].astype(str)) < pd.to_datetime("16:00:00"))

            # map to game_id sides
            games_q = """
            SELECT eg.game_id, eg.date::date AS d, eg.home_team, eg.away_team
            FROM enhanced_games eg
            WHERE eg.date BETWEEN %s AND %s
            """
            g = pd.read_sql(games_q, conn, params=(start_date, end_date))

            # join by (team, date)
            mh = g.merge(merged, left_on=['home_team','d'], right_on=['team','d'], how='left')
            ma = g.merge(merged, left_on=['away_team','d'], right_on=['team','d'], how='left')

            out = {}
            for _, r in mh.iterrows():
                out[(int(r.game_id), 'home')] = (None if pd.isna(r.tz_shift) else int(r.tz_shift), bool(r.day_after_night) if not pd.isna(r.day_after_night) else False)
            for _, r in ma.iterrows():
                out[(int(r.game_id), 'away')] = (None if pd.isna(r.tz_shift) else int(r.tz_shift), bool(r.day_after_night) if not pd.isna(r.day_after_night) else False)
            return out
        except Exception as e:
            logger.warning(f"⚠️  Travel map query failed (team_schedule may not exist): {e}")
            return {}
        finally:
            conn.close()

    def build_missing_regulars_and_corexwoba(self, start_date, end_date):
        conn = self.get_db_connection()
        bio_pk = self._first_present_column('player_bio', ['mlb_id','player_id','id'])
        # if no bio_pk, we can still get SP ids from enhanced_games, but throws default to 'R'
        join_home = f"LEFT JOIN player_bio hsp ON hsp.{bio_pk} = eg.home_sp_id" if bio_pk else ""
        join_away = f"LEFT JOIN player_bio asp ON asp.{bio_pk} = eg.away_sp_id" if bio_pk else ""

        gq = f"""
        SELECT eg.game_id, eg.date::date AS d, eg.home_team, eg.away_team,
            eg.home_sp_id, eg.away_sp_id,
            COALESCE(hsp.throws,'R') AS home_sp_throw, COALESCE(asp.throws,'R') AS away_sp_throw
        FROM enhanced_games eg
        {join_home}
        {join_away}
        WHERE eg.date BETWEEN %s AND %s
        """

        
        try:
            games = pd.read_sql(gq, conn, params=(start_date, end_date))
            if games.empty:
                return {}

            # Latest lineup per game (we created a helper view)
            player_id_col = self._first_present_column("lineups_enh_latest", ["player_id", "mlb_id", "id"])
            if not player_id_col:
                logger.warning("No player ID column found in lineups_enh_latest")
                return {int(row.game_id): (0, 0, 0.300, 0.300, 0, 0) for _, row in games.iterrows()}
                
            lq = f"""
            SELECT l.game_id, l.spot, l.{player_id_col} as player_id, l.bats
            FROM lineups_enh_latest l
            JOIN enhanced_games eg ON eg.game_id::text = l.game_id::text
            WHERE eg.date BETWEEN %s AND %s
            """
            lineups = pd.read_sql(lq, conn, params=(start_date, end_date))

            # Historical starts last 30d to define "core" (simplified - use current lineups if no history)
            if lineups.empty:
                # No lineup data available, return defaults
                return {int(row.game_id): (0, 0, 0.300, 0.300, 0, 0) for _, row in games.iterrows()}

            # Batter rolling xwOBA (we'll use recent entry; fallback to 0.300)
            bxq = """
            SELECT player_id, date::date AS d, xwoba_v_rhp, xwoba_v_lhp
            FROM batter_daily_rolling
            WHERE date BETWEEN %s - INTERVAL '7 days' AND %s
            """
            try:
                bx = pd.read_sql(bxq, conn, params=(start_date, end_date))
            except:
                # batter_daily_rolling table doesn't exist, use defaults
                bx = pd.DataFrame()

            # Build lineup sets per game & helper functions
            lu = lineups.groupby('game_id')['player_id'].apply(list).reset_index().rename(columns={'player_id':'players'})
            lu_bats = lineups.groupby('game_id')['bats'].apply(list).reset_index().rename(columns={'bats':'bats'})

            # xwOBA vs SP hand helper
            bx_idx = {}
            if not bx.empty:
                bx['d'] = pd.to_datetime(bx['d']).dt.date
                for _, r in bx.iterrows():
                    bx_idx[(int(r.player_id), r['d'])] = (r.xwoba_v_rhp, r.xwoba_v_lhp)

            def lineup_xwoba(lineup_players, hand, d):
                vals = []
                for pid in lineup_players:
                    xr = bx_idx.get((int(pid), d))
                    if xr:
                        vals.append(xr[0] if hand=='R' else xr[1])
                    else:
                        vals.append(0.300)
                return float(sum(vals)/len(vals)) if vals else 0.300

            # platoon advantage: bats vs SP throw
            def platoon_adv(bats_list, hand):
                # advantage if L vs R or R vs L; switch counts as advantage vs both
                adv = 0
                for b in bats_list:
                    if b == 'S': adv += 1
                    elif b == 'L' and hand == 'R': adv += 1
                    elif b == 'R' and hand == 'L': adv += 1
                return int(adv)

            out = {}

            # Process each game
            gh = games.merge(lu, on='game_id', how='left').merge(lu_bats, on='game_id', how='left')
            for _, r in gh.iterrows():
                players = r.players if isinstance(r.players, list) else []
                bats = r.bats if isinstance(r.bats, list) else []
                d = pd.to_datetime(r.d).date()
                
                # Home team calculations
                home_xw = lineup_xwoba(players, r.away_sp_throw, d)  # home vs away SP
                home_padv = platoon_adv(bats, r.away_sp_throw)
                
                # Away team calculations  
                away_xw = lineup_xwoba(players, r.home_sp_throw, d)  # away vs home SP
                away_padv = platoon_adv(bats, r.home_sp_throw)
                
                # For now, set missing regulars to 0 (would need 30d history)
                out[int(r.game_id)] = (0, 0, home_xw, away_xw, home_padv, away_padv)

            return out
            
        except Exception as e:
            logger.warning(f"⚠️  Missing regulars/core xwOBA query failed: {e}")
            return {}
        finally:
            conn.close()

    def build_ip_cap_map(self, start_date: date, end_date: date) -> Dict[Tuple[int,date], float]:
        """
        Returns {(pitcher_id, game_date): ip_cap} using pitcher_game_logs (no IL feed needed).
        Heuristics:
          - Compute last 3 starts IP avg (L3_IP) and last start pitches.
          - If long gap (>=20d since last start) → cap = min(4.0, 0.8*L3_IP)
          - Else base cap = min(6.5, L3_IP + 0.5)
          - Pitch cap derived from last_pitches: ip_from_pitches = min(last_pitches + 15, 100)/15
          - Final = min(base cap, ip_from_pitches), not less than 2.0
        """
        if not self._table_exists('pitcher_game_logs'):
            logger.info("ℹ️  Skipping IP cap map: pitcher_game_logs not present")
            return {}
            
        conn = self.get_db_connection()
        q = """
        WITH span AS (
          SELECT %s::date AS d0, %s::date AS d1
        ),
        starts AS (
          SELECT pgl.pitcher_id, pgl.game_date::date AS d, pgl.innings_pitched AS ip, COALESCE(pgl.pitches_thrown, 15*pgl.innings_pitched)::int AS pitches
          FROM pitcher_game_logs pgl
          JOIN span s ON pgl.game_date BETWEEN s.d0 - INTERVAL '60 days' AND s.d1
          WHERE pgl.role IN ('SP','OPENER','BULK') AND pgl.innings_pitched > 0
        ),
        last3 AS (
          SELECT pitcher_id, d,
                 AVG(ip) OVER (PARTITION BY pitcher_id ORDER BY d ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS l3_ip,
                 LAG(d,1)  OVER (PARTITION BY pitcher_id ORDER BY d) AS prev_start_date,
                 LAG(pitches,1) OVER (PARTITION BY pitcher_id ORDER BY d) AS last_pitches
          FROM starts
        )
        SELECT pitcher_id, d AS date, l3_ip, prev_start_date, last_pitches
        FROM last3
        WHERE d BETWEEN %s AND %s
        """
        try:
            df = pd.read_sql(q, conn, params=(start_date, end_date, start_date, end_date))
            
            out = {}
            for _, r in df.iterrows():
                pid = int(r.pitcher_id)
                d = pd.to_datetime(r['date']).date()
                l3 = float(r.l3_ip) if not pd.isna(r.l3_ip) else 4.5
                prev_d = pd.to_datetime(r.prev_start_date).date() if pd.notna(r.prev_start_date) else None
                gap = (d - prev_d).days if prev_d else 999
                last_p = int(r.last_pitches) if not pd.isna(r.last_pitches) else 75
                if gap >= 20:
                    base = min(4.0, 0.8 * l3)
                else:
                    base = min(6.5, l3 + 0.5)
                ip_from_p = min(last_p + 15, 100) / 15.0
                cap = max(2.0, min(base, ip_from_p))
                out[(pid, d)] = cap
            return out
        except Exception as e:
            logger.warning(f"⚠️  IP cap query failed: {e}")
            return {}
        finally:
            conn.close()
    
    def populate_reliever_usage(self, start_dt, end_dt):
        """Populate reliever_usage table from pitcher_game_logs"""
        
        if not self._table_exists('pitcher_game_logs'):
            logger.info("⏭️  Skipping reliever usage: pitcher_game_logs not present")
            return
        
        logger.info(f"🔄 Populating reliever usage from {start_dt} to {end_dt}...")
        
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Extract reliever usage from pitcher game logs
        relief_extract_query = """
        INSERT INTO reliever_usage (
            pitcher_id, team, date, role, innings, pitches, 
            score_diff_entry, inning_entry, created_at, updated_at
        )
        SELECT 
            pgl.pitcher_id,
            pgl.team,
            pgl.game_date as date,
            pgl.role,
            pgl.innings_pitched as innings,
            COALESCE(pgl.pitches_thrown, CEIL(pgl.innings_pitched * 15))::int as pitches,
            NULL::int as score_diff_entry,
            NULL::int as inning_entry,
            NOW() as created_at,
            NOW() as updated_at
        FROM pitcher_game_logs pgl
        WHERE pgl.game_date BETWEEN %s AND %s
          AND pgl.role IN ('RP','CL','SU','OPENER','BULK')
          AND pgl.innings_pitched > 0
        ON CONFLICT (date, pitcher_id) DO UPDATE
        SET role      = EXCLUDED.role,
            team      = EXCLUDED.team,
            innings   = EXCLUDED.innings,
            pitches   = EXCLUDED.pitches,
            updated_at= NOW()
        """
        
        try:
            cursor.execute(relief_extract_query, (start_dt, end_dt))
            relief_count = cursor.rowcount
            conn.commit()
            logger.info(f"✅ Populated {relief_count} reliever usage records")
            
        except Exception as e:
            logger.error(f"❌ Error populating reliever usage: {str(e)}")
            conn.rollback()
        
        cursor.close()
        conn.close()
    
    def enhance_park_factors(self):
        """Enhance park factors with additional data"""
        
        logger.info("🔄 Enhancing park factors...")
        
        conn = self.get_db_connection()
        
        # Check current park factors coverage
        parks_query = """
        SELECT 
            venue_name,
            runs_factor,
            hr_factor,
            COUNT(*) as games_count
        FROM parks_dim
        JOIN venue_xref ON parks_dim.venue_name = venue_xref.normalized_venue
        JOIN enhanced_games eg ON eg.venue = venue_xref.raw_venue
        WHERE eg.date >= '2024-08-01'
        GROUP BY venue_name, runs_factor, hr_factor
        ORDER BY games_count DESC
        """
        
        parks_df = pd.read_sql(parks_query, conn)
        
        logger.info(f"📊 Current park factors:")
        for _, row in parks_df.iterrows():
            logger.info(f"   {row['venue_name']}: Runs={row['runs_factor']:.3f}, HR={row['hr_factor']:.3f} ({row['games_count']} games)")
        
        # TODO: Enhance with weather data, elevation, dimensions, etc.
        logger.info("⏳ Weather/conditions enhancement - TODO")
        
        conn.close()
    
    def run_quick_backfill(self):
        """Run all quick backfill operations"""
        
        logger.info("🚀 Starting quick historical data backfill...")
        
        # 1. Fill pitcher rolling gaps
        self.fill_pitcher_rolling_gaps()
        
        # 2. Collect missing handedness data
        self.collect_missing_handedness_data()
        
        # 3. Enhance park factors
        self.enhance_park_factors()
        
        logger.info("✅ Quick backfill completed!")
    
    def analyze_improvements(self):
        """Analyze data quality improvements after backfill (reflects actual computed data)."""
        logger.info("📊 Analyzing data quality improvements...")
        
        conn = self.get_db_connection()
        try:
            # What venue column do we have?
            eg_venue_col = self._first_present_column('enhanced_games', ['venue','venue_name']) or 'venue'
            # Parks join (optional)
            parks_join = f"LEFT JOIN park_meta pm ON pm.venue_name = eg.{eg_venue_col}"

            q = f"""
            WITH rg AS (
              SELECT
                eg.game_id, eg.date::date AS d,
                gc.home_lineup_core_xwoba, gc.away_lineup_core_xwoba,
                gc.home_sp_velo_delta_L3, gc.away_sp_velo_delta_L3,
                gc.home_bp_fatigue, gc.away_bp_fatigue,
                pm.venue_name
              FROM enhanced_games eg
              JOIN game_conditions gc ON CAST(gc.game_id AS TEXT) = CAST(eg.game_id AS TEXT)
              {parks_join}
              WHERE eg.date >= '2024-08-15' AND eg.total_runs IS NOT NULL
            )
            SELECT
              COUNT(*) AS total_games,
              COUNT(home_lineup_core_xwoba) + COUNT(away_lineup_core_xwoba) AS lu_core_present_twice,
              ROUND(100.0 * (COUNT(home_lineup_core_xwoba) + COUNT(away_lineup_core_xwoba)) / (2*COUNT(*)), 1) AS lineup_core_pct,
              COUNT(venue_name) AS park_meta_rows,
              ROUND(100.0 * COUNT(venue_name) / COUNT(*), 1) AS park_meta_pct
            FROM rg
            """
            df = pd.read_sql(q, conn)
            r = df.iloc[0]
            logger.info("\n=== POST-BACKFILL DATA QUALITY REPORT ===")
            logger.info(f"📊 Analysis of {int(r['total_games'])} recent games:")
            logger.info(f"   🧰 Lineup-vs-hand (from game_conditions): {r['lineup_core_pct']}% coverage")
            logger.info(f"   🏟️  Park meta present: {r['park_meta_pct']}% coverage")
        finally:
            conn.close()

def main():
    """Main execution function"""
    
    print("🔄 ENHANCED HISTORICAL DATA BACKFILL")
    print("=" * 45)
    
    backfiller = EnhancedHistoricalBackfill()
    
    if len(sys.argv) >= 3:
        backfiller.run_enhanced_backfill(sys.argv[1], sys.argv[2])
    else:
        backfiller.run_enhanced_backfill()

if __name__ == "__main__":
    main()
