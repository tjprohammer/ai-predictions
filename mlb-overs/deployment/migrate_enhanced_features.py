#!/usr/bin/env python3
"""
Enhanced Features Database Migration Script
==========================================
Adds missing database columns to support enhanced feature pipeline

This script safely adds columns using IF NOT EXISTS logic to avoid conflicts.
"""

import os
import sys
import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_connection():
    """Get database connection from environment variables or default settings"""
    try:
        # Try environment variables first - unified to 'mlb' database
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'mlb'),  # Fixed: unified to mlb database
            user=os.getenv('DB_USER', 'mlbuser'),   # Updated default user
            password=os.getenv('DB_PASSWORD', 'mlbpass')  # Updated default password
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def column_exists(cursor, table_name, column_name, schema='public'):
    """Check if a column exists in a table with schema qualification"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        )
    """, (schema, table_name, column_name))
    return cursor.fetchone()[0]

def table_exists(cursor, table_name, schema='public'):
    """Check if a table exists with schema qualification"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = %s
        )
    """, (schema, table_name))
    return cursor.fetchone()[0]

def add_column_safe(cursor, table_name, column_name, column_definition):
    """Safely add a column if it doesn't exist - avoids table rewrites with no DEFAULT"""
    if not column_exists(cursor, table_name, column_name):
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
            logger.info(f"✅ Added column {column_name} to {table_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add column {column_name} to {table_name}: {e}")
            return False
    else:
        logger.info(f"⏭️  Column {column_name} already exists in {table_name}")
        return True

def create_index_concurrent(cursor, index_name, create_sql):
    """Create index concurrently to avoid table locks"""
    try:
        cursor.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} {create_sql}")
        logger.info(f"✅ Created/verified concurrent index {index_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create concurrent index {index_name}: {e}")
        return False

def create_table_safe(cursor, table_name, create_sql):
    """Safely create a table if it doesn't exist"""
    if not table_exists(cursor, table_name):
        try:
            cursor.execute(create_sql)
            logger.info(f"✅ Created table {table_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create table {table_name}: {e}")
            return False
    else:
        logger.info(f"⏭️  Table {table_name} already exists")
        return True

def create_index_safe(cursor, index_name, create_sql):
    """Safely create an index if it doesn't exist"""
    try:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} {create_sql}")
        logger.info(f"✅ Created/verified index {index_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create index {index_name}: {e}")
        return False

def run_migration():
    """Run the complete database migration"""
    conn = get_database_connection()
    if not conn:
        logger.error("Could not connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        
        logger.info("🚀 Starting Enhanced Features Database Migration")
        logger.info("=" * 60)
        
        # ===================================================================
        # ENHANCED_GAMES TABLE MIGRATIONS
        # ===================================================================
        logger.info("📋 Migrating enhanced_games table...")
        
        enhanced_games_columns = [
            # Core umpire and venue columns (no defaults to avoid table rewrite)
            ("plate_umpire", "TEXT"),
            ("umpire_ou_tendency", "DOUBLE PRECISION"),
            ("venue", "TEXT"),
            ("ballpark", "TEXT"),
            ("roof_type", "TEXT"),
            ("roof_status", "TEXT"),
            
            # Enhanced weather columns
            ("humidity", "INTEGER"),
            ("wind_direction_deg", "INTEGER"),
            ("wind_speed", "DOUBLE PRECISION"),     # Add missing wind_speed for index
            ("air_pressure", "DOUBLE PRECISION"),    # Use DOUBLE PRECISION for continuous features
            ("dew_point", "INTEGER"),
            ("wind_gust", "DOUBLE PRECISION"),
            ("precip_prob", "DOUBLE PRECISION"),
            
            # Enhanced ballpark data
            ("park_cf_bearing_deg", "SMALLINT"),  # 0-359 degrees to center field
            ("game_time_utc", "TIMESTAMPTZ"),     # scheduled start in UTC
            ("game_timezone", "TEXT"),            # e.g., 'America/Chicago'
            
            # Pitcher information (no defaults)
            ("home_sp_hand", "CHAR(1)"),
            ("away_sp_hand", "CHAR(1)"),
            ("home_sp_days_rest", "INTEGER"),
            ("away_sp_days_rest", "INTEGER"),
            
            # Catcher information for framing stats
            ("home_catcher", "TEXT"),
            ("away_catcher", "TEXT"),
            
            # Game context (no defaults)
            ("series_game", "INTEGER"),
            ("getaway_day", "BOOLEAN"),
            ("doubleheader", "BOOLEAN"),
            ("day_after_night", "BOOLEAN")
        ]
        
        for column_name, column_def in enhanced_games_columns:
            add_column_safe(cursor, "enhanced_games", column_name, column_def)
        
        # ===================================================================
        # LINEUPS TABLE MIGRATIONS  
        # ===================================================================
        logger.info("👥 Migrating lineups table...")
        
        lineups_columns = [
            ("date", "DATE"),
            ("lineup_wrcplus", "INTEGER"),
            ("vs_lhp_ops", "DOUBLE PRECISION"),    # Use DOUBLE PRECISION for continuous stats
            ("vs_rhp_ops", "DOUBLE PRECISION"),    # Use DOUBLE PRECISION for continuous stats
            ("lhb_count", "INTEGER"),
            ("rhb_count", "INTEGER"),
            ("star_players_out", "INTEGER"),
            ("lineup_confirmed", "BOOLEAN")
        ]
        
        for column_name, column_def in lineups_columns:
            add_column_safe(cursor, "lineups", column_name, column_def)
        
        # ===================================================================
        # CREATE SUPPORTING TABLES
        # ===================================================================
        logger.info("🏗️  Creating supporting tables...")
        
        # Pitcher comprehensive stats table
        pitcher_stats_sql = """
        CREATE TABLE pitcher_comprehensive_stats (
            id SERIAL PRIMARY KEY,
            pitcher_name TEXT NOT NULL,
            date DATE NOT NULL,
            game_id TEXT,
            team TEXT,
            opponent TEXT,
            ip NUMERIC,
            er INTEGER,
            h INTEGER,
            bb INTEGER,
            k INTEGER,
            hr INTEGER,
            pitches INTEGER,
            strikes INTEGER,
            balls INTEGER,
            game_score INTEGER,
            fip DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        create_table_safe(cursor, "pitcher_comprehensive_stats", pitcher_stats_sql)
        
        # Catcher framing stats table
        catcher_framing_sql = """
        CREATE TABLE catcher_framing_stats (
            id SERIAL PRIMARY KEY,
            catcher_name TEXT NOT NULL,
            team TEXT,
            date DATE NOT NULL,
            framing_runs DOUBLE PRECISION DEFAULT 0.0,
            strike_rate DOUBLE PRECISION DEFAULT 0.5,
            called_strikes INTEGER DEFAULT 0,
            called_balls INTEGER DEFAULT 0,
            edge_calls INTEGER DEFAULT 0,
            edge_strikes INTEGER DEFAULT 0,
            csaa DOUBLE PRECISION DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        create_table_safe(cursor, "catcher_framing_stats", catcher_framing_sql)
        
        # Team travel log table
        travel_log_sql = """
        CREATE TABLE team_travel_log (
            id SERIAL PRIMARY KEY,
            team TEXT NOT NULL,
            date DATE NOT NULL,
            venue TEXT,
            venue_city TEXT,
            venue_state TEXT,
            venue_timezone TEXT,
            travel_distance_miles INTEGER DEFAULT 0,
            timezone_change INTEGER DEFAULT 0,
            games_in_last_7 INTEGER DEFAULT 0,
            home_away_switch BOOLEAN DEFAULT FALSE,
            cross_country_travel BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        create_table_safe(cursor, "team_travel_log", travel_log_sql)
        
        # ===================================================================
        # SET DEFAULTS AND BACKFILL DATA
        # ===================================================================
        logger.info("🔄 Setting defaults and backfilling data...")
        
        # Set defaults for new columns (done after adding to avoid table rewrite)
        default_updates = [
            ("umpire_ou_tendency", "0.0", "umpire_ou_tendency IS NULL"),
            ("roof_type", "'open'", "roof_type IS NULL"),
            ("roof_status", "'open'", "roof_status IS NULL"),
            ("home_sp_days_rest", "4", "home_sp_days_rest IS NULL"),
            ("away_sp_days_rest", "4", "away_sp_days_rest IS NULL"),
            ("getaway_day", "FALSE", "getaway_day IS NULL"),
            ("doubleheader", "FALSE", "doubleheader IS NULL"),
            ("day_after_night", "FALSE", "day_after_night IS NULL")
        ]
        
        for column, default_value, condition in default_updates:
            if column_exists(cursor, "enhanced_games", column):
                cursor.execute(f"UPDATE enhanced_games SET {column} = {default_value} WHERE {condition}")
                updated_count = cursor.rowcount
                logger.info(f"✅ Set default {default_value} for {updated_count} rows in {column}")
        
        # Lineup defaults
        lineup_defaults = [
            ("lineup_wrcplus", "100", "lineup_wrcplus IS NULL"),
            ("vs_lhp_ops", "0.750", "vs_lhp_ops IS NULL"),
            ("vs_rhp_ops", "0.750", "vs_rhp_ops IS NULL"),
            ("lhb_count", "4", "lhb_count IS NULL"),
            ("rhb_count", "5", "rhb_count IS NULL"),
            ("star_players_out", "0", "star_players_out IS NULL"),
            ("lineup_confirmed", "FALSE", "lineup_confirmed IS NULL")
        ]
        
        for column, default_value, condition in lineup_defaults:
            if column_exists(cursor, "lineups", column):
                cursor.execute(f"UPDATE lineups SET {column} = {default_value} WHERE {condition}")
                updated_count = cursor.rowcount
                logger.info(f"✅ Set default {default_value} for {updated_count} rows in {column}")
        
        # Update venue column with existing venue_name data (only when NULL)
        cursor.execute("""
            UPDATE enhanced_games 
            SET venue = venue_name 
            WHERE venue IS NULL AND venue_name IS NOT NULL
        """)
        updated_venues = cursor.rowcount
        logger.info(f"✅ Updated {updated_venues} venue fields from venue_name")
        
        # Set roof types for known stadiums (only when NULL to preserve live data)
        cursor.execute("""
            UPDATE enhanced_games 
            SET roof_type = CASE 
                WHEN venue_name ILIKE '%Tropicana%' THEN 'dome'
                WHEN venue_name ILIKE '%Minute Maid%' THEN 'retractable'
                WHEN venue_name ILIKE '%Rogers Centre%' THEN 'retractable'
                WHEN venue_name ILIKE '%Chase Field%' THEN 'retractable'
                WHEN venue_name ILIKE '%T-Mobile Park%' THEN 'retractable'
                WHEN venue_name ILIKE '%American Family Field%' THEN 'retractable'
                WHEN venue_name ILIKE '%Marlins Park%' THEN 'retractable'
                ELSE 'open'
            END
            WHERE roof_type IS NULL
        """)
        updated_roofs = cursor.rowcount
        logger.info(f"✅ Updated {updated_roofs} roof type fields")
        
        # Set roof status (only when NULL to preserve live statuses)
        cursor.execute("""
            UPDATE enhanced_games 
            SET roof_status = CASE 
                WHEN roof_type = 'dome' THEN 'closed'
                ELSE 'open'
            END
            WHERE roof_status IS NULL
        """)
        updated_status = cursor.rowcount
        logger.info(f"✅ Updated {updated_status} roof status fields")
        
        # Set defaults for new columns after creation
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN umpire_ou_tendency SET DEFAULT 0.0")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN roof_type SET DEFAULT 'open'")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN roof_status SET DEFAULT 'open'")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN home_sp_days_rest SET DEFAULT 4")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN away_sp_days_rest SET DEFAULT 4")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN getaway_day SET DEFAULT FALSE")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN doubleheader SET DEFAULT FALSE")
        cursor.execute("ALTER TABLE enhanced_games ALTER COLUMN day_after_night SET DEFAULT FALSE")
        
        cursor.execute("ALTER TABLE lineups ALTER COLUMN lineup_wrcplus SET DEFAULT 100")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN vs_lhp_ops SET DEFAULT 0.750")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN vs_rhp_ops SET DEFAULT 0.750")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN lhb_count SET DEFAULT 4")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN rhb_count SET DEFAULT 5")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN star_players_out SET DEFAULT 0")
        cursor.execute("ALTER TABLE lineups ALTER COLUMN lineup_confirmed SET DEFAULT FALSE")
        
        logger.info("✅ Set column defaults for future inserts")
        
        # Commit DDL changes before creating concurrent indexes
        conn.commit()
        logger.info("✅ Committed schema changes")
        
        # ===================================================================
        # CREATE PERFORMANCE VIEWS
        # ===================================================================
        logger.info("📊 Creating performance views...")
        
        # Team travel snapshot view (optimized with DISTINCT ON)
        cursor.execute("""
            CREATE OR REPLACE VIEW team_travel_snapshot AS
            SELECT DISTINCT ON (team, date)
                   team, date, travel_distance_miles, timezone_change, games_in_last_7,
                   home_away_switch, cross_country_travel
            FROM team_travel_log
            ORDER BY team, date, created_at DESC
        """)
        logger.info("✅ Created optimized team_travel_snapshot view")
        
        # Enhanced games features view for one-stop queries
        cursor.execute("""
            CREATE OR REPLACE VIEW enhanced_games_features_v AS
            SELECT eg.*,
                   lh.lineup_wrcplus  AS home_lineup_wrcplus,
                   la.lineup_wrcplus  AS away_lineup_wrcplus,
                   lh.vs_lhp_ops      AS home_vs_lhp_ops,
                   lh.vs_rhp_ops      AS home_vs_rhp_ops,
                   la.vs_lhp_ops      AS away_vs_lhp_ops,
                   la.vs_rhp_ops      AS away_vs_rhp_ops,
                   lh.lhb_count       AS home_lhb_count,
                   lh.rhb_count       AS home_rhb_count,
                   la.lhb_count       AS away_lhb_count,
                   la.rhb_count       AS away_rhb_count,
                   lh.star_players_out AS home_star_players_out,
                   la.star_players_out AS away_star_players_out,
                   th.travel_distance_miles AS home_travel_distance_miles,
                   th.timezone_change       AS home_timezone_change,
                   th.games_in_last_7       AS home_games_in_last_7,
                   th.home_away_switch      AS home_home_away_switch,
                   th.cross_country_travel  AS home_cross_country_travel,
                   ta.travel_distance_miles AS away_travel_distance_miles,
                   ta.timezone_change       AS away_timezone_change,
                   ta.games_in_last_7       AS away_games_in_last_7,
                   ta.home_away_switch      AS away_home_away_switch,
                   ta.cross_country_travel  AS away_cross_country_travel
            FROM enhanced_games eg
            LEFT JOIN lineups lh ON lh.game_id = eg.game_id AND lh.team = eg.home_team AND lh.date = eg."date"
            LEFT JOIN lineups la ON la.game_id = eg.game_id AND la.team = eg.away_team AND la.date = eg."date"
            LEFT JOIN team_travel_snapshot th ON th.team = eg.home_team AND th.date = eg."date"
            LEFT JOIN team_travel_snapshot ta ON ta.team = eg.away_team AND ta.date = eg."date"
        """)
        logger.info("✅ Created enhanced_games_features_v view")
        
        # Commit view changes
        conn.commit()
        logger.info("✅ Committed view changes")
        
        # ===================================================================
        # CREATE CRITICAL UNIQUE INDEX (CONCURRENT)
        # ===================================================================
        logger.info("🔑 Creating critical unique index concurrently...")
        
        # Switch to autocommit for concurrent index creation
        conn.set_session(autocommit=True)
        
        # Try concurrent unique index first, fall back to non-unique
        try:
            cursor.execute('CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_enhanced_games_gid_date ON enhanced_games(game_id, "date")')
            logger.info("✅ Created unique concurrent index on (game_id, date)")
        except Exception as e:
            logger.warning(f"⚠️ Unique concurrent index failed (likely duplicates): {e}")
            cursor.execute('CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_games_gid_date ON enhanced_games(game_id, "date")')
            logger.info("✅ Created non-unique concurrent index on (game_id, date)")
        
        # ===================================================================
        # CREATE CONCURRENT INDEXES (NON-BLOCKING)
        # ===================================================================
        logger.info("📇 Creating concurrent indexes (non-blocking)...")
        
        # Create concurrent indexes to avoid table locks
        concurrent_indexes = [
            ("idx_enhanced_games_plate_umpire", "ON enhanced_games USING btree (plate_umpire)"),
            ("idx_enhanced_games_venue", "ON enhanced_games USING btree (venue)"),
            ("idx_enhanced_games_roof", "ON enhanced_games USING btree (roof_type, roof_status)"),
            ("idx_enhanced_games_pitcher_names", "ON enhanced_games USING btree (home_sp_name, away_sp_name)"),
            ("idx_enhanced_games_catchers", "ON enhanced_games USING btree (home_catcher, away_catcher)"),
            ("idx_enhanced_games_weather", "ON enhanced_games USING btree (wind_direction_deg, wind_speed)"),
            ("idx_lineups_team_date", "ON lineups USING btree (team, date DESC)"),
            ("idx_lineups_game_team", "ON lineups USING btree (game_id, team)"),
            ("idx_lineups_gid_team_date", "ON lineups USING btree (game_id, team, date)"),  # Matches JOIN pattern
            ("idx_travel_team_date_created", "ON team_travel_log USING btree (team, date, created_at DESC)"),  # Matches view ORDER BY
            ("idx_pitcher_stats_name_date", "ON pitcher_comprehensive_stats USING btree (pitcher_name, date DESC)"),
            ("idx_pitcher_stats_game_id", "ON pitcher_comprehensive_stats USING btree (game_id)"),
            ("idx_catcher_framing_name_date", "ON catcher_framing_stats USING btree (catcher_name, date DESC)")
        ]
        
        for index_name, index_sql in concurrent_indexes:
            create_index_concurrent(cursor, index_name, index_sql)
        
        # ===================================================================
        # ANALYZE TABLES FOR PLANNER STATS
        # ===================================================================
        logger.info("📊 Updating table statistics...")
        
        analyze_tables = ["enhanced_games", "lineups", "team_travel_log", "pitcher_comprehensive_stats", "catcher_framing_stats"]
        for table in analyze_tables:
            if table_exists(cursor, table):
                cursor.execute(f"ANALYZE {table}")
                logger.info(f"✅ Analyzed {table} for planner statistics")
        
        # Return to transaction mode and commit
        conn.set_session(autocommit=False)
        conn.commit()
        
        logger.info("=" * 60)
        logger.info("🎉 Enhanced Features Database Migration COMPLETE!")
        logger.info("=" * 60)
        logger.info("Added support for:")
        logger.info("  • Umpire tendencies and plate umpire tracking")
        logger.info("  • Enhanced weather data (humidity, wind direction, pressure, gusts)")
        logger.info("  • Ballpark details (center field bearing, roof info, timezones)")
        logger.info("  • Catcher tracking for framing statistics")
        logger.info("  • Pitcher handedness and rest tracking")
        logger.info("  • Game context (series position, travel factors)")
        logger.info("  • Advanced lineup analysis (handedness splits, wRC+)")
        logger.info("  • Pitcher recent form tracking")
        logger.info("  • Team travel and fatigue tracking")
        logger.info("  • Performance views for one-stop feature queries")
        logger.info("")
        logger.info("🚀 Model-ready features now available:")
        logger.info("  • wind_out_to_cf: wind_direction_deg + wind_speed + park_cf_bearing_deg")
        logger.info("  • roof_open: roof_status = 'open'")
        logger.info("  • air_density_index: temperature + humidity + air_pressure")
        logger.info("  • lineup_handedness: home/away_lhb_count vs SP hand")
        logger.info("  • travel_fatigue: distance_miles + timezone_change + games_in_last_7")
        logger.info("  • catcher_framing: home/away_catcher + framing_stats")
        logger.info("")
        logger.info("📊 New views available:")
        logger.info("  • team_travel_snapshot: Latest travel data per team/date")
        logger.info("  • enhanced_games_features_v: One-stop shop for all features")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Update data collection scripts to populate new columns")
        logger.info("  2. Use enhanced_games_features_v for faster feature queries")
        logger.info("  3. Validate new feature signals in model performance")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    if run_migration():
        logger.info("✅ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed")
        sys.exit(1)
