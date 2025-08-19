-- Enhanced Feature Pipeline Database Migrations
-- ===============================================
-- Adds missing columns identified during enhanced feature development
-- Run these migrations to enable full enhanced feature pipeline functionality

-- Enable timing for migrations
\timing

BEGIN;

-- ============================================================================
-- 1. ENHANCED_GAMES TABLE ENHANCEMENTS
-- ============================================================================

-- Add umpire-related columns
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='plate_umpire') THEN
        ALTER TABLE enhanced_games ADD COLUMN plate_umpire TEXT;
        RAISE NOTICE 'Added plate_umpire column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='umpire_ou_tendency') THEN
        ALTER TABLE enhanced_games ADD COLUMN umpire_ou_tendency NUMERIC DEFAULT 0.0;
        RAISE NOTICE 'Added umpire_ou_tendency column to enhanced_games';
    END IF;
END $$;

-- Add venue/ballpark details
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='venue') THEN
        ALTER TABLE enhanced_games ADD COLUMN venue TEXT;
        RAISE NOTICE 'Added venue column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='ballpark') THEN
        ALTER TABLE enhanced_games ADD COLUMN ballpark TEXT;
        RAISE NOTICE 'Added ballpark column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='roof_type') THEN
        ALTER TABLE enhanced_games ADD COLUMN roof_type TEXT DEFAULT 'open'; -- 'open', 'retractable', 'dome'
        RAISE NOTICE 'Added roof_type column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='roof_status') THEN
        ALTER TABLE enhanced_games ADD COLUMN roof_status TEXT DEFAULT 'open'; -- 'open', 'closed'
        RAISE NOTICE 'Added roof_status column to enhanced_games';
    END IF;
END $$;

-- Add enhanced weather columns  
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='humidity') THEN
        ALTER TABLE enhanced_games ADD COLUMN humidity INTEGER;
        RAISE NOTICE 'Added humidity column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='wind_direction_deg') THEN
        ALTER TABLE enhanced_games ADD COLUMN wind_direction_deg INTEGER;
        RAISE NOTICE 'Added wind_direction_deg column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='air_pressure') THEN
        ALTER TABLE enhanced_games ADD COLUMN air_pressure NUMERIC;
        RAISE NOTICE 'Added air_pressure column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='dew_point') THEN
        ALTER TABLE enhanced_games ADD COLUMN dew_point INTEGER;
        RAISE NOTICE 'Added dew_point column to enhanced_games';
    END IF;
END $$;

-- Add pitcher information columns
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='home_sp_hand') THEN
        ALTER TABLE enhanced_games ADD COLUMN home_sp_hand CHAR(1); -- 'L' or 'R'
        RAISE NOTICE 'Added home_sp_hand column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='away_sp_hand') THEN
        ALTER TABLE enhanced_games ADD COLUMN away_sp_hand CHAR(1); -- 'L' or 'R' 
        RAISE NOTICE 'Added away_sp_hand column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='home_sp_days_rest') THEN
        ALTER TABLE enhanced_games ADD COLUMN home_sp_days_rest INTEGER DEFAULT 4;
        RAISE NOTICE 'Added home_sp_days_rest column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='away_sp_days_rest') THEN
        ALTER TABLE enhanced_games ADD COLUMN away_sp_days_rest INTEGER DEFAULT 4;
        RAISE NOTICE 'Added away_sp_days_rest column to enhanced_games';
    END IF;
END $$;

-- Add game context columns
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='series_game') THEN
        ALTER TABLE enhanced_games ADD COLUMN series_game INTEGER; -- 1, 2, 3, 4 for game number in series
        RAISE NOTICE 'Added series_game column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='getaway_day') THEN
        ALTER TABLE enhanced_games ADD COLUMN getaway_day BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added getaway_day column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='doubleheader') THEN
        ALTER TABLE enhanced_games ADD COLUMN doubleheader BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added doubleheader column to enhanced_games';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='enhanced_games' AND column_name='day_after_night') THEN
        ALTER TABLE enhanced_games ADD COLUMN day_after_night BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added day_after_night column to enhanced_games';
    END IF;
END $$;

-- ============================================================================
-- 2. LINEUPS TABLE ENHANCEMENTS
-- ============================================================================

-- Add missing lineup analysis columns
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='date') THEN
        ALTER TABLE lineups ADD COLUMN date DATE;
        RAISE NOTICE 'Added date column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='lineup_wrcplus') THEN
        ALTER TABLE lineups ADD COLUMN lineup_wrcplus INTEGER DEFAULT 100;
        RAISE NOTICE 'Added lineup_wrcplus column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='vs_lhp_ops') THEN
        ALTER TABLE lineups ADD COLUMN vs_lhp_ops NUMERIC DEFAULT 0.750;
        RAISE NOTICE 'Added vs_lhp_ops column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='vs_rhp_ops') THEN
        ALTER TABLE lineups ADD COLUMN vs_rhp_ops NUMERIC DEFAULT 0.750;
        RAISE NOTICE 'Added vs_rhp_ops column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='lhb_count') THEN
        ALTER TABLE lineups ADD COLUMN lhb_count INTEGER DEFAULT 4;
        RAISE NOTICE 'Added lhb_count column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='rhb_count') THEN
        ALTER TABLE lineups ADD COLUMN rhb_count INTEGER DEFAULT 5;
        RAISE NOTICE 'Added rhb_count column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='star_players_out') THEN
        ALTER TABLE lineups ADD COLUMN star_players_out INTEGER DEFAULT 0;
        RAISE NOTICE 'Added star_players_out column to lineups';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='lineups' AND column_name='lineup_confirmed') THEN
        ALTER TABLE lineups ADD COLUMN lineup_confirmed BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added lineup_confirmed column to lineups';
    END IF;
END $$;

-- ============================================================================
-- 3. PITCHER_COMPREHENSIVE_STATS TABLE ENHANCEMENTS (if exists)
-- ============================================================================

-- Check if table exists and add columns for recent form tracking
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='pitcher_comprehensive_stats') THEN
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='pitcher_comprehensive_stats' AND column_name='ip') THEN
            ALTER TABLE pitcher_comprehensive_stats ADD COLUMN ip NUMERIC;
            RAISE NOTICE 'Added ip column to pitcher_comprehensive_stats';
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='pitcher_comprehensive_stats' AND column_name='er') THEN
            ALTER TABLE pitcher_comprehensive_stats ADD COLUMN er INTEGER;
            RAISE NOTICE 'Added er column to pitcher_comprehensive_stats';
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='pitcher_comprehensive_stats' AND column_name='pitches') THEN
            ALTER TABLE pitcher_comprehensive_stats ADD COLUMN pitches INTEGER;
            RAISE NOTICE 'Added pitches column to pitcher_comprehensive_stats';
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='pitcher_comprehensive_stats' AND column_name='pitcher_name') THEN
            ALTER TABLE pitcher_comprehensive_stats ADD COLUMN pitcher_name TEXT;
            RAISE NOTICE 'Added pitcher_name column to pitcher_comprehensive_stats';
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='pitcher_comprehensive_stats' AND column_name='date') THEN
            ALTER TABLE pitcher_comprehensive_stats ADD COLUMN date DATE;
            RAISE NOTICE 'Added date column to pitcher_comprehensive_stats';
        END IF;
        
    ELSE
        RAISE NOTICE 'pitcher_comprehensive_stats table does not exist - creating it';
        
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
            fip NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_pitcher_stats_name_date ON pitcher_comprehensive_stats(pitcher_name, date DESC);
        CREATE INDEX idx_pitcher_stats_game_id ON pitcher_comprehensive_stats(game_id);
        
        RAISE NOTICE 'Created pitcher_comprehensive_stats table with indexes';
    END IF;
END $$;

-- ============================================================================
-- 4. CREATE ENHANCED FEATURE SUPPORT TABLES (if needed)
-- ============================================================================

-- Catcher framing stats table
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='catcher_framing_stats') THEN
        CREATE TABLE catcher_framing_stats (
            id SERIAL PRIMARY KEY,
            catcher_name TEXT NOT NULL,
            team TEXT,
            date DATE NOT NULL,
            framing_runs NUMERIC DEFAULT 0.0,
            strike_rate NUMERIC DEFAULT 0.5,
            called_strikes INTEGER DEFAULT 0,
            called_balls INTEGER DEFAULT 0,
            edge_calls INTEGER DEFAULT 0,
            edge_strikes INTEGER DEFAULT 0,
            csaa NUMERIC DEFAULT 0.0, -- Called Strike Above Average
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_catcher_framing_name_date ON catcher_framing_stats(catcher_name, date DESC);
        
        RAISE NOTICE 'Created catcher_framing_stats table';
    END IF;
END $$;

-- Team travel log table
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='team_travel_log') THEN
        CREATE TABLE team_travel_log (
            id SERIAL PRIMARY KEY,
            team TEXT NOT NULL,
            date DATE NOT NULL,
            venue TEXT,
            venue_city TEXT,
            venue_state TEXT,
            venue_timezone TEXT,
            travel_distance_miles INTEGER DEFAULT 0,
            timezone_change INTEGER DEFAULT 0, -- hours difference
            games_in_last_7 INTEGER DEFAULT 0,
            home_away_switch BOOLEAN DEFAULT FALSE,
            cross_country_travel BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_travel_team_date ON team_travel_log(team, date DESC);
        
        RAISE NOTICE 'Created team_travel_log table';
    END IF;
END $$;

-- ============================================================================
-- 5. UPDATE EXISTING DATA WITH DEFAULTS
-- ============================================================================

-- Update enhanced_games with venue names from venue_name where venue is null
UPDATE enhanced_games 
SET venue = venue_name 
WHERE venue IS NULL AND venue_name IS NOT NULL;

-- Set default roof types for known stadiums
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
WHERE roof_type IS NULL;

-- Set default roof status (closed for domes, open for others initially)
UPDATE enhanced_games 
SET roof_status = CASE 
    WHEN roof_type = 'dome' THEN 'closed'
    ELSE 'open'
END
WHERE roof_status IS NULL;

-- ============================================================================
-- 6. CREATE INDEXES FOR PERFORMANCE
-- ============================================================================

-- Enhanced Games indexes for new columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_games_plate_umpire 
ON enhanced_games(plate_umpire) WHERE plate_umpire IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_games_venue 
ON enhanced_games(venue) WHERE venue IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_games_roof 
ON enhanced_games(roof_type, roof_status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_games_pitcher_names 
ON enhanced_games(home_sp_name, away_sp_name);

-- Lineups indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lineups_team_date 
ON lineups(team, date DESC) WHERE date IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lineups_game_team 
ON lineups(game_id, team);

-- ============================================================================
-- 7. GRANT PERMISSIONS (if needed)
-- ============================================================================

-- Grant permissions to application user (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- ============================================================================
-- 8. MIGRATION SUMMARY
-- ============================================================================

DO $$ 
BEGIN 
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'ENHANCED FEATURE PIPELINE MIGRATION COMPLETE';
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Added columns to support:';
    RAISE NOTICE '- Umpire tendencies (plate_umpire, umpire_ou_tendency)';
    RAISE NOTICE '- Enhanced weather (humidity, wind_direction_deg, air_pressure, dew_point)';
    RAISE NOTICE '- Venue details (venue, ballpark, roof_type, roof_status)';
    RAISE NOTICE '- Pitcher info (handedness, days_rest)';
    RAISE NOTICE '- Game context (series_game, getaway_day, doubleheader, day_after_night)';
    RAISE NOTICE '- Lineup analysis (lineup_wrcplus, vs_lhp_ops, vs_rhp_ops, lhb_count, etc.)';
    RAISE NOTICE '- Pitcher recent form tracking (pitcher_comprehensive_stats enhancements)';
    RAISE NOTICE '- Catcher framing stats (new table)';
    RAISE NOTICE '- Team travel tracking (new table)';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Populate new columns with actual data as it becomes available';
    RAISE NOTICE '2. Update data collection scripts to populate these columns';
    RAISE NOTICE '3. Re-run enhanced feature pipeline to utilize new data';
    RAISE NOTICE '============================================================================';
END $$;

COMMIT;

-- Optional: Show final table structures
\d enhanced_games
\d lineups
\d pitcher_comprehensive_stats
\d catcher_framing_stats  
\d team_travel_log
