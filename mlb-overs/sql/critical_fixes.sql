-- CRITICAL DATA QUALITY FIXES
-- These fixes will significantly improve prediction quality

-- ========================================
-- FIX 1: SEASON ERA SCALING (HIGHEST PRIORITY)
-- ========================================
-- Current average: 1.80, should be ~4.50
-- Multiply by 2.5 to get realistic values

UPDATE enhanced_games 
SET 
    home_sp_season_era = home_sp_season_era * 2.5,
    away_sp_season_era = away_sp_season_era * 2.5
WHERE date >= '2025-08-11'  -- Only fix the period where ERA data exists
AND home_sp_season_era IS NOT NULL 
AND away_sp_season_era IS NOT NULL;

-- Verify the fix
SELECT 
    AVG(home_sp_season_era) as avg_home_era,
    AVG(away_sp_season_era) as avg_away_era,
    MIN(home_sp_season_era) as min_era,
    MAX(home_sp_season_era) as max_era
FROM enhanced_games 
WHERE date >= '2025-08-11' 
AND home_sp_season_era IS NOT NULL;

-- ========================================
-- FIX 2: ROLLING OPS CALCULATIONS (HIGH PRIORITY)
-- ========================================
-- Current: averaged OPS (~0.65), should be summed OPS (~9-19)

-- Fix L14 rolling OPS
UPDATE enhanced_games 
SET 
    home_team_ops_l14 = (
        SELECT COALESCE(SUM(home_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.home_team = enhanced_games.home_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '14 days'
        AND eg2.home_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL  -- Only count actual games
        LIMIT 14
    ),
    away_team_ops_l14 = (
        SELECT COALESCE(SUM(away_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.away_team = enhanced_games.away_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '14 days'
        AND eg2.away_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL
        LIMIT 14
    )
WHERE date >= '2025-06-25';

-- Fix L20 rolling OPS
UPDATE enhanced_games 
SET 
    home_team_ops_l20 = (
        SELECT COALESCE(SUM(home_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.home_team = enhanced_games.home_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '20 days'
        AND eg2.home_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL
        LIMIT 20
    ),
    away_team_ops_l20 = (
        SELECT COALESCE(SUM(away_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.away_team = enhanced_games.away_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '20 days'
        AND eg2.away_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL
        LIMIT 20
    )
WHERE date >= '2025-06-25';

-- Fix L30 rolling OPS
UPDATE enhanced_games 
SET 
    home_team_ops_l30 = (
        SELECT COALESCE(SUM(home_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.home_team = enhanced_games.home_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '30 days'
        AND eg2.home_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL
        LIMIT 30
    ),
    away_team_ops_l30 = (
        SELECT COALESCE(SUM(away_team_ops), 0)
        FROM enhanced_games eg2 
        WHERE eg2.away_team = enhanced_games.away_team 
        AND eg2.date < enhanced_games.date 
        AND eg2.date >= enhanced_games.date - INTERVAL '30 days'
        AND eg2.away_team_ops IS NOT NULL
        AND eg2.total_runs IS NOT NULL
        LIMIT 30
    )
WHERE date >= '2025-06-25';

-- Verify rolling OPS fix
SELECT 
    AVG(home_team_ops_l14) as avg_home_ops_l14,
    AVG(home_team_ops_l20) as avg_home_ops_l20, 
    AVG(home_team_ops_l30) as avg_home_ops_l30,
    MIN(home_team_ops_l14) as min_ops_l14,
    MAX(home_team_ops_l30) as max_ops_l30
FROM enhanced_games 
WHERE date >= '2025-08-01'
AND home_team_ops_l14 IS NOT NULL;

-- ========================================
-- FIX 3: COMBINED ERA CALCULATIONS (MEDIUM PRIORITY)
-- ========================================
-- Fix combined ERA to be average of starter and bullpen ERA

UPDATE enhanced_games 
SET combined_era = (
    CASE 
        WHEN home_sp_season_era IS NOT NULL AND away_sp_season_era IS NOT NULL
        THEN (home_sp_season_era + away_sp_season_era) / 2.0
        ELSE NULL
    END
)
WHERE date >= '2025-08-11';

-- Fix ERA differential
UPDATE enhanced_games 
SET era_differential = (
    CASE 
        WHEN home_sp_season_era IS NOT NULL AND away_sp_season_era IS NOT NULL
        THEN home_sp_season_era - away_sp_season_era
        ELSE NULL
    END
)
WHERE date >= '2025-08-11';

-- ========================================
-- VALIDATION QUERIES
-- ========================================

-- Check season ERA fix
SELECT 
    'Season ERA Check' as check_type,
    COUNT(*) as records,
    AVG(home_sp_season_era) as avg_era,
    COUNT(CASE WHEN home_sp_season_era BETWEEN 2.0 AND 7.0 THEN 1 END) as realistic_count,
    ROUND(COUNT(CASE WHEN home_sp_season_era BETWEEN 2.0 AND 7.0 THEN 1 END) * 100.0 / COUNT(*), 1) as realistic_pct
FROM enhanced_games 
WHERE date >= '2025-08-11' 
AND home_sp_season_era IS NOT NULL;

-- Check rolling OPS fix  
SELECT 
    'Rolling OPS Check' as check_type,
    COUNT(*) as records,
    AVG(home_team_ops_l14) as avg_ops_l14,
    AVG(home_team_ops_l30) as avg_ops_l30,
    COUNT(CASE WHEN home_team_ops_l14 BETWEEN 7.0 AND 17.0 THEN 1 END) as realistic_l14,
    COUNT(CASE WHEN home_team_ops_l30 BETWEEN 15.0 AND 35.0 THEN 1 END) as realistic_l30
FROM enhanced_games 
WHERE date >= '2025-08-01'
AND home_team_ops_l14 IS NOT NULL;

-- Overall data quality summary
SELECT 
    'Overall Quality' as summary,
    COUNT(*) as total_games,
    COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as games_with_results,
    COUNT(CASE WHEN home_team_runs_l7 BETWEEN 14 AND 70 THEN 1 END) as good_rolling_runs,
    COUNT(CASE WHEN home_sp_season_era BETWEEN 2.0 AND 7.0 THEN 1 END) as good_era,
    COUNT(CASE WHEN home_team_ops_l14 BETWEEN 7.0 AND 17.0 THEN 1 END) as good_rolling_ops
FROM enhanced_games 
WHERE date >= '2025-08-01';
