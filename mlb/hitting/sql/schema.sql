-- ============================================================================
-- MLB Hitting Props Database Schema
-- Complete schema for comprehensive hitting analysis and props betting
-- ============================================================================

-- ============================================================================
-- 1. Core Player Game Logs Table (Enhanced)
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_game_logs (
    -- Identifiers
    game_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT,
    team TEXT,
    opponent TEXT,
    home_away CHAR(1), -- 'H' or 'A'
    lineup_spot INTEGER,
    
    -- Game context
    starting_pitcher_id BIGINT,
    starting_pitcher_name TEXT,
    starting_pitcher_hand CHAR(1), -- 'L' or 'R'
    
    -- Core hitting stats
    plate_appearances INTEGER DEFAULT 0,
    at_bats INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    singles INTEGER DEFAULT 0,
    doubles INTEGER DEFAULT 0,
    triples INTEGER DEFAULT 0,
    home_runs INTEGER DEFAULT 0,
    runs INTEGER DEFAULT 0,
    runs_batted_in INTEGER DEFAULT 0,
    walks INTEGER DEFAULT 0,
    intentional_walks INTEGER DEFAULT 0,
    strikeouts INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0, -- stolen bases
    cs INTEGER DEFAULT 0, -- caught stealing
    
    -- Advanced metrics
    total_bases INTEGER DEFAULT 0,
    extra_base_hits INTEGER DEFAULT 0,
    
    -- Rate stats (calculated per game)
    batting_avg NUMERIC(4,3) DEFAULT 0.000,
    on_base_pct NUMERIC(4,3) DEFAULT 0.000,
    slugging_pct NUMERIC(4,3) DEFAULT 0.000,
    ops NUMERIC(4,3) DEFAULT 0.000,
    
    -- Contact quality (when available)
    ground_balls INTEGER DEFAULT 0,
    fly_balls INTEGER DEFAULT 0,
    line_drives INTEGER DEFAULT 0,
    pop_ups INTEGER DEFAULT 0,
    
    -- Game situation
    left_on_base INTEGER DEFAULT 0,
    double_plays INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    PRIMARY KEY (player_id, game_id),
    CONSTRAINT valid_home_away CHECK (home_away IN ('H', 'A'))
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_pgl_date ON player_game_logs(date);
CREATE INDEX IF NOT EXISTS idx_pgl_player_date ON player_game_logs(player_id, date);
CREATE INDEX IF NOT EXISTS idx_pgl_sp_hand ON player_game_logs(starting_pitcher_id, starting_pitcher_hand);
CREATE INDEX IF NOT EXISTS idx_pgl_team_date ON player_game_logs(team, date);

-- ============================================================================
-- 2. Batter vs Pitcher Aggregates (Comprehensive) 
-- ============================================================================
DROP MATERIALIZED VIEW IF EXISTS mv_bvp_agg;
CREATE MATERIALIZED VIEW mv_bvp_agg AS
SELECT
  player_id,
  starting_pitcher_id AS pitcher_id,
  MAX(date)           AS through_date,
  COUNT(*)            AS g,
  SUM(plate_appearances) AS pa,
  SUM(at_bats)        AS ab,
  SUM(hits)           AS h,
  SUM(home_runs)      AS hr,
  SUM(runs_batted_in) AS rbi,
  SUM(walks)          AS bb,
  SUM(strikeouts)     AS k,
  SUM(total_bases)    AS tb,
  SUM(extra_base_hits) AS xbh,
  -- Calculated rates
  CASE WHEN SUM(at_bats) > 0 THEN 
    ROUND(SUM(hits)::NUMERIC / SUM(at_bats), 3) 
    ELSE 0 
  END AS avg_vs_pitcher,
  CASE WHEN SUM(at_bats) > 0 THEN 
    ROUND(SUM(total_bases)::NUMERIC / SUM(at_bats), 3) 
    ELSE 0 
  END AS slg_vs_pitcher
FROM player_game_logs
WHERE starting_pitcher_id IS NOT NULL
GROUP BY player_id, starting_pitcher_id;

CREATE INDEX IF NOT EXISTS idx_bvp_player_pitcher ON mv_bvp_agg(player_id, pitcher_id);

-- ============================================================================
-- 3. Pitcher Handedness Reference
-- ============================================================================
CREATE TABLE IF NOT EXISTS pitchers (
    pitcher_id BIGINT PRIMARY KEY,
    pitcher_name TEXT,
    throws CHAR(1) -- 'L' or 'R'
);

CREATE INDEX IF NOT EXISTS idx_pitchers_hand ON pitchers(throws);

-- ============================================================================
-- 4. Expected Plate Appearances Distribution
-- ============================================================================
DROP MATERIALIZED VIEW IF EXISTS mv_pa_distribution;
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_pa_distribution AS
SELECT 
  lineup_spot,
  AVG(plate_appearances) AS avg_pa,
  STDDEV(plate_appearances) AS std_pa,
  COUNT(*) AS games
FROM player_game_logs 
WHERE lineup_spot BETWEEN 1 AND 9
  AND plate_appearances > 0
GROUP BY lineup_spot;

CREATE INDEX IF NOT EXISTS idx_pa_dist_lineup ON mv_pa_distribution(lineup_spot);

-- ============================================================================
-- 5. Rolling Form Windows (5/10/15 games) - Expanded Stats
-- ============================================================================
DROP MATERIALIZED VIEW IF EXISTS mv_hitter_form;
CREATE MATERIALIZED VIEW mv_hitter_form AS
WITH ordered AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date) AS rn
  FROM player_game_logs
  WHERE at_bats > 0  -- Only games with actual plate appearances
),
form_windows AS (
  SELECT 
    o.player_id,
    o.date,
    -- Last 5 games
    SUM(helper.hits) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS hits_l5,
    SUM(helper.at_bats) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS ab_l5,
    SUM(helper.home_runs) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS hr_l5,
    SUM(helper.runs_batted_in) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS rbi_l5,
    SUM(helper.walks) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS bb_l5,
    SUM(helper.strikeouts) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS k_l5,
    SUM(helper.total_bases) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS tb_l5,
    SUM(helper.extra_base_hits) FILTER (WHERE helper.rn BETWEEN o.rn-4 AND o.rn) AS xbh_l5,
    -- Last 10 games  
    SUM(helper.hits) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS hits_l10,
    SUM(helper.at_bats) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS ab_l10,
    SUM(helper.home_runs) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS hr_l10,
    SUM(helper.runs_batted_in) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS rbi_l10,
    SUM(helper.walks) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS bb_l10,
    SUM(helper.strikeouts) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS k_l10,
    SUM(helper.total_bases) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS tb_l10,
    SUM(helper.extra_base_hits) FILTER (WHERE helper.rn BETWEEN o.rn-9 AND o.rn) AS xbh_l10,
    -- Last 15 games
    SUM(helper.hits) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS hits_l15,
    SUM(helper.at_bats) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS ab_l15,
    SUM(helper.home_runs) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS hr_l15,
    SUM(helper.runs_batted_in) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS rbi_l15,
    SUM(helper.walks) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS bb_l15,
    SUM(helper.strikeouts) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS k_l15,
    SUM(helper.total_bases) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS tb_l15,
    SUM(helper.extra_base_hits) FILTER (WHERE helper.rn BETWEEN o.rn-14 AND o.rn) AS xbh_l15
  FROM ordered o
  JOIN ordered helper ON helper.player_id = o.player_id AND helper.rn <= o.rn
  GROUP BY o.player_id, o.date, o.rn
)
SELECT * FROM form_windows;

CREATE INDEX IF NOT EXISTS idx_hitter_form_player_date ON mv_hitter_form(player_id, date);
CREATE INDEX IF NOT EXISTS idx_hitter_form_date ON mv_hitter_form(date);

-- ============================================================================
-- 6. Player Props Odds Storage
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_props_odds (
    date DATE NOT NULL,
    game_id VARCHAR(50) NOT NULL,
    player_id BIGINT NOT NULL,
    market VARCHAR(20) NOT NULL, -- 'HITS_0.5', 'HITS_1.5', 'HR_0.5', 'RBI_0.5', etc.
    over_price NUMERIC(5,2),
    under_price NUMERIC(5,2),
    line NUMERIC(4,1),
    sportsbook VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (date, game_id, player_id, market, sportsbook)
);

CREATE INDEX IF NOT EXISTS idx_props_date_market ON player_props_odds(date, market);
CREATE INDEX IF NOT EXISTS idx_props_player_date ON player_props_odds(player_id, date);

-- ============================================================================
-- 7. Hitter Props Predictions Storage
-- ============================================================================
CREATE TABLE IF NOT EXISTS hitter_prop_predictions (
    date DATE NOT NULL,
    game_id VARCHAR(50) NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT,
    team TEXT,
    opponent TEXT,
    market VARCHAR(20) NOT NULL, -- 'HITS_0.5', 'HITS_1.5', etc.
    line NUMERIC(4,1),
    
    -- Model predictions
    prob_over NUMERIC(5,4),
    prob_under NUMERIC(5,4),
    
    -- Expected value calculations
    over_price NUMERIC(5,2),
    under_price NUMERIC(5,2),
    ev_over NUMERIC(6,4),
    ev_under NUMERIC(6,4),
    
    -- Kelly criterion betting
    kelly_over NUMERIC(6,4),
    kelly_under NUMERIC(6,4),
    
    -- Model metadata
    model_version VARCHAR(20) DEFAULT 'v1',
    confidence_score NUMERIC(4,3),
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (date, game_id, player_id, market)
);

CREATE INDEX IF NOT EXISTS idx_hitter_preds_date ON hitter_prop_predictions(date);
CREATE INDEX IF NOT EXISTS idx_hitter_preds_ev ON hitter_prop_predictions(date) WHERE (ev_over > 0.05 OR ev_under > 0.05);

-- ============================================================================
-- 8. API Views for Easy Access
-- ============================================================================

-- Today's best hitting props picks
CREATE OR REPLACE VIEW api_best_hitprops_today AS
SELECT 
  p.player_name,
  p.team,
  p.opponent,
  p.market,
  p.line,
  p.prob_over,
  p.prob_under,
  p.over_price,
  p.under_price,
  GREATEST(p.ev_over, p.ev_under) as best_ev,
  CASE 
    WHEN p.ev_over > p.ev_under THEN p.kelly_over
    ELSE p.kelly_under
  END as best_kelly,
  CASE 
    WHEN p.ev_over > p.ev_under THEN 'OVER'
    ELSE 'UNDER'
  END as recommendation
FROM hitter_prop_predictions p
LEFT JOIN player_props_odds o
  ON o.date=p.date AND o.game_id=p.game_id AND o.player_id=p.player_id AND o.market=p.market
WHERE p.date = CURRENT_DATE
  AND (p.ev_over > 0.05 OR p.ev_under > 0.05)  -- Only show positive EV
ORDER BY best_ev DESC NULLS LAST;

-- Player performance streaks view
CREATE OR REPLACE VIEW api_player_streaks_today AS
SELECT 
  f.player_id,
  f.date,
  -- Hit percentages for different windows
  CASE WHEN f.ab_l5 > 0 THEN ROUND((f.hits_l5::NUMERIC / f.ab_l5) * 100, 1) ELSE NULL END as hit_pct_l5,
  CASE WHEN f.ab_l10 > 0 THEN ROUND((f.hits_l10::NUMERIC / f.ab_l10) * 100, 1) ELSE NULL END as hit_pct_l10,
  CASE WHEN f.ab_l15 > 0 THEN ROUND((f.hits_l15::NUMERIC / f.ab_l15) * 100, 1) ELSE NULL END as hit_pct_l15,
  -- Raw stats
  f.hits_l5, f.ab_l5,
  f.hits_l10, f.ab_l10,
  f.hits_l15, f.ab_l15,
  -- Hot/Cold indicators
  CASE 
    WHEN f.ab_l5 >= 15 AND (f.hits_l5::NUMERIC / f.ab_l5) > 0.35 THEN 'HOT'
    WHEN f.ab_l5 >= 15 AND (f.hits_l5::NUMERIC / f.ab_l5) < 0.20 THEN 'COLD'
    ELSE 'NEUTRAL'
  END as streak_status_l5,
  CASE 
    WHEN f.ab_l10 >= 25 AND (f.hits_l10::NUMERIC / f.ab_l10) > 0.33 THEN 'HOT'
    WHEN f.ab_l10 >= 25 AND (f.hits_l10::NUMERIC / f.ab_l10) < 0.22 THEN 'COLD'  
    ELSE 'NEUTRAL'
  END as streak_status_l10
FROM mv_hitter_form f
WHERE f.date = CURRENT_DATE
  AND f.ab_l5 > 0  -- Must have at least some at-bats
ORDER BY CASE WHEN f.ab_l10 > 0 THEN (f.hits_l10::NUMERIC / f.ab_l10) ELSE 0 END DESC;

-- ============================================================================
-- 9. Refresh Functions (to be called daily)
-- ============================================================================
CREATE OR REPLACE FUNCTION refresh_hitting_views() RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW mv_bvp_agg;
  REFRESH MATERIALIZED VIEW mv_hitter_form;
  REFRESH MATERIALIZED VIEW mv_pa_distribution;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 10. Data Quality Functions
-- ============================================================================
CREATE OR REPLACE FUNCTION check_hitting_data_quality(target_date DATE DEFAULT CURRENT_DATE) 
RETURNS TABLE(
  metric VARCHAR(50),
  value BIGINT,
  status VARCHAR(20)
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    'player_logs'::VARCHAR(50) as metric,
    COUNT(*)::BIGINT as value,
    CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20) as status
  FROM player_game_logs 
  WHERE date = target_date
  
  UNION ALL
  
  SELECT 
    'mv_hitter_form_rows'::VARCHAR(50),
    COUNT(*)::BIGINT,
    CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'NEEDS_REFRESH' END::VARCHAR(20)
  FROM mv_hitter_form 
  WHERE date = target_date
  
  UNION ALL
  
  SELECT 
    'predictions'::VARCHAR(50),
    COUNT(*)::BIGINT,
    CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20)
  FROM hitter_prop_predictions 
  WHERE date = target_date;
END;
$$ LANGUAGE plpgsql;
