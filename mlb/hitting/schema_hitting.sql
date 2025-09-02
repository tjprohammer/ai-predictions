-- MLB Hitting Stats Database Schema
-- This file contains all the tables and views needed for player hitting props

-- ============================================================================
-- 1. Player Game Logs (Comprehensive Batting Stats)
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_game_logs (
  game_id         VARCHAR(50) NOT NULL,
  date            DATE        NOT NULL,
  player_id       BIGINT      NOT NULL,
  player_name     TEXT,
  team            TEXT,
  team_id         INTEGER,
  opponent        TEXT,
  home_away       CHAR(1),              -- 'H'/'A'
  lineup_spot     INT,                  -- 1..9 if available
  -- Core hitting stats
  plate_appearances INT DEFAULT 0,
  at_bats         INT DEFAULT 0,
  hits            INT DEFAULT 0,
  singles         INT DEFAULT 0,         -- Calculated as hits - doubles - triples - home_runs
  doubles         INT DEFAULT 0,
  triples         INT DEFAULT 0,
  home_runs       INT DEFAULT 0,
  runs            INT DEFAULT 0,
  runs_batted_in  INT DEFAULT 0,
  -- Plate discipline
  walks           INT DEFAULT 0,
  intentional_walks INT DEFAULT 0,
  strikeouts      INT DEFAULT 0,
  hit_by_pitch    INT DEFAULT 0,
  sacrifice_hits  INT DEFAULT 0,        -- Sacrifice bunts
  sacrifice_flies INT DEFAULT 0,
  -- Base running
  stolen_bases    INT DEFAULT 0,
  caught_stealing INT DEFAULT 0,
  -- Advanced metrics (calculated)
  total_bases     INT DEFAULT 0,        -- Singles + 2*Doubles + 3*Triples + 4*HR
  extra_base_hits INT DEFAULT 0,        -- Doubles + Triples + HR
  -- Rate stats (calculated)
  batting_avg     NUMERIC(4,3) DEFAULT 0.000,  -- hits/at_bats
  on_base_pct     NUMERIC(4,3) DEFAULT 0.000,  -- (H+BB+HBP)/(AB+BB+HBP+SF)
  slugging_pct    NUMERIC(4,3) DEFAULT 0.000,  -- total_bases/at_bats
  ops             NUMERIC(4,3) DEFAULT 0.000,  -- on_base_pct + slugging_pct
  -- Contact quality (if available from advanced data)
  ground_balls    INT DEFAULT 0,
  fly_balls       INT DEFAULT 0,
  line_drives     INT DEFAULT 0,
  pop_ups         INT DEFAULT 0,
  -- Game situation
  left_on_base    INT DEFAULT 0,
  double_plays    INT DEFAULT 0,
  -- Pitcher info
  starting_pitcher_id BIGINT,           -- opposing SP
  starting_pitcher_name TEXT,
  starting_pitcher_hand CHAR(1),        -- 'R'/'L'
  -- Meta
  created_at      TIMESTAMP DEFAULT NOW(),
  updated_at      TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (player_id, game_id, date)
);

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
-- Pitcher Reference (Handedness)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pitchers (
    pitcher_id INTEGER PRIMARY KEY,
    pitcher_name TEXT,
    throws_hand CHAR(1) -- 'L' or 'R'
);

CREATE INDEX IF NOT EXISTS idx_pitchers_hand ON pitchers(throws_hand);

-- ============================================================================
-- Expected PA/AB Distribution by Lineup Spot
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_pa_distribution AS
SELECT
  lineup_spot,
  home_away,
  COUNT(*)                         AS g,
  AVG(plate_appearances)          AS exp_pa,
  AVG(at_bats)                    AS exp_ab
FROM player_game_logs
WHERE lineup_spot BETWEEN 1 AND 9
  AND plate_appearances > 0
GROUP BY lineup_spot, home_away;

CREATE INDEX IF NOT EXISTS idx_pa_dist_lineup ON mv_pa_distribution(lineup_spot, home_away);

-- ============================================================================
-- 3. Rolling Form Windows (5/10/15 games) - Expanded Stats
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

CREATE INDEX IF NOT EXISTS idx_form_player_date ON mv_hitter_form(player_id, date);
CREATE INDEX IF NOT EXISTS idx_form_date ON mv_hitter_form(date);

-- ============================================================================
-- 4. Player Props Odds Storage (Multi-Market)
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_props_odds (
  date        DATE NOT NULL,
  game_id     VARCHAR(50) NOT NULL,
  player_id   BIGINT NOT NULL,
  player_name TEXT,
  market      TEXT NOT NULL,       -- e.g., 'HITS_0.5', 'HITS_1.5', 'HR_0.5', 'RBI_0.5', 'TB_1.5'
  line        NUMERIC NOT NULL,    -- 0.5, 1.5, 2.5...
  over_odds   INT,                 -- American odds
  under_odds  INT,
  book        TEXT DEFAULT 'FanDuel',
  updated_at  TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (date, game_id, player_id, market, book)
);

CREATE INDEX IF NOT EXISTS idx_props_date_market ON player_props_odds(date, market);
CREATE INDEX IF NOT EXISTS idx_props_player_date ON player_props_odds(player_id, date);

-- ============================================================================
-- 5. Hitter Prop Predictions (Multi-Market)
-- ============================================================================
CREATE TABLE IF NOT EXISTS hitter_prop_predictions (
  date          DATE NOT NULL,
  game_id       VARCHAR(50) NOT NULL,
  player_id     BIGINT NOT NULL,
  player_name   TEXT,
  market        TEXT NOT NULL,           -- 'HITS_0.5', 'HITS_1.5', 'HR_0.5', 'RBI_0.5', 'TB_1.5', etc.
  line          NUMERIC NOT NULL,        -- 0.5, 1.5, etc.
  p_over        NUMERIC,                 -- predicted prob Over (≥ line)
  p_under       NUMERIC,                 -- predicted prob Under (< line)
  ev_over       NUMERIC,                 -- Expected value Over
  ev_under      NUMERIC,                 -- Expected value Under
  kelly_over    NUMERIC,                 -- Kelly criterion Over
  kelly_under   NUMERIC,                 -- Kelly criterion Under
  confidence    NUMERIC,                 -- Model confidence 0-100
  recommendation TEXT,                   -- 'OVER', 'UNDER', 'HOLD'
  edge          NUMERIC,                 -- Best edge available
  features      JSONB,                   -- Feature bundle for audit
  model_version TEXT DEFAULT 'v2.0',
  created_at    TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (date, game_id, player_id, market)
);

CREATE INDEX IF NOT EXISTS idx_hitter_preds_date ON hitter_prop_predictions(date);
CREATE INDEX IF NOT EXISTS idx_hitter_preds_ev ON hitter_prop_predictions(date, GREATEST(COALESCE(ev_over,0), COALESCE(ev_under,0)) DESC);

-- ============================================================================
-- 6. API View for Recommended Players
-- ============================================================================
CREATE OR REPLACE VIEW api_hit_props_today AS
SELECT
  p.date, 
  p.game_id, 
  p.player_id,
  p.player_name,
  p.market,
  p.line,
  p.p_over, 
  p.p_under,
  o.over_odds, 
  o.under_odds, 
  o.book,
  p.ev_over, 
  p.ev_under,
  p.kelly_over, 
  p.kelly_under,
  p.confidence,
  p.recommendation,
  p.edge,
  CASE 
    WHEN p.ev_over > p.ev_under THEN 'OVER'
    WHEN p.ev_under > p.ev_over THEN 'UNDER'
    ELSE 'HOLD'
  END as best_bet,
  GREATEST(COALESCE(p.ev_over,0), COALESCE(p.ev_under,0)) as best_ev,
  CASE 
    WHEN p.ev_over > p.ev_under THEN p.kelly_over
    WHEN p.ev_under > p.ev_over THEN p.kelly_under
    ELSE 0
  END as best_kelly
FROM hitter_prop_predictions p
LEFT JOIN player_props_odds o
  ON o.date=p.date AND o.game_id=p.game_id AND o.player_id=p.player_id AND o.market=p.market
WHERE p.date = CURRENT_DATE
  AND (p.ev_over > 0.05 OR p.ev_under > 0.05)  -- Only show positive EV
ORDER BY best_ev DESC NULLS LAST;

-- ============================================================================
-- 7. Player Performance Streaks View
-- ============================================================================
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
-- Refresh Functions (to be called daily)
-- ============================================================================
CREATE OR REPLACE FUNCTION refresh_hitting_views() RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY mv_bvp_agg;
  REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hitter_form;
  REFRESH MATERIALIZED VIEW CONCURRENTLY mv_pa_distribution;
END;
$$ LANGUAGE plpgsql;
