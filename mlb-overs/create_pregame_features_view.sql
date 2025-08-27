DROP MATERIALIZED VIEW IF EXISTS pregame_features_v1;

CREATE MATERIALIZED VIEW pregame_features_v1 AS
WITH
-- All SP starts (home & away) with side flag
starts AS (
  SELECT date, game_id, home_sp_id AS sp_id,
         home_sp_er::numeric AS er, home_sp_ip::numeric AS ip,
         'home'::text AS side
  FROM enhanced_games
  WHERE home_sp_id IS NOT NULL AND home_sp_ip IS NOT NULL
  UNION ALL
  SELECT date, game_id, away_sp_id,
         away_sp_er::numeric, away_sp_ip::numeric,
         'away'
  FROM enhanced_games
  WHERE away_sp_id IS NOT NULL AND away_sp_ip IS NOT NULL
),

-- Bullpen (home & away) normalized to team/date
bp AS (
  SELECT date, home_team AS team, home_bp_er::numeric AS er, home_bp_ip::numeric AS ip
  FROM enhanced_games
  WHERE home_bp_ip IS NOT NULL
  UNION ALL
  SELECT date, away_team, away_bp_er::numeric, away_bp_ip::numeric
  FROM enhanced_games
  WHERE away_bp_ip IS NOT NULL
),

-- Team scoring by game (home & away) for rolling means
team_games AS (
  SELECT date, home_team AS team, home_team_runs_pg::numeric AS runs
  FROM enhanced_games WHERE home_team_runs_pg IS NOT NULL
  UNION ALL
  SELECT date, away_team, away_team_runs_pg::numeric
  FROM enhanced_games WHERE away_team_runs_pg IS NOT NULL
)

SELECT
  g.game_id,
  g.date,

  /* ===== SP ERA L3 (as-of) ===== */
  (SELECT 9.0 * SUM(s.er) / NULLIF(SUM(s.ip), 0)
     FROM (
       SELECT er, ip
       FROM starts s
       WHERE s.sp_id = g.home_sp_id AND s.date < g.date
       ORDER BY s.date DESC
       LIMIT 3
     ) s
  ) AS home_sp_era_l3_asof,

  (SELECT 9.0 * SUM(s.er) / NULLIF(SUM(s.ip), 0)
     FROM (
       SELECT er, ip
       FROM starts s
       WHERE s.sp_id = g.away_sp_id AND s.date < g.date
       ORDER BY s.date DESC
       LIMIT 3
     ) s
  ) AS away_sp_era_l3_asof,

  /* ===== SP WHIP L3 (as-of) =====
     Pick H/BB from the correct columns depending on whether the pitcher's start
     was as the home or away starter in that historical game. */
  (SELECT (SUM(t.h) + SUM(t.bb)) / NULLIF(SUM(t.ip), 0)
     FROM (
       SELECT
         CASE WHEN s.side = 'home' THEN e.home_sp_h::numeric ELSE e.away_sp_h::numeric END AS h,
         CASE WHEN s.side = 'home' THEN e.home_sp_bb::numeric ELSE e.away_sp_bb::numeric END AS bb,
         s.ip
       FROM starts s
       JOIN enhanced_games e ON e.game_id = s.game_id
       WHERE s.sp_id = g.home_sp_id AND s.date < g.date
       ORDER BY s.date DESC
       LIMIT 3
     ) t
  ) AS home_sp_whip_l3_asof,

  (SELECT (SUM(t.h) + SUM(t.bb)) / NULLIF(SUM(t.ip), 0)
     FROM (
       SELECT
         CASE WHEN s.side = 'home' THEN e.home_sp_h::numeric ELSE e.away_sp_h::numeric END AS h,
         CASE WHEN s.side = 'home' THEN e.home_sp_bb::numeric ELSE e.away_sp_bb::numeric END AS bb,
         s.ip
       FROM starts s
       JOIN enhanced_games e ON e.game_id = s.game_id
       WHERE s.sp_id = g.away_sp_id AND s.date < g.date
       ORDER BY s.date DESC
       LIMIT 3
     ) t
  ) AS away_sp_whip_l3_asof,
  /* ===== Bullpen load & ERA (as-of) ===== */
  (SELECT COALESCE(SUM(ip), 0)
     FROM bp b
     WHERE b.team = g.home_team AND b.date < g.date
       AND b.date >= g.date - INTERVAL '3 days'
  ) AS home_bp_ip_3d_asof,

  (SELECT 9.0 * SUM(er) / NULLIF(SUM(ip), 0)
     FROM bp b
     WHERE b.team = g.home_team AND b.date < g.date
       AND b.date >= g.date - INTERVAL '30 days'
  ) AS home_bp_era_30d_asof,

  (SELECT COALESCE(SUM(ip), 0)
     FROM bp b
     WHERE b.team = g.away_team AND b.date < g.date
       AND b.date >= g.date - INTERVAL '3 days'
  ) AS away_bp_ip_3d_asof,

  (SELECT 9.0 * SUM(er) / NULLIF(SUM(ip), 0)
     FROM bp b
     WHERE b.team = g.away_team AND b.date < g.date
       AND b.date >= g.date - INTERVAL '30 days'
  ) AS away_bp_era_30d_asof,

  /* ===== Team offense L14 (as-of) ===== */
  (SELECT AVG(runs) FROM (
     SELECT runs
     FROM team_games t
     WHERE t.team = g.home_team AND t.date < g.date
     ORDER BY t.date DESC
     LIMIT 14
  ) x) AS home_runs_pg_14_asof,

  (SELECT AVG(runs) FROM (
     SELECT runs
     FROM team_games t
     WHERE t.team = g.away_team AND t.date < g.date
     ORDER BY t.date DESC
     LIMIT 14
  ) x) AS away_runs_pg_14_asof

FROM enhanced_games g
WHERE g.date >= DATE '2024-01-01';

-- helpful indexes for the view
CREATE INDEX IF NOT EXISTS pregame_features_v1_game_id_idx ON pregame_features_v1 (game_id);
CREATE INDEX IF NOT EXISTS pregame_features_v1_date_idx    ON pregame_features_v1 (date);
