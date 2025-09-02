-- mlb/hitting/sql/schema.sql
CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id BIGINT NOT NULL,
    game_date DATE NOT NULL,
    game_id   VARCHAR(64),
    team      TEXT,
    opp_team  TEXT,
    lineup_slot INT,
    pitcher_id  BIGINT,
    pitcher_hand CHAR(1),
    hits INT,
    at_bats INT,
    plate_appearances INT,
    PRIMARY KEY (player_id, game_date, game_id)
);

CREATE TABLE IF NOT EXISTS batter_vs_pitcher_logs (
    player_id  BIGINT NOT NULL,
    pitcher_id BIGINT NOT NULL,
    game_date  DATE NOT NULL,
    game_id    VARCHAR(64),
    hits INT,
    at_bats INT,
    PRIMARY KEY (player_id, pitcher_id, game_id)
);

CREATE TABLE IF NOT EXISTS player_props_odds (
    date        DATE NOT NULL,
    game_id     VARCHAR(64) NOT NULL,
    player_id   BIGINT NOT NULL,
    player_name TEXT,
    team        TEXT,
    market      TEXT NOT NULL,
    line        TEXT NOT NULL,
    over_odds   INT,
    under_odds  INT,
    book        TEXT,
    updated_at  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, game_id, player_id, market, line, COALESCE(book,''))
);

CREATE TABLE IF NOT EXISTS hitter_prop_predictions (
    date        DATE NOT NULL,
    game_id     VARCHAR(64) NOT NULL,
    player_id   BIGINT NOT NULL,
    player_name TEXT,
    team        TEXT,
    opp_team    TEXT,
    lineup_slot INT,
    pitcher_id  BIGINT,
    pitcher_hand CHAR(1),
    ab_expected NUMERIC,
    pa_expected NUMERIC,
    per_pa_hit_rate NUMERIC,
    p_over_0_5  NUMERIC,
    p_over_1_5  NUMERIC,
    over_0_5_odds INT,
    under_0_5_odds INT,
    over_1_5_odds INT,
    under_1_5_odds INT,
    ev_over_0_5 NUMERIC,
    ev_over_1_5 NUMERIC,
    kelly_over_0_5 NUMERIC,
    kelly_over_1_5 NUMERIC,
    created_at  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (date, game_id, player_id)
);

DROP MATERIALIZED VIEW IF EXISTS mv_hitter_form;
CREATE MATERIALIZED VIEW mv_hitter_form AS
WITH base AS (
  SELECT player_id, game_date, team, opp_team, pitcher_id, pitcher_hand, lineup_slot,
         COALESCE(hits,0) AS hits,
         GREATEST(COALESCE(at_bats,0),0) AS ab,
         GREATEST(COALESCE(plate_appearances,0),0) AS pa
  FROM player_game_logs
),
roll AS (
  SELECT
    player_id, game_date, team, opp_team, pitcher_id, pitcher_hand, lineup_slot,
    SUM(hits) OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS hits_l5,
    SUM(ab)   OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS ab_l5,
    SUM(hits) OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)  AS hits_l10,
    SUM(ab)   OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)  AS ab_l10,
    SUM(hits) OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS hits_l15,
    SUM(ab)   OVER (PARTITION BY player_id ORDER BY game_date ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS ab_l15,
    SUM(hits) OVER (PARTITION BY player_id) AS hits_season,
    SUM(ab)   OVER (PARTITION BY player_id) AS ab_season
  FROM base
)
SELECT * FROM roll
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_mv_hitter_form_player_date ON mv_hitter_form(player_id, game_date);

DROP MATERIALIZED VIEW IF EXISTS mv_bvp_agg;
CREATE MATERIALIZED VIEW mv_bvp_agg AS
SELECT player_id, pitcher_id, MAX(game_date) AS asof_date,
       SUM(hits) AS hits_bvp, SUM(at_bats) AS ab_bvp
FROM batter_vs_pitcher_logs
GROUP BY player_id, pitcher_id
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_mv_bvp_agg_player_pitcher ON mv_bvp_agg(player_id, pitcher_id);

DROP MATERIALIZED VIEW IF EXISTS mv_pa_distribution;
CREATE MATERIALIZED VIEW mv_pa_distribution AS
SELECT team, lineup_slot, pitcher_hand,
       COUNT(*) AS games,
       AVG(plate_appearances) AS pa_avg,
       AVG(at_bats)           AS ab_avg
FROM player_game_logs
WHERE lineup_slot BETWEEN 1 AND 9
GROUP BY team, lineup_slot, pitcher_hand
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_mv_pa_dist_team_slot ON mv_pa_distribution(team, lineup_slot, pitcher_hand);
