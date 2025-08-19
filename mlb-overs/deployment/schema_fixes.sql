-- Schema fixes for probability_predictions table
-- 1) Drop existing table and recreate with proper composite primary key

DROP TABLE IF EXISTS probability_predictions;

CREATE TABLE probability_predictions (
  game_id         varchar NOT NULL,
  game_date       date NOT NULL,
  market_total    numeric,
  predicted_total numeric,
  adj_edge        numeric,
  p_over          double precision,
  p_under         double precision,
  over_odds       integer,
  under_odds      integer,
  ev_over         double precision,
  ev_under        double precision,
  kelly_over      double precision,
  kelly_under     double precision,
  model_version   text,
  created_at      timestamp DEFAULT now(),
  PRIMARY KEY (game_id, game_date)
);

-- 2) Recreate api_games_today view with computed confidence from probabilities
DROP VIEW IF EXISTS api_games_today;

CREATE VIEW api_games_today AS
SELECT 
  eg.game_id,
  eg."date" AS game_date,
  eg.home_team,
  eg.away_team,
  eg.predicted_total,
  eg.market_total,
  ROUND(100 * GREATEST(pp.p_over, pp.p_under)) AS confidence,
  CASE
    WHEN pp.ev_over > pp.ev_under THEN 'OVER'
    WHEN pp.ev_under > pp.ev_over THEN 'UNDER'
    ELSE 'NO BET'
  END AS recommendation,
  ROUND((eg.predicted_total - eg.market_total), 2) AS edge,
  pp.p_over  AS over_probability,
  pp.p_under AS under_probability,
  pp.ev_over AS expected_value_over,
  pp.ev_under AS expected_value_under,
  pp.kelly_over AS kelly_fraction_over,
  pp.kelly_under AS kelly_fraction_under,
  COALESCE(eg.temperature, 70) AS temperature,
  COALESCE(eg.wind_speed, 5) AS wind_speed,
  eg.weather_condition
FROM enhanced_games eg
LEFT JOIN probability_predictions pp
  ON pp.game_id  = eg.game_id
 AND pp.game_date = eg."date"
WHERE eg."date" = CURRENT_DATE
ORDER BY eg.game_id;
