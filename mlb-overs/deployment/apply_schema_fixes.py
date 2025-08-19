#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def apply_schema_fixes():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            print("🔧 Ensuring probability_predictions exists...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS probability_predictions (
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
                  created_at      timestamp DEFAULT now()
                )
            """))

            print("🔧 Add adj_edge if missing...")
            conn.execute(text("""
                ALTER TABLE probability_predictions
                ADD COLUMN IF NOT EXISTS adj_edge numeric
            """))

            print("🔧 Set composite primary key (game_id, game_date)...")
            conn.execute(text("DROP INDEX IF EXISTS ux_prob_preds_gid_date"))
            conn.execute(text("ALTER TABLE probability_predictions DROP CONSTRAINT IF EXISTS probability_predictions_pkey"))
            conn.execute(text("ALTER TABLE probability_predictions ADD PRIMARY KEY (game_id, game_date)"))

            print("🔧 Dropping api_games_today view (if present)...")
            conn.execute(text("DROP VIEW IF EXISTS api_games_today"))

            print("🔧 Creating api_games_today view with confidence...")
            conn.execute(text("""
                CREATE VIEW api_games_today (
                  game_id, game_date, home_team, away_team,
                  predicted_total, market_total, confidence,
                  recommendation, edge,
                  over_probability, under_probability,
                  expected_value_over, expected_value_under,
                  kelly_fraction_over, kelly_fraction_under,
                  temperature, wind_speed, weather_condition
                ) AS
                SELECT 
                  eg.game_id,
                  eg."date"::date AS game_date,
                  eg.home_team,
                  eg.away_team,
                  eg.predicted_total,
                  eg.market_total,
                  /* Confidence from probabilities; NULL if probs missing */
                  CASE
                    WHEN pp.p_over IS NULL OR pp.p_under IS NULL THEN NULL
                    ELSE (ROUND(100 * GREATEST(pp.p_over, pp.p_under)))::int
                  END AS confidence,
                  CASE
                    WHEN pp.ev_over > pp.ev_under THEN 'OVER'
                    WHEN pp.ev_under > pp.ev_over THEN 'UNDER'
                    ELSE 'NO BET'
                  END AS recommendation,
                  ROUND((eg.predicted_total - eg.market_total)::numeric, 2) AS edge,
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
                ORDER BY eg.game_id
            """))

            print("✅ Schema fixes applied successfully!")
        return True
    except Exception as e:
        print(f"❌ Error applying schema fixes: {e}")
        return False
    finally:
        engine.dispose()

if __name__ == "__main__":
    apply_schema_fixes()
