#!/usr/bin/env python3
"""
Create the api_outcomes_all view that combines predictions with actual results.
This view is the foundation for outcomes tracking and performance analysis.
"""

import os
from sqlalchemy import create_engine, text

def create_outcomes_view():
    """Create or replace the api_outcomes_all database view"""
    
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # Create the comprehensive outcomes view
    view_sql = """
    CREATE OR REPLACE VIEW api_outcomes_all AS
    WITH latest_preds AS (
      SELECT DISTINCT ON (pp.game_id, pp.game_date)
             pp.game_id,
             pp.game_date,
             pp.created_at,
             pp.run_id,
             pp.model_version,
             pp.p_over, pp.p_under,
             pp.ev_over, pp.ev_under,
             pp.kelly_over, pp.kelly_under,
             pp.over_odds, pp.under_odds,
             pp.predicted_total,
             pp.market_total,
             CASE WHEN pp.ev_over >= pp.ev_under THEN 'OVER' ELSE 'UNDER' END AS rec_side,
             CASE WHEN pp.ev_over >= pp.ev_under THEN pp.p_over  ELSE pp.p_under  END AS rec_prob,
             CASE WHEN pp.ev_over >= pp.ev_under THEN pp.kelly_over ELSE pp.kelly_under END AS kelly_best,
             CASE WHEN pp.ev_over >= pp.ev_under THEN pp.over_odds ELSE pp.under_odds END AS side_odds
      FROM probability_predictions pp
      ORDER BY pp.game_id, pp.game_date, pp.created_at DESC
    ),
    finals AS (
      SELECT eg.game_id,
             eg.date AS game_date,
             eg.home_team, eg.away_team,
             COALESCE(eg.market_total, lp.market_total) as market_total,
             COALESCE(eg.predicted_total, lp.predicted_total) as predicted_total,
             COALESCE(lgf.total_runs,
                      CASE WHEN eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL
                           THEN eg.home_score + eg.away_score END) AS total_runs,
             eg.venue_name,
             eg.weather_condition,
             eg.temperature
      FROM enhanced_games eg
      LEFT JOIN latest_preds lp ON lp.game_id = eg.game_id AND lp.game_date = eg.date
      LEFT JOIN legitimate_game_features lgf
        ON lgf.game_id = eg.game_id AND lgf.date = eg.date
    )
    SELECT
      f.game_id, f.game_date, f.home_team, f.away_team,
      f.market_total, f.predicted_total, f.total_runs,
      f.venue_name, f.weather_condition, f.temperature,
      lp.run_id, lp.created_at, lp.model_version,
      lp.p_over, lp.p_under, lp.ev_over, lp.ev_under,
      lp.kelly_over, lp.kelly_under,
      lp.over_odds, lp.under_odds,
      lp.rec_side, lp.rec_prob, lp.kelly_best, lp.side_odds,
      CASE
        WHEN f.total_runs IS NULL THEN NULL
        WHEN f.total_runs = f.market_total THEN 'PUSH'
        WHEN f.total_runs > f.market_total THEN 'OVER'
        ELSE 'UNDER'
      END AS actual_side,
      CASE
        WHEN f.total_runs IS NULL OR f.total_runs = f.market_total THEN NULL
        WHEN (f.total_runs > f.market_total AND lp.rec_side='OVER')
          OR (f.total_runs < f.market_total AND lp.rec_side='UNDER') THEN TRUE
        ELSE FALSE
      END AS won,
      ROUND((f.predicted_total - f.market_total)::numeric, 2) AS edge_pred,
      ROUND((f.total_runs - f.market_total)::numeric, 2) AS edge_actual,
      ABS(f.predicted_total - f.total_runs) AS abs_error_model,
      ABS(f.market_total - f.total_runs) AS abs_error_market
    FROM finals f
    LEFT JOIN latest_preds lp
      ON lp.game_id = f.game_id AND lp.game_date = f.game_date;
    """

    with engine.begin() as conn:
        conn.execute(text(view_sql))
        print('✅ Created api_outcomes_all view')
        
        # Test the view
        test_result = conn.execute(text('SELECT COUNT(*) as count FROM api_outcomes_all')).fetchone()
        print(f'✅ View contains {test_result.count} records')
        
        # Show sample for recent completed games
        sample = conn.execute(text("""
            SELECT game_id, home_team, away_team, rec_side, won, total_runs, market_total, edge_pred
            FROM api_outcomes_all 
            WHERE game_date >= CURRENT_DATE - INTERVAL '7 days'
            AND total_runs IS NOT NULL
            AND rec_side IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 5
        """)).fetchall()
        
        if sample:
            print('\n📊 Sample recent outcomes:')
            for row in sample:
                result = "WIN" if row.won else "LOSS" if row.won is False else "PUSH"
                edge = f"Edge: {row.edge_pred:+.1f}" if row.edge_pred else "No edge"
                print(f'  {row.away_team} @ {row.home_team}: {row.rec_side} - {result} (actual: {row.total_runs}, market: {row.market_total}) | {edge}')
        else:
            print('\n📝 No recent completed games with predictions found')

if __name__ == "__main__":
    create_outcomes_view()
