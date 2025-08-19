#!/usr/bin/env python3
"""
CLV Outcomes Logger

Logs realized bet outcomes and CLV for completed games.
Matches picks to closing odds and actual results using the exact priced line and book.
"""

import os, argparse
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def log_outcomes(target_date, run_id=None):
    """Log outcomes for a specific date and optional run_id"""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    with engine.begin() as conn:
        # If no run_id specified, get the latest run for the date
        if not run_id:
            latest_run = conn.execute(text("""
                SELECT run_id FROM probability_predictions 
                WHERE game_date = :d 
                ORDER BY created_at DESC LIMIT 1
            """), {"d": target_date}).fetchone()
            
            if not latest_run:
                print(f"❌ No predictions found for {target_date}")
                return
            
            run_id = str(latest_run[0])
            print(f"🎯 Using latest run: {run_id[:8]}...")
        
        # Execute the enhanced CLV logging query that matches priced lines
        result = conn.execute(text("""
            WITH picks AS (
              SELECT pp.run_id, pp.game_id, pp.game_date,
                     CASE WHEN pp.ev_over >= pp.ev_under THEN 'OVER' ELSE 'UNDER' END AS side,
                     CASE WHEN pp.ev_over >= pp.ev_under THEN pp.p_over ELSE pp.p_under END AS rec_prob,
                     CASE WHEN pp.ev_over >= pp.ev_under THEN pp.over_odds ELSE pp.under_odds END AS placed_odds,
                     pp.fair_odds, 
                     COALESCE(pp.priced_total, pp.market_total) AS priced_total, 
                     COALESCE(pp.priced_book, 'market') AS priced_book, 
                     COALESCE(pp.stake, 0.0) AS stake
              FROM probability_predictions pp
              WHERE pp.run_id = :rid AND pp.game_date = :d
            ),
            closing AS (
              SELECT o.game_id, o."date", o.book, o.total, o.over_odds, o.under_odds,
                     ROW_NUMBER() OVER (PARTITION BY game_id, "date", book, total ORDER BY collected_at DESC) rn
              FROM totals_odds_close o
              WHERE o."date" = :d
            ),
            finals AS (
              SELECT eg.game_id, eg."date", 
                     COALESCE(eg.total_runs, eg.home_score + eg.away_score) AS total_runs
              FROM enhanced_games eg
              WHERE eg."date" = :d
            )
            INSERT INTO bet_outcomes(run_id, game_id, game_date, side, rec_prob, odds, fair_odds, clv, won, stake, pnl)
            SELECT p.run_id, p.game_id, p.game_date, p.side, p.rec_prob,
                   p.placed_odds AS odds,
                   p.fair_odds,
                   /* CLV vs closing on same exact line+book; fallback 0 if missing */
                   COALESCE(
                     CASE WHEN p.side='OVER' THEN c.over_odds - p.placed_odds
                          ELSE c.under_odds - p.placed_odds END, 0
                   ) AS clv,
                   /* Grade vs exact priced line (handles pushes) */
                   CASE WHEN f.total_runs = p.priced_total THEN NULL  -- push
                        WHEN (p.side='OVER'  AND f.total_runs >  p.priced_total) OR
                             (p.side='UNDER' AND f.total_runs <  p.priced_total)
                        THEN TRUE
                        ELSE FALSE END AS won,
                   p.stake AS stake,
                   /* Calculate PnL: (win_amount * won) - (stake * (1-won)) */
                   CASE WHEN f.total_runs = p.priced_total THEN 0.0  -- push = 0 PnL
                        WHEN (p.side='OVER'  AND f.total_runs >  p.priced_total) OR
                             (p.side='UNDER' AND f.total_runs <  p.priced_total)
                        THEN p.stake * (CASE WHEN p.placed_odds < 0 THEN 100.0/ABS(p.placed_odds) ELSE p.placed_odds/100.0 END)
                        ELSE -p.stake END AS pnl
            FROM picks p
            LEFT JOIN closing c
              ON c.game_id=p.game_id AND c."date"=p.game_date 
             AND c.book=p.priced_book AND c.total=p.priced_total AND c.rn=1
            LEFT JOIN finals f ON f.game_id=p.game_id AND f."date"=p.game_date
            ON CONFLICT (run_id, game_id) DO NOTHING
        """), {"rid": run_id, "d": target_date})
        
        rows_inserted = result.rowcount
        print(f"✅ Logged {rows_inserted} bet outcomes for {target_date}")
        
        # Show summary
        summary = conn.execute(text("""
            SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN won = TRUE THEN 1 END) as wins,
                COUNT(CASE WHEN won = FALSE THEN 1 END) as losses,
                COUNT(CASE WHEN won IS NULL THEN 1 END) as pushes,
                SUM(stake) as total_stake,
                SUM(pnl) as total_pnl,
                AVG(clv) as avg_clv
            FROM bet_outcomes 
            WHERE run_id = :rid AND game_date = :d
        """), {"rid": run_id, "d": target_date}).fetchone()
        
        if summary and summary[0] > 0:
            total, wins, losses, pushes, stake, pnl, avg_clv = summary
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            roi = (pnl / stake) if stake and stake > 0 else 0
            
            print(f"\n📊 Outcomes Summary for {target_date}:")
            print(f"   Total bets: {total}")
            print(f"   Record: {wins}-{losses}-{pushes} ({win_rate:.1%})")
            print(f"   Total stake: ${stake:.0f}")
            print(f"   Total P&L: ${pnl:+.0f}")
            print(f"   ROI: {roi:+.1%}")
            print(f"   Average CLV: {avg_clv:+.0f} cents" if avg_clv else "   Average CLV: N/A")

def main():
    ap = argparse.ArgumentParser(description="Log CLV outcomes for completed games")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--run-id", help="Specific run ID (optional, defaults to latest)")
    args = ap.parse_args()
    
    target_date = args.date
    print(f"🎯 Logging outcomes for {target_date}")
    
    log_outcomes(target_date, args.run_id)

if __name__ == "__main__":
    main()
