#!/usr/bin/env python3
"""
Log outcomes and PnL for completed games on a given date.
This script reads from the api_outcomes_all view and writes to bet_outcomes table.

Usage:
    python log_outcomes.py --date 2025-08-17 --bankroll 10000
    python log_outcomes.py --date 2025-08-17 --bankroll 10000 --stake-scale 0.5
"""

import os
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def american_win_return(odds: float) -> float:
    """Convert American odds to win multiplier (e.g. -110 -> 0.909, +110 -> 1.1)"""
    return (100/abs(odds)) if odds < 0 else (odds/100)

def main():
    ap = argparse.ArgumentParser(description="Log outcomes and PnL for a completed date")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (completed games)")
    ap.add_argument("--bankroll", type=float, default=float(os.getenv("BANKROLL", 10000)))
    ap.add_argument("--stake-scale", type=float, default=float(os.getenv("STAKE_SCALE", 1.0)),
                    help="Multiplier if you scaled Kelly in production (e.g. 0.5 for half-Kelly)")
    ap.add_argument("--min-kelly", type=float, default=0.01, 
                    help="Minimum Kelly fraction to consider a bet (default: 0.01)")
    args = ap.parse_args()

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with engine.begin() as conn:
        # Ensure bet_outcomes table exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bet_outcomes (
              run_id uuid,
              game_id varchar,
              game_date date,
              home_team text,
              away_team text,
              side text,
              rec_prob double precision,
              odds integer,
              fair_odds integer,
              stake_frac double precision,
              stake_dollars double precision,
              b_return double precision,
              won boolean,
              push boolean,
              pnl_units double precision,
              pnl_dollars double precision,
              clv integer,
              model_version text,
              edge_pred double precision,
              abs_error_model double precision,
              created_at timestamp DEFAULT now(),
              PRIMARY KEY (run_id, game_id)
            )
        """))

        # Pull outcomes from the view
        df = pd.read_sql(
            text("""
                SELECT *
                FROM api_outcomes_all
                WHERE game_date = :d
                  AND rec_side IS NOT NULL
                  AND total_runs IS NOT NULL
                  AND kelly_best > :min_kelly
            """),
            conn, params={"d": args.date, "min_kelly": args.min_kelly}
        )

        if df.empty:
            print(f"❌ No completed games with qualifying predictions for {args.date}")
            print(f"   (Looking for games with rec_side, total_runs, and kelly_best > {args.min_kelly})")
            return

        print(f"📊 Found {len(df)} qualifying games for {args.date}")

        # Compute per-bet economics
        out_rows = []
        for _, r in df.iterrows():
            side = str(r["rec_side"])
            odds = int(r["side_odds"]) if pd.notna(r["side_odds"]) else None
            won = bool(r["won"]) if r["won"] is not None else None
            push = (str(r["actual_side"]) == "PUSH")

            if odds is None:
                print(f"⚠️  Skipping {r['away_team']} @ {r['home_team']}: No odds available")
                continue

            p = float(r["rec_prob"]) if pd.notna(r["rec_prob"]) else np.nan
            if pd.isna(p):
                print(f"⚠️  Skipping {r['away_team']} @ {r['home_team']}: No probability available")
                continue
                
            # Calculate fair odds from probability
            fair_odds = (-int(round(100*p/(1-p)))) if p > 0.5 else int(round(100*(1-p)/p))
            
            kelly_frac = float(r["kelly_best"]) if pd.notna(r["kelly_best"]) else 0.0
            stake_frac = max(0.0, kelly_frac) * float(args.stake_scale)
            b = american_win_return(float(odds))

            # Calculate PnL
            if push:
                pnl_units = 0.0
            elif won:
                pnl_units = stake_frac * b
            else:
                pnl_units = -stake_frac

            stake_dollars = stake_frac * float(args.bankroll)
            pnl_dollars = pnl_units * float(args.bankroll)

            out_rows.append({
                "run_id": r["run_id"],
                "game_id": r["game_id"],
                "game_date": r["game_date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "side": side,
                "rec_prob": p,
                "odds": odds,
                "fair_odds": fair_odds,
                "stake_frac": stake_frac,
                "stake_dollars": stake_dollars,
                "b_return": b,
                "won": None if push else won,
                "push": push,
                "pnl_units": pnl_units,
                "pnl_dollars": pnl_dollars,
                "clv": None,  # Fill later with CLV calculation
                "model_version": r.get("model_version"),
                "edge_pred": float(r["edge_pred"]) if pd.notna(r["edge_pred"]) else None,
                "abs_error_model": float(r["abs_error_model"]) if pd.notna(r["abs_error_model"]) else None
            })

        if not out_rows:
            print("❌ Nothing to log (missing odds or probabilities)")
            return

        # Upsert to bet_outcomes table
        upsert_sql = text("""
            INSERT INTO bet_outcomes (
              run_id, game_id, game_date, home_team, away_team, side,
              rec_prob, odds, fair_odds, stake_frac, stake_dollars, b_return,
              won, push, pnl_units, pnl_dollars, clv, model_version,
              edge_pred, abs_error_model
            ) VALUES (
              :run_id, :game_id, :game_date, :home_team, :away_team, :side,
              :rec_prob, :odds, :fair_odds, :stake_frac, :stake_dollars, :b_return,
              :won, :push, :pnl_units, :pnl_dollars, :clv, :model_version,
              :edge_pred, :abs_error_model
            )
            ON CONFLICT (run_id, game_id) DO UPDATE SET
              rec_prob      = EXCLUDED.rec_prob,
              odds          = EXCLUDED.odds,
              fair_odds     = EXCLUDED.fair_odds,
              stake_frac    = EXCLUDED.stake_frac,
              stake_dollars = EXCLUDED.stake_dollars,
              b_return      = EXCLUDED.b_return,
              won           = EXCLUDED.won,
              push          = EXCLUDED.push,
              pnl_units     = EXCLUDED.pnl_units,
              pnl_dollars   = EXCLUDED.pnl_dollars,
              clv           = EXCLUDED.clv,
              model_version = EXCLUDED.model_version,
              edge_pred     = EXCLUDED.edge_pred,
              abs_error_model = EXCLUDED.abs_error_model,
              created_at    = now()
        """)
        
        with engine.begin() as conn2:
            for row in out_rows:
                conn2.execute(upsert_sql, row)

        # Calculate and display summary
        n = len(out_rows)
        wins = sum(1 for r in out_rows if r["won"] is True)
        losses = sum(1 for r in out_rows if r["won"] is False)
        pushes = sum(1 for r in out_rows if r["push"])
        
        total_stake = sum(r["stake_frac"] for r in out_rows)
        total_pnl = sum(r["pnl_units"] for r in out_rows)
        roi = (total_pnl / total_stake) if total_stake > 0 else 0.0
        
        total_dollars = sum(r["stake_dollars"] for r in out_rows)
        total_pnl_dollars = sum(r["pnl_dollars"] for r in out_rows)
        
        print(f"\n✅ Logged {n} outcomes for {args.date}")
        print(f"📊 Results: W-L-P {wins}-{losses}-{pushes}")
        print(f"💰 ROI: {roi:.1%}")
        print(f"💵 Total staked: ${total_dollars:,.2f} ({total_stake:.3f} units)")
        print(f"📈 Total PnL: ${total_pnl_dollars:+,.2f} ({total_pnl:+.3f} units)")
        
        if wins + losses > 0:
            win_rate = wins / (wins + losses)
            print(f"🎯 Win rate: {win_rate:.1%}")

        # Show individual results
        print(f"\n📋 Individual results:")
        for row in out_rows:
            result = "PUSH" if row["push"] else ("WIN" if row["won"] else "LOSS")
            pnl_str = f"${row['pnl_dollars']:+.2f}" if abs(row['pnl_dollars']) >= 1 else f"{row['pnl_units']:+.3f}u"
            print(f"  {row['away_team']} @ {row['home_team']}: {row['side']} {row['odds']} - {result} ({pnl_str})")

if __name__ == "__main__":
    main()
