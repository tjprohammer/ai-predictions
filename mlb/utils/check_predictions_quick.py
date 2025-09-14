#!/usr/bin/env python3
"""Quick utility to inspect all predictions for a given date (default: today).

Shows whitelist (primary) prediction, learning system prediction, market line, and edges.
Usage:
  python mlb/utils/check_predictions_quick.py            # today
  python mlb/utils/check_predictions_quick.py --date 2025-09-11
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
from datetime import date, datetime
import math

def fmt(v):
    return "-" if v is None else f"{v:.2f}" if isinstance(v, (int,float)) and not math.isnan(v) else str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', '-d', help='YYYY-MM-DD (defaults to today)')
    args = ap.parse_args()
    target = args.date or date.today().isoformat()

    root = Path(__file__).resolve().parents[2]
    db_path = root / 'mlb.db'
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Pull all games with any prediction signal
    rows = cur.execute(
        '''SELECT game_id, home_team, away_team, market_total,
                  predicted_total, predicted_total_learning, predicted_total_whitelist,
                  ROUND(CASE WHEN predicted_total IS NOT NULL AND market_total IS NOT NULL
                             THEN predicted_total - market_total END, 2) as edge_primary,
                  ROUND(CASE WHEN predicted_total_learning IS NOT NULL AND market_total IS NOT NULL
                             THEN predicted_total_learning - market_total END, 2) as edge_learning
           FROM enhanced_games
           WHERE date = ?
           ORDER BY COALESCE(predicted_total, predicted_total_learning) DESC NULLS LAST''', (target,)
    ).fetchall()

    if not rows:
        print(f"No games found for {target}.")
        return

    print(f"Predictions for {target} (total games: {len(rows)})")
    print("Game                          Mkt  WL/Prim  Learn  WL_col  Edge  L_Edge")
    print("-"*78)
    diff_count = 0
    prim_count = 0
    learn_count = 0
    wl_count = 0
    prim_vals = []
    learn_vals = []
    for r in rows:
        prim = r['predicted_total']
        learn = r['predicted_total_learning']
        wl = r['predicted_total_whitelist']
        if prim is not None: prim_count += 1; prim_vals.append(prim)
        if learn is not None: learn_count += 1; learn_vals.append(learn)
        if wl is not None: wl_count += 1
        if prim is not None and learn is not None and abs(prim - learn) > 1e-6:
            diff_count += 1
        label = f"{r['away_team']} @ {r['home_team']}"[:26].ljust(26)
        print(f"{label}  {fmt(r['market_total']).rjust(4)}  {fmt(prim).rjust(6)}  {fmt(learn).rjust(5)}  {fmt(wl).rjust(6)}  {fmt(r['edge_primary']).rjust(5)}  {fmt(r['edge_learning']).rjust(6)}")

    def mean_std(vals):
        if not vals: return (None, None)
        m = sum(vals)/len(vals)
        v = sum((x-m)**2 for x in vals)/len(vals)
        return m, math.sqrt(v)

    prim_mean, prim_std = mean_std(prim_vals)
    learn_mean, learn_std = mean_std(learn_vals)
    print("\nSummary:")
    print(f"  Primary predictions (predicted_total): {prim_count}/{len(rows)}  mean={fmt(prim_mean)} sd={fmt(prim_std)}")
    print(f"  Learning predictions (predicted_total_learning): {learn_count}/{len(rows)}  mean={fmt(learn_mean)} sd={fmt(learn_std)}")
    print(f"  Whitelist column populated: {wl_count}/{len(rows)}")
    print(f"  Primary vs Learning different: {diff_count} games")

    conn.close()

if __name__ == '__main__':
    main()
