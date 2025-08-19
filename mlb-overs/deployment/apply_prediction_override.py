#!/usr/bin/env python3
import os, numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import argparse

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--league_avg", type=float, default=8.7, help="Fallback baseline")
    ap.add_argument("--clip_lo", type=float, default=6.5)
    ap.add_argument("--clip_hi", type=float, default=12.0)
    ap.add_argument("--shrink", type=float, default=0.33, help="weight on (raw - base)")
    args = ap.parse_args()

    eng = create_engine(DATABASE_URL, pool_pre_ping=True)

    with eng.begin() as conn:
        df = pd.read_sql(text("""
            SELECT game_id, "date", predicted_total, market_total, over_odds, under_odds
            FROM enhanced_games
            WHERE "date" = :d AND predicted_total IS NOT NULL
            ORDER BY game_id
        """), conn, params={"d": args.date})

    if df.empty:
        print(f"❌ No predictions found for {args.date}")
        return

    # Define "looks like pregame" filter
    mt = pd.to_numeric(df["market_total"], errors="coerce")
    oo = pd.to_numeric(df["over_odds"], errors="coerce")
    uo = pd.to_numeric(df["under_odds"], errors="coerce")
    looks_pregame = (
        mt.between(6.5, 12.5)
        & oo.between(-150, 150)
        & uo.between(-150, 150)
    )

    # Baseline: use pregame line if available; else league baseline
    base = np.where(looks_pregame, mt, args.league_avg)

    # Clip crazy raw preds, then shrink toward base
    raw = np.clip(pd.to_numeric(df["predicted_total"], errors="coerce"), args.clip_lo-0.5, args.clip_hi+1.0)
    corrected = base + args.shrink * (raw - base)
    corrected = np.clip(corrected, args.clip_lo, args.clip_hi)

    # Extra global sanity pass
    if float(np.nanmean(corrected)) > 9.8 or (corrected > 10).mean() > 0.60:
        corrected = base + 0.15 * (raw - base)
        corrected = np.clip(corrected, args.clip_lo, args.clip_hi)

    out = df[["game_id", "date"]].copy()
    out["predicted_total_new"] = corrected.astype(float)

    print("Before override: mean={:.2f}, >10 runs={:.0f}%".format(
        float(df["predicted_total"].mean()),
        100.0 * float((df["predicted_total"] > 10).mean())
    ))
    print("After  override: mean={:.2f}, >10 runs={:.0f}%".format(
        float(out["predicted_total_new"].mean()),
        100.0 * float((out["predicted_total_new"] > 10).mean())
    ))

    # Write back
    with eng.begin() as conn:
        upd = text("""
            UPDATE enhanced_games
               SET predicted_total = :p
             WHERE game_id = :gid AND "date" = :d
        """)
        for r in out.to_dict("records"):
            conn.execute(upd, {"p": r["predicted_total_new"], "gid": r["game_id"], "d": r["date"]})

    print(f"✅ Overrode {len(out)} predictions for {args.date}")

if __name__ == "__main__":
    main()
