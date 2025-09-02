import argparse, os, joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.isotonic import IsotonicRegression

def _get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url, pool_pre_ping=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    engine = _get_engine()
    q = text("""
      SELECT h.date, h.player_id, h.p_over_0_5, h.p_over_1_5, pgl.hits
      FROM hitter_prop_predictions h
      JOIN player_game_logs pgl
        ON pgl.player_id = h.player_id AND pgl.game_date = h.date
      WHERE h.date BETWEEN :s AND :e
        AND h.p_over_0_5 IS NOT NULL AND h.p_over_1_5 IS NOT NULL
        AND pgl.hits IS NOT NULL
    """)
    df = pd.read_sql(q, engine, params={"s": args.start, "e": args.end})
    if df.empty:
        print("[calibration] No data spanning that range.")
        return

    y05 = (df["hits"] >= 1).astype(int).values
    y15 = (df["hits"] >= 2).astype(int).values
    x05 = np.clip(df["p_over_0_5"].values, 1e-6, 1-1e-6)
    x15 = np.clip(df["p_over_1_5"].values, 1e-6, 1-1e-6)

    iso05 = IsotonicRegression(out_of_bounds="clip").fit(x05, y05)
    iso15 = IsotonicRegression(out_of_bounds="clip").fit(x15, y15)

    joblib.dump({"HITS_0.5": iso05, "HITS_1.5": iso15}, args.out)
    print(f"[calibration] Saved to {args.out}")

if __name__ == "__main__":
    main()
