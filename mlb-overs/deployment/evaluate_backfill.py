#!/usr/bin/env python3
"""
Evaluate model predictions over a date range and produce simple charts.

Usage examples:
  python evaluate_backfill.py --start 2025-07-15 --end 2025-08-14
  python evaluate_backfill.py --start 2025-07-15 --end 2025-08-14 --threshold 0.75 --odds -110

Outputs (in ./reports):
  - backtest_<start>_<end>.csv
  - scatter_pred_vs_actual_<start>_<end>.png
  - hist_residuals_<start>_<end>.png
  - bar_winrate_by_edge_<start>_<end>.png
  - line_cum_roi_<start>_<end>.png
"""
import os
import math
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

def american_payout_per_1u(odds: float) -> float:
    """Return net winnings for a 1.0u risk at given American odds if the bet wins."""
    if odds < 0:
        return 100.0 / abs(odds)     # e.g., -110 -> +0.9091
    else:
        return odds / 100.0          # e.g., +120 -> +1.2

def fetch(engine, start, end):
    sql = text("""
        SELECT
          date,
          game_id,
          home_team,
          away_team,
          market_total,
          predicted_total,
          total_runs,
          recommendation
        FROM enhanced_games
        WHERE date BETWEEN :start AND :end
    """)
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params={"start": start, "end": end})
    return df

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions and visualize outcomes")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--threshold", type=float, default=1.0, help="Edge threshold for placing bets (abs(pred-market))")
    ap.add_argument("--odds", type=float, default=-110, help="Assumed price for totals bets")
    ap.add_argument("--use-recommendation", action="store_true",
                    help="Bet only when recommendation is OVER/UNDER (ignores --threshold).")
    args = ap.parse_args()

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    start = args.start
    end = args.end
    outdir = Path("./reports")
    outdir.mkdir(parents=True, exist_ok=True)

    df = fetch(engine, start, end)
    # Only rows where we have both prediction and truth
    df = df[(df["predicted_total"].notna()) & (df["total_runs"].notna())].copy()

    if df.empty:
        print(f"No rows with both predictions and final totals between {start} and {end}.")
        return

    # Basic fields
    df["edge"] = df["predicted_total"].astype(float) - df["market_total"].astype(float)
    df["residual"] = df["predicted_total"].astype(float) - df["total_runs"].astype(float)
    df["abs_err"] = df["residual"].abs()

    # Truth vs market labels (push when equal)
    def truth_label(r):
        if pd.isna(r["market_total"]) or pd.isna(r["total_runs"]):
            return None
        if r["total_runs"] > r["market_total"]:
            return "OVER"
        if r["total_runs"] < r["market_total"]:
            return "UNDER"
        return "PUSH"

    df["result"] = df.apply(truth_label, axis=1)

    # Betting rule
    if args.use_recommendation:
        placed = df["recommendation"].isin(["OVER", "UNDER"])
        pick = df["recommendation"].where(placed, None)
    else:
        placed = df["edge"].abs() >= args.threshold
        pick = np.where(df["edge"] >= 0, "OVER", "UNDER")
        pick = pd.Series(pick).where(placed, None)

    df["placed_bet"] = placed
    df["bet_pick"] = pick

    # Per-bet returns (1.0u risk per bet)
    win_net = american_payout_per_1u(args.odds)
    def per_bet_return(r):
        if not r["placed_bet"]:
            return 0.0
        if r["result"] == "PUSH":
            return 0.0
        if r["bet_pick"] == r["result"]:
            return win_net
        return -1.0

    df["unit_return"] = df.apply(per_bet_return, axis=1)

    # Metrics
    mae_model = float(df["abs_err"].mean())
    mae_market = float((df["market_total"].astype(float) - df["total_runs"].astype(float)).abs().mean())
    bias = float(df["residual"].mean())
    n_all = len(df)
    n_bets = int(df["placed_bet"].sum())
    n_wins = int(((df["placed_bet"]) & (df["bet_pick"] == df["result"])).sum())
    n_push = int(((df["placed_bet"]) & (df["result"] == "PUSH")).sum())
    win_rate = (n_wins / n_bets) if n_bets > 0 else float("nan")
    roi = df.loc[df["placed_bet"], "unit_return"].sum() / max(n_bets, 1)

    print(f"\nEvaluation {start} → {end}")
    print(f"Rows with truth+prediction: {n_all}")
    print(f"Model MAE vs truth:   {mae_model:.3f}")
    print(f"Market MAE vs truth:  {mae_market:.3f}")
    print(f"Model bias (pred-actual): {bias:+.3f}")
    print(f"\nBetting rule: {'recommendation' if args.use_recommendation else f'|edge|>={args.threshold}'} @ odds {args.odds}")
    print(f"Placed bets: {n_bets}  |  Wins: {n_wins}  Push: {n_push}")
    print(f"Win rate: {win_rate:.1%} (pushes excluded)")
    print(f"Avg ROI per bet: {roi:+.3f}u")

    # Save per-game CSV
    csv_path = outdir / f"backtest_{start}_{end}.csv"
    cols = ["date","game_id","away_team","home_team","market_total","predicted_total","total_runs",
            "edge","recommendation","result","placed_bet","bet_pick","unit_return","residual","abs_err"]
    df[cols].to_csv(csv_path, index=False)
    print(f"Saved per-game results → {csv_path}")

    # Plots (one per figure)
    # 1) Scatter Predicted vs Actual
    plt.figure()
    plt.scatter(df["predicted_total"], df["total_runs"], s=18)
    lims = [
        math.floor(min(df["predicted_total"].min(), df["total_runs"].min()) - 0.5),
        math.ceil(max(df["predicted_total"].max(), df["total_runs"].max()) + 0.5),
    ]
    xs = np.linspace(lims[0], lims[1], 100)
    plt.plot(xs, xs)
    plt.xlabel("Predicted total")
    plt.ylabel("Actual total (final runs)")
    plt.title(f"Predicted vs Actual ({start} → {end})")
    fig1 = outdir / f"scatter_pred_vs_actual_{start}_{end}.png"
    plt.savefig(fig1, bbox_inches="tight", dpi=140)
    plt.close()
    print(f"Saved plot → {fig1}")

    # 2) Histogram of residuals
    plt.figure()
    plt.hist(df["residual"].values, bins=20)
    plt.xlabel("Residual = Pred − Actual")
    plt.ylabel("Count")
    plt.title(f"Residuals ({start} → {end})")
    fig2 = outdir / f"hist_residuals_{start}_{end}.png"
    plt.savefig(fig2, bbox_inches="tight", dpi=140)
    plt.close()
    print(f"Saved plot → {fig2}")

    # 3) Win rate by edge bucket (for placed bets)
    placed_df = df[df["placed_bet"]].copy()
    if not placed_df.empty:
        # 0.5-run buckets of |edge|
        def bucket(v):
            v = abs(v)
            b = math.floor(v*2)/2.0
            return f"[{b:.1f},{b+0.5:.1f})"
        placed_df["edge_bucket"] = placed_df["edge"].apply(bucket)
        grp = placed_df.groupby("edge_bucket", sort=True)
        wr = (grp.apply(lambda g: (g["bet_pick"]==g["result"]).mean())
                    .rename("win_rate")).reset_index()
        plt.figure()
        plt.bar(wr["edge_bucket"], wr["win_rate"])
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.xlabel("|edge| bucket")
        plt.ylabel("Win rate")
        plt.title(f"Win rate by |edge| bucket ({start} → {end})")
        fig3 = outdir / f"bar_winrate_by_edge_{start}_{end}.png"
        plt.savefig(fig3, bbox_inches="tight", dpi=140)
        plt.close()
        print(f"Saved plot → {fig3}")

        # 4) Cumulative ROI over time
        ts = placed_df.sort_values(["date","game_id"]).copy()
        ts["cum_roi"] = ts["unit_return"].cumsum()
        plt.figure()
        plt.plot(ts["cum_roi"].values)
        plt.xlabel("Bets in chronological order")
        plt.ylabel("Cumulative ROI (units)")
        plt.title(f"Cumulative ROI ({start} → {end})")
        fig4 = outdir / f"line_cum_roi_{start}_{end}.png"
        plt.savefig(fig4, bbox_inches="tight", dpi=140)
        plt.close()
        print(f"Saved plot → {fig4}")

if __name__ == "__main__":
    main()
