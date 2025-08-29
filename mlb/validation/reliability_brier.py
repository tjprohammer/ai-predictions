#!/usr/bin/env python3
"""
Reliability and Brier Score Analysis

Builds a 10-bin reliability diagram and calculates Brier score for the last N days.
Pulls the latest prediction per game, grades vs the priced_total if present (else market_total),
excludes pushes, and reports Brier, ECE, and a per-bin table.

Saves two figures: a reliability curve and a bin count histogram.
"""

# Windows-safe Unicode handling
import sys, os
if os.name == "nt":
    try:
        # Use UTF-8 and never crash on printing emojis
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def fetch_latest_preds(eng, start_date, end_date, model_version=None):
    """Fetch latest predictions per game in the date range"""
    q = """
    WITH ranked AS (
      SELECT pp.*,
             ROW_NUMBER() OVER (
               PARTITION BY game_id, game_date
               ORDER BY created_at DESC
             ) rn
      FROM probability_predictions pp
      WHERE game_date BETWEEN :s AND :e
        AND (:mv IS NULL OR model_version = :mv)
    )
    SELECT * FROM ranked WHERE rn = 1
    """
    return pd.read_sql(text(q), eng, params={"s": start_date, "e": end_date, "mv": model_version})

def fetch_finals(eng, start_date, end_date):
    """Fetch final game results"""
    q = """
    SELECT eg.game_id, eg."date" AS game_date,
           COALESCE(eg.total_runs, eg.home_score + eg.away_score) AS total_runs
    FROM enhanced_games eg
    WHERE eg."date" BETWEEN :s AND :e
    """
    return pd.read_sql(text(q), eng, params={"s": start_date, "e": end_date})

def main():
    ap = argparse.ArgumentParser(description="10-bin reliability + Brier over last N days")
    ap.add_argument("--days", type=int, default=30, help="Lookback window (days)")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (inclusive)")
    ap.add_argument("--model-version", default=None, help="Optional filter (e.g., enhanced_bullpen_v1)")
    ap.add_argument("--outfile-prefix", default="calibration", help="Prefix for saved PNG/CSV")
    args = ap.parse_args()

    end_dt   = datetime.strptime(args.end, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=args.days - 1)

    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    preds = fetch_latest_preds(eng, start_dt, end_dt, args.model_version)
    finals = fetch_finals(eng, start_dt, end_dt)

    if preds.empty:
        print("No predictions found in window."); return

    # Join with finals
    df = preds.merge(finals, on=["game_id","game_date"], how="left")
    
    # Choose PRICED line if available, else market_total
    if "priced_total" in df.columns and df["priced_total"].notna().any():
        df["line_total"] = df["priced_total"].astype(float)
        print(f"📊 Using priced_total for {df['priced_total'].notna().sum()}/{len(df)} games")
    else:
        df["line_total"] = pd.to_numeric(df["market_total"], errors="coerce")
        print(f"📊 Using market_total for all {len(df)} games")

    # Choose side (the side with higher EV)
    choose_over = df["ev_over"] >= df["ev_under"]
    df["side"]   = np.where(choose_over, "OVER", "UNDER")
    df["p_win"]  = np.where(choose_over, df["p_over"], df["p_under"])

    # Outcome vs line; exclude pushes
    df["won"] = np.where(df["side"].eq("OVER"),
                         (df["total_runs"] > df["line_total"]).astype(float),
                         (df["total_runs"] < df["line_total"]).astype(float))
    
    mask_valid = df["total_runs"].notna() & df["line_total"].notna() & (df["total_runs"] != df["line_total"])
    df = df.loc[mask_valid].copy()

    if df.empty:
        print("No graded games (likely all pushes/missing)."); return

    # Brier score
    df["brier"] = (df["p_win"] - df["won"])**2
    brier = float(df["brier"].mean())

    # 10-bin reliability
    bins = np.linspace(0.0, 1.0, 11)
    df["bin"] = pd.cut(df["p_win"], bins=bins, include_lowest=True, right=False)
    g = df.groupby("bin", observed=True)
    calib = pd.DataFrame({
        "n": g.size(),
        "avg_pred": g["p_win"].mean(),
        "emp_rate": g["won"].mean(),
    }).reset_index()
    calib["abs_gap"] = (calib["emp_rate"] - calib["avg_pred"]).abs()
    N = len(df)
    ece = float((calib["n"] / N * calib["abs_gap"]).sum())

    # Print summary and table
    print(f"📅 Window: {start_dt} → {end_dt}  (N={N})")
    print(f"🎯 Brier score: {brier:.4f}   |   ECE: {ece:.4f}")
    print(f"📈 Model version: {args.model_version or 'all'}")
    print("\nPer-bin calibration:")
    # Convert categorical columns to string before fillna to avoid TypeError
    calib_copy = calib.copy()
    for col in calib_copy.select_dtypes(include=['category']).columns:
        calib_copy[col] = calib_copy[col].astype(str)
    print(calib_copy.fillna('0').round(4).to_string(index=False))

    # Save CSV files
    calib_csv = f"{args.outfile_prefix}_bins_{end_dt}.csv".replace(" ", "_")
    df_csv    = f"{args.outfile_prefix}_events_{end_dt}.csv".replace(" ", "_")
    calib.to_csv(calib_csv, index=False)
    df[["game_id","game_date","side","p_win","won","line_total","total_runs","priced_book"]].to_csv(df_csv, index=False)
    print(f"\n💾 Saved: {calib_csv}")
    print(f"💾 Saved: {df_csv}")

    # Reliability curve
    fig1, ax1 = plt.subplots(figsize=(8,6))
    x = calib["avg_pred"].values
    y = calib["emp_rate"].values
    
    # Perfect calibration line
    ax1.plot([0,1],[0,1], linestyle="--", color="gray", alpha=0.7, label="Perfect calibration")
    
    # Scatter plot with size proportional to bin count
    sizes = np.maximum(30, 3*np.sqrt(calib["n"].values + 1))
    scatter = ax1.scatter(x, y, s=sizes, alpha=0.7, c=calib["n"], cmap="viridis")
    
    # Add count labels on points
    for xi, yi, ni in zip(x, y, calib["n"].values):
        if not np.isnan(xi) and not np.isnan(yi):
            ax1.text(xi, yi+0.02, str(int(ni)), fontsize=9, ha="center", va="bottom", fontweight="bold")
    
    ax1.set_xlabel("Predicted probability", fontsize=12)
    ax1.set_ylabel("Empirical win rate", fontsize=12)
    ax1.set_title(f"Reliability Curve (N={N})\nBrier={brier:.4f}  ECE={ece:.4f}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax1, label="Bin count")
    
    out_png1 = f"{args.outfile_prefix}_reliability_{end_dt}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(out_png1, dpi=150, bbox_inches="tight")
    print(f"📊 Saved: {out_png1}")
    plt.close()

    # Bin counts histogram
    fig2, ax2 = plt.subplots(figsize=(10,5))
    centers = (bins[:-1] + bins[1:]) / 2.0
    
    # Align calib to all bins (including empty)
    counts = np.zeros(10, dtype=int)
    for i, b in enumerate(pd.IntervalIndex.from_breaks(bins, closed="left")):
        if b in calib["bin"].values:
            counts[i] = int(calib.loc[calib["bin"].eq(b), "n"].values[0])
    
    bars = ax2.bar(centers, counts, width=0.08, alpha=0.7, edgecolor="black")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha="center", va="bottom", fontweight="bold")
    
    ax2.set_xlabel("Predicted probability (bin centers)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Prediction Distribution - 10 Bins (N={N})", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim(-0.05, 1.05)
    
    out_png2 = f"{args.outfile_prefix}_histogram_{end_dt}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"📊 Saved: {out_png2}")
    plt.close()

    print(f"\n✅ Calibration analysis complete!")
    print(f"📈 {N} graded predictions over {args.days} days")
    print(f"🎯 Brier: {brier:.4f} (lower is better)")
    print(f"📐 ECE: {ece:.4f} (lower is better)")

if __name__ == "__main__":
    main()
