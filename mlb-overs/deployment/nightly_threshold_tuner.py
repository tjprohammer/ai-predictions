#!/usr/bin/env python3
"""
Nightly threshold tuner - automatically selects optimal edge threshold
based on trailing 30-day performance and updates model_config.

Run this nightly after games finish to keep threshold optimized.
"""

import os
import math
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
ODDS = -110  # American odds for ROI calculation

def roi_from_record(wins, losses, pushes=0, odds=-110):
    """Calculate ROI per bet from W-L record and odds"""
    price = abs(odds)
    win_payoff = 100/price if odds < 0 else price/100
    # For -110: win +0.909u, lose -1u
    total = wins * win_payoff - losses * 1.0
    bets = wins + losses  # pushes excluded from bet count
    return (total/bets) if bets > 0 else 0.0

def tune_threshold():
    """Find optimal threshold from last 30 days of data"""
    with engine.begin() as conn:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        
        print(f"📊 Analyzing threshold performance: {start_date} to {end_date}")
        
        # Get last 30 days of completed games
        df = pd.read_sql(text("""
            WITH base AS (
              SELECT "date", market_total, predicted_total, total_runs
              FROM enhanced_games
              WHERE "date" BETWEEN :s AND :e
                AND market_total IS NOT NULL 
                AND predicted_total IS NOT NULL 
                AND total_runs IS NOT NULL
            )
            SELECT * FROM base
        """), conn, params={"s": start_date, "e": end_date})
        
        if df.empty:
            print("⚠️  No data available for threshold tuning")
            return None
            
        print(f"   Found {len(df)} completed games")

    best_thr, best_roi = THRESHOLDS[0], -1e9
    summary = []
    
    for thr in THRESHOLDS:
        g = df.copy()
        g["edge"] = g["predicted_total"] - g["market_total"]
        g["pick"] = g["edge"].apply(lambda x: "OVER" if x >= thr else ("UNDER" if x <= -thr else "NO BET"))
        g = g[g["pick"] != "NO BET"].copy()
        
        if g.empty:
            summary.append((thr, 0, 0, 0, 0.0))
            continue
            
        g["result"] = g.apply(lambda r: "OVER" if r["total_runs"] > r["market_total"]
                                        else ("UNDER" if r["total_runs"] < r["market_total"] else "PUSH"), axis=1)
        
        wins = (g["pick"] == g["result"]).sum()
        pushes = (g["result"] == "PUSH").sum()
        losses = len(g) - wins - pushes
        roi = roi_from_record(wins, losses, pushes, ODDS)
        
        summary.append((thr, len(g), wins, pushes, roi))
        
        # Update best if this threshold is better (require at least 5 bets)
        if roi > best_roi and len(g) >= 5:
            best_roi, best_thr = roi, thr

    print("\n📈 Threshold Performance Summary (last 30 days):")
    print("   Thr  | Bets | Wins | Push |   ROI")
    print("   -----|------|------|------|-------")
    for thr, n, w, p, roi in summary:
        marker = " 🎯" if thr == best_thr else "   "
        print(f"   {thr:>4} | {n:>4} | {w:>4} | {p:>4} | {roi:+.3f}u{marker}")

    return best_thr, best_roi, summary

def update_threshold(new_threshold):
    """Update the threshold in model_config"""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_config(key, value) VALUES('edge_threshold', :v)
            ON CONFLICT (key) DO UPDATE SET 
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP
        """), {"v": str(new_threshold)})

def main():
    print("🔧 Nightly Threshold Tuner")
    print("=" * 50)
    
    result = tune_threshold()
    if result is None:
        print("❌ Could not tune threshold - insufficient data")
        return
        
    best_thr, best_roi, summary = result
    
    # Get current threshold
    with engine.begin() as conn:
        current = conn.execute(text(
            "SELECT value FROM model_config WHERE key='edge_threshold'"
        )).scalar()
    
    print(f"\n🎯 Recommendation:")
    print(f"   Current threshold: {current}")
    print(f"   Optimal threshold: {best_thr}")
    print(f"   Expected ROI: {best_roi:+.3f}u per bet")
    
    if current != str(best_thr):
        update_threshold(best_thr)
        print(f"\n✅ Updated edge_threshold: {current} → {best_thr}")
        print("   Frontend will automatically use new threshold")
    else:
        print(f"\n✅ Threshold unchanged (already optimal)")

if __name__ == "__main__":
    main()
