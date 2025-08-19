#!/usr/bin/env python3
"""
Comprehensive Game Analysis - Shows all games before filtering

This version shows the full decision process for every game today,
including those that get filtered out by our strict criteria.
"""

import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import math
import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from sqlalchemy import create_engine, text
from sklearn.isotonic import IsotonicRegression
from math import erf, isfinite
from scipy.stats import norm
from uuid import uuid4

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def american_win_return(odds):
    """Calculate profit (in units) on a 1u stake when you win at American odds"""
    if isinstance(odds, (pd.Series, np.ndarray)):
        return np.where(odds < 0, 100/np.abs(odds), odds/100)
    else:
        return (100/abs(odds)) if odds < 0 else (odds/100)

def kelly_fraction_robust(p, odds):
    """Calculate Kelly criterion fraction with 5% cap"""
    b = american_win_return(odds)
    if isinstance(b, (pd.Series, np.ndarray)):
        f = np.where(b > 0, (p * (b + 1) - 1) / b, 0)
        return np.clip(f, 0.0, 0.05)
    else:
        f = (p * (b + 1) - 1) / b if b > 0 else 0
        return max(0.0, min(0.05, f))

def prob_for_side(row):
    """Get probability for the recommended side"""
    return row["p_over"] if row["recommendation"] == "OVER" else row["p_under"]

def fair_american(p):
    """Convert probability to fair American odds"""
    if p <= 0 or p >= 1:
        return 0
    return -int(round(100*p/(1-p))) if p > 0.5 else int(round(100*(1-p)/p))

def main():
    target_date = "2025-08-17"
    end_dt = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=30 - 1)

    print(f"🎯 Comprehensive Analysis for {target_date}")
    print(f"📊 Using calibration window: {start_dt} to {end_dt}")

    with engine.begin() as conn:
        # Calibration window
        calib = pd.read_sql(text("""
            SELECT predicted_total, market_total, total_runs
            FROM enhanced_games
            WHERE "date" BETWEEN :s AND :e
              AND predicted_total IS NOT NULL
              AND market_total IS NOT NULL
              AND total_runs IS NOT NULL
        """), conn, params={"s": start_dt, "e": end_dt})

        # All games today
        today = pd.read_sql(text("""
            SELECT game_id, "date", home_team, away_team, predicted_total, market_total, 
                   over_odds, under_odds
            FROM enhanced_games
            WHERE "date" = :d
              AND predicted_total IS NOT NULL
              AND market_total IS NOT NULL
        """), conn, params={"d": target_date})

    print(f"📈 Calibration: {len(calib)} games")
    print(f"🎲 All games today: {len(today)} games")

    # Same calibration as production system
    resid = calib["total_runs"] - calib["predicted_total"]
    sig = np.percentile(np.abs(resid.dropna()), 68.27) if len(resid) else 1.8
    sigma = float(np.clip(sig, 1.9, 3.0))
    bias = float(resid.mean())
    
    z_cal = (calib["market_total"].values - (calib["predicted_total"].values + bias)) / sigma
    y = (calib["total_runs"] > calib["market_total"]).astype(int).values

    def nll(s):
        p = 1.0 - norm.cdf(s * z_cal)
        p = np.clip(p, 1e-4, 1-1e-4)
        return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

    S = np.linspace(0.6, 1.6, 41)
    s = float(S[np.argmin([nll(si) for si in S])])
    
    print(f"🔬 σ = {sigma:.2f}, bias = {bias:+.2f}, temp_s = {s:.3f}")

    # Score ALL games (no filtering yet)
    out = today.copy()
    use_total = out["market_total"].astype(float).values
    
    z_today = (use_total - (out["predicted_total"].values + bias)) / sigma
    z_today = np.clip(z_today, np.percentile(z_cal,1), np.percentile(z_cal,99))
    p_over_raw_today = 1.0 - norm.cdf(s * z_today)
    p_over_raw_today = 0.5 + 0.55*(p_over_raw_today - 0.5)

    # Isotonic calibration
    p_over_raw_calib = 1.0 - norm.cdf(s * z_cal)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_over_raw_calib, y)
    out["p_over"] = iso.transform(p_over_raw_today)
    out["p_under"] = 1.0 - out["p_over"]

    # Apply probability clamps
    out["p_over"] = out["p_over"].clip(0.35, 0.65)
    out["p_under"] = 1 - out["p_over"]

    # Calculate edge and other metrics
    out["adj_edge"] = ((out["predicted_total"].values + bias) - use_total).round(2)
    out["over_odds"] = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    
    # Remove games with missing odds
    missing_odds = out["over_odds"].isna() | out["under_odds"].isna()
    if missing_odds.sum() > 0:
        print(f"⚠️  Removing {missing_odds.sum()} games with missing odds")
        out = out[~missing_odds].copy()

    # Calculate EV and Kelly
    out["ev_over"] = out["p_over"] * american_win_return(out["over_odds"]) - (1 - out["p_over"])
    out["ev_under"] = out["p_under"] * american_win_return(out["under_odds"]) - (1 - out["p_under"])
    
    base_k_over = kelly_fraction_robust(out["p_over"], out["over_odds"])
    base_k_under = kelly_fraction_robust(out["p_under"], out["under_odds"])
    out["kelly_over"] = np.clip(0.33*base_k_over, 0, 0.03)
    out["kelly_under"] = np.clip(0.33*base_k_under, 0, 0.03)
    out["best_kelly"] = out[["kelly_over","kelly_under"]].max(axis=1)
    
    # Determine recommendations
    out["recommendation"] = "PASS"
    better_over = out["ev_over"] > out["ev_under"]
    out.loc[better_over, "recommendation"] = "OVER"
    out.loc[~better_over, "recommendation"] = "UNDER"

    # Drop sign-mismatch games immediately (don't show in analysis)
    agree = np.sign(out["adj_edge"]) == np.sign(out["p_over"] - 0.5)
    sign_mismatches = (~agree).sum()
    out = out[agree].copy()
    print(f"🚫 Filtered out {sign_mismatches} sign-mismatch games")

    # Add side-specific metrics for cleaner display
    out["p_side"] = out.apply(prob_for_side, axis=1)
    out["fair_odds"] = out["p_side"].apply(fair_american)
    out["best_ev"] = out[["ev_over","ev_under"]].max(axis=1)
    out["side_odds"] = np.where(out["recommendation"] == "OVER", out["over_odds"], out["under_odds"])

    # Add filter flags (but don't filter yet)
    out["knife_edge"] = out["p_over"].sub(0.5).abs() >= 0.04
    out["min_kelly"] = out["best_kelly"] >= 0.005
    out["min_edge"] = out["adj_edge"].abs() >= 0.30
    out["min_ev"] = out["best_ev"] >= 0.05
    
    out["passes_all"] = (out["knife_edge"] & out["min_kelly"] & out["min_edge"] & out["min_ev"])

    # Sort by best Kelly for display
    out = out.sort_values("best_kelly", ascending=False)

    print(f"\n📋 COMPREHENSIVE ANALYSIS - {len(out)} Games (After Sign Filter):")
    print("   Game ID | Teams           | Rec  | P(Side) | Edge | EV    | Kelly | Fair  | Filters")
    print("   --------|-----------------|------|---------|----- |-------|-------|-------|--------")
    
    for _, row in out.iterrows():
        away = row['away_team'][:3].upper() if pd.notna(row['away_team']) else '???'
        home = row['home_team'][:3].upper() if pd.notna(row['home_team']) else '???'
        
        # Filter status
        filters = ""
        if not row["knife_edge"]: filters += "K"
        if not row["min_kelly"]: filters += "k"
        if not row["min_edge"]: filters += "E"
        if not row["min_ev"]: filters += "V"
        if row["passes_all"]: filters = "✅"
        if filters == "": filters = "✅"
        
        print(f"   {row['game_id']:>7} | {away}@{home:<12} | {row['recommendation']:>4} | {row['p_side']:>7.3f} | {row['adj_edge']:>+4.1f} | {row['best_ev']:>+5.3f} | {row['best_kelly']:>5.3f} | {row['fair_odds']:>+5.0f} | {filters:<7}")

    # Summary
    total_games = len(out)
    knife_fails = (~out["knife_edge"]).sum()
    kelly_fails = (~out["min_kelly"]).sum()
    edge_fails = (~out["min_edge"]).sum()
    ev_fails = (~out["min_ev"]).sum()
    passes_all = out["passes_all"].sum()
    
    print(f"\n📊 Filter Analysis:")
    print(f"   Analyzed games: {total_games} (after sign filter)")
    print(f"   Knife-edge: {knife_fails} (K)")
    print(f"   Low Kelly: {kelly_fails} (k)")
    print(f"   Small edge: {edge_fails} (E)")
    print(f"   Low EV: {ev_fails} (V)")
    print(f"   ✅ Passes all filters: {passes_all}")
    
    if passes_all > 0:
        actionable = out[out["passes_all"]].head(5)  # Top 5 like production
        print(f"\n🎯 ACTIONABLE BETS (Top 5):")
        print("   Teams        | Side  | Prob | EV    | Kelly | Fair  | Book")
        print("   -------------|-------|------|-------|-------|-------|------")
        for _, row in actionable.iterrows():
            away = row['away_team'][:3].upper()
            home = row['home_team'][:3].upper()
            
            print(f"   {away}@{home:<8} | {row['recommendation']:>5} | {row['p_side']:>4.1%} | {row['best_ev']:>+5.1%} | {row['best_kelly']:>5.1%} | {row['fair_odds']:>+5.0f} | {row['side_odds']:>+4.0f}")

if __name__ == "__main__":
    main()
