#!/usr/bin/env python3
"""
Probabilities and Expected Value Calculator

Converts predicted totals to calibrated probabilities using isotonic regression
and calculates expected value and Kelly criterion for optimal bet sizing.

This transforms our edge-based approach into proper probability-based betting.
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

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def american_win_return(odds):
    """Calculate profit (in units) on a 1u stake when you win at American odds"""
    if isinstance(odds, (pd.Series, np.ndarray)):
        # Vectorized calculation for pandas Series
        return np.where(odds < 0, 100/np.abs(odds), odds/100)
    else:
        # Scalar calculation
        return (100/abs(odds)) if odds < 0 else (odds/100)

def ev_from_prob(p, odds):
    """Calculate expected value from probability and American odds"""
    b = american_win_return(odds)
    return p * b - (1 - p) * 1.0

def kelly_fraction_robust(p, odds):
    """Calculate Kelly criterion fraction with 5% cap"""
    b = american_win_return(odds)
    if isinstance(b, (pd.Series, np.ndarray)):
        f = np.where(b > 0, (p * (b + 1) - 1) / b, 0)
        return np.clip(f, 0.0, 0.05)  # Cap at 5% instead of 10%
    else:
        f = (p * (b + 1) - 1) / b if b > 0 else 0
        return max(0.0, min(0.05, f))

def main():
    ap = argparse.ArgumentParser(description="Calculate probabilities and EV for MLB totals")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--window-days", type=int, default=30, help="Calibration window in days")
    ap.add_argument("--model-version", default="enhanced_bullpen_v1", help="Model version identifier")
    args = ap.parse_args()

    target_date = args.date
    end_dt = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=args.window_days - 1)

    print(f"🎯 Calculating probabilities for {target_date}")
    print(f"📊 Using calibration window: {start_dt} to {end_dt}")

    with engine.begin() as conn:
        # Calibration window: games with market, prediction, and actual results
        calib = pd.read_sql(text("""
            SELECT predicted_total, market_total, total_runs
            FROM enhanced_games
            WHERE "date" BETWEEN :s AND :e
              AND predicted_total IS NOT NULL
              AND market_total IS NOT NULL
              AND total_runs IS NOT NULL
        """), conn, params={"s": start_dt, "e": end_dt})

        # Target day: games to score
        today = pd.read_sql(text("""
            SELECT game_id, "date", predicted_total, market_total, 
                   over_odds, under_odds
            FROM enhanced_games
            WHERE "date" = :d
              AND predicted_total IS NOT NULL
              AND market_total IS NOT NULL
        """), conn, params={"d": target_date})

    if calib.empty:
        print(f"❌ No calibration data available ({start_dt} to {end_dt})")
        return
        
    if today.empty:
        print(f"❌ No games to score for {target_date}")
        return

    print(f"📈 Calibration: {len(calib)} games")
    print(f"🎲 Scoring: {len(today)} games")

    # 1) Robust sigma with higher floor 
    resid = calib["total_runs"] - calib["predicted_total"]
    sig = np.percentile(np.abs(resid.dropna()), 68.27) if len(resid) else 1.8
    sigma = float(np.clip(sig, 1.6, 3.0))  # ↑ floor, ↓ cap
    print(f"🔬 σ (robust): {sigma:.2f} runs")

    # Bias correction: mean(actual - predicted)
    bias = float(resid.mean())
    print(f"🔄 Bias correction: mean(actual - pred) = {bias:+.2f} runs")

    # 2) Calibration z-range to avoid extrapolation
    mu_cal = calib["predicted_total"].values + bias
    z_cal = (calib["market_total"].values - mu_cal) / sigma
    z_lo, z_hi = np.percentile(z_cal, [1, 99])

    # Build calibration probabilities
    p_over_raw_calib = 1.0 - norm.cdf(z_cal)
    y = (calib["total_runs"] > calib["market_total"]).astype(int).values

    # Score today's games
    out = today.copy()
    
    # A) Use exact line you priced: after matching odds, overwrite market total
    # TODO: Add book_total matching logic here when available
    # For now, use market_total as the line we're pricing
    use_total = out["market_total"].astype(float).values
    
    # B) Tame z before isotonic + raise σ floor (already done above)
    # Today: clamp z, shrink raw p toward 0.5, then isotonic
    mu_today = (out["predicted_total"].values + bias)
    z_today = (use_total - mu_today) / sigma
    z_today = np.clip(z_today, z_lo, z_hi)  # clamp to seen range
    
    p_over_raw_today = 1.0 - norm.cdf(z_today)
    p_over_raw_today = 0.5 + 0.70*(p_over_raw_today - 0.5)  # shrink toward 0.5

    # 3) Fit isotonic with minimum data fallback
    if len(calib) >= 200:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_over_raw_calib, y)
        print(f"📐 Isotonic calibrated on {len(p_over_raw_calib)} samples")
        
        # Score today's games with isotonic calibration
        out["p_over"] = iso.transform(p_over_raw_today)
    else:
        # Fallback to Normal model with slight shrinkage to 0.5
        print(f"⚠️  Too few calibration samples ({len(calib)} < 200), using Normal fallback")
        out["p_over"] = 0.5 + 0.85 * (p_over_raw_today - 0.5)
    
    out["p_under"] = 1.0 - out["p_over"]

    # Diagnostic "adjusted edge" that matches the priced line
    out["adj_edge"] = (mu_today - use_total).round(2)

    # Clip away pathologies
    eps = 1e-3
    out["p_over"]  = np.clip(out["p_over"],  eps, 1.0 - eps)
    out["p_under"] = 1.0 - out["p_over"]

    # Optional: quick sanity prints
    edge = out["adj_edge"]
    bad = ((edge > 0.5) & (out["p_over"] < 0.5)).sum() + ((edge < -0.5) & (out["p_over"] > 0.5)).sum()
    print(f"🧪 Sanity: sign mismatches after calibration = {bad}")

    # Handle missing odds - SKIP games with missing odds
    out["over_odds"]  = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    
    missing_odds = out["over_odds"].isna() | out["under_odds"].isna()
    if missing_odds.sum() > 0:
        print(f"⚠️  Skipping {missing_odds.sum()} games with missing odds")
        out = out[~missing_odds].copy()
    
    if out.empty:
        print("❌ No games with valid odds to analyze")
        return

    # Calculate Expected Value
    out["ev_over"] = out["p_over"] * american_win_return(out["over_odds"]) - (1 - out["p_over"])
    out["ev_under"] = out["p_under"] * american_win_return(out["under_odds"]) - (1 - out["p_under"])

    # C) Guardrails (skip coin-flips, cap Kelly, pick top 5)
    print("\n🛡️  Applying betting guardrails...")
    
    # 1. Skip knife-edge lines (probabilities too close to 50%)
    mask = (out["p_over"].sub(0.5).abs() >= 0.04)  # skip knife-edges
    out = out.loc[mask].copy()
    
    if out.empty:
        print("❌ No games pass knife-edge filter")
        return
    
    # 2. Calculate Kelly fractions (cap at 5%)
    out["kelly_over"]  = np.clip(kelly_fraction_robust(out["p_over"],  out["over_odds"]),  0, 0.05)
    out["kelly_under"] = np.clip(kelly_fraction_robust(out["p_under"], out["under_odds"]), 0, 0.05)
    out["best_kelly"]  = out[["kelly_over","kelly_under"]].max(axis=1)
    
    # 3. Take only the top 5 by Kelly
    out = out.sort_values("best_kelly", ascending=False).head(5)
    
    # Mark final recommendations based on better EV
    out["recommendation"] = "PASS"
    better_over = out["ev_over"] > out["ev_under"]
    out.loc[better_over, "recommendation"] = "OVER"
    out.loc[~better_over, "recommendation"] = "UNDER"

    # Quick sanity check
    print("≥0.90 probs:", int((out["p_over"]>=0.90).sum() + (out["p_under"]>=0.90).sum()))
    if not out.empty:
        # Add book_total column for display (same as market_total for now)
        out["book_total"] = out["market_total"]
        print(out[["game_id","market_total","book_total","p_over","ev_over","ev_under","best_kelly"]].head(10))

    # Create/update probability_predictions table
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS probability_predictions (
              game_id         varchar NOT NULL,
              game_date       date NOT NULL,
              market_total    numeric,
              predicted_total numeric,
              adj_edge        numeric,
              p_over          double precision,
              p_under         double precision,
              over_odds       integer,
              under_odds      integer,
              ev_over         double precision,
              ev_under        double precision,
              kelly_over      double precision,
              kelly_under     double precision,
              model_version   text,
              created_at      timestamp DEFAULT now(),
              PRIMARY KEY (game_id, game_date)
            )
        """))
        
        upsert_sql = text("""
            INSERT INTO probability_predictions (
                game_id, game_date,
                market_total, predicted_total,
                p_over, p_under,
                over_odds, under_odds,
                ev_over, ev_under,
                kelly_over, kelly_under,
                model_version, created_at,
                adj_edge
            ) VALUES (
                :gid, :d,
                :mt, :pt,
                :pov, :pun,
                :oo, :uo,
                :evo, :evu,
                :ko, :ku,
                :mv, NOW(),
                :ae
            )
            ON CONFLICT (game_id, game_date) DO UPDATE SET
                market_total    = EXCLUDED.market_total,
                predicted_total = EXCLUDED.predicted_total,
                p_over          = EXCLUDED.p_over,
                p_under         = EXCLUDED.p_under,
                over_odds       = EXCLUDED.over_odds,
                under_odds      = EXCLUDED.under_odds,
                ev_over         = EXCLUDED.ev_over,
                ev_under        = EXCLUDED.ev_under,
                kelly_over      = EXCLUDED.kelly_over,
                kelly_under     = EXCLUDED.kelly_under,
                model_version   = EXCLUDED.model_version,
                created_at      = NOW(),
                adj_edge        = EXCLUDED.adj_edge
        """)
        
        for r in out.to_dict("records"):
            conn.execute(upsert_sql, {
                "gid": r["game_id"], "d": r["date"],
                "mt": r["market_total"], "pt": r["predicted_total"],
                "ae": float(r["adj_edge"]),
                "pov": float(r["p_over"]), "pun": float(r["p_under"]),
                "oo": int(r["over_odds"]), "uo": int(r["under_odds"]),
                "evo": float(r["ev_over"]), "evu": float(r["ev_under"]),
                "ko": float(r["kelly_over"]), "ku": float(r["kelly_under"]),
                "mv": args.model_version
            })

    print(f"\n✅ Scored {len(out)} games for {target_date}")
    
    if out.empty:
        print("❌ No games with valid data for betting")
        return
    
    print("\n📊 Betting Recommendations:")
    print("   Game ID | Rec  | P(Over) | EV    | Kelly | Odds")
    print("   --------|------|---------|-------|-------|------")
    
    for _, row in out.iterrows():
        rec = row["recommendation"]
        if rec != "PASS":
            p_display = row["p_over"] if rec == "OVER" else row["p_under"]
            ev_display = row["ev_over"] if rec == "OVER" else row["ev_under"]
            kelly_display = row["kelly_over"] if rec == "OVER" else row["kelly_under"]
            odds_display = row["over_odds"] if rec == "OVER" else row["under_odds"]
            print(f"   {row['game_id']:>7} | {rec:>4} | {p_display:>7.3f} | {ev_display:>+5.3f} | {kelly_display:>5.3f} | {odds_display:>+4.0f}")

    # Summary statistics
    total_bets = (out["recommendation"] != "PASS").sum()
    over_bets = (out["recommendation"] == "OVER").sum()
    under_bets = (out["recommendation"] == "UNDER").sum()
    
    if total_bets > 0:
        avg_ev = out.loc[out["recommendation"] != "PASS", ["ev_over", "ev_under"]].max(axis=1).mean()
        avg_kelly = out.loc[out["recommendation"] != "PASS", "best_kelly"].mean()
        
        print(f"\n📈 Summary:")
        print(f"   Total bets recommended: {total_bets}")
        print(f"   OVER bets: {over_bets}")
        print(f"   UNDER bets: {under_bets}")
        print(f"   Average EV: {avg_ev:+.3f}")
        print(f"   Average Kelly: {avg_kelly:.3f}")
    else:
        print(f"\n📈 Summary:")
        print(f"   No bets meet guardrail criteria")
        print(f"   Games analyzed: {len(out)}")

if __name__ == "__main__":
    main()
