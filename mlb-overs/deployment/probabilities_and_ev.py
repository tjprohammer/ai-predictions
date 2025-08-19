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
from uuid import uuid4

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def calibration_bins(p_pred, y, n_bins=10):
    """Return DataFrame of reliability bins with counts/means/gaps/Brier."""
    p = np.asarray(p_pred, float)
    y = np.asarray(y, int)
    # equal-count bins are more stable than fixed edges
    qs = np.quantile(p, np.linspace(0, 1, n_bins+1))
    # ensure monotone & unique edges
    qs = np.unique(np.round(qs, 6))
    # fallback if duplicates collapse bins
    if len(qs) < 3:
        qs = np.linspace(0, 1, min(5, n_bins)+1)
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (p >= lo) & (p <= hi if hi == qs[-1] else p < hi)
        if not mask.any():
            continue
        pp = p[mask].mean()
        rr = y[mask].mean()
        gap = rr - pp
        brier = np.mean((y[mask] - p[mask])**2)
        rows.append({"bin_low": float(lo), "bin_high": float(hi),
                     "count": int(mask.sum()), "avg_pred": float(pp),
                     "emp_rate": float(rr), "gap": float(gap), "brier": float(brier)})
    return pd.DataFrame(rows).sort_values("bin_low").reset_index(drop=True)

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

def prob_for_side(row):
    """Get probability for the recommended side"""
    return row["p_over"] if row["recommendation"] == "OVER" else row["p_under"]

def fair_american(p):
    """Convert probability to fair American odds"""
    if p <= 0 or p >= 1:
        return 0
    return -int(round(100*p/(1-p))) if p > 0.5 else int(round(100*(1-p)/p))

def save_all_probabilities_to_db(all_games_df, target_date, run_id, model_version, sigma, temp_s, bias, engine):
    """Save probability calculations for ALL games, not just betting recommendations"""
    with engine.begin() as conn:
        # Delete existing predictions for this run
        conn.execute(text("""
            DELETE FROM probability_predictions 
            WHERE game_date = :d AND run_id = :rid
        """), {"d": target_date, "rid": str(run_id)})
        
        upsert_sql = text("""
            INSERT INTO probability_predictions (
                game_id, game_date, run_id,
                market_total, predicted_total,
                p_over, p_under,
                over_odds, under_odds,
                ev_over, ev_under,
                kelly_over, kelly_under,
                recommendation, model_version, created_at,
                adj_edge, sigma, temp_s, bias,
                p_side, fair_odds, priced_total, priced_book, stake,
                n_books, spread_cents, pass_reason
            ) VALUES (
                :gid, :d, :rid,
                :mt, :pt,
                :pov, :pun,
                :oo, :uo,
                :evo, :evu,
                :ko, :ku,
                :rec, :mv, NOW(),
                :ae, :sig, :temp_s, :bias,
                :p_side, :fair_odds, :priced_total, :priced_book, :stake,
                :n_books, :spread_cents, :pass_reason
            )
        """)
        
        for _, r in all_games_df.iterrows():
            # Calculate EV and Kelly for both sides
            ev_over = ev_from_prob(r["p_over"], r["over_odds"])
            ev_under = ev_from_prob(r["p_under"], r["under_odds"])
            kelly_over = kelly_fraction_robust(r["p_over"], r["over_odds"])
            kelly_under = kelly_fraction_robust(r["p_under"], r["under_odds"])
            
            # Determine recommendation (but mark as analysis-only if not a betting rec)
            if ev_over > ev_under and ev_over > 0:
                rec_side = "OVER"
                rec_prob = r["p_over"]
                rec_odds = r["over_odds"]
            elif ev_under > ev_over and ev_under > 0:
                rec_side = "UNDER"
                rec_prob = r["p_under"]
                rec_odds = r["under_odds"]
            else:
                rec_side = "HOLD"
                rec_prob = 0.5
                rec_odds = -110
            
            fair_odds = fair_american(rec_prob)
            
            conn.execute(upsert_sql, {
                "gid": r["game_id"], "d": r["date"], "rid": str(run_id),
                "mt": float(r["market_total"]), "pt": float(r["predicted_total"]),
                "pov": float(r["p_over"]), "pun": float(r["p_under"]),
                "oo": int(r["over_odds"]), "uo": int(r["under_odds"]),
                "evo": float(ev_over), "evu": float(ev_under),
                "ko": float(kelly_over), "ku": float(kelly_under),
                "rec": rec_side, "mv": model_version,
                "ae": float(r["adj_edge"]), "sig": float(sigma), "temp_s": float(temp_s), "bias": float(bias),
                "p_side": float(rec_prob), "fair_odds": int(fair_odds),
                "priced_total": float(r.get("priced_total", r["market_total"])),
                "priced_book": r.get("priced_book", "consensus"),
                "stake": 0.0,  # Will be calculated later for actual bets
                "n_books": int(r.get("n_books", 1)) if pd.notna(r.get("n_books", 1)) else 1,
                "spread_cents": int(r.get("spread_cents", 0)) if pd.notna(r.get("spread_cents", 0)) else 0,
                "pass_reason": str(r.get("pass_reason", ""))
            })
            
        print(f"✅ Saved probabilities for {len(all_games_df)} games to database")

def main():
    ap = argparse.ArgumentParser(description="Calculate probabilities and EV for MLB totals")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--window-days", type=int, default=30, help="Calibration window in days")
    ap.add_argument("--model-version", default="enhanced_bullpen_v1", help="Model version identifier")
    args = ap.parse_args()

    target_date = args.date
    end_dt = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=args.window_days - 1)
    
    # Generate unique run ID for traceability
    run_id = uuid4()

    print(f"🎯 Calculating probabilities for {target_date}")
    print(f"📊 Using calibration window: {start_dt} to {end_dt}")
    print(f"🆔 Run ID: {str(run_id)[:8]}...")

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

        # Target day: games to score (include games without market_total)
        today = pd.read_sql(text("""
            SELECT game_id, "date", predicted_total, market_total, 
                   over_odds, under_odds
            FROM enhanced_games
            WHERE "date" = :d
              AND predicted_total IS NOT NULL
        """), conn, params={"d": target_date})

    if calib.empty:
        print(f"❌ No calibration data available ({start_dt} to {end_dt})")
        return
        
    if today.empty:
        print(f"❌ No games to score for {target_date}")
        return

    print(f"📈 Calibration: {len(calib)} games")
    print(f"🎲 Scoring: {len(today)} games")
    
    # Handle games without market_total by using reasonable default (8.5)
    missing_market = today["market_total"].isna().sum()
    if missing_market > 0:
        print(f"⚠️  {missing_market} games missing market_total, using 8.5 as default")
        # Use reasonable default for missing market totals (don't use predicted_total!)
        today["market_total"] = today["market_total"].fillna(8.5)

    # 1) Robust sigma with higher floor 
    resid = calib["total_runs"] - calib["predicted_total"]
    sig = np.percentile(np.abs(resid.dropna()), 68.27) if len(resid) else 1.8
    sigma = float(np.clip(sig, 1.9, 3.0))   # ↑ was 1.6, now 1.9 (cooler)
    print(f"🔬 σ (robust): {sigma:.2f} runs")

    # Bias correction: mean(actual - predicted) with shrinkage and capping
    raw_bias = float(resid.mean())
    n = len(resid)
    # James-Stein style shrink toward 0 + hard cap
    w = n / (n + 1000.0)           # tune 500-1500 if needed
    bias = float(np.clip(raw_bias * w, -0.25, 0.25))
    print(f"🔄 Bias correction (shrunk): mean={raw_bias:+.2f} → used={bias:+.2f} runs")

    # 2) Temperature scaling on z (learn scale s on calibration window)
    z_cal = (calib["market_total"].values - (calib["predicted_total"].values + bias)) / sigma
    y = (calib["total_runs"] > calib["market_total"]).astype(int).values

    # Fit temperature s on calibration (p = Φ(s * z))
    def nll(s):
        p = 1.0 - norm.cdf(s * z_cal)
        p = np.clip(p, 1e-4, 1-1e-4)
        return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

    # Simple grid search (robust, no SciPy optimize dependency)
    S = np.linspace(0.6, 1.6, 41)  # widen if needed
    s = float(S[np.argmin([nll(si) for si in S])])
    print(f"🧪 Temperature s = {s:.3f}")

    # Raw probs for isotonic fit (using s)
    p_over_raw_calib = 1.0 - norm.cdf(s * z_cal)

    # Score today's games
    out = today.copy()
    
    # A) Use exact line you priced: match book totals from totals_odds table
    has_odds = False
    with engine.begin() as conn:
        odds = pd.read_sql(text("""
            SELECT game_id, "date", book, total AS book_total,
                   over_odds, under_odds, collected_at
            FROM totals_odds
            WHERE "date" = :d
        """), conn, params={"d": target_date})
    
    # Force consistent types just in case
    today["game_id"] = today["game_id"].astype(str)
    odds["game_id"] = odds["game_id"].astype(str)
    
    if not odds.empty:
        # Merge today's market_total onto every odds row for the same game
        m = odds.merge(
            today[["game_id", "market_total"]],
            on="game_id", how="inner", suffixes=("", "_mt")
        )

        # Distance between the book line and your market_total
        m["tot_diff"] = (m["book_total"].astype(float) - m["market_total"].astype(float)).abs()

        # Keep only reasonably close lines (<= 0.5). If you want to be looser, set to 1.0
        m = m[m["tot_diff"] <= 0.5].copy()

        # Prefer smallest diff, then newest snapshot
        m = (m.sort_values(["game_id", "tot_diff", "collected_at"],
                           ascending=[True, True, False])
               .drop_duplicates("game_id", keep="first"))

        # Merge the chosen book line back onto today's rows
        out = today.merge(
            m[["game_id", "book", "book_total", "over_odds", "under_odds"]],
            on="game_id", how="left"
        )

        # Use the exact priced line when present; otherwise fallback
        use_total = out["book_total"].fillna(out["market_total"]).astype(float).values
        out["priced_total"] = use_total
        out["priced_book"] = out["book"].fillna("market")
        has_odds = True

        print(f"📊 Using book totals for {out['book_total'].notna().sum()}/{len(out)} games "
              f"from books: {sorted(out['book'].dropna().unique().tolist())}")
        
        # ✅ FIXED consensus block - use book_total not total
        consensus = (odds.groupby(["game_id","book_total"])
                   .agg(n_books=("book","nunique"),
                        spread_cents=("over_odds", lambda x: x.max()-x.min() if len(x) > 1 else 0))
                   .reset_index())
        
        # Merge consensus data onto our games
        out = out.merge(consensus, 
                      left_on=["game_id", "priced_total"], 
                      right_on=["game_id", "book_total"], 
                      how="left", suffixes=("", "_consensus"))
        print(f"📈 Market consensus: {out['n_books'].fillna(1).mean():.1f} avg books per game")
    else:
        print(f"⚠️  No odds in totals_odds for this date; falling back to market_total")
        out = today.copy()
        use_total = out["market_total"].astype(float).values
        out["priced_total"] = use_total
        out["priced_book"] = "market"
        # Add default consensus values when no odds data available
        out["n_books"] = 1
        out["spread_cents"] = 0
    
    # B) Today: de-mean day's edges, then clamp, scale, shrink, and isotonic
    # Edge before any filtering (to detect day-level drift)
    edge_all = (out["predicted_total"].values + bias) - use_total
    # Robust "center": trimmed median, then cap
    global_shift = float(np.clip(np.median(edge_all), -0.75, 0.75))
    print(f"🧭 Day de-mean shift applied: {global_shift:+.2f} runs")
    
    pred_centered = out["predicted_total"].values - global_shift
    
    z_today = (use_total - (pred_centered + bias)) / sigma
    z_today = np.clip(z_today, np.percentile(z_cal,1), np.percentile(z_cal,99))
    p_over_raw_today = 1.0 - norm.cdf(s * z_today)
    p_over_raw_today = 0.5 + 0.55*(p_over_raw_today - 0.5)  # ↓ was 0.65, now 0.55 (harder shrink)

    # Quick sanity checks for balanced edges
    edges = (pred_centered + bias) - use_total
    print(f"🧪 Day edges: median={np.median(edges):+.2f}, p90={np.percentile(edges,90):+.2f}, "
          f"pos%={(edges>0).mean():.1%}, |edge|>1 runs={(np.abs(edges)>1).mean():.1%}")

    # Record what we priced (exact line & book)
    out["priced_total"] = pd.Series(use_total, index=out.index)
    out["priced_book"] = out.get("book", pd.Series("market", index=out.index)).fillna("market")

    # Safety prints for debugging
    missing = out["book_total"].isna().sum() if "book_total" in out.columns else len(out)
    if missing:
        print(f"ℹ️  No close book line for {missing} game(s). Using market_total for those.")
    
    print("📋 First 5 games with pricing info:")
    if len(out) > 0:
        # Only show columns that actually exist
        cols_to_show = ["game_id","market_total","priced_total","priced_book"]
        for col in ["book_total","over_odds","under_odds"]:
            if col in out.columns:
                cols_to_show.append(col)
        
        print(out[cols_to_show].head(5).to_string(index=False))

    # 3) Robust calibration: orientation check, anchors, OOF isotonic, blend
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    def fit_iso_oof(p_raw, y, n_splits=5, anchor_weight=50.0):
        """
        Returns (iso_model, p_oof), where:
          - iso_model is refit on all data (for deployment),
          - p_oof are out-of-fold calibrated probs for diagnostics.
        Applies orientation correction, anchor points, and class-balanced weights.
        """
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y, dtype=int)

        # --- 1) Orientation check (make sure higher p_raw -> more overs) ---
        rho, _ = spearmanr(p_raw, y)
        flip = (rho < 0)
        if flip:
            p_use = 1.0 - p_raw
        else:
            p_use = p_raw.copy()

        # --- 2) Class-balanced weights (avoid class bias) ---
        pos = y.sum()
        neg = len(y) - pos
        # Avoid div by zero if extreme
        w_pos = 0.5 / max(pos, 1)
        w_neg = 0.5 / max(neg, 1)
        base_w = np.where(y == 1, w_pos, w_neg)

        # --- 3) Anchor points (pull toward identity & prevent edge pathologies) ---
        # Anchors at 0->0, 0.5->0.5, 1->1
        p_aug = np.concatenate([p_use, [0.0, 0.5, 1.0]])
        y_aug = np.concatenate([y,     [0,   1,   1]])  # for monotone 0<=p<=1, mid pulls to 0.5
        # Better mid anchor: split 0.5 target to 0 and 1 with half weight each, or set 0.5 exactly.
        # We'll use exact 0.5 via two half-weights to keep binary targets:
        y_aug = np.concatenate([y, [0, 1, 0, 1]])
        p_aug = np.concatenate([p_use, [0.5, 0.5, 0.0, 1.0]])
        w_aug = np.concatenate([base_w, [anchor_weight/2, anchor_weight/2, anchor_weight, anchor_weight]])

        # --- 4) Out-of-fold isotonic for stability diagnostics ---
        kf = KFold(n_splits=min(n_splits, max(2, len(y)//40)), shuffle=True, random_state=42)
        p_oof = np.zeros_like(p_use, dtype=float)
        for tr, va in kf.split(p_use):
            iso_cv = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_cv.fit(p_aug[np.r_[tr, -4, -3, -2, -1]],  # train fold + 4 anchors
                       y_aug[np.r_[tr, -4, -3, -2, -1]],
                       sample_weight=w_aug[np.r_[tr, -4, -3, -2, -1]])
            p_oof[va] = iso_cv.transform(p_use[va])

        # --- 5) Final model: refit on ALL data + anchors ---
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_aug, y_aug, sample_weight=w_aug)

        # Undo flip if we flipped inputs (ensure output is "prob over")
        if flip:
            # If we trained on 1-p, calibrated prob for "over" is 1 - iso(1-p)
            def deploy_transform(p_now):
                return 1.0 - iso.transform(1.0 - p_now)
        else:
            def deploy_transform(p_now):
                return iso.transform(p_now)

        return deploy_transform, p_oof

    # --- Build calibration inputs you already had ---
    p_over_raw_calib = 1.0 - norm.cdf(s * z_cal)

    # Safety: if too small window, keep your Normal fallback
    MIN_CAL_SAMPLES = 200
    clamp_range = 0.15  # Default clamp range
    if len(calib) < MIN_CAL_SAMPLES:
        print(f"⚠️  Too few calibration samples ({len(calib)} < {MIN_CAL_SAMPLES}), using Normal fallback")
        p_tmp_cal = 0.5 + 0.85 * (p_over_raw_calib - 0.5)
        p_tmp = 0.5 + 0.85 * (p_over_raw_today - 0.5)
    else:
        deploy_transform, p_oof = fit_iso_oof(p_over_raw_calib, y)
        print(f"📐 Robust isotonic calibrated on {len(p_over_raw_calib)} samples")

        # --- Enhanced Reliability Report with Bins ---
        # Build reliability bins on OOF predictions
        bins_df = calibration_bins(p_oof, y, n_bins=10)

        # Brier vs identity
        brier_iso = float(np.mean((y - p_oof)**2))
        brier_id  = float(np.mean((y - p_over_raw_calib)**2))
        improve   = (brier_id - brier_iso) / max(brier_id, 1e-9)

        print("📊 Reliability Report:")
        print(f"   Brier (isotonic): {brier_iso:.4f}")
        print(f"   Brier (identity): {brier_id:.4f}")
        print(f"   Improvement: {improve:+.1%}")

        # Worst 3 gaps
        if not bins_df.empty:
            worst = (bins_df.assign(abs_gap=bins_df["gap"].abs())
                             .sort_values("abs_gap", ascending=False).head(3))
            print("   📈 Calibration bins (worst 3):")
            for _, r in worst.iterrows():
                print(f"      ({r.bin_low:.1f},{r.bin_high:.1f}]: {r['count']} games, gap={r.gap:+.3f}")

        # AUC calculation for dynamic clamping
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = roc_auc_score(y, p_oof)
            print(f"   🎯 OOF AUC: {auc_score:.3f}")
            
            # Dynamic clamp adjustment
            if auc_score > 0.65 and brier_iso < brier_id * 0.95:  # Strong performance
                clamp_range = 0.20  # 0.30-0.70 range
                print(f"   📏 Dynamic clamp: Strong performance, widening to ±{clamp_range}")
            else:
                clamp_range = 0.15  # Default 0.35-0.65 range
                print(f"   📏 Dynamic clamp: Standard range ±{clamp_range}")
        except:
            clamp_range = 0.15
            print(f"   📏 Dynamic clamp: Fallback to standard ±{clamp_range}")

        # --- If isotonic underperforms, force identity ---
        α_base = max(0.2, min(0.8, len(calib) / 2000.0))  # 0.2..0.8
        if brier_iso >= brier_id:
            print("   ⚠️  Isotonic underperforms → using identity (α=0)")
            p_tmp = p_over_raw_today
        else:
            α = α_base
            print(f"   ✅ Isotonic outperforms, α = {α:.2f}")
            p_iso_today = deploy_transform(p_over_raw_today)
            p_tmp = α * p_iso_today + (1 - α) * p_over_raw_today

        # Optional: small symmetric shrink to 0.5
        p_tmp = 0.5 + 0.9 * (p_tmp - 0.5)

    # Do sign-mismatch check BEFORE hard clamping to avoid false mismatches
    edge = ((out["predicted_total"].values + bias) - use_total).round(2)
    agree = np.sign(edge) == np.sign(p_tmp - 0.5)
    
    # CREATE ALL_GAMES DATASET FIRST (before any filtering) to ensure all games get probabilities
    all_games = out.copy()
    
    # Apply dynamic realism clamp to all games
    delta_all = (p_tmp - 0.5).clip(-clamp_range, clamp_range)
    all_games["p_over"] = 0.5 + delta_all
    all_games["p_under"] = 0.5 - delta_all
    all_games["adj_edge"] = edge
    
    # Add market depth columns for all games
    all_games["n_books"] = all_games.get("n_books", pd.Series(1, index=all_games.index)).fillna(1)
    all_games["spread_cents"] = all_games.get("spread_cents", pd.Series(0, index=all_games.index)).fillna(0)
    all_games["pass_reason"] = ""
    
    # Clip away pathologies for all games
    eps = 1e-3
    all_games["p_over"]  = np.clip(all_games["p_over"],  eps, 1.0 - eps)
    all_games["p_under"] = 1.0 - all_games["p_over"]
    
    # Handle missing odds for all games
    if "over_odds" not in all_games.columns:
        all_games = all_games.merge(today[["game_id", "over_odds", "under_odds"]], on="game_id", how="left")
    
    all_games["over_odds"]  = pd.to_numeric(all_games["over_odds"], errors="coerce")
    all_games["under_odds"] = pd.to_numeric(all_games["under_odds"], errors="coerce")
    
    # Fill missing odds with defaults (-110)
    all_games["over_odds"]  = all_games["over_odds"].fillna(-110)
    all_games["under_odds"] = all_games["under_odds"].fillna(-110)
    
    # SAVE ALL GAMES TO DATABASE FIRST (before any filtering for betting)
    save_all_probabilities_to_db(all_games, target_date, run_id, args.model_version, 
                                sigma, s, bias, engine)
    
    # Now continue with filtered games for betting recommendations
    out = out[agree].copy()
    print(f"🎯 Kept {len(out)} games after sign agreement filter (for betting)")
    print(f"📊 Saved probabilities for all {len(all_games)} games to database")

    if out.empty:
        print("❌ No games pass sign agreement filter for betting")
        return

    # THEN apply realism clamp symmetrically to betting games
    delta = (p_tmp[out.index] - 0.5).clip(-clamp_range, clamp_range)
    out["p_over"] = 0.5 + delta
    out["p_under"] = 0.5 - delta
    out["adj_edge"] = edge[out.index]  # keep aligned to priced line
    
    # Add market depth columns for betting games
    out["n_books"] = out.get("n_books", 1)
    out["spread_cents"] = out.get("spread_cents", 0)
    out["pass_reason"] = ""

    # Clip away pathologies
    eps = 1e-3
    out["p_over"]  = np.clip(out["p_over"],  eps, 1.0 - eps)
    out["p_under"] = 1.0 - out["p_over"]

    # Handle missing odds - ensure odds columns exist from original today dataframe
    if "over_odds" not in out.columns:
        # Get odds from the original today dataframe
        out = out.merge(today[["game_id", "over_odds", "under_odds"]], on="game_id", how="left")
    
    out["over_odds"]  = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    
    # BEFORE guardrails, ensure odds exist for betting subset
    filled_over  = out["over_odds"].isna().sum()
    filled_under = out["under_odds"].isna().sum()
    out["over_odds"]  = out["over_odds"].fillna(-110).astype(int)
    out["under_odds"] = out["under_odds"].fillna(-110).astype(int)
    if filled_over or filled_under:
        print(f"ℹ️ Filled default odds (-110) for {max(filled_over, filled_under)} game(s) without book odds")
    
    if out.empty:
        print("❌ No games with valid odds to analyze")
        return

    # Calculate Expected Value
    out["ev_over"] = out["p_over"] * american_win_return(out["over_odds"]) - (1 - out["p_over"])
    out["ev_under"] = out["p_under"] * american_win_return(out["under_odds"]) - (1 - out["p_under"])

    # C) Guardrails (stricter entry filter for meaningful edges only)
    print("\n🛡️  Applying betting guardrails...")
    
    # 1. Skip knife-edge lines (probabilities too close to 50%)
    mask = (out["p_over"].sub(0.5).abs() >= 0.04)  # skip knife-edges
    out = out.loc[mask].copy()
    
    if out.empty:
        print("❌ No games pass knife-edge filter")
        return
    
    # 2. Calculate Kelly fractions (fractional Kelly, not full)
    base_k_over  = kelly_fraction_robust(out["p_over"],  out["over_odds"])
    base_k_under = kelly_fraction_robust(out["p_under"], out["under_odds"])
    out["kelly_over"]  = np.clip(0.33*base_k_over,  0, 0.03)   # 1/3 Kelly, 3% cap
    out["kelly_under"] = np.clip(0.33*base_k_under, 0, 0.03)
    out["best_kelly"]  = out[["kelly_over","kelly_under"]].max(axis=1)
    
    # 3. Explicit bankroll math with automatic caps
    BANKROLL = float(os.getenv("BANKROLL", "10000"))
    out["stake"] = (out["best_kelly"] * BANKROLL).round(2)
    
    # Enforce daily cap automatically
    DAILY_CAP = 0.10 * BANKROLL
    if out["stake"].sum() > DAILY_CAP:
        scale = DAILY_CAP / out["stake"].sum()
        out["stake"] = (out["stake"] * scale).round(2)
        out["best_kelly"] = (out["best_kelly"] * scale)
        print(f"💰 Scaled stakes by {scale:.3f} to respect ${DAILY_CAP:.0f} daily cap")
    
    # Optional: per-game cap (3% of bankroll max)
    GAME_CAP = 0.03 * BANKROLL
    out["stake"] = out["stake"].clip(upper=GAME_CAP)
    
    # 4. Stricter entry filter (meaningful edges only)
    min_ev = 0.05
    out = out[(out["best_kelly"] >= 0.005) &
              (out["adj_edge"].abs() >= 0.30) &
              (out[["ev_over","ev_under"]].max(axis=1) >= min_ev)].copy()
    
    if out.empty:
        print("❌ No games meet strict entry criteria")
        return
    
    # 5. Take only the top 5 by Kelly
    out = out.sort_values("best_kelly", ascending=False).head(5)
    
    # Mark final recommendations based on better EV
    out["recommendation"] = "PASS"
    better_over = out["ev_over"] > out["ev_under"]
    out.loc[better_over, "recommendation"] = "OVER"
    out.loc[~better_over, "recommendation"] = "UNDER"

    # Add side-specific metrics for cleaner display
    out["p_side"] = out.apply(prob_for_side, axis=1)
    out["fair_odds"] = out["p_side"].apply(fair_american)
    out["best_ev"] = out[["ev_over","ev_under"]].max(axis=1)
    out["side_odds"] = np.where(out["recommendation"] == "OVER", out["over_odds"], out["under_odds"])

    # --- Price sanity checks with consensus guard ---
    print("\n🔍 Price Sanity Checks:")
    
    # Data quality assertions
    try:
        assert np.all(np.isfinite(out["side_odds"])), "Non-finite side_odds detected"
        assert np.all((out["p_side"] > 0) & (out["p_side"] < 1)), "Invalid p_side values detected"
        print("   ✅ Data quality checks passed")
    except AssertionError as e:
        print(f"   🚨 Data quality error: {e}")
        # Log problematic rows
        bad_odds = out[~np.isfinite(out["side_odds"])]
        bad_probs = out[~((out["p_side"] > 0) & (out["p_side"] < 1))]
        if len(bad_odds) > 0:
            print(f"      Bad odds in games: {bad_odds['game_id'].tolist()}")
        if len(bad_probs) > 0:
            print(f"      Bad probabilities in games: {bad_probs['game_id'].tolist()}")
    
    # Multi-book consensus guard for high EV games
    high_ev_threshold = 0.25
    extreme_ev_threshold = 0.30
    
    # Record auto-pass reason BEFORE filtering
    auto_pass = (out["best_ev"] > high_ev_threshold) & ((out["n_books"].fillna(1) < 2) | (out["spread_cents"].fillna(999) > 15))
    out.loc[auto_pass, "pass_reason"] = "high_EV_weak_market"
    out.loc[~auto_pass, "pass_reason"] = ""
    
    # Zero the Kelly for auto-pass games
    out.loc[auto_pass, ["kelly_over", "kelly_under", "best_kelly", "stake"]] = 0
    
    if auto_pass.sum() > 0:
        print(f"   🛡️  AUTO-PASS for {auto_pass.sum()} high-EV games with weak market consensus")
    
    # Remove auto-passed games from betting consideration
    out = out[~auto_pass].copy()
    
    # Flag extreme EV games for review
    extreme_ev = out["best_ev"] > extreme_ev_threshold
    outliers = out[extreme_ev].copy()
    
    if len(outliers) > 0:
        print(f"   ⚠️  {len(outliers)} games with EV > {extreme_ev_threshold:.0%} (potential market inefficiencies):")
        for _, row in outliers.iterrows():
            fair_book_diff = row["fair_odds"] - row["side_odds"]
            n_books = row.get("n_books", 1)
            spread = row.get("spread_cents", 999)
            print(f"      Game {row['game_id']}: {row['recommendation']} EV={row['best_ev']:+.3f}, "
                  f"Fair={row['fair_odds']:+d} vs Book={row['side_odds']:+d} (diff={fair_book_diff:+d}), "
                  f"Market: {n_books} books, {spread}¢ spread")
    else:
        print(f"   ✅ No extreme EV outliers (all < {extreme_ev_threshold:.0%})")
    
    # Log any negative fair odds vs positive book odds (potential data issues)
    sign_mismatches = out[
        ((out["fair_odds"] < 0) & (out["side_odds"] > 0)) |
        ((out["fair_odds"] > 0) & (out["side_odds"] < 0))
    ].copy()
    
    if len(sign_mismatches) > 0:
        print(f"   🚨 {len(sign_mismatches)} games with fair/book odds sign mismatches:")
        for _, row in sign_mismatches.iterrows():
            print(f"      Game {row['game_id']}: Fair={row['fair_odds']:+d}, Book={row['side_odds']:+d}")
    
    # Summary of EV distribution
    ev_summary = out["best_ev"].describe()
    print(f"   📊 EV Distribution: Mean={ev_summary['mean']:+.3f}, "
          f"Median={ev_summary['50%']:+.3f}, Max={ev_summary['max']:+.3f}")

    print("≥0.90 probs:", int((out["p_over"]>=0.90).sum() + (out["p_under"]>=0.90).sum()))
    if not out.empty:
        # Add book_total column for display (same as market_total for now)
        out["book_total"] = out["market_total"]
        print(out[["game_id","market_total","book_total","p_side","best_ev","best_kelly"]].head(10))

    print(f"\n✅ Scored {len(out)} games for {target_date}")
    
    if out.empty:
        print("❌ No games with valid data for betting")
        return

    # Create/update database tables (only for games that passed all filters)
    with engine.begin() as conn:
        # Create enhanced database schema
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS totals_odds (
              game_id varchar NOT NULL,
              "date" date NOT NULL,
              book text NOT NULL,
              total numeric NOT NULL,
              over_odds integer,
              under_odds integer,
              collected_at timestamp NOT NULL DEFAULT now(),
              PRIMARY KEY (game_id, "date", book, total, collected_at)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS totals_odds_idx 
            ON totals_odds (game_id, "date", total, collected_at)
        """))
        
        # Create closing odds table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS totals_odds_close (LIKE totals_odds INCLUDING ALL)
        """))
        
        # Create calibration metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS calibration_meta (
              run_id uuid PRIMARY KEY,
              game_date date NOT NULL,
              sigma double precision NOT NULL,
              temp_s double precision NOT NULL,
              bias double precision NOT NULL,
              calibration_samples integer NOT NULL,
              created_at timestamp DEFAULT now()
            )
        """))
        
        # Create calibration bins table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS calibration_bins (
              run_id uuid NOT NULL,
              game_date date NOT NULL,
              bin_low double precision NOT NULL,
              bin_high double precision NOT NULL,
              count integer NOT NULL,
              avg_pred double precision NOT NULL,
              emp_rate double precision NOT NULL,
              gap double precision NOT NULL,
              brier double precision NOT NULL,
              created_at timestamp DEFAULT now(),
              PRIMARY KEY (run_id, game_date, bin_low, bin_high)
            )
        """))
        
        # Store today's calibration parameters
        conn.execute(text("""
            INSERT INTO calibration_meta (
                run_id, game_date, sigma, temp_s, bias, calibration_samples
            ) VALUES (:rid, :date, :sig, :temp_s, :bias, :cal_samples)
        """), {
            "rid": str(run_id),
            "date": target_date,
            "sig": sigma,
            "temp_s": s,
            "bias": bias,
            "cal_samples": len(calib)
        })
        
        # Save calibration bins to database (only if we have them from isotonic path)
        if 'bins_df' in locals() and not bins_df.empty:
            conn.execute(text("""
                DELETE FROM calibration_bins WHERE run_id=:rid AND game_date=:d
            """), {"rid": str(run_id), "d": target_date})
            
            for _, r in bins_df.iterrows():
                conn.execute(text("""
                    INSERT INTO calibration_bins
                      (run_id, game_date, bin_low, bin_high, count, avg_pred, emp_rate, gap, brier)
                    VALUES
                      (:rid, :date, :lo, :hi, :cnt, :pp, :rr, :gap, :br)
                """), {
                    "rid": str(run_id), "date": target_date,
                    "lo": float(r.bin_low), "hi": float(r.bin_high), "cnt": int(r["count"]),
                    "pp": float(r.avg_pred), "rr": float(r.emp_rate), "gap": float(r.gap), "br": float(r.brier)
                })
            print(f"📋 Saved {len(bins_df)} calibration bins to database")
        
        # Create probability_predictions table with enhanced schema
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS probability_predictions (
              game_id         varchar NOT NULL,
              game_date       date NOT NULL,
              run_id          uuid NOT NULL,
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
              recommendation  text,
              model_version   text,
              sigma           double precision,
              temp_s          double precision,
              bias            double precision,
              p_side          double precision,
              fair_odds       integer,
              priced_total    numeric,
              priced_book     text,
              stake           double precision,
              n_books         integer,
              spread_cents    integer,
              pass_reason     text,
              created_at      timestamp DEFAULT now(),
              PRIMARY KEY (game_id, game_date, run_id)
            )
        """))
        
        # Add new columns if they don't exist (for existing tables)
        # Use separate transactions to avoid rollback issues
        for col_def in [
            "ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS n_books integer DEFAULT 1",
            "ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS spread_cents integer DEFAULT 0", 
            "ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS pass_reason text DEFAULT ''"
        ]:
            try:
                with engine.begin() as alt_conn:
                    alt_conn.execute(text(col_def))
            except Exception:
                pass  # Column already exists or other error
        
        # Create bet outcomes table for CLV tracking
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bet_outcomes (
              run_id uuid, 
              game_id varchar, 
              game_date date,
              side text, 
              rec_prob double precision, 
              odds integer,
              fair_odds integer, 
              clv integer, 
              won boolean,
              stake double precision, 
              pnl double precision,
              created_at timestamp DEFAULT now(),
              PRIMARY KEY (run_id, game_id)
            )
        """))
        
        # TODO: Implement outcome/CLV tracking page
        # Query to join probability_predictions with:
        # - enhanced_games.total_runs for W/L determination
        # - totals_odds_close for closing line value (CLV)
        # - Group by date for daily calibration analysis
        # Example query structure:
        """
        SELECT 
            pp.game_date,
            pp.recommendation,
            pp.p_side,
            pp.fair_odds,
            pp.stake,
            CASE WHEN pp.recommendation = 'OVER' THEN eg.total_runs > pp.market_total
                 WHEN pp.recommendation = 'UNDER' THEN eg.total_runs < pp.market_total
                 ELSE NULL END as won,
            (pp.stake * CASE WHEN won THEN american_return(pp.side_odds) ELSE -1 END) as pnl,
            (close_odds.odds - pp.side_odds) as clv
        FROM probability_predictions pp
        JOIN enhanced_games eg ON pp.game_id = eg.game_id AND pp.game_date = eg.date
        LEFT JOIN totals_odds_close close_odds ON pp.game_id = close_odds.game_id
        WHERE pp.recommendation != 'HOLD'
        ORDER BY pp.game_date DESC
        """
        
        # Clear previous predictions for this run
        conn.execute(text("""
            DELETE FROM probability_predictions 
            WHERE game_date = :d AND run_id = :rid
        """), {"d": target_date, "rid": str(run_id)})
        
        upsert_sql = text("""
            INSERT INTO probability_predictions (
                game_id, game_date, run_id,
                market_total, predicted_total,
                p_over, p_under,
                over_odds, under_odds,
                ev_over, ev_under,
                kelly_over, kelly_under,
                recommendation, model_version, created_at,
                adj_edge, sigma, temp_s, bias,
                p_side, fair_odds, priced_total, priced_book, stake,
                n_books, spread_cents, pass_reason
            ) VALUES (
                :gid, :d, :rid,
                :mt, :pt,
                :pov, :pun,
                :oo, :uo,
                :evo, :evu,
                :ko, :ku,
                :rec, :mv, NOW(),
                :ae, :sig, :temp_s, :bias,
                :p_side, :fair_odds, :priced_total, :priced_book, :stake,
                :n_books, :spread_cents, :pass_reason
            )
        """)
        
        # DISABLED: Old save operation that only saved filtered betting recommendations
        # We now save ALL games at the end of the function instead
        """
        for r in out.to_dict("records"):
            conn.execute(upsert_sql, {
                "gid": r["game_id"], "d": r["date"], "rid": str(run_id),
                "mt": r["market_total"], "pt": r["predicted_total"],
                "ae": float(r["adj_edge"]),
                "pov": float(r["p_over"]), "pun": float(r["p_under"]),
                "oo": int(r["over_odds"]), "uo": int(r["under_odds"]),
                "evo": float(r["ev_over"]), "evu": float(r["ev_under"]),
                "ko": float(r["kelly_over"]), "ku": float(r["kelly_under"]),
                "rec": r["recommendation"], "mv": args.model_version,
                "sig": sigma, "temp_s": s, "bias": bias,
                "p_side": float(r["p_side"]), "fair_odds": int(r["fair_odds"]),
                "priced_total": float(r["priced_total"]),
                "priced_book": str(r["priced_book"]),
                "stake": float(r["stake"])
            })
        """
    
    print("\n📊 Betting Recommendations:")
    print("   Game ID | Rec  | P(Side) | EV    | Kelly | Stake | Fair  | Book")
    print("   --------|------|---------|-------|-------|-------|-------|------")
    
    for _, row in out.iterrows():
        rec = row["recommendation"]
        if rec != "PASS":
            print(f"   {row['game_id']:>7} | {rec:>4} | {row['p_side']:>7.3f} | {row['best_ev']:>+5.3f} | {row['best_kelly']:>5.3f} | ${row['stake']:>5.0f} | {row['fair_odds']:>+5.0f} | {row['side_odds']:>+4.0f}")

    # Summary statistics
    total_bets = (out["recommendation"] != "PASS").sum()
    over_bets = (out["recommendation"] == "OVER").sum()
    under_bets = (out["recommendation"] == "UNDER").sum()
    
    if total_bets > 0:
        avg_ev = out["best_ev"].mean()
        avg_kelly = out["best_kelly"].mean()
        total_stake_pct = out["best_kelly"].sum()
        total_stake_dollars = out["stake"].sum()
        
        print(f"\n📈 Summary:")
        print(f"   Total bets recommended: {total_bets}")
        print(f"   OVER bets: {over_bets}")
        print(f"   UNDER bets: {under_bets}")
        print(f"   Average EV: {avg_ev:+.3f}")
        print(f"   Average Kelly: {avg_kelly:.3f}")
        print(f"   Total stake: {total_stake_pct:.3f} ({total_stake_pct:.1%})")
        print(f"   Total dollars: ${total_stake_dollars:.0f}")
        print(f"   Bankroll: ${BANKROLL:.0f}")
        print(f"   Run ID: {str(run_id)[:8]}")
        
        # Risk control warnings
        if total_stake_pct > 0.10:
            print(f"   ⚠️  WARNING: Total stake {total_stake_pct:.1%} exceeds 10% daily limit")
        if total_bets > 5:
            print(f"   ⚠️  WARNING: {total_bets} bets exceeds recommended 5-game limit")
    else:
        print(f"\n📈 Summary:")
        print(f"   No bets meet guardrail criteria")
        print(f"   Games analyzed: {len(out)}")
        print(f"   Run ID: {str(run_id)[:8]}")
    
    print(f"\n✅ ALL {len(all_games)} games saved to database with probability calculations")

if __name__ == "__main__":
    main()
