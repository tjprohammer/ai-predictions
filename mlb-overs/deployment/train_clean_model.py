#!/usr/bin/env python3

# Windows-safe Unicode handling for emoji printing
import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os, json, joblib, argparse, numpy as np, pandas as pd
import re
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

DB_URL = os.getenv("DATABASE_URL","postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
MODELS_DIR = Path("../models")

# Outcome/label leakage patterns (tightened)
OUTCOME_PATTERNS = [
    r"(^|_)total_runs($|_)",
    r"(^|_)home_score($|_)", r"(^|_)away_score($|_)",
    r"(^|_)final(_|$)", r"(^|_)result(_|$)",
    r"(^|_)extra_innings($|_)", r"(^|_)innings_played($|_)", r"(^|_)completed_innings($|_)",
    r"(^|_)winner($|_)", r"(^|_)loser($|_)", r"(^|_)outcome($|_)",
]

# Market/serving leakage patterns
LEAK_MARKET_PATTERNS = [
    r"market_total", r"expected_total", r"over_odds", r"under_odds", 
    r"implied_total", r"predicted_total"
]

# Safe allow-list substrings (never treat as leakage)
SAFE_FEATURE_HINTS = [
    "bullpen", "relief", "prev", "previous", "last_", "_l7", "_l14", "_l30"
]

LEAKY_REGEXES = [
    r"(^|_)(market_total|expected_total|over_odds|under_odds)(_|$)"
]

DROP_EXPLICIT = {"market_total","expected_total","over_odds","under_odds"}

# Raw columns to drop before feature engineering
RAW_LEAKY_COLS = {
    "total_runs","home_score","away_score","final_score",
    "winner","loser","result","outcome","late_game_scoring_factor",
    "label_total"  # Always drop the label column from features
}

# Leak detection patterns
LEAK_NAME_PATTERNS = [
    "market_total","expected_total","over_odds","under_odds","implied_total",
    "predicted_total","home_score","away_score","final_score","total_runs",
    "winner","loser","result","outcome","late_game","extra_innings",
    "innings_played","completed_innings"
]
SAFE_HINTS = ["bullpen","relief","prev","previous","last_","_l7","_l14","_l30"]

def check_for_leakage(features, target):
    """Detect outcome and market leakage in features"""
    outcome_leaks = []
    market_leaks = []
    
    for col in features.columns:
        low = col.lower()
        
        # Skip obvious safe pregame workload features
        if any(s in low for s in SAFE_FEATURE_HINTS):
            continue
        
        # Check for outcome leakage
        if any(re.search(p, col, re.I) for p in OUTCOME_PATTERNS):
            outcome_leaks.append(col)
            continue
        
        # Check for market leakage
        if any(p in low for p in LEAK_MARKET_PATTERNS):
            market_leaks.append(col)
    
    if outcome_leaks:
        print(f"ERROR: Outcome leakage detected in {len(outcome_leaks)} columns:")
        for col in sorted(outcome_leaks):
            print(f"  - {col}")
        
        # Only compute correlations on numeric intersection
        num_cols = [c for c in outcome_leaks if pd.api.types.is_numeric_dtype(features[c])]
        if num_cols:
            correlations = features[num_cols].corrwith(target).abs().sort_values(ascending=False)
            print("\nCorrelations with target:")
            for col, corr in correlations.head().items():
                print(f"  {col}: {corr:.3f}")
        
        raise ValueError("Outcome leakage found - these columns must be removed")
    
    if market_leaks:
        print(f"WARNING: Market leakage detected in {len(market_leaks)} columns:")
        for col in sorted(market_leaks):
            print(f"  - {col}")
    
    return outcome_leaks, market_leaks

def add_historical_pitcher_stats(feat_df, engine):
    """
    Hydrate SP rolling stats by *game date* using an as-of join.
    Auto-detects date/id columns and gracefully degrades when fields are missing.
    """
    cand_tables = [
        "pitcher_daily_rolling",
        "pitcher_rolling_stats_materialized",
        "pitcher_rolling_stats_mv",
        "pitcher_rolling_stats",
    ]

    # pick an available table
    tbl = None
    with engine.begin() as conn:
        for t in cand_tables:
            try:
                if conn.execute(text(f"SELECT 1 FROM {t} LIMIT 1")).fetchone():
                    tbl = t
                    break
            except Exception:
                pass
    if not tbl:
        print("⚠️  No pitcher rolling table found; leaving SP stats as-is.")
        return feat_df

    # introspect columns
    with engine.begin() as conn:
        cols = {r[0] for r in conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
            {"t": tbl}
        )}

    # detect id / date columns with more options
    date_options = ["stat_date", "game_date", "date"]
    pid_options = ["pitcher_id", "pitcher_name", "player_id", "name"]
    
    date_col = next((c for c in date_options if c in cols), None)
    pid_col = next((c for c in pid_options if c in cols), None)
    
    if not date_col or not pid_col:
        print(f"⚠️  Missing date ({date_options}) or pitcher id/name ({pid_options}) on {tbl}; skipping hydration.")
        print(f"    Available columns: {sorted(cols)}")
        return feat_df

    # map canonical -> actual column names via synonyms
    synonyms = {
        "era":      ["era"],
        "whip":     ["whip"],
        "k_per_9":  ["k_per_9", "k9", "k_9", "k_per_nine"],
        "bb_per_9": ["bb_per_9", "bb9", "bb_9", "bb_per_nine"],
        "starts":   ["starts", "gs", "games_started", "start_count"],
    }
    colmap = {}
    for canon, opts in synonyms.items():
        colmap[canon] = next((c for c in opts if c in cols), None)

    # build SELECT with only available fields
    select_parts = [f"{pid_col} AS pid", f"{date_col}::date AS asof_date"]
    for canon, actual in colmap.items():
        if actual:
            select_parts.append(f"{actual} AS {canon}")

    print(f"📊 Using pitcher stats table: {tbl}")
    print(f"🔍 Schema detected: date_col='{date_col}', pid_col='{pid_col}', fields={ [p.split(' AS ')[1] for p in select_parts[2:]] }")

    q = text(f"""
        SELECT {", ".join(select_parts)}
        FROM {tbl}
        WHERE {date_col}::date <= :max_d
        ORDER BY pid, asof_date
    """)

    max_d = pd.to_datetime(feat_df["date"]).max().date()
    roll = pd.read_sql(q, engine, params={"max_d": str(max_d)})

    # need ids & dates on feature frame (fallback to names if ids missing)
    need_any = ({"home_sp_id","away_sp_id"} & set(feat_df.columns)) \
               or ({"home_sp_name","away_sp_name"} & set(feat_df.columns))
    if not need_any or "date" not in feat_df.columns:
        print("⚠️  Missing SP id/name or date; skipping hydration.")
        return feat_df

    f = feat_df.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.date
    roll["asof_date"] = pd.to_datetime(roll["asof_date"]).dt.date

    def get_pid_series(side):
        if "home_sp_id" in f.columns and "away_sp_id" in f.columns and pid_col == "pitcher_id":
            return f[f"{side}_sp_id"]
        # fallback to names if available and table uses names
        if "home_sp_name" in f.columns and "away_sp_name" in f.columns and pid_col == "pitcher_name":
            return f[f"{side}_sp_name"]
        # last resort: return a NA series to skip merge
        return pd.Series([pd.NA]*len(f))

    # asof-join per side
    for side in ["home","away"]:
        pid_series = get_pid_series(side)
        if pid_series.isna().all():
            continue
        left = pd.DataFrame({
            "pid": pid_series, 
            "date": pd.to_datetime(f["date"])
        }).sort_values("date")
        right = roll.sort_values("asof_date").rename(columns={"asof_date":"date"})
        right["date"] = pd.to_datetime(right["date"])  # Ensure datetime type
        tmp = pd.merge_asof(left, right, by="pid", on="date", direction="backward")

        # fill available canonical metrics
        mapping = [
            ("era",      f"{side}_sp_era"),
            ("whip",     f"{side}_sp_whip"),
            ("k_per_9",  f"{side}_sp_k_per_9"),
            ("bb_per_9", f"{side}_sp_bb_per_9"),
            ("starts",   f"{side}_sp_starts"),
        ]
        for src, dst in mapping:
            if src in tmp.columns:
                if dst in f.columns:
                    f[dst] = pd.to_numeric(f[dst], errors="coerce")
                    f[dst] = f[dst].fillna(pd.to_numeric(tmp[src], errors="coerce"))
                else:
                    f[dst] = pd.to_numeric(tmp[src], errors="coerce")

    # priors for any remaining gaps
    priors = {
        "home_sp_era": 4.30, "away_sp_era": 4.30,
        "home_sp_whip": 1.30, "away_sp_whip": 1.30,
        "home_sp_k_per_9": 8.3, "away_sp_k_per_9": 8.3,
        "home_sp_bb_per_9": 3.1, "away_sp_bb_per_9": 3.1,
        "home_sp_starts": 10.0, "away_sp_starts": 10.0,
    }
    for c, v in priors.items():
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce").fillna(v)

    cov_home = f.get("home_sp_era", pd.Series(dtype=float)).notna().mean() if "home_sp_era" in f else 0.0
    cov_away = f.get("away_sp_era", pd.Series(dtype=float)).notna().mean() if "away_sp_era" in f else 0.0
    print(f"✅ SP hydration coverage – home ERA: {cov_home:.1%}, away ERA: {cov_away:.1%}")

    return f

def fetch_training_rows(engine, end_date, window_days=150, min_total=5, max_total=13):
    start_date = (datetime.strptime(end_date,"%Y-%m-%d") - timedelta(days=window_days-1)).strftime("%Y-%m-%d")
    q = text("""
      SELECT
        lgf.*,
        eg.market_total,
        eg.home_sp_id, eg.away_sp_id,
        eg.home_sp_name, eg.away_sp_name,
        COALESCE(
          lgf.total_runs,
          CASE WHEN eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL
               THEN eg.home_score + eg.away_score END
        ) AS label_total
      FROM legitimate_game_features lgf
      LEFT JOIN enhanced_games eg
        ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
      WHERE lgf."date" BETWEEN :start AND :end
        AND COALESCE(lgf.total_runs, (eg.home_score + eg.away_score)) IS NOT NULL
        AND (eg.market_total IS NULL OR (eg.market_total BETWEEN :lo AND :hi))
    """)
    return pd.read_sql(q, engine, params={"start":start_date,"end":end_date,"lo":min_total,"hi":max_total})

def engineer_features(df):
    # use your deployed feature builder but *don't* rely on market features
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    pred = EnhancedBullpenPredictor()
    feat = pred.engineer_features(df)
    
    # drop name-based leaks
    to_drop = []
    for c in list(feat.columns):
        low = c.lower()
        if any(p in low for p in LEAK_NAME_PATTERNS) and not any(h in low for h in SAFE_HINTS):
            to_drop.append(c)
    if to_drop:
        print(f"� Dropping {len(to_drop)} name-based leak columns: {to_drop[:8]}{'...' if len(to_drop)>8 else ''}")
        feat = feat.drop(columns=to_drop, errors="ignore")
    return feat

def auto_leak_scrub(X, y, corr=0.98, tol=0.10, equal_frac=0.30):
    """Remove features that behave suspiciously like the label"""
    bad = []
    # numeric only
    Xn = X.select_dtypes(include=[np.number])
    
    # 1) high absolute correlation with label
    corr_s = Xn.corrwith(y).abs().sort_values(ascending=False)
    bad += corr_s[corr_s >= corr].index.tolist()
    
    # 2) near-identity to label
    diff = (Xn.subtract(y, axis=0)).abs()
    near = diff.median() < tol
    bad += near[near].index.tolist()
    
    # 3) rounded equality on a big fraction of rows
    y3 = y.round(3)
    for c in Xn.columns:
        if pd.api.types.is_numeric_dtype(Xn[c]):
            same = (Xn[c].round(3) == y3).mean()
            if same >= equal_frac:
                bad.append(c)
    
    bad = sorted(set(bad))
    if bad:
        print(f"🚫 Auto-scrubbing {len(bad)} leakage-like features:", bad[:12], "..." if len(bad)>12 else "")
        Xn = Xn.drop(columns=bad, errors="ignore")
        # keep original non-numeric columns if any
        X = pd.concat([Xn, X.drop(columns=Xn.columns, errors="ignore")], axis=1)
    return X, bad

def drop_leaky(X):
    keep = X.columns.copy()
    leaky_patterns = ["market_total", "expected_total", "over_odds", "under_odds", "implied_total", 
                     "predicted_total", "home_score", "away_score", "total_runs", "final_score",
                     "winner", "result", "outcome", "actual_"]
    for col in list(keep):
        low = col.lower()
        if any(p in low for p in leaky_patterns):
            keep = keep.drop(col)
    return X[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (last day included)")
    ap.add_argument("--window-days", type=int, default=150)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    eng = create_engine(DB_URL, pool_pre_ping=True)
    raw = fetch_training_rows(eng, args.end, args.window_days)
    if raw.empty:
        raise SystemExit("no training rows returned")

    # label
    y = pd.to_numeric(raw["label_total"], errors="coerce")
    ok = y.notna()
    raw = raw.loc[ok].copy()
    y = y.loc[ok].astype(float)

    # unify date column if needed
    if "date" not in raw.columns and "game_date" in raw.columns:
        raw = raw.rename(columns={"game_date": "date"})

    # nuke post-game cols early
    raw_leaky_found = [c for c in RAW_LEAKY_COLS if c in raw.columns]
    if raw_leaky_found:
        print(f"� Dropping {len(raw_leaky_found)} raw leaky columns: {raw_leaky_found}")
        raw = raw.drop(columns=raw_leaky_found, errors="ignore")

    print("🏃‍♂️ Hydrating historical pitcher rolling stats (pre-engineering)...")
    raw = add_historical_pitcher_stats(raw, eng)

    # now engineer features on hydrated raw
    feat = engineer_features(raw)
    
    # numeric selection
    X = feat.select_dtypes(include=[np.number]).copy()
    
    # scrub name-based leaks again on X
    name_leak_cols = [c for c in X.columns
                      if any(p in c.lower() for p in LEAK_NAME_PATTERNS) and not any(h in c.lower() for h in SAFE_HINTS)]
    if name_leak_cols:
        print(f"🚫 Dropping {len(name_leak_cols)} additional name-based leak columns: {name_leak_cols[:8]}{'...' if len(name_leak_cols)>8 else ''}")
        X = X.drop(columns=name_leak_cols, errors="ignore")

    # *** automatic leak scrub against y ***
    X, auto_bad = auto_leak_scrub(X, y, corr=0.98, tol=0.10, equal_frac=0.30)
    
    # Extra safety: ensure label_total never makes it through
    if "label_total" in X.columns:
        print("🚨 Removing label_total from features (emergency catch)")
        X = X.drop(columns=["label_total"])
    
    # Comprehensive leakage detection (as backup)
    print("🔍 Final leakage check...")
    outcome_leaks, market_leaks = check_for_leakage(X, y)
    
    # Clean up infinities and extreme values
    X = X.replace([np.inf, -np.inf], np.nan).clip(-1000, 1000)

    # fill values
    fill_values = X.median(numeric_only=True).to_dict()
    X = X.fillna(fill_values)
    
    # Hard check for non-finite values
    if not np.isfinite(X.values).all():
        raise ValueError("Non-finite values remain after cleaning.")
    
    print(f"Training features: {X.shape[1]} features, {len(X)} rows")
    print(f"Features after cleaning: {X.isnull().sum().sum()} NaN values, {np.isinf(X.values).sum()} inf values")

    # chronological split (last N days holdout)
    cutoff = (pd.to_datetime(args.end) - pd.Timedelta(days=args.holdout_days-1)).date()
    raw_dates = pd.to_datetime(raw["date"]).dt.date
    train_idx = raw_dates < cutoff
    test_idx  = raw_dates >= cutoff

    # Keep evaluation sidecar from raw data BEFORE feature cleaning (for market comparison)
    eval_cols = ["game_id", "date"]
    if "market_total" in raw.columns:
        eval_cols.append("market_total")
    if "expected_total" in raw.columns:
        eval_cols.append("expected_total")
    
    # Build eval_te from raw data, not cleaned features
    eval_te = raw.loc[test_idx, eval_cols].copy().reset_index(drop=True)
    print(f"📋 Built evaluation sidecar: {len(eval_te)} rows with columns {eval_cols}")

    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    ytr, yte = y.loc[train_idx], y.loc[test_idx]

    # model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ).fit(Xtr, ytr)

    pred_tr = rf.predict(Xtr)
    pred_te = rf.predict(Xte)

    mae_tr = float(mean_absolute_error(ytr, pred_tr))
    mae_te = float(mean_absolute_error(yte, pred_te))
    print(f"MAE train={mae_tr:.3f}  holdout={mae_te:.3f}  n_tr={len(ytr)} n_te={len(yte)}")
    
    # Enhanced MAE safety check
    if mae_te < 1.0:
        print(f"❌ Holdout MAE {mae_te:.3f} is still too low after scrubbing — dumping top features for inspection.")
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
        print("Top 20 feature importances:")
        print(fi.to_string())
        raise SystemExit(1)
    
    # 2. Predictions should be reasonable (6-12 runs typical)
    pred_mean = pred_te.mean()
    pred_std = pred_te.std()
    if pred_mean < 6.0 or pred_mean > 12.0:
        print(f"⚠️  Unusual prediction mean: {pred_mean:.2f} runs")
    if pred_std < 0.5 or pred_std > 3.0:
        print(f"⚠️  Unusual prediction spread: {pred_std:.2f} runs")
    
    # 3. Feature importance should make sense
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top features should be pitcher/team/park related, not market
    top_features = feature_importance.head(10)['feature'].tolist()
    pitcher_features = [f for f in top_features if any(x in f.lower() for x in ['era', 'whip', 'sp_', 'pitcher'])]
    
    if len(pitcher_features) < 3:
        print(f"⚠️  Few pitcher features in top 10: {pitcher_features}")
    else:
        print(f"✅ Pitcher features prominent: {len(pitcher_features)}/10 in top features")
    
    print(f"✅ Safety checks passed - MAE: {mae_te:.3f}, pred_mean: {pred_mean:.2f}")

    # compare to market baseline if available (use evaluation sidecar)
    # Defensive fix for eval_te structure
    if not isinstance(eval_te, pd.DataFrame):
        eval_te = pd.DataFrame(eval_te)
    
    def get_numeric_series(df, col, default=np.nan):
        """Safely extract a numeric Series from a DataFrame column"""
        if col in df.columns:
            s = df[col]
            if not isinstance(s, (pd.Series, list, tuple, np.ndarray)):
                # scalar -> expand to Series
                s = pd.Series([s] * len(df), index=df.index)
        else:
            # missing column -> fill with NaN
            s = pd.Series(default, index=df.index)
        return pd.to_numeric(s, errors="coerce")
    
    # Add sanity checks before evaluation
    assert len(eval_te) > 0, "Empty holdout set"
    print(f"📊 Evaluation sidecar: {len(eval_te)} rows, columns: {list(eval_te.columns)}")
    
    if "market_total" in eval_te.columns:
        try:
            mkt_te = get_numeric_series(eval_te, "market_total")
            mask = mkt_te.notna()
            if mask.any():
                # Align with actual labels (both should be same length)
                yte_aligned = yte.reset_index(drop=True)
                mkt_aligned = mkt_te.reset_index(drop=True)
                mkt_mae = float(mean_absolute_error(yte_aligned[mask], mkt_aligned[mask]))
                print(f"market MAE (holdout) = {mkt_mae:.3f}  (model - market = {mae_te - mkt_mae:+.3f})")
                
                # Prediction vs market spread analysis
                pred_te_aligned = pd.Series(pred_te).reset_index(drop=True)
                spread = np.abs(pred_te_aligned[mask] - mkt_aligned[mask])
                spread_desc = spread.describe()
                print(f"🔎 Holdout | abs(pred - market) | mean={spread_desc['mean']:.2f}, std={spread_desc['std']:.2f}, max={spread_desc['max']:.2f}")
            else:
                print("⚠️  No valid market_total values for holdout; skipping market comparison.")
        except Exception as e:
            print(f"⚠️  Error processing market_total: {e}")
            print("⚠️  Skipping market comparison.")
    else:
        print("⚠️  market_total column not available; skipping market comparison.")

    # permutation importance (sanity)
    r = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=0, n_jobs=-1)
    perm = pd.DataFrame({"feature": Xte.columns, "perm_importance": r.importances_mean}) \
            .sort_values("perm_importance", ascending=False)
    print("top perm-importance:")
    print(perm.head(15).to_string(index=False))

    # bundle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": rf,
        "model_type": "legitimate_random_forest_clean",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "training_period": {"end": args.end, "window_days": args.window_days, "holdout_days": args.holdout_days},
        "feature_columns": list(X.columns),
        "feature_fill_values": fill_values,
        "label_definition": "final total runs (home+away), pre-game features only",
        "evaluation_metrics": {"mae_train": mae_tr, "mae_holdout": mae_te},
        "training_feature_snapshot": X.sample(min(5000, len(X)), random_state=1),  # for drift checks
        "bias_correction": 0.0,
    }
    out = MODELS_DIR / "legitimate_model_latest.joblib"
    joblib.dump(bundle, out)
    print(f"✅ saved: {out.resolve()}")

if __name__ == "__main__":
    main()
