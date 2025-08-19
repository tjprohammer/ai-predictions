#!/usr/bin/env python3
"""
Predict for ALL games in a date range (including completed games),
writing predictions + edge + recommendation back into enhanced_games.

Usage:
  python predict_from_range.py --start 2025-05-01 --end 2025-08-15 --thr 1.0

Notes:
- Uses the same EnhancedBullpenPredictor pipeline as daily serving.
- No inserts: UPDATEs only, so requires rows already seeded in enhanced_games.
"""

import os, sys, argparse
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def safe_align(df, feature_columns, fill_values=None):
    X = df.reindex(columns=feature_columns)
    if isinstance(fill_values, dict):
        for c, v in fill_values.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
    med = X.median(numeric_only=True)
    return X.fillna(med).fillna(0.0)

def predict_for_day(engine, ds, thr):
    # Load LGF for the day (NO filter on total_runs)
    q = text("""SELECT * FROM legitimate_game_features WHERE "date" = :d ORDER BY game_id""")
    base = pd.read_sql(q, engine, params={"d": ds})
    if base.empty:
        print(f"{ds}: no LGF rows; skipping")
        return 0, 0

    # Bring market totals (helps expected_total / edge downstream)
    mk = pd.read_sql(text("""SELECT game_id, market_total FROM enhanced_games WHERE "date" = :d"""),
                     engine, params={"d": ds})
    if "market_total_final" in base.columns:
        base["market_total"] = base.pop("market_total_final")
    
    # Clean merge to avoid duplicate columns
    base = base.merge(mk, on="game_id", how="left", suffixes=('', '_from_enhanced'))
    if "market_total_from_enhanced" in base.columns:
        # Use the enhanced_games version as it's more reliable
        base["market_total"] = base["market_total_from_enhanced"]
        base = base.drop("market_total_from_enhanced", axis=1)

    # ✅ Ensure the model truly sees the bookmaker line
    if "market_total" in base.columns:
        needs_fix = (
            "expected_total" not in base.columns
            or base["expected_total"].isna().all()
            or base["expected_total"].nunique(dropna=True) <= 1  # constant/flat
        )
        if needs_fix:
            base["expected_total"] = base["market_total"]
            print(f"🔧 expected_total set from market_total for {len(base)} games on {ds}")

    # Use the full prediction pipeline with rolling stats integration
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    P = EnhancedBullpenPredictor()

    # Use the same method as daily predictions (includes rolling stats)
    try:
        predictions = P.predict_today_games(ds)
        if not predictions:
            print(f"{ds}: no predictions generated; skipping")
            return 0, 0
        
        # Extract predictions and game IDs
        yhat = [float(p['predicted_total']) for p in predictions]
        game_ids = [int(p['game_id']) for p in predictions]
        
        # Create DataFrame for batch update
        pred_df = pd.DataFrame({
            'game_id': game_ids,
            'predicted_total': yhat
        })
        
    except Exception as e:
        print(f"{ds}: predict_today_games failed → using fallback method ({e})")
        # Fallback to old method without rolling stats
        feats = P.engineer_features(base)
        try:
            X = P.align_serving_features(feats, strict=False)
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=P.feature_columns)
        except Exception as e2:
            print(f"{ds}: align_serving_features failed → final fallback ({e2})")
            X = safe_align(feats, P.feature_columns, getattr(P, "fill_values", None))

        # Transform + predict
        M = X
        if getattr(P, "preproc", None) is not None:
            M = P.preproc.transform(M)
        elif getattr(P, "scaler", None) is not None:
            sc = P.scaler
            if not hasattr(sc, "n_features_in_") or sc.n_features_in_ == M.shape[1]:
                M = sc.transform(M)

        yhat = P.model.predict(M).astype(float)

        # apply holdout bias from the trained bundle if present
        bias = float(getattr(P, "bias_correction", 0.0) or 0.0)
        if bias:
            print(f"🔧 Applying bias correction: {bias:+.3f} runs")
            yhat = yhat + bias
        
        # Create DataFrame for batch update
        pred_df = pd.DataFrame({
            'game_id': base['game_id'].values,
            'predicted_total': yhat
        })

    # Ensure game_id data types match
    pred_df['game_id'] = pred_df['game_id'].astype(str)
    base_copy = base[["game_id","date"]].copy()
    base_copy['game_id'] = base_copy['game_id'].astype(str)
    
    # Merge back with base data for edge calculation
    out = base_copy.merge(pred_df, on="game_id", how="left")
    
    # edge / recommendation if market present
    if "market_total" in base.columns:
        market_data = base[["game_id","market_total"]].copy()
        market_data['game_id'] = market_data['game_id'].astype(str)
        out = out.merge(market_data, on="game_id", how="left")
        out["edge"] = (out["predicted_total"] - out["market_total"]).round(2)
        def rec(predicted, market):
            if pd.isna(predicted) or pd.isna(market): return None
            diff = predicted - market
            if diff >=  thr:  return "OVER"
            if diff <= -thr:  return "UNDER"
            return "HOLD"
        out["recommendation"] = out.apply(lambda row: rec(row["predicted_total"], row["market_total"]), axis=1)
    else:
        out["edge"] = None
        out["recommendation"] = None

    # Write back
    upd_sql = text("""
        UPDATE enhanced_games
           SET predicted_total = :pred,
               edge            = :edge,
               recommendation  = :rec
         WHERE game_id = :gid AND "date" = :d
    """)
    updated = 0
    with engine.begin() as conn:
        for r in out.to_dict(orient="records"):
            params = {
                "pred": r["predicted_total"],
                "edge": r["edge"],
                "rec":  r["recommendation"],
                "gid":  r["game_id"],
                "d":    r["date"]
            }
            updated += conn.execute(upd_sql, params).rowcount

    print(f"{ds}: predicted {len(out)} | updated {updated}")
    return len(out), updated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)  # YYYY-MM-DD
    ap.add_argument("--end",   required=True)
    ap.add_argument("--thr", type=float, default=1.0, help="edge threshold for OVER/UNDER labels")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    eng = get_engine()
    tot_pred = tot_upd = 0
    for d in daterange(start, end):
        p, u = predict_for_day(eng, d.strftime("%Y-%m-%d"), args.thr)
        tot_pred += p; tot_upd += u
    print(f"Done. predictions={tot_pred} | rows_updated={tot_upd}")

if __name__ == "__main__":
    main()
