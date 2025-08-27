#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import os

# Reuse your feature/prep + predict code from the pipelines directory
from pipelines.ultra_sharp_pipeline import (
    _prep_features, _is_closed_row, _predict_with_bundle, _harmonize_serve_columns
)

def _pick_bundle(row, model_dir: str) -> Path:
    md = Path(model_dir)
    if _is_closed_row(row):
        p = md / "ultra_bundle_closed.joblib"
        if p.exists(): return p
    else:
        p = md / "ultra_bundle_open.joblib"
        if p.exists(): return p
    return md / "ultra_bundle_all.joblib"

def _jitter_predict(df, bundle_path: Path, trials=0, jitter=0.05, seed=42):
    """Return base preds and (optional) sign-agreement stability with feature jitter."""
    base = _predict_with_bundle(bundle_path, df)
    if trials <= 0:
        return base, np.ones(len(base))
    rng = np.random.default_rng(seed)
    signs = []
    for _ in range(trials):
        noisy = df.copy()
        for col in ["wind_out_cf","air_density_proxy","altitude_ft",
                    "home_bp_fatigue","away_bp_fatigue",
                    "home_arms_avail","away_arms_avail"]:
            if col in noisy.columns:
                x = pd.to_numeric(noisy[col], errors="coerce")
                noisy[col] = x * (1 + rng.uniform(-jitter, jitter, len(noisy)))
        p = _predict_with_bundle(bundle_path, noisy)
        signs.append(np.sign(p))
    signs = np.stack(signs, axis=1)  # [n, trials]
    agree = (np.sign(base).reshape(-1,1) == signs).mean(axis=1)
    return base, agree

def main():
    ap = argparse.ArgumentParser(description="Backtest ultra residual model vs market")
    ap.add_argument("--db", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--min_edge", type=float, default=0.8, help="abs(residual) threshold to simulate bets")
    ap.add_argument("--price", type=int, default=-110, help="American price used for ROI sim")
    ap.add_argument("--stability_trials", type=int, default=0, help="0 = off; else number of jitter trials")
    ap.add_argument("--stability_min", type=float, default=0.80, help="min sign-agreement to keep a bet")
    args = ap.parse_args()

    # Target guard (for when switching to STEAM or OPENING residual later)
    target = os.getenv("ULTRA_TARGET", "closing_resid")
    print(f"Eval target: {target}")

    # Check for in-sample leakage
    import joblib
    from pathlib import Path
    
    reg = joblib.load(Path(args.model_dir) / "ultra_registry.joblib")
    tw = reg.get("train_window")
    if tw:
        bt_start = pd.to_datetime(args.start)
        bt_end   = pd.to_datetime(args.end)
        tr_start = pd.to_datetime(tw["start"])
        tr_end   = pd.to_datetime(tw["end"])
        if not (bt_end < tr_start or bt_start > tr_end):
            raise SystemExit(f"Backtest window [{bt_start.date()}..{bt_end.date()}] overlaps training window "
                             f"[{tr_start.date()}..{tr_end.date()}]. Retrain with an earlier --end.")

    eng = create_engine(args.db)
    q = """
      SELECT
        gc.*,
        eg.total_runs,
        eg.market_total,
        eg.date,
        -- tiny pregame whitelist from EG:
        eg.plate_umpire,
        eg.plate_umpire_bb_pct,
        eg.plate_umpire_strike_zone_consistency,
        eg.plate_umpire_rpg,
        eg.roof_type,
        eg.roof_status,
        -- pregame as-of features (CRITICAL for parity):
        pf.home_sp_era_l3_asof,
        pf.away_sp_era_l3_asof,
        pf.home_sp_whip_l3_asof,
        pf.away_sp_whip_l3_asof,
        pf.home_bp_ip_3d_asof,
        pf.away_bp_ip_3d_asof,
        pf.home_bp_era_30d_asof,
        pf.away_bp_era_30d_asof,
        pf.home_runs_pg_14_asof,
        pf.away_runs_pg_14_asof
      FROM game_conditions gc
      JOIN enhanced_games eg ON eg.game_id::text = gc.game_id::text
      LEFT JOIN pregame_features_v1 pf ON pf.game_id = eg.game_id
      WHERE eg.date BETWEEN :s AND :e
        AND eg.market_total IS NOT NULL
        AND eg.total_runs IS NOT NULL
    """
    df = pd.read_sql(text(q), eng, params={"s": args.start, "e": args.end})
    df = _harmonize_serve_columns(df)
    
    # Fix roof bucketing alias
    if "roof_status" in df.columns and "roof_state" not in df.columns:
        df["roof_state"] = df["roof_status"]
    
    # ---- Deduplicate: keep latest snapshot per game_id if duplicates exist ----
    if "game_id" in df.columns:
        dupes = df["game_id"].duplicated(keep=False).sum()
        if dupes:
            print(f"WARNING: {dupes} duplicated rows across game_id — keeping latest per game.")
            sort_keys = [c for c in ["game_time_utc","created_at","date"] if c in df.columns]
            if sort_keys:
                df = df.sort_values(["game_id"] + sort_keys).groupby("game_id", as_index=False).tail(1)
            else:
                df = df.drop_duplicates(subset=["game_id"], keep="last")
    
    # Feature parity check for as-of features
    asof_cols = [
        "home_sp_era_l3_asof","away_sp_era_l3_asof",
        "home_sp_whip_l3_asof","away_sp_whip_l3_asof",
        "home_bp_ip_3d_asof","away_bp_ip_3d_asof",
        "home_bp_era_30d_asof","away_bp_era_30d_asof",
        "home_runs_pg_14_asof","away_runs_pg_14_asof",
    ]
    present = [c for c in asof_cols if c in df.columns]
    print(f"As-of features present: {len(present)}/{len(asof_cols)} → {present[:6]}{'...' if len(present)>6 else ''}")
    
    if len(present) < len(asof_cols) * 0.7:
        print(f"⚠️  WARNING: Only {len(present)}/{len(asof_cols)} as-of features present!")
    
    # Check for postgame snapshot leakage
    if {"created_at","game_time_utc"} <= set(df.columns):
        try:
            bad = (pd.to_datetime(df["created_at"], errors="coerce") >
                   pd.to_datetime(df["game_time_utc"], errors="coerce")).mean()
            if bad > 0.05:
                print(f"WARNING: {bad:.1%} rows look like postgame snapshots → high leakage risk.")
        except:
            pass
        
    if df.empty:
        print("No finished games with market_total in range.")
        return

    parts = []
    # bucket by roof state
    buckets = df.apply(lambda r: "closed" if _is_closed_row(r) else "open", axis=1)
    for bucket, g in df.groupby(buckets):
        # Inspect feature coverage vs bundle before predicting
        bundle_path = _pick_bundle(g.iloc[0], args.model_dir)
        import joblib
        b = joblib.load(bundle_path)
        served = _prep_features(g).columns.tolist()
        missing = [c for c in b["feature_cols"] if c not in served]
        miss_frac = len(missing) / max(1, len(b["feature_cols"]))
        print(f"[{bucket}] serving {len(b['feature_cols'])-len(missing)}/{len(b['feature_cols'])} features "
              f"(missing {miss_frac:.0%}).")
        if miss_frac > 0.30:
            print(f"⚠️  [{bucket}] >30% of trained features missing → expect flatter preds.")
        bundle = _pick_bundle(g.iloc[0], args.model_dir)
        resid_hat, agree = _jitter_predict(g, bundle, trials=args.stability_trials)
        parts.append(pd.DataFrame({
            "game_id": g["game_id"].astype(str),
            "date": pd.to_datetime(g["date"]).dt.date.astype(str),
            "bucket": bucket,
            "market_total": pd.to_numeric(g["market_total"], errors="coerce"),
            "total_runs": pd.to_numeric(g["total_runs"], errors="coerce"),
            "resid_hat": resid_hat.astype(float),
            "stability": agree.astype(float),
        }))
    out = pd.concat(parts, ignore_index=True)
    out["pred_total"] = out["market_total"] + out["resid_hat"]
    out["true_resid"] = out["total_runs"] - out["market_total"]
    out["edge"] = out["resid_hat"]
    out["side"] = np.where(out["edge"] > 0, "OVER", "UNDER")
    out["result"] = np.where(out["total_runs"] > out["market_total"], "OVER",
                      np.where(out["total_runs"] < out["market_total"], "UNDER", "PUSH"))

    # --- Accuracy vs market ---
    mae_resid = np.mean(np.abs(out["resid_hat"] - out["true_resid"]))
    mae_total = np.mean(np.abs(out["pred_total"] - out["total_runs"]))
    mae_market = np.mean(np.abs(out["market_total"] - out["total_runs"]))

    non_push = out[out["result"] != "PUSH"]
    dir_acc = (non_push["side"] == non_push["result"]).mean() if len(non_push) else np.nan

    print("\n=== BACKTEST ACCURACY ===")
    print(f"Games: {len(out)}  (non-push: {len(non_push)})")
    print(f"MAE (model total):  {mae_total:.3f}")
    print(f"MAE (market total): {mae_market:.3f}")
    print(f"MAE (residual):     {mae_resid:.3f}")
    print(f"Direction accuracy: {dir_acc*100:.1f}%")

    # ---- Signal diagnostics ----
    print("\nBACKTEST ANALYSIS:")
    print(f"σ(resid_hat) = {out['resid_hat'].std():.3f}   σ(true_resid) = {out['true_resid'].std():.3f}")
    if out['resid_hat'].nunique() < 5:
        print(f"Unique resid_hat values: {out['resid_hat'].nunique()} → likely underpowered features on serve.")
    print(f"corr(resid_hat, true_resid) = {out[['resid_hat','true_resid']].corr().iloc[0,1]:.3f}")
    print("By bucket std:\n", out.groupby('bucket')['resid_hat'].std())

    # --- Simple ROI sim (flat 1u, -110) with selectivity + stability ---
    bets = out[out["edge"].abs() >= args.min_edge].copy()
    if args.stability_trials > 0:
        bets = bets[bets["stability"] >= args.stability_min]
    if bets.empty:
        print("\nNo bets passed filters.")
    else:
        price = args.price  # negative for favorites (e.g., -110)
        win_ret = 100.0 / abs(price)  # e.g., 0.909 for -110
        bets["ret"] = np.where(
            bets["result"] == "PUSH", 0.0,
            np.where(bets["side"] == bets["result"], win_ret, -1.0)
        )
        roi = bets["ret"].mean()
        print("\n=== BET SIM (flat 1u each) ===")
        print(f"Filters: |edge|>={args.min_edge} "
              f"{'(stability>=' + str(args.stability_min) + ')' if args.stability_trials>0 else ''}")
        print(f"Bets: {len(bets)}  ROI: {roi*100:.1f}%")
        print(f"Wins: {(bets['ret']>0).sum()}  Losses: {(bets['ret']<0).sum()}  Pushes: {(bets['ret']==0).sum()}")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nWrote detailed rows → {args.out}")

if __name__ == "__main__":
    main()
