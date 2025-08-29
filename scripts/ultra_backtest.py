#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add the mlb-overs directory to the path so we can import from it
sys.path.append(str(Path(__file__).parent / "mlb-overs" / "pipelines"))

# Reuse your feature/prep + predict code
from ultra_sharp_pipeline import _prep_features, _is_closed_row, _predict_with_bundle

def _pick_bundle(row, model_dir: str) -> Path:
    md = Path(model_dir)
    if _is_closed_row(row):
        p = md / "ultra_bundle_closed.joblib"
        if p.exists(): return p
    else:
        p = md / "ultra_bundle_open.joblib"
        if p.exists(): return p
    return md / "ultra_bundle_all.joblib"

def _jitter_predict(df, bundle_path: Path, trials=0, jitter=0.05):
    """Return base preds and (optional) sign-agreement stability with feature jitter."""
    base = _predict_with_bundle(bundle_path, df)
    if trials <= 0:
        return base, np.ones(len(base))
    signs = []
    for _ in range(trials):
        noisy = df.copy()
        for col in ["wind_out_cf","air_density_proxy","altitude_ft",
                    "home_bp_fatigue","away_bp_fatigue",
                    "home_arms_avail","away_arms_avail"]:
            if col in noisy.columns:
                x = pd.to_numeric(noisy[col], errors="coerce")
                noisy[col] = x * (1 + np.random.uniform(-jitter, jitter, len(noisy)))
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

    eng = create_engine(args.db)
    q = """
      SELECT gc.*, eg.market_total, eg.total_runs, eg.date
      FROM game_conditions gc
      JOIN enhanced_games eg ON gc.game_id::text = eg.game_id::text
      WHERE eg.date BETWEEN :s AND :e
        AND eg.market_total IS NOT NULL
        AND eg.total_runs IS NOT NULL
    """
    df = pd.read_sql(text(q), eng, params={"s": args.start, "e": args.end})
    if df.empty:
        print("No finished games with market_total in range.")
        return

    parts = []
    # bucket by roof state
    buckets = df.apply(lambda r: "closed" if _is_closed_row(r) else "open", axis=1)
    for bucket, g in df.groupby(buckets):
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
