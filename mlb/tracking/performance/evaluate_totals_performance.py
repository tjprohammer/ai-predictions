#!/usr/bin/env python3
"""
Comprehensive Totals Prediction Performance Evaluator
=====================================================

Answers the questions:
  * "Show me ALL the games and how the model did"
  * "What's the win percentage?" (directional & betting picks)
  * "How accurate are the raw totals?" (MAE / RMSE / bias / hit rates)

Models evaluated (if columns populated):
  - predicted_total              (Learning Adaptive / Enhanced Bullpen)
  - predicted_total_learning     (Ultra 80 Incremental)
  - predicted_total_whitelist    (Pruned whitelist model)

Definitions:
  Directional Outcome vs Market:
     actual > market_total  => OVER wins
     actual < market_total  => UNDER wins
     actual == market_total => PUSH (excluded from directional accuracy denominator)

  Recommendation ("pick") set:
     A game where |prediction - market_total| >= edge_threshold (default 0.75 runs).
     Bet side = sign(prediction - market_total). (positive => OVER, negative => UNDER)

  Pick Win%:
     wins / (wins + losses) ignoring pushes.

  ROI (flat 1u per pick):
     Sum(profit) / number_of_bets  (loss = -1u, win = win_return_u, push = 0; pushes not in denominator)
     Uses over_odds/under_odds when present else assumes -110.

Outputs:
  * Console summary per model (overall + pick subset)
  * CSV of every evaluated game with per-model errors & pick results
  * Aggregated daily performance CSV (optional)

Usage Examples:
  python mlb/tracking/performance/evaluate_totals_performance.py --all
  python mlb/tracking/performance/evaluate_totals_performance.py --model predicted_total --start 2025-04-01 --end 2025-09-10
  python mlb/tracking/performance/evaluate_totals_performance.py --all --edge-threshold 0.5 --export-daily

Exit codes: 0 success, 1 no data / error.
"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
PREDICTION_COLS = ["predicted_total", "predicted_total_learning", "predicted_total_whitelist"]


def american_win_return(odds: float) -> float:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        odds = -110
    return (100 / abs(odds)) if odds < 0 else (odds / 100)


def fetch_date_bounds(engine) -> Optional[tuple]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT MIN(date), MAX(date) FROM enhanced_games WHERE total_runs IS NOT NULL")).fetchone()
    if row and row[0] and row[1]:
        return row[0], row[1]
    return None


def fetch_games(engine, start: date, end: date) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, game_id, home_team, away_team,
                       market_total, total_runs,
                       predicted_total, predicted_total_learning, predicted_total_whitelist,
                       over_odds, under_odds
                FROM enhanced_games
                WHERE date BETWEEN :s AND :e
                  AND total_runs IS NOT NULL
                  AND (predicted_total IS NOT NULL OR predicted_total_learning IS NOT NULL OR predicted_total_whitelist IS NOT NULL)
                ORDER BY date, game_id
                """
            ),
            conn,
            params={"s": start, "e": end},
        )
    return df


def core_stats(pred: pd.Series, actual: pd.Series) -> Dict[str, float]:
    err = pred - actual
    abs_err = err.abs()
    return {
        "n": int(len(err)),
        "mae": float(abs_err.mean()),
        "rmse": float(math.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "corr": float(pred.corr(actual)) if len(err) > 1 else float("nan"),
        "hit_0_5": float((abs_err <= 0.5).mean()),
        "hit_1_0": float((abs_err <= 1.0).mean()),
        "hit_1_5": float((abs_err <= 1.5).mean()),
        "hit_2_0": float((abs_err <= 2.0).mean()),
    }


def directional_performance(df: pd.DataFrame, col: str) -> Dict[str, float]:
    out = {"dir_n": 0, "dir_acc": float("nan")}
    if "market_total" not in df.columns or df["market_total"].isna().all():
        return out
    mt = df["market_total"]
    actual_sign = np.sign(df["total_runs"] - mt)
    model_sign = np.sign(df[col] - mt)
    mask = (actual_sign != 0) & (model_sign != 0)
    if mask.any():
        out["dir_n"] = int(mask.sum())
        out["dir_acc"] = float((actual_sign[mask] == model_sign[mask]).mean())
    return out


def picks_performance(df: pd.DataFrame, col: str, edge_threshold: float) -> Dict[str, float]:
    out = {
        "picks_n": 0,
        "picks_win_pct": float("nan"),
        "picks_roi": float("nan"),
        "picks_w": 0,
        "picks_l": 0,
        "picks_p": 0,
    }
    if "market_total" not in df.columns or df["market_total"].isna().all():
        return out
    mt = df["market_total"]
    pred = df[col]
    edge = pred - mt
    mask = edge.abs() >= edge_threshold
    if not mask.any():
        return out

    subset = df[mask].copy()
    out["picks_n"] = int(len(subset))
    # Determine bet side
    subset["bet_side"] = np.where(edge[mask] > 0, "OVER", "UNDER")
    # Determine outcome
    actual = subset["total_runs"]
    pushes = actual == mt[mask]
    overs_hit = actual > mt[mask]
    unders_hit = actual < mt[mask]
    wins = ((subset["bet_side"] == "OVER") & overs_hit) | ((subset["bet_side"] == "UNDER") & unders_hit)
    losses = ((subset["bet_side"] == "OVER") & unders_hit) | ((subset["bet_side"] == "UNDER") & overs_hit)

    out["picks_w"] = int(wins.sum())
    out["picks_l"] = int(losses.sum())
    out["picks_p"] = int(pushes.sum())
    denom = wins.sum() + losses.sum()
    if denom > 0:
        out["picks_win_pct"] = float(wins.sum() / denom)

    # ROI calc using available odds (choose correct column per bet side)
    profits = []
    for idx, row in subset.iterrows():
        if pushes.loc[idx]:
            profits.append(0.0)
            continue
        side = row["bet_side"]
        if side == "OVER":
            odds = row.get("over_odds", -110)
            win_ret = american_win_return(odds)
            profits.append(win_ret if ((row["total_runs"] > row["market_total"])) else -1.0)
        else:
            odds = row.get("under_odds", -110)
            win_ret = american_win_return(odds)
            profits.append(win_ret if ((row["total_runs"] < row["market_total"])) else -1.0)
    bet_count = wins.sum() + losses.sum()  # exclude pushes
    if bet_count > 0:
        out["picks_roi"] = float(sum(profits) / bet_count)
    return out


def evaluate_model(df: pd.DataFrame, col: str, edge_threshold: float) -> Dict[str, float]:
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return {"model": col, "n": 0}
    stats = core_stats(sub[col], sub["total_runs"])  # type: ignore
    dir_stats = directional_performance(sub, col)
    pick_stats = picks_performance(sub, col, edge_threshold)
    merged = {"model": col}
    merged.update(stats)
    merged.update(dir_stats)
    merged.update(pick_stats)
    return merged


def augment_detailed(df: pd.DataFrame, models: List[str], edge_threshold: float) -> pd.DataFrame:
    out = df.copy()
    for m in models:
        if m not in out.columns:
            continue
        out[f"{m}_error"] = out[m] - out["total_runs"]
        out[f"{m}_abs_error"] = out[f"{m}_error"].abs()
        if "market_total" in out.columns:
            edge = out[m] - out["market_total"]
            out[f"{m}_edge"] = edge
            out[f"{m}_pick_flag"] = edge.abs() >= edge_threshold
            out[f"{m}_pick_side"] = np.where(edge > 0, "OVER", "UNDER")
            # outcome
            actual = out["total_runs"]
            over_hit = actual > out["market_total"]
            under_hit = actual < out["market_total"]
            push = actual == out["market_total"]
            win = ((out[f"{m}_pick_side"] == "OVER") & over_hit) | ((out[f"{m}_pick_side"] == "UNDER") & under_hit)
            loss = ((out[f"{m}_pick_side"] == "OVER") & under_hit) | ((out[f"{m}_pick_side"] == "UNDER") & over_hit)
            out[f"{m}_pick_result"] = np.where(~out[f"{m}_pick_flag"], "NO_PICK", np.where(push, "PUSH", np.where(win, "WIN", np.where(loss, "LOSS", "PUSH"))))
    return out


def format_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "NA"
    return f"{v*100:.1f}%"


def print_model_summary(metrics: Dict[str, float]):
    if metrics.get("n", 0) == 0:
        print(f"Model {metrics['model']}: no data")
        return
    print(f"\n=== {metrics['model']} ===")
    print(f"Games: {metrics['n']}")
    print(f"MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f} | Bias {metrics['bias']:+.2f} | Corr {metrics['corr']:.3f}")
    print(
        "Hit Rates (abs error ≤ x): 0.5:" f"{format_pct(metrics['hit_0_5'])}  1.0:{format_pct(metrics['hit_1_0'])}  1.5:{format_pct(metrics['hit_1_5'])}  2.0:{format_pct(metrics['hit_2_0'])}"
    )
    if metrics.get("dir_n", 0) > 0:
        print(
            f"Directional (non-push): {metrics['dir_n']} | Accuracy {format_pct(metrics['dir_acc'])}"
        )
    if metrics.get("picks_n", 0) > 0:
        print(
            f"Picks (|edge|≥thr): {metrics['picks_n']}  W:{metrics['picks_w']} L:{metrics['picks_l']} P:{metrics['picks_p']} | Win% {format_pct(metrics['picks_win_pct'])} | ROI/Bet {metrics['picks_roi']:.3f}"
        )


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate historical totals prediction performance")
    ap.add_argument("--db", default=DEFAULT_DB, help="Database URL")
    ap.add_argument("--start", help="Start date YYYY-MM-DD (default: first game with results)")
    ap.add_argument("--end", help="End date YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--model", choices=PREDICTION_COLS, default="predicted_total", help="Single model to evaluate (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="Evaluate all available model columns")
    ap.add_argument("--edge-threshold", type=float, default=0.75, help="Edge threshold in runs for pick set (default 0.75)")
    ap.add_argument("--export-daily", action="store_true", help="Export daily aggregated performance CSV")
    ap.add_argument("--export-rolling", action="store_true", help="Export rolling 30-day window metrics CSV (per model)")
    ap.add_argument("--persist-history", action="store_true", help="Persist rolling + overall metrics to totals_performance_history table")
    ap.add_argument("--rolling-windows", default="30", help="Comma-separated rolling window sizes in days (e.g. 7,14,30,60). Default 30")
    ap.add_argument("--export-detailed", action="store_true", help="Export per-game detailed CSV (default True)")
    ap.add_argument("--no-export-detailed", action="store_true", help="Disable per-game export")
    args = ap.parse_args()

    engine = create_engine(args.db, pool_pre_ping=True)
    bounds = fetch_date_bounds(engine)
    if not bounds:
        print("No historical games with results found.")
        return 1

    first_date, last_date = bounds
    start = date.fromisoformat(args.start) if args.start else first_date
    # Default end is yesterday (avoid partial in-progress day) if no explicit end
    default_end = date.today() - timedelta(days=1)
    end = date.fromisoformat(args.end) if args.end else min(last_date, default_end)

    if end < start:
        print("End date precedes start date.")
        return 1

    print("Evaluating totals prediction performance")
    print(f"Date range: {start} → {end}")
    print(f"Edge threshold: ±{args.edge_threshold} runs")

    df = fetch_games(engine, start, end)
    if df.empty:
        print("No games with predictions in selected range.")
        return 1

    available_models = [c for c in PREDICTION_COLS if c in df.columns and df[c].notna().any()]
    if not available_models:
        print("No populated prediction columns.")
        return 1

    models = available_models if args.all else [m for m in [args.model] if m in available_models]
    if not models:
        print(f"Requested model {args.model} not available; available: {available_models}")
        return 1

    summaries = []
    for m in models:
        metrics = evaluate_model(df, m, args.edge_threshold)
        summaries.append(metrics)
        print_model_summary(metrics)

    # Export per-game detailed
    export_detailed = not args.no_export_detailed if not args.export_detailed else True
    if export_detailed:
        detailed = augment_detailed(df, models, args.edge_threshold)
        os.makedirs("exports", exist_ok=True)
        path = os.path.join(
            "exports",
            f"totals_performance_detailed_{start}_{end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        detailed.to_csv(path, index=False)
        print(f"\nPer-game detailed results saved → {path}")
        # Show sample
        print("Sample (first 5 rows):")
        print(detailed.head().to_string(index=False))

    # Export daily aggregates
    if args.export_daily:
        daily_rows = []
        for m in models:
            tmp = df[df[m].notna()].copy()
            if tmp.empty:
                continue
            tmp["abs_err"] = (tmp[m] - tmp["total_runs"]).abs()
            day_grp = (
                tmp.groupby("date")
                .agg(
                    games=("game_id", "count"),
                    mae=("abs_err", "mean"),
                    avg_pred=(m, "mean"),
                    avg_actual=("total_runs", "mean"),
                )
                .reset_index()
            )
            day_grp["model"] = m
            daily_rows.append(day_grp)
        if daily_rows:
            daily_df = pd.concat(daily_rows, ignore_index=True)
            daily_path = os.path.join(
                "exports",
                f"totals_performance_daily_{start}_{end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            daily_df.to_csv(daily_path, index=False)
            print(f"Daily aggregates saved → {daily_path}")

    # Compute multi-window rolling metrics if requested
    roll_df = None
    if args.export_rolling or args.persist_history:
        try:
            windows = sorted({int(w.strip()) for w in args.rolling_windows.split(',') if w.strip()})
        except ValueError:
            print(f"Invalid --rolling-windows value: {args.rolling_windows}")
            windows = [30]
        roll_entries = []
        for m in models:
            model_df = df[df[m].notna()].copy()
            if model_df.empty:
                continue
            model_df = model_df.sort_values("date")
            unique_dates = sorted(model_df["date"].unique())
            for window_days in windows:
                if window_days <= 0:
                    continue
                for d in unique_dates:
                    start_window = d - timedelta(days=window_days - 1)
                    slice_df = model_df[(model_df["date"] >= start_window) & (model_df["date"] <= d)]
                    # Require at least 5 games in window to report to avoid noise
                    if slice_df.empty or len(slice_df) < 5:
                        continue
                    metrics = evaluate_model(slice_df, m, args.edge_threshold)
                    roll_entries.append({
                        "as_of_date": d,
                        "model": m,
                        "window_days": window_days,
                        "games": metrics.get("n", 0),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "bias": metrics.get("bias"),
                        "dir_acc": metrics.get("dir_acc"),
                        "picks_win_pct": metrics.get("picks_win_pct"),
                        "picks_roi": metrics.get("picks_roi"),
                        "picks_n": metrics.get("picks_n"),
                    })
        if roll_entries:
            roll_df = pd.DataFrame(roll_entries)
            if args.export_rolling:
                win_str = "-".join(map(str, windows))
                roll_path = os.path.join(
                    "exports", f"totals_performance_rolling_{win_str}_{start}_{end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                roll_df.to_csv(roll_path, index=False)
                print(f"Rolling window metrics saved → {roll_path}")

    # Persist history if requested
    if args.persist_history:
        from sqlalchemy import text as _txt
        with engine.begin() as conn:
            conn.execute(_txt(
                """
                CREATE TABLE IF NOT EXISTS totals_performance_history (
                  id SERIAL PRIMARY KEY,
                  as_of_date DATE NOT NULL,
                  model TEXT NOT NULL,
                  window_days INT NOT NULL,
                  games INT NOT NULL,
                  mae DOUBLE PRECISION,
                  rmse DOUBLE PRECISION,
                  bias DOUBLE PRECISION,
                  dir_acc DOUBLE PRECISION,
                  picks_win_pct DOUBLE PRECISION,
                  picks_roi DOUBLE PRECISION,
                  picks_n INT,
                  created_at TIMESTAMP DEFAULT NOW(),
                  UNIQUE(as_of_date, model, window_days)
                )
                """
            ))
            # Overall summary rows (window_days = -1)
            for s in summaries:
                conn.execute(_txt(
                    """
                    INSERT INTO totals_performance_history(as_of_date, model, window_days, games, mae, rmse, bias, dir_acc, picks_win_pct, picks_roi, picks_n)
                    VALUES(:as_of_date, :model, :window_days, :games, :mae, :rmse, :bias, :dir_acc, :picks_win_pct, :picks_roi, :picks_n)
                    ON CONFLICT (as_of_date, model, window_days) DO UPDATE SET
                      games=EXCLUDED.games, mae=EXCLUDED.mae, rmse=EXCLUDED.rmse, bias=EXCLUDED.bias,
                      dir_acc=EXCLUDED.dir_acc, picks_win_pct=EXCLUDED.picks_win_pct, picks_roi=EXCLUDED.picks_roi, picks_n=EXCLUDED.picks_n,
                      created_at=NOW()
                    """
                ), {
                    "as_of_date": end,
                    "model": s.get("model"),
                    "window_days": -1,
                    "games": s.get("n", 0),
                    "mae": s.get("mae"),
                    "rmse": s.get("rmse"),
                    "bias": s.get("bias"),
                    "dir_acc": s.get("dir_acc"),
                    "picks_win_pct": s.get("picks_win_pct"),
                    "picks_roi": s.get("picks_roi"),
                    "picks_n": s.get("picks_n"),
                })
            # Rolling rows (multi-window)
            if roll_df is not None and not roll_df.empty:
                for _, r in roll_df.iterrows():
                    conn.execute(_txt(
                        """
                        INSERT INTO totals_performance_history(as_of_date, model, window_days, games, mae, rmse, bias, dir_acc, picks_win_pct, picks_roi, picks_n)
                        VALUES(:as_of_date, :model, :window_days, :games, :mae, :rmse, :bias, :dir_acc, :picks_win_pct, :picks_roi, :picks_n)
                        ON CONFLICT (as_of_date, model, window_days) DO UPDATE SET
                          games=EXCLUDED.games, mae=EXCLUDED.mae, rmse=EXCLUDED.rmse, bias=EXCLUDED.bias,
                          dir_acc=EXCLUDED.dir_acc, picks_win_pct=EXCLUDED.picks_win_pct, picks_roi=EXCLUDED.picks_roi, picks_n=EXCLUDED.picks_n,
                          created_at=NOW()
                        """
                    ), {
                        "as_of_date": r.as_of_date,
                        "model": r.model,
                        "window_days": int(r.window_days),
                        "games": int(r.games),
                        "mae": r.mae,
                        "rmse": r.rmse,
                        "bias": r.bias,
                        "dir_acc": r.dir_acc,
                        "picks_win_pct": r.picks_win_pct,
                        "picks_roi": r.picks_roi,
                        "picks_n": r.picks_n,
                    })
        print("Persisted performance history (overall + rolling) to totals_performance_history")

    # Summary table export
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(
        "exports", f"totals_performance_summary_{start}_{end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary metrics saved → {summary_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
