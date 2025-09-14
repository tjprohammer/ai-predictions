#!/usr/bin/env python3
"""Performance Alert Script.

Reads latest entries from totals_performance_history and evaluates simple guardrails.
Exit code 0 => all good, 2 => warnings present.

Rules (evaluated per model & window):
  * Bias magnitude > 0.75 => alert
  * MAE > 3.25 => alert
  * picks_win_pct < 0.55 with picks_n >= 50 => alert
  * picks_roi < 0 with picks_n >= 50 => alert

Prioritizes shorter windows (7,14,30) to catch drift quickly.
"""

from __future__ import annotations

import os
import argparse
from datetime import date
from typing import List, Dict

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")


def fetch_latest(engine, windows=(7, 14, 30)) -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT t.* FROM totals_performance_history t
                INNER JOIN (
                  SELECT model, window_days, MAX(as_of_date) AS max_date
                  FROM totals_performance_history
                  WHERE window_days IN :wins
                  GROUP BY model, window_days
                ) m
                ON t.model = m.model AND t.window_days = m.window_days AND t.as_of_date = m.max_date
                ORDER BY t.model, t.window_days
                """
            ),
            conn,
            params={"wins": tuple(windows)},
        )
    return df


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    alerts = []
    for _, r in df.iterrows():
        issues: List[str] = []
        if r.get("bias") is not None and abs(r.bias) > 0.75:
            issues.append(f"Bias {r.bias:+.2f} > 0.75")
        if r.get("mae") is not None and r.mae > 3.25:
            issues.append(f"MAE {r.mae:.2f} > 3.25")
        if r.get("picks_n") is not None and r.picks_n >= 50:
            if r.get("picks_win_pct") is not None and r.picks_win_pct < 0.55:
                issues.append(f"Pick Win% {r.picks_win_pct:.3f} < 0.55")
            if r.get("picks_roi") is not None and r.picks_roi < 0:
                issues.append(f"Pick ROI {r.picks_roi:.3f} < 0")
        alerts.append({
            "as_of_date": r.as_of_date,
            "model": r.model,
            "window_days": int(r.window_days),
            "issues": "; ".join(issues) if issues else "",
        })
    return pd.DataFrame(alerts)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Check performance guardrails")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--windows", default="7,14,30", help="Comma-separated window_days to evaluate")
    ap.add_argument("--fail-on-alert", action="store_true", help="Exit 2 when any alerts present")
    args = ap.parse_args()

    try:
        windows = tuple(sorted({int(x.strip()) for x in args.windows.split(',') if x.strip()}))
    except ValueError:
        print(f"Invalid --windows value: {args.windows}")
        return 1
    engine = create_engine(args.db, pool_pre_ping=True)
    df = fetch_latest(engine, windows)
    if df.empty:
        print("No performance history rows found for supplied windows.")
        return 1
    alerts_df = evaluate(df)
    print("Latest performance guardrail evaluation:")
    print(alerts_df.to_string(index=False))
    any_alerts = alerts_df["issues"].str.len() > 0
    if any_alerts.any():
        print("ALERTS DETECTED")
        if args.fail_on_alert:
            return 2
    else:
        print("All clear.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
