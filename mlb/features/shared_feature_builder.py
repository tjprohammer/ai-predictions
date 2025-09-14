"""Shared Feature Builder
===========================
Lightweight, dependency-minimal feature builder used to move the retraining
script toward parity with the serving pipeline (`daily_api_workflow.py`).

Design Goals:
  * Pure SQL + pandas (no hidden side-effects) – safe to call in batch retraining
  * Resilient to missing columns (older historical rows) – fills with neutral defaults
  * Produces a stable, explicit feature column order (persisted in model metadata)
  * Adds a few derived features the live predictor relies on (pitcher / environment deltas)

This is an incremental step – not full parity with the in-workflow dynamic
`attach_recency_and_matchup_features`, but it centralises overlapping logic so
we can iterate without copy/paste divergence.

Usage (in retrain script):
  from mlb.features.shared_feature_builder import fetch_training_frame, build_feature_matrix
  raw = fetch_training_frame(engine, days=180)
  X, y = build_feature_matrix(raw)

Returned:
  X: pandas.DataFrame (numeric feature matrix)
  y: pandas.Series (total runs target)

Neutral Defaults Strategy:
  * ERA / WHIP → league-ish averages (4.20 / 1.30)
  * Bullpen ERA windows → 4.20
  * Park / weather → 1.00 / 70F / 5 mph
  * Percent-style columns (lineup splits) → 0.5
  * wRC+ → 100

Future Extensions (tracked):
  - Incorporate empirical Bayes blended splits used in serving
  - Add rolling form stats (7/14/30) with shrinkage
  - Weather interaction features (temp * wind, park_factor * temp)
  - Probabilistic residual model / sigma estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd
from sqlalchemy import text

NEUTRAL = {
    "home_sp_era": 4.20,
    "away_sp_era": 4.20,
    "home_sp_whip": 1.30,
    "away_sp_whip": 1.30,
    "bullpen_era_7d_home": 4.20,
    "bullpen_era_14d_home": 4.20,
    "bullpen_era_30d_home": 4.20,
    "bullpen_era_7d_away": 4.20,
    "bullpen_era_14d_away": 4.20,
    "bullpen_era_30d_away": 4.20,
    "ballpark_run_factor": 1.00,
    "ballpark_hr_factor": 1.00,
    "temperature": 70.0,
    "wind_speed": 5.0,
}

OPTIONAL_COLUMNS: List[str] = [
    # Environment
    "ballpark_run_factor", "ballpark_hr_factor", "temperature", "wind_speed",
    # Pitchers
    "home_sp_era", "away_sp_era", "home_sp_whip", "away_sp_whip",
    # Bullpen rolling ERAs
    "bullpen_era_7d_home", "bullpen_era_14d_home", "bullpen_era_30d_home",
    "bullpen_era_7d_away", "bullpen_era_14d_away", "bullpen_era_30d_away",
    # Team splits (gracefully optional)
    "team_wrc_plus_vs_rhp_7d_home", "team_wrc_plus_vs_rhp_7d_away",
    "team_wrc_plus_vs_lhp_7d_home", "team_wrc_plus_vs_lhp_7d_away",
    "lineup_r_pct_home", "lineup_r_pct_away",
]

BASE_REQUIRED = ["date", "home_team", "away_team", "market_total", "total_runs"]


def fetch_training_frame(engine, days: int) -> pd.DataFrame:
    """Pull historical rows from enhanced_games for the past N days.

    We intentionally select a superset of potentially useful columns; absent
    columns are tolerated (older DB snapshots)."""
    start_date = date.today() - timedelta(days=days)
    # Build dynamic SELECT list so we don't error if some columns are missing.
    # We rely on Postgres ignoring unknown columns if we probe information_schema first.
    with engine.begin() as conn:
        existing = conn.execute(text(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'enhanced_games' AND table_schema = current_schema()
            """
        )).scalars().all()
    cols = [c for c in (BASE_REQUIRED + OPTIONAL_COLUMNS) if c in existing]
    select_list = ",".join(cols)
    q = text(f"""
        SELECT {select_list}
        FROM enhanced_games
        WHERE date >= :start
          AND total_runs IS NOT NULL
          AND market_total IS NOT NULL
        ORDER BY date
    """)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"start": start_date})
    return df


def _apply_neutral_defaults(df: pd.DataFrame) -> pd.DataFrame:
    for col, val in NEUTRAL.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)
    # Soft fills for percentage / split style columns
    for c in df.columns:
        lc = c.lower()
        if "wrc_plus" in lc:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(100.0)
        elif lc.endswith("_pct_home") or lc.endswith("_pct_away") or lc.endswith("_pct"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5)
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create numeric feature matrix & target.

    Adds derived features:
      * era_diff = home_sp_era - away_sp_era
      * whip_diff = home_sp_whip - away_sp_whip
      * bullpen_era_diff_7d / 14d / 30d
      * park_temp_interaction = ballpark_run_factor * temperature (if both exist)
      * market_total retained as anchoring context (model learns additive offset)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    work = df.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.month
    work["dow"] = pd.to_datetime(work["date"]).dt.dayofweek

    work = _apply_neutral_defaults(work)

    # Derived deltas
    if set(["home_sp_era", "away_sp_era"]).issubset(work.columns):
        work["era_diff"] = work["home_sp_era"] - work["away_sp_era"]
    if set(["home_sp_whip", "away_sp_whip"]).issubset(work.columns):
        work["whip_diff"] = work["home_sp_whip"] - work["away_sp_whip"]
    if set(["bullpen_era_7d_home", "bullpen_era_7d_away"]).issubset(work.columns):
        work["bullpen_era_diff_7d"] = work["bullpen_era_7d_home"] - work["bullpen_era_7d_away"]
    if set(["bullpen_era_14d_home", "bullpen_era_14d_away"]).issubset(work.columns):
        work["bullpen_era_diff_14d"] = work["bullpen_era_14d_home"] - work["bullpen_era_14d_away"]
    if set(["bullpen_era_30d_home", "bullpen_era_30d_away"]).issubset(work.columns):
        work["bullpen_era_diff_30d"] = work["bullpen_era_30d_home"] - work["bullpen_era_30d_away"]
    if set(["ballpark_run_factor", "temperature"]).issubset(work.columns):
        work["park_temp_interaction"] = work["ballpark_run_factor"] * work["temperature"]

    # One-hot encode teams (same approach as baseline for consistency)
    teams = pd.concat([work["home_team"], work["away_team"]]).astype(str).unique()
    for t in teams:
        work[f"home_{t}"] = (work["home_team"].astype(str) == t).astype(int)
        work[f"away_{t}"] = (work["away_team"].astype(str) == t).astype(int)

    drop_cols = {"date", "total_runs", "home_team", "away_team"}
    feature_cols = [c for c in work.columns if c not in drop_cols]

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = work["total_runs"].astype(float)
    return X, y


__all__ = [
    "fetch_training_frame",
    "build_feature_matrix",
]
