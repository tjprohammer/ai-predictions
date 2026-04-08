from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import delete_for_date_range, query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _outs_from_baseball_ip(value: float | None) -> int:
    if value is None or pd.isna(value):
        return 0
    whole = int(float(value))
    tenths = int(round((float(value) - whole) * 10))
    return whole * 3 + tenths


def _baseball_ip_from_outs(outs: int) -> float:
    whole, remainder = divmod(int(outs), 3)
    return float(f"{whole}.{remainder}")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    valid = frame[[value_col, weight_col]].dropna()
    if valid.empty:
        return None
    weights = valid[weight_col].astype(float)
    if float(weights.sum()) <= 0:
        return float(valid[value_col].mean())
    return float(np.average(valid[value_col].astype(float), weights=weights))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh canonical bullpens_daily rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    frame = query_df(
        """
        SELECT *
        FROM player_game_pitching
        WHERE game_date BETWEEN :start_date AND :end_date
          AND COALESCE(is_starter, FALSE) = FALSE
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        log.info("No bullpen pitching rows available for refresh")
        return 0

    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    rows = []
    for (game_date, team), group in frame.groupby(["game_date", "team"], dropna=False):
        outs = int(group["innings_pitched"].apply(_outs_from_baseball_ip).sum())
        late_innings_series = group["late_innings_pitched"] if "late_innings_pitched" in group.columns else pd.Series(0.0, index=group.index)
        late_runs_series = group["late_runs_allowed"] if "late_runs_allowed" in group.columns else pd.Series(0, index=group.index)
        late_earned_runs_series = group["late_earned_runs"] if "late_earned_runs" in group.columns else pd.Series(0, index=group.index)
        late_hits_series = group["late_hits_allowed"] if "late_hits_allowed" in group.columns else pd.Series(0, index=group.index)
        late_outs = int(late_innings_series.fillna(0).apply(_outs_from_baseball_ip).sum())
        late_usage_mask = (
            late_innings_series.fillna(0).apply(_outs_from_baseball_ip) > 0
        ) | (late_runs_series.fillna(0).astype(int) > 0) | (late_earned_runs_series.fillna(0).astype(int) > 0) | (late_hits_series.fillna(0).astype(int) > 0)
        pitches = group["pitches_thrown"].fillna(0).astype(int)
        late_earned_runs = int(late_earned_runs_series.fillna(0).sum())
        late_innings_decimal = late_outs / 3 if late_outs else 0
        rows.append(
            {
                "game_date": game_date,
                "season": game_date.year,
                "team": team,
                "innings_pitched": _baseball_ip_from_outs(outs),
                "pitches_thrown": int(pitches.sum()),
                "relievers_used": int(group["player_id"].nunique()),
                "hits_allowed": int(group["hits_allowed"].fillna(0).sum()),
                "earned_runs": int(group["earned_runs"].fillna(0).sum()),
                "walks": int(group["walks"].fillna(0).sum()),
                "strikeouts": int(group["strikeouts"].fillna(0).sum()),
                "hard_hit_pct": _weighted_mean(group, "hard_hit_pct", "pitches_thrown"),
                "late_innings_pitched": _baseball_ip_from_outs(late_outs),
                "late_relievers_used": int(group.loc[late_usage_mask, "player_id"].nunique()),
                "late_runs_allowed": int(late_runs_series.fillna(0).sum()),
                "late_earned_runs": late_earned_runs,
                "late_hits_allowed": int(late_hits_series.fillna(0).sum()),
                "late_era": None if late_innings_decimal <= 0 else float((late_earned_runs * 9) / late_innings_decimal),
            }
        )

    delete_for_date_range("bullpens_daily", start_date, end_date)
    inserted = upsert_rows("bullpens_daily", rows, ["game_date", "team"])
    log.info("Refreshed %s bullpens_daily rows", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())