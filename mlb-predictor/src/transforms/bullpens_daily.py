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
        pitches = group["pitches_thrown"].fillna(0).astype(int)
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
            }
        )

    delete_for_date_range("bullpens_daily", start_date, end_date)
    inserted = upsert_rows("bullpens_daily", rows, ["game_date", "team"])
    log.info("Refreshed %s bullpens_daily rows", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())