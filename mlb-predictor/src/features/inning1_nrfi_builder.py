"""Build inning-1 NRFI training rows by joining first-five feature snapshots with boxscore labels."""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from src.features.contracts import (
    FIRST5_TOTALS_FEATURE_COLUMNS,
    FIRST5_TOTALS_META_COLUMNS,
    INNING1_NRFI_TARGET_COLUMN,
    validate_columns,
)
from src.models.common import load_feature_snapshots
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df
from src.utils.logging import get_logger

from src.features.common import write_feature_snapshot


log = get_logger(__name__)


def _fetch_inning1_labels(game_ids: list[int]) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame(columns=["game_id", "total_runs_inning1"])
    chunks: list[pd.DataFrame] = []
    step = 400
    for idx in range(0, len(game_ids), step):
        part = game_ids[idx : idx + step]
        placeholders = ", ".join(str(int(g)) for g in part)
        frame = query_df(
            f"""
            SELECT game_id, total_runs_inning1
            FROM games
            WHERE game_id IN ({placeholders})
            """,
        )
        chunks.append(frame)
    if not chunks:
        return pd.DataFrame(columns=["game_id", "total_runs_inning1"])
    return pd.concat(chunks, ignore_index=True)


def build_frame(start_date: date, end_date: date) -> pd.DataFrame:
    f5 = load_feature_snapshots("first5_totals")
    if f5.empty:
        return pd.DataFrame()
    f5["game_date"] = pd.to_datetime(f5["game_date"]).dt.date
    mask = (f5["game_date"] >= start_date) & (f5["game_date"] <= end_date)
    f5 = f5.loc[mask].copy()
    if f5.empty:
        return pd.DataFrame()

    labels = _fetch_inning1_labels([int(g) for g in f5["game_id"].unique().tolist()])
    if labels.empty:
        merged = f5.copy()
        merged["total_runs_inning1"] = np.nan
    else:
        merged = f5.merge(labels, on="game_id", how="left")

    tr = merged["total_runs_inning1"]
    merged[INNING1_NRFI_TARGET_COLUMN] = np.where(
        tr.isna(),
        np.nan,
        (tr == 0).astype(float),
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build inning-1 NRFI feature snapshots (first5 features + actual_nrfi label)")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    frame = build_frame(start_date, end_date)
    if frame.empty:
        log.info("No inning1_nrfi rows for %s to %s (need first5_totals features)", start_date, end_date)
        return 0

    required = list(FIRST5_TOTALS_META_COLUMNS) + list(FIRST5_TOTALS_FEATURE_COLUMNS) + [INNING1_NRFI_TARGET_COLUMN]
    validate_columns(frame, required, "inning1_nrfi")
    output_path = write_feature_snapshot(frame, "inning1_nrfi", start_date, end_date)
    labeled = frame[frame[INNING1_NRFI_TARGET_COLUMN].notna()]
    log.info(
        "Wrote %s inning1_nrfi rows (%s with inning-1 label) -> %s",
        len(frame),
        len(labeled),
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
