from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import delete_for_date_range, query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    valid = frame[[value_col, weight_col]].dropna()
    if valid.empty:
        return None
    weights = valid[weight_col].astype(float)
    if float(weights.sum()) <= 0:
        return float(valid[value_col].mean())
    return float(np.average(valid[value_col].astype(float), weights=weights))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh canonical team_offense_daily rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    frame = query_df(
        """
        SELECT *
        FROM player_game_batting
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        log.info("No player batting rows available for offense_daily refresh")
        return 0

    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    rows = []
    for (game_date, team), group in frame.groupby(["game_date", "team"], dropna=False):
        at_bats = int(group["at_bats"].fillna(0).sum())
        hits = int(group["hits"].fillna(0).sum())
        walks = int(group["walks"].fillna(0).sum())
        hbp = int(group["hbp"].fillna(0).sum())
        sac_flies = int(group["sac_flies"].fillna(0).sum())
        plate_appearances = int(group["plate_appearances"].fillna(0).sum())
        total_bases = int(
            group["singles"].fillna(0).sum()
            + 2 * group["doubles"].fillna(0).sum()
            + 3 * group["triples"].fillna(0).sum()
            + 4 * group["home_runs"].fillna(0).sum()
        )
        ba = None if at_bats == 0 else hits / at_bats
        obp_denominator = at_bats + walks + hbp + sac_flies
        obp = None if obp_denominator == 0 else (hits + walks + hbp) / obp_denominator
        slg = None if at_bats == 0 else total_bases / at_bats
        rows.append(
            {
                "game_date": game_date,
                "season": game_date.year,
                "team": team,
                "games_played": 1,
                "plate_appearances": plate_appearances,
                "at_bats": at_bats,
                "runs": int(group["runs"].fillna(0).sum()),
                "hits": hits,
                "walks": walks,
                "strikeouts": int(group["strikeouts"].fillna(0).sum()),
                "singles": int(group["singles"].fillna(0).sum()),
                "doubles": int(group["doubles"].fillna(0).sum()),
                "triples": int(group["triples"].fillna(0).sum()),
                "home_runs": int(group["home_runs"].fillna(0).sum()),
                "ba": ba,
                "obp": obp,
                "slg": slg,
                "iso": None if ba is None or slg is None else slg - ba,
                "xba": _weighted_mean(group, "xba", "plate_appearances"),
                "xwoba": _weighted_mean(group, "xwoba", "plate_appearances"),
                "bb_pct": None if plate_appearances == 0 else walks / plate_appearances,
                "k_pct": None if plate_appearances == 0 else int(group["strikeouts"].fillna(0).sum()) / plate_appearances,
                "hard_hit_pct": _weighted_mean(group, "hard_hit_pct", "plate_appearances"),
            }
        )

    delete_for_date_range("team_offense_daily", start_date, end_date)
    inserted = upsert_rows("team_offense_daily", rows, ["game_date", "team"])
    log.info("Refreshed %s team_offense_daily rows", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())