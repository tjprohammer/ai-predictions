from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestors.common import player_dimension_row
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


REQUIRED_COLUMNS = {
    "game_date",
    "team",
    "player_id",
    "player_name",
    "lineup_slot",
    "source_name",
    "snapshot_ts",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_games() -> pd.DataFrame:
    return query_df(
        """
        SELECT game_id, game_date, home_team, away_team
        FROM games
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import manual lineup CSV into canonical lineups table")
    parser.add_argument("--csv", help="Override manual lineups csv path")
    args = parser.parse_args()
    settings = get_settings()
    csv_path = Path(args.csv) if args.csv else settings.manual_lineups_csv

    if not csv_path.exists():
        log.info("No manual lineup CSV found at %s; skipping", csv_path)
        return 0

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing lineup columns: {sorted(missing)}")

    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True)
    frame["player_id"] = frame["player_id"].astype(int)
    frame["lineup_slot"] = frame["lineup_slot"].astype(int)

    games = _load_games()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    if "game_id" in frame.columns:
        merged = frame.merge(
            games[["game_id", "game_date", "home_team", "away_team"]],
            on="game_id",
            how="left",
            suffixes=("", "_game"),
        )
        if "game_date_game" in merged.columns:
            merged["game_date"] = merged["game_date"].where(merged["game_date"].notna(), merged["game_date_game"])
    else:
        candidates = pd.concat(
            [
                games[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
                games[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"}),
            ],
            ignore_index=True,
        )
        merged = frame.merge(candidates, on=["game_date", "team"], how="left")

    if merged["game_id"].isna().any():
        unresolved = merged[merged["game_id"].isna()][["game_date", "team", "player_name"]]
        log.warning("Skipped %s unresolved lineup rows", len(unresolved))
        merged = merged[merged["game_id"].notna()].copy()

    player_rows = [
        player_dimension_row(
            int(row.player_id),
            full_name_override=row.player_name,
            team_abbr_override=row.team,
            position_override=getattr(row, "position", None),
        )
        for row in merged.itertuples(index=False)
    ]
    lineup_rows = []
    for row in merged.itertuples(index=False):
        lineup_rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "team": row.team,
                "player_id": int(row.player_id),
                "player_name": row.player_name,
                "lineup_slot": int(row.lineup_slot),
                "field_position": getattr(row, "position", None),
                "batting_order": int(row.lineup_slot),
                "is_confirmed": _as_bool(getattr(row, "confirmed", False)),
                "source_name": row.source_name,
                "source_url": getattr(row, "source_url", None),
                "snapshot_ts": row.snapshot_ts.to_pydatetime(),
            }
        )

    upsert_rows("dim_players", player_rows, ["player_id"])
    inserted = upsert_rows("lineups", lineup_rows, ["game_id", "player_id", "source_name", "snapshot_ts"])
    log.info("Imported %s lineup rows from %s", inserted, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())