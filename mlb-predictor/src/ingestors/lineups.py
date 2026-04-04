from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestors.common import player_dimension_row, record_ingest_event
from src.ingestors.prepare_slate_inputs import LINEUP_COLUMNS, build_lineup_input_frame
from src.utils.cli import add_date_range_args, resolve_date_range
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


def _load_games_for_range(start_date, end_date) -> pd.DataFrame:
    return query_df(
        """
        SELECT game_id, game_date, home_team, away_team
        FROM games
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )


def _coerce_lineup_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in LINEUP_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[LINEUP_COLUMNS].copy()


def _filter_lineup_date_range(frame: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if frame.empty:
        return frame
    resolved_start_date = pd.to_datetime(start_date, errors="coerce").date()
    resolved_end_date = pd.to_datetime(end_date, errors="coerce").date()
    filtered = frame.copy()
    filtered["game_date"] = pd.to_datetime(filtered["game_date"], errors="coerce").dt.date
    filtered = filtered[filtered["game_date"].notna()].copy()
    return filtered[
        (filtered["game_date"] >= resolved_start_date) & (filtered["game_date"] <= resolved_end_date)
    ].copy()


def _build_lineup_import_frame(csv_path: Path, start_date, end_date) -> pd.DataFrame:
    games = _load_games_for_range(start_date, end_date)
    if games.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date

    existing = pd.DataFrame(columns=LINEUP_COLUMNS)
    if csv_path.exists():
        existing = _filter_lineup_date_range(_coerce_lineup_columns(pd.read_csv(csv_path)), start_date, end_date)
    else:
        log.info(
            "No manual lineup CSV found at %s; using projected lineup inference for %s to %s",
            csv_path,
            start_date,
            end_date,
        )

    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return build_lineup_input_frame(games, existing=existing, snapshot_ts=snapshot_ts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import lineup inputs into the canonical lineups table")
    parser.add_argument("--csv", help="Override manual lineups csv path")
    add_date_range_args(parser)
    args = parser.parse_args()
    settings = get_settings()
    csv_path = Path(args.csv) if args.csv else settings.manual_lineups_csv
    start_date, end_date = resolve_date_range(args)

    frame = _build_lineup_import_frame(csv_path, start_date, end_date)
    if frame.empty:
        log.info("No lineup rows available for %s to %s", start_date, end_date)
        return 0

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing lineup columns: {sorted(missing)}")

    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True)
    frame["player_id"] = frame["player_id"].astype(int)
    frame["lineup_slot"] = frame["lineup_slot"].astype(int)

    games = _load_games_for_range(start_date, end_date)
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
    log.info("Imported %s lineup rows for %s to %s", inserted, start_date, end_date)
    record_ingest_event(
        source_name="lineup_csv",
        ingestor_module="src.ingestors.lineups",
        target_date=start_date.isoformat(),
        row_count=inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())