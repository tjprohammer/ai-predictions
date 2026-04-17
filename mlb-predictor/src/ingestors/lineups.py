from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestors.common import BASE_URL, player_dimension_row, record_ingest_event, statsapi_get
from src.ingestors.prepare_slate_inputs import LINEUP_COLUMNS, build_lineup_input_frame
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger
from src.utils.pregame_lock import filter_games_dataframe_pregame_unlocked, locked_game_ids_from_db
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


def _lineup_slot_from_batting_order(batting_order: Any) -> int | None:
    if batting_order in (None, ""):
        return None
    batting_order_str = str(batting_order).strip()
    if not batting_order_str.isdigit():
        return None
    if len(batting_order_str) >= 3:
        return int(batting_order_str[:-2])
    return int(batting_order_str)


def _load_games_for_range(start_date, end_date) -> pd.DataFrame:
    return query_df(
        """
        SELECT game_id, game_date, home_team, away_team, game_start_ts
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


def _statsapi_lineup_rows_for_game(
    game_id: int,
    game_date,
    away_team: str,
    home_team: str,
    snapshot_ts: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        feed = statsapi_get(f"/api/v1.1/game/{game_id}/feed/live")
    except Exception as exc:  # noqa: BLE001
        log.warning("StatsAPI lineup fetch failed for game %s: %s", game_id, exc)
        return rows

    teams_box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    source_url = f"{BASE_URL}/api/v1.1/game/{game_id}/feed/live"
    for side, team_abbr in (("away", away_team), ("home", home_team)):
        players = (teams_box.get(side) or {}).get("players") or {}
        starters_by_slot: dict[int, dict[str, object]] = {}
        for player in players.values():
            person = player.get("person") or {}
            player_id = person.get("id")
            lineup_slot = _lineup_slot_from_batting_order(player.get("battingOrder"))
            if player_id is None or lineup_slot is None:
                continue

            batting_order_value = int(str(player.get("battingOrder")))
            positions = player.get("allPositions") or []
            field_position = (
                (player.get("position") or {}).get("abbreviation")
                or (positions[0] or {}).get("abbreviation") if positions else None
            )
            starter_row = {
                "game_id": game_id,
                "game_date": game_date,
                "team": team_abbr,
                "player_id": int(player_id),
                "player_name": person.get("fullName") or str(player_id),
                "lineup_slot": int(lineup_slot),
                "position": field_position,
                "confirmed": True,
                "source_name": "mlb_statsapi_lineups",
                "source_url": source_url,
                "snapshot_ts": snapshot_ts,
                "_batting_order_value": batting_order_value,
            }
            existing = starters_by_slot.get(int(lineup_slot))
            if existing is None or batting_order_value < int(existing["_batting_order_value"]):
                starters_by_slot[int(lineup_slot)] = starter_row

        if len(starters_by_slot) != 9:
            continue

        for lineup_slot in sorted(starters_by_slot):
            row = starters_by_slot[lineup_slot].copy()
            row.pop("_batting_order_value", None)
            rows.append(row)

    return rows


def _fetch_statsapi_lineup_frame(games: pd.DataFrame, snapshot_ts: str) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    game_rows: list[tuple[int, object, str, str]] = []
    for game in games.itertuples(index=False):
        game_rows.append((int(game.game_id), game.game_date, str(game.away_team), str(game.home_team)))

    workers = min(8, max(1, len(game_rows)))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_statsapi_lineup_rows_for_game, gid, gd, aw, hw, snapshot_ts): gid
            for gid, gd, aw, hw in game_rows
        }
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                log.warning("StatsAPI lineup worker failed for game %s: %s", futures[fut], exc)

    return pd.DataFrame(rows, columns=LINEUP_COLUMNS)


def _lineup_team_keys(frame: pd.DataFrame) -> set[tuple[int, str]]:
    if frame.empty:
        return set()
    keys: set[tuple[int, str]] = set()
    for row in frame.itertuples(index=False):
        game_id = getattr(row, "game_id", None)
        team = getattr(row, "team", None)
        if pd.isna(game_id) or team is None:
            continue
        keys.add((int(game_id), str(team)))
    return keys


def _prefer_statsapi_lineups(statsapi_frame: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if statsapi_frame.empty:
        return existing.copy()
    if existing.empty:
        return statsapi_frame.copy()

    covered_keys = _lineup_team_keys(statsapi_frame)
    filtered_existing = existing[
        existing.apply(
            lambda row: (int(row["game_id"]), str(row["team"])) not in covered_keys if pd.notna(row["game_id"]) else True,
            axis=1,
        )
    ].copy()
    return pd.concat([statsapi_frame, filtered_existing], ignore_index=True)


def _build_lineup_import_frame(csv_path: Path, start_date, end_date, *, pregame_lock_minutes: int) -> pd.DataFrame:
    games = _load_games_for_range(start_date, end_date)
    if games.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    games = filter_games_dataframe_pregame_unlocked(games, pregame_lock_minutes, log_name="lineup ingest games")
    if games.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    existing = pd.DataFrame(columns=LINEUP_COLUMNS)
    if csv_path.exists():
        existing = _filter_lineup_date_range(_coerce_lineup_columns(pd.read_csv(csv_path)), start_date, end_date)
    else:
        log.info(
            "No manual lineup CSV found at %s; using StatsAPI lineups when available and projected inference otherwise for %s to %s",
            csv_path,
            start_date,
            end_date,
        )

    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    statsapi_existing = _fetch_statsapi_lineup_frame(games, snapshot_ts)
    preferred_existing = _prefer_statsapi_lineups(statsapi_existing, existing)
    return build_lineup_input_frame(games, existing=preferred_existing, snapshot_ts=snapshot_ts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import lineup inputs into the canonical lineups table")
    parser.add_argument("--csv", help="Override manual lineups csv path")
    add_date_range_args(parser)
    args = parser.parse_args()
    settings = get_settings()
    csv_path = Path(args.csv) if args.csv else settings.manual_lineups_csv
    start_date, end_date = resolve_date_range(args)

    frame = _build_lineup_import_frame(
        csv_path,
        start_date,
        end_date,
        pregame_lock_minutes=settings.pregame_ingest_lock_minutes,
    )
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

    locked_ids = locked_game_ids_from_db(start_date, end_date, settings.pregame_ingest_lock_minutes)
    if locked_ids:
        before = len(frame)
        frame = frame.loc[~frame["game_id"].isin(locked_ids)].copy()
        dropped = before - len(frame)
        if dropped:
            log.info(
                "Pregame ingest lock: skipped %s lineup row(s) for %s locked game(s) (CSV / merged)",
                dropped,
                len(locked_ids),
            )

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