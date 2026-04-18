from __future__ import annotations

import argparse
from datetime import date, datetime

from src.ingestors.common import compute_payload_hash, iter_schedule_games, record_ingest_event, team_dimension_row, venue_dimension_row
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _official_game_date(game: dict[str, object], game_start_ts: datetime) -> date:
    official_date = game.get("officialDate")
    if official_date:
        return date.fromisoformat(str(official_date))
    return game_start_ts.date()


def build_date_row(game_date, game_type: str) -> dict[str, object]:
    season_phase = "regular"
    if game_type == "S":
        season_phase = "spring"
    elif game_type == "F":
        season_phase = "postseason"
    return {
        "date_id": game_date,
        "year": game_date.year,
        "month": game_date.month,
        "day": game_date.day,
        "day_of_week": game_date.weekday(),
        "week_of_year": int(game_date.strftime("%U")),
        "is_weekend": game_date.weekday() >= 5,
        "season_phase": season_phase,
    }


def normalize_game(game: dict[str, object]) -> dict[str, object]:
    game_start_ts = datetime.fromisoformat(str(game["gameDate"]).replace("Z", "+00:00"))
    game_date = _official_game_date(game, game_start_ts)
    home_team_id = int(game["teams"]["home"]["team"]["id"])
    away_team_id = int(game["teams"]["away"]["team"]["id"])
    home_team = team_dimension_row(home_team_id)["team_abbr"]
    away_team = team_dimension_row(away_team_id)["team_abbr"]
    venue = game.get("venue") or {}
    home_score = game["teams"]["home"].get("score")
    away_score = game["teams"]["away"].get("score")
    total_runs = None
    if home_score is not None and away_score is not None:
        total_runs = int(home_score) + int(away_score)
    status = (game.get("status") or {}).get("abstractGameState", "scheduled").lower()
    return {
        "game_id": int(game["gamePk"]),
        "game_date": game_date,
        "season": int(game.get("season", game_date.year)),
        "game_type": game.get("gameType", "R"),
        "status": status,
        "home_team": home_team,
        "away_team": away_team,
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "game_start_ts": game_start_ts,
        "double_header": 0 if game.get("doubleHeader") == "N" else 1,
        "innings": 9,
        "home_runs": home_score,
        "away_runs": away_score,
        "total_runs": total_runs,
        "home_win": None if total_runs is None else int(home_score) > int(away_score),
    }


def _preserve_existing_team_matchups(game_rows: list[dict[str, object]]) -> None:
    """Keep home/away team codes stable across re-ingests if Stats API drifts for the same game_id."""
    if not game_rows or not table_exists("games"):
        return
    ids = sorted({int(r["game_id"]) for r in game_rows})
    existing: dict[int, tuple[str, str]] = {}
    chunk_size = 400
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        placeholders = ", ".join(str(int(x)) for x in chunk)
        frame = query_df(
            f"SELECT game_id, home_team, away_team FROM games WHERE game_id IN ({placeholders})",
            {},
        )
        for rec in frame.to_dict("records"):
            gid = int(rec["game_id"])
            existing[gid] = (str(rec["home_team"]), str(rec["away_team"]))
    for row in game_rows:
        gid = int(row["game_id"])
        prev = existing.get(gid)
        if not prev:
            continue
        home_db, away_db = prev
        home_new = str(row["home_team"])
        away_new = str(row["away_team"])
        if home_db == home_new and away_db == away_new:
            continue
        log.warning(
            "Stats API returned a different matchup for game_id=%s than the database (keeping DB teams): DB %s @ %s, API %s @ %s",
            gid,
            away_db,
            home_db,
            away_new,
            home_new,
        )
        row["home_team"] = home_db
        row["away_team"] = away_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MLB schedule and game rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = iter_schedule_games(start_date.isoformat(), end_date.isoformat())
    if not games:
        log.info("No schedule rows returned for %s to %s", start_date, end_date)
        return 0

    # Deterministic order: schedule payload order can vary; upsert dedupes by game_id (last wins).
    games.sort(key=lambda g: int(g.get("gamePk", 0)))

    team_ids = sorted(
        {
            int(game["teams"]["home"]["team"]["id"])
            for game in games
            if game.get("teams")
        }
        | {
            int(game["teams"]["away"]["team"]["id"])
            for game in games
            if game.get("teams")
        }
    )
    venue_ids = sorted({int(game["venue"]["id"]) for game in games if game.get("venue", {}).get("id")})

    team_rows = [team_dimension_row(team_id) for team_id in team_ids]
    venue_rows = [venue_dimension_row(venue_id) for venue_id in venue_ids]
    game_rows = [normalize_game(game) for game in games]
    _preserve_existing_team_matchups(game_rows)
    date_rows = [build_date_row(row["game_date"], row["game_type"]) for row in game_rows]

    upsert_rows("dim_teams", team_rows, ["team_abbr"])
    upsert_rows("dim_venues", venue_rows, ["venue_id"])
    upsert_rows("dim_dates", date_rows, ["date_id"])
    inserted = upsert_rows("games", game_rows, ["game_id"])
    log.info("Upserted %s game rows for %s to %s", inserted, start_date, end_date)
    record_ingest_event(
        source_name="mlb_statsapi",
        ingestor_module="src.ingestors.games",
        target_date=start_date.isoformat(),
        row_count=inserted,
        payload_hash=compute_payload_hash(game_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())