from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.features.common import infer_lineup_from_history
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

LINEUP_COLUMNS = [
    "game_id",
    "game_date",
    "team",
    "player_id",
    "player_name",
    "lineup_slot",
    "position",
    "confirmed",
    "source_name",
    "source_url",
    "snapshot_ts",
]
MARKET_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "team",
    "player_id",
    "player_name",
    "sportsbook",
    "market_type",
    "line_value",
    "over_price",
    "under_price",
    "snapshot_ts",
    "is_opening",
    "is_closing",
    "source_name",
]
HITTER_PROP_MARKET_TYPE = "player_hits"
HITTER_PROP_LINE_VALUE = 0.5
FIRST5_TOTAL_MARKET_TYPE = "first_five_total"


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns].copy()


def _load_games(start_date, end_date) -> pd.DataFrame:
    frame = query_df(
        """
        SELECT game_id, game_date, home_team, away_team, game_start_ts
        FROM games
        WHERE game_date BETWEEN :start_date AND :end_date
        ORDER BY game_date, game_start_ts NULLS LAST, away_team, home_team
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return frame
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    return frame


def _load_starting_pitchers(start_date, end_date) -> pd.DataFrame:
    frame = query_df(
        """
        WITH ranked_starters AS (
            SELECT
                s.game_id,
                s.game_date,
                s.team,
                s.pitcher_id,
                COALESCE(dp.full_name, CAST(s.pitcher_id AS TEXT)) AS pitcher_name,
                s.is_probable,
                ROW_NUMBER() OVER (
                    PARTITION BY s.game_id, s.team
                    ORDER BY COALESCE(s.is_probable, FALSE) DESC, s.pitcher_id
                ) AS row_rank
            FROM pitcher_starts s
            LEFT JOIN dim_players dp ON dp.player_id = s.pitcher_id
            WHERE s.game_date BETWEEN :start_date AND :end_date
        )
        SELECT game_id, game_date, team, pitcher_id, pitcher_name, is_probable
        FROM ranked_starters
        WHERE row_rank = 1
        ORDER BY game_date, game_id, team
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return frame
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    return frame


def _load_lineup_history(end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_start = end_date - timedelta(days=400)
    player_batting = query_df(
        """
        SELECT game_date, player_id, team, lineup_slot, plate_appearances
        FROM player_game_batting
        WHERE game_date >= :history_start AND game_date < :end_date
        """,
        {"history_start": history_start, "end_date": end_date},
    )
    if not player_batting.empty:
        player_batting["game_date"] = pd.to_datetime(player_batting["game_date"], errors="coerce")
    players = query_df("SELECT player_id, full_name FROM dim_players")
    return player_batting, players


def _build_lineup_templates(games: pd.DataFrame, existing: pd.DataFrame, snapshot_ts: str) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=LINEUP_COLUMNS)
    existing = existing.copy()
    if not existing.empty:
        existing["game_date"] = pd.to_datetime(existing["game_date"], errors="coerce").dt.date
    player_batting, players = _load_lineup_history(max(games["game_date"]) + timedelta(days=1))
    covered_keys = set(zip(existing.get("game_date", pd.Series(dtype=object)), existing.get("team", pd.Series(dtype=object))))
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        for team in (game.away_team, game.home_team):
            key = (game.game_date, team)
            if key in covered_keys:
                continue
            inferred = infer_lineup_from_history(team, game.game_date, player_batting, players)
            if inferred.empty:
                continue
            for row in inferred.itertuples(index=False):
                rows.append(
                    {
                        "game_id": int(game.game_id),
                        "game_date": game.game_date.isoformat(),
                        "team": team,
                        "player_id": int(row.player_id),
                        "player_name": row.player_name,
                        "lineup_slot": int(row.lineup_slot),
                        "position": None,
                        "confirmed": False,
                        "source_name": "projected_template",
                        "source_url": None,
                        "snapshot_ts": snapshot_ts,
                    }
                )
    generated = pd.DataFrame(rows, columns=LINEUP_COLUMNS)
    if existing.empty:
        return generated
    return pd.concat([existing[LINEUP_COLUMNS], generated], ignore_index=True)


def build_lineup_input_frame(
    games: pd.DataFrame,
    existing: pd.DataFrame | None = None,
    snapshot_ts: str | None = None,
) -> pd.DataFrame:
    existing_frame = pd.DataFrame(columns=LINEUP_COLUMNS) if existing is None else existing.copy()
    resolved_snapshot_ts = snapshot_ts or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return _build_lineup_templates(games, existing_frame, resolved_snapshot_ts)


def _build_market_templates(
    games: pd.DataFrame,
    starters: pd.DataFrame,
    lineups: pd.DataFrame,
    existing: pd.DataFrame,
    snapshot_ts: str,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    existing = existing.copy()
    if not existing.empty:
        existing["game_date"] = pd.to_datetime(existing["game_date"], errors="coerce").dt.date
    existing_keys = {
        (
            int(game_id) if pd.notna(game_id) else None,
            str(market_type or "").strip().lower(),
            int(player_id) if pd.notna(player_id) else None,
        )
        for game_id, market_type, player_id in zip(
            existing.get("game_id", pd.Series(dtype=object)),
            existing.get("market_type", pd.Series(dtype=object)),
            existing.get("player_id", pd.Series(dtype=object)),
        )
    }
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        total_key = (int(game.game_id), "total", None)
        if total_key not in existing_keys:
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date.isoformat(),
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "team": None,
                    "player_id": None,
                    "player_name": None,
                    "sportsbook": "manual",
                    "market_type": "total",
                    "line_value": None,
                    "over_price": None,
                    "under_price": None,
                    "snapshot_ts": snapshot_ts,
                    "is_opening": False,
                    "is_closing": False,
                    "source_name": "manual_template",
                }
            )

        first5_total_key = (int(game.game_id), FIRST5_TOTAL_MARKET_TYPE, None)
        if first5_total_key not in existing_keys:
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date.isoformat(),
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "team": None,
                    "player_id": None,
                    "player_name": None,
                    "sportsbook": "manual",
                    "market_type": FIRST5_TOTAL_MARKET_TYPE,
                    "line_value": None,
                    "over_price": None,
                    "under_price": None,
                    "snapshot_ts": snapshot_ts,
                    "is_opening": False,
                    "is_closing": False,
                    "source_name": "manual_template",
                }
            )

        game_starters = starters[starters["game_id"] == int(game.game_id)] if not starters.empty else pd.DataFrame()
        for starter in game_starters.itertuples(index=False):
            prop_key = (int(game.game_id), "pitcher_strikeouts", int(starter.pitcher_id))
            if prop_key in existing_keys:
                continue
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date.isoformat(),
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "team": starter.team,
                    "player_id": int(starter.pitcher_id),
                    "player_name": starter.pitcher_name,
                    "sportsbook": "manual",
                    "market_type": "pitcher_strikeouts",
                    "line_value": None,
                    "over_price": None,
                    "under_price": None,
                    "snapshot_ts": snapshot_ts,
                    "is_opening": False,
                    "is_closing": False,
                    "source_name": "manual_template",
                }
            )

        game_lineups = lineups[lineups["game_id"] == int(game.game_id)] if not lineups.empty else pd.DataFrame()
        if not game_lineups.empty:
            for hitter in game_lineups.itertuples(index=False):
                if pd.isna(getattr(hitter, "player_id", None)):
                    continue
                prop_key = (int(game.game_id), HITTER_PROP_MARKET_TYPE, int(hitter.player_id))
                if prop_key in existing_keys:
                    continue
                rows.append(
                    {
                        "game_id": int(game.game_id),
                        "game_date": game.game_date.isoformat(),
                        "home_team": game.home_team,
                        "away_team": game.away_team,
                        "team": getattr(hitter, "team", None),
                        "player_id": int(hitter.player_id),
                        "player_name": getattr(hitter, "player_name", None),
                        "sportsbook": "manual",
                        "market_type": HITTER_PROP_MARKET_TYPE,
                        "line_value": HITTER_PROP_LINE_VALUE,
                        "over_price": None,
                        "under_price": None,
                        "snapshot_ts": snapshot_ts,
                        "is_opening": False,
                        "is_closing": False,
                        "source_name": "manual_template",
                    }
                )
    generated = pd.DataFrame(rows, columns=MARKET_COLUMNS)
    if existing.empty:
        return generated
    if generated.empty:
        return existing[MARKET_COLUMNS].copy()
    return pd.concat(
        [existing[MARKET_COLUMNS].astype(object), generated.astype(object)],
        ignore_index=True,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate lineup and market template CSVs for the current slate")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    settings = get_settings()
    games = _load_games(start_date, end_date)
    if games.empty:
        log.info("No games found for %s to %s; nothing to template", start_date, end_date)
        return 0

    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lineups_existing = _read_csv(settings.manual_lineups_csv, LINEUP_COLUMNS)
    markets_existing = _read_csv(settings.manual_markets_csv, MARKET_COLUMNS)
    starters = _load_starting_pitchers(start_date, end_date)

    lineup_frame = build_lineup_input_frame(games, lineups_existing, snapshot_ts)
    market_frame = _build_market_templates(games, starters, lineup_frame, markets_existing, snapshot_ts)

    _write_csv(lineup_frame, settings.manual_lineups_csv)
    _write_csv(market_frame, settings.manual_markets_csv)

    log.info(
        "Prepared slate input templates for %s to %s: %s lineup rows in %s, %s market rows in %s",
        start_date,
        end_date,
        len(lineup_frame),
        settings.manual_lineups_csv,
        len(market_frame),
        settings.manual_markets_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())