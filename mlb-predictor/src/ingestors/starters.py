from __future__ import annotations

import argparse
from datetime import date, datetime

from src.ingestors.common import iter_schedule_games, player_dimension_row, team_dimension_row
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _official_game_date(game: dict[str, object], game_start_ts: datetime) -> date:
    official_date = game.get("officialDate")
    if official_date:
        return date.fromisoformat(str(official_date))
    return game_start_ts.date()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest probable starters from MLB StatsAPI")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = iter_schedule_games(start_date.isoformat(), end_date.isoformat())
    starter_rows: list[dict[str, object]] = []
    player_ids: set[int] = set()

    prior_starts = query_df(
        """
        SELECT pitcher_id, max(game_date) AS last_game_date
        FROM pitcher_starts
        GROUP BY pitcher_id
        """
    )
    prior_map = {
        int(row.pitcher_id): row.last_game_date.date() if hasattr(row.last_game_date, "date") else row.last_game_date
        for row in prior_starts.itertuples(index=False)
    }

    for game in games:
        game_start_ts = datetime.fromisoformat(str(game["gameDate"]).replace("Z", "+00:00"))
        game_date = _official_game_date(game, game_start_ts)
        home_team_id = int(game["teams"]["home"]["team"]["id"])
        away_team_id = int(game["teams"]["away"]["team"]["id"])
        home_team = team_dimension_row(home_team_id)["team_abbr"]
        away_team = team_dimension_row(away_team_id)["team_abbr"]
        for side in ("home", "away"):
            probable = game["teams"][side].get("probablePitcher")
            if not probable:
                continue
            pitcher_id = int(probable["id"])
            team_abbr = home_team if side == "home" else away_team
            last_start = prior_map.get(pitcher_id)
            days_rest = None if last_start is None else (game_date - last_start).days
            starter_rows.append(
                {
                    "game_id": int(game["gamePk"]),
                    "game_date": game_date,
                    "pitcher_id": pitcher_id,
                    "team": team_abbr,
                    "side": side,
                    "is_probable": True,
                    "days_rest": days_rest,
                }
            )
            player_ids.add(pitcher_id)

    player_rows = [player_dimension_row(player_id) for player_id in sorted(player_ids)]
    upsert_rows("dim_players", player_rows, ["player_id"])
    inserted = upsert_rows("pitcher_starts", starter_rows, ["game_id", "pitcher_id"])
    log.info("Upserted %s probable starter rows for %s to %s", inserted, start_date, end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())