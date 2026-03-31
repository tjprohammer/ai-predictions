from __future__ import annotations

import argparse
from datetime import datetime

from src.ingestors.common import player_dimension_row, statsapi_get
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _parse_baseball_innings(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _player_row(player: dict[str, object], team: str) -> dict[str, object]:
    person = player.get("person") or {}
    positions = player.get("allPositions") or []
    return player_dimension_row(
        int(person["id"]),
        full_name_override=person.get("fullName") or str(person["id"]),
        team_abbr_override=team,
        position_override=positions[0]["abbreviation"] if positions else None,
    )


def _pitching_row(game_id: int, game_date, team: str, opponent: str, side_code: str, player: dict[str, object]) -> dict[str, object] | None:
    person = player.get("person") or {}
    stats = (player.get("stats") or {}).get("pitching") or {}
    if not stats:
        return None
    player_id = int(person["id"])
    return {
        "game_id": game_id,
        "game_date": game_date,
        "player_id": player_id,
        "team": team,
        "opponent": opponent,
        "home_away": side_code,
        "is_starter": int(stats.get("gamesStarted", 0) or 0) > 0,
        "innings_pitched": _parse_baseball_innings(stats.get("inningsPitched")),
        "batters_faced": stats.get("battersFaced"),
        "hits_allowed": stats.get("hits"),
        "runs_allowed": stats.get("runs"),
        "earned_runs": stats.get("earnedRuns"),
        "walks": stats.get("baseOnBalls"),
        "strikeouts": stats.get("strikeOuts"),
        "home_runs_allowed": stats.get("homeRuns"),
        "pitches_thrown": stats.get("pitchesThrown"),
        "strikes_thrown": stats.get("strikes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest final or in-progress boxscore rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = query_df(
        """
        SELECT
            game_id,
            game_date,
            season,
            game_type,
            home_team,
            away_team,
            venue_id,
            venue_name,
            game_start_ts,
            resumed_from,
            double_header,
            innings
        FROM games
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if games.empty:
        log.info("No game rows available for %s to %s", start_date, end_date)
        return 0

    game_rows = []
    pitcher_rows = []
    player_rows = []
    starter_rows = []

    for game in games.itertuples(index=False):
        feed = statsapi_get(f"/api/v1.1/game/{int(game.game_id)}/feed/live")
        game_data = feed.get("gameData") or {}
        live_data = feed.get("liveData") or {}
        teams_box = (live_data.get("boxscore") or {}).get("teams") or {}
        status = ((game_data.get("status") or {}).get("abstractGameState") or "scheduled").lower()
        home_box = teams_box.get("home") or {}
        away_box = teams_box.get("away") or {}
        home_batting = (home_box.get("teamStats") or {}).get("batting") or {}
        away_batting = (away_box.get("teamStats") or {}).get("batting") or {}
        home_fielding = (home_box.get("teamStats") or {}).get("fielding") or {}
        away_fielding = (away_box.get("teamStats") or {}).get("fielding") or {}

        home_score = home_box.get("teamStats", {}).get("batting", {}).get("runs")
        away_score = away_box.get("teamStats", {}).get("batting", {}).get("runs")
        if home_score is None or away_score is None:
            linescore = live_data.get("linescore") or {}
            teams_line = linescore.get("teams") or {}
            home_score = (teams_line.get("home") or {}).get("runs")
            away_score = (teams_line.get("away") or {}).get("runs")

        game_rows.append(
            {
                "game_id": int(game.game_id),
                "game_date": game.game_date,
                "season": game.season,
                "game_type": game.game_type,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "venue_id": game.venue_id,
                "venue_name": game.venue_name,
                "game_start_ts": game.game_start_ts,
                "resumed_from": game.resumed_from,
                "double_header": game.double_header,
                "innings": game.innings,
                "status": status,
                "home_runs": home_score,
                "away_runs": away_score,
                "total_runs": None if home_score is None or away_score is None else int(home_score) + int(away_score),
                "home_hits": home_batting.get("hits"),
                "away_hits": away_batting.get("hits"),
                "home_errors": home_fielding.get("errors"),
                "away_errors": away_fielding.get("errors"),
                "home_win": None if home_score is None or away_score is None else int(home_score) > int(away_score),
            }
        )

        for side, team_abbr, opponent, side_code in (
            ("home", game.home_team, game.away_team, "H"),
            ("away", game.away_team, game.home_team, "A"),
        ):
            team_box = teams_box.get(side) or {}
            players = team_box.get("players") or {}
            for player in players.values():
                person = player.get("person") or {}
                if not person.get("id"):
                    continue
                player_rows.append(_player_row(player, team_abbr))
                pitching_row = _pitching_row(int(game.game_id), game.game_date, team_abbr, opponent, side_code, player)
                if not pitching_row:
                    continue
                pitcher_rows.append(pitching_row)
                if pitching_row["is_starter"]:
                    starter_rows.append(
                        {
                            "game_id": int(game.game_id),
                            "game_date": game.game_date,
                            "pitcher_id": pitching_row["player_id"],
                            "team": team_abbr,
                            "side": "home" if side == "home" else "away",
                            "is_probable": False,
                            "ip": pitching_row["innings_pitched"],
                            "hits_allowed": pitching_row["hits_allowed"],
                            "runs_allowed": pitching_row["runs_allowed"],
                            "earned_runs": pitching_row["earned_runs"],
                            "walks": pitching_row["walks"],
                            "strikeouts": pitching_row["strikeouts"],
                            "home_runs_allowed": pitching_row["home_runs_allowed"],
                            "pitch_count": pitching_row["pitches_thrown"],
                        }
                    )

    upsert_rows("dim_players", player_rows, ["player_id"])
    upsert_rows("games", game_rows, ["game_id"])
    upsert_rows("player_game_pitching", pitcher_rows, ["game_id", "player_id"])
    upsert_rows("pitcher_starts", starter_rows, ["game_id", "pitcher_id"])
    log.info("Updated %s games and %s pitching rows", len(game_rows), len(pitcher_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())