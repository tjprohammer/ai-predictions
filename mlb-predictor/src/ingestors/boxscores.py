from __future__ import annotations

import argparse
from datetime import datetime

from src.ingestors.common import statsapi_get
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _parse_baseball_innings(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _first_five_team_runs(linescore: dict[str, object] | None) -> tuple[int | None, int | None]:
    innings = (linescore or {}).get("innings") or []
    if len(innings) < 5:
        return None, None

    home_runs = 0
    away_runs = 0
    for inning in innings[:5]:
        away_half = (inning or {}).get("away") or {}
        home_half = (inning or {}).get("home") or {}
        away_inning_runs = away_half.get("runs")
        home_inning_runs = home_half.get("runs")
        if away_inning_runs is None or home_inning_runs is None:
            return None, None
        home_runs += int(home_inning_runs)
        away_runs += int(away_inning_runs)
    return home_runs, away_runs


def _player_row(
    player: dict[str, object],
    team: str,
    existing_players: dict[int, dict[str, object]],
) -> dict[str, object]:
    person = player.get("person") or {}
    positions = player.get("allPositions") or []
    player_id = int(person["id"])
    existing = existing_players.get(player_id) or {}
    return {
        "player_id": player_id,
        "full_name": existing.get("full_name") or person.get("fullName") or str(player_id),
        "first_name": existing.get("first_name") or person.get("firstName"),
        "last_name": existing.get("last_name") or person.get("lastName"),
        "bats": existing.get("bats"),
        "throws": existing.get("throws"),
        "position": existing.get("position") or (positions[0]["abbreviation"] if positions else None),
        "team_abbr": team or existing.get("team_abbr"),
        "active": existing.get("active") if existing.get("active") is not None else True,
    }


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

    existing_players_frame = query_df(
        """
        SELECT
            player_id,
            full_name,
            first_name,
            last_name,
            bats,
            throws,
            position,
            team_abbr,
            active
        FROM dim_players
        """
    )
    existing_players = {
        int(row.player_id): {
            "full_name": row.full_name,
            "first_name": row.first_name,
            "last_name": row.last_name,
            "bats": row.bats,
            "throws": row.throws,
            "position": row.position,
            "team_abbr": row.team_abbr,
            "active": row.active,
        }
        for row in existing_players_frame.itertuples(index=False)
        if row.player_id is not None
    }

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
        linescore = live_data.get("linescore") or {}
        if home_score is None or away_score is None:
            teams_line = linescore.get("teams") or {}
            home_score = (teams_line.get("home") or {}).get("runs")
            away_score = (teams_line.get("away") or {}).get("runs")
        home_runs_first5, away_runs_first5 = _first_five_team_runs(linescore)

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
                "home_runs_first5": home_runs_first5,
                "away_runs_first5": away_runs_first5,
                "total_runs_first5": None
                if home_runs_first5 is None or away_runs_first5 is None
                else int(home_runs_first5) + int(away_runs_first5),
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
                player_rows.append(_player_row(player, team_abbr, existing_players))
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
                            "batters_faced": pitching_row["batters_faced"],
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