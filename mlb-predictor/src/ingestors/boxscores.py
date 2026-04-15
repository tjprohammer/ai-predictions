from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from src.ingestors.common import record_ingest_event, statsapi_get
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)

LATE_INNING_HIT_EVENTS = {"single", "double", "triple", "home_run"}


def _parse_baseball_innings(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _baseball_ip_from_outs(outs: int) -> float:
    whole, remainder = divmod(int(outs), 3)
    return float(f"{whole}.{remainder}")


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


def _first_inning_team_runs(linescore: dict[str, object] | None) -> tuple[int | None, int | None]:
    """Runs scored in the top and bottom of the 1st inning (NRFI/YRFI actuals).

    Returns (None, None) if the first inning is not complete in the linescore
    (e.g. suspended before the home half).
    """
    innings = (linescore or {}).get("innings") or []
    if len(innings) < 1:
        return None, None
    inning = innings[0] or {}
    away_half = (inning.get("away") or {})
    home_half = (inning.get("home") or {})
    away_inning_runs = away_half.get("runs")
    home_inning_runs = home_half.get("runs")
    if away_inning_runs is None or home_inning_runs is None:
        return None, None
    return int(away_inning_runs), int(home_inning_runs)


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


def _late_bullpen_stats_by_pitcher(
    feed: dict[str, Any],
    home_team: str,
    away_team: str,
    starter_ids_by_team: dict[str, int | None],
) -> dict[int, dict[str, int | float]]:
    all_plays = (((feed.get("liveData") or {}).get("plays") or {}).get("allPlays")) or []
    pitcher_stats: dict[int, dict[str, int]] = {}
    previous_home_score = 0
    previous_away_score = 0

    for play in all_plays:
        result = play.get("result") or {}
        home_score = int(result.get("homeScore") or previous_home_score or 0)
        away_score = int(result.get("awayScore") or previous_away_score or 0)

        about = play.get("about") or {}
        inning = about.get("inning")
        half_inning = str(about.get("halfInning") or "").strip().lower()
        pitcher_id = ((play.get("matchup") or {}).get("pitcher") or {}).get("id")

        defensive_team = None
        if half_inning == "top":
            defensive_team = home_team
        elif half_inning == "bottom":
            defensive_team = away_team

        if (
            defensive_team is not None
            and pitcher_id is not None
            and inning is not None
            and int(inning) >= 7
            and int(pitcher_id) != starter_ids_by_team.get(defensive_team)
        ):
            runners = play.get("runners") or []
            outs_recorded = sum(
                1
                for runner in runners
                if bool((runner.get("movement") or {}).get("isOut"))
            )
            if outs_recorded == 0 and bool(result.get("isOut")):
                outs_recorded = 1

            runs_allowed = sum(
                1
                for runner in runners
                if bool((runner.get("details") or {}).get("isScoringEvent"))
            )
            earned_runs = sum(
                1
                for runner in runners
                if bool((runner.get("details") or {}).get("isScoringEvent"))
                and bool((runner.get("details") or {}).get("earned"))
            )
            if runs_allowed == 0:
                runs_allowed = max(home_score - previous_home_score, 0) + max(away_score - previous_away_score, 0)

            event_type = str(result.get("eventType") or "").strip().lower()
            hits_allowed = 1 if event_type in LATE_INNING_HIT_EVENTS else 0

            bucket = pitcher_stats.setdefault(
                int(pitcher_id),
                {
                    "late_outs_recorded": 0,
                    "late_runs_allowed": 0,
                    "late_earned_runs": 0,
                    "late_hits_allowed": 0,
                },
            )
            bucket["late_outs_recorded"] += int(outs_recorded)
            bucket["late_runs_allowed"] += int(runs_allowed)
            bucket["late_earned_runs"] += int(earned_runs)
            bucket["late_hits_allowed"] += int(hits_allowed)

        previous_home_score = home_score
        previous_away_score = away_score

    return {
        pitcher_id: {
            "late_innings_pitched": _baseball_ip_from_outs(int(stats["late_outs_recorded"])),
            "late_runs_allowed": int(stats["late_runs_allowed"]),
            "late_earned_runs": int(stats["late_earned_runs"]),
            "late_hits_allowed": int(stats["late_hits_allowed"]),
        }
        for pitcher_id, stats in pitcher_stats.items()
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
        starter_ids_by_team: dict[str, int | None] = {}
        for side, team_abbr in (("home", game.home_team), ("away", game.away_team)):
            players = ((teams_box.get(side) or {}).get("players") or {}).values()
            starter_id = next(
                (
                    int((player.get("person") or {}).get("id"))
                    for player in players
                    if int((((player.get("stats") or {}).get("pitching") or {}).get("gamesStarted", 0) or 0)) > 0
                ),
                None,
            )
            starter_ids_by_team[team_abbr] = starter_id
        late_bullpen_stats = _late_bullpen_stats_by_pitcher(
            feed,
            game.home_team,
            game.away_team,
            starter_ids_by_team,
        )

        home_score = home_box.get("teamStats", {}).get("batting", {}).get("runs")
        away_score = away_box.get("teamStats", {}).get("batting", {}).get("runs")
        linescore = live_data.get("linescore") or {}
        if home_score is None or away_score is None:
            teams_line = linescore.get("teams") or {}
            home_score = (teams_line.get("home") or {}).get("runs")
            away_score = (teams_line.get("away") or {}).get("runs")
        home_runs_first5, away_runs_first5 = _first_five_team_runs(linescore)
        away_runs_inning1, home_runs_inning1 = _first_inning_team_runs(linescore)
        total_runs_inning1 = (
            None
            if away_runs_inning1 is None or home_runs_inning1 is None
            else int(away_runs_inning1) + int(home_runs_inning1)
        )

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
                "home_runs_inning1": home_runs_inning1,
                "away_runs_inning1": away_runs_inning1,
                "total_runs_inning1": total_runs_inning1,
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
                late_stats = late_bullpen_stats.get(int(pitching_row["player_id"]), {})
                pitching_row.update(
                    {
                        "late_innings_pitched": late_stats.get("late_innings_pitched", 0.0),
                        "late_runs_allowed": late_stats.get("late_runs_allowed", 0),
                        "late_earned_runs": late_stats.get("late_earned_runs", 0),
                        "late_hits_allowed": late_stats.get("late_hits_allowed", 0),
                    }
                )
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
    record_ingest_event(
        source_name="mlb_statsapi",
        ingestor_module="src.ingestors.boxscores",
        target_date=start_date.isoformat(),
        row_count=len(game_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())