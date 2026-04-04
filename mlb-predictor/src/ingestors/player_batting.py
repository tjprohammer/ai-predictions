from __future__ import annotations

import argparse

from src.ingestors.common import player_dimension_row, record_ingest_event, statsapi_get
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _lineup_slot(player: dict[str, object]) -> int | None:
    batting_order = player.get("battingOrder")
    if not batting_order:
        return None
    batting_order_str = str(batting_order)
    if len(batting_order_str) >= 3:
        return int(batting_order_str[:-2])
    return int(batting_order_str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest player batting game logs from boxscores")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = query_df(
        """
        SELECT game_id, game_date, home_team, away_team
        FROM games
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if games.empty:
        log.info("No games available for batting ingest")
        return 0

    player_rows = []
    batting_rows = []
    for game in games.itertuples(index=False):
        feed = statsapi_get(f"/api/v1.1/game/{int(game.game_id)}/feed/live")
        teams_box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side, team_abbr, opponent, side_code in (
            ("home", game.home_team, game.away_team, "H"),
            ("away", game.away_team, game.home_team, "A"),
        ):
            players = (teams_box.get(side) or {}).get("players") or {}
            for player in players.values():
                person = player.get("person") or {}
                player_id = person.get("id")
                if not player_id:
                    continue
                batting = (player.get("stats") or {}).get("batting") or {}
                batting_order = player.get("battingOrder")
                has_batting_entry = bool(batting) or batting_order is not None
                if not has_batting_entry:
                    continue
                plate_appearances = int(batting.get("plateAppearances", 0) or 0)
                at_bats = int(batting.get("atBats", 0) or 0)
                hits = int(batting.get("hits", 0) or 0)
                walks = int(batting.get("baseOnBalls", 0) or 0)
                hbp = int(batting.get("hitByPitch", 0) or 0)
                doubles = int(batting.get("doubles", 0) or 0)
                triples = int(batting.get("triples", 0) or 0)
                home_runs = int(batting.get("homeRuns", 0) or 0)
                singles = hits - doubles - triples - home_runs
                positions = player.get("allPositions") or []
                player_rows.append(
                    player_dimension_row(
                        int(player_id),
                        full_name_override=person.get("fullName") or str(player_id),
                        team_abbr_override=team_abbr,
                        position_override=positions[0]["abbreviation"] if positions else None,
                    )
                )
                batting_rows.append(
                    {
                        "game_id": int(game.game_id),
                        "game_date": game.game_date,
                        "player_id": int(player_id),
                        "team": team_abbr,
                        "opponent": opponent,
                        "home_away": side_code,
                        "lineup_slot": _lineup_slot(player),
                        "plate_appearances": plate_appearances,
                        "at_bats": at_bats,
                        "hits": hits,
                        "singles": singles,
                        "doubles": doubles,
                        "triples": triples,
                        "home_runs": home_runs,
                        "runs": int(batting.get("runs", 0) or 0),
                        "rbi": int(batting.get("rbi", 0) or 0),
                        "walks": walks,
                        "strikeouts": int(batting.get("strikeOuts", 0) or 0),
                        "hbp": hbp,
                        "sac_flies": int(batting.get("sacFlies", 0) or 0),
                        "stolen_bases": int(batting.get("stolenBases", 0) or 0),
                    }
                )

    upsert_rows("dim_players", player_rows, ["player_id"])
    inserted = upsert_rows("player_game_batting", batting_rows, ["game_id", "player_id"])
    log.info("Upserted %s player batting rows", inserted)
    record_ingest_event(
        source_name="mlb_statsapi",
        ingestor_module="src.ingestors.player_batting",
        target_date=start_date.isoformat(),
        row_count=inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())