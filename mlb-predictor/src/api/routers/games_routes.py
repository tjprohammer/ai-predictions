"""Game board, per-game detail, and matchup splits."""
from __future__ import annotations

from typing import Any

import src.api.app_logic as app_logic
from fastapi import APIRouter

from src.utils.matchup_keys import team_abbr_to_opponent_id

from src.api.constants import (
    MATCHUP_BVP_ADEQUATE_MIN_AB,
    MATCHUP_BVP_STRONG_MIN_AB,
    MATCHUP_PLATOON_ADEQUATE_MIN_PA,
    MATCHUP_PLATOON_STRONG_MIN_PA,
    MATCHUP_PVT_ADEQUATE_MIN_IP,
    MATCHUP_PVT_STRONG_MIN_IP,
)

router = APIRouter()


def _tier_three_way(value: float | int | None, adequate: float | int, strong: float | int) -> str:
    v = float(value or 0)
    if v < float(adequate):
        return "low"
    if v < float(strong):
        return "adequate"
    return "strong"


@router.get("/api/games/board")
def games_board(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    hit_limit_per_team: int = app_logic.Query(default=4, ge=1, le=9),
    min_probability: float = app_logic.Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = app_logic.Query(default=False),
    include_inferred: bool = app_logic.Query(default=True),
) -> app_logic.JSONResponse:
    rows = app_logic._fetch_game_board(target_date, hit_limit_per_team, min_probability, confirmed_only, include_inferred)
    green_pick_limit = app_logic._green_pick_board_limit(target_date)
    return app_logic._json_response(
        {
            "target_date": target_date.isoformat(),
            "summary": app_logic._summarize_board_rows(rows, target_date),
            "best_bets": app_logic._flatten_best_bets(rows),
            "watchlist_markets": app_logic._flatten_watchlist_markets(rows),
            "experimental_markets": app_logic._fetch_experimental_market_cards(target_date),
            "green_picks": app_logic._fetch_ai_pick_results(
                target_date,
                limit=green_pick_limit,
            ),
            "games": rows,
        }
    )


@router.get("/api/games/{game_id}/detail")
def game_detail(
    game_id: int,
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    include_inferred: bool = app_logic.Query(default=True),
) -> app_logic.JSONResponse:
    payload = app_logic._fetch_game_detail(game_id, target_date, include_inferred=include_inferred)
    if payload is None:
        return app_logic._json_response({"target_date": target_date.isoformat(), "game": None}, status_code=404)
    return app_logic._json_response({"target_date": target_date.isoformat(), "game": payload})


@router.get("/api/games/{game_id}/matchups")
def game_matchups(game_id: int) -> app_logic.JSONResponse:
    """Return batter-vs-pitcher and pitcher-vs-team matchup splits for a game."""

    # ── BvP: career + game-log fallback ─────────────────────────────
    # 1. Get lineup batters & opposing starters for this game
    roster_frame = app_logic._safe_frame(
        """
        WITH latest_lineup AS (
            SELECT player_id, team,
                   MAX(lineup_slot) AS lineup_slot
            FROM lineups
            WHERE game_id = :game_id
            GROUP BY player_id, team
        )
        SELECT l.player_id  AS batter_id,
               dp.full_name AS batter_name,
               l.lineup_slot,
               l.team       AS batter_team,
               ps.pitcher_id,
               dp2.full_name AS pitcher_name
        FROM latest_lineup l
        JOIN pitcher_starts ps ON ps.game_id = :game_id
                               AND ps.team != l.team
        LEFT JOIN dim_players dp  ON dp.player_id = l.player_id
        LEFT JOIN dim_players dp2 ON dp2.player_id = ps.pitcher_id
        ORDER BY l.team, l.lineup_slot
        """,
        {"game_id": game_id},
    )
    roster_rows = app_logic._frame_records(roster_frame)

    # 2. Look up game_date for the fallback function
    game_date_frame = app_logic._safe_frame(
        "SELECT game_date FROM games WHERE game_id = :game_id",
        {"game_id": game_id},
    )
    game_date_rows = app_logic._frame_records(game_date_frame)
    target_date = app_logic.date.fromisoformat(str(game_date_rows[0]["game_date"])) if game_date_rows else app_logic.date.today()

    # 3. Use app_logic._fetch_batter_vs_pitcher_map for career data + game-log fallback
    matchup_pairs = [(int(r["batter_id"]), int(r["pitcher_id"])) for r in roster_rows]
    bvp_map = app_logic._fetch_batter_vs_pitcher_map(target_date, matchup_pairs) if matchup_pairs else {}

    # 4. Merge BvP stats with roster metadata
    bvp_records: list[dict[str, app_logic.Any]] = []
    for r in roster_rows:
        key = (int(r["batter_id"]), int(r["pitcher_id"]))
        stats = bvp_map.get(key, {})
        ab = stats.get("at_bats", 0)
        bvp_records.append({
            "batter_id": r["batter_id"],
            "batter_name": r["batter_name"],
            "pitcher_id": r["pitcher_id"],
            "pitcher_name": r["pitcher_name"],
            "lineup_slot": r["lineup_slot"],
            "batter_team": r["batter_team"],
            "games": stats.get("games"),
            "plate_appearances": ab + stats.get("walks", 0) if ab else 0,
            "at_bats": ab,
            "hits": stats.get("hits", 0) if ab else None,
            "home_runs": stats.get("home_runs", 0) if ab else None,
            "walks": stats.get("walks", 0) if ab else None,
            "strikeouts": stats.get("strikeouts", 0) if ab else None,
            "rbi": stats.get("rbi", 0) if ab else None,
            "runs": stats.get("runs"),
            "doubles": stats.get("doubles", 0) if ab else None,
            "triples": stats.get("triples", 0) if ab else None,
            "batting_avg": stats.get("batting_avg") if ab else None,
            "obp": stats.get("obp") if ab else None,
            "slg": stats.get("slg") if ab else None,
            "ops": stats.get("ops") if ab else None,
            "source": stats.get("source", "none"),
            "sample_tier": _tier_three_way(ab, MATCHUP_BVP_ADEQUATE_MIN_AB, MATCHUP_BVP_STRONG_MIN_AB),
        })

    teams_frame = app_logic._safe_frame(
        "SELECT home_team, away_team FROM games WHERE game_id = :game_id",
        {"game_id": game_id},
    )
    team_rows = app_logic._frame_records(teams_frame)
    home_team = str(team_rows[0]["home_team"] or "") if team_rows else ""
    away_team = str(team_rows[0]["away_team"] or "") if team_rows else ""
    pvt_params: dict[str, Any] = {"game_id": game_id}
    if team_rows:
        # Home starter faces the away lineup; away starter faces the home lineup.
        pvt_params["oid_home_pitcher_faces"] = team_abbr_to_opponent_id(away_team)
        pvt_params["oid_away_pitcher_faces"] = team_abbr_to_opponent_id(home_team)
    else:
        pvt_params["oid_home_pitcher_faces"] = -1
        pvt_params["oid_away_pitcher_faces"] = -1

    pvt_frame = app_logic._safe_frame(
        """
        SELECT ms.player_id   AS pitcher_id,
               dp.full_name   AS pitcher_name,
               ps.team        AS pitcher_team,
               ms.games, ms.era, ms.strikeouts,
               ms.innings_pitched, ms.earned_runs,
               ms.walks, ms.hits,
               ms.whip, ms.k_per_9,
               ms.opponent_id
        FROM matchup_splits ms
        JOIN pitcher_starts ps ON ps.pitcher_id = ms.player_id
                                AND ps.game_id = :game_id
        JOIN games g ON g.game_id = ps.game_id
        LEFT JOIN dim_players dp ON dp.player_id = ms.player_id
        WHERE ms.split_type = 'pitcher_vs_team'
          AND ms.season = 0
          AND ms.opponent_id = CASE
            WHEN ps.team = g.home_team THEN :oid_home_pitcher_faces
            ELSE :oid_away_pitcher_faces
          END
        """,
        pvt_params,
    )

    # ── Platoon splits: only the relevant split for the opposing pitcher's hand ──
    platoon_frame = app_logic._safe_frame(
        """
        WITH latest_lineup AS (
            SELECT player_id, team,
                   MAX(lineup_slot) AS lineup_slot
            FROM lineups
            WHERE game_id = :game_id
            GROUP BY player_id, team
        )
        SELECT ms.player_id   AS batter_id,
               dp.full_name   AS batter_name,
               ms.split_type,
               ms.season,
               l.lineup_slot,
               l.team           AS batter_team,
               ms.games, ms.plate_appearances, ms.at_bats,
               ms.hits, ms.home_runs, ms.walks, ms.strikeouts,
               ms.rbi, ms.doubles, ms.triples,
               ms.batting_avg, ms.obp, ms.slg, ms.ops
        FROM matchup_splits ms
        JOIN latest_lineup l ON l.player_id = ms.player_id
        /* Restrict to the split matching the opposing starter's throwing hand */
        JOIN pitcher_starts ps ON ps.game_id = :game_id
                               AND ps.team != l.team
        JOIN dim_players dp_opp ON dp_opp.player_id = ps.pitcher_id
        LEFT JOIN dim_players dp ON dp.player_id = ms.player_id
        WHERE ms.plate_appearances > 0
          AND (  (dp_opp.throws = 'L' AND ms.split_type = 'platoon_lhp')
              OR (dp_opp.throws = 'R' AND ms.split_type = 'platoon_rhp'))
        ORDER BY l.team, l.lineup_slot
        """,
        {"game_id": game_id},
    )

    # ── Team head-to-head (shared with game detail) ────────────────────
    h2h: dict[str, Any] = app_logic._fetch_h2h_totals_for_game(game_id) or {}

    pvt_rows = app_logic._frame_records(pvt_frame)
    for row in pvt_rows:
        ip = row.get("innings_pitched")
        row["sample_tier"] = _tier_three_way(
            float(ip) if ip is not None else 0.0,
            MATCHUP_PVT_ADEQUATE_MIN_IP,
            MATCHUP_PVT_STRONG_MIN_IP,
        )
        pt = str(row.get("pitcher_team") or "")
        row["opponent_team_abbr"] = away_team if pt == home_team else home_team

    platoon_rows = app_logic._frame_records(platoon_frame)
    for row in platoon_rows:
        pa = row.get("plate_appearances")
        row["sample_tier"] = _tier_three_way(
            int(pa) if pa is not None else 0,
            MATCHUP_PLATOON_ADEQUATE_MIN_PA,
            MATCHUP_PLATOON_STRONG_MIN_PA,
        )

    return app_logic._json_response({
        "game_id": game_id,
        "batter_vs_pitcher": bvp_records,
        "pitcher_vs_team": pvt_rows,
        "platoon_splits": platoon_rows,
        "head_to_head": h2h,
        "sample_tier_legend": {
            "low": "Small sample — directional only; lean on process stats.",
            "adequate": "Usable context; still combine with xwOBA, lineup, park.",
            "strong": "Heavier sample — matchup history is more informative.",
        },
    })
