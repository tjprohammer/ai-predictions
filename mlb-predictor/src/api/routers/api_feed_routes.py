"""Predictions, players, experiments, trends, and related JSON feeds."""
from __future__ import annotations

import src.api.app_logic as app_logic
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/predictions/totals")
def totals_predictions(target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today)) -> app_logic.JSONResponse:
    rows = app_logic._fetch_totals_predictions(target_date)
    return app_logic._json_response({"target_date": target_date.isoformat(), "rows": rows})


@router.get("/api/totals/board")
def totals_board(target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today)) -> app_logic.JSONResponse:
    return app_logic._json_response({"target_date": target_date.isoformat(), **app_logic._fetch_totals_board(target_date)})


@router.get("/api/predictions/hits")
def hit_predictions(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    limit: int = app_logic.Query(default=40, ge=1, le=200),
    min_probability: float = app_logic.Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = app_logic.Query(default=False),
    include_inferred: bool = app_logic.Query(default=True),
) -> app_logic.JSONResponse:
    rows = app_logic._fetch_hit_predictions(target_date, limit, min_probability, confirmed_only, include_inferred)
    return app_logic._json_response({"target_date": target_date.isoformat(), "rows": rows})


@router.get("/api/hot-hitters")
def hot_hitters(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    limit: int = app_logic.Query(default=60, ge=1, le=200),
    min_probability: float = app_logic.Query(default=0.35, ge=0.0, le=1.0),
    confirmed_only: bool = app_logic.Query(default=False),
    include_inferred: bool = app_logic.Query(default=True),
    streak_only: bool = app_logic.Query(default=False),
) -> app_logic.JSONResponse:
    payload = app_logic._fetch_hot_hitters(
        target_date,
        min_probability,
        confirmed_only,
        limit,
        include_inferred,
        streak_only,
    )
    return app_logic._json_response({"target_date": target_date.isoformat(), **payload})


@router.get("/api/players/search")
def player_search(
    q: str = app_logic.Query(default="", max_length=100),
    team: str = app_logic.Query(default="", max_length=10),
    position: str = app_logic.Query(default="", max_length=10),
    starting_only: bool = app_logic.Query(default=False),
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    limit: int = app_logic.Query(default=25, ge=1, le=50),
) -> app_logic.JSONResponse:
    """Search non-pitcher players by name with optional lineup enrichment."""
    if not app_logic._table_exists("dim_players"):
        return app_logic._json_response({"players": []})
    safe_q = q.strip()
    safe_team = team.strip()
    safe_pos = position.strip()
    if not safe_q and not safe_team and not safe_pos and not starting_only:
        return app_logic._json_response({"players": []})
    clauses = [
        "dp.position IS NOT NULL",
        "dp.position != 'P'",
        "dp.active = 1",
    ]
    params: dict[str, object] = {"limit": limit, "target_date": target_date.isoformat()}
    if safe_q and len(safe_q) >= 2:
        clauses.append("dp.full_name LIKE :pattern")
        params["pattern"] = f"%{safe_q}%"
    if safe_team:
        clauses.append("dp.team_abbr = :team")
        params["team"] = safe_team
    if safe_pos:
        clauses.append("dp.position = :position")
        params["position"] = safe_pos
    has_lineups = app_logic._table_exists("lineups")
    has_games = app_logic._table_exists("games")
    has_pitcher_starts = app_logic._table_exists("pitcher_starts")
    if starting_only:
        if not has_lineups:
            return app_logic._json_response({"players": []})
        clauses.append("lu.player_id IS NOT NULL")
    where = " AND ".join(clauses)
    lineup_join = ""
    pitcher_join = ""
    lineup_cols = ", NULL AS lineup_slot, 0 AS is_confirmed, NULL AS game_id"
    game_cols = ", NULL AS opponent, NULL AS game_start_ts"
    pitcher_cols = ", NULL AS opposing_pitcher_name, NULL AS opposing_pitcher_throws"
    if has_lineups:
        lineup_join = """
        LEFT JOIN (
            SELECT player_id, lineup_slot, is_confirmed, game_id,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_ts DESC) AS rn
            FROM lineups
            WHERE game_date = :target_date
        ) lu ON lu.player_id = dp.player_id AND lu.rn = 1"""
        lineup_cols = ", lu.lineup_slot, COALESCE(lu.is_confirmed, 0) AS is_confirmed, lu.game_id"
    if has_lineups and has_games:
        game_cols = """, CASE WHEN dp.team_abbr = g.home_team THEN g.away_team
                             WHEN dp.team_abbr = g.away_team THEN g.home_team
                             ELSE NULL END AS opponent, g.game_start_ts"""
        lineup_join += """
        LEFT JOIN games g ON g.game_id = lu.game_id"""
    if has_lineups and has_games and has_pitcher_starts:
        pitcher_join = """
        LEFT JOIN (
            SELECT ps.game_id, ps.team, dp2.full_name AS pitcher_name, dp2.throws AS pitcher_throws
            FROM pitcher_starts ps
            JOIN dim_players dp2 ON dp2.player_id = ps.pitcher_id
            WHERE ps.game_date = :target_date
        ) opp_p ON opp_p.game_id = lu.game_id AND opp_p.team != dp.team_abbr"""
        pitcher_cols = ", opp_p.pitcher_name AS opposing_pitcher_name, opp_p.pitcher_throws AS opposing_pitcher_throws"
    order_clause = "dp.full_name"
    if has_lineups:
        order_clause = """
            CASE WHEN lu.player_id IS NOT NULL THEN 0 ELSE 1 END,
            lu.lineup_slot ASC,
            dp.full_name"""
    frame = app_logic._safe_frame(
        f"""
        SELECT dp.player_id, dp.full_name, dp.position, dp.team_abbr,
               dp.bats, dp.throws
               {lineup_cols}{game_cols}{pitcher_cols}
        FROM dim_players dp
        {lineup_join}
        {pitcher_join}
        WHERE {where}
        ORDER BY {order_clause}
        LIMIT :limit
        """,
        params,
    )
    return app_logic._json_response({"players": app_logic._frame_records(frame)})


@router.get("/api/players/{player_id}/recent-stats")
def player_recent_stats(
    player_id: int,
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    limit: int = app_logic.Query(default=10, ge=1, le=30),
) -> app_logic.JSONResponse:
    """Return recent game batting lines for a single player, with opposing pitcher hand."""
    if not app_logic._table_exists("player_game_batting"):
        return app_logic._json_response({"games": []})
    has_pitcher_starts = app_logic._table_exists("pitcher_starts")
    pitcher_join = ""
    pitcher_col = ", NULL AS opposing_pitcher_throws"
    if has_pitcher_starts:
        pitcher_join = """
        LEFT JOIN pitcher_starts ps
          ON ps.game_id = b.game_id AND ps.team = b.opponent
        LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id"""
        pitcher_col = ", dp.throws AS opposing_pitcher_throws"
    frame = app_logic._safe_frame(
        f"""
        SELECT
            b.game_date,
            b.opponent,
            b.hits,
            b.at_bats,
            b.home_runs,
            b.runs,
            b.rbi,
            b.walks,
            b.stolen_bases,
            b.strikeouts,
            (
                COALESCE(b.singles, 0)
                + 2 * COALESCE(b.doubles, 0)
                + 3 * COALESCE(b.triples, 0)
                + 4 * COALESCE(b.home_runs, 0)
            ) AS total_bases
            {pitcher_col}
        FROM player_game_batting b
        {pitcher_join}
        WHERE b.player_id = :player_id
          AND b.game_date <= :target_date
        ORDER BY b.game_date DESC, b.game_id DESC
        LIMIT :limit
        """,
        {"player_id": player_id, "target_date": target_date.isoformat(), "limit": limit},
    )
    return app_logic._json_response({"games": app_logic._frame_records(frame)})


@router.get("/api/results/daily")
def daily_results(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    hit_min_probability: float = app_logic.Query(default=0.35, ge=0.0, le=1.0),
    hitter_top_n: int = app_logic.Query(default=24, ge=1, le=200),
) -> app_logic.JSONResponse:
    return app_logic._json_response(
        {
            "target_date": target_date.isoformat(),
            **app_logic._fetch_daily_results(target_date, hit_min_probability, hitter_top_n),
        }
    )


@router.get("/api/experiments/summary")
def experiments_summary(
    window_days: int = app_logic.Query(default=14, ge=1, le=90),
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
) -> app_logic.JSONResponse:
    return app_logic._json_response(app_logic._fetch_experiment_summary(target_date, window_days))


@router.get("/api/experiments/daily-detail")
def experiments_daily_detail(
    target_date: app_logic.date = app_logic.Query(...),
) -> app_logic.JSONResponse:
    return app_logic._json_response(app_logic._fetch_experiment_daily_detail(target_date))


@router.get("/api/model-scorecards")
def model_scorecards(target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today), window_days: int = app_logic.Query(default=14, ge=1, le=60)) -> app_logic.JSONResponse:
    return app_logic._json_response({"target_date": target_date.isoformat(), **app_logic._fetch_model_scorecards(target_date, window_days)})


@router.get("/api/leaders/season")
def season_leaderboards(target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today), limit: int = app_logic.Query(default=10, ge=3, le=30)) -> app_logic.JSONResponse:
    return app_logic._json_response({"target_date": target_date.isoformat(), **app_logic._fetch_season_leaderboards(target_date, limit)})


@router.get("/api/trends/players/{player_id}")
def player_trend(player_id: int, target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today), limit: int = app_logic.Query(default=10, ge=1, le=30)) -> app_logic.JSONResponse:
    return app_logic._json_response({"target_date": target_date.isoformat(), "player_id": player_id, "rows": app_logic._fetch_player_trend(player_id, target_date, limit)})


@router.get("/api/trends/pitchers/{pitcher_id}")
def pitcher_trend(pitcher_id: int, target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today), limit: int = app_logic.Query(default=10, ge=1, le=30)) -> app_logic.JSONResponse:
    return app_logic._json_response({"target_date": target_date.isoformat(), "pitcher_id": pitcher_id, "rows": app_logic._fetch_pitcher_trend(pitcher_id, target_date, limit)})


@router.get("/api/pitchers/{pitcher_id}/recent-starts")
def pitcher_recent_starts(pitcher_id: int, target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today), limit: int = app_logic.Query(default=5, ge=1, le=15)) -> app_logic.JSONResponse:
    return app_logic._json_response(
        {
            "target_date": target_date.isoformat(),
            "pitcher_id": pitcher_id,
            "rows": app_logic._fetch_pitcher_recent_starts(pitcher_id, target_date, limit),
        }
    )
