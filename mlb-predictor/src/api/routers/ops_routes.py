"""Readiness, review, source health, calibration, and recommendations."""
from __future__ import annotations

import src.api.app_logic as app_logic
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/game-readiness")
def game_readiness(target_date: app_logic.date | None = None) -> app_logic.JSONResponse:
    """Return the latest validation readiness for each game on the target date."""
    return app_logic._json_response(app_logic._fetch_game_readiness_payload(target_date))


@router.get("/api/review/top-misses")
def review_top_misses(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    market: str = "totals",
    limit: int = app_logic.Query(default=6, ge=1, le=20),
) -> app_logic.JSONResponse:
    """Return the biggest misses and best calls for the selected date."""
    return app_logic._json_response(app_logic._fetch_top_review_payload(target_date, market=market, limit=limit))


@router.get("/api/review/clv")
def review_clv(
    start_date: app_logic.date | None = None,
    end_date: app_logic.date | None = None,
    market: str = "totals",
    limit: int = app_logic.Query(default=8, ge=1, le=25),
) -> app_logic.JSONResponse:
    """Return best and worst closing-line-value rows over a date range."""
    sd = start_date or (app_logic.date.today() - app_logic.timedelta(days=7))
    ed = end_date or app_logic.date.today()
    return app_logic._json_response(app_logic._fetch_clv_review_payload(sd, ed, market=market, limit=limit))


@router.get("/api/source-health")
def source_health(hours: int = 24) -> app_logic.JSONResponse:
    """Return recent source health checks grouped by source."""
    return app_logic._json_response(app_logic._fetch_source_health_payload(hours))


@router.get("/api/calibration")
def calibration_bins(
    market: str = "totals",
    start_date: app_logic.date | None = None,
    end_date: app_logic.date | None = None,
) -> app_logic.JSONResponse:
    """Return calibration bins for a market over a date range."""
    if not app_logic._table_exists("prediction_calibration_bins"):
        return app_logic._json_response({"bins": []})
    sd = str(start_date or (app_logic.date.today() - app_logic.timedelta(days=30)))
    ed = str(end_date or app_logic.date.today())
    frame = app_logic._safe_frame(
        """
        SELECT bin_label, bin_lower, bin_upper,
               SUM(count) AS count,
               CASE WHEN SUM(count) > 0
                    THEN CAST(SUM(actual_hit_rate * count) AS DOUBLE PRECISION) / SUM(count)
                    ELSE NULL END AS actual_hit_rate,
               CASE WHEN SUM(count) > 0
                    THEN CAST(SUM(mean_predicted_prob * count) AS DOUBLE PRECISION) / SUM(count)
                    ELSE NULL END AS mean_predicted_prob,
               SUM(brier_score_sum) AS brier_score_sum
        FROM prediction_calibration_bins
        WHERE market = :market
          AND score_date BETWEEN :start_date AND :end_date
        GROUP BY bin_label, bin_lower, bin_upper
        ORDER BY bin_lower
        """,
        {"market": market, "start_date": sd, "end_date": ed},
    )
    return app_logic._json_response({"bins": app_logic._frame_records(frame), "market": market, "start_date": sd, "end_date": ed})


@router.get("/api/recommendations/history")
def recommendation_history_endpoint(
    market: str | None = None,
    start_date: app_logic.date | None = None,
    end_date: app_logic.date | None = None,
    graded_only: bool = False,
    limit: int = 50,
) -> app_logic.JSONResponse:
    """Return recommendation history with optional filters."""
    if not app_logic._table_exists("recommendation_history"):
        return app_logic._json_response({"recommendations": []})
    sd = str(start_date or (app_logic.date.today() - app_logic.timedelta(days=7)))
    ed = str(end_date or app_logic.date.today())
    conditions = ["game_date BETWEEN :start_date AND :end_date"]
    params: dict[str, app_logic.Any] = {"start_date": sd, "end_date": ed, "lim": max(1, min(limit, 200))}
    if market:
        conditions.append("market = :market")
        params["market"] = market
    if graded_only:
        conditions.append("graded = TRUE")
    where = " AND ".join(conditions)
    frame = app_logic._safe_frame(
        f"""
        SELECT game_date, game_id, market, entity_type, entity_id, player_id,
               team, away_team, home_team, model_name, model_version,
               recommended_side, probability, market_line, predicted_value,
               entry_market_sportsbook, closing_market_sportsbook, closing_market_same_sportsbook,
               closing_market_line, clv_line_delta, clv_side_value, beat_closing_line,
               actual_value, actual_side, graded, success, edge
        FROM recommendation_history
        WHERE {where}
        ORDER BY game_date DESC, game_id
        LIMIT :lim
        """,
        params,
    )
    records = app_logic._frame_records(frame)
    summary = {
        "total": len(records),
        "graded": sum(1 for r in records if r.get("graded")),
        "wins": sum(1 for r in records if r.get("success") is True),
        "losses": sum(1 for r in records if r.get("success") is False),
    }
    return app_logic._json_response({"recommendations": records, "summary": summary})


@router.get("/api/recommendations/best-bets-history")
def best_bet_history_endpoint(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    window_days: int = app_logic.Query(default=14, ge=1, le=60),
    limit: int = app_logic.Query(default=12, ge=1, le=100),
    graded_only: bool = app_logic.Query(default=False),
) -> app_logic.JSONResponse:
    return app_logic._json_response(
        app_logic._fetch_best_bet_history_payload(
            target_date,
            window_days=window_days,
            limit=limit,
            graded_only=graded_only,
        )
    )

