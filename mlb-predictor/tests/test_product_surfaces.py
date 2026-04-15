from datetime import date

import pandas as pd

from src.transforms import product_surfaces


def test_grade_experimental_first_inning():
    assert product_surfaces._grade_experimental_first_inning(
        game_status="Final",
        total_runs_inning1=0,
        line_value=0.5,
        market_key="nrfi",
    ) == (True, 0.0, "under", True)
    assert product_surfaces._grade_experimental_first_inning(
        game_status="Final",
        total_runs_inning1=1,
        line_value=0.5,
        market_key="nrfi",
    ) == (True, 1.0, "over", False)
    assert product_surfaces._grade_experimental_first_inning(
        game_status="Final",
        total_runs_inning1=2,
        line_value=0.5,
        market_key="yrfi",
    ) == (True, 2.0, "over", True)
    assert product_surfaces._grade_experimental_first_inning(
        game_status="Live",
        total_runs_inning1=0,
        line_value=0.5,
        market_key="nrfi",
    ) == (False, None, None, None)


def test_build_pitcher_trends_qualifies_order_by_columns(monkeypatch):
    captured: dict[str, str] = {}

    def fake_query_df(query: str, params=None):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(product_surfaces, "query_df", fake_query_df)

    result = product_surfaces._build_pitcher_trends(date(2026, 4, 2), date(2026, 4, 2))

    assert result == 0
    assert "ORDER BY ps.pitcher_id, ps.game_date, ps.game_id" in captured["query"]


def test_build_prediction_outcomes_normalizes_game_date(monkeypatch):
    totals_frame = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "game_id": 1,
                "away_team": "NYY",
                "home_team": "BOS",
                "game_status": "final",
                "model_name": "totals",
                "model_version": "v1",
                "prediction_ts": pd.Timestamp("2026-04-02T12:00:00Z"),
                "predicted_total_runs": 8.5,
                "total_runs": 9,
                "market_total": 8.0,
                "market_sportsbook": "FanDuel",
                "market_snapshot_ts": pd.Timestamp("2026-04-02T11:45:00Z"),
                "over_probability": 0.56,
                "under_probability": 0.44,
                "edge": 0.05,
            }
        ]
    )

    def fake_latest_rows(query: str, params=None):
        if "predictions_totals" in query:
            return totals_frame
        return pd.DataFrame()

    captured: dict[str, object] = {}

    monkeypatch.setattr(product_surfaces, "_latest_rows", fake_latest_rows)
    monkeypatch.setattr(
        product_surfaces,
        "_fetch_closing_market_by_game",
        lambda *args, **kwargs: {
            1: {
                "sportsbook": "FanDuel",
                "line_value": 8.5,
                "snapshot_ts": pd.Timestamp("2026-04-02T12:30:00Z"),
                "same_sportsbook": True,
            }
        },
    )
    monkeypatch.setattr(product_surfaces, "_fetch_weather_by_game", lambda *args, **kwargs: {})
    monkeypatch.setattr(product_surfaces, "delete_for_date_range", lambda *args, **kwargs: None)

    def fake_upsert_rows(table_name, rows, conflict_columns):
        captured["table_name"] = table_name
        captured["rows"] = rows
        return len(rows)

    monkeypatch.setattr(product_surfaces, "upsert_rows", fake_upsert_rows)
    monkeypatch.setattr(product_surfaces, "_build_best_bet_outcomes", lambda *_a, **_k: [])
    monkeypatch.setattr(product_surfaces, "_build_experimental_market_outcomes", lambda *_a, **_k: [])

    result = product_surfaces._build_prediction_outcomes(date(2026, 4, 2), date(2026, 4, 2))

    assert result == 1
    assert captured["table_name"] == "prediction_outcomes_daily"
    assert captured["rows"][0]["game_date"] == date(2026, 4, 2)
    assert captured["rows"][0]["entry_market_sportsbook"] == "FanDuel"
    assert captured["rows"][0]["closing_market_same_sportsbook"] is True


def test_build_prediction_outcomes_skips_live_games_for_grading(monkeypatch):
    totals_frame = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "game_id": 1,
                "away_team": "NYY",
                "home_team": "BOS",
                "game_status": "live",
                "model_name": "totals",
                "model_version": "v1",
                "prediction_ts": pd.Timestamp("2026-04-02T12:00:00Z"),
                "predicted_total_runs": 8.5,
                "total_runs": 5,
                "market_total": 8.0,
                "market_sportsbook": "FanDuel",
                "market_snapshot_ts": pd.Timestamp("2026-04-02T11:45:00Z"),
                "over_probability": 0.56,
                "under_probability": 0.44,
            }
        ]
    )
    hits_frame = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "game_id": 1,
                "player_id": 42,
                "team": "NYY",
                "opponent": "BOS",
                "game_status": "in progress",
                "model_name": "hits",
                "model_version": "v1",
                "prediction_ts": pd.Timestamp("2026-04-02T12:00:00Z"),
                "predicted_hit_probability": 0.61,
                "fair_price": -115,
                "market_price": -110,
                "edge": 0.02,
                "actual_hits": 2,
            }
        ]
    )
    strikeouts_frame = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "game_id": 1,
                "pitcher_id": 77,
                "team": "BOS",
                "opponent": "NYY",
                "game_status": "live",
                "model_name": "strikeouts",
                "model_version": "v1",
                "prediction_ts": pd.Timestamp("2026-04-02T12:00:00Z"),
                "predicted_strikeouts": 6.2,
                "market_line": 5.5,
                "over_probability": 0.58,
                "under_probability": 0.42,
                "edge": 0.03,
                "actual_strikeouts": 4,
            }
        ]
    )

    def fake_latest_rows(query: str, params=None):
        if "predictions_totals" in query:
            return totals_frame
        if "predictions_player_hits" in query:
            return hits_frame
        if "predictions_pitcher_strikeouts" in query:
            return strikeouts_frame
        return pd.DataFrame()

    captured: dict[str, object] = {}

    monkeypatch.setattr(product_surfaces, "_latest_rows", fake_latest_rows)
    monkeypatch.setattr(product_surfaces, "_fetch_closing_market_by_game", lambda *args, **kwargs: {})
    monkeypatch.setattr(product_surfaces, "_fetch_weather_by_game", lambda *args, **kwargs: {})
    monkeypatch.setattr(product_surfaces, "delete_for_date_range", lambda *args, **kwargs: None)

    def fake_upsert_rows(table_name, rows, conflict_columns):
        captured["table_name"] = table_name
        captured["rows"] = rows
        return len(rows)

    monkeypatch.setattr(product_surfaces, "upsert_rows", fake_upsert_rows)
    monkeypatch.setattr(product_surfaces, "_build_best_bet_outcomes", lambda *_a, **_k: [])
    monkeypatch.setattr(product_surfaces, "_build_experimental_market_outcomes", lambda *_a, **_k: [])

    result = product_surfaces._build_prediction_outcomes(date(2026, 4, 2), date(2026, 4, 2))

    assert result == 3
    assert captured["table_name"] == "prediction_outcomes_daily"
    assert len(captured["rows"]) == 3
    assert all(row["graded"] is False for row in captured["rows"])
    assert all(row["actual_value"] is None for row in captured["rows"])


def test_fetch_closing_market_by_game_prefers_entry_sportsbook(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_id": 1,
                "sportsbook": "DraftKings",
                "line_value": 8.5,
                "snapshot_ts": "2026-04-02T18:00:00Z",
                "source_name": "oddsapi",
                "over_price": -110,
                "under_price": -110,
                "game_start_ts": "2026-04-02T19:00:00Z",
            },
            {
                "game_id": 1,
                "sportsbook": "FanDuel",
                "line_value": 9.0,
                "snapshot_ts": "2026-04-02T17:30:00Z",
                "source_name": "oddsapi",
                "over_price": -108,
                "under_price": -112,
                "game_start_ts": "2026-04-02T19:00:00Z",
            },
        ]
    )

    monkeypatch.setattr(product_surfaces, "table_exists", lambda _name: True)
    monkeypatch.setattr(product_surfaces, "query_df", lambda *_args, **_kwargs: frame)

    result = product_surfaces._fetch_closing_market_by_game(
        date(2026, 4, 2),
        date(2026, 4, 2),
        preferred_sportsbook_by_game={1: "FanDuel"},
    )

    assert result[1]["sportsbook"] == "FanDuel"
    assert result[1]["line_value"] == 9.0
    assert result[1]["same_sportsbook"] is True


def test_fetch_closing_market_by_game_falls_back_to_game_markets(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_id": 2,
                "sportsbook": "DraftKings",
                "line_value": 8.0,
                "snapshot_ts": "2026-04-01T18:15:00Z",
                "source_name": "covers_html",
                "over_price": -110,
                "under_price": -110,
                "game_start_ts": "2026-04-01T19:00:00Z",
            },
            {
                "game_id": 2,
                "sportsbook": "FanDuel",
                "line_value": 8.5,
                "snapshot_ts": "2026-04-01T18:30:00Z",
                "source_name": "covers_html",
                "over_price": -108,
                "under_price": -112,
                "game_start_ts": "2026-04-01T19:00:00Z",
            },
        ]
    )

    monkeypatch.setattr(product_surfaces, "table_exists", lambda name: name == "game_markets")
    monkeypatch.setattr(product_surfaces, "query_df", lambda *_args, **_kwargs: frame)

    result = product_surfaces._fetch_closing_market_by_game(
        date(2026, 4, 1),
        date(2026, 4, 1),
        preferred_sportsbook_by_game={2: "FanDuel"},
    )

    assert result[2]["sportsbook"] == "FanDuel"
    assert result[2]["line_value"] == 8.5
    assert result[2]["same_sportsbook"] is True


def test_build_model_scorecards_normalizes_score_date(monkeypatch):
    scorecard_source = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp("2026-04-02"),
                "market": "totals",
                "model_name": "totals",
                "model_version": "v1",
                "graded": True,
                "success": True,
                "beat_market": True,
                "absolute_error": 0.5,
                "predicted_value": 8.5,
                "actual_value": 9.0,
                "brier_score": 0.2,
            }
        ]
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(product_surfaces, "query_df", lambda query, params=None: scorecard_source)
    monkeypatch.setattr(product_surfaces, "delete_for_date_range", lambda *args, **kwargs: None)

    def fake_upsert_rows(table_name, rows, conflict_columns):
        captured["table_name"] = table_name
        captured["rows"] = rows
        return len(rows)

    monkeypatch.setattr(product_surfaces, "upsert_rows", fake_upsert_rows)

    result = product_surfaces._build_model_scorecards(date(2026, 4, 2), date(2026, 4, 2))

    assert result == 1
    assert captured["table_name"] == "model_scorecards_daily"
    assert captured["rows"][0]["score_date"] == date(2026, 4, 2)