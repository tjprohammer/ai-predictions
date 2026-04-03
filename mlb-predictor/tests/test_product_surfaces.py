from datetime import date

import pandas as pd

from src.transforms import product_surfaces


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
                "model_name": "totals",
                "model_version": "v1",
                "prediction_ts": pd.Timestamp("2026-04-02T12:00:00Z"),
                "predicted_total": 8.5,
                "actual_total_runs": 9,
                "market_total": 8.0,
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
    monkeypatch.setattr(product_surfaces, "delete_for_date_range", lambda *args, **kwargs: None)

    def fake_upsert_rows(table_name, rows, conflict_columns):
        captured["table_name"] = table_name
        captured["rows"] = rows
        return len(rows)

    monkeypatch.setattr(product_surfaces, "upsert_rows", fake_upsert_rows)

    result = product_surfaces._build_prediction_outcomes(date(2026, 4, 2), date(2026, 4, 2))

    assert result == 1
    assert captured["table_name"] == "prediction_outcomes_daily"
    assert captured["rows"][0]["game_date"] == date(2026, 4, 2)


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