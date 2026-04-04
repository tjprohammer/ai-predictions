import importlib

import pandas as pd


app_module = importlib.import_module("src.api.app")


def test_sql_bind_list_populates_named_parameters():
    params: dict[str, int] = {}

    placeholders = app_module._sql_bind_list("player_id", [17, 42], params)

    assert placeholders == ":player_id_0, :player_id_1"
    assert params == {"player_id_0": 17, "player_id_1": 42}


def test_sqlite_helper_fragments_use_portable_casts():
    assert app_module._sql_real("metric", dialect="sqlite") == "CAST(metric AS REAL)"
    assert app_module._sql_integer("slot", dialect="sqlite") == "CAST(slot AS INTEGER)"
    assert app_module._sql_year("game_date", dialect="sqlite") == "CAST(strftime('%Y', game_date) AS INTEGER)"
    assert app_module._sql_order_nulls_last("lineup_slot") == "CASE WHEN lineup_slot IS NULL THEN 1 ELSE 0 END, lineup_slot ASC"


def test_sqlite_boolean_helper_normalizes_text_values():
    fragment = app_module._sql_boolean("flag_text", dialect="sqlite")

    assert "'true'" in fragment
    assert "'false'" in fragment
    assert "THEN 1" in fragment
    assert "THEN 0" in fragment


def test_build_totals_review_block_surfaces_rationale_and_grading():
    detail = {
        "totals": {
            "predicted_total_runs": 9.1,
            "market_total": 8.0,
            "over_probability": 0.61,
            "under_probability": 0.39,
            "venue_run_factor": 1.05,
            "away_lineup_top5_xwoba": 0.344,
            "home_lineup_top5_xwoba": 0.331,
            "away_bullpen_pitches_last3": 92,
            "home_bullpen_pitches_last3": 88,
            "market_locked": True,
        },
        "weather": {"temperature_f": 82},
        "starters": {"away": {"name": "Away SP"}, "home": {"name": "Home SP"}},
        "actual_result": {"is_final": True, "total_runs": 10},
    }
    outcome = {
        "recommended_side": "over",
        "actual_side": "over",
        "graded": True,
        "beat_market": True,
        "beat_closing_line": True,
        "absolute_error": 0.9,
        "clv_side_value": 0.5,
        "weather_delta_temperature_f": 4.0,
    }

    review = app_module._build_totals_review_block(detail, outcome)

    assert review["rationale"]["direction"] == "over"
    assert review["rationale"]["signals"]
    assert review["grading"]["result"] == "won"
    assert review["grading"]["beat_market"] is True
    assert review["grading"]["beat_closing_line"] is True
    assert "Observed weather landed" in review["grading"]["weather_shift"]


def test_fetch_top_review_payload_sorts_losses_before_best_calls(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-03",
                "game_id": 1,
                "market": "totals",
                "recommended_side": "over",
                "actual_side": "under",
                "graded": True,
                "success": False,
                "beat_market": False,
                "probability": 0.62,
                "predicted_value": 9.5,
                "market_line": 8.5,
                "actual_value": 5.0,
                "absolute_error": 4.5,
                "weather_delta_temperature_f": -3.0,
                "closing_market_line": 9.0,
                "clv_line_delta": 0.5,
                "clv_side_value": 0.5,
                "beat_closing_line": True,
                "meta_payload": {"away_team": "STL", "home_team": "DET"},
            },
            {
                "game_date": "2026-04-03",
                "game_id": 2,
                "market": "totals",
                "recommended_side": "under",
                "actual_side": "over",
                "graded": True,
                "success": False,
                "beat_market": False,
                "probability": 0.58,
                "predicted_value": 7.2,
                "market_line": 8.0,
                "actual_value": 10.0,
                "absolute_error": 2.8,
                "weather_delta_temperature_f": None,
                "closing_market_line": 7.5,
                "clv_line_delta": -0.5,
                "clv_side_value": -0.5,
                "beat_closing_line": False,
                "meta_payload": {"away_team": "ATL", "home_team": "NYM"},
            },
            {
                "game_date": "2026-04-03",
                "game_id": 3,
                "market": "totals",
                "recommended_side": "over",
                "actual_side": "over",
                "graded": True,
                "success": True,
                "beat_market": True,
                "probability": 0.64,
                "predicted_value": 9.0,
                "market_line": 8.0,
                "actual_value": 11.0,
                "absolute_error": 2.0,
                "weather_delta_temperature_f": 1.0,
                "closing_market_line": 8.5,
                "clv_line_delta": 0.5,
                "clv_side_value": 0.5,
                "beat_closing_line": True,
                "meta_payload": {"away_team": "LAD", "home_team": "SF"},
            },
        ]
    )

    monkeypatch.setattr(app_module, "_table_exists", lambda _name: True)
    monkeypatch.setattr(app_module, "_safe_frame", lambda *_args, **_kwargs: frame)

    payload = app_module._fetch_top_review_payload(app_module.date(2026, 4, 3), limit=2)

    assert payload["summary"]["graded_games"] == 3
    assert payload["misses"][0]["game_id"] == 1
    assert payload["best_calls"][0]["game_id"] == 3


def test_fetch_clv_review_payload_sorts_best_and_worst(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-03",
                "game_id": 10,
                "recommended_side": "over",
                "actual_side": "over",
                "success": True,
                "beat_market": True,
                "market_line": 8.0,
                "closing_market_line": 9.0,
                "clv_line_delta": 1.0,
                "clv_side_value": 1.0,
                "beat_closing_line": True,
                "meta_payload": {"away_team": "SEA", "home_team": "HOU"},
            },
            {
                "game_date": "2026-04-03",
                "game_id": 11,
                "recommended_side": "under",
                "actual_side": "under",
                "success": True,
                "beat_market": False,
                "market_line": 8.5,
                "closing_market_line": 7.5,
                "clv_line_delta": -1.0,
                "clv_side_value": 1.0,
                "beat_closing_line": True,
                "meta_payload": {"away_team": "CHC", "home_team": "MIL"},
            },
            {
                "game_date": "2026-04-03",
                "game_id": 12,
                "recommended_side": "over",
                "actual_side": "under",
                "success": False,
                "beat_market": False,
                "market_line": 8.5,
                "closing_market_line": 7.5,
                "clv_line_delta": -1.0,
                "clv_side_value": -1.0,
                "beat_closing_line": False,
                "meta_payload": {"away_team": "NYY", "home_team": "BOS"},
            },
        ]
    )

    monkeypatch.setattr(app_module, "_table_exists", lambda _name: True)
    monkeypatch.setattr(app_module, "_safe_frame", lambda *_args, **_kwargs: frame)

    payload = app_module._fetch_clv_review_payload(app_module.date(2026, 4, 3), app_module.date(2026, 4, 3), limit=2)

    assert payload["summary"]["rows"] == 3
    assert payload["summary"]["positive_clv_rate"] == 2 / 3
    assert payload["best_clv"][0]["clv_side_value"] == 1.0
    assert payload["worst_clv"][0]["game_id"] == 12