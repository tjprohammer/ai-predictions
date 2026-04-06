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
    assert app_module._sql_year_param("target_date", dialect="sqlite") == "CAST(strftime('%Y', :target_date) AS INTEGER)"
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


def test_summarize_team_lineup_prefers_confirmed_rows_when_available():
    summary = app_module._summarize_team_lineup(
        [
            {
                "player_name": "Confirmed One",
                "is_confirmed_lineup": True,
                "has_lineup_snapshot": True,
                "is_inferred_lineup": False,
                "lineup_source_name": "mlb_statsapi",
            },
            {
                "player_name": "Projected Extra",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": False,
                "is_inferred_lineup": True,
                "lineup_source_name": None,
            },
        ]
    )

    assert summary["lineup_scope"] == "confirmed"
    assert [row["player_name"] for row in summary["lineup"]] == ["Confirmed One"]
    assert summary["lineup_source_summary"] == "Confirmed lineup"
    assert summary["lineup_counts"] == {
        "total_rows": 2,
        "displayed_rows": 1,
        "confirmed_rows": 1,
        "snapshot_rows": 1,
        "inferred_rows": 1,
    }


def test_summarize_team_lineup_prefers_snapshot_rows_before_inferred_rows():
    summary = app_module._summarize_team_lineup(
        [
            {
                "player_name": "Snapshot One",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": True,
                "is_inferred_lineup": False,
                "lineup_source_name": "rotowire",
            },
            {
                "player_name": "Projected Extra",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": False,
                "is_inferred_lineup": True,
                "lineup_source_name": "projected_template",
            },
        ]
    )

    assert summary["lineup_scope"] == "snapshot"
    assert [row["player_name"] for row in summary["lineup"]] == ["Snapshot One"]
    assert summary["lineup_source_summary"] == "Rotowire lineup snapshot"


def test_fetch_pitcher_recent_starts_returns_prior_outings(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-03",
                "game_id": 100,
                "team": "SEA",
                "ip": 6.0,
                "strikeouts": 8,
                "pitch_count": 94,
                "opponent": "LAA",
            },
            {
                "game_date": "2026-03-28",
                "game_id": 90,
                "team": "SEA",
                "ip": 5.1,
                "strikeouts": 6,
                "pitch_count": 88,
                "opponent": "CLE",
            },
        ]
    )

    monkeypatch.setattr(app_module, "_table_exists", lambda name: name in {"pitcher_starts", "games"})
    monkeypatch.setattr(app_module, "_safe_frame", lambda *_args, **_kwargs: frame)

    rows = app_module._fetch_pitcher_recent_starts(123, app_module.date(2026, 4, 4), limit=5)

    assert len(rows) == 2
    assert rows[0]["game_date"] == "2026-04-03"
    assert rows[0]["strikeouts"] == 8
    assert rows[0]["pitch_count"] == 94
    assert rows[1]["opponent"] == "CLE"