import importlib

import pandas as pd


app_module = importlib.import_module("src.api.app")
app_logic = importlib.import_module("src.api.app_logic")


def _patch_pair(monkeypatch, name: str, value: object) -> None:
    monkeypatch.setattr(app_module, name, value)
    monkeypatch.setattr(app_logic, name, value)


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

    _patch_pair(monkeypatch, "_table_exists", lambda _name: True)
    _patch_pair(monkeypatch, "_safe_frame", lambda *_args, **_kwargs: frame)

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

    _patch_pair(monkeypatch, "_table_exists", lambda _name: True)
    _patch_pair(monkeypatch, "_safe_frame", lambda *_args, **_kwargs: frame)

    payload = app_module._fetch_clv_review_payload(app_module.date(2026, 4, 3), app_module.date(2026, 4, 3), limit=2)

    assert payload["summary"]["rows"] == 3
    assert payload["summary"]["positive_clv_rate"] == 2 / 3
    assert payload["best_clv"][0]["clv_side_value"] == 1.0
    assert payload["worst_clv"][0]["game_id"] == 12


def test_scale_expected_run_split_keeps_first5_total_consistent():
    away, home = app_module._scale_expected_run_split(5.0737114168, 3.7666666667, 4.8)

    assert away == 2.231
    assert home == 2.843
    assert round(away + home, 3) == 5.074


def test_team_expected_runs_coherent_with_game_total_rescales_feature_rates():
    row = {
        "predicted_total_runs": 8.0,
        "predicted_total_fundamentals": 9.29,
        "away_expected_runs": 5.3,
        "home_expected_runs": 5.17,
    }
    away, home = app_module._team_expected_runs_coherent_with_game_total(row)

    assert away is not None and home is not None
    assert round(float(away) + float(home), 2) == 8.0


def test_apply_market_freeze_payload_overlays_frozen_line_and_grading():
    payload = {
        "predicted_total_runs": 5.07,
        "market_total": 4.5,
        "market_backed": True,
        "actual_total_runs": 3.0,
        "recommended_side": "over",
        "actual_side": "under",
        "result": "lost",
        "delta_vs_market": 0.57,
    }

    updated = app_module._apply_market_freeze_payload(
        payload,
        {
            "frozen_sportsbook": "consensus",
            "frozen_snapshot_ts": "2026-04-07T22:00:00Z",
            "frozen_line_value": 3.5,
        },
    )

    assert updated["market_total"] == 3.5
    assert updated["market_locked"] is True
    assert updated["locked_sportsbook"] == "consensus"
    assert updated["recommended_side"] == "over"
    assert updated["actual_side"] == "under"
    assert updated["result"] == "lost"
    assert updated["delta_vs_market"] == 1.57


def test_summarize_team_lineup_prefers_confirmed_rows_when_available():
    summary = app_module._summarize_team_lineup(
        [
            {
                "player_id": 1,
                "lineup_slot": 1,
                "player_name": "Confirmed One",
                "is_confirmed_lineup": True,
                "has_lineup_snapshot": True,
                "is_inferred_lineup": False,
                "lineup_source_name": "mlb_statsapi",
            },
            {
                "player_id": 2,
                "lineup_slot": 2,
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


def test_summarize_team_lineup_does_not_pad_confirmed_rows_with_projected_fallbacks():
    summary = app_module._summarize_team_lineup(
        [
            {
                "player_id": 11,
                "lineup_slot": 1,
                "player_name": "Confirmed Leadoff",
                "is_confirmed_lineup": True,
                "has_lineup_snapshot": True,
                "is_inferred_lineup": False,
                "lineup_source_name": "mlb_statsapi",
            },
            {
                "player_id": 12,
                "lineup_slot": 2,
                "player_name": "Projected Two-Hole",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": False,
                "is_inferred_lineup": True,
                "lineup_source_name": "projected_template",
            },
            {
                "player_id": 13,
                "lineup_slot": 3,
                "player_name": "Projected Three-Hole",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": False,
                "is_inferred_lineup": True,
                "lineup_source_name": "projected_template",
            },
        ]
    )

    assert summary["lineup_scope"] == "confirmed"
    assert [row["player_name"] for row in summary["lineup"]] == ["Confirmed Leadoff"]
    assert summary["lineup_counts"]["displayed_rows"] == 1


def test_summarize_team_lineup_prefers_snapshot_rows_before_inferred_rows():
    summary = app_module._summarize_team_lineup(
        [
            {
                "player_id": 21,
                "lineup_slot": 1,
                "player_name": "Snapshot One",
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": True,
                "is_inferred_lineup": False,
                "lineup_source_name": "rotowire",
            },
            {
                "player_id": 22,
                "lineup_slot": 2,
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

    _patch_pair(monkeypatch, "_table_exists", lambda name: name in {"pitcher_starts", "games"})
    _patch_pair(monkeypatch, "_safe_frame", lambda *_args, **_kwargs: frame)

    rows = app_module._fetch_pitcher_recent_starts(123, app_module.date(2026, 4, 4), limit=5)

    assert len(rows) == 2
    assert rows[0]["game_date"] == "2026-04-03"
    assert rows[0]["strikeouts"] == 8
    assert rows[0]["pitch_count"] == 94
    assert rows[1]["opponent"] == "CLE"


def test_fetch_pitcher_recent_starts_falls_back_to_player_game_pitching(monkeypatch):
    outing_row = pd.DataFrame(
        [
            {
                "game_date": "2026-04-02",
                "game_id": 900,
                "team": "TOR",
                "ip": 1.0,
                "earned_runs": 0,
                "strikeouts": 2,
                "walks": 0,
                "pitch_count": 17,
                "opponent": "NYY",
            },
        ]
    )

    def fake_safe(sql: str, params: object) -> pd.DataFrame:
        if "FROM pitcher_starts" in sql:
            return pd.DataFrame()
        if "FROM player_game_pitching" in sql:
            return outing_row
        return pd.DataFrame()

    _patch_pair(
        monkeypatch,
        "_table_exists",
        lambda name: name in {"pitcher_starts", "games", "player_game_pitching"},
    )
    _patch_pair(monkeypatch, "_safe_frame", fake_safe)

    rows = app_module._fetch_pitcher_recent_starts(55, app_module.date(2026, 4, 5), limit=5)

    assert len(rows) == 1
    assert rows[0]["opponent"] == "NYY"
    assert rows[0]["pitch_count"] == 17


def test_starter_records_prefer_boxscore_overrides_probable(monkeypatch):
    ranked = [
        {
            "game_id": 1,
            "team": "PHI",
            "pitcher_id": 111,
            "pitcher_name": "Reliever Listed",
            "throws": "R",
            "is_probable": True,
            "days_rest": 1,
            "xwoba_against": 0.4,
            "csw_pct": 0.2,
            "avg_fb_velo": 93.0,
            "whiff_pct": 0.2,
        }
    ]
    box_row = {
        "game_id": 1,
        "team": "PHI",
        "pitcher_id": 222,
        "pitcher_name": "Actual Starter",
        "throws": "R",
        "is_probable": False,
        "days_rest": 5,
        "ip": 6.0,
        "strikeouts": 7,
        "walks": 1,
        "pitch_count": 95,
        "xwoba_against": 0.31,
        "csw_pct": 0.29,
        "avg_fb_velo": 96.0,
        "whiff_pct": 0.22,
    }

    def box_map(*_a, **_k):
        return {(1, "PHI"): box_row}

    _patch_pair(monkeypatch, "_fetch_boxscore_primary_starter_map", box_map)

    out = app_module._starter_records_prefer_boxscore(
        ranked,
        app_module.date(2026, 4, 1),
    )

    assert out[0]["pitcher_id"] == 222
    assert out[0]["pitcher_name"] == "Actual Starter"
    assert out[0]["ip"] == 6.0


def test_starter_records_prefer_boxscore_noop_when_no_box(monkeypatch):
    ranked = [{"game_id": 1, "team": "PHI", "pitcher_id": 1, "pitcher_name": "A"}]

    def empty_map(*_a, **_k):
        return {}

    _patch_pair(monkeypatch, "_fetch_boxscore_primary_starter_map", empty_map)

    out = app_module._starter_records_prefer_boxscore(
        ranked,
        app_module.date(2026, 4, 1),
    )

    assert out == ranked