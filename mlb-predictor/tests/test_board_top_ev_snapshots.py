"""Frozen Top EV snapshots: Daily Results should reuse the lock-time pick, not recompute."""

from datetime import date, datetime, timedelta, timezone

import src.api.app_logic as app_logic


def test_top_ev_snapshot_lock_respects_optional_env_minutes(monkeypatch):
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES", raising=False)
    monkeypatch.setenv("MLB_PREGAME_INGEST_LOCK_MINUTES", "30")
    get_settings.cache_clear()
    start = datetime.now(timezone.utc) + timedelta(minutes=20)
    row = {"game_start_ts": start}
    assert app_logic._is_game_top_ev_snapshot_lock_active(row) is True

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES", "10")
    get_settings.cache_clear()
    row_far = {"game_start_ts": datetime.now(timezone.utc) + timedelta(minutes=45)}
    assert app_logic._is_game_top_ev_snapshot_lock_active(row_far) is False
    row_close = {"game_start_ts": datetime.now(timezone.utc) + timedelta(minutes=8)}
    assert app_logic._is_game_top_ev_snapshot_lock_active(row_close) is True

    monkeypatch.delenv("BOARD_TOP_EV_SNAPSHOT_LOCK_MINUTES", raising=False)
    get_settings.cache_clear()


def test_live_top_ev_daily_results_uses_frozen_snapshot_pick(monkeypatch):
    target = date(2026, 4, 17)
    frozen_pick = {
        "market_key": "first_five_spread",
        "bet_side": "home",
        "selection_label": "F5 HOME -0.5",
        "weighted_ev": 0.12,
        "probability_edge": 0.06,
        "model_probability": 0.58,
        "price": -110,
        "opposing_price": -110,
        "sportsbook": "book_a",
        "input_trust": {"grade": "A", "score": 0.8},
        "top_ev_candidate_count": 50,
    }
    monkeypatch.setattr(app_logic, "_fetch_board_top_ev_snapshots_map", lambda d: {9001: frozen_pick})
    monkeypatch.setattr(app_logic, "_fetch_board_top_ev_run_snapshots_map", lambda d: {})

    seen: list[dict] = []

    def capture_grade(detail, pick):
        seen.append(dict(pick))
        meta = {
            "underlying_market_key": frozen_pick["market_key"],
            "price": pick.get("price"),
            "weighted_ev": pick.get("weighted_ev"),
        }
        return (
            {
                "recommended_side": pick.get("bet_side"),
                "graded": True,
            },
            "won",
            meta,
        )

    monkeypatch.setattr(app_logic, "_grade_top_ev_pick_for_daily_results", capture_grade)

    board_row = {
        "game_id": 9001,
        "away_team": "AAA",
        "home_team": "HHH",
        "game_date": target.isoformat(),
        "certainty": {
            "starter_certainty": 0.9,
            "lineup_certainty": 0.9,
            "market_freshness": 0.9,
            "weather_freshness": 0.9,
            "bullpen_completeness": 0.9,
        },
        "totals": {"predicted_total_runs": 8.5},
        "first5_totals": {},
        "hit_targets": {},
        "starters": {"away": None, "home": None},
        "actual_result": {"is_final": True, "away_runs": 5, "home_runs": 4},
    }

    rows = app_logic._live_top_ev_rows_for_daily_results(target, board_rows=[board_row])
    assert len(rows) == 1
    assert rows[0].get("top_ev_frozen") is True
    assert rows[0].get("top_ev_snapshot_kind") == "lock"
    assert "frozen" in str(rows[0].get("green_reason") or "").lower()
    assert seen and seen[0]["selection_label"] == "F5 HOME -0.5"


def test_live_top_ev_daily_results_prefers_lock_over_run_snapshot(monkeypatch):
    """Lock snapshot wins when both exist; run is fallback when lock row missing."""
    target = date(2026, 4, 17)
    lock_pick = {"market_key": "total", "bet_side": "over", "selection_label": "LOCK", "weighted_ev": 0.2}
    run_pick = {"market_key": "total", "bet_side": "under", "selection_label": "RUN", "weighted_ev": 0.1}
    board_row = {
        "game_id": 42,
        "away_team": "A",
        "home_team": "H",
        "game_date": target.isoformat(),
        "certainty": {
            "starter_certainty": 0.9,
            "lineup_certainty": 0.9,
            "market_freshness": 0.9,
            "weather_freshness": 0.9,
            "bullpen_completeness": 0.9,
        },
        "totals": {"predicted_total_runs": 8.0},
        "first5_totals": {},
        "hit_targets": {},
        "starters": {"away": None, "home": None},
        "actual_result": {},
    }
    monkeypatch.setattr(app_logic, "_fetch_board_top_ev_snapshots_map", lambda d: {42: lock_pick})
    monkeypatch.setattr(app_logic, "_fetch_board_top_ev_run_snapshots_map", lambda d: {42: run_pick})
    monkeypatch.setattr(app_logic, "_grade_top_ev_pick_for_daily_results", lambda d, p: ({}, "pending", {}))
    rows = app_logic._live_top_ev_rows_for_daily_results(target, board_rows=[board_row])
    assert len(rows) == 1
    assert rows[0].get("pick_label") == "LOCK"
    assert rows[0].get("top_ev_snapshot_kind") == "lock"

    monkeypatch.setattr(app_logic, "_fetch_board_top_ev_snapshots_map", lambda d: {})
    rows2 = app_logic._live_top_ev_rows_for_daily_results(target, board_rows=[board_row])
    assert rows2[0].get("pick_label") == "RUN"
    assert rows2[0].get("top_ev_snapshot_kind") == "run"
