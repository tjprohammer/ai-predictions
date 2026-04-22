"""Daily Results green-board section falls back to live board when archived outcomes are empty."""

from datetime import date
from unittest.mock import patch

import pytest

from src.api import app_logic


@pytest.fixture
def sample_live_row():
    return {
        "game_id": 1,
        "game_date": "2026-04-12",
        "game_start_ts": None,
        "market": "moneyline",
        "market_label": "Moneyline",
        "subject_label": "A at B",
        "subject_subtitle": "Moneyline",
        "pick_label": "Away +120",
        "model_display": "55%",
        "actual_display": "Pending",
        "notes_display": "Moneyline",
        "confidence": 0.55,
        "edge": 0.02,
        "result": "pending",
        "market_backed": True,
        "is_green_pick": True,
        "is_watchlist_pick": False,
        "is_experimental_pick": False,
        "green_rank": None,
        "watchlist_rank": None,
        "green_reason": "x",
        "input_trust_grade": None,
        "promotion_tier": None,
    }


def test_fetch_daily_results_sets_live_fallback_when_archived_empty(sample_live_row):
    # Not calendar-today so archived+live merge runs; archived empty → live only.
    target = date(1985, 7, 15)
    with (
        patch.object(
            app_logic,
            "_live_green_board_rows_for_daily_results",
            return_value=[sample_live_row],
        ),
        patch.object(app_logic, "_fetch_watchlist_pick_results", return_value=[]),
        patch.object(
            app_logic,
            "_live_watchlist_board_rows_for_daily_results",
            return_value=[],
        ),
        patch.object(app_logic, "_fetch_experimental_pick_results", return_value=[]),
        patch.object(app_logic, "_live_top_ev_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_table_exists", return_value=False),
    ):
        out = app_logic._fetch_daily_results(target, hit_min_probability=0.5)

    assert out["summary"]["live_green_board_fallback"] is True
    assert out["summary"]["ai_picks"]["live_board_fallback"] is True
    assert len(out["ai_picks"]) == 1
    assert out["ai_picks"][0]["game_id"] == 1


def test_merge_green_drops_live_when_archived_already_has_that_game(sample_live_row):
    """Archived ``prediction_outcomes_daily`` row wins; do not also show a different live green for same game_id."""
    archived = [
        dict(
            sample_live_row,
            market="first5",
            market_label="First 5",
            pick_label="F5 Home +0.5",
            notes_display="archived f5",
        ),
    ]
    live_tt = dict(
        sample_live_row,
        market="away_team_total",
        market_label="Away Team Total",
        pick_label="Away TT Over 3.5",
        notes_display="live tt",
    )
    merged, _ = app_logic._merge_green_daily_results(archived, [live_tt])
    assert len(merged) == 1
    assert merged[0]["pick_label"] == "F5 Home +0.5"


def test_fetch_daily_results_prefers_archived_green_picks_over_live(sample_live_row):
    """Past slates: archived ``prediction_outcomes_daily`` wins over live when both exist."""
    target = date(1985, 7, 15)
    # Same row identity (game_id, market, pick_label) as live; archived row wins and live duplicate is dropped.
    archived = [dict(sample_live_row, edge=0.01, notes_display="from prediction_outcomes_daily")]
    live_row = dict(sample_live_row, edge=0.99)
    with (
        patch.object(app_logic, "_fetch_ai_pick_results", return_value=archived),
        patch.object(
            app_logic,
            "_live_green_board_rows_for_daily_results",
            return_value=[live_row],
        ),
        patch.object(app_logic, "_fetch_watchlist_pick_results", return_value=[]),
        patch.object(
            app_logic,
            "_live_watchlist_board_rows_for_daily_results",
            return_value=[],
        ),
        patch.object(app_logic, "_fetch_experimental_pick_results", return_value=[]),
        patch.object(app_logic, "_live_top_ev_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_table_exists", return_value=False),
    ):
        out = app_logic._fetch_daily_results(target, hit_min_probability=0.5)

    assert out["summary"]["live_green_board_fallback"] is False
    assert len(out["ai_picks"]) == 1
    assert out["ai_picks"][0]["edge"] == 0.01
    assert out["ai_picks"][0]["notes_display"] == "from prediction_outcomes_daily"


def _minimal_watchlist_row(game_id: int, pick_label: str) -> dict:
    return {
        "game_id": game_id,
        "game_date": "2026-04-12",
        "market": "moneyline",
        "pick_label": pick_label,
        "result": "pending",
        "market_backed": True,
        "is_watchlist_pick": True,
        "is_green_pick": False,
    }


def test_fetch_daily_results_merges_live_watchlist_picks():
    target = date(1985, 7, 15)
    archived = [_minimal_watchlist_row(1, "Away ML")]
    live_extra = [_minimal_watchlist_row(2, "Home ML")]
    with (
        patch.object(app_logic, "_fetch_ai_pick_results", return_value=[]),
        patch.object(app_logic, "_live_green_board_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_fetch_watchlist_pick_results", return_value=archived),
        patch.object(
            app_logic,
            "_live_watchlist_board_rows_for_daily_results",
            return_value=live_extra,
        ),
        patch.object(app_logic, "_fetch_experimental_pick_results", return_value=[]),
        patch.object(app_logic, "_live_top_ev_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_table_exists", return_value=False),
    ):
        out = app_logic._fetch_daily_results(target, hit_min_probability=0.5)

    assert out["summary"]["live_watchlist_board_supplement"] is True
    assert len(out["watchlist"]) == 2
    assert {r["game_id"] for r in out["watchlist"]} == {1, 2}


def test_merge_pick_of_day_renumbers_and_one_row_per_game() -> None:
    archived = [
        {"game_id": 10, "pick_of_day_rank": 1, "pick_label": "A"},
        {"game_id": 20, "pick_of_day_rank": 2, "pick_label": "B"},
        {"game_id": 30, "pick_of_day_rank": 3, "pick_label": "C"},
    ]
    live = [
        {"game_id": 20, "pick_of_day_rank": 1, "pick_label": "live duplicate game"},
        {"game_id": 40, "pick_of_day_rank": 1, "pick_label": "D"},
        {"game_id": 50, "pick_of_day_rank": 2, "pick_label": "E"},
    ]
    merged, supplemental = app_logic._merge_pick_of_day_daily_results(archived, live)
    assert supplemental is True
    assert {r["game_id"] for r in merged} == {10, 20, 30, 40, 50}
    assert [r["pick_of_day_rank"] for r in merged] == [1, 2, 3, 4, 5]


def test_daily_results_pick_label_pitcher_without_selection_meta() -> None:
    meta = {"pitcher_name": "Kris Bubic"}
    lbl = app_logic._daily_results_pick_label(
        "pitcher_strikeouts",
        "under",
        6.5,
        6.5,
        "BAL",
        "KC",
        meta,
    )
    assert "Under" in lbl
    assert "6.5" in lbl
    assert "Bubic" in lbl


def test_fetch_daily_results_calendar_today_uses_live_green_not_archived(sample_live_row, monkeypatch):
    """Today's slate: Daily Results green strip must match the live board, not stale prediction_outcomes_daily."""
    target = date(2026, 7, 20)

    class _FixedDate:
        today = staticmethod(lambda: target)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(app_logic, "date", _FixedDate)
    archived = [dict(sample_live_row, edge=0.01, notes_display="from archive")]
    live_row = dict(sample_live_row, edge=0.99, notes_display="from live board")
    with (
        patch.object(app_logic, "_fetch_ai_pick_results", return_value=archived),
        patch.object(
            app_logic,
            "_live_green_board_rows_for_daily_results",
            return_value=[live_row],
        ),
        patch.object(app_logic, "_fetch_pick_of_day_results", return_value=[]),
        patch.object(app_logic, "_live_pick_of_day_board_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_fetch_watchlist_pick_results", return_value=[]),
        patch.object(app_logic, "_live_watchlist_board_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_fetch_experimental_pick_results", return_value=[]),
        patch.object(app_logic, "_live_top_ev_rows_for_daily_results", return_value=[]),
        patch.object(app_logic, "_table_exists", return_value=False),
    ):
        out = app_logic._fetch_daily_results(target, hit_min_probability=0.5)

    assert out["summary"]["live_green_board_fallback"] is True
    assert len(out["ai_picks"]) == 1
    assert out["ai_picks"][0]["edge"] == 0.99
    assert out["ai_picks"][0]["notes_display"] == "from live board"
