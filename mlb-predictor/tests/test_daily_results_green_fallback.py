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
    target = date(2026, 4, 12)
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


def test_fetch_daily_results_prefers_archived_green_picks_over_live(sample_live_row):
    """After product_surfaces, green picks come from ``prediction_outcomes_daily`` so they do not drift on refresh."""
    target = date(2026, 4, 12)
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
    target = date(2026, 4, 12)
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
