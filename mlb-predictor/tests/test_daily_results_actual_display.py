"""Regression: Daily Results actual line must show real K total, not win/loss flag (0/1)."""

import importlib

app_logic = importlib.import_module("src.api.app_logic")
best_bets = importlib.import_module("src.utils.best_bets")


def test_pitcher_k_actual_display_uses_measure_not_win_loss_binary():
    """``grade_best_bet_pick`` sets ``actual_value`` to 0/1; UI must use ``actual_measure`` for Ks."""
    record = {
        "recommended_side": "under",
        "actual_side": "over",
        "actual_value": 0.0,
        "graded": True,
        "game_final": True,
    }
    meta = {
        "underlying_market_key": best_bets.PITCHER_STRIKEOUTS_MARKET_KEY,
        "actual_measure": 6.0,
    }
    out = app_logic._daily_results_actual_display(record, "top_ev", "TOR", "MIL", meta)
    assert out == "6 Ks · OVER"


def test_pitcher_k_row_uses_actual_strikeouts_when_present():
    record = {
        "recommended_side": "under",
        "actual_side": "over",
        "actual_value": 0.0,
        "actual_strikeouts": 6,
        "graded": True,
        "game_final": True,
    }
    meta: dict = {}
    out = app_logic._daily_results_actual_display(
        record, best_bets.PITCHER_STRIKEOUTS_MARKET_KEY, "TOR", "MIL", meta
    )
    assert out == "6 Ks · OVER"
