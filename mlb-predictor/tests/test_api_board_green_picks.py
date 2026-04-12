from datetime import date, timedelta
import importlib

from src.utils import best_bets


app_module = importlib.import_module("src.api.app")


def test_green_pick_board_limit_is_unbounded_for_historical_dates():
    assert app_module._green_pick_board_limit(date.today() - timedelta(days=1)) is None


def test_green_pick_board_limit_keeps_board_cap_for_today_and_future():
    assert app_module._green_pick_board_limit(date.today()) == best_bets.BOARD_BEST_BET_LIMIT
    assert app_module._green_pick_board_limit(date.today() + timedelta(days=1)) == best_bets.BOARD_BEST_BET_LIMIT