from datetime import date, timedelta
import importlib

app_module = importlib.import_module("src.api.app")


def test_green_pick_board_limit_is_unbounded():
    assert app_module._green_pick_board_limit(date.today()) is None
    assert app_module._green_pick_board_limit(date.today() - timedelta(days=1)) is None
    assert app_module._green_pick_board_limit(date.today() + timedelta(days=1)) is None
