"""Matchup API sample_tier thresholds."""

from src.api.routers.games_routes import _tier_three_way


def test_tier_three_way_brackets():
    assert _tier_three_way(10, 25, 80) == "low"
    assert _tier_three_way(30, 25, 80) == "adequate"
    assert _tier_three_way(90, 25, 80) == "strong"
    assert _tier_three_way(None, 5, 15) == "low"
