"""Strikeout projection vs market line (pitchers board)."""

from src.api import app_logic


def test_merge_strikeout_uses_nonnegative_fundamentals_over_bad_calibrated():
    merged = app_logic._merge_strikeout_market_context(
        {
            "projected_strikeouts": -93.17,
            "projected_strikeouts_fundamentals": 6.5,
            "market": {},
        },
        {"consensus_line": 6.5},
    )
    assert merged is not None
    assert merged["display_projected_strikeouts"] == 6.5
    assert merged["market"]["projection_delta"] == 0.0


def test_merge_strikeout_projection_delta_matches_display_primary():
    merged = app_logic._merge_strikeout_market_context(
        {
            "public_projected_strikeouts": 7.2,
            "projected_strikeouts": 1.0,
            "market": {},
        },
        {"consensus_line": 6.5},
    )
    assert merged["display_projected_strikeouts"] == 7.2
    assert merged["market"]["projection_delta"] == 0.7
