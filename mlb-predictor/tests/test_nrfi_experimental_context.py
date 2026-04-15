"""NRFI/YRFI experimental inning-1 context copy (app_logic)."""

from src.api import app_logic


def test_format_nrfi_context_includes_starters_and_env():
    block = app_logic._format_nrfi_experimental_context_block(
        {
            "away_team": "NYM",
            "home_team": "LAD",
            "away_lineup_top5_xwoba": 0.335,
            "home_lineup_top5_xwoba": 0.328,
            "venue_run_factor": 1.04,
            "roof_type": "outdoor",
            "temperature_f": 74,
            "wind_speed_mph": 10,
            "starters": {
                "away": {
                    "pitcher_name": "A. Ace",
                    "xwoba_against": 0.29,
                    "csw_pct": 0.31,
                    "whiff_pct": 0.28,
                    "avg_walks": 2.1,
                    "avg_strikeouts": 6.2,
                    "sample_starts": 5,
                },
                "home": {
                    "pitcher_name": "B. Bull",
                    "xwoba_against": 0.31,
                    "csw_pct": 29,
                    "avg_walks": 3.4,
                    "sample_starts": 5,
                },
            },
        }
    )
    assert "heuristic" in block.lower()
    assert "NYM" in block and "LAD" in block
    assert "A. Ace" in block
    assert "Environment" in block
    assert "park run factor" in block.lower()


def test_format_nrfi_context_empty():
    assert app_logic._format_nrfi_experimental_context_block(None) == ""
    assert app_logic._format_nrfi_experimental_context_block({}) == ""
