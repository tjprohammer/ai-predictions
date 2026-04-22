"""Unit tests for src.utils.input_trust (grades aligned with API certainty payload)."""

import math

from src.utils.input_trust import ACTIONABLE_TRUST_GRADES, input_trust_from_certainty
from src.utils.best_bets import card_input_trust_from_game, promotion_tier_for_card


def test_input_trust_grade_a_when_signals_strong():
    cert = {
        "starter_certainty": 1.0,
        "lineup_certainty": 1.0,
        "market_freshness": 1.0,
        "weather_freshness": 1.0,
        "bullpen_completeness": 1.0,
        "missing_fallback_count": 0,
        "board_state": "complete",
    }
    out = input_trust_from_certainty(cert)
    assert out["grade"] == "A"
    assert out["score"] >= 0.99
    assert "Strong pregame" in out["summary"]


def test_input_trust_nan_missing_fallback_treated_as_zero():
    """Stored certainty rows may carry NaN for missing_fallback_count (pandas/JSON edge cases)."""
    cert = {
        "starter_certainty": 1.0,
        "lineup_certainty": 1.0,
        "market_freshness": 1.0,
        "weather_freshness": 1.0,
        "bullpen_completeness": 1.0,
        "missing_fallback_count": float("nan"),
        "board_state": "complete",
    }
    out = input_trust_from_certainty(cert)
    assert out["grade"] == "A"
    assert not math.isnan(out["score"])


def test_actionable_grades_constant():
    assert ACTIONABLE_TRUST_GRADES == {"A", "B"}


def test_promotion_tier_actionable_vs_edge():
    assert (
        promotion_tier_for_card(positive=True, input_trust={"grade": "A"})
        == "actionable"
    )
    assert (
        promotion_tier_for_card(positive=True, input_trust={"grade": "C"})
        == "edge_only"
    )
    assert promotion_tier_for_card(positive=False, input_trust={"grade": "A"}) == "none"


def test_card_input_trust_from_nested_certainty():
    game = {
        "certainty": {
            "input_trust": {"grade": "B", "score": 0.71, "summary": "ok"},
        }
    }
    assert card_input_trust_from_game(game)["grade"] == "B"


def test_input_trust_strong_chip_not_grade_d_when_feature_row_sparse():
    """Many NaNs in totals certainty key fields can coexist with strong freshness scores."""
    cert = {
        "starter_certainty": 0.95,
        "lineup_certainty": 0.90,
        "market_freshness": 0.92,
        "weather_freshness": 0.88,
        "bullpen_completeness": 0.90,
        "missing_fallback_count": 6,
        "board_state": "minimal",
    }
    out = input_trust_from_certainty(cert)
    assert out["grade"] == "B"
    assert out["grade"] != "D"

