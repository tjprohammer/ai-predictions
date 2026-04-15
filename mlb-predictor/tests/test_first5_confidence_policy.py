"""First-five publish tiers vs starter/lineup certainty (MODEL_REWORK_PLAN)."""

from types import SimpleNamespace

import pytest

from src.models.predict_first5_totals import _compute_confidence_level


def _row(
    *,
    starter_certainty_score: float,
    lineup_certainty_score: float | None = 0.6,
    starter_asymmetry_score: float = 0.2,
    board_state: str = "complete",
    market_total: float = 8.5,
    starter_quality_gap: float = 0.02,
):
    return SimpleNamespace(
        starter_certainty_score=starter_certainty_score,
        lineup_certainty_score=lineup_certainty_score,
        starter_quality_gap=starter_quality_gap,
        starter_asymmetry_score=starter_asymmetry_score,
        board_state=board_state,
        market_total=market_total,
    )


def test_high_confidence_requires_strong_lineup_when_score_present():
    level, _ = _compute_confidence_level(
        _row(starter_certainty_score=1.0, lineup_certainty_score=0.2),
    )
    assert level != "high"


def test_high_confidence_when_starter_and_lineup_strong():
    level, reason = _compute_confidence_level(
        _row(starter_certainty_score=1.0, lineup_certainty_score=0.6),
    )
    assert level == "high"
    assert reason is None


def test_schedule_probable_starter_cannot_reach_high_tier():
    # 0.62 average from probable rows — below 0.75 gate
    level, _ = _compute_confidence_level(
        _row(starter_certainty_score=0.62, lineup_certainty_score=0.9),
    )
    assert level != "high"


def test_suppress_without_market():
    level, reason = _compute_confidence_level(
        SimpleNamespace(
            starter_certainty_score=1.0,
            lineup_certainty_score=0.8,
            starter_asymmetry_score=0.2,
            board_state="complete",
            market_total=None,
        )
    )
    assert level == "suppress"
    assert reason == "no_market_line"
