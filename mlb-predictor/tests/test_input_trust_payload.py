"""API input_trust payload (projection vs certainty separation)."""

import importlib

import pytest

app_logic = importlib.import_module("src.api.app_logic")


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
    out = app_logic._input_trust_from_certainty(cert)
    assert out["grade"] == "A"
    assert out["score"] >= 0.99
    assert "Strong pregame" in out["summary"]


def test_input_trust_grade_d_when_many_fallbacks():
    cert = {
        "starter_certainty": 0.2,
        "lineup_certainty": 0.2,
        "market_freshness": 0.2,
        "weather_freshness": 0.2,
        "bullpen_completeness": 0.2,
        "missing_fallback_count": 12,
        "board_state": "minimal",
    }
    out = app_logic._input_trust_from_certainty(cert)
    assert out["grade"] == "D"
    assert "informational" in out["summary"].lower()


def test_build_certainty_payload_includes_input_trust():
    p = app_logic._build_certainty_payload(
        starter_certainty=1.0,
        lineup_certainty=1.0,
        weather_freshness=1.0,
        market_freshness=1.0,
        bullpen_completeness=1.0,
        missing_fallback_count=0,
        board_state="complete",
    )
    assert "input_trust" in p
    assert p["input_trust"]["grade"] in {"A", "B", "C", "D"}
    assert "summary" in p["input_trust"]


def test_board_state_minimal_downgrades():
    cert = {
        "starter_certainty": 1.0,
        "lineup_certainty": 1.0,
        "market_freshness": 1.0,
        "weather_freshness": 1.0,
        "bullpen_completeness": 1.0,
        "missing_fallback_count": 0,
        "board_state": "minimal",
    }
    out = app_logic._input_trust_from_certainty(cert)
    assert out["grade"] == "B"
