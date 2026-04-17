"""Unit tests for hitter form classification (hot / warm / steady / cold)."""

from __future__ import annotations

from src.utils.hitter_form import HOT_HITTER_PAGE_FORM_KEYS, classify_hitter_form


def _base_player(**kwargs: object) -> dict:
    row = {
        "hit_rate_7": 0.28,
        "hit_rate_30": 0.28,
        "xwoba_14": 0.32,
        "hard_hit_pct_14": 0.38,
        "batting_avg_last7": 0.26,
        "hit_games_last7": 3,
        "games_last7": 6,
        "streak_len": 0,
        "streak_len_capped": 0,
    }
    row.update(kwargs)
    return row


def test_classify_hot_when_strong_positive_signals() -> None:
    p = _base_player(
        hit_rate_7=0.42,
        hit_rate_30=0.22,
        games_last7=6,
        xwoba_14=0.395,
        streak_len=5,
    )
    out = classify_hitter_form(p)
    assert out["form_key"] == "hot"
    assert out["label"] == "Hot"
    assert out["tone"] == "good"


def test_classify_warm_tier() -> None:
    p = _base_player(
        hit_rate_7=0.34,
        hit_rate_30=0.26,
        games_last7=5,
        streak_len=3,
    )
    out = classify_hitter_form(p)
    assert out["form_key"] == "warm"
    assert out["label"] == "Heating up"


def test_classify_cold_when_lagging() -> None:
    p = _base_player(
        hit_rate_7=0.18,
        hit_rate_30=0.32,
        games_last7=6,
        xwoba_14=0.27,
    )
    out = classify_hitter_form(p)
    assert out["form_key"] == "cold"
    assert out["label"] == "Cold"
    assert out["tone"] == "warn"


def test_classify_steady_when_no_clear_signal() -> None:
    p = _base_player(
        hit_rate_7=0.29,
        hit_rate_30=0.27,
        games_last7=6,
        xwoba_14=0.32,
        streak_len=1,
    )
    out = classify_hitter_form(p)
    assert out["form_key"] == "steady"
    assert out["label"] == "Steady"


def test_hot_hitters_page_accepts_hot_and_warm() -> None:
    assert "hot" in HOT_HITTER_PAGE_FORM_KEYS
    assert "warm" in HOT_HITTER_PAGE_FORM_KEYS
    assert "cold" not in HOT_HITTER_PAGE_FORM_KEYS
