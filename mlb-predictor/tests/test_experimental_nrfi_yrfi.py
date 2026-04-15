"""NRFI/YRFI experimental carousel pick + reasoning (app_logic)."""

from src.api import app_logic


def test_pick_nrfi_when_both_posted_matches_dedupe_convention():
    nrfi = {"under_price": -150, "over_price": 130}
    yrfi = {"over_price": -140, "under_price": 120}
    m, row, reason = app_logic._pick_experimental_nrfi_yrfi_row(
        nrfi_row=nrfi,
        yrfi_row=yrfi,
    )
    assert m == "nrfi"
    assert row is nrfi
    assert reason == "both_posted_prefer_nrfi"


def test_pick_single_side_yrfi():
    r = {"over_price": -140}
    m, row, reason = app_logic._pick_experimental_nrfi_yrfi_row(
        nrfi_row=None,
        yrfi_row=r,
    )
    assert m == "yrfi"
    assert row is r
    assert reason == "yrfi_only"


def test_pick_single_side_nrfi():
    r = {"under_price": -150}
    m, row, reason = app_logic._pick_experimental_nrfi_yrfi_row(
        nrfi_row=r,
        yrfi_row=None,
    )
    assert m == "nrfi"
    assert row is r
    assert reason == "nrfi_only"


def test_reasoning_notes_implied_odds():
    text = app_logic._experimental_first_inning_reasoning_notes(
        market="nrfi",
        pick_reason="both_posted_prefer_nrfi",
        row={"under_price": -150, "over_price": 130},
    )
    assert "posted odds" in text.lower()
    assert "implied" in text.lower()


def test_reasoning_notes_with_model_probability():
    text = app_logic._experimental_first_inning_reasoning_notes(
        market="yrfi",
        pick_reason="model_pick",
        row={"under_price": -150, "over_price": 130},
        model_nrfi_probability=0.42,
        model_confidence="medium",
    )
    assert "trained inning-1 model" in text.lower()
    assert "42%" in text
    assert "yrfi" in text.lower()
