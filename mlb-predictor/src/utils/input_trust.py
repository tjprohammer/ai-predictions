"""Input trust grades (A–D) separate from raw projection — see docs/MODEL_REWORK_PLAN.md."""

from __future__ import annotations

import math
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def composite_certainty_score_for_input_trust(certainty: dict[str, Any] | None) -> float:
    """Weighted composite matching the board/API certainty stack (not EV `game_certainty_weight`)."""
    certainty = certainty or {}
    weighted_values = [
        (_to_float(certainty.get("starter_certainty")), 0.30),
        (_to_float(certainty.get("lineup_certainty")), 0.26),
        (_to_float(certainty.get("market_freshness")), 0.18),
        (_to_float(certainty.get("weather_freshness")), 0.12),
        (_to_float(certainty.get("bullpen_completeness")), 0.14),
    ]
    numerator = 0.0
    denominator = 0.0
    for value, weight in weighted_values:
        if value is None:
            continue
        numerator += max(0.0, min(1.0, value)) * weight
        denominator += weight
    if denominator <= 0:
        return 0.6
    return numerator / denominator


def weakest_certainty_axis_label(certainty: dict[str, Any]) -> str:
    """Human label for the weakest (or missing) input-trust component."""
    items: list[tuple[str, float | None]] = [
        ("Starter clarity", _to_float(certainty.get("starter_certainty"))),
        ("Lineup confirmation", _to_float(certainty.get("lineup_certainty"))),
        ("Market freshness", _to_float(certainty.get("market_freshness"))),
        ("Weather freshness", _to_float(certainty.get("weather_freshness"))),
        ("Bullpen context", _to_float(certainty.get("bullpen_completeness"))),
    ]
    worst: tuple[str, float] | None = None
    for label, val in items:
        if val is None:
            return label
        score = max(0.0, min(1.0, float(val)))
        gap = 1.0 - score
        if worst is None or gap > worst[1]:
            worst = (label, gap)
    return worst[0] if worst else "Pregame inputs"


def _to_nonneg_int(value: Any) -> int:
    """Count-like ints; NaN/None/invalid → 0 (avoids `nan or 0` staying NaN)."""
    f = _to_float(value)
    if f is None:
        return 0
    return max(0, int(f))


def input_trust_from_certainty(certainty: dict[str, Any]) -> dict[str, Any]:
    """Separate **input trust** from raw projection (see docs/MODEL_REWORK_PLAN.md).

    The weighted composite (``score``) reflects the same five signals as the board certainty chip.
    ``missing_fallback_count`` counts NaNs in separate *model numeric* fields (e.g. blended xwoba
    columns); those can be sparse while freshness scores are still strong. Grade is driven
    primarily by ``score``; high ``missing`` only tightens the letter when the composite is weak.
    """
    score = composite_certainty_score_for_input_trust(certainty)
    missing = _to_nonneg_int(certainty.get("missing_fallback_count"))
    board = str(certainty.get("board_state") or "").strip().lower()

    if score < 0.28 or (missing >= 8 and score < 0.40):
        grade = "D"
    elif score < 0.42 or (missing >= 4 and score < 0.52):
        grade = "C"
    elif score < 0.62 or (missing >= 2 and score < 0.52):
        grade = "B"
    else:
        grade = "A"

    if board == "minimal":
        grade = {"A": "B", "B": "C", "C": "D", "D": "D"}.get(grade, grade)
    elif board == "partial" and grade == "A":
        grade = "B"

    axis = weakest_certainty_axis_label(certainty)
    if grade == "D":
        summary = (
            "Major gaps in pregame context; treat the run projection as informational, not actionable."
        )
    elif grade == "C":
        summary = (
            "Several inputs are still projected or stale — watchlist tier until lineups and markets firm up."
        )
    elif grade == "B":
        summary = (
            f"Projection is usable; weakest signal: {axis}. Edge and totals can still move before lock."
        )
    else:
        summary = (
            f"Strong pregame input alignment ({score:.0%} composite). {axis} is the first place to watch if anything shifts."
        )

    return {
        "grade": grade,
        "score": round(float(score), 4),
        "summary": summary,
    }


ACTIONABLE_TRUST_GRADES = frozenset({"A", "B"})
