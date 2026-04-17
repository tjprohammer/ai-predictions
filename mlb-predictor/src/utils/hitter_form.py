"""Hitter form tiers (hot / warm / steady / cold) for API and product surfaces.

Uses a **point score** with minimum sample sizes so short noisy samples do not
solo-trigger "Hot" (previous logic OR'd many weak signals). Thresholds are module
constants for tuning.
"""

from __future__ import annotations

import math
from typing import Any

# --- Tuning knobs (document when changing product behavior) ---
MIN_GAMES_7_FOR_DELTA = 4  # require this many games before 7G vs 30G delta counts fully
HOT_SCORE_MIN = 5
WARM_SCORE_MIN = 3
COLD_SCORE_MIN = 3  # strong cold
COLD_SCORE_SOFT = 2  # cold if hot does not clearly dominate


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _format_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def hitter_heat_score(player: dict[str, Any]) -> float:
    """Scalar for sorting — comparable across players; not a probability."""
    hr7 = _to_float(player.get("hit_rate_7")) or 0.0
    hr30 = _to_float(player.get("hit_rate_30")) or 0.0
    xw = _to_float(player.get("xwoba_14")) or 0.0
    hh = _to_float(player.get("hard_hit_pct_14")) or 0.0
    streak = max(
        int(player.get("streak_len") or 0),
        int(player.get("streak_len_capped") or 0),
    )
    games_7 = int(player.get("games_last7") or 0)
    sample_w = min(1.0, games_7 / 7.0) if games_7 else 0.0
    delta = hr7 - hr30 if _to_float(player.get("hit_rate_7")) is not None else 0.0
    return (
        hr7 * 1.15
        + delta * 1.4 * sample_w
        + xw * 1.05
        + hh * 0.45
        + streak * 0.035
    )


def _accumulate_scores(player: dict[str, Any]) -> tuple[int, int, list[str], list[str]]:
    """Return (hot_score, cold_score, hot_reasons, cold_reasons)."""
    hit_rate_7 = _to_float(player.get("hit_rate_7"))
    hit_rate_30 = _to_float(player.get("hit_rate_30"))
    xwoba_14 = _to_float(player.get("xwoba_14"))
    hard_hit_pct_14 = _to_float(player.get("hard_hit_pct_14"))
    batting_avg_last7 = _to_float(player.get("batting_avg_last7"))
    hit_games_last7 = int(player.get("hit_games_last7") or 0)
    games_last7 = int(player.get("games_last7") or 0)
    streak = max(
        int(player.get("streak_len") or 0),
        int(player.get("streak_len_capped") or 0),
    )
    hit_delta = None if hit_rate_7 is None or hit_rate_30 is None else hit_rate_7 - hit_rate_30

    hot: list[str] = []
    cold: list[str] = []
    hs = 0
    cs = 0

    delta_trusted = games_last7 >= MIN_GAMES_7_FOR_DELTA
    if hit_delta is not None:
        if delta_trusted:
            if hit_delta >= 0.10:
                hs += 3
                hot.append(
                    f"7G hit rate {_format_rate(hit_rate_7)} is {hit_delta * 100:+.0f} pts vs 30G baseline"
                )
            elif hit_delta >= 0.06:
                hs += 2
                hot.append(
                    f"7G vs 30G up {hit_delta * 100:+.0f} pts (solid sample)"
                )
            elif hit_delta <= -0.10:
                cs += 3
                cold.append(
                    f"7G hit rate {_format_rate(hit_rate_7)} is {hit_delta * 100:+.0f} pts vs 30G"
                )
            elif hit_delta <= -0.06:
                cs += 2
                cold.append(
                    f"7G vs 30G down {hit_delta * 100:+.0f} pts"
                )
        else:
            # Small sample: only allow large deltas to contribute a little
            if hit_delta >= 0.18:
                hs += 1
                hot.append("Short 7G sample but hit rate well above 30G trend")
            elif hit_delta <= -0.18:
                cs += 1
                cold.append("Short 7G sample but hit rate well below 30G trend")

    if xwoba_14 is not None:
        if xwoba_14 >= 0.39:
            hs += 2
            hot.append(f"xwOBA (14G) {_format_metric(xwoba_14)} — elite contact quality")
        elif xwoba_14 >= 0.355:
            hs += 1
            hot.append(f"xwOBA (14G) {_format_metric(xwoba_14)}")
        elif xwoba_14 <= 0.275:
            cs += 3
            cold.append(f"xwOBA (14G) only {_format_metric(xwoba_14)}")
        elif xwoba_14 <= 0.305:
            cs += 2
            cold.append(f"xwOBA (14G) {_format_metric(xwoba_14)} — soft contact")

    if streak >= 5:
        hs += 2
        hot.append(f"{streak}-game hit streak")
    elif streak >= 3:
        hs += 1
        hot.append(f"{streak}-game hit streak")

    if hard_hit_pct_14 is not None:
        if hard_hit_pct_14 >= 0.47:
            hs += 1
            hot.append(f"Hard-hit% (14G) {_format_rate(hard_hit_pct_14)}")
        elif hard_hit_pct_14 <= 0.32 and games_last7 >= 4:
            cs += 1
            cold.append(f"Hard-hit% (14G) {_format_rate(hard_hit_pct_14)}")

    if games_last7 >= 5 and hit_games_last7 >= 4:
        hs += 1
        hot.append(f"Hits in {hit_games_last7} of last {games_last7} games")
    if games_last7 >= 5 and hit_games_last7 <= 1 and streak == 0:
        cs += 1
        cold.append(f"Only {hit_games_last7} hit games in last {games_last7}")

    if batting_avg_last7 is not None and batting_avg_last7 >= 0.330 and games_last7 >= 4:
        hs += 1
        hot.append(f"L7 AVG {_format_metric(batting_avg_last7)}")
    if batting_avg_last7 is not None and batting_avg_last7 <= 0.175 and games_last7 >= 5:
        cs += 1
        cold.append(f"L7 AVG {_format_metric(batting_avg_last7)}")

    return hs, cs, hot, cold


def classify_hitter_form(player: dict[str, Any]) -> dict[str, Any]:
    """Return display label, ``form_key``, tone, reasons, and ``heat_score``."""
    hit_rate_7 = _to_float(player.get("hit_rate_7"))
    hit_rate_30 = _to_float(player.get("hit_rate_30"))
    xwoba_14 = _to_float(player.get("xwoba_14"))
    hard_hit_pct_14 = _to_float(player.get("hard_hit_pct_14"))
    streak = max(
        int(player.get("streak_len") or 0),
        int(player.get("streak_len_capped") or 0),
    )
    hit_delta = None if hit_rate_7 is None or hit_rate_30 is None else hit_rate_7 - hit_rate_30

    evidence = [
        f"7G {_format_rate(hit_rate_7)} vs 30G {_format_rate(hit_rate_30)}"
        + (f" ({hit_delta * 100:+.0f} pts)" if hit_delta is not None else ""),
        f"xwOBA14 {_format_metric(xwoba_14)}",
        f"HH14 {_format_rate(hard_hit_pct_14)}",
        f"Streak {streak}",
    ]

    hot_score, cold_score, hot_reasons, cold_reasons = _accumulate_scores(player)
    # De-duplicate reasons while keeping order
    def _uniq(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in seq:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    hot_reasons = _uniq(hot_reasons)
    cold_reasons = _uniq(cold_reasons)

    form_key = "steady"
    label = "Steady"
    tone = ""
    reasons: list[str] = []

    # Cold wins if clearly stronger than hot signal
    if cold_score >= COLD_SCORE_MIN and cold_score > hot_score:
        form_key = "cold"
        label = "Cold"
        tone = "warn"
        reasons = cold_reasons
    elif cold_score >= COLD_SCORE_SOFT and hot_score <= 1:
        form_key = "cold"
        label = "Cold"
        tone = "warn"
        reasons = cold_reasons
    elif hot_score >= HOT_SCORE_MIN and cold_score <= 1:
        form_key = "hot"
        label = "Hot"
        tone = "good"
        reasons = hot_reasons
    elif hot_score >= WARM_SCORE_MIN and cold_score == 0:
        form_key = "warm"
        label = "Heating up"
        tone = "good"
        reasons = hot_reasons
    else:
        reasons = evidence[:2]

    summary = reasons[0] if reasons else evidence[0]
    detail = " · ".join(reasons[1:4] if len(reasons) > 1 else evidence[1:3])

    return {
        "label": label,
        "form_key": form_key,
        "tone": tone,
        "summary": summary,
        "detail": detail,
        "reasons": reasons,
        "heat_score": round(hitter_heat_score(player), 4),
        "evidence": evidence,
        "hot_score": hot_score,
        "cold_score": cold_score,
    }


HOT_HITTER_PAGE_FORM_KEYS = frozenset({"hot", "warm"})
