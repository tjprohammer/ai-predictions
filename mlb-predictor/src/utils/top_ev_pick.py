"""Select the single highest weighted-EV line across team markets, batter TB O/U, pitcher K, 1+ hit, and HR yes."""

from __future__ import annotations

from typing import Any

from src.utils.best_bets import (
    TOP_EV_ELIGIBLE_MARKET_KEYS,
    TOP_EV_FULL_MARKET_KEYS,
    _top_ev_f5_team_total_fallback_enabled,
    to_float,
)


def collect_top_ev_candidates(
    detail: dict[str, Any],
    game_market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Priced EV candidates: team markets, batter TB O/U, pitcher K, 1+ hit (same pool as board / game detail Top EV)."""
    from src.utils import best_bets as best_bets_utils

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        best_bets_utils.collect_all_team_market_ev_candidates(detail, game_market_rows_by_type),
    )
    for side in ("away", "home"):
        for row in (detail.get("teams") or {}).get(side, {}).get("lineup") or []:
            hit_row = {
                **row,
                "away_team": detail.get("away_team"),
                "home_team": detail.get("home_team"),
            }
            card = best_bets_utils.build_player_hits_board_card(hit_row)
            if card:
                candidates.append(best_bets_utils.annotate_market_card_for_display(card))
    for side in ("away", "home"):
        st = (detail.get("starters") or {}).get(side)
        candidates.extend(best_bets_utils.build_pitcher_strikeouts_ev_cards_from_starter(detail, st))
    for side in ("away", "home"):
        for row in (detail.get("teams") or {}).get(side, {}).get("lineup") or []:
            candidates.extend(best_bets_utils.build_batter_total_bases_ev_cards_from_row(detail, row))
    return [c for c in candidates if not best_bets_utils._excluded_for_overs_only_sportsbooks(c)]


def _select_top_weighted_ev_for_allowed_keys(
    candidates: list[dict[str, Any]],
    allowed_keys: frozenset[str],
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for card in candidates:
        if not card or card.get("hr_model_only"):
            continue
        if card.get("lane_research_only"):
            continue
        mk = str(card.get("market_key") or "")
        if mk not in allowed_keys:
            continue
        if to_float(card.get("weighted_ev")) is None:
            continue
        eligible.append(card)

    if not eligible:
        return None

    eligible.sort(
        key=lambda c: (
            float(c.get("weighted_ev") or -999.0),
            float(c.get("probability_edge") or -999.0),
        ),
        reverse=True,
    )
    best = dict(eligible[0])
    best["top_ev_rank"] = 1
    best["top_ev_candidate_count"] = len(eligible)
    return best


def select_top_weighted_ev_pick(
    candidates: list[dict[str, Any]],
    *,
    allow_f5_team_total_fallback: bool | None = None,
) -> dict[str, Any] | None:
    """Return a copy of the best row by ``weighted_ev`` (tie-break ``probability_edge``).

    Skips HR model-only rows (no priced line), any candidate without ``weighted_ev``, and
    markets outside ``TOP_EV_ELIGIBLE_MARKET_KEYS`` (HR yes is not eligible).

    When the primary pool is empty but priced F5 team-total rows exist (common before full-game
    lines are posted), optionally falls back to those so Top EV / Daily Results are not blank.
    Disable with env ``TOP_EV_F5_TEAM_TOTAL_FALLBACK=false`` or ``allow_f5_team_total_fallback=False``.
    """
    primary = _select_top_weighted_ev_for_allowed_keys(candidates, TOP_EV_ELIGIBLE_MARKET_KEYS)
    if primary is not None:
        return primary
    use_fb = _top_ev_f5_team_total_fallback_enabled() if allow_f5_team_total_fallback is None else bool(allow_f5_team_total_fallback)
    if not use_fb:
        return None
    return _select_top_weighted_ev_for_allowed_keys(candidates, TOP_EV_FULL_MARKET_KEYS)
