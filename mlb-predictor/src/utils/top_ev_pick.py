"""Select the single highest weighted-EV line across team markets, 1+ hit, and pitcher K for a game."""

from __future__ import annotations

from typing import Any

from src.utils.best_bets import TOP_EV_ELIGIBLE_MARKET_KEYS, to_float


def collect_top_ev_candidates(
    detail: dict[str, Any],
    game_market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Priced EV candidates: team markets, 1+ hit, pitcher K (same pool as board / game detail Top EV)."""
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
    return candidates


def select_top_weighted_ev_pick(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a copy of the best row by ``weighted_ev`` (tie-break ``probability_edge``).

    Skips HR model-only rows (no priced line), any candidate without ``weighted_ev``, and
    markets outside ``TOP_EV_ELIGIBLE_MARKET_KEYS`` (HR yes is not eligible).
    """
    eligible: list[dict[str, Any]] = []
    for card in candidates:
        if not card or card.get("hr_model_only"):
            continue
        mk = str(card.get("market_key") or "")
        if mk not in TOP_EV_ELIGIBLE_MARKET_KEYS:
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
