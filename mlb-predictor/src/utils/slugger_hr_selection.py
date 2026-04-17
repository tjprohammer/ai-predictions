"""Shared logic for HR slugger picks: top batters per game, then optional global cap."""

from __future__ import annotations

from typing import Any

from src.utils import best_bets as best_bets_utils

# Show at most this many batters per game on the board / daily results / outcomes.
SLUGGER_HR_PER_GAME = 3

# Live board (`GET /api/games/board`): global cap after per-game ranking + round-robin across games.
# Keep small so the strip is scannable; elite sluggers still dominate whenever they play.
SLUGGER_BOARD_MAX_CARDS = 10

# Hot hitters page (`GET /api/hot-hitters`): allow a few more rows than the live board.
SLUGGER_HOT_HITTERS_PAGE_MAX_CARDS = 18

# Daily Results page: cap slugger HR rows (same round-robin family as the board; keep the table short).
SLUGGER_DAILY_RESULTS_MAX_CARDS = 8

# Skip obvious bench / pinch-only rows when projected PA is known (from HR feature snapshot).
SLUGGER_MIN_PROJECTED_PA = 2.0

# --- Blended ranking ---
# Older versions overweighted raw HR/PA and career HR skill (huge multipliers), so the same elite
# names surfaced almost every slate. We now anchor on **today's** price edge (WE / prob edge), add
# **matchup context** (park + opposing starter HR profile), and keep batter power / hot streaks as
# **mild** tie-breakers. ``_slugger_context_score`` is the primary sort key (higher = better).
_LEAGUE_HR_PER_PA_FLOOR = 0.022


def _compute_slugger_context_score(rec: dict[str, Any], card: dict[str, Any]) -> float:
    """Prioritize today's betting value, then park/starter context; power and BvP are tie-breakers."""
    wev_raw = _coerce_float(card.get("weighted_ev"))
    if wev_raw is None:
        wev_fit = -0.18
    else:
        wev_fit = max(-0.55, min(wev_raw, 0.95))
    pe_raw = _coerce_float(card.get("probability_edge"))
    if pe_raw is None:
        pe_fit = -0.12
    else:
        pe_fit = max(-0.45, min(pe_raw, 0.4))

    cw = _coerce_float(card.get("certainty_weight"))
    cw_fit = max(0.0, min(float(cw if cw is not None else 0.65), 1.0))

    hrpa = _coerce_float(rec.get("hr_per_pa_blended")) or 0.0
    prior_hr = _coerce_float(rec.get("season_prior_hr_per_pa")) or hrpa
    xw = _coerce_float(rec.get("xwoba_14")) or 0.0
    hh = _coerce_float(rec.get("hard_hit_pct_14")) or 0.0
    gr30 = _coerce_float(rec.get("hr_game_rate_30")) or 0.0
    hot7 = _coerce_float(rec.get("hit_rate_7")) or 0.0
    hot14 = _coerce_float(rec.get("hit_rate_14")) or 0.0
    streak = _coerce_float(rec.get("streak_len_capped")) or 0.0

    park = _coerce_float(rec.get("park_hr_factor"))
    op_hr9 = _coerce_float(rec.get("opposing_starter_hr_per_9"))
    barrel = _coerce_float(rec.get("opposing_starter_barrel_pct"))

    ab = int(rec.get("bvp_ab") or 0)
    ops = _coerce_float(rec.get("bvp_ops"))
    bvp_hr = _coerce_float(rec.get("bvp_home_runs")) or 0.0

    # Dominant: model vs book on this prop today (already lineup/market aware via WE pipeline).
    value_core = 14.0 * wev_fit + 5.0 * pe_fit + 0.45 * (cw_fit - 0.66)

    # Second: environment and opposing starter — changes by game even when the batter is familiar.
    matchup = 0.0
    if park is not None:
        matchup += max(0.0, park - 1.0) * 2.15
        matchup -= max(0.0, 0.97 - park) * 0.65
    if op_hr9 is not None:
        matchup += max(0.0, op_hr9 - 1.12) * 0.5
    if barrel is not None:
        matchup += max(0.0, barrel - 0.078) * 2.4

    # Mild: intrinsic HR skill (avoid 300–400× multipliers that froze the same stars on top).
    power_bonus = max(0.0, hrpa - _LEAGUE_HR_PER_PA_FLOOR) * 18.0
    pedigree_bonus = max(0.0, prior_hr - _LEAGUE_HR_PER_PA_FLOOR) * 10.0
    qc_lift = max(0.0, xw - 0.30) * 0.6 + max(0.0, hh - 0.40) * 0.12
    hot_lift = hot7 * 0.9 + hot14 * 0.35 + min(streak, 10.0) * 0.02
    hr_prop_lift = gr30 * 0.5

    bvp_lift = 0.0
    if ops is not None and ab > 0:
        sample = min(1.0, ab / 32.0)
        hr_vs_p = bvp_hr / float(ab)
        bvp_lift = sample * (0.38 * ops + 0.28 * hr_vs_p * 12.0)

    return (
        value_core
        + matchup
        + power_bonus
        + pedigree_bonus
        + qc_lift
        + hot_lift
        + hr_prop_lift
        + bvp_lift
    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_slugger_internal_fields(card: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in card.items() if not str(k).startswith("_slugger_")}


def _enrich_slugger_card(card: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """Attach ranking signals not present on the board card (features + market implied)."""
    out = dict(card)
    out["_slugger_hr_per_pa"] = _coerce_float(rec.get("hr_per_pa_blended"))
    out["_slugger_proj_pa"] = _coerce_float(rec.get("projected_plate_appearances"))
    out["_slugger_implied"] = best_bets_utils.american_implied_probability(rec.get("market_price"))
    out["_slugger_context_score"] = _compute_slugger_context_score(rec, out)
    return out


def slugger_hr_card_sort_key(card: dict[str, Any]) -> tuple[float, ...]:
    """Blend EV with power, hot bat, HR pedigree, and BvP; then EV/edge tie-breakers."""
    ctx = _coerce_float(card.get("_slugger_context_score"))
    wev = _coerce_float(card.get("weighted_ev"))
    pe = _coerce_float(card.get("probability_edge"))
    mp = _coerce_float(card.get("model_probability"))
    hrb = _coerce_float(card.get("_slugger_hr_per_pa"))
    impl = _coerce_float(card.get("_slugger_implied"))
    ppa = _coerce_float(card.get("_slugger_proj_pa"))
    return (
        float(ctx if ctx is not None else -1e18),
        float(wev if wev is not None else -1e9),
        float(pe if pe is not None else -1e9),
        float(hrb if hrb is not None else -1.0),
        float(impl if impl is not None else -1.0),
        float(ppa if ppa is not None else 0.0),
        float(mp if mp is not None else 0.0),
    )


def _round_robin_interleave_games(
    by_game: dict[int, list[dict[str, Any]]],
    max_cards: int,
) -> list[dict[str, Any]]:
    """Take 1st pick from each game, then 2nd, … so the board spans many matchups."""
    if max_cards <= 0:
        return []
    games = sorted(
        by_game.keys(),
        key=lambda gid: slugger_hr_card_sort_key(by_game[gid][0]) if by_game[gid] else (-1e9,) * 7,
        reverse=True,
    )
    merged: list[dict[str, Any]] = []
    round_idx = 0
    while len(merged) < max_cards:
        progressed = False
        for gid in games:
            if len(merged) >= max_cards:
                break
            lst = by_game.get(gid, [])
            if round_idx < len(lst):
                merged.append(lst[round_idx])
                progressed = True
        if not progressed:
            break
        round_idx += 1
    return merged


def iter_slugger_tracked_cards(
    hr_recs: list[dict[str, Any]],
    *,
    per_game: int = SLUGGER_HR_PER_GAME,
    max_cards: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build HR board cards from DB-shaped rows, keep up to ``per_game`` best per ``game_id``.

    Ranks by a **context score** anchored on weighted EV / probability edge (today's price vs
    model), plus park and opposing-starter matchup context; HR/PA and hot hitting are mild
    tie-breakers. Per-game sort still uses ``slugger_hr_card_sort_key`` (EV and implied as
    fallbacks when internal scores tie).

    With ``max_cards`` (board strip), interleaves games round-robin. Each card includes
    ``slugger_rank_in_game`` (1..per_game). Internal ``_slugger_*`` keys are stripped before return.
    """
    by_game: dict[int, list[dict[str, Any]]] = {}
    for rec in hr_recs:
        ppa = _coerce_float(rec.get("projected_plate_appearances"))
        if ppa is not None and ppa < SLUGGER_MIN_PROJECTED_PA:
            continue
        card = best_bets_utils.build_player_hr_board_card(rec)
        if not card:
            continue
        card = best_bets_utils.annotate_market_card_for_display(card)
        card = _enrich_slugger_card(card, rec)
        gid = int(card.get("game_id") or 0)
        if not gid:
            continue
        by_game.setdefault(gid, []).append(card)

    ranked_by_game: dict[int, list[dict[str, Any]]] = {}
    for gid, cards in by_game.items():
        cards.sort(key=slugger_hr_card_sort_key, reverse=True)
        ranked: list[dict[str, Any]] = []
        for rank, card in enumerate(cards[:per_game], start=1):
            c = dict(card)
            c["slugger_rank_in_game"] = rank
            ranked.append(c)
        ranked_by_game[gid] = ranked

    if max_cards is not None and max_cards >= 0:
        interleaved = _round_robin_interleave_games(ranked_by_game, max_cards)
        return [_strip_slugger_internal_fields(dict(c)) for c in interleaved]

    out: list[dict[str, Any]] = []
    for lst in ranked_by_game.values():
        out.extend(lst)
    out.sort(key=slugger_hr_card_sort_key, reverse=True)
    return [_strip_slugger_internal_fields(dict(c)) for c in out]
