from __future__ import annotations

import math
from statistics import median
from typing import Any

from src.utils.input_trust import ACTIONABLE_TRUST_GRADES
from src.utils.settings import get_settings


SUPPLEMENTAL_GAME_MARKET_TYPES = (
    "total",
    "home_team_total",
    "away_team_total",
    "moneyline",
    "run_line",
    "first_five_moneyline",
    "first_five_total",
    "first_five_spread",
    "first_five_team_total_away",
    "first_five_team_total_home",
    "alt_game_total",
)

BEST_BET_MARKET_KEYS = (
    "moneyline",
    "run_line",
    "game_total",
    "away_team_total",
    "home_team_total",
    "first_five_moneyline",
    "first_five_total",
    "first_five_spread",
    "first_five_team_total_away",
    "first_five_team_total_home",
)

BEST_BET_SELECTION_LIMIT_PER_GAME = 1
# Legacy tests may pass limit=5 explicitly. Board/API default is no fixed cap (use one green per game
# up to slate size — see ``flatten_best_bets``).
BOARD_BEST_BET_LIMIT = 5
# Slate-wide green strip: at most this many distinct picks per game (same pass logic for both passes).
BOARD_GREEN_STRIP_MAX_PER_GAME_FIRST_PASS = 1
# Main-board watchlist: include all per-game candidates up to this slate-wide cap.
BOARD_WATCHLIST_LIMIT = 500

# When strict `positive` is rare (slates, recalibrated models), still show meaningful edges on the green strip.
# These sit between "watchlist noise" and full `passes_best_bet_thresholds` gates.
# Soft backfill also requires actionable (A/B) input trust — strict positives use EV gates only so the
# strip can show one pick per game when the model clears the bar, even if lineups/markets are still B/C.
# Slightly relaxed vs original (2026) so pre-lock / full slates surface more than a single soft green.
# Tuned so thin slates are not reduced to a single green when models sit just below the strict gates.
BOARD_GREEN_SOFT_MIN_WEIGHTED_EV = 0.013
BOARD_GREEN_SOFT_MIN_PROB_EDGE = 0.024
BOARD_GREEN_SOFT_MIN_CERTAINTY = 0.70
BOARD_GREEN_SOFT_MIN_MODEL_PROB = 0.505

# First-inning experimental lines: nrfi and yrfi sort order when deduping / displaying.
EXPERIMENTAL_FIRST_INNING_MARKETS_ORDER = ("nrfi", "yrfi")

# Hitter props (1+ hits, HR yes) use EV machinery for grading/detail cards; they never belong on the team best-bet board.
PLAYER_HITS_MARKET_KEY = "player_hits"
# Priced "to hit a home run" / batter HR yes props.
PLAYER_HOME_RUN_MARKET_KEY = "player_home_run"
# HR probabilities are tiny; rounding to 4 decimals can collapse them to 0.0 in JSON/UI.
HR_MODEL_PROBABILITY_DECIMALS = 8


def is_player_prop_yes_market(market_key: str | None) -> bool:
    mk = str(market_key or "")
    return mk in (PLAYER_HITS_MARKET_KEY, PLAYER_HOME_RUN_MARKET_KEY)


def excluded_from_team_best_pick_board(market_key: str | None) -> bool:
    """True for markets that must not appear on green strip / team watchlist (best bets = team markets)."""
    return str(market_key or "") in (PLAYER_HITS_MARKET_KEY, PLAYER_HOME_RUN_MARKET_KEY)


BEST_BET_THRESHOLD_MAP: dict[str, dict[str, float]] = {
    "moneyline": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    "run_line": {
        "weighted_ev": 0.047,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    "game_total": {
        "weighted_ev": 0.047,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    PLAYER_HITS_MARKET_KEY: {
        "weighted_ev": 0.042,
        "probability_edge": 0.052,
        "certainty_weight": 0.70,
        "model_probability": 0.505,
    },
    PLAYER_HOME_RUN_MARKET_KEY: {
        "weighted_ev": 0.030,
        "probability_edge": 0.034,
        "certainty_weight": 0.66,
        "model_probability": 0.12,
    },
    "away_team_total": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.565,
    },
    "home_team_total": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.565,
    },
    "first_five_moneyline": {
        "weighted_ev": 0.047,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    "first_five_total": {
        "weighted_ev": 0.047,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    "first_five_spread": {
        "weighted_ev": 0.047,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.535,
    },
    "first_five_team_total_away": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.565,
    },
    "first_five_team_total_home": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.565,
    },
    "pitcher_strikeouts": {
        "weighted_ev": 0.038,
        "probability_edge": 0.065,
        "certainty_weight": 0.80,
        "model_probability": 0.525,
    },
}

# Team + pitcher K props + 1+ hit for Top EV. HR yes is excluded — rare events are poor anchors
# for the single headline EV pick vs team markets and contact props.
PITCHER_STRIKEOUTS_MARKET_KEY = "pitcher_strikeouts"
# First-five *team* O/U lines are omitted from product surfaces — often unavailable at regional
# books (e.g. FanDuel state menus) while another book in the consolidated feed still has a number.
_TOP_EV_EXCLUDED_MARKET_KEYS = frozenset({"first_five_team_total_away", "first_five_team_total_home"})
TOP_EV_ELIGIBLE_MARKET_KEYS = (
    frozenset(BEST_BET_MARKET_KEYS)
    | frozenset((PLAYER_HITS_MARKET_KEY, PITCHER_STRIKEOUTS_MARKET_KEY))
) - _TOP_EV_EXCLUDED_MARKET_KEYS

MARKET_SIM_MAX_RUNS = 16


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def format_hr_probability_pct_display(mp: float) -> str:
    """HR is a rare event — avoid rounding tiny probabilities to 0.00% or 0.000%."""
    pct = float(mp) * 100.0
    if pct <= 0 and mp and float(mp) > 0:
        pct = float(mp) * 100.0
    if pct <= 0:
        return "0%"
    if pct < 0.01:
        s = f"{pct:.8f}".rstrip("0").rstrip(".")
        return f"{s}%" if s else "0%"
    if pct < 0.1:
        return f"{pct:.3f}%"
    if pct < 1.0:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"


def _sanitize_hr_fair_american_for_display(value: Any) -> int | None:
    """Hide absurd placeholder fair odds from legacy rows (tiny p → huge American prices)."""
    v = to_float(value)
    if v is None:
        return None
    if abs(v) > 500_000:
        return None
    return int(round(v))


def format_price_text(price: Any) -> str:
    value = to_float(price)
    if value is None:
        return "-"
    rounded = int(round(value))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def format_market_line_text(value: Any) -> str:
    line = to_float(value)
    if line is None:
        return "-"
    return f"{line:+.1f}" if abs(line) >= 0.05 else "0.0"


def scale_expected_run_split(
    predicted_total: Any,
    away_weight: Any,
    home_weight: Any,
) -> tuple[float | None, float | None]:
    total = to_float(predicted_total)
    away = to_float(away_weight)
    home = to_float(home_weight)
    if total is None:
        return None, None
    if away is None and home is None:
        return None, None
    if away is None or away <= 0:
        away = home if home is not None and home > 0 else 1.0
    if home is None or home <= 0:
        home = away if away is not None and away > 0 else 1.0
    denominator = away + home
    if denominator <= 0:
        return None, None
    away_share = round(total * away / denominator, 3)
    home_share = round(total - away_share, 3)
    return away_share, home_share


def market_focus_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    line_values = [
        value
        for value in (to_float(row.get("line_value")) for row in rows)
        if value is not None
    ]
    if not line_values:
        return rows, None
    consensus_line = round(float(median(line_values)), 2)
    distances = [
        abs(value - consensus_line)
        for value in (to_float(row.get("line_value")) for row in rows)
        if value is not None
    ]
    if not distances:
        return rows, consensus_line
    min_distance = min(distances)
    focused = [
        row
        for row in rows
        if to_float(row.get("line_value")) is not None
        and abs(float(to_float(row.get("line_value")) or 0.0) - consensus_line) <= min_distance + 1e-9
    ]
    return focused or rows, consensus_line


def aggregate_supplemental_market_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        game_id = row.get("game_id")
        market_type = str(row.get("market_type") or "")
        if game_id is None or not market_type:
            continue
        grouped.setdefault(int(game_id), {}).setdefault(market_type, []).append(row)

    result: dict[int, dict[str, Any]] = {}
    for game_id, market_groups in grouped.items():
        result[game_id] = {}
        for market_type, market_rows in market_groups.items():
            focused_rows, consensus_line = market_focus_rows(market_rows)
            best_over_row = max(
                (row for row in focused_rows if row.get("over_price") is not None),
                key=lambda row: int(row.get("over_price")),
                default=None,
            )
            best_under_row = max(
                (row for row in focused_rows if row.get("under_price") is not None),
                key=lambda row: int(row.get("under_price")),
                default=None,
            )
            result[game_id][market_type] = {
                "line_value": consensus_line,
                "over_price": None if best_over_row is None else best_over_row.get("over_price"),
                "under_price": None if best_under_row is None else best_under_row.get("under_price"),
                "sportsbook": (best_over_row or best_under_row or {}).get("sportsbook"),
                "best_over_sportsbook": None if best_over_row is None else best_over_row.get("sportsbook"),
                "best_under_sportsbook": None if best_under_row is None else best_under_row.get("sportsbook"),
                "sportsbook_count": len(market_rows),
                "latest_snapshot_ts": max(
                    (row.get("snapshot_ts") for row in market_rows if row.get("snapshot_ts") is not None),
                    default=None,
                ),
            }
    return result


def american_implied_probability(price: Any) -> float | None:
    value = to_float(price)
    if value is None or abs(value) < 1e-9:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def american_profit_per_unit(price: Any) -> float | None:
    value = to_float(price)
    if value is None or abs(value) < 1e-9:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


def no_vig_pair(price_a: Any, price_b: Any) -> tuple[float | None, float | None]:
    implied_a = american_implied_probability(price_a)
    implied_b = american_implied_probability(price_b)
    if implied_a is None or implied_b is None:
        return None, None
    total = implied_a + implied_b
    if total <= 0:
        return None, None
    return implied_a / total, implied_b / total


def poisson_distribution(mean_runs: Any, max_runs: int = MARKET_SIM_MAX_RUNS) -> list[float] | None:
    mean_value = to_float(mean_runs)
    if mean_value is None or mean_value < 0:
        return None
    probabilities = [math.exp(-mean_value)]
    for run_total in range(1, max_runs + 1):
        probabilities.append(probabilities[-1] * mean_value / float(run_total))
    total_probability = sum(probabilities)
    if total_probability <= 0:
        return None
    if total_probability < 1.0:
        probabilities[-1] += 1.0 - total_probability
    else:
        probabilities = [probability / total_probability for probability in probabilities]
    return probabilities


def joint_run_distribution(away_mean: Any, home_mean: Any) -> dict[str, float] | None:
    away_probs = poisson_distribution(away_mean)
    home_probs = poisson_distribution(home_mean)
    if away_probs is None or home_probs is None:
        return None
    away_win = 0.0
    home_win = 0.0
    tie = 0.0
    for away_runs, away_probability in enumerate(away_probs):
        for home_runs, home_probability in enumerate(home_probs):
            joint_probability = away_probability * home_probability
            if away_runs > home_runs:
                away_win += joint_probability
            elif home_runs > away_runs:
                home_win += joint_probability
            else:
                tie += joint_probability
    total_probability = away_win + home_win + tie
    if total_probability <= 0:
        return None
    scale = 1.0 / total_probability
    return {
        "away_win": away_win * scale,
        "home_win": home_win * scale,
        "tie": tie * scale,
    }


def team_total_side_probabilities(team_mean: Any, line_value: Any) -> dict[str, dict[str, float]] | None:
    distribution = poisson_distribution(team_mean)
    line = to_float(line_value)
    if distribution is None or line is None:
        return None
    over_win = 0.0
    under_win = 0.0
    push = 0.0
    for runs, probability in enumerate(distribution):
        if runs > line:
            over_win += probability
        elif runs < line:
            under_win += probability
        else:
            push += probability
    return {
        "over": {"win": over_win, "loss": under_win, "push": push},
        "under": {"win": under_win, "loss": over_win, "push": push},
    }


def moneyline_side_probabilities(
    away_mean: Any,
    home_mean: Any,
    *,
    push_on_tie: bool,
) -> dict[str, dict[str, float]] | None:
    joint = joint_run_distribution(away_mean, home_mean)
    if joint is None:
        return None
    if push_on_tie:
        return {
            "away": {"win": joint["away_win"], "loss": joint["home_win"], "push": joint["tie"]},
            "home": {"win": joint["home_win"], "loss": joint["away_win"], "push": joint["tie"]},
        }
    away_mean_value = to_float(away_mean) or 0.0
    home_mean_value = to_float(home_mean) or 0.0
    total_mean = away_mean_value + home_mean_value
    home_tie_share = 0.5 if total_mean <= 0 else home_mean_value / total_mean
    away_tie_share = 1.0 - home_tie_share
    return {
        "away": {
            "win": joint["away_win"] + (joint["tie"] * away_tie_share),
            "loss": joint["home_win"] + (joint["tie"] * home_tie_share),
            "push": 0.0,
        },
        "home": {
            "win": joint["home_win"] + (joint["tie"] * home_tie_share),
            "loss": joint["away_win"] + (joint["tie"] * away_tie_share),
            "push": 0.0,
        },
    }


def run_line_side_probabilities(
    away_mean: Any,
    home_mean: Any,
    home_line_value: Any,
) -> dict[str, dict[str, float]] | None:
    joint = joint_run_distribution(away_mean, home_mean)
    home_line = to_float(home_line_value)
    away_probs = poisson_distribution(away_mean)
    home_probs = poisson_distribution(home_mean)
    if joint is None or home_line is None or away_probs is None or home_probs is None:
        return None
    home_cover = 0.0
    away_cover = 0.0
    push = 0.0
    for away_runs, away_probability in enumerate(away_probs):
        for home_runs, home_probability in enumerate(home_probs):
            joint_probability = away_probability * home_probability
            adjusted_home = float(home_runs) + home_line
            if adjusted_home > float(away_runs):
                home_cover += joint_probability
            elif adjusted_home < float(away_runs):
                away_cover += joint_probability
            else:
                push += joint_probability
    total_probability = home_cover + away_cover + push
    if total_probability <= 0:
        return None
    scale = 1.0 / total_probability
    home_cover *= scale
    away_cover *= scale
    push *= scale
    return {
        "home": {"win": home_cover, "loss": away_cover, "push": push},
        "away": {"win": away_cover, "loss": home_cover, "push": push},
    }


def game_certainty_weight(certainty: dict[str, Any] | None) -> float:
    certainty = certainty or {}
    weighted_values = [
        (to_float(certainty.get("starter_certainty")), 0.30),
        (to_float(certainty.get("lineup_certainty")), 0.26),
        (to_float(certainty.get("market_freshness")), 0.18),
        (to_float(certainty.get("weather_freshness")), 0.12),
        (to_float(certainty.get("bullpen_completeness")), 0.14),
    ]
    numerator = 0.0
    denominator = 0.0
    for index, (value, weight) in enumerate(weighted_values):
        if value is None:
            continue
        # A future slate often has no confirmed lineups yet; treat a zero lineup
        # certainty as unavailable data rather than a hard negative signal.
        if index == 1 and value <= 0:
            continue
        numerator += max(0.0, min(1.0, value)) * weight
        denominator += weight
    if denominator <= 0:
        return 0.6
    return numerator / denominator


def card_input_trust_from_game(game: dict[str, Any]) -> dict[str, Any]:
    cert = game.get("certainty")
    if isinstance(cert, dict):
        nested = cert.get("input_trust")
        if isinstance(nested, dict) and nested:
            return dict(nested)
    it = game.get("input_trust")
    if isinstance(it, dict) and it:
        return dict(it)
    return {}


def promotion_tier_for_card(*, positive: bool, input_trust: dict[str, Any]) -> str:
    if not positive:
        return "none"
    grade = str((input_trust or {}).get("grade") or "").strip().upper()
    if grade in ACTIONABLE_TRUST_GRADES:
        return "actionable"
    return "edge_only"


def market_thresholds(market_key: str) -> dict[str, float]:
    return BEST_BET_THRESHOLD_MAP.get(
        market_key,
        {
            "weighted_ev": 0.012,
            "probability_edge": 0.020,
            "certainty_weight": 0.80,
            "model_probability": 0.530,
        },
    )


def passes_best_bet_thresholds(
    market_key: str,
    *,
    weighted_ev: float,
    probability_edge: float,
    certainty_weight: float,
    model_probability: float,
) -> bool:
    thresholds = market_thresholds(market_key)
    return bool(
        weighted_ev >= float(thresholds["weighted_ev"])
        and probability_edge >= float(thresholds["probability_edge"])
        and certainty_weight >= float(thresholds["certainty_weight"])
        and model_probability >= float(thresholds["model_probability"])
    )


def _strict_ev_gate_hints(card: dict[str, Any]) -> list[str]:
    """Human-readable reasons a card is not `positive` (full gates), for game-detail UI."""
    market_key = str(card.get("market_key") or "")
    if (
        market_key not in BEST_BET_MARKET_KEYS
        and not is_player_prop_yes_market(market_key)
        and market_key != PITCHER_STRIKEOUTS_MARKET_KEY
    ) or card.get("positive"):
        return []
    thresholds = market_thresholds(market_key)
    wev = to_float(card.get("weighted_ev"))
    pe = to_float(card.get("probability_edge"))
    cw = to_float(card.get("certainty_weight"))
    mp = to_float(card.get("model_probability"))
    t_wev = float(thresholds["weighted_ev"])
    t_pe = float(thresholds["probability_edge"])
    t_cw = float(thresholds["certainty_weight"])
    t_mp = float(thresholds["model_probability"])
    hints: list[str] = []
    if wev is None or wev < t_wev:
        hints.append(
            f"Weighted EV {('—' if wev is None else f'{wev * 100:.1f}%')} vs gate {t_wev * 100:.0f}%"
        )
    if pe is None or pe < t_pe:
        hints.append(
            f"No-vig edge {('—' if pe is None else f'{pe * 100:.1f} pts')} vs gate {t_pe * 100:.0f} pts"
        )
    if cw is None or cw < t_cw:
        hints.append(
            f"Input certainty {('—' if cw is None else f'{cw * 100:.0f}%')} vs gate {t_cw * 100:.0f}%"
        )
    if mp is None or mp < t_mp:
        hints.append(
            f"Model prob {('—' if mp is None else f'{mp * 100:.1f}%')} vs gate {t_mp * 100:.0f}%"
        )
    return hints


def annotate_market_card_for_display(card: dict[str, Any]) -> dict[str, Any]:
    """Add `board_pick_tier`, `board_badges`, and optional `ev_gate_hints` for UI transparency."""
    if not card:
        return card
    mk = str(card.get("market_key") or "")
    badges: list[dict[str, str]] = []
    tier = "monitor"

    if card.get("hr_model_only"):
        badges.append({"key": "model_only", "label": "Model only (no book line)"})

    if card.get("positive"):
        tier = "strict"
        badges.append({"key": "strict", "label": "Full EV gates"})
    elif mk in BEST_BET_MARKET_KEYS and qualifies_board_green_strip(
        card
    ):
        tier = "soft_green"
        badges.append({"key": "soft_green", "label": "Soft green strip"})
    elif _is_watchlist_candidate(card):
        tier = "watchlist"
        badges.append({"key": "watchlist", "label": "Watchlist"})
    else:
        wev = to_float(card.get("weighted_ev"))
        pe = to_float(card.get("probability_edge"))
        if wev is not None and wev > 0:
            tier = "raw_edge"
            badges.append({"key": "raw_edge", "label": "Raw EV only"})
        elif pe is not None and pe >= 0.015:
            tier = "lean"
            badges.append({"key": "lean", "label": "Lean vs no-vig"})
        if not badges and (card.get("weighted_ev") is None and card.get("model_probability") is None):
            badges.append({"key": "incomplete", "label": "Incomplete line"})

    out = dict(card)
    out["board_pick_tier"] = tier
    out["board_badges"] = badges
    if not card.get("positive") and (
        mk in BEST_BET_MARKET_KEYS or is_player_prop_yes_market(mk) or mk == PITCHER_STRIKEOUTS_MARKET_KEY
    ):
        hints = _strict_ev_gate_hints(card)
        if hints:
            out["ev_gate_hints"] = hints
    if mk in BEST_BET_MARKET_KEYS:
        try:
            from src.models.board_action_score import maybe_attach_action_score

            out = maybe_attach_action_score(out)
        except Exception:
            pass
    return out


def _game_certainty_pct_from_game(game: dict[str, Any]) -> float | None:
    dq = game.get("data_quality") or {}
    v = dq.get("certainty_pct")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _meets_board_green_min_game_certainty(card: dict[str, Any]) -> bool:
    """Optional gate: per-game ``data_quality.certainty_pct`` (0–100) vs settings."""
    min_pct = get_settings().board_green_min_game_certainty_pct
    if min_pct is None:
        return True
    gcp = card.get("game_certainty_pct")
    if gcp is None:
        return False
    try:
        return float(gcp) >= float(min_pct)
    except (TypeError, ValueError):
        return False


def _input_trust_actionable_for_green_strip(card: dict[str, Any]) -> bool:
    """Green strip: only A/B input trust — same bar as ``promotion_tier`` actionable (not edge-only C/D)."""
    it = card.get("input_trust")
    if not isinstance(it, dict):
        return False
    grade = str(it.get("grade") or "").strip().upper()
    return grade in ACTIONABLE_TRUST_GRADES


def qualifies_board_green_strip(card: dict[str, Any]) -> bool:
    """Strict green (`positive`) uses full EV gates only. Soft backfill needs soft thresholds + A/B trust.

    Rationale: `positive` already encodes strong model + price edges; input trust mainly filters
    marginal soft greens when pregame context is weak.
    """
    mk = str(card.get("market_key") or "")
    if excluded_from_team_best_pick_board(mk):
        return False
    if not card or mk not in BEST_BET_MARKET_KEYS:
        return False
    if not _meets_board_green_min_game_certainty(card):
        return False
    if card.get("positive"):
        return True
    wev = to_float(card.get("weighted_ev"))
    pe = to_float(card.get("probability_edge"))
    cw = to_float(card.get("certainty_weight"))
    mp = to_float(card.get("model_probability"))
    if wev is None or pe is None or cw is None or mp is None:
        return False
    if not (
        cw >= BOARD_GREEN_SOFT_MIN_CERTAINTY
        and wev >= BOARD_GREEN_SOFT_MIN_WEIGHTED_EV
        and pe >= BOARD_GREEN_SOFT_MIN_PROB_EDGE
        and mp >= BOARD_GREEN_SOFT_MIN_MODEL_PROB
    ):
        return False
    return _input_trust_actionable_for_green_strip(card)


def build_market_candidate(
    *,
    game: dict[str, Any],
    market_key: str,
    market_label: str,
    selection_label: str,
    bet_side: str,
    sportsbook: Any,
    line_value: Any,
    price: Any,
    opposing_price: Any,
    model_probability: float | None,
    model_loss_probability: float | None,
    push_probability: float | None,
    certainty_weight: float,
    market_summary: str,
    model_summary: str,
) -> dict[str, Any] | None:
    fair_probability, _ = no_vig_pair(price, opposing_price)
    profit_per_unit = american_profit_per_unit(price)
    if fair_probability is None or profit_per_unit is None or model_probability is None:
        return None
    loss_probability = model_loss_probability
    push_probability = 0.0 if push_probability is None else max(0.0, push_probability)
    if loss_probability is None:
        loss_probability = max(0.0, 1.0 - model_probability - push_probability)
    raw_ev = (model_probability * profit_per_unit) - loss_probability
    probability_edge = model_probability - fair_probability
    weighted_ev = raw_ev * certainty_weight
    positive = passes_best_bet_thresholds(
        market_key,
        weighted_ev=weighted_ev,
        probability_edge=probability_edge,
        certainty_weight=certainty_weight,
        model_probability=model_probability,
    )
    it = card_input_trust_from_game(game)
    gcp = _game_certainty_pct_from_game(game)
    return {
        "game_id": int(game.get("game_id") or 0),
        "away_team": game.get("away_team"),
        "home_team": game.get("home_team"),
        "game_certainty_pct": gcp,
        "market_key": market_key,
        "market_label": market_label,
        "selection_label": selection_label,
        "bet_side": bet_side,
        "sportsbook": sportsbook,
        "line_value": to_float(line_value),
        "price": None if price is None else int(price),
        "opposing_price": None if opposing_price is None else int(opposing_price),
        "model_probability": round(float(model_probability), 4),
        "no_vig_probability": round(float(fair_probability), 4),
        "probability_edge": round(float(probability_edge), 4),
        "raw_ev": round(float(raw_ev), 4),
        "weighted_ev": round(float(weighted_ev), 4),
        "certainty_weight": round(float(certainty_weight), 4),
        "push_probability": round(float(push_probability), 4),
        "positive": positive,
        "input_trust": it,
        "promotion_tier": promotion_tier_for_card(positive=positive, input_trust=it),
        "market_summary": market_summary,
        "model_summary": model_summary,
    }


def fallback_market_card(
    *,
    game: dict[str, Any],
    market_key: str,
    market_label: str,
    market_summary: str,
    model_summary: str,
) -> dict[str, Any]:
    return {
        "game_id": int(game.get("game_id") or 0),
        "away_team": game.get("away_team"),
        "home_team": game.get("home_team"),
        "game_certainty_pct": _game_certainty_pct_from_game(game),
        "market_key": market_key,
        "market_label": market_label,
        "selection_label": None,
        "bet_side": None,
        "sportsbook": None,
        "line_value": None,
        "price": None,
        "opposing_price": None,
        "model_probability": None,
        "no_vig_probability": None,
        "probability_edge": None,
        "raw_ev": None,
        "weighted_ev": None,
        "certainty_weight": None,
        "push_probability": None,
        "positive": False,
        "input_trust": card_input_trust_from_game(game),
        "promotion_tier": "none",
        "market_summary": market_summary,
        "model_summary": model_summary,
    }


def best_market_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate.get("weighted_ev") or -999.0),
            float(candidate.get("probability_edge") or -999.0),
            float(candidate.get("raw_ev") or -999.0),
            float(candidate.get("model_probability") or -999.0),
        ),
    )


def hitter_lineup_certainty_weight(record: dict[str, Any]) -> float:
    if record.get("is_confirmed_lineup"):
        return 0.85
    if record.get("has_lineup_snapshot"):
        return 0.78
    return 0.65


def build_player_hits_board_card(record: dict[str, Any]) -> dict[str, Any] | None:
    """1+ hit yes-side card aligned with team market EV tiers (hot hitter / detail surfaces)."""
    mp = to_float(record.get("predicted_hit_probability"))
    price = record.get("market_price")
    if mp is None:
        return None
    price_f = to_float(price)
    if price_f is None:
        return None
    implied = american_implied_probability(price_f)
    pe = to_float(record.get("edge"))
    if pe is None and implied is not None:
        pe = mp - implied
    if pe is None:
        return None
    profit = american_profit_per_unit(price_f)
    if profit is None:
        return None
    loss_p = max(0.0, 1.0 - mp)
    raw_ev = (mp * profit) - loss_p
    cw = hitter_lineup_certainty_weight(record)
    weighted_ev = raw_ev * cw
    fair_p = implied if implied is not None else max(0.0, min(1.0, mp - pe))
    player_name = str(record.get("player_name") or "Player")
    team = str(record.get("team") or "")
    away_team = str(record.get("away_team") or "")
    home_team = str(record.get("home_team") or "")
    market_summary = (
        f"Yes (1+ H) {format_price_text(price_f)}"
        + (f" · Fair {format_price_text(record.get('fair_price'))}" if record.get("fair_price") is not None else "")
    )
    model_summary = (
        f"Model P(1+ H) {mp * 100:.1f}% · Market implied {fair_p * 100:.1f}%"
        if fair_p is not None
        else f"Model P(1+ H) {mp * 100:.1f}%"
    )
    positive = passes_best_bet_thresholds(
        PLAYER_HITS_MARKET_KEY,
        weighted_ev=float(weighted_ev),
        probability_edge=float(pe),
        certainty_weight=float(cw),
        model_probability=float(mp),
    )
    trust_grade = "A" if record.get("is_confirmed_lineup") else ("B" if record.get("has_lineup_snapshot") else "C")
    it = {"grade": trust_grade}
    return {
        "game_id": int(record.get("game_id") or 0),
        "player_id": int(record["player_id"]) if record.get("player_id") is not None else None,
        "away_team": away_team,
        "home_team": home_team,
        "market_key": PLAYER_HITS_MARKET_KEY,
        "market_label": "1+ Hits",
        "selection_label": f"{player_name} Yes (1+ H)",
        "pick_label": f"{player_name} · 1+ Hits",
        "subject_label": f"{player_name} ({team})" if team else player_name,
        "subject_subtitle": f"{away_team} at {home_team}" if away_team and home_team else "",
        "bet_side": "yes",
        "sportsbook": record.get("sportsbook"),
        "line_value": None,
        "price": int(round(price_f)),
        "opposing_price": None,
        "model_probability": round(float(mp), 4),
        "no_vig_probability": round(float(fair_p), 4) if fair_p is not None else None,
        "probability_edge": round(float(pe), 4),
        "raw_ev": round(float(raw_ev), 4),
        "weighted_ev": round(float(weighted_ev), 4),
        "certainty_weight": round(float(cw), 4),
        "push_probability": 0.0,
        "positive": positive,
        "input_trust": it,
        "promotion_tier": promotion_tier_for_card(positive=positive, input_trust=it),
        "market_summary": market_summary,
        "model_summary": model_summary,
    }


def _build_player_hr_board_card_model_only(record: dict[str, Any], mp: float) -> dict[str, Any]:
    """HR card when the model scored the prop but no sportsbook line was ingested for that batter."""
    player_name = str(record.get("player_name") or "Player")
    team = str(record.get("team") or "")
    away_team = str(record.get("away_team") or "")
    home_team = str(record.get("home_team") or "")
    cw = hitter_lineup_certainty_weight(record)
    trust_grade = "A" if record.get("is_confirmed_lineup") else ("B" if record.get("has_lineup_snapshot") else "C")
    it = {"grade": trust_grade}
    fair_bits = ""
    fair_disp = _sanitize_hr_fair_american_for_display(record.get("fair_price"))
    if fair_disp is not None:
        fair_bits = f"Fair {format_price_text(fair_disp)} · "
    market_summary = (
        f"{fair_bits}No priced HR line in player_prop_markets for this matchup — model projection only."
    )
    phr = format_hr_probability_pct_display(mp)
    model_summary = f"Model P(HR) {phr}"
    return {
        "game_id": int(record.get("game_id") or 0),
        "player_id": int(record["player_id"]) if record.get("player_id") is not None else None,
        "away_team": away_team,
        "home_team": home_team,
        "market_key": PLAYER_HOME_RUN_MARKET_KEY,
        "market_label": "To hit a HR",
        "selection_label": f"{player_name} · HR (model)",
        "pick_label": f"{player_name} · HR",
        "subject_label": f"{player_name} ({team})" if team else player_name,
        "subject_subtitle": f"{away_team} at {home_team}" if away_team and home_team else "",
        "bet_side": "yes",
        "sportsbook": record.get("sportsbook"),
        "line_value": None,
        "price": None,
        "opposing_price": None,
        "model_probability": round(float(mp), HR_MODEL_PROBABILITY_DECIMALS),
        "no_vig_probability": None,
        "probability_edge": None,
        "raw_ev": None,
        "weighted_ev": None,
        "certainty_weight": round(float(cw), 4),
        "push_probability": 0.0,
        "positive": False,
        "input_trust": it,
        "promotion_tier": promotion_tier_for_card(positive=False, input_trust=it),
        "market_summary": market_summary,
        "model_summary": model_summary,
        "hr_probability_display": phr,
        "hr_reasoning": record.get("hr_reasoning") or [],
        "hr_model_only": True,
    }


def build_player_hr_board_card(record: dict[str, Any]) -> dict[str, Any] | None:
    """Yes-side HR prop card (priced over / yes lines). Rare events — thresholds are looser than 1+ hits."""
    mp = to_float(record.get("predicted_hr_probability"))
    price = record.get("market_price")
    if mp is None:
        return None
    price_f = to_float(price)
    if price_f is None:
        return _build_player_hr_board_card_model_only(record, float(mp))
    implied = american_implied_probability(price_f)
    pe = to_float(record.get("edge"))
    if pe is None and implied is not None:
        pe = mp - implied
    if pe is None:
        return None
    profit = american_profit_per_unit(price_f)
    if profit is None:
        return None
    loss_p = max(0.0, 1.0 - mp)
    raw_ev = (mp * profit) - loss_p
    cw = hitter_lineup_certainty_weight(record)
    weighted_ev = raw_ev * cw
    fair_p = implied if implied is not None else max(0.0, min(1.0, mp - pe))
    player_name = str(record.get("player_name") or "Player")
    team = str(record.get("team") or "")
    away_team = str(record.get("away_team") or "")
    home_team = str(record.get("home_team") or "")
    fair_disp = _sanitize_hr_fair_american_for_display(record.get("fair_price"))
    market_summary = (
        f"Yes (HR) {format_price_text(price_f)}"
        + (f" · Fair {format_price_text(fair_disp)}" if fair_disp is not None else "")
    )
    phr = format_hr_probability_pct_display(mp)
    model_summary = (
        f"Model P(HR) {phr} · Market implied {fair_p * 100:.2f}%"
        if fair_p is not None
        else f"Model P(HR) {phr}"
    )
    positive = passes_best_bet_thresholds(
        PLAYER_HOME_RUN_MARKET_KEY,
        weighted_ev=float(weighted_ev),
        probability_edge=float(pe),
        certainty_weight=float(cw),
        model_probability=float(mp),
    )
    trust_grade = "A" if record.get("is_confirmed_lineup") else ("B" if record.get("has_lineup_snapshot") else "C")
    it = {"grade": trust_grade}
    return {
        "game_id": int(record.get("game_id") or 0),
        "player_id": int(record["player_id"]) if record.get("player_id") is not None else None,
        "away_team": away_team,
        "home_team": home_team,
        "market_key": PLAYER_HOME_RUN_MARKET_KEY,
        "market_label": "To hit a HR",
        "selection_label": f"{player_name} Yes (HR)",
        "pick_label": f"{player_name} · HR",
        "subject_label": f"{player_name} ({team})" if team else player_name,
        "subject_subtitle": f"{away_team} at {home_team}" if away_team and home_team else "",
        "bet_side": "yes",
        "sportsbook": record.get("sportsbook"),
        "line_value": None,
        "price": int(round(price_f)),
        "opposing_price": None,
        "model_probability": round(float(mp), HR_MODEL_PROBABILITY_DECIMALS),
        "no_vig_probability": round(float(fair_p), 4) if fair_p is not None else None,
        "probability_edge": round(float(pe), 4),
        "raw_ev": round(float(raw_ev), 4),
        "weighted_ev": round(float(weighted_ev), 4),
        "certainty_weight": round(float(cw), 4),
        "push_probability": 0.0,
        "positive": positive,
        "input_trust": it,
        "promotion_tier": promotion_tier_for_card(positive=positive, input_trust=it),
        "market_summary": market_summary,
        "model_summary": model_summary,
        "hr_probability_display": phr,
        "hr_reasoning": record.get("hr_reasoning") or [],
    }


def build_market_cards_for_game(
    game: dict[str, Any],
    market_rows_by_type: dict[str, list[dict[str, Any]]],
    *,
    all_scored_team_candidates: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    away_team = str(game.get("away_team") or "Away")
    home_team = str(game.get("home_team") or "Home")
    totals = dict(game.get("totals") or {})
    first5_totals = dict(game.get("first5_totals") or {})
    certainty_weight = game_certainty_weight(game.get("certainty"))

    away_expected_runs = to_float(totals.get("away_expected_runs"))
    home_expected_runs = to_float(totals.get("home_expected_runs"))
    full_game_total_mean = to_float(totals.get("predicted_total_runs"))
    if full_game_total_mean is None and away_expected_runs is not None and home_expected_runs is not None:
        full_game_total_mean = away_expected_runs + home_expected_runs
    first5_away_runs = to_float(first5_totals.get("away_runs") or first5_totals.get("away_expected_runs"))
    first5_home_runs = to_float(first5_totals.get("home_runs") or first5_totals.get("home_expected_runs"))
    first5_total_runs = to_float(first5_totals.get("total_runs") or first5_totals.get("predicted_total_runs"))
    if first5_total_runs is None and first5_away_runs is not None and first5_home_runs is not None:
        first5_total_runs = first5_away_runs + first5_home_runs

    full_game_moneyline = moneyline_side_probabilities(away_expected_runs, home_expected_runs, push_on_tie=False)
    first5_moneyline = moneyline_side_probabilities(first5_away_runs, first5_home_runs, push_on_tie=True)

    cards: list[dict[str, Any]] = []

    moneyline_rows = market_rows_by_type.get("moneyline") or []
    if moneyline_rows:
        candidates = []
        for row in moneyline_rows:
            market_summary = f"{away_team} {format_price_text(row.get('under_price'))} / {home_team} {format_price_text(row.get('over_price'))}"
            model_summary = (
                f"Model win prob {away_team} {format(full_game_moneyline['away']['win'] * 100, '.1f') if full_game_moneyline else '-'}% / "
                f"{home_team} {format(full_game_moneyline['home']['win'] * 100, '.1f') if full_game_moneyline else '-'}%"
            )
            if full_game_moneyline is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key="moneyline", market_label="Moneyline", selection_label=f"{away_team} ML", bet_side="away", sportsbook=row.get("sportsbook"), line_value=None, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=full_game_moneyline["away"]["win"], model_loss_probability=full_game_moneyline["away"]["loss"], push_probability=full_game_moneyline["away"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key="moneyline", market_label="Moneyline", selection_label=f"{home_team} ML", bet_side="home", sportsbook=row.get("sportsbook"), line_value=None, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=full_game_moneyline["home"]["win"], model_loss_probability=full_game_moneyline["home"]["loss"], push_probability=full_game_moneyline["home"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key="moneyline", market_label="Moneyline", market_summary=f"{away_team} / {home_team}", model_summary="Win probability unavailable."))

    run_line_rows, _ = market_focus_rows(market_rows_by_type.get("run_line") or [])
    if run_line_rows:
        candidates = []
        for row in run_line_rows:
            home_line = to_float(row.get("line_value"))
            probabilities = run_line_side_probabilities(away_expected_runs, home_expected_runs, home_line)
            market_summary = f"{home_team} {format_market_line_text(home_line)} {format_price_text(row.get('over_price'))} / {away_team} {format_market_line_text(None if home_line is None else -home_line)} {format_price_text(row.get('under_price'))}"
            model_summary = (
                f"Cover prob {home_team} {format(probabilities['home']['win'] * 100, '.1f') if probabilities else '-'}% / "
                f"{away_team} {format(probabilities['away']['win'] * 100, '.1f') if probabilities else '-'}%"
            )
            if probabilities is not None and home_line is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key="run_line", market_label="Run Line", selection_label=f"{home_team} {format_market_line_text(home_line)}", bet_side="home", sportsbook=row.get("sportsbook"), line_value=home_line, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=probabilities["home"]["win"], model_loss_probability=probabilities["home"]["loss"], push_probability=probabilities["home"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key="run_line", market_label="Run Line", selection_label=f"{away_team} {format_market_line_text(-home_line)}", bet_side="away", sportsbook=row.get("sportsbook"), line_value=-home_line, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=probabilities["away"]["win"], model_loss_probability=probabilities["away"]["loss"], push_probability=probabilities["away"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key="run_line", market_label="Run Line", market_summary=f"{away_team} / {home_team}", model_summary="Run-line probability unavailable."))

    total_rows, _ = market_focus_rows(market_rows_by_type.get("total") or [])
    # Book odds live in ``game_markets`` (``market_type='total'``). When that ingest is empty but
    # ``predictions_totals`` already has ``market_total`` + projected runs, still build a game_total
    # card using the model's consensus line and neutral −110/−110 for no-vig EV (same idea as a
    # placeholder until ``game_markets`` rows land).
    total_rows_effective: list[dict[str, Any]] = list(total_rows)
    synthetic_total_prices = False
    if not total_rows_effective and full_game_total_mean is not None:
        mt = to_float(totals.get("market_total"))
        if mt is not None:
            total_rows_effective = [
                {
                    "line_value": mt,
                    "over_price": -110,
                    "under_price": -110,
                    "sportsbook": totals.get("market_sportsbook") or "model_market_total",
                }
            ]
            synthetic_total_prices = True

    if total_rows_effective:
        if full_game_total_mean is not None:
            candidates = []
            for row in total_rows_effective:
                line_value = to_float(row.get("line_value"))
                probabilities = team_total_side_probabilities(full_game_total_mean, line_value)
                price_note = (
                    " · EV vs −110/−110 placeholder (no game_markets total row)"
                    if synthetic_total_prices
                    else ""
                )
                market_summary = (
                    f"Full-game run total {format(line_value, '.1f') if line_value is not None else '-'} · "
                    f"Over {format_price_text(row.get('over_price'))} / Under {format_price_text(row.get('under_price'))}"
                    f"{price_note}"
                )
                model_summary = (
                    f"Mean {format(full_game_total_mean, '.2f') if full_game_total_mean is not None else '-'} runs · Over {format(probabilities['over']['win'] * 100, '.1f') if probabilities else '-'}% / "
                    f"Under {format(probabilities['under']['win'] * 100, '.1f') if probabilities else '-'}%"
                )
                if probabilities is not None and line_value is not None:
                    candidates.extend(
                        filter(
                            None,
                            [
                                build_market_candidate(
                                    game=game,
                                    market_key="game_total",
                                    market_label="Game Total (Runs)",
                                    selection_label=f"Over {line_value:.1f} runs",
                                    bet_side="over",
                                    sportsbook=row.get("sportsbook"),
                                    line_value=line_value,
                                    price=row.get("over_price"),
                                    opposing_price=row.get("under_price"),
                                    model_probability=probabilities["over"]["win"],
                                    model_loss_probability=probabilities["over"]["loss"],
                                    push_probability=probabilities["over"]["push"],
                                    certainty_weight=certainty_weight,
                                    market_summary=market_summary,
                                    model_summary=model_summary,
                                ),
                                build_market_candidate(
                                    game=game,
                                    market_key="game_total",
                                    market_label="Game Total (Runs)",
                                    selection_label=f"Under {line_value:.1f} runs",
                                    bet_side="under",
                                    sportsbook=row.get("sportsbook"),
                                    line_value=line_value,
                                    price=row.get("under_price"),
                                    opposing_price=row.get("over_price"),
                                    model_probability=probabilities["under"]["win"],
                                    model_loss_probability=probabilities["under"]["loss"],
                                    push_probability=probabilities["under"]["push"],
                                    certainty_weight=certainty_weight,
                                    market_summary=market_summary,
                                    model_summary=model_summary,
                                ),
                            ],
                        )
                    )
            if all_scored_team_candidates:
                cards.extend([c for c in candidates if c])
            else:
                best_candidate = best_market_candidate(candidates)
                cards.append(
                    best_candidate
                    if best_candidate is not None
                    else fallback_market_card(
                        game=game,
                        market_key="game_total",
                        market_label="Game Total (Runs)",
                        market_summary="Game total (runs)",
                        model_summary="Full-game total probability unavailable.",
                    )
                )
        elif not all_scored_team_candidates:
            cards.append(
                fallback_market_card(
                    game=game,
                    market_key="game_total",
                    market_label="Game Total (Runs)",
                    market_summary="Game total (runs)",
                    model_summary="Projected full-game runs missing — run totals features + predict so EV can be scored.",
                )
            )

    for market_key, team_name, team_mean in (("away_team_total", away_team, away_expected_runs), ("home_team_total", home_team, home_expected_runs)):
        focused_rows, _ = market_focus_rows(market_rows_by_type.get(market_key) or [])
        if not focused_rows:
            continue
        candidates = []
        for row in focused_rows:
            line_value = to_float(row.get("line_value"))
            probabilities = team_total_side_probabilities(team_mean, line_value)
            market_summary = f"{team_name} TT {format(line_value, '.1f') if line_value is not None else '-'} · Over {format_price_text(row.get('over_price'))} / Under {format_price_text(row.get('under_price'))}"
            model_summary = (
                f"Mean {format(team_mean, '.2f') if team_mean is not None else '-'} runs · Over {format(probabilities['over']['win'] * 100, '.1f') if probabilities else '-'}% / "
                f"Under {format(probabilities['under']['win'] * 100, '.1f') if probabilities else '-'}%"
            )
            if probabilities is not None and line_value is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key=market_key, market_label=("Away Team Total" if market_key == "away_team_total" else "Home Team Total"), selection_label=f"{team_name} TT Over {line_value:.1f}", bet_side="over", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=probabilities["over"]["win"], model_loss_probability=probabilities["over"]["loss"], push_probability=probabilities["over"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key=market_key, market_label=("Away Team Total" if market_key == "away_team_total" else "Home Team Total"), selection_label=f"{team_name} TT Under {line_value:.1f}", bet_side="under", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=probabilities["under"]["win"], model_loss_probability=probabilities["under"]["loss"], push_probability=probabilities["under"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key=market_key, market_label="Away Team Total" if market_key == "away_team_total" else "Home Team Total", market_summary=f"{team_name} team total", model_summary="Team-total probability unavailable."))

    first5_moneyline_rows = market_rows_by_type.get("first_five_moneyline") or []
    if first5_moneyline_rows:
        candidates = []
        for row in first5_moneyline_rows:
            market_summary = f"F5 {away_team} {format_price_text(row.get('under_price'))} / {home_team} {format_price_text(row.get('over_price'))}"
            model_summary = f"F5 run context {away_team} {format(first5_away_runs, '.2f') if first5_away_runs is not None else '-'} / {home_team} {format(first5_home_runs, '.2f') if first5_home_runs is not None else '-'}"
            if first5_moneyline is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key="first_five_moneyline", market_label="First 5 Moneyline", selection_label=f"F5 {away_team} ML", bet_side="away", sportsbook=row.get("sportsbook"), line_value=None, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=first5_moneyline["away"]["win"], model_loss_probability=first5_moneyline["away"]["loss"], push_probability=first5_moneyline["away"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key="first_five_moneyline", market_label="First 5 Moneyline", selection_label=f"F5 {home_team} ML", bet_side="home", sportsbook=row.get("sportsbook"), line_value=None, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=first5_moneyline["home"]["win"], model_loss_probability=first5_moneyline["home"]["loss"], push_probability=first5_moneyline["home"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key="first_five_moneyline", market_label="First 5 Moneyline", market_summary=f"F5 {away_team} / {home_team}", model_summary="First-five moneyline probability unavailable."))

    first5_total_rows, _ = market_focus_rows(market_rows_by_type.get("first_five_total") or [])
    if first5_total_rows:
        candidates = []
        for row in first5_total_rows:
            line_value = to_float(row.get("line_value"))
            probabilities = team_total_side_probabilities(first5_total_runs, line_value)
            market_summary = f"F5 Total {format(line_value, '.1f') if line_value is not None else '-'} · Over {format_price_text(row.get('over_price'))} / Under {format_price_text(row.get('under_price'))}"
            model_summary = (
                f"Mean {format(first5_total_runs, '.2f') if first5_total_runs is not None else '-'} runs · Over {format(probabilities['over']['win'] * 100, '.1f') if probabilities else '-'}% / "
                f"Under {format(probabilities['under']['win'] * 100, '.1f') if probabilities else '-'}%"
            )
            if probabilities is not None and line_value is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key="first_five_total", market_label="First 5 Combined Total", selection_label=f"F5 Over {line_value:.1f}", bet_side="over", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=probabilities["over"]["win"], model_loss_probability=probabilities["over"]["loss"], push_probability=probabilities["over"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key="first_five_total", market_label="First 5 Combined Total", selection_label=f"F5 Under {line_value:.1f}", bet_side="under", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=probabilities["under"]["win"], model_loss_probability=probabilities["under"]["loss"], push_probability=probabilities["under"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key="first_five_total", market_label="First 5 Combined Total", market_summary="First 5 combined total", model_summary="First-five total probability unavailable."))

    first5_spread_rows, _ = market_focus_rows(market_rows_by_type.get("first_five_spread") or [])
    if first5_spread_rows:
        candidates = []
        for row in first5_spread_rows:
            home_line = to_float(row.get("line_value"))
            probabilities = run_line_side_probabilities(first5_away_runs, first5_home_runs, home_line)
            market_summary = f"F5 {home_team} {format_market_line_text(home_line)} {format_price_text(row.get('over_price'))} / {away_team} {format_market_line_text(None if home_line is None else -home_line)} {format_price_text(row.get('under_price'))}"
            model_summary = (
                f"F5 cover prob {home_team} {format(probabilities['home']['win'] * 100, '.1f') if probabilities else '-'}% / "
                f"{away_team} {format(probabilities['away']['win'] * 100, '.1f') if probabilities else '-'}%"
            )
            if probabilities is not None and home_line is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key="first_five_spread", market_label="First 5 Run Line", selection_label=f"F5 {home_team} {format_market_line_text(home_line)}", bet_side="home", sportsbook=row.get("sportsbook"), line_value=home_line, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=probabilities["home"]["win"], model_loss_probability=probabilities["home"]["loss"], push_probability=probabilities["home"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key="first_five_spread", market_label="First 5 Run Line", selection_label=f"F5 {away_team} {format_market_line_text(-home_line)}", bet_side="away", sportsbook=row.get("sportsbook"), line_value=-home_line, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=probabilities["away"]["win"], model_loss_probability=probabilities["away"]["loss"], push_probability=probabilities["away"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key="first_five_spread", market_label="First 5 Run Line", market_summary=f"F5 {away_team} / {home_team}", model_summary="First-five spread probability unavailable."))

    for market_key, team_name, team_mean in (("first_five_team_total_away", away_team, first5_away_runs), ("first_five_team_total_home", home_team, first5_home_runs)):
        focused_rows, _ = market_focus_rows(market_rows_by_type.get(market_key) or [])
        if not focused_rows:
            continue
        candidates = []
        for row in focused_rows:
            line_value = to_float(row.get("line_value"))
            probabilities = team_total_side_probabilities(team_mean, line_value)
            market_summary = f"F5 {team_name} TT {format(line_value, '.1f') if line_value is not None else '-'} · Over {format_price_text(row.get('over_price'))} / Under {format_price_text(row.get('under_price'))}"
            model_summary = (
                f"F5 mean {format(team_mean, '.2f') if team_mean is not None else '-'} runs · Over {format(probabilities['over']['win'] * 100, '.1f') if probabilities else '-'}% / "
                f"Under {format(probabilities['under']['win'] * 100, '.1f') if probabilities else '-'}%"
            )
            if probabilities is not None and line_value is not None:
                candidates.extend(
                    filter(
                        None,
                        [
                            build_market_candidate(game=game, market_key=market_key, market_label=("First 5 Away Team Total" if market_key == "first_five_team_total_away" else "First 5 Home Team Total"), selection_label=f"F5 {team_name} TT Over {line_value:.1f}", bet_side="over", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("over_price"), opposing_price=row.get("under_price"), model_probability=probabilities["over"]["win"], model_loss_probability=probabilities["over"]["loss"], push_probability=probabilities["over"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                            build_market_candidate(game=game, market_key=market_key, market_label=("First 5 Away Team Total" if market_key == "first_five_team_total_away" else "First 5 Home Team Total"), selection_label=f"F5 {team_name} TT Under {line_value:.1f}", bet_side="under", sportsbook=row.get("sportsbook"), line_value=line_value, price=row.get("under_price"), opposing_price=row.get("over_price"), model_probability=probabilities["under"]["win"], model_loss_probability=probabilities["under"]["loss"], push_probability=probabilities["under"]["push"], certainty_weight=certainty_weight, market_summary=market_summary, model_summary=model_summary),
                        ],
                    )
                )
        if all_scored_team_candidates:
            cards.extend([c for c in candidates if c])
        else:
            best_candidate = best_market_candidate(candidates)
            cards.append(best_candidate if best_candidate is not None else fallback_market_card(game=game, market_key=market_key, market_label="First 5 Away Team Total" if market_key == "first_five_team_total_away" else "First 5 Home Team Total", market_summary=f"F5 {team_name} team total", model_summary="First-five team-total probability unavailable."))

    if all_scored_team_candidates:
        scored = [
            annotate_market_card_for_display(card)
            for card in cards
            if card.get("market_key") in BEST_BET_MARKET_KEYS and to_float(card.get("weighted_ev")) is not None
        ]
        scored.sort(
            key=lambda card: (
                float(card.get("weighted_ev") or -999.0),
                float(card.get("probability_edge") or -999.0),
            ),
            reverse=True,
        )
        return scored, []

    cards = [card for card in cards if card.get("market_key") in BEST_BET_MARKET_KEYS]
    cards = [annotate_market_card_for_display(card) for card in cards]
    positive_cards = [card for card in cards if card.get("positive")]
    positive_cards.sort(key=lambda card: (float(card.get("weighted_ev") or -999.0), float(card.get("probability_edge") or -999.0)), reverse=True)
    return cards, positive_cards[:BEST_BET_SELECTION_LIMIT_PER_GAME]


def collect_all_team_market_ev_candidates(
    game: dict[str, Any],
    market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Every scored team-market side (ML/RL/totals/F5…) for Top EV selection — not one pick per market type."""
    scored, _ = build_market_cards_for_game(game, market_rows_by_type, all_scored_team_candidates=True)
    return scored


def build_pitcher_strikeouts_ev_cards_from_starter(
    game: dict[str, Any],
    starter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Over/under K props for one starter when ``k_projection`` has probabilities and book prices."""
    if not starter:
        return []
    kp = starter.get("k_projection")
    if not isinstance(kp, dict):
        return []
    market = kp.get("market") or {}
    line = to_float(market.get("consensus_line"))
    over_price = market.get("best_over_price")
    under_price = market.get("best_under_price")
    op = to_float(kp.get("over_probability"))
    un = to_float(kp.get("under_probability"))
    if line is None or op is None or un is None:
        return []
    pname = str(starter.get("pitcher_name") or "Pitcher")
    pid = starter.get("pitcher_id")
    team = str(starter.get("team") or "")
    cw = game_certainty_weight(game.get("certainty"))
    out: list[dict[str, Any]] = []
    for bet_side, mp, loss_p, price, opp, label in (
        ("over", op, un, over_price, under_price, f"{pname} Over {line:g} K"),
        ("under", un, op, under_price, over_price, f"{pname} Under {line:g} K"),
    ):
        if price is None:
            continue
        ms = (
            f"{pname} K line {line:g} · Over {format_price_text(over_price)} / "
            f"Under {format_price_text(under_price)}"
        )
        msum = f"Model P(over) {op * 100:.1f}% · P(under) {un * 100:.1f}%"
        card = build_market_candidate(
            game=game,
            market_key=PITCHER_STRIKEOUTS_MARKET_KEY,
            market_label="Pitcher strikeouts",
            selection_label=label,
            bet_side=bet_side,
            sportsbook=None,
            line_value=line,
            price=price,
            opposing_price=opp,
            model_probability=mp,
            model_loss_probability=loss_p,
            push_probability=0.0,
            certainty_weight=cw,
            market_summary=ms,
            model_summary=msum,
        )
        if not card:
            continue
        card["player_id"] = int(pid) if pid is not None else None
        card["subject_label"] = f"{pname} ({team})" if team else pname
        out.append(annotate_market_card_for_display(card))
    return out


def _green_strip_row_key(card: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(card.get("game_id") or 0),
        str(card.get("market_key") or ""),
        str(card.get("selection_label") or ""),
    )


def _select_green_strip_with_slate_spread(
    sorted_candidates: list[dict[str, Any]],
    *,
    limit: int,
    max_per_game_first_pass: int,
    seed: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Prefer one pick per matchup (by EV order) before allowing multiple markets from the same game."""
    if limit <= 0:
        return []
    if max_per_game_first_pass <= 0:
        base = list(seed) if seed else []
        return (base + sorted_candidates)[:limit]

    picked_keys: set[tuple[int, str, str]] = set()
    per_game_counts: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    if seed:
        for bet in seed:
            key = _green_strip_row_key(bet)
            picked_keys.add(key)
            gid = int(bet.get("game_id") or 0)
            per_game_counts[gid] = per_game_counts.get(gid, 0) + 1
        selected = list(seed)
        if len(selected) >= limit:
            return selected[:limit]

    for bet in sorted_candidates:
        key = _green_strip_row_key(bet)
        if key in picked_keys:
            continue
        gid = int(bet.get("game_id") or 0)
        if per_game_counts.get(gid, 0) >= max_per_game_first_pass:
            continue
        selected.append(bet)
        picked_keys.add(key)
        per_game_counts[gid] = per_game_counts.get(gid, 0) + 1
        if len(selected) >= limit:
            return selected

    for bet in sorted_candidates:
        if len(selected) >= limit:
            break
        key = _green_strip_row_key(bet)
        if key in picked_keys:
            continue
        gid = int(bet.get("game_id") or 0)
        # Second pass used to fill the strip without a per-game cap, so one matchup could
        # dominate (3-4 greens). Keep the same max-per-game rule as the first pass.
        if per_game_counts.get(gid, 0) >= max_per_game_first_pass:
            continue
        selected.append(bet)
        picked_keys.add(key)
        per_game_counts[gid] = per_game_counts.get(gid, 0) + 1
    return selected[:limit]


def board_pick_composite_score_v1(card: dict[str, Any]) -> float:
    """Blend EV, edge, model/certainty, game data quality, trust, optional learning overlay (action_score).

    Used when ``BOARD_GREEN_SELECTION_MODE=composite_v1``. Weights are explicit and tunable;
    typical scores fall in ~0.35–1.15 for board-eligible cards.
    """
    wev = to_float(card.get("weighted_ev")) or 0.0
    pe = to_float(card.get("probability_edge")) or 0.0
    cw = to_float(card.get("certainty_weight"))
    if cw is None:
        cw = 0.0
    gcp = to_float(card.get("game_certainty_pct"))
    mp = to_float(card.get("model_probability")) or 0.0
    it = card.get("input_trust") if isinstance(card.get("input_trust"), dict) else {}
    it_score = to_float(it.get("score"))
    grade = str(it.get("grade") or "").strip().upper()
    action = to_float(card.get("action_score"))

    wev_n = min(max(wev, 0.0) / 0.10, 1.0)
    pe_n = min(max(pe, 0.0) / 0.10, 1.0)
    gcp_n = min(max((gcp / 100.0) if gcp is not None else 0.55, 0.0), 1.0)
    if it_score is not None:
        trust_n = min(max(it_score, 0.0), 1.0)
    elif grade == "A":
        trust_n = 0.88
    elif grade == "B":
        trust_n = 0.74
    else:
        trust_n = 0.58
    grade_adj = 0.08 if grade == "A" else (0.04 if grade == "B" else 0.0)
    act_part = 0.08 * min(max(action, 0.0), 1.0) if action is not None else 0.0

    score = (
        0.26 * wev_n
        + 0.22 * pe_n
        + 0.18 * min(max(cw, 0.0), 1.0)
        + 0.16 * gcp_n
        + 0.10 * min(max(mp, 0.0), 1.0)
        + 0.06 * trust_n
        + grade_adj
        + act_part
    )
    return round(float(score), 4)


def _green_strip_sort_key(card: dict[str, Any], *, mode: str) -> tuple[float, ...]:
    wev = float(card.get("weighted_ev") or -999.0)
    pe = float(card.get("probability_edge") or -999.0)
    if mode == "composite_v1":
        comp = board_pick_composite_score_v1(card)
        return (comp, wev, pe)
    return (wev, pe)


def flatten_best_bets(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """Board green strip: full-gate (`positive`) picks first, then soft-qualified backfill.

    Strict picks are used first; only the remainder can come from softer edge bands
    (see ``BOARD_GREEN_SOFT_*``). Within each band, candidates are ranked by EV (``ev_gates``) or
    by ``board_pick_composite_score_v1`` (``composite_v1``), then we spread across games.

    Pass an explicit ``limit`` to cap the strip (tests use small limits).
    """
    if limit is None:
        limit = len(rows)
    settings = get_settings()
    mode = getattr(settings, "board_green_selection_mode", "ev_gates")
    if mode not in ("ev_gates", "composite_v1"):
        mode = "composite_v1"
    composite_min = getattr(settings, "board_green_composite_min", None)

    seen: set[tuple[int, str, str]] = set()
    flattened: list[dict[str, Any]] = []
    for row in rows:
        gid = int(row.get("game_id") or 0)

        def _add(card: dict[str, Any]) -> None:
            if excluded_from_team_best_pick_board(card.get("market_key")):
                return
            if not qualifies_board_green_strip(card):
                return
            key = (gid, str(card.get("market_key") or ""), str(card.get("selection_label") or ""))
            if key in seen:
                return
            seen.add(key)
            tier = "strict" if card.get("positive") else "soft"
            out = {**card, "green_strip_tier": tier}
            if mode == "composite_v1":
                sc = board_pick_composite_score_v1(out)
                out["green_composite_score"] = sc
                if composite_min is not None and sc < float(composite_min):
                    return
            flattened.append(out)

        for bet in row.get("best_bets") or []:
            _add(bet)
        for card in row.get("market_cards") or []:
            _add(card)

    sort_key = lambda bet: _green_strip_sort_key(bet, mode=mode)
    strict_flat = [b for b in flattened if b.get("positive")]
    soft_flat = [b for b in flattened if not b.get("positive")]
    strict_flat.sort(key=sort_key, reverse=True)
    soft_flat.sort(key=sort_key, reverse=True)

    # Prefer full-gate picks for every strip slot; only backfill with soft-qualified cards if needed.
    from_strict = _select_green_strip_with_slate_spread(
        strict_flat,
        limit=limit,
        max_per_game_first_pass=BOARD_GREEN_STRIP_MAX_PER_GAME_FIRST_PASS,
    )
    if len(from_strict) >= limit:
        return from_strict
    return _select_green_strip_with_slate_spread(
        soft_flat,
        limit=limit,
        max_per_game_first_pass=BOARD_GREEN_STRIP_MAX_PER_GAME_FIRST_PASS,
        seed=from_strict,
    )


def _is_watchlist_candidate(card: dict[str, Any]) -> bool:
    if not card or card.get("positive"):
        return False
    mk = str(card.get("market_key") or "")
    if excluded_from_team_best_pick_board(mk):
        return False
    if mk not in BEST_BET_MARKET_KEYS:
        return False
    model_probability = to_float(card.get("model_probability"))
    certainty_weight = to_float(card.get("certainty_weight"))
    weighted_ev = to_float(card.get("weighted_ev"))
    probability_edge = to_float(card.get("probability_edge"))
    if model_probability is None or certainty_weight is None:
        return False
    if certainty_weight < 0.52:
        return False
    if weighted_ev is not None and weighted_ev > 0:
        return True
    if probability_edge is not None and probability_edge >= 0.025 and model_probability >= 0.52:
        return True
    return False


def flatten_watchlist_markets(
    rows: list[dict[str, Any]],
    limit: int = BOARD_WATCHLIST_LIMIT,
    *,
    secondary_lines_only: bool = False,
) -> list[dict[str, Any]]:
    """Board strip: games with **no** green-strip pick (legacy behavior).

    When ``secondary_lines_only`` is True (Daily Results / outcome snapshots), include
    watchlist-tier team markets from **every** game **except** picks that already appear
    on the green strip — so slates that fill the green board still have secondary lines to track.
    """
    if secondary_lines_only:
        green_strip = flatten_best_bets(rows, limit=None)
        green_keys = {_green_strip_row_key(c) for c in green_strip}
        selected: list[dict[str, Any]] = []
        for row in rows:
            gid = int(row.get("game_id") or 0)
            for card in row.get("market_cards") or []:
                if excluded_from_team_best_pick_board(card.get("market_key")):
                    continue
                if not _is_watchlist_candidate(card):
                    continue
                keyed = dict(card)
                if keyed.get("game_id") is None:
                    keyed["game_id"] = gid
                if _green_strip_row_key(keyed) in green_keys:
                    continue
                selected.append(keyed)
        selected.sort(
            key=lambda card: (
                float(card.get("weighted_ev") or -999.0),
                float(card.get("probability_edge") or -999.0),
                float(card.get("raw_ev") or -999.0),
                float(card.get("model_probability") or -999.0),
            ),
            reverse=True,
        )
        return selected[:limit]

    selected = []
    green_game_ids = {
        int(card.get("game_id"))
        for card in flatten_best_bets(rows, limit=None)
        if card.get("game_id") is not None
    }
    for row in rows:
        game_id = row.get("game_id")
        if game_id is not None and int(game_id) in green_game_ids:
            continue
        # Per-game `best_bets` are strict positives (≤1/game). If present, they are the game's
        # primary pick even when they miss the global green strip top-N — still show on watchlist.
        raw_best = [
            card
            for card in (row.get("best_bets") or [])
            if not excluded_from_team_best_pick_board(card.get("market_key"))
        ]
        if raw_best:
            market_cards = raw_best
        else:
            market_cards = [
                card
                for card in (row.get("market_cards") or [])
                if _is_watchlist_candidate(card)
                and not excluded_from_team_best_pick_board(card.get("market_key"))
            ]
        if not market_cards:
            continue
        market_cards.sort(
            key=lambda card: (
                float(card.get("weighted_ev") or -999.0),
                float(card.get("probability_edge") or -999.0),
                float(card.get("raw_ev") or -999.0),
                float(card.get("model_probability") or -999.0),
            ),
            reverse=True,
        )
        selected.extend(
            card
            for card in market_cards
            if not excluded_from_team_best_pick_board(card.get("market_key"))
        )
    selected.sort(
        key=lambda card: (
            float(card.get("weighted_ev") or -999.0),
            float(card.get("probability_edge") or -999.0),
            float(card.get("raw_ev") or -999.0),
            float(card.get("model_probability") or -999.0),
        ),
        reverse=True,
    )
    return selected[:limit]


def recommendation_card_identity(card: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(card.get("game_id") or 0),
        str(card.get("market_key") or ""),
        str(card.get("bet_side") or ""),
        str(card.get("selection_label") or ""),
    )


def snapshot_recommendation_tiers(
    rows: list[dict[str, Any]],
    *,
    green_limit: int | None = None,
    watchlist_limit: int = BOARD_WATCHLIST_LIMIT,
    watchlist_secondary_lines: bool = False,
) -> dict[str, Any]:
    green_cards = flatten_best_bets(rows, limit=green_limit)
    watchlist_cards = flatten_watchlist_markets(
        rows,
        limit=watchlist_limit,
        secondary_lines_only=watchlist_secondary_lines,
    )
    return {
        "green_cards": green_cards,
        "watchlist_cards": watchlist_cards,
        "green_lookup": {
            recommendation_card_identity(card): index + 1
            for index, card in enumerate(green_cards)
        },
        "watchlist_lookup": {
            recommendation_card_identity(card): index + 1
            for index, card in enumerate(watchlist_cards)
        },
    }


def dedupe_experimental_first_inning_by_game(
    rows: list[dict[str, Any]],
    *,
    market_field: str = "market_type",
    game_id_field: str = "game_id",
) -> list[dict[str, Any]]:
    """Keep at most one row per (game_id, nrfi|yrfi). Both sides may appear for the same game."""
    order = {m: i for i, m in enumerate(EXPERIMENTAL_FIRST_INNING_MARKETS_ORDER)}
    best_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        gid_raw = row.get(game_id_field)
        if gid_raw is None:
            continue
        gid = int(gid_raw)
        mkt = str(row.get(market_field) or "").lower()
        if mkt not in order:
            continue
        key = (gid, mkt)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = row
            continue
        ts_new = str(row.get("snapshot_ts") or "")
        ts_old = str(existing.get("snapshot_ts") or "")
        if ts_new > ts_old:
            best_by_key[key] = row

    def _sort_key(r: dict[str, Any]) -> tuple[str, str, int]:
        ts = r.get("game_start_ts") or r.get("snapshot_ts") or r.get("game_date") or ""
        mkt = str(r.get(market_field) or "").lower()
        return (str(ts), mkt, int(r.get(game_id_field) or 0))

    return sorted(best_by_key.values(), key=_sort_key)


def selected_team_for_card(card: dict[str, Any]) -> str | None:
    market_key = str(card.get("market_key") or "")
    bet_side = str(card.get("bet_side") or "")
    if market_key == "away_team_total":
        return str(card.get("away_team") or "") or None
    if market_key == "home_team_total":
        return str(card.get("home_team") or "") or None
    if market_key == "first_five_team_total_away":
        return str(card.get("away_team") or "") or None
    if market_key == "first_five_team_total_home":
        return str(card.get("home_team") or "") or None
    if bet_side == "away":
        return str(card.get("away_team") or "") or None
    if bet_side == "home":
        return str(card.get("home_team") or "") or None
    return None


def opposite_team_for_card(card: dict[str, Any]) -> str | None:
    selected_team = selected_team_for_card(card)
    away_team = str(card.get("away_team") or "") or None
    home_team = str(card.get("home_team") or "") or None
    if selected_team is None:
        return None
    if away_team == selected_team:
        return home_team
    if home_team == selected_team:
        return away_team
    return None


def grade_best_bet_pick(
    card: dict[str, Any],
    *,
    actual_result: dict[str, Any] | None = None,
    first5_result: dict[str, Any] | None = None,
    player_prop_actuals: dict[str, Any] | None = None,
    pitcher_strikeouts_actual: float | None = None,
) -> dict[str, Any]:
    market_key = str(card.get("market_key") or "")
    bet_side = str(card.get("bet_side") or "")
    line_value = to_float(card.get("line_value"))
    actual_result = actual_result or {}
    first5_result = first5_result or {}
    player_prop_actuals = player_prop_actuals or {}

    actual_side: str | None = None
    actual_measure: float | None = None
    graded = False

    if market_key == "moneyline":
        away_runs = to_float(actual_result.get("away_runs"))
        home_runs = to_float(actual_result.get("home_runs"))
        if away_runs is not None and home_runs is not None and bool(actual_result.get("is_final")):
            graded = True
            if away_runs > home_runs:
                actual_side = "away"
            elif home_runs > away_runs:
                actual_side = "home"
            else:
                actual_side = "push"
    elif market_key == "run_line":
        away_runs = to_float(actual_result.get("away_runs"))
        home_runs = to_float(actual_result.get("home_runs"))
        if away_runs is not None and home_runs is not None and line_value is not None and bool(actual_result.get("is_final")):
            graded = True
            if bet_side == "home":
                adjusted = home_runs + line_value
                actual_side = "home" if adjusted > away_runs else "away" if adjusted < away_runs else "push"
            elif bet_side == "away":
                adjusted = away_runs + line_value
                actual_side = "away" if adjusted > home_runs else "home" if adjusted < home_runs else "push"
    elif market_key in {"away_team_total", "home_team_total"}:
        team_runs = to_float(actual_result.get("away_runs" if market_key == "away_team_total" else "home_runs"))
        if team_runs is not None and line_value is not None and bool(actual_result.get("is_final")):
            graded = True
            actual_measure = team_runs
            if team_runs > line_value:
                actual_side = "over"
            elif team_runs < line_value:
                actual_side = "under"
            else:
                actual_side = "push"
    elif market_key == "game_total":
        total_runs = to_float(actual_result.get("total_runs"))
        if total_runs is None and bool(actual_result.get("is_final")):
            away_runs = to_float(actual_result.get("away_runs"))
            home_runs = to_float(actual_result.get("home_runs"))
            if away_runs is not None and home_runs is not None:
                total_runs = away_runs + home_runs
        if total_runs is not None and line_value is not None and bool(actual_result.get("is_final")):
            graded = True
            actual_measure = total_runs
            if total_runs > line_value:
                actual_side = "over"
            elif total_runs < line_value:
                actual_side = "under"
            else:
                actual_side = "push"
    elif market_key == "first_five_moneyline":
        away_runs = to_float(first5_result.get("away_runs"))
        home_runs = to_float(first5_result.get("home_runs"))
        if away_runs is not None and home_runs is not None:
            graded = True
            if away_runs > home_runs:
                actual_side = "away"
            elif home_runs > away_runs:
                actual_side = "home"
            else:
                actual_side = "push"
    elif market_key == "first_five_total":
        total_runs = to_float(first5_result.get("total_runs"))
        if total_runs is not None and line_value is not None:
            graded = True
            actual_measure = total_runs
            if total_runs > line_value:
                actual_side = "over"
            elif total_runs < line_value:
                actual_side = "under"
            else:
                actual_side = "push"
    elif market_key == "first_five_spread":
        away_runs = to_float(first5_result.get("away_runs"))
        home_runs = to_float(first5_result.get("home_runs"))
        if away_runs is not None and home_runs is not None and line_value is not None:
            graded = True
            if bet_side == "home":
                adjusted = home_runs + line_value
                actual_side = "home" if adjusted > away_runs else "away" if adjusted < away_runs else "push"
            elif bet_side == "away":
                adjusted = away_runs + line_value
                actual_side = "away" if adjusted > home_runs else "home" if adjusted < home_runs else "push"
    elif market_key in {"first_five_team_total_away", "first_five_team_total_home"}:
        team_runs = to_float(first5_result.get("away_runs" if market_key == "first_five_team_total_away" else "home_runs"))
        if team_runs is not None and line_value is not None:
            graded = True
            actual_measure = team_runs
            if team_runs > line_value:
                actual_side = "over"
            elif team_runs < line_value:
                actual_side = "under"
            else:
                actual_side = "push"
    elif market_key == PLAYER_HITS_MARKET_KEY:
        hit_ct = to_float(player_prop_actuals.get("hits"))
        if hit_ct is not None and bool(actual_result.get("is_final")):
            graded = True
            actual_side = "yes" if hit_ct >= 1.0 else "no"
    elif market_key == PLAYER_HOME_RUN_MARKET_KEY:
        hr_ct = to_float(player_prop_actuals.get("home_runs"))
        if hr_ct is not None and bool(actual_result.get("is_final")):
            graded = True
            actual_side = "yes" if hr_ct >= 1.0 else "no"
    elif market_key == PITCHER_STRIKEOUTS_MARKET_KEY:
        ks = to_float(pitcher_strikeouts_actual)
        if ks is not None and line_value is not None and bool(actual_result.get("is_final")):
            graded = True
            actual_measure = ks
            if ks > line_value:
                actual_side = "over"
            elif ks < line_value:
                actual_side = "under"
            else:
                actual_side = "push"

    if not graded or actual_side is None:
        return {"graded": graded, "actual_side": actual_side, "success": None, "result": "pending", "actual_value": None, "actual_measure": actual_measure}
    if actual_side == "push":
        return {"graded": True, "actual_side": "push", "success": None, "result": "push", "actual_value": None, "actual_measure": actual_measure}
    success = actual_side == bet_side
    return {
        "graded": True,
        "actual_side": actual_side,
        "success": success,
        "result": "won" if success else "lost",
        "actual_value": 1.0 if success else 0.0,
        "actual_measure": actual_measure,
    }