from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.common import (
    baseball_ip_to_outs,
    build_pitcher_priors,
    build_team_priors,
    coerce_utc_timestamp_series,
    compute_board_state,
    compute_freshness_score,
    compute_starter_certainty,
    count_missing_fallbacks,
    default_cutoff,
    infer_lineup_from_history,
    offense_snapshot,
    outs_to_baseball_ip,
    persist_strikeout_features,
    pitcher_snapshot,
    write_feature_snapshot,
)
from src.features.contracts import (
    STRIKEOUTS_CERTAINTY_KEY_FIELDS,
    STRIKEOUTS_FEATURE_COLUMNS,
    STRIKEOUTS_META_COLUMNS,
    STRIKEOUTS_TARGET_COLUMN,
    validate_columns,
)
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _load_frames(start_date, end_date, settings):
    history_start = pd.Timestamp(start_date) - pd.Timedelta(days=45)
    prior_start = f"{settings.prior_season}-01-01"
    return {
        "games": query_df(
            """
            SELECT game_id, game_date, game_start_ts, season, home_team, away_team
            FROM games
            WHERE game_date BETWEEN :start_date AND :end_date
            ORDER BY game_date, game_id
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "games_history": query_df(
            """
            SELECT game_id, game_date, home_team, away_team, venue_id
            FROM games
            WHERE game_date >= :prior_start AND game_date <= :end_date
            """,
            {"prior_start": prior_start, "end_date": end_date},
        ),
        "pitcher_starts": query_df(
            """
            SELECT *
            FROM pitcher_starts
            WHERE game_date >= :prior_start AND game_date <= :end_date
            """,
            {"prior_start": prior_start, "end_date": end_date},
        ),
        "team_offense": query_df(
            """
            SELECT *
            FROM team_offense_daily
            WHERE game_date >= :prior_start AND game_date <= :end_date
            """,
            {"prior_start": prior_start, "end_date": end_date},
        ),
        "lineups": query_df(
            """
            SELECT *
            FROM lineups
            WHERE game_date BETWEEN :history_start AND :end_date
            """,
            {"history_start": history_start.date(), "end_date": end_date},
        ),
        "players": query_df(
            """
            SELECT player_id, full_name, team_abbr, bats, throws, position
            FROM dim_players
            """
        ),
        "player_batting": query_df(
            """
            SELECT game_id, game_date, player_id, team, opponent, lineup_slot, hits, at_bats, plate_appearances, strikeouts
            FROM player_game_batting
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start.date(), "end_date": end_date},
        ),
        "prop_markets": query_df(
            """
            SELECT *
            FROM player_prop_markets
            WHERE game_date BETWEEN :start_date AND :end_date
              AND market_type = 'pitcher_strikeouts'
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
    }


def _latest_lineup_for_team(
    game_id: int,
    team: str,
    game_date,
    cutoff_ts,
    lineups: pd.DataFrame,
    player_batting: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    team_lineups = lineups[
        (lineups["game_id"] == game_id)
        & (lineups["team"] == team)
        & (lineups["snapshot_ts"] <= cutoff_ts)
    ].copy()
    if not team_lineups.empty:
        latest_snapshot = team_lineups["snapshot_ts"].max()
        latest = team_lineups[team_lineups["snapshot_ts"] == latest_snapshot].copy()
        latest = latest.merge(
            players[["player_id", "bats", "full_name"]].drop_duplicates(subset=["player_id"]),
            on="player_id",
            how="left",
        )
        latest["player_name"] = latest["player_name"].fillna(latest["full_name"])
        latest = latest.drop(columns=["full_name"])
        latest["is_confirmed"] = latest["is_confirmed"].fillna(False)
        return latest[["player_id", "player_name", "lineup_slot", "is_confirmed", "bats"]]

    inferred = infer_lineup_from_history(team, game_date, player_batting, players)
    if inferred.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "lineup_slot", "is_confirmed", "bats"])
    inferred = inferred.merge(
        players[["player_id", "bats"]].drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="left",
    )
    return inferred[["player_id", "player_name", "lineup_slot", "is_confirmed", "bats"]]


def _summarize_lineup_handedness(lineup: pd.DataFrame) -> dict[str, float | int | None]:
    if lineup.empty:
        return {
            "same_hand_share": None,
            "opposite_hand_share": None,
            "switch_share": None,
            "known_hitters": 0,
            "confirmed_hitters": 0,
            "counts": {"R": 0, "L": 0, "S": 0},
        }
    bats = lineup["bats"].astype(str).str.upper()
    counts = {
        "R": int((bats == "R").sum()),
        "L": int((bats == "L").sum()),
        "S": int((bats == "S").sum()),
    }
    known_hitters = counts["R"] + counts["L"] + counts["S"]
    return {
        "known_hitters": known_hitters,
        "confirmed_hitters": int(lineup["is_confirmed"].fillna(False).sum()),
        "counts": counts,
    }


def _same_hand_shares(throw_hand: str | None, handedness: dict[str, float | int | None]) -> dict[str, float | int | None]:
    counts = handedness.get("counts") or {"R": 0, "L": 0, "S": 0}
    known_hitters = int(handedness.get("known_hitters") or 0)
    if throw_hand not in {"R", "L"} or known_hitters <= 0:
        return {
            "same_hand_share": None,
            "opposite_hand_share": None,
            "switch_share": None,
        }
    same_count = counts[throw_hand]
    opposite_count = counts["L" if throw_hand == "R" else "R"]
    switch_count = counts["S"]
    return {
        "same_hand_share": same_count / known_hitters,
        "opposite_hand_share": opposite_count / known_hitters,
        "switch_share": switch_count / known_hitters,
    }


def _recent_pitcher_form(history: pd.DataFrame) -> dict[str, float | None]:
    recent_3 = history.tail(3)
    recent_5 = history.tail(5)
    outs_3 = recent_3["ip"].apply(pd.to_numeric, errors="coerce")
    outs_5 = recent_5["ip"].apply(pd.to_numeric, errors="coerce")
    batters_3 = _usage_opportunity_series(recent_3)
    batters_5 = _usage_opportunity_series(recent_5)
    return {
        "recent_avg_ip_3": float(outs_3.mean()) if not outs_3.dropna().empty else None,
        "recent_avg_ip_5": float(outs_5.mean()) if not outs_5.dropna().empty else None,
        "recent_avg_strikeouts_3": float(pd.to_numeric(recent_3["strikeouts"], errors="coerce").mean()) if not recent_3.empty else None,
        "recent_avg_strikeouts_5": float(pd.to_numeric(recent_5["strikeouts"], errors="coerce").mean()) if not recent_5.empty else None,
        "recent_k_per_batter_3": _safe_rate_sum(recent_3, "strikeouts", "batters_faced"),
        "recent_k_per_batter_5": _safe_rate_sum(recent_5, "strikeouts", "batters_faced"),
        "recent_avg_pitch_count_3": float(batters_3.mean()) if not batters_3.dropna().empty else None,
        "recent_whiff_pct_5": float(pd.to_numeric(recent_5["whiff_pct"], errors="coerce").mean()) if not recent_5.empty else None,
        "recent_csw_pct_5": float(pd.to_numeric(recent_5["csw_pct"], errors="coerce").mean()) if not recent_5.empty else None,
        "recent_xwoba_5": float(pd.to_numeric(recent_5["xwoba_against"], errors="coerce").mean()) if not recent_5.empty else None,
    }


def _season_pitcher_context(history: pd.DataFrame, season: int) -> dict[str, float | int | None]:
    season_history = history[history["game_date"].dt.year == int(season)].copy()
    if season_history.empty:
        return {
            "season_starts": 0,
            "season_innings": None,
            "season_strikeouts": 0,
            "season_k_per_start": None,
            "season_k_per_batter": None,
        }

    strikeout_series = pd.to_numeric(season_history["strikeouts"], errors="coerce").fillna(0)
    strikeout_total = int(strikeout_series.sum())
    starts = int(len(season_history.index))
    outs = int(pd.to_numeric(season_history["ip"], errors="coerce").apply(baseball_ip_to_outs).sum())
    batters_faced = pd.to_numeric(season_history.get("batters_faced"), errors="coerce")
    batters_faced_total = float(batters_faced.fillna(0).sum()) if batters_faced is not None else 0.0

    return {
        "season_starts": starts,
        "season_innings": outs_to_baseball_ip(outs) if outs > 0 else None,
        "season_strikeouts": strikeout_total,
        "season_k_per_start": None if starts <= 0 else float(strikeout_total) / float(starts),
        "season_k_per_batter": None if batters_faced_total <= 0 else float(strikeout_total) / batters_faced_total,
    }


def _usage_opportunity_series(frame: pd.DataFrame) -> pd.Series:
    batters_faced = pd.to_numeric(frame.get("batters_faced"), errors="coerce")
    if batters_faced.dropna().empty:
        return pd.to_numeric(frame["pitch_count"], errors="coerce")
    return batters_faced


def _safe_rate_sum(frame: pd.DataFrame, numerator: str, denominator: str) -> float | None:
    if frame.empty:
        return None
    numerator_sum = pd.to_numeric(frame[numerator], errors="coerce").fillna(0).sum()
    if denominator == "batters_faced":
        denominator_series = _usage_opportunity_series(frame)
    else:
        denominator_series = pd.to_numeric(frame[denominator], errors="coerce")
    denominator_sum = denominator_series.fillna(0).sum()
    if denominator_sum <= 0:
        return None
    return float(numerator_sum) / float(denominator_sum)


def _latest_prop_market_snapshot(game_id: int, pitcher_id: int, cutoff_ts, prop_markets: pd.DataFrame) -> dict[str, object]:
    frame = prop_markets[
        (prop_markets["game_id"] == game_id)
        & (prop_markets["player_id"] == pitcher_id)
        & (prop_markets["snapshot_ts"] <= cutoff_ts)
    ].copy()
    if frame.empty:
        return {"market_line": None, "line_snapshot_ts": None}
    latest = frame.sort_values("snapshot_ts").groupby("sportsbook", as_index=False).tail(1)
    line_values = pd.to_numeric(latest["line_value"], errors="coerce").dropna()
    return {
        "market_line": round(float(line_values.median()), 2) if not line_values.empty else None,
        "line_snapshot_ts": latest["snapshot_ts"].max(),
    }


# ---------------------------------------------------------------------------
# New differentiation features (experiment branch)
# ---------------------------------------------------------------------------

def _pitcher_vs_team_k_rate(
    pitcher_id: int,
    opponent: str,
    game_date,
    pitcher_starts: pd.DataFrame,
    games_history: pd.DataFrame,
) -> float | None:
    """Historical K/start for this pitcher against this specific opponent."""
    history = pitcher_starts[
        (pitcher_starts["pitcher_id"] == pitcher_id)
        & (pitcher_starts["game_date"].dt.date < game_date)
    ].copy()
    if history.empty:
        return None
    # Join with games_history to determine the opponent for each start
    merged = history.merge(
        games_history[["game_id", "home_team", "away_team"]],
        on="game_id",
        how="left",
    )
    # Determine opponent: if pitcher's team is home_team → opponent is away_team, else home_team
    merged["opponent_team"] = np.where(
        merged["team"] == merged["home_team"],
        merged["away_team"],
        merged["home_team"],
    )
    vs_team = merged[merged["opponent_team"] == opponent]
    if vs_team.empty:
        return None
    k_vals = pd.to_numeric(vs_team["strikeouts"], errors="coerce").dropna()
    if k_vals.empty:
        return None
    return float(k_vals.mean())


def _opponent_lineup_k_pct_recent(
    opponent_lineup: pd.DataFrame,
    player_batting: pd.DataFrame,
    game_date,
    lookback_games: int = 7,
) -> float | None:
    """Average K% of opponent's lineup batters over their recent games."""
    if opponent_lineup.empty:
        return None
    batter_ids = opponent_lineup["player_id"].dropna().unique()
    if len(batter_ids) == 0:
        return None
    batting = player_batting[
        (player_batting["player_id"].isin(batter_ids))
        & (player_batting["game_date"].dt.date < game_date)
    ].copy()
    if batting.empty:
        return None
    # Take each batter's most recent N games
    batting = batting.sort_values("game_date")
    recent = batting.groupby("player_id").tail(lookback_games)
    total_k = pd.to_numeric(recent["strikeouts"], errors="coerce").fillna(0).sum()
    total_pa = pd.to_numeric(recent["plate_appearances"], errors="coerce").fillna(0).sum()
    if total_pa <= 0:
        return None
    return float(total_k) / float(total_pa)


def _venue_k_factor(
    venue_id: int | None,
    game_date,
    pitcher_starts: pd.DataFrame,
    games_history: pd.DataFrame,
    min_games: int = 20,
) -> float | None:
    """K-per-game at this venue vs league average, as a ratio (>1 = K-friendly)."""
    if venue_id is None:
        return None
    venue_games = games_history[
        (games_history["venue_id"] == venue_id)
        & (games_history["game_date"].dt.date < game_date)
    ]
    if len(venue_games) < min_games:
        return None
    venue_game_ids = set(venue_games["game_id"])
    venue_starts = pitcher_starts[
        (pitcher_starts["game_id"].isin(venue_game_ids))
        & (pitcher_starts["game_date"].dt.date < game_date)
    ]
    if venue_starts.empty:
        return None
    venue_k = pd.to_numeric(venue_starts["strikeouts"], errors="coerce").dropna()
    if venue_k.empty:
        return None
    venue_k_per_start = float(venue_k.mean())

    # League average K/start for the same time window
    all_starts = pitcher_starts[pitcher_starts["game_date"].dt.date < game_date]
    all_k = pd.to_numeric(all_starts["strikeouts"], errors="coerce").dropna()
    if all_k.empty:
        return None
    league_k_per_start = float(all_k.mean())
    if league_k_per_start <= 0:
        return None
    return round(venue_k_per_start / league_k_per_start, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical pitcher strikeout feature rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)
    settings = get_settings()
    frames = _load_frames(start_date, end_date, settings)

    games = frames["games"]
    if games.empty:
        log.info("Games missing for strikeout feature build")
        return 0

    for frame_name in ("pitcher_starts", "team_offense", "lineups", "player_batting", "prop_markets"):
        if not frames[frame_name].empty and "game_date" in frames[frame_name].columns:
            frames[frame_name]["game_date"] = pd.to_datetime(frames[frame_name]["game_date"])
    if not frames["games_history"].empty and "game_date" in frames["games_history"].columns:
        frames["games_history"]["game_date"] = pd.to_datetime(frames["games_history"]["game_date"])
    for frame_name in ("lineups", "prop_markets"):
        if not frames[frame_name].empty and "snapshot_ts" in frames[frame_name].columns:
            frames[frame_name]["snapshot_ts"] = coerce_utc_timestamp_series(frames[frame_name]["snapshot_ts"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["game_start_ts"] = pd.to_datetime(games["game_start_ts"], utc=True)

    # Build venue map for current games
    venue_map = {}
    if not frames["games_history"].empty:
        for g_row in frames["games_history"].itertuples(index=False):
            venue_map[int(g_row.game_id)] = getattr(g_row, "venue_id", None)
    player_meta_map = (
        frames["players"][["player_id", "full_name", "throws"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")
        .to_dict("index")
    )

    team_priors = build_team_priors(frames["team_offense"], settings.prior_season)
    pitcher_priors = build_pitcher_priors(frames["pitcher_starts"], settings.prior_season)
    prediction_ts = datetime.now(timezone.utc)
    feature_version = f"{settings.model_version_prefix}_strikeouts_core"

    rows = []
    for game in games.itertuples(index=False):
        cutoff_ts = default_cutoff(game.game_date, game.game_start_ts)
        starters = frames["pitcher_starts"][frames["pitcher_starts"]["game_id"] == game.game_id].copy()
        if starters.empty:
            continue

        home_lineup = _latest_lineup_for_team(
            int(game.game_id),
            game.home_team,
            game.game_date,
            cutoff_ts,
            frames["lineups"],
            frames["player_batting"],
            frames["players"],
        )
        away_lineup = _latest_lineup_for_team(
            int(game.game_id),
            game.away_team,
            game.game_date,
            cutoff_ts,
            frames["lineups"],
            frames["player_batting"],
            frames["players"],
        )
        home_handedness = _summarize_lineup_handedness(home_lineup)
        away_handedness = _summarize_lineup_handedness(away_lineup)

        home_offense = offense_snapshot(
            game.home_team,
            game.game_date,
            frames["team_offense"],
            team_priors,
            full_weight_games=settings.team_full_weight_games,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        away_offense = offense_snapshot(
            game.away_team,
            game.game_date,
            frames["team_offense"],
            team_priors,
            full_weight_games=settings.team_full_weight_games,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )

        for starter in starters.sort_values(["side", "pitcher_id"]).itertuples(index=False):
            team = starter.team
            opponent = game.away_team if team == game.home_team else game.home_team
            opponent_lineup = away_lineup if team == game.home_team else home_lineup
            opponent_handedness = away_handedness if team == game.home_team else home_handedness
            opponent_offense = away_offense if team == game.home_team else home_offense
            pitcher_meta = player_meta_map.get(int(starter.pitcher_id), {})
            pitcher_name = pitcher_meta.get("full_name") or getattr(starter, "pitcher_name", None) or str(starter.pitcher_id)
            throw_hand = str(pitcher_meta.get("throws") or getattr(starter, "throws", "") or "").strip().upper()[:1]

            history = frames["pitcher_starts"][
                (frames["pitcher_starts"]["pitcher_id"] == starter.pitcher_id)
                & (frames["pitcher_starts"]["game_date"].dt.date < game.game_date)
            ].sort_values("game_date")
            recent_form = _recent_pitcher_form(history)
            season_context = _season_pitcher_context(history, int(game.game_date.year))
            pitcher_snapshot(
                int(starter.pitcher_id),
                game.game_date,
                frames["pitcher_starts"],
                pitcher_priors,
                full_weight_starts=settings.pitcher_full_weight_starts,
                prior_blend_mode=settings.prior_blend_mode,
                prior_weight_multiplier=settings.prior_weight_multiplier,
            )
            shares = _same_hand_shares(throw_hand, opponent_handedness)
            market = _latest_prop_market_snapshot(int(game.game_id), int(starter.pitcher_id), cutoff_ts, frames["prop_markets"])
            baseline_strikeouts = recent_form["recent_avg_strikeouts_5"] or recent_form["recent_avg_strikeouts_3"] or season_context["season_k_per_start"]
            projected_innings = recent_form["recent_avg_ip_5"] or recent_form["recent_avg_ip_3"] or season_context["season_innings"]
            projected_innings = projected_innings if projected_innings and not pd.isna(projected_innings) else None
            handedness_adjustment_applied = shares["same_hand_share"] is not None and shares["opposite_hand_share"] is not None
            handedness_data_missing = int(opponent_handedness["known_hitters"] or 0) == 0
            game_start = game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None
            starter_cert = compute_starter_certainty(int(starter.pitcher_id), bool(getattr(starter, "is_probable", False)))
            market_freshness = compute_freshness_score(
                market["line_snapshot_ts"], game_start, decay_hours=(1, 6, 12, 24),
            )

            # --- New differentiation features ---
            pvt_k_rate = _pitcher_vs_team_k_rate(
                int(starter.pitcher_id), opponent, game.game_date,
                frames["pitcher_starts"], frames["games_history"],
            )
            opp_lineup_k_recent = _opponent_lineup_k_pct_recent(
                opponent_lineup, frames["player_batting"], game.game_date,
            )
            v_k_factor = _venue_k_factor(
                venue_map.get(int(game.game_id)), game.game_date,
                frames["pitcher_starts"], frames["games_history"],
            )

            rows.append(
                {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date,
                    "pitcher_id": int(starter.pitcher_id),
                    "team": team,
                    "opponent": opponent,
                    "prediction_ts": prediction_ts,
                    "game_start_ts": game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None,
                    "line_snapshot_ts": market["line_snapshot_ts"],
                    "feature_cutoff_ts": cutoff_ts,
                    "feature_version": feature_version,
                    "pitcher_name": pitcher_name,
                    "throws": throw_hand or None,
                    "days_rest": starter.days_rest,
                    "projected_innings": projected_innings,
                    "season_starts": season_context["season_starts"],
                    "season_innings": season_context["season_innings"],
                    "season_strikeouts": season_context["season_strikeouts"],
                    "season_k_per_start": season_context["season_k_per_start"],
                    "season_k_per_batter": season_context["season_k_per_batter"],
                    "recent_avg_ip_3": recent_form["recent_avg_ip_3"],
                    "recent_avg_ip_5": recent_form["recent_avg_ip_5"],
                    "recent_avg_strikeouts_3": recent_form["recent_avg_strikeouts_3"],
                    "recent_avg_strikeouts_5": recent_form["recent_avg_strikeouts_5"],
                    "recent_k_per_batter_3": recent_form["recent_k_per_batter_3"],
                    "recent_k_per_batter_5": recent_form["recent_k_per_batter_5"],
                    "recent_avg_pitch_count_3": recent_form["recent_avg_pitch_count_3"],
                    "recent_whiff_pct_5": recent_form["recent_whiff_pct_5"],
                    "recent_csw_pct_5": recent_form["recent_csw_pct_5"],
                    "recent_xwoba_5": recent_form["recent_xwoba_5"],
                    "baseline_strikeouts": baseline_strikeouts,
                    "market_line": market["market_line"],
                    "opponent_lineup_k_pct": None,
                    "opponent_k_pct_blended": opponent_offense["k_pct"],
                    "pitcher_vs_team_k_rate": pvt_k_rate,
                    "opponent_lineup_k_pct_recent": opp_lineup_k_recent,
                    "venue_k_factor": v_k_factor,
                    "same_hand_share": shares["same_hand_share"],
                    "opposite_hand_share": shares["opposite_hand_share"],
                    "switch_share": shares["switch_share"],
                    "lineup_right_count": opponent_handedness["counts"]["R"],
                    "lineup_left_count": opponent_handedness["counts"]["L"],
                    "lineup_switch_count": opponent_handedness["counts"]["S"],
                    "known_hitters": opponent_handedness["known_hitters"],
                    "confirmed_hitters": opponent_handedness["confirmed_hitters"],
                    "total_hitters": len(opponent_lineup.index),
                    "handedness_adjustment_applied": handedness_adjustment_applied,
                    "handedness_data_missing": handedness_data_missing,
                    "actual_strikeouts": starter.strikeouts,
                }
            )
            row = rows[-1]
            row["starter_certainty_score"] = starter_cert
            row["market_freshness_score"] = market_freshness
            missing = count_missing_fallbacks(row, STRIKEOUTS_CERTAINTY_KEY_FIELDS)
            row["missing_fallback_count"] = missing
            row["board_state"] = compute_board_state(missing, threshold_minimal=2)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        log.info("No strikeout feature rows were built")
        return 0

    player_batting = frames["player_batting"].copy()
    player_batting["game_date"] = pd.to_datetime(player_batting["game_date"])
    opponent_k = (
        player_batting.assign(strikeouts=0)
        if "strikeouts" not in player_batting.columns
        else player_batting
    )
    if "strikeouts" in opponent_k.columns:
        opponent_k["strikeouts"] = pd.to_numeric(opponent_k["strikeouts"], errors="coerce").fillna(0)
        opponent_k["plate_appearances"] = pd.to_numeric(opponent_k["plate_appearances"], errors="coerce").fillna(0)
        matchup_rates = (
            opponent_k.groupby(["game_id", "team"], as_index=False)
            .agg(strikeouts=("strikeouts", "sum"), plate_appearances=("plate_appearances", "sum"))
        )
        matchup_rates["opponent_lineup_k_pct"] = np.where(
            matchup_rates["plate_appearances"] > 0,
            matchup_rates["strikeouts"] / matchup_rates["plate_appearances"],
            np.nan,
        )
        feature_frame = feature_frame.merge(
            matchup_rates[["game_id", "team", "opponent_lineup_k_pct"]].rename(columns={"team": "opponent"}),
            on=["game_id", "opponent"],
            how="left",
            suffixes=("", "_computed"),
        )
        computed_k_pct = feature_frame["opponent_lineup_k_pct_computed"]
        feature_frame["opponent_lineup_k_pct"] = computed_k_pct.where(computed_k_pct.notna(), feature_frame["opponent_lineup_k_pct"])
        feature_frame = feature_frame.drop(columns=["opponent_lineup_k_pct_computed"])

    validate_columns(
        feature_frame,
        STRIKEOUTS_META_COLUMNS + STRIKEOUTS_FEATURE_COLUMNS + [STRIKEOUTS_TARGET_COLUMN],
        "strikeouts",
    )
    output_path = write_feature_snapshot(feature_frame, "strikeouts", start_date, end_date)
    persisted = persist_strikeout_features(feature_frame, start_date, end_date)
    log.info("Built %s strikeout rows -> %s and persisted %s DB rows", len(feature_frame), output_path, persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())