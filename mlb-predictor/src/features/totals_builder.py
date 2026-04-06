from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.features.common import (
    build_hitter_priors,
    build_pitcher_priors,
    build_team_priors,
    bullpen_snapshot,
    coerce_utc_timestamp_series,
    compute_board_state,
    compute_freshness_score,
    compute_starter_certainty,
    count_missing_fallbacks,
    default_cutoff,
    latest_market_snapshot,
    latest_weather_snapshot,
    lineup_snapshot,
    offense_snapshot,
    park_snapshot,
    persist_totals_features,
    pitcher_snapshot,
    write_feature_snapshot,
)
from src.features.contracts import TOTALS_CERTAINTY_KEY_FIELDS, TOTALS_FEATURE_COLUMNS, TOTALS_META_COLUMNS, TOTALS_TARGET_COLUMN, validate_columns
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _load_frames(start_date, end_date, settings):
    history_start = f"{settings.prior_season}-01-01"
    frames = {
        "games": query_df(
            """
            SELECT game_id, game_date, game_start_ts, status, home_team, away_team, total_runs, season
            FROM games
            WHERE game_date BETWEEN :start_date AND :end_date
            ORDER BY game_date, game_id
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "team_offense": query_df(
            """
            SELECT *
            FROM team_offense_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "bullpens": query_df(
            """
            SELECT *
            FROM bullpens_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "pitcher_starts": query_df(
            """
            SELECT *
            FROM pitcher_starts
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "lineups": query_df(
            """
            SELECT *
            FROM lineups
            WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "player_batting": query_df(
            """
            SELECT *
            FROM player_game_batting
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "weather": query_df(
            """
            SELECT *
            FROM game_weather
            WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "markets": query_df(
            """
            SELECT *
            FROM game_markets
            WHERE game_date BETWEEN :start_date AND :end_date
              AND market_type = 'total'
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "market_freezes": query_df("SELECT * FROM market_selection_freezes WHERE market_type = 'total'")
        if table_exists("market_selection_freezes")
        else pd.DataFrame(),
        "parks": query_df("SELECT * FROM park_factors"),
    }
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical full-game totals feature rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)
    settings = get_settings()

    frames = _load_frames(start_date, end_date, settings)
    games = frames["games"]
    if games.empty:
        log.info("No games found for totals feature build")
        return 0

    for frame_name in ("team_offense", "bullpens", "pitcher_starts", "player_batting"):
        if not frames[frame_name].empty:
            frames[frame_name]["game_date"] = pd.to_datetime(frames[frame_name]["game_date"])
    for frame_name in ("lineups", "weather", "markets"):
        if not frames[frame_name].empty and "snapshot_ts" in frames[frame_name].columns:
            frames[frame_name]["snapshot_ts"] = coerce_utc_timestamp_series(frames[frame_name]["snapshot_ts"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["game_start_ts"] = pd.to_datetime(games["game_start_ts"], utc=True)

    team_priors = build_team_priors(frames["team_offense"], settings.prior_season)
    pitcher_priors = build_pitcher_priors(frames["pitcher_starts"], settings.prior_season)
    hitter_priors = build_hitter_priors(frames["player_batting"], settings.prior_season)
    prediction_ts = datetime.now(timezone.utc)
    feature_version = f"{settings.model_version_prefix}_totals_core"

    rows = []
    for game in games.itertuples(index=False):
        cutoff_ts = default_cutoff(game.game_date, game.game_start_ts)
        starters = frames["pitcher_starts"][frames["pitcher_starts"]["game_id"] == game.game_id].copy()
        home_starter_id = None
        away_starter_id = None
        if not starters.empty:
            starters = starters.sort_values(["is_probable", "pitcher_id"])
            home_rows = starters[starters["side"] == "home"]
            away_rows = starters[starters["side"] == "away"]
            home_starter_id = int(home_rows.iloc[0]["pitcher_id"]) if not home_rows.empty else None
            away_starter_id = int(away_rows.iloc[0]["pitcher_id"]) if not away_rows.empty else None
            home_starter_probable = bool(home_rows.iloc[0]["is_probable"]) if not home_rows.empty else None
            away_starter_probable = bool(away_rows.iloc[0]["is_probable"]) if not away_rows.empty else None
        else:
            home_starter_probable = None
            away_starter_probable = None

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
        home_starter = pitcher_snapshot(
            home_starter_id,
            game.game_date,
            frames["pitcher_starts"],
            pitcher_priors,
            full_weight_starts=settings.pitcher_full_weight_starts,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        away_starter = pitcher_snapshot(
            away_starter_id,
            game.game_date,
            frames["pitcher_starts"],
            pitcher_priors,
            full_weight_starts=settings.pitcher_full_weight_starts,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        home_bullpen = bullpen_snapshot(game.home_team, game.game_date, frames["bullpens"])
        away_bullpen = bullpen_snapshot(game.away_team, game.game_date, frames["bullpens"])
        home_lineup = lineup_snapshot(
            game.game_id,
            game.home_team,
            cutoff_ts,
            frames["lineups"],
            frames["player_batting"],
            hitter_priors,
            game.game_date,
            settings.min_pa_full_weight,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        away_lineup = lineup_snapshot(
            game.game_id,
            game.away_team,
            cutoff_ts,
            frames["lineups"],
            frames["player_batting"],
            hitter_priors,
            game.game_date,
            settings.min_pa_full_weight,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        market = latest_market_snapshot(
            game.game_id,
            cutoff_ts,
            frames["markets"],
            freezes=frames["market_freezes"],
            market_type="total",
        )
        weather = latest_weather_snapshot(game.game_id, cutoff_ts, frames["weather"])
        park = park_snapshot(game.home_team, int(game.season or game.game_date.year), frames["parks"], settings.prior_season)

        game_start = game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None
        starter_certainty = (
            compute_starter_certainty(home_starter_id, home_starter_probable)
            + compute_starter_certainty(away_starter_id, away_starter_probable)
        ) / 2.0
        lineup_certainty = 0.0
        home_total = home_lineup["total_count"]
        away_total = away_lineup["total_count"]
        if home_total or away_total:
            home_ratio = home_lineup["confirmed_count"] / home_total if home_total else 0.0
            away_ratio = away_lineup["confirmed_count"] / away_total if away_total else 0.0
            lineup_certainty = (home_ratio + away_ratio) / 2.0
        weather_freshness = compute_freshness_score(
            weather["weather_snapshot_ts"], game_start, decay_hours=(3, 12, 24, 48),
        )
        market_freshness = compute_freshness_score(
            market["line_snapshot_ts"], game_start, decay_hours=(1, 6, 12, 24),
        )
        bullpen_completeness = (home_bullpen["completeness_3"] + away_bullpen["completeness_3"]) / 2.0

        rows.append(
            {
                "game_id": int(game.game_id),
                "game_date": game.game_date,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "prediction_ts": prediction_ts,
                "game_start_ts": game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None,
                "line_snapshot_ts": market["line_snapshot_ts"],
                "feature_cutoff_ts": cutoff_ts,
                "feature_version": feature_version,
                "home_runs_rate_blended": home_offense["runs_rate"],
                "away_runs_rate_blended": away_offense["runs_rate"],
                "home_hits_rate_blended": home_offense["hits_rate"],
                "away_hits_rate_blended": away_offense["hits_rate"],
                "home_xwoba_blended": home_offense["xwoba"],
                "away_xwoba_blended": away_offense["xwoba"],
                "home_iso_blended": home_offense["iso"],
                "away_iso_blended": away_offense["iso"],
                "home_bb_pct_blended": home_offense["bb_pct"],
                "away_bb_pct_blended": away_offense["bb_pct"],
                "home_k_pct_blended": home_offense["k_pct"],
                "away_k_pct_blended": away_offense["k_pct"],
                "home_starter_xwoba_blended": home_starter["xwoba"],
                "away_starter_xwoba_blended": away_starter["xwoba"],
                "home_starter_csw_blended": home_starter["csw"],
                "away_starter_csw_blended": away_starter["csw"],
                "home_starter_rest_days": home_starter["days_rest"],
                "away_starter_rest_days": away_starter["days_rest"],
                "home_bullpen_pitches_last3": home_bullpen["pitches_last3"],
                "away_bullpen_pitches_last3": away_bullpen["pitches_last3"],
                "home_bullpen_innings_last3": home_bullpen["innings_last3"],
                "away_bullpen_innings_last3": away_bullpen["innings_last3"],
                "home_bullpen_b2b": home_bullpen["b2b"],
                "away_bullpen_b2b": away_bullpen["b2b"],
                "home_bullpen_runs_allowed_last3": home_bullpen["runs_allowed_last3"],
                "away_bullpen_runs_allowed_last3": away_bullpen["runs_allowed_last3"],
                "home_bullpen_earned_runs_last3": home_bullpen["earned_runs_last3"],
                "away_bullpen_earned_runs_last3": away_bullpen["earned_runs_last3"],
                "home_bullpen_hits_allowed_last3": home_bullpen["hits_allowed_last3"],
                "away_bullpen_hits_allowed_last3": away_bullpen["hits_allowed_last3"],
                "home_bullpen_era_last3": home_bullpen["era_last3"],
                "away_bullpen_era_last3": away_bullpen["era_last3"],
                "home_lineup_top5_xwoba": home_lineup["top5_xwoba"],
                "away_lineup_top5_xwoba": away_lineup["top5_xwoba"],
                "home_lineup_k_pct": home_lineup["lineup_k_pct"],
                "away_lineup_k_pct": away_lineup["lineup_k_pct"],
                "venue_run_factor": park["run_factor"],
                "venue_hr_factor": park["hr_factor"],
                "temperature_f": weather["temperature_f"],
                "wind_speed_mph": weather["wind_speed_mph"],
                "wind_direction_deg": weather["wind_direction_deg"],
                "humidity_pct": weather["humidity_pct"],
                "market_sportsbook": market.get("market_sportsbook"),
                "market_total": market["market_total"],
                "market_over_price": market["market_over_price"],
                "market_under_price": market["market_under_price"],
                "line_movement": market["line_movement"],
                "actual_total_runs": game.total_runs,
            }
        )
        row = rows[-1]
        missing = count_missing_fallbacks(row, TOTALS_CERTAINTY_KEY_FIELDS)
        row["starter_certainty_score"] = starter_certainty
        row["lineup_certainty_score"] = lineup_certainty
        row["weather_freshness_score"] = weather_freshness
        row["market_freshness_score"] = market_freshness
        row["bullpen_completeness_score"] = bullpen_completeness
        row["missing_fallback_count"] = missing
        row["board_state"] = compute_board_state(missing, threshold_minimal=3)

    feature_frame = pd.DataFrame(rows)
    validate_columns(feature_frame, TOTALS_META_COLUMNS + TOTALS_FEATURE_COLUMNS + [TOTALS_TARGET_COLUMN], "totals")
    output_path = write_feature_snapshot(feature_frame, "totals", start_date, end_date)
    persisted = persist_totals_features(feature_frame, start_date, end_date)
    log.info("Built %s totals rows -> %s and persisted %s DB rows", len(feature_frame), output_path, persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())