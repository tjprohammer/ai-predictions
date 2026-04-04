from __future__ import annotations

import pandas as pd


TOTALS_META_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "market_sportsbook",
    "prediction_ts",
    "game_start_ts",
    "line_snapshot_ts",
    "feature_cutoff_ts",
    "feature_version",
]

TOTALS_FEATURE_COLUMNS = [
    "home_runs_rate_blended",
    "away_runs_rate_blended",
    "home_hits_rate_blended",
    "away_hits_rate_blended",
    "home_xwoba_blended",
    "away_xwoba_blended",
    "home_iso_blended",
    "away_iso_blended",
    "home_bb_pct_blended",
    "away_bb_pct_blended",
    "home_k_pct_blended",
    "away_k_pct_blended",
    "home_starter_xwoba_blended",
    "away_starter_xwoba_blended",
    "home_starter_csw_blended",
    "away_starter_csw_blended",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "home_bullpen_pitches_last3",
    "away_bullpen_pitches_last3",
    "home_bullpen_innings_last3",
    "away_bullpen_innings_last3",
    "home_bullpen_b2b",
    "away_bullpen_b2b",
    "home_lineup_top5_xwoba",
    "away_lineup_top5_xwoba",
    "home_lineup_k_pct",
    "away_lineup_k_pct",
    "venue_run_factor",
    "venue_hr_factor",
    "temperature_f",
    "wind_speed_mph",
    "wind_direction_deg",
    "humidity_pct",
    "market_total",
    "market_over_price",
    "market_under_price",
    "line_movement",
]

TOTALS_TARGET_COLUMN = "actual_total_runs"

FIRST5_TOTALS_META_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "market_sportsbook",
    "prediction_ts",
    "game_start_ts",
    "line_snapshot_ts",
    "feature_cutoff_ts",
    "feature_version",
]

FIRST5_TOTALS_FEATURE_COLUMNS = [
    "home_runs_rate_blended",
    "away_runs_rate_blended",
    "home_hits_rate_blended",
    "away_hits_rate_blended",
    "home_xwoba_blended",
    "away_xwoba_blended",
    "home_iso_blended",
    "away_iso_blended",
    "home_bb_pct_blended",
    "away_bb_pct_blended",
    "home_k_pct_blended",
    "away_k_pct_blended",
    "home_starter_xwoba_blended",
    "away_starter_xwoba_blended",
    "home_starter_csw_blended",
    "away_starter_csw_blended",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "home_lineup_top5_xwoba",
    "away_lineup_top5_xwoba",
    "home_lineup_k_pct",
    "away_lineup_k_pct",
    "venue_run_factor",
    "venue_hr_factor",
    "temperature_f",
    "wind_speed_mph",
    "wind_direction_deg",
    "humidity_pct",
    "market_total",
    "market_over_price",
    "market_under_price",
    "line_movement",
]

FIRST5_TOTALS_TARGET_COLUMN = "actual_total_runs_first5"

HITS_META_COLUMNS = [
    "game_id",
    "game_date",
    "player_id",
    "team",
    "opponent",
    "prediction_ts",
    "game_start_ts",
    "line_snapshot_ts",
    "feature_cutoff_ts",
    "feature_version",
]

HITS_FEATURE_COLUMNS = [
    "player_name",
    "home_away",
    "lineup_slot",
    "is_confirmed_lineup",
    "projected_plate_appearances",
    "hit_rate_7",
    "hit_rate_14",
    "hit_rate_30",
    "hit_rate_blended",
    "xba_14",
    "xwoba_14",
    "hard_hit_pct_14",
    "k_pct_14",
    "season_prior_hit_rate",
    "season_prior_xba",
    "season_prior_xwoba",
    "opposing_starter_xwoba",
    "opposing_starter_csw",
    "opposing_bullpen_pitches_last3",
    "opposing_bullpen_innings_last3",
    "venue_run_factor",
    "park_hr_factor",
    "temperature_f",
    "wind_speed_mph",
    "team_run_environment",
    "streak_len_capped",
]

HITS_TARGET_COLUMN = "got_hit"

STRIKEOUTS_META_COLUMNS = [
    "game_id",
    "game_date",
    "pitcher_id",
    "team",
    "opponent",
    "prediction_ts",
    "game_start_ts",
    "line_snapshot_ts",
    "feature_cutoff_ts",
    "feature_version",
]

STRIKEOUTS_FEATURE_COLUMNS = [
    "throws",
    "days_rest",
    "projected_innings",
    "recent_avg_ip_3",
    "recent_avg_ip_5",
    "recent_avg_strikeouts_3",
    "recent_avg_strikeouts_5",
    "recent_k_per_batter_3",
    "recent_k_per_batter_5",
    "recent_avg_pitch_count_3",
    "recent_whiff_pct_5",
    "recent_csw_pct_5",
    "recent_xwoba_5",
    "baseline_strikeouts",
    "opponent_k_pct_blended",
    "same_hand_share",
    "opposite_hand_share",
    "switch_share",
    "lineup_right_count",
    "lineup_left_count",
    "lineup_switch_count",
    "known_hitters",
    "confirmed_hitters",
    "total_hitters",
    "handedness_adjustment_applied",
    "handedness_data_missing",
]

STRIKEOUTS_TARGET_COLUMN = "actual_strikeouts"


def validate_columns(frame: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")