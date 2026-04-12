from __future__ import annotations

import pandas as pd


FIELD_ROLE_CORE_PREDICTOR = "core predictor"
FIELD_ROLE_CALIBRATION_INPUT = "calibration input"
FIELD_ROLE_CERTAINTY_SIGNAL = "certainty signal"
FIELD_ROLE_DIAGNOSTIC_FLAG = "diagnostic flag"
FIELD_ROLE_PRODUCT_ONLY = "product-only field"
FIELD_ROLE_ENVIRONMENT_CONTEXT = "environment context"


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
    "home_starter_fb_velo_blended",
    "away_starter_fb_velo_blended",
    "home_starter_whiff_pct_blended",
    "away_starter_whiff_pct_blended",
    "home_starter_hard_hit_pct_blended",
    "away_starter_hard_hit_pct_blended",
    "home_starter_avg_ip_blended",
    "away_starter_avg_ip_blended",
    "home_starter_k_per_9_blended",
    "away_starter_k_per_9_blended",
    "home_bullpen_pitches_last3",
    "away_bullpen_pitches_last3",
    "home_bullpen_innings_last3",
    "away_bullpen_innings_last3",
    "home_bullpen_b2b",
    "away_bullpen_b2b",
    "home_bullpen_runs_allowed_last3",
    "away_bullpen_runs_allowed_last3",
    "home_bullpen_earned_runs_last3",
    "away_bullpen_earned_runs_last3",
    "home_bullpen_hits_allowed_last3",
    "away_bullpen_hits_allowed_last3",
    "home_bullpen_era_last3",
    "away_bullpen_era_last3",
    "home_bullpen_late_innings_last3",
    "away_bullpen_late_innings_last3",
    "home_bullpen_late_runs_allowed_last3",
    "away_bullpen_late_runs_allowed_last3",
    "home_bullpen_late_earned_runs_last3",
    "away_bullpen_late_earned_runs_last3",
    "home_bullpen_late_hits_allowed_last3",
    "away_bullpen_late_hits_allowed_last3",
    "home_bullpen_late_era_last3",
    "away_bullpen_late_era_last3",
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
    "ump_run_value",
    "market_total",
    "market_over_price",
    "market_under_price",
    "line_movement",
    "starter_certainty_score",
    "lineup_certainty_score",
    "weather_freshness_score",
    "market_freshness_score",
    "bullpen_completeness_score",
    "missing_fallback_count",
    "board_state",
]

TOTALS_TARGET_COLUMN = "actual_total_runs"
TOTALS_HOME_RUNS_COLUMN = "actual_home_runs"
TOTALS_AWAY_RUNS_COLUMN = "actual_away_runs"

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
    "home_starter_fb_velo_blended",
    "away_starter_fb_velo_blended",
    "home_starter_whiff_pct_blended",
    "away_starter_whiff_pct_blended",
    "home_starter_hard_hit_pct_blended",
    "away_starter_hard_hit_pct_blended",
    "home_starter_avg_ip_blended",
    "away_starter_avg_ip_blended",
    "home_starter_k_per_9_blended",
    "away_starter_k_per_9_blended",
    "starter_fb_velo_diff",
    "starter_k_per_9_diff",
    "starter_xwoba_diff",
    "starter_csw_diff",
    "starter_quality_gap",
    "starter_asymmetry_score",
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
    "ump_run_value",
    "market_total",
    "market_over_price",
    "market_under_price",
    "line_movement",
    "starter_certainty_score",
    "lineup_certainty_score",
    "weather_freshness_score",
    "market_freshness_score",
    "missing_fallback_count",
    "board_state",
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
    "streak_len",
    "starter_certainty_score",
    "lineup_certainty_score",
    "weather_freshness_score",
    "market_freshness_score",
    "bullpen_completeness_score",
    "missing_fallback_count",
    "board_state",
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
    "season_starts",
    "season_innings",
    "season_strikeouts",
    "season_k_per_start",
    "season_k_per_batter",
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
    "matchup_k_factor",
    "matchup_baseline_strikeouts",
    "matchup_recent_strikeouts_3",
    "matchup_recent_strikeouts_5",
    "matchup_season_k_per_start",
    "opponent_k_pct_blended",
    "pitcher_vs_team_k_rate",
    "opponent_lineup_k_pct_recent",
    "venue_k_factor",
    "ump_k_rate_adj",
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
    "starter_certainty_score",
    "market_freshness_score",
    "missing_fallback_count",
    "board_state",
]

STRIKEOUTS_TARGET_COLUMN = "actual_strikeouts"


TOTALS_FIELD_ROLES = {
    "home_runs_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_runs_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_hits_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_hits_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_iso_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_iso_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_bb_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_bb_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_k_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_k_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_csw_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_csw_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_rest_days": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_rest_days": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_fb_velo_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_fb_velo_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_whiff_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_whiff_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_hard_hit_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_hard_hit_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_avg_ip_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_avg_ip_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_k_per_9_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_k_per_9_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_bullpen_pitches_last3": FIELD_ROLE_CORE_PREDICTOR,
    "away_bullpen_pitches_last3": FIELD_ROLE_CORE_PREDICTOR,
    "home_bullpen_innings_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_innings_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_b2b": FIELD_ROLE_CORE_PREDICTOR,
    "away_bullpen_b2b": FIELD_ROLE_CORE_PREDICTOR,
    "home_bullpen_runs_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_runs_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_earned_runs_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_earned_runs_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_hits_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_hits_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_era_last3": FIELD_ROLE_CORE_PREDICTOR,
    "away_bullpen_era_last3": FIELD_ROLE_CORE_PREDICTOR,
    "home_bullpen_late_innings_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_late_innings_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_late_runs_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_late_runs_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_late_earned_runs_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_late_earned_runs_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_late_hits_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_late_hits_allowed_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_bullpen_late_era_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "away_bullpen_late_era_last3": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "home_lineup_top5_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "away_lineup_top5_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "home_lineup_k_pct": FIELD_ROLE_CORE_PREDICTOR,
    "away_lineup_k_pct": FIELD_ROLE_CORE_PREDICTOR,
    "venue_run_factor": FIELD_ROLE_CORE_PREDICTOR,
    "venue_hr_factor": FIELD_ROLE_CORE_PREDICTOR,
    "temperature_f": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_speed_mph": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_direction_deg": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "humidity_pct": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "ump_run_value": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "market_total": FIELD_ROLE_CALIBRATION_INPUT,
    "market_over_price": FIELD_ROLE_CALIBRATION_INPUT,
    "market_under_price": FIELD_ROLE_CALIBRATION_INPUT,
    "line_movement": FIELD_ROLE_CALIBRATION_INPUT,
    "starter_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "lineup_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "weather_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "market_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "bullpen_completeness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "missing_fallback_count": FIELD_ROLE_CERTAINTY_SIGNAL,
    "board_state": FIELD_ROLE_CERTAINTY_SIGNAL,
}

FIRST5_TOTALS_FIELD_ROLES = {
    "home_runs_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_runs_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_hits_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_hits_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_iso_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_iso_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_bb_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_bb_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_k_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_k_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_xwoba_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_csw_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_csw_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_rest_days": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_rest_days": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_fb_velo_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_fb_velo_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_whiff_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_whiff_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_hard_hit_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_hard_hit_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_avg_ip_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_avg_ip_blended": FIELD_ROLE_CORE_PREDICTOR,
    "home_starter_k_per_9_blended": FIELD_ROLE_CORE_PREDICTOR,
    "away_starter_k_per_9_blended": FIELD_ROLE_CORE_PREDICTOR,
    "starter_fb_velo_diff": FIELD_ROLE_CORE_PREDICTOR,
    "starter_k_per_9_diff": FIELD_ROLE_CORE_PREDICTOR,
    "starter_xwoba_diff": FIELD_ROLE_CORE_PREDICTOR,
    "starter_csw_diff": FIELD_ROLE_CORE_PREDICTOR,
    "starter_quality_gap": FIELD_ROLE_CORE_PREDICTOR,
    "starter_asymmetry_score": FIELD_ROLE_CORE_PREDICTOR,
    "home_lineup_top5_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "away_lineup_top5_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "home_lineup_k_pct": FIELD_ROLE_CORE_PREDICTOR,
    "away_lineup_k_pct": FIELD_ROLE_CORE_PREDICTOR,
    "venue_run_factor": FIELD_ROLE_CORE_PREDICTOR,
    "venue_hr_factor": FIELD_ROLE_CORE_PREDICTOR,
    "temperature_f": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_speed_mph": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_direction_deg": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "humidity_pct": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "ump_run_value": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "market_total": FIELD_ROLE_CALIBRATION_INPUT,
    "market_over_price": FIELD_ROLE_CALIBRATION_INPUT,
    "market_under_price": FIELD_ROLE_CALIBRATION_INPUT,
    "line_movement": FIELD_ROLE_CALIBRATION_INPUT,
    "starter_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "lineup_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "weather_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "market_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "missing_fallback_count": FIELD_ROLE_CERTAINTY_SIGNAL,
    "board_state": FIELD_ROLE_CERTAINTY_SIGNAL,
}

HITS_FIELD_ROLES = {
    "player_name": FIELD_ROLE_PRODUCT_ONLY,
    "home_away": FIELD_ROLE_CORE_PREDICTOR,
    "lineup_slot": FIELD_ROLE_CORE_PREDICTOR,
    "is_confirmed_lineup": FIELD_ROLE_CERTAINTY_SIGNAL,
    "projected_plate_appearances": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_7": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_14": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_30": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "xba_14": FIELD_ROLE_CORE_PREDICTOR,
    "xwoba_14": FIELD_ROLE_CORE_PREDICTOR,
    "hard_hit_pct_14": FIELD_ROLE_CORE_PREDICTOR,
    "k_pct_14": FIELD_ROLE_CORE_PREDICTOR,
    "season_prior_hit_rate": FIELD_ROLE_CORE_PREDICTOR,
    "season_prior_xba": FIELD_ROLE_CORE_PREDICTOR,
    "season_prior_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_starter_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_starter_csw": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_bullpen_pitches_last3": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_bullpen_innings_last3": FIELD_ROLE_CORE_PREDICTOR,
    "venue_run_factor": FIELD_ROLE_CORE_PREDICTOR,
    "park_hr_factor": FIELD_ROLE_CORE_PREDICTOR,
    "temperature_f": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_speed_mph": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "team_run_environment": FIELD_ROLE_CORE_PREDICTOR,
    "streak_len_capped": FIELD_ROLE_PRODUCT_ONLY,
    "streak_len": FIELD_ROLE_PRODUCT_ONLY,
    "starter_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "lineup_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "weather_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "market_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "bullpen_completeness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "missing_fallback_count": FIELD_ROLE_CERTAINTY_SIGNAL,
    "board_state": FIELD_ROLE_CERTAINTY_SIGNAL,
}

STRIKEOUTS_FIELD_ROLES = {
    "throws": FIELD_ROLE_CORE_PREDICTOR,
    "days_rest": FIELD_ROLE_CORE_PREDICTOR,
    "projected_innings": FIELD_ROLE_CORE_PREDICTOR,
    "season_starts": FIELD_ROLE_CORE_PREDICTOR,
    "season_innings": FIELD_ROLE_CORE_PREDICTOR,
    "season_strikeouts": FIELD_ROLE_CORE_PREDICTOR,
    "season_k_per_start": FIELD_ROLE_CORE_PREDICTOR,
    "season_k_per_batter": FIELD_ROLE_CORE_PREDICTOR,
    "recent_avg_ip_3": FIELD_ROLE_CORE_PREDICTOR,
    "recent_avg_ip_5": FIELD_ROLE_CORE_PREDICTOR,
    "recent_avg_strikeouts_3": FIELD_ROLE_CORE_PREDICTOR,
    "recent_avg_strikeouts_5": FIELD_ROLE_CORE_PREDICTOR,
    "recent_k_per_batter_3": FIELD_ROLE_CORE_PREDICTOR,
    "recent_k_per_batter_5": FIELD_ROLE_CORE_PREDICTOR,
    "recent_avg_pitch_count_3": FIELD_ROLE_CORE_PREDICTOR,
    "recent_whiff_pct_5": FIELD_ROLE_CORE_PREDICTOR,
    "recent_csw_pct_5": FIELD_ROLE_CORE_PREDICTOR,
    "recent_xwoba_5": FIELD_ROLE_CORE_PREDICTOR,
    "baseline_strikeouts": FIELD_ROLE_CORE_PREDICTOR,
    "matchup_k_factor": FIELD_ROLE_CORE_PREDICTOR,
    "matchup_baseline_strikeouts": FIELD_ROLE_CORE_PREDICTOR,
    "matchup_recent_strikeouts_3": FIELD_ROLE_CORE_PREDICTOR,
    "matchup_recent_strikeouts_5": FIELD_ROLE_CORE_PREDICTOR,
    "matchup_season_k_per_start": FIELD_ROLE_CORE_PREDICTOR,
    "opponent_k_pct_blended": FIELD_ROLE_CORE_PREDICTOR,
    "pitcher_vs_team_k_rate": FIELD_ROLE_CORE_PREDICTOR,
    "opponent_lineup_k_pct_recent": FIELD_ROLE_CORE_PREDICTOR,
    "venue_k_factor": FIELD_ROLE_CORE_PREDICTOR,
    "ump_k_rate_adj": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "same_hand_share": FIELD_ROLE_CORE_PREDICTOR,
    "opposite_hand_share": FIELD_ROLE_CORE_PREDICTOR,
    "switch_share": FIELD_ROLE_CORE_PREDICTOR,
    "lineup_right_count": FIELD_ROLE_CORE_PREDICTOR,
    "lineup_left_count": FIELD_ROLE_CORE_PREDICTOR,
    "lineup_switch_count": FIELD_ROLE_CORE_PREDICTOR,
    "known_hitters": FIELD_ROLE_CERTAINTY_SIGNAL,
    "confirmed_hitters": FIELD_ROLE_CERTAINTY_SIGNAL,
    "total_hitters": FIELD_ROLE_CERTAINTY_SIGNAL,
    "handedness_adjustment_applied": FIELD_ROLE_DIAGNOSTIC_FLAG,
    "handedness_data_missing": FIELD_ROLE_CERTAINTY_SIGNAL,
    "starter_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "market_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "missing_fallback_count": FIELD_ROLE_CERTAINTY_SIGNAL,
    "board_state": FIELD_ROLE_CERTAINTY_SIGNAL,
}

TOTAL_BASES_META_COLUMNS = [
    "game_id", "game_date", "player_id", "team", "opponent",
    "prediction_ts", "game_start_ts", "line_snapshot_ts",
    "feature_cutoff_ts", "feature_version", "player_name",
]

TOTAL_BASES_FEATURE_COLUMNS = [
    "home_away", "lineup_slot", "is_confirmed_lineup", "projected_plate_appearances",
    "hit_rate_7", "hit_rate_14", "hit_rate_30", "hit_rate_blended",
    "xba_14", "xwoba_14", "hard_hit_pct_14", "k_pct_14",
    "season_prior_hit_rate", "season_prior_xwoba",
    "tb_rate_7", "tb_rate_14", "tb_rate_30", "iso_14", "hr_rate_14",
    "opposing_starter_xwoba", "opposing_starter_hard_hit_pct", "opposing_starter_k_pct",
    "opposing_bullpen_pitches_last3",
    "venue_run_factor", "park_hr_factor",
    "temperature_f", "wind_speed_mph",
    "team_run_environment", "game_total_line",
    "ump_run_value",
    "market_tb_line", "market_tb_over_price", "market_tb_under_price",
    "starter_certainty_score", "lineup_certainty_score",
    "weather_freshness_score", "market_freshness_score",
    "missing_fallback_count", "board_state",
]

TOTAL_BASES_TARGET_COLUMN = "actual_total_bases"

TOTAL_BASES_CERTAINTY_KEY_FIELDS = [
    "opposing_starter_xwoba", "xwoba_14", "tb_rate_14",
    "temperature_f", "venue_run_factor",
]

TOTAL_BASES_FIELD_ROLES = {
    "home_away": FIELD_ROLE_PRODUCT_ONLY,
    "lineup_slot": FIELD_ROLE_CORE_PREDICTOR,
    "is_confirmed_lineup": FIELD_ROLE_CERTAINTY_SIGNAL,
    "projected_plate_appearances": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_7": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_14": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_30": FIELD_ROLE_CORE_PREDICTOR,
    "hit_rate_blended": FIELD_ROLE_CORE_PREDICTOR,
    "xba_14": FIELD_ROLE_CORE_PREDICTOR,
    "xwoba_14": FIELD_ROLE_CORE_PREDICTOR,
    "hard_hit_pct_14": FIELD_ROLE_CORE_PREDICTOR,
    "k_pct_14": FIELD_ROLE_CORE_PREDICTOR,
    "season_prior_hit_rate": FIELD_ROLE_CORE_PREDICTOR,
    "season_prior_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "tb_rate_7": FIELD_ROLE_CORE_PREDICTOR,
    "tb_rate_14": FIELD_ROLE_CORE_PREDICTOR,
    "tb_rate_30": FIELD_ROLE_CORE_PREDICTOR,
    "iso_14": FIELD_ROLE_CORE_PREDICTOR,
    "hr_rate_14": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_starter_xwoba": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_starter_hard_hit_pct": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_starter_k_pct": FIELD_ROLE_CORE_PREDICTOR,
    "opposing_bullpen_pitches_last3": FIELD_ROLE_CORE_PREDICTOR,
    "venue_run_factor": FIELD_ROLE_CORE_PREDICTOR,
    "park_hr_factor": FIELD_ROLE_CORE_PREDICTOR,
    "temperature_f": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "wind_speed_mph": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "team_run_environment": FIELD_ROLE_CORE_PREDICTOR,
    "game_total_line": FIELD_ROLE_CALIBRATION_INPUT,
    "ump_run_value": FIELD_ROLE_ENVIRONMENT_CONTEXT,
    "market_tb_line": FIELD_ROLE_CALIBRATION_INPUT,
    "market_tb_over_price": FIELD_ROLE_CALIBRATION_INPUT,
    "market_tb_under_price": FIELD_ROLE_CALIBRATION_INPUT,
    "starter_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "lineup_certainty_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "weather_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "market_freshness_score": FIELD_ROLE_CERTAINTY_SIGNAL,
    "missing_fallback_count": FIELD_ROLE_CERTAINTY_SIGNAL,
    "board_state": FIELD_ROLE_CERTAINTY_SIGNAL,
}

LANE_FEATURE_COLUMNS = {
    "totals": TOTALS_FEATURE_COLUMNS,
    "first5_totals": FIRST5_TOTALS_FEATURE_COLUMNS,
    "hits": HITS_FEATURE_COLUMNS,
    "strikeouts": STRIKEOUTS_FEATURE_COLUMNS,
    "total_bases": TOTAL_BASES_FEATURE_COLUMNS,
}

LANE_FIELD_ROLES = {
    "totals": TOTALS_FIELD_ROLES,
    "first5_totals": FIRST5_TOTALS_FIELD_ROLES,
    "hits": HITS_FIELD_ROLES,
    "strikeouts": STRIKEOUTS_FIELD_ROLES,
    "total_bases": TOTAL_BASES_FIELD_ROLES,
}

TOTALS_CERTAINTY_KEY_FIELDS = [
    "home_starter_xwoba_blended", "away_starter_xwoba_blended",
    "home_lineup_top5_xwoba", "away_lineup_top5_xwoba",
    "temperature_f", "market_total", "venue_run_factor",
]

FIRST5_TOTALS_CERTAINTY_KEY_FIELDS = [
    "home_starter_xwoba_blended", "away_starter_xwoba_blended",
    "home_lineup_top5_xwoba", "away_lineup_top5_xwoba",
    "temperature_f", "market_total", "venue_run_factor",
]

HITS_CERTAINTY_KEY_FIELDS = [
    "opposing_starter_xwoba", "xwoba_14",
    "temperature_f", "venue_run_factor",
]

STRIKEOUTS_CERTAINTY_KEY_FIELDS = [
    "recent_avg_strikeouts_5", "baseline_strikeouts", "opponent_k_pct_blended",
]


def feature_field_roles(lane: str) -> dict[str, str]:
    try:
        return dict(LANE_FIELD_ROLES[lane])
    except KeyError as exc:
        raise ValueError(f"Unsupported lane for field roles: {lane}") from exc


def feature_columns_for_roles(
    lane: str,
    allowed_roles: list[str] | tuple[str, ...],
    available_columns: list[str] | None = None,
) -> list[str]:
    try:
        feature_columns = LANE_FEATURE_COLUMNS[lane]
        field_roles = LANE_FIELD_ROLES[lane]
    except KeyError as exc:
        raise ValueError(f"Unsupported lane for field roles: {lane}") from exc

    allowed_role_set = set(allowed_roles)
    selected = [column for column in feature_columns if field_roles[column] in allowed_role_set]
    if available_columns is None:
        return selected
    available_column_set = set(available_columns)
    return [column for column in selected if column in available_column_set]


def _validate_field_role_maps() -> None:
    for lane, feature_columns in LANE_FEATURE_COLUMNS.items():
        role_map = LANE_FIELD_ROLES[lane]
        missing = [column for column in feature_columns if column not in role_map]
        extras = [column for column in role_map if column not in feature_columns]
        if missing or extras:
            raise ValueError(
                f"Field role map mismatch for {lane}: missing={missing}, extras={extras}"
            )


_validate_field_role_maps()


def validate_columns(frame: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")