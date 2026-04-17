import pandas as pd
import pytest

from src.features.contracts import (
    FIELD_ROLE_CALIBRATION_INPUT,
    FIELD_ROLE_CERTAINTY_SIGNAL,
    FIELD_ROLE_CORE_PREDICTOR,
    FIELD_ROLE_DIAGNOSTIC_FLAG,
    FIELD_ROLE_PRODUCT_ONLY,
    HITS_FEATURE_COLUMNS,
    STRIKEOUTS_FEATURE_COLUMNS,
    STRIKEOUTS_META_COLUMNS,
    TOTALS_FEATURE_COLUMNS,
    TOTALS_META_COLUMNS,
    feature_columns_for_roles,
    feature_field_roles,
    validate_columns,
)


def test_validate_columns_accepts_complete_frame():
    frame = pd.DataFrame([{column: 1 for column in TOTALS_META_COLUMNS}])
    validate_columns(frame, TOTALS_META_COLUMNS, "totals")


def test_validate_columns_raises_for_missing_columns():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame([{"game_id": 1}]), TOTALS_META_COLUMNS, "totals")


def test_validate_columns_accepts_strikeout_meta_frame():
    frame = pd.DataFrame([{column: 1 for column in STRIKEOUTS_META_COLUMNS}])
    validate_columns(frame, STRIKEOUTS_META_COLUMNS, "strikeouts")


def test_totals_feature_roles_cover_every_feature_column():
    roles = feature_field_roles("totals")

    assert set(roles) == set(TOTALS_FEATURE_COLUMNS)


def test_feature_columns_for_roles_excludes_calibration_inputs_from_totals_training():
    selected = feature_columns_for_roles(
        "totals",
        [FIELD_ROLE_CORE_PREDICTOR],
    )

    assert "market_total" not in selected
    assert "line_movement" not in selected
    assert "home_runs_rate_blended" in selected


def test_feature_columns_for_roles_excludes_non_core_hits_fields_from_training():
    selected = feature_columns_for_roles(
        "hits",
        [FIELD_ROLE_CORE_PREDICTOR],
    )

    assert "player_name" not in selected
    assert "is_confirmed_lineup" not in selected
    assert "streak_len_capped" not in selected
    assert "projected_plate_appearances" in selected


def test_hr_lane_includes_hr_per_pa_and_barrel_context():
    selected = feature_columns_for_roles(
        "hr",
        [FIELD_ROLE_CORE_PREDICTOR],
    )

    assert "player_name" not in selected
    assert "hr_per_pa_blended" in selected
    assert "opposing_starter_barrel_pct" in selected


def test_feature_columns_for_roles_excludes_certainty_and_diagnostic_strikeout_fields_from_training():
    selected = feature_columns_for_roles(
        "strikeouts",
        [FIELD_ROLE_CORE_PREDICTOR],
    )
    roles = feature_field_roles("strikeouts")

    assert set(roles) == set(STRIKEOUTS_FEATURE_COLUMNS)
    assert roles["throws"] == FIELD_ROLE_CORE_PREDICTOR
    assert roles["confirmed_hitters"] == FIELD_ROLE_CERTAINTY_SIGNAL
    assert roles["handedness_adjustment_applied"] == FIELD_ROLE_DIAGNOSTIC_FLAG
    assert "throws" in selected
    assert "season_strikeouts" in selected
    assert "season_k_per_start" in selected
    assert "confirmed_hitters" not in selected
    assert "handedness_adjustment_applied" not in selected
    assert "baseline_strikeouts" in selected


def test_feature_field_roles_exposes_all_supported_roles():
    hits_roles = set(feature_field_roles("hits").values())

    assert FIELD_ROLE_CORE_PREDICTOR in hits_roles
    assert FIELD_ROLE_CERTAINTY_SIGNAL in hits_roles
    assert FIELD_ROLE_PRODUCT_ONLY in hits_roles
    assert FIELD_ROLE_CALIBRATION_INPUT in set(feature_field_roles("totals").values())