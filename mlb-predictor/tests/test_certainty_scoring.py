"""Tests for certainty scoring helpers and contract registration."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.common import (
    STARTER_CERTAINTY_SCHEDULE_PROBABLE,
    STARTER_CERTAINTY_UNSPECIFIED,
    bullpen_snapshot,
    compute_board_state,
    compute_freshness_score,
    compute_starter_certainty,
    count_missing_fallbacks,
    latest_weather_snapshot,
    lineup_snapshot,
)
from src.features.contracts import (
    FIELD_ROLE_CERTAINTY_SIGNAL,
    FIRST5_TOTALS_CERTAINTY_KEY_FIELDS,
    FIRST5_TOTALS_FEATURE_COLUMNS,
    FIRST5_TOTALS_FIELD_ROLES,
    HITS_CERTAINTY_KEY_FIELDS,
    HITS_FEATURE_COLUMNS,
    HITS_FIELD_ROLES,
    STRIKEOUTS_CERTAINTY_KEY_FIELDS,
    STRIKEOUTS_FEATURE_COLUMNS,
    STRIKEOUTS_FIELD_ROLES,
    TOTALS_CERTAINTY_KEY_FIELDS,
    TOTALS_FEATURE_COLUMNS,
    TOTALS_FIELD_ROLES,
)


# ---------------------------------------------------------------------------
# compute_starter_certainty
# ---------------------------------------------------------------------------

class TestComputeStarterCertainty:
    def test_no_starter(self):
        assert compute_starter_certainty(None, None) == 0.0

    def test_starter_schedule_probable(self):
        assert compute_starter_certainty(12345, True) == STARTER_CERTAINTY_SCHEDULE_PROBABLE

    def test_starter_boxscore_confirmed(self):
        assert compute_starter_certainty(12345, False) == 1.0

    def test_starter_flag_unspecified(self):
        assert compute_starter_certainty(12345, None) == STARTER_CERTAINTY_UNSPECIFIED


# ---------------------------------------------------------------------------
# compute_freshness_score
# ---------------------------------------------------------------------------

class TestComputeFreshnessScore:
    def _ts(self, hours_before_ref: float) -> datetime:
        ref = datetime(2024, 7, 1, 19, 0, tzinfo=timezone.utc)
        from datetime import timedelta
        return ref - timedelta(hours=hours_before_ref)

    @property
    def ref(self):
        return datetime(2024, 7, 1, 19, 0, tzinfo=timezone.utc)

    def test_none_snapshot(self):
        assert compute_freshness_score(None, self.ref) == 0.0

    def test_none_reference(self):
        assert compute_freshness_score(self.ref, None) == 0.0

    def test_zero_age(self):
        assert compute_freshness_score(self.ref, self.ref) == 1.0

    def test_observed_future_snapshot(self):
        from datetime import timedelta
        future = self.ref + timedelta(hours=1)
        assert compute_freshness_score(future, self.ref) == 1.0

    def test_default_decay_1h(self):
        score = compute_freshness_score(self._ts(1), self.ref)
        assert score == 0.9

    def test_default_decay_6h(self):
        score = compute_freshness_score(self._ts(6), self.ref)
        assert score == 0.7

    def test_default_decay_18h(self):
        score = compute_freshness_score(self._ts(18), self.ref)
        assert score == 0.5

    def test_default_decay_36h(self):
        score = compute_freshness_score(self._ts(36), self.ref)
        assert score == 0.3

    def test_very_old(self):
        score = compute_freshness_score(self._ts(100), self.ref)
        assert score == 0.1

    def test_market_decay_hours(self):
        score = compute_freshness_score(self._ts(0.5), self.ref, decay_hours=(1, 6, 12, 24))
        assert score == 0.9

    def test_weather_decay_hours(self):
        score = compute_freshness_score(self._ts(2), self.ref, decay_hours=(3, 12, 24, 48))
        assert score == 0.9


# ---------------------------------------------------------------------------
# count_missing_fallbacks
# ---------------------------------------------------------------------------

class TestCountMissingFallbacks:
    def test_no_missing(self):
        row = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert count_missing_fallbacks(row, ["a", "b", "c"]) == 0

    def test_all_missing(self):
        row = {"a": None, "b": None}
        assert count_missing_fallbacks(row, ["a", "b"]) == 2

    def test_nan_treated_as_missing(self):
        row = {"a": float("nan"), "b": 1.0}
        assert count_missing_fallbacks(row, ["a", "b"]) == 1

    def test_missing_key_treated_as_missing(self):
        row = {"a": 1.0}
        assert count_missing_fallbacks(row, ["a", "b"]) == 1

    def test_zero_not_missing(self):
        row = {"a": 0, "b": 0.0}
        assert count_missing_fallbacks(row, ["a", "b"]) == 0

    def test_string_not_missing(self):
        row = {"a": "hello"}
        assert count_missing_fallbacks(row, ["a"]) == 0


# ---------------------------------------------------------------------------
# compute_board_state
# ---------------------------------------------------------------------------

class TestComputeBoardState:
    def test_complete(self):
        assert compute_board_state(0) == "complete"

    def test_partial_default(self):
        assert compute_board_state(1) == "partial"
        assert compute_board_state(2) == "partial"

    def test_minimal_default(self):
        assert compute_board_state(3) == "minimal"
        assert compute_board_state(7) == "minimal"

    def test_custom_threshold(self):
        assert compute_board_state(1, threshold_minimal=2) == "partial"
        assert compute_board_state(2, threshold_minimal=2) == "minimal"


# ---------------------------------------------------------------------------
# Extended snapshot helpers return new keys
# ---------------------------------------------------------------------------

class TestBullpenSnapshotCompleteness:
    def test_empty_bullpen_returns_zero(self):
        bp = pd.DataFrame(columns=["team", "game_date", "innings_pitched", "pitches_thrown",
                                     "earned_runs", "hits_allowed", "runs_allowed"])
        from datetime import date
        result = bullpen_snapshot("NYY", date(2024, 7, 1), bp)
        assert result["completeness_3"] == 0.0

    def test_partial_bullpen(self):
        from datetime import date
        bp = pd.DataFrame({
            "team": ["NYY", "NYY"],
            "game_date": [date(2024, 6, 29), date(2024, 6, 30)],
            "innings_pitched": [2.0, 3.0],
            "pitches_thrown": [30, 40],
            "earned_runs": [1, 2],
            "hits_allowed": [2, 3],
            "runs_allowed": [1, 2],
        })
        result = bullpen_snapshot("NYY", date(2024, 7, 1), bp)
        assert abs(result["completeness_3"] - 2 / 3) < 1e-9

    def test_full_bullpen(self):
        from datetime import date
        bp = pd.DataFrame({
            "team": ["NYY"] * 3,
            "game_date": [date(2024, 6, 28), date(2024, 6, 29), date(2024, 6, 30)],
            "innings_pitched": [2.0, 3.0, 1.0],
            "pitches_thrown": [30, 40, 20],
            "earned_runs": [1, 2, 0],
            "hits_allowed": [2, 3, 1],
            "runs_allowed": [1, 2, 0],
        })
        result = bullpen_snapshot("NYY", date(2024, 7, 1), bp)
        assert result["completeness_3"] == 1.0


class TestWeatherSnapshotTimestamp:
    def test_empty_returns_none_ts(self):
        weather = pd.DataFrame(columns=["game_id", "snapshot_ts", "temperature_f",
                                          "wind_speed_mph", "wind_direction_deg", "humidity_pct"])
        result = latest_weather_snapshot(1, datetime(2024, 7, 1, tzinfo=timezone.utc), weather)
        assert result["weather_snapshot_ts"] is None

    def test_returns_snapshot_ts(self):
        ts = pd.Timestamp("2024-07-01 18:00", tz="UTC")
        weather = pd.DataFrame({
            "game_id": [1],
            "snapshot_ts": [ts],
            "temperature_f": [72],
            "wind_speed_mph": [5],
            "wind_direction_deg": [180],
            "humidity_pct": [50],
        })
        result = latest_weather_snapshot(1, datetime(2024, 7, 1, 20, 0, tzinfo=timezone.utc), weather)
        assert result["weather_snapshot_ts"] == ts


class TestLineupSnapshotCounts:
    def test_empty_lineup_returns_zero_counts(self):
        lineups = pd.DataFrame(columns=["game_id", "team", "snapshot_ts", "player_id",
                                          "lineup_slot", "is_confirmed", "player_name"])
        batting = pd.DataFrame(columns=["game_id", "game_date", "player_id", "team",
                                          "lineup_slot", "plate_appearances"])
        from datetime import date
        hitter_priors = pd.DataFrame()
        result = lineup_snapshot(1, "NYY", datetime(2024, 7, 1, tzinfo=timezone.utc),
                                  lineups, batting, hitter_priors, date(2024, 7, 1), 50)
        assert result["confirmed_count"] == 0
        assert result["total_count"] == 0


# ---------------------------------------------------------------------------
# Contract registration checks
# ---------------------------------------------------------------------------

CERTAINTY_FIELDS_TOTALS = [
    "starter_certainty_score", "lineup_certainty_score", "weather_freshness_score",
    "market_freshness_score", "bullpen_completeness_score", "missing_fallback_count",
    "board_state",
]

CERTAINTY_FIELDS_FIRST5 = [
    "starter_certainty_score", "lineup_certainty_score", "weather_freshness_score",
    "market_freshness_score", "missing_fallback_count", "board_state",
]

CERTAINTY_FIELDS_HITS = [
    "starter_certainty_score", "lineup_certainty_score", "weather_freshness_score",
    "market_freshness_score", "bullpen_completeness_score", "missing_fallback_count",
    "board_state",
]

CERTAINTY_FIELDS_STRIKEOUTS = [
    "starter_certainty_score", "market_freshness_score",
    "missing_fallback_count", "board_state",
]


class TestCertaintyContractRegistration:
    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_TOTALS)
    def test_totals_feature_columns(self, field):
        assert field in TOTALS_FEATURE_COLUMNS

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_TOTALS)
    def test_totals_field_roles(self, field):
        assert TOTALS_FIELD_ROLES[field] == FIELD_ROLE_CERTAINTY_SIGNAL

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_FIRST5)
    def test_first5_feature_columns(self, field):
        assert field in FIRST5_TOTALS_FEATURE_COLUMNS

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_FIRST5)
    def test_first5_field_roles(self, field):
        assert FIRST5_TOTALS_FIELD_ROLES[field] == FIELD_ROLE_CERTAINTY_SIGNAL

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_HITS)
    def test_hits_feature_columns(self, field):
        assert field in HITS_FEATURE_COLUMNS

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_HITS)
    def test_hits_field_roles(self, field):
        assert HITS_FIELD_ROLES[field] == FIELD_ROLE_CERTAINTY_SIGNAL

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_STRIKEOUTS)
    def test_strikeouts_feature_columns(self, field):
        assert field in STRIKEOUTS_FEATURE_COLUMNS

    @pytest.mark.parametrize("field", CERTAINTY_FIELDS_STRIKEOUTS)
    def test_strikeouts_field_roles(self, field):
        assert STRIKEOUTS_FIELD_ROLES[field] == FIELD_ROLE_CERTAINTY_SIGNAL


class TestCertaintyKeyFields:
    def test_totals_key_fields_in_feature_columns(self):
        for field in TOTALS_CERTAINTY_KEY_FIELDS:
            assert field in TOTALS_FEATURE_COLUMNS, f"{field} not in TOTALS_FEATURE_COLUMNS"

    def test_first5_key_fields_in_feature_columns(self):
        for field in FIRST5_TOTALS_CERTAINTY_KEY_FIELDS:
            assert field in FIRST5_TOTALS_FEATURE_COLUMNS, f"{field} not in FIRST5_TOTALS_FEATURE_COLUMNS"

    def test_hits_key_fields_in_feature_columns(self):
        for field in HITS_CERTAINTY_KEY_FIELDS:
            assert field in HITS_FEATURE_COLUMNS, f"{field} not in HITS_FEATURE_COLUMNS"

    def test_strikeouts_key_fields_in_feature_columns(self):
        for field in STRIKEOUTS_CERTAINTY_KEY_FIELDS:
            assert field in STRIKEOUTS_FEATURE_COLUMNS, f"{field} not in STRIKEOUTS_FEATURE_COLUMNS"
