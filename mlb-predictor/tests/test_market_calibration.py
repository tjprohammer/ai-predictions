import numpy as np
import pandas as pd
import pytest

from src.models.common import calibrate_with_market, fit_market_calibrator


def _make_data(n=50, market_coverage=1.0, seed=42):
    rng = np.random.RandomState(seed)
    actuals = rng.normal(8.5, 2.5, n)
    raw_predictions = actuals + rng.normal(0, 1.5, n)
    market_values = actuals + rng.normal(0, 0.8, n)
    market_series = pd.Series(market_values)
    # mask out some market values
    n_missing = int(n * (1 - market_coverage))
    if n_missing > 0:
        missing_idx = rng.choice(n, n_missing, replace=False)
        market_series.iloc[missing_idx] = np.nan
    return raw_predictions, market_series, pd.Series(actuals)


class TestFitMarketCalibrator:
    def test_returns_calibrator_with_enough_data(self):
        raw, market, actuals = _make_data(n=50)
        result = fit_market_calibrator(raw, market, actuals)
        assert result is not None
        assert "calibrator" in result
        assert "model_weight" in result
        assert "market_weight" in result
        assert result["calibration_rows"] == 50

    def test_returns_none_below_min_rows(self):
        raw, market, actuals = _make_data(n=10)
        result = fit_market_calibrator(raw, market, actuals, min_rows=20)
        assert result is None

    def test_returns_none_when_market_all_nan(self):
        raw, _, actuals = _make_data(n=50)
        market = pd.Series([np.nan] * 50)
        result = fit_market_calibrator(raw, market, actuals)
        assert result is None

    def test_calibration_residual_std_is_positive(self):
        raw, market, actuals = _make_data(n=50)
        result = fit_market_calibrator(raw, market, actuals)
        assert result["calibration_residual_std"] > 0

    def test_calibrator_reduces_error(self):
        raw, market, actuals = _make_data(n=100)
        result = fit_market_calibrator(raw, market, actuals)
        calibrated = result["calibrator"].predict(
            np.column_stack([raw, market.astype(float).values])
        )
        raw_mae = np.abs(actuals.values - raw).mean()
        cal_mae = np.abs(actuals.values - calibrated).mean()
        assert cal_mae <= raw_mae

    def test_partial_market_coverage(self):
        raw, market, actuals = _make_data(n=60, market_coverage=0.5)
        result = fit_market_calibrator(raw, market, actuals)
        assert result is not None
        assert result["calibration_rows"] == 30


class TestCalibrateWithMarket:
    def test_returns_raw_when_no_calibrator(self):
        raw = np.array([7.0, 8.0, 9.0])
        market = pd.Series([8.5, 9.0, np.nan])
        calibrated, mask = calibrate_with_market(raw, market, None)
        np.testing.assert_array_equal(calibrated, raw)
        assert not mask.any()

    def test_applies_calibration_where_market_available(self):
        raw_train, market_train, actuals_train = _make_data(n=50)
        cal_info = fit_market_calibrator(raw_train, market_train, actuals_train)

        raw = np.array([7.0, 8.0, 9.0])
        market = pd.Series([8.5, np.nan, 9.0])
        calibrated, mask = calibrate_with_market(raw, market, cal_info)

        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.True_
        # Row without market keeps raw value
        assert calibrated[1] == 8.0
        # Calibrated rows should differ from raw
        assert calibrated[0] != 7.0 or calibrated[2] != 9.0

    def test_all_nan_market_returns_raw(self):
        raw_train, market_train, actuals_train = _make_data(n=50)
        cal_info = fit_market_calibrator(raw_train, market_train, actuals_train)

        raw = np.array([7.0, 8.0])
        market = pd.Series([np.nan, np.nan])
        calibrated, mask = calibrate_with_market(raw, market, cal_info)
        np.testing.assert_array_equal(calibrated, raw)
        assert not mask.any()

    def test_does_not_mutate_input(self):
        raw_train, market_train, actuals_train = _make_data(n=50)
        cal_info = fit_market_calibrator(raw_train, market_train, actuals_train)

        raw = np.array([7.0, 8.0, 9.0])
        raw_copy = raw.copy()
        market = pd.Series([8.5, 8.5, 8.5])
        calibrate_with_market(raw, market, cal_info)
        np.testing.assert_array_equal(raw, raw_copy)
