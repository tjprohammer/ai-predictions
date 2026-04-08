import importlib

import numpy as np
import pandas as pd


common_module = importlib.import_module("src.models.common")


class _NegativeWeightCalibrationRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = np.array([-0.25, 0.9])
        self.intercept_ = 0.1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 5.0)


class _UnusedPredictor:
    def predict(self, X):
        raise AssertionError("predict() should not be called when calibration is gated off")


def test_fit_market_calibrator_discards_negative_raw_model_weight(monkeypatch):
    monkeypatch.setattr(common_module, "_CalibrationRidge", _NegativeWeightCalibrationRidge)

    calibrator = common_module.fit_market_calibrator(
        np.array([4.0, 5.0, 6.0]),
        pd.Series([4.5, 5.5, 6.5]),
        pd.Series([4.2, 5.1, 6.3]),
        min_rows=1,
    )

    assert calibrator is None


def test_calibrate_with_market_skips_negative_weight_calibrator():
    raw_predictions = np.array([4.0, 6.0])
    market_values = pd.Series([4.5, 6.5])

    calibrated, mask = common_module.calibrate_with_market(
        raw_predictions,
        market_values,
        {
            "model_weight": -0.1,
            "calibrator": _UnusedPredictor(),
        },
    )

    assert np.array_equal(calibrated, raw_predictions)
    assert not mask.any()