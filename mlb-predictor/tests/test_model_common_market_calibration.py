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


def test_mean_pinball_median_is_half_mae_for_tau_half():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.2])
    pin = common_module.mean_pinball_loss(y, p, tau=0.5)
    mae = float(np.mean(np.abs(y - p)))
    assert abs(pin - 0.5 * mae) < 1e-9


def test_log_loss_ou_vs_market_line_returns_metrics_when_enough_rows():
    y = np.array([9.0, 6.0, 10.0])
    pred = np.array([8.5, 6.5, 9.5])
    mkt = pd.Series([8.0, 7.5, 9.0])
    out = common_module.log_loss_ou_vs_market_line(y, pred, mkt, std=1.0, min_rows=3)
    assert out is not None
    assert out["n"] == 3.0
    assert 0.0 <= out["log_loss"] <= 2.0