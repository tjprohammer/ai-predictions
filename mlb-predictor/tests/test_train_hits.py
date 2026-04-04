from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import train_hits


def test_build_logistic_classifier_scales_features_before_fit():
    model = train_hits._build_logistic_classifier()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["standardscaler"], StandardScaler)
    assert isinstance(model.named_steps["logisticregression"], LogisticRegression)
    assert model.named_steps["logisticregression"].max_iter == train_hits.LOGISTIC_MAX_ITER


def test_sigmoid_calibrator_uses_extended_iteration_budget():
    calibrator = train_hits._fit_sigmoid_calibrator(
        train_hits.np.array([0.1, 0.2, 0.8, 0.9]),
        [0, 0, 1, 1],
    )

    assert calibrator is not None
    assert calibrator.max_iter == train_hits.LOGISTIC_MAX_ITER