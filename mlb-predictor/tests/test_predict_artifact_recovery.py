from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models import predict_hits, predict_strikeouts


def test_predict_hits_retries_with_retrained_artifact(monkeypatch, tmp_path):
    target_date = "2026-04-03"
    monkeypatch.setattr(predict_hits.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(target_date=target_date))
    monkeypatch.setattr(predict_hits, "get_settings", lambda: SimpleNamespace(report_dir=tmp_path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)
    monkeypatch.setattr(
        predict_hits,
        "load_feature_snapshots",
        lambda lane: pd.DataFrame([
            {"game_date": target_date, "game_id": 1, "player_id": 2, "team": "SEA", "feature_a": 0.4}
        ]),
    )
    monkeypatch.setattr(predict_hits, "encode_frame", lambda frame, category_columns, training_columns: frame)
    monkeypatch.setattr(predict_hits, "_fetch_market_map", lambda target_date: {(1, 2): -110})
    monkeypatch.setattr(predict_hits, "run_sql", lambda *args, **kwargs: None)
    saved_rows: list[dict] = []
    monkeypatch.setattr(predict_hits, "upsert_rows", lambda table, rows, keys: saved_rows.extend(rows))

    class FailingModel:
        def predict_proba(self, X):
            raise AttributeError("stale artifact")

    class WorkingModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    failing_artifact = {
        "feature_columns": ["feature_a"],
        "category_columns": [],
        "training_columns": ["feature_a"],
        "model": FailingModel(),
    }
    working_artifact = failing_artifact | {
        "model": WorkingModel(),
        "model_name": "logistic",
        "model_version": "hits_test",
        "calibration_method": "identity",
        "calibrator": None,
    }

    reload_calls: list[str] = []
    monkeypatch.setattr(predict_hits, "_load_or_train_artifact", lambda: failing_artifact)
    monkeypatch.setattr(
        predict_hits,
        "_reload_artifact_after_failure",
        lambda exc: reload_calls.append(str(exc)) or working_artifact,
    )

    assert predict_hits.main() == 0
    assert reload_calls == ["stale artifact"]
    assert len(saved_rows) == 1
    assert saved_rows[0]["model_version"] == "hits_test"


def test_predict_strikeouts_retries_with_retrained_artifact(monkeypatch, tmp_path):
    target_date = "2026-04-03"
    monkeypatch.setattr(predict_strikeouts.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(target_date=target_date))
    monkeypatch.setattr(predict_strikeouts, "get_settings", lambda: SimpleNamespace(report_dir=tmp_path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)
    monkeypatch.setattr(
        predict_strikeouts,
        "load_feature_snapshots",
        lambda lane: pd.DataFrame([
            {"game_date": target_date, "game_id": 1, "pitcher_id": 3, "team": "SEA", "feature_a": 0.2}
        ]),
    )
    monkeypatch.setattr(predict_strikeouts, "encode_frame", lambda frame, category_columns, training_columns: frame)
    monkeypatch.setattr(predict_strikeouts, "_fetch_market_map", lambda target_date: {(1, 3): 5.5})
    monkeypatch.setattr(predict_strikeouts, "run_sql", lambda *args, **kwargs: None)
    saved_rows: list[dict] = []
    monkeypatch.setattr(predict_strikeouts, "upsert_rows", lambda table, rows, keys: saved_rows.extend(rows))

    class FailingModel:
        def predict(self, X):
            raise AttributeError("stale artifact")

    class WorkingModel:
        def predict(self, X):
            return np.array([6.2])

    failing_artifact = {
        "feature_columns": ["feature_a"],
        "category_columns": [],
        "training_columns": ["feature_a"],
        "model": FailingModel(),
        "residual_std": 1.0,
    }
    working_artifact = failing_artifact | {"model": WorkingModel(), "model_name": "ridge", "model_version": "strikeouts_test"}

    reload_calls: list[str] = []
    monkeypatch.setattr(predict_strikeouts, "_load_or_train_artifact", lambda: failing_artifact)
    monkeypatch.setattr(
        predict_strikeouts,
        "_reload_artifact_after_failure",
        lambda exc: reload_calls.append(str(exc)) or working_artifact,
    )

    assert predict_strikeouts.main() == 0
    assert reload_calls == ["stale artifact"]
    assert len(saved_rows) == 1
    assert saved_rows[0]["model_version"] == "strikeouts_test"