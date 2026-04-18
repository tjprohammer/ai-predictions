from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models import predict_first5_totals, predict_hits, predict_hr, predict_strikeouts, predict_totals


def _capture_delete(monkeypatch, module):
    captured: dict[str, object] = {}

    def fake_run_sql(query, params=None, *args, **kwargs):
        captured["query"] = query
        captured["params"] = params

    monkeypatch.setattr(module, "run_sql", fake_run_sql)
    return captured


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
    delete_capture = _capture_delete(monkeypatch, predict_hits)
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
    assert "model_version" not in str(delete_capture["query"])
    assert delete_capture["params"] == {"target_date": pd.Timestamp(target_date).date()}


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
    delete_capture = _capture_delete(monkeypatch, predict_strikeouts)
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
    assert "model_version" not in str(delete_capture["query"])
    assert delete_capture["params"] == {"target_date": pd.Timestamp(target_date).date()}


def test_predict_totals_replaces_selected_date_across_versions(monkeypatch, tmp_path):
    target_date = "2026-04-03"
    monkeypatch.setattr(predict_totals.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(target_date=target_date))
    monkeypatch.setattr(predict_totals, "get_settings", lambda: SimpleNamespace(report_dir=tmp_path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)
    monkeypatch.setattr(
        predict_totals,
        "load_feature_snapshots",
        lambda lane: pd.DataFrame([
            {"game_date": target_date, "game_id": 1, "feature_a": 0.4, "market_total": 8.5, "market_sportsbook": "FD", "line_snapshot_ts": None}
        ]),
    )
    monkeypatch.setattr(predict_totals, "encode_frame", lambda frame, category_columns, training_columns: frame)
    delete_capture = _capture_delete(monkeypatch, predict_totals)
    saved_rows: list[dict] = []
    monkeypatch.setattr(predict_totals, "upsert_rows", lambda table, rows, keys: saved_rows.extend(rows))

    class WorkingModel:
        def predict(self, X):
            return np.array([8.9])

    artifact = {
        "feature_columns": ["feature_a"],
        "category_columns": [],
        "training_columns": ["feature_a"],
        "model": WorkingModel(),
        "model_name": "gbr",
        "model_version": "totals_test",
        "residual_std": 1.0,
    }

    monkeypatch.setattr(predict_totals, "_load_or_train_artifact", lambda: artifact)

    assert predict_totals.main() == 0
    assert len(saved_rows) == 1
    assert saved_rows[0]["model_version"] == "totals_test"
    assert "model_version" not in str(delete_capture["query"])
    assert delete_capture["params"] == {"target_date": pd.Timestamp(target_date).date()}


def test_predict_first5_totals_replaces_selected_date_across_versions(monkeypatch, tmp_path):
    target_date = "2026-04-03"
    monkeypatch.setattr(predict_first5_totals.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(target_date=target_date))
    monkeypatch.setattr(predict_first5_totals, "get_settings", lambda: SimpleNamespace(report_dir=tmp_path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)
    monkeypatch.setattr(
        predict_first5_totals,
        "load_feature_snapshots",
        lambda lane: pd.DataFrame([
            {"game_date": target_date, "game_id": 1, "feature_a": 0.4, "market_total": 4.5, "market_sportsbook": "FD", "line_snapshot_ts": None}
        ]),
    )
    monkeypatch.setattr(predict_first5_totals, "encode_frame", lambda frame, category_columns, training_columns: frame)
    delete_capture = _capture_delete(monkeypatch, predict_first5_totals)
    saved_rows: list[dict] = []
    monkeypatch.setattr(predict_first5_totals, "upsert_rows", lambda table, rows, keys: saved_rows.extend(rows))

    class WorkingModel:
        def predict(self, X):
            return np.array([4.9])

    artifact = {
        "feature_columns": ["feature_a"],
        "category_columns": [],
        "training_columns": ["feature_a"],
        "model": WorkingModel(),
        "model_name": "gbr",
        "model_version": "first5_test",
        "residual_std": 1.0,
    }

    monkeypatch.setattr(predict_first5_totals, "_load_or_train_artifact", lambda: artifact)

    assert predict_first5_totals.main() == 0
    assert len(saved_rows) == 1
    assert saved_rows[0]["model_version"] == "first5_test"
    assert "model_version" not in str(delete_capture["query"])
    assert delete_capture["params"] == {"target_date": pd.Timestamp(target_date).date()}


def test_predict_hr_exits_zero_when_training_raises_after_missing_artifact(monkeypatch):
    monkeypatch.setattr(predict_hr.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(target_date=None))
    load_calls: list[str] = []

    def load_side_effect(lane: str):
        load_calls.append(lane)
        raise FileNotFoundError(f"No model artifacts found for {lane}")

    monkeypatch.setattr(predict_hr, "load_latest_artifact", load_side_effect)

    def train_raises(argv: list[str]) -> int:
        raise RuntimeError("training unavailable")

    monkeypatch.setattr("src.models.train_hr.main", train_raises)

    assert predict_hr.main() == 0
    assert load_calls == ["hr", "hr"]


def test_predict_strikeouts_market_expectation_skips_opener_like_rows():
    opener_row = SimpleNamespace(
        projected_innings=1.0,
        recent_avg_ip_3=1.0,
        recent_avg_ip_5=1.0,
        baseline_strikeouts=1.0,
    )
    starter_row = SimpleNamespace(
        projected_innings=5.4,
        recent_avg_ip_3=5.7,
        recent_avg_ip_5=5.4,
        baseline_strikeouts=5.8,
    )

    assert predict_strikeouts._expects_market_line(opener_row) is False
    assert predict_strikeouts._expects_market_line(starter_row) is True