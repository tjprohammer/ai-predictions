from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.models import common


def test_save_artifact_writes_metadata_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_settings", lambda: SimpleNamespace(model_dir=tmp_path))

    artifact = {
        "model_name": "ridge",
        "model_version": "totals_test",
        "trained_at": datetime(2026, 4, 4, tzinfo=timezone.utc),
        "model": object(),
    }

    artifact_path = common.save_artifact("totals", "totals_test", artifact)
    metadata_path = artifact_path.with_suffix(".meta.json")

    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["lane"] == "totals"
    assert metadata["model_version"] == "totals_test"
    assert metadata["sklearn_version"] == common._current_sklearn_version()


def test_load_latest_artifact_skips_incompatible_newer_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_settings", lambda: SimpleNamespace(model_dir=tmp_path))
    lane_dir = tmp_path / "hits"
    lane_dir.mkdir(parents=True, exist_ok=True)

    compatible_artifact = {"model_name": "logistic", "model_version": "compatible", "model": "ok"}
    compatible_path = lane_dir / "compatible.pkl"
    with compatible_path.open("wb") as handle:
        pickle.dump(compatible_artifact, handle)
    compatible_path.with_suffix(".meta.json").write_text(
        json.dumps({"sklearn_version": common._current_sklearn_version()}),
        encoding="utf-8",
    )

    incompatible_artifact = {"model_name": "logistic", "model_version": "stale", "model": "stale"}
    incompatible_path = lane_dir / "stale.pkl"
    with incompatible_path.open("wb") as handle:
        pickle.dump(incompatible_artifact, handle)
    incompatible_path.with_suffix(".meta.json").write_text(
        json.dumps({"sklearn_version": "0.0"}),
        encoding="utf-8",
    )

    loaded = common.load_latest_artifact("hits")

    assert loaded["model_version"] == "compatible"


def test_load_latest_artifact_rejects_legacy_artifacts_without_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_settings", lambda: SimpleNamespace(model_dir=tmp_path))
    lane_dir = tmp_path / "strikeouts"
    lane_dir.mkdir(parents=True, exist_ok=True)

    legacy_path = lane_dir / "legacy.pkl"
    with legacy_path.open("wb") as handle:
        pickle.dump({"model_version": "legacy", "model": "old"}, handle)

    with pytest.raises(common.ArtifactRuntimeMismatchError):
        common.load_latest_artifact("strikeouts")