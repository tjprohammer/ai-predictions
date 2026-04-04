import importlib
import json
import types

from src.utils import doctor as doctor_module


app_module = importlib.import_module("src.api.app")


def test_redact_database_url_masks_password():
    value = app_module._redact_database_url("postgresql+psycopg2://user:secret@localhost:5432/mlb")

    assert value == "postgresql+psycopg2://user:***@localhost:5432/mlb"


def test_doctor_payload_reports_warning_and_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_module,
        "_fetch_status",
        lambda target_date: {
            "target_date": target_date.isoformat(),
            "db_connected": True,
            "totals_artifact_ready": True,
            "hits_artifact_ready": True,
            "strikeouts_artifact_ready": False,
            "totals_predictions": 5,
            "hits_predictions": 2,
            "strikeouts_predictions": 0,
            "rebuild_blocker": None,
        },
    )
    monkeypatch.setattr(app_module, "_fetch_game_readiness_payload", lambda *_args, **_kwargs: {"games": [], "summary": {"green": 0, "yellow": 0, "red": 0, "total": 0}})
    monkeypatch.setattr(
        app_module,
        "_fetch_source_health_payload",
        lambda *_args, **_kwargs: {"sources": [], "summary": {"total_sources": 0, "healthy_sources": 0, "sources_with_failures": 1}},
    )
    monkeypatch.setattr(app_module, "_fetch_pipeline_runs", lambda *_args, **_kwargs: [{"job_id": "run-1", "status": "succeeded"}])
    monkeypatch.setattr(app_module, "_active_update_job_payload", lambda: None)
    monkeypatch.setattr(app_module, "_update_job_history_payload", lambda: [{"job_id": "job-1", "status": "failed"}])
    monkeypatch.setattr(
        app_module,
        "settings",
        types.SimpleNamespace(
            database_url="sqlite:///runtime/db.sqlite3",
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            report_dir=tmp_path / "reports",
            feature_dir=tmp_path / "features",
        ),
    )

    payload = app_module._doctor_payload(app_module.date(2026, 4, 2))

    assert payload["overall"]["status"] == "warn"
    assert payload["runtime"]["database_url"] == "sqlite:///runtime/db.sqlite3"
    assert payload["pipeline_runs"]["runs"][0]["job_id"] == "run-1"


def test_doctor_cli_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor_module.app_module,
        "_doctor_payload",
        lambda *args, **kwargs: {"overall": {"status": "ok"}, "checks": [], "target_date": "2026-04-02"},
    )
    monkeypatch.setattr(
        doctor_module,
        "build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": staticmethod(
                    lambda: type(
                        "Args",
                        (),
                        {
                            "target_date": app_module.date(2026, 4, 2),
                            "source_health_hours": 24,
                            "pipeline_limit": 5,
                            "update_history_limit": 5,
                            "json": True,
                        },
                    )()
                )
            },
        )(),
    )

    result = doctor_module.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["overall"]["status"] == "ok"
