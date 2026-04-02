import json
import importlib


app_module = importlib.import_module("src.api.app")


def _reset_update_jobs():
    with app_module.UPDATE_JOB_LOCK:
        app_module.UPDATE_JOBS.clear()
    if app_module.UPDATE_JOB_STORE_PATH.exists():
        app_module.UPDATE_JOB_STORE_PATH.unlink()


def _install_success_stubs(monkeypatch):
    _reset_update_jobs()
    calls: list[tuple[str, list[str]]] = []

    def fake_run_module(module_name: str, *args: str):
        calls.append((module_name, list(args)))
        return {
            "module": module_name,
            "command": ["python", "-m", module_name, *args],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(app_module, "_run_module", fake_run_module)
    monkeypatch.setattr(
        app_module,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "ok": True},
    )
    return calls


def test_update_job_prepare_slate_runs_expected_modules(monkeypatch):
    calls = _install_success_stubs(monkeypatch)

    response = app_module.run_update_job(
        app_module.UpdateJobRunRequest(
            action="prepare_slate",
            target_date="2026-04-02",
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["label"] == "Prepare slate"
    assert calls == [
        ("src.ingestors.games", ["--target-date", "2026-04-02"]),
        ("src.ingestors.starters", ["--target-date", "2026-04-02"]),
        ("src.ingestors.prepare_slate_inputs", ["--target-date", "2026-04-02"]),
    ]


def test_update_job_stops_after_failed_step(monkeypatch):
    _reset_update_jobs()
    calls: list[str] = []

    def fake_run_module(module_name: str, *args: str):
        calls.append(module_name)
        return {
            "module": module_name,
            "command": ["python", "-m", module_name, *args],
            "returncode": 1 if module_name == "src.ingestors.market_totals" else 0,
            "stdout": "",
            "stderr": "boom" if module_name == "src.ingestors.market_totals" else "",
        }

    monkeypatch.setattr(app_module, "_run_module", fake_run_module)
    monkeypatch.setattr(
        app_module,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "ok": True},
    )

    response = app_module.run_update_job(
        app_module.UpdateJobRunRequest(
            action="import_manual_inputs",
            target_date="2026-04-02",
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload["ok"] is False
    assert calls == ["src.ingestors.lineups", "src.ingestors.market_totals"]


def test_legacy_pipeline_includes_first5_lane(monkeypatch):
    calls = _install_success_stubs(monkeypatch)

    response = app_module.run_pipeline(
        app_module.PipelineRunRequest(
            target_date="2026-04-02",
            refresh_aggregates=True,
            rebuild_features=True,
        )
    )

    assert response.status_code == 200
    assert [module for module, _ in calls] == [
        "src.transforms.offense_daily",
        "src.transforms.bullpens_daily",
        "src.features.totals_builder",
        "src.features.first5_totals_builder",
        "src.features.hits_builder",
        "src.features.strikeouts_builder",
        "src.models.predict_totals",
        "src.models.predict_first5_totals",
        "src.models.predict_hits",
        "src.models.predict_strikeouts",
        "src.transforms.product_surfaces",
    ]


def test_start_update_job_returns_queued_job(monkeypatch):
    _reset_update_jobs()
    launched: list[str] = []

    monkeypatch.setattr(app_module, "_launch_update_job", lambda job_id: launched.append(job_id))

    response = app_module.start_update_job(
        app_module.UpdateJobRunRequest(
            action="prepare_slate",
            target_date="2026-04-02",
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 202
    assert payload["ok"] is True
    assert payload["job"]["status"] == "queued"
    assert payload["job"]["total_steps"] == 3
    assert launched == [payload["job"]["job_id"]]


def test_start_update_job_rejects_when_another_job_is_active(monkeypatch):
    _reset_update_jobs()
    monkeypatch.setattr(app_module, "_launch_update_job", lambda job_id: None)

    first_response = app_module.start_update_job(
        app_module.UpdateJobRunRequest(
            action="prepare_slate",
            target_date="2026-04-02",
        )
    )
    first_payload = json.loads(first_response.body)

    second_response = app_module.start_update_job(
        app_module.UpdateJobRunRequest(
            action="refresh_results",
            target_date="2026-04-02",
        )
    )
    second_payload = json.loads(second_response.body)

    assert second_response.status_code == 409
    assert second_payload["active_job"]["job_id"] == first_payload["job"]["job_id"]


def test_background_update_job_marks_success(monkeypatch):
    calls = _install_success_stubs(monkeypatch)
    job = app_module._create_update_job("prepare_slate", "2026-04-02")

    app_module._run_update_job_background(job["job_id"])
    stored_job = app_module._get_update_job(job["job_id"])

    assert stored_job is not None
    assert stored_job["status"] == "succeeded"
    assert stored_job["completed_steps"] == stored_job["total_steps"] == 3
    assert stored_job["status_snapshot"]["ok"] is True
    assert [module for module, _ in calls] == [
        "src.ingestors.games",
        "src.ingestors.starters",
        "src.ingestors.prepare_slate_inputs",
    ]


def test_update_job_history_persists_to_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "UPDATE_JOB_STORE_PATH", tmp_path / "update_jobs.json")
    calls = _install_success_stubs(monkeypatch)

    job = app_module._create_update_job("prepare_slate", "2026-04-02")
    app_module._run_update_job_background(job["job_id"])

    stored_payload = json.loads(app_module.UPDATE_JOB_STORE_PATH.read_text(encoding="utf-8"))

    assert calls
    assert stored_payload[0]["job_id"] == job["job_id"]
    assert stored_payload[0]["status"] == "succeeded"


def test_load_persisted_update_jobs_recovers_history_and_marks_interrupted_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "UPDATE_JOB_STORE_PATH", tmp_path / "update_jobs.json")
    _reset_update_jobs()
    persisted_jobs = [
        {
            "job_id": "finished-job",
            "action": "prepare_slate",
            "label": "Prepare slate",
            "target_date": "2026-04-02",
            "status": "succeeded",
            "created_at": "2026-04-02T10:00:00+00:00",
            "started_at": "2026-04-02T10:00:01+00:00",
            "finished_at": "2026-04-02T10:00:03+00:00",
            "current_step": None,
            "completed_steps": 3,
            "total_steps": 3,
            "steps": [],
            "error": None,
            "status_snapshot": {"ok": True},
        },
        {
            "job_id": "interrupted-job",
            "action": "refresh_results",
            "label": "Refresh results and stats",
            "target_date": "2026-04-03",
            "status": "running",
            "created_at": "2026-04-02T11:00:00+00:00",
            "started_at": "2026-04-02T11:00:01+00:00",
            "finished_at": None,
            "current_step": "src.ingestors.boxscores",
            "completed_steps": 1,
            "total_steps": 5,
            "steps": [],
            "error": None,
            "status_snapshot": None,
        },
    ]
    app_module.UPDATE_JOB_STORE_PATH.write_text(
        json.dumps(persisted_jobs, indent=2),
        encoding="utf-8",
    )

    app_module._load_persisted_update_jobs()
    history = app_module._update_job_history_payload()

    assert len(history) == 2
    assert history[0]["job_id"] == "interrupted-job"
    assert history[0]["status"] == "failed"
    assert "restarted" in str(history[0]["error"]).lower()
    assert history[1]["job_id"] == "finished-job"
    assert app_module._active_update_job_payload() is None