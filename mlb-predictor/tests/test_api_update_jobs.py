import json
import importlib
import types
from datetime import date

import pytest

from src.api.update_job_sequences import build_pipeline_run_sequence, build_update_job_sequence


app_module = importlib.import_module("src.api.app")
app_logic = importlib.import_module("src.api.app_logic")


def _patch_pair(monkeypatch, name: str, value: object) -> None:
    """Patch callables used from ``app_logic`` helpers (their globals stay on app_logic)."""
    monkeypatch.setattr(app_module, name, value)
    monkeypatch.setattr(app_logic, name, value)


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

    _patch_pair(monkeypatch, "_run_module", fake_run_module)
    _patch_pair(
        monkeypatch,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "ok": True},
    )
    return calls


def test_update_job_prepare_slate_runs_expected_modules(monkeypatch):
    calls = _install_success_stubs(monkeypatch)

    response = app_module.run_update_job(
        action="prepare_slate",
        target_date=date(2026, 4, 2),
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["label"] == "Prepare Slate"
    assert calls == build_update_job_sequence("prepare_slate", "2026-04-02")


def test_update_job_refresh_everything_runs_full_sequence(monkeypatch):
    calls = _install_success_stubs(monkeypatch)

    response = app_module.run_update_job(
        action="refresh_everything",
        target_date=date(2026, 4, 2),
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["label"] == "Refresh Everything"
    assert calls == build_update_job_sequence("refresh_everything", "2026-04-02")


def test_update_lineups_only_skips_market_totals_odds_ingest():
    modules = [m for m, _ in build_update_job_sequence("update_lineups_only", "2026-04-02")]
    assert "src.ingestors.lineups" in modules
    assert "src.ingestors.market_totals" not in modules


def test_update_markets_only_runs_market_totals_not_lineups():
    modules = [m for m, _ in build_update_job_sequence("update_markets_only", "2026-04-02")]
    assert "src.ingestors.market_totals" in modules
    assert "src.ingestors.lineups" not in modules


@pytest.mark.parametrize(
    ("action", "target_date", "expected_label"),
    [
        ("refresh_everything", date(2026, 4, 2), "Refresh Everything"),
        ("prepare_slate", date(2026, 4, 2), "Prepare Slate"),
        ("import_manual_inputs", date(2026, 4, 2), "Update Lineups & Markets"),
        ("update_lineups_only", date(2026, 4, 2), "Update Lineups"),
        ("update_markets_only", date(2026, 4, 2), "Update Markets"),
        ("refresh_results", date(2026, 4, 2), "Refresh Daily Results"),
        ("rebuild_predictions", date(2026, 4, 2), "Rebuild Predictions"),
        ("grade_predictions", date(2026, 4, 3), "Grade Predictions"),
    ],
)
def test_update_job_sequences_cover_every_dashboard_action(
    monkeypatch,
    action,
    target_date,
    expected_label,
):
    _patch_pair(monkeypatch, "_action_blocker", lambda a, d: None)
    calls = _install_success_stubs(monkeypatch)
    expected_modules = [m for m, _ in build_update_job_sequence(action, target_date.isoformat())]

    response = app_module.run_update_job(
        action=action,
        target_date=target_date,
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["label"] == expected_label
    assert [module for module, _ in calls] == expected_modules


def test_update_job_grade_predictions_targets_yesterday(monkeypatch):
    calls = _install_success_stubs(monkeypatch)

    response = app_module.run_update_job(
        action="grade_predictions",
        target_date=date(2026, 4, 3),
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["label"] == "Grade Predictions"
    assert [module for module, _ in calls] == [
        m for m, _ in build_update_job_sequence("grade_predictions", "2026-04-03")
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

    _patch_pair(monkeypatch, "_run_module", fake_run_module)
    _patch_pair(
        monkeypatch,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "ok": True},
    )

    response = app_module.run_update_job(
        action="import_manual_inputs",
        target_date=date(2026, 4, 2),
    )
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload["ok"] is False
    assert calls == [
        "src.ingestors.lineups",
        "src.ingestors.player_status",
        "src.ingestors.market_totals",
    ]


def test_legacy_pipeline_includes_first5_lane(monkeypatch):
    _patch_pair(monkeypatch, "_pipeline_blocker", lambda target_date: None)
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
        m
        for m, _ in build_pipeline_run_sequence(
            "2026-04-02",
            refresh_aggregates=True,
            rebuild_features=True,
        )
    ]


def test_start_update_job_returns_queued_job(monkeypatch):
    _reset_update_jobs()
    launched: list[str] = []

    _patch_pair(monkeypatch, "_launch_update_job", lambda job_id: launched.append(job_id))

    response = app_module.start_update_job(
        action="prepare_slate",
        target_date=date(2026, 4, 2),
    )
    payload = json.loads(response.body)

    assert response.status_code == 202
    assert payload["ok"] is True
    assert payload["job"]["status"] == "queued"
    assert payload["job"]["total_steps"] == len(build_update_job_sequence("prepare_slate", "2026-04-02"))
    assert launched == [payload["job"]["job_id"]]


def test_start_update_job_rejects_when_another_job_is_active(monkeypatch):
    _reset_update_jobs()
    _patch_pair(monkeypatch, "_launch_update_job", lambda job_id: None)

    first_response = app_module.start_update_job(
        action="prepare_slate",
        target_date=date(2026, 4, 2),
    )
    first_payload = json.loads(first_response.body)

    second_response = app_module.start_update_job(
        action="refresh_results",
        target_date=date(2026, 4, 2),
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
    slate = build_update_job_sequence("prepare_slate", "2026-04-02")
    assert stored_job["completed_steps"] == stored_job["total_steps"] == len(slate)
    assert stored_job["status_snapshot"]["ok"] is True
    assert calls == slate


def test_start_update_job_blocks_rebuild_predictions_when_desktop_history_missing(monkeypatch):
    _reset_update_jobs()
    blocker = {"message": "Desktop historical data is incomplete."}

    _patch_pair(
        monkeypatch,
        "_action_blocker",
        lambda action, target_date: blocker if action == "rebuild_predictions" else None,
    )
    _patch_pair(
        monkeypatch,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "rebuild_blocker": blocker},
    )

    response = app_module.start_update_job(
        action="rebuild_predictions",
        target_date=date(2026, 4, 2),
    )
    payload = json.loads(response.body)

    assert response.status_code == 409
    assert payload["ok"] is False
    assert payload["message"] == blocker["message"]
    assert payload["blocker"] == blocker


def test_action_blocker_allows_refresh_everything():
    """refresh_everything is never blocked — it populates the history tables."""
    from datetime import date

    blocker = app_module._action_blocker("refresh_everything", date(2026, 4, 2))
    assert blocker is None



def test_update_job_history_persists_to_disk(monkeypatch, tmp_path):
    store = tmp_path / "update_jobs.json"
    _patch_pair(monkeypatch, "UPDATE_JOB_STORE_PATH", store)
    calls = _install_success_stubs(monkeypatch)

    job = app_module._create_update_job("prepare_slate", "2026-04-02")
    app_module._run_update_job_background(job["job_id"])

    stored_payload = json.loads(app_module.UPDATE_JOB_STORE_PATH.read_text(encoding="utf-8"))

    assert calls
    assert stored_payload[0]["job_id"] == job["job_id"]
    assert stored_payload[0]["status"] == "succeeded"


def test_load_persisted_update_jobs_recovers_history_and_marks_interrupted_failed(monkeypatch, tmp_path):
    store = tmp_path / "update_jobs.json"
    _patch_pair(monkeypatch, "UPDATE_JOB_STORE_PATH", store)
    _reset_update_jobs()
    persisted_jobs = [
        {
            "job_id": "finished-job",
            "action": "prepare_slate",
            "label": "Prepare Slate",
            "target_date": "2026-04-02",
            "status": "succeeded",
            "created_at": "2026-04-02T10:00:00+00:00",
            "started_at": "2026-04-02T10:00:01+00:00",
            "finished_at": "2026-04-02T10:00:03+00:00",
            "current_step": None,
            "completed_steps": 15,
            "total_steps": 15,
            "steps": [],
            "error": None,
            "status_snapshot": {"ok": True},
        },
        {
            "job_id": "interrupted-job",
            "action": "refresh_results",
            "label": "Refresh Daily Results",
            "target_date": "2026-04-03",
            "status": "running",
            "created_at": "2026-04-02T11:00:00+00:00",
            "started_at": "2026-04-02T11:00:01+00:00",
            "finished_at": None,
            "current_step": "src.ingestors.boxscores",
            "completed_steps": 1,
            "total_steps": 6,
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


def test_run_module_uses_in_process_runner_when_frozen(monkeypatch):
    events: list[tuple[str, list[str]]] = []

    def fake_main():
        import sys

        events.append(("argv", list(sys.argv)))
        print("stdout ok")
        return 0

    fake_module = types.SimpleNamespace(main=fake_main)

    monkeypatch.setattr(app_logic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_logic.importlib, "import_module", lambda name: fake_module)

    result = app_module._run_module("fake.module", "--target-date", "2026-04-02")

    assert result["returncode"] == 0
    assert result["stdout"] == "stdout ok"
    assert result["stderr"] == ""
    assert events == [("argv", ["fake.module", "--target-date", "2026-04-02"])]


def test_background_update_job_failure_does_not_deadlock(monkeypatch):
    """Regression: _persist_update_jobs inside UPDATE_JOB_LOCK caused deadlock on failure."""
    _reset_update_jobs()

    def fake_run_module(module_name: str, *args: str):
        return {
            "module": module_name,
            "command": ["python", "-m", module_name, *args],
            "returncode": 1,
            "stdout": "",
            "stderr": "simulated crash",
        }

    _patch_pair(monkeypatch, "_run_module", fake_run_module)
    _patch_pair(
        monkeypatch,
        "_fetch_status",
        lambda target_date: {"target_date": target_date.isoformat(), "ok": True},
    )

    job = app_module._create_update_job("prepare_slate", "2026-04-02")
    app_module._run_update_job_background(job["job_id"])

    # If we reach here the lock was released (no deadlock)
    stored_job = app_module._get_update_job(job["job_id"])
    assert stored_job is not None
    assert stored_job["status"] == "failed"
    assert stored_job["completed_steps"] == 1
    # Lock must be free — acquire/release proves no deadlock
    assert app_module.UPDATE_JOB_LOCK.acquire(timeout=1)
    app_module.UPDATE_JOB_LOCK.release()


def test_run_module_in_process_reports_exceptions(monkeypatch):
    def fake_main():
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(main=fake_main)

    monkeypatch.setattr(app_logic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_logic.importlib, "import_module", lambda name: fake_module)

    result = app_module._run_module("fake.module")

    assert result["returncode"] == 1
    assert "RuntimeError: boom" in result["stderr"]