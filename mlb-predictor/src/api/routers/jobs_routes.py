"""Pipeline runs, update jobs, and pipeline run history."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any

import src.api.app_logic as app_logic
from fastapi import APIRouter, Body
from src.api.update_job_sequences import UPDATE_JOB_ACTION_KEYS

router = APIRouter()


@router.get("/api/update-jobs/actions")
def list_update_job_actions() -> app_logic.JSONResponse:
    """Which `action` strings `/api/update-jobs/start` accepts (for debugging stale servers)."""
    return app_logic._json_response({"actions": sorted(UPDATE_JOB_ACTION_KEYS)})


def _resolve_target_date(value: Any) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip()[:10])
    raise TypeError("target_date must be a date or ISO date string")


def _validate_update_job_action(action: str | None) -> tuple[str, None] | tuple[None, dict[str, Any]]:
    """Return (normalized_action, None) or (None, error_payload) for 422."""
    key = (action or "").strip()
    if not key:
        return (
            None,
            {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "action"],
                        "msg": "Field required",
                    }
                ]
            },
        )
    if key not in UPDATE_JOB_ACTION_KEYS:
        return (
            None,
            {
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["body", "action"],
                        "msg": f"Unknown action {key!r}; allowed: {sorted(UPDATE_JOB_ACTION_KEYS)}",
                        "input": action,
                    }
                ]
            },
        )
    return (key, None)


@router.post("/api/pipeline/run")
def run_pipeline(request: app_logic.PipelineRunRequest) -> app_logic.JSONResponse:
    blocker = app_logic._pipeline_blocker(request.target_date)
    if blocker is not None:
        return app_logic._json_response(app_logic._blocked_pipeline_payload(request.target_date, blocker), status_code=409)

    target_date = request.target_date.isoformat()
    steps, failed_step = app_logic._run_module_sequence(
        app_logic._pipeline_sequence(target_date, request.refresh_aggregates, request.rebuild_features)
    )
    if failed_step is not None:
        return app_logic._json_response({"ok": False, "steps": steps}, status_code=500)
    return app_logic._json_response(
        {
            "ok": True,
            "target_date": target_date,
            "steps": steps,
            "status": app_logic._fetch_status(request.target_date),
        }
    )


@router.post("/api/update-jobs/run")
def run_update_job(
    action: Annotated[str, Body()],
    target_date: Annotated[date | None, Body()] = None,
) -> app_logic.JSONResponse:
    action_key, err = _validate_update_job_action(action)
    if err is not None:
        return app_logic._json_response(err, status_code=422)
    assert action_key is not None
    td = _resolve_target_date(target_date)

    blocker = app_logic._action_blocker(action_key, td)
    if blocker is not None:
        return app_logic._json_response(app_logic._blocked_update_payload(action_key, td, blocker), status_code=409)

    target_date_str = td.isoformat()
    steps, failed_step = app_logic._run_module_sequence(app_logic._update_job_sequence(action_key, target_date_str))
    payload = {
        "ok": failed_step is None,
        "action": action_key,
        "label": app_logic._update_job_label(action_key),
        "target_date": target_date_str,
        "steps": steps,
        "status": app_logic._fetch_status(td),
    }
    if failed_step is not None:
        return app_logic._json_response(payload, status_code=500)
    return app_logic._json_response(payload)


@router.post("/api/update-jobs/start")
def start_update_job(
    action: Annotated[str, Body()],
    target_date: Annotated[date | None, Body()] = None,
) -> app_logic.JSONResponse:
    action_key, err = _validate_update_job_action(action)
    if err is not None:
        return app_logic._json_response(err, status_code=422)
    assert action_key is not None
    td = _resolve_target_date(target_date)

    blocker = app_logic._action_blocker(action_key, td)
    if blocker is not None:
        return app_logic._json_response(
            {
                "ok": False,
                "message": blocker["message"],
                "blocker": blocker,
                "status": app_logic._fetch_status(td),
            },
            status_code=409,
        )

    active_job = app_logic._active_update_job_payload()
    if active_job is not None:
        return app_logic._json_response(
            {
                "ok": False,
                "message": "Another update job is already running.",
                "active_job": active_job,
            },
            status_code=409,
        )

    job = app_logic._create_update_job(action_key, td.isoformat())
    app_logic._launch_update_job(job["job_id"])
    created_job = app_logic._get_update_job(job["job_id"])
    return app_logic._json_response({"ok": True, "job": created_job}, status_code=202)


@router.get("/api/update-jobs/active")
def active_update_job() -> app_logic.JSONResponse:
    return app_logic._json_response({"job": app_logic._active_update_job_payload()})


@router.get("/api/update-jobs/history")
def update_job_history() -> app_logic.JSONResponse:
    return app_logic._json_response({"jobs": app_logic._update_job_history_payload()})


@router.get("/api/update-jobs/{job_id}")
def update_job_status(job_id: str) -> app_logic.JSONResponse:
    job = app_logic._get_update_job(job_id)
    if job is None:
        return app_logic._json_response({"job": None}, status_code=404)
    return app_logic._json_response({"job": job})


@router.get("/api/pipeline-runs")
def pipeline_run_history(limit: int = 20) -> app_logic.JSONResponse:
    """Return recent pipeline runs from the DB (not in-memory)."""
    return app_logic._json_response({"runs": app_logic._fetch_pipeline_runs(limit)})


@router.get("/api/pipeline-runs/{job_id}/steps")
def pipeline_run_steps(job_id: str) -> app_logic.JSONResponse:
    """Return steps for a specific pipeline run from the DB."""
    if not app_logic._table_exists("pipeline_run_steps"):
        return app_logic._json_response({"steps": []})
    steps_frame = app_logic._safe_frame(
        """
        SELECT step_index, module_name, returncode, stdout, stderr,
               started_at, finished_at
        FROM pipeline_run_steps
        WHERE job_id = :job_id
        ORDER BY step_index
        """,
        {"job_id": job_id},
    )
    return app_logic._json_response({"steps": app_logic._frame_records(steps_frame)})
