"""Liveness and lightweight status JSON (/health, /api/status, /api/doctor)."""
from __future__ import annotations

import src.api.app_logic as app_logic
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
@router.get("/api/health")
def health() -> app_logic.JSONResponse:
    return app_logic._json_response(app_logic._fetch_status(app_logic.date.today()))


@router.get("/api/status")
def status(target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today)) -> app_logic.JSONResponse:
    return app_logic._json_response(app_logic._fetch_status(target_date))


@router.get("/api/doctor")
def doctor_status(
    target_date: app_logic.date = app_logic.Query(default_factory=app_logic.date.today),
    source_health_hours: int = app_logic.Query(default=24, ge=1, le=168),
    pipeline_limit: int = app_logic.Query(default=5, ge=1, le=20),
    update_history_limit: int = app_logic.Query(default=5, ge=1, le=20),
) -> app_logic.JSONResponse:
    return app_logic._json_response(
        app_logic._doctor_payload(
            target_date,
            source_health_hours=source_health_hours,
            pipeline_limit=pipeline_limit,
            update_history_limit=update_history_limit,
        )
    )
