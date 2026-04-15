"""Static HTML shell pages (/, favicon, feature pages)."""
from __future__ import annotations

import src.api.app_logic as app_logic
from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api.constants import STATIC_DIR

router = APIRouter()

_VENUE_FIELD_GEOMETRY = STATIC_DIR / "data" / "venue_field_geometry.json"


@router.get("/")
def index() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.INDEX_FILE)


@router.get("/favicon.ico", include_in_schema=False)
@router.get("/favicon.svg", include_in_schema=False)
def favicon() -> app_logic.FileResponse:
    return app_logic.FileResponse(app_logic.FAVICON_FILE, media_type="image/svg+xml")


@router.get("/hot-hittes")
@router.get("/hot-hittes/")
@router.get("/hot-hitters/")
@router.get("/hot-hitters")
def hot_hitters_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.HOT_HITTERS_FILE)


@router.get("/results/")
@router.get("/results")
def results_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.RESULTS_FILE)


@router.get("/doctor/")
@router.get("/doctor")
def doctor_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.DOCTOR_FILE)


@router.get("/experiments/")
@router.get("/experiments")
def experiments_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.EXPERIMENTS_FILE)


@router.get("/totals/")
@router.get("/totals")
def totals_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.TOTALS_FILE)


@router.get("/pitchers/")
@router.get("/pitchers")
def pitchers_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.PITCHERS_FILE)


@router.get("/game/")
@router.get("/game")
def game_page() -> app_logic.FileResponse:
    return app_logic._html_file_response(app_logic.GAME_FILE)


@router.get("/data/venue_field_geometry.json", include_in_schema=False)
def venue_field_geometry_json() -> FileResponse:
    """Per-park outfield orientation + fence labels for wind visualization (see file ``_readme``)."""
    return FileResponse(_VENUE_FIELD_GEOMETRY, media_type="application/json")
