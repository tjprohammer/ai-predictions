"""Aggregated HTTP API router (composed sub-routers)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routers.api_feed_routes import router as api_feed_router
from src.api.routers.games_routes import router as games_router
from src.api.routers.html_routes import router as html_router
from src.api.routers.jobs_routes import router as jobs_router
from src.api.routers.meta_routes import router as meta_router
from src.api.routers.ops_routes import router as ops_router

router = APIRouter()
router.include_router(meta_router)
router.include_router(html_router)
router.include_router(api_feed_router)
router.include_router(games_router)
router.include_router(jobs_router)
router.include_router(ops_router)
