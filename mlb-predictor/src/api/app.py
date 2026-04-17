"""FastAPI application: merges ``app_logic`` into this module, then registers HTTP routes."""

from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.routing import APIRoute

from src.api.security import add_local_app_middleware

import src.api.app_logic as app_logic
from src.api.routers.http_routes import router as http_router

_mod = sys.modules[__name__]

for _key, _value in app_logic.__dict__.items():
    if _key.startswith("__") and _key.endswith("__") and _key not in ("__annotations__",):
        continue
    if _key in ("__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"):
        continue
    setattr(_mod, _key, _value)

app = FastAPI(title="MLB Predictor", version="1.1.0")
_mod.__dict__["app"] = app
add_local_app_middleware(app)

app.include_router(http_router)

for _route in app.routes:
    if isinstance(_route, APIRoute) and _route.endpoint is not None:
        _mod.__dict__[_route.endpoint.__name__] = _route.endpoint
