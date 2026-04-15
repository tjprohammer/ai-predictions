"""
Local-first HTTP hardening for the FastAPI app.

- Binds are controlled separately (see Makefile / launcher); this layer assumes
  the common case: dashboard and API are used from the same machine.

- All ``/api/*`` routes require either:
  - a trusted client address (loopback or the ASGI test client), or
  - a valid ``MLB_PREDICTOR_API_KEY`` sent as
    ``Authorization: Bearer <key>`` or ``X-MLB-Predictor-Api-Key: <key>``.

Set ``MLB_PREDICTOR_API_KEY`` when you intentionally bind beyond localhost and
need remote access to JSON endpoints.
"""

from __future__ import annotations

import os

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

_TRUSTED_CLIENT_HOSTS = frozenset(
    {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
        None,
    }
)

# Browsers and pywebview typically use http://127.0.0.1:<port> or http://localhost:<port>
_CORS_ORIGIN_REGEX = (
    r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$"
    r"|^https?://\[::1\](:\d+)?$"
)


def _api_key_from_env() -> str:
    return (os.environ.get("MLB_PREDICTOR_API_KEY") or "").strip()


def _trusted_client(request: Request) -> bool:
    if request.client is None:
        return True
    host = (request.client.host or "").strip().lower()
    if host in _TRUSTED_CLIENT_HOSTS:
        return True
    # Some stacks report IPv4-mapped IPv6
    if host.startswith("127.") or host.endswith("127.0.0.1"):
        return True
    return False


def _valid_api_key(request: Request, expected: str) -> bool:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth == f"Bearer {expected}":
        return True
    header_key = (request.headers.get("X-MLB-Predictor-Api-Key") or "").strip()
    return header_key == expected


def add_local_app_middleware(app: ASGIApp) -> None:
    """Register security + tight CORS on the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if not isinstance(app, FastAPI):
        raise TypeError("Expected a FastAPI instance")

    # CORS outermost (registered second) so OPTIONS preflight is handled first.
    app.add_middleware(LocalApiSecurityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class LocalApiSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path or ""
        if not path.startswith("/api/"):
            return await call_next(request)

        key = _api_key_from_env()
        if key:
            if _valid_api_key(request, key):
                return await call_next(request)
            if _trusted_client(request):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Invalid or missing API key. Set MLB_PREDICTOR_API_KEY in the environment "
                        "and send Authorization: Bearer <key> or X-MLB-Predictor-Api-Key."
                    )
                },
            )

        if _trusted_client(request):
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "API access is restricted to localhost. "
                    "Set MLB_PREDICTOR_API_KEY and pass the key in headers for remote access, "
                    "or use http://127.0.0.1 / http://localhost from this machine."
                )
            },
        )


CORS_ALLOW_ORIGIN_REGEX = _CORS_ORIGIN_REGEX
