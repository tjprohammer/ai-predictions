from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.desktop import launcher as launcher_module


REQUEST_TIMEOUT_SECONDS = 15
UPDATE_ACTION_TIMEOUT_SECONDS = 900
UPDATE_ACTIONS = (
    "refresh_everything",
    "prepare_slate",
    "import_manual_inputs",
    "refresh_results",
    "rebuild_predictions",
    "grade_predictions",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local desktop/API smoke test against the current runtime")
    parser.add_argument("--base-url", help="Existing app base URL, for example http://127.0.0.1:8126/")
    parser.add_argument("--target-date", type=date.fromisoformat, help="Target slate date in YYYY-MM-DD format")
    parser.add_argument(
        "--exercise-update-actions",
        action="store_true",
        help="Also run each dashboard Update Center action and collect the results",
    )
    parser.add_argument(
        "--update-action",
        action="append",
        choices=UPDATE_ACTIONS,
        dest="update_actions",
        help="Run only selected update action smokes. Can be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full smoke payload as JSON")
    return parser


def _normalize_base_url(base_url: str) -> str:
    return base_url if base_url.endswith("/") else f"{base_url}/"


def _request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    params: dict[str, object] | None = None,
    json_body: dict[str, object] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    allow_error: bool = False,
) -> tuple[int, dict[str, Any]]:
    response = requests.request(
        method,
        f"{_normalize_base_url(base_url)}{path.lstrip('/')}",
        params=params,
        json=json_body,
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    if not allow_error:
        response.raise_for_status()
    normalized = dict(payload) if isinstance(payload, dict) else {"value": payload}
    return response.status_code, normalized


def _fetch_json(base_url: str, path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    _status_code, payload = _request_json("GET", base_url, path, params=params)
    return payload


def _post_json(
    base_url: str,
    path: str,
    body: dict[str, object],
    *,
    timeout: int = UPDATE_ACTION_TIMEOUT_SECONDS,
    allow_error: bool = False,
) -> tuple[int, dict[str, Any]]:
    return _request_json(
        "POST",
        base_url,
        path,
        json_body=body,
        timeout=timeout,
        allow_error=allow_error,
    )


def _resolve_update_actions(update_actions: Iterable[str] | None) -> list[str]:
    requested = [action for action in (update_actions or []) if action]
    return requested or list(UPDATE_ACTIONS)


def _exercise_update_action(base_url: str, action: str, target_date: date) -> dict[str, Any]:
    status_code, payload = _post_json(
        base_url,
        "/api/update-jobs/run",
        {
            "action": action,
            "target_date": target_date.isoformat(),
        },
        allow_error=True,
    )
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    status_snapshot = payload.get("status") if isinstance(payload.get("status"), dict) else None
    failed_step = next(
        (
            step
            for step in reversed(steps)
            if isinstance(step, dict) and step.get("returncode") not in {0, None}
        ),
        None,
    )
    error = payload.get("message") or payload.get("error")
    if error is None and isinstance(failed_step, dict):
        error = f"{failed_step.get('module', action)} exited with code {failed_step.get('returncode')}"
    return {
        "action": action,
        "label": payload.get("label") or action,
        "ok": bool(payload.get("ok")) and status_code < 400,
        "http_status": status_code,
        "step_count": len(steps),
        "failed_step": failed_step.get("module") if isinstance(failed_step, dict) else None,
        "error": error,
        "status_snapshot": status_snapshot,
    }


def run_smoke(
    base_url: str,
    target_date: date,
    *,
    exercise_update_actions: bool = False,
    update_actions: Iterable[str] | None = None,
) -> dict[str, Any]:
    target_date_raw = target_date.isoformat()
    payload = {
        "base_url": _normalize_base_url(base_url),
        "target_date": target_date_raw,
        "health": _fetch_json(base_url, "/api/health"),
        "doctor": _fetch_json(base_url, "/api/doctor", {"target_date": target_date_raw}),
        "board": _fetch_json(base_url, "/api/games/board", {"target_date": target_date_raw}),
        "scorecards": _fetch_json(base_url, "/api/model-scorecards", {"target_date": target_date_raw}),
        "top_misses": _fetch_json(base_url, "/api/review/top-misses", {"target_date": target_date_raw}),
        "clv": _fetch_json(
            base_url,
            "/api/review/clv",
            {"start_date": target_date_raw, "end_date": target_date_raw},
        ),
    }
    payload["summary"] = {
        "board_games": len(payload["board"].get("games") or []),
        "scorecard_rows": len(payload["scorecards"].get("scorecards") or []),
        "top_miss_rows": len(payload["top_misses"].get("misses") or []),
        "clv_best_rows": len(payload["clv"].get("best_clv") or []),
        "doctor_status": payload["doctor"].get("overall", {}).get("status"),
    }
    if exercise_update_actions:
        action_results = [
            _exercise_update_action(base_url, action, target_date)
            for action in _resolve_update_actions(update_actions)
        ]
        payload["update_actions"] = action_results
        payload["post_update_health"] = _fetch_json(base_url, "/api/health")
        payload["summary"].update(
            {
                "update_action_count": len(action_results),
                "update_action_failures": [
                    result["action"] for result in action_results if not result["ok"]
                ],
                "post_update_totals_predictions": payload["post_update_health"].get("totals_predictions"),
                "post_update_hits_predictions": payload["post_update_health"].get("hits_predictions"),
                "post_update_strikeouts_predictions": payload["post_update_health"].get("strikeouts_predictions"),
            }
        )
    return payload


def _bootstrap_local_server() -> tuple[launcher_module.AppServer, str]:
    launcher_module.ensure_standard_streams()
    bundle_dir = launcher_module.bundle_root()
    user_dir = launcher_module.runtime_root()
    launcher_module.ensure_bundle_on_sys_path(bundle_dir)
    log_path = launcher_module.runtime_log_path(user_dir)
    launcher_module.bootstrap_runtime_environment(bundle_dir, user_dir, log_path=log_path)
    launcher_module.maybe_run_startup_migrations(log_path)
    launcher_module.maybe_run_startup_reference_bootstrap(log_path)
    app = launcher_module.load_fastapi_app()
    server = launcher_module.AppServer(launcher_module.DEFAULT_HOST, launcher_module.find_open_port(), app)
    server.start()
    server.wait_until_ready()
    return server, server.url


def main() -> int:
    args = build_parser().parse_args()
    target_date = args.target_date or date.today()
    server = None
    base_url = args.base_url
    try:
        if not base_url:
            server, base_url = _bootstrap_local_server()
        payload = run_smoke(
            str(base_url),
            target_date,
            exercise_update_actions=args.exercise_update_actions,
            update_actions=args.update_actions,
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                json.dumps(
                    {
                        "base_url": payload["base_url"],
                        "target_date": payload["target_date"],
                        "summary": payload["summary"],
                    },
                    indent=2,
                )
            )
        return 0
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())