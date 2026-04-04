from __future__ import annotations

import argparse
import importlib
import json
from datetime import date


app_module = importlib.import_module("src.api.app")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local runtime health and operator readiness checks")
    parser.add_argument("--target-date", type=date.fromisoformat, help="Target slate date in YYYY-MM-DD format")
    parser.add_argument("--source-health-hours", type=int, default=24, help="Recent source-health window to summarize")
    parser.add_argument("--pipeline-limit", type=int, default=5, help="Number of recent pipeline runs to include")
    parser.add_argument(
        "--update-history-limit",
        type=int,
        default=5,
        help="Number of recent update jobs to include",
    )
    parser.add_argument("--json", action="store_true", help="Print the full doctor payload as JSON")
    return parser


def render_text(payload: dict[str, object]) -> str:
    overall = dict(payload.get("overall") or {})
    status = str(overall.get("status") or "unknown").upper()
    checks = list(payload.get("checks") or [])
    runtime = dict(payload.get("runtime") or {})

    lines = [
        f"MLB Predictor doctor: {status}",
        f"Target date: {payload.get('target_date')}",
        f"Database: {runtime.get('database_url')}",
        f"Dialect: {runtime.get('db_dialect')}",
        "",
        "Checks:",
    ]
    for check in checks:
        check_dict = dict(check or {})
        marker = "PASS" if check_dict.get("ok") else "FAIL"
        lines.append(f"- {marker} {check_dict.get('label')}: {check_dict.get('detail')}")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    target_date = args.target_date or date.today()
    payload = app_module._doctor_payload(
        target_date,
        source_health_hours=args.source_health_hours,
        pipeline_limit=args.pipeline_limit,
        update_history_limit=args.update_history_limit,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())