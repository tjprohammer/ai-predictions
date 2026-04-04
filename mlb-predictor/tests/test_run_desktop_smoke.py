import datetime as dt
import json

from scripts import run_desktop_smoke as smoke_module


def test_run_smoke_hits_expected_endpoints(monkeypatch):
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_fetch_json(base_url, path, params=None):
        calls.append((path, params))
        mapping = {
            "/api/health": {"ok": True},
            "/api/doctor": {"overall": {"status": "ok"}},
            "/api/games/board": {"games": [{"game_id": 1}]},
            "/api/model-scorecards": {"scorecards": [{"market": "totals"}]},
            "/api/review/top-misses": {"misses": [{"game_id": 1}]},
            "/api/review/clv": {"best_clv": [{"game_id": 1}]},
        }
        return mapping[path]

    monkeypatch.setattr(smoke_module, "_fetch_json", fake_fetch_json)

    payload = smoke_module.run_smoke("http://127.0.0.1:8126", dt.date(2026, 4, 2))

    assert payload["summary"]["doctor_status"] == "ok"
    assert payload["summary"]["board_games"] == 1
    assert [path for path, _params in calls] == [
        "/api/health",
        "/api/doctor",
        "/api/games/board",
        "/api/model-scorecards",
        "/api/review/top-misses",
        "/api/review/clv",
    ]


def test_run_smoke_can_exercise_update_actions(monkeypatch):
    monkeypatch.setattr(
        smoke_module,
        "_fetch_json",
        lambda _base_url, path, _params=None: {
            "/api/health": {
                "ok": True,
                "totals_predictions": 12,
                "hits_predictions": 34,
                "strikeouts_predictions": 8,
            },
            "/api/doctor": {"overall": {"status": "ok"}},
            "/api/games/board": {"games": [{"game_id": 1}]},
            "/api/model-scorecards": {"scorecards": [{"market": "totals"}]},
            "/api/review/top-misses": {"misses": []},
            "/api/review/clv": {"best_clv": []},
        }[path],
    )
    monkeypatch.setattr(
        smoke_module,
        "_exercise_update_action",
        lambda _base_url, action, _target_date: {
            "action": action,
            "label": action,
            "ok": action != "rebuild_predictions",
            "http_status": 200 if action != "rebuild_predictions" else 500,
            "step_count": 4,
            "failed_step": None if action != "rebuild_predictions" else "src.models.predict_hits",
            "error": None if action != "rebuild_predictions" else "failed",
            "status_snapshot": {
                "totals_predictions": 12,
                "hits_predictions": 34,
                "strikeouts_predictions": 8,
            },
        },
    )

    payload = smoke_module.run_smoke(
        "http://127.0.0.1:8126",
        dt.date(2026, 4, 2),
        exercise_update_actions=True,
        update_actions=["prepare_slate", "rebuild_predictions"],
    )

    assert payload["summary"]["update_action_count"] == 2
    assert payload["summary"]["update_action_failures"] == ["rebuild_predictions"]
    assert payload["summary"]["post_update_totals_predictions"] == 12
    assert payload["summary"]["post_update_hits_predictions"] == 34
    assert payload["summary"]["post_update_strikeouts_predictions"] == 8
    assert [result["action"] for result in payload["update_actions"]] == [
        "prepare_slate",
        "rebuild_predictions",
    ]


def test_main_bootstraps_local_server_when_base_url_missing(monkeypatch, capsys):
    class FakeServer:
        url = "http://127.0.0.1:9001/"

        def start(self):
            return None

        def wait_until_ready(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(smoke_module.launcher_module, "ensure_standard_streams", lambda: None)
    monkeypatch.setattr(smoke_module.launcher_module, "bundle_root", lambda: "bundle")
    monkeypatch.setattr(smoke_module.launcher_module, "runtime_root", lambda: "runtime")
    monkeypatch.setattr(smoke_module.launcher_module, "ensure_bundle_on_sys_path", lambda _bundle: None)
    monkeypatch.setattr(smoke_module.launcher_module, "runtime_log_path", lambda _runtime: "runtime.log")
    monkeypatch.setattr(smoke_module.launcher_module, "bootstrap_runtime_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(smoke_module.launcher_module, "maybe_run_startup_migrations", lambda *_args: None)
    monkeypatch.setattr(smoke_module.launcher_module, "maybe_run_startup_reference_bootstrap", lambda *_args: None)
    monkeypatch.setattr(smoke_module.launcher_module, "load_fastapi_app", lambda: object())
    monkeypatch.setattr(smoke_module.launcher_module, "find_open_port", lambda: 9001)
    monkeypatch.setattr(smoke_module.launcher_module, "AppServer", lambda *_args: FakeServer())
    monkeypatch.setattr(
        smoke_module,
        "run_smoke",
        lambda base_url, target_date, **_kwargs: {
            "base_url": base_url,
            "target_date": target_date.isoformat(),
            "summary": {"doctor_status": "ok"},
        },
    )
    monkeypatch.setattr(
        smoke_module,
        "build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": staticmethod(
                    lambda: type(
                        "Args",
                        (),
                        {
                            "base_url": None,
                            "target_date": dt.date(2026, 4, 2),
                            "exercise_update_actions": False,
                            "update_actions": None,
                            "json": False,
                        },
                    )()
                )
            },
        )(),
    )

    result = smoke_module.main()
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["summary"]["doctor_status"] == "ok"
