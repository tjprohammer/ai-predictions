import os
import sqlite3
import sys
from pathlib import Path

import src.desktop.launcher as launcher_module
from src.desktop.launcher import bootstrap_runtime_environment


def _write_history_seed(database_path: Path, row_count: int) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        for table_name in launcher_module.DESKTOP_HISTORY_REQUIRED_TABLES:
            connection.execute(f"CREATE TABLE {table_name} (game_date TEXT)")
            for index in range(row_count):
                connection.execute(
                    f"INSERT INTO {table_name} (game_date) VALUES (?)",
                    (f"2026-04-{index + 1:02d}",),
                )
        connection.commit()
    finally:
        connection.close()


def test_bootstrap_runtime_environment_seeds_data_and_config(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    user_root = tmp_path / "user"

    (bundle_root / "data" / "raw").mkdir(parents=True)
    (bundle_root / "db" / "seeds").mkdir(parents=True)
    (bundle_root / "config").mkdir(parents=True)

    (bundle_root / "data" / "raw" / "manual_lineups.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "data" / "raw" / "manual_market_totals.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "config" / ".env.example").write_text(
        "DATABASE_URL=postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb\n",
        encoding="utf-8",
    )
    (bundle_root / "db" / "seeds" / "park_factors.csv").write_text("venue_id\n", encoding="utf-8")

    for key in (
        "DATABASE_URL",
        "DATA_DIR",
        "MODEL_DIR",
        "REPORT_DIR",
        "FEATURE_DIR",
        "MANUAL_LINEUPS_CSV",
        "MANUAL_MARKETS_CSV",
        "PARK_FACTORS_CSV",
    ):
        monkeypatch.delenv(key, raising=False)

    user_env = bootstrap_runtime_environment(bundle_root, user_root)

    assert user_env == user_root / "config" / ".env"
    assert (user_root / "data" / "raw" / "manual_lineups.csv").exists()
    assert (user_root / "data" / "raw" / "manual_market_totals.csv").exists()
    assert (user_root / "db" / "mlb_predictor.sqlite3").exists()
    assert Path(os.environ["DATA_DIR"]) == user_root / "data"
    assert os.environ["DATABASE_URL"].startswith("sqlite:///")
    assert "mlb_predictor.sqlite3" in user_env.read_text(encoding="utf-8")
    assert Path(os.environ["PARK_FACTORS_CSV"]).exists()


def test_bootstrap_runtime_environment_preserves_explicit_database_url(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    user_root = tmp_path / "user"

    (bundle_root / "data" / "raw").mkdir(parents=True)
    (bundle_root / "db" / "seeds").mkdir(parents=True)
    (bundle_root / "config").mkdir(parents=True)

    (bundle_root / "data" / "raw" / "manual_lineups.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "data" / "raw" / "manual_market_totals.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "config" / ".env.example").write_text(
        "DATABASE_URL=postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb\n",
        encoding="utf-8",
    )
    (bundle_root / "db" / "seeds" / "park_factors.csv").write_text("venue_id\n", encoding="utf-8")

    custom_database_url = "postgresql+psycopg2://custom-user:custom-pass@dbhost:5432/mlb"
    monkeypatch.setenv("DATABASE_URL", custom_database_url)

    user_env = bootstrap_runtime_environment(bundle_root, user_root)

    assert user_env is not None
    assert os.environ["DATABASE_URL"] == custom_database_url
    assert "mlb_predictor.sqlite3" in user_env.read_text(encoding="utf-8")


def test_bootstrap_runtime_environment_refreshes_incomplete_default_runtime_seed(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    user_root = tmp_path / "user"

    (bundle_root / "data" / "raw").mkdir(parents=True)
    (bundle_root / "db" / "seeds").mkdir(parents=True)
    (bundle_root / "config").mkdir(parents=True)
    (user_root / "db").mkdir(parents=True)

    (bundle_root / "data" / "raw" / "manual_lineups.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "data" / "raw" / "manual_market_totals.csv").write_text("game_id\n", encoding="utf-8")
    (bundle_root / "config" / ".env.example").write_text(
        "DATABASE_URL=postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb\n",
        encoding="utf-8",
    )
    (bundle_root / "db" / "seeds" / "park_factors.csv").write_text("venue_id\n", encoding="utf-8")
    _write_history_seed(bundle_root / "db" / "mlb_predictor.sqlite3", row_count=2)
    _write_history_seed(user_root / "db" / "mlb_predictor.sqlite3", row_count=0)

    for key in (
        "DATABASE_URL",
        "DATA_DIR",
        "MODEL_DIR",
        "REPORT_DIR",
        "FEATURE_DIR",
        "MANUAL_LINEUPS_CSV",
        "MANUAL_MARKETS_CSV",
        "PARK_FACTORS_CSV",
    ):
        monkeypatch.delenv(key, raising=False)

    bootstrap_runtime_environment(bundle_root, user_root, log_path=user_root / "logs" / "launcher.log")

    connection = sqlite3.connect(user_root / "db" / "mlb_predictor.sqlite3")
    try:
        counts = {
            table_name: int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
            for table_name in launcher_module.DESKTOP_HISTORY_REQUIRED_TABLES
        }
    finally:
        connection.close()

    assert all(count == 2 for count in counts.values())
    assert (user_root / "db" / "mlb_predictor.pre_refresh.sqlite3").exists()


def test_ensure_standard_streams_supplies_devnull_fallbacks(monkeypatch):
    original_stdout = launcher_module.sys.stdout
    original_stderr = launcher_module.sys.stderr
    monkeypatch.setattr(launcher_module.sys, "stdout", None)
    monkeypatch.setattr(launcher_module.sys, "stderr", None)

    launcher_module.ensure_standard_streams()

    assert launcher_module.sys.stdout is not None
    assert launcher_module.sys.stderr is not None
    assert hasattr(launcher_module.sys.stdout, "isatty")
    assert hasattr(launcher_module.sys.stderr, "isatty")

    launcher_module.sys.stdout = original_stdout
    launcher_module.sys.stderr = original_stderr


def test_load_fastapi_app_returns_fastapi_instance():
    app = launcher_module.load_fastapi_app()

    assert app.title == "MLB Predictor"


def test_maybe_run_startup_migrations_only_for_sqlite(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_migrations():
        calls.append("ran")

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'desktop.sqlite3').as_posix()}")
    monkeypatch.setattr("src.utils.db_migrate.run_migrations", fake_run_migrations)

    launcher_module.maybe_run_startup_migrations(tmp_path / "launcher.log")

    assert calls == ["ran"]


def test_maybe_run_startup_migrations_skips_non_sqlite(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_run_migrations():
        calls.append("ran")

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/mlb")
    monkeypatch.setattr("src.utils.db_migrate.run_migrations", fake_run_migrations)

    launcher_module.maybe_run_startup_migrations(tmp_path / "launcher.log")

    assert calls == []


def test_maybe_run_startup_reference_bootstrap_only_for_sqlite(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_seed():
        calls.append("ran")
        return {"imported": 30, "bootstrapped": 0, "target_ready": True}

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'desktop.sqlite3').as_posix()}")
    monkeypatch.setattr("src.ingestors.park_factors.ensure_park_factors_seeded", fake_seed)

    launcher_module.maybe_run_startup_reference_bootstrap(tmp_path / "launcher.log")

    assert calls == ["ran"]


def test_maybe_run_startup_reference_bootstrap_skips_non_sqlite(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_seed():
        calls.append("ran")
        return {"imported": 0, "bootstrapped": 0, "target_ready": False}

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/mlb")
    monkeypatch.setattr("src.ingestors.park_factors.ensure_park_factors_seeded", fake_seed)

    launcher_module.maybe_run_startup_reference_bootstrap(tmp_path / "launcher.log")

    assert calls == []


def test_app_server_uses_direct_app_object(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.install_signal_handlers = None

    monkeypatch.setattr(launcher_module.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(launcher_module.uvicorn, "Server", FakeServer)

    fake_app = object()
    launcher_module.AppServer("127.0.0.1", 8126, fake_app)

    assert captured["app"] is fake_app
    assert captured["kwargs"]["log_config"] is None


def test_launch_window_returns_false_when_pywebview_runtime_fails(monkeypatch, tmp_path):
    class FakeWebviewModule:
        @staticmethod
        def create_window(*args, **kwargs):
            return None

        @staticmethod
        def start():
            raise RuntimeError("pythonnet loader failed")

    monkeypatch.setitem(sys.modules, "webview", FakeWebviewModule)

    log_path = tmp_path / "launcher.log"

    launched = launcher_module.launch_window("http://127.0.0.1:8126/", 1480, 980, log_path=log_path)

    assert launched is False
    assert "pywebview launch failed (RuntimeError): pythonnet loader failed" in log_path.read_text(encoding="utf-8")


def test_launch_window_returns_false_when_pywebview_missing(monkeypatch, tmp_path):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "webview":
            raise ImportError("No module named 'webview'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    log_path = tmp_path / "launcher.log"

    launched = launcher_module.launch_window("http://127.0.0.1:8126/", 1480, 980, log_path=log_path)

    assert launched is False
    assert "pywebview unavailable; falling back to browser launch" in log_path.read_text(encoding="utf-8")


def test_auto_refresh_enabled_defaults_to_non_headless(monkeypatch):
    monkeypatch.delenv("MLB_PREDICTOR_AUTO_REFRESH", raising=False)

    assert launcher_module.auto_refresh_enabled(headless=False) is True
    assert launcher_module.auto_refresh_enabled(headless=True) is False


def test_auto_refresh_enabled_honors_explicit_env(monkeypatch):
    monkeypatch.setenv("MLB_PREDICTOR_AUTO_REFRESH", "true")
    assert launcher_module.auto_refresh_enabled(headless=True) is True

    monkeypatch.setenv("MLB_PREDICTOR_AUTO_REFRESH", "off")
    assert launcher_module.auto_refresh_enabled(headless=False) is False


def test_should_run_startup_refresh_requires_missing_predictions():
    assert launcher_module.should_run_startup_refresh({"db_connected": False}) is False
    assert (
        launcher_module.should_run_startup_refresh(
            {
                "db_connected": True,
                "totals_predictions": 3,
                "hits_predictions": 12,
                "strikeouts_predictions": 4,
            }
        )
        is False
    )
    assert (
        launcher_module.should_run_startup_refresh(
            {
                "db_connected": True,
                "totals_predictions": 0,
                "hits_predictions": 12,
                "strikeouts_predictions": 4,
            }
        )
        is True
    )


def test_run_startup_refresh_runs_update_sequence(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    get_payloads = [
        {"db_connected": True, "totals_predictions": 0, "hits_predictions": 0, "strikeouts_predictions": 0},
        {"db_connected": True, "totals_predictions": 3, "hits_predictions": 12, "strikeouts_predictions": 4},
    ]
    post_calls: list[dict[str, object]] = []

    def fake_get(url, timeout):
        return FakeResponse(get_payloads.pop(0))

    def fake_post(url, json, timeout):
        post_calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(launcher_module.requests, "get", fake_get)
    monkeypatch.setattr(launcher_module.requests, "post", fake_post)

    log_path = tmp_path / "launcher.log"
    launcher_module.run_startup_refresh("http://127.0.0.1:8126/", log_path, target_date="2026-04-02")

    assert [call["json"]["action"] for call in post_calls] == list(launcher_module.AUTO_REFRESH_ACTIONS)
    assert all(call["json"]["target_date"] == "2026-04-02" for call in post_calls)
    log_text = log_path.read_text(encoding="utf-8")
    assert "Startup auto-refresh started for 2026-04-02" in log_text
    assert "Startup auto-refresh finished for 2026-04-02" in log_text


def test_run_startup_refresh_skips_when_predictions_exist(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    post_calls: list[dict[str, object]] = []

    def fake_get(url, timeout):
        return FakeResponse(
            {"db_connected": True, "totals_predictions": 3, "hits_predictions": 12, "strikeouts_predictions": 4}
        )

    def fake_post(url, json, timeout):
        post_calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(launcher_module.requests, "get", fake_get)
    monkeypatch.setattr(launcher_module.requests, "post", fake_post)

    log_path = tmp_path / "launcher.log"
    launcher_module.run_startup_refresh("http://127.0.0.1:8126/", log_path, target_date="2026-04-02")

    assert post_calls == []
    assert "Startup auto-refresh skipped for 2026-04-02" in log_path.read_text(encoding="utf-8")


def test_run_startup_refresh_logs_clear_warning_when_rebuild_is_blocked(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self.payload

    post_calls: list[str] = []

    def fake_get(url, timeout):
        return FakeResponse(
            {"db_connected": True, "totals_predictions": 0, "hits_predictions": 0, "strikeouts_predictions": 0}
        )

    def fake_post(url, json, timeout):
        post_calls.append(str(json["action"]))
        if json["action"] == "refresh_everything":
            return FakeResponse({"message": "Desktop historical data is incomplete."}, status_code=409)
        return FakeResponse({"ok": True}, status_code=200)

    monkeypatch.setattr(launcher_module.requests, "get", fake_get)
    monkeypatch.setattr(launcher_module.requests, "post", fake_post)

    log_path = tmp_path / "launcher.log"
    launcher_module.run_startup_refresh("http://127.0.0.1:8126/", log_path, target_date="2026-04-02")

    assert post_calls == list(launcher_module.AUTO_REFRESH_ACTIONS)
    assert "Startup auto-refresh blocked during refresh_everything for 2026-04-02" in log_path.read_text(encoding="utf-8")