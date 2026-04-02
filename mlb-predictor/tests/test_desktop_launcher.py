import os
import sys
from pathlib import Path

import src.desktop.launcher as launcher_module
from src.desktop.launcher import bootstrap_runtime_environment


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