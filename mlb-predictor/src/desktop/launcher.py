from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import requests
import uvicorn
from dotenv import load_dotenv


APP_TITLE = "MLB Predictor"
RUNTIME_DIRNAME = "MLBPredictor"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_WIDTH = 1480
DEFAULT_HEIGHT = 980
SERVER_BOOT_TIMEOUT_SECONDS = 30.0
_STDOUT_FALLBACK = None
_STDERR_FALLBACK = None


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def runtime_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / RUNTIME_DIRNAME
    return Path.home() / ".mlb-predictor"


def runtime_log_path(user_dir: Path) -> Path:
    return user_dir / "logs" / "launcher.log"


def append_runtime_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def ensure_standard_streams() -> None:
    global _STDOUT_FALLBACK, _STDERR_FALLBACK
    if sys.stdout is None:
        _STDOUT_FALLBACK = open(os.devnull, "a", encoding="utf-8")
        sys.stdout = _STDOUT_FALLBACK
    if sys.stderr is None:
        _STDERR_FALLBACK = open(os.devnull, "a", encoding="utf-8")
        sys.stderr = _STDERR_FALLBACK


def _copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def bootstrap_runtime_environment(bundle_dir: Path, user_dir: Path) -> Path | None:
    user_dir.mkdir(parents=True, exist_ok=True)

    data_source = bundle_dir / "data"
    config_source = bundle_dir / "config"
    db_source = bundle_dir / "db"

    data_target = user_dir / "data"
    config_target = user_dir / "config"
    db_target = user_dir / "db"

    _copy_tree_if_missing(data_source, data_target)
    _copy_tree_if_missing(db_source, db_target)

    if data_target.exists():
        for relative_dir in ("raw", "features", "models", "reports", "staged"):
            (data_target / relative_dir).mkdir(parents=True, exist_ok=True)

    config_target.mkdir(parents=True, exist_ok=True)
    user_env = config_target / ".env"
    if not user_env.exists():
        _copy_if_missing(config_source / ".env", user_env)
        _copy_if_missing(config_source / ".env.example", user_env)

    if user_env.exists():
        load_dotenv(user_env, override=False)

    os.environ.setdefault("DATA_DIR", str(data_target))
    os.environ.setdefault("MODEL_DIR", str(data_target / "models"))
    os.environ.setdefault("REPORT_DIR", str(data_target / "reports"))
    os.environ.setdefault("FEATURE_DIR", str(data_target / "features"))
    os.environ.setdefault("MANUAL_LINEUPS_CSV", str(data_target / "raw" / "manual_lineups.csv"))
    os.environ.setdefault("MANUAL_MARKETS_CSV", str(data_target / "raw" / "manual_market_totals.csv"))

    park_factors_path = db_target / "seeds" / "park_factors.csv"
    if not park_factors_path.exists():
        fallback_park_factors = bundle_dir / "db" / "seeds" / "park_factors.csv"
        if fallback_park_factors.exists():
            park_factors_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_park_factors, park_factors_path)
    if park_factors_path.exists():
        os.environ.setdefault("PARK_FACTORS_CSV", str(park_factors_path))

    return user_env if user_env.exists() else None


def find_open_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def load_fastapi_app():
    from src.api.app import app as fastapi_app

    return fastapi_app


class AppServer:
    def __init__(self, host: str, port: int, app) -> None:
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/"
        self._config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info",
            log_config=None,
        )
        self._server = uvicorn.Server(self._config)
        self._server.install_signal_handlers = lambda: None
        self._thread_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="mlb-predictor-server", daemon=True)

    def _run(self) -> None:
        try:
            self._server.run()
        except BaseException as exc:  # noqa: BLE001
            self._thread_error = exc
            raise

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout_seconds: float = SERVER_BOOT_TIMEOUT_SECONDS) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            if not self._thread.is_alive():
                if self._thread_error is not None:
                    raise RuntimeError("The app server exited before becoming ready.") from self._thread_error
                raise RuntimeError("The app server exited before becoming ready.")
            try:
                response = requests.get(f"{self.url}api/health", timeout=1.0)
                if response.ok:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for the app server to start: {last_error}")

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


def launch_window(url: str, width: int, height: int) -> bool:
    try:
        import webview
    except ImportError:
        return False

    webview.create_window(APP_TITLE, url, width=width, height=height, min_size=(1120, 760))
    webview.start()
    return True


def hold_browser_session(url: str) -> None:
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch MLB Predictor as a desktop app shell")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface for the embedded API server")
    parser.add_argument("--port", type=int, help="Port for the embedded API server")
    parser.add_argument("--headless", action="store_true", help="Run the packaged server without opening a desktop window")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Desktop window width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Desktop window height")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    user_dir = runtime_root()
    log_path = runtime_log_path(user_dir)
    bundled_root = bundle_root()
    append_runtime_log(log_path, f"Launcher starting from {bundled_root}")
    bootstrap_runtime_environment(bundled_root, user_dir)
    ensure_standard_streams()

    server: AppServer | None = None

    try:
        port = args.port or find_open_port(args.host)
        fastapi_app = load_fastapi_app()
        server = AppServer(args.host, port, fastapi_app)
        server.start()
        server.wait_until_ready()
        append_runtime_log(log_path, f"Server ready at {server.url}")
        if args.headless:
            print(server.url)
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                return 0

        if not launch_window(server.url, args.width, args.height):
            append_runtime_log(log_path, "pywebview unavailable; falling back to browser launch")
            print("pywebview is not installed; opening the app in a browser window instead.")
            hold_browser_session(server.url)
        return 0
    except Exception as exc:  # noqa: BLE001
        append_runtime_log(log_path, f"Launcher failed: {exc}")
        append_runtime_log(log_path, traceback.format_exc())
        raise
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())