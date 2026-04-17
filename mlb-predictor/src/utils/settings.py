from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]

for env_path in (BASE_DIR / "config" / ".env", BASE_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

_LEGACY_DEFAULT_DATABASE_URL = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"


def _default_project_sqlite_url() -> str:
    project_db = BASE_DIR / "db" / "mlb_predictor.sqlite3"
    project_db.parent.mkdir(parents=True, exist_ok=True)
    if not project_db.exists():
        project_db.touch()
    url = f"sqlite:///{project_db.resolve().as_posix()}"
    os.environ["DATABASE_URL"] = url
    return url


def _resolve_database_url() -> str:
    """Return DATABASE_URL: explicit env wins; else prefer existing SQLite; else create project SQLite."""
    raw = (os.environ.get("DATABASE_URL") or "").strip()
    if raw and raw != _LEGACY_DEFAULT_DATABASE_URL:
        return raw
    # Prefer well-known SQLite locations (desktop runtime, then project dev DB).
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        runtime_db = Path(local_app_data) / "MLBPredictor" / "db" / "mlb_predictor.sqlite3"
    else:
        runtime_db = Path.home() / ".mlb-predictor" / "db" / "mlb_predictor.sqlite3"
    project_db = BASE_DIR / "db" / "mlb_predictor.sqlite3"
    for candidate in (runtime_db, project_db):
        if candidate.exists():
            url = f"sqlite:///{candidate.resolve().as_posix()}"
            os.environ["DATABASE_URL"] = url
            return url
    return _default_project_sqlite_url()


def _parse_windows(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    database_url: str
    prior_season: int
    current_season: int
    rolling_windows: tuple[int, ...]
    team_full_weight_games: int
    pitcher_full_weight_starts: int
    min_pa_full_weight: int
    prior_blend_mode: str
    prior_weight_multiplier: float
    data_dir: Path
    model_dir: Path
    report_dir: Path
    feature_dir: Path
    manual_lineups_csv: Path
    manual_markets_csv: Path
    park_factors_csv: Path
    model_version_prefix: str
    odds_api_key: str | None
    odds_api_key_fallback: str | None
    log_level: str
    # Skip mutating markets / lineups / probable starters for a game once this many minutes
    # before first pitch (0 disables). Uses ``games.game_start_ts`` (UTC).
    pregame_ingest_lock_minutes: int
    # If >0, market_totals may skip The Odds API pulls when DB snapshots for the slate are newer
    # than this many minutes (see MARKET_INGEST_ODDS_API_FRESH_MINUTES). 0 disables.
    market_ingest_odds_api_fresh_minutes: int
    # Minimum per-game data_quality.certainty_pct (0–100) required for a pick to appear on the
    # green strip. Empty / unset = no extra gate (legacy behavior).
    board_green_min_game_certainty_pct: float | None
    # When true and ``board_green_snapshots`` exists, freeze each game's first qualifying green
    # pick once the game enters the pregame ingest lock window.
    board_green_snapshot_enabled: bool


def _parse_optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return float(str(raw).strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = BASE_DIR / os.getenv("DATA_DIR", "data")
    model_dir = BASE_DIR / os.getenv("MODEL_DIR", "data/models")
    report_dir = BASE_DIR / os.getenv("REPORT_DIR", "data/reports")
    feature_dir = BASE_DIR / os.getenv("FEATURE_DIR", "data/features")
    manual_lineups_csv = BASE_DIR / os.getenv("MANUAL_LINEUPS_CSV", "data/raw/manual_lineups.csv")
    manual_markets_csv = BASE_DIR / os.getenv("MANUAL_MARKETS_CSV", "data/raw/manual_market_totals.csv")
    park_factors_csv = BASE_DIR / os.getenv("PARK_FACTORS_CSV", "db/seeds/park_factors.csv")
    return Settings(
        base_dir=BASE_DIR,
        database_url=_resolve_database_url(),
        prior_season=int(os.getenv("PRIOR_SEASON", "2025")),
        current_season=int(os.getenv("CURRENT_SEASON", "2026")),
        rolling_windows=_parse_windows(os.getenv("ROLLING_WINDOWS", "7,14,30")),
        team_full_weight_games=int(os.getenv("TEAM_FULL_WEIGHT_GAMES", "30")),
        pitcher_full_weight_starts=int(os.getenv("PITCHER_FULL_WEIGHT_STARTS", "10")),
        min_pa_full_weight=int(os.getenv("MIN_PA_FULL_WEIGHT", "120")),
        prior_blend_mode=os.getenv("PRIOR_BLEND_MODE", "standard").strip().lower(),
        prior_weight_multiplier=float(os.getenv("PRIOR_WEIGHT_MULTIPLIER", "1.0")),
        data_dir=data_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        feature_dir=feature_dir,
        manual_lineups_csv=manual_lineups_csv,
        manual_markets_csv=manual_markets_csv,
        park_factors_csv=park_factors_csv,
        model_version_prefix=os.getenv("MODEL_VERSION_PREFIX", "v1"),
        odds_api_key=os.getenv("THE_ODDS_API_KEY") or None,
        odds_api_key_fallback=os.getenv("THE_ODDS_API_KEY_FALLBACK") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        pregame_ingest_lock_minutes=int(os.getenv("MLB_PREGAME_INGEST_LOCK_MINUTES", "30")),
        market_ingest_odds_api_fresh_minutes=int(os.getenv("MARKET_INGEST_ODDS_API_FRESH_MINUTES", "0")),
        board_green_min_game_certainty_pct=_parse_optional_float_env("BOARD_GREEN_MIN_GAME_CERTAINTY_PCT"),
        board_green_snapshot_enabled=(
            os.getenv("BOARD_GREEN_SNAPSHOT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        ),
    )