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
    log_level: str


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
        database_url=os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"),
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
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )