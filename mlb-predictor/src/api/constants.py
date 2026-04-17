from __future__ import annotations

from pathlib import Path

from src.features.first5_totals_builder import main as build_first5_totals_features_main
from src.features.hits_builder import main as build_hits_features_main
from src.features.hr_builder import main as build_hr_features_main
from src.features.strikeouts_builder import main as build_strikeouts_features_main
from src.features.total_bases_builder import main as build_total_bases_features_main
from src.features.totals_builder import main as build_totals_features_main
from src.ingestors.boxscores import main as ingest_boxscores_main
from src.ingestors.games import main as ingest_games_main
from src.ingestors.lineups import main as import_lineups_main
from src.ingestors.lineups_backfill import main as lineups_backfill_main
from src.ingestors.matchup_splits import main as matchup_splits_main
from src.ingestors.market_totals import main as import_market_totals_main
from src.ingestors.player_batting import main as ingest_player_batting_main
from src.ingestors.player_status import main as ingest_player_status_main
from src.ingestors.prepare_slate_inputs import main as prepare_slate_inputs_main
from src.ingestors.starters import main as ingest_starters_main
from src.ingestors.umpire import main as ingest_umpire_main
from src.models.predict_first5_totals import main as predict_first5_totals_main
from src.models.predict_hits import main as predict_hits_main
from src.models.predict_hr import main as predict_hr_main
from src.models.predict_strikeouts import main as predict_strikeouts_main
from src.models.predict_total_bases import main as predict_total_bases_main
from src.models.predict_totals import main as predict_totals_main
from src.models.train_total_bases import main as train_total_bases_main
from src.transforms.bullpens_daily import main as refresh_bullpens_daily_main
from src.transforms.freeze_markets import main as freeze_markets_main
from src.transforms.offense_daily import main as refresh_offense_daily_main
from src.transforms.product_surfaces import main as refresh_product_surfaces_main

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
HOT_HITTERS_FILE = STATIC_DIR / "hot-hitters.html"
PITCHERS_FILE = STATIC_DIR / "pitchers.html"
RESULTS_FILE = STATIC_DIR / "results.html"
TOTALS_FILE = STATIC_DIR / "totals.html"
GAME_FILE = STATIC_DIR / "game.html"
DOCTOR_FILE = STATIC_DIR / "doctor.html"
EXPERIMENTS_FILE = STATIC_DIR / "experiments.html"
FAVICON_FILE = STATIC_DIR / "favicon.svg"

HTML_SHELL_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Matchup API: sample-size tiers for UI trust (not model thresholds).
# "low" = noisy; "adequate" = directional; "strong" = more stable.
MATCHUP_BVP_ADEQUATE_MIN_AB = 25
MATCHUP_BVP_STRONG_MIN_AB = 80
MATCHUP_H2H_ADEQUATE_MIN_GAMES = 5
MATCHUP_H2H_STRONG_MIN_GAMES = 15
MATCHUP_PVT_ADEQUATE_MIN_IP = 15.0
MATCHUP_PVT_STRONG_MIN_IP = 45.0
MATCHUP_PLATOON_ADEQUATE_MIN_PA = 30
MATCHUP_PLATOON_STRONG_MIN_PA = 100

# Main game board default filters — keep aligned with index.html (hitLimit / hitMinProbability / checkboxes).
GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM = 3
GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY = 0.5
GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY = False
GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED = True

UPDATE_MODULE_MAINS = {
    "src.ingestors.games": ingest_games_main,
    "src.ingestors.starters": ingest_starters_main,
    "src.ingestors.prepare_slate_inputs": prepare_slate_inputs_main,
    "src.ingestors.lineups": import_lineups_main,
    "src.ingestors.lineups_backfill": lineups_backfill_main,
    "src.ingestors.matchup_splits": matchup_splits_main,
    "src.ingestors.player_status": ingest_player_status_main,
    "src.ingestors.market_totals": import_market_totals_main,
    "src.ingestors.umpire": ingest_umpire_main,
    "src.ingestors.boxscores": ingest_boxscores_main,
    "src.ingestors.player_batting": ingest_player_batting_main,
    "src.transforms.offense_daily": refresh_offense_daily_main,
    "src.transforms.bullpens_daily": refresh_bullpens_daily_main,
    "src.transforms.freeze_markets": freeze_markets_main,
    "src.transforms.product_surfaces": refresh_product_surfaces_main,
    "src.features.totals_builder": build_totals_features_main,
    "src.features.first5_totals_builder": build_first5_totals_features_main,
    "src.features.hits_builder": build_hits_features_main,
    "src.features.hr_builder": build_hr_features_main,
    "src.features.total_bases_builder": build_total_bases_features_main,
    "src.features.strikeouts_builder": build_strikeouts_features_main,
    "src.models.predict_totals": predict_totals_main,
    "src.models.predict_first5_totals": predict_first5_totals_main,
    "src.models.predict_hits": predict_hits_main,
    "src.models.predict_hr": predict_hr_main,
    "src.models.predict_strikeouts": predict_strikeouts_main,
    "src.models.train_total_bases": train_total_bases_main,
    "src.models.predict_total_bases": predict_total_bases_main,
}
