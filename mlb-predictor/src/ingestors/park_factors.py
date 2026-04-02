from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

from src.ingestors.common import team_dimension_row
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

STATCAST_PARK_FACTORS_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={year}"

REQUIRED_COLUMNS = {"season", "team_abbr"}
FACTOR_COLUMNS = [
    "run_factor",
    "hr_factor",
    "singles_factor",
    "doubles_factor",
    "triples_factor",
]
OPTIONAL_COLUMNS = ["venue_id", "venue_name", "source_name"]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "team_abbr", *OPTIONAL_COLUMNS, *FACTOR_COLUMNS])


def _write_seed_file(frame: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def _fetch_statcast_frame(season: int) -> pd.DataFrame:
    response = requests.get(
        STATCAST_PARK_FACTORS_URL.format(year=season),
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    match = re.search(r"var data = (\[.*?\]);", response.text, re.S)
    if not match:
        raise ValueError(f"Could not locate embedded Statcast park factor data for season {season}")
    payload = json.loads(match.group(1))

    rows: list[dict[str, object]] = []
    for item in payload:
        team_id_raw = item.get("main_team_id")
        if team_id_raw is None:
            continue
        team_abbr = team_dimension_row(int(team_id_raw))["team_abbr"]
        rows.append(
            {
                "season": season,
                "team_abbr": team_abbr,
                "venue_id": pd.to_numeric(item.get("venue_id"), errors="coerce"),
                "venue_name": item.get("venue_name"),
                "run_factor": pd.to_numeric(item.get("index_runs"), errors="coerce") / 100.0,
                "hr_factor": pd.to_numeric(item.get("index_hr"), errors="coerce") / 100.0,
                "singles_factor": pd.to_numeric(item.get("index_1b"), errors="coerce") / 100.0,
                "doubles_factor": pd.to_numeric(item.get("index_2b"), errors="coerce") / 100.0,
                "triples_factor": pd.to_numeric(item.get("index_3b"), errors="coerce") / 100.0,
                "source_name": f"statcast_{item.get('year_range', season)}",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Statcast park factor payload was empty for season {season}")
    normalized = _normalize_frame(frame)

    team_lookup = _latest_home_venues()
    missing_teams = sorted(set(team_lookup["team_abbr"].tolist()) - set(normalized["team_abbr"].tolist()))
    if missing_teams:
        neutral_rows = team_lookup[team_lookup["team_abbr"].isin(missing_teams)].copy()
        neutral_rows["season"] = season
        neutral_rows["run_factor"] = 1.0
        neutral_rows["hr_factor"] = 1.0
        neutral_rows["singles_factor"] = 1.0
        neutral_rows["doubles_factor"] = 1.0
        neutral_rows["triples_factor"] = 1.0
        neutral_rows["source_name"] = "statcast_missing_neutral"
        neutral_rows = neutral_rows[["season", "team_abbr", *OPTIONAL_COLUMNS, *FACTOR_COLUMNS]]
        normalized = pd.concat([normalized, neutral_rows], ignore_index=True)

    log.info("Fetched %s Statcast park factor rows for season %s", len(normalized), season)
    return normalized


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "source" in normalized.columns and "source_name" not in normalized.columns:
        normalized = normalized.rename(columns={"source": "source_name"})
    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing park factor columns: {sorted(missing)}")
    for column in OPTIONAL_COLUMNS + FACTOR_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["season"] = pd.to_numeric(normalized["season"], errors="coerce")
    normalized["team_abbr"] = normalized["team_abbr"].astype(str).str.upper().str.strip()
    normalized["venue_id"] = pd.to_numeric(normalized["venue_id"], errors="coerce")
    for column in FACTOR_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["source_name"] = normalized["source_name"].fillna("seed_csv")
    normalized = normalized.dropna(subset=["season"])
    normalized["season"] = normalized["season"].astype(int)
    normalized = normalized[normalized["team_abbr"] != ""]
    return normalized[["season", "team_abbr", *OPTIONAL_COLUMNS, *FACTOR_COLUMNS]].drop_duplicates(
        subset=["season", "team_abbr"], keep="last"
    )


def _frame_to_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        payload: dict[str, object] = {
            "season": int(row.season),
            "team_abbr": row.team_abbr,
            "venue_id": None if pd.isna(row.venue_id) else int(row.venue_id),
            "venue_name": row.venue_name,
            "source_name": row.source_name,
        }
        for column in FACTOR_COLUMNS:
            value = getattr(row, column)
            payload[column] = None if pd.isna(value) else float(value)
        rows.append(payload)
    return rows


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        log.info("No park factor CSV found at %s", csv_path)
        return _empty_frame()
    frame = pd.read_csv(csv_path)
    normalized = _normalize_frame(frame)
    log.info("Loaded %s park factor rows from %s", len(normalized), csv_path)
    return normalized


def _latest_home_venues() -> pd.DataFrame:
    teams = query_df("SELECT team_abbr FROM dim_teams ORDER BY team_abbr")
    if teams.empty:
        return pd.DataFrame(columns=["team_abbr", "venue_id", "venue_name"])
    games = query_df(
        """
        SELECT home_team AS team_abbr, venue_id, venue_name, game_date, season
        FROM games
        WHERE venue_id IS NOT NULL
        """
    )
    if games.empty:
        teams["venue_id"] = None
        teams["venue_name"] = None
        return teams
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.sort_values(["team_abbr", "game_date"])
    latest = games.drop_duplicates(subset=["team_abbr"], keep="last")
    return teams.merge(latest[["team_abbr", "venue_id", "venue_name"]], on="team_abbr", how="left")


def _build_bootstrap_frame(target_season: int, fallback_season: int) -> pd.DataFrame:
    venue_lookup = _latest_home_venues()
    existing = query_df("SELECT * FROM park_factors ORDER BY season, team_abbr")
    existing = _normalize_frame(existing) if not existing.empty else _empty_frame()

    fallback = existing[existing["season"] == fallback_season].copy()
    if fallback.empty and not existing.empty:
        fallback = existing.sort_values("season").drop_duplicates(subset=["team_abbr"], keep="last")

    if venue_lookup.empty:
        if fallback.empty:
            return _empty_frame()
        fallback_rows = fallback.copy()
        fallback_rows["season"] = target_season
        fallback_rows["source_name"] = f"bootstrap_{fallback_season}"
        return fallback_rows[["season", "team_abbr", *OPTIONAL_COLUMNS, *FACTOR_COLUMNS]].drop_duplicates(
            subset=["season", "team_abbr"],
            keep="last",
        )

    merged = venue_lookup.merge(
        fallback[["team_abbr", "venue_id", "venue_name", "source_name", *FACTOR_COLUMNS]],
        on="team_abbr",
        how="left",
        suffixes=("_game", "_factor"),
    )

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        has_fallback = any(pd.notna(getattr(row, column)) for column in FACTOR_COLUMNS)
        payload: dict[str, object] = {
            "season": target_season,
            "team_abbr": row.team_abbr,
            "venue_id": None,
            "venue_name": None,
            "source_name": f"bootstrap_{fallback_season}" if has_fallback else "neutral_bootstrap",
        }
        venue_id_game = getattr(row, "venue_id_game", None)
        venue_id_factor = getattr(row, "venue_id_factor", None)
        if pd.notna(venue_id_game):
            payload["venue_id"] = int(venue_id_game)
        elif pd.notna(venue_id_factor):
            payload["venue_id"] = int(venue_id_factor)
        venue_name_game = getattr(row, "venue_name_game", None)
        venue_name_factor = getattr(row, "venue_name_factor", None)
        payload["venue_name"] = venue_name_game if pd.notna(venue_name_game) else venue_name_factor
        for column in FACTOR_COLUMNS:
            value = getattr(row, column)
            payload[column] = float(value) if pd.notna(value) else 1.0
        rows.append(payload)
    return pd.DataFrame(rows, columns=["season", "team_abbr", *OPTIONAL_COLUMNS, *FACTOR_COLUMNS])


def ensure_park_factors_seeded(
    *,
    csv_path: Path | None = None,
    source: str = "auto",
    target_season: int | None = None,
    fallback_season: int | None = None,
    skip_bootstrap: bool = False,
    write_seed: bool = False,
    force_bootstrap: bool = False,
) -> dict[str, int | bool]:
    settings = get_settings()
    resolved_csv_path = csv_path or settings.park_factors_csv
    resolved_target_season = target_season or settings.current_season
    resolved_fallback_season = fallback_season or settings.prior_season

    imported = 0
    imported_frame = _empty_frame()
    if source in {"auto", "csv"}:
        imported_frame = _load_csv(resolved_csv_path)
    if imported_frame.empty and source in {"auto", "statcast"}:
        imported_frame = _fetch_statcast_frame(resolved_fallback_season)
        if write_seed or not resolved_csv_path.exists():
            _write_seed_file(imported_frame, resolved_csv_path)
            log.info("Wrote %s park factor rows to %s", len(imported_frame), resolved_csv_path)
    if not imported_frame.empty:
        imported = upsert_rows("park_factors", _frame_to_rows(imported_frame), ["season", "team_abbr"])
        log.info("Imported %s park factor rows into the database", imported)

    if skip_bootstrap:
        return {
            "imported": imported,
            "bootstrapped": 0,
            "target_ready": imported > 0,
            "bootstrap_attempted": False,
        }

    existing_target = query_df(
        "SELECT COUNT(*) AS row_count FROM park_factors WHERE season = :season",
        {"season": resolved_target_season},
    )
    if int(existing_target.iloc[0]["row_count"]) > 0 and not force_bootstrap:
        log.info("Park factors for season %s already exist; skipping bootstrap", resolved_target_season)
        return {
            "imported": imported,
            "bootstrapped": 0,
            "target_ready": True,
            "bootstrap_attempted": False,
        }

    bootstrap = _build_bootstrap_frame(resolved_target_season, resolved_fallback_season)
    if bootstrap.empty:
        log.warning("Could not bootstrap park factors for season %s; no team or venue data found", resolved_target_season)
        return {
            "imported": imported,
            "bootstrapped": 0,
            "target_ready": False,
            "bootstrap_attempted": True,
        }

    inserted = upsert_rows("park_factors", _frame_to_rows(bootstrap), ["season", "team_abbr"])
    log.info(
        "Bootstrapped %s park factor rows for season %s using fallback season %s",
        inserted,
        resolved_target_season,
        resolved_fallback_season,
    )
    return {
        "imported": imported,
        "bootstrapped": inserted,
        "target_ready": inserted > 0,
        "bootstrap_attempted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load park factors from CSV and bootstrap the current season")
    parser.add_argument("--csv", help="Override park factors csv path")
    parser.add_argument(
        "--source",
        choices=["auto", "csv", "statcast"],
        default="auto",
        help="Reference source to use before bootstrapping current-season rows",
    )
    parser.add_argument("--target-season", type=int, help="Season to ensure is populated")
    parser.add_argument("--fallback-season", type=int, help="Season to copy from when bootstrapping")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Only import CSV rows; do not bootstrap")
    parser.add_argument("--write-seed", action="store_true", help="Write fetched reference data back to the configured CSV path")
    parser.add_argument("--force-bootstrap", action="store_true", help="Rebuild the target season even if rows already exist")
    args = parser.parse_args()

    ensure_park_factors_seeded(
        csv_path=Path(args.csv) if args.csv else None,
        source=args.source,
        target_season=args.target_season,
        fallback_season=args.fallback_season,
        skip_bootstrap=args.skip_bootstrap,
        write_seed=args.write_seed,
        force_bootstrap=args.force_bootstrap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())