"""Cross-source validator — checks data completeness per game for a target date."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)


def _is_stale(iso_ts: object, *, max_age_hours: int) -> bool:
    if not iso_ts:
        return True
    raw = str(iso_ts).replace("Z", "+00:00")
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts > timedelta(hours=max_age_hours)


def validate_slate(target_date: str) -> list[dict[str, Any]]:
    """Run all validation checks and return one row per game."""
    games = query_df(
        """
        SELECT g.game_id, g.game_date, g.away_team, g.home_team,
               g.venue_id, g.status, g.game_start_ts
        FROM games g
        WHERE g.game_date = :target_date
          AND g.game_type = 'R'
        ORDER BY g.game_start_ts
        """,
        {"target_date": target_date},
    )
    if games.empty:
        log.info("No games found for %s", target_date)
        return []

    # --- Starters ---
    starters = query_df(
        "SELECT game_id, team FROM pitcher_starts WHERE game_date = :d",
        {"d": target_date},
    )
    starters_by_game: dict[int, set[str]] = {}
    for _, row in starters.iterrows():
        starters_by_game.setdefault(int(row["game_id"]), set()).add(row["team"])

    # --- Markets ---
    markets = query_df(
        "SELECT game_id, line_value, snapshot_ts FROM game_markets WHERE game_date = :d AND market_type = 'total'",
        {"d": target_date},
    )
    games_with_market = set(markets["game_id"].tolist()) if not markets.empty else set()
    latest_market_by_game: dict[int, dict[str, Any]] = {}
    if not markets.empty:
        sorted_markets = markets.sort_values("snapshot_ts")
        for _, row in sorted_markets.iterrows():
            latest_market_by_game[int(row["game_id"])] = row.to_dict()
    any_market_rows = query_df(
        "SELECT DISTINCT game_id FROM game_markets WHERE game_date = :d AND market_type = 'total'",
        {"d": target_date},
    )
    games_with_any_market_rows = set(any_market_rows["game_id"].tolist()) if not any_market_rows.empty else set()

    # --- Venue ---
    venues = query_df(
        """
        SELECT g.game_id, v.venue_id, v.latitude
        FROM games g LEFT JOIN dim_venues v ON v.venue_id = g.venue_id
        WHERE g.game_date = :d
        """,
        {"d": target_date},
    )
    venue_mapped: dict[int, bool] = {}
    for _, row in venues.iterrows():
        venue_mapped[int(row["game_id"])] = row.get("latitude") is not None

    # --- Lineups ---
    lineups = query_df(
        """
        SELECT game_id, team, COUNT(*) as cnt,
               MIN(lineup_slot) AS min_slot,
               MAX(lineup_slot) AS max_slot,
               COUNT(DISTINCT lineup_slot) AS slot_count
        FROM lineups
        WHERE game_date = :d
        GROUP BY game_id, team
        """,
        {"d": target_date},
    )
    lineups_by_game: dict[int, dict[str, dict[str, int]]] = {}
    for _, row in lineups.iterrows():
        lineups_by_game.setdefault(int(row["game_id"]), {})[str(row["team"])] = {
            "cnt": int(row["cnt"]),
            "min_slot": int(row["min_slot"]),
            "max_slot": int(row["max_slot"]),
            "slot_count": int(row["slot_count"]),
        }

    # --- Weather ---
    weather = query_df(
        "SELECT game_id, temperature_f, wind_speed_mph, snapshot_ts FROM game_weather WHERE game_date = :d",
        {"d": target_date},
    )
    games_with_weather = set(weather["game_id"].tolist()) if not weather.empty else set()
    latest_weather_by_game: dict[int, dict[str, Any]] = {}
    if not weather.empty:
        sorted_weather = weather.sort_values("snapshot_ts")
        for _, row in sorted_weather.iterrows():
            latest_weather_by_game[int(row["game_id"])] = row.to_dict()

    freshness = query_df(
        """
        SELECT source_name, MAX(ingested_at) AS latest_ingested_at
        FROM raw_ingest_events
        WHERE DATE(ingested_at) = :d
          AND source_name IN ('market_totals', 'open_meteo', 'lineup_csv')
        GROUP BY source_name
        """,
        {"d": target_date},
    )
    freshness_by_source = {
        str(row["source_name"]): row.get("latest_ingested_at")
        for _, row in freshness.iterrows()
    }

    # --- Build results ---
    results: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        gid = int(game["game_id"])
        away = game["away_team"]
        home = game["home_team"]
        starter_teams = starters_by_game.get(gid, set())
        lineup_teams = lineups_by_game.get(gid, {})
        latest_market = latest_market_by_game.get(gid, {})
        latest_weather = latest_weather_by_game.get(gid, {})

        has_away_starter = away in starter_teams
        has_home_starter = home in starter_teams
        has_market = gid in games_with_market
        has_venue = venue_mapped.get(gid, False)
        away_lineup = lineup_teams.get(away, {})
        home_lineup = lineup_teams.get(home, {})
        has_away_lineup = away_lineup.get("cnt", 0) >= 9
        has_home_lineup = home_lineup.get("cnt", 0) >= 9
        has_weather = gid in games_with_weather

        warnings: list[str] = []
        base_failures = 0
        if not has_away_starter:
            warnings.append("away_starter_missing")
            base_failures += 1
        if not has_home_starter:
            warnings.append("home_starter_missing")
            base_failures += 1
        if not has_market:
            warnings.append("market_only_post_start" if gid in games_with_any_market_rows else "no_market_line")
            base_failures += 1
        if not has_venue:
            warnings.append("venue_not_mapped")
            base_failures += 1
        if not has_away_lineup:
            warnings.append("away_lineup_incomplete")
            base_failures += 1
        if not has_home_lineup:
            warnings.append("home_lineup_incomplete")
            base_failures += 1
        if not has_weather:
            warnings.append("no_weather")
            base_failures += 1

        if has_market:
            market_total = latest_market.get("line_value")
            if market_total is not None and (float(market_total) < 5.0 or float(market_total) > 14.0):
                warnings.append("implausible_market_total")
        if has_weather:
            temp_f = latest_weather.get("temperature_f")
            wind_mph = latest_weather.get("wind_speed_mph")
            if (temp_f is not None and (float(temp_f) < 20.0 or float(temp_f) > 110.0)) or (
                wind_mph is not None and float(wind_mph) > 60.0
            ):
                warnings.append("implausible_weather")
        if has_away_lineup and not (
            away_lineup.get("min_slot") == 1 and away_lineup.get("max_slot") == 9 and away_lineup.get("slot_count") == 9
        ):
            warnings.append("away_lineup_slots_invalid")
        if has_home_lineup and not (
            home_lineup.get("min_slot") == 1 and home_lineup.get("max_slot") == 9 and home_lineup.get("slot_count") == 9
        ):
            warnings.append("home_lineup_slots_invalid")

        target_dt = date.fromisoformat(target_date)
        if target_dt >= date.today():
            if _is_stale(freshness_by_source.get("market_totals"), max_age_hours=4):
                warnings.append("market_source_stale")
            if _is_stale(freshness_by_source.get("open_meteo"), max_age_hours=6):
                warnings.append("weather_source_stale")
            if _is_stale(freshness_by_source.get("lineup_csv"), max_age_hours=6):
                warnings.append("lineup_source_stale")

        check_count = 7
        pass_count = check_count - base_failures
        if pass_count == check_count and not warnings:
            badge = "green"
        elif pass_count >= 4:
            badge = "yellow"
        else:
            badge = "red"

        results.append({
            "game_id": gid,
            "game_date": target_date,
            "away_team": away,
            "home_team": home,
            "has_away_starter": has_away_starter,
            "has_home_starter": has_home_starter,
            "has_market": has_market,
            "has_venue": has_venue,
            "has_away_lineup": has_away_lineup,
            "has_home_lineup": has_home_lineup,
            "has_weather": has_weather,
            "checks_passed": pass_count,
            "checks_total": check_count,
            "warnings": ";".join(warnings) if warnings else None,
            "badge": badge,
        })
    return results


def persist_validation(results: list[dict[str, Any]]) -> int:
    """Write validation results to game_readiness table."""
    if not results or not table_exists("game_readiness"):
        return 0
    return upsert_rows("game_readiness", results, ["game_id", "game_date"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data completeness for a slate")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    all_results: list[dict[str, Any]] = []
    current = start_date
    while current <= end_date:
        results = validate_slate(current.isoformat())
        all_results.extend(results)
        current += __import__("datetime").timedelta(days=1)

    if all_results:
        written = persist_validation(all_results)
        log.info("Validated %s games, wrote %s readiness rows", len(all_results), written)
        greens = sum(1 for r in all_results if r["badge"] == "green")
        yellows = sum(1 for r in all_results if r["badge"] == "yellow")
        reds = sum(1 for r in all_results if r["badge"] == "red")
        log.info("Readiness: %s green, %s yellow, %s red", greens, yellows, reds)
    else:
        log.info("No games to validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
