#!/usr/bin/env python3
"""
Backfill a date range into Postgres using your existing ingestors,
then (optionally) run predictions for each day so the frontend can show history.

Examples:
  python backfill_range.py --start 2025-07-15 --end 2025-08-14 --predict
  python backfill_range.py --start 2025-07-15 --end 2025-08-14 --market-ingestor enhanced --no-weather
"""
import sys, subprocess, logging, os
from datetime import datetime, timedelta
import argparse
from pathlib import Path
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("backfill")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

# Map each script to the date flag it understands
DATE_FLAG = {
    "working_games_ingestor.py": "--target-date",
    "working_pitcher_ingestor.py": "--target-date",
    "working_team_ingestor.py": "--target-date",
    "working_weather_ingestor.py": "--target-date",  # adjust if your weather script expects --date
    "enhanced_market_collector.py": "--date",
    "real_market_ingestor.py": "--date",
    "daily_api_workflow.py": "--date",
}

# Child env to avoid Windows console Unicode crashes (emoji prints)
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

def run(cmd):
    log.info("» %s", " ".join(cmd))
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, env=CHILD_ENV, encoding='utf-8', errors='replace')
        if cp.stdout and cp.stdout.strip():
            # keep debug-level to avoid noisy logs unless needed
            log.debug(cp.stdout.rstrip())
        if cp.returncode != 0:
            log.error("Command failed (code %s): %s", cp.returncode, " ".join(cmd))
            if cp.stderr and cp.stderr.strip():
                log.error(cp.stderr.rstrip())
        return cp.returncode
    except Exception as e:
        log.error("Command execution error: %s", e)
        return 1

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def verify_day(engine, ds):
    """Light sanity checks so you know data actually landed."""
    with engine.begin() as conn:
        q = lambda sql: conn.execute(text(sql), {"d": ds}).scalar() or 0

        lgf_total = q("""SELECT COUNT(*) FROM legitimate_game_features WHERE "date" = :d""")
        eg_seeded = q("""SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d""")
        mk_count  = q("""SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d AND market_total IS NOT NULL""")
        sp_named  = q("""SELECT COUNT(*) FROM enhanced_games
                         WHERE "date" = :d AND home_sp_name IS NOT NULL AND away_sp_name IS NOT NULL""")
        preds     = q("""SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d AND predicted_total IS NOT NULL""")
        finals    = q("""SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d AND total_runs IS NOT NULL""")

    log.info("✔ %s summary: LGF=%d | EG=%d | markets=%d | SP named=%d | preds=%d | finals=%d",
             ds, lgf_total, eg_seeded, mk_count, sp_named, preds, finals)

def call_with_date(script_path: Path, date_str: str):
    """Call a script with the correct date flag based on its filename."""
    name = script_path.name
    flag = DATE_FLAG.get(name, "--date")
    return run([sys.executable, str(script_path), flag, date_str])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--predict", dest="predict", action="store_true",
                    help="Also run model predictions per day")
    ap.add_argument("--no-predict", dest="predict", action="store_false")
    ap.add_argument("--no-weather", action="store_true",
                    help="Skip weather backfill if your weather source lacks history")
    ap.add_argument("--market-ingestor", choices=["enhanced","real"], default="enhanced",
                    help="Which market ingestor to use")
    ap.set_defaults(predict=False)
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
    if end < start:
        log.error("End date must be >= start date"); sys.exit(2)

    here = Path(__file__).resolve().parent
    py = sys.executable

    # Ingestors live in data_collection
    data_collection_dir = here.parent / "data_collection"
    games_ing   = data_collection_dir / "working_games_ingestor.py"
    market_ing  = data_collection_dir / ("enhanced_market_collector.py" if args.market_ingestor=="enhanced" else "real_market_ingestor.py")
    pitcher_ing = data_collection_dir / "working_pitcher_ingestor.py"
    team_ing    = data_collection_dir / "working_team_ingestor.py"
    weather_ing = data_collection_dir / "working_weather_ingestor.py"
    daily       = here / "daily_api_workflow.py"

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    total_days = (end - start).days + 1
    log.info("Starting backfill for %d days: %s → %s", total_days, start, end)

    failures = 0

    for i, d in enumerate(daterange(start, end), 1):
        ds = d.strftime("%Y-%m-%d")
        log.info("===== Backfilling %s (%d/%d) =====", ds, i, total_days)

        # 1) GAMES & FINAL SCORES
        log.info("1/6: Games & scores for %s", ds)
        failures += call_with_date(games_ing, ds) != 0

        # 2) MARKETS
        log.info("2/6: Market data for %s (%s)", ds, args.market_ingestor)
        failures += call_with_date(market_ing, ds) != 0

        # 3) PITCHERS
        log.info("3/6: Pitcher data for %s", ds)
        failures += call_with_date(pitcher_ing, ds) != 0

        # 4) TEAMS
        log.info("4/6: Team data for %s", ds)
        failures += call_with_date(team_ing, ds) != 0

        # 5) WEATHER (optional)
        if not args.no_weather:
            log.info("5/6: Weather data for %s", ds)
            failures += call_with_date(weather_ing, ds) != 0
        else:
            log.info("5/6: Skipping weather for %s (--no-weather)", ds)

        # 6) PREDICTIONS (optional)
        if args.predict:
            log.info("6/6: Generating predictions for %s", ds)
            failures += run([py, str(daily), DATE_FLAG["daily_api_workflow.py"], ds, "--stages", "features,predict"]) != 0
        else:
            log.info("6/6: Skipping predictions for %s (--no-predict)", ds)

        # Verify this day landed in DB
        verify_day(engine, ds)

        log.info("Completed %s (%d/%d)", ds, i, total_days)

    log.info("✅ Backfill complete: %s → %s (%d days processed, failures=%d)",
             args.start, args.end, total_days, failures)

    log.info("Next: retrain on the backfilled window, e.g.:")
    log.info("  python retrain_model.py --end %s --window-days %d --holdout-days 7 --audit --deploy",
             args.end, total_days)

if __name__ == "__main__":
    main()
