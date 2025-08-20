#!/usr/bin/env python3
"""
Complete Historical Backfill Workflow
====================================

This script orchestrates the complete historical data collection for MLB predictions:
1. Games, pitchers, teams, weather data collection
2. Real historical market totals (if The Odds API key available)
3. Retro predictions for all games
4. Model evaluation and reporting

Usage:
  python complete_historical_workflow.py --start 2025-05-01 --end 2025-08-15 --use-real-markets

Requirements:
- THE_ODDS_API_KEY (for real historical markets, paid plan required)
- OPENWEATHER_API_KEY (for historical weather data)
- DATABASE_URL (PostgreSQL connection)
"""

import os, sys, argparse, subprocess, time
from datetime import datetime
from pathlib import Path

# Use current interpreter and force UTF-8 for child processes
PY = sys.executable
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

def run_command(cmd, description, cwd=None, fail_fast=False, logfile=None):
    """Run a command, stream output live, and optionally tee to a logfile."""
    # Make Python children unbuffered
    if cmd and (cmd[0].endswith("python") or cmd[0].endswith("python.exe") or cmd[0] == PY):
        cmd = cmd[:1] + ["-u"] + cmd[1:]

    print("\n" + "="*60)
    print(f"STEP: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("="*60, flush=True)

    f = open(logfile, "a", encoding="utf-8") if logfile else None
    try:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=CHILD_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        last = time.time()
        for line in p.stdout:
            line = line.rstrip("\n")
            print(line)
            if f: f.write(line + "\n")
            last = time.time()

        p.wait()
        if p.returncode == 0:
            print(f"✅ DONE: {description}\n", flush=True)
            return True
        else:
            print(f"❌ FAILED (exit {p.returncode}): {description}\n", flush=True)
            if fail_fast:
                sys.exit(p.returncode)
            return False
    finally:
        if f: f.close()

def check_environment():
    """Check required environment variables"""
    required_vars = ["DATABASE_URL"]
    optional_vars = ["THE_ODDS_API_KEY", "OPENWEATHER_API_KEY"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        print(f"ERROR: Missing required environment variables: {missing_required}")
        return False
    
    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_optional:
        print(f"WARNING: Missing optional environment variables: {missing_optional}")
        print("This may limit functionality (real markets, weather data)")
    
    return True

def create_frontend_view():
    """Create the API view for frontend consumption"""
    from sqlalchemy import create_engine, text
    
    engine = create_engine(os.getenv('DATABASE_URL'))
    
    view_sql = """
    CREATE OR REPLACE VIEW api_games_today AS
    SELECT eg.game_id, eg.date, eg.home_team, eg.away_team, eg.game_time_utc,
           eg.venue_name, eg.roof_type, eg.market_total, eg.predicted_total,
           ROUND(eg.predicted_total - eg.market_total,2) AS edge,
           CASE WHEN eg.predicted_total IS NULL OR eg.market_total IS NULL THEN NULL
                WHEN (eg.predicted_total - eg.market_total) >=  1.0 THEN 'OVER'
                WHEN (eg.predicted_total - eg.market_total) <= -1.0 THEN 'UNDER'
                ELSE 'NO BET' END AS recommendation,
           eg.home_sp_name, eg.away_sp_name, eg.home_sp_season_era, eg.away_sp_season_era,
           eg.temperature, eg.wind_speed, eg.wind_direction, eg.humidity
    FROM enhanced_games eg
    WHERE eg.date = CURRENT_DATE AND eg.total_runs IS NULL
    ORDER BY eg.game_time_utc NULLS LAST, eg.game_id
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(view_sql))
        print("✅ Created api_games_today view")
        return True
    except Exception as e:
        print(f"❌ Failed to create view: {e}")
        return False

def verify_coverage(start_date, end_date):
    """Check data coverage for the backfilled period"""
    from sqlalchemy import create_engine, text
    
    engine = create_engine(os.getenv('DATABASE_URL'))
    
    query = """
    SELECT "date",
           COUNT(*) AS games,
           COUNT(market_total) AS markets,
           COUNT(predicted_total) AS preds,
           COUNT(total_runs) AS finals
    FROM enhanced_games
    WHERE "date" BETWEEN :start AND :end
    GROUP BY 1 ORDER BY 1
    """
    
    try:
        with engine.begin() as conn:
            result = conn.execute(text(query), {"start": start_date, "end": end_date}).fetchall()
        
        print(f"\n{'='*80}")
        print("DATA COVERAGE SUMMARY")
        print(f"{'='*80}")
        print(f"{'Date':<12} {'Games':<6} {'Markets':<8} {'Preds':<6} {'Finals':<7}")
        print("-" * 80)
        
        total_games = total_markets = total_preds = total_finals = 0
        for row in result:
            date, games, markets, preds, finals = row
            total_games += games
            total_markets += markets
            total_preds += preds
            total_finals += finals
            print(f"{date:<12} {games:<6} {markets:<8} {preds:<6} {finals:<7}")
        
        print("-" * 80)
        print(f"{'TOTAL':<12} {total_games:<6} {total_markets:<8} {total_preds:<6} {total_finals:<7}")
        
        market_coverage = (total_markets / total_games * 100) if total_games > 0 else 0
        pred_coverage = (total_preds / total_games * 100) if total_games > 0 else 0
        finals_coverage = (total_finals / total_games * 100) if total_games > 0 else 0
        
        print(f"\nCOVERAGE PERCENTAGES:")
        print(f"Markets: {market_coverage:.1f}%")
        print(f"Predictions: {pred_coverage:.1f}%")
        print(f"Finals: {finals_coverage:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Failed to verify coverage: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete historical backfill workflow")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--use-real-markets", action="store_true", help="Use The Odds API for real historical markets (requires paid plan)")
    parser.add_argument("--bookmaker", default="fanduel", help="Bookmaker for real markets (fanduel, draftkings, etc.)")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather data collection")
    parser.add_argument("--skip-predictions", action="store_true", help="Skip prediction generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation")
    parser.add_argument("--threshold", type=float, default=1.0, help="Edge threshold for OVER/UNDER recommendations")
    parser.add_argument("--fail-fast", action="store_true", help="Exit immediately on first error instead of continuing")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        return 1
    
    if end_date < start_date:
        print("ERROR: End date must be after start date")
        return 1
    
    # Check environment
    if not check_environment():
        return 1
    
    # Get script directory and create logs directory
    script_dir = Path(__file__).parent
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting complete historical backfill: {args.start} to {args.end}")
    print(f"   Real markets: {args.use_real_markets}")
    print(f"   Skip weather: {args.skip_weather}")
    print(f"   Skip predictions: {args.skip_predictions}")
    print(f"   Skip evaluation: {args.skip_evaluation}")
    print(f"   Fail fast: {args.fail_fast}")
    print(f"   Logs directory: {logs_dir}")
    
    success = True
    
    # Step 1: Create frontend view
    print("\n🔧 Creating frontend view...")
    if not create_frontend_view():
        if args.fail_fast:
            print("FAIL-FAST: Exiting due to view creation failure")
            return 1
        success = False
    
    # Step 2: Seed games/pitchers/teams/weather with enhanced markets
    market_ingestor = "enhanced"
    weather_flag = ["--no-weather"] if args.skip_weather else []
    predict_flag = ["--no-predict"] if args.skip_predictions else []
    
    cmd = [
        PY, "backfill_range.py",
        "--start", args.start,
        "--end", args.end,
        "--market-ingestor", market_ingestor
    ] + weather_flag + predict_flag
    
    if not run_command(cmd, "Seed games/pitchers/teams/weather data", cwd=script_dir, 
                       fail_fast=args.fail_fast, logfile=logs_dir / "backfill_seed.log"):
        success = False
    
    # Step 3: Real historical markets (if requested and API key available)
    if args.use_real_markets and os.getenv("THE_ODDS_API_KEY"):
        cmd = [
            PY, "historical_markets_backfill.py",
            "--start", args.start,
            "--end", args.end,
            "--bookmaker", args.bookmaker,
            "--sleep", "0.8"
        ]
        
        if not run_command(cmd, f"Fetch real historical markets ({args.bookmaker})", cwd=script_dir, 
                           fail_fast=False, logfile=logs_dir / "historical_markets.log"):
            print("⚠️  Real markets failed, continuing with enhanced markets")
    elif args.use_real_markets:
        print("⚠️  THE_ODDS_API_KEY not set, skipping real markets")
    
    # Step 4: Retro predictions (if not skipped)
    if not args.skip_predictions:
        cmd = [
            PY, "predict_from_range.py",
            "--start", args.start,
            "--end", args.end,
            "--thr", str(args.threshold)
        ]
        
        if not run_command(cmd, "Generate retro predictions", cwd=script_dir, 
                           fail_fast=args.fail_fast, logfile=logs_dir / "predictions.log"):
            success = False
    
    # Step 5: Model evaluation (if not skipped)
    if not args.skip_evaluation:
        # Check if evaluate_backfill.py exists
        eval_script = script_dir / "evaluate_backfill.py"
        if eval_script.exists():
            cmd = [
                PY, "evaluate_backfill.py",
                "--start", args.start,
                "--end", args.end,
                "--threshold", str(args.threshold),
                "--odds", "-110"
            ]
            
            if not run_command(cmd, "Evaluate model performance", cwd=script_dir, 
                               fail_fast=False, logfile=logs_dir / "evaluation.log"):
                print("⚠️  Evaluation failed, but continuing")
        else:
            print("⚠️  evaluate_backfill.py not found, skipping evaluation")
    
    # Step 6: Verify coverage
    print("\n📊 Verifying data coverage...")
    verify_coverage(args.start, args.end)
    
    # Summary
    print(f"\n{'='*80}")
    if success:
        print("✅ COMPLETE HISTORICAL BACKFILL FINISHED SUCCESSFULLY!")
    else:
        print("⚠️  BACKFILL COMPLETED WITH SOME ERRORS")
    
    print(f"📅 Period: {args.start} to {args.end}")
    print(f"💰 Real markets: {'Yes' if args.use_real_markets and os.getenv('THE_ODDS_API_KEY') else 'No'}")
    print(f"🌤️  Weather data: {'No' if args.skip_weather else 'Yes'}")
    print(f"🎯 Predictions: {'No' if args.skip_predictions else 'Yes'}")
    print(f"📈 Evaluation: {'No' if args.skip_evaluation else 'Yes'}")
    print(f"{'='*80}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
