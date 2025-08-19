#!/usr/bin/env python3
"""
Daily Production Runbook

Comprehensive script that executes the full daily betting workflow:
1. Feature engineering and data validation
2. Odds collection (or market total fallback)  
3. Probability calculation with risk controls
4. Analysis and reporting
5. Optional: outcomes logging for previous days

Usage:
  python daily_runbook.py --date 2025-08-17 --mode predictions
  python daily_runbook.py --date 2025-08-16 --mode outcomes --fail-fast
"""

import argparse
import subprocess
import sys
import os
import shlex
import datetime
from datetime import datetime as dt, timedelta
from pathlib import Path

# Anchor paths to script directory
BASE = Path(__file__).resolve().parent

def run_command(argv, description, cwd=None, env=None):
    """Run a command (list form) with streaming output and capture to log."""
    from subprocess import Popen, PIPE
    
    print(f"\n🚀 {description}")
    print(f"   Command: {' '.join(shlex.quote(x) for x in argv)}")

    # Ensure UTF-8 encoding for subprocess environment (Windows Unicode safety)
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        subprocess_env.update(env)

    logs_dir = BASE / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"{dt.now().strftime('%Y%m%d_%H%M%S')}_{argv[1].replace('.py','')}.log"

    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            p = Popen(argv, cwd=cwd or BASE, env=subprocess_env, stdout=PIPE, stderr=PIPE, text=True, encoding='utf-8', errors='replace')
            # stream both stdout and stderr
            out, err = p.communicate()
            logf.write(out or "")
            logf.write(err or "")
            rc = p.returncode
        except Exception as e:
            print(f"❌ {description} failed with exception: {e}")
            return False

    if rc == 0:
        print(f"✅ {description} completed successfully")
        # show tail of stdout if present, otherwise tail of stderr
        tail = (out or err or "").strip().splitlines()[-10:]
        for line in tail:
            print(f"   {line}")
        print(f"   📝 Log: {log_path}")
        return True
    else:
        print(f"❌ {description} failed (rc={rc})")
        print(f"   📝 Log: {log_path}")
        # print last few error lines to console
        tail_err = (err or out or "").strip().splitlines()[-10:]
        for line in tail_err:
            print(f"   {line}")
        return False

def validate_environment():
    """Check that required files and environment are ready"""
    print("🔍 Validating environment...")
    
    # Anchored paths relative to script directory
    required_files = [
        BASE / "daily_api_workflow.py",
        BASE / "enhanced_analysis.py", 
        BASE / "enhanced_bullpen_predictor.py",
        BASE / "production_status.py",
        BASE.parent / "data_collection" / "working_games_ingestor.py",
        BASE.parent / "data_collection" / "real_market_ingestor.py",
        BASE.parent / "data_collection" / "weather_ingestor.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not file_path.exists():
            missing.append(str(file_path.relative_to(BASE.parent)))
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    # Check database connection
    try:
        from sqlalchemy import create_engine, text
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

def run_predictions_workflow(target_date, fail_fast=False):
    """Execute the full predictions workflow"""
    print(f"\n🎯 DAILY PREDICTIONS WORKFLOW for {target_date}")
    print("=" * 60)
    
    py = sys.executable
    
    # Step 1: Feature engineering
    if not run_command(
        [py, "daily_api_workflow.py", "--stages", "features", "--target-date", target_date],
        "Feature engineering and data validation"
    ):
        return False
    
    # Step 2: Health gate check (calibration drift)
    ok = run_command(
        [py, "health_gate.py", "--date", target_date, "--days", "30"],
        "Calibration health check"
    )
    if not ok:
        if fail_fast:
            print("❌ Health gate failed and --fail-fast enabled")
            return False
        else:
            print("⚠️  Health gate failed but continuing")
    
    # Step 3: Enhanced predictions with risk controls
    if not run_command(
        [py, "probabilities_and_ev.py", "--date", target_date, "--model-version", "enhanced_bullpen_v1"],
        "Probability calculation and bet sizing"
    ):
        return False
    
    # Step 4: Analysis and reporting
    if not run_command(
        [py, "enhanced_analysis.py"],
        "Enhanced analysis and traceability"
    ):
        return False
    
    print(f"\n🎉 Predictions workflow completed for {target_date}")
    print("📋 Next steps:")
    print("   1. Review betting recommendations above")
    print("   2. Validate park factors and starting pitcher stats")
    print("   3. Cross-reference with real-time odds feeds")
    print("   4. Execute trades with conservative position sizing")
    print(f"   5. Run outcomes logging tomorrow: python daily_runbook.py --date {target_date} --mode outcomes")
    
    return True

def run_outcomes_workflow(target_date):
    """Log outcomes for completed games"""
    print(f"\n📊 OUTCOMES LOGGING WORKFLOW for {target_date}")
    print("=" * 60)
    
    py = sys.executable
    
    # Log CLV and outcomes
    if not run_command(
        [py, "log_outcomes.py", "--date", target_date],
        "CLV and outcomes logging"
    ):
        return False
    
    # Run reliability analysis for last 30 days (soft-fail allowed)
    ok = run_command(
        [py, "reliability_brier.py", "--end", target_date, "--days", "30", "--model-version", "enhanced_bullpen_v1"],
        "30-day reliability and Brier analysis"
    )
    if not ok:
        print("⚠️  Reliability analysis failed (may be normal if insufficient data)")
    
    print(f"\n🎉 Outcomes workflow completed for {target_date}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Daily production runbook for MLB betting system")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["predictions", "outcomes"], default="predictions",
                       help="Workflow mode: predictions (live betting) or outcomes (post-game analysis)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip environment validation")
    parser.add_argument("--fail-fast", action="store_true", help="Stop if the health gate fails")
    parser.add_argument("--force", action="store_true", help="Override date validation checks")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        target_date = dt.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return 1
    
    # Date sanity checks (unless --force)
    if not args.force:
        today = dt.now().date()
        if args.mode == "predictions" and target_date < today:
            print(f"❌ Cannot run predictions for past date {target_date} (use --force to override)")
            return 1
        elif args.mode == "outcomes" and target_date >= today:
            print(f"❌ Cannot log outcomes for future date {target_date} (use --force to override)")
            return 1
    
    print(f"🎯 MLB BETTING SYSTEM - Daily Runbook")
    print(f"📅 Date: {args.date}")
    print(f"🔧 Mode: {args.mode}")
    print(f"🕒 Started: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.fail_fast:
        print(f"⚡ Fail-fast mode enabled")
    
    # Environment validation
    if not args.skip_validation and not validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    # Execute workflow
    if args.mode == "predictions":
        success = run_predictions_workflow(args.date, args.fail_fast)
    else:
        success = run_outcomes_workflow(args.date)
    
    if success:
        print(f"\n🎉 Daily runbook completed successfully!")
        return 0
    else:
        print(f"\n❌ Daily runbook failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
