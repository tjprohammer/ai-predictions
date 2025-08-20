#!/usr/bin/env python3
"""
complete_backfill_retrain.py
=============================
Complete workflow: backfill historical data → check coverage → retrain → deploy

This script demonstrates the full historical learning loop:
1. Backfill a month of historical data
2. Check training data coverage  
3. Retrain model on backfilled data
4. Deploy new model if validation passes

Examples:
  # Backfill last month and retrain
  python complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14

  # Include predictions during backfill
  python complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14 --predict

  # Skip weather data (if historical weather unavailable)
  python complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14 --no-weather
"""
import sys
import os
import subprocess
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("complete_workflow")

def run_command(cmd, description, timeout=3600):
    """Run a command with logging and error handling."""
    log.info(f"🚀 {description}")
    log.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        
        if result.returncode != 0:
            log.error(f"❌ {description} failed (code {result.returncode})")
            if result.stderr:
                log.error(f"Error: {result.stderr}")
            return False
        else:
            log.info(f"✅ {description} completed successfully")
            if result.stdout.strip():
                # Print key lines from output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # Last 10 lines
                    if any(keyword in line.lower() for keyword in ['complete', 'total', 'success', 'coverage', 'mae']):
                        log.info(f"Output: {line}")
            return True
            
    except subprocess.TimeoutExpired:
        log.error(f"❌ {description} timed out")
        return False
    except Exception as e:
        log.error(f"❌ {description} error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete backfill and retrain workflow")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--predict", action="store_true", help="Generate predictions during backfill")
    parser.add_argument("--no-weather", action="store_true", help="Skip weather during backfill")
    parser.add_argument("--require-market", action="store_true", help="Only train on games with market data")
    parser.add_argument("--dry-run", action="store_true", help="Show commands but don't execute")
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:
        log.error(f"❌ Invalid date format: {e}")
        return False
    
    if end_date < start_date:
        log.error("❌ End date must be >= start date")
        return False
    
    # Calculate window parameters
    window_days = (datetime.strptime(end_date, "%Y-%m-%d") - 
                   datetime.strptime(start_date, "%Y-%m-%d")).days + 1
    holdout_days = min(7, max(3, window_days // 5))  # 20% holdout, min 3, max 7 days
    
    log.info(f"📅 Workflow Parameters:")
    log.info(f"   Date range: {start_date} → {end_date} ({window_days} days)")
    log.info(f"   Holdout: {holdout_days} days")
    log.info(f"   Include predictions: {args.predict}")
    log.info(f"   Skip weather: {args.no_weather}")
    log.info(f"   Require market data: {args.require_market}")
    
    # Get script paths
    here = Path(__file__).resolve().parent
    py = sys.executable
    
    backfill_script = str(here / "backfill_range.py")
    coverage_script = str(here / "coverage_check.py")
    retrain_script = str(here / "retrain_model.py")
    
    if args.dry_run:
        log.info("🧪 DRY RUN - Commands that would be executed:")
        
        # Show backfill command
        backfill_cmd = [py, backfill_script, "--start", start_date, "--end", end_date]
        if args.predict: backfill_cmd.append("--predict")
        if args.no_weather: backfill_cmd.append("--no-weather")
        if args.require_market: backfill_cmd.extend(["--market-ingestor", "enhanced"])
        log.info(f"1. Backfill: {' '.join(backfill_cmd)}")
        
        # Show coverage check
        coverage_cmd = [py, coverage_script, "--start", start_date, "--end", end_date]
        if args.require_market: coverage_cmd.append("--require-market")
        log.info(f"2. Coverage: {' '.join(coverage_cmd)}")
        
        # Show retrain command
        retrain_cmd = [py, retrain_script, "--end", end_date, 
                      "--window-days", str(window_days), "--holdout-days", str(holdout_days),
                      "--audit", "--deploy"]
        if args.require_market: retrain_cmd.append("--require-market")
        log.info(f"3. Retrain: {' '.join(retrain_cmd)}")
        return True
    
    log.info("🔄 Starting Complete Backfill and Retrain Workflow")
    log.info("=" * 60)
    
    # Step 1: Backfill historical data
    backfill_cmd = [py, backfill_script, "--start", start_date, "--end", end_date]
    if args.predict:
        backfill_cmd.append("--predict")
    if args.no_weather:
        backfill_cmd.append("--no-weather")
    if args.require_market:
        backfill_cmd.extend(["--market-ingestor", "enhanced"])  # or "real" based on preference
    
    if not run_command(backfill_cmd, f"Backfill historical data ({start_date} → {end_date})", timeout=3600):
        log.error("❌ Backfill failed - aborting workflow")
        return False
    
    # Step 2: Check training data coverage
    coverage_cmd = [py, coverage_script, "--start", start_date, "--end", end_date]
    if args.require_market:
        coverage_cmd.append("--require-market")
    
    if not run_command(coverage_cmd, "Check training data coverage", timeout=60):
        log.error("❌ Coverage check failed - aborting workflow")
        return False
    
    # Step 3: Retrain model on backfilled data
    retrain_cmd = [py, retrain_script, 
                   "--end", end_date,
                   "--window-days", str(window_days),
                   "--holdout-days", str(holdout_days),
                   "--audit", "--deploy"]
    if args.require_market:
        retrain_cmd.append("--require-market")
    
    if not run_command(retrain_cmd, f"Retrain model (window={window_days}d, holdout={holdout_days}d)", timeout=1800):
        log.error("❌ Retraining failed - aborting workflow")
        return False
    
    # Success!
    log.info("🎉 Complete workflow finished successfully!")
    log.info("=" * 60)
    log.info("✅ Historical data backfilled")
    log.info("✅ Training coverage validated")
    log.info("✅ Model retrained and deployed")
    log.info(f"📊 Training window: {window_days} days")
    log.info(f"🎯 Holdout validation: {holdout_days} days")
    log.info("🚀 New model is now active for predictions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
