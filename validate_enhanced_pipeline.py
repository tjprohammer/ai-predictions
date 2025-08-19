#!/usr/bin/env python3
"""
Enhanced Pipeline Validation Script
Tests all components of the enhanced MLB prediction pipeline
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

def run_command(cmd, description):
    """Run a CLI command and return success status"""
    print(f"\n🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    # Set environment variables for UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=".", env=env, encoding='utf-8', errors='ignore')
        print("   ✅ SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED: {e}")
        if e.stdout:
            print(f"   STDOUT: {e.stdout}")
        if e.stderr:
            print(f"   STDERR: {e.stderr}")
        return False

def check_data_status(date_str):
    """Check current data status in database"""
    engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    
    query = f'''
        SELECT 
            COUNT(*) as total_games,
            COUNT(market_total) as market_data,
            COUNT(temperature) as weather_data,
            COUNT(predicted_total) as predictions,
            COUNT(home_sp_name) as pitcher_data
        FROM enhanced_games 
        WHERE date = '{date_str}'
    '''
    
    df = pd.read_sql(query, engine)
    status = df.iloc[0]
    
    print(f"\n📊 DATA STATUS for {date_str}:")
    print(f"   📅 Total Games: {status['total_games']}")
    print(f"   💰 Market Data: {status['market_data']} games")
    print(f"   🌤️  Weather Data: {status['weather_data']} games")
    print(f"   🤖 Predictions: {status['predictions']} games")
    print(f"   ⚾ Pitcher Data: {status['pitcher_data']} games")
    
    return status

def main():
    print("🚀 ENHANCED MLB PIPELINE VALIDATION")
    print("=" * 50)
    
    date_str = "2025-08-17"
    success_count = 0
    total_tests = 0
    
    # Test 1: Games Collection
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/data_collection/working_games_ingestor.py", 
        "--target-date", date_str
    ], "Collect MLB Games Data"):
        success_count += 1
    
    # Test 2: Market Data Collection
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/data_collection/real_market_ingestor.py", 
        "--date", date_str
    ], "Collect Real Market Data (Fixed Bug)"):
        success_count += 1
    
    # Test 3: Weather Data Collection
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/data_collection/weather_ingestor.py", 
        "--date", date_str, "--force-update"
    ], "Collect Enhanced Weather Data"):
        success_count += 1
    
    # Test 4: ML Predictions
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/deployment/enhanced_bullpen_predictor.py", 
        "--target-date", date_str
    ], "Generate Enhanced ML Predictions"):
        success_count += 1
    
    # Test 5: Betting Analysis
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/deployment/enhanced_analysis.py", 
        "--date", date_str
    ], "Run Enhanced Betting Analysis"):
        success_count += 1
    
    # Test 6: Production Status
    total_tests += 1
    if run_command([
        sys.executable, 
        "mlb-overs/deployment/production_status.py"
    ], "Check Production Status"):
        success_count += 1
    
    # Check final data status
    try:
        status = check_data_status(date_str)
        data_completeness = (
            status['market_data'] > 0 and 
            status['weather_data'] > 0 and 
            status['predictions'] > 0
        )
    except Exception as e:
        print(f"\n❌ Database check failed: {e}")
        data_completeness = False
    
    # Final Report
    print(f"\n🏁 ENHANCED PIPELINE VALIDATION COMPLETE")
    print("=" * 50)
    print(f"✅ CLI Tests Passed: {success_count}/{total_tests}")
    print(f"📊 Data Pipeline: {'✅ COMPLETE' if data_completeness else '❌ INCOMPLETE'}")
    
    if success_count == total_tests and data_completeness:
        print("\n🎉 ENHANCED PIPELINE FULLY OPERATIONAL!")
        print("Ready for production betting analysis.")
        return 0
    else:
        print("\n⚠️  Some components need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
