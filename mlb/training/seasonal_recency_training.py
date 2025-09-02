#!/usr/bin/env python3
"""
Seasonal Training with Recency Bias
===================================
Train the Ultra-80 system on the full season but with stronger weighting 
for recent games. This balances historical knowledge with recent trends.

Usage:
    python seasonal_recency_training.py [--start-date YYYY-MM-DD]
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the systems directory to the path
sys.path.append(str(Path(__file__).parent / "mlb" / "systems"))
from incremental_ultra_80_system import IncrementalUltra80System

def seasonal_recency_training(season_start_date=None):
    """Train using full season with recency bias"""
    
    print("🏟️  SEASONAL TRAINING WITH RECENCY BIAS")
    print("=" * 50)
    
    # Calculate date range
    if season_start_date:
        start_date = season_start_date
    else:
        # Default to start of current season (April 1st)
        current_year = datetime.now().year
        start_date = f"{current_year}-04-01"
    
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"🗓️  Training Window: {start_date} to {end_date}")
    print(f"📈 Strategy: Full season data with stronger recent weighting")
    
    # Initialize system
    system = IncrementalUltra80System()
    
    # Load existing state if available (maintains accumulated knowledge)
    state_path = Path("models") / "seasonal_recency_ultra80_state.joblib"
    state_loaded = system.load_state(str(state_path))
    
    if state_loaded:
        print("📁 Loaded existing seasonal state")
        
        # Only train on recent games to update the model
        recent_start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        print(f"🔄 Updating with recent games: {recent_start} to {end_date}")
        
        results = system.team_level_incremental_learn(
            start_date=recent_start,
            end_date=end_date
        )
    else:
        print("🚀 Training new seasonal model from scratch")
        
        results = system.team_level_incremental_learn(
            start_date=start_date,
            end_date=end_date
        )
    
    if results:
        # Save the updated seasonal model
        system.save_state(str(state_path))
        
        print(f"✅ Seasonal Recency Model Updated Successfully!")
        print(f"📊 Games processed: {len(results.get('predictions', []))}")
        print(f"🎯 Coverage: {results.get('final_coverage', 0):.1%}")
        print(f"📈 MAE: {results.get('final_mae', 0):.2f}")
        print(f"💰 ROI: {results.get('final_roi', 0):+.2%}")
        print(f"💾 State saved: {state_path}")
        
        return True
    else:
        print("❌ Training failed - no results")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with seasonal data and recency bias")
    parser.add_argument("--start-date", help="Season start date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    success = seasonal_recency_training(args.start_date)
    exit(0 if success else 1)
