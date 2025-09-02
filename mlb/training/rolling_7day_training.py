#!/usr/bin/env python3
"""
Rolling 7-Day Training System
=============================
Train the Ultra-80 system using only the most recent 7 days of data.
This creates a highly responsive model that adapts quickly to recent trends.

Usage:
    python rolling_7day_training.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the systems directory to the path
sys.path.append(str(Path(__file__).parent / "mlb" / "systems"))
from incremental_ultra_80_system import IncrementalUltra80System

def rolling_7day_training():
    """Train using only the last 7 days of completed games"""
    
    print("🔄 ROLLING 7-DAY TRAINING SYSTEM")
    print("=" * 50)
    
    # Calculate date range (last 7 days)
    today = datetime.now()
    end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
    start_date = (today - timedelta(days=8)).strftime('%Y-%m-%d')  # 8 days ago to get 7 full days
    
    print(f"🗓️  Training Window: {start_date} to {end_date}")
    
    # Initialize system
    system = IncrementalUltra80System()
    
    # Option 1: Fresh training on 7 days only (no state loading)
    print("🧠 Training fresh model on 7-day window...")
    results = system.team_level_incremental_learn(
        start_date=start_date,
        end_date=end_date
    )
    
    if results:
        # Save the 7-day trained model
        state_path = Path("models") / "rolling_7day_ultra80_state.joblib"
        system.save_state(str(state_path))
        
        print(f"✅ 7-Day Rolling Model Trained Successfully!")
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
    success = rolling_7day_training()
    exit(0 if success else 1)
