#!/usr/bin/env python3
"""
Incremental Learning Update
==========================

Updates the incremental learning system with recent completed games.
Run this before the daily workflow to ensure models are up-to-date.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add pipeline path
sys.path.append(str(Path(__file__).parent / 'mlb-overs' / 'pipelines'))

def update_incremental_learning(days_back: int = 3):
    """Update incremental learning with recent completed games"""
    
    try:
        from incremental_ultra_80_system import IncrementalUltra80System
    except ImportError as e:
        print(f"❌ Cannot import incremental system: {e}")
        return False
    
    print("🧠 UPDATING INCREMENTAL LEARNING FROM RECENT GAMES")
    print("=" * 60)
    
    # Create system and load existing state
    system = IncrementalUltra80System()
    state_loaded = system.load_state()
    print(f"📁 State loaded: {state_loaded}")
    print(f"🏗️ Is fitted: {system.is_fitted}")
    
    # Calculate date range for learning
    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday - timedelta(days=days_back)
    
    print(f"📅 Learning from: {start_date.strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')}")
    
    try:
        # Run incremental learning
        results = system.team_level_incremental_learn(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=yesterday.strftime('%Y-%m-%d')
        )
        
        if results:
            games_count = len(results.get('predictions', []))
            coverage = results['final_coverage']
            mae = results['final_mae']
            roi = results['final_roi']
            
            print(f"✅ Learned from {games_count} completed games")
            print(f"📊 Performance: Coverage={coverage:.1%} | MAE={mae:.2f} | ROI={roi:+.2%}")
            
            # Save updated state
            system.save_state()
            print("💾 Updated model state saved")
            
            return True
        else:
            print("⚠️ No recent completed games found for learning")
            print("💡 This is normal if no games completed in the last few days")
            return True  # Not an error, just no new data
            
    except Exception as e:
        print(f"❌ Error during incremental learning: {e}")
        return False

def main():
    success = update_incremental_learning()
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
