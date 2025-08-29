#!/usr/bin/env python3
"""
Test Incremental System Integration
==================================

Quick test to verify the incremental system can generate predictions for today.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add pipeline path
sys.path.append(str(Path(__file__).parent / 'mlb-overs' / 'pipelines'))

try:
    from incremental_ultra_80_system import IncrementalUltra80System
    print("✅ Successfully imported IncrementalUltra80System")
except ImportError as e:
    print(f"❌ Failed to import IncrementalUltra80System: {e}")
    sys.exit(1)

def test_incremental_system():
    """Test the incremental system for today's predictions"""
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"🎯 Testing incremental system for: {today}")
    
    # Create system
    system = IncrementalUltra80System()
    print("✅ Created IncrementalUltra80System instance")
    
    # Try to load existing state
    state_loaded = system.load_state()
    print(f"📁 State loaded: {state_loaded}")
    print(f"🏗️ Is fitted: {system.is_fitted}")
    
    if not system.is_fitted:
        print("⚠️ System not fitted - need to train on historical data first")
        print("💡 Run incremental learning on historical data to fit the models")
        return False
    
    # Try to predict for today
    try:
        print(f"🔮 Attempting to predict games for {today}...")
        predictions = system.predict_future_slate(today, outdir='outputs')
        
        if predictions is not None and not predictions.empty:
            print(f"✅ Generated {len(predictions)} predictions!")
            print("\n📋 PREDICTION SUMMARY:")
            print("-" * 60)
            
            for _, pred in predictions.head(5).iterrows():  # Show first 5
                print(f"{pred['away_team']} @ {pred['home_team']}")
                print(f"  Market: {pred['market_total']} | Predicted: {pred['pred_total']}")
                print(f"  Edge: {pred['diff']:+.2f} | EV: {pred['ev']:+.1%} | Trust: {pred['trust']:.2f}")
                print()
            
            if len(predictions) > 5:
                print(f"... and {len(predictions) - 5} more games")
            
            return True
        else:
            print(f"⚠️ No predictions generated for {today}")
            print("💡 This could mean no games scheduled or no market data available")
            return False
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return False

def main():
    print("🧪 TESTING INCREMENTAL SYSTEM INTEGRATION")
    print("=" * 50)
    
    success = test_incremental_system()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Incremental system test PASSED!")
        print("🎯 System is ready for daily workflow integration")
    else:
        print("❌ Incremental system test FAILED!")
        print("💡 Check if historical training is needed or if today has scheduled games")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
