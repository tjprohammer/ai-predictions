#!/usr/bin/env python3
"""
Quick test of the organized tracking system
"""

import sys
import os
from pathlib import Path

# Add tracking paths
sys.path.append(str(Path(__file__).parent / "mlb" / "tracking" / "performance"))
sys.path.append(str(Path(__file__).parent / "mlb" / "tracking" / "validation"))
sys.path.append(str(Path(__file__).parent / "mlb" / "tracking" / "results"))
sys.path.append(str(Path(__file__).parent / "mlb" / "tracking" / "monitoring"))

def test_tracking_system():
    """Test our organized tracking system"""
    print("🔍 Testing Organized MLB Tracking System")
    print("=" * 50)
    
    # Test 1: Performance Tracking
    print("\n📊 Testing Performance Tracking...")
    try:
        from enhanced_prediction_tracker import EnhancedPredictionTracker
        tracker = EnhancedPredictionTracker()
        
        # Get basic performance data
        performance = tracker.get_comprehensive_performance_analysis(days=7)
        print(f"✅ Performance tracker loaded successfully")
        print(f"   📈 Analysis covers {performance.get('days_analyzed', 'N/A')} days")
        
    except Exception as e:
        print(f"❌ Performance tracking failed: {e}")
    
    # Test 2: Validation
    print("\n🔍 Testing Validation...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "mlb/tracking/validation/check_predictions_final.py"], 
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Validation script executed successfully")
            lines = result.stdout.split('\n')
            for line in lines[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ Validation failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    # Test 3: API Integration
    print("\n🌐 Testing API Integration...")
    try:
        # Test if we can import the tracking components like the API does
        sys.path.append("mlb/tracking/performance")
        from enhanced_prediction_tracker import EnhancedPredictionTracker
        
        api_tracker = EnhancedPredictionTracker()
        print("✅ API-style tracking import successful")
        print("   🔗 Ready for /api/comprehensive-tracking endpoint")
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
    
    # Test 4: Directory Structure
    print("\n📁 Testing Directory Structure...")
    directories = [
        "mlb/tracking/performance",
        "mlb/tracking/results", 
        "mlb/tracking/validation",
        "mlb/tracking/monitoring"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            files = os.listdir(directory)
            print(f"✅ {directory}: {len(files)} files")
        else:
            print(f"❌ {directory}: Missing")
    
    print("\n🎯 Tracking System Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_tracking_system()
