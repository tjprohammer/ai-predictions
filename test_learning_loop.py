#!/usr/bin/env python3
"""
test_learning_loop.py
=====================
Test the complete learning loop: evaluation and retraining
"""
import sys
import os
from datetime import datetime, timedelta

# Add the deployment directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlb-overs", "deployment"))

def test_evaluation():
    """Test the evaluation stage"""
    print("Testing evaluation stage...")
    
    from daily_api_workflow import stage_eval
    
    # Use yesterday as test date
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        result = stage_eval(target_date)
        if result:
            print(f"✅ Evaluation stage completed successfully for {target_date}")
        else:
            print(f"❌ Evaluation stage failed for {target_date}")
        return result
    except Exception as e:
        print(f"❌ Evaluation stage error: {e}")
        return False

def test_retraining():
    """Test the retraining stage"""
    print("\nTesting retraining stage...")
    
    from daily_api_workflow import stage_retrain
    
    # Use yesterday as end date for training window
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        # Test with smaller parameters for faster execution
        result = stage_retrain(
            target_date=target_date,
            window_days=90,  # Smaller window for testing
            holdout_days=14,
            deploy=False,    # Don't deploy during testing
            audit=False      # Skip audit for now
        )
        if result:
            print(f"✅ Retraining stage completed successfully for {target_date}")
        else:
            print(f"❌ Retraining stage failed for {target_date}")
        return result
    except Exception as e:
        print(f"❌ Retraining stage error: {e}")
        return False

def main():
    """Run all learning loop tests"""
    print("🔄 Testing MLB Learning Loop Components")
    print("=" * 50)
    
    # Test evaluation
    eval_success = test_evaluation()
    
    # Test retraining
    retrain_success = test_retraining()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Evaluation: {'✅ PASS' if eval_success else '❌ FAIL'}")
    print(f"  Retraining: {'✅ PASS' if retrain_success else '❌ FAIL'}")
    
    if eval_success and retrain_success:
        print("\n🎉 All learning loop components working!")
        return True
    else:
        print("\n⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
