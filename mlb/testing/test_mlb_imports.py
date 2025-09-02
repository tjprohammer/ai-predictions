#!/usr/bin/env python3
"""
Test script to verify all MLB module imports work correctly after reorganization
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all key modules can be imported from the new structure"""
    
    # Add the mlb directory to Python path
    mlb_path = Path(__file__).parent / "mlb"
    sys.path.append(str(mlb_path))
    
    print("🧪 Testing MLB module imports...")
    
    try:
        # Test core imports
        sys.path.append(str(mlb_path / "core"))
        
        print("  ✅ Testing enhanced_bullpen_predictor...")
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
        
        print("  ✅ Testing learning_model_predictor...")
        from learning_model_predictor import predict_and_upsert_learning
        
        # Test systems imports  
        sys.path.append(str(mlb_path / "systems"))
        
        print("  ✅ Testing incremental_ultra_80_system...")
        from incremental_ultra_80_system import IncrementalUltra80System
        
        print("  ✅ Testing ultra_80_percent_system...")
        from ultra_80_percent_system import UltraModel
        
        # Test validation imports
        sys.path.append(str(mlb_path / "validation"))
        
        print("  ✅ Testing health_gate...")
        import health_gate
        
        print("  ✅ Testing probabilities_and_ev...")
        import probabilities_and_ev
        
        print("✅ All imports successful! MLB module reorganization is working.")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
