#!/usr/bin/env python3
"""
Bootstrap Ultra 80 System for Daily Workflow
Creates state file from the deployment directory context
"""

import sys
import os
from pathlib import Path

# Add the pipelines directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))

from incremental_ultra_80_system import IncrementalUltra80System


def bootstrap_ultra80():
    """Bootstrap the Ultra 80 system for daily workflow usage"""
    print("🚀 Bootstrapping Ultra 80 system for daily workflow...")
    
    # State file path (in root directory)
    state_path = Path("../../incremental_ultra80_state.joblib")
    
    # Initialize the system
    system = IncrementalUltra80System()
    
    # Try to load existing state first
    if state_path.exists():
        print(f"📁 Found existing state file: {state_path}")
        try:
            # Load from the original location and re-save from this context
            original_loaded = system.load_state(str(state_path))
            if original_loaded:
                print("✅ Loaded existing state successfully")
                # Re-save from this module context
                system.save_state(str(state_path))
                print(f"💾 Re-saved state file from deployment context: {state_path}")
                return True
        except Exception as e:
            print(f"⚠️  Could not load existing state: {e}")
    
    print("🏗️  No valid state found, running fresh bootstrap...")
    
    # If no valid state, run a fresh bootstrap
    try:
        system.bootstrap_from_models()
        system.save_state(str(state_path))
        print(f"💾 Fresh bootstrap complete, state saved: {state_path}")
        return True
    except Exception as e:
        print(f"❌ Bootstrap failed: {e}")
        return False


if __name__ == "__main__":
    success = bootstrap_ultra80()
    if success:
        print("🎉 Ultra 80 system ready for daily workflow!")
    else:
        print("💥 Bootstrap failed - check logs above")
        sys.exit(1)
