#!/usr/bin/env python3
"""
Apply bias correction to the model bundle based on audit recommendations.
"""

import joblib
import logging
from pathlib import Path

def apply_bias_correction(bundle_path: str, bias_value: float):
    """Apply bias correction to model bundle."""
    logger = logging.getLogger(__name__)
    
    # Load the bundle
    logger.info(f"Loading bundle from {bundle_path}")
    bundle = joblib.load(bundle_path)
    
    # Update bias correction
    old_bias = bundle.get('bias_correction', 0.0)
    bundle['bias_correction'] = bias_value
    
    logger.info(f"Updated bias correction: {old_bias} → {bias_value}")
    
    # Save updated bundle
    backup_path = bundle_path.replace('.joblib', '_backup.joblib')
    logger.info(f"Creating backup at {backup_path}")
    joblib.dump(bundle, backup_path)
    
    logger.info(f"Saving updated bundle to {bundle_path}")
    joblib.dump(bundle, bundle_path)
    
    logger.info("✅ Bias correction applied successfully")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # From audit: "Add bias correction: 0.542"
    bias_correction = 0.542
    
    bundle_path = "../models/legitimate_model_latest.joblib"
    
    if Path(bundle_path).exists():
        apply_bias_correction(bundle_path, bias_correction)
    else:
        print(f"❌ Bundle not found: {bundle_path}")
        sys.exit(1)
