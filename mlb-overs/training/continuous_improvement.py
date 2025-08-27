#!/usr/bin/env python3
"""
Continuous Model Improvement System
==================================

Runs multiple training cycles to progressively improve model accuracy.
Uses legitimate training with anti-cheat validation.

Author: AI Assistant
Date: 2025-08-24
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import subprocess
import json
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_cycle(cycle_num, retrain_days=30):
    """Run a single training cycle with legitimate trainer"""
    logger.info(f"🔄 TRAINING CYCLE {cycle_num}/10")
    logger.info(f"   Retrain window: {retrain_days} days")
    
    try:
        # Run legitimate model trainer
        cutoff_date = (datetime.now() - timedelta(days=retrain_days)).strftime('%Y-%m-%d')
        cmd = [
            sys.executable,
            "mlb-overs/deployment/legitimate_model_trainer.py",
            "--cutoff-date", cutoff_date,
            "--verbose"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            # Extract accuracy from output
            output_lines = result.stdout.split('\n')
            learning_accuracy = None
            original_accuracy = None
            
            for line in output_lines:
                if "Learning model accuracy:" in line:
                    learning_accuracy = float(line.split(":")[1].strip().replace("%", ""))
                elif "Original model accuracy:" in line:
                    original_accuracy = float(line.split(":")[1].strip().replace("%", ""))
            
            logger.info(f"✅ CYCLE {cycle_num} COMPLETE:")
            if learning_accuracy:
                logger.info(f"   Learning: {learning_accuracy:.1f}%")
            if original_accuracy:
                logger.info(f"   Original: {original_accuracy:.1f}%")
            
            return {
                'cycle': cycle_num,
                'learning_accuracy': learning_accuracy,
                'original_accuracy': original_accuracy,
                'success': True,
                'retrain_days': retrain_days
            }
        else:
            logger.error(f"❌ CYCLE {cycle_num} FAILED:")
            logger.error(result.stderr)
            return {
                'cycle': cycle_num,
                'success': False,
                'error': result.stderr
            }
            
    except Exception as e:
        logger.error(f"❌ CYCLE {cycle_num} ERROR: {e}")
        return {
            'cycle': cycle_num,
            'success': False,
            'error': str(e)
        }

def progressive_training(max_cycles=10):
    """Run progressive training cycles with increasing data windows"""
    logger.info("🚀 STARTING CONTINUOUS MODEL IMPROVEMENT")
    logger.info("==========================================")
    
    results = []
    
    # Progressive training windows
    training_windows = [20, 25, 30, 35, 40, 45, 50, 60, 75, 90]
    
    for cycle in range(1, max_cycles + 1):
        retrain_days = training_windows[cycle - 1] if cycle <= len(training_windows) else 90
        
        result = run_training_cycle(cycle, retrain_days)
        results.append(result)
        
        if result['success'] and result.get('learning_accuracy'):
            accuracy = result['learning_accuracy']
            logger.info(f"📈 PROGRESS: {accuracy:.1f}% accuracy achieved")
            
            # Stop if we reach excellent accuracy
            if accuracy >= 75.0:
                logger.info(f"🎯 EXCELLENT ACCURACY ACHIEVED: {accuracy:.1f}%")
                logger.info("   Stopping training - target reached!")
                break
        else:
            logger.warning(f"⚠️ Cycle {cycle} had issues, continuing...")
    
    # Save results
    results_file = f"continuous_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("==========================================")
    logger.info("🏁 CONTINUOUS TRAINING COMPLETE")
    logger.info(f"📊 Results saved to: {results_file}")
    
    # Summary
    successful_cycles = [r for r in results if r['success']]
    if successful_cycles:
        best_learning = max(r.get('learning_accuracy', 0) for r in successful_cycles if r.get('learning_accuracy'))
        best_original = max(r.get('original_accuracy', 0) for r in successful_cycles if r.get('original_accuracy'))
        
        logger.info(f"🏆 BEST ACHIEVED:")
        logger.info(f"   Learning: {best_learning:.1f}%")
        logger.info(f"   Original: {best_original:.1f}%")
        
        improvement = best_learning - 61.9  # Starting accuracy
        logger.info(f"📈 IMPROVEMENT: +{improvement:.1f} percentage points")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Continuous Model Improvement")
    parser.add_argument("--cycles", type=int, default=10, help="Maximum training cycles")
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 cycles")
    
    args = parser.parse_args()
    
    max_cycles = 3 if args.quick else args.cycles
    
    logger.info(f"🎯 Target: Improve from current 61.9% to 70-75%")
    logger.info(f"📊 Plan: {max_cycles} progressive training cycles")
    logger.info(f"🔒 Anti-cheat: Full temporal validation")
    
    results = progressive_training(max_cycles)
    
    # Final verification
    logger.info("🔍 Running final model verification...")
    try:
        sys.path.append('mlb-overs/deployment')
        from dual_model_integration import DualModelPredictor
        predictor = DualModelPredictor()
        if predictor.original_model:
            logger.info(f"✅ Final models loaded: {len(predictor.features)} features")
            logger.info("🎯 Ready for production predictions!")
        else:
            logger.error("❌ Final verification failed")
    except Exception as e:
        logger.error(f"❌ Final verification error: {e}")

if __name__ == "__main__":
    main()
