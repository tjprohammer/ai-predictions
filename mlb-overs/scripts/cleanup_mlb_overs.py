#!/usr/bin/env python3
"""
MLB-Overs Directory Cleanup Script
Move one-time use files to archive based on production documentation
"""

import os
import shutil
from pathlib import Path

# Define the mlb-overs directory
MLB_OVERS_ROOT = Path(__file__).parent / "mlb-overs"

# KEEP these important production files in mlb-overs root
KEEP_FILES = {
    # Production models and config
    "models/",
    "api/",
    "deployment/",
    "README.md",
    ".env",
    "__init__.py",
    
    # Key production scripts mentioned in documentation
    "comprehensive_learning_backtester.py",
    "feature_importance_analyzer.py", 
    "model_performance_verifier.py",
    
    # Current data files
    "daily_market_totals.json",
    "daily_predictions.json",
    "backtest_results_20250822_145615.json",  # Latest backtest results
    
    # Production directories
    "data/",
    "features/",
    "scripts/",
    "tests/",
    "migrations/",
}

# Archive these one-time use files
ARCHIVE_FILES = {
    # Model refinement (one-time use)
    "advanced_model_refinement.py",
    "efficient_model_refinement.py", 
    "simplified_model_refinement.py",
    "improved_learning_model.py",
    "enhanced_model_trainer.py",
    
    # Debug files
    "debug_isotonic.py",
    "debug_rpg.py",
    
    # Validation files (one-time use)
    "clean_model_validator.py",
    "data_reality_check.py",
    "todays_reality_check.py",
    
    # Tracking files (one-time use)
    "performance_tracker.py",
    "prediction_tracker.py",
    "learning_model_backtester.py",
    
    # Testing files (one-time use)
    "test_calibrated_api.py",
    "todays_predictions.py",
    
    # Feature building (one-time use)
    "build_enhanced_features.py",
    
    # Bias correction (one-time use)
    "urgent_bias_correction.py",
}

# Archive old prediction/data files
ARCHIVE_DATA = {
    "betting_summary_2025-08-21.json",
    "betting_summary_2025-08-22.json", 
    "enhanced_predictions_2025-08-21.json",
    "enhanced_predictions_2025-08-22.json",
    "daily_predictions_2025-08-22.csv",
    "daily_predictions_2025-08-22.json",
    "data_investigation_20250822_233910.json",
}

def create_archive_directory():
    """Create archive directory if it doesn't exist"""
    archive_dir = MLB_OVERS_ROOT / "archive_cleanup"
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def move_files_to_archive(file_set, archive_dir, description):
    """Move a set of files to archive directory"""
    moved_count = 0
    for filename in file_set:
        source_path = MLB_OVERS_ROOT / filename
        if source_path.exists() and source_path.is_file():
            dest_path = archive_dir / filename
            try:
                shutil.move(str(source_path), str(dest_path))
                print(f"Moved {filename}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    print(f"\n{description}: {moved_count} files moved to archive")
    return moved_count

def list_directory_contents():
    """List remaining contents in mlb-overs directory"""
    print("\nRemaining contents in mlb-overs directory:")
    items = []
    for item in os.listdir(MLB_OVERS_ROOT):
        if item.startswith('.') or item == '__pycache__' or item.startswith('archive'):
            continue
        item_path = MLB_OVERS_ROOT / item
        if item_path.is_dir():
            items.append(f"📁 {item}/")
        else:
            items.append(f"📄 {item}")
    
    for item in sorted(items):
        print(f"  {item}")

def main():
    """Main cleanup function"""
    print("🧹 Starting MLB-Overs Directory Cleanup")
    print("=" * 50)
    
    # Create archive directory
    archive_dir = create_archive_directory()
    print(f"Created archive directory: {archive_dir}")
    
    total_moved = 0
    
    # Move different categories of files
    total_moved += move_files_to_archive(ARCHIVE_FILES, archive_dir, "Python script files")
    total_moved += move_files_to_archive(ARCHIVE_DATA, archive_dir, "Old data/prediction files")
    
    print("\n" + "=" * 50)
    print(f"🎉 MLB-Overs Cleanup Complete!")
    print(f"Total files moved to archive: {total_moved}")
    
    # List remaining contents
    list_directory_contents()
    
    print(f"\nArchived files are in: {archive_dir}")

if __name__ == "__main__":
    main()
