#!/usr/bin/env python3
"""
Training Directories Cleanup Script
Clean up redundant training files, old model backups, and obsolete training systems
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Define the mlb-overs directory
MLB_OVERS_ROOT = Path(__file__).parent / "mlb-overs"

def create_archive_directory():
    """Create archive directory for training cleanup"""
    archive_dir = MLB_OVERS_ROOT / "archive_training_cleanup"
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def cleanup_training_systems():
    """Clean up the training_systems directory"""
    training_systems_dir = MLB_OVERS_ROOT / "training_systems"
    archive_dir = create_archive_directory()
    training_archive = archive_dir / "training_systems"
    training_archive.mkdir(exist_ok=True)
    
    # Files to KEEP in training_systems (only current/essential)
    keep_files = {
        "daily_learning_pipeline.py",  # Current pipeline
        "production_model.joblib",     # Production model
    }
    
    # Archive everything else
    archived_count = 0
    if training_systems_dir.exists():
        for item in training_systems_dir.iterdir():
            if item.name in keep_files or item.name.startswith('.') or item.name == '__pycache__':
                continue
                
            dest_path = training_archive / item.name
            try:
                if item.is_dir():
                    shutil.move(str(item), str(dest_path))
                    print(f"Moved directory: training_systems/{item.name}/")
                else:
                    shutil.move(str(item), str(dest_path))
                    print(f"Moved file: training_systems/{item.name}")
                archived_count += 1
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    return archived_count

def cleanup_model_backups():
    """Clean up excessive model backups, keeping only recent ones"""
    models_dir = MLB_OVERS_ROOT / "models"
    archive_dir = create_archive_directory()
    models_archive = archive_dir / "models_old_backups"
    models_archive.mkdir(exist_ok=True)
    
    # Current production models to KEEP
    keep_models = {
        "adaptive_learning_model.joblib",     # Main learning model (84.5%)
        "legitimate_model_latest.joblib",     # Current production model
        "optimized_features.json",            # Selected features
        "comprehensive_features.json",        # Feature metadata
        "comprehensive_metadata.json",        # Model metadata
        "verification_report_20250822_155344.json",  # Latest verification
    }
    
    # Keep current pipeline scripts
    keep_scripts = {
        "adaptive_learning_pipeline.py",
        "dual_model_predictor.py", 
        "enhanced_infer.py",
        "infer.py",
        "learning_impact_tracker.py",
    }
    
    archived_count = 0
    if models_dir.exists():
        for item in models_dir.iterdir():
            # Skip directories, __pycache__, and files we want to keep
            if (item.is_dir() or 
                item.name.startswith('.') or 
                item.name == '__pycache__' or
                item.name in keep_models or 
                item.name in keep_scripts):
                continue
            
            # Archive old backup models and redundant files
            if (item.name.startswith('backup_') or
                item.name.startswith('learning_model_retrained_') or
                item.name.endswith('.bak.') or
                'backup' in item.name or
                item.name in [
                    'clean_mlb_model.joblib',
                    'comprehensive_model.joblib', 
                    'efficient_refined_model.joblib',  # Data leakage model
                    'enhanced_leak_free_features.joblib',
                    'enhanced_leak_free_model.joblib',
                    'fixed_model_latest.joblib',
                    'legitimate_model_08-15-2025.joblib',
                    'legitimate_model_2025-08-15.joblib',
                    'legitimate_model_20250816_174512.joblib',
                    'legitimate_model_20250816_175551.joblib',
                    'legitimate_model_20250816_181102.joblib',
                    'legitimate_model_20250816_181515.joblib',
                    'legitimate_model_20250817_170325.joblib',
                    'legitimate_model_latest_backup.joblib',
                    'production_clean_model.joblib',
                    'refined_learning_model.joblib',
                    'enhanced_feature_importance.csv',
                ]):
                
                dest_path = models_archive / item.name
                try:
                    shutil.move(str(item), str(dest_path))
                    print(f"Moved old model: {item.name}")
                    archived_count += 1
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    return archived_count

def cleanup_models_backup():
    """Clean up the models_backup directory"""
    models_backup_dir = MLB_OVERS_ROOT / "models_backup"
    archive_dir = create_archive_directory()
    
    archived_count = 0
    if models_backup_dir.exists():
        backup_archive = archive_dir / "models_backup"
        try:
            shutil.move(str(models_backup_dir), str(backup_archive))
            print(f"Moved entire models_backup/ directory")
            archived_count = 1
        except Exception as e:
            print(f"Error moving models_backup directory: {e}")
    
    return archived_count

def cleanup_training_data():
    """Clean up old training data files"""
    training_data_dir = MLB_OVERS_ROOT / "training_data"
    archive_dir = create_archive_directory()
    training_data_archive = archive_dir / "training_data"
    training_data_archive.mkdir(exist_ok=True)
    
    # Keep only essential training data
    keep_files = {
        "complete_historical_data.csv",  # Main historical dataset
    }
    
    archived_count = 0
    if training_data_dir.exists():
        for item in training_data_dir.iterdir():
            if item.name in keep_files or item.name.startswith('.'):
                continue
                
            dest_path = training_data_archive / item.name
            try:
                shutil.move(str(item), str(dest_path))
                print(f"Moved training data: {item.name}")
                archived_count += 1
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    return archived_count

def list_remaining_files():
    """List what remains after cleanup"""
    print("\n" + "="*60)
    print("REMAINING FILES AFTER CLEANUP")
    print("="*60)
    
    directories_to_check = [
        "training_systems",
        "training_data", 
        "models",
        "models_backup"
    ]
    
    for dir_name in directories_to_check:
        dir_path = MLB_OVERS_ROOT / dir_name
        if dir_path.exists():
            print(f"\n📁 {dir_name}/:")
            items = list(dir_path.iterdir())
            if not items:
                print("  (empty)")
            else:
                for item in sorted(items):
                    if item.name.startswith('.') or item.name == '__pycache__':
                        continue
                    if item.is_dir():
                        print(f"  📁 {item.name}/")
                    else:
                        print(f"  📄 {item.name}")
        else:
            print(f"\n📁 {dir_name}/: (directory not found)")

def main():
    """Main cleanup function"""
    print("🧹 Starting Training Directories Cleanup")
    print("=" * 60)
    
    total_archived = 0
    
    # Clean up each directory
    print("\n1. Cleaning up training_systems/...")
    total_archived += cleanup_training_systems()
    
    print("\n2. Cleaning up old model backups...")
    total_archived += cleanup_model_backups()
    
    print("\n3. Cleaning up models_backup/...")
    total_archived += cleanup_models_backup()
    
    print("\n4. Cleaning up training_data/...")
    total_archived += cleanup_training_data()
    
    print("\n" + "="*60)
    print(f"🎉 Training Cleanup Complete!")
    print(f"Total items archived: {total_archived}")
    
    # List remaining structure
    list_remaining_files()
    
    archive_dir = MLB_OVERS_ROOT / "archive_training_cleanup"
    print(f"\nArchived files are in: {archive_dir}")
    print("\nIMPORTANT: Review archived files before deleting permanently.")

if __name__ == "__main__":
    main()
