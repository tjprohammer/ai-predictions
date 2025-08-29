#!/usr/bin/env python3
"""
MLB Module Migration Helper
===========================
Updates file paths and imports after reorganizing MLB prediction system into mlb/ directory

Usage:
    python migrate_mlb_references.py [--dry-run]
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Path mappings for the migration
PATH_MAPPINGS = {
    # Core files moved to mlb/core/
    'mlb-overs/deployment/daily_api_workflow.py': 'mlb/core/daily_api_workflow.py',
    'mlb-overs/deployment/enhanced_bullpen_predictor.py': 'mlb/core/enhanced_bullpen_predictor.py', 
    'mlb-overs/models/learning_model_predictor.py': 'mlb/core/learning_model_predictor.py',
    
    # Systems moved to mlb/systems/
    'mlb-overs/pipelines/incremental_ultra_80_system.py': 'mlb/systems/incremental_ultra_80_system.py',
    'mlb-overs/pipelines/ultra_80_percent_system.py': 'mlb/systems/ultra_80_percent_system.py',
    
    # Ingestion moved to mlb/ingestion/
    'mlb-overs/deployment/ingestion/': 'mlb/ingestion/',
    
    # Validation moved to mlb/validation/
    'mlb-overs/deployment/health_gate.py': 'mlb/validation/health_gate.py',
    'mlb-overs/deployment/probabilities_and_ev.py': 'mlb/validation/probabilities_and_ev.py',
    
    # Training moved to mlb/training/
    'mlb-overs/deployment/training/training_bundle_audit.py': 'mlb/training/training_bundle_audit.py',
    'mlb-overs/deployment/retrain_model.py': 'mlb/training/retrain_model.py',
    'mlb-overs/deployment/data_preparation/backfill_range.py': 'mlb/training/backfill_range.py',
    
    # Models moved to mlb/models/
    'mlb-overs/deployment/models/': 'mlb/models/',
    'mlb-overs/deployment/incremental_ultra80_state.joblib': 'mlb/models/incremental_ultra80_state.joblib',
    
    # Config moved to mlb/config/
    'mlb-overs/deployment/model_bias_corrections.json': 'mlb/config/model_bias_corrections.json',
    'mlb-overs/deployment/daily_market_totals.json': 'mlb/config/daily_market_totals.json',
    
    # Utils moved to mlb/utils/
    'mlb-overs/deployment/apply_prediction_override.py': 'mlb/utils/apply_prediction_override.py',
}

# Import statement mappings
IMPORT_MAPPINGS = {
    'from enhanced_bullpen_predictor import': 'sys.path.append("mlb/core"); from enhanced_bullpen_predictor import',
    'from learning_model_predictor import': 'sys.path.append("mlb/core"); from learning_model_predictor import', 
    'from incremental_ultra_80_system import': 'sys.path.append("mlb/systems"); from incremental_ultra_80_system import',
    'import health_gate': 'sys.path.append("mlb/validation"); import health_gate',
    'import probabilities_and_ev': 'sys.path.append("mlb/validation"); import probabilities_and_ev',
}

def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files that might need updating"""
    python_files = []
    
    for path in root_dir.rglob("*.py"):
        # Skip the new mlb directory since it's already updated
        if "mlb" not in str(path) or "mlb-overs" in str(path):
            python_files.append(path)
    
    return python_files

def update_file_paths(content: str) -> Tuple[str, List[str]]:
    """Update file paths in content and return updated content + changes made"""
    updated_content = content
    changes = []
    
    for old_path, new_path in PATH_MAPPINGS.items():
        # Match quoted paths
        old_quoted = f'"{old_path}"'
        new_quoted = f'"{new_path}"'
        
        if old_quoted in updated_content:
            updated_content = updated_content.replace(old_quoted, new_quoted)
            changes.append(f"  📁 {old_quoted} → {new_quoted}")
        
        # Match single quoted paths  
        old_single = f"'{old_path}'"
        new_single = f"'{new_path}'"
        
        if old_single in updated_content:
            updated_content = updated_content.replace(old_single, new_single)
            changes.append(f"  📁 {old_single} → {new_single}")
    
    return updated_content, changes

def update_imports(content: str) -> Tuple[str, List[str]]:
    """Update import statements and return updated content + changes made"""
    updated_content = content
    changes = []
    
    for old_import, new_import in IMPORT_MAPPINGS.items():
        if old_import in updated_content:
            updated_content = updated_content.replace(old_import, new_import)
            changes.append(f"  🔗 {old_import} → {new_import}")
    
    return updated_content, changes

def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single file and return True if changes were made"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
        
        # Apply updates
        updated_content = original_content
        all_changes = []
        
        # Update file paths
        updated_content, path_changes = update_file_paths(updated_content)
        all_changes.extend(path_changes)
        
        # Update import statements
        updated_content, import_changes = update_imports(updated_content)
        all_changes.extend(import_changes)
        
        # Check if any changes were made
        if updated_content != original_content:
            print(f"📝 {file_path}")
            for change in all_changes:
                print(change)
            
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"  ✅ Updated")
            else:
                print(f"  🔍 Would update (dry run)")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Migrate MLB file references after reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--root", default=".", help="Root directory to search for files")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    print(f"🔍 Searching for Python files in: {root_dir}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    python_files = find_python_files(root_dir)
    print(f"📄 Found {len(python_files)} Python files to check")
    
    updated_count = 0
    
    for file_path in python_files:
        if process_file(file_path, args.dry_run):
            updated_count += 1
    
    print(f"\n✅ Migration complete! Updated {updated_count} files")
    
    if args.dry_run:
        print("🔍 Run without --dry-run to apply changes")

if __name__ == "__main__":
    main()
