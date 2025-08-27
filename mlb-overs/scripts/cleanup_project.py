#!/usr/bin/env python3
"""
Project Cleanup Script - Move one-time use files to archive
Based on LEARNING_MODEL_DOCUMENTATION.md analysis
"""

import os
import shutil
from pathlib import Path

# Define the project root
PROJECT_ROOT = Path(__file__).parent

# Files to KEEP in root directory (production/important files)
KEEP_FILES = {
    # Documentation
    "README.md",
    "LEARNING_MODEL_DOCUMENTATION.md",
    "HandyCommands.md",
    
    # Environment and config
    ".env",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    
    # Current production data files
    "comprehensive_real_umpire_stats_2025.csv",
    "comprehensive_real_umpire_stats_2025.json",
    "daily_market_totals.json",
    "daily_predictions.json",
    
    # Migration files (might be needed)
    "add_umpire_columns.sql",
    "migration_add_umpire_columns.sql",
    
    # Current workflow files
    "enhanced_daily_workflow.py",
    "daily_learning_pipeline.py",
}

# Files to ARCHIVE (one-time use, debug, analysis files)
ARCHIVE_FILES = {
    # Debug files
    "debug_actual_features.py",
    "debug_bias_application.py", 
    "debug_boxscore.py",
    "debug_confidence.py",
    "debug_market_matching.py",
    "debug_model_features.py",
    "debug_query.py",
    "debug_starters.py",
    "debug_time_filtering.py",
    "diagnose_model.py",
    
    # Analysis files (one-time use)
    "analyze_30_day_bias.py",
    "analyze_enhanced_results.py",
    "analyze_features.py",
    "analyze_feature_mismatch.py",
    "analyze_thresholds.py",
    "analyze_todays_performance.py",
    "advanced_feature_fixes.py",
    "advanced_performance_analyzer.py",
    
    # Fix/surgical files (one-time use)
    "apply_surgical_fixes.py",
    "comprehensive_critical_fixes.py",
    "corrected_surgical_fix.py",
    "create_stronger_bias_corrections.py",
    "create_stronger_corrections.py",
    "feature_quality_fixes.py",
    "focused_critical_fixes.py",
    "surgical_critical_fixes.py",
    "surgical_fix_complete.py",
    "working_surgical_runner.py",
    
    # Data collection/enhancement (one-time use)
    "comprehensive_data_enhancement.py",
    "comprehensive_real_umpire_collector.py",
    "collect_all_umpire_assignments.py",
    "collect_final_scores.py",
    "real_data_feature_fixer.py",
    "real_umpire_stats_collector.py",
    "thorough_real_data_fixer.py",
    
    # Training/model building (one-time use)
    "comprehensive_training_builder.py",
    "enhanced_model_training.py",
    "historical_training_builder.py",
    "model_performance_enhancer.py",
    "model_calibration_solution.py",
    
    # Verification/checking (one-time use)
    "check_schema.py",
    "check_umpire_data_exists.py",
    "verify_date_fix.py",
    "verify_detailed_umpire_stats.py",
    "verify_umpire_stats.py",
    "inspect_broken_model.py",
    "inspect_schema.py",
    
    # Data exploration (one-time use)
    "explore_data_sources.py",
    "data_requirements_analysis.py",
    "compare_features.py",
    
    # Historical analysis (one-time use)
    "historical_backtesting.py",
    "historical_performance_analysis.py",
    
    # Learning system files (replaced by production versions)
    "continuous_learning_system.py",
    "fixed_20_session_learning_system.py",
    "ten_session_learning_system.py",
    "integrated_learning_workflow.py",
    "live_learning_predictor.py",
    
    # Team analysis (one-time use)
    "comprehensive_team_analysis.py",
    "final_team_analysis.py",
    "team_data_backfill.py",
    "team_data_integration.py",
    "fix_team_data_integration.py",
    
    # Bias correction (one-time use)
    "reset_bias_corrections.py",
    "update_bias_corrections.py",
    "urgent_bias_correction.py",
    
    # Backfill/progress tracking (one-time use)
    "backfill_progress.py",
    "phase2_data_recollection_runner.py",
    
    # Umpire data generation (one-time use)
    "generate_umpire_stats_database.py",
    "database_real_umpire_analyzer.py",
    "ingest_umpire_stats.py",
    
    # Result tracking/updating (one-time use)
    "game_result_tracker.py",
    "manual_result_updater.py",
    "simple_results_checker.py",
    "recent_prediction_tracker.py",
    
    # Performance tracking (one-time use)
    "enhanced_prediction_tracker.py",
    "enhanced_score_collector.py",
    "learning_impact_monitor.py",
    "weekly_performance_tracker.py",
    "working_14_day_analyzer.py",
    "workflow_impact_analysis.py",
    
    # Pipeline integration (one-time use)
    "pipeline_integration.py",
    "enhanced_learning_integration.py",
    
    # Model fixes (one-time use)
    "fix_model_retrain.py",
    
    # Game display (one-time use)
    "enhanced_game_display.py",
    
    # Test files (one-time use)
    "test_api.py",
    
    # PowerShell scripts (one-time use)
    "enhanced_gameday.ps1",
    "enhanced_learning_gameday.ps1",
    "enhanced_learning_gameday_fixed.ps1",
}

# JSON/CSV files to archive (old predictions/analysis)
ARCHIVE_JSON_CSV = {
    "betting_summary_2025-08-21.json",
    "betting_summary_2025-08-22.json",
    "enhanced_predictions_2025-08-21.json", 
    "enhanced_predictions_2025-08-22.json",
    "high_accuracy_predictions_2025-08-22.json",
    "learning_predictions_2025-08-20.json",
    "daily_learning_log.json",
    "historical_backtest_results.json",
    "missing_odds.csv",
    "model_bias_corrections.json",
    "model_bias_corrections_new.json",
    
    # Performance analysis files
    "performance_analysis_20250819_132322.json",
    "performance_analysis_20250819_214432.json", 
    "performance_analysis_20250819_214930.json",
    "performance_analysis_20250819_220703.json",
    "performance_analysis_20250819_233707.json",
    
    # Prediction analysis CSV files
    "prediction_analysis_20250819_131346.csv",
    "prediction_analysis_20250819_132249.csv",
    "prediction_analysis_20250819_214943.csv", 
    "prediction_analysis_20250821_213040.csv",
}

# Documentation files to archive (one-time guides)
ARCHIVE_DOCS = {
    "daily_workflow_transition_guide.md",
    "DATASET_ENHANCEMENT_STATUS.md",
    "ENHANCED_MODEL_LEARNING_RESULTS.md",
    "ENHANCED_WORKFLOW_COMPLETE.md",
    "LEARNING_INTEGRATION_GUIDE.md",
    "LEARNING_LOOP_COMPLETE.md",
    "LIVE_GAME_FILTERING_COMPLETE.md",
    "MODEL_TRAINING_COMPLETE.md",
    "PIPELINE_STATUS_FIXED.md",
    "SCHEMA_AWARE_IMPLEMENTATION.md",
    "SYSTEM_ANALYSIS.md",
}

def create_archive_directory():
    """Create archive directory if it doesn't exist"""
    archive_dir = PROJECT_ROOT / "archive_root_cleanup"
    archive_dir.mkdir(exist_ok=True)
    return archive_dir

def move_files_to_archive(file_set, archive_dir, description):
    """Move a set of files to archive directory"""
    moved_count = 0
    for filename in file_set:
        source_path = PROJECT_ROOT / filename
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

def main():
    """Main cleanup function"""
    print("🧹 Starting AI Predictions Project Cleanup")
    print("=" * 50)
    
    # Create archive directory
    archive_dir = create_archive_directory()
    print(f"Created archive directory: {archive_dir}")
    
    total_moved = 0
    
    # Move different categories of files
    total_moved += move_files_to_archive(ARCHIVE_FILES, archive_dir, "Python script files")
    total_moved += move_files_to_archive(ARCHIVE_JSON_CSV, archive_dir, "JSON/CSV data files")
    total_moved += move_files_to_archive(ARCHIVE_DOCS, archive_dir, "Documentation files")
    
    print("\n" + "=" * 50)
    print(f"🎉 Cleanup Complete!")
    print(f"Total files moved to archive: {total_moved}")
    
    # List remaining files in root
    remaining_files = [f for f in os.listdir(PROJECT_ROOT) 
                      if os.path.isfile(PROJECT_ROOT / f) 
                      and not f.startswith('.') 
                      and f != 'cleanup_project.py']
    
    print(f"\nRemaining files in root directory: {len(remaining_files)}")
    for file in sorted(remaining_files):
        print(f"  ✓ {file}")
    
    print(f"\nArchived files are in: {archive_dir}")
    print("Review the archive directory and delete if confirmed the files are no longer needed.")

if __name__ == "__main__":
    main()
