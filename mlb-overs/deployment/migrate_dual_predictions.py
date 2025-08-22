#!/usr/bin/env python3
"""
Database Migration: Add Dual Prediction Columns
==============================================
Adds columns to enhanced_games table to track both original and learning model predictions.

New columns:
- predicted_total_original: Original EnhancedBullpenPredictor prediction
- predicted_total_learning: 203-feature learning model prediction  
- prediction_timestamp: When predictions were made
- prediction_comparison: JSON field for storing comparison metrics

This allows tracking both models in the UI and comparing their performance.
"""

import os
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment or default"""
    return os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

def migrate_dual_predictions():
    """Add dual prediction columns to enhanced_games table"""
    
    log.info("Starting dual prediction column migration...")
    
    # Connect to database
    engine = create_engine(get_database_url())
    
    migration_queries = [
        # Add new columns for dual predictions
        """
        ALTER TABLE enhanced_games 
        ADD COLUMN IF NOT EXISTS predicted_total_original NUMERIC;
        """,
        
        """
        ALTER TABLE enhanced_games 
        ADD COLUMN IF NOT EXISTS predicted_total_learning NUMERIC;
        """,
        
        """
        ALTER TABLE enhanced_games 
        ADD COLUMN IF NOT EXISTS prediction_timestamp TIMESTAMP DEFAULT NOW();
        """,
        
        """
        ALTER TABLE enhanced_games 
        ADD COLUMN IF NOT EXISTS prediction_comparison JSONB;
        """,
        
        # Add helpful comments
        """
        COMMENT ON COLUMN enhanced_games.predicted_total_original 
        IS 'Prediction from original EnhancedBullpenPredictor model';
        """,
        
        """
        COMMENT ON COLUMN enhanced_games.predicted_total_learning 
        IS 'Prediction from 203-feature adaptive learning model';
        """,
        
        """
        COMMENT ON COLUMN enhanced_games.prediction_timestamp 
        IS 'Timestamp when predictions were generated';
        """,
        
        """
        COMMENT ON COLUMN enhanced_games.prediction_comparison 
        IS 'JSON metadata about prediction comparison (correlation, difference, etc.)';
        """,
        
        # Create index for efficient querying
        """
        CREATE INDEX IF NOT EXISTS idx_enhanced_games_prediction_timestamp 
        ON enhanced_games(prediction_timestamp);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_enhanced_games_dual_predictions 
        ON enhanced_games(date, predicted_total_original, predicted_total_learning) 
        WHERE predicted_total_original IS NOT NULL AND predicted_total_learning IS NOT NULL;
        """
    ]
    
    try:
        with engine.begin() as conn:
            for i, query in enumerate(migration_queries, 1):
                log.info(f"Executing migration step {i}/{len(migration_queries)}...")
                conn.execute(text(query))
                log.info(f"✅ Step {i} completed")
        
        log.info("🎉 Migration completed successfully!")
        
        # Verify the migration
        verify_migration(engine)
        
    except Exception as e:
        log.error(f"❌ Migration failed: {e}")
        raise
    finally:
        engine.dispose()

def verify_migration(engine):
    """Verify that the migration was successful"""
    log.info("Verifying migration...")
    
    # Check that columns exist
    verification_query = text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        AND column_name IN (
            'predicted_total_original',
            'predicted_total_learning', 
            'prediction_timestamp',
            'prediction_comparison'
        )
        ORDER BY column_name;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(verification_query)
        columns = result.fetchall()
        
        expected_columns = {
            'predicted_total_original': 'numeric',
            'predicted_total_learning': 'numeric', 
            'prediction_timestamp': 'timestamp without time zone',
            'prediction_comparison': 'jsonb'
        }
        
        found_columns = {row[0]: row[1] for row in columns}
        
        log.info("Migration verification:")
        for col_name, expected_type in expected_columns.items():
            if col_name in found_columns:
                actual_type = found_columns[col_name]
                if expected_type in actual_type:
                    log.info(f"✅ {col_name}: {actual_type}")
                else:
                    log.warning(f"⚠️ {col_name}: expected {expected_type}, got {actual_type}")
            else:
                log.error(f"❌ {col_name}: NOT FOUND")
        
        # Check row count to make sure table still works
        count_query = text("SELECT COUNT(*) FROM enhanced_games;")
        row_count = conn.execute(count_query).scalar()
        log.info(f"📊 enhanced_games table has {row_count} rows")

def create_dual_prediction_view():
    """Create a view for easy dual prediction analysis"""
    log.info("Creating dual prediction analysis view...")
    
    engine = create_engine(get_database_url())
    
    view_query = text("""
        CREATE OR REPLACE VIEW dual_prediction_analysis AS
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            predicted_total,
            predicted_total_original,
            predicted_total_learning,
            prediction_timestamp,
            total_runs,
            market_total,
            
            -- Prediction differences
            (predicted_total_learning - predicted_total_original) AS learning_vs_original_diff,
            (predicted_total_learning - market_total) AS learning_vs_market_diff,
            (predicted_total_original - market_total) AS original_vs_market_diff,
            
            -- Accuracy metrics (when actual results available)
            CASE 
                WHEN total_runs IS NOT NULL THEN ABS(predicted_total_original - total_runs)
                ELSE NULL 
            END AS original_error,
            
            CASE 
                WHEN total_runs IS NOT NULL THEN ABS(predicted_total_learning - total_runs)
                ELSE NULL 
            END AS learning_error,
            
            CASE 
                WHEN total_runs IS NOT NULL THEN ABS(market_total - total_runs)
                ELSE NULL 
            END AS market_error,
            
            -- Performance indicators
            CASE 
                WHEN total_runs IS NOT NULL AND predicted_total_learning IS NOT NULL AND predicted_total_original IS NOT NULL
                THEN (ABS(predicted_total_learning - total_runs) < ABS(predicted_total_original - total_runs))
                ELSE NULL
            END AS learning_model_better
            
        FROM enhanced_games
        WHERE predicted_total_original IS NOT NULL 
           OR predicted_total_learning IS NOT NULL
        ORDER BY date DESC, prediction_timestamp DESC;
    """)
    
    try:
        with engine.begin() as conn:
            conn.execute(view_query)
        log.info("✅ dual_prediction_analysis view created")
    except Exception as e:
        log.error(f"Failed to create view: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    try:
        migrate_dual_predictions()
        create_dual_prediction_view()
        
        print("\n" + "="*60)
        print("🎉 DUAL PREDICTION MIGRATION COMPLETE!")
        print("="*60)
        print("\nNew columns added to enhanced_games:")
        print("  - predicted_total_original: Original model predictions")
        print("  - predicted_total_learning: Learning model predictions") 
        print("  - prediction_timestamp: When predictions were made")
        print("  - prediction_comparison: Comparison metadata")
        print("\nNew view created:")
        print("  - dual_prediction_analysis: Easy comparison queries")
        print("\nYou can now run your daily workflow with dual predictions!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
