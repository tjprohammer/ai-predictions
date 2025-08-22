#!/usr/bin/env python3
"""
Phase 2 Data Re-Collection Runner - REAL DATA ONLY
==================================================

This script re-runs Phase 2 data collection for all training games 
to replace the corrupted constant values with real trend data.

NO FALLBACKS - Uses only actual game performance data.

Steps:
1. Clear existing Phase 2 trends data 
2. Re-collect using real data only system
3. Validate data quality improvement
4. Generate quality report

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime, timedelta
import subprocess

# Add the data_collection directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mlb-overs', 'data_collection'))

from phase2_real_trends_only import RealTrendsCollector

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'mlb',
    'user': 'mlbuser',
    'password': 'mlbpass'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clear_phase2_data():
    """Clear existing Phase 2 trends data to start fresh"""
    
    logging.info("🧹 Clearing existing Phase 2 trends data...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Clear all L7, L14, L20, L30 trend columns
        cursor.execute("""
            UPDATE enhanced_games SET
                home_team_runs_l7 = NULL,
                home_team_runs_allowed_l7 = NULL,
                away_team_runs_l7 = NULL, 
                away_team_runs_allowed_l7 = NULL,
                home_team_ops_l14 = NULL,
                away_team_ops_l14 = NULL,
                home_team_runs_l20 = NULL,
                home_team_runs_allowed_l20 = NULL,
                home_team_ops_l20 = NULL,
                away_team_runs_l20 = NULL,
                away_team_runs_allowed_l20 = NULL,
                away_team_ops_l20 = NULL,
                home_team_runs_l30 = NULL,
                home_team_ops_l30 = NULL,
                away_team_runs_l30 = NULL,
                away_team_ops_l30 = NULL,
                home_team_form_rating = NULL,
                away_team_form_rating = NULL
            WHERE date >= '2025-03-01'
        """)
        
        rows_cleared = cursor.rowcount
        conn.commit()
        
        logging.info(f"✅ Cleared Phase 2 data from {rows_cleared} games")
        
        return rows_cleared
        
    except Exception as e:
        logging.error(f"❌ Error clearing Phase 2 data: {str(e)}")
        conn.rollback()
        return 0
        
    finally:
        cursor.close()
        conn.close()

def check_data_quality_before():
    """Check data quality before re-collection"""
    
    logging.info("🔍 Checking data quality before re-collection...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Check for constant values (the previous problem)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_training_games,
                COUNT(home_team_runs_l7) as games_with_l7,
                COUNT(CASE WHEN home_team_runs_l7 = 4.8 THEN 1 END) as constant_48_values,
                COUNT(CASE WHEN away_team_runs_l7 = 4.8 THEN 1 END) as away_constant_48
            FROM enhanced_games 
            WHERE date >= '2025-08-12' AND date <= '2025-08-21'
            AND (home_team_runs_l7 IS NOT NULL OR away_team_runs_l7 IS NOT NULL)
        """)
        
        result = cursor.fetchone()
        
        if result:
            total_games, games_with_l7, home_constant, away_constant = result
            logging.info(f"📊 Before cleanup: {total_games} total games")
            logging.info(f"📊 Games with L7 data: {games_with_l7}")
            logging.info(f"📊 Home constant 4.8 values: {home_constant}")
            logging.info(f"📊 Away constant 4.8 values: {away_constant}")
            
            return {
                'total_games': total_games,
                'games_with_data': games_with_l7,
                'constant_values': home_constant + away_constant
            }
    
    except Exception as e:
        logging.error(f"❌ Error checking data quality: {str(e)}")
        
    finally:
        cursor.close()
        conn.close()
    
    return None

def check_data_quality_after():
    """Check data quality after re-collection"""
    
    logging.info("🔍 Checking data quality after re-collection...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Check for variety in L7 runs values
        cursor.execute("""
            SELECT 
                COUNT(*) as total_training_games,
                COUNT(home_team_runs_l7) as games_with_l7,
                COUNT(DISTINCT home_team_runs_l7) as unique_home_l7_values,
                COUNT(DISTINCT away_team_runs_l7) as unique_away_l7_values,
                MIN(home_team_runs_l7) as min_home_l7,
                MAX(home_team_runs_l7) as max_home_l7,
                AVG(home_team_runs_l7) as avg_home_l7,
                COUNT(CASE WHEN home_team_runs_l7 = 4.8 THEN 1 END) as remaining_constant_48
            FROM enhanced_games 
            WHERE date >= '2025-08-12' AND date <= '2025-08-21'
            AND home_team_runs_l7 IS NOT NULL
        """)
        
        result = cursor.fetchone()
        
        if result:
            total, with_l7, unique_home, unique_away, min_l7, max_l7, avg_l7, constant_48 = result
            
            logging.info(f"📊 After cleanup: {total} total games")
            logging.info(f"📊 Games with L7 data: {with_l7}")
            logging.info(f"📊 Unique home L7 values: {unique_home}")
            logging.info(f"📊 Unique away L7 values: {unique_away}")
            logging.info(f"📊 L7 range: {min_l7:.1f} to {max_l7:.1f} (avg: {avg_l7:.1f})")
            logging.info(f"📊 Remaining constant 4.8 values: {constant_48}")
            
            return {
                'total_games': total,
                'games_with_data': with_l7,
                'unique_values': unique_home,
                'value_range': (min_l7, max_l7),
                'average': avg_l7,
                'constant_values': constant_48
            }
    
    except Exception as e:
        logging.error(f"❌ Error checking data quality: {str(e)}")
        
    finally:
        cursor.close()
        conn.close()
    
    return None

def main():
    """Main execution function"""
    
    print("🎯 PHASE 2 DATA RE-COLLECTION - REAL DATA ONLY")
    print("=" * 60)
    print("This will replace corrupted constant values with real trend data")
    print("🚫 NO FALLBACKS - ONLY ACTUAL GAME PERFORMANCE DATA")
    print()
    
    # Step 1: Check data quality before cleanup
    print("📊 STEP 1: Checking current data quality...")
    before_stats = check_data_quality_before()
    
    if before_stats and before_stats['constant_values'] > 0:
        print(f"⚠️ Found {before_stats['constant_values']} constant values - cleanup needed")
    
    # Step 2: Clear existing Phase 2 data
    print("\n🧹 STEP 2: Clearing existing Phase 2 trends data...")
    cleared_count = clear_phase2_data()
    
    if cleared_count == 0:
        print("❌ Failed to clear existing data")
        return False
    
    # Step 3: Re-collect using real data only system
    print("\n🚀 STEP 3: Re-collecting Phase 2 data with real values only...")
    
    collector = RealTrendsCollector()
    
    try:
        # Process training period (August 12-21, 2025)
        summary = collector.process_recent_games('2025-08-12', '2025-08-21')
        
        print(f"✅ Re-collection complete: {summary['updated']}/{summary['total_games']} games updated")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Re-collection failed: {str(e)}")
        return False
        
    finally:
        collector.close()
    
    # Step 4: Validate data quality improvement
    print("\n🔍 STEP 4: Validating data quality improvement...")
    after_stats = check_data_quality_after()
    
    if after_stats:
        print(f"✅ Data quality after cleanup:")
        print(f"   - Games with L7 data: {after_stats['games_with_data']}")
        print(f"   - Unique L7 values: {after_stats['unique_values']}")
        print(f"   - Value range: {after_stats['value_range'][0]:.1f} to {after_stats['value_range'][1]:.1f}")
        print(f"   - Average L7 runs: {after_stats['average']:.1f}")
        print(f"   - Remaining constant values: {after_stats['constant_values']}")
        
        # Success criteria
        if after_stats['unique_values'] >= 10 and after_stats['constant_values'] == 0:
            print("\n🎉 SUCCESS: Data quality significantly improved!")
            print("✅ Variety in L7 values restored")
            print("✅ Constant 4.8 values eliminated")
            print("✅ Ready for 120-day model training")
            return True
        else:
            print("\n⚠️ Data quality partially improved but issues remain")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Phase 2 data re-collection COMPLETE!")
        print("Ready to proceed with comprehensive 120-day model training")
    else:
        print("\n❌ Phase 2 data re-collection had issues")
        print("Manual investigation may be required")
