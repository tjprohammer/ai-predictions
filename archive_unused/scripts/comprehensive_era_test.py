#!/usr/bin/env python3
"""
Comprehensive ERA Pipeline Test
Tests the ERA ingestor with real data and shows actual calculations
"""

imp        logger.info(f"\n🔄 Testing rolling calculations:")
        for window in [3, 5, 10]:
            rolling_data = calculate_rolling_era(pitcher_id, test_date, window, DATABASE_URL_PSYCOPG2) os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add the mlb-overs directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mlb-overs'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Correct database URLs for different connection types
DATABASE_URL_PSYCOPG2 = 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'
DATABASE_URL_SQLALCHEMY = 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'

def test_database_connection():
    """Test database connectivity and show data overview"""
    logger.info("Testing database connection...")
    
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL_PSYCOPG2)
        cursor = conn.cursor()
        
        # Basic connection test
        cursor.execute("SELECT current_database(), current_user")
        db_info = cursor.fetchone()
        logger.info(f"✅ Connected to database: {db_info[0]} as user: {db_info[1]}")
        
        # Check pitchers_starts data
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN ip IS NOT NULL AND er IS NOT NULL THEN 1 END) as complete_records,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM pitchers_starts
        """)
        stats = cursor.fetchone()
        logger.info(f"Total records: {stats[0]}")
        logger.info(f"Complete IP/ER records: {stats[1]}")
        logger.info(f"Date range: {stats[2]} to {stats[3]}")
        
        # Find pitchers with most data
        cursor.execute("""
            SELECT pitcher_id, COUNT(*) as games, 
                   COUNT(CASE WHEN ip IS NOT NULL AND er IS NOT NULL THEN 1 END) as complete_games
            FROM pitchers_starts 
            WHERE ip IS NOT NULL AND er IS NOT NULL
            GROUP BY pitcher_id 
            HAVING COUNT(CASE WHEN ip IS NOT NULL AND er IS NOT NULL THEN 1 END) >= 5
            ORDER BY complete_games DESC 
            LIMIT 10
        """)
        
        pitchers = cursor.fetchall()
        logger.info("Top pitchers with complete data:")
        for pitcher in pitchers:
            logger.info(f"  Pitcher {pitcher[0]}: {pitcher[2]} complete games")
        
        conn.close()
        return True, pitchers[0][0] if pitchers else "686752"  # Return best pitcher ID
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False, None

def test_era_calculations_detailed(pitcher_id):
    """Test ERA calculations with detailed output"""
    logger.info(f"Testing detailed ERA calculations for pitcher {pitcher_id}...")
    
    try:
        from ingestors.era_ingestor import get_pitcher_era_stats, calculate_rolling_era
        import psycopg2
        
        # Get raw data to verify calculations
        conn = psycopg2.connect(DATABASE_URL_PSYCOPG2)
        cursor = conn.cursor()
        
        # Get pitcher's recent games
        cursor.execute("""
            SELECT date, ip, er, 
                   CASE WHEN ip > 0 THEN ROUND((er * 9.0 / ip)::numeric, 2) ELSE NULL END as game_era
            FROM pitchers_starts 
            WHERE pitcher_id = %s AND ip IS NOT NULL AND er IS NOT NULL
            ORDER BY date DESC 
            LIMIT 10
        """, (pitcher_id,))
        
        games = cursor.fetchall()
        logger.info(f"Recent games for pitcher {pitcher_id}:")
        logger.info("Date       | IP   | ER | Game ERA")
        logger.info("-" * 35)
        
        total_ip = 0
        total_er = 0
        for i, game in enumerate(games):
            date, ip, er, game_era = game
            total_ip += ip
            total_er += er
            logger.info(f"{date} | {ip:4.1f} | {er:2d} | {game_era if game_era else 'N/A':7}")
            
            # Show rolling calculations
            if i == 2:  # After 3 games (L3)
                l3_era = (total_er * 9.0) / total_ip if total_ip > 0 else None
                logger.info(f"  → L3 ERA (manual): {l3_era:.2f}")
            elif i == 4:  # After 5 games (L5)
                l5_era = (total_er * 9.0) / total_ip if total_ip > 0 else None
                logger.info(f"  → L5 ERA (manual): {l5_era:.2f}")
        
        logger.info(f"Total IP: {total_ip:.1f}, Total ER: {total_er}")
        if total_ip > 0:
            manual_era = (float(total_er) * 9.0) / float(total_ip)
            logger.info(f"Manual season ERA: {manual_era:.2f}")
        
        conn.close()
        
        # Now test our ERA ingestor
        test_date = "2025-08-12"  # Today
        era_stats = get_pitcher_era_stats(pitcher_id, test_date, DATABASE_URL_PSYCOPG2)
        
        if era_stats:
            logger.info(f"\n🎯 ERA Ingestor Results for pitcher {pitcher_id}:")
            for key, value in era_stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.warning("❌ No ERA stats returned from ingestor")
            return False
        
        # Test individual rolling calculations
        logger.info(f"\n🔄 Testing rolling calculations:")
        for window in [3, 5, 10]:
            rolling_data = calculate_rolling_era(pitcher_id, test_date, window, DATABASE_URL)
            if rolling_data and rolling_data.get('era'):
                logger.info(f"  L{window} ERA: {rolling_data['era']:.2f} ({rolling_data['games']} games)")
            else:
                logger.info(f"  L{window} ERA: No data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ERA calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test the pipeline integration"""
    logger.info("Testing pipeline integration...")
    
    try:
        # Test build_features integration
        logger.info("Testing build_features.py integration...")
        import features.build_features
        from ingestors.era_ingestor import backfill_todays_pitchers
        
        # Test daily backfill function
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Testing backfill for date: {today}")
        
        # This would normally backfill today's pitchers
        # For testing, we'll just verify the function exists and is callable
        logger.info("✅ backfill_todays_pitchers function available")
        
        # Test infer.py integration
        logger.info("Testing infer.py integration...")
        from models.infer import get_pitcher_era_stats as infer_era_stats
        from sqlalchemy import create_engine
        
        engine = create_engine(DATABASE_URL)
        test_pitcher_id = "686752"
        test_date = "2025-08-12"
        
        # Test the updated infer function
        infer_stats = infer_era_stats(engine, test_pitcher_id, test_date)
        if infer_stats:
            logger.info("✅ infer.py ERA integration working:")
            for key, value in infer_stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.warning("❌ infer.py returned no ERA stats")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive ERA pipeline test"""
    logger.info("="*60)
    logger.info("COMPREHENSIVE ERA PIPELINE TEST")
    logger.info("="*60)
    
    results = {}
    
    # Test database connection and get a good pitcher ID
    db_success, best_pitcher_id = test_database_connection()
    results['Database Connection'] = db_success
    
    if db_success and best_pitcher_id:
        # Test detailed ERA calculations
        results['ERA Calculations'] = test_era_calculations_detailed(best_pitcher_id)
        
        # Test pipeline integration
        results['Pipeline Integration'] = test_pipeline_integration()
    else:
        logger.error("Skipping further tests due to database connection failure")
        results['ERA Calculations'] = False
        results['Pipeline Integration'] = False
    
    # Summary
    logger.info("="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ERA pipeline is fully functional!")
        logger.info("📊 You can now see actual ERA calculations and verify accuracy!")
    else:
        logger.warning("⚠️ Some tests failed - check the details above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
