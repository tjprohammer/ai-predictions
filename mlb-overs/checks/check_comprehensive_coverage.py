import psycopg2
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost",
        database="mlb",
        user="postgres",
        password="mlbpass"
    )

def check_comprehensive_coverage():
    """Check comprehensive data coverage after all enhancements."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    logger.info("🔍 Checking comprehensive data coverage...")
    
    # Get total games count
    cursor.execute("SELECT COUNT(*) FROM enhanced_games")
    total_games = cursor.fetchone()[0]
    logger.info(f"📊 Total games in dataset: {total_games:,}")
    
    # Check offensive stats coverage (our backfill results)
    offensive_stats = ['team_obp', 'team_slg', 'team_iso', 'team_woba', 'team_wrc_plus',
                      'opp_obp', 'opp_slg', 'opp_iso', 'opp_woba', 'opp_wrc_plus']
    
    logger.info("\n🏏 OFFENSIVE STATS COVERAGE:")
    for stat in offensive_stats:
        cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {stat} IS NOT NULL")
        count = cursor.fetchone()[0]
        coverage = (count / total_games) * 100
        logger.info(f"  {stat}: {count:,}/{total_games:,} ({coverage:.1f}%)")
    
    # Check pitcher stats coverage  
    pitcher_stats = ['home_pitcher_whip', 'home_pitcher_season_ip', 'home_pitcher_era',
                    'away_pitcher_whip', 'away_pitcher_season_ip', 'away_pitcher_era']
    
    logger.info("\n⚾ PITCHER STATS COVERAGE:")
    for stat in pitcher_stats:
        cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {stat} IS NOT NULL")
        count = cursor.fetchone()[0]
        coverage = (count / total_games) * 100
        logger.info(f"  {stat}: {count:,}/{total_games:,} ({coverage:.1f}%)")
    
    # Check weather data coverage
    weather_stats = ['humidity', 'pressure', 'wind_speed', 'temperature']
    
    logger.info("\n🌤️ WEATHER DATA COVERAGE:")
    for stat in weather_stats:
        cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {stat} IS NOT NULL")
        count = cursor.fetchone()[0]
        coverage = (count / total_games) * 100
        logger.info(f"  {stat}: {count:,}/{total_games:,} ({coverage:.1f}%)")
    
    # Check venue distribution
    logger.info("\n🏟️ VENUE DISTRIBUTION:")
    cursor.execute("""
        SELECT venue, COUNT(*) as game_count 
        FROM enhanced_games 
        GROUP BY venue 
        ORDER BY game_count DESC
        LIMIT 10
    """)
    
    venue_data = cursor.fetchall()
    for venue, count in venue_data:
        logger.info(f"  {venue}: {count:,} games")
    
    # Check complete feature coverage (games with ALL enhanced features)
    logger.info("\n🎯 COMPLETE FEATURE COVERAGE:")
    
    # Games with ALL offensive stats
    cursor.execute("""
        SELECT COUNT(*) FROM enhanced_games 
        WHERE team_obp IS NOT NULL AND team_slg IS NOT NULL 
        AND team_iso IS NOT NULL AND team_woba IS NOT NULL 
        AND opp_obp IS NOT NULL AND opp_slg IS NOT NULL 
        AND opp_iso IS NOT NULL AND opp_woba IS NOT NULL
    """)
    complete_offensive = cursor.fetchone()[0]
    offensive_coverage = (complete_offensive / total_games) * 100
    logger.info(f"  Complete offensive stats: {complete_offensive:,}/{total_games:,} ({offensive_coverage:.1f}%)")
    
    # Games with ALL pitcher stats
    cursor.execute("""
        SELECT COUNT(*) FROM enhanced_games 
        WHERE home_pitcher_whip IS NOT NULL AND home_pitcher_era IS NOT NULL 
        AND away_pitcher_whip IS NOT NULL AND away_pitcher_era IS NOT NULL
    """)
    complete_pitcher = cursor.fetchone()[0]
    pitcher_coverage = (complete_pitcher / total_games) * 100
    logger.info(f"  Complete pitcher stats: {complete_pitcher:,}/{total_games:,} ({pitcher_coverage:.1f}%)")
    
    # Games with humidity (key weather indicator)
    cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE humidity IS NOT NULL")
    complete_weather = cursor.fetchone()[0]
    weather_coverage = (complete_weather / total_games) * 100
    logger.info(f"  Weather data: {complete_weather:,}/{total_games:,} ({weather_coverage:.1f}%)")
    
    # Games with ALL enhanced features
    cursor.execute("""
        SELECT COUNT(*) FROM enhanced_games 
        WHERE team_obp IS NOT NULL AND team_slg IS NOT NULL 
        AND team_iso IS NOT NULL AND team_woba IS NOT NULL 
        AND opp_obp IS NOT NULL AND opp_slg IS NOT NULL 
        AND opp_iso IS NOT NULL AND opp_woba IS NOT NULL
        AND home_pitcher_whip IS NOT NULL AND home_pitcher_era IS NOT NULL 
        AND away_pitcher_whip IS NOT NULL AND away_pitcher_era IS NOT NULL
        AND humidity IS NOT NULL
    """)
    complete_all = cursor.fetchone()[0]
    all_coverage = (complete_all / total_games) * 100
    logger.info(f"  🚀 ALL enhanced features: {complete_all:,}/{total_games:,} ({all_coverage:.1f}%)")
    
    # Date range coverage
    logger.info("\n📅 DATE RANGE COVERAGE:")
    cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM enhanced_games")
    min_date, max_date = cursor.fetchone()
    logger.info(f"  Date range: {min_date} to {max_date}")
    
    # Monthly distribution of enhanced games
    cursor.execute("""
        SELECT EXTRACT(MONTH FROM game_date) as month, COUNT(*) as game_count
        FROM enhanced_games 
        WHERE team_obp IS NOT NULL AND humidity IS NOT NULL
        GROUP BY EXTRACT(MONTH FROM game_date)
        ORDER BY month
    """)
    
    month_data = cursor.fetchall()
    logger.info("\n📊 ENHANCED GAMES BY MONTH:")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in month_data:
        month_name = month_names[int(month)]
        logger.info(f"  {month_name}: {count:,} games")
    
    logger.info(f"\n✅ Coverage analysis complete!")
    logger.info(f"🎯 Dataset ready for enhanced model training with {all_coverage:.1f}% complete feature coverage!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_comprehensive_coverage()
