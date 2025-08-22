#!/usr/bin/env python3
"""
Analyze current dataset to determine Phase 3 enhancement opportunities
"""

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_phase3_opportunities():
    """Analyze what Phase 3 enhancements are needed"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        print("🎯 PHASE 3 ENHANCEMENT ANALYSIS")
        print("=" * 60)
        
        # Check current column coverage
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            ORDER BY column_name
        """)
        
        all_columns = cursor.fetchall()
        total_columns = len(all_columns)
        print(f"📊 Current dataset: {total_columns} columns")
        
        # Categorize existing enhancements
        bullpen_cols = [col for col, _ in all_columns if 'bullpen' in col]
        trend_cols = [col for col, _ in all_columns if any(x in col for x in ['l7', 'l14', 'l20'])]
        weather_cols = [col for col, _ in all_columns if any(x in col for x in ['temp', 'wind', 'humidity', 'weather'])]
        lineup_cols = [col for col, _ in all_columns if any(x in col for x in ['lineup', 'injury', 'roster'])]
        
        print(f"\n📈 COMPLETED ENHANCEMENTS:")
        print(f"✅ Phase 1 - Bullpen data: {len(bullpen_cols)} columns")
        print(f"✅ Phase 2 - Recent trends: {len(trend_cols)} columns")
        
        print(f"\n🔍 POTENTIAL PHASE 3 OPPORTUNITIES:")
        print(f"🌤️  Weather data: {len(weather_cols)} columns")
        print(f"👥 Lineup/injury data: {len(lineup_cols)} columns")
        
        # Check for missing critical data
        cursor.execute("""
            SELECT 
                COUNT(*) as total_games,
                COUNT(home_sp_name) as with_starting_pitchers,
                COUNT(home_bp_er) as with_bullpen_game_stats,
                COUNT(wind_speed) as with_weather,
                COUNT(home_team_runs_l7) as with_trends,
                COUNT(home_bullpen_era_l30) as with_bullpen_trends
            FROM enhanced_games
            WHERE date >= '2025-03-20'
        """)
        
        coverage = cursor.fetchone()
        total, sp, bp_game, weather, trends, bp_trends = coverage
        
        print(f"\n📊 DATA COVERAGE ANALYSIS (Recent games):")
        print(f"Total games: {total}")
        print(f"Starting pitcher data: {sp}/{total} ({100*sp/total:.1f}%)")
        print(f"Bullpen game stats: {bp_game}/{total} ({100*bp_game/total:.1f}%)")
        print(f"Weather data: {weather}/{total} ({100*weather/total:.1f}%)")
        print(f"Recent trends: {trends}/{total} ({100*trends/total:.1f}%)")
        print(f"Bullpen trends: {bp_trends}/{total} ({100*bp_trends/total:.1f}%)")
        
        # Identify Phase 3 priorities
        print(f"\n🎯 PHASE 3 ENHANCEMENT PRIORITIES:")
        
        if weather < total * 0.8:
            weather_gap = total - weather
            print(f"🌤️  HIGH PRIORITY: Weather enhancement ({weather_gap} games missing weather data)")
            print(f"     - Temperature, wind speed, humidity, conditions")
            print(f"     - Dome vs outdoor stadium factors")
            print(f"     - Weather impact on over/under performance")
        
        # Check for advanced pitcher metrics
        cursor.execute("""
            SELECT 
                COUNT(home_sp_xera) as with_advanced_sp,
                COUNT(home_sp_whip_l5) as with_recent_sp_trends,
                COUNT(home_closer_saves) as with_closer_data
            FROM enhanced_games 
            WHERE date >= '2025-03-20'
            LIMIT 1
        """)
        
        try:
            advanced = cursor.fetchone()
            if advanced and advanced[0] is not None:
                sp_advanced, sp_recent, closer = advanced
            else:
                sp_advanced = sp_recent = closer = 0
        except:
            sp_advanced = sp_recent = closer = 0
        
        if sp_advanced < total * 0.8:
            print(f"⚾ MEDIUM PRIORITY: Advanced pitcher metrics")
            print(f"     - xERA, FIP, SIERRA for starting pitchers")
            print(f"     - Recent form trends (L5 games)")
            print(f"     - Platoon splits (vs LHB/RHB)")
        
        if closer < total * 0.8:
            print(f"🎯 MEDIUM PRIORITY: Closer/save situation data")
            print(f"     - Closer availability and recent usage")
            print(f"     - Save situation performance")
            print(f"     - High-leverage relief pitcher data")
        
        # Check for team-level advanced metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(home_team_wrc_plus) as with_wrc,
                COUNT(home_team_era_plus) as with_era_plus,
                COUNT(home_team_babip) as with_babip
            FROM enhanced_games 
            WHERE date >= '2025-03-20'
            LIMIT 1
        """)
        
        try:
            team_advanced = cursor.fetchone()
            if team_advanced and team_advanced[1] is not None:
                wrc_count = team_advanced[1]
            else:
                wrc_count = 0
        except:
            wrc_count = 0
        
        if wrc_count < total * 0.8:
            print(f"📈 LOW PRIORITY: Team advanced metrics")
            print(f"     - wRC+, ERA+, BABIP")
            print(f"     - Park factors and adjustments")
            print(f"     - Strength of schedule metrics")
        
        # Sample current data to see what's working well
        cursor.execute("""
            SELECT home_team, away_team, date, total_runs,
                   home_bullpen_era_l30, home_team_runs_l7,
                   CASE WHEN wind_speed IS NOT NULL THEN 'Yes' ELSE 'No' END as has_weather
            FROM enhanced_games 
            WHERE date >= '2025-08-15'
            ORDER BY date DESC
            LIMIT 5
        """)
        
        recent_samples = cursor.fetchall()
        print(f"\n📋 RECENT GAMES DATA SAMPLE:")
        print("Date       | Teams                    | Total | BP ERA | L7 Runs | Weather")
        print("-" * 75)
        for sample in recent_samples:
            home, away, date, total, bp_era, l7_runs, weather = sample
            bp_era_str = f"{bp_era:.2f}" if bp_era else "NULL"
            l7_str = f"{l7_runs:.1f}" if l7_runs else "NULL"
            print(f"{date} | {away[:8]:8} @ {home[:8]:8} | {total:5} | {bp_era_str:6} | {l7_str:7} | {weather}")
        
        print(f"\n🚀 RECOMMENDED PHASE 3: WEATHER & STADIUM ENHANCEMENT")
        print("=" * 60)
        print("🌤️  Priority: Add comprehensive weather data")
        print("   - Temperature, wind speed/direction, humidity")
        print("   - Precipitation, weather conditions")
        print("   - Stadium type (dome, outdoor, retractable roof)")
        print("   - Historical weather impact on scoring")
        print("\n📊 Expected Benefits:")
        print("   - Weather significantly impacts over/under outcomes")
        print("   - Wind direction affects home run rates")
        print("   - Temperature affects ball flight distance")
        print("   - Humidity impacts pitcher grip and stamina")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Phase 3 analysis failed: {e}")

if __name__ == "__main__":
    analyze_phase3_opportunities()
