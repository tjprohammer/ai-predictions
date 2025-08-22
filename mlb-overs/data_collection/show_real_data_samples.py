#!/usr/bin/env python3
"""
Display real data samples from the enhanced dataset
"""

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_real_data_samples():
    """Show 10 samples of the real enhanced data"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        print("🎯 REAL ENHANCED MLB DATA SAMPLES")
        print("=" * 80)
        
        # Get 10 random samples with both bullpen and trend data
        cursor.execute("""
            SELECT 
                date, home_team, away_team, 
                home_score, away_score, total_runs,
                -- Phase 1: Real Bullpen Data
                home_bullpen_era_l30, away_bullpen_era_l30,
                home_bullpen_whip_l30, away_bullpen_whip_l30,
                home_bullpen_usage_rate, away_bullpen_usage_rate,
                home_bullpen_rest_status, away_bullpen_rest_status,
                -- Phase 2: Recent Trends Data
                home_team_runs_l7, away_team_runs_l7,
                home_team_runs_l20, away_team_runs_l20,
                home_team_runs_allowed_l7, away_team_runs_allowed_l7,
                home_team_runs_allowed_l20, away_team_runs_allowed_l20,
                home_team_ops_l14, away_team_ops_l14,
                home_team_ops_l20, away_team_ops_l20
            FROM enhanced_games 
            WHERE home_bullpen_era_l30 IS NOT NULL 
            AND home_team_runs_l7 IS NOT NULL
            AND date >= '2025-03-20'
            ORDER BY RANDOM()
            LIMIT 10
        """)
        
        samples = cursor.fetchall()
        
        for i, sample in enumerate(samples, 1):
            (date, home_team, away_team, home_score, away_score, total_runs,
             home_bp_era, away_bp_era, home_bp_whip, away_bp_whip,
             home_bp_usage, away_bp_usage, home_bp_rest, away_bp_rest,
             home_runs_l7, away_runs_l7, home_runs_l20, away_runs_l20,
             home_allow_l7, away_allow_l7, home_allow_l20, away_allow_l20,
             home_ops_l14, away_ops_l14, home_ops_l20, away_ops_l20) = sample
            
            print(f"\n🏟️  SAMPLE {i}: {date}")
            print(f"🆚 {away_team} @ {home_team}")
            print(f"📊 Final Score: {away_team} {away_score} - {home_score} {home_team} (Total: {total_runs})")
            
            print(f"\n   📈 PHASE 1 - REAL BULLPEN DATA:")
            print(f"      Home Bullpen: ERA={home_bp_era}, WHIP={home_bp_whip}, Usage={home_bp_usage}%, Rest={home_bp_rest}")
            print(f"      Away Bullpen: ERA={away_bp_era}, WHIP={away_bp_whip}, Usage={away_bp_usage}%, Rest={away_bp_rest}")
            
            print(f"\n   📊 PHASE 2 - RECENT TRENDS DATA:")
            print(f"      {home_team} Recent Performance:")
            
            # Handle potential NULL values
            home_l7 = f"{home_runs_l7:.2f}" if home_runs_l7 is not None else "NULL"
            home_l20 = f"{home_runs_l20:.2f}" if home_runs_l20 is not None else "NULL"
            home_allow7 = f"{home_allow_l7:.2f}" if home_allow_l7 is not None else "NULL"
            home_allow20 = f"{home_allow_l20:.2f}" if home_allow_l20 is not None else "NULL"
            home_ops14 = f"{home_ops_l14:.3f}" if home_ops_l14 is not None else "NULL"
            home_ops20 = f"{home_ops_l20:.3f}" if home_ops_l20 is not None else "NULL"
            
            away_l7 = f"{away_runs_l7:.2f}" if away_runs_l7 is not None else "NULL"
            away_l20 = f"{away_runs_l20:.2f}" if away_runs_l20 is not None else "NULL"
            away_allow7 = f"{away_allow_l7:.2f}" if away_allow_l7 is not None else "NULL"
            away_allow20 = f"{away_allow_l20:.2f}" if away_allow_l20 is not None else "NULL"
            away_ops14 = f"{away_ops_l14:.3f}" if away_ops_l14 is not None else "NULL"
            away_ops20 = f"{away_ops_l20:.3f}" if away_ops_l20 is not None else "NULL"
            
            print(f"        Runs: L7={home_l7}, L20={home_l20}")
            print(f"        Runs Allowed: L7={home_allow7}, L20={home_allow20}")
            print(f"        OPS: L14={home_ops14}, L20={home_ops20}")
            print(f"      {away_team} Recent Performance:")
            print(f"        Runs: L7={away_l7}, L20={away_l20}")
            print(f"        Runs Allowed: L7={away_allow7}, L20={away_allow20}")
            print(f"        OPS: L14={away_ops14}, L20={away_ops20}")
            
            print("-" * 80)
        
        # Show data variety statistics
        print(f"\n📊 DATA VARIETY VERIFICATION:")
        print("=" * 50)
        
        # Bullpen ERA variety
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT home_bullpen_era_l30) as unique_eras,
                MIN(home_bullpen_era_l30) as min_era,
                MAX(home_bullpen_era_l30) as max_era,
                AVG(home_bullpen_era_l30) as avg_era,
                COUNT(*) as total_games
            FROM enhanced_games 
            WHERE home_bullpen_era_l30 IS NOT NULL
        """)
        
        era_stats = cursor.fetchone()
        print(f"🎯 Bullpen ERA Variety:")
        print(f"   Unique ERA values: {era_stats[0]}")
        print(f"   Range: {era_stats[1]:.2f} - {era_stats[2]:.2f}")
        print(f"   Average: {era_stats[3]:.2f}")
        print(f"   Total games: {era_stats[4]}")
        
        # Recent trends variety
        cursor.execute("""
            SELECT 
                AVG(home_team_runs_l7) as avg_home_l7,
                AVG(away_team_runs_l7) as avg_away_l7,
                MIN(home_team_runs_l7) as min_runs_l7,
                MAX(home_team_runs_l7) as max_runs_l7,
                AVG(home_team_ops_l14) as avg_ops
            FROM enhanced_games 
            WHERE home_team_runs_l7 IS NOT NULL
        """)
        
        trend_stats = cursor.fetchone()
        print(f"\n🎯 Recent Trends Variety:")
        print(f"   L7 Runs Average: Home={trend_stats[0]:.2f}, Away={trend_stats[1]:.2f}")
        print(f"   L7 Runs Range: {trend_stats[2]:.2f} - {trend_stats[3]:.2f}")
        print(f"   L14 OPS Average: {trend_stats[4]:.3f}")
        
        # Check for any fake data remnants
        cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE home_bullpen_era_l30 = 4.50")
        fake_count = cursor.fetchone()[0]
        print(f"\n✅ Fake 4.50 ERA placeholders remaining: {fake_count}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Data sampling failed: {e}")

if __name__ == "__main__":
    show_real_data_samples()
