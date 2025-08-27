#!/usr/bin/env python3
"""
Check what offensive stats we currently have in the dataset
"""

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_offensive_stats():
    """Check current offensive stats coverage"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        print("🔍 CHECKING OFFENSIVE STATS IN CURRENT DATASET:")
        print("=" * 60)
        
        # Check what offensive columns exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            AND (column_name LIKE '%avg%' OR column_name LIKE '%ops%' OR 
                 column_name LIKE '%obp%' OR column_name LIKE '%slg%' OR
                 column_name LIKE '%wrc%' OR column_name LIKE '%offense%' OR
                 column_name LIKE '%batting%' OR column_name LIKE '%hit%')
            ORDER BY column_name
        """)
        
        offensive_cols = cursor.fetchall()
        print(f"📊 Found {len(offensive_cols)} offensive-related columns:")
        for col in offensive_cols:
            print(f"  - {col[0]}")
        
        # Check team-level stats
        print(f"\n🎯 CHECKING TEAM OFFENSIVE DATA:")
        print("-" * 50)
        
        # Try to get team offensive stats
        try:
            cursor.execute("""
                SELECT home_team, away_team, date, total_runs,
                       home_team_avg, away_team_avg,
                       home_team_obp, away_team_obp,
                       home_team_ops, away_team_ops
                FROM enhanced_games 
                WHERE date >= '2025-08-15'
                ORDER BY date DESC
                LIMIT 5
            """)
            
            samples = cursor.fetchall()
            if samples:
                print("Recent games with team offensive stats:")
                print("Date       | Teams              | Runs | Home AVG | Away AVG | Home OPS | Away OPS")
                print("-" * 75)
                for sample in samples:
                    home, away, date, runs, h_avg, a_avg, h_obp, a_obp, h_ops, a_ops = sample
                    h_avg_str = f"{h_avg:.3f}" if h_avg else "NULL"
                    a_avg_str = f"{a_avg:.3f}" if a_avg else "NULL"
                    h_ops_str = f"{h_ops:.3f}" if h_ops else "NULL"
                    a_ops_str = f"{a_ops:.3f}" if a_ops else "NULL"
                    print(f"{date} | {away[:8]:8}@{home[:8]:8} | {runs:4} | {h_avg_str:8} | {a_avg_str:8} | {h_ops_str:8} | {a_ops_str:8}")
            else:
                print("No team offensive stats found")
                
        except Exception as e:
            print(f"Team offensive stats check failed: {e}")
        
        # Check what team columns actually exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            AND column_name LIKE '%team%'
            ORDER BY column_name
        """)
        
        team_cols = cursor.fetchall()
        print(f"\n📋 Available team columns ({len(team_cols)} total):")
        for i, col in enumerate(team_cols):
            if i < 20:  # Show first 20
                print(f"  - {col[0]}")
        if len(team_cols) > 20:
            print(f"  ... and {len(team_cols) - 20} more")
        
        # Check individual player stats
        print(f"\n⚾ INDIVIDUAL PLAYER OFFENSIVE STATS:")
        print("-" * 50)
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            AND (column_name LIKE '%lineup%' OR column_name LIKE '%batter%' OR
                 column_name LIKE '%h1_%' OR column_name LIKE '%h2_%' OR
                 column_name LIKE '%dh_%')
            ORDER BY column_name
        """)
        
        player_offensive = cursor.fetchall()
        print(f"Individual player offensive columns: {len(player_offensive)}")
        for i, col in enumerate(player_offensive):
            if i < 15:  # Show first 15
                print(f"  - {col[0]}")
        if len(player_offensive) > 15:
            print(f"  ... and {len(player_offensive) - 15} more")
        
        # Check coverage stats
        print(f"\n📈 OFFENSIVE STATS COVERAGE:")
        print("-" * 40)
        
        cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE date >= '2025-03-20'")
        total = cursor.fetchone()[0]
        print(f"Total recent games: {total}")
        
        # Check specific offensive metrics
        offensive_metrics = [
            ('home_team_avg', 'Team Batting Average'),
            ('home_team_ops', 'Team OPS'),
            ('home_team_wrc_plus', 'Team wRC+'),
            ('home_lineup_avg', 'Lineup Average'),
            ('home_team_runs_l7', 'Recent Runs (L7)'),
            ('home_team_ops_l14', 'Recent OPS (L14)')
        ]
        
        for col_name, desc in offensive_metrics:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {col_name} IS NOT NULL AND date >= '2025-03-20'")
                count = cursor.fetchone()[0]
                pct = 100 * count / total if total > 0 else 0
                print(f"{desc:20}: {count:4}/{total} ({pct:5.1f}%)")
            except Exception as e:
                print(f"{desc:20}: Column not found")
        
        # Sample some actual data
        print(f"\n🎯 SAMPLE OFFENSIVE DATA:")
        print("-" * 40)
        
        cursor.execute("""
            SELECT home_team, away_team, date, home_score, away_score,
                   home_team_runs_l7, away_team_runs_l7,
                   home_team_ops_l14, away_team_ops_l14
            FROM enhanced_games 
            WHERE date >= '2025-08-15' 
            AND home_team_runs_l7 IS NOT NULL
            ORDER BY date DESC
            LIMIT 3
        """)
        
        recent_data = cursor.fetchall()
        if recent_data:
            print("Recent games with offensive trends:")
            for game in recent_data:
                home, away, date, h_score, a_score, h_l7, a_l7, h_ops14, a_ops14 = game
                print(f"{date}: {away} {a_score} @ {home} {h_score}")
                print(f"  L7 Runs: {away} {a_l7:.1f}, {home} {h_l7:.1f}")
                print(f"  L14 OPS: {away} {a_ops14:.3f}, {home} {h_ops14:.3f}")
        else:
            print("No recent offensive trend data found")
        
        print(f"\n🚀 OFFENSIVE ENHANCEMENT RECOMMENDATIONS:")
        print("=" * 50)
        print("Based on current data analysis:")
        print("✓ We have recent offensive trends (L7/L14/L20) from Phase 2")
        print("? Need to verify season-long team offensive stats")
        print("? Need individual lineup batting averages")
        print("? Need advanced metrics (wRC+, wOBA, ISO)")
        print("? Need platoon splits (vs LHP/RHP)")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Offensive stats check failed: {e}")

if __name__ == "__main__":
    check_offensive_stats()
