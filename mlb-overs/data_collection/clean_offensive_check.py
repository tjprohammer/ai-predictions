#!/usr/bin/env python3
"""
Clean check of offensive stats coverage
"""

import psycopg2

def clean_offensive_check():
    """Clean check of offensive stats"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        print("🎯 OFFENSIVE STATS STATUS REPORT")
        print("=" * 50)
        
        # Check total games
        cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE date >= '2025-03-20'")
        total = cursor.fetchone()[0]
        print(f"Total recent games: {total}")
        
        # Check specific offensive coverage
        offensive_checks = [
            ('home_team_avg', 'Team Batting Average'),
            ('home_team_ops', 'Team OPS'),
            ('home_team_slg', 'Team Slugging'),
            ('home_team_obp', 'Team On-Base %'),
            ('home_team_wrc_plus', 'Team wRC+'),
            ('home_team_iso', 'Team ISO'),
            ('home_team_runs_l7', 'Recent Runs L7'),
            ('home_team_ops_l14', 'Recent OPS L14'),
            ('home_team_ops_l20', 'Recent OPS L20')
        ]
        
        print(f"\n📊 OFFENSIVE STATS COVERAGE:")
        print("-" * 40)
        
        for col_name, desc in offensive_checks:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {col_name} IS NOT NULL AND date >= '2025-03-20'")
                count = cursor.fetchone()[0]
                pct = 100 * count / total if total > 0 else 0
                status = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
                print(f"{status} {desc:20}: {count:4}/{total} ({pct:5.1f}%)")
            except:
                print(f"❌ {desc:20}: Column not found")
        
        # Sample recent data
        print(f"\n🎯 SAMPLE RECENT OFFENSIVE DATA:")
        print("-" * 50)
        
        cursor.execute("""
            SELECT home_team, away_team, date, total_runs,
                   home_team_ops, away_team_ops,
                   home_team_wrc_plus, away_team_wrc_plus,
                   home_team_runs_l7, away_team_runs_l7
            FROM enhanced_games 
            WHERE date >= '2025-08-15' 
            AND home_team_ops IS NOT NULL
            ORDER BY date DESC
            LIMIT 5
        """)
        
        samples = cursor.fetchall()
        if samples:
            print("Date       | Teams             | Runs | Home OPS | Away OPS | Home wRC+ | Away wRC+")
            print("-" * 75)
            for sample in samples:
                home, away, date, runs, h_ops, a_ops, h_wrc, a_wrc, h_l7, a_l7 = sample
                h_ops_str = f"{h_ops:.3f}" if h_ops else "NULL"
                a_ops_str = f"{a_ops:.3f}" if a_ops else "NULL"
                h_wrc_str = f"{h_wrc}" if h_wrc else "NULL"
                a_wrc_str = f"{a_wrc}" if a_wrc else "NULL"
                print(f"{date} | {away[:7]:7}@{home[:7]:7} | {runs:4} | {h_ops_str:8} | {a_ops_str:8} | {h_wrc_str:9} | {a_wrc_str:9}")
        
        # Check gaps in data
        print(f"\n🔍 CHECKING DATA GAPS:")
        print("-" * 30)
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM enhanced_games 
            WHERE date >= '2025-03-20' 
            AND home_team_avg IS NULL
        """)
        missing_avg = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM enhanced_games 
            WHERE date >= '2025-03-20' 
            AND home_team_ops IS NULL
        """)
        missing_ops = cursor.fetchone()[0]
        
        print(f"Missing team AVG: {missing_avg}/{total} games")
        print(f"Missing team OPS: {missing_ops}/{total} games")
        
        if missing_avg > total * 0.1:  # More than 10% missing
            print(f"\n⚠️  NEEDS ENHANCEMENT: Team batting averages")
        
        if missing_ops > total * 0.1:
            print(f"\n⚠️  NEEDS ENHANCEMENT: Team OPS data")
        else:
            print(f"\n✅ Team OPS coverage is good ({100*(total-missing_ops)/total:.1f}%)")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_offensive_check()
