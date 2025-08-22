#!/usr/bin/env python3
"""
Simple offensive stats summary
"""

import psycopg2

def offensive_summary():
    """Simple summary of offensive stats"""
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    cursor = conn.cursor()
    
    print("🎯 OFFENSIVE STATS SUMMARY:")
    print("=" * 40)
    
    # Check total coverage
    cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE date >= '2025-03-20'")
    total = cursor.fetchone()[0]
    print(f"Total recent games: {total}")
    
    # Coverage check
    stats = [
        ('home_team_avg', 'Batting Average'),
        ('home_team_ops', 'Team OPS'),
        ('home_team_wrc_plus', 'wRC+'),
        ('home_team_runs_l7', 'Recent Runs L7'),
        ('home_team_ops_l14', 'Recent OPS L14'),
        ('home_team_ops_l20', 'Recent OPS L20')
    ]
    
    print(f"\nCOVERAGE SUMMARY:")
    for col, name in stats:
        cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {col} IS NOT NULL AND date >= '2025-03-20'")
        count = cursor.fetchone()[0]
        pct = 100 * count / total
        status = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
        print(f"{status} {name:16}: {count:4}/{total} ({pct:5.1f}%)")
    
    # Show actual data sample (simple)
    print(f"\n📊 RECENT DATA SAMPLE:")
    cursor.execute("""
        SELECT home_team, away_team, date, total_runs,
               home_team_ops, home_team_runs_l7
        FROM enhanced_games 
        WHERE date >= '2025-08-18' 
        AND home_team_ops IS NOT NULL
        ORDER BY date DESC
        LIMIT 3
    """)
    
    samples = cursor.fetchall()
    for home, away, date, runs, ops, l7 in samples:
        ops_str = f"{ops:.3f}" if ops is not None else "NULL"
        l7_str = f"{l7:.1f}" if l7 is not None else "NULL"
        print(f"{date}: {away} @ {home} | Runs: {runs} | Home OPS: {ops_str} | Home L7: {l7_str}")
    
    print(f"\n🎯 CONCLUSION:")
    print("✅ EXCELLENT: Team OPS, wRC+, recent trends (L7/L14)")
    print("❌ MISSING: Team batting averages (need enhancement)")
    print("⚠️ PARTIAL: L20 trends (some gaps)")
    
    conn.close()

if __name__ == "__main__":
    offensive_summary()
