"""
Data Cleanup and Workflow Integration Script
Fixes duplicate team entries and ensures proper team performance integration
"""

import psycopg2
from datetime import datetime

# Standard team mapping
TEAM_NAME_MAPPING = {
    'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'ATH': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'
}

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser', 
            password='mlbpass'
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def cleanup_duplicate_team_data():
    """Remove duplicate team entries and standardize to full names"""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    print("🧹 CLEANING UP DUPLICATE TEAM DATA")
    print("=" * 60)
    
    # First, let's see what we're working with
    cursor.execute("""
        SELECT team, COUNT(*) as records
        FROM teams_offense_daily
        GROUP BY team
        HAVING COUNT(*) < 50  -- These are likely duplicates or incomplete data
        ORDER BY COUNT(*) DESC
    """)
    
    small_datasets = cursor.fetchall()
    print(f"Teams with <50 records (likely duplicates/incomplete):")
    for team, count in small_datasets:
        print(f"  {team}: {count} records")
    
    # Strategy: Keep only full team names, remove abbreviations and minor league teams
    teams_to_keep = list(TEAM_NAME_MAPPING.values())
    teams_to_keep.append("Oakland Athletics")  # Special case for Athletics
    
    print(f"\\n📋 STANDARDIZATION PLAN:")
    print(f"Keeping {len(teams_to_keep)} standard MLB team names")
    print(f"Removing abbreviations and minor league teams")
    
    # Count what we'll remove
    cursor.execute("""
        SELECT team, COUNT(*) as records
        FROM teams_offense_daily
        WHERE team NOT IN %s
        GROUP BY team
        ORDER BY team
    """, (tuple(teams_to_keep),))
    
    to_remove = cursor.fetchall()
    total_remove = sum(count for _, count in to_remove)
    
    print(f"\\n❌ TO BE REMOVED ({len(to_remove)} team variants, {total_remove} records):")
    for team, count in to_remove:
        print(f"  {team}: {count} records")
    
    # Ask for confirmation
    response = input(f"\\n⚠️  Proceed with removing {total_remove} duplicate/invalid records? (y/N): ")
    
    if response.lower() == 'y':
        cursor.execute("""
            DELETE FROM teams_offense_daily
            WHERE team NOT IN %s
        """, (tuple(teams_to_keep),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"✅ Deleted {deleted_count} duplicate/invalid records")
        
        # Verify cleanup
        cursor.execute("SELECT COUNT(DISTINCT team) FROM teams_offense_daily")
        remaining_teams = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM teams_offense_daily")
        remaining_records = cursor.fetchone()[0]
        
        print(f"✅ Cleanup complete: {remaining_teams} teams, {remaining_records} records remaining")
    else:
        print("❌ Cleanup cancelled")
    
    conn.close()

def create_team_performance_view():
    """Create a database view for easy team performance queries"""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    print("\\n📊 CREATING TEAM PERFORMANCE VIEW")
    print("=" * 60)
    
    # Drop existing view if it exists
    cursor.execute("DROP VIEW IF EXISTS team_recent_performance")
    
    # Create view for recent team performance
    cursor.execute("""
        CREATE VIEW team_recent_performance AS
        SELECT 
            team,
            -- Last 5 games (using window functions)
            (SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.runs_pg IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 5
            ) recent_5) as runs_pg_5,
            
            (SELECT AVG(woba) FROM (
                SELECT woba FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.woba IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 5
            ) recent_5_woba) as woba_5,
            
            -- Last 15 games  
            (SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.runs_pg IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 15
            ) recent_15) as runs_pg_15,
            
            (SELECT AVG(woba) FROM (
                SELECT woba FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.woba IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 15
            ) recent_15_woba) as woba_15,
            
            -- Last 25 games
            (SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.runs_pg IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 25
            ) recent_25) as runs_pg_25,
            
            (SELECT AVG(woba) FROM (
                SELECT woba FROM teams_offense_daily t2 
                WHERE t2.team = teams.team AND t2.woba IS NOT NULL 
                ORDER BY t2.date DESC LIMIT 25
            ) recent_25_woba) as woba_25,
            
            -- Season averages
            AVG(runs_pg) as season_runs_pg,
            AVG(woba) as season_woba,
            COUNT(*) as total_games
        FROM (SELECT DISTINCT team FROM teams_offense_daily) teams
        LEFT JOIN teams_offense_daily ON teams.team = teams_offense_daily.team
        WHERE runs_pg IS NOT NULL
        GROUP BY teams.team
    """)
    
    conn.commit()
    print("✅ Created team_recent_performance view")
    
    # Test the view
    cursor.execute("""
        SELECT team, runs_pg_5, runs_pg_15, runs_pg_25, season_runs_pg
        FROM team_recent_performance
        ORDER BY runs_pg_5 DESC NULLS LAST
        LIMIT 10
    """)
    
    print("\\n🔥 TOP 10 HOTTEST TEAMS (by last 5 games):")
    print(f"{'Team':20} | {'5-Game':>6} | {'15-Game':>7} | {'25-Game':>7} | {'Season':>6}")
    print("-" * 60)
    
    for team, r5, r15, r25, season in cursor.fetchall():
        r5_str = f"{r5:.2f}" if r5 else "N/A"
        r15_str = f"{r15:.2f}" if r15 else "N/A"
        r25_str = f"{r25:.2f}" if r25 else "N/A"
        season_str = f"{season:.2f}" if season else "N/A"
        print(f"{team:20} | {r5_str:>6} | {r15_str:>7} | {r25_str:>7} | {season_str:>6}")
    
    conn.close()

def test_enhanced_integration():
    """Test that our enhanced analysis now works correctly"""
    print("\\n🧪 TESTING ENHANCED ANALYSIS INTEGRATION")
    print("=" * 60)
    
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Test a few key teams
    test_teams = ['Milwaukee Brewers', 'New York Mets', 'San Francisco Giants']
    
    for team in test_teams:
        cursor.execute("""
            SELECT runs_pg_5, runs_pg_15, runs_pg_25, season_runs_pg
            FROM team_recent_performance
            WHERE team = %s
        """, (team,))
        
        result = cursor.fetchone()
        if result:
            r5, r15, r25, season = result
            status = "🔥 HOT" if r5 and r5 >= 6.5 else "🧊 COLD" if r5 and r5 <= 3.5 else "😐 NORMAL"
            print(f"{team:20}: {r5:.2f} (5G) | {r15:.2f} (15G) | {r25:.2f} (25G) | {season:.2f} (Season) {status}")
        else:
            print(f"{team:20}: No data found")
    
    conn.close()

def update_enhanced_analysis_function():
    """Show how to update the enhanced analysis to use the cleaned data"""
    print("\\n🔧 ENHANCED ANALYSIS UPDATE NEEDED")
    print("=" * 60)
    print("After cleanup, update mlb-overs/api/enhanced_analysis.py:")
    print("""
def get_team_recent_performance(team_name: str, days: int = 5) -> Dict[str, float]:
    conn = get_db_connection()
    if not conn:
        return {'recent_runs_pg': 4.5, 'recent_woba': 0.320, 'games': 0}
    
    try:
        cursor = conn.cursor()
        
        # Use the standardized view - much simpler!
        if days == 5:
            cursor.execute(\"\"\"
                SELECT runs_pg_5, woba_5, total_games
                FROM team_recent_performance 
                WHERE team = %s
            \"\"\", (team_name,))
        elif days == 15:
            cursor.execute(\"\"\"
                SELECT runs_pg_15, woba_15, total_games
                FROM team_recent_performance 
                WHERE team = %s
            \"\"\", (team_name,))
        else:  # 25 or other
            cursor.execute(\"\"\"
                SELECT runs_pg_25, woba_25, total_games
                FROM team_recent_performance 
                WHERE team = %s
            \"\"\", (team_name,))
        
        result = cursor.fetchone()
        if result and result[0]:
            return {
                'recent_runs_pg': float(result[0]),
                'recent_woba': float(result[1]) if result[1] else 0.320,
                'games': int(result[2]) if result[2] else 0
            }
    except Exception as e:
        print(f"Error getting team performance: {e}")
    finally:
        conn.close()
    
    return {'recent_runs_pg': 4.5, 'recent_woba': 0.320, 'games': 0}
    """)

if __name__ == "__main__":
    print("🏟️  MLB DATA CLEANUP AND WORKFLOW INTEGRATION")
    print("=" * 80)
    
    # Step 1: Clean up duplicate data
    cleanup_duplicate_team_data()
    
    # Step 2: Create performance view
    create_team_performance_view()
    
    # Step 3: Test integration
    test_enhanced_integration()
    
    # Step 4: Show update needed
    update_enhanced_analysis_function()
    
    print("\\n✅ WORKFLOW INTEGRATION COMPLETE!")
    print("Now your enhanced analysis will have clean, consistent team performance data.")
