"""
Comprehensive Team Performance Analysis
Shows recent performance trends for MLB teams over 5, 15, and 25 game periods
"""

import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Standard MLB teams mapping
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

# Reverse mapping for abbreviations
ABBREV_TO_FULL = TEAM_NAME_MAPPING.copy()
FULL_TO_ABBREV = {v: k for k, v in TEAM_NAME_MAPPING.items()}

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

def clean_team_data():
    """Clean and standardize team data by removing duplicates and minor league teams"""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get all teams currently in the database
    cursor.execute("SELECT DISTINCT team FROM teams_offense_daily ORDER BY team")
    all_teams = [row[0] for row in cursor.fetchall()]
    
    print("CURRENT TEAMS IN DATABASE:")
    print("=" * 60)
    
    mlb_teams = []
    duplicates = []
    minor_league = []
    
    for team in all_teams:
        if team in ABBREV_TO_FULL:
            # This is an MLB abbreviation
            mlb_teams.append(team)
        elif team in FULL_TO_ABBREV:
            # This is an MLB full name
            mlb_teams.append(team)
        elif any(mlb_team.lower() in team.lower() for mlb_team in ABBREV_TO_FULL.values()):
            # Potential duplicate or variation
            duplicates.append(team)
        else:
            # Likely minor league or other
            minor_league.append(team)
    
    print(f"MLB Teams (valid): {len(mlb_teams)}")
    print(f"Potential Duplicates: {len(duplicates)}")
    print(f"Minor League/Other: {len(minor_league)}")
    
    if duplicates:
        print(f"\nPOTENTIAL DUPLICATES:")
        for team in duplicates:
            print(f"  - {team}")
    
    if minor_league:
        print(f"\nMINOR LEAGUE/OTHER TEAMS:")
        for team in minor_league:
            print(f"  - {team}")
    
    conn.close()
    return mlb_teams, duplicates, minor_league

def get_team_performance_summary(team_name: str, periods: List[int] = [5, 15, 25]) -> Dict:
    """Get team performance over multiple time periods"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    cursor = conn.cursor()
    
    # Handle both abbreviations and full names
    search_names = [team_name]
    if team_name in ABBREV_TO_FULL:
        search_names.append(ABBREV_TO_FULL[team_name])
    elif team_name in FULL_TO_ABBREV:
        search_names.append(FULL_TO_ABBREV[team_name])
    
    results = {}
    
    for period in periods:
        cutoff_date = (datetime.now() - timedelta(days=period*2)).strftime('%Y-%m-%d')  # Rough estimate
        
        placeholders = ','.join(['%s'] * len(search_names))
        query = f"""
        SELECT 
            AVG(runs_pg) as avg_runs_pg,
            AVG(woba) as avg_woba,
            AVG(wrcplus) as avg_wrcplus,
            AVG(iso) as avg_iso,
            AVG(bb_pct) as avg_bb_pct,
            AVG(k_pct) as avg_k_pct,
            COUNT(*) as games
        FROM (
            SELECT * FROM teams_offense_daily 
            WHERE team IN ({placeholders})
            AND date >= %s
            AND runs_pg IS NOT NULL
            ORDER BY date DESC
            LIMIT %s
        ) recent_games
        """
        
        cursor.execute(query, search_names + [cutoff_date, period])
        result = cursor.fetchone()
        
        if result and result[0]:
            results[f'last_{period}_games'] = {
                'runs_pg': float(result[0]) if result[0] else 0,
                'woba': float(result[1]) if result[1] else 0,
                'wrcplus': float(result[2]) if result[2] else 0,
                'iso': float(result[3]) if result[3] else 0,
                'bb_pct': float(result[4]) if result[4] else 0,
                'k_pct': float(result[5]) if result[5] else 0,
                'games': int(result[6]) if result[6] else 0
            }
    
    conn.close()
    return results

def analyze_all_teams():
    """Analyze performance for all MLB teams"""
    conn = get_db_connection()
    if not conn:
        return
    
    # Get all valid MLB teams from our database
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT team, COUNT(*) as records 
        FROM teams_offense_daily 
        WHERE team IN %s OR team IN %s
        GROUP BY team
        HAVING COUNT(*) > 50  -- Only teams with substantial data
        ORDER BY team
    """, (tuple(ABBREV_TO_FULL.keys()), tuple(ABBREV_TO_FULL.values())))
    
    teams_data = cursor.fetchall()
    
    print("COMPREHENSIVE TEAM PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"{'Team':20} | {'Last 5':>8} | {'Last 15':>8} | {'Last 25':>8} | {'Status':>8}")
    print("-" * 80)
    
    hot_teams = []
    cold_teams = []
    
    for team, records in teams_data:
        if records < 50:  # Skip teams with insufficient data
            continue
            
        performance = get_team_performance_summary(team)
        
        if performance:
            runs_5 = performance.get('last_5_games', {}).get('runs_pg', 0)
            runs_15 = performance.get('last_15_games', {}).get('runs_pg', 0) 
            runs_25 = performance.get('last_25_games', {}).get('runs_pg', 0)
            
            # Determine if team is hot or cold
            status = "NORMAL"
            if runs_5 >= 6.5:
                status = "🔥 HOT"
                hot_teams.append((team, runs_5))
            elif runs_5 <= 3.5:
                status = "🧊 COLD"
                cold_teams.append((team, runs_5))
            elif runs_5 >= 5.5:
                status = "WARM"
            elif runs_5 <= 4.0:
                status = "COOL"
            
            print(f"{team:20} | {runs_5:7.2f} | {runs_15:7.2f} | {runs_25:7.2f} | {status:>8}")
    
    # Summary of hot and cold teams
    if hot_teams:
        print(f"\n🔥 HOT TEAMS (6.5+ R/G last 5):")
        hot_teams.sort(key=lambda x: x[1], reverse=True)
        for team, rpg in hot_teams:
            print(f"   {team}: {rpg:.2f} R/G")
    
    if cold_teams:
        print(f"\n🧊 COLD TEAMS (3.5 or less R/G last 5):")
        cold_teams.sort(key=lambda x: x[1])
        for team, rpg in cold_teams:
            print(f"   {team}: {rpg:.2f} R/G")
    
    conn.close()

def check_workflow_integration():
    """Check if team data is properly integrated into our workflow"""
    print("\nWORKFLOW INTEGRATION STATUS:")
    print("=" * 60)
    
    # Check if we're using team data in predictions
    try:
        from api.enhanced_analysis import get_team_recent_performance
        test_result = get_team_recent_performance("New York Yankees", 5)
        print(f"✅ Team data function working: {test_result}")
    except Exception as e:
        print(f"❌ Team data function error: {e}")
    
    # Check recent API predictions for team data usage
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT home_team, away_team, ai_analysis 
            FROM enhanced_games 
            WHERE date = %s 
            AND ai_analysis IS NOT NULL
            LIMIT 3
        """, (datetime.now().strftime('%Y-%m-%d'),))
        
        games = cursor.fetchall()
        if games:
            print(f"\n✅ Found {len(games)} games with AI analysis today")
            for home, away, analysis in games:
                if 'recent' in analysis.lower() or 'form' in analysis.lower():
                    print(f"✅ {away} @ {home}: Team form detected in analysis")
                else:
                    print(f"⚠️  {away} @ {home}: No team form detected")
        else:
            print("❌ No games with AI analysis found for today")
        
        conn.close()

if __name__ == "__main__":
    print("MLB Team Performance Analysis")
    print("=" * 80)
    
    # 1. Clean and identify data issues
    print("\n1. CHECKING DATA QUALITY:")
    mlb_teams, duplicates, minor_league = clean_team_data()
    
    # 2. Analyze all teams
    print(f"\n2. TEAM PERFORMANCE ANALYSIS:")
    analyze_all_teams()
    
    # 3. Check workflow integration
    print(f"\n3. WORKFLOW INTEGRATION:")
    check_workflow_integration()
