"""
Enhanced Game Analysis with Team Performance Integration
Shows current games with team hot/cold status and comprehensive recommendations
"""

import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Team name mapping for standardization
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

def get_team_performance(team_name: str, periods: List[int] = [5, 15, 25]) -> Dict:
    """Get team performance over multiple periods"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    cursor = conn.cursor()
    
    # Map full team name to potential variants in our data
    search_names = [team_name]
    
    # Add abbreviation if we know it
    for abbr, full_name in TEAM_NAME_MAPPING.items():
        if full_name == team_name:
            search_names.append(abbr)
            break
    
    # Also try "Athletics" mapping
    if team_name == "Oakland Athletics":
        search_names.append("Athletics")
    
    results = {}
    
    for period in periods:
        # Get recent performance with proper limit
        placeholders = ','.join(['%s'] * len(search_names))
        query = f"""
        SELECT 
            AVG(runs_pg) as avg_runs_pg,
            AVG(woba) as avg_woba,
            AVG(wrcplus) as avg_wrcplus,
            COUNT(*) as games
        FROM (
            SELECT * FROM teams_offense_daily 
            WHERE team IN ({placeholders})
            AND runs_pg IS NOT NULL
            ORDER BY date DESC
            LIMIT %s
        ) recent_games
        """
        
        cursor.execute(query, search_names + [period])
        result = cursor.fetchone()
        
        if result and result[0]:
            results[f'last_{period}'] = {
                'runs_pg': float(result[0]),
                'woba': float(result[1]) if result[1] else 0,
                'wrcplus': float(result[2]) if result[2] else 0,
                'games': int(result[3])
            }
    
    conn.close()
    return results

def get_team_status(runs_pg: float) -> str:
    """Determine team status based on recent scoring"""
    if runs_pg >= 7.0:
        return "🔥🔥 BLAZING"
    elif runs_pg >= 6.5:
        return "🔥 HOT"
    elif runs_pg >= 5.5:
        return "🌡️ WARM"
    elif runs_pg >= 4.5:
        return "😐 NORMAL"
    elif runs_pg >= 3.5:
        return "❄️ COOL"
    elif runs_pg >= 2.5:
        return "🧊 COLD"
    else:
        return "🧊🧊 FROZEN"

def get_enhanced_game_analysis():
    """Get today's games with enhanced team performance analysis"""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get today's games
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT 
            home_team, away_team, 
            predicted_total, market_total, 
            recommendation, confidence, edge,
            over_odds, under_odds
        FROM enhanced_games 
        WHERE date = %s
        ORDER BY confidence DESC
    """, (today,))
    
    games = cursor.fetchall()
    
    if not games:
        print(f"No games found for {today}")
        return
    
    print("🏟️  TODAY'S MLB GAMES WITH TEAM PERFORMANCE ANALYSIS")
    print("=" * 100)
    print(f"{'MATCHUP':35} | {'TEAMS STATUS':25} | {'PRED':>4} | {'MKT':>4} | {'REC':>4} | {'CONF':>4} | {'EDGE':>5}")
    print("-" * 100)
    
    strong_picks = []
    premium_picks = []
    hot_team_games = []
    
    for home, away, pred, market, rec, conf, edge, over_odds, under_odds in games:
        # Get team performance
        home_perf = get_team_performance(home)
        away_perf = get_team_performance(away)
        
        home_runs_5 = home_perf.get('last_5', {}).get('runs_pg', 4.5)
        away_runs_5 = away_perf.get('last_5', {}).get('runs_pg', 4.5)
        
        home_status = get_team_status(home_runs_5)
        away_status = get_team_status(away_runs_5)
        
        # Enhanced recommendation analysis
        pick_type = ""
        if conf >= 75:
            pick_type = "⭐ PREMIUM"
            premium_picks.append((away, home, rec, conf, edge))
        elif conf >= 65:
            pick_type = "💪 STRONG"
            strong_picks.append((away, home, rec, conf, edge))
        
        # Flag games with hot teams
        if home_runs_5 >= 6.5 or away_runs_5 >= 6.5:
            hot_team_games.append((away, home, home_runs_5, away_runs_5, rec, conf))
        
        # Display matchup
        matchup = f"{away:15} @ {home:15}"
        team_status = f"{away_status[:8]:8} vs {home_status[:8]:8}"
        
        print(f"{matchup:35} | {team_status:25} | {pred:4.1f} | {market:4.1f} | {rec:>4} | {conf:4.0f}% | {edge:+4.1f} {pick_type}")
    
    # Summary sections
    if premium_picks:
        print(f"\n⭐ PREMIUM PICKS (75%+ Confidence):")
        for away, home, rec, conf, edge in premium_picks:
            print(f"   {away} @ {home}: {rec} ({conf:.0f}% confidence, {edge:+.2f} edge)")
    
    if strong_picks:
        print(f"\n💪 STRONG PICKS (65%+ Confidence):")
        for away, home, rec, conf, edge in strong_picks:
            print(f"   {away} @ {home}: {rec} ({conf:.0f}% confidence, {edge:+.2f} edge)")
    
    if hot_team_games:
        print(f"\n🔥 GAMES WITH HOT TEAMS (6.5+ R/G last 5):")
        for away, home, away_rpg, home_rpg, rec, conf in hot_team_games:
            hot_info = []
            if home_rpg >= 6.5:
                hot_info.append(f"{home} ({home_rpg:.1f} R/G)")
            if away_rpg >= 6.5:
                hot_info.append(f"{away} ({away_rpg:.1f} R/G)")
            
            print(f"   {away} @ {home}: Hot teams: {', '.join(hot_info)} → {rec} ({conf:.0f}%)")
    
    # Show the top hot and cold teams today
    print(f"\n🌡️  TEAM TEMPERATURE CHECK (Last 5 Games):")
    all_teams_today = set()
    for home, away, *_ in games:
        all_teams_today.add(home)
        all_teams_today.add(away)
    
    team_temps = []
    for team in all_teams_today:
        perf = get_team_performance(team)
        runs_5 = perf.get('last_5', {}).get('runs_pg', 4.5)
        team_temps.append((team, runs_5, get_team_status(runs_5)))
    
    team_temps.sort(key=lambda x: x[1], reverse=True)
    
    print("Hottest Teams Today:")
    for team, runs, status in team_temps[:5]:
        print(f"   {team:20}: {runs:.2f} R/G {status}")
    
    print("Coldest Teams Today:")
    for team, runs, status in team_temps[-5:]:
        print(f"   {team:20}: {runs:.2f} R/G {status}")
    
    conn.close()

def check_data_integration():
    """Check how well our team data is integrated"""
    print("\n📊 DATA INTEGRATION STATUS:")
    print("=" * 60)
    
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Check team data coverage
    cursor.execute("""
        SELECT COUNT(DISTINCT team) as unique_teams,
               COUNT(*) as total_records,
               MAX(date) as latest_date
        FROM teams_offense_daily
        WHERE team IN %s
    """, (tuple(TEAM_NAME_MAPPING.values()),))
    
    teams, records, latest = cursor.fetchone()
    print(f"✅ Full team names: {teams} teams, {records} records, latest: {latest}")
    
    # Check abbreviation coverage  
    cursor.execute("""
        SELECT COUNT(DISTINCT team) as unique_teams,
               COUNT(*) as total_records
        FROM teams_offense_daily
        WHERE team IN %s
    """, (tuple(TEAM_NAME_MAPPING.keys()),))
    
    abbrev_teams, abbrev_records = cursor.fetchone()
    print(f"⚠️  Abbreviations: {abbrev_teams} teams, {abbrev_records} records (duplicates)")
    
    # Check games coverage
    cursor.execute("""
        SELECT COUNT(*) as todays_games
        FROM enhanced_games
        WHERE date = %s
    """, (datetime.now().strftime('%Y-%m-%d'),))
    
    todays_games = cursor.fetchone()[0]
    print(f"🎮 Today's games: {todays_games}")
    
    conn.close()

if __name__ == "__main__":
    print("🏟️  MLB Enhanced Game Analysis with Team Performance")
    print("=" * 80)
    
    # Show data integration status
    check_data_integration()
    
    # Show enhanced game analysis
    get_enhanced_game_analysis()
