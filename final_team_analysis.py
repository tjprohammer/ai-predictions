"""
Final Team Performance Analysis with Clean Data
"""

import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def analyze_clean_team_data():
    """Analyze team performance with cleaned data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("📊 FINAL TEAM PERFORMANCE ANALYSIS - CLEAN DATA")
    print("=" * 70)
    
    teams_to_analyze = [
        'Milwaukee Brewers', 'New York Mets', 'Arizona Diamondbacks', 
        'San Francisco Giants', 'Chicago Cubs', 'Pittsburgh Pirates',
        'Los Angeles Dodgers', 'Colorado Rockies', 'Toronto Blue Jays'
    ]
    
    print(f"{'Team':20} | {'Last 5':>6} | {'Last 15':>7} | {'Last 25':>7} | {'Status':>10}")
    print("-" * 70)
    
    hot_teams = []
    cold_teams = []
    
    for team in teams_to_analyze:
        # Get last 5 games average
        cursor.execute("""
            SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily 
                WHERE team = %s AND runs_pg IS NOT NULL 
                ORDER BY date DESC LIMIT 5
            ) recent_5
        """, (team,))
        runs_5 = cursor.fetchone()[0]
        
        # Get last 15 games average
        cursor.execute("""
            SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily 
                WHERE team = %s AND runs_pg IS NOT NULL 
                ORDER BY date DESC LIMIT 15
            ) recent_15
        """, (team,))
        runs_15 = cursor.fetchone()[0]
        
        # Get last 25 games average
        cursor.execute("""
            SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily 
                WHERE team = %s AND runs_pg IS NOT NULL 
                ORDER BY date DESC LIMIT 25
            ) recent_25
        """, (team,))
        runs_25 = cursor.fetchone()[0]
        
        if runs_5:
            if runs_5 >= 7.0:
                status = "🔥🔥 BLAZING"
            elif runs_5 >= 6.5:
                status = "🔥 HOT"
                hot_teams.append((team, runs_5))
            elif runs_5 >= 5.5:
                status = "🌡️ WARM"
            elif runs_5 >= 4.0:
                status = "😐 NORMAL"
            elif runs_5 >= 3.0:
                status = "🧊 COLD"
                cold_teams.append((team, runs_5))
            else:
                status = "🧊🧊 FROZEN"
                cold_teams.append((team, runs_5))
                
            print(f"{team:20} | {runs_5:6.2f} | {runs_15:7.2f} | {runs_25:7.2f} | {status:>10}")
    
    print(f"\n🔥 HOT TEAMS (6.5+ R/G):")
    for team, rpg in hot_teams:
        print(f"   {team}: {rpg:.2f} R/G")
    
    print(f"\n🧊 COLD TEAMS (3.0 or less R/G):")
    for team, rpg in cold_teams:
        print(f"   {team}: {rpg:.2f} R/G")
    
    # Now check today's games
    print(f"\n🏟️ TODAY'S GAMES WITH ACCURATE TEAM STATUS:")
    print("=" * 70)
    
    cursor.execute("""
        SELECT home_team, away_team, predicted_total, market_total, 
               recommendation, confidence, edge
        FROM enhanced_games 
        WHERE date = CURRENT_DATE
        ORDER BY confidence DESC
    """)
    
    games = cursor.fetchall()
    
    for home, away, pred, market, rec, conf, edge in games[:5]:  # Top 5 by confidence
        # Get home team performance
        cursor.execute("""
            SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily 
                WHERE team = %s AND runs_pg IS NOT NULL 
                ORDER BY date DESC LIMIT 5
            ) recent
        """, (home,))
        home_runs = cursor.fetchone()[0] or 4.5
        
        # Get away team performance  
        cursor.execute("""
            SELECT AVG(runs_pg) FROM (
                SELECT runs_pg FROM teams_offense_daily 
                WHERE team = %s AND runs_pg IS NOT NULL 
                ORDER BY date DESC LIMIT 5
            ) recent
        """, (away,))
        away_runs = cursor.fetchone()[0] or 4.5
        
        home_status = "🔥" if home_runs >= 6.5 else "🧊" if home_runs <= 3.5 else "😐"
        away_status = "🔥" if away_runs >= 6.5 else "🧊" if away_runs <= 3.5 else "😐"
        
        pick_strength = "PREMIUM" if conf >= 75 else "STRONG" if conf >= 65 else "MODERATE"
        
        print(f"{away:15} {away_status} @ {home:15} {home_status} | {rec:5} {conf:2.0f}% | {pick_strength}")
    
    conn.close()

if __name__ == "__main__":
    analyze_clean_team_data()
