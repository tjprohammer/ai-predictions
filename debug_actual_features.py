"""
Debug model features using actual database columns
"""

import psycopg2

def debug_actual_features():
    """Debug using the actual database schema"""
    
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    cursor = conn.cursor()
    
    # Get a sample game with all relevant data
    cursor.execute("""
        SELECT 
            home_team, away_team, predicted_total, market_total,
            home_team_avg, away_team_avg,
            home_sp_season_era, away_sp_season_era,
            home_sp_whip, away_sp_whip,
            home_sp_season_k, away_sp_season_k,
            home_sp_season_bb, away_sp_season_bb,
            home_sp_season_ip, away_sp_season_ip,
            temperature, wind_speed,
            ballpark_run_factor, ballpark_hr_factor
        FROM enhanced_games 
        WHERE date = '2025-08-20'
        AND home_team = 'Los Angeles Angels'
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    
    if not result:
        print("No data found")
        return
    
    cols = ['home_team', 'away_team', 'predicted_total', 'market_total',
            'home_team_avg', 'away_team_avg', 'home_sp_season_era', 'away_sp_season_era',
            'home_sp_whip', 'away_sp_whip', 'home_sp_season_k', 'away_sp_season_k',
            'home_sp_season_bb', 'away_sp_season_bb', 'home_sp_season_ip', 'away_sp_season_ip',
            'temperature', 'wind_speed', 'ballpark_run_factor', 'ballpark_hr_factor']
    
    data = dict(zip(cols, result))
    
    print("🔍 DEBUGGING MODEL FEATURES")
    print("=" * 60)
    print(f"Game: {data['away_team']} @ {data['home_team']}")
    print(f"Predicted: {data['predicted_total']}")
    print(f"Market: {data['market_total']}")
    print()
    
    print("📊 TEAM BATTING AVERAGES:")
    print("-" * 40)
    print(f"Home ({data['home_team']}): {data['home_team_avg']:.3f}")
    print(f"Away ({data['away_team']}): {data['away_team_avg']:.3f}")
    
    print("\n🥎 PITCHER STATS:")
    print("-" * 40)
    print(f"Home Pitcher:")
    print(f"  ERA: {data['home_sp_season_era']:.2f}")
    print(f"  WHIP: {data['home_sp_whip']:.2f}")
    print(f"  K: {data['home_sp_season_k']} in {data['home_sp_season_ip']:.1f} IP")
    print(f"  K/9: {(data['home_sp_season_k'] * 9 / data['home_sp_season_ip']):.1f}")
    print(f"  BB: {data['home_sp_season_bb']} in {data['home_sp_season_ip']:.1f} IP")
    print(f"  BB/9: {(data['home_sp_season_bb'] * 9 / data['home_sp_season_ip']):.1f}")
    
    print(f"\nAway Pitcher:")
    print(f"  ERA: {data['away_sp_season_era']:.2f}")
    print(f"  WHIP: {data['away_sp_whip']:.2f}")
    print(f"  K: {data['away_sp_season_k']} in {data['away_sp_season_ip']:.1f} IP")
    print(f"  K/9: {(data['away_sp_season_k'] * 9 / data['away_sp_season_ip']):.1f}")
    print(f"  BB: {data['away_sp_season_bb']} in {data['away_sp_season_ip']:.1f} IP")
    print(f"  BB/9: {(data['away_sp_season_bb'] * 9 / data['away_sp_season_ip']):.1f}")
    
    print(f"\n🏟️ BALLPARK FACTORS:")
    print("-" * 40)
    print(f"Run Factor: {data['ballpark_run_factor']:.2f}")
    print(f"HR Factor: {data['ballpark_hr_factor']:.2f}")
    
    print(f"\n🌤️ WEATHER:")
    print("-" * 40)
    print(f"Temperature: {data['temperature']}°F")
    print(f"Wind Speed: {data['wind_speed']} mph")
    
    # Now let's calculate what a reasonable prediction should be
    print(f"\n🧮 MANUAL CALCULATION CHECK:")
    print("-" * 40)
    
    # Basic calculation: league average runs per team per game is about 4.5
    # Good pitchers reduce that, bad pitchers increase it
    
    league_avg_rpg = 4.5
    
    # ERA impact: league average is about 4.00
    home_era_factor = 4.00 / float(data['home_sp_season_era'])  # < 1 if pitcher is good
    away_era_factor = 4.00 / float(data['away_sp_season_era'])  # < 1 if pitcher is good
    
    print(f"League average per team: {league_avg_rpg:.1f} RPG")
    print(f"Home ERA factor: {home_era_factor:.2f} (ERA {data['home_sp_season_era']:.2f})")
    print(f"Away ERA factor: {away_era_factor:.2f} (ERA {data['away_sp_season_era']:.2f})")
    
    # Rough estimate: home team scores vs away pitcher, away team scores vs home pitcher
    expected_home_runs = league_avg_rpg * away_era_factor * float(data['ballpark_run_factor'])
    expected_away_runs = league_avg_rpg * home_era_factor * float(data['ballpark_run_factor'])
    
    manual_total = expected_home_runs + expected_away_runs
    
    print(f"\nRough manual calculation:")
    print(f"  Home team vs away pitcher: {expected_home_runs:.1f}")
    print(f"  Away team vs home pitcher: {expected_away_runs:.1f}")
    print(f"  Total: {manual_total:.1f}")
    print(f"  Market: {data['market_total']:.1f}")
    print(f"  Our prediction: {data['predicted_total']:.1f}")
    
    if manual_total > 7.0 and data['predicted_total'] < 7.0:
        print(f"\n🚨 PROBLEM IDENTIFIED:")
        print(f"  Manual calculation gives reasonable result: {manual_total:.1f}")
        print(f"  Our model gives unreasonably low result: {data['predicted_total']:.1f}")
        print(f"  This suggests a problem with our feature engineering or model")
    
    # Let's also check if we're getting team offensive stats correctly
    print(f"\n📈 TEAM OFFENSIVE PERFORMANCE CHECK:")
    print("-" * 40)
    
    # Get recent team performance
    cursor.execute("""
        SELECT AVG(home_team_runs) as avg_home_runs, AVG(away_team_runs) as avg_away_runs
        FROM enhanced_games 
        WHERE (home_team = %s OR away_team = %s)
        AND date >= '2025-01-01'
        AND home_team_runs IS NOT NULL
    """, (data['home_team'], data['home_team']))
    
    team_stats = cursor.fetchone()
    
    if team_stats:
        avg_home, avg_away = team_stats
        avg_total = (avg_home or 0) + (avg_away or 0)
        print(f"{data['home_team']} recent average runs per game: {avg_total:.1f}")
        
        if avg_total > 0 and avg_total < 3.0:
            print(f"🚨 SUSPICIOUS: Team averaging only {avg_total:.1f} runs per game")
            print(f"  This is unusually low for MLB teams")
    
    conn.close()

if __name__ == "__main__":
    debug_actual_features()
