"""
Enhanced Pitcher API Helper Functions
Leverages comprehensive pitcher stats table for accurate ERA calculations
Provides fallbacks to original game-level calculations when needed
"""

from sqlalchemy import text
import pandas as pd
from datetime import datetime, timedelta

def get_comprehensive_pitcher_era(engine, pitcher_id, era_type="season"):
    """
    Get pitcher ERA from comprehensive stats table
    
    Args:
        engine: Database connection
        pitcher_id: Pitcher ID (string)
        era_type: "season", "l3", "l5", "l10", or "recent"
    
    Returns:
        ERA value or None if not available
    """
    try:
        with engine.connect() as cx:
            # Get current season year
            current_year = datetime.now().year
            
            if era_type == "season":
                query = text("""
                    SELECT era_season 
                    FROM pitcher_comprehensive_stats 
                    WHERE pitcher_id = :pid AND season_year = :year
                """)
                result = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "year": current_year})
                return result.iloc[0]['era_season'] if not result.empty and result.iloc[0]['era_season'] is not None else None
            
            elif era_type in ["l3", "l5", "l10"]:
                col_name = f"era_{era_type}"
                query = text(f"""
                    SELECT {col_name} 
                    FROM pitcher_comprehensive_stats 
                    WHERE pitcher_id = :pid AND season_year = :year
                """)
                result = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "year": current_year})
                return result.iloc[0][col_name] if not result.empty and result.iloc[0][col_name] is not None else None
            
            elif era_type == "recent":
                # Get most recent 5 games ERA
                return get_comprehensive_pitcher_era(engine, pitcher_id, "l5")
    
    except Exception:
        return None

def enhanced_era_from_rows(engine, pitcher_id, rows_df=None):
    """
    Enhanced ERA calculation with comprehensive stats fallback
    
    First tries comprehensive stats, then falls back to game-level calculation
    """
    # Try comprehensive season ERA first
    comprehensive_era = get_comprehensive_pitcher_era(engine, pitcher_id, "season")
    if comprehensive_era is not None:
        return comprehensive_era
    
    # Fallback to original game-level calculation
    if rows_df is not None and not rows_df.empty:
        total_ip = rows_df['ip'].sum()
        total_er = rows_df['er'].sum()
        if total_ip > 0:
            return round((total_er * 9.0) / total_ip, 2)
    
    # Final fallback to era_game average
    try:
        with engine.connect() as cx:
            query = text("""
                SELECT AVG(era_game) as avg_era
                FROM pitchers_starts 
                WHERE pitcher_id = :pid AND era_game IS NOT NULL
                LIMIT 10
            """)
            result = pd.read_sql(query, cx, params={"pid": str(pitcher_id)})
            if not result.empty and result.iloc[0]['avg_era'] is not None:
                return round(result.iloc[0]['avg_era'], 2)
    except Exception:
        pass
    
    return None

def enhanced_lastN_era(engine, pitcher_id, n=5):
    """
    Enhanced last N games ERA with comprehensive stats support
    """
    # Map common N values to comprehensive stats
    if n == 3:
        comp_era = get_comprehensive_pitcher_era(engine, pitcher_id, "l3")
        if comp_era is not None:
            return comp_era
    elif n == 5:
        comp_era = get_comprehensive_pitcher_era(engine, pitcher_id, "l5")
        if comp_era is not None:
            return comp_era
    elif n == 10:
        comp_era = get_comprehensive_pitcher_era(engine, pitcher_id, "l10")
        if comp_era is not None:
            return comp_era
    
    # Fallback to game-level calculation
    try:
        with engine.connect() as cx:
            query = text("""
                SELECT ip, er, era_game, date
                FROM pitchers_starts 
                WHERE pitcher_id = :pid AND ip IS NOT NULL AND er IS NOT NULL
                ORDER BY date DESC
                LIMIT :n
            """)
            df = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "n": n})
            
            if not df.empty:
                total_ip = df['ip'].sum()
                total_er = df['er'].sum()
                if total_ip > 0:
                    return round((total_er * 9.0) / total_ip, 2)
    except Exception:
        pass
    
    return None

def enhanced_season_era_until(engine, pitcher_id, until_date=None):
    """
    Enhanced season ERA calculation with comprehensive stats
    """
    # Try comprehensive season ERA first
    comp_era = get_comprehensive_pitcher_era(engine, pitcher_id, "season")
    if comp_era is not None:
        return comp_era
    
    # Fallback to game-level calculation with date filter
    try:
        with engine.connect() as cx:
            if until_date:
                query = text("""
                    SELECT ip, er FROM pitchers_starts 
                    WHERE pitcher_id = :pid AND date <= :until_date
                    AND ip IS NOT NULL AND er IS NOT NULL
                """)
                df = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "until_date": until_date})
            else:
                query = text("""
                    SELECT ip, er FROM pitchers_starts 
                    WHERE pitcher_id = :pid 
                    AND ip IS NOT NULL AND er IS NOT NULL
                """)
                df = pd.read_sql(query, cx, params={"pid": str(pitcher_id)})
            
            if not df.empty:
                total_ip = df['ip'].sum()
                total_er = df['er'].sum()
                if total_ip > 0:
                    return round((total_er * 9.0) / total_ip, 2)
    except Exception:
        pass
    
    return None

def enhanced_vs_team_era(engine, pitcher_id, opp_team):
    """
    Enhanced vs team ERA calculation
    """
    try:
        with engine.connect() as cx:
            query = text("""
                SELECT ip, er, era_game
                FROM pitchers_starts 
                WHERE pitcher_id = :pid AND opp_team = :opp_team
                AND ip IS NOT NULL AND er IS NOT NULL
                ORDER BY date DESC
                LIMIT 5
            """)
            df = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "opp_team": opp_team})
            
            if not df.empty:
                total_ip = df['ip'].sum()
                total_er = df['er'].sum()
                if total_ip > 0:
                    return round((total_er * 9.0) / total_ip, 2)
    except Exception:
        pass
    
    return None

def get_pitcher_profile(engine, pitcher_id):
    """
    Get complete pitcher profile including comprehensive stats
    """
    try:
        with engine.connect() as cx:
            # Get comprehensive stats
            comp_query = text("""
                SELECT * FROM pitcher_comprehensive_stats 
                WHERE pitcher_id = :pid 
                ORDER BY season_year DESC 
                LIMIT 1
            """)
            comp_stats = pd.read_sql(comp_query, cx, params={"pid": str(pitcher_id)})
            
            # Get recent game performance
            recent_query = text("""
                SELECT date, opp_team, ip, er, era_game
                FROM pitchers_starts 
                WHERE pitcher_id = :pid 
                AND ip IS NOT NULL AND er IS NOT NULL
                ORDER BY date DESC
                LIMIT 5
            """)
            recent_games = pd.read_sql(recent_query, cx, params={"pid": str(pitcher_id)})
            
            profile = {
                "pitcher_id": str(pitcher_id),
                "has_comprehensive_stats": not comp_stats.empty,
                "recent_games_count": len(recent_games)
            }
            
            if not comp_stats.empty:
                comp_row = comp_stats.iloc[0]
                profile.update({
                    "season_era": comp_row.get('era_season'),
                    "era_l3": comp_row.get('era_l3'),
                    "era_l5": comp_row.get('era_l5'), 
                    "era_l10": comp_row.get('era_l10'),
                    "season_ip": comp_row.get('ip_season'),
                    "season_er": comp_row.get('er_season'),
                    "wins": comp_row.get('wins'),
                    "losses": comp_row.get('losses'),
                    "games_started": comp_row.get('games_started'),
                    "whip": comp_row.get('whip'),
                    "last_updated": comp_row.get('last_updated')
                })
            
            if not recent_games.empty:
                profile["recent_games"] = recent_games.to_dict('records')
            
            return profile
    
    except Exception as e:
        return {"pitcher_id": str(pitcher_id), "error": str(e)}

def test_enhanced_api_functions(engine):
    """
    Test the enhanced API functions with sample pitcher data
    """
    print("Testing Enhanced Pitcher API Functions")
    print("=" * 50)
    
    # Get a sample pitcher from the database
    with engine.connect() as cx:
        sample_query = text("""
            SELECT DISTINCT pitcher_id 
            FROM pitcher_comprehensive_stats 
            WHERE era_season IS NOT NULL 
            LIMIT 3
        """)
        sample_pitchers = pd.read_sql(sample_query, cx)
    
    if sample_pitchers.empty:
        print("No comprehensive pitcher data found for testing")
        return
    
    for _, row in sample_pitchers.iterrows():
        pitcher_id = row['pitcher_id']
        print(f"\nTesting Pitcher ID: {pitcher_id}")
        print("-" * 30)
        
        # Test all ERA functions
        season_era = enhanced_era_from_rows(engine, pitcher_id)
        l3_era = enhanced_lastN_era(engine, pitcher_id, 3)
        l5_era = enhanced_lastN_era(engine, pitcher_id, 5)
        l10_era = enhanced_lastN_era(engine, pitcher_id, 10)
        
        print(f"Season ERA: {season_era}")
        print(f"Last 3 ERA: {l3_era}")
        print(f"Last 5 ERA: {l5_era}")
        print(f"Last 10 ERA: {l10_era}")
        
        # Get full profile
        profile = get_pitcher_profile(engine, pitcher_id)
        print(f"Has Comprehensive Stats: {profile.get('has_comprehensive_stats')}")
        print(f"Recent Games: {profile.get('recent_games_count')}")
        
        if profile.get('wins') is not None:
            print(f"W-L Record: {profile.get('wins')}-{profile.get('losses')}")
        if profile.get('whip') is not None:
            print(f"WHIP: {profile.get('whip')}")

if __name__ == "__main__":
    from configs.connection import get_connection
    
    engine = get_connection()
    test_enhanced_api_functions(engine)
