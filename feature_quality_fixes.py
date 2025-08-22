#!/usr/bin/env python3
"""
Feature Quality Fix Script
=========================
Fixes problematic features identified in comprehensive training:

1. Starting Pitcher Days Rest - Calculate real rest from previous starts
2. Recent Team Performance (L7/L20/L30) - Calculate rolling averages
3. Form Rating - Calculate from recent win/loss streaks
4. Missing data imputation for key features

This will improve the 96-feature model to include more meaningful features.
"""

import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def fix_starting_pitcher_days_rest():
    """
    Calculate real starting pitcher days rest instead of constant 4.0
    """
    print("🔧 FIXING STARTING PITCHER DAYS REST")
    print("-" * 50)
    
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    cursor = conn.cursor()
    
    # Get all games in training period
    cursor.execute("""
        SELECT game_id, date, home_team, away_team, home_sp_name, away_sp_name
        FROM enhanced_games 
        WHERE date >= '2025-03-20' AND date <= '2025-08-21'
        AND home_sp_name IS NOT NULL AND away_sp_name IS NOT NULL
        ORDER BY date, game_id
    """)
    
    games = cursor.fetchall()
    updates_made = 0
    
    print(f"   Processing {len(games)} games...")
    
    for game_id, date, home_team, away_team, home_sp, away_sp in games:
        # Calculate home SP days rest
        cursor.execute("""
            SELECT MAX(date) as last_start
            FROM enhanced_games 
            WHERE date < %s 
            AND ((home_team = %s AND home_sp_name = %s) OR (away_team = %s AND away_sp_name = %s))
        """, (date, home_team, home_sp, home_team, home_sp))
        
        home_last_start = cursor.fetchone()[0]
        home_rest = 4  # Default
        if home_last_start:
            home_rest = (date - home_last_start).days
        
        # Calculate away SP days rest  
        cursor.execute("""
            SELECT MAX(date) as last_start
            FROM enhanced_games 
            WHERE date < %s 
            AND ((home_team = %s AND home_sp_name = %s) OR (away_team = %s AND away_sp_name = %s))
        """, (date, away_team, away_sp, away_team, away_sp))
        
        away_last_start = cursor.fetchone()[0]
        away_rest = 4  # Default
        if away_last_start:
            away_rest = (date - away_last_start).days
        
        # Update the game
        cursor.execute("""
            UPDATE enhanced_games 
            SET home_sp_days_rest = %s, away_sp_days_rest = %s
            WHERE game_id = %s
        """, (home_rest, away_rest, game_id))
        
        updates_made += 1
        if updates_made % 200 == 0:
            print(f"   Updated {updates_made} games...")
    
    conn.commit()
    conn.close()
    
    print(f"   ✅ Fixed days rest for {updates_made} games")

def fix_recent_team_performance():
    """
    Calculate recent team performance metrics (L7, L14, L20, L30)
    """
    print("\n🔧 FIXING RECENT TEAM PERFORMANCE METRICS")
    print("-" * 50)
    
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    cursor = conn.cursor()
    
    # Get all games to update
    cursor.execute("""
        SELECT game_id, date, home_team, away_team
        FROM enhanced_games 
        WHERE date >= '2025-03-20' AND date <= '2025-08-21'
        ORDER BY date, game_id
    """)
    
    games = cursor.fetchall()
    updates_made = 0
    
    print(f"   Processing {len(games)} games...")
    
    for game_id, date, home_team, away_team in games:
        # Calculate recent performance for home team
        for days in [7, 14, 20, 30]:
            start_date = date - timedelta(days=days)
            
            # Home team runs scored in last N days
            cursor.execute("""
                SELECT AVG(CASE WHEN home_team = %s THEN home_score ELSE away_score END) as avg_runs
                FROM enhanced_games 
                WHERE date BETWEEN %s AND %s
                AND date < %s
                AND (home_team = %s OR away_team = %s)
                AND total_runs IS NOT NULL
            """, (home_team, start_date, date, date, home_team, home_team))
            
            home_runs_avg = cursor.fetchone()[0] or 4.5  # Default MLB average
            
            # Home team runs allowed in last N days
            cursor.execute("""
                SELECT AVG(CASE WHEN home_team = %s THEN away_score ELSE home_score END) as avg_runs_allowed
                FROM enhanced_games 
                WHERE date BETWEEN %s AND %s
                AND date < %s
                AND (home_team = %s OR away_team = %s)
                AND total_runs IS NOT NULL
            """, (home_team, start_date, date, date, home_team, home_team))
            
            home_runs_allowed_avg = cursor.fetchone()[0] or 4.5
            
            # Away team runs scored in last N days
            cursor.execute("""
                SELECT AVG(CASE WHEN home_team = %s THEN home_score ELSE away_score END) as avg_runs
                FROM enhanced_games 
                WHERE date BETWEEN %s AND %s
                AND date < %s
                AND (home_team = %s OR away_team = %s)
                AND total_runs IS NOT NULL
            """, (away_team, start_date, date, date, away_team, away_team))
            
            away_runs_avg = cursor.fetchone()[0] or 4.5
            
            # Away team runs allowed in last N days
            cursor.execute("""
                SELECT AVG(CASE WHEN home_team = %s THEN away_score ELSE home_score END) as avg_runs_allowed
                FROM enhanced_games 
                WHERE date BETWEEN %s AND %s
                AND date < %s
                AND (home_team = %s OR away_team = %s)
                AND total_runs IS NOT NULL
            """, (away_team, start_date, date, date, away_team, away_team))
            
            away_runs_allowed_avg = cursor.fetchone()[0] or 4.5
            
            # Update the specific columns
            if days == 7:
                cursor.execute("""
                    UPDATE enhanced_games 
                    SET home_team_runs_l7 = %s, 
                        home_team_runs_allowed_l7 = %s,
                        away_team_runs_l7 = %s,
                        away_team_runs_allowed_l7 = %s
                    WHERE game_id = %s
                """, (home_runs_avg, home_runs_allowed_avg, away_runs_avg, away_runs_allowed_avg, game_id))
            elif days == 20:
                cursor.execute("""
                    UPDATE enhanced_games 
                    SET home_team_runs_l20 = %s, 
                        home_team_runs_allowed_l20 = %s,
                        away_team_runs_l20 = %s,
                        away_team_runs_allowed_l20 = %s
                    WHERE game_id = %s
                """, (home_runs_avg, home_runs_allowed_avg, away_runs_avg, away_runs_allowed_avg, game_id))
            elif days == 30:
                cursor.execute("""
                    UPDATE enhanced_games 
                    SET home_team_runs_l30 = %s, 
                        away_team_runs_l30 = %s
                    WHERE game_id = %s
                """, (home_runs_avg, away_runs_avg, game_id))
        
        updates_made += 1
        if updates_made % 100 == 0:
            print(f"   Updated {updates_made} games...")
    
    conn.commit()
    conn.close()
    
    print(f"   ✅ Fixed recent performance for {updates_made} games")

def fix_form_rating():
    """
    Calculate team form rating based on recent wins/losses
    """
    print("\n🔧 FIXING TEAM FORM RATING")
    print("-" * 50)
    
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    cursor = conn.cursor()
    
    # Get all games to update
    cursor.execute("""
        SELECT game_id, date, home_team, away_team
        FROM enhanced_games 
        WHERE date >= '2025-03-20' AND date <= '2025-08-21'
        ORDER BY date, game_id
    """)
    
    games = cursor.fetchall()
    updates_made = 0
    
    print(f"   Processing {len(games)} games...")
    
    for game_id, date, home_team, away_team in games:
        # Calculate form rating for last 10 games
        start_date = date - timedelta(days=20)  # Look back 20 days for ~10 games
        
        # Home team form
        cursor.execute("""
            SELECT 
                COUNT(*) as games_played,
                SUM(CASE 
                    WHEN (home_team = %s AND home_score > away_score) OR 
                         (away_team = %s AND away_score > home_score) 
                    THEN 1 ELSE 0 
                END) as wins
            FROM enhanced_games 
            WHERE date BETWEEN %s AND %s
            AND date < %s
            AND (home_team = %s OR away_team = %s)
            AND total_runs IS NOT NULL
        """, (home_team, home_team, start_date, date, date, home_team, home_team))
        
        home_games, home_wins = cursor.fetchone()
        home_form = 5.0  # Default neutral
        if home_games and home_games > 0:
            home_form = (home_wins / home_games) * 10  # Scale 0-10
        
        # Away team form
        cursor.execute("""
            SELECT 
                COUNT(*) as games_played,
                SUM(CASE 
                    WHEN (home_team = %s AND home_score > away_score) OR 
                         (away_team = %s AND away_score > home_score) 
                    THEN 1 ELSE 0 
                END) as wins
            FROM enhanced_games 
            WHERE date BETWEEN %s AND %s
            AND date < %s
            AND (home_team = %s OR away_team = %s)
            AND total_runs IS NOT NULL
        """, (away_team, away_team, start_date, date, date, away_team, away_team))
        
        away_games, away_wins = cursor.fetchone()
        away_form = 5.0  # Default neutral
        if away_games and away_games > 0:
            away_form = (away_wins / away_games) * 10  # Scale 0-10
        
        # Update form ratings
        cursor.execute("""
            UPDATE enhanced_games 
            SET home_team_form_rating = %s, away_team_form_rating = %s
            WHERE game_id = %s
        """, (home_form, away_form, game_id))
        
        updates_made += 1
        if updates_made % 200 == 0:
            print(f"   Updated {updates_made} games...")
    
    conn.commit()
    conn.close()
    
    print(f"   ✅ Fixed form rating for {updates_made} games")

def verify_fixes():
    """
    Verify that the fixes improved feature quality
    """
    print("\n🔍 VERIFYING FEATURE FIXES")
    print("-" * 50)
    
    conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
    cursor = conn.cursor()
    
    features_to_check = [
        'home_sp_days_rest',
        'away_sp_days_rest',
        'home_team_runs_l7',
        'away_team_runs_l7', 
        'home_team_runs_l20',
        'away_team_runs_l20',
        'home_team_form_rating',
        'away_team_form_rating'
    ]
    
    for feature in features_to_check:
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                COUNT({feature}) as non_null,
                COUNT(DISTINCT {feature}) as unique_vals,
                MIN({feature}) as min_val,
                MAX({feature}) as max_val,
                AVG({feature}) as avg_val
            FROM enhanced_games 
            WHERE date >= '2025-03-20' AND date <= '2025-08-21'
        """)
        
        total, non_null, unique, min_val, max_val, avg_val = cursor.fetchone()
        missing_pct = ((total - non_null) / total * 100) if total > 0 else 0
        
        print(f"   {feature}:")
        print(f"     Non-null: {non_null}/{total} ({100-missing_pct:.1f}%)")
        print(f"     Unique values: {unique}")
        if min_val is not None:
            print(f"     Range: {min_val:.2f} - {max_val:.2f} (avg: {avg_val:.2f})")
        
        if missing_pct < 5 and unique > 10:
            print(f"     ✅ FIXED!")
        elif missing_pct < 20:
            print(f"     🔧 IMPROVED")
        else:
            print(f"     ❌ Still needs work")
        print()
    
    conn.close()

def main():
    print("🚀 FEATURE QUALITY FIX SCRIPT")
    print("=" * 70)
    print("Fixing problematic features identified in comprehensive training")
    print()
    
    try:
        # Fix the main problematic features
        fix_starting_pitcher_days_rest()
        fix_recent_team_performance() 
        fix_form_rating()
        
        # Verify improvements
        verify_fixes()
        
        print("\n🎉 FEATURE FIXES COMPLETE!")
        print("   Ready to retrain comprehensive model with improved features")
        print("   Should have better variance and more meaningful data")
        
    except Exception as e:
        print(f"❌ Error during feature fixes: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
