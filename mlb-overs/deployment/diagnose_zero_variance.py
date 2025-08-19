#!/usr/bin/env python3
"""Diagnostic script to investigate zero-variance features."""

from sqlalchemy import create_engine, text
import pandas as pd
import os

def check_ballpark_data():
    """Check ballpark factor data availability."""
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    print("=== BALLPARK FACTORS INVESTIGATION ===")
    
    # Check if ballpark_factors table exists
    try:
        ballpark_query = text("SELECT table_name FROM information_schema.tables WHERE table_name = 'ballpark_factors'")
        result = pd.read_sql(ballpark_query, engine)
        if len(result) > 0:
            print("✅ ballpark_factors table exists")
            
            # Check sample data
            sample_query = text("SELECT * FROM ballpark_factors LIMIT 5")
            sample = pd.read_sql(sample_query, engine)
            print(f"Sample ballpark factors ({len(sample)} rows):")
            print(sample.to_string())
            
            # Check distinct parks
            parks_query = text("SELECT DISTINCT ballpark FROM ballpark_factors")
            parks = pd.read_sql(parks_query, engine)
            print(f"\nDistinct ballparks: {len(parks)}")
            
        else:
            print("❌ ballpark_factors table does not exist")
    except Exception as e:
        print(f"❌ Error checking ballpark_factors: {e}")
    
    # Check enhanced_games for ballpark data
    try:
        eg_query = text("""
            SELECT 
                ballpark_run_factor,
                ballpark_hr_factor,
                COUNT(*) as count
            FROM enhanced_games 
            WHERE date = '2025-08-16'
            GROUP BY ballpark_run_factor, ballpark_hr_factor
        """)
        eg_ballpark = pd.read_sql(eg_query, engine)
        print(f"\nBallpark factors in enhanced_games for 2025-08-16:")
        print(eg_ballpark.to_string())
        
    except Exception as e:
        print(f"❌ Error checking enhanced_games ballpark data: {e}")

def check_weather_data():
    """Check weather data availability."""
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    print("\n=== WEATHER DATA INVESTIGATION ===")
    
    # Check weather_game table
    try:
        weather_query = text("SELECT table_name FROM information_schema.tables WHERE table_name = 'weather_game'")
        result = pd.read_sql(weather_query, engine)
        if len(result) > 0:
            print("✅ weather_game table exists")
            
            # Check for today's data
            today_query = text("SELECT * FROM weather_game WHERE date = '2025-08-16' LIMIT 3")
            today_weather = pd.read_sql(today_query, engine)
            print(f"Weather data for 2025-08-16 ({len(today_weather)} rows):")
            if len(today_weather) > 0:
                print(today_weather[['temp_f', 'wind_mph', 'humidity_pct']].to_string())
            else:
                print("No weather data found for 2025-08-16")
                
        else:
            print("❌ weather_game table does not exist")
    except Exception as e:
        print(f"❌ Error checking weather_game: {e}")
    
    # Check enhanced_games for weather columns
    try:
        eg_weather_query = text("""
            SELECT 
                temperature,
                wind_speed,
                humidity,
                COUNT(*) as count
            FROM enhanced_games 
            WHERE date = '2025-08-16'
            GROUP BY temperature, wind_speed, humidity
        """)
        eg_weather = pd.read_sql(eg_weather_query, engine)
        print(f"\nWeather data in enhanced_games for 2025-08-16:")
        print(eg_weather.to_string())
        
    except Exception as e:
        print(f"❌ Error checking enhanced_games weather: {e}")

def check_pitcher_quality_data():
    """Check pitcher quality/experience data."""
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    print("\n=== PITCHER EXPERIENCE INVESTIGATION ===")
    
    try:
        # Check for pitcher quality columns in enhanced_games
        quality_query = text("""
            SELECT 
                home_pitcher_quality,
                away_pitcher_quality,
                pitching_dominance,
                COUNT(*) as count
            FROM enhanced_games 
            WHERE date = '2025-08-16'
            GROUP BY home_pitcher_quality, away_pitcher_quality, pitching_dominance
        """)
        quality_data = pd.read_sql(quality_query, engine)
        print(f"Pitcher quality data in enhanced_games for 2025-08-16:")
        print(quality_data.to_string())
        
    except Exception as e:
        print(f"❌ Error checking pitcher quality: {e}")

def check_offense_imbalance():
    """Check offense imbalance calculation."""
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    print("\n=== OFFENSE IMBALANCE INVESTIGATION ===")
    
    try:
        # Check team offense data
        offense_query = text("""
            SELECT 
                home_team,
                away_team,
                home_team_runs_per_game,
                away_team_runs_per_game,
                (home_team_runs_per_game - away_team_runs_per_game) as offense_diff
            FROM enhanced_games 
            WHERE date = '2025-08-16'
            LIMIT 5
        """)
        offense_data = pd.read_sql(offense_query, engine)
        print(f"Team offense data for 2025-08-16:")
        print(offense_data.to_string())
        
        # Check if all teams have same offense values
        rpg_query = text("""
            SELECT 
                DISTINCT home_team_runs_per_game,
                DISTINCT away_team_runs_per_game,
                COUNT(*) as count
            FROM enhanced_games 
            WHERE date = '2025-08-16'
            GROUP BY home_team_runs_per_game, away_team_runs_per_game
        """)
        rpg_data = pd.read_sql(rpg_query, engine)
        print(f"\nDistinct RPG values:")
        print(rpg_data.to_string())
        
    except Exception as e:
        print(f"❌ Error checking offense data: {e}")

def main():
    print("INVESTIGATING ZERO-VARIANCE FEATURES")
    print("=" * 50)
    
    check_ballpark_data()
    check_weather_data()
    check_pitcher_quality_data()
    check_offense_imbalance()
    
    print("\n" + "=" * 50)
    print("SUMMARY: Zero variance likely caused by:")
    print("1. Missing ballpark_factors table or all factors = 1.0")
    print("2. Missing weather_game data or all weather values identical")
    print("3. Missing pitcher quality metrics")
    print("4. All teams having identical offense stats")

if __name__ == "__main__":
    main()
