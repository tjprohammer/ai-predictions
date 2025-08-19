#!/usr/bin/env python3
"""
Quick script to check if weather data was saved to database
"""

from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

# Get database engine
engine = get_engine()

def check_weather_data():
    """Check current weather data in enhanced_games table"""
    print('🔍 Checking weather data in enhanced_games table...')
    print('=' * 60)
    
    try:
        with engine.begin() as conn:
            # Check current weather data
            query = text("""
                SELECT game_id, venue_name, weather_condition, temperature, 
                       wind_speed, wind_direction, date,
                       predicted_total, market_total, recommendation
                FROM enhanced_games 
                WHERE date = CURRENT_DATE
                ORDER BY game_id
            """)
            
            result = conn.execute(query)
            games_found = 0
            
            for row in result:
                games_found += 1
                print(f'🏟️ Game {row.game_id}: {row.venue_name}')
                print(f'   Weather: {row.weather_condition}, {row.temperature}°F')
                print(f'   Wind: {row.wind_speed}mph {row.wind_direction}')
                print(f'   ML Prediction: {row.predicted_total} vs Market: {row.market_total} - {row.recommendation}')
                print()
            
            if games_found == 0:
                print("❌ No games found for today")
            else:
                print(f"✅ Found {games_found} games with data")
                
    except Exception as e:
        print(f"❌ Error checking database: {e}")

def check_pipeline_completeness():
    """Check if all pipeline data is complete"""
    print('\n🔧 Checking pipeline data completeness...')
    print('=' * 60)
    
    try:
        with engine.begin() as conn:
            # Check for missing data
            query = text("""
                SELECT 
                    COUNT(*) as total_games,
                    COUNT(predicted_total) as games_with_predictions,
                    COUNT(weather_condition) as games_with_weather,
                    COUNT(CASE WHEN weather_condition != 'Clear' OR wind_speed != 8 THEN 1 END) as games_with_varied_weather,
                    COUNT(market_total) as games_with_odds,
                    COUNT(over_odds) as games_with_betting_odds
                FROM enhanced_games 
                WHERE date = CURRENT_DATE
            """)
            
            result = conn.execute(query).fetchone()
            
            print(f"📊 Pipeline Data Status:")
            print(f"   Total games: {result.total_games}")
            print(f"   Games with ML predictions: {result.games_with_predictions}/{result.total_games}")
            print(f"   Games with weather data: {result.games_with_weather}/{result.total_games}")
            print(f"   Games with varied weather: {result.games_with_varied_weather}/{result.total_games}")
            print(f"   Games with market odds: {result.games_with_odds}/{result.total_games}")
            print(f"   Games with betting odds: {result.games_with_betting_odds}/{result.total_games}")
            
            # Check for completeness
            if (result.games_with_predictions == result.total_games and 
                result.games_with_weather == result.total_games and
                result.games_with_odds == result.total_games):
                print("\n✅ ALL PIPELINE FUNCTIONS ARE UPDATING DATABASE CORRECTLY!")
            else:
                print("\n⚠️ Some pipeline data may be missing")
                
    except Exception as e:
        print(f"❌ Error checking pipeline: {e}")

if __name__ == "__main__":
    check_weather_data()
    check_pipeline_completeness()
