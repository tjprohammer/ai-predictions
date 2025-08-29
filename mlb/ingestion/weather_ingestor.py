#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Weather Ingestor with Real-Time API Support
==================================================

Provides realistic weather data for MLB stadiums with multiple data sources:
1. Real-time weather APIs (OpenWeather, WeatherAPI) - when API keys available
2. Stadium-specific realistic fallbacks - when APIs unavailable
3. Automatic weather monitoring and updates throughout the day

Usage:
  python weather_ingestor.py --date 2025-08-18 --realtime
  python weather_ingestor.py --date 2025-08-18 --force-update

Environment Variables:
  OPENWEATHER_API_KEY=your_key_here
  WEATHERAPI_KEY=your_key_here (alternative)
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import argparse
import sys
import random
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Load environment variables
load_dotenv()

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def get_stadium_coordinates():
    """Get real stadium coordinates for weather API calls"""
    return {
        'Nationals Park': (38.8730, -77.0074),
        'Yankee Stadium': (40.8296, -73.9262),
        'Fenway Park': (42.3467, -71.0972),
        'Wrigley Field': (41.9484, -87.6553),
        'Dodger Stadium': (34.0739, -118.2400),
        'Oracle Park': (37.7786, -122.3893),
        'Coors Field': (39.7559, -104.9942),
        'Petco Park': (32.7073, -117.1566),
        'Minute Maid Park': (29.7572, -95.3558),
        'Tropicana Field': (27.7682, -82.6534),
        'T-Mobile Park': (47.5914, -122.3326),
        'Camden Yards': (39.2838, -76.6217),
        'Progressive Field': (41.4958, -81.6852),
        'Comerica Park': (42.3390, -83.0485),
        'Kauffman Stadium': (39.0517, -94.4803),
        'Target Field': (44.9817, -93.2776),
        'Guaranteed Rate Field': (41.8299, -87.6338),
        'Globe Life Field': (32.7470, -97.0832),
        'Angel Stadium': (33.8003, -117.8827),
        'Oakland Coliseum': (37.7516, -122.2008),
        'Marlins Park': (25.7781, -80.2195),
        'Truist Park': (33.8906, -84.4677),
        'Great American Ball Park': (39.0974, -84.5066),
        'PNC Park': (40.4469, -80.0057),
        'American Family Field': (43.0280, -87.9712),
        'Busch Stadium': (38.6226, -90.1928),
        'Citizens Bank Park': (39.9061, -75.1665),
        'Citi Field': (40.7571, -73.8458),
        'Chase Field': (33.4453, -112.0667),
        'Coors Field': (39.7559, -104.9942)
    }

def get_real_weather_openweather(lat, lon, api_key):
    """Get real weather from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'imperial'  # Fahrenheit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert wind direction from degrees to cardinal
        wind_deg = data.get('wind', {}).get('deg', 0)
        wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        wind_direction = wind_directions[int((wind_deg + 11.25) / 22.5) % 16]
        
        weather_data = {
            'temperature': round(data['main']['temp']),
            'weather_condition': data['weather'][0]['main'],
            'wind_speed': round(data.get('wind', {}).get('speed', 0)),
            'wind_direction': wind_direction,
            'humidity': data['main'].get('humidity', 50),
            'pressure': data['main'].get('pressure', 1013.25),
            'source': 'OpenWeather_API'
        }
        
        return weather_data
        
    except Exception as e:
        print(f"   ⚠️ OpenWeather API failed: {e}")
        return None

def get_real_weather_weatherapi(lat, lon, api_key):
    """Get real weather from WeatherAPI (alternative)"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json"
        params = {
            'key': api_key,
            'q': f"{lat},{lon}",
            'aqi': 'no'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        current = data['current']
        
        weather_data = {
            'temperature': round(current['temp_f']),
            'weather_condition': current['condition']['text'],
            'wind_speed': round(current['wind_mph']),
            'wind_direction': current['wind_dir'],
            'humidity': current['humidity'],
            'pressure': current['pressure_mb'],
            'source': 'WeatherAPI'
        }
        
        return weather_data
        
    except Exception as e:
        print(f"   ⚠️ WeatherAPI failed: {e}")
        return None

def get_stadium_weather_data():
    """Get realistic weather data for MLB stadiums"""
    # Stadium locations with realistic weather for August
    stadium_weather = {
        'Nationals Park': {
            'temp_range': (78, 88),
            'conditions': ['Clear', 'Partly Cloudy', 'Cloudy'],
            'wind_range': (3, 12),
            'wind_directions': ['NE', 'E', 'SE', 'S', 'SW', 'W']
        },
        'Coors Field': {
            'temp_range': (68, 78),  # Denver - higher altitude, cooler
            'conditions': ['Clear', 'Partly Cloudy'],
            'wind_range': (5, 15),
            'wind_directions': ['W', 'NW', 'SW']
        },
        'Progressive Field': {
            'temp_range': (72, 82),
            'conditions': ['Clear', 'Partly Cloudy', 'Cloudy'],
            'wind_range': (4, 14),
            'wind_directions': ['NE', 'E', 'SE', 'SW']
        },
        'Rogers Centre': {
            'temp_range': (70, 80),  # Toronto
            'conditions': ['Clear', 'Partly Cloudy', 'Cloudy'],
            'wind_range': (6, 16),
            'wind_directions': ['N', 'NE', 'E', 'SE']
        },
        'Citi Field': {
            'temp_range': (76, 86),  # New York
            'conditions': ['Clear', 'Partly Cloudy', 'Hazy'],
            'wind_range': (5, 15),
            'wind_directions': ['NE', 'E', 'SE', 'S']
        },
        'Oriole Park at Camden Yards': {
            'temp_range': (78, 88),  # Baltimore
            'conditions': ['Clear', 'Partly Cloudy', 'Humid'],
            'wind_range': (4, 12),
            'wind_directions': ['NE', 'E', 'SE', 'S', 'SW']
        },
        'T-Mobile Park': {
            'temp_range': (65, 75),  # Seattle - cooler, marine climate
            'conditions': ['Cloudy', 'Partly Cloudy', 'Clear'],
            'wind_range': (6, 18),
            'wind_directions': ['W', 'NW', 'SW']
        },
        'Tropicana Field': {
            'temp_range': (72, 72),  # Indoor dome
            'conditions': ['Dome'],
            'wind_range': (0, 0),
            'wind_directions': ['Calm']
        }
    }
    
    # Default weather for stadiums not specifically listed
    default_weather = {
        'temp_range': (72, 82),
        'conditions': ['Clear', 'Partly Cloudy', 'Cloudy'],
        'wind_range': (5, 15),
        'wind_directions': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    }
    
    return stadium_weather, default_weather

def generate_realistic_weather(venue_name, stadium_weather, default_weather):
    """Generate realistic weather for a venue"""
    # Get venue-specific weather or use default
    weather_params = stadium_weather.get(venue_name, default_weather)
    
    # Generate random weather within realistic ranges
    temp_min, temp_max = weather_params['temp_range']
    temperature = random.randint(temp_min, temp_max)
    
    condition = random.choice(weather_params['conditions'])
    
    wind_min, wind_max = weather_params['wind_range']
    wind_speed = random.randint(wind_min, wind_max)
    
    wind_direction = random.choice(weather_params['wind_directions'])
    
    return {
        'temperature': temperature,
        'weather_condition': condition,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction
    }

def get_weather_for_venue(venue_name, use_realtime=True, force_update=False):
    """Get weather for a venue using real-time APIs first, then fallback to realistic data"""
    
    # Get API keys
    openweather_key = os.environ.get('OPENWEATHER_API_KEY', '').strip()
    weatherapi_key = os.environ.get('WEATHERAPI_KEY', '').strip()
    
    # Get coordinates for this venue
    coordinates = get_stadium_coordinates()
    coords = coordinates.get(venue_name)
    
    # Try real-time weather if API key available and coordinates found
    if use_realtime and coords and (openweather_key or weatherapi_key):
        lat, lon = coords
        print(f"   🌐 Fetching real-time weather for {venue_name} ({lat}, {lon})")
        
        # Try OpenWeather first
        if openweather_key:
            weather = get_real_weather_openweather(lat, lon, openweather_key)
            if weather:
                print(f"   ✅ Real-time weather: {weather['temperature']}°F, {weather['weather_condition']}, Wind {weather['wind_speed']}mph {weather['wind_direction']}")
                return weather
        
        # Try WeatherAPI as backup
        if weatherapi_key:
            weather = get_real_weather_weatherapi(lat, lon, weatherapi_key)
            if weather:
                print(f"   ✅ Real-time weather: {weather['temperature']}°F, {weather['weather_condition']}, Wind {weather['wind_speed']}mph {weather['wind_direction']}")
                return weather
    
    # Fallback to stadium-specific realistic weather
    print(f"   🏟️ Using stadium-specific weather for {venue_name}")
    stadium_weather, default_weather = get_stadium_weather_data()
    weather = generate_realistic_weather(venue_name, stadium_weather, default_weather)
    weather['source'] = 'Stadium_Specific'
    return weather

def update_weather_for_games(target_date=None, use_realtime=True, force_update=False):
    """Update weather data for games with real-time API support"""
    if target_date is None:
        target_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Get games needing weather updates
    if force_update:
        # Update all games regardless of existing weather
        engine = get_engine()
        with engine.begin() as conn:
            query = text("""
                SELECT game_id, venue_name 
                FROM enhanced_games 
                WHERE date = :target_date
            """)
            result = conn.execute(query, {'target_date': target_date})
            games_needing_weather = [(game.game_id, game.venue_name) for game in result.fetchall()]
    else:
        # Only update games missing weather
        games_needing_weather = get_games_needing_weather(target_date)
    
    if not games_needing_weather:
        print(f"All games for {target_date} already have weather data")
        return 0
    
    print(f"[WEATHER] Updating weather for {len(games_needing_weather)} games")
    if use_realtime:
        print("🌐 Attempting real-time weather collection...")
    
    engine = get_engine()
    updated_count = 0
    api_calls = 0
    
    try:
        with engine.begin() as conn:
            for game_id, venue_name in games_needing_weather:
                # Get weather for this venue
                weather_data = get_weather_for_venue(venue_name, use_realtime, force_update)
                
                if weather_data.get('source') in ['OpenWeather_API', 'WeatherAPI']:
                    api_calls += 1
                    # Add small delay between API calls to be respectful
                    time.sleep(0.1)
                
                # Update the game's weather
                update_sql = text("""
                    UPDATE enhanced_games 
                    SET weather_condition = :weather_condition,
                        temperature = :temperature,
                        wind_speed = :wind_speed,
                        wind_direction = :wind_direction,
                        humidity = :humidity
                    WHERE game_id = :game_id AND date = :target_date
                """)
                
                params = {
                    'game_id': game_id,
                    'target_date': target_date,
                    'weather_condition': weather_data['weather_condition'],
                    'temperature': weather_data['temperature'],
                    'wind_speed': weather_data['wind_speed'],
                    'wind_direction': weather_data['wind_direction'],
                    'humidity': weather_data.get('humidity', 50)
                }
                
                result = conn.execute(update_sql, params)
                
                if result.rowcount > 0:
                    updated_count += 1
                    source_icon = "🌐" if weather_data.get('source') in ['OpenWeather_API', 'WeatherAPI'] else "🏟️"
                    print(f"{source_icon} {venue_name}: {weather_data['temperature']}°F, {weather_data['weather_condition']}, Wind {weather_data['wind_speed']}mph {weather_data['wind_direction']}")
                else:
                    print(f"⚠️ Failed to update weather for game {game_id}")
        
        if api_calls > 0:
            print(f"📡 Made {api_calls} real-time weather API calls")
        
        return updated_count
        
    except Exception as e:
        print(f"Error updating weather: {e}")
        return 0

def get_games_needing_weather(target_date=None):
    """Get games from database that need weather updates"""
    if target_date is None:
        target_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    engine = get_engine()
    
    try:
        with engine.begin() as conn:
            query = text("""
                SELECT game_id, venue_name, weather_condition, wind_speed, wind_direction
                FROM enhanced_games 
                WHERE date = :target_date
                AND (weather_condition IS NULL OR weather_condition = 'Unknown' 
                     OR wind_direction IS NULL OR wind_direction = 'Unknown'
                     OR wind_speed IS NULL OR wind_speed = 0)
            """)
            
            result = conn.execute(query, {'target_date': target_date})
            games = result.fetchall()
            
            return [(game.game_id, game.venue_name) for game in games]
            
    except Exception as e:
        print(f"Error getting games needing weather: {e}")
        return []

def main():
    """Main function with real-time weather support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update weather data with real-time API support')
    parser.add_argument('--date', help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--realtime', action='store_true', help='Use real-time weather APIs when available')
    parser.add_argument('--force-update', action='store_true', help='Update weather for all games, even if already set')
    parser.add_argument('--offline', action='store_true', help='Use only stadium-specific data (no API calls)')
    args = parser.parse_args()
    
    target_date = args.date if args.date else pd.Timestamp.now().strftime('%Y-%m-%d')
    use_realtime = args.realtime and not args.offline
    force_update = args.force_update
    
    print(f"[WEATHER] Enhanced Weather Update for {target_date}")
    print("=" * 50)
    
    if use_realtime:
        print("🌐 Real-time weather mode: Will attempt API calls")
        # Check for API keys
        ow_key = os.environ.get('OPENWEATHER_API_KEY', '').strip()
        wa_key = os.environ.get('WEATHERAPI_KEY', '').strip()
        if not ow_key and not wa_key:
            print("No weather API keys found. Set OPENWEATHER_API_KEY or WEATHERAPI_KEY")
            print("   Falling back to stadium-specific weather data")
    else:
        print("Stadium-specific weather mode: No API calls")
    
    if force_update:
        print("Force update mode: Will update all games")
    
    updated = update_weather_for_games(target_date, use_realtime, force_update)
    
    if updated > 0:
        print(f"✅ Updated weather for {updated} games")
        
        # Show weather variance for model validation
        engine = get_engine()
        with engine.begin() as conn:
            weather_stats = pd.read_sql(text("""
                SELECT 
                    COUNT(*) as games,
                    AVG(temperature) as avg_temp,
                    MIN(temperature) as min_temp,
                    MAX(temperature) as max_temp,
                    AVG(wind_speed) as avg_wind,
                    MAX(wind_speed) as max_wind
                FROM enhanced_games 
                WHERE date = :date AND temperature IS NOT NULL
            """), conn, params={'date': target_date})
            
            if len(weather_stats) > 0:
                stats = weather_stats.iloc[0]
                print(f"📊 Weather summary: {stats.games} games, temp {stats.min_temp}-{stats.max_temp}°F (avg {stats.avg_temp:.1f}°F), wind 0-{stats.max_wind}mph (avg {stats.avg_wind:.1f}mph)")
    else:
        print("ℹ️ No weather updates needed")

if __name__ == "__main__":
    main()
