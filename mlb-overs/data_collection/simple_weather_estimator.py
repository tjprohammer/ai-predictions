#!/usr/bin/env python3
"""
Simple Weather Data Estimator
============================
Cost-effective approach to estimate weather conditions for MLB games
Uses current weather API + seasonal adjustments + geographic patterns
"""

import requests
import psycopg2
import pandas as pd
from datetime import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleWeatherEstimator:
    def __init__(self):
        self.weather_api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.weather_base_url = "http://api.openweathermap.org/data/2.5"
        
        # Database connection
        self.conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        
        # MLB Ballpark coordinates and characteristics
        self.ballpark_data = {
            # Regular Season MLB Stadiums with geographic characteristics
            'Angel Stadium': {
                'coords': (33.8003, -117.8827),
                'climate': 'dry_warm',
                'avg_humidity': 65,
                'coastal': True
            },
            'Minute Maid Park': {
                'coords': (29.7572, -95.3552),
                'climate': 'humid_warm',
                'avg_humidity': 78,
                'coastal': True
            },
            'Daikin Park': {
                'coords': (29.7572, -95.3552),
                'climate': 'humid_warm',
                'avg_humidity': 78,
                'coastal': True
            },
            'Oakland Coliseum': {
                'coords': (37.7516, -122.2008),
                'climate': 'mild_dry',
                'avg_humidity': 70,
                'coastal': True
            },
            'Tropicana Field': {
                'coords': (27.7682, -82.6534),
                'climate': 'dome',  # Indoor
                'avg_humidity': 45,  # Climate controlled
                'coastal': True
            },
            'Yankee Stadium': {
                'coords': (40.8296, -73.9262),
                'climate': 'continental',
                'avg_humidity': 68,
                'coastal': False
            },
            'Progressive Field': {
                'coords': (41.4962, -81.6852),
                'climate': 'continental',
                'avg_humidity': 72,
                'coastal': False
            },
            'Comerica Park': {
                'coords': (42.3391, -83.0485),
                'climate': 'continental',
                'avg_humidity': 70,
                'coastal': False
            },
            'Kauffman Stadium': {
                'coords': (39.0517, -94.4803),
                'climate': 'continental',
                'avg_humidity': 65,
                'coastal': False
            },
            'Target Field': {
                'coords': (44.9817, -93.2776),
                'climate': 'continental',
                'avg_humidity': 68,
                'coastal': False
            },
            'Fenway Park': {
                'coords': (42.3467, -71.0972),
                'climate': 'coastal_cool',
                'avg_humidity': 72,
                'coastal': True
            },
            'Oriole Park at Camden Yards': {
                'coords': (39.2838, -76.6214),
                'climate': 'humid_continental',
                'avg_humidity': 70,
                'coastal': True
            },
            'Guaranteed Rate Field': {
                'coords': (41.8300, -87.6338),
                'climate': 'continental',
                'avg_humidity': 68,
                'coastal': False
            },
            'Rogers Centre': {
                'coords': (43.6414, -79.3894),
                'climate': 'dome',  # Retractable roof
                'avg_humidity': 50,
                'coastal': False
            },
            'T-Mobile Park': {
                'coords': (47.5914, -122.3326),
                'climate': 'marine',
                'avg_humidity': 75,
                'coastal': True
            },
            'Globe Life Field': {
                'coords': (32.7511, -97.0825),
                'climate': 'hot_dry',
                'avg_humidity': 60,
                'coastal': False
            },
            'Chase Field': {
                'coords': (33.4453, -112.0667),
                'climate': 'dome',  # Indoor
                'avg_humidity': 35,  # Desert + AC
                'coastal': False
            },
            'Coors Field': {
                'coords': (39.7559, -104.9942),
                'climate': 'high_altitude',
                'avg_humidity': 45,  # Mile high
                'coastal': False
            },
            'Dodger Stadium': {
                'coords': (34.0739, -118.2400),
                'climate': 'mediterranean',
                'avg_humidity': 65,
                'coastal': True
            },
            'Petco Park': {
                'coords': (32.7073, -117.1566),
                'climate': 'mediterranean',
                'avg_humidity': 68,
                'coastal': True
            },
            'Oracle Park': {
                'coords': (37.7786, -122.3893),
                'climate': 'marine',
                'avg_humidity': 75,
                'coastal': True
            },
            'Truist Park': {
                'coords': (33.8906, -84.4677),
                'climate': 'humid_subtropical',
                'avg_humidity': 75,
                'coastal': False
            },
            'loanDepot park': {
                'coords': (25.7781, -80.2197),
                'climate': 'tropical',
                'avg_humidity': 80,
                'coastal': True
            },
            'Citi Field': {
                'coords': (40.7571, -73.8458),
                'climate': 'humid_continental',
                'avg_humidity': 70,
                'coastal': True
            },
            'Citizens Bank Park': {
                'coords': (39.9061, -75.1665),
                'climate': 'humid_continental',
                'avg_humidity': 68,
                'coastal': True
            },
            'Nationals Park': {
                'coords': (38.8730, -77.0078),
                'climate': 'humid_continental',
                'avg_humidity': 70,
                'coastal': False
            },
            'PNC Park': {
                'coords': (40.4469, -80.0057),
                'climate': 'humid_continental',
                'avg_humidity': 72,
                'coastal': False
            },
            'Great American Ball Park': {
                'coords': (39.0975, -84.5061),
                'climate': 'humid_continental',
                'avg_humidity': 70,
                'coastal': False
            },
            'American Family Field': {
                'coords': (43.0280, -87.9712),
                'climate': 'continental',
                'avg_humidity': 70,
                'coastal': False
            },
            'Wrigley Field': {
                'coords': (41.9484, -87.6553),
                'climate': 'continental',
                'avg_humidity': 68,
                'coastal': False
            },
            'Busch Stadium': {
                'coords': (38.6226, -90.1928),
                'climate': 'humid_continental',
                'avg_humidity': 68,
                'coastal': False
            }
        }
        
        # Seasonal humidity adjustments (percentage points)
        self.seasonal_adjustments = {
            3: -5,   # March: drier
            4: -3,   # April: mild
            5: 0,    # May: baseline
            6: +5,   # June: more humid
            7: +8,   # July: peak humidity
            8: +8,   # August: peak humidity
            9: +3,   # September: still warm
            10: -2,  # October: cooler/drier
        }

    def get_representative_weather(self, venue: str) -> Optional[Dict]:
        """Get representative weather data for a venue using current weather API"""
        if venue not in self.ballpark_data:
            logger.warning(f"Unknown venue: {venue}")
            return None
            
        venue_data = self.ballpark_data[venue]
        lat, lon = venue_data['coords']
        
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'imperial'
            }
            
            response = requests.get(f"{self.weather_base_url}/weather", params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Weather API error {response.status_code} for {venue}")
                return self.get_fallback_weather(venue_data)
            
            data = response.json()
            main = data.get('main', {})
            wind = data.get('wind', {})
            weather_info = data.get('weather', [{}])[0]
            
            return {
                'temperature': main.get('temp', 75),
                'humidity': main.get('humidity', venue_data['avg_humidity']),
                'pressure': main.get('pressure', 1013),
                'feels_like_temp': main.get('feels_like', main.get('temp', 75)),
                'wind_speed': wind.get('speed', 5),
                'wind_direction': wind.get('deg', 180),
                'weather_description': weather_info.get('description', 'clear'),
                'venue_climate': venue_data['climate']
            }
            
        except Exception as e:
            logger.warning(f"Error getting weather for {venue}: {e}")
            return self.get_fallback_weather(venue_data)

    def get_fallback_weather(self, venue_data: Dict) -> Dict:
        """Generate reasonable fallback weather based on venue characteristics"""
        climate = venue_data['climate']
        avg_humidity = venue_data['avg_humidity']
        
        # Climate-based defaults
        defaults = {
            'dome': {'temp': 72, 'humidity': 45, 'pressure': 1013},
            'desert': {'temp': 85, 'humidity': 35, 'pressure': 1010},
            'coastal_cool': {'temp': 68, 'humidity': 75, 'pressure': 1015},
            'tropical': {'temp': 82, 'humidity': 80, 'pressure': 1012},
            'continental': {'temp': 75, 'humidity': 68, 'pressure': 1013},
            'high_altitude': {'temp': 70, 'humidity': 45, 'pressure': 1000},
        }
        
        base = defaults.get(climate, {'temp': 75, 'humidity': 65, 'pressure': 1013})
        
        return {
            'temperature': base['temp'],
            'humidity': avg_humidity,
            'pressure': base['pressure'],
            'feels_like_temp': base['temp'],
            'wind_speed': 5,
            'wind_direction': 180,
            'weather_description': 'clear',
            'venue_climate': climate
        }

    def estimate_seasonal_weather(self, venue: str, game_date: str, base_weather: Dict) -> Dict:
        """Adjust weather data based on season and venue"""
        date_obj = datetime.strptime(game_date, '%Y-%m-%d')
        month = date_obj.month
        
        # Apply seasonal humidity adjustment
        humidity_adj = self.seasonal_adjustments.get(month, 0)
        adjusted_humidity = max(25, min(95, base_weather['humidity'] + humidity_adj))
        
        # Temperature adjustments by month
        temp_adjustments = {
            3: -15, 4: -8, 5: -3, 6: +5, 7: +10, 8: +8, 9: +2, 10: -5
        }
        temp_adj = temp_adjustments.get(month, 0)
        adjusted_temp = base_weather['temperature'] + temp_adj
        
        # Special venue adjustments
        venue_data = self.ballpark_data.get(venue, {})
        climate = venue_data.get('climate', 'continental')
        
        if climate == 'dome':
            # Climate controlled
            adjusted_humidity = min(50, adjusted_humidity)
            adjusted_temp = 72
        elif climate == 'high_altitude':
            # Coors Field - lower humidity, bigger temperature swings
            adjusted_humidity = max(30, adjusted_humidity - 10)
        elif climate == 'tropical':
            # Miami - consistently high humidity
            adjusted_humidity = max(75, adjusted_humidity)
        
        return {
            **base_weather,
            'humidity': adjusted_humidity,
            'temperature': adjusted_temp,
            'feels_like_temp': adjusted_temp + 2 if adjusted_humidity > 75 else adjusted_temp,
            'seasonal_month': month
        }

    def run_weather_estimation(self):
        """Run the weather estimation process for all games"""
        cursor = self.conn.cursor()
        
        # Get games that need weather data
        cursor.execute("""
            SELECT DISTINCT venue 
            FROM enhanced_games 
            WHERE date >= '2025-04-01' 
            AND date <= '2025-08-21'
            AND humidity IS NULL
            ORDER BY venue
        """)
        
        venues = [row[0] for row in cursor.fetchall()]
        logger.info(f"Getting representative weather for {len(venues)} venues...")
        
        # Get representative weather for each venue
        venue_weather = {}
        for venue in venues:
            logger.info(f"Getting weather data for {venue}...")
            weather = self.get_representative_weather(venue)
            if weather:
                venue_weather[venue] = weather
                time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Successfully got weather data for {len(venue_weather)} venues")
        
        # Now apply to all games
        cursor.execute("""
            SELECT game_id, date, venue 
            FROM enhanced_games 
            WHERE date >= '2025-04-01' 
            AND date <= '2025-08-21'
            AND humidity IS NULL
            ORDER BY date DESC
        """)
        
        games = cursor.fetchall()
        logger.info(f"Estimating weather for {len(games)} games...")
        
        updated = 0
        for game_id, game_date, venue in games:
            base_weather = venue_weather.get(venue)
            if not base_weather:
                continue
                
            # Apply seasonal adjustments
            final_weather = self.estimate_seasonal_weather(venue, str(game_date), base_weather)
            
            # Update database
            cursor.execute("""
                UPDATE enhanced_games 
                SET humidity = %s,
                    pressure = %s,
                    weather_description = %s,
                    feels_like_temp = %s
                WHERE game_id = %s
            """, (
                final_weather['humidity'],
                final_weather['pressure'],
                final_weather['weather_description'],
                final_weather['feels_like_temp'],
                game_id
            ))
            
            updated += 1
            
            if updated % 100 == 0:
                self.conn.commit()
                logger.info(f"Updated {updated}/{len(games)} games...")
        
        self.conn.commit()
        logger.info(f"🎯 Weather estimation complete! Updated {updated} games")
        
        # Validation
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(humidity) as with_humidity,
                (COUNT(humidity) * 100.0 / COUNT(*)) as coverage
            FROM enhanced_games 
            WHERE date >= '2025-04-01' AND date <= '2025-08-21'
        """)
        
        total, with_humidity, coverage = cursor.fetchone()
        logger.info(f"Final humidity coverage: {with_humidity}/{total} ({coverage:.1f}%)")
        
        cursor.close()

def main():
    logger.info("🌤️ Starting cost-effective weather estimation...")
    
    estimator = SimpleWeatherEstimator()
    estimator.run_weather_estimation()
    
    logger.info("✅ Weather estimation complete!")

if __name__ == "__main__":
    main()
