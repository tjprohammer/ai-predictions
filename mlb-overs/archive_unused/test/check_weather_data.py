from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

with engine.begin() as conn:
    print('WEATHER DATA VERIFICATION FOR 2025-08-13')
    print('=' * 50)
    
    # Get all weather data for today
    weather_data = pd.read_sql("""
        SELECT game_id, home_team, away_team, weather_condition, 
               temperature, wind_speed, wind_direction 
        FROM enhanced_games 
        WHERE date = '2025-08-13' 
        AND weather_condition IS NOT NULL 
        ORDER BY wind_speed DESC
    """, conn)
    
    print(f'Games with weather data: {len(weather_data)}')
    print()
    
    if len(weather_data) > 0:
        print('DETAILED WEATHER DATA:')
        print('-' * 80)
        for _, row in weather_data.iterrows():
            wind_info = f"{row['wind_speed']}mph {row['wind_direction']}" if row['wind_speed'] is not None else "No wind data"
            print(f"{row['away_team']} @ {row['home_team']}")
            print(f"  Weather: {row['temperature']}°F, {row['weather_condition']}, Wind: {wind_info}")
            print()
        
        # Summary stats
        print('WEATHER SUMMARY:')
        print('-' * 30)
        print(f"Temperature range: {weather_data['temperature'].min()}°F - {weather_data['temperature'].max()}°F")
        print(f"Wind speed range: {weather_data['wind_speed'].min()}-{weather_data['wind_speed'].max()} mph")
        print()
        print("Weather conditions:")
        conditions = weather_data['weather_condition'].value_counts()
        for condition, count in conditions.items():
            print(f"  {condition}: {count} games")
        print()
        print("Wind directions:")
        directions = weather_data['wind_direction'].value_counts()
        for direction, count in directions.items():
            print(f"  {direction}: {count} games")
    else:
        print('❌ No weather data found!')
