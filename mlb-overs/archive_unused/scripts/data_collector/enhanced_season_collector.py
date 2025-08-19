"""
Enhanced season data collector with comprehensive team and player statistics.
Collects batting averages, team stats, weather data, and more for better ML predictions.
"""
import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import json

def fetch_enhanced_games_for_month(start_date, end_date):
    """Fetch games with enhanced statistics for a date range"""
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=boxscore,team,linescore,weather,venue"
    
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        games = []
        game_count = 0
        
        if 'dates' in data:
            for date_obj in data['dates']:
                if 'games' not in date_obj:
                    continue
                    
                for game in date_obj['games']:
                    # Only completed games
                    if game.get('status', {}).get('statusCode') != 'F':
                        continue
                    
                    game_count += 1
                    if game_count % 25 == 0:
                        print(f"  Processing game {game_count}...")
                    
                    home_score = game['teams']['home'].get('score')
                    away_score = game['teams']['away'].get('score')
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    # Basic game data
                    game_data = {
                        'game_id': game['gamePk'],
                        'date': game['gameDate'][:10],
                        'home_team': game['teams']['home']['team']['name'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': home_score + away_score,
                    }
                    
                    # Extract weather data
                    weather_data = {
                        'weather_condition': None,
                        'temperature': None,
                        'wind_speed': None,
                        'wind_direction': None,
                    }
                    
                    if 'weather' in game and game['weather']:
                        weather = game['weather']
                        weather_data['weather_condition'] = weather.get('condition')
                        weather_data['temperature'] = weather.get('temp')
                        
                        # Parse wind data (e.g., "8 mph, Out To LF")
                        wind = weather.get('wind', '')
                        if wind:
                            wind_parts = wind.split(',')
                            if len(wind_parts) >= 2:
                                # Extract speed (e.g., "8 mph")
                                speed_part = wind_parts[0].strip()
                                if 'mph' in speed_part:
                                    try:
                                        weather_data['wind_speed'] = int(speed_part.split()[0])
                                    except:
                                        pass
                                
                                # Extract direction (e.g., "Out To LF")
                                weather_data['wind_direction'] = wind_parts[1].strip()
                    
                    # Add venue information
                    venue_data = {
                        'venue_id': game.get('venue', {}).get('id'),
                        'venue_name': game.get('venue', {}).get('name'),
                    }
                    
                    # Merge weather and venue data
                    game_data.update(weather_data)
                    game_data.update(venue_data)
                    
                    # Get enhanced boxscore data
                    try:
                        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game['gamePk']}/boxscore"
                        box_resp = requests.get(boxscore_url, timeout=10)
                        boxscore = box_resp.json()
                        
                        # Initialize enhanced stats
                        enhanced_stats = {
                            # Pitcher stats
                            'home_sp_id': None, 'home_sp_er': None, 'home_sp_ip': None,
                            'home_sp_k': None, 'home_sp_bb': None, 'home_sp_h': None,
                            'away_sp_id': None, 'away_sp_er': None, 'away_sp_ip': None, 
                            'away_sp_k': None, 'away_sp_bb': None, 'away_sp_h': None,
                            
                            # Team batting stats
                            'home_team_hits': None, 'home_team_runs': None, 'home_team_rbi': None,
                            'home_team_lob': None, 'away_team_hits': None, 'away_team_runs': None,
                            'away_team_rbi': None, 'away_team_lob': None,
                            
                            # Game situation
                            'game_type': game.get('gameType', 'R'),
                            'day_night': game.get('dayNight', 'D'),
                        }
                        
                        # Extract pitcher data (existing logic)
                        if 'teams' in boxscore:
                            # Home team pitchers
                            home_pitchers = []
                            if 'home' in boxscore['teams'] and 'players' in boxscore['teams']['home']:
                                for pid, pdata in boxscore['teams']['home']['players'].items():
                                    if 'stats' in pdata and 'pitching' in pdata['stats']:
                                        stats = pdata['stats']['pitching']
                                        ip = stats.get('inningsPitched', '0')
                                        try:
                                            ip_float = float(ip) if ip else 0.0
                                            if ip_float > 0:
                                                home_pitchers.append((
                                                    ip_float, 
                                                    pdata['person']['id'],
                                                    stats.get('earnedRuns'),
                                                    ip,
                                                    stats.get('strikeOuts', 0),
                                                    stats.get('baseOnBalls', 0),
                                                    stats.get('hits', 0)
                                                ))
                                        except:
                                            continue
                            
                            if home_pitchers:
                                home_pitchers.sort(reverse=True)  # Most innings first
                                enhanced_stats.update({
                                    'home_sp_id': home_pitchers[0][1],
                                    'home_sp_er': home_pitchers[0][2],
                                    'home_sp_ip': home_pitchers[0][3],
                                    'home_sp_k': home_pitchers[0][4],
                                    'home_sp_bb': home_pitchers[0][5],
                                    'home_sp_h': home_pitchers[0][6],
                                })
                            
                            # Away team pitchers
                            away_pitchers = []
                            if 'away' in boxscore['teams'] and 'players' in boxscore['teams']['away']:
                                for pid, pdata in boxscore['teams']['away']['players'].items():
                                    if 'stats' in pdata and 'pitching' in pdata['stats']:
                                        stats = pdata['stats']['pitching']
                                        ip = stats.get('inningsPitched', '0')
                                        try:
                                            ip_float = float(ip) if ip else 0.0
                                            if ip_float > 0:
                                                away_pitchers.append((
                                                    ip_float, 
                                                    pdata['person']['id'],
                                                    stats.get('earnedRuns'),
                                                    ip,
                                                    stats.get('strikeOuts', 0),
                                                    stats.get('baseOnBalls', 0),
                                                    stats.get('hits', 0)
                                                ))
                                        except:
                                            continue
                            
                            if away_pitchers:
                                away_pitchers.sort(reverse=True)  # Most innings first
                                enhanced_stats.update({
                                    'away_sp_id': away_pitchers[0][1],
                                    'away_sp_er': away_pitchers[0][2],
                                    'away_sp_ip': away_pitchers[0][3],
                                    'away_sp_k': away_pitchers[0][4],
                                    'away_sp_bb': away_pitchers[0][5],
                                    'away_sp_h': away_pitchers[0][6],
                                })
                            
                            # Extract team batting stats
                            if 'home' in boxscore['teams'] and 'teamStats' in boxscore['teams']['home']:
                                home_batting = boxscore['teams']['home']['teamStats'].get('batting', {})
                                enhanced_stats.update({
                                    'home_team_hits': home_batting.get('hits'),
                                    'home_team_runs': home_batting.get('runs'),
                                    'home_team_rbi': home_batting.get('rbi'),
                                    'home_team_lob': home_batting.get('leftOnBase'),
                                })
                            
                            if 'away' in boxscore['teams'] and 'teamStats' in boxscore['teams']['away']:
                                away_batting = boxscore['teams']['away']['teamStats'].get('batting', {})
                                enhanced_stats.update({
                                    'away_team_hits': away_batting.get('hits'),
                                    'away_team_runs': away_batting.get('runs'),
                                    'away_team_rbi': away_batting.get('rbi'),
                                    'away_team_lob': away_batting.get('leftOnBase'),
                                })
                        
                        # Merge enhanced stats with basic game data
                        game_data.update(enhanced_stats)
                        
                    except Exception as e:
                        print(f"Warning: Could not get enhanced data for game {game['gamePk']}: {e}")
                        # Continue with basic game data
                    
                    games.append(game_data)
                    time.sleep(0.1)  # Be respectful to the API
        
        print(f"✅ Collected {len(games)} games from {start_date} to {end_date}")
        return games
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Enhanced MLB season data collector")
    parser.add_argument("--output-file", default="enhanced_historical_games_2025.parquet", help="Output file name")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    args = parser.parse_args()
    
    print("🏆 ENHANCED MLB 2025 SEASON DATA COLLECTION")
    print("=" * 60)
    print("📊 Collecting: Pitcher stats, team batting, game situation data")
    print()
    
    # Define season periods (same as before)
    periods = [
        ("2025-03-20", "2025-04-30"),  # Spring/Early season
        ("2025-05-01", "2025-05-31"),  # May
        ("2025-06-01", "2025-06-30"),  # June  
        ("2025-07-01", "2025-07-31"),  # July
        ("2025-08-01", "2025-08-13"),  # August (current)
    ]
    
    all_games = []
    
    for i, (start_date, end_date) in enumerate(periods, 1):
        print(f"📊 Period {i}/{len(periods)}: {start_date} to {end_date}")
        print(f"📅 Fetching enhanced game data from {start_date} to {end_date}...")
        
        games = fetch_enhanced_games_for_month(start_date, end_date)
        all_games.extend(games)
        
        print(f"✅ Period {i} complete: {len(games)} games collected")
        print()
        
        # Small delay between periods
        time.sleep(2)
    
    if all_games:
        df = pd.DataFrame(all_games)
        
        # Save data
        df.to_parquet(args.output_file, index=False)
        print(f"💾 Saved {len(all_games)} enhanced games to {args.output_file}")
        
        # Show summary
        print("\n📈 ENHANCED DATASET SUMMARY")
        print("=" * 40)
        print(f"📊 Total games: {len(all_games)}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📋 Columns: {len(df.columns)}")
        print("🏗️ Enhanced features:")
        print("   • Pitcher: ERA, IP, K, BB, Hits allowed")
        print("   • Team batting: Hits, Runs, RBI, LOB")
        print("   • Weather: Temperature, wind speed/direction, conditions")
        print("   • Venue: Ballpark ID and name")
        print("   • Game context: Day/night, game type")
        print(f"   • All columns: {df.columns.tolist()}")
        
    else:
        print("❌ No games collected!")

if __name__ == "__main__":
    main()
