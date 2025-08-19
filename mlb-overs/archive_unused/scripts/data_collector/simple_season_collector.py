#!/usr/bin/env python3
"""
Simple and reliable MLB season data collector.
Focuses on getting the data without complex progress bars that cause issues.
"""

import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path

def fetch_games_for_month(start_date, end_date):
    """Fetch games for a month range."""
    print(f"📅 Fetching games from {start_date} to {end_date}...")
    
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'sportId': 1,
        'hydrate': 'boxscore'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
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
                    if game_count % 50 == 0:
                        print(f"  Processing game {game_count}...")
                    
                    home_score = game['teams']['home'].get('score')
                    away_score = game['teams']['away'].get('score')
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    # Get pitcher data from boxscore
                    home_sp_id, home_sp_er, home_sp_ip = None, None, None
                    away_sp_id, away_sp_er, away_sp_ip = None, None, None
                    
                    try:
                        # Fetch boxscore
                        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game['gamePk']}/boxscore"
                        box_resp = requests.get(boxscore_url, timeout=10)
                        box_resp.raise_for_status()
                        boxscore = box_resp.json()
                        
                        # Find home starting pitcher (pitcher with most innings)
                        if 'teams' in boxscore and 'home' in boxscore['teams']:
                            home_pitchers = []
                            players = boxscore['teams']['home'].get('players', {})
                            for pid, pdata in players.items():
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
                                                ip
                                            ))
                                    except:
                                        continue
                            
                            if home_pitchers:
                                home_pitchers.sort(reverse=True)  # Most innings first
                                home_sp_id = home_pitchers[0][1]
                                home_sp_er = home_pitchers[0][2]
                                home_sp_ip = home_pitchers[0][3]
                        
                        # Find away starting pitcher
                        if 'teams' in boxscore and 'away' in boxscore['teams']:
                            away_pitchers = []
                            players = boxscore['teams']['away'].get('players', {})
                            for pid, pdata in players.items():
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
                                                ip
                                            ))
                                    except:
                                        continue
                            
                            if away_pitchers:
                                away_pitchers.sort(reverse=True)  # Most innings first
                                away_sp_id = away_pitchers[0][1]
                                away_sp_er = away_pitchers[0][2]
                                away_sp_ip = away_pitchers[0][3]
                    
                    except Exception as e:
                        # Continue without pitcher data if boxscore fails
                        pass
                    
                    # Add game to collection
                    games.append({
                        'game_id': game['gamePk'],
                        'date': date_obj['date'],
                        'home_team': game['teams']['home']['team']['name'],
                        'away_team': game['teams']['away']['team']['name'],
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': home_score + away_score,
                        'home_sp_id': home_sp_id,
                        'away_sp_id': away_sp_id,
                        'home_sp_er': home_sp_er,
                        'home_sp_ip': home_sp_ip,
                        'away_sp_er': away_sp_er,
                        'away_sp_ip': away_sp_ip
                    })
                    
                    time.sleep(0.05)  # Small delay
        
        print(f"✅ Collected {len(games)} games from {start_date} to {end_date}")
        return games
        
    except Exception as e:
        print(f"❌ Error fetching games for {start_date}-{end_date}: {e}")
        return []

def main():
    print("🏆 MLB 2025 SEASON DATA COLLECTION")
    print("="*50)
    
    # Month ranges for 2025 season
    month_ranges = [
        ('2025-03-20', '2025-04-30'),  # March-April
        ('2025-05-01', '2025-05-31'),  # May
        ('2025-06-01', '2025-06-30'),  # June
        ('2025-07-01', '2025-07-31'),  # July
        ('2025-08-01', '2025-08-12'),  # August so far
    ]
    
    all_games = []
    
    for i, (start_date, end_date) in enumerate(month_ranges, 1):
        print(f"\n📊 Period {i}/{len(month_ranges)}: {start_date} to {end_date}")
        games = fetch_games_for_month(start_date, end_date)
        all_games.extend(games)
        print(f"📈 Total collected so far: {len(all_games)} games")
        time.sleep(2)  # Pause between months
    
    if not all_games:
        print("❌ No games collected!")
        return
    
    # Convert to DataFrame and save
    print(f"\n💾 Processing {len(all_games)} games...")
    df = pd.DataFrame(all_games)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add estimated betting lines
    df['market_total'] = 8.5  # Simple estimate
    
    # Stats
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Total games: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Average runs: {df['total_runs'].mean():.2f}")
    
    # Pitcher data coverage
    pitcher_games = df[(df['home_sp_id'].notna()) | (df['away_sp_id'].notna())]
    coverage = len(pitcher_games) / len(df) * 100
    print(f"Pitcher data: {coverage:.1f}% ({len(pitcher_games)}/{len(df)} games)")
    
    # Save files
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    parquet_path = output_dir / 'historical_games_2025.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Saved: {parquet_path}")
    
    csv_path = output_dir / 'historical_games_2025.csv'
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")
    
    return df

if __name__ == '__main__':
    main()
