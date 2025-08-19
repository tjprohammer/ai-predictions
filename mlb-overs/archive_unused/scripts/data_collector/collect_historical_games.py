#!/usr/bin/env python3
"""
Simpler season data collector - gets games month by month.
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from tqdm import tqdm

def fetch_games_for_date_range(start_date, end_date):
    """Fetch games for a specific date range."""
    print(f"📅 Fetching games from {start_date} to {end_date}...")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'sportId': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        games = []
        total_games = 0
        
        # Count total games first for progress bar
        if 'dates' in data:
            for date_obj in data['dates']:
                if 'games' in date_obj:
                    for game in date_obj['games']:
                        if game.get('status', {}).get('statusCode') == 'F':
                            total_games += 1
        
        print(f"🎯 Found {total_games} completed games to process...")
        
        if total_games == 0:
            return games
            
        # Process games with progress bar
        with tqdm(total=total_games, desc="Processing games", unit="game") as pbar:
            if 'dates' in data:
                for date_obj in data['dates']:
                    if 'games' not in date_obj:
                        continue
                        
                    for game in date_obj['games']:
                        # Only completed games
                        if game.get('status', {}).get('statusCode') != 'F':
                            continue
                        
                        pbar.set_description(f"Processing game {game['gamePk']}")
                        
                        home_score = game['teams']['home'].get('score')
                        away_score = game['teams']['away'].get('score')

                        # Fetch boxscore for pitcher stats and starting pitcher IDs
                        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game['gamePk']}/boxscore"
                        pitcher_stats = {
                            'home_sp_id': None, 'away_sp_id': None,
                            'home_sp_er': None, 'home_sp_ip': None, 'home_sp_so': None, 'home_sp_bb': None, 'home_sp_h': None,
                            'away_sp_er': None, 'away_sp_ip': None, 'away_sp_so': None, 'away_sp_bb': None, 'away_sp_h': None
                        }
                        try:
                            boxscore_resp = requests.get(boxscore_url, timeout=30)
                            boxscore_resp.raise_for_status()
                            boxscore = boxscore_resp.json()
                            
                            # Find starting pitchers (first pitcher in batting order or with most innings)
                            # Home starter
                            home_pitchers = []
                            if 'teams' in boxscore and 'home' in boxscore['teams'] and 'players' in boxscore['teams']['home']:
                                for pid, pdata in boxscore['teams']['home']['players'].items():
                                    if 'stats' in pdata and 'pitching' in pdata['stats']:
                                        ip_str = pdata['stats']['pitching'].get('inningsPitched', '0')
                                        try:
                                            ip = float(ip_str) if ip_str else 0.0
                                            if ip > 0:
                                                home_pitchers.append((ip, pdata['person']['id'], pdata['stats']['pitching']))
                                        except (ValueError, TypeError):
                                            continue
                            
                            if home_pitchers:
                                # Starting pitcher is usually the one with most innings (or first to pitch)
                                home_pitchers.sort(key=lambda x: x[0], reverse=True)
                                pitcher_stats['home_sp_id'] = home_pitchers[0][1]
                                pstats = home_pitchers[0][2]
                                pitcher_stats['home_sp_er'] = pstats.get('earnedRuns')
                                pitcher_stats['home_sp_ip'] = pstats.get('inningsPitched')
                                pitcher_stats['home_sp_so'] = pstats.get('strikeOuts')
                                pitcher_stats['home_sp_bb'] = pstats.get('baseOnBalls')
                                pitcher_stats['home_sp_h'] = pstats.get('hits')
                            
                            # Away starter
                            away_pitchers = []
                            if 'teams' in boxscore and 'away' in boxscore['teams'] and 'players' in boxscore['teams']['away']:
                                for pid, pdata in boxscore['teams']['away']['players'].items():
                                    if 'stats' in pdata and 'pitching' in pdata['stats']:
                                        ip_str = pdata['stats']['pitching'].get('inningsPitched', '0')
                                        try:
                                            ip = float(ip_str) if ip_str else 0.0
                                            if ip > 0:
                                                away_pitchers.append((ip, pdata['person']['id'], pdata['stats']['pitching']))
                                        except (ValueError, TypeError):
                                            continue
                            
                            if away_pitchers:
                                # Starting pitcher is usually the one with most innings
                                away_pitchers.sort(key=lambda x: x[0], reverse=True)
                                pitcher_stats['away_sp_id'] = away_pitchers[0][1]
                                pstats = away_pitchers[0][2]
                                pitcher_stats['away_sp_er'] = pstats.get('earnedRuns')
                                pitcher_stats['away_sp_ip'] = pstats.get('inningsPitched')
                                pitcher_stats['away_sp_so'] = pstats.get('strikeOuts')
                                pitcher_stats['away_sp_bb'] = pstats.get('baseOnBalls')
                                pitcher_stats['away_sp_h'] = pstats.get('hits')
                                
                        except Exception as e:
                            # Continue without pitcher stats if boxscore fails
                            pass

                        if home_score is not None and away_score is not None:
                            games.append({
                                'game_id': game['gamePk'],
                                'date': date_obj['date'],
                                'home_team': game['teams']['home']['team']['name'],
                                'away_team': game['teams']['away']['team']['name'],
                                'home_score': home_score,
                                'away_score': away_score,
                                'total_runs': home_score + away_score,
                                'home_sp_id': pitcher_stats['home_sp_id'],
                                'away_sp_id': pitcher_stats['away_sp_id'],
                                'home_sp_er': pitcher_stats['home_sp_er'],
                                'home_sp_ip': pitcher_stats['home_sp_ip'],
                                'home_sp_so': pitcher_stats['home_sp_so'],
                                'home_sp_bb': pitcher_stats['home_sp_bb'],
                                'home_sp_h': pitcher_stats['home_sp_h'],
                                'away_sp_er': pitcher_stats['away_sp_er'],
                                'away_sp_ip': pitcher_stats['away_sp_ip'],
                                'away_sp_so': pitcher_stats['away_sp_so'],
                                'away_sp_bb': pitcher_stats['away_sp_bb'],
                                'away_sp_h': pitcher_stats['away_sp_h']
                            })
                        
                        pbar.update(1)
                        time.sleep(0.1)  # Small delay to be nice to API
        
        return games

    except Exception as e:
        print(f"Error fetching games for {start_date}-{end_date}: {e}")
        return []

def collect_season_games():
    """Collect games month by month to avoid API issues."""
    all_games = []
    
    # Define month ranges for 2025 season (adjust based on actual season)
    month_ranges = [
        ('2025-03-20', '2025-04-30'),  # March-April (Opening Day typically late March)
        ('2025-05-01', '2025-05-31'),  # May
        ('2025-06-01', '2025-06-30'),  # June
        ('2025-07-01', '2025-07-31'),  # July
        ('2025-08-01', '2025-08-12'),  # August so far
    ]
    
    print(f"\n🏟️  MLB 2025 SEASON DATA COLLECTION")
    print(f"📊 Collecting data from {len(month_ranges)} time periods...")
    print(f"🎯 Estimated total games: ~2000+ games")
    print(f"⚾ Including pitcher stats (ERA, IP, SO, BB, H)")
    print("="*60)
    
    for i, (start_date, end_date) in enumerate(month_ranges, 1):
        print(f"\n📅 Period {i}/{len(month_ranges)}: {start_date} to {end_date}")
        games = fetch_games_for_date_range(start_date, end_date)
        all_games.extend(games)
        print(f"✅ Collected {len(games)} games for this period")
        print(f"📈 Total collected so far: {len(all_games)} games")
        time.sleep(1)  # Be nice to the API
    
    return all_games

def estimate_betting_lines(games_df):
    """Estimate betting lines based on team and historical averages."""
    print("Estimating betting lines...")
    
    # Start with league average
    games_df['k_close'] = 8.5
    
    # Adjust based on actual scoring patterns
    # High-scoring teams/parks
    high_scoring = ['Rockies', 'Rangers', 'Astros', 'Red Sox', 'Blue Jays', 'Yankees']
    low_scoring = ['Rays', 'Giants', 'Marlins', 'Athletics', 'Guardians']
    
    for idx, row in games_df.iterrows():
        base_total = 8.5
        
        # Adjust for team tendencies
        if any(team in row['home_team'] for team in high_scoring):
            base_total += 0.5
        if any(team in row['away_team'] for team in high_scoring):
            base_total += 0.5
        if any(team in row['home_team'] for team in low_scoring):
            base_total -= 0.5
        if any(team in row['away_team'] for team in low_scoring):
            base_total -= 0.5
        
        # Add some variation based on month (weather effects)
        month = pd.to_datetime(row['date']).month
        if month in [4, 5]:  # Cooler months
            base_total -= 0.2
        elif month in [7, 8]:  # Hot months
            base_total += 0.2
        
        games_df.at[idx, 'k_close'] = round(base_total * 2) / 2  # Round to nearest 0.5

    return games_df

def main():
    print("🏆 SIMPLIFIED SEASON DATA COLLECTION")
    print("="*60)
    
    # Collect all games
    all_games = collect_season_games()
    
    if not all_games:
        print("❌ No games collected")
        return
    
    # Convert to DataFrame
    games_df = pd.DataFrame(all_games)
    games_df['date'] = pd.to_datetime(games_df['date'])

    # Print columns to confirm pitcher stats are present
    print("\n--- DataFrame Columns After Collection ---")
    print(games_df.columns.tolist())

    # Sort by date
    games_df = games_df.sort_values('date')

    # Estimate betting lines
    games_df = estimate_betting_lines(games_df)

    # Calculate some statistics
    print(f"\n📊 SEASON DATA SUMMARY:")
    print(f"Total games collected: {len(games_df)}")
    print(f"Date range: {games_df['date'].min().date()} to {games_df['date'].max().date()}")
    print(f"Average total runs: {games_df['total_runs'].mean():.2f}")
    print(f"Average estimated line: {games_df['k_close'].mean():.2f}")
    print(f"Actual vs line bias: {(games_df['total_runs'] - games_df['k_close']).mean():.2f}")
    
    # Show pitcher data quality
    games_with_pitchers = games_df[(games_df['home_sp_id'].notna()) | (games_df['away_sp_id'].notna())]
    pitcher_coverage = len(games_with_pitchers) / len(games_df) * 100
    print(f"Pitcher data coverage: {pitcher_coverage:.1f}% ({len(games_with_pitchers)}/{len(games_df)} games)")
    
    # Show over/under record
    games_df['went_over'] = games_df['total_runs'] > games_df['k_close']
    over_pct = games_df['went_over'].mean() * 100
    print(f"Over percentage: {over_pct:.1f}%")    # Distribution of totals
    print(f"\n🎯 TOTAL RUNS DISTRIBUTION:")
    for total in sorted(games_df['total_runs'].unique()):
        count = (games_df['total_runs'] == total).sum()
        pct = count / len(games_df) * 100
        print(f"{total:2d} runs: {count:3d} games ({pct:4.1f}%)")

    # Save the data
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    parquet_path = output_dir / 'historical_games_2025.parquet'
    games_df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Saved to: {parquet_path}")

    csv_path = output_dir / 'historical_games_2025.csv'
    games_df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")

    return games_df

if __name__ == '__main__':
    main()
