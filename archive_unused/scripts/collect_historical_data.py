"""
Collects historical MLB game data for a given season using the statsapi.
"""
import statsapi
import pandas as pd
from datetime import date
import argparse

def collect_season_data(year: int):
    """
    Collects all game data for a given year, including game IDs, teams,
    scores, and starting pitchers.
    """
    print(f"Collecting historical data for the {year} season...")
    
    games_data = []
    
    # Fetch schedule for the entire season
    schedule = statsapi.schedule(start_date=f'01/01/{year}', end_date=f'12/31/{year}')
    
    total_games = len(schedule)
    print(f"Found {total_games} games scheduled for {year}.")

    for i, game in enumerate(schedule):
        game_id = game['game_id']
        game_date = game['game_date']
        home_team = game['home_name']
        away_team = game['away_name']
        home_score = game['home_score']
        away_score = game['away_score']
        
        # Skip games that haven't been played or have no score
        if game['status'] != 'Final':
            continue

        # Get starting pitchers
        try:
            game_details = statsapi.get('game', {'gamePk': game_id})
            home_sp = game_details.get('gameData', {}).get('probablePitchers', {}).get('home', {}).get('id')
            away_sp = game_details.get('gameData', {}).get('probablePitchers', {}).get('away', {}).get('id')
        except Exception:
            home_sp = None
            away_sp = None

        games_data.append({
            'game_id': game_id,
            'date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'total_runs': home_score + away_score,
            'home_sp_id': home_sp,
            'away_sp_id': away_sp,
            'market_total': 8.5 # Placeholder, to be updated later
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_games} games...")

    df = pd.DataFrame(games_data)
    print(f"Successfully collected data for {len(df)} completed games.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Collect historical MLB game data.")
    parser.add_argument("--year", type=int, default=2024, help="Year to collect data for.")
    parser.add_argument("--out-dir", type=str, default="s:/Projects/AI_Predictions/mlb-overs/data", help="Output directory.")
    args = parser.parse_args()

    df = collect_season_data(args.year)
    
    # Save to parquet file
    output_path = f"{args.out_dir}/historical_games_{args.year}.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Data saved to {output_path}")
    print("\nSample of collected data:")
    print(df.head())

if __name__ == "__main__":
    main()
