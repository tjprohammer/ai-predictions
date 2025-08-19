"""
Inspects the collected historical data for a specific date.
"""
import pandas as pd
import argparse
from sqlalchemy import create_engine
import os
import statsapi

# --- Copied from models/infer.py to avoid import issues ---
def get_pitcher_era_stats_infer(eng, pitcher_id, end_date, opponent_team=None):
    """Get comprehensive ERA statistics for a pitcher using the ERA ingestor"""
    if not pitcher_id:
        return {}
    
    url_parts = eng.url
    database_url = f"postgresql://{url_parts.username}:{url_parts.password}@{url_parts.host}:{url_parts.port}/{url_parts.database}"
    
    # This is a simplified version for inspection purposes
    try:
        # Use statsapi to get season stats as a fallback
        stats = statsapi.player_stats(pitcher_id, group='pitching', type='season', season=end_date.year)
        # statsapi returns a dict with 'stats' key containing a list of stat dicts
        if stats and 'stats' in stats and isinstance(stats['stats'], list) and len(stats['stats']) > 0:
            # Sometimes the stats list contains multiple entries, look for one with 'era'
            for entry in stats['stats']:
                era = entry.get('stats', {}).get('era')
                if era is not None:
                    return {'era_season': era}
        # If no ERA found, log warning
        print(f"Warning: No ERA found for pitcher_id {pitcher_id} in season {end_date.year}")
    except Exception as e:
        print(f"Error fetching ERA for pitcher_id {pitcher_id}: {e}")
    return {}
# --- End of copied code ---


def inspect_data_for_date(data_path: str, target_date: str, eng):
    """
    Loads the historical data and displays games for a specific date.
    """
    print(f"Loading historical data from {data_path}...")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.date

    print("\n--- DataFrame Columns ---")
    print(df.columns.tolist())

    print(f"Filtering for games on {target_date}...")
    target_games = df[df['date'] == pd.to_datetime(target_date).date()]

    if target_games.empty:
        print(f"No games found for {target_date}.")
        return

    print(f"\n--- Games on {target_date} ---")
    # Helper to calculate ERA for a pitcher up to a given date
    def calc_pitcher_era(df, pitcher_id, up_to_date):
        # Filter games where pitcher started and date is before or equal to up_to_date
        games = df[(df['date'] <= up_to_date) & ((df['home_sp_id'] == pitcher_id) | (df['away_sp_id'] == pitcher_id))]
        
        # Sum earned runs and innings pitched (convert strings to floats)
        home_games = games[games['home_sp_id'] == pitcher_id]
        away_games = games[games['away_sp_id'] == pitcher_id]
        
        # Convert to numeric and handle None values
        home_er = pd.to_numeric(home_games['home_sp_er'], errors='coerce').fillna(0).sum()
        home_ip = pd.to_numeric(home_games['home_sp_ip'], errors='coerce').fillna(0).sum()
        away_er = pd.to_numeric(away_games['away_sp_er'], errors='coerce').fillna(0).sum()
        away_ip = pd.to_numeric(away_games['away_sp_ip'], errors='coerce').fillna(0).sum()
        
        total_earned_runs = home_er + away_er
        total_innings_pitched = home_ip + away_ip
        
        if total_innings_pitched > 0:
            era = round((total_earned_runs * 9) / total_innings_pitched, 2)
        else:
            era = 'N/A'
        return era

    for _, game in target_games.iterrows():
        home_sp_id = game['home_sp_id']
        away_sp_id = game['away_sp_id']

        home_sp_name = "N/A"
        home_sp_era = "N/A"
        if pd.notna(home_sp_id):
            try:
                home_sp_name = statsapi.lookup_player(int(home_sp_id))[0]['fullName']
            except Exception:
                home_sp_name = f"ID: {int(home_sp_id)}"
            home_sp_era = calc_pitcher_era(df, int(home_sp_id), game['date'])

        away_sp_name = "N/A"
        away_sp_era = "N/A"
        if pd.notna(away_sp_id):
            try:
                away_sp_name = statsapi.lookup_player(int(away_sp_id))[0]['fullName']
            except Exception:
                away_sp_name = f"ID: {int(away_sp_id)}"
            away_sp_era = calc_pitcher_era(df, int(away_sp_id), game['date'])

        print(
            f"Matchup: {game['away_team']} @ {game['home_team']}\n"
            f"  Score: {game['away_score']} - {game['home_score']} (Total: {game['total_runs']})\n"
            f"  Away Pitcher: {away_sp_name} (ERA: {away_sp_era})\n"
            f"  Home Pitcher: {home_sp_name} (ERA: {home_sp_era})\n"
        )

def main():
    parser = argparse.ArgumentParser(description="Inspect historical MLB game data.")
    parser.add_argument("--date", type=str, default="2025-08-11", help="Date to inspect (YYYY-MM-DD).")
    parser.add_argument("--data-path", type=str, default="S:/Projects/AI_Predictions/mlb-overs/data/historical_games_2025.parquet", help="Path to historical data.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
    args = parser.parse_args()

    eng = create_engine(args.database_url)
    inspect_data_for_date(args.data_path, args.date, eng)

if __name__ == "__main__":
    main()
